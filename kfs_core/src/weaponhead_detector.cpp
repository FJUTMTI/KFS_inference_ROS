#include "weaponhead_detector.h"

#include <opencv2/imgproc.hpp>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <cmath>

WeaponheadDetector::WeaponheadDetector(const Params& params)
    : params_(params)
{
    // 保持最小依赖, 仅使用 OpenCV 传统函数
}

FrameResult WeaponheadDetector::detect(const cv::Mat& frame) {
    FrameResult result;
    result.frame_size = frame.size();
    result.inference_ms = 0.0;
    if (frame.empty()) return result;

    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat gray, grayF;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else
        gray = frame.clone();
    gray.convertTo(grayF, CV_32F);

    const int w = gray.cols, h = gray.rows;
    const int cx = w / 2, cy = h / 2;

    // ─── A路: 逐行梯度扫描 ───
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred,
        cv::Size(params_.blurKernel | 1, params_.blurKernel | 1), 0);

    int searchTop = std::max(0, cy - params_.searchBandV);
    int searchBot = std::min(h - 1, cy + params_.searchBandV);
    int bandH = searchBot - searchTop + 1;

    cv::Mat bandF, bandSm, grad, gradSm;
    blurred(cv::Range(searchTop, searchBot + 1), cv::Range::all()).convertTo(bandF, CV_32F);
    cv::GaussianBlur(bandF, bandSm, cv::Size(9, 1), 0);

    grad = cv::Mat(bandH, w, CV_32F);
    for (int x = 1; x < w - 1; ++x) {
        cv::Mat col_dst = grad.col(x);
        cv::absdiff(bandSm.col(x + 1), bandSm.col(x - 1), col_dst);
        col_dst *= 0.5f;
    }
    grad.col(1).copyTo(grad.col(0));
    grad.col(w - 2).copyTo(grad.col(w - 1));
    cv::GaussianBlur(grad, gradSm, cv::Size(21, 1), 0);

    struct Row { int y, left, right; };
    std::vector<Row> rows;
    rows.reserve(bandH);

    for (int yi = 0; yi < bandH; ++yi) {
        int yAbs = searchTop + yi;
        double gmin, gmax;
        cv::minMaxLoc(gradSm.row(yi), &gmin, &gmax);
        if (gmax < 1.5) continue;
        float gthresh = std::max(2.0f, float(gmax) * params_.gradRatio);

        const float* gp = gradSm.ptr<float>(yi);
        int left = 0, right = w - 1;
        for (int x = cx; x >= 0; --x)   { if (gp[x] > gthresh) { left  = x; break; } }
        if (left == 0)
            for (int x = 0; x < cx; ++x) { if (gp[x] > gthresh) { left  = x; break; } }
        for (int x = cx; x < w; ++x)    { if (gp[x] > gthresh) { right = x; break; } }
        if (right == w - 1)
            for (int x = w-1; x > cx; --x){ if (gp[x] > gthresh) { right = x; break; } }

        int rw = right - left;
        if (rw < params_.minWidth || rw > params_.maxWidth) continue;
        if (left == 0 || right == w - 1) continue;

        float rm = float(cv::mean(grayF.row(yAbs).colRange(left, right))[0]);
        if (rm > params_.darkMaxGray) continue;

        int m = 12;
        float outL = float(cv::mean(grayF.row(yAbs).colRange(std::max(0,left-m), left))[0]);
        float inL  = float(cv::mean(grayF.row(yAbs).colRange(left, std::min(left+m, right)))[0]);
        float outR = float(cv::mean(grayF.row(yAbs).colRange(right, std::min(w,right+m)))[0]);
        float inR  = float(cv::mean(grayF.row(yAbs).colRange(std::max(left,right-m), right))[0]);
        bool lc = (inL  > 0) ? (outL / inL) >= params_.contrastRatio : (outL > 20);
        bool rc = (inR  > 0) ? (outR / inR) >= params_.contrastRatio : (outR > 20);
        if (!lc && !rc) continue;

        rows.push_back({yAbs, left, right});
    }

    // ─── B路: 二值图 blob 检测 (始终运行, 供验证用) ───
    struct Blob { cv::Rect box; double area; float sat, solidity, grayMean; };
    std::vector<Blob> blobs;

    {
        cv::Mat mask;
        cv::threshold(gray, mask, params_.blobGrayThr, 255, cv::THRESH_BINARY_INV);
        cv::Mat maskCopy = mask.clone();  // findContours 会修改输入
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(maskCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& cnt : contours) {
            double area = cv::contourArea(cnt);
            if (area < params_.blobMinArea) continue;

            cv::Rect box = cv::boundingRect(cnt);
            std::vector<cv::Point> hull;
            cv::convexHull(cnt, hull);
            double hullArea = cv::contourArea(hull);
            float solidity = (hullArea > 0) ? float(area / hullArea) : 0;
            if (solidity < params_.blobMinSolidity) continue;

            cv::Mat roiHSV;
            cv::cvtColor(frame(box), roiHSV, cv::COLOR_BGR2HSV);
            cv::Mat roiMask = mask(box);  // 用原始 mask 的 ROI
            cv::Scalar sMean = cv::mean(roiHSV, roiMask);  // (H,S,V)
            if (sMean[1] > params_.blobMaxSat) continue;

            float gm = float(cv::mean(grayF(box), roiMask)[0]);
            blobs.push_back({box, area, float(sMean[0]), solidity, gm});
        }
    }

    // ─── 决策: 梯度聚类优先, blob 独立兜底 ───
    Detection det;
    det.class_id   = 99;
    det.class_name = "WEAPONHEAD";
    det.confidence = 1.0f;
    bool found = false;

    // 策略A: 梯度聚类 (内置暗灰度+对比度, 无需额外验证)
    if (!rows.empty()) {
        struct Cluster { int t, b; std::vector<int> ls, rs; };
        std::vector<Cluster> clusters;
        auto mid = [](std::vector<int>& v) {
            if (v.empty()) return 0;
            std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
            return v[v.size()/2];
        };
        Cluster cur2;
        cur2.t = rows[0].y; cur2.b = rows[0].y;
        cur2.ls.push_back(rows[0].left); cur2.rs.push_back(rows[0].right);
        for (size_t i = 1; i < rows.size(); ++i) {
            int mL = mid(cur2.ls), mR = mid(cur2.rs);
            int gap = rows[i].y - rows[i-1].y;
            if (gap <= 2 && std::abs(rows[i].left-mL) <= params_.maxDrift
                       && std::abs(rows[i].right-mR) <= params_.maxDrift) {
                cur2.b = rows[i].y;
                cur2.ls.push_back(rows[i].left); cur2.rs.push_back(rows[i].right);
            } else {
                clusters.push_back(std::move(cur2));
                cur2 = Cluster{};
                cur2.t = rows[i].y; cur2.b = rows[i].y;
                cur2.ls.push_back(rows[i].left); cur2.rs.push_back(rows[i].right);
            }
        }
        clusters.push_back(std::move(cur2));
        Cluster* best = nullptr;
        int bestH = 0;
        for (auto& cl : clusters) {
            int ch = cl.b - cl.t + 1;
            if (ch > bestH) { bestH = ch; best = &cl; }
        }
        if (best && bestH >= params_.minHeight) {
            int fL = mid(best->ls), fR = mid(best->rs), fT = best->t, fB = best->b;
            det.corner_tl = cv::Point2f(float(fL), float(fT));
            det.corner_tr = cv::Point2f(float(fR), float(fT));
            det.corner_br = cv::Point2f(float(fR), float(fB));
            det.corner_bl = cv::Point2f(float(fL), float(fB));
            found = true;
        }
    }

    // 策略B: 梯度失败时, 用 blob 独立兜底 (选最靠近中心的合格 blob)
    if (!found && !blobs.empty()) {
        Blob* bestBlob = nullptr;
        float bestScore = 1e9f;
        for (auto& b : blobs) {
            float bcx = b.box.x + b.box.width * 0.5f;
            float bcy = b.box.y + b.box.height * 0.5f;
            float dist = std::sqrt((bcx - cx) * (bcx - cx) + (bcy - cy) * (bcy - cy));
            float score = dist * 1.0f - b.area * 0.002f + b.grayMean * 0.5f;
            if (score < bestScore) { bestScore = score; bestBlob = &b; }
        }
        if (bestBlob) {
            det.corner_tl = cv::Point2f(float(bestBlob->box.x), float(bestBlob->box.y));
            det.corner_tr = cv::Point2f(float(bestBlob->box.x + bestBlob->box.width), float(bestBlob->box.y));
            det.corner_br = cv::Point2f(float(bestBlob->box.x + bestBlob->box.width), float(bestBlob->box.y + bestBlob->box.height));
            det.corner_bl = cv::Point2f(float(bestBlob->box.x), float(bestBlob->box.y + bestBlob->box.height));
            found = true;
        }
    }

    if (found) result.detections.push_back(det);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.inference_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}
