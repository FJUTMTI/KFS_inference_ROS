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

    if (frame.empty()) {
        return result;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame.clone();
    }

    const int w = gray.cols;
    const int h = gray.rows;
    const int cx = w / 2;
    const int cy = h / 2;

    // 1. 重度高斯模糊以适应虚焦物体 (软轮廓)
    cv::Mat blurred;
    int k = std::max(3, params_.blurKernel | 1);  // 奇数
    cv::GaussianBlur(gray, blurred, cv::Size(k, k), 0);

    // 2. 取中心水平带多行平均 profile (鲁棒)
    const int band = 8;
    cv::Mat profile_row(1, w, CV_32F, cv::Scalar(0.0f));
    int valid_rows = 0;
    for (int dy = -band; dy <= band; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= h) continue;
        cv::Mat row;
        blurred.row(y).convertTo(row, CV_32F);
        profile_row += row;
        ++valid_rows;
    }
    if (valid_rows > 0) {
        profile_row /= static_cast<float>(valid_rows);
    } else {
        blurred.row(cy).convertTo(profile_row, CV_32F);
    }

    // 3. 计算中心值 + 偏差
    float cval = profile_row.at<float>(0, cx);
    cv::Mat diff;
    cv::absdiff(profile_row, cval, diff);

    // 轻微平滑偏差曲线 (1D)
    cv::Mat diff_sm;
    cv::GaussianBlur(diff, diff_sm, cv::Size(25, 1), 0);

    // 阈值: 取峰值一定比例, 或最小 10
    double minv, maxv;
    cv::minMaxLoc(diff_sm, &minv, &maxv);
    float thresh = std::max(10.0f, static_cast<float>(maxv) * params_.devRatio);
    if (thresh < 8.0f) thresh = 8.0f;

    // 4. 从中心向左扫描第一个超过阈值的点 (左边界)
    int left = 0;
    for (int x = cx; x >= 0; --x) {
        if (diff_sm.at<float>(0, x) > thresh) {
            left = x;
            break;
        }
    }

    // 5. 从中心向右扫描 (右边界)
    int right = w - 1;
    for (int x = cx; x < w; ++x) {
        if (diff_sm.at<float>(0, x) > thresh) {
            right = x;
            break;
        }
    }

    // 6. 宽度保护
    int width = right - left;
    if (width < params_.minWidth) {
        // 扩展到最小宽度, 保持中心
        int need = (params_.minWidth - width) / 2;
        left  = std::max(0, cx - (width / 2 + need));
        right = std::min(w - 1, cx + (width / 2 + need));
    }
    // 裁剪
    left  = std::max(0, left);
    right = std::min(w - 1, right);

    // 7. 垂直范围: 中心附近带状 (表示物体大致高度范围, 非精确上下边缘)
    int half_h = std::max(params_.minBandH / 2, 30);
    int top = std::max(0, cy - half_h);
    int bot = std::min(h - 1, cy + half_h);

    // 8. 组装 Detection (松耦合: 使用统一结构, 特殊 class 区分)
    Detection det;
    det.class_id   = 99;            // 特殊 ID, 避开 0/1/2
    det.class_name = "OBJ";
    det.confidence = 1.0f;

    // 用 tl.x / br.x 直接表达左右边界像素 (上/下 y 取带状)
    det.corner_tl = cv::Point2f(static_cast<float>(left),  static_cast<float>(top));
    det.corner_tr = cv::Point2f(static_cast<float>(right), static_cast<float>(top));
    det.corner_br = cv::Point2f(static_cast<float>(right), static_cast<float>(bot));
    det.corner_bl = cv::Point2f(static_cast<float>(left),  static_cast<float>(bot));

    result.detections.push_back(det);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.inference_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 调试输出 (仅在需要时)
    // std::cout << "[OpenCV] left=" << left << " right=" << right
    //           << " width=" << (right-left) << " ms=" << result.inference_ms << std::endl;

    return result;
}
