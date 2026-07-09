#include "lightbar_detector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>

// ============================================================
// 构造 / 工具
// ============================================================

LightbarDetector::LightbarDetector(const kfs::LightbarConfig& cfg)
    : cfg_(cfg)
{
    std::cout << "[LIGHTBAR] 检测器已初始化"
              << " | thr=[" << cfg_.coreThreshMin << "," << cfg_.coreThreshMax << "]"
              << " min_len=" << cfg_.minLength
              << " min_aspect=" << cfg_.minAspect
              << " max_short=" << cfg_.maxShortSide
              << std::endl;
}

const std::vector<std::string>& LightbarDetector::colorNames() {
    static const std::vector<std::string> kNames = {
        "LIGHTBAR_RED", "LIGHTBAR_GREEN", "LIGHTBAR_BLUE", "LIGHTBAR_YELLOW"
    };
    return kNames;
}

float LightbarDetector::longAxisAngleDeg(float width, float height, float angle) {
    // OpenCV: angle 是 width 边相对水平的角度 ∈ [-90, 0)
    float ang = angle;
    if (width < height) {
        ang += 90.f;  // 长轴方向
    }
    while (ang <= -90.f) ang += 180.f;
    while (ang >   90.f) ang -= 180.f;
    return ang;
}

cv::Vec2f LightbarDetector::unitFromAngle(float deg) {
    const float r = deg * static_cast<float>(CV_PI) / 180.f;
    return cv::Vec2f(std::cos(r), std::sin(r));
}

cv::Mat LightbarDetector::multiOrientClose(const cv::Mat& mask, int klen) {
    klen = std::max(3, klen | 1);
    cv::Mat closed = cv::Mat::zeros(mask.size(), CV_8UC1);
    const int mid = klen / 2;

    auto apply = [&](const cv::Mat& kernel) {
        cv::Mat tmp;
        cv::morphologyEx(mask, tmp, cv::MORPH_CLOSE, kernel);
        cv::max(closed, tmp, closed);
    };

    // 0°
    {
        cv::Mat k = cv::Mat::zeros(klen, klen, CV_8UC1);
        k.row(mid).setTo(1);
        apply(k);
    }
    // 90°
    {
        cv::Mat k = cv::Mat::zeros(klen, klen, CV_8UC1);
        k.col(mid).setTo(1);
        apply(k);
    }
    // 45°
    {
        cv::Mat k = cv::Mat::zeros(klen, klen, CV_8UC1);
        for (int i = 0; i < klen; ++i) k.at<uchar>(i, i) = 1;
        apply(k);
    }
    // 135°
    {
        cv::Mat k = cv::Mat::zeros(klen, klen, CV_8UC1);
        for (int i = 0; i < klen; ++i) k.at<uchar>(i, klen - 1 - i) = 1;
        apply(k);
    }
    return closed;
}

// ============================================================
// 核心提取: 自适应阈值 + 多方向形态学
// ============================================================

cv::Mat LightbarDetector::extractCoreMask(const cv::Mat& gray, int* thr_out) const {
    CV_Assert(gray.type() == CV_8UC1);
    const int H = gray.rows, W = gray.cols;
    const int total = H * W;

    // 高分位估计饱和亮部
    std::vector<uchar> buf(gray.ptr<uchar>(), gray.ptr<uchar>() + total);
    const float pct = std::min(99.9f, std::max(99.0f, cfg_.corePercentile));
    const size_t idx = static_cast<size_t>(
        std::min(total - 1, std::max(0, static_cast<int>(total * (pct / 100.f)))));
    std::nth_element(buf.begin(), buf.begin() + static_cast<long>(idx), buf.end());
    const float p = static_cast<float>(buf[idx]);

    // 起始阈值偏保守 (避免一上来就切碎 green 类灯条);
    // 仅当核心过胖 (黄灯 bloom) 时抬高阈值
    int thr = static_cast<int>(std::max(
        static_cast<float>(cfg_.coreThreshMin),
        std::min(static_cast<float>(cfg_.coreThreshMax),
                 p * cfg_.coreThreshScale)));

    const int ok = std::max(3, cfg_.openKsize | 1);
    cv::Mat k_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ok, ok));
    const int close_len = std::max(7, cfg_.closeLength | 1);

    cv::Mat best_core;
    int best_thr = thr;
    float best_score = -1e9f;

    // 搜索 thr: 细长段优先。黄灯 bloom 需要较高 thr (≈245–252),
    // 步长 2 并强制扫到 thr_hi, 避免跳过最佳区间。
    const int thr_hi = std::min(252, std::max(thr + 30, cfg_.coreThreshMax + 25));
    bool saw_fat = false;

    for (int t = thr; t <= thr_hi; t += 2) {
        cv::Mat core;
        cv::threshold(gray, core, t, 255, cv::THRESH_BINARY);
        cv::morphologyEx(core, core, cv::MORPH_OPEN, k_open);
        core = multiOrientClose(core, close_len);
        cv::morphologyEx(core, core, cv::MORPH_OPEN, k_open);

        const float frac = static_cast<float>(cv::countNonZero(core)) / static_cast<float>(total);
        if (frac > cfg_.maxCoreFrac * 1.2f) saw_fat = true;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(core, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 只统计「合格细长段」(与 findSegments 一致), 避免胖 bloom 虚高分
        int n_thin = 0;
        float max_short_thin = 0.f;
        float sum_long_thin = 0.f;
        float best_two = 0.f;
        std::vector<float> longs;
        longs.reserve(8);

        for (const auto& c : contours) {
            if (cv::contourArea(c) < cfg_.minArea) continue;
            const cv::RotatedRect r = cv::minAreaRect(c);
            const float ls = std::max(r.size.width, r.size.height);
            const float ss = std::min(r.size.width, r.size.height);
            if (ls < cfg_.minLength) continue;
            if (ls / std::max(1.f, ss) < cfg_.minAspect) continue;
            if (ss > cfg_.maxShortSide) continue;  // 拒绝过胖
            ++n_thin;
            max_short_thin = std::max(max_short_thin, ss);
            sum_long_thin += ls;
            longs.push_back(ls);
        }
        std::sort(longs.begin(), longs.end(), std::greater<float>());
        if (longs.size() >= 2) {
            best_two = longs[0] + longs[1];
        }

        float sc = -1e6f;
        if (n_thin >= 2) {
            // 奖励两段总长 (真实灯条通常 >400px), 惩罚短边与面积
            sc = 5.f * best_two + 200.f * n_thin
               - 6.f * max_short_thin
               - 15000.f * frac;
            // 长度比接近也更好 (用 top2)
            if (longs.size() >= 2) {
                const float lr = longs[1] / longs[0];
                sc += 300.f * lr;
            }
        } else if (n_thin == 1) {
            sc = sum_long_thin - 1000.f;
        }

        if (sc > best_score) {
            best_score = sc;
            best_core  = core.clone();
            best_thr   = t;
        }

        // 已得到两段够长且够细的结果: 若曾见过胖 bloom, 继续搜更高 thr;
        // 否则可提前结束。
        if (n_thin >= 2 && best_two > 700.f && max_short_thin <= cfg_.maxShortSide * 0.9f
            && frac <= cfg_.maxCoreFrac) {
            if (!saw_fat || t >= 246) break;
        }
    }

    if (best_core.empty()) {
        cv::threshold(gray, best_core, thr, 255, cv::THRESH_BINARY);
        cv::morphologyEx(best_core, best_core, cv::MORPH_OPEN, k_open);
        best_core = multiOrientClose(best_core, close_len);
        best_thr = thr;
    }

    if (thr_out) *thr_out = best_thr;
    return best_core;
}

// ============================================================
// 细长段
// ============================================================

std::vector<LightbarDetector::Segment>
LightbarDetector::findSegments(const cv::Mat& mask) const {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Segment> segs;
    segs.reserve(contours.size());

    for (const auto& c : contours) {
        const double area = cv::contourArea(c);
        if (area < cfg_.minArea) continue;

        const cv::RotatedRect rect = cv::minAreaRect(c);
        const float ww = rect.size.width;
        const float hh = rect.size.height;
        const float long_s  = std::max(ww, hh);
        const float short_s = std::max(1.f, std::min(ww, hh));
        const float aspect  = long_s / short_s;

        if (long_s < cfg_.minLength || aspect < cfg_.minAspect) continue;
        // 拒绝过胖 bloom 块
        if (short_s > cfg_.maxShortSide) continue;

        Segment s;
        s.center     = rect.center;
        s.long_side  = long_s;
        s.short_side = short_s;
        s.aspect     = aspect;
        s.area       = static_cast<float>(area);
        s.long_angle = longAxisAngleDeg(ww, hh, rect.angle);
        s.rect       = rect;
        segs.push_back(s);
    }

    std::sort(segs.begin(), segs.end(),
              [](const Segment& a, const Segment& b) {
                  return a.long_side > b.long_side;
              });
    return segs;
}

// ============================================================
// 等长共线配对 (旋转方向投影)
// ============================================================

bool LightbarDetector::findBestPair(const std::vector<Segment>& segs,
                                    int img_w, int img_h,
                                    PairResult& out) const {
    const int n = static_cast<int>(std::min<size_t>(segs.size(), 30));
    float best_sc = -1.f;
    PairResult best;

    auto angDiff = [](float x, float y) {
        float d = std::abs(x - y);
        return std::min(d, 180.f - d);
    };

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const Segment& a = segs[i];
            const Segment& b = segs[j];

            const float lr = std::min(a.long_side, b.long_side)
                           / std::max(a.long_side, b.long_side);
            if (lr < cfg_.minLengthRatio) continue;

            // 两段自身朝向一致
            const float seg_ang_diff = angDiff(a.long_angle, b.long_angle);
            if (seg_ang_diff > cfg_.maxAngleDiff) continue;

            // 平均长轴单位向量
            cv::Vec2f u1 = unitFromAngle(a.long_angle);
            cv::Vec2f u2 = unitFromAngle(b.long_angle);
            if (u1.dot(u2) < 0.f) u2 = -u2;
            cv::Vec2f u = u1 + u2;
            const float nu = std::sqrt(u.dot(u));
            if (nu < 1e-6f) continue;
            u *= (1.f / nu);

            const cv::Vec2f d(b.center.x - a.center.x, b.center.y - a.center.y);
            const float dist = std::sqrt(d.dot(d));
            if (dist < 50.f) continue;

            // 沿长轴 / 垂直长轴分解 —— 支持任意旋转的共线对
            const float along   = std::abs(d.dot(u));
            const float lateral = std::abs(d[0] * u[1] - d[1] * u[0]);

            const float max_lat = (a.short_side + b.short_side) * 1.2f + 20.f;
            if (lateral > max_lat) continue;
            if (along < std::max(a.long_side, b.long_side) * 0.40f) continue;
            // 中心连线应主要沿长轴 (共线)
            if (along / (dist + 1e-6f) < 0.88f) continue;

            const float mid_x = (a.center.x + b.center.x) * 0.5f;
            const float mid_y = (a.center.y + b.center.y) * 0.5f;
            const float cx_pen = std::abs(mid_x - img_w * 0.5f) / (img_w * 0.5f);
            const float cy_pen = std::abs(mid_y - img_h * 0.5f) / (img_h * 0.5f);
            if (cx_pen > 0.65f || cy_pen > 0.60f) continue;

            float sc = lr * (a.long_side + b.long_side) * std::sqrt(a.aspect * b.aspect);
            sc /= (1.f + lateral / 4.f);
            sc /= (1.f + seg_ang_diff / 4.f);
            sc *= (1.25f - cx_pen) * (1.25f - cy_pen);
            sc *= (1.f + std::min(a.long_side, b.long_side) / 350.f);

            if (sc > best_sc) {
                best_sc = sc;
                best.a = a;
                best.b = b;
                best.score = sc;
                best.lateral = lateral;
                best.along = along;

                // 合并旋转矩形
                std::vector<cv::Point2f> pts;
                pts.reserve(8);
                cv::Point2f box[4];
                a.rect.points(box);
                for (int k = 0; k < 4; ++k) pts.push_back(box[k]);
                b.rect.points(box);
                for (int k = 0; k < 4; ++k) pts.push_back(box[k]);
                best.union_rect = cv::minAreaRect(pts);
            }
        }
    }

    if (best_sc < 0.f) return false;
    out = best;
    return true;
}

// ============================================================
// 颜色: 外环 HSV + 通道差
// ============================================================

bool LightbarDetector::classifyColor(const cv::Mat& bgr,
                                     const cv::Mat& gray,
                                     const cv::Mat& core_mask,
                                     const PairResult& pair,
                                     std::string& color_name,
                                     float& confidence,
                                     std::map<std::string, float>* scores_out) const {
    CV_Assert(bgr.type() == CV_8UC3);
    const int H = bgr.rows, W = bgr.cols;

    cv::Mat region = cv::Mat::zeros(H, W, CV_8UC1);
    auto fillSeg = [&](const Segment& s) {
        cv::Point2f pts[4];
        s.rect.points(pts);
        std::vector<cv::Point> poly = {
            cv::Point(cvRound(pts[0].x), cvRound(pts[0].y)),
            cv::Point(cvRound(pts[1].x), cvRound(pts[1].y)),
            cv::Point(cvRound(pts[2].x), cvRound(pts[2].y)),
            cv::Point(cvRound(pts[3].x), cvRound(pts[3].y)),
        };
        cv::fillConvexPoly(region, poly, 255);
    };
    fillSeg(pair.a);
    fillSeg(pair.b);

    // 外环: 大膨胀 - 小膨胀, 取有色 bloom (避开过曝灯芯)
    const int outer_k = std::max(21, cfg_.bloomKsize | 1);
    const int inner_k = std::max(5, std::min(outer_k / 3, 15) | 1);
    cv::Mat k_outer = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(outer_k, outer_k));
    cv::Mat k_inner = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(inner_k, inner_k));
    cv::Mat outer, inner, ring;
    cv::dilate(region, outer, k_outer);
    cv::dilate(region, inner, k_inner);
    cv::subtract(outer, inner, ring);

    // 也合并稍外的非 core 区域, 提高样本数
    cv::Mat not_core;
    cv::bitwise_not(core_mask, not_core);
    cv::Mat bloom;
    cv::bitwise_and(outer, not_core, bloom);
    cv::Mat bright;
    cv::threshold(gray, bright, 40, 255, cv::THRESH_BINARY);
    cv::bitwise_and(bloom, bright, bloom);
    // 优先用 ring∩bloom, 不够再退到 bloom
    cv::Mat sample;
    cv::bitwise_and(ring, bloom, sample);
    if (cv::countNonZero(sample) < 80) {
        sample = bloom;
    }
    if (cv::countNonZero(sample) < 40) {
        sample = outer;
    }

    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    double sum_red = 0, sum_green = 0, sum_blue = 0, sum_yellow = 0;
    double sum_r = 0, sum_g = 0, sum_b = 0;
    double sum_h = 0, sum_s = 0;
    int count = 0;
    int hue_bins[18] = {0};  // 0..179 → 18 bins of 10

    for (int y = 0; y < H; ++y) {
        const uchar* sm = sample.ptr<uchar>(y);
        const cv::Vec3b* brow = bgr.ptr<cv::Vec3b>(y);
        const cv::Vec3b* hrow = hsv.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x) {
            if (!sm[x]) continue;
            const int bb = brow[x][0];
            const int gg = brow[x][1];
            const int rr = brow[x][2];
            const int hh = hrow[x][0];
            const int ss = hrow[x][1];
            const int vv = hrow[x][2];

            sum_r += rr; sum_g += gg; sum_b += bb;
            sum_red    += std::max(0, rr - std::max(gg, bb));
            sum_green  += std::max(0, gg - std::max(rr, bb));
            sum_blue   += std::max(0, bb - std::max(rr, gg));
            sum_yellow += std::max(0, std::min(rr, gg) - bb);
            ++count;

            // HSV 仅统计有饱和度的像素 (过曝白芯无效)
            if (ss > 40 && vv > 50) {
                sum_h += hh;
                sum_s += ss;
                hue_bins[std::min(17, hh / 10)]++;
            }
        }
    }

    if (count < 10) {
        color_name = "LIGHTBAR_UNKNOWN";
        confidence = 0.f;
        return false;
    }

    const float inv = 1.f / static_cast<float>(count);
    float red_s    = static_cast<float>(sum_red)    * inv;
    float green_s  = static_cast<float>(sum_green)  * inv;
    float blue_s   = static_cast<float>(sum_blue)   * inv;
    float yellow_s = static_cast<float>(sum_yellow) * inv;
    const float mean_r = static_cast<float>(sum_r) * inv;
    const float mean_g = static_cast<float>(sum_g) * inv;
    const float mean_b = static_cast<float>(sum_b) * inv;

    // HSV 峰值色相
    int peak_bin = 0, peak_cnt = 0, sat_cnt = 0;
    for (int i = 0; i < 18; ++i) {
        sat_cnt += hue_bins[i];
        if (hue_bins[i] > peak_cnt) {
            peak_cnt = hue_bins[i];
            peak_bin = i;
        }
    }
    const float peak_h = peak_bin * 10.f + 5.f;  // bin 中心

    // 通道差消歧
    if (yellow_s > 18.f && std::abs(mean_r - mean_g) < 30.f
        && mean_r > mean_b + 25.f && mean_g > mean_b + 25.f) {
        yellow_s *= 1.8f;
        red_s    *= 0.35f;
    }
    if (red_s > 18.f && mean_r > mean_g + 25.f) {
        red_s    *= 1.4f;
        yellow_s *= 0.30f;
    }
    // 蓝灯近芯常呈青绿 (G≈B 高): 给 blue 补偿
    if (mean_b > 150.f && mean_g > 150.f && mean_r < 120.f
        && std::abs(mean_b - mean_g) < 40.f) {
        blue_s = std::max(blue_s, 25.f);
        green_s *= 0.4f;
    }

    // HSV 投票加成 (比通道差更稳, 尤其蓝/青)
    if (sat_cnt > 50) {
        // OpenCV H: R≈0/170, Y≈20-35, G≈40-85, cyan/blue≈85-130
        if (peak_h <= 12.f || peak_h >= 160.f) {
            red_s += 40.f;
        } else if (peak_h >= 15.f && peak_h < 40.f) {
            yellow_s += 40.f;
        } else if (peak_h >= 40.f && peak_h < 88.f) {
            green_s += 40.f;
        } else if (peak_h >= 88.f && peak_h < 140.f) {
            blue_s += 40.f;  // 含青色 bloom
        }
    }

    std::map<std::string, float> scores = {
        {"LIGHTBAR_RED",    red_s},
        {"LIGHTBAR_GREEN",  green_s},
        {"LIGHTBAR_BLUE",   blue_s},
        {"LIGHTBAR_YELLOW", yellow_s},
    };
    if (scores_out) *scores_out = scores;

    auto best_it = std::max_element(
        scores.begin(), scores.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    if (best_it->second < cfg_.minColorScore) {
        color_name = "LIGHTBAR_UNKNOWN";
        confidence = 0.f;
        return false;
    }

    color_name = best_it->first;

    float second = 0.f;
    for (const auto& kv : scores) {
        if (kv.first == color_name) continue;
        second = std::max(second, kv.second);
    }
    const float top = best_it->second;
    confidence = std::min(1.f, top / (top + second + 1e-3f));
    confidence = std::min(1.f, confidence * (0.55f + 0.45f * std::min(1.f, top / 50.f)));
    return true;
}

// ============================================================
// Detection: 旋转矩形四角
// ============================================================

Detection LightbarDetector::toDetection(const PairResult& pair,
                                        const std::string& color_name,
                                        float confidence) const {
    cv::Point2f pts[4];
    pair.union_rect.points(pts);

    // 按 y 再 x 排序找近似 tl, 然后按角度绕序
    int tl_i = 0;
    for (int i = 1; i < 4; ++i) {
        if (pts[i].y < pts[tl_i].y - 1e-3f
            || (std::abs(pts[i].y - pts[tl_i].y) < 1e-3f && pts[i].x < pts[tl_i].x)) {
            tl_i = i;
        }
    }
    // OpenCV boxPoints 顺序为连续四边, 从 tl_i 起沿环取
    // 保证顺时针/逆时针一致即可
    Detection det;
    if (color_name == "LIGHTBAR_RED")         det.class_id = 10;
    else if (color_name == "LIGHTBAR_GREEN")  det.class_id = 11;
    else if (color_name == "LIGHTBAR_BLUE")   det.class_id = 12;
    else if (color_name == "LIGHTBAR_YELLOW") det.class_id = 13;
    else                                      det.class_id = 19;

    det.class_name = color_name;
    det.confidence = confidence;

    // 使用旋转矩形四点: 0→1→2→3 环序映射到 tl,tr,br,bl (相对)
    det.corner_tl = pts[tl_i];
    det.corner_tr = pts[(tl_i + 1) % 4];
    det.corner_br = pts[(tl_i + 2) % 4];
    det.corner_bl = pts[(tl_i + 3) % 4];

    // 若 (tl→tr) 与 (tl→bl) 叉积方向不对, 交换 tr/bl 以保持一致绕向
    const cv::Point2f v1 = det.corner_tr - det.corner_tl;
    const cv::Point2f v2 = det.corner_bl - det.corner_tl;
    if (v1.x * v2.y - v1.y * v2.x > 0) {
        // 逆时针, 交换使 tr 为另一邻点
        std::swap(det.corner_tr, det.corner_bl);
    }
    return det;
}

// ============================================================
// 主入口
// ============================================================

FrameResult LightbarDetector::detect(const cv::Mat& frame) {
    FrameResult result;
    result.frame_size = frame.empty() ? cv::Size() : frame.size();
    result.inference_ms = 0.0;

    if (frame.empty() || frame.channels() != 3) {
        return result;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat bgr = frame.isContinuous() ? frame : frame.clone();
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    int thr_used = 0;
    cv::Mat core = extractCoreMask(gray, &thr_used);
    auto segs = findSegments(core);

    PairResult pair;
    const bool found = findBestPair(segs, bgr.cols, bgr.rows, pair);

    if (found) {
        std::string color;
        float conf = 0.f;
        if (classifyColor(bgr, gray, core, pair, color, conf, nullptr)) {
            result.detections.push_back(toDetection(pair, color, conf));
        } else {
            result.detections.push_back(toDetection(pair, "LIGHTBAR_UNKNOWN", conf));
        }

        if (cfg_.saveDebugMask) {
            cv::Mat dbg;
            cv::cvtColor(core, dbg, cv::COLOR_GRAY2BGR);
            auto drawSeg = [&](const Segment& s, const cv::Scalar& col) {
                cv::Point2f pts[4];
                s.rect.points(pts);
                for (int i = 0; i < 4; ++i) {
                    cv::line(dbg, pts[i], pts[(i + 1) % 4], col, 2);
                }
            };
            drawSeg(pair.a, cv::Scalar(0, 255, 0));
            drawSeg(pair.b, cv::Scalar(0, 255, 255));
            // 合并旋转矩形 (品红)
            cv::Point2f upts[4];
            pair.union_rect.points(upts);
            for (int i = 0; i < 4; ++i) {
                cv::line(dbg, upts[i], upts[(i + 1) % 4], cv::Scalar(255, 0, 255), 2);
            }
            // 共线中心连线
            cv::line(dbg, pair.a.center, pair.b.center, cv::Scalar(255, 0, 255), 2);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "thr=%d lat=%.1f", thr_used, pair.lateral);
            cv::putText(dbg, buf, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX,
                        1.0, cv::Scalar(0, 255, 255), 2);
            result.debug_image = dbg;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.inference_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}
