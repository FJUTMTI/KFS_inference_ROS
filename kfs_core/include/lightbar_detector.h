#pragma once

#include "yolo_detector.h"  // Detection / FrameResult
#include "kfs_core/config.h"

#include <opencv2/core.hpp>
#include <map>
#include <string>
#include <vector>

// ============================================================
// 灯条检测器 (传统 OpenCV)
//
// 场景特征: 图像中部两条等长、共线的 LED 灯条 (可为任意旋转角度),
// 颜色 red / green / blue / yellow。
//
// 流程:
//   1) 自适应高亮灯芯阈值 (过胖 bloom 自动抬阈值)
//   2) 多方向形态学闭运算 + 开运算, 得到干净细长段
//   3) 按长轴方向投影判定等长共线对 (支持旋转矩形)
//   4) 外环 HSV + 通道差判定颜色
// ============================================================

class LightbarDetector {
public:
    explicit LightbarDetector(const kfs::LightbarConfig& cfg = {});
    ~LightbarDetector() = default;

    FrameResult detect(const cv::Mat& frame);

    const kfs::LightbarConfig& config() const { return cfg_; }

    static const std::vector<std::string>& colorNames();

private:
    struct Segment {
        cv::Point2f center;
        float       long_side  = 0.f;
        float       short_side = 0.f;
        float       aspect     = 0.f;
        float       area       = 0.f;
        /// 长轴方向角 (度), 范围 (-90, 90], 0=水平, +90≈竖直 (图像 y 向下)
        float       long_angle = 0.f;
        cv::RotatedRect rect;
    };

    struct PairResult {
        Segment a, b;
        float   score   = 0.f;
        float   lateral = 0.f;
        float   along   = 0.f;
        cv::RotatedRect union_rect;  // 两段共线合并的旋转矩形
    };

    kfs::LightbarConfig cfg_;

    /// 多方向线状闭运算 (0/45/90/135°), 桥接 LED 点间隙且适配斜向灯条
    static cv::Mat multiOrientClose(const cv::Mat& mask, int klen);

    /// 长轴角度: OpenCV minAreaRect → (-90, 90]
    static float longAxisAngleDeg(float width, float height, float angle);

    static cv::Vec2f unitFromAngle(float deg);

    /// 提取高亮灯芯 (自适应阈值 + 形态学)
    cv::Mat extractCoreMask(const cv::Mat& gray, int* thr_out = nullptr) const;

    std::vector<Segment> findSegments(const cv::Mat& mask) const;

    bool findBestPair(const std::vector<Segment>& segs,
                      int img_w, int img_h,
                      PairResult& out) const;

    bool classifyColor(const cv::Mat& bgr,
                       const cv::Mat& gray,
                       const cv::Mat& core_mask,
                       const PairResult& pair,
                       std::string& color_name,
                       float& confidence,
                       std::map<std::string, float>* scores_out = nullptr) const;

    /// 输出旋转矩形四角 (按 boxPoints 顺序整理为 tl-tr-br-bl 近似)
    Detection toDetection(const PairResult& pair,
                          const std::string& color_name,
                          float confidence) const;
};
