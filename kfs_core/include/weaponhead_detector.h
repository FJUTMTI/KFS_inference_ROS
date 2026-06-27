#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "yolo_detector.h"   // 复用 Detection / FrameResult 结构 (松耦合, 仅为结果类型)

/**
 * @brief Weaponhead 传统 OpenCV 目标识别器
 *
 * 双路融合算法:
 *   A路 — 逐行梯度扫描 (模糊→梯度峰→暗灰度→边缘对比→行聚类)
 *   B路 — 二值图 blob 检测 (灰度阈值→连通域→形状/颜色验证)
 *   最终: A路结果用B路特征验证; A路失败时B路独立输出
 *
 * - 不依赖 ONNX / YOLO
 * - 返回结果兼容 FrameResult / Detection, class_name="WEAPONHEAD"
 */
class WeaponheadDetector {
public:
    /**
     * @brief 可调参数 (可选, 目前内部使用合理默认)
     */
    struct Params {
        // === A路: 梯度扫描 ===
        int   blurKernel    = 21;     // 高斯模糊核
        float gradRatio     = 0.35f;  // 梯度阈值比例
        int   searchBandV   = 140;    // 垂直搜索范围
        int   minWidth      = 16;     // 最小行宽度
        int   maxWidth      = 300;    // 最大行宽度
        int   darkMaxGray   = 100;    // 行内平均灰度上限
        float contrastRatio = 1.15f;  // 边缘外/内灰度比
        int   minHeight     = 14;     // 最小有效行数
        int   maxDrift      = 35;     // 行聚类最大漂移

        // === B路: blob 验证 ===
        int   blobGrayThr    = 50;    // 二值化阈值 (gray < thr → 暗区)
        int   blobMinArea    = 500;   // blob 最小面积
        float blobMaxSat     = 80.0f; // blob 饱和度上限 (排除彩色噪声)
        float blobMinSolidity = 0.80f; // blob 凸包紧密度下限
    };

    WeaponheadDetector() : WeaponheadDetector(Params{}) {}
    explicit WeaponheadDetector(const Params& params);

    // 不可拷贝 (同 YoloDetector 风格)
    WeaponheadDetector(const WeaponheadDetector&) = delete;
    WeaponheadDetector& operator=(const WeaponheadDetector&) = delete;

    /**
     * @brief 对 BGR 图像执行传统 CV 检测
     * @return 结果中通常包含 0 或 1 个 Detection (class_name = "WEAPONHEAD", conf=1.0)
     *         左右边界像素位于 corner_tl.x 和 corner_br.x
     */
    FrameResult detect(const cv::Mat& frame);

private:
    Params params_;
};
