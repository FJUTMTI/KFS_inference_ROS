#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "yolo_detector.h"   // 复用 Detection / FrameResult 结构 (松耦合, 仅为结果类型)

/**
 * @brief Weaponhead 传统 OpenCV 目标识别器 (用于 USB 相机)
 *
 * 针对“虚焦” (defocus blur) 物体的简单轮廓检测示例。
 * 目标: 识别图像正中物体的大致轮廓, 输出其左右两侧边界像素。
 *
 * - 不依赖 ONNX / YOLO
 * - 完全独立于原有 YOLO 代码路径 (松耦合添加)
 * - 返回结果兼容 FrameResult / Detection, 使用 class_name="OBJ" 表示
 *   左右边界通过 corner_tl.x / corner_br.x 给出 (y 取中心附近带状表示物体高度范围)
 *
 * 示例图片验证: assets/01.png , assets/02.png
 *
 * 配置中 detector.type = "weaponhead_detector"
 */
class WeaponheadDetector {
public:
    /**
     * @brief 可调参数 (可选, 目前内部使用合理默认)
     */
    struct Params {
        int   blurKernel   = 21;     // 高斯模糊核, 越大越 tolerant to 虚焦
        float devRatio     = 0.22f;  // 中心行 profile 偏差阈值比例 (相对 max dev)
        int   minBandH     = 60;     // 物体表示高度 (中心上下各一半)
        int   minWidth     = 20;     // 最小报告宽度
    };

    WeaponheadDetector() : WeaponheadDetector(Params{}) {}
    explicit WeaponheadDetector(const Params& params);

    // 不可拷贝 (同 YoloDetector 风格)
    WeaponheadDetector(const WeaponheadDetector&) = delete;
    WeaponheadDetector& operator=(const WeaponheadDetector&) = delete;

    /**
     * @brief 对 BGR 图像执行传统 CV 检测
     * @return 结果中通常包含 0 或 1 个 Detection (class_name = "OBJ", conf=1.0)
     *         左右边界像素位于 corner_tl.x 和 corner_br.x
     */
    FrameResult detect(const cv::Mat& frame);

private:
    Params params_;
};
