#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

// ============================================================
// 前向声明 (避免对 kfs_core 的循环依赖)
// ============================================================
namespace kfs {
struct ModelConfig;
}

// ============================================================
// 检测结果结构体
// ============================================================

/**
 * @brief 单个检测框信息
 *
 * 角点 (corner points) 以像素坐标给出:
 *   (x1,y1) — 左上角
 *   (x2,y2) — 右上角
 *   (x3,y3) — 右下角
 *   (x4,y4) — 左下角
 */
struct Detection {
    int class_id;           // 类别 ID: 0=R1, 1=T, 2=F
    std::string class_name; // 类别名: "R1", "T", "F"
    float confidence;       // 置信度 [0, 1]

    // 4 个角点 (像素坐标, YOLO/ONNX 模型空间的 xywh 转回原图)
    cv::Point2f corner_tl;  // 左上 Top-Left
    cv::Point2f corner_tr;  // 右上 Top-Right
    cv::Point2f corner_br;  // 右下 Bottom-Right
    cv::Point2f corner_bl;  // 左下 Bottom-Left

    // 便捷接口
    cv::Rect2f bbox() const {
        return cv::Rect2f(corner_tl.x, corner_tl.y,
                          corner_br.x - corner_tl.x,
                          corner_br.y - corner_tl.y);
    }
};

/**
 * @brief 单帧检测结果
 */
struct FrameResult {
    std::vector<Detection> detections;  // 所有检测框
    cv::Size frame_size;                // 原始帧尺寸 (1920×1080)
    double inference_ms;                // 推理耗时 (毫秒)
    cv::Mat debug_image;                // 可选调试图像 (weaponhead 二值化等)
};

// ============================================================
// YOLO11 ONNX 检测器
// ============================================================

class YoloDetector {
public:
    /**
     * @brief 从原始参数构造
     * @param model_path   ONNX 模型路径
     * @param class_names  类别名列表 (长度 = 模型输出类别数)
     * @param input_size   模型输入尺寸 (默认 640)
     * @param conf_thresh  置信度阈值 (默认 0.25)
     * @param iou_thresh   NMS IOU 阈值 (默认 0.45)
     * @param use_cuda     是否尝试 CUDA (默认 true, 失败回退 CPU)
     */
    YoloDetector(const std::string& model_path,
                 const std::vector<std::string>& class_names = {"R1", "T", "F"},
                 int   input_size  = 640,
                 float conf_thresh = 0.25f,
                 float iou_thresh  = 0.45f,
                 bool  use_cuda    = true);

    /**
     * @brief 从 ModelConfig 构造 (推荐, 用于 ROS2 集成)
     */
    explicit YoloDetector(const kfs::ModelConfig& cfg);

    ~YoloDetector();

    // 不可拷贝
    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;

    /**
     * @brief 对 BGR 图像执行推理
     * @param frame  输入图像 (BGR, 任意尺寸)
     * @return       检测结果 (含 4 角点)
     */
    FrameResult detect(const cv::Mat& frame);

    /// 返回类别名 (构造时设定)
    const std::vector<std::string>& classNames() const;

    /// 类别数量
    int numClasses() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};
