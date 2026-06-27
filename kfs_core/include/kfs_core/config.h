#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace kfs {

// ============================================================
// 相机配置
// ============================================================

/// USB 相机子配置
struct USBConfig {
    int         device = 0;
    int         width  = 1280;
    int         height = 1024;
    int         fps    = 30;
    std::string fourcc = "";       // "MJPG", "YUYV", 或空(自动)
    std::string calibration_file = "";  // e.g. "config/ost.yaml" (ROS ost.yaml / camera_info yaml 格式)
    bool        undistort = false;      // 是否对 getFrame() 返回的图像做畸变矫正 (需要 calibration_file)
};

/// V4L2 控制参数 (值 < 0 表示不设置)
struct CameraControlsConfig {
    int autoExposure = -1;         // -1=自动, 0=自动光圈优先, 1=手动
    int exposure     = -1;         // 手动曝光值
    int gain         = -1;
    int brightness   = -1;
    int contrast     = -1;
    int saturation   = -1;
    int whiteBalance = -1;         // -1=自动
    int sharpness    = -1;
};

/// 模型推理配置
struct ModelConfig {
    std::string path         = "models/kfs_yolo11_3class.onnx";
    int         inputSize    = 640;
    float       confThresh   = 0.25f;
    float       iouThresh    = 0.30f;
    bool        useCUDA      = true;    // 是否尝试 CUDA (失败回退 CPU)
    std::vector<std::string> classNames = {"R1", "T", "F"};
};

/// 显示配置
struct DisplayConfig {
    bool debug = true;
};

/// weaponhead_detector 参数
struct WeaponheadConfig {
    // A路: 梯度扫描
    int   blurKernel    = 21;
    float gradRatio     = 0.35f;
    int   searchBandV   = 140;
    int   minWidth      = 16;
    int   maxWidth      = 300;
    int   darkMaxGray   = 100;
    float contrastRatio = 1.15f;
    int   minHeight     = 14;
    int   maxDrift      = 35;
    // B路: blob 验证
    int   blobGrayThr   = 50;
    int   blobMinArea   = 500;
    float blobMaxSat    = 80.0f;
    float blobSolidity  = 0.80f;  // blob 紧密度下限
};

// ============================================================
// 全局配置 (YAML 映射)
// ============================================================

struct Config {
    std::string          cameraType = "usb";   // "usb" | "realsense"
    USBConfig            usb;
    CameraControlsConfig controls;
    ModelConfig          model;
    DisplayConfig        display;

    // 检测器类型 (保留向后, 默认 yolo)
    std::string          detectorType = "yolo";

    // 松耦合添加: weaponhead_detector (传统 OpenCV 虚焦物体左右边界检测)
    // 当为 true 时, 与 YOLO 并行运行, 结果合并到同一帧 detections 中 (不同目标)
    // 主要用于 USB 相机场景, 识别 assets/ 图片示例中的中央虚焦 weaponhead 轮廓
    bool                 enableWeaponheadDetector = false;

    // weaponhead_detector 可调参数 (支持 YAML 配置)
    WeaponheadConfig     weaponhead;
};

// ============================================================
// YAML 加载
// ============================================================

/// 从 YAML 文件加载配置, 解析失败抛出 std::exception
Config loadConfig(const std::string& yamlPath);

} // namespace kfs
