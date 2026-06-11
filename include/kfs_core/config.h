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

// ============================================================
// 全局配置 (YAML 映射)
// ============================================================

struct Config {
    std::string          cameraType = "usb";   // "usb" | "realsense"
    USBConfig            usb;
    CameraControlsConfig controls;
    ModelConfig          model;
    DisplayConfig        display;
};

// ============================================================
// YAML 加载
// ============================================================

/// 从 YAML 文件加载配置, 解析失败抛出 std::exception
Config loadConfig(const std::string& yamlPath);

} // namespace kfs
