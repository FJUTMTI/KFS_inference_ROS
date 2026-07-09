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

/// 视频文件输入配置 (用于 camera.type == "video" 调试模式, 代替 USB/RealSense)
struct VideoConfig {
    std::string path = "";           // 视频文件路径, e.g. "kfs_core/assets/001.mp4"
    bool        loop = true;         // 播放到末尾是否循环 (调试推荐 true)
    std::string calibration_file = "";  // 可选, 与 usb 同格式的 ost.yaml
    bool        undistort = false;      // 是否对帧做畸变矫正
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

/// weaponhead YOLO 模型配置 (与3class并行推理)
struct WeaponheadConfig {
    std::string whModelPath = "";  // 空=不使用
};

/// 灯条检测器配置 (传统 OpenCV 颜色通道 + 形态学)
struct LightbarConfig {
    bool  enabled         = false;

    // 高亮灯芯提取 (自适应搜索 thr, 过胖 bloom 自动抬高)
    float corePercentile  = 99.8f;  // 灰度分位数 (估饱和亮部)
    int   coreThreshMin   = 200;    // 搜索下界
    int   coreThreshMax   = 235;    // 起始上界 (可搜索到 ~252)
    float coreThreshScale = 0.90f;  // 起始 thr ≈ percentile * scale
    float maxCoreFrac     = 0.015f; // 核心占画面面积上限 (黄灯 bloom 约束)
    float maxShortSide    = 65.f;   // 段短边上限 (拒绝过胖连通域)

    // 形态学
    int   openKsize       = 3;      // 开运算核 (去噪)
    int   closeLength     = 15;     // 多方向线状闭运算长度 (桥接 LED 点)
    int   closeVertical   = 15;     // 兼容旧字段, 若 >0 可覆盖 closeLength

    // 细长段过滤
    float minLength       = 50.f;
    float minAspect       = 3.5f;
    float minArea         = 50.f;

    // 等长共线配对 (按长轴投影, 支持任意旋转)
    float minLengthRatio  = 0.55f;  // 两段长度比下限
    float maxAngleDiff    = 18.f;   // 两段长轴最大夹角 (度)

    // 颜色 bloom / 外环
    int   bloomKsize      = 41;     // 外环膨胀核
    float minColorScore   = 5.f;    // 颜色得分下限

    // 调试
    bool  saveDebugMask   = false;  // 在 FrameResult.debug_image 输出核心掩膜
};

// ============================================================
// 全局配置 (YAML 映射)
// ============================================================

struct Config {
    std::string          cameraType = "usb";   // "usb" | "realsense" | "video"
    USBConfig            usb;
    VideoConfig          video;
    CameraControlsConfig controls;
    ModelConfig          model;
    DisplayConfig        display;

    // 检测器类型: "yolo" | "lightbar" | "yolo_lightbar"
    // lightbar 时跳过 YOLO (仅灯条); yolo_lightbar 时两者并行
    std::string          detectorType = "yolo";

    // 松耦合添加: weaponhead_detector (传统 OpenCV 虚焦物体左右边界检测)
    // 当为 true 时, 与 YOLO 并行运行, 结果合并到同一帧 detections 中 (不同目标)
    // 主要用于 USB 相机场景, 识别 assets/ 图片示例中的中央虚焦 weaponhead 轮廓
    bool                 enableWeaponheadDetector = false;

    // weaponhead_detector 可调参数 (支持 YAML 配置)
    WeaponheadConfig     weaponhead;

    // 灯条检测器 (与 YOLO 可并行; detectorType=="lightbar" 时仅跑灯条)
    bool                 enableLightbarDetector = false;
    LightbarConfig       lightbar;
};

// ============================================================
// YAML 加载
// ============================================================

/// 从 YAML 文件加载配置, 解析失败抛出 std::exception
Config loadConfig(const std::string& yamlPath);

} // namespace kfs
