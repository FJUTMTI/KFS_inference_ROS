# KFS 视觉算法 — ROBCON 2026 武林探秘

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![C++17](https://img.shields.io/badge/C++-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-✓-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.19-4C4C4C.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **CURC ROBCON 2026 "武林探秘" 实时视觉算法系统**

本仓库包含 ROBCON 2026 比赛的两套核心视觉识别功能：

| 功能 | 模型 | 用途 |
|------|------|------|
| 🏷️ **功夫卷轴识别** | `kfs_yolo11_3class.onnx` | 识别赛场上的功夫卷轴类别 (R1 / T / F) |
| ⚔️ **武器头识别** | `kfs_weaponhead_v1.onnx` | 识别武器头顶端区域，输出左右边界像素坐标 |

两套模型通过 **双 YOLO 并行推理** 同时运行，每帧整合为统一结果。

---

## 特性

| 特性 | 说明 |
|------|------|
| 🎯 推理后端 | ONNX Runtime (CUDA / CPU 自动切换) |
| ⚡ 推理性能 | CUDA ≈ 8~15ms / CPU ≈ 25~50ms (640×640) |
| 🔀 双模型并行 | 3class 卷轴模型 + weaponhead 武器头模型同时推理，合并发布 |
| 📷 相机支持 | Intel RealSense D415/D435 / V4L2 USB 摄像头 / 本地视频文件 |
| 🤖 ROS2 接口 | 自定义 msg/srv + rqt_reconfigure 动态参数调节 |
| 🧩 模块化设计 | `libkfs_core` 共享库，可被任意 ROS2 节点链接 |
| 📐 畸变矫正 | 支持 USB/Video 相机标定 + 在线去畸变 |

---

## 检测类别

### 功夫卷轴 (3class YOLO)

| ID | 名称 | 涵盖范围 |
|----|------|----------|
| 0 | R1 | R_R1 + B_R1 |
| 1 | T | T_03 ~ T_17 |
| 2 | F | F_18 ~ F_32 |

### 武器头 (weaponhead YOLO)

| ID | 名称 | 说明 |
|----|------|------|
| 0 | weaponhead | 武器头顶端区域，结果中包含 `corner_tl.x` / `corner_br.x` 即左右边界像素 |

> 全部类别名可通过 YAML 配置或 ROS2 参数动态修改。

---

## 算法结构

```
┌─────────────────────────────────────────────────────┐
│                   kfs_infer_node                    │
│                                                     │
│  ┌─────────────┐  ┌──────────────────────────────┐  │
│  │ ICameraCapture│  │ frame                       │  │
│  │  · USBCapture │──▶│  ┌──────────────────────┐  │  │
│  │  · RealSense  │  │  │ YoloDetector (3class) │  │  │
│  │  · VideoFile  │  │  │  · kfs_yolo11_3class  │  │  │
│  └─────────────┘  │  │  · 卷轴 R1 / T / F     │  │  │
│                    │  └──────────┬─────────────┘  │  │
│                    │             │ detections[]    │  │
│                    │  ┌──────────▼─────────────┐  │  │
│                    │  │ YoloDetector (weaponhd)│  │  │
│                    │  │  · kfs_weaponhead_v1   │  │  │
│                    │  │  · 武器头 左右边界     │  │  │
│                    │  └──────────┬─────────────┘  │  │
│                    │             │ detections[]    │  │
│                    │  ┌──────────▼─────────────┐  │  │
│                    │  │    合并 → FrameResult   │  │  │
│                    │  └──────────┬─────────────┘  │  │
│                    └─────────────┼────────────────┘  │
│                                  │                    │
│  ┌───────────────────────────────▼────────────────┐  │
│  │              ROS2 通信接口                      │  │
│  │  [Pub]  ~/result       InferResults (数组)     │  │
│  │  [Pub]  ~/debug_image  sensor_msgs/Image       │  │
│  │  [Pub]  ~/status       std_msgs/String         │  │
│  │  [Sub]  ~/enable       std_msgs/Bool           │  │
│  │  [Srv]  ~/set_state    SetInferState           │  │
│  │  [Srv]  ~/trigger      std_srvs/Trigger        │  │
│  │  [Param]  rqt_reconfigure 动态参数             │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### ROS2 消息

| 话题/服务 | 类型 | 方向 | 说明 |
|-----------|------|------|------|
| `~/result` | `kfs_core/InferResults` | 发布 | 每帧所有检测结果（含 `inference_ms`、帧尺寸元信息），推荐下游订阅此话题 |
| `~/debug_image` | `sensor_msgs/Image` | 发布 | 标注后的调试画面（rqt/rviz 可视化） |
| `~/status` | `std_msgs/String` | 发布 | 运行状态文本 |
| `~/enable` | `std_msgs/Bool` | 订阅 | 控制是否执行推理 |
| `~/set_state` | `kfs_core/SetInferState` | 服务 | 显式启停推理 |
| `~/trigger` | `std_srvs/Trigger` | 服务 | 触发单次推理 |

---

## 依赖

### 系统依赖

| 依赖 | 版本 | 必需 | 安装命令 |
|------|------|------|----------|
| ROS2 Humble | — | ✅ | [官方安装指南](https://docs.ros.org/en/humble/Installation.html) |
| CMake | ≥ 3.16 | ✅ | `sudo apt install cmake` |
| GCC (C++17) | ≥ 8 | ✅ | `sudo apt install build-essential` |
| OpenCV | ≥ 4.x | ✅ | `sudo apt install libopencv-dev` |
| ONNX Runtime | ≥ 1.16 | ✅ | 见下方安装说明 |
| yaml-cpp | — | ✅ | `sudo apt install libyaml-cpp-dev` |
| librealsense2 | ≥ 2.50 | ❌ | `sudo apt install librealsense2-dev` |

> ❌ = 可选，仅 RealSense 相机需要。USB 摄像头无需此项。

### Python 依赖 (仅导出 ONNX 时需要)

```bash
pip install ultralytics onnx
```

---

## 一键安装

```bash
# 1. 系统依赖
sudo apt install -y cmake build-essential libopencv-dev libyaml-cpp-dev

# 2. ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz
tar xzf onnxruntime-linux-x64-1.19.2.tgz
echo 'export ONNXRUNTIME_DIR='"$(pwd)"'/onnxruntime-linux-x64-1.19.2' >> ~/.bashrc
source ~/.bashrc

# 3. (可选) RealSense D415 / D435
sudo apt install -y librealsense2-dev

# 4. ROS2 Humble (如未安装)
# 参见 https://docs.ros.org/en/humble/Installation.html
```

---

## 编译

### colcon (推荐，ROS2 原生)

```bash
cd /home/lee/realsense_inference
source /opt/ros/humble/setup.zsh
colcon build --packages-select kfs_core --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.zsh
```

### 独立 CMake (不需要 ROS2 环境，仅编译库和 CLI)

```bash
cd /home/lee/realsense_inference/kfs_core
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 产物:
#   libkfs_core_lib.so  — 核心共享库
#   kfs_detect          — CLI 测试程序
#   kfs_infer_node      — ROS2 节点 (需要 ROS2 环境)
```

---

## 运行

### ROS2 节点

```bash
# 启动推理节点
ros2 run kfs_core kfs_infer_node --ros-args \
  -p model_path:=/absolute/path/to/kfs_yolo11_3class.onnx \
  -p config_path:=/absolute/path/to/kfs_config.yaml

# 查看标注画面
rqt_image_view /kfs_infer_node/debug_image

# 动态调参
ros2 run rqt_reconfigure rqt_reconfigure

# 查看检测结果（一帧一个数组消息）
ros2 topic echo /kfs_infer_node/result --once

# 保存截图
ros2 service call /kfs_infer_node/snapshot std_srvs/srv/Trigger
```

### 推理控制

```bash
# 启停推理
ros2 service call /kfs_infer_node/set_state kfs_core/srv/SetInferState "{enable: true}"
ros2 service call /kfs_infer_node/set_state kfs_core/srv/SetInferState "{enable: false}"

# 单次推理触发
ros2 service call /kfs_infer_node/trigger std_srvs/srv/Trigger
```

### CLI Demo (无 ROS2 环境)

```bash
cd /home/lee/realsense_inference
build/kfs_core/kfs_detect --config kfs_core/config/kfs_config.yaml

# 按键:
#   SPACE — 单次推理
#   D     — 切换持续推理
#   S     — 截图保存
#   Q/ESC — 退出
```

### rqt_reconfigure 可调参数

| 参数 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `model_path` | string | — | 3class 卷轴 ONNX 模型路径 |
| `conf_threshold` | float | 0.0~1.0 | 置信度阈值 |
| `iou_threshold` | float | 0.0~1.0 | NMS IoU 阈值 |
| `use_cuda` | bool | — | CUDA 推理开关 |
| `class_names` | string | — | 逗号分隔类别名，如 `"R1,T,F"` |
| `camera_type` | string | — | `usb` / `realsense` / `video` |
| `debug_image` | bool | — | 是否发布标注画面 |
| `inference_enabled` | bool | — | 推理使能 |
| `usb_device` | int | 0~63 | USB 摄像头设备号 |
| `usb_width` | int | 160~3840 | 分辨率宽 |
| `usb_height` | int | 120~2160 | 分辨率高 |
| `usb_fps` | int | 1~240 | 帧率 |
| `video_path` | string | — | 视频文件路径 (camera_type=video 时) |
| `video_loop` | bool | — | 视频文件循环播放 |
| `wh_model_path` | string | — | weaponhead ONNX 模型路径（空=禁用武器头检测） |

---

## 配置

编辑 `kfs_core/config/kfs_config.yaml`：

```yaml
# ============================================================
# KFS 视觉算法 — 全局配置
# ============================================================

# --- 相机 ---
camera:
  type: usb                     # "usb" | "realsense" | "video"
  usb:
    device: 0
    width: 1920
    height: 1080
    fps: 30
    fourcc: "MJPG"
    calibration_file: "kfs_core/config/ost.yaml"   # 可选，相机标定文件
    undistort: true                                 # 是否在线去畸变
  video:                        # camera.type=video 时生效
    path: "kfs_core/assets/002.mp4"
    loop: true
    calibration_file: "kfs_core/config/ost.yaml"
    undistort: true
  controls:
    auto_exposure: 0
    brightness: 0
    contrast: 50
    saturation: 64
    sharpness: 50

# --- 功夫卷轴识别模型 ---
model:
  path: kfs_core/models/kfs_yolo11_3class.onnx
  input_size: 640
  conf_threshold: 0.25
  iou_threshold: 0.30
  use_cuda: true
  class_names: ["R1", "T", "F"]

# --- 武器头识别模型 (与卷轴模型并行) ---
weaponhead_detector:
  wh_model_path: "kfs_core/models/kfs_weaponhead_v1.onnx"   # 空=禁用

# --- 调试显示 ---
display:
  debug: true
```

> `camera.type: video` + `video.path` 可指向本地视频文件代替摄像头，适合离线调试。

---

## 模型

仓库提供的预训练 ONNX 模型位于 `kfs_core/models/`：

| 文件 | 用途 | 训练数据 |
|------|------|----------|
| `kfs_yolo11_3class.onnx` | 功夫卷轴 3 分类 | — |
| `kfs_weaponhead_v1.onnx` | 武器头检测 (多边形标注原图训练) | 多边形 JSON 标注 |
| `kfs_weaponhead_v2.onnx` | 武器头检测 (去畸变 bbox 训练) | 去畸变 YOLO bbox |

> 推荐使用 `kfs_weaponhead_v1.onnx`（mAP50-95 = 0.8913，效果更优）。

### 导出 ONNX

```bash
cd /home/lee/realsense_inference/kfs_core

# 自动查找 best.pt 并导出 ONNX
python scripts/export_onnx.py

# 指定路径
python scripts/export_onnx.py \
    --pt /path/to/best.pt \
    --output models/my_model.onnx \
    --imgsz 640
```

---

## 相机标定

USB 摄像头可通过棋盘格标定生成 `ost.yaml`，用于在线去畸变。

```bash
cd /home/lee/realsense_inference

python3 kfs_core/scripts/calibrate_camera.py \
  --video kfs_core/assets/calib.mp4 \
  --size 11x8 \
  --square 0.02 \
  --output kfs_core/config/ost.yaml \
  --scale 1.0 \
  --sample-every 10 \
  --max-samples 20
```

标定完成后在 YAML 中启用：

```yaml
camera:
  usb:
    calibration_file: "kfs_core/config/ost.yaml"
    undistort: true
```

> 注意：开启去畸变后，推理模型应使用去畸变数据重新训练或验证效果。

---

## 目录结构

```
realsense_inference/
├── README.md
├── kfs_core/                          # ROS2 功能包
│   ├── CMakeLists.txt                 # CMake 构建 (colcon + 独立模式)
│   ├── package.xml                    # ROS2 包描述
│   ├── cmake/
│   │   └── FindOnnxRuntime.cmake      # ONNX Runtime 查找模块
│   ├── include/
│   │   ├── yolo_detector.h            # YOLO ONNX 推理器
│   │   ├── usb_capture.h             # USB 摄像头采集
│   │   ├── rs_capture.h              # RealSense 采集
│   │   └── kfs_core/
│   │       ├── config.h              # 配置结构 + YAML 加载
│   │       ├── icamera_capture.h     # 相机抽象接口
│   │       └── camera_factory.h      # 相机工厂
│   ├── src/
│   │   ├── kfs_infer_node.cpp        # ROS2 推理节点 (双 YOLO 并行主循环)
│   │   ├── yolo_detector.cpp         # ONNX Runtime 推理核心
│   │   ├── usb_capture.cpp           # V4L2 USB 捕获
│   │   ├── rs_capture.cpp            # RealSense 捕获
│   │   ├── config.cpp                # YAML 配置加载
│   │   ├── camera_factory.cpp        # 相机工厂实现
│   │   └── main.cpp                  # CLI Demo 入口
│   ├── msg/
│   │   ├── InferResult.msg           # 单检测框结构
│   │   └── InferResults.msg          # 一帧完整检测结果 (数组消息)
│   ├── srv/
│   │   └── SetInferState.srv         # 推理启停控制
│   ├── config/
│   │   └── kfs_config.yaml           # 默认配置文件
│   ├── models/                        # ONNX 模型
│   │   ├── kfs_yolo11_3class.onnx     # 卷轴识别
│   │   ├── kfs_weaponhead_v1.onnx     # 武器头识别 (推荐)
│   │   └── kfs_weaponhead_v2.onnx     # 武器头识别 (备选)
│   ├── scripts/
│   │   ├── export_onnx.py            # YOLO → ONNX 导出
│   │   └── calibrate_camera.py       # 相机标定脚本
│   └── assets/                        # 测试图片、数据集等 (gitignored)
├── build/                             # colcon 构建产物
├── install/                           # colcon 安装产物
└── log/                               # colcon 日志
```

---

## 集成到你的 ROS2 功能包

```cmake
# 在你的 CMakeLists.txt 中:
find_package(kfs_core REQUIRED)
target_link_libraries(your_node kfs::kfs_core)
```

```cpp
// 在你的 ROS2 节点中:
#include "kfs_core/config.h"
#include "kfs_core/camera_factory.h"
#include "yolo_detector.h"

auto cfg = kfs::loadConfig("config.yaml");
auto cam = kfs::CameraFactory::create(cfg);

// 卷轴识别
YoloDetector scroll_detector(cfg.model);
// 武器头识别 (并行)
std::unique_ptr<YoloDetector> wh_detector;
if (!cfg.weaponhead.whModelPath.empty())
    wh_detector = std::make_unique<YoloDetector>(cfg.weaponhead.whModelPath);

cam->start();
while (rclcpp::ok()) {
    cv::Mat frame;
    cam->getFrame(frame);

    auto result = scroll_detector.detect(frame);        // 卷轴检测
    if (wh_detector) {
        auto wh_result = wh_detector->detect(frame);    // 武器头检测
        result.detections.insert(result.detections.end(),
            wh_result.detections.begin(), wh_result.detections.end());
    }
    // result 同时包含 R1/T/F 和 weaponhead
}
```

---

## License

MIT © FJUTMTI
