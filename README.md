# KFS_inference_ROS

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![C++17](https://img.shields.io/badge/C++-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-✓-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**基于 YOLO11 ONNX + RealSense / USB 摄像头的实时目标检测 ROS2 功能包**

| 特性 | 说明 |
|------|------|
| 🎯 推理后端 | ONNX Runtime (CUDA / CPU 自动切换) |
| 📷 相机支持 | Intel RealSense D415 / D435 / 任意 V4L2 USB 摄像头 |
| 🤖 ROS2 接口 | 自定义 msg/srv + rqt_reconfigure 动态调参 |
| 🧩 模块化设计 | `libkfs_core` 共享库 — 可被任何 ROS2 节点链接 |
| ⚡ 推理性能 | CUDA ≈ 8~15ms / CPU ≈ 25~50ms (640×640) |

---

## 检测类别

| ID | 名称 | 说明 |
|----|------|------|
| 0 | R1 | (R_R1 + B_R1) |
| 1 | T | (T_03 ~ T_17) |
| 2 | F | (F_18 ~ F_32) |

> 类别名可通过 YAML 配置或 ROS2 参数动态修改。

---

## 架构

```
kfs_infer_node (ROS2 Node)
├── kfs::ICameraCapture  ← 相机抽象接口
│   ├── RealSenseCapture (librealsense2)
│   └── USBCapture       (OpenCV VideoCapture / V4L2)
├── YoloDetector         ← ONNX Runtime 推理
├── kfs::Config          ← YAML + ROS 参数配置
└── kfs::CameraFactory   ← 相机工厂

ROS2 通信:
  [Pub]  ~/result        kfs_core/InferResult    — 检测结果
  [Pub]  ~/debug_image   sensor_msgs/Image       — 标注画面
  [Pub]  ~/status        std_msgs/String         — 运行状态
  [Sub]  ~/enable        std_msgs/Bool           — 推理使能
  [Srv]  ~/set_state     kfs_core/SetInferState  — 启停控制
  [Srv]  ~/trigger       std_srvs/Trigger        — 单次推理
  [Param] 19 项可动态调节 (rqt_reconfigure)
```

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
pip install ultralytics onnx onnx-simplifier
```

---

## 一键安装

```bash
# 1. 系统依赖
sudo apt install -y cmake build-essential libopencv-dev libyaml-cpp-dev

# 2. ONNX Runtime (推荐手动安装以控制版本)
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

### 方式一：colcon (推荐，ROS2 原生)

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/<your-org>/KFS_inference_ROS.git kfs_core
cd ~/ros2_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select kfs_core --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.zsh
```

### 方式二：独立 CMake (不依赖 colcon，但无 ROS2 节点)

```bash
cd KFS_inference_ROS
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 产物:
#   libkfs_core_lib.so  — 核心库
#   kfs_detect          — CLI Demo
#   kfs_infer_node      — ROS2 节点
```

---

## 模型导出

```bash
cd KFS_inference_ROS

# 自动查找 best.pt 并导出 ONNX
python scripts/export_onnx.py

# 指定路径
python scripts/export_onnx.py \
    --pt /path/to/best.pt \
    --output models/kfs_yolo11_3class.onnx \
    --imgsz 640
```

---

## 配置

编辑 `config/kfs_config.yaml`，或通过 ROS2 参数覆盖：

```yaml
camera:
  type: usb               # "usb" 或 "realsense"
  usb:
    device: 0
    width: 640
    height: 480
    fps: 60
    fourcc: "MJPG"
  controls:
    auto_exposure: 0
    brightness: 0
    contrast: 50
    saturation: 64
    sharpness: 50

model:
  path: models/kfs_yolo11_3class.onnx
  input_size: 640
  conf_threshold: 0.25
  iou_threshold: 0.30
  use_cuda: true
  class_names: ["R1", "T", "F"]

display:
  debug: true             # CLI Demo 用, ROS2 节点忽略
```

---

## 运行

### ROS2 节点 (推荐)

```bash
# 启动推理节点
ros2 run kfs_core kfs_infer_node --ros-args \
  -p model_path:=/absolute/path/to/kfs_yolo11_3class.onnx \
  -p config_path:=/absolute/path/to/kfs_config.yaml

# 查看标注画面
rqt_image_view /kfs_infer_node/debug_image

# 动态调参
ros2 run rqt_reconfigure rqt_reconfigure

# 查看检测结果
ros2 topic echo /kfs_infer_node/result

# 控制推理启停
ros2 service call /kfs_infer_node/set_state kfs_core/srv/SetInferState "{enable: true}"
ros2 service call /kfs_infer_node/trigger std_srvs/srv/Trigger
```

### rqt_reconfigure 可调参数

| 参数 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `conf_threshold` | float | 0.0 ~ 1.0 | 置信度阈值 |
| `iou_threshold` | float | 0.0 ~ 1.0 | NMS IoU 阈值 |
| `use_cuda` | bool | — | CUDA 推理开关 |
| `model_path` | string | — | ONNX 模型路径 |
| `class_names` | string | — | 逗号分隔类别名, 如 `"R1,T,F"` |
| `camera_type` | string | — | `usb` / `realsense` |
| `usb_device` | int | 0 ~ 63 | USB 摄像头设备号 |
| `usb_width` | int | 160 ~ 3840 | 分辨率宽 |
| `usb_height` | int | 120 ~ 2160 | 分辨率高 |
| `usb_fps` | int | 1 ~ 240 | 帧率 |
| `debug_image` | bool | — | 是否发布标注画面 |
| `inference_enabled` | bool | — | 推理使能 |

### CLI Demo (无 ROS2 环境)

```bash
./build/kfs_detect --config config/kfs_config.yaml

# 按键:
#   SPACE — 单次推理
#   D     — 切换持续推理
#   S     — 截图保存
#   Q/ESC — 退出
```

---

## 目录结构

```
KFS_inference_ROS/
├── CMakeLists.txt              # CMake 构建 (colcon + 独立模式)
├── package.xml                 # ROS2 包描述
│
├── cmake/
│   └── FindOnnxRuntime.cmake   # ONNX Runtime 查找模块 (可复用)
│
├── include/
│   ├── yolo_detector.h         # YOLO 推理器 (public)
│   ├── usb_capture.h           # USB 摄像头 (public)
│   ├── rs_capture.h            # RealSense 摄像头 (public, 条件编译)
│   └── kfs_core/
│       ├── icamera_capture.h   # 相机抽象接口
│       ├── config.h            # 配置结构 + YAML 加载
│       └── camera_factory.h    # 相机工厂
│
├── src/
│   ├── yolo_detector.cpp       # ONNX Runtime 推理核心
│   ├── usb_capture.cpp         # V4L2 USB 捕获
│   ├── rs_capture.cpp          # RealSense 捕获 (条件编译)
│   ├── config.cpp              # YAML 配置加载
│   ├── camera_factory.cpp      # 相机工厂实现
│   ├── main.cpp                # CLI Demo 薄层
│   └── kfs_infer_node.cpp      # ROS2 推理节点
│
├── msg/
│   └── InferResult.msg         # 自定义消息: 检测结果
│
├── srv/
│   └── SetInferState.srv       # 自定义服务: 推理启停
│
├── config/
│   └── kfs_config.yaml         # 默认配置文件
│
├── models/                     # ONNX 模型 (gitignore)
├── scripts/
│   └── export_onnx.py          # YOLO → ONNX 导出
└── README.md
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
YoloDetector detector(cfg.model);

cv::Mat frame;
cam->start();
while (rclcpp::ok()) {
    cam->getFrame(frame);
    auto result = detector.detect(frame);
    // 处理 result.detections ...
}
```

---

## License

MIT © KFS Team

```bash
./build/kfs_detect                          # 使用默认配置
./build/kfs_detect --config my_config.yaml  # 指定配置
./build/kfs_detect --list-cameras           # 列出 USB 摄像头
```

---

## Debug 模式按键

| 按键 | 功能 |
|------|------|
| `q` / `ESC` | 退出 |
| `s` | 截图保存为 PNG |
| `SPACE` | 触发单次推理 / 切换连续模式 |
| `m` | 切换 IDLE / CONTINUOUS 模式 |

---

## 输出格式

```cpp
struct Detection {
    int class_id;           // 0=R1, 1=T, 2=F
    string class_name;      // "R1", "T", "F"
    float confidence;       // [0, 1]
    Point2f corner_tl;      // 左上角 (像素坐标)
    Point2f corner_tr;      // 右上角
    Point2f corner_br;      // 右下角
    Point2f corner_bl;      // 左下角
};
```

终端输出示例：

```
═══════════════════════════════════════════
  KFS 实时检测结果
  帧尺寸: 640×480 | 推理: 8.2 ms
───────────────────────────────────────────
  # | 类别 | 置信度 | 左上角点 (x,y) | 右下角点 (x,y)
───────────────────────────────────────────
  0 |   R1 |  0.92 | ( 320, 240) | ( 580, 460)
  1 |    T |  0.87 | ( 640, 300) | ( 920, 520)
  2 |    F |  0.76 | (1000, 400) | (1280, 620)
═══════════════════════════════════════════
```

---

## 目录结构

```
realsense_inference/
├── CMakeLists.txt
├── config/
│   └── kfs_config.yaml         # 全局 YAML 配置
├── models/
│   └── kfs_yolo11_3class.onnx  # ONNX 模型
├── include/
│   ├── yolo_detector.h         # 推理核心
│   ├── rs_capture.h            # RealSense D415 采集
│   └── usb_capture.h           # USB 摄像头采集
├── src/
│   ├── main.cpp                # 主程序
│   ├── yolo_detector.cpp       # ONNX 推理实现
│   ├── rs_capture.cpp          # D415 实现
│   └── usb_capture.cpp         # USB 实现
├── scripts/
│   └── export_onnx.py          # .pt → .onnx 导出
└── README.md
```
sudo usermod -aG video $USER
# 重新登录生效
```

### RealSense D415

```bash
# 添加 udev 规则
sudo bash -c 'echo "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"8086\", MODE=\"0666\"" > /etc/udev/rules.d/99-realsense.rules'
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## 🧪 测试 (无相机时用视频/图片)

如果没有连接摄像头，可以写一个简单的测试程序用图片或视频文件验证 ONNX 推理:

```cpp
// test_standalone.cpp
#include "yolo_detector.h"
#include <opencv2/highgui.hpp>

int main() {
    YoloDetector detector("models/kfs_yolo11_3class.onnx");
    cv::Mat img = cv::imread("test.jpg");
    auto result = detector.detect(img);

    for (auto& d : result.detections) {
        printf("[%s] conf=%.2f  tl=(%.0f,%.0f) br=(%.0f,%.0f)\n",
               d.class_name.c_str(), d.confidence,
               d.corner_tl.x, d.corner_tl.y,
               d.corner_br.x, d.corner_br.y);
    }
    return 0;
}
```

## ❔ ONNX 导出脚本详细帮助

```bash
cd ~/KFS_training/realsense_inference
python scripts/export_onnx.py --help
```

支持的导出格式:
- `onnx` — ONNX (CPU/GPU 通用)
- `engine` — NVIDIA TensorRT (最快, 需 GPU + TensorRT)
- `openvino` — Intel OpenVINO (Intel CPU/GPU 优化)
- `tflite` — TensorFlow Lite (移动端/嵌入式)
- `coreml` — Apple Core ML (macOS/iOS)

### 常见问题

| 问题 | 解决方法 |
|---|---|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| ONNX 简化失败 | 加 `--no-simplify` 或 `pip install onnx-simplifier` |
| opset 版本不兼容 | 尝试 `--opset 11` |
| .pt 文件找不到 | 先运行 `train_yolo11_3class.py` 训练, 或 `--list-pt` 查看
