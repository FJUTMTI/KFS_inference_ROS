# KFS 目标检测 (realsense_inference workspace)

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
├── WeaponheadDetector   ← 传统 OpenCV (可选, 松耦合, USB 虚焦 weaponhead 左右边界)
├── kfs::Config          ← YAML + ROS 参数配置
└── kfs::CameraFactory   ← 相机工厂

ROS2 通信:
  [Pub]  ~/result        kfs_core/InferResults   — 检测结果（数组，一帧一条，推荐）
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

当前源码布局：工作空间根 `realsense_inference/` 下有一个 `kfs_core/` 子目录作为 ROS2 功能包（包名 `kfs_core`）。

### 方式一：colcon (推荐，ROS2 原生)

```bash
# 假设当前在 realsense_inference/ 根目录（kfs_core/ 是包源码）
source /opt/ros/humble/setup.zsh
colcon build --base-paths kfs_core --packages-select kfs_core --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.zsh
```

### 方式二：独立 CMake (不依赖 colcon，但无 ROS2 节点)

```bash
cd realsense_inference/kfs_core
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 产物（在 kfs_core/build/ 下）:
#   libkfs_core_lib.so  — 核心库
#   kfs_detect          — CLI Demo
#   kfs_infer_node      — ROS2 节点（需 ROS2 环境）
```

从工作空间根可直接运行 CLI（无需 install）：
```
build/kfs_core/kfs_detect --config kfs_core/config/kfs_config.yaml ...
```

---

## 模型导出

```bash
cd realsense_inference/kfs_core

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

编辑 `kfs_core/config/kfs_config.yaml`（从工作空间根运行时路径），或通过 ROS2 参数覆盖。

关键新增：
- `detector.enable_weaponhead: true` 可与 YOLO 并行启用传统 OpenCV weaponhead 检测（USB 相机下识别虚焦中央物体轮廓左右边界）。
- `camera.type: video` + `video.path` 支持用本地视频文件 (如 assets/001.mp4) 代替摄像头，用于调试和效果验证（支持 loop 循环）。

```yaml
camera:
  type: usb               # "usb" | "realsense" | "video"
  usb:
    device: 0
    width: 640
    height: 480
    fps: 60
    fourcc: "MJPG"
    calibration_file: "kfs_core/config/ost.yaml"  # 可选，标定后生成
    undistort: false                            # 是否对捕获帧做畸变矫正预处理 (见下方说明)
  controls:
    auto_exposure: 0
    brightness: 0
    contrast: 50
    saturation: 64
    sharpness: 50

# 视频文件调试模式（推荐用于可复现的开发/验证，无需连接摄像头）
# camera:
#   type: video
#   video:
#     path: "kfs_core/assets/001.mp4"   # 替换为你的测试视频
#     loop: true                        # 循环播放
#     calibration_file: "kfs_core/config/ost.yaml"
#     undistort: false
# （此时 usb/controls 会被忽略；ROS2 节点也支持通过 -p video_path:=... 等参数动态切换）

model:
  path: kfs_core/models/kfs_yolo11_3class.onnx
  input_size: 640
  conf_threshold: 0.25
  iou_threshold: 0.30
  use_cuda: true
  class_names: ["R1", "T", "F"]

display:
  debug: true             # CLI Demo 用, ROS2 节点忽略

detector:
  type: yolo
  enable_weaponhead: false   # 启用后与 YOLO 并行 (weaponhead_detector 传统 CV)
```

## 相机标定 (USB 摄像头)

本项目支持使用 ROS2 `camera_calibration` 对 USB 摄像头进行张正友标定，生成 `ost.yaml` 供运行时加载真实内参 (fx/fy/cx/cy)。

### 准备标定视频

- 准备 11x8 棋盘格标定板 (interior corners 11x8，即 12x9 方格)，方格边长例如 20mm。
- 录制视频时缓慢平移/旋转/倾斜标定板，覆盖画面不同区域和距离。示例视频已放在 `kfs_core/assets/calib.mp4`。

### 执行标定

使用仓库提供的脚本 (内部调用 ROS2 camera_calibration 的 MonoCalibrator)：

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

- `--square` : 方格物理尺寸 (米)，根据你的标定板修改 (20mm=0.02)。
- 默认以视频原生分辨率 (本例 1920×1080) 处理并输出 ost.yaml；如需更快可加 `--scale 0.5`（结果会按比例对应较低分辨率，运行时仍自动缩放内参）。
- 输出 `ost.yaml` 即标准 ROS 格式，包含 camera_matrix / distortion / projection 等。

标定完成后，`kfs_core/config/kfs_config.yaml` 默认已指向它：

```yaml
camera:
  usb:
    ...
    calibration_file: "kfs_core/config/ost.yaml"
    undistort: true          # 启用畸变矫正预处理 (推荐用于视觉任务)
```

USB 捕获器启动时会加载标定并打印 `[USB] 已加载相机标定...`。

如果 `undistort: true`，每次 `getFrame()` 会自动用 `cv::initUndistortRectifyMap` + `remap` 做畸变矫正，返回无畸变图像（同时内部会把返回的内参更新为 getOptimalNewCameraMatrix 后的新 K）。

注意：YOLO / weaponhead_detector 模型通常在带畸变的原始画面上训练/调参，开启 undistort 后可能需要重新验证检测效果。推荐先设 false 跑 baseline，再打开 true 对比。

脚本默认 `--scale 1.0`（原生）。如果标定视频很大，可用 `--scale 0.5 --sample-every 5` 加速。

当前 ost.yaml 已基于 calib.mp4 的 1920×1080 原生分辨率生成。

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

# 查看检测结果（现在是一帧一个数组消息）
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
| `detector_type` | string | — | `yolo` / `weaponhead_detector` |
| `enable_weaponhead` | bool | — | 并行启用 weaponhead_detector (传统 OpenCV 虚焦物体左右边界) |
| `usb_device` | int | 0 ~ 63 | USB 摄像头设备号 |
| `usb_width` | int | 160 ~ 3840 | 分辨率宽 |
| `usb_height` | int | 120 ~ 2160 | 分辨率高 |
| `usb_fps` | int | 1 ~ 240 | 帧率 |
| `debug_image` | bool | — | 是否发布标注画面 |
| `inference_enabled` | bool | — | 推理使能 |

### CLI Demo (无 ROS2 环境)

从工作空间根目录运行（推荐）：

```bash
cd /home/lee/realsense_inference
build/kfs_core/kfs_detect --config kfs_core/config/kfs_config.yaml

# 按键:
#   SPACE — 单次推理
#   D     — 切换持续推理
#   S     — 截图保存
#   Q/ESC — 退出
```

( colcon build 后 `source install/setup.bash` 即可在 PATH 中直接使用 `kfs_detect`，但建议显式 --config 指向 kfs_core/config/kfs_config.yaml )

#### 使用测试图片验证 weaponhead_detector (推荐用于确认功能，无需相机)

新结构下推荐命令会**自动创建临时文件夹并输出带标注的结果图像**（红线+文字标出左右边界像素）：

```bash
cd /home/lee/realsense_inference

# 临时启用（测试后恢复）
cp kfs_core/config/kfs_config.yaml /tmp/kfs_config.bak
sed -i 's/enable_weaponhead: false/enable_weaponhead: true/' kfs_core/config/kfs_config.yaml

# 运行（支持目录，一次测试 assets 全部图片）
# 会创建 /tmp/kfs_weaponhead_test_YYYYMMDD_HHMMSS/ 并保存 01_marked.png 等
build/kfs_core/kfs_detect --image kfs_core/assets

# 恢复
mv /tmp/kfs_config.bak kfs_core/config/kfs_config.yaml
```

输出示例:

```
[WEAPONHEAD] 左右边界像素: left=150  right=444  (宽度=294)
...
[SAVE] 标注图像已输出: /tmp/kfs_weaponhead_test_.../01_marked.png
[INFO] 所有测试完成。临时文件夹: /tmp/kfs_weaponhead_test_...
```

这精确模拟了从 USB 相机视频流中识别武器头 (weaponhead) 虚焦物体轮廓的左右边界像素。
```

---

## 目录结构 (工作空间视图)

当前 README 已移至工作空间根目录。包源码位于 `kfs_core/` 子目录下。

```
realsense_inference/                 # 工作空间根 (本 README 所在)
├── README.md                        # 主文档（已移至上级）
├── kfs_core/                    # 功能包 (包名仍为 kfs_core)
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── assets/                      # 测试图片 (用于 weaponhead 左右边界验证)
│   ├── config/kfs_config.yaml
│   ├── include/...
│   ├── models/...
│   ├── src/ (包含 weaponhead_detector.cpp 等)
│   └── ...
├── build/ install/ log/             # colcon 构建产物
└── ...
```

├── CMakeLists.txt              # CMake 构建 (colcon + 独立模式)
├── package.xml                 # ROS2 包描述
│
├── cmake/
│   └── FindOnnxRuntime.cmake   # ONNX Runtime 查找模块 (可复用)
│
├── include/
│   ├── yolo_detector.h         # YOLO 推理器 (public)
│   ├── weaponhead_detector.h   # 传统 OpenCV weaponhead (松耦合, public)
│   ├── usb_capture.h           # USB 摄像头 (public)
│   ├── rs_capture.h            # RealSense 摄像头 (public, 条件编译)
│   └── kfs_core/
│       ├── icamera_capture.h   # 相机抽象接口
│       ├── config.h            # 配置结构 + YAML 加载
│       └── camera_factory.h    # 相机工厂
│
├── src/
│   ├── yolo_detector.cpp       # ONNX Runtime 推理核心
│   ├── weaponhead_detector.cpp # 传统 OpenCV weaponhead 轮廓检测 (松耦合)
│   ├── usb_capture.cpp         # V4L2 USB 捕获
│   ├── rs_capture.cpp          # RealSense 捕获 (条件编译)
│   ├── config.cpp              # YAML 配置加载
│   ├── camera_factory.cpp      # 相机工厂实现
│   ├── main.cpp                # CLI Demo 薄层
│   └── kfs_infer_node.cpp      # ROS2 推理节点
│
├── msg/
│   ├── InferResult.msg         # 单检测框结构（InferResults 的元素类型）
│   └── InferResults.msg        # 一帧完整检测结果（数组消息，推荐使用）
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
#include "weaponhead_detector.h"   // 可选, weaponhead_detector

auto cfg = kfs::loadConfig("config.yaml");
auto cam = kfs::CameraFactory::create(cfg);
YoloDetector detector(cfg.model);

// weaponhead_detector 松耦合可选 (并行使用, e.g. for usb camera + 虚焦 weaponhead)
std::unique_ptr<WeaponheadDetector> wh;
if (cfg.enableWeaponheadDetector) wh = std::make_unique<WeaponheadDetector>();

cv::Mat frame;
cam->start();
while (rclcpp::ok()) {
    cam->getFrame(frame);
    auto result = detector.detect(frame);
    if (wh) {
        auto whr = wh->detect(frame);
        for (auto& d : whr.detections) result.detections.push_back(d);  // 合并 WEAPONHEAD
    }
    // 处理 result.detections ... (可能同时有 YOLO 的 R1/T/F 和 weaponhead 的 WEAPONHEAD)
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

## 真实部署提示

实际项目中把 `kfs_core/` 目录放到 ROS2 workspace 的 `src/` 下即可

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

### 验证 weaponhead_detector (传统 OpenCV 武器头虚焦轮廓左右边界)

由于文件结构已调整为更标准的 ROS2 功能包布局（源码位于 `kfs_core/` 下），测试时请从工作空间根目录 (`realsense_inference/`) 运行，并使用完整相对路径。

推荐新的测试命令（会自动建立临时文件夹并输出带标注的结果图像）：

```bash
cd /home/lee/realsense_inference

# 1. 临时启用 weaponhead (测试后可改回 false)
sed -i 's/enable_weaponhead: false/enable_weaponhead: true/' kfs_core/config/kfs_config.yaml

# 2. 运行测试 (推荐：传入 assets 目录，一次测试所有图片)
#    命令会自动创建 /tmp/kfs_weaponhead_test_<时间戳>/ 并保存带红线标注左右边界的 PNG
build/kfs_core/kfs_detect --image kfs_core/assets

# 单独测试一张:
# build/kfs_core/kfs_detect --image kfs_core/assets/01.png

# 3. 测试完恢复
sed -i 's/enable_weaponhead: true/enable_weaponhead: false/' kfs_core/config/kfs_config.yaml
```

**新行为**：
- 每次 `--image` 测试会自动在 `/tmp/` 下创建唯一临时文件夹，例如 `/tmp/kfs_weaponhead_test_20260624_171255/`
- 在该文件夹中为每张输入图片生成 `<basename>_marked.png` （已用红线精确标出左右边界像素值、文字标签 "L:150" "R:444" 等）
- 同时在终端清晰打印左右边界像素，方便确认。
- 即使 `display.debug: false` 也会保存标注图；设为 true 可即时弹出窗口查看。

示例终端输出片段：

```
[INFO] 图片测试模式启动
       输入: kfs_core/assets (共 2 张图)
       临时输出文件夹: /tmp/kfs_weaponhead_test_20260624_171255
...
[WEAPONHEAD] 左右边界像素: left=150  right=444  (宽度=294)
...
[SAVE] 标注图像已输出: /tmp/kfs_weaponhead_test_.../01_marked.png
[INFO] 所有测试完成。临时文件夹: /tmp/kfs_weaponhead_test_20260624_171255
       请检查其中的 *_marked.png 确认 weaponhead 左右边界是否正确标出。
```

实际从 USB 相机获取视频流时，weaponhead_detector 会在每帧与 YOLO 并行运行，结果合并发布。

## ❔ ONNX 导出脚本详细帮助

```bash
cd ~/realsense_inference/kfs_core
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
