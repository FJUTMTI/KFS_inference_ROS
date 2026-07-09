# KFS 视觉算法 — ROBCON 2026 武林探秘

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![C++17](https://img.shields.io/badge/C++-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-✓-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.19-4C4C4C.svg)](https://onnxruntime.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **CURC ROBCON 2026「武林探秘」实时视觉算法系统**

本仓库 `kfs_core` 提供比赛所需的视觉识别能力，统一封装相机采集、配置加载、检测器与 ROS2 节点。

| 功能 | 实现 | 用途 |
|------|------|------|
| 🏷️ **功夫卷轴识别** | YOLO11 ONNX (`kfs_yolo11_3class.onnx`) | 识别赛场卷轴类别 R1 / T / F |
| ⚔️ **武器头识别** | YOLO ONNX (`whv4.onnx` 等) | 识别武器头顶端，输出边界框 |
| 💡 **灯条颜色识别** | 传统 OpenCV (`LightbarDetector`) | 识别中央共线双段 LED 灯条颜色 (红/绿/蓝/黄) |

检测器可单独或并行运行：YOLO 与灯条结果合并到同一帧 `FrameResult` / `InferResults`。

---

## 特性

| 特性 | 说明 |
|------|------|
| 🎯 推理后端 | ONNX Runtime (CUDA / CPU 自动切换) |
| ⚡ 推理性能 | YOLO CUDA ≈ 8~15ms；灯条 OpenCV ≈ 50~200ms (1900×2532 全图) |
| 🔀 多检测器并行 | 3class 卷轴 + weaponhead + 灯条可同时运行并合并发布 |
| 📷 相机支持 | Intel RealSense D415/D435 / V4L2 USB / 本地视频文件 |
| 🖼️ 离线测图 | CLI `--image` 支持单图或目录批量测试（灯条样例见 `assets/lightbar`） |
| 🤖 ROS2 接口 | 自定义 msg/srv + rqt_reconfigure 动态参数 |
| 🧩 模块化 | `libkfs_core_lib` 共享库，可被任意节点链接 |
| 📐 畸变矫正 | USB/Video 支持 `ost.yaml` 标定 + 在线去畸变 |
| ⚙️ 全配置驱动 | 相机类型、检测器开关、阈值等均由 YAML / ROS 参数控制 |

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
| 99 | WEAPONHEAD | 与 3class 并行时统一为 class_id=99，避免 ID 冲突 |

### 灯条 (OpenCV 传统视觉)

| ID | 名称 | 说明 |
|----|------|------|
| 10 | LIGHTBAR_RED | 红色双段共线 LED |
| 11 | LIGHTBAR_GREEN | 绿色 |
| 12 | LIGHTBAR_BLUE | 蓝色 |
| 13 | LIGHTBAR_YELLOW | 黄色 |
| 19 | LIGHTBAR_UNKNOWN | 几何配对成功但颜色不确定 |

输出为**旋转矩形四角**（`corner_tl/tr/br/bl`），覆盖两段等长共线灯条（支持任意倾斜角度）。

---

## 算法结构

```
┌──────────────────────────────────────────────────────────────┐
│                      kfs_infer_node / kfs_detect             │
│                                                              │
│  ┌────────────────┐     frame                                │
│  │ ICameraCapture │──────────────────────────────────┐       │
│  │  · USBCapture  │                                  │       │
│  │  · RealSense   │   ┌──────────────────────────────▼────┐  │
│  │  · VideoFile   │   │  YoloDetector (3class)  [可选]     │  │
│  └────────────────┘   │  · R1 / T / F                      │  │
│                       ├────────────────────────────────────┤  │
│                       │  YoloDetector (weaponhead) [可选]  │  │
│                       │  · WEAPONHEAD                      │  │
│                       ├────────────────────────────────────┤  │
│                       │  LightbarDetector (OpenCV) [可选]  │  │
│                       │  · 高亮灯芯 + 形态学 + 共线配对    │  │
│                       │  · 外环 HSV / 通道差 → 颜色        │  │
│                       └──────────────────┬─────────────────┘  │
│                                          │ 合并 FrameResult    │
│  ┌───────────────────────────────────────▼─────────────────┐  │
│  │ ROS2: ~/result  ~/debug_image  ~/status  服务/参数        │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 灯条检测流程

1. **自适应高亮阈值**：从灰度高分位起搜 thr；黄灯等大面积 bloom 自动抬高阈值，压成干净灯芯  
2. **多方向形态学**：开运算去噪 + 0°/45°/90°/135° 线状闭运算，适配斜向 LED 间隙  
3. **细长段过滤**：长宽比、短边上限，拒绝 bloom 胖块  
4. **等长共线配对**：按长轴单位向量分解 `along` / `lateral`，支持旋转共线（非仅竖直）  
5. **颜色判定**：段外环 HSV 峰值 + 通道差（红/绿/蓝/黄消歧）  
6. **输出**：两段合并的旋转矩形 + `LIGHTBAR_*` 类别

测试样例：`kfs_core/assets/lightbar/`（红/绿/蓝/黄共 24 张）。  
标注/掩膜参考：`kfs_core/assets/lightbar_debug/`。

### ROS2 消息

| 话题/服务 | 类型 | 方向 | 说明 |
|-----------|------|------|------|
| `~/result` | `kfs_core/InferResults` | 发布 | 一帧全部检测（含 `inference_ms`、帧尺寸） |
| `~/debug_image` | `sensor_msgs/Image` | 发布 | 标注调试画面 |
| `~/status` | `std_msgs/String` | 发布 | 状态文本 |
| `~/enable` | `std_msgs/Bool` | 订阅 | 推理使能 |
| `~/set_state` | `kfs_core/SetInferState` | 服务 | 显式启停推理 |
| `~/trigger` | `std_srvs/Trigger` | 服务 | 触发单次推理 |
| `~/snapshot` | `std_srvs/Trigger` | 服务 | 保存当前帧截图 |

---

## 依赖

### 系统依赖

| 依赖 | 版本 | 必需 | 安装 |
|------|------|------|------|
| ROS2 Humble | — | 节点需要 | [官方指南](https://docs.ros.org/en/humble/Installation.html) |
| CMake | ≥ 3.16 | ✅ | `sudo apt install cmake` |
| GCC (C++17) | ≥ 8 | ✅ | `sudo apt install build-essential` |
| OpenCV | ≥ 4.x | ✅ | `sudo apt install libopencv-dev` |
| ONNX Runtime | ≥ 1.16 | ✅（仅灯条模式可无模型推理） | 见下方 |
| yaml-cpp | — | ✅ | `sudo apt install libyaml-cpp-dev` |
| librealsense2 | ≥ 2.50 | ❌ | `sudo apt install librealsense2-dev` |

> 仅跑灯条 CLI（`detector.type: lightbar`）时不加载 ONNX，但当前库仍链接 ONNX Runtime。

### 一键安装依赖

```bash
sudo apt install -y cmake build-essential libopencv-dev libyaml-cpp-dev

# ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz
tar xzf onnxruntime-linux-x64-1.19.2.tgz
echo 'export ONNXRUNTIME_DIR='"$(pwd)"'/onnxruntime-linux-x64-1.19.2' >> ~/.bashrc
source ~/.bashrc

# 可选 RealSense
sudo apt install -y librealsense2-dev
```

---

## 编译

### colcon（推荐）

```bash
cd /path/to/realsense_inference
source /opt/ros/humble/setup.bash   # 或 setup.zsh
colcon build --packages-select kfs_core --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

产物：

- `install/kfs_core/lib/libkfs_core_lib.so` — 核心库  
- `install/kfs_core/bin/kfs_detect` — CLI  
- `install/kfs_core/lib/kfs_core/kfs_infer_node` — ROS2 节点  

若运行 CLI 提示找不到 `.so`：

```bash
export LD_LIBRARY_PATH="$(pwd)/install/kfs_core/lib:${LD_LIBRARY_PATH}"
```

### 独立 CMake（无 ROS2，仅库 + CLI）

```bash
cd kfs_core
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# 产物: libkfs_core_lib.so, kfs_detect
```

---

## 运行

### CLI Demo

在**工作空间根目录**执行（路径相对仓库根）：

```bash
export LD_LIBRARY_PATH="$(pwd)/install/kfs_core/lib:${LD_LIBRARY_PATH}"

# 默认配置（相机 / 视频 + 已启用的检测器）
./install/kfs_core/bin/kfs_detect --config kfs_core/config/kfs_config.yaml

# 列出 USB 摄像头
./install/kfs_core/bin/kfs_detect --list-cameras

# 静态图片 / 目录测试
./install/kfs_core/bin/kfs_detect \
  --config kfs_core/config/kfs_config.yaml \
  --image kfs_core/assets/lightbar
```

**按键（debug 窗口）：**

| 键 | 功能 |
|----|------|
| `SPACE` | 单次推理 |
| `D` | 切换持续推理 |
| `S` | 截图 |
| `Q` / `ESC` | 退出 |

路径名含 `lightbar` 时，CLI 会自动切到灯条检测（无需改 YOLO 配置即可测图）。  
批量测图结果写入 `/tmp/kfs_test_YYYYMMDD_HHMMSS/`（`*_marked.png`，可选 `*_mask.png`）。

#### 仅测灯条（不加载 YOLO）

在 `kfs_config.yaml` 中设置：

```yaml
detector:
  type: lightbar          # 不加载 ONNX
  enable_lightbar: true

lightbar_detector:
  enabled: true
  save_debug_mask: true   # 输出核心掩膜到 debug_image
```

或并行：

```yaml
detector:
  type: yolo_lightbar     # YOLO + 灯条
  enable_lightbar: true
```

### ROS2 节点

```bash
source install/setup.bash

ros2 run kfs_core kfs_infer_node --ros-args \
  -p config_path:=$(pwd)/kfs_core/config/kfs_config.yaml

# 视频代替摄像头
ros2 run kfs_core kfs_infer_node --ros-args \
  -p camera_type:=video \
  -p video_path:=kfs_core/assets/005.mp4

# 启用灯条
ros2 run kfs_core kfs_infer_node --ros-args \
  -p enable_lightbar:=true \
  -p detector_type:=yolo_lightbar

# 查看结果
rqt_image_view /kfs_infer_node/debug_image
ros2 topic echo /kfs_infer_node/result --once
ros2 run rqt_reconfigure rqt_reconfigure
```

### 推理控制

```bash
ros2 service call /kfs_infer_node/set_state kfs_core/srv/SetInferState "{enable: true}"
ros2 service call /kfs_infer_node/trigger std_srvs/srv/Trigger
ros2 service call /kfs_infer_node/snapshot std_srvs/srv/Trigger
```

### 主要动态参数

| 参数 | 说明 |
|------|------|
| `config_path` | YAML 配置路径 |
| `camera_type` | `usb` / `realsense` / `video` |
| `model_path` / `conf_threshold` / `iou_threshold` / `use_cuda` | 3class YOLO |
| `wh_model_path` | weaponhead 模型（空=禁用） |
| `detector_type` | `yolo` / `lightbar` / `yolo_lightbar` |
| `enable_lightbar` | 是否并行灯条检测 |
| `enable_weaponhead` | 是否启用 weaponhead 相关逻辑 |
| `video_path` / `video_loop` | 视频输入 |
| `usb_device` / `usb_width` / `usb_height` / `usb_fps` | USB 相机 |
| `inference_enabled` / `debug_image` | 推理与调试图 |

---

## 配置

主配置：`kfs_core/config/kfs_config.yaml`。

```yaml
camera:
  type: video                 # usb | realsense | video
  video:
    path: "kfs_core/assets/005.mp4"
    loop: true
    calibration_file: "kfs_core/config/ost.yaml"
    undistort: true
  usb:
    device: 0
    width: 1920
    height: 1080
    fps: 30
    fourcc: "MJPG"
    calibration_file: "kfs_core/config/ost.yaml"
    undistort: true

model:
  path: kfs_core/models/kfs_yolo11_3class.onnx
  input_size: 640
  conf_threshold: 0.25
  iou_threshold: 0.30
  use_cuda: true
  class_names: ["R1", "T", "F"]

detector:
  type: yolo                  # yolo | lightbar | yolo_lightbar
  enable_weaponhead: true
  enable_lightbar: false

weaponhead_detector:
  enabled: true
  wh_model_path: "kfs_core/models/whv4.onnx"

lightbar_detector:
  enabled: false
  # 高亮灯芯（过胖 bloom 自动抬 thr）
  core_percentile: 99.8
  core_thresh_min: 200
  core_thresh_max: 235
  core_thresh_scale: 0.90
  max_core_frac: 0.015
  max_short_side: 65.0
  # 形态学
  open_ksize: 3
  close_length: 15
  # 细长段 / 共线
  min_length: 50.0
  min_aspect: 3.5
  min_area: 50.0
  min_length_ratio: 0.55
  max_angle_diff: 18.0
  # 颜色外环
  bloom_ksize: 41
  min_color_score: 5.0
  save_debug_mask: false
```

| 配置段 | 作用 |
|--------|------|
| `camera` | 输入源：USB / RealSense / 视频，及曝光等 V4L2 控制 |
| `model` | 3class YOLO 路径与阈值 |
| `detector` | 检测器组合开关 |
| `weaponhead_detector` | 第二路 YOLO 模型 |
| `lightbar_detector` | 灯条 OpenCV 算法可调参数 |
| `display.debug` | CLI 是否弹窗显示 |

---

## 模型

预训练权重在 `kfs_core/models/`：

| 文件 | 用途 |
|------|------|
| `kfs_yolo11_3class.onnx` | 功夫卷轴 3 分类 |
| `whv4.onnx` | 武器头检测（当前配置默认） |
| `kfs_weaponhead_v1.onnx` / `v2` | 武器头备选 |
| `*_fp16.onnx` | FP16 变体（导出脚本生成） |

### 导出 ONNX

```bash
cd kfs_core
python scripts/export_onnx.py
python scripts/export_onnx.py --pt /path/to/best.pt --output models/my_model.onnx --imgsz 640
python scripts/export_fp16.py   # 如需 FP16
```

---

## 相机标定

```bash
python3 kfs_core/scripts/calibrate_camera.py \
  --video kfs_core/assets/calib.mp4 \
  --size 11x8 \
  --square 0.02 \
  --output kfs_core/config/ost.yaml \
  --sample-every 10 \
  --max-samples 20
```

在 YAML 中启用：

```yaml
camera:
  usb:
    calibration_file: "kfs_core/config/ost.yaml"
    undistort: true
```

---

## 目录结构

```
realsense_inference/
├── README.md
├── kfs_core/
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── cmake/FindOnnxRuntime.cmake
│   ├── include/
│   │   ├── yolo_detector.h            # YOLO ONNX + Detection / FrameResult
│   │   ├── lightbar_detector.h        # 灯条 OpenCV 检测器
│   │   ├── usb_capture.h / rs_capture.h
│   │   └── kfs_core/
│   │       ├── config.h               # Config / LightbarConfig / YAML
│   │       ├── icamera_capture.h
│   │       └── camera_factory.h
│   ├── src/
│   │   ├── kfs_infer_node.cpp         # ROS2 节点
│   │   ├── main.cpp                   # CLI
│   │   ├── yolo_detector.cpp
│   │   ├── lightbar_detector.cpp
│   │   ├── config.cpp / camera_factory.cpp
│   │   └── usb_capture.cpp / rs_capture.cpp
│   ├── msg/  InferResult.msg  InferResults.msg
│   ├── srv/  SetInferState.srv
│   ├── config/
│   │   ├── kfs_config.yaml
│   │   └── ost.yaml
│   ├── models/                        # ONNX 权重
│   ├── scripts/
│   │   ├── export_onnx.py / export_fp16.py
│   │   └── calibrate_camera.py
│   └── assets/
│       ├── lightbar/                  # 灯条测试图 (红/绿/蓝/黄)
│       ├── lightbar_debug/            # 标注与掩膜参考
│       ├── *.mp4 / *.png              # 其它测试素材
│       └── yolo_dataset/              # 训练数据样例
├── build/  install/  log/             # colcon 产物
```

---

## 集成到其它 ROS2 包

```cmake
find_package(kfs_core REQUIRED)
target_link_libraries(your_node kfs_core_lib)  # 或导出目标 kfs::kfs_core（视安装导出而定）
```

```cpp
#include "kfs_core/config.h"
#include "kfs_core/camera_factory.h"
#include "yolo_detector.h"
#include "lightbar_detector.h"

auto cfg = kfs::loadConfig("kfs_core/config/kfs_config.yaml");
auto cam = kfs::CameraFactory::create(cfg);

std::unique_ptr<YoloDetector> yolo;
if (cfg.detectorType != "lightbar")
    yolo = std::make_unique<YoloDetector>(cfg.model);

std::unique_ptr<LightbarDetector> lightbar;
if (cfg.enableLightbarDetector || cfg.detectorType == "lightbar")
    lightbar = std::make_unique<LightbarDetector>(cfg.lightbar);

cam->start();
cv::Mat frame;
while (/* ... */) {
    if (!cam->getFrame(frame) || frame.empty()) continue;

    FrameResult result;
    if (yolo) result = yolo->detect(frame);
    if (lightbar) {
        auto lb = lightbar->detect(frame);
        result.detections.insert(result.detections.end(),
            lb.detections.begin(), lb.detections.end());
        result.inference_ms += lb.inference_ms;
    }
    // result.detections: R1/T/F, WEAPONHEAD, LIGHTBAR_*
}
```

---

## 常见问题

| 现象 | 处理 |
|------|------|
| `libkfs_core_lib.so: cannot open shared object` | `export LD_LIBRARY_PATH=.../install/kfs_core/lib:$LD_LIBRARY_PATH` |
| 无 USB 相机 | `camera.type: video` + 本地 mp4，或 `--image` 测图 |
| 仅测灯条仍加载 YOLO | `detector.type: lightbar` |
| 黄灯框过大 | 提高 `max_short_side` 约束已内置；可调高 `core_thresh_min` / 查看 `save_debug_mask` |
| 斜向灯条漏检 | 已支持任意角度共线；检查 `min_length_ratio`、`max_angle_diff` |
| CUDA 不可用 | `use_cuda: false`，自动回退 CPU |

---

## License

MIT © FJUTMTI
