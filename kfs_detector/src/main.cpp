/**
 * @file main.cpp
 * @brief KFS 目标检测 — CLI Demo (薄层, 核心逻辑在 libkfs_core)
 *
 * 用法:
 *   ./kfs_detect                                    # 使用默认配置
 *   ./kfs_detect --config my_config.yaml            # 指定配置
 *   ./kfs_detect --list-cameras                    # 查看可用摄像头
 *
 * 按键 (debug 模式):
 *   q / ESC — 退出
 *   s      — 截图保存
 *   SPACE  — 单次推理
 *   d      — 切换持续推理
 */

#include "kfs_core/config.h"
#include "kfs_core/camera_factory.h"
#include "kfs_core/icamera_capture.h"
#include "yolo_detector.h"
#include "weaponhead_detector.h"
#include "usb_capture.h"       // USBCapture::listDevices / listResolutions (静态方法)

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <csignal>
#include <atomic>
#include <thread>
#include <cstdio>
#include <memory>
#include <filesystem>
#include <algorithm>

// ============================================================
// 全局标志
// ============================================================
static std::atomic<bool> g_running{true};

static void signalHandler(int) {
    g_running = false;
}

// ============================================================
// 命令行解析
// ============================================================

static void printUsage() {
    std::cout << "用法: ./kfs_detect [--config PATH] [--list-cameras] [--image PATH] [--help]\n\n"
              << "== 选项 ==\n"
              << "  --config PATH       指定配置文件 (默认: config/kfs_config.yaml)\n"
              << "  --list-cameras      列出所有 USB 摄像头及支持的分辨率\n"
              << "  --image PATH        使用静态图片/目录测试 (无相机)。\n"
              << "                      - 单文件: --image assets/01.png\n"
              << "                      - 目录:   --image assets/   (测试所有图片)\n"
              << "                      每次运行会自动创建 /tmp/kfs_weaponhead_test_YYYYMMDD_HHMMSS/ \n"
              << "                      并在其中保存带标注的 _marked.png 图像 (左右边界用红线+文字标出)\n"
              << "                      (建议在配置中设置 detector.enable_weaponhead: true 及 display.debug: true)\n"
              << "  --help / -h         显示帮助\n\n"
              << "== 配置文件格式 ==\n"
              << "  所有参数统一在 YAML 文件中设置: config/kfs_config.yaml\n"
              << "  包含: 相机类型/分辨率/帧率/编码/V4L2控制/模型参数/类别名/显示 + detector.enable_weaponhead\n\n"
              << "== ROS2 集成 ==\n"
              << "  此 CLI 仅为 Demo。ROS2 节点应直接链接 libkfs_core:\n"
              << "    target_link_libraries(your_ros2_node kfs::kfs_core)\n"
              << "  然后使用 kfs::Config + kfs::CameraFactory + YoloDetector (+ WeaponheadDetector) 即可。\n\n";
    exit(0);
}

// ============================================================
// 检测模式 (CLI 交互专用)
// ============================================================
enum class DetectMode {
    IDLE,          // 仅显示视频流, 不推理
    SINGLE_SHOT,   // 触发一次推理, 完成后回到 IDLE
    CONTINUOUS     // 持续推理每一帧
};

static const char* modeLabel(DetectMode m) {
    switch (m) {
        case DetectMode::IDLE:       return "IDLE (press SPACE to detect)";
        case DetectMode::SINGLE_SHOT:return "SINGLE-SHOT";
        case DetectMode::CONTINUOUS: return "CONTINUOUS";
    }
    return "";
}

// ============================================================
// Debug 可视化
// ============================================================

// 类别对应颜色 — 动态生成, 支持任意数量的类别
static cv::Scalar classColor(int classId) {
    // 使用色相环均匀分布颜色 (BGR)
    static const cv::Scalar PALETTE[] = {
        cv::Scalar(0, 255, 0),    // 绿色
        cv::Scalar(255, 0, 0),    // 蓝色
        cv::Scalar(0, 0, 255),    // 红色
        cv::Scalar(0, 255, 255),  // 黄色
        cv::Scalar(255, 0, 255),  // 品红
        cv::Scalar(255, 255, 0),  // 青色
        cv::Scalar(128, 0, 255),  // 紫色
        cv::Scalar(255, 128, 0),  // 橙色
    };
    constexpr int N = sizeof(PALETTE) / sizeof(PALETTE[0]);
    return PALETTE[classId % N];
}

static void drawDebug(cv::Mat& frame, const FrameResult& result, double fps,
                      DetectMode mode) {
    const int thickness = 2;
    const double fontScale = 0.7;
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;

    for (const auto& det : result.detections) {
        const auto& color = classColor(det.class_id);

        // 绘制 4 条边 (用 4 个角点)
        std::vector<cv::Point> corners = {
            cv::Point(static_cast<int>(det.corner_tl.x), static_cast<int>(det.corner_tl.y)),
            cv::Point(static_cast<int>(det.corner_tr.x), static_cast<int>(det.corner_tr.y)),
            cv::Point(static_cast<int>(det.corner_br.x), static_cast<int>(det.corner_br.y)),
            cv::Point(static_cast<int>(det.corner_bl.x), static_cast<int>(det.corner_bl.y)),
        };
        cv::polylines(frame, corners, true, color, thickness);

        // 绘制 4 个角点小圆
        for (const auto& pt : corners) {
            cv::circle(frame, pt, 4, color, -1);
        }

        // 标签文本
        std::ostringstream label;
        label << det.class_name << " " << std::fixed << std::setprecision(2)
              << det.confidence;

        cv::Size textSz = cv::getTextSize(label.str(), fontFace, fontScale, 1, nullptr);
        int labelY = std::max(corners[0].y - 8, textSz.height + 4);

        cv::rectangle(frame,
                      cv::Point(corners[0].x, labelY - textSz.height - 4),
                      cv::Point(corners[0].x + textSz.width + 4, labelY + 2),
                      color, -1);
        cv::putText(frame, label.str(),
                    cv::Point(corners[0].x + 2, labelY),
                    fontFace, fontScale, cv::Scalar(255, 255, 255), 1);
    }

    // 左上角状态信息
    std::ostringstream ss;
    ss << "FPS: " << std::fixed << std::setprecision(1) << fps
       << " | Inference: " << result.inference_ms << " ms"
       << " | Detections: " << result.detections.size();
    cv::putText(frame, ss.str(),
                cv::Point(10, 30), fontFace, 0.6,
                cv::Scalar(0, 255, 255), 1);

    // 底部: 模式 + 按键提示
    cv::Scalar modeColor = (mode == DetectMode::IDLE) ? cv::Scalar(100, 200, 100)
                           : (mode == DetectMode::CONTINUOUS) ? cv::Scalar(0, 200, 255)
                           : cv::Scalar(0, 255, 255);
    cv::putText(frame, modeLabel(mode),
                cv::Point(10, frame.rows - 36), fontFace, 0.55,
                modeColor, 1);
    cv::putText(frame, "SPACE: detect once | D: toggle continuous | S: screenshot | Q/ESC: quit",
                cv::Point(10, frame.rows - 12), fontFace, 0.45,
                cv::Scalar(200, 200, 200), 1);
}

// ============================================================
// 终端输出
// ============================================================

static void printResults(const FrameResult& result) {
    std::cout << "\033[2J\033[H";  // 清屏
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "  KFS RealSense D415 检测结果\n";
    std::cout << "  帧尺寸: " << result.frame_size.width
              << "×" << result.frame_size.height
              << " | 推理: " << std::fixed << std::setprecision(1)
              << result.inference_ms << " ms\n";
    std::cout << "───────────────────────────────────────────\n";

    if (result.detections.empty()) {
        std::cout << "  (无检测)\n";
    } else {
        std::cout << "  # | 类别 | 置信度 | 左上角点 (x,y) | 右下角点 (x,y)\n";
        std::cout << "───────────────────────────────────────────\n";
        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto& d = result.detections[i];
            std::cout << "  " << std::setw(2) << i
                      << " | " << std::setw(4) << d.class_name
                      << " | " << std::fixed << std::setprecision(2)
                      << std::setw(5) << d.confidence
                      << " | (" << std::setw(4) << static_cast<int>(d.corner_tl.x)
                      << ","  << std::setw(4) << static_cast<int>(d.corner_tl.y)
                      << ") | (" << std::setw(4) << static_cast<int>(d.corner_br.x)
                      << ","  << std::setw(4) << static_cast<int>(d.corner_br.y)
                      << ")\n";
        }
    }
    std::cout << "═══════════════════════════════════════════\n";
}

// ============================================================
// 主函数 (CLI Demo 薄层)
// ============================================================

int main(int argc, char** argv) {
    // 重定向 stderr 抑制 libjpeg "Corrupt JPEG data" 警告
    if (freopen("/dev/null", "w", stderr) == nullptr) { /* 忽略 */ }

    // 信号处理
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);

    // 解析命令行
    // 默认使用新结构下的路径 (从工作空间根目录运行时有效)
    std::string configPath = "kfs_detector/config/kfs_config.yaml";
    std::string testImagePath;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            configPath = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            testImagePath = argv[++i];
        } else if (arg == "--list-cameras") {
            std::cout << "═══════════════════════════════════════════\n";
            std::cout << "  系统 USB 摄像头列表\n";
            std::cout << "═══════════════════════════════════════════\n\n";
            auto devices = USBCapture::listDevices();
            if (devices.empty()) {
                std::cout << "  未检测到 USB 摄像头\n\n";
                std::cout << "  提示:\n";
                std::cout << "    - 检查 USB 连接: ls /dev/video*\n";
                std::cout << "    - 检查权限: sudo usermod -aG video $USER\n";
            } else {
                for (int dev : devices) {
                    std::cout << "  /dev/video" << dev << ":\n";
                    auto resList = USBCapture::listResolutions(dev);
                    if (resList.empty()) {
                        std::cout << "    (无法获取分辨率列表)\n";
                    } else {
                        for (const auto& r : resList) {
                            std::cout << "    - " << r.width << "×" << r.height
                                      << " @ " << static_cast<int>(r.fps) << " fps\n";
                        }
                    }
                }
            }
            std::cout << "\n═══════════════════════════════════════════\n";
            exit(0);
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
        } else {
            std::cout << "[WARN] 未知参数: " << arg << " (用 --help 查看用法)\n";
        }
    }

    // 加载 YAML 配置 (来自 kfs_core)
    kfs::Config cfg = kfs::loadConfig(configPath);
    std::cout << "[CONFIG] 已加载: " << configPath
              << " | detector: " << cfg.detectorType << std::endl;

    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║   KFS 目标检测 — CLI Demo                ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    // 1) YOLO 始终加载 (主流程), weaponhead_detector 作为松耦合附加 (并行, 不同目标)
    std::unique_ptr<YoloDetector> yolo_det = std::make_unique<YoloDetector>(cfg.model);
    std::cout << "[INFO] 模型类别: ";
    for (const auto& name : yolo_det->classNames()) {
        std::cout << name << " ";
    }
    std::cout << std::endl;

    std::unique_ptr<WeaponheadDetector> wh_det;
    if (cfg.enableWeaponheadDetector || cfg.detectorType == "weaponhead_detector") {
        wh_det = std::make_unique<WeaponheadDetector>();
        std::cout << "[INFO] weaponhead_detector 已启用 (传统 OpenCV 并行) — 检测虚焦物体左右边界" << std::endl;
    }

    // 2) 创建相机 (通过 CameraFactory, 返回 ICameraCapture 接口)
    std::string windowName;

    // === 图片测试模式 (无相机, 专为验证 weaponhead_detector 左右边界) ===
    // 支持单文件或目录；自动创建 /tmp/kfs_weaponhead_test_时间戳/ 并保存带标注的图像
    if (!testImagePath.empty()) {
        std::vector<std::string> image_paths;
        if (std::filesystem::is_directory(testImagePath)) {
            for (const auto& entry : std::filesystem::directory_iterator(testImagePath)) {
                if (entry.is_regular_file()) {
                    auto p = entry.path();
                    std::string ext = p.extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                        image_paths.push_back(p.string());
                    }
                }
            }
            std::sort(image_paths.begin(), image_paths.end());
        } else if (std::filesystem::exists(testImagePath)) {
            image_paths.push_back(testImagePath);
        }

        if (image_paths.empty()) {
            std::cerr << "[ERROR] 无效的 --image 路径 (不是图片或目录内无图片): " << testImagePath << std::endl;
            return 1;
        }

        // 创建唯一临时测试文件夹
        auto now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::stringstream ts_ss;
        ts_ss << std::put_time(std::localtime(&now_time), "%Y%m%d_%H%M%S");
        std::string test_dir = "/tmp/kfs_weaponhead_test_" + ts_ss.str();
        std::filesystem::create_directories(test_dir);
        std::cout << "[INFO] 图片测试模式启动\n";
        std::cout << "       输入: " << testImagePath << " (共 " << image_paths.size() << " 张图)\n";
        std::cout << "       临时输出文件夹: " << test_dir << std::endl;

        for (const auto& img_path : image_paths) {
            cv::Mat frame = cv::imread(img_path);
            if (frame.empty()) {
                std::cerr << "[WARN] 跳过无法读取的图片: " << img_path << std::endl;
                continue;
            }

            FrameResult result;
            if (yolo_det) {
                result = yolo_det->detect(frame);
            }
            if (wh_det) {
                auto wh_res = wh_det->detect(frame);
                for (auto& d : wh_res.detections) {
                    result.detections.push_back(std::move(d));
                }
                if (result.inference_ms <= 0) result.inference_ms = wh_res.inference_ms;
                else result.inference_ms += wh_res.inference_ms;
            }

            // 特别输出 weaponhead 的左右边界 (用户确认用)
            std::cout << "\n═══════════════════════════════════════════\n";
            std::cout << "  WeaponheadDetector 测试结果\n";
            std::cout << "  文件: " << img_path << "  (" << frame.cols << "×" << frame.rows << ")\n";
            std::cout << "───────────────────────────────────────────\n";
            bool found_wh = false;
            for (const auto& d : result.detections) {
                if (d.class_name == "OBJ" || d.class_name == "weaponhead") {
                    found_wh = true;
                    int left  = static_cast<int>(std::round(d.corner_tl.x));
                    int right = static_cast<int>(std::round(d.corner_br.x));
                    int top   = static_cast<int>(std::round(d.corner_tl.y));
                    int bot   = static_cast<int>(std::round(d.corner_br.y));
                    std::cout << "  [weaponhead] 左右边界像素: left=" << left
                              << "  right=" << right
                              << "  (宽度=" << (right - left) << ")\n";
                    std::cout << "                 垂直带: top=" << top << " bot=" << bot << "\n";
                    std::cout << "                 中心: x=" << (left + right)/2 << "\n";
                }
            }
            if (!found_wh) {
                std::cout << "  (未启用 weaponhead_detector 或未检测到 OBJ)\n";
                std::cout << "  提示: 在 config 中设置 detector.enable_weaponhead: true 再测试\n";
            }
            std::cout << "  总检测数: " << result.detections.size()
                      << " | 耗时: " << result.inference_ms << " ms\n";
            std::cout << "═══════════════════════════════════════════\n";

            // 生成并保存带标注的结果图像 (始终保存到临时文件夹)
            cv::Mat marked = frame.clone();
            drawDebug(marked, result, 0.0, DetectMode::SINGLE_SHOT);

            // 额外用红线 + 文字突出显示 weaponhead 左右边界像素
            for (const auto& d : result.detections) {
                if (d.class_name == "OBJ" || d.class_name == "weaponhead") {
                    int left = static_cast<int>(std::round(d.corner_tl.x));
                    int right = static_cast<int>(std::round(d.corner_br.x));
                    // 红色垂直边界线
                    cv::line(marked, cv::Point(left, 0), cv::Point(left, marked.rows - 1), cv::Scalar(0, 0, 255), 2);
                    cv::line(marked, cv::Point(right, 0), cv::Point(right, marked.rows - 1), cv::Scalar(0, 0, 255), 2);
                    // 像素标签
                    std::string lbl_l = "L:" + std::to_string(left);
                    std::string lbl_r = "R:" + std::to_string(right);
                    cv::putText(marked, lbl_l, cv::Point(left + 8, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                    int r_text_x = std::max(5, right - 90);
                    cv::putText(marked, lbl_r, cv::Point(r_text_x, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                    // 中心水平指示线
                    int mid_y = (static_cast<int>(std::round(d.corner_tl.y)) + static_cast<int>(std::round(d.corner_br.y))) / 2;
                    cv::line(marked, cv::Point(left, mid_y), cv::Point(right, mid_y), cv::Scalar(0, 255, 0), 1);
                }
            }

            std::string base = std::filesystem::path(img_path).stem().string();
            std::string out_path = test_dir + "/" + base + "_marked.png";
            if (cv::imwrite(out_path, marked)) {
                std::cout << "[SAVE] 标注图像已输出: " << out_path << std::endl;
            } else {
                std::cerr << "[WARN] 保存标注图像失败: " << out_path << std::endl;
            }

            // 如果 debug，显示窗口 (方便即时查看)
            if (cfg.display.debug) {
                cv::imshow("KFS Test - " + base, marked);
                std::cout << "[INFO] 按任意键继续下一个 / 关闭窗口...\n";
                cv::waitKey(0);
            }
        }

        if (cfg.display.debug) {
            cv::destroyAllWindows();
        }
        std::cout << "\n[INFO] 所有测试完成。临时文件夹: " << test_dir << std::endl;
        std::cout << "       请检查其中的 *_marked.png 确认 weaponhead 左右边界是否正确标出。\n\n";
        return 0;
    }

    // === 正常相机模式 ===
    auto camera = kfs::CameraFactory::create(cfg);
    if (!camera) {
        std::cerr << "[ERROR] 无法创建相机 (类型: " << cfg.cameraType << ")\n";
        return 1;
    }

    if (!camera->start()) {
        std::cerr << "[ERROR] 无法启动相机\n"
                  << "  请检查: 1) 设备连接  2) 权限 (sudo usermod -aG video $USER)\n"
                  << "  提示: 运行 ./kfs_detect --list-cameras 查看可用摄像头\n";
        return 1;
    }

    windowName = std::string("KFS Detection - ") + cfg.cameraType;
    std::cout << "[INFO] 实际分辨率: "
              << camera->getWidth() << "×"
              << camera->getHeight() << std::endl;

    auto intrinsics = camera->getIntrinsics();
    if (intrinsics.fx > 0) {
        std::cout << "[INFO] 相机内参: fx=" << intrinsics.fx
                  << " fy=" << intrinsics.fy
                  << " cx=" << intrinsics.cx
                  << " cy=" << intrinsics.cy << std::endl;
    }

    // 3) 主循环
    cv::Mat frame;
    int    frameCount = 0;
    auto   tStart     = std::chrono::high_resolution_clock::now();
    double fps        = 0.0;

    DetectMode detectMode = DetectMode::IDLE;
    FrameResult lastResult;

    if (cfg.display.debug) {
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    }

    std::cout << "\n[INFO] 检测器已就绪, 等待触发推理...\n"
              << (cfg.display.debug
                    ? "  按 SPACE 单次推理 | 按 D 持续推理 | Q/ESC 退出\n"
                    : "  按 Ctrl+C 退出\n")
              << "\n";

    while (g_running) {
        if (!g_running) break;
        if (!camera->getFrame(frame) || frame.empty()) {
            if (!g_running) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        bool shouldDetect = (detectMode == DetectMode::CONTINUOUS)
                         || (detectMode == DetectMode::SINGLE_SHOT);

        if (shouldDetect) {
            if (yolo_det) {
                lastResult = yolo_det->detect(frame);
            }
            if (wh_det) {
                auto wh_res = wh_det->detect(frame);
                // 松耦合并: weaponhead 结果追加 (不同目标 OBJ)
                for (auto& d : wh_res.detections) {
                    lastResult.detections.push_back(std::move(d));
                }
                if (lastResult.inference_ms <= 0) lastResult.inference_ms = wh_res.inference_ms;
                else lastResult.inference_ms += wh_res.inference_ms;
            }
            if (detectMode == DetectMode::SINGLE_SHOT) {
                detectMode = DetectMode::IDLE;
            }
        }

        // FPS 计算
        frameCount++;
        {
            auto tFps = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(tFps - tStart).count();
            if (elapsed >= 1.0) {
                fps = frameCount / elapsed;
                frameCount = 0;
                tStart = tFps;
            }
        }

        if (cfg.display.debug) {
            drawDebug(frame, lastResult, fps, detectMode);
            cv::imshow(windowName, frame);

            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {
                g_running = false;
            } else if (key == ' ') {
                detectMode = DetectMode::SINGLE_SHOT;
                std::cout << "[INFO] 触发单次推理" << std::endl;
            } else if (key == 'd' || key == 'D') {
                detectMode = (detectMode == DetectMode::CONTINUOUS)
                           ? DetectMode::IDLE : DetectMode::CONTINUOUS;
                std::cout << "[INFO] 检测模式: " << modeLabel(detectMode) << std::endl;
            } else if (key == 's') {
                auto now = std::chrono::system_clock::now();
                auto ts  = std::chrono::duration_cast<std::chrono::seconds>(
                    now.time_since_epoch()).count();
                std::string fname = "screenshot_" + std::to_string(ts) + ".png";
                cv::imwrite(fname, frame);
                std::cout << "[SAVE] 截图已保存: " << fname << std::endl;
            }
        } else {
            if (shouldDetect) {
                printResults(lastResult);
            }
        }
    }

    // 4) 清理
    camera->stop();
    if (cfg.display.debug) {
        cv::destroyAllWindows();
    }

    std::cout << "\n[INFO] 程序正常退出\n";
    return 0;
}
