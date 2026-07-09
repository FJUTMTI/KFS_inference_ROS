/**
 * @file kfs_infer_node.cpp
 * @brief KFS 推理 ROS2 节点
 *
 * 与其他功能包通信:
 *
 *   [订阅]
 *     ~/enable (std_msgs/Bool)          — 推理使能
 *
 *   [服务]
 *     ~/set_state (kfs_core/SetInferState) — 显式启停推理
 *     ~/trigger  (std_srvs/Trigger)        — 触发单次推理
 *
 *   [发布]
 *     ~/result       (kfs_core/InferResults) — 检测结果（一帧一个数组消息，推荐订阅此话题）
 *     ~/debug_image  (sensor_msgs/Image)     — 标注画面 (rqt/rviz 可视化)
 *     ~/status       (std_msgs/String)       — 状态文本
 *
 *     (旧的单框 InferResult 话题已废弃，改用数组形式)
 *
 *   [参数 — rqt_reconfigure 动态调节]
 *     config_path         — YAML 配置文件
 *     debug_image         — 是否发布 debug 画面
 *     conf_threshold      — 置信度阈值 (0.0~1.0)
 *     iou_threshold       — NMS IOU 阈值 (0.0~1.0)
 *     use_cuda            — 是否使用 CUDA
 *     model_path          — ONNX 模型路径
 *     camera_type         — 相机类型 (usb / realsense)
 *     usb_device          — USB 摄像头设备号
 *     usb_width           — USB 分辨率宽
 *     usb_height          — USB 分辨率高
 *     usb_fps             — USB 帧率
 *     usb_fourcc          — USB 编码 (MJPG / YUYV)
 *     video_path          — 视频文件路径 (camera_type=video 时启用, 代替摄像头；默认 kfs_core/assets/001.mp4)
 *     video_loop          — 视频循环播放
 *     video_calibration_file — 视频对应的标定文件
 *     video_undistort     — 视频帧是否做畸变矫正
 *     detector_type       — 检测器类型 (yolo / lightbar / yolo_lightbar)
 *     enable_weaponhead   — 是否并行启用 weaponhead YOLO
 *     enable_lightbar     — 是否并行启用灯条检测 (传统 OpenCV)
 *     inference_enabled   — 推理使能
 */

#include "kfs_core/config.h"
#include "kfs_core/camera_factory.h"
#include "kfs_core/icamera_capture.h"
#include "yolo_detector.h"
#include "lightbar_detector.h"

// ROS2
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <sensor_msgs/msg/image.hpp>

// 自定义接口
#include "kfs_core/msg/infer_result.hpp"
#include "kfs_core/msg/infer_results.hpp"
#include "kfs_core/srv/set_infer_state.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <memory>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <mutex>

// ============================================================
// 可视化工具 (debug_image 话题用)
// ============================================================

static cv::Scalar classColor(int classId) {
    static const cv::Scalar PALETTE[] = {
        {0, 255, 0}, {255, 0, 0}, {0, 0, 255}, {0, 255, 255},
        {255, 0, 255}, {255, 255, 0}, {128, 0, 255}, {255, 128, 0},
    };
    constexpr int N = sizeof(PALETTE) / sizeof(PALETTE[0]);
    return PALETTE[classId % N];
}

static cv::Mat drawDetections(const cv::Mat& src, const FrameResult& result,
                              double fps, bool inferActive) {
    cv::Mat disp = src.clone();
    if (disp.empty()) return disp;

    const int thickness = 2;
    const double fontScale = 0.7;
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;

    for (const auto& det : result.detections) {
        auto color = classColor(det.class_id);

        std::vector<cv::Point> corners = {
            {int(det.corner_tl.x), int(det.corner_tl.y)},
            {int(det.corner_tr.x), int(det.corner_tr.y)},
            {int(det.corner_br.x), int(det.corner_br.y)},
            {int(det.corner_bl.x), int(det.corner_bl.y)},
        };
        cv::polylines(disp, corners, true, color, thickness);
        for (const auto& pt : corners)
            cv::circle(disp, pt, 4, color, -1);

        std::string label = det.class_name + " "
                          + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        cv::Size textSz = cv::getTextSize(label, fontFace, fontScale, 1, nullptr);
        int labelY = std::max(corners[0].y - 8, textSz.height + 4);
        cv::rectangle(disp,
                      {corners[0].x, labelY - textSz.height - 4},
                      {corners[0].x + textSz.width + 4, labelY + 2},
                      color, -1);
        cv::putText(disp, label,
                    {corners[0].x + 2, labelY},
                    fontFace, fontScale, {255, 255, 255}, 1);

        // weaponhead 特殊高亮：在 ROS debug_image 中实时绘制左右边界红线 + 像素标签
        if (det.class_name == "WEAPONHEAD" || det.class_name == "OBJ" || det.class_name == "weaponhead") {
            int left = static_cast<int>(std::round(det.corner_tl.x));
            int right = static_cast<int>(std::round(det.corner_br.x));
            // 红色垂直边界线 (全高度)
            cv::line(disp, cv::Point(left, 0), cv::Point(left, disp.rows - 1), cv::Scalar(0, 0, 255), 2);
            cv::line(disp, cv::Point(right, 0), cv::Point(right, disp.rows - 1), cv::Scalar(0, 0, 255), 2);
            // 顶部像素标签
            std::string lbl_l = "L:" + std::to_string(left);
            std::string lbl_r = "R:" + std::to_string(right);
            cv::putText(disp, lbl_l, cv::Point(left + 8, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            int r_text_x = std::max(5, right - 90);
            cv::putText(disp, lbl_r, cv::Point(r_text_x, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            // 物体带中心水平指示线
            int mid_y = (static_cast<int>(std::round(det.corner_tl.y)) + static_cast<int>(std::round(det.corner_br.y))) / 2;
            cv::line(disp, cv::Point(left, mid_y), cv::Point(right, mid_y), cv::Scalar(0, 255, 0), 1);
        }
    }

    // 状态带
    std::string status = "FPS:" + std::to_string(int(fps))
                       + " Inf:" + std::to_string(int(result.inference_ms)) + "ms"
                       + " Det:" + std::to_string(result.detections.size())
                       + (inferActive ? " ON" : " IDLE");
    cv::putText(disp, status, {10, 30}, fontFace, 0.6, {0, 255, 255}, 1);

    return disp;
}

// ============================================================
// KfsInferNode
// ============================================================

class KfsInferNode : public rclcpp::Node {
public:
    KfsInferNode() : Node("kfs_infer_node") {
        // 默认路径: 从工作空间根运行时使用的相对路径
        std::string config_default = "kfs_core/config/kfs_config.yaml";
        std::string model_default  = "kfs_core/models/kfs_yolo11_3class.onnx";
        // 安装后优先使用 share 目录路径
        try {
            auto share_dir = ament_index_cpp::get_package_share_directory("kfs_core");
            config_default = share_dir + "/config/kfs_config.yaml";
            model_default  = share_dir + "/models/kfs_yolo11_3class.onnx";
        } catch (const std::exception&) {
            // 未安装时回退到相对路径 (从工作空间根运行)
        }

        // 尽早声明 config_path（允许命令行 -p config_path:=... 覆盖）
        declareParam<std::string>("config_path",        config_default);

        // 预加载 YAML 以便将 YAML 中的值作为其他参数的 declare 默认值。
        // 这样用户只需在 kfs_config.yaml 中把 camera.type 设为 video 即可生效（ROS 节点也会尊重它），
        // 同时命令行 -p camera_type:=xxx 等仍可覆盖。
        // 如果未提供 config_path 或加载失败，则回退到硬编码默认。
        kfs::Config seed_cfg;
        bool seed_loaded = false;
        {
            std::string yaml_for_seed = config_default;
            try {
                rclcpp::Parameter p;
                if (this->get_parameter("config_path", p)) {
                    yaml_for_seed = p.as_string();
                }
            } catch (const std::exception&) {}
            if (!yaml_for_seed.empty()) {
                try {
                    seed_cfg = kfs::loadConfig(yaml_for_seed);
                    seed_loaded = true;
                } catch (const std::exception& e) {
                    RCLCPP_WARN(this->get_logger(), "用于参数默认值的 YAML 加载失败 (%s)，将使用内置默认值", e.what());
                }
            }
        }

        // ---- 声明全部参数 (支持 rqt_reconfigure 动态调节) ----
        // 大部分默认值来自 seed_cfg（即 YAML），以便配置文件修改立即影响节点默认行为。
        declareParam<bool>       ("debug_image",         true);
        declareParam<float>      ("conf_threshold",      seed_loaded ? seed_cfg.model.confThresh : 0.25f, 0.0f, 1.0f);
        declareParam<float>      ("iou_threshold",       seed_loaded ? seed_cfg.model.iouThresh : 0.30f, 0.0f, 1.0f);
        declareParam<bool>       ("use_cuda",            seed_loaded ? seed_cfg.model.useCUDA : true);
        declareParam<std::string>("model_path",          seed_loaded ? seed_cfg.model.path : model_default);
        declareParam<std::string>("camera_type",         "");  // 空=跟随YAML, 非空=覆盖YAML
        declareParam<int>        ("usb_device",          seed_loaded ? seed_cfg.usb.device : 0, 0, 63);
        declareParam<int>        ("usb_width",           seed_loaded ? seed_cfg.usb.width : 640, 160, 3840);
        declareParam<int>        ("usb_height",          seed_loaded ? seed_cfg.usb.height : 480, 120, 2160);
        declareParam<int>        ("usb_fps",             seed_loaded ? seed_cfg.usb.fps : 60, 1, 240);
        declareParam<std::string>("usb_fourcc",          seed_loaded ? seed_cfg.usb.fourcc : "MJPG");
        declareParam<std::string>("video_path",          seed_loaded ? seed_cfg.video.path : "kfs_core/assets/001.mp4");
        declareParam<bool>       ("video_loop",          seed_loaded ? seed_cfg.video.loop : true);
        declareParam<std::string>("video_calibration_file", seed_loaded ? seed_cfg.video.calibration_file : "");
        declareParam<bool>       ("video_undistort",     seed_loaded ? seed_cfg.video.undistort : false);
        declareParam<bool>       ("inference_enabled",   true);
        declareParam<int>        ("input_size",          seed_loaded ? seed_cfg.model.inputSize : 640, 320, 1280);
        declareParam<std::string>("detector_type",       seed_loaded ? seed_cfg.detectorType : "yolo");  // "yolo" | "lightbar" | "yolo_lightbar"
        declareParam<bool>       ("enable_weaponhead",   seed_loaded ? seed_cfg.enableWeaponheadDetector : true);
        declareParam<bool>       ("enable_lightbar",     seed_loaded ? seed_cfg.enableLightbarDetector : false);

        // weaponhead YOLO 模型 (与3class并行推理)
        declareParam<std::string>("wh_model_path",       seed_loaded ? seed_cfg.weaponhead.whModelPath : "");

        // 类别名作为 string 列表(逗号分隔), rqt 字符串参数编辑
        {
            std::string class_names_def = "R1,T,F";
            if (seed_loaded && !seed_cfg.model.classNames.empty()) {
                std::ostringstream oss;
                for (size_t i = 0; i < seed_cfg.model.classNames.size(); ++i) {
                    if (i > 0) oss << ",";
                    oss << seed_cfg.model.classNames[i];
                }
                class_names_def = oss.str();
            }
            declareParam<std::string>("class_names", class_names_def);
        }

        // ---- 加载 YAML 作为初始值, 然后 ROS 参数覆盖 ----
        // 注意: 即使相机创建/启动失败 (例如无硬件摄像头)，也继续初始化节点。
        // 这样可以通过 ros2 param set / ros2 param load / rqt_reconfigure 动态切换
        // camera_type + video_path (或 usb_* 参数) 来恢复/切换输入源，无需重启节点。
        initCore();

        // ---- 发布者 ----
        pub_results_    = this->create_publisher<kfs_core::msg::InferResults>("~/result", 10);
        pub_debug_      = this->create_publisher<sensor_msgs::msg::Image>("~/debug_image", 10);
        pub_status_     = this->create_publisher<std_msgs::msg::String>("~/status", 10);

        // ---- 订阅者 ----
        sub_enable_ = this->create_subscription<std_msgs::msg::Bool>(
            "~/enable", 10,
            [this](std_msgs::msg::Bool::ConstSharedPtr msg) {
                inference_enabled_.store(msg->data);
            });

        // ---- 服务 ----
        srv_set_state_ = this->create_service<kfs_core::srv::SetInferState>(
            "~/set_state",
            [this](const std::shared_ptr<kfs_core::srv::SetInferState::Request> req,
                   std::shared_ptr<kfs_core::srv::SetInferState::Response> res) {
                inference_enabled_.store(req->enable);
                res->success = true;
                res->message = req->enable ? "推理已启动" : "推理已停止";
            });

        srv_trigger_ = this->create_service<std_srvs::srv::Trigger>(
            "~/trigger",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
                trigger_single_shot_.store(true);
                res->success = true;
                res->message = "单次推理已触发";
            });

        srv_snapshot_ = this->create_service<std_srvs::srv::Trigger>(
            "~/snapshot",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
                std::lock_guard<std::mutex> lock(mtx_);
                if (last_raw_frame_.empty()) {
                    res->success = false;
                    res->message = "尚无帧数据";
                    return;
                }
                // 1) 原始帧
                cv::imwrite("/tmp/snapshot_raw.png", last_raw_frame_);
                // 2) 标注画面 (含检测框、FPS、状态)
                if (!last_debug_image_.empty()) {
                    cv::imwrite("/tmp/snapshot_debug.png", last_debug_image_);
                }
                // 3) 诊断分析图 (梯度热力图)
                cv::Mat diag = buildDiagnosticImage(last_raw_frame_);
                if (!diag.empty()) {
                    cv::imwrite("/tmp/snapshot_diag.png", diag);
                }
                // 4) 检测结果文本 (等同于 ~/result 话题内容)
                std::ostringstream oss;
                oss << "frame: " << last_result_.frame_size.width
                    << "x" << last_result_.frame_size.height << "\n";
                oss << "inference_ms: " << last_result_.inference_ms << "\n";
                oss << "detections: " << last_result_.detections.size() << "\n";
                for (size_t i = 0; i < last_result_.detections.size(); ++i) {
                    const auto& d = last_result_.detections[i];
                    float cx_d = (d.corner_tl.x + d.corner_tr.x + d.corner_br.x + d.corner_bl.x) * 0.25f;
                    float cy_d = (d.corner_tl.y + d.corner_tr.y + d.corner_br.y + d.corner_bl.y) * 0.25f;
                    oss << "  [" << i << "] class=" << d.class_name
                        << " id=" << d.class_id
                        << " conf=" << d.confidence
                        << " center=(" << int(cx_d) << "," << int(cy_d) << ")"
                        << " bbox=" << int(d.corner_br.x - d.corner_tl.x)
                        << "x" << int(d.corner_br.y - d.corner_tl.y)
                        << " tl=(" << int(d.corner_tl.x) << "," << int(d.corner_tl.y) << ")"
                        << " br=(" << int(d.corner_br.x) << "," << int(d.corner_br.y) << ")"
                        << "\n";
                }
                oss << "fps: " << int(fps_) << "\n";
                {
                    std::ofstream ofs("/tmp/snapshot_result.txt");
                    ofs << oss.str();
                }
                std::string msg = "OK | /tmp/snapshot_raw.png /tmp/snapshot_debug.png /tmp/snapshot_diag.png /tmp/snapshot_result.txt";
                res->success = true;
                res->message = msg;
            });

        // ---- 动态参数回调 ----
        param_cb_handle_ = this->add_on_set_parameters_callback(
            [this](const std::vector<rclcpp::Parameter>& params) {
                return onParamChange(params);
            });

        // ---- 定时器: 主循环 (30Hz) ----
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            [this]() { loopOnce(); });

        size_t cls_count = yolo_detector_ ? yolo_detector_->classNames().size() : 0;
        RCLCPP_INFO(this->get_logger(),
                    "节点启动 | 相机:%s | detector:%s | 类别/模式:%zu | debug_image:%s",
                    cfg_.cameraType.c_str(),
                    cfg_.detectorType.c_str(),
                    cls_count,
                    pub_debug_->get_subscription_count() > 0 ? "ON" : "OFF");
    }

    ~KfsInferNode() override {
        if (camera_) camera_->stop();
    }

private:
    // ----------------------------------------------------------
    // 参数声明辅助
    // ----------------------------------------------------------
    template<typename T>
    void declareParam(const std::string& name, T defaultVal) {
        this->declare_parameter<T>(name, defaultVal);
    }
    template<typename T>
    void declareParam(const std::string& name, T defaultVal, T lo, T hi) {
        rcl_interfaces::msg::IntegerRange ir;
        rcl_interfaces::msg::FloatingPointRange fr;
        auto desc = rcl_interfaces::msg::ParameterDescriptor{};
        if constexpr (std::is_integral_v<T>) {
            ir.from_value = static_cast<int64_t>(lo);
            ir.to_value   = static_cast<int64_t>(hi);
            ir.step       = 1;
            desc.integer_range = {ir};
        } else {
            fr.from_value = static_cast<double>(lo);
            fr.to_value   = static_cast<double>(hi);
            fr.step       = 0.0;
            desc.floating_point_range = {fr};
        }
        this->declare_parameter<T>(name, defaultVal, desc);
    }

    // ----------------------------------------------------------
    // 从参数 (YAML + ROS 覆盖) 构建 Config
    // ----------------------------------------------------------
    kfs::Config buildConfig() {
        kfs::Config cfg;

        // 先加载 YAML
        auto yamlPath = getParam<std::string>("config_path");
        if (!yamlPath.empty()) {
            try {
                cfg = kfs::loadConfig(yamlPath);
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "YAML 加载失败, 使用所有默认值: %s", e.what());
            }
        }

        // ROS 参数覆盖 (优先): 仅当 ROS 参数非空/非默认时才覆盖 YAML
        {
            auto camType = getParam<std::string>("camera_type");
            if (!camType.empty()) cfg.cameraType = camType;
        }
        cfg.usb.device               = getParam<int>("usb_device");
        cfg.usb.width                = getParam<int>("usb_width");
        cfg.usb.height               = getParam<int>("usb_height");
        cfg.usb.fps                  = getParam<int>("usb_fps");
        cfg.usb.fourcc               = getParam<std::string>("usb_fourcc");
        cfg.video.path               = getParam<std::string>("video_path");
        cfg.video.loop               = getParam<bool>("video_loop");
        cfg.video.calibration_file   = getParam<std::string>("video_calibration_file");
        cfg.video.undistort          = getParam<bool>("video_undistort");
        cfg.model.path               = getParam<std::string>("model_path");
        cfg.model.inputSize          = getParam<int>("input_size");
        cfg.model.confThresh         = getParam<float>("conf_threshold");
        cfg.model.iouThresh          = getParam<float>("iou_threshold");
        cfg.model.useCUDA            = getParam<bool>("use_cuda");
        cfg.display.debug            = false; // 无头模式

        cfg.detectorType             = getParam<std::string>("detector_type");
        cfg.enableWeaponheadDetector = getParam<bool>("enable_weaponhead");
        cfg.enableLightbarDetector   = getParam<bool>("enable_lightbar");
        if (cfg.detectorType == "lightbar" || cfg.detectorType == "yolo_lightbar") {
            cfg.enableLightbarDetector = true;
        }
        cfg.lightbar.enabled = cfg.enableLightbarDetector;

        // weaponhead ROS 参数覆盖 (优先于 YAML)
        cfg.weaponhead.whModelPath      = getParam<std::string>("wh_model_path");

        // 类别名解析
        auto clsStr = getParam<std::string>("class_names");
        cfg.model.classNames.clear();
        std::istringstream iss(clsStr);
        std::string token;
        while (std::getline(iss, token, ',')) {
            if (!token.empty()) cfg.model.classNames.push_back(token);
        }
        if (cfg.model.classNames.empty())
            cfg.model.classNames = {"R1", "T", "F"};

        inference_enabled_.store(getParam<bool>("inference_enabled"));
        return cfg;
    }

    template<typename T>
    T getParam(const std::string& name) {
        return this->get_parameter(name).get_value<T>();
    }

    // ----------------------------------------------------------
    // 初始化核心模块
    // ----------------------------------------------------------
    bool initCore() {
        cfg_ = buildConfig();

        // 检测器始终创建，即使相机暂不可用
        rebuildDetectors();

        camera_ = kfs::CameraFactory::create(cfg_);
        if (!camera_) {
            RCLCPP_ERROR(this->get_logger(),
                "相机创建失败 (类型: %s)。节点将继续运行，可通过动态参数切换输入源。",
                cfg_.cameraType.c_str());
        } else if (!camera_->start()) {
            RCLCPP_ERROR(this->get_logger(),
                "相机启动失败 (类型: %s)。\n"
                "  常见原因: 无 USB 摄像头、/dev/video0 不可用、权限不足 (需加入 video 组)。\n"
                "  调试/验证建议 (使用视频文件代替摄像头):\n"
                "    ros2 param set /kfs_infer_node camera_type video\n"
                "    ros2 param set /kfs_infer_node video_path kfs_core/assets/001.mp4\n"
                "  启动时指定: ros2 run kfs_core kfs_infer_node --ros-args -p camera_type:=video",
                cfg_.cameraType.c_str());
            camera_.reset();
        } else {
            RCLCPP_INFO(this->get_logger(),
                        "相机:%dx%d | YOLO:%s | whYOLO:%s | lightbar:%s | conf:%.2f iou:%.2f cuda:%d",
                        camera_->getWidth(), camera_->getHeight(),
                        (yolo_detector_ ? cfg_.model.path.c_str() : "OFF"),
                        (wh_yolo_detector_ ? "ON" : "OFF"),
                        (lightbar_detector_ ? "ON" : "OFF"),
                        cfg_.model.confThresh, cfg_.model.iouThresh, cfg_.model.useCUDA);
        }

        return true;
    }

    void rebuildDetectors() {
        yolo_detector_.reset();
        wh_yolo_detector_.reset();
        lightbar_detector_.reset();

        const bool use_yolo = (cfg_.detectorType != "lightbar");
        const bool use_lightbar = cfg_.enableLightbarDetector
            || cfg_.detectorType == "lightbar"
            || cfg_.detectorType == "yolo_lightbar";

        if (use_yolo) {
            yolo_detector_ = std::make_unique<YoloDetector>(cfg_.model);
            if (!cfg_.weaponhead.whModelPath.empty()) {
                std::vector<std::string> whClasses = {"WEAPONHEAD"};
                wh_yolo_detector_ = std::make_unique<YoloDetector>(
                    cfg_.weaponhead.whModelPath, whClasses,
                    cfg_.model.inputSize,
                    cfg_.model.confThresh, cfg_.model.iouThresh,
                    cfg_.model.useCUDA);
            }
        }
        if (use_lightbar) {
            lightbar_detector_ = std::make_unique<LightbarDetector>(cfg_.lightbar);
        }
    }

    // ----------------------------------------------------------
    // 动态参数回调 (rqt_reconfigure)
    // ----------------------------------------------------------
    rcl_interfaces::msg::SetParametersResult onParamChange(
        const std::vector<rclcpp::Parameter>& params)
    {
        bool needRebuildDetector = false;
        bool needRebuildCamera   = false;

        for (const auto& p : params) {
            auto n = p.get_name();
            if (n == "conf_threshold" || n == "iou_threshold" ||
                n == "use_cuda" || n == "model_path" ||
                n == "input_size" || n == "class_names" ||
                n == "wh_model_path") {
                needRebuildDetector = true;
            }
            if (n == "camera_type" || n == "usb_device" || n == "usb_width" ||
                n == "usb_height" || n == "usb_fps" || n == "usb_fourcc" ||
                n == "video_path" || n == "video_loop" || n == "video_calibration_file" ||
                n == "video_undistort") {
                needRebuildCamera = true;
            }
            if (n == "detector_type" || n == "enable_weaponhead" || n == "enable_lightbar"
                || n == "config_path") {
                needRebuildDetector = true;
            }
            if (n == "inference_enabled") {
                inference_enabled_.store(p.get_value<bool>());
            }
            if (n == "config_path") {
                needRebuildCamera = true;
            }
        }

        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;

        // 重建相机 (需要 stop + start)
        if (needRebuildCamera) {
            std::lock_guard<std::mutex> lock(mtx_);
            cfg_ = buildConfig();
            if (camera_) camera_->stop();
            camera_.reset();
            camera_ = kfs::CameraFactory::create(cfg_);
            if (camera_ && camera_->start()) {
                RCLCPP_INFO(this->get_logger(), "相机已重载: %dx%d",
                            camera_->getWidth(), camera_->getHeight());
            } else {
                if (camera_) {
                    RCLCPP_ERROR(this->get_logger(), "相机重载启动失败 (类型: %s)", cfg_.cameraType.c_str());
                    camera_.reset();
                }
                result.successful = false;
                result.reason = "相机重建失败";
            }
        }

        // 重建检测器 (YOLO + 可选 weaponhead / lightbar)
        if (needRebuildDetector) {
            std::lock_guard<std::mutex> lock(mtx_);
            if (!needRebuildCamera) cfg_ = buildConfig();
            try {
                rebuildDetectors();
                RCLCPP_INFO(this->get_logger(),
                            "检测器已重载 | YOLO:%s | whYOLO:%s | lightbar:%s | conf:%.2f iou:%.2f cuda:%d",
                            (yolo_detector_ ? "ON" : "OFF"),
                            (wh_yolo_detector_ ? "ON" : "OFF"),
                            (lightbar_detector_ ? "ON" : "OFF"),
                            cfg_.model.confThresh, cfg_.model.iouThresh,
                            cfg_.model.useCUDA);
            } catch (const std::exception& e) {
                result.successful = false;
                result.reason = std::string("检测器重建失败: ") + e.what();
            }
        }

        return result;
    }

    // ----------------------------------------------------------
    // 主循环
    // ----------------------------------------------------------
    void loopOnce() {
        std::lock_guard<std::mutex> lock(mtx_);

        // 取帧
        cv::Mat frame;
        if (!camera_ || !camera_->getFrame(frame) || frame.empty())
            return;

        last_raw_frame_ = frame.clone();  // 留底供 snapshot 服务使用

        bool shouldInfer = inference_enabled_.load();
        bool singleShot  = trigger_single_shot_.exchange(false);
        bool pubDebug    = getParam<bool>("debug_image") &&
                           pub_debug_->get_subscription_count() > 0;

        FrameResult result;
        result.frame_size = frame.size();

        bool did_inference = false;
        if (shouldInfer || singleShot) {
            if (yolo_detector_) {
                result = yolo_detector_->detect(frame);
            }
            if (wh_yolo_detector_) {
                auto why_res = wh_yolo_detector_->detect(frame);
                for (auto& d : why_res.detections) {
                    // 标记为 WEAPONHEAD class，避免与3class ID冲突
                    d.class_id = 99;
                    d.class_name = "WEAPONHEAD";
                    result.detections.push_back(std::move(d));
                }
                result.inference_ms += why_res.inference_ms;
            }
            if (lightbar_detector_) {
                auto lb_res = lightbar_detector_->detect(frame);
                for (auto& d : lb_res.detections) {
                    result.detections.push_back(std::move(d));
                }
                result.inference_ms += lb_res.inference_ms;
            }
            did_inference = true;
        }

        // 发布检测结果（数组形式，一帧只发布一条消息）
        if (did_inference) {
            publishResults(result);
        }

        last_result_ = result;  // 缓存供 snapshot 使用

        // FPS 计算 (每秒平滑更新)
        {
            double fps = frameCount_ > 0
                ? frameCount_ / std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - lastFpsTime_).count()
                : 0.0;
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - lastFpsTime_).count() >= 1.0) {
                fps_ = fps;
                lastFpsTime_ = now;
                frameCount_ = 0;
            }
        }

        // 发布 debug 画面 (有订阅者时才绘制, 节省 CPU)
        {
            auto disp = drawDetections(frame, result, fps_, shouldInfer);
            last_debug_image_ = disp.clone();  // 缓存供 snapshot 使用
            if (pubDebug) {
                auto imgMsg = cv_bridge::CvImage(
                    std_msgs::msg::Header(), "bgr8", disp).toImageMsg();
                imgMsg->header.stamp    = this->now();
                imgMsg->header.frame_id = cfg_.cameraType + "_camera";
                pub_debug_->publish(*imgMsg);
            }
        }

        // 状态 (每 30 帧)
        frameCount_++;
        if (frameCount_ % 30 == 0) {
            publishStatus(result);
        }
    }

    // ----------------------------------------------------------
    // 发布检测结果（数组消息形式）
    // ----------------------------------------------------------
    void publishResults(const FrameResult& result) {
        auto stamp = this->now();
        std::string frame_id = cfg_.cameraType + "_camera";

        kfs_core::msg::InferResults msg;
        msg.header.stamp    = stamp;
        msg.header.frame_id = frame_id;

        msg.inference_ms  = static_cast<float>(result.inference_ms);
        msg.frame_width   = static_cast<uint32_t>(result.frame_size.width);
        msg.frame_height  = static_cast<uint32_t>(result.frame_size.height);

        // 把内部 Detection 转为 InferResult 并填入数组
        for (const auto& det : result.detections) {
            kfs_core::msg::InferResult det_msg;

            // 元素消息的 header 留空（顶层 header 已包含时间戳和 frame_id）
            // 如需可在此设置 det_msg.header = msg.header;

            det_msg.class_id   = det.class_id;
            det_msg.class_name = det.class_name;
            det_msg.confidence = det.confidence;

            // 中心点取 4 个角点的平均值
            det_msg.center_x = (det.corner_tl.x + det.corner_tr.x +
                                det.corner_br.x + det.corner_bl.x) * 0.25f;
            det_msg.center_y = (det.corner_tl.y + det.corner_tr.y +
                                det.corner_br.y + det.corner_bl.y) * 0.25f;

            det_msg.bbox_width  = det.corner_br.x - det.corner_tl.x;
            det_msg.bbox_height = det.corner_br.y - det.corner_tl.y;

            msg.detections.push_back(det_msg);
        }

        pub_results_->publish(msg);
    }

    // ----------------------------------------------------------
    // 诊断用图像: 2x3 六面板 — 匹配新 joint 算法
    // ----------------------------------------------------------
    cv::Mat buildDiagnosticImage(const cv::Mat& frame) {
        // 简化诊断图: 原始帧(左) + 标注帧(右)
        if (frame.empty()) return {};
        cv::Mat board;
        cv::hconcat(frame, last_debug_image_.empty() ? frame : last_debug_image_, board);
        return board;
    }

    // ----------------------------------------------------------
    // 发布状态
    // ----------------------------------------------------------
    void publishStatus(const FrameResult& result) {
        auto msg = std_msgs::msg::String();
        msg.data = "Det: " + std::to_string(result.detections.size())
                 + " | Inf: " + std::to_string(int(result.inference_ms)) + "ms";
        pub_status_->publish(msg);
    }

    // ---- 核心 ----
    kfs::Config                          cfg_;
    std::unique_ptr<kfs::ICameraCapture> camera_;
    // 检测器: YOLO 3class + weaponhead YOLO + lightbar 并行
    std::unique_ptr<YoloDetector>        yolo_detector_;
    std::unique_ptr<YoloDetector>        wh_yolo_detector_;
    std::unique_ptr<LightbarDetector>    lightbar_detector_;

    // ---- ROS2 ----
    rclcpp::Publisher<kfs_core::msg::InferResults>::SharedPtr  pub_results_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr     pub_debug_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr       pub_status_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr      sub_enable_;
    rclcpp::Service<kfs_core::srv::SetInferState>::SharedPtr  srv_set_state_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr        srv_trigger_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr        srv_snapshot_;
    rclcpp::TimerBase::SharedPtr                              timer_;
    OnSetParametersCallbackHandle::SharedPtr                  param_cb_handle_;

    // ---- 状态 ----
    std::atomic<bool>    inference_enabled_{true};
    std::atomic<bool>    trigger_single_shot_{false};
    int                  frameCount_  = 0;
    double               fps_         = 0.0;
    std::chrono::steady_clock::time_point lastFpsTime_{std::chrono::steady_clock::now()};
    cv::Mat              last_raw_frame_;
    cv::Mat              last_debug_image_;
    FrameResult          last_result_;
    std::mutex           mtx_;
};

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KfsInferNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}