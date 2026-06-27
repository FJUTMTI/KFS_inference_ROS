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
 *     detector_type       — 检测器类型 (yolo / weaponhead_detector)
 *     enable_weaponhead   — 是否并行启用 weaponhead_detector (传统 CV 虚焦 weaponhead 左右边界)
 *     inference_enabled   — 推理使能
 */

#include "kfs_core/config.h"
#include "kfs_core/camera_factory.h"
#include "kfs_core/icamera_capture.h"
#include "yolo_detector.h"
#include "weaponhead_detector.h"

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
        declareParam<std::string>("camera_type",         seed_loaded ? seed_cfg.cameraType : "usb");
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
        declareParam<std::string>("detector_type",       seed_loaded ? seed_cfg.detectorType : "yolo");  // "yolo" | "weaponhead_detector"
        declareParam<bool>       ("enable_weaponhead",   seed_loaded ? seed_cfg.enableWeaponheadDetector : true);   // 与 YOLO 并行启用 weaponhead_detector (CV 虚焦左右边界)

        // weaponhead_detector 参数 (动态调节)
        declareParam<int>        ("wh_blur_kernel",      seed_loaded ? seed_cfg.weaponhead.blurKernel : 21,   3, 51);
        declareParam<float>      ("wh_grad_ratio",       seed_loaded ? seed_cfg.weaponhead.gradRatio : 0.35f, 0.05f, 0.95f);
        declareParam<int>        ("wh_search_band_v",    seed_loaded ? seed_cfg.weaponhead.searchBandV : 140,  20, 240);
        declareParam<int>        ("wh_min_width",        seed_loaded ? seed_cfg.weaponhead.minWidth : 16,   4,  100);
        declareParam<int>        ("wh_max_width",        seed_loaded ? seed_cfg.weaponhead.maxWidth : 300,  30, 640);
        declareParam<int>        ("wh_dark_max_gray",    seed_loaded ? seed_cfg.weaponhead.darkMaxGray : 100,   5,  200);
        declareParam<float>      ("wh_contrast_ratio",   seed_loaded ? seed_cfg.weaponhead.contrastRatio : 1.15f, 1.0f, 10.0f);
        declareParam<int>        ("wh_min_height",       seed_loaded ? seed_cfg.weaponhead.minHeight : 14,   2,  100);
        declareParam<int>        ("wh_max_drift",        seed_loaded ? seed_cfg.weaponhead.maxDrift : 35,   2,  80);
        declareParam<int>        ("wh_blob_gray_thr",    seed_loaded ? seed_cfg.weaponhead.blobGrayThr : 50,   10, 150);
        declareParam<int>        ("wh_blob_min_area",    seed_loaded ? seed_cfg.weaponhead.blobMinArea : 500,  100, 5000);
        declareParam<float>      ("wh_blob_max_sat",     seed_loaded ? seed_cfg.weaponhead.blobMaxSat : 80.0f, 0.0f, 255.0f);
        declareParam<float>      ("wh_blob_solidity",    seed_loaded ? seed_cfg.weaponhead.blobSolidity : 0.80f, 0.50f, 1.0f);

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

        // ROS 参数覆盖 (优先)
        cfg.cameraType               = getParam<std::string>("camera_type");
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

        // weaponhead ROS 参数覆盖 (优先于 YAML)
        cfg.weaponhead.blurKernel    = getParam<int>("wh_blur_kernel");
        cfg.weaponhead.gradRatio     = getParam<float>("wh_grad_ratio");
        cfg.weaponhead.searchBandV   = getParam<int>("wh_search_band_v");
        cfg.weaponhead.minWidth      = getParam<int>("wh_min_width");
        cfg.weaponhead.maxWidth      = getParam<int>("wh_max_width");
        cfg.weaponhead.darkMaxGray   = getParam<int>("wh_dark_max_gray");
        cfg.weaponhead.contrastRatio = getParam<float>("wh_contrast_ratio");
        cfg.weaponhead.minHeight     = getParam<int>("wh_min_height");
        cfg.weaponhead.maxDrift      = getParam<int>("wh_max_drift");
        cfg.weaponhead.blobGrayThr   = getParam<int>("wh_blob_gray_thr");
        cfg.weaponhead.blobMinArea   = getParam<int>("wh_blob_min_area");
        cfg.weaponhead.blobMaxSat    = getParam<float>("wh_blob_max_sat");
        cfg.weaponhead.blobSolidity  = getParam<float>("wh_blob_solidity");

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

        // 检测器 (YOLO + 可选 weaponhead) 始终创建，即使相机暂不可用
        // 这样节点可保持运行，支持通过 rqt/ param set 动态切换相机/视频源后恢复
        yolo_detector_ = std::make_unique<YoloDetector>(cfg_.model);
        if (cfg_.enableWeaponheadDetector || cfg_.detectorType == "weaponhead_detector") {
            WeaponheadDetector::Params whParams;
            whParams.blurKernel    = cfg_.weaponhead.blurKernel;
            whParams.gradRatio     = cfg_.weaponhead.gradRatio;
            whParams.searchBandV   = cfg_.weaponhead.searchBandV;
            whParams.minWidth      = cfg_.weaponhead.minWidth;
            whParams.maxWidth      = cfg_.weaponhead.maxWidth;
            whParams.darkMaxGray   = cfg_.weaponhead.darkMaxGray;
            whParams.contrastRatio = cfg_.weaponhead.contrastRatio;
            whParams.minHeight     = cfg_.weaponhead.minHeight;
            whParams.maxDrift      = cfg_.weaponhead.maxDrift;
            whParams.blobGrayThr  = cfg_.weaponhead.blobGrayThr;
            whParams.blobMinArea  = cfg_.weaponhead.blobMinArea;
            whParams.blobMaxSat   = cfg_.weaponhead.blobMaxSat;
            whParams.blobMinSolidity = cfg_.weaponhead.blobSolidity;
            wh_detector_ = std::make_unique<WeaponheadDetector>(whParams);
        }

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
                        "相机:%dx%d | 模型:%s | wh:%s | conf:%.2f iou:%.2f cuda:%d",
                        camera_->getWidth(), camera_->getHeight(),
                        cfg_.model.path.c_str(),
                        (wh_detector_ ? "ON" : "OFF"),
                        cfg_.model.confThresh, cfg_.model.iouThresh, cfg_.model.useCUDA);
        }

        return true;
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
                n == "input_size" || n == "class_names") {
                needRebuildDetector = true;
            }
            if (n == "camera_type" || n == "usb_device" || n == "usb_width" ||
                n == "usb_height" || n == "usb_fps" || n == "usb_fourcc" ||
                n == "video_path" || n == "video_loop" || n == "video_calibration_file" ||
                n == "video_undistort") {
                needRebuildCamera = true;
            }
            if (n == "detector_type" || n == "enable_weaponhead" || n == "config_path" ||
                n.rfind("wh_", 0) == 0) {
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

        // 重建检测器 (YOLO + 可选 weaponhead_detector 松耦合并行)
        if (needRebuildDetector) {
            std::lock_guard<std::mutex> lock(mtx_);
            if (!needRebuildCamera) cfg_ = buildConfig();
            yolo_detector_.reset();
            wh_detector_.reset();
            try {
                yolo_detector_ = std::make_unique<YoloDetector>(cfg_.model);
                if (cfg_.enableWeaponheadDetector || cfg_.detectorType == "weaponhead_detector") {
                    WeaponheadDetector::Params whParams;
                    whParams.blurKernel    = cfg_.weaponhead.blurKernel;
                    whParams.gradRatio     = cfg_.weaponhead.gradRatio;
                    whParams.searchBandV   = cfg_.weaponhead.searchBandV;
                    whParams.minWidth      = cfg_.weaponhead.minWidth;
                    whParams.maxWidth      = cfg_.weaponhead.maxWidth;
                    whParams.darkMaxGray   = cfg_.weaponhead.darkMaxGray;
                    whParams.contrastRatio = cfg_.weaponhead.contrastRatio;
                    whParams.minHeight     = cfg_.weaponhead.minHeight;
                    whParams.maxDrift      = cfg_.weaponhead.maxDrift;
                    whParams.blobGrayThr  = cfg_.weaponhead.blobGrayThr;
                    whParams.blobMinArea  = cfg_.weaponhead.blobMinArea;
                    whParams.blobMaxSat   = cfg_.weaponhead.blobMaxSat;
                    whParams.blobMinSolidity = cfg_.weaponhead.blobSolidity;
                    wh_detector_ = std::make_unique<WeaponheadDetector>(whParams);
                }
                RCLCPP_INFO(this->get_logger(),
                            "检测器已重载 | wh:%s | conf:%.2f iou:%.2f cuda:%d",
                            (wh_detector_ ? "ON" : "OFF"),
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
            if (wh_detector_) {
                auto wh_res = wh_detector_->detect(frame);
                // 松耦合: weaponhead 结果追加到 YOLO 结果中 (不同目标, e.g. WEAPONHEAD)
                for (auto& d : wh_res.detections) {
                    result.detections.push_back(std::move(d));
                }
                if (result.inference_ms <= 0.0) {
                    result.inference_ms = wh_res.inference_ms;
                } else {
                    result.inference_ms += wh_res.inference_ms;
                }
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
    // 诊断用图像: 2x3 六面板 (Gray | Blur | DarkMask | Grad | Pass | Laplacian)
    // ----------------------------------------------------------
    cv::Mat buildDiagnosticImage(const cv::Mat& frame) {
        if (frame.empty()) return {};

        cv::Mat gray, gray3;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, gray3, cv::COLOR_GRAY2BGR);
        } else {
            gray = frame.clone();
            cv::cvtColor(gray, gray3, cv::COLOR_GRAY2BGR);
        }

        const int w = gray.cols, h = gray.rows, cx = w / 2, cy = h / 2;

        // 1) 模糊
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(21, 21), 0);
        cv::Mat blur3; cv::cvtColor(blurred, blur3, cv::COLOR_GRAY2BGR);

        // 2) 暗部 mask (< 50, 同算法)
        cv::Mat darkMask, darkColor;
        cv::threshold(gray, darkMask, 50, 255, cv::THRESH_BINARY_INV);
        cv::applyColorMap(darkMask, darkColor, cv::COLORMAP_HOT);

        // 3) 逐行梯度 → Pass/Fail 图 (搜索带)
        int vBand = 140;
        int vTop = std::max(0, cy - vBand);
        int vBot = std::min(h - 1, cy + vBand);
        int vH = vBot - vTop + 1;

        cv::Mat bandF, bandSm, grad;
        blurred(cv::Range(vTop, vBot + 1), cv::Range::all()).convertTo(bandF, CV_32F);
        cv::GaussianBlur(bandF, bandSm, cv::Size(9, 1), 0);

        grad = cv::Mat::zeros(vH, w, CV_32F);
        for (int x = 1; x < w - 1; ++x) {
            cv::Mat d; cv::absdiff(bandSm.col(x + 1), bandSm.col(x - 1), d);
            d.copyTo(grad.col(x)); grad.col(x) *= 0.5f;
        }
        cv::GaussianBlur(grad, grad, cv::Size(21, 1), 0);

        // 全局梯度热力图
        double globalGmax;
        cv::minMaxLoc(grad, nullptr, &globalGmax);
        cv::Mat gradColor;
        grad.convertTo(gradColor, CV_8UC1, globalGmax > 0 ? 255.0 / globalGmax : 1.0);
        cv::applyColorMap(gradColor, gradColor, cv::COLORMAP_INFERNO);

        // 逐行扫描 Pass/Fail
        cv::Mat passMap = cv::Mat::zeros(vH, w, CV_8UC3);
        cv::Mat grayF; gray.convertTo(grayF, CV_32F);
        int passCount = 0;

        for (int yi = 0; yi < vH; ++yi) {
            int yAbs = vTop + yi;
            double gmin, gmax;
            cv::minMaxLoc(grad.row(yi), &gmin, &gmax);
            if (gmax < 1.5) continue;
            float gthresh = std::max(2.0f, float(gmax) * 0.35f);

            const float* gptr = grad.ptr<float>(yi);
            int left = 0, right = w - 1;
            for (int x = cx; x >= 0; --x)   { if (gptr[x] > gthresh) { left = x; break; } }
            for (int x = cx; x < w; ++x)    { if (gptr[x] > gthresh) { right = x; break; } }

            int rw = right - left;
            if (rw < 16 || rw > 300) continue;
            if (left == 0 || right == w - 1) continue;

            cv::Scalar rm = cv::mean(grayF.row(yAbs).colRange(left, right));
            if (rm[0] > 100) continue;

            float outL = float(cv::mean(grayF.row(yAbs).colRange(std::max(0,left-12), left))[0]);
            float inL  = float(cv::mean(grayF.row(yAbs).colRange(left, std::min(left+12, right)))[0]);
            float outR = float(cv::mean(grayF.row(yAbs).colRange(right, std::min(w,right+12)))[0]);
            float inR  = float(cv::mean(grayF.row(yAbs).colRange(std::max(left,right-12), right))[0]);
            bool lc = (inL > 0) ? (outL / inL) >= 1.15f : (outL > 20);
            bool rc = (inR > 0) ? (outR / inR) >= 1.15f : (outR > 20);
            if (!lc && !rc) continue;

            cv::line(passMap, {left, yi}, {right, yi}, {0, 255, 0}, 1);
            cv::circle(passMap, {left, yi}, 1, {255, 0, 0}, -1);
            cv::circle(passMap, {right, yi}, 1, {0, 0, 255}, -1);
            passCount++;
        }

        // 4) Blob 检测可视化
        cv::Mat blobVis = cv::Mat::zeros(h, w, CV_8UC3);
        {
            cv::Mat mask50;
            cv::threshold(gray, mask50, 50, 255, cv::THRESH_BINARY_INV);
            cv::Mat maskCopy = mask50.clone();
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(maskCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            for (const auto& cnt : contours) {
                double area = cv::contourArea(cnt);
                if (area < 500) continue;
                std::vector<cv::Point> hull;
                cv::convexHull(cnt, hull);
                double hullArea = cv::contourArea(hull);
                float solidity = (hullArea > 0) ? float(area / hullArea) : 0;
                cv::Rect box = cv::boundingRect(cnt);
                cv::Mat roiM = mask50(box);
                cv::Mat roiH; cv::cvtColor(frame(box), roiH, cv::COLOR_BGR2HSV);
                float s = float(cv::mean(roiH, roiM)[1]);
                cv::Scalar col = (solidity >= 0.80f && s <= 80.0f) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                cv::drawContours(blobVis, std::vector<std::vector<cv::Point>>{cnt}, -1, col, 2);
                cv::rectangle(blobVis, box, col, 1);
            }
        }

        // ── 拼图: 2 行 × 3 列 ──
        int pad = 10, labelH = 20;
        int cellW = w, cellH = h;
        int totalW = cellW * 3 + pad * 4;
        int totalH = labelH + cellH + pad + labelH + cellH + pad;
        cv::Mat board(totalH, totalW, CV_8UC3, cv::Scalar(30, 30, 30));

        auto putLabel = [&](int x, int y, const std::string& t, cv::Scalar c = {200, 200, 200}) {
            cv::putText(board, t, {x, y + 15}, cv::FONT_HERSHEY_SIMPLEX, 0.5, c, 1);
        };
        auto drawCenter = [&](int x, int y) {
            cv::line(board, {x + cx, y}, {x + cx, y + cellH - 1}, {0, 255, 255}, 1);
        };

        // Row0
        int r0y = pad;
        int c0x = pad, c1x = pad * 2 + cellW, c2x = pad * 3 + cellW * 2;

        putLabel(c0x, r0y, "1. Gray"); gray3.copyTo(board(cv::Rect(c0x, r0y + labelH, cellW, cellH))); drawCenter(c0x, r0y + labelH);
        putLabel(c1x, r0y, "2. Blurred k=21"); blur3.copyTo(board(cv::Rect(c1x, r0y + labelH, cellW, cellH)));
        putLabel(c2x, r0y, "3. Dark mask (thr=50)"); darkColor.copyTo(board(cv::Rect(c2x, r0y + labelH, cellW, cellH)));

        // Row1
        int r1y = r0y + labelH + cellH + pad;
        putLabel(c0x, r1y, "4. |Gradient| (max=" + std::to_string(int(globalGmax)) + ")");
        gradColor.copyTo(board(cv::Rect(c0x, r1y + labelH, cellW, vH)));

        putLabel(c1x, r1y, "5. Pass rows (green=pass, " + std::to_string(passCount) + " rows)");
        passMap.copyTo(board(cv::Rect(c1x, r1y + labelH, cellW, vH)));

        putLabel(c2x, r1y, "6. Blob (green=valid, red=reject)");
        blobVis.copyTo(board(cv::Rect(c2x, r1y + labelH, cellW, cellH)));

        // 如果最后有检测结果, 叠加到 panel 1/2/3
        if (!last_result_.detections.empty()) {
            for (const auto& d : last_result_.detections) {
                int l = int(d.corner_tl.x), t = int(d.corner_tl.y);
                int r = int(d.corner_br.x), b = int(d.corner_br.y);
                cv::Scalar col = (d.class_name == "WEAPONHEAD") ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 0, 0);
                cv::rectangle(board, {c0x + l, r0y + labelH + t}, {c0x + r, r0y + labelH + b}, col, 2);
                cv::rectangle(board, {c1x + l, r0y + labelH + t}, {c1x + r, r0y + labelH + b}, col, 2);
                cv::rectangle(board, {c2x + l, r0y + labelH + t}, {c2x + r, r0y + labelH + b}, col, 2);
            }
        }

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
    // 检测器: 两种实现松耦合共存, 运行时根据 cfg.detectorType 选择其一
    std::unique_ptr<YoloDetector>        yolo_detector_;
    std::unique_ptr<WeaponheadDetector>  wh_detector_;

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