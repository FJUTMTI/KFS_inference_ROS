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

#include <memory>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>
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
        // ---- 声明全部参数 (支持 rqt_reconfigure 动态调节) ----
        declareParam<std::string>("config_path",        "config/kfs_config.yaml");
        declareParam<bool>       ("debug_image",         true);
        declareParam<float>      ("conf_threshold",      0.25f, 0.0f, 1.0f);
        declareParam<float>      ("iou_threshold",       0.30f, 0.0f, 1.0f);
        declareParam<bool>       ("use_cuda",            true);
        declareParam<std::string>("model_path",          "models/kfs_yolo11_3class.onnx");
        declareParam<std::string>("camera_type",         "usb");
        declareParam<int>        ("usb_device",          0, 0, 63);
        declareParam<int>        ("usb_width",           640, 160, 3840);
        declareParam<int>        ("usb_height",          480, 120, 2160);
        declareParam<int>        ("usb_fps",             60, 1, 240);
        declareParam<std::string>("usb_fourcc",          "MJPG");
        declareParam<bool>       ("inference_enabled",   true);
        declareParam<int>        ("input_size",          640, 320, 1280);
        declareParam<std::string>("detector_type",      "yolo");  // "yolo" | "weaponhead_detector"
        declareParam<bool>       ("enable_weaponhead",    false);   // 与 YOLO 并行启用 weaponhead_detector (CV 虚焦左右边界)

        // 类别名作为 string 列表(逗号分隔), rqt 字符串参数编辑
        declareParam<std::string>("class_names", "R1,T,F");

        // ---- 加载 YAML 作为初始值, 然后 ROS 参数覆盖 ----
        if (!initCore()) {
            RCLCPP_FATAL(this->get_logger(), "核心初始化失败");
            return;
        }

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
        cfg.model.path               = getParam<std::string>("model_path");
        cfg.model.inputSize          = getParam<int>("input_size");
        cfg.model.confThresh         = getParam<float>("conf_threshold");
        cfg.model.iouThresh          = getParam<float>("iou_threshold");
        cfg.model.useCUDA            = getParam<bool>("use_cuda");
        cfg.display.debug            = false; // 无头模式

        cfg.detectorType             = getParam<std::string>("detector_type");
        cfg.enableWeaponheadDetector = getParam<bool>("enable_weaponhead");

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

        camera_ = kfs::CameraFactory::create(cfg_);
        if (!camera_) {
            RCLCPP_ERROR(this->get_logger(), "相机创建失败");
            return false;
        }
        if (!camera_->start()) {
            RCLCPP_ERROR(this->get_logger(), "相机启动失败");
            return false;
        }

        // YOLO 始终创建 (主检测)
        yolo_detector_ = std::make_unique<YoloDetector>(cfg_.model);
        // weaponhead_detector 松耦合附加 (与 YOLO 并行, 检测不同目标)
        if (cfg_.enableWeaponheadDetector || cfg_.detectorType == "weaponhead_detector") {
            wh_detector_ = std::make_unique<WeaponheadDetector>();
        }

        RCLCPP_INFO(this->get_logger(),
                    "相机:%dx%d | 模型:%s | wh:%s | conf:%.2f iou:%.2f cuda:%d",
                    camera_->getWidth(), camera_->getHeight(),
                    cfg_.model.path.c_str(),
                    (wh_detector_ ? "ON" : "OFF"),
                    cfg_.model.confThresh, cfg_.model.iouThresh, cfg_.model.useCUDA);
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
                n == "usb_height" || n == "usb_fps" || n == "usb_fourcc") {
                needRebuildCamera = true;
            }
            if (n == "detector_type" || n == "enable_weaponhead" || n == "config_path") {
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
            if (camera_) {
                camera_->start();
                RCLCPP_INFO(this->get_logger(), "相机已重载: %dx%d",
                            camera_->getWidth(), camera_->getHeight());
            } else {
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
                    wh_detector_ = std::make_unique<WeaponheadDetector>();
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
                // 松耦合: weaponhead 结果追加到 YOLO 结果中 (不同目标, e.g. OBJ)
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

        // 发布 debug 画面 (有订阅者时才绘制, 节省 CPU)
        if (pubDebug) {
            publishDebugImage(frame, result, shouldInfer);
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
    // 发布 debug 画面
    // ----------------------------------------------------------
    void publishDebugImage(const cv::Mat& frame, const FrameResult& result,
                           bool inferActive) {
        double fps = frameCount_ > 0
            ? frameCount_ / std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - lastFpsTime_).count()
            : 0.0;

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - lastFpsTime_).count() >= 1.0) {
            fps_ = fps;
            lastFpsTime_ = now;
        }

        auto disp = drawDetections(frame, result, fps_, inferActive);

        auto imgMsg = cv_bridge::CvImage(
            std_msgs::msg::Header(), "bgr8", disp).toImageMsg();
        imgMsg->header.stamp    = this->now();
        imgMsg->header.frame_id = cfg_.cameraType + "_camera";
        pub_debug_->publish(*imgMsg);
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
    rclcpp::TimerBase::SharedPtr                              timer_;
    OnSetParametersCallbackHandle::SharedPtr                  param_cb_handle_;

    // ---- 状态 ----
    std::atomic<bool>    inference_enabled_{true};
    std::atomic<bool>    trigger_single_shot_{false};
    int                  frameCount_  = 0;
    double               fps_         = 0.0;
    std::chrono::steady_clock::time_point lastFpsTime_{std::chrono::steady_clock::now()};
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