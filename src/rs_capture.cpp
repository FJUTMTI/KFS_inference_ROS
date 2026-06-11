#ifdef HAS_REALSENSE

#include "rs_capture.h"
#include <librealsense2/rs.hpp>
#include <iostream>
#include <stdexcept>

struct RealSenseCapture::Impl {
    rs2::pipeline       pipe;
    rs2::pipeline_profile profile;
    rs2::config         cfg;
    bool                running = false;

    // D415 彩色流规格
    int width  = 1920;
    int height = 1080;
    int fps    = 30;

    Impl(int w, int h, int f) : width(w), height(h), fps(f) {}

    bool start() {
        if (running) return true;

        try {
            // 仅启用彩色流
            cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);

            profile = pipe.start(cfg);

            // 获取实际流参数 (可能与请求值略有不同)
            auto stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
            width  = stream.width();
            height = stream.height();
            fps    = stream.fps();

            std::cout << "[RealSense] D415 彩色流已启动: "
                      << width << "×" << height << " @ " << fps << " fps" << std::endl;

            running = true;
            return true;
        }
        catch (const rs2::error& e) {
            std::cout << "[RealSense] 启动失败: " << e.what() << std::endl;
            return false;
        }
    }

    void stop() {
        if (!running) return;
        try {
            pipe.stop();
            running = false;
            std::cout << "[RealSense] 流已停止" << std::endl;
        }
        catch (const rs2::error& e) {
            std::cout << "[RealSense] 停止异常: " << e.what() << std::endl;
        }
    }
};

RealSenseCapture::RealSenseCapture(int width, int height, int fps)
    : pImpl(std::make_unique<Impl>(width, height, fps))
{}

RealSenseCapture::~RealSenseCapture() {
    pImpl->stop();
}

bool RealSenseCapture::start()        { return pImpl->start(); }
void RealSenseCapture::stop()         { pImpl->stop(); }
bool RealSenseCapture::isRunning() const { return pImpl->running; }

bool RealSenseCapture::getFrame(cv::Mat& frame) {
    if (!pImpl->running) return false;

    try {
        rs2::frameset fs = pImpl->pipe.wait_for_frames();
        rs2::video_frame color = fs.get_color_frame();

        int w = color.get_width();
        int h = color.get_height();

        // 直接将 BGR8 数据包到 cv::Mat (零拷贝, 只读)
        frame = cv::Mat(h, w, CV_8UC3,
                        const_cast<void*>(color.get_data())).clone();

        return !frame.empty();
    }
    catch (const rs2::error& e) {
        std::cout << "[RealSense] 取帧错误: " << e.what() << std::endl;
        return false;
    }
}

int  RealSenseCapture::getWidth()  const { return pImpl->width; }
int  RealSenseCapture::getHeight() const { return pImpl->height; }

kfs::CameraIntrinsics RealSenseCapture::getIntrinsics() const {
    kfs::CameraIntrinsics intr{};
    if (!pImpl->running) return intr;

    auto stream = pImpl->profile.get_stream(RS2_STREAM_COLOR)
                      .as<rs2::video_stream_profile>();
    auto in = stream.get_intrinsics();

    intr.fx     = in.fx;
    intr.fy     = in.fy;
    intr.cx     = in.ppx;
    intr.cy     = in.ppy;
    intr.width  = in.width;
    intr.height = in.height;

    return intr;
}

#endif // HAS_REALSENSE
