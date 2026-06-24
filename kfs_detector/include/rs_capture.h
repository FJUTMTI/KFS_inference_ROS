#pragma once

#include "kfs_core/icamera_capture.h"
#include <opencv2/core.hpp>
#include <memory>
#include <string>

/**
 * @brief RealSense D415 彩色流捕获器
 *
 * 规格:
 *   - 分辨率: 1920×1080
 *   - 帧率:   30 fps
 *   - 传感器: Rolling Shutter, 2MP
 *   - FOV:    69°×42°
 *
 * 实现 kfs::ICameraCapture 接口，可被 CameraFactory 透明创建。
 */
class RealSenseCapture : public kfs::ICameraCapture {
public:
    /**
     * @param width   彩色流宽 (默认 1920)
     * @param height  彩色流高 (默认 1080)
     * @param fps     帧率 (默认 30)
     */
    RealSenseCapture(int width  = 1920,
                     int height = 1080,
                     int fps    = 30);

    ~RealSenseCapture() override;

    // 不可拷贝
    RealSenseCapture(const RealSenseCapture&) = delete;
    RealSenseCapture& operator=(const RealSenseCapture&) = delete;

    // ---- ICameraCapture 接口 ----
    bool start()                override;
    void stop()                 override;
    bool isRunning() const      override;
    bool getFrame(cv::Mat& frame) override;

    int getWidth()  const override;
    int getHeight() const override;

    kfs::CameraIntrinsics getIntrinsics() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};
