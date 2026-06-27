#pragma once

#include "kfs_core/icamera_capture.h"
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief 摄像头分辨率描述
 */
struct CameraResolution {
    int width;
    int height;
    double fps;
};

/**
 * @brief USB / Video0 摄像头捕获器 (基于 OpenCV VideoCapture)
 *
 * 支持任意 V4L2 USB 摄像头, 可配置分辨率、帧率、设备号、曝光等参数。
 * 实现 kfs::ICameraCapture 接口，可被 CameraFactory 透明创建。
 */
class USBCapture : public kfs::ICameraCapture {
public:
    /**
     * @brief 相机控制参数
     * 值 < 0 表示保持自动/默认, >= 0 则手动设置
     */
    struct CameraControls {
        int autoExposure = -1;   // -1=自动, 1=手动
        int exposure     = 120;  // 手动曝光值 (0~10000), 推荐 50~500
        int gain         = 20;   // 增益 (0~100)
        int brightness   = 128;  // 亮度 (0~255)
        int contrast     = 128;  // 对比度 (0~255)
        int saturation   = 128;  // 饱和度 (0~255)
        int whiteBalance = -1;   // -1=自动, 2000~6500
        int sharpness    = 128;  // 锐度 (0~255)
    };

    /**
     * @param deviceId  /dev/videoN 编号 (默认 0)
     * @param width     期望分辨率宽 (默认 1920)
     * @param height    期望分辨率高 (默认 1080)
     * @param fps       期望帧率 (默认 30)
     * @param fourcc    编码格式, "MJPG" 或 "" (自动), 推荐 MJPG 获得高帧率
     */
    USBCapture(int deviceId = 0,
               int width    = 1920,
               int height   = 1080,
               int fps      = 30,
               const std::string& fourcc = "");

    ~USBCapture() override;

    // 不可拷贝
    USBCapture(const USBCapture&) = delete;
    USBCapture& operator=(const USBCapture&) = delete;

    // ---- ICameraCapture 接口 ----
    bool start()                override;
    void stop()                 override;
    bool isRunning() const      override;
    bool getFrame(cv::Mat& frame) override;

    int getWidth()  const override;
    int getHeight() const override;

    kfs::CameraIntrinsics getIntrinsics() const override;

    // ---- USBCapture 特有 ----
    int getFPS()    const;
    int getDeviceId() const;

    /// 设置相机控制参数 (曝光/增益/亮度等), 在 start() 前调用
    void applyControls(const CameraControls& ctrl);

    // ----------------------------------------------------------
    // 静态工具方法
    // ----------------------------------------------------------

    /**
     * @brief 列出指定摄像头支持的分辨率
     * @param deviceId 摄像头设备号
     * @return 支持的分辨率列表
     */
    static std::vector<CameraResolution> listResolutions(int deviceId = 0);

    /**
     * @brief 列出系统所有可用的摄像头
     * @return 可用的 /dev/videoN 编号列表
     */
    static std::vector<int> listDevices();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};
