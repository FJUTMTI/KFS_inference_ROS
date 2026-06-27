#include "kfs_core/camera_factory.h"

#include "usb_capture.h"
#ifdef HAS_REALSENSE
#include "rs_capture.h"
#endif

#include <iostream>

namespace kfs {

std::unique_ptr<ICameraCapture> CameraFactory::create(const Config& cfg) {
    if (cfg.cameraType == "usb") {
        USBCapture::CameraControls ctrl;
        ctrl.autoExposure = cfg.controls.autoExposure;
        ctrl.exposure     = cfg.controls.exposure;
        ctrl.gain         = cfg.controls.gain;
        ctrl.brightness   = cfg.controls.brightness;
        ctrl.contrast     = cfg.controls.contrast;
        ctrl.saturation   = cfg.controls.saturation;
        ctrl.whiteBalance = cfg.controls.whiteBalance;
        ctrl.sharpness    = cfg.controls.sharpness;

        auto cam = std::make_unique<USBCapture>(
            cfg.usb.device,
            cfg.usb.width, cfg.usb.height, cfg.usb.fps,
            cfg.usb.fourcc,
            cfg.usb.calibration_file,
            cfg.usb.undistort);
        cam->applyControls(ctrl);
        return cam;
    }

    if (cfg.cameraType == "video") {
        auto cam = std::make_unique<USBCapture>(
            cfg.video.path,
            cfg.video.calibration_file,
            cfg.video.undistort,
            cfg.video.loop);
        return cam;
    }

#ifdef HAS_REALSENSE
    if (cfg.cameraType == "realsense") {
        return std::make_unique<RealSenseCapture>(1920, 1080, 30);
    }
#else
    if (cfg.cameraType == "realsense") {
        std::cerr << "[CameraFactory] 此编译版本不支持 RealSense (未链接 librealsense2)\n";
    }
#endif

    std::cerr << "[CameraFactory] 未知相机类型: " << cfg.cameraType << " (支持: usb, realsense, video)\n";
    return nullptr;
}

} // namespace kfs
