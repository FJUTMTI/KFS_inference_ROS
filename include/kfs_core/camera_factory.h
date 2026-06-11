#pragma once

#include "kfs_core/icamera_capture.h"
#include "kfs_core/config.h"
#include <memory>

namespace kfs {

/**
 * @brief 相机工厂 — 根据配置创建 ICameraCapture 实例
 *
 * 用法:
 *   auto cam = CameraFactory::create(cfg);
 *   cam->start();
 *
 * 新增相机类型只需实现 ICameraCapture 并在工厂中注册即可，
 * 上层代码 (ROS2 节点 / CLI) 无需改动。
 */
struct CameraFactory {
    /**
     * @brief 根据配置创建相机
     * @param cfg  全局配置 (主要使用 cameraType 和 USB/controls 子项)
     * @return     相机实例, 失败返回 nullptr
     */
    static std::unique_ptr<ICameraCapture> create(const Config& cfg);

    CameraFactory() = delete;
};

} // namespace kfs
