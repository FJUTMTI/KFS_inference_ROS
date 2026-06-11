#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>

namespace kfs {

/**
 * @brief 相机内参 (用于坐标系转换)
 */
struct CameraIntrinsics {
    float fx = 0.f, fy = 0.f;   // 焦距
    float cx = 0.f, cy = 0.f;   // 主点
    int   width = 0, height = 0;
};

/**
 * @brief 相机抽象接口 (多态)
 *
 * 所有相机实现 (RealSense / USB / 模拟) 实现此接口，
 * ROS2 节点或其他上层代码仅依赖此接口，无需关心具体设备。
 *
 * 用法:
 *   std::unique_ptr<ICameraCapture> cam = CameraFactory::create(cfg);
 *   cam->start();
 *   cv::Mat frame;
 *   while (cam->getFrame(frame)) { ... }
 */
class ICameraCapture {
public:
    virtual ~ICameraCapture() = default;

    /// 启动相机流
    virtual bool start() = 0;

    /// 停止相机流
    virtual void stop() = 0;

    /// 是否正在运行
    virtual bool isRunning() const = 0;

    /**
     * @brief 获取最新一帧
     * @param frame  输出 BGR 图像
     * @return       成功获取返回 true
     */
    virtual bool getFrame(cv::Mat& frame) = 0;

    /// 获取实际分辨率
    virtual int getWidth()  const = 0;
    virtual int getHeight() const = 0;

    /// 获取相机内参
    virtual CameraIntrinsics getIntrinsics() const = 0;
};

} // namespace kfs
