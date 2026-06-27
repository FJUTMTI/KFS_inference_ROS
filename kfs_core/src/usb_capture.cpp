#include "usb_capture.h"
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <dirent.h>
#include <yaml-cpp/yaml.h>
#include <fstream>

struct USBCapture::Impl {
    int  deviceId = -1;
    int  requestW, requestH, requestFPS;
    int  realWidth = 0, realHeight = 0, realFPS = 30;
    int  fourccCode;
    bool running = false;
    CameraControls pendingCtrls;
    cv::VideoCapture cap;
    std::string calibrationFile;
    kfs::CameraIntrinsics calibratedIntrinsics;  // loaded if calibration_file valid
    bool intrinsicsLoaded = false;
    bool doUndistort = false;
    std::vector<double> distortionCoeffs;        // k1,k2,p1,p2,k3,... (up to 8 for rational)
    cv::Mat undistMap1, undistMap2;
    bool mapsReady = false;

    // video file mode (type=video)
    bool        isFile = false;
    std::string videoPath;
    bool        loop = true;

    Impl(int devId, int w, int h, int f, const std::string& fourcc, const std::string& calib_file, bool undist)
        : deviceId(devId), requestW(w), requestH(h), requestFPS(f), calibrationFile(calib_file), doUndistort(undist) {
        if (fourcc == "MJPG" || fourcc == "mjpg")
            fourccCode = cv::VideoWriter::fourcc('M','J','P','G');
        else if (fourcc == "YUYV" || fourcc == "yuyv")
            fourccCode = cv::VideoWriter::fourcc('Y','U','Y','V');
        else fourccCode = -1;
        loadCalibration();
    }

    Impl(const std::string& path, const std::string& calib_file, bool undist, bool loop_play)
        : requestW(0), requestH(0), requestFPS(0), calibrationFile(calib_file), doUndistort(undist),
          isFile(true), videoPath(path), loop(loop_play) {
        fourccCode = -1;
        loadCalibration();
    }

    void loadCalibration() {
        if (calibrationFile.empty() || intrinsicsLoaded) return;
        try {
            YAML::Node root = YAML::LoadFile(calibrationFile);
            // ROS ost.yaml or camera_info.yaml style
            if (root["camera_matrix"] && root["camera_matrix"]["data"]) {
                auto data = root["camera_matrix"]["data"];
                if (data.IsSequence() && data.size() >= 9) {
                    float fx = data[0].as<float>();
                    float fy = data[4].as<float>();
                    float cx = data[2].as<float>();
                    float cy = data[5].as<float>();
                    calibratedIntrinsics.fx = fx;
                    calibratedIntrinsics.fy = fy;
                    calibratedIntrinsics.cx = cx;
                    calibratedIntrinsics.cy = cy;
                    if (root["image_width"]) calibratedIntrinsics.width = root["image_width"].as<int>();
                    if (root["image_height"]) calibratedIntrinsics.height = root["image_height"].as<int>();
                    intrinsicsLoaded = true;
                    std::cout << "[Capture] 已加载相机标定: " << calibrationFile
                              << " fx=" << fx << " fy=" << fy << " cx=" << cx << " cy=" << cy << std::endl;
                }
            }
            // Fallback: some yaml use top level K: [..] or camera_matrix as flat
            if (!intrinsicsLoaded && root["K"] && root["K"].IsSequence() && root["K"].size() >= 9) {
                YAML::Node k = root["K"];
                calibratedIntrinsics.fx = k[0].as<float>();
                calibratedIntrinsics.fy = k[4].as<float>();
                calibratedIntrinsics.cx = k[2].as<float>();
                calibratedIntrinsics.cy = k[5].as<float>();
                intrinsicsLoaded = true;
                std::cout << "[Capture] 已加载相机标定(K): " << calibrationFile << std::endl;
            }

            // Load distortion coefficients (supports plumb_bob 5 or rational 8)
            YAML::Node distNode;
            if (root["distortion_coefficients"] && root["distortion_coefficients"]["data"])
                distNode = root["distortion_coefficients"]["data"];
            else if (root["D"] && root["D"].IsSequence())
                distNode = root["D"];
            else if (root["distortion"] && root["distortion"].IsSequence())
                distNode = root["distortion"];

            if (distNode && distNode.IsSequence() && distNode.size() >= 4) {
                distortionCoeffs.clear();
                for (std::size_t i = 0; i < distNode.size(); ++i) {
                    distortionCoeffs.push_back(distNode[i].as<double>());
                }
                // Trim trailing zeros for common 5-coeff case
                while (distortionCoeffs.size() > 5 && distortionCoeffs.back() == 0.0)
                    distortionCoeffs.pop_back();
            }

            if (!distortionCoeffs.empty() && intrinsicsLoaded) {
                std::cout << "[Capture] 畸变系数已加载 (" << distortionCoeffs.size() << " 个)\n";
            }
        } catch (const std::exception& e) {
            std::cout << "[Capture] 标定文件加载失败 '" << calibrationFile << "': " << e.what() << " (回退默认内参)\n";
        }
    }

    void initUndistortMaps() {
        if (!doUndistort || !intrinsicsLoaded || mapsReady || distortionCoeffs.empty()) return;

        int w = realWidth > 0 ? realWidth : requestW;
        int h = realHeight > 0 ? realHeight : requestH;
        if (w <= 0 || h <= 0) return;

        // Use the (possibly already scaled in getIntrinsics logic) current intrinsics
        kfs::CameraIntrinsics cur = calibratedIntrinsics;
        if (cur.width <= 0) cur.width = w;
        if (cur.height <= 0) cur.height = h;

        cv::Mat K = (cv::Mat_<double>(3,3) <<
            cur.fx, 0, cur.cx,
            0, cur.fy, cur.cy,
            0, 0, 1);

        int dsize = (int)distortionCoeffs.size();
        cv::Mat D(1, dsize, CV_64F);
        for (int i = 0; i < dsize; ++i) {
            D.at<double>(0, i) = distortionCoeffs[i];
        }

        // Compute optimal new camera matrix (alpha=0 keeps all pixels valid, smaller FOV)
        cv::Mat newK = cv::getOptimalNewCameraMatrix(K, D, cv::Size(w, h), 0.0);

        cv::initUndistortRectifyMap(K, D, cv::Mat(), newK, cv::Size(w, h),
                                    CV_32FC1, undistMap1, undistMap2);

        mapsReady = !undistMap1.empty() && !undistMap2.empty();
        if (mapsReady) {
            // Update returned intrinsics to the new (undistorted) camera matrix for consistency
            calibratedIntrinsics.fx = static_cast<float>(newK.at<double>(0,0));
            calibratedIntrinsics.fy = static_cast<float>(newK.at<double>(1,1));
            calibratedIntrinsics.cx = static_cast<float>(newK.at<double>(0,2));
            calibratedIntrinsics.cy = static_cast<float>(newK.at<double>(1,2));
            calibratedIntrinsics.width = w;
            calibratedIntrinsics.height = h;

            std::cout << "[Capture] 畸变矫正地图已初始化 (" << w << "x" << h << ")\n";
        }
    }

    bool start() {
        if (running) return true;

        if (isFile) {
            if (videoPath.empty() || !cap.open(videoPath)) {
                std::cout << "[Video] 无法打开视频文件: " << videoPath << "\n";
                return false;
            }
        } else {
            // 打开设备 — OpenCV 全权管理 V4L2
            if (!cap.open(deviceId, cv::CAP_V4L2))
                cap.open(deviceId);
            if (!cap.isOpened()) {
                std::cout << "[USB] 无法打开 /dev/video" << deviceId << "\n";
                return false;
            }

            // 设置 FOURCC
            if (fourccCode != -1)
                cap.set(cv::CAP_PROP_FOURCC, static_cast<double>(fourccCode));

            // 设置分辨率
            cap.set(cv::CAP_PROP_FRAME_WIDTH,  static_cast<double>(requestW));
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(requestH));

            // V4L2 控制通过 OpenCV (部分有效)
            if (pendingCtrls.autoExposure == 1)
                cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
            if (pendingCtrls.brightness >= 0)
                cap.set(cv::CAP_PROP_BRIGHTNESS, static_cast<double>(pendingCtrls.brightness));
            if (pendingCtrls.contrast >= 0)
                cap.set(cv::CAP_PROP_CONTRAST, static_cast<double>(pendingCtrls.contrast));
            if (pendingCtrls.saturation >= 0)
                cap.set(cv::CAP_PROP_SATURATION, static_cast<double>(pendingCtrls.saturation));
            if (pendingCtrls.gain >= 0)
                cap.set(cv::CAP_PROP_GAIN, static_cast<double>(pendingCtrls.gain));
        }

        // 读实际参数
        realWidth  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        realHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        realFPS    = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
        if (realWidth  <= 0) realWidth  = requestW;
        if (realHeight <= 0) realHeight = requestH;
        if (realFPS    <= 0) realFPS    = requestFPS;

        // Prepare undistort maps if requested and we have calibration
        initUndistortMaps();

        int af = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
        char a=(char)(af&0xFF),b=(char)((af>>8)&0xFF),c=(char)((af>>16)&0xFF),d=(char)((af>>24)&0xFF);
        std::string cc{a,b,c,d};
        for (auto& ch:cc) if (ch<32||ch>126) ch='?';

        running = true;
        if (isFile) {
            std::cout << "[Video] " << videoPath << " " << realWidth << "×" << realHeight
                      << " @ " << realFPS << "fps" << (loop ? " (loop)" : "") << std::endl;
        } else {
            std::cout << "[USB] " << cc << " " << realWidth << "×" << realHeight
                      << " @ " << realFPS << "fps" << std::endl;
        }
        return true;
    }

    void stop() {
        if (!running) return;
        running = false;
        cap.release();
        if (isFile) {
            std::cout << "[Video] 已释放\n";
        } else {
            std::cout << "[USB] 已释放\n";
        }
    }

    bool getFrame(cv::Mat& f) {
        if (!running) return false;
        if (!cap.read(f)) {
            if (isFile && loop && !videoPath.empty()) {
                // 循环播放: seek 到开头重试
                cap.set(cv::CAP_PROP_POS_FRAMES, 0.0);
                if (!cap.read(f)) return false;
            } else {
                return false;
            }
        }

        if (doUndistort && mapsReady && !undistMap1.empty() && !undistMap2.empty()) {
            cv::Mat undistorted;
            cv::remap(f, undistorted, undistMap1, undistMap2, cv::INTER_LINEAR);
            f = undistorted;
        }
        return true;
    }
};

USBCapture::USBCapture(int d,int w,int h,int fps,const std::string& fc, const std::string& calib, bool undist)
    : pImpl(std::make_unique<Impl>(d,w,h,fps,fc,calib,undist)) {}
USBCapture::USBCapture(const std::string& path, const std::string& calib, bool undist, bool loop)
    : pImpl(std::make_unique<Impl>(path, calib, undist, loop)) {}
USBCapture::~USBCapture(){pImpl->stop();}
bool USBCapture::start(){return pImpl->start();}
void USBCapture::stop(){pImpl->stop();}
bool USBCapture::isRunning()const{return pImpl->running;}
bool USBCapture::getFrame(cv::Mat& f){return pImpl->getFrame(f);}
int USBCapture::getWidth()const{return pImpl->realWidth;}
int USBCapture::getHeight()const{return pImpl->realHeight;}
int USBCapture::getFPS()const{return pImpl->realFPS;}
int USBCapture::getDeviceId()const{return pImpl->isFile ? -1 : pImpl->deviceId;}
void USBCapture::applyControls(const CameraControls& c){pImpl->pendingCtrls=c;}
kfs::CameraIntrinsics USBCapture::getIntrinsics()const{
    if (pImpl->intrinsicsLoaded) {
        kfs::CameraIntrinsics in = pImpl->calibratedIntrinsics;
        int lw = in.width > 0 ? in.width : pImpl->requestW;
        int lh = in.height > 0 ? in.height : pImpl->requestH;
        int rw = pImpl->realWidth > 0 ? pImpl->realWidth : pImpl->requestW;
        int rh = pImpl->realHeight > 0 ? pImpl->realHeight : pImpl->requestH;
        if (lw > 0 && lh > 0 && rw > 0 && rh > 0 && (lw != rw || lh != rh)) {
            // Scale intrinsics from calib resolution to actual running resolution
            float sx = static_cast<float>(rw) / static_cast<float>(lw);
            float sy = static_cast<float>(rh) / static_cast<float>(lh);
            in.fx *= sx;
            in.fy *= sy;
            in.cx *= sx;
            in.cy *= sy;
        }
        // Always report the intrinsics for the resolution we are actually running at
        in.width = rw;
        in.height = rh;
        return in;
    }
    kfs::CameraIntrinsics in{}; in.width=pImpl->realWidth; in.height=pImpl->realHeight;
    in.fx=in.cx=(float)pImpl->realWidth*.5f; in.fy=in.cy=(float)pImpl->realHeight*.5f;
    return in;
}

static bool isVid(const std::string& n){
    if(n.find("video")==std::string::npos)return false;
    for(char c:n)if(c>='0'&&c<='9')return true;
    return false;
}
std::vector<int> USBCapture::listDevices(){
    std::vector<int> r;DIR*d=opendir("/dev");if(!d)return r;
    struct dirent*e;while((e=readdir(d))){if(!isVid(e->d_name))continue;
    std::string ns;for(char c:e->d_name)if(c>='0'&&c<='9')ns+=c;
    if(ns.empty())continue;int id=std::stoi(ns);
    cv::VideoCapture t(id,cv::CAP_V4L2);if(t.isOpened()){r.push_back(id);t.release();}}
    closedir(d);return r;
}
std::vector<CameraResolution> USBCapture::listResolutions(int devId){
    std::vector<CameraResolution> rl;
    cv::VideoCapture cap(devId,cv::CAP_V4L2);if(!cap.isOpened())return rl;
    static const int cr[][2]={{1920,1080},{1280,720},{960,540},{800,600},{640,480},{480,360},{320,240}};
    double ow=cap.get(cv::CAP_PROP_FRAME_WIDTH),oh=cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    for(auto&r:cr){cap.set(cv::CAP_PROP_FRAME_WIDTH,(double)r[0]);cap.set(cv::CAP_PROP_FRAME_HEIGHT,(double)r[1]);
    int gw=(int)cap.get(cv::CAP_PROP_FRAME_WIDTH),gh=(int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    if(gw==r[0]&&gh==r[1]){double gf=cap.get(cv::CAP_PROP_FPS);if(gf<=0)gf=30;
    bool dup=false;for(auto&x:rl)if(x.width==gw&&x.height==gh){dup=true;break;}if(!dup)rl.push_back({gw,gh,gf});}}
    cap.set(cv::CAP_PROP_FRAME_WIDTH,ow);cap.set(cv::CAP_PROP_FRAME_HEIGHT,oh);cap.release();return rl;
}
