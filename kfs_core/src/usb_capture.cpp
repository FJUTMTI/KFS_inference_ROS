#include "usb_capture.h"
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <dirent.h>

struct USBCapture::Impl {
    int  deviceId, requestW, requestH, requestFPS;
    int  realWidth = 0, realHeight = 0, realFPS = 30;
    int  fourccCode;
    bool running = false;
    CameraControls pendingCtrls;
    cv::VideoCapture cap;

    Impl(int devId, int w, int h, int f, const std::string& fourcc)
        : deviceId(devId), requestW(w), requestH(h), requestFPS(f) {
        if (fourcc == "MJPG" || fourcc == "mjpg")
            fourccCode = cv::VideoWriter::fourcc('M','J','P','G');
        else if (fourcc == "YUYV" || fourcc == "yuyv")
            fourccCode = cv::VideoWriter::fourcc('Y','U','Y','V');
        else fourccCode = -1;
    }

    bool start() {
        if (running) return true;

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

        // 读实际参数
        realWidth  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        realHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        realFPS    = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
        if (realWidth  <= 0) realWidth  = requestW;
        if (realHeight <= 0) realHeight = requestH;
        if (realFPS    <= 0) realFPS    = requestFPS;

        int af = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
        char a=(char)(af&0xFF),b=(char)((af>>8)&0xFF),c=(char)((af>>16)&0xFF),d=(char)((af>>24)&0xFF);
        std::string cc{a,b,c,d};
        for (auto& ch:cc) if (ch<32||ch>126) ch='?';

        running = true;
        std::cout << "[USB] " << cc << " " << realWidth << "×" << realHeight
                  << " @ " << realFPS << "fps" << std::endl;
        return true;
    }

    void stop() {
        if (!running) return;
        running = false;
        cap.release();
        std::cout << "[USB] 已释放\n";
    }

    bool getFrame(cv::Mat& f) {
        return running && cap.read(f);
    }
};

USBCapture::USBCapture(int d,int w,int h,int fps,const std::string& fc)
    : pImpl(std::make_unique<Impl>(d,w,h,fps,fc)) {}
USBCapture::~USBCapture(){pImpl->stop();}
bool USBCapture::start(){return pImpl->start();}
void USBCapture::stop(){pImpl->stop();}
bool USBCapture::isRunning()const{return pImpl->running;}
bool USBCapture::getFrame(cv::Mat& f){return pImpl->getFrame(f);}
int USBCapture::getWidth()const{return pImpl->realWidth;}
int USBCapture::getHeight()const{return pImpl->realHeight;}
int USBCapture::getFPS()const{return pImpl->realFPS;}
int USBCapture::getDeviceId()const{return pImpl->deviceId;}
void USBCapture::applyControls(const CameraControls& c){pImpl->pendingCtrls=c;}
kfs::CameraIntrinsics USBCapture::getIntrinsics()const{
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
