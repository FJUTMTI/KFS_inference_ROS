#include "kfs_core/config.h"
#include <iostream>

namespace kfs {

// ---- YAML 辅助 (节点缺失时返回默认值) ----
namespace {

static int yamlInt(const YAML::Node& node, const std::string& key, int defval) {
    if (node && node[key]) return node[key].as<int>();
    return defval;
}
static float yamlFloat(const YAML::Node& node, const std::string& key, float defval) {
    if (node && node[key]) return node[key].as<float>();
    return defval;
}
static std::string yamlStr(const YAML::Node& node, const std::string& key,
                           const std::string& defval) {
    if (node && node[key]) return node[key].as<std::string>();
    return defval;
}
static bool yamlBool(const YAML::Node& node, const std::string& key, bool defval) {
    if (node && node[key]) return node[key].as<bool>();
    return defval;
}

static std::vector<std::string> yamlStrVec(const YAML::Node& node,
                                            const std::string& key,
                                            const std::vector<std::string>& defval) {
    if (node && node[key] && node[key].IsSequence()) {
        std::vector<std::string> result;
        for (const auto& item : node[key]) {
            result.push_back(item.as<std::string>());
        }
        return result;
    }
    return defval;
}

} // anonymous namespace

Config loadConfig(const std::string& yamlPath) {
    Config cfg;
    YAML::Node root = YAML::LoadFile(yamlPath);

    // ---- camera ----
    if (root["camera"]) {
        auto cam = root["camera"];
        cfg.cameraType = yamlStr(cam, "type", cfg.cameraType);

        if (cam["usb"]) {
            auto usb = cam["usb"];
            cfg.usb.device = yamlInt(usb, "device", cfg.usb.device);
            cfg.usb.width  = yamlInt(usb, "width",  cfg.usb.width);
            cfg.usb.height = yamlInt(usb, "height", cfg.usb.height);
            cfg.usb.fps    = yamlInt(usb, "fps",    cfg.usb.fps);
            cfg.usb.fourcc = yamlStr(usb, "fourcc", cfg.usb.fourcc);
            cfg.usb.calibration_file = yamlStr(usb, "calibration_file", cfg.usb.calibration_file);
            cfg.usb.undistort = yamlBool(usb, "undistort", cfg.usb.undistort);
        }

        if (cam["video"]) {
            auto vid = cam["video"];
            cfg.video.path = yamlStr(vid, "path", cfg.video.path);
            cfg.video.loop = yamlBool(vid, "loop", cfg.video.loop);
            cfg.video.calibration_file = yamlStr(vid, "calibration_file", cfg.video.calibration_file);
            cfg.video.undistort = yamlBool(vid, "undistort", cfg.video.undistort);
        }

        if (cam["controls"]) {
            auto ctrl = cam["controls"];
            cfg.controls.autoExposure = yamlInt(ctrl, "auto_exposure", cfg.controls.autoExposure);
            cfg.controls.exposure     = yamlInt(ctrl, "exposure",      cfg.controls.exposure);
            cfg.controls.gain         = yamlInt(ctrl, "gain",          cfg.controls.gain);
            cfg.controls.brightness   = yamlInt(ctrl, "brightness",    cfg.controls.brightness);
            cfg.controls.contrast     = yamlInt(ctrl, "contrast",      cfg.controls.contrast);
            cfg.controls.saturation   = yamlInt(ctrl, "saturation",    cfg.controls.saturation);
            cfg.controls.whiteBalance = yamlInt(ctrl, "white_balance", cfg.controls.whiteBalance);
            cfg.controls.sharpness    = yamlInt(ctrl, "sharpness",     cfg.controls.sharpness);
        }
    }

    // ---- model ----
    if (root["model"]) {
        auto mdl = root["model"];
        cfg.model.path       = yamlStr(mdl, "path",           cfg.model.path);
        cfg.model.inputSize  = yamlInt(mdl, "input_size",     cfg.model.inputSize);
        cfg.model.confThresh = yamlFloat(mdl, "conf_threshold", cfg.model.confThresh);
        cfg.model.iouThresh  = yamlFloat(mdl, "iou_threshold",  cfg.model.iouThresh);
        cfg.model.useCUDA    = yamlBool(mdl, "use_cuda",       cfg.model.useCUDA);
        cfg.model.classNames = yamlStrVec(mdl, "class_names",  cfg.model.classNames);
    }

    // ---- display ----
    if (root["display"]) {
        auto disp = root["display"];
        cfg.display.debug = yamlBool(disp, "debug", cfg.display.debug);
    }

    // ---- detector (松耦合: weaponhead / lightbar) ----
    if (root["detector"]) {
        auto det = root["detector"];
        cfg.detectorType = yamlStr(det, "type", cfg.detectorType);
        cfg.enableWeaponheadDetector = yamlBool(det, "enable_weaponhead", cfg.enableWeaponheadDetector);
        cfg.enableLightbarDetector   = yamlBool(det, "enable_lightbar",   cfg.enableLightbarDetector);
    }

    // detector.type 快捷开关
    if (cfg.detectorType == "lightbar" || cfg.detectorType == "yolo_lightbar") {
        cfg.enableLightbarDetector = true;
    }

    // ---- weaponhead_detector 参数 ----
    if (root["weaponhead_detector"]) {
        auto wh = root["weaponhead_detector"];
        if (wh["enabled"]) {
            cfg.enableWeaponheadDetector = wh["enabled"].as<bool>();
        }
        cfg.weaponhead.whModelPath      = yamlStr(wh, "wh_model_path",     cfg.weaponhead.whModelPath);
    }

    // ---- lightbar_detector 参数 ----
    if (root["lightbar_detector"]) {
        auto lb = root["lightbar_detector"];
        if (lb["enabled"]) {
            cfg.enableLightbarDetector = lb["enabled"].as<bool>();
            cfg.lightbar.enabled = cfg.enableLightbarDetector;
        }
        cfg.lightbar.corePercentile  = yamlFloat(lb, "core_percentile",   cfg.lightbar.corePercentile);
        cfg.lightbar.coreThreshMin   = yamlInt  (lb, "core_thresh_min",   cfg.lightbar.coreThreshMin);
        cfg.lightbar.coreThreshMax   = yamlInt  (lb, "core_thresh_max",   cfg.lightbar.coreThreshMax);
        cfg.lightbar.coreThreshScale = yamlFloat(lb, "core_thresh_scale", cfg.lightbar.coreThreshScale);
        cfg.lightbar.maxCoreFrac     = yamlFloat(lb, "max_core_frac",     cfg.lightbar.maxCoreFrac);
        cfg.lightbar.maxShortSide    = yamlFloat(lb, "max_short_side",    cfg.lightbar.maxShortSide);
        cfg.lightbar.openKsize       = yamlInt  (lb, "open_ksize",        cfg.lightbar.openKsize);
        cfg.lightbar.closeLength     = yamlInt  (lb, "close_length",      cfg.lightbar.closeLength);
        cfg.lightbar.closeVertical   = yamlInt  (lb, "close_vertical",    cfg.lightbar.closeVertical);
        // 兼容: close_vertical 旧字段覆盖 close_length
        if (lb["close_vertical"] && !lb["close_length"]) {
            cfg.lightbar.closeLength = cfg.lightbar.closeVertical;
        }
        cfg.lightbar.minLength       = yamlFloat(lb, "min_length",        cfg.lightbar.minLength);
        cfg.lightbar.minAspect       = yamlFloat(lb, "min_aspect",        cfg.lightbar.minAspect);
        cfg.lightbar.minArea         = yamlFloat(lb, "min_area",          cfg.lightbar.minArea);
        cfg.lightbar.minLengthRatio  = yamlFloat(lb, "min_length_ratio",  cfg.lightbar.minLengthRatio);
        cfg.lightbar.maxAngleDiff    = yamlFloat(lb, "max_angle_diff",    cfg.lightbar.maxAngleDiff);
        cfg.lightbar.bloomKsize      = yamlInt  (lb, "bloom_ksize",       cfg.lightbar.bloomKsize);
        cfg.lightbar.minColorScore   = yamlFloat(lb, "min_color_score",   cfg.lightbar.minColorScore);
        cfg.lightbar.saveDebugMask   = yamlBool (lb, "save_debug_mask",   cfg.lightbar.saveDebugMask);
    } else {
        cfg.lightbar.enabled = cfg.enableLightbarDetector;
    }

    return cfg;
}

} // namespace kfs
