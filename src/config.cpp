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

    return cfg;
}

} // namespace kfs
