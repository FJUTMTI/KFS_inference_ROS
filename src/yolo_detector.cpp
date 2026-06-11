#include "yolo_detector.h"
#include "kfs_core/config.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <iostream>
#include <cassert>

// ============================================================
// YoloDetector::Impl — ONNX Runtime 推理核心
// ============================================================

struct YoloDetector::Impl {
    Ort::Env           env;
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;

    // 模型输入/输出名
    std::string inputName;
    std::string outputName;

    // 输入张量 shape (1, 3, inputH, inputW)
    int64_t inputC, inputH, inputW;

    // 参数
    float confThresh;
    float iouThresh;
    std::vector<std::string> classNames_;
    int numClasses_;

    // 预处理内存池 (CPU, 复用)
    std::vector<float> blob;
    std::vector<int64_t> inputShape;

    // 内存信息
    Ort::MemoryInfo cpuMemInfo;
    bool   useCUDA;

    Impl(const std::string& modelPath,
         const std::vector<std::string>& classNames,
         int inputSize,
         float confThresh,
         float iouThresh,
         bool  tryCUDA)
        : env(ORT_LOGGING_LEVEL_WARNING, "KFS_YOLO")
        , confThresh(confThresh)
        , iouThresh(iouThresh)
        , classNames_(classNames)
        , numClasses_(static_cast<int>(classNames.size()))
        , inputC(3), inputH(inputSize), inputW(inputSize)
        , cpuMemInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU))
        , useCUDA(false)
    {
        // 尝试 CUDA GPU 推理 (仅在 tryCUDA=true 时)
        if (tryCUDA) {
            try {
                OrtCUDAProviderOptions cudaOpts;
                cudaOpts.device_id = 0;
                opts.AppendExecutionProvider_CUDA(cudaOpts);
                useCUDA = true;
                opts.SetIntraOpNumThreads(2);
                std::cout << "[YOLO] CUDA GPU 推理已启用" << std::endl;
            }
            catch (const std::exception& e) {
                useCUDA = false;
                opts.SetIntraOpNumThreads(12);
                std::cout << "[YOLO] CUDA 不可用, 回退 CPU: " << e.what() << std::endl;
            }
        } else {
            opts.SetIntraOpNumThreads(12);
            std::cout << "[YOLO] CUDA 已禁用, 使用 CPU 推理 (线程数=12)" << std::endl;
        }

        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 加载模型
        session = std::make_unique<Ort::Session>(env, modelPath.c_str(), opts);

        // 获取输入信息
        {
            // ORT 1.26: GetInputTypeInfo 返回值类型 (非指针)
            Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            Ort::AllocatorWithDefaultOptions allocator;
            inputName = session->GetInputNameAllocated(0, allocator).get();
            auto shape = tensorInfo.GetShape();

            assert(shape.size() == 4);
            // ONNX 导出时 batch=1
            inputC = shape[1];
            inputH = shape[2];
            inputW = shape[3];
        }

        // 获取输出信息
        {
            Ort::AllocatorWithDefaultOptions allocator;
            outputName = session->GetOutputNameAllocated(0, allocator).get();
        }

        inputShape = {1, inputC, inputH, inputW};
        blob.resize(inputC * inputH * inputW);

        std::cout << "[YOLO] 模型加载成功: " << modelPath << std::endl;
        std::cout << "[YOLO] 输入尺寸: " << inputW << "×" << inputH
                  << "×" << inputC << std::endl;
    }

    // ----------------------------------------------------------
    // 预处理: BGR → RGB, resize, normalize, HWC→CHW
    // ----------------------------------------------------------
    void preprocess(const cv::Mat& frame) {
        // 1) BGR→RGB + resize 到 (inputW, inputH)
        cv::Mat rgb, resized;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, cv::Size(inputW, inputH));

        // 2) 归一化到 [0, 1]，转 CHW (float32)
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);

        // 3) 分离通道写入 blob
        std::vector<cv::Mat> channels(inputC);
        cv::split(resized, channels);

        size_t planeSize = inputH * inputW;
        for (int c = 0; c < inputC; ++c) {
            std::memcpy(blob.data() + c * planeSize,
                        channels[c].data,
                        planeSize * sizeof(float));
        }
    }

    // ----------------------------------------------------------
    // 后处理: 解析 YOLO11 输出 → 检测框列表
    // ONNX 输出已转置为 (N, C) 布局: 每行 = [cx, cy, w, h, cls0, cls1, cls2]
    // ----------------------------------------------------------
    std::vector<Detection> postprocess(const std::vector<float>& output,
                                       const cv::Size& origSize) {
        int totalElements = static_cast<int>(output.size());

        // 从数据总量反推: total = numAnchors * stride, 其中 stride = 4 + numClasses
        int stride = 4 + numClasses_;  // e.g. 4+3=7 for 3-class model

        if (totalElements % stride != 0) {
            std::cout << "[YOLO] 输出维度不匹配: total=" << totalElements
                      << " stride=" << stride << std::endl;
            return {};
        }
        int numAnchors = totalElements / stride;

        float scaleX = static_cast<float>(origSize.width)  / inputW;
        float scaleY = static_cast<float>(origSize.height) / inputH;

        std::vector<Detection> raw;

        for (int i = 0; i < numAnchors; ++i) {
            const float* row = output.data() + i * stride;

            float cx = row[0];
            float cy = row[1];
            float w  = row[2];
            float h  = row[3];

            // 找最大类别置信度
            float maxScore = 0.f;
            int   bestCls  = -1;
            for (int c = 0; c < numClasses_; ++c) {
                float score = row[4 + c];
                if (score > maxScore) {
                    maxScore = score;
                    bestCls  = c;
                }
            }

            if (maxScore < confThresh) continue;

            // 转回原图坐标
            float x1 = (cx - w * 0.5f) * scaleX;
            float y1 = (cy - h * 0.5f) * scaleY;
            float x2 = (cx + w * 0.5f) * scaleX;
            float y2 = (cy + h * 0.5f) * scaleY;

            // 裁剪到图像范围内
            x1 = std::max(0.f, std::min(x1, static_cast<float>(origSize.width)));
            y1 = std::max(0.f, std::min(y1, static_cast<float>(origSize.height)));
            x2 = std::max(0.f, std::min(x2, static_cast<float>(origSize.width)));
            y2 = std::max(0.f, std::min(y2, static_cast<float>(origSize.height)));

            Detection det;
            det.class_id   = bestCls;
            det.class_name = classNames_[bestCls];
            det.confidence = maxScore;
            det.corner_tl  = {x1, y1};
            det.corner_tr  = {x2, y1};
            det.corner_br  = {x2, y2};
            det.corner_bl  = {x1, y2};

            raw.push_back(det);
        }

        // NMS (按类别分组)
        return nms(raw);
    }

    // ----------------------------------------------------------
    // NMS (类别内) + 中心距离去重
    // ----------------------------------------------------------
    std::vector<Detection> nms(std::vector<Detection>& dets) {
        // 按置信度降序
        std::sort(dets.begin(), dets.end(),
                  [](const Detection& a, const Detection& b) {
                      return a.confidence > b.confidence;
                  });

        std::vector<Detection> keep;
        std::vector<bool> suppressed(dets.size(), false);

        for (size_t i = 0; i < dets.size(); ++i) {
            if (suppressed[i]) continue;
            keep.push_back(dets[i]);

            float cx_i = (dets[i].corner_tl.x + dets[i].corner_br.x) * 0.5f;
            float cy_i = (dets[i].corner_tl.y + dets[i].corner_br.y) * 0.5f;

            for (size_t j = i + 1; j < dets.size(); ++j) {
                if (suppressed[j]) continue;
                if (dets[i].class_id != dets[j].class_id) continue;

                // 计算 IOU
                float iouVal = iou(dets[i], dets[j]);

                // 计算中心距离 (归一化到框的对角线)
                float cx_j = (dets[j].corner_tl.x + dets[j].corner_br.x) * 0.5f;
                float cy_j = (dets[j].corner_tl.y + dets[j].corner_br.y) * 0.5f;
                float diag_i = std::hypot(dets[i].corner_br.x - dets[i].corner_tl.x,
                                          dets[i].corner_br.y - dets[i].corner_tl.y);
                float centerDist = std::hypot(cx_i - cx_j, cy_i - cy_j) / std::max(diag_i, 1.f);

                // 抑制条件: IOU > 阈值 或 中心很近 + 小IOU
                if (iouVal > iouThresh || (iouVal > 0.15f && centerDist < 0.15f)) {
                    suppressed[j] = true;
                }
            }
        }
        return keep;
    }

    static float iou(const Detection& a, const Detection& b) {
        float ax1 = a.corner_tl.x, ay1 = a.corner_tl.y;
        float ax2 = a.corner_br.x, ay2 = a.corner_br.y;
        float bx1 = b.corner_tl.x, by1 = b.corner_tl.y;
        float bx2 = b.corner_br.x, by2 = b.corner_br.y;

        float interX1 = std::max(ax1, bx1);
        float interY1 = std::max(ay1, by1);
        float interX2 = std::min(ax2, bx2);
        float interY2 = std::min(ay2, by2);

        float interW = std::max(0.f, interX2 - interX1);
        float interH = std::max(0.f, interY2 - interY1);
        float interArea = interW * interH;

        float areaA = (ax2 - ax1) * (ay2 - ay1);
        float areaB = (bx2 - bx1) * (by2 - by1);
        float unionArea = areaA + areaB - interArea;

        return (unionArea > 0.f) ? interArea / unionArea : 0.f;
    }

    // ----------------------------------------------------------
    // ONNX 推理 + 后处理 (CPU/CUDA 共用)
    // ----------------------------------------------------------
    FrameResult runInference(const cv::Mat& frame) {
        FrameResult result;
        result.frame_size = frame.size();
        if (frame.empty()) return result;

        auto t0 = std::chrono::high_resolution_clock::now();

        // 1) 预处理 (CPU)
        preprocess(frame);

        // 2) 创建输入 tensor (CPU pinned memory for fast H2D)
        auto inputTensor = Ort::Value::CreateTensor<float>(
            cpuMemInfo,
            blob.data(),
            blob.size(),
            inputShape.data(),
            inputShape.size()
        );

        // 3) 运行推理
        const char* inputNames[]  = { inputName.c_str() };
        const char* outputNames[] = { outputName.c_str() };
        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            inputNames, &inputTensor, 1,
            outputNames, 1
        );

        // 4) 读取输出 (D2H if CUDA)
        auto& outTensor = outputs.front();
        auto* outData   = outTensor.GetTensorMutableData<float>();
        auto  outShape  = outTensor.GetTensorTypeAndShapeInfo().GetShape();

        // 转置: (1, C, N) → (N, C) flatten
        int stride     = static_cast<int>(outShape[1]);
        int numAnchors = static_cast<int>(outShape[2]);
        std::vector<float> outVec(numAnchors * stride);

        if (stride < numAnchors) {
            // 通道优先 → 行优先 转置
            for (int i = 0; i < numAnchors; ++i)
                for (int c = 0; c < stride; ++c)
                    outVec[i * stride + c] = outData[c * numAnchors + i];
        } else {
            // 已经是行优先, 直接复制
            std::copy(outData, outData + numAnchors * stride, outVec.data());
        }

        result.detections = postprocess(outVec, frame.size());

        auto t1 = std::chrono::high_resolution_clock::now();
        result.inference_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return result;
    }
};

// ============================================================
// YoloDetector public 接口
// ============================================================

YoloDetector::YoloDetector(const std::string& modelPath,
                           const std::vector<std::string>& classNames,
                           int inputSize,
                           float confThresh,
                           float iouThresh,
                           bool  useCUDA)
    : pImpl(std::make_unique<Impl>(modelPath, classNames,
                                    inputSize, confThresh, iouThresh, useCUDA))
{}

YoloDetector::YoloDetector(const kfs::ModelConfig& cfg)
    : YoloDetector(cfg.path, cfg.classNames, cfg.inputSize,
                   cfg.confThresh, cfg.iouThresh, cfg.useCUDA)
{}

YoloDetector::~YoloDetector() = default;

FrameResult YoloDetector::detect(const cv::Mat& frame) {
    return pImpl->runInference(frame);
}

const std::vector<std::string>& YoloDetector::classNames() const {
    return pImpl->classNames_;
}

int YoloDetector::numClasses() const {
    return pImpl->numClasses_;
}
