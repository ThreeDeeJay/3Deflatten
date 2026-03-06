// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten – ONNX Runtime depth estimator (Depth Anything V2)
#pragma once
#include <windows.h>
#include <string>
#include <vector>
#include <memory>
// vcpkg onnxruntime header – try both install layouts
#if __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
#  include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#elif __has_include(<onnxruntime_cxx_api.h>)
#  include <onnxruntime_cxx_api.h>
#else
#  error "Cannot find onnxruntime_cxx_api.h. Is onnxruntime installed via vcpkg?"
#endif
#include "ideflatten.h"

struct DepthResult {
    std::vector<float> data;   // row-major float depth [0,1], size = w*h
    int width;
    int height;
};

class DepthEstimator {
public:
    DepthEstimator();
    ~DepthEstimator();

    // Load (or reload) the ONNX model.
    // modelPath: path to .onnx file; "" -> search default locations.
    HRESULT Load(const std::wstring& modelPath,
                 GPUProvider         provider,
                 std::wstring&       outGPUInfo);

    // Estimate depth from an RGBA/BGRA frame in system memory.
    HRESULT Estimate(const BYTE* srcData,
                     int         srcWidth,
                     int         srcHeight,
                     int         srcStride,
                     bool        isBGR,
                     bool        flipDepth,
                     float       smoothAlpha,
                     DepthResult& result);

    bool         IsLoaded()    const { return m_loaded; }
    std::wstring GetModelPath() const { return m_modelPath; }

private:
    void BuildSessionOptions(GPUProvider provider, std::wstring& outInfo);
    void PreprocessFrame(const BYTE* src, int w, int h, int stride,
                         bool isBGR, std::vector<float>& tensor,
                         int& modelW, int& modelH);
    void PostprocessDepth(const float* raw, int rawW, int rawH,
                          int dstW, int dstH,
                          bool flipDepth,
                          std::vector<float>& depth);
    void BilinearResize(const float* src, int sw, int sh,
                        float* dst,       int dw, int dh);
    void TemporalSmooth(std::vector<float>& current, float alpha);

    Ort::Env            m_env;
    Ort::SessionOptions m_sessionOpts;
    std::unique_ptr<Ort::Session> m_session;

    std::wstring m_modelPath;
    bool         m_loaded        = false;
    int          m_estimateCount = 0;

    std::string  m_inputName;
    std::string  m_outputName;

    int64_t m_modelInputW  = 518;
    int64_t m_modelInputH  = 518;
    bool    m_dynamicInput = false;

    std::vector<float> m_prevDepth;
    int                m_prevW = 0, m_prevH = 0;

    // ImageNet normalisation constants (RGB channel order)
    static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};
};
