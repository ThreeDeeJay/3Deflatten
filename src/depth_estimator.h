// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten – ONNX Runtime depth estimator (Depth Anything V2 / V3)
#pragma once
#include <windows.h>
#include <string>
#include <vector>
#include <deque>
#include <memory>
#if __has_include(<onnxruntime_cxx_api.h>)
#  include <onnxruntime_cxx_api.h>
#elif __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
#  include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#else
#  error "onnxruntime_cxx_api.h not found."
#endif
#include "ideflatten.h"

struct DepthResult {
    std::vector<float> data;   // row-major float depth [0,1], size = w*h
    int width;
    int height;
};

class DepthEstimator {
public:
    // Sentinel model path: resolves to da3-small.onnx with streaming mode.
    // Colons are illegal in Windows file paths so this can never be a real file.
    static constexpr wchar_t STREAMING_SENTINEL[] = L":da3-streaming:";

    DepthEstimator();
    ~DepthEstimator();

    HRESULT Load(const std::wstring& modelPath,
                 GPUProvider         provider,
                 InferenceRuntime    runtime,
                 std::wstring&       outGPUInfo);

    HRESULT Estimate(const BYTE* srcData,
                     int         srcWidth,
                     int         srcHeight,
                     int         srcStride,
                     bool        isBGR,
                     bool        flipDepth,
                     float       smoothAlpha,
                     DepthResult& result);

    bool         IsLoaded()     const { return m_loaded; }
    bool         IsStreaming()  const { return m_streaming; }
    bool         IsDA3Stream()  const { return m_da3StreamMode; }
    std::wstring GetModelPath() const { return m_modelPath; }

    // Reset streaming context (seek/stop).  Safe on non-streaming models.
    void ResetStreamingContext();

private:
    void BuildSessionOptions(GPUProvider provider, std::wstring& outInfo);
    void PreprocessFrame(const BYTE* src, int w, int h, int stride,
                         bool isBGR, std::vector<float>& tensor,
                         int& modelW, int& modelH);
    void PostprocessDepth(const float* raw, int rawW, int rawH,
                          int dstW, int dstH, bool flipDepth,
                          std::vector<float>& depth);
    void BilinearResize(const float* src, int sw, int sh,
                        float* dst,       int dw, int dh);
    void TemporalSmooth(std::vector<float>& current, float alpha);

    // ── Recurrent-context streaming (future recurrent ONNX models) ───────────
    HRESULT EstimateStreaming(const BYTE* srcData,
                              int srcW, int srcH, int srcStride,
                              bool isBGR, bool flipDepth, float smoothAlpha,
                              DepthResult& result);
    bool DetectStreamingModel();
    void InitStreamingContext(int modelW, int modelH);

    // ── DA3-Streaming sliding-window algorithm ───────────────────────────────
    // Implements the core temporal consistency technique from
    //   ByteDance-Seed/Depth-Anything-3/da3_streaming/
    // Uses any single-frame model (da3-small.onnx by default).
    //
    // Algorithm:
    //  1. Run inference on each frame normally to get raw un-normalised depth.
    //  2. Keep a ring buffer of the last STREAM_WINDOW raw depth outputs.
    //  3. For each new frame, affine-align the new depth to the anchor (the
    //     running median of the ring buffer) to eliminate per-frame scale/shift
    //     jitter.  This is the temporal consistency mechanism from DA3-Streaming.
    //  4. Blend the aligned depth with the ring-buffer average (EWA) for
    //     additional flicker suppression.
    //
    // The anchor is recomputed every ANCHOR_RESET_FRAMES frames using the
    // buffer median to handle gradual scene changes without drift.
    void DA3StreamAccumulate(std::vector<float>& depth);
    static void AffineAlignTo(const std::vector<float>& anchor,
                               std::vector<float>& depth);

    static constexpr int  STREAM_WINDOW       = 8;   // sliding window size
    static constexpr int  ANCHOR_RESET_FRAMES = 16;  // re-anchor every N frames
    static constexpr float STREAM_EWA_ALPHA   = 0.35f; // blend weight for new frame

    bool                       m_da3StreamMode = false;  // DA3-Streaming algorithm active
    std::deque<std::vector<float>> m_streamBuf;           // ring buffer of raw depths
    std::vector<float>         m_streamAnchor;            // current affine anchor
    int                        m_streamFrameCount = 0;
    int                        m_streamW = 0, m_streamH = 0;

    // ── Session ──────────────────────────────────────────────────────────────
    Ort::Env            m_env;
    Ort::SessionOptions m_sessionOpts;
    std::unique_ptr<Ort::Session> m_session;

    std::wstring m_modelPath;
    std::string  m_trtCacheDir;
    bool         m_loaded        = false;
    int          m_estimateCount = 0;

    std::string  m_inputName;
    std::string  m_outputName;
    std::string  m_ctxInName;
    std::string  m_ctxOutName;

    int64_t m_modelInputW  = 518;
    int64_t m_modelInputH  = 518;
    bool    m_dynamicInput = false;

    // Recurrent-context streaming state (future models)
    bool                m_streaming = false;
    std::vector<float>  m_ctxTensor;
    int64_t             m_ctxC = 0, m_ctxH = 0, m_ctxW = 0;
    bool                m_ctxReady  = false;

    std::vector<float>  m_prevDepth;
    int                 m_prevW = 0, m_prevH = 0;

    // ── Native TRT-RTX session (ORT_ENABLE_TRTRTX builds only) ──────────────
    // When inferenceRuntime == TensorRTRtx this is used instead of the ORT session.
    struct TrtRtxSession;
    std::unique_ptr<TrtRtxSession> m_trtRtx;
#ifdef ORT_ENABLE_TRTRTX
    HRESULT LoadTrtRtxNative(const std::wstring& onnxPath, std::wstring& outInfo,
                              InferenceRuntime runtime);
    HRESULT EstimateTrtRtx(const BYTE* srcData, int srcW, int srcH, int srcStride,
                            bool isBGR, bool flipDepth, float smoothAlpha,
                            DepthResult& result);
#endif

    static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};
};
