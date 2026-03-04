// SPDX-License-Identifier: GPL-3.0-or-later
#include "depth_estimator.h"
#include "logger.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <shlobj.h>

constexpr float DepthEstimator::MEAN[3];
constexpr float DepthEstimator::STD[3];

// ── Helper: locate default model ─────────────────────────────────────────────
static std::wstring FindDefaultModel() {
    // 1. %APPDATA%\3Deflatten\models\depth_anything_v2_small.onnx
    wchar_t appdata[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathW(nullptr, CSIDL_APPDATA, nullptr, 0, appdata))) {
        auto p = std::filesystem::path(appdata)
                 / L"3Deflatten" / L"models"
                 / L"depth_anything_v2_small.onnx";
        if (std::filesystem::exists(p)) return p.wstring();
    }
    // 2. Alongside the running executable
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    auto p = std::filesystem::path(exePath).parent_path()
             / L"depth_anything_v2_small.onnx";
    if (std::filesystem::exists(p)) return p.wstring();

    return {};
}

// ── ctor / dtor ──────────────────────────────────────────────────────────────
DepthEstimator::DepthEstimator()
    : m_env(ORT_LOGGING_LEVEL_WARNING, "3Deflatten") {}

DepthEstimator::~DepthEstimator() {
    m_session.reset();
}

// ── Load ─────────────────────────────────────────────────────────────────────
HRESULT DepthEstimator::Load(const std::wstring& modelPath,
                              GPUProvider        provider,
                              std::wstring&      outInfo) {
    std::wstring path = modelPath.empty() ? FindDefaultModel() : modelPath;
    if (path.empty() || !std::filesystem::exists(path)) {
        LOG_ERR("Depth model not found: ",
                std::string(path.begin(), path.end()));
        return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
    }

    LOG_INFO("Loading depth model: ", std::string(path.begin(), path.end()));

    try {
        BuildSessionOptions(provider, outInfo);

        m_session = std::make_unique<Ort::Session>(
            m_env, path.c_str(), m_sessionOpts);

        Ort::AllocatorWithDefaultOptions alloc;
        auto iname = m_session->GetInputNameAllocated(0, alloc);
        auto oname = m_session->GetOutputNameAllocated(0, alloc);
        m_inputName  = iname.get();
        m_outputName = oname.get();

        auto inShape = m_session->GetInputTypeInfo(0)
                           .GetTensorTypeAndShapeInfo()
                           .GetShape();
        if (inShape.size() == 4) {
            m_dynamicInput = (inShape[2] < 0 || inShape[3] < 0);
            if (!m_dynamicInput) {
                m_modelInputH = inShape[2];
                m_modelInputW = inShape[3];
            }
        }

        m_modelPath = path;
        m_loaded    = true;
        LOG_INFO("Model loaded OK – input: ", m_inputName,
                 "  output: ", m_outputName,
                 "  provider: ", std::string(outInfo.begin(), outInfo.end()));
        return S_OK;

    } catch (const Ort::Exception& e) {
        LOG_ERR("ORT exception during Load: ", e.what());
        m_loaded = false;
        return E_FAIL;
    }
}

// ── BuildSessionOptions ───────────────────────────────────────────────────────
void DepthEstimator::BuildSessionOptions(GPUProvider provider,
                                          std::wstring& outInfo) {
    m_sessionOpts = Ort::SessionOptions();
    m_sessionOpts.SetIntraOpNumThreads(4);
    m_sessionOpts.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    m_sessionOpts.EnableMemPattern();
    m_sessionOpts.EnableCpuMemArena();

    auto tryEP = [&](GPUProvider ep) -> bool {
        try {
            if (ep == GPUProvider::CUDA) {
                OrtCUDAProviderOptions cuda{};
                cuda.device_id = 0;
                m_sessionOpts.AppendExecutionProvider_CUDA(cuda);
                outInfo = L"NVIDIA CUDA";
                LOG_INFO("Execution provider: CUDA");
                return true;
            } else if (ep == GPUProvider::DirectML) {
                m_sessionOpts.AppendExecutionProvider_DML(0);
                outInfo = L"DirectML (DX12 GPU)";
                LOG_INFO("Execution provider: DirectML");
                return true;
            }
        } catch (const Ort::Exception& e) {
            LOG_WARN("EP init failed: ", e.what());
        }
        return false;
    };

    if (provider == GPUProvider::CUDA) {
        if (!tryEP(GPUProvider::CUDA)) provider = GPUProvider::DirectML;
    }
    if (provider == GPUProvider::DirectML) {
        if (!tryEP(GPUProvider::DirectML)) provider = GPUProvider::CPU;
    }
    if (provider == GPUProvider::CPU) {
        outInfo = L"CPU";
        LOG_INFO("Execution provider: CPU");
        return;
    }
    if (provider == GPUProvider::Auto) {
        if (tryEP(GPUProvider::CUDA))    return;
        if (tryEP(GPUProvider::DirectML)) return;
        outInfo = L"CPU (fallback)";
        LOG_INFO("Execution provider: CPU (auto fallback)");
    }
}

// ── Estimate ─────────────────────────────────────────────────────────────────
HRESULT DepthEstimator::Estimate(const BYTE* srcData,
                                  int   srcWidth,
                                  int   srcHeight,
                                  int   srcStride,
                                  bool  isBGR,
                                  bool  flipDepth,
                                  float smoothAlpha,
                                  DepthResult& result) {
    if (!m_loaded || !m_session) return E_FAIL;

    try {
        std::vector<float> inputTensor;
        int mw, mh;
        PreprocessFrame(srcData, srcWidth, srcHeight, srcStride,
                        isBGR, inputTensor, mw, mh);

        std::array<int64_t, 4> shape{1, 3, (int64_t)mh, (int64_t)mw};
        auto memInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        auto inputVal = Ort::Value::CreateTensor<float>(
            memInfo,
            inputTensor.data(), inputTensor.size(),
            shape.data(), shape.size());

        const char* inNames[]  = {m_inputName.c_str()};
        const char* outNames[] = {m_outputName.c_str()};
        auto outputs = m_session->Run(
            Ort::RunOptions{nullptr},
            inNames, &inputVal, 1,
            outNames, 1);

        const float* rawDepth = outputs[0].GetTensorData<float>();
        auto rawShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int  rawH = (int)rawShape[rawShape.size() - 2];
        int  rawW = (int)rawShape[rawShape.size() - 1];

        std::vector<float> depth(srcWidth * srcHeight);
        PostprocessDepth(rawDepth, rawW, rawH,
                         srcWidth, srcHeight,
                         flipDepth, depth);

        if (smoothAlpha > 0.f && smoothAlpha < 1.f)
            TemporalSmooth(depth, smoothAlpha);

        result.data   = std::move(depth);
        result.width  = srcWidth;
        result.height = srcHeight;
        return S_OK;

    } catch (const Ort::Exception& e) {
        LOG_ERR("ORT Estimate exception: ", e.what());
        return E_FAIL;
    }
}

// ── PreprocessFrame ───────────────────────────────────────────────────────────
void DepthEstimator::PreprocessFrame(const BYTE* src,
                                      int w, int h, int stride,
                                      bool isBGR,
                                      std::vector<float>& tensor,
                                      int& mw, int& mh) {
    if (m_dynamicInput) {
        const int base = 14;
        mw = std::min(((w + base - 1) / base) * base, 1022);
        mh = std::min(((h + base - 1) / base) * base, 1022);
    } else {
        mw = (int)m_modelInputW;
        mh = (int)m_modelInputH;
    }

    tensor.resize(3 * mh * mw);
    float* R = tensor.data();
    float* G = R + mh * mw;
    float* B = G + mh * mw;

    for (int dy = 0; dy < mh; ++dy) {
        float fy  = (dy + 0.5f) * h / mh - 0.5f;
        int   sy0 = std::max(0, std::min((int)fy, h - 1));
        float ty  = fy - sy0;
        int   sy1 = std::min(sy0 + 1, h - 1);

        for (int dx = 0; dx < mw; ++dx) {
            float fx  = (dx + 0.5f) * w / mw - 0.5f;
            int   sx0 = std::max(0, std::min((int)fx, w - 1));
            float tx  = fx - sx0;
            int   sx1 = std::min(sx0 + 1, w - 1);

            auto px = [&](int sx, int sy) -> const BYTE* {
                return src + sy * stride + sx * 4;
            };
            float w00 = (1-tx)*(1-ty), w10 = tx*(1-ty);
            float w01 = (1-tx)*ty,     w11 = tx*ty;

            // BGRA: B=0, G=1, R=2  |  RGBA: R=0, G=1, B=2
            int ri = isBGR ? 2 : 0;
            int gi = 1;
            int bi = isBGR ? 0 : 2;

            float r = w00*px(sx0,sy0)[ri] + w10*px(sx1,sy0)[ri]
                    + w01*px(sx0,sy1)[ri] + w11*px(sx1,sy1)[ri];
            float g = w00*px(sx0,sy0)[gi] + w10*px(sx1,sy0)[gi]
                    + w01*px(sx0,sy1)[gi] + w11*px(sx1,sy1)[gi];
            float b = w00*px(sx0,sy0)[bi] + w10*px(sx1,sy0)[bi]
                    + w01*px(sx0,sy1)[bi] + w11*px(sx1,sy1)[bi];

            int idx = dy * mw + dx;
            R[idx] = (r / 255.f - MEAN[0]) / STD[0];
            G[idx] = (g / 255.f - MEAN[1]) / STD[1];
            B[idx] = (b / 255.f - MEAN[2]) / STD[2];
        }
    }
}

// ── PostprocessDepth ──────────────────────────────────────────────────────────
void DepthEstimator::PostprocessDepth(const float* raw,
                                       int rawW, int rawH,
                                       int dstW, int dstH,
                                       bool flipDepth,
                                       std::vector<float>& depth) {
    std::vector<float> resized(dstW * dstH);
    BilinearResize(raw, rawW, rawH, resized.data(), dstW, dstH);

    float mn = resized[0], mx = resized[0];
    for (float v : resized) { mn = std::min(mn,v); mx = std::max(mx,v); }
    float range = (mx - mn) > 1e-6f ? (mx - mn) : 1e-6f;

    depth.resize(dstW * dstH);
    for (int i = 0; i < (int)depth.size(); ++i) {
        float v = (resized[i] - mn) / range;
        depth[i] = flipDepth ? (1.f - v) : v;
    }
}

// ── BilinearResize ────────────────────────────────────────────────────────────
void DepthEstimator::BilinearResize(const float* src, int sw, int sh,
                                     float* dst,       int dw, int dh) {
    for (int dy = 0; dy < dh; ++dy) {
        float fy  = (dy + 0.5f) * sh / dh - 0.5f;
        int   sy0 = std::max(0, std::min((int)fy, sh-1));
        float ty  = fy - sy0;
        int   sy1 = std::min(sy0+1, sh-1);

        for (int dx = 0; dx < dw; ++dx) {
            float fx  = (dx + 0.5f) * sw / dw - 0.5f;
            int   sx0 = std::max(0, std::min((int)fx, sw-1));
            float tx  = fx - sx0;
            int   sx1 = std::min(sx0+1, sw-1);

            dst[dy*dw+dx] = src[sy0*sw+sx0]*(1-tx)*(1-ty)
                           + src[sy0*sw+sx1]*   tx *(1-ty)
                           + src[sy1*sw+sx0]*(1-tx)*   ty
                           + src[sy1*sw+sx1]*   tx *    ty;
        }
    }
}

// ── TemporalSmooth ────────────────────────────────────────────────────────────
void DepthEstimator::TemporalSmooth(std::vector<float>& cur, float alpha) {
    int n = (int)cur.size();
    if ((int)m_prevDepth.size() != n) {
        m_prevDepth = cur;
        return;
    }
    for (int i = 0; i < n; ++i)
        cur[i] = alpha * cur[i] + (1.f - alpha) * m_prevDepth[i];
    m_prevDepth = cur;
}
