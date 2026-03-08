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
// Search order:
//  1. DLL folder       (e.g. Win64/ or Win32/)
//  2. DLL parent       (install root containing both Win32/ and Win64/)
//  3. Host EXE folder
//  4. %APPDATA%\3Deflatten\models
// In each folder the well-known name is tried first, then any *.onnx file.

static std::wstring FirstOnnxIn(const std::filesystem::path& dir) {
    auto named = dir / L"depth_anything_v2_small.onnx";
    if (std::filesystem::exists(named)) return named.wstring();
    std::error_code ec;
    for (auto& e : std::filesystem::directory_iterator(dir, ec))
        if (e.path().extension() == L".onnx")
            return e.path().wstring();
    return {};
}

static std::wstring FindDefaultModel() {
    namespace fs = std::filesystem;

    // 1 & 2: DLL folder and its parent.
    {
        wchar_t dllPath[MAX_PATH] = {};
        HMODULE hSelf = nullptr;
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&FindDefaultModel), &hSelf);
        if (hSelf) GetModuleFileNameW(hSelf, dllPath, MAX_PATH);
        if (dllPath[0]) {
            fs::path dllDir = fs::path(dllPath).parent_path();
            auto r = FirstOnnxIn(dllDir);
            if (!r.empty()) return r;
            r = FirstOnnxIn(dllDir.parent_path());
            if (!r.empty()) return r;
        }
    }

    // 3: Host EXE folder.
    {
        wchar_t exePath[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        if (exePath[0]) {
            auto r = FirstOnnxIn(fs::path(exePath).parent_path());
            if (!r.empty()) return r;
        }
    }

    // 4: %APPDATA%\3Deflatten\models
    {
        wchar_t appdata[MAX_PATH] = {};
        if (SUCCEEDED(SHGetFolderPathW(nullptr, CSIDL_APPDATA, nullptr, 0, appdata))) {
            auto r = FirstOnnxIn(fs::path(appdata) / L"3Deflatten" / L"models");
            if (!r.empty()) return r;
        }
    }

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
        std::string shapeStr;
        for (auto d : inShape) shapeStr += std::to_string(d) + " ";
        LOG_INFO("Model loaded OK");
        LOG_INFO("  path    : ", std::string(path.begin(), path.end()));
        LOG_INFO("  input   : ", m_inputName, "  shape: [", shapeStr, "]");
        LOG_INFO("  output  : ", m_outputName);
        LOG_INFO("  dynamic : ", m_dynamicInput ? "yes" : "no");
        if (!m_dynamicInput)
            LOG_INFO("  fixed   : ", m_modelInputW, "x", m_modelInputH);
        LOG_INFO("  provider: ", std::string(outInfo.begin(), outInfo.end()));
        return S_OK;

    } catch (const Ort::Exception& e) {
        LOG_ERR("ORT exception during Load: ", e.what());
        m_loaded = false;
        return E_FAIL;
    }
}

// ── BuildSessionOptions ───────────────────────────────────────────────────────

// Returns the directory that contains this DLL.
// Used to locate bundled provider DLLs (onnxruntime_providers_*.dll).
static std::wstring GetDllDir() {
    wchar_t path[MAX_PATH] = {};
    HMODULE hm = nullptr;
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&GetDllDir), &hm);
    if (hm) GetModuleFileNameW(hm, path, MAX_PATH);
    wchar_t* sl = wcsrchr(path, L'\\');
    if (!sl) sl = wcsrchr(path, L'/');
    if (sl) *sl = L'\0';
    return path;
}

// Probe whether a DLL can be fully loaded (all its own dependencies resolved).
// LOAD_LIBRARY_AS_DATAFILE only maps the file without resolving imports, so it
// succeeds even when a DLL's dependencies are missing.  We need a real load to
// confirm the DLL is actually usable.
// Returns: 0 = not found, ERROR_MOD_NOT_FOUND(126) = found but deps missing, S_OK = ok
static DWORD DllLoadable(const wchar_t* path) {
    HMODULE h = LoadLibraryExW(path, nullptr,
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
        LOAD_LIBRARY_SEARCH_USER_DIRS    |
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
    if (h) { FreeLibrary(h); return 0; }
    return GetLastError();
}

// Probe for nvcuda.dll (NVIDIA driver).  Also checks that the CUDA *runtime*
// version required by this ORT build is available.
// ORT 1.21 GPU build requires CUDA 12.x (cudart64_12.dll).
static bool CudaDriverPresent() {
    // nvcuda.dll = kernel-mode driver proxy, present if any NVIDIA driver installed
    DWORD e = DllLoadable(L"nvcuda.dll");
    if (e != 0) {
        LOG_WARN("nvcuda.dll not loadable (error ", e, ") – no NVIDIA driver detected. "
                 "CUDA/TRT EPs unavailable.");
        return false;
    }
    // cudart64_12.dll = CUDA 12.x runtime, required by ORT 1.21 GPU build.
    // CUDA 11.x ships cudart64_110.dll which is NOT compatible.
    e = DllLoadable(L"cudart64_12.dll");
    if (e != 0) {
        // Try generic cudart64 (older naming)
        e = DllLoadable(L"cudart64.dll");
    }
    if (e != 0) {
        LOG_WARN("CUDA 12.x runtime (cudart64_12.dll) not found (error ", e, ").");
        LOG_WARN("  This ORT build requires CUDA 12.x. You appear to have an older version.");
        LOG_WARN("  Download: https://developer.nvidia.com/cuda-downloads");
        LOG_WARN("  Minimum driver for CUDA 12: 527.41 (Windows). Your driver: check Device Manager.");
        return false;
    }
    return true;
}

// Probe a provider DLL for loadability and log a clear diagnostic on failure.
// name = short name for logging, path = full path to the DLL.
static bool ProviderDllLoadable(const char* name, const std::wstring& path) {
    DWORD e = DllLoadable(path.c_str());
    if (e == 0) return true;
    if (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND || e == ERROR_MOD_NOT_FOUND) {
        LOG_WARN(name, " not found at '",
                 std::string(path.begin(), path.end()), "'");
    } else if (e == 126) { // ERROR_MOD_NOT_FOUND when file exists = dependency missing
        LOG_WARN(name, " found but its dependencies could not be loaded (error 126).");
        if (std::wstring(path).find(L"tensorrt") != std::wstring::npos) {
            LOG_WARN("  TensorRT runtime is not installed or is the wrong version.");
            LOG_WARN("  ORT 1.21 requires TensorRT 10.x + CUDA 12.x + driver >= 520.");
            LOG_WARN("  Driver 426.06 is too old for TensorRT -- minimum is ~520.61.");
            LOG_WARN("  Download TensorRT: https://developer.nvidia.com/tensorrt");
        } else {
            LOG_WARN("  A required dependency (cuDNN, CUDA runtime, etc.) is missing.");
            LOG_WARN("  ORT 1.21 GPU requires: CUDA 12.x + cuDNN 9.x + driver >= 527.");
            LOG_WARN("  Download cuDNN: https://developer.nvidia.com/cudnn");
        }
    } else {
        LOG_WARN(name, " load failed with error ", e);
    }
    return false;
}

// Returns %LOCALAPPDATA%\3Deflatten\trt_engines as a UTF-8 string.
// TRT compiles CUDA kernels on first use; the cache avoids that on subsequent
// runs (first-run cost: 30-120 s on an RTX 2080 Ti for 1022x1022 input).
static std::string TrtEngineCacheDir() {
    wchar_t appdata[MAX_PATH] = {};
    if (SUCCEEDED(SHGetFolderPathW(nullptr, CSIDL_LOCAL_APPDATA, nullptr, 0, appdata))) {
        std::wstring dir = std::wstring(appdata) + L"\\3Deflatten\\trt_engines";
        CreateDirectoryW(dir.c_str(), nullptr);
        // Convert to narrow UTF-8 string (ORT TRT options take const char*)
        int n = WideCharToMultiByte(CP_UTF8, 0, dir.c_str(), -1, nullptr, 0, nullptr, nullptr);
        std::string out(n, '\0');
        WideCharToMultiByte(CP_UTF8, 0, dir.c_str(), -1, out.data(), n, nullptr, nullptr);
        out.pop_back(); // remove null terminator
        return out;
    }
    return ".\\trt_engines";
}

void DepthEstimator::BuildSessionOptions(GPUProvider provider,
                                          std::wstring& outInfo) {
    m_sessionOpts = Ort::SessionOptions();
    m_sessionOpts.SetIntraOpNumThreads(4);
    m_sessionOpts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    m_sessionOpts.EnableMemPattern();
    m_sessionOpts.EnableCpuMemArena();

    // Returns true if the EP was successfully appended.
    // On failure (DLL absent, CUDA not installed, ORT exception) returns false.
    auto tryEP = [&](GPUProvider ep) -> bool {

        // ── TensorRT ─────────────────────────────────────────────────────────
        if (ep == GPUProvider::TensorRT) {
#ifndef ORT_ENABLE_TENSORRT
            LOG_INFO("TensorRT EP: not compiled in (build uses DirectML backend)");
            return false;
#else
            if (!CudaDriverPresent()) return false;

            // onnxruntime_providers_tensorrt.dll must be next to the .ax
            std::wstring trtDll = GetDllDir() + L"\\onnxruntime_providers_tensorrt.dll";
            if (!ProviderDllLoadable("onnxruntime_providers_tensorrt.dll", trtDll))
                return false;

            try {
                // Store the cache path so the member lives long enough for
                // AppendExecutionProvider_TensorRT to copy it.
                m_trtCacheDir = TrtEngineCacheDir();

                OrtTensorRTProviderOptions trt{};
                trt.device_id                 = 0;
                trt.trt_max_workspace_size    = 2LL * 1024 * 1024 * 1024; // 2 GB
                trt.trt_fp16_enable           = 1;   // ~2x faster on RTX cards
                trt.trt_engine_cache_enable   = 1;
                trt.trt_engine_cache_path     = m_trtCacheDir.c_str();
                trt.trt_dump_subgraphs        = 0;
                m_sessionOpts.AppendExecutionProvider_TensorRT(trt);

                outInfo = L"NVIDIA TensorRT (FP16, engine cache: "
                        + std::wstring(m_trtCacheDir.begin(), m_trtCacheDir.end())
                        + L")";
                LOG_INFO("Execution provider: TensorRT (FP16)");
                LOG_INFO("  Engine cache: ", m_trtCacheDir);
                LOG_WARN("  NOTE: first inference compiles TRT kernels (30-120 s). "
                         "Subsequent runs use the cache and are fast.");
                return true;
            } catch (const Ort::Exception& e) {
                LOG_WARN("TensorRT EP init failed: ", e.what());
                return false;
            }
#endif // ORT_ENABLE_TENSORRT
        }

        // ── CUDA ─────────────────────────────────────────────────────────────
        if (ep == GPUProvider::CUDA) {
#ifndef ORT_ENABLE_CUDA
            LOG_INFO("CUDA EP: not compiled in (build uses DirectML backend)");
            return false;
#else
            if (!CudaDriverPresent()) return false;

            std::wstring cudaDll = GetDllDir() + L"\\onnxruntime_providers_cuda.dll";
            if (!ProviderDllLoadable("onnxruntime_providers_cuda.dll", cudaDll))
                return false;

            try {
                OrtCUDAProviderOptions cuda{};
                cuda.device_id                = 0;
                cuda.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchExhaustive;
                cuda.do_copy_in_default_stream = 1;
                m_sessionOpts.AppendExecutionProvider_CUDA(cuda);
                outInfo = L"NVIDIA CUDA";
                LOG_INFO("Execution provider: CUDA");
                return true;
            } catch (const Ort::Exception& e) {
                LOG_WARN("CUDA EP init failed: ", e.what());
                return false;
            }
#endif // ORT_ENABLE_CUDA
        }

        // ── DirectML ─────────────────────────────────────────────────────────
        if (ep == GPUProvider::DirectML) {
#ifndef ORT_ENABLE_DML
            LOG_INFO("DirectML EP: not compiled in (build uses GPU/CUDA backend)");
            return false;
#else
            // directml.dll ships with Windows 10 1903+ or is bundled next to the .ax
            DWORD _dmlErr = DllLoadable(L"directml.dll");
            if (_dmlErr != 0) {
                LOG_WARN("directml.dll not loadable (error ", _dmlErr, ") – DirectML EP skipped");
                return false;
            }
            try {
                m_sessionOpts.AppendExecutionProvider("DML", {{"device_id", "0"}});
                outInfo = L"DirectML (DX12 GPU)";
                LOG_INFO("Execution provider: DirectML");
                return true;
            } catch (const Ort::Exception& e) {
                LOG_WARN("DirectML EP init failed: ", e.what());
                return false;
            }
#endif // ORT_ENABLE_DML
        }

        return false; // unknown EP
    };

    // ── Explicit provider selection ───────────────────────────────────────────
    // Walk down the fallback chain from the requested provider.
    if (provider == GPUProvider::TensorRT) {
        if (tryEP(GPUProvider::TensorRT)) return;
        LOG_INFO("TensorRT requested but unavailable – trying CUDA");
        provider = GPUProvider::CUDA;
    }
    if (provider == GPUProvider::CUDA) {
        if (tryEP(GPUProvider::CUDA)) return;
        LOG_INFO("CUDA requested but unavailable – trying DirectML");
        provider = GPUProvider::DirectML;
    }
    if (provider == GPUProvider::DirectML) {
        if (tryEP(GPUProvider::DirectML)) return;
        LOG_INFO("DirectML requested but unavailable – falling back to CPU");
        provider = GPUProvider::CPU;
    }
    if (provider == GPUProvider::CPU) {
        outInfo = L"CPU";
        LOG_INFO("Execution provider: CPU");
        return;
    }

    // ── Auto: try best available in order ────────────────────────────────────
    // Auto = 0, so we reach here only when provider == Auto from the start.
    if (tryEP(GPUProvider::TensorRT)) return;
    if (tryEP(GPUProvider::CUDA))     return;
    if (tryEP(GPUProvider::DirectML)) return;
    outInfo = L"CPU (auto fallback – no GPU EP available)";
    LOG_INFO("Execution provider: CPU (auto fallback)");
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
    if (!m_loaded || !m_session) {
        LOG_ERR("Estimate called but model not loaded");
        return E_FAIL;
    }

    try {
        std::vector<float> inputTensor;
        int mw, mh;
        PreprocessFrame(srcData, srcWidth, srcHeight, srcStride,
                        isBGR, inputTensor, mw, mh);

        if (m_estimateCount == 0)
            LOG_INFO("First Estimate call:"
                     " src=", srcWidth, "x", srcHeight,
                     " isBGR=", isBGR ? "yes" : "no",
                     " -> model input=", mw, "x", mh,
                     " tensor_elems=", inputTensor.size());
        ++m_estimateCount;

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
        if (m_estimateCount == 1)
            LOG_INFO("First ORT output: raw depth map=", rawW, "x", rawH,
                     " -> resample to ", srcWidth, "x", srcHeight);

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
