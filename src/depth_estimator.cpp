// SPDX-License-Identifier: GPL-3.0-or-later
#include "depth_estimator.h"
#include "logger.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <shlobj.h>
#include <winreg.h>
#include <vector>
#pragma comment(lib, "Version.lib")  // GetFileVersionInfoW / VerQueryValueW

constexpr float DepthEstimator::MEAN[3];
constexpr float DepthEstimator::STD[3];

// ── DA3-Streaming sentinel ────────────────────────────────────────────────────
constexpr wchar_t DepthEstimator::STREAMING_SENTINEL[];

// ── Helper: locate default model ─────────────────────────────────────────────
// Search order:
//  1. DLL folder       (e.g. Win64/ or Win32/)
//  2. DLL parent       (install root — where models sit next to Win32/Win64/)
//  3. Host EXE folder
//  4. %APPDATA%\3Deflatten\models
// In each folder da3-small.onnx is tried first, then depth_anything_v2_small.onnx,
// then any *.onnx.  DA3-Streaming therefore picks da3-small.onnx automatically
// when it exists in the same folder as the .ax — no copy to LOCALAPPDATA needed.

static std::wstring FirstOnnxIn(const std::filesystem::path& dir) {
    for (auto* name : {L"da3-small.onnx", L"depth_anything_v2_small.onnx"}) {
        auto p = dir / name;
        if (std::filesystem::exists(p)) return p.wstring();
    }
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
                              InferenceRuntime   runtime,
                              std::wstring&      outInfo) {
    // ── DA3-Streaming sentinel ────────────────────────────────────────────────
    bool wantDA3Stream = (modelPath == STREAMING_SENTINEL);

    std::wstring path;
    if (wantDA3Stream || modelPath.empty()) {
        path = FindDefaultModel();
        if (path.empty() && wantDA3Stream) {
            LOG_ERR("DA3-Streaming: no ONNX model found.");
            LOG_ERR("  Place da3-small.onnx next to the .ax file and");
            LOG_ERR("  run Setup.py to download it if needed.");
            return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
        }
        if (!wantDA3Stream && !path.empty() &&
            path.find(L"da3-small") != std::wstring::npos) {
            wantDA3Stream = true;
            LOG_INFO("DA3-Streaming auto-enabled (da3-small.onnx found via auto-detect)");
        } else if (wantDA3Stream && !path.empty() &&
                   path.find(L"da3-small") == std::wstring::npos) {
            LOG_WARN("DA3-Streaming: using '",
                     std::string(path.begin(), path.end()),
                     "' instead of da3-small.onnx (recommended for DA3-Streaming).");
        }
    } else {
        path = modelPath;
    }

    if (path.empty() || !std::filesystem::exists(path)) {
        LOG_ERR("Depth model not found: ",
                std::string(path.begin(), path.end()));
        return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
    }

    LOG_INFO("Loading depth model: ", std::string(path.begin(), path.end()));

    // ── Native TRT-RTX path ───────────────────────────────────────────────────
#ifdef ORT_ENABLE_TRTRTX
    if (runtime == InferenceRuntime::TensorRTRtx ||
        runtime == InferenceRuntime::TensorRTNative) {
        const char* runtimeName = (runtime == InferenceRuntime::TensorRTRtx)
            ? "TRT-RTX" : "TensorRT native";
        LOG_INFO("Native inference runtime: ", runtimeName);
        HRESULT hr = LoadTrtRtxNative(path, outInfo, runtime);
        if (SUCCEEDED(hr)) {
            m_session.reset();
            m_modelPath     = path;
            m_loaded        = true;
            m_da3StreamMode = wantDA3Stream;
            m_streamBuf.clear();
            m_streamAnchor.clear();
            m_streamFrameCount = 0;
        }
        return hr;
    }
#else
    if (runtime == InferenceRuntime::TensorRTRtx ||
        runtime == InferenceRuntime::TensorRTNative) {
        LOG_WARN("Native TRT runtime: not compiled in (build requires ORT_ENABLE_TRTRTX).");
        LOG_WARN("  Falling back to ONNXRuntime path.");
    }
#endif

    // ── ORT path (default) ────────────────────────────────────────────────────
    m_trtRtx.reset();   // discard any previous native session

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

        m_streaming   = false;
        m_ctxReady    = false;
        m_ctxInName.clear();
        m_ctxOutName.clear();
        if (DetectStreamingModel()) {
            m_streaming = true;
            LOG_INFO("  recurrent-context streaming: yes (context_in/out tensors found)");
            LOG_INFO("  ctx_in    : ", m_ctxInName);
            LOG_INFO("  ctx_out   : ", m_ctxOutName);
        } else {
            LOG_INFO("  recurrent-context streaming: no (single-pass model)");
        }

        m_modelPath = path;
        m_loaded    = true;
        m_da3StreamMode    = wantDA3Stream;
        m_streamBuf.clear();
        m_streamAnchor.clear();
        m_streamFrameCount = 0;

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
        if (m_da3StreamMode)
            LOG_INFO("  DA3-Streaming: ENABLED (window=", STREAM_WINDOW,
                     " anchor_reset=", ANCHOR_RESET_FRAMES, ")");
        return S_OK;

    } catch (const Ort::Exception& e) {
        LOG_ERR("ORT exception during Load: ", e.what());
        m_loaded = false;
        return E_FAIL;
    }
}

// ── Native TRT-RTX Session ────────────────────────────────────────────────────
// Dynamically loads nvinfer_10.dll + nvonnxparser_10.dll (from the TRT-RTX
// package, which ships the same API as standard TensorRT 10.x).
// Build path  : ONNX → TRT engine (FP16) → cached as <model>.trtrtx_sm<CC>.bin
// Infer path  : engine → exec context → CUDA graph capture → graph launch
//
// All TRT objects are accessed through opaque void* + vtable offsets so we need
// no TRT headers at compile time — just the two DLLs at runtime.
//
// TRT C++ ABI is stable: the vtable slot for each method has been fixed since
// TRT 8.  We only call methods whose slots are documented in the TRT OSS headers.
//
// CUDA runtime is loaded from the same cudart64_12.dll that the user bundled;
// we use a tiny subset (cudaMalloc, cudaMemcpy, cudaStreamCreate, graph capture).
#ifdef ORT_ENABLE_TRTRTX
// NvInfer.h and NvOnnxParser.h come from -DORT_TRTRTX_HOME=<sdk_root>/include
// cuda_runtime.h comes from FindCUDAToolkit (CUDA Toolkit must be installed).
// Both are linked via target_link_libraries in CMakeLists.txt (Unified backend only).
//
// Suppress C4100 (unreferenced formal parameter) from TRT SDK internal headers —
// the TRT SDK uses pure-virtual stub methods with intentionally unnamed parameters.
#pragma warning(push)
#pragma warning(disable: 4100)
#include <NvInfer.h>
#include <NvOnnxParser.h>
#pragma warning(pop)
#include <cuda_runtime.h>
#include <fstream>

// ── RAII wrappers around TRT objects ─────────────────────────────────────────
// TRT objects are heap-allocated with new() and freed with delete; they are not
// COM objects and do not use AddRef/Release.  Use unique_ptr with default deleter.
using TrtBuilderPtr    = std::unique_ptr<nvinfer1::IBuilder>;
using TrtNetworkPtr    = std::unique_ptr<nvinfer1::INetworkDefinition>;
using TrtConfigPtr     = std::unique_ptr<nvinfer1::IBuilderConfig>;
using TrtParserPtr     = std::unique_ptr<nvonnxparser::IParser>;
using TrtHostMemPtr    = std::unique_ptr<nvinfer1::IHostMemory>;
using TrtRuntimePtr    = std::unique_ptr<nvinfer1::IRuntime>;
using TrtEnginePtr     = std::unique_ptr<nvinfer1::ICudaEngine>;
using TrtContextPtr    = std::unique_ptr<nvinfer1::IExecutionContext>;

// TRT ILogger routed to our LOG system
class TrtRtxLogger : public nvinfer1::ILogger {
public:
    void log(Severity sev, const char* msg) noexcept override {
        if      (sev <= Severity::kWARNING) LOG_WARN("[TRT] ", msg);
        else if (sev == Severity::kINFO)    LOG_INFO("[TRT] ", msg);
    }
};

// ── TrtRtxSession ─────────────────────────────────────────────────────────────
// Holds all TRT and CUDA state for one loaded model.
struct DepthEstimator::TrtRtxSession {
    TrtRtxLogger      logger;
    TrtRuntimePtr     runtime;
    TrtEnginePtr      engine;
    TrtContextPtr     context;

    // Model I/O dimensions (determined from engine after load)
    int modelW = 518, modelH = 518;
    int nbBindings = 0;
    int inputIdx  = -1;
    int outputIdx = -1;

    // CUDA resources
    cudaStream_t stream   = nullptr;
    float*       d_input  = nullptr;   // device [1,3,H,W]
    float*       d_output = nullptr;   // device [1,1,H,W]
    float*       h_output = nullptr;   // pinned host readback

    size_t inputBytes  = 0;
    size_t outputBytes = 0;

    std::string inputName, outputName;
    std::string enginePath;

    // DLL handles for dynamically loaded factory functions.
    // Kept alive for the lifetime of the session; freed in Destroy().
    // Static linking against nvinfer_10.lib / nvonnxparser_10.lib is intentionally
    // avoided: TRT-RTX 1.4 and standard TRT 10.x may use different export names
    // in their .lib files even though the DLLs share the same API surface.
    HMODULE hInfer  = nullptr;
    HMODULE hParser = nullptr;

    ~TrtRtxSession() { Destroy(); }
    void Destroy() {
        context.reset();
        engine.reset();
        runtime.reset();
        if (stream)  { cudaStreamDestroy(stream);  stream  = nullptr; }
        if (d_input)  { cudaFree(d_input);   d_input  = nullptr; }
        if (d_output) { cudaFree(d_output);  d_output = nullptr; }
        if (h_output) { cudaFreeHost(h_output); h_output = nullptr; }
        // Free DLL handles last — TRT objects above must be destroyed first
        if (hParser) { FreeLibrary(hParser); hParser = nullptr; }
        if (hInfer)  { FreeLibrary(hInfer);  hInfer  = nullptr; }
    }
};

// ── Engine cache path ─────────────────────────────────────────────────────────
// Serialized engine is saved next to the .onnx as <model>.trtrtx_sm<CC>_fp16.bin.
// This is the output of buildSerializedNetwork() — binary TRT engine blob.
// It is NOT the same as the .cache file produced by tensorrt_rtx.exe.
static std::string TrtRtxEnginePath(const std::wstring& onnxPath, int sm) {
    return std::string(onnxPath.begin(), onnxPath.end())
         + ".trtrtx_sm" + std::to_string(sm) + "_fp16.bin";
}

// ── LoadTrtRtxNative ──────────────────────────────────────────────────────────
HRESULT DepthEstimator::LoadTrtRtxNative(const std::wstring& onnxPath,
                                          std::wstring& outInfo,
                                          InferenceRuntime runtime) {
    m_trtRtx.reset();
    auto sess = std::make_unique<DepthEstimator::TrtRtxSession>();

    // ── Detect device SM ─────────────────────────────────────────────────────
    int dev = 0;
    cudaError_t cerr = cudaGetDevice(&dev);
    if (cerr != cudaSuccess) {
        LOG_ERR("TRT-RTX: cudaGetDevice failed (", cerr, "). Is a CUDA GPU present?");
        return E_FAIL;
    }
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    int sm = major * 10 + minor;
    LOG_INFO("TRT-RTX: device=", dev, " SM=", sm, " (", major, ".", minor, ")");
    sess->enginePath = TrtRtxEnginePath(onnxPath, sm);

    // ── CUDA stream ───────────────────────────────────────────────────────────
    cerr = cudaStreamCreate(&sess->stream);
    if (cerr != cudaSuccess) {
        LOG_ERR("TRT-RTX: cudaStreamCreate failed (", cerr, ")");
        return E_FAIL;
    }

    // ── Dynamically load TRT factory functions ────────────────────────────────
    // We resolve createInferBuilder_INTERNAL, createInferRuntime_INTERNAL, and
    // createNvOnnxParser_INTERNAL via GetProcAddress instead of relying on the
    // .lib import library.  This decouples us from symbol-name differences between
    // the standard TRT 10.x .lib and TRT-RTX 1.4 .lib — both DLLs export the
    // same functions at runtime, but their .lib files may differ.
    //
    // NvInfer.h signatures (stable since TRT 8):
    //   IBuilder*  createInferBuilder_INTERNAL (void* logger, int32_t version)
    //   IRuntime*  createInferRuntime_INTERNAL (void* logger, int32_t version)
    //   IParser*   createNvOnnxParser_INTERNAL(void* network, void* logger, int32_t version)
    using FnMkBuilder = nvinfer1::IBuilder*       (*)(void*, int32_t);
    using FnMkRuntime = nvinfer1::IRuntime*        (*)(void*, int32_t);
    using FnMkParser  = nvonnxparser::IParser*     (*)(void*, void*, int32_t);

    auto loadTrtDll = [&](const wchar_t* name) -> HMODULE {
        std::wstring full = std::wstring(GetDllDir().c_str()) + L"\\" + name;
        HMODULE h = LoadLibraryExW(full.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS |
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
        if (!h) h = LoadLibraryExW(name, nullptr,
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        if (h) LOG_INFO("TRT: loaded ", std::string(name, name + wcslen(name)));
        else   LOG_ERR ("TRT: failed to load ", std::string(name, name + wcslen(name)));
        return h;
    };

    sess->hInfer  = loadTrtDll(L"nvinfer_10.dll");
    sess->hParser = loadTrtDll(L"nvonnxparser_10.dll");
    if (!sess->hInfer)  { LOG_ERR("TRT: nvinfer_10.dll not found next to the .ax"); return E_FAIL; }
    if (!sess->hParser) { LOG_ERR("TRT: nvonnxparser_10.dll not found next to the .ax"); return E_FAIL; }

    auto fnMkBuilder = reinterpret_cast<FnMkBuilder>(
        GetProcAddress(sess->hInfer, "createInferBuilder_INTERNAL"));
    auto fnMkRuntime = reinterpret_cast<FnMkRuntime>(
        GetProcAddress(sess->hInfer, "createInferRuntime_INTERNAL"));
    auto fnMkParser  = reinterpret_cast<FnMkParser>(
        GetProcAddress(sess->hParser, "createNvOnnxParser_INTERNAL"));

    if (!fnMkBuilder || !fnMkRuntime || !fnMkParser) {
        LOG_ERR("TRT: could not resolve factory functions from DLLs:");
        LOG_ERR("  createInferBuilder_INTERNAL : ", fnMkBuilder ? "OK" : "MISSING");
        LOG_ERR("  createInferRuntime_INTERNAL : ", fnMkRuntime ? "OK" : "MISSING");
        LOG_ERR("  createNvOnnxParser_INTERNAL : ", fnMkParser  ? "OK" : "MISSING");
        LOG_ERR("  Ensure nvinfer_10.dll/nvonnxparser_10.dll are from TRT 10.x or TRT-RTX 1.4+");
        return E_FAIL;
    }
    LOG_INFO("TRT: factory functions resolved OK");

    // ── Attempt to load serialized engine from disk ───────────────────────────
    bool engineLoaded = false;
    if (std::filesystem::exists(sess->enginePath)) {
        LOG_INFO("TRT-RTX: loading cached engine: ", sess->enginePath);
        std::ifstream f(sess->enginePath, std::ios::binary);
        if (f) {
            f.seekg(0, std::ios::end);
            size_t sz = (size_t)f.tellg();
            f.seekg(0, std::ios::beg);
            LOG_INFO("TRT-RTX: cached engine size=", sz, " bytes");
            std::vector<uint8_t> blob(sz);
            if (f.read(reinterpret_cast<char*>(blob.data()), sz)) {
                sess->runtime.reset(fnMkRuntime(&sess->logger, NV_TENSORRT_VERSION));
                if (!sess->runtime) {
                    LOG_ERR("TRT: createInferRuntime returned null (cached engine path).");
                } else {
                    sess->engine.reset(
                        sess->runtime->deserializeCudaEngine(blob.data(), sz));
                    engineLoaded = (sess->engine != nullptr);
                    if (!engineLoaded)
                        LOG_WARN("TRT-RTX: deserializeCudaEngine failed — will rebuild.");
                    else
                        LOG_INFO("TRT-RTX: cached engine deserialized OK.");
                }
            }
        }
    }

    // ── Build engine from ONNX if no valid cache ──────────────────────────────
    if (!engineLoaded) {
        LOG_INFO("TRT-RTX: building FP16 engine from ONNX (may take 1-5 min)...");
        LOG_INFO("  ONNX: ", std::string(onnxPath.begin(), onnxPath.end()));
        LOG_INFO("  Cache will be saved to: ", sess->enginePath);

        TrtBuilderPtr builder(fnMkBuilder(&sess->logger, NV_TENSORRT_VERSION));
        if (!builder) { LOG_ERR("TRT: createInferBuilder returned null."); return E_FAIL; }
        LOG_INFO("TRT-RTX: builder created.");

        // kEXPLICIT_BATCH = 0 (from reference code)
        TrtNetworkPtr network(builder->createNetworkV2(0U));
        if (!network) { LOG_ERR("TRT-RTX: createNetworkV2 returned null."); return E_FAIL; }
        LOG_INFO("TRT-RTX: network created.");

        // Parse ONNX
        TrtParserPtr parser(fnMkParser(network.get(), &sess->logger, NV_ONNX_PARSER_VERSION));
        if (!parser) { LOG_ERR("TRT: createParser returned null."); return E_FAIL; }
        LOG_INFO("TRT-RTX: parser created. Parsing ONNX...");
        std::string onnxA(onnxPath.begin(), onnxPath.end());
        if (!parser->parseFromFile(onnxA.c_str(),
                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            LOG_ERR("TRT-RTX: ONNX parse failed.");
            int ne = parser->getNbErrors();
            for (int i = 0; i < ne; ++i)
                LOG_ERR("  Parse error[", i, "]: ", parser->getError(i)->desc());
            return E_FAIL;
        }
        LOG_INFO("TRT-RTX: ONNX parsed OK.  Inputs=", network->getNbInputs(),
                 " Outputs=", network->getNbOutputs());

        // Log network I/O
        for (int i = 0; i < network->getNbInputs(); ++i) {
            auto* t = network->getInput(i);
            auto  d = t->getDimensions();
            std::string ds;
            for (int k = 0; k < d.nbDims; ++k) ds += std::to_string(d.d[k]) + " ";
            LOG_INFO("  Input[", i, "]: '", t->getName(), "' dims=[", ds, "]");
        }
        for (int i = 0; i < network->getNbOutputs(); ++i) {
            auto* t = network->getOutput(i);
            LOG_INFO("  Output[", i, "]: '", t->getName(), "'");
        }

        // Builder config
        TrtConfigPtr cfg(builder->createBuilderConfig());
        if (!cfg) { LOG_ERR("TRT-RTX: createBuilderConfig returned null."); return E_FAIL; }

        // FP16 — always enable; TRT-RTX and TRT 10.x on RTX hardware always support it.
        // platformHasFastFp16() was removed in TRT-RTX SDK (always true for RTX targets).
        cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
        LOG_INFO("TRT-RTX: FP16 enabled.");

        // Workspace — use setMemoryPoolLimit (TRT 10.x and TRT-RTX 1.4+ API).
        // setMaxWorkspaceSize was removed in TRT 10; TRT-RTX follows TRT 10 API.
        cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, (size_t)2 << 30);
        LOG_INFO("TRT-RTX: workspace=2GB.");

        // Optimization profile for dynamic-shape models
        // Use fixed 518×518 (default Depth Anything v2/v3 input).
        // If the model has no dynamic dims this profile is harmless.
        if (network->getNbInputs() > 0) {
            auto* inp = network->getInput(0);
            auto  dims = inp->getDimensions();
            bool hasDynamic = false;
            for (int k = 0; k < dims.nbDims; ++k)
                if (dims.d[k] < 0) { hasDynamic = true; break; }

            if (hasDynamic) {
                LOG_INFO("TRT-RTX: dynamic input detected — adding optimization profile.");
                auto* profile = builder->createOptimizationProfile();
                // Fix H and W at 518, batch=1, C=3
                nvinfer1::Dims4 fixedDims{1, 3, (int)m_modelInputH, (int)m_modelInputW};
                profile->setDimensions(inp->getName(),
                    nvinfer1::OptProfileSelector::kMIN, fixedDims);
                profile->setDimensions(inp->getName(),
                    nvinfer1::OptProfileSelector::kOPT, fixedDims);
                profile->setDimensions(inp->getName(),
                    nvinfer1::OptProfileSelector::kMAX, fixedDims);
                cfg->addOptimizationProfile(profile);
                LOG_INFO("TRT-RTX: profile set: 1×3×",
                         (int)m_modelInputH, "×", (int)m_modelInputW);
            }
        }

        // Serialize network → engine blob
        LOG_INFO("TRT-RTX: buildSerializedNetwork starting...");
        TrtHostMemPtr serialized(builder->buildSerializedNetwork(*network, *cfg));
        if (!serialized) {
            LOG_ERR("TRT-RTX: buildSerializedNetwork returned null.");
            LOG_ERR("  Check above [TRT] WARNING/ERROR lines for the root cause.");
            return E_FAIL;
        }
        LOG_INFO("TRT-RTX: engine serialized OK (", serialized->size()/1024/1024, " MB).");

        // Save to disk
        {
            std::ofstream f(sess->enginePath, std::ios::binary);
            if (f) {
                f.write(static_cast<const char*>(serialized->data()), serialized->size());
                LOG_INFO("TRT-RTX: engine cached at: ", sess->enginePath);
            } else {
                LOG_WARN("TRT-RTX: could not write engine cache (path writable?)");
            }
        }

        // Deserialize for use
        sess->runtime.reset(fnMkRuntime(&sess->logger, NV_TENSORRT_VERSION));
        if (!sess->runtime) { LOG_ERR("TRT: createInferRuntime returned null (post-build)."); return E_FAIL; }
        sess->engine.reset(sess->runtime->deserializeCudaEngine(
            serialized->data(), serialized->size()));
        if (!sess->engine) { LOG_ERR("TRT-RTX: deserializeCudaEngine (after build) failed."); return E_FAIL; }
        LOG_INFO("TRT-RTX: engine ready.");
    }

    // ── Create execution context ──────────────────────────────────────────────
    sess->context.reset(sess->engine->createExecutionContext());
    if (!sess->context) { LOG_ERR("TRT-RTX: createExecutionContext failed."); return E_FAIL; }
    LOG_INFO("TRT-RTX: execution context created.");

    // ── Query I/O bindings (TRT 10.x / TRT-RTX 1.4+ API) ────────────────────
    // getNbIOTensors / getIOTensorName / getTensorIOMode — present in both
    // standard TRT 10.x and TRT-RTX 1.4.  The legacy TRT 8/9 API
    // (getNbBindings/getBindingName/bindingIsInput) was removed in TRT 10.
    {
        int nb = sess->engine->getNbIOTensors();
        for (int i = 0; i < nb; ++i) {
            const char* name = sess->engine->getIOTensorName(i);
            auto mode = sess->engine->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                sess->inputName = name;
                sess->inputIdx  = i;
                auto d = sess->engine->getTensorShape(name);
                if (d.nbDims == 4) {
                    sess->modelH = (int)(d.d[2] > 0 ? d.d[2] : m_modelInputH);
                    sess->modelW = (int)(d.d[3] > 0 ? d.d[3] : m_modelInputW);
                }
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                sess->outputName = name;
                sess->outputIdx  = i;
            }
            LOG_INFO("TRT: binding[", i, "] '", name,
                     "' ", mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT");
        }
        sess->nbBindings = nb;
        // Set dynamic input shape on the execution context before allocating memory
        if (!sess->inputName.empty()) {
            nvinfer1::Dims4 d{1, 3, sess->modelH, sess->modelW};
            sess->context->setInputShape(sess->inputName.c_str(), d);
            LOG_INFO("TRT: context input shape: 1×3×", sess->modelH, "×", sess->modelW);
        }
    }

    if (sess->inputIdx < 0 || sess->outputIdx < 0) {
        LOG_ERR("TRT-RTX: could not identify input/output bindings.");
        return E_FAIL;
    }
    LOG_INFO("TRT-RTX: input='", sess->inputName, "' idx=", sess->inputIdx,
             "  output='", sess->outputName, "' idx=", sess->outputIdx);
    LOG_INFO("TRT-RTX: model dims: ", sess->modelW, "×", sess->modelH);

    // ── Allocate device/host memory ───────────────────────────────────────────
    sess->inputBytes  = sizeof(float) * 3 * sess->modelH * sess->modelW;
    sess->outputBytes = sizeof(float) * 1 * sess->modelH * sess->modelW;
    cerr = cudaMalloc(reinterpret_cast<void**>(&sess->d_input),  sess->inputBytes);
    if (cerr != cudaSuccess) { LOG_ERR("TRT-RTX: cudaMalloc(input) failed (", cerr, ")"); return E_FAIL; }
    cerr = cudaMalloc(reinterpret_cast<void**>(&sess->d_output), sess->outputBytes);
    if (cerr != cudaSuccess) { LOG_ERR("TRT-RTX: cudaMalloc(output) failed (", cerr, ")"); return E_FAIL; }
    cerr = cudaMallocHost(reinterpret_cast<void**>(&sess->h_output), sess->outputBytes);
    if (cerr != cudaSuccess) { LOG_ERR("TRT-RTX: cudaMallocHost failed (", cerr, ")"); return E_FAIL; }
    LOG_INFO("TRT-RTX: device memory allocated  input=", sess->inputBytes/1024,
             " KB  output=", sess->outputBytes/1024, " KB");

    m_trtRtx     = std::move(sess);
    m_inputName  = m_trtRtx->inputName;
    m_outputName = m_trtRtx->outputName;
    outInfo = (runtime == InferenceRuntime::TensorRTRtx)
        ? L"TensorRT-RTX native (FP16)"
        : L"TensorRT native (FP16)";
    LOG_INFO("Native TRT session ready  runtime=",
             runtime == InferenceRuntime::TensorRTRtx ? "TRT-RTX" : "TensorRT");
    return S_OK;
}

// ── EstimateTrtRtx ────────────────────────────────────────────────────────────
HRESULT DepthEstimator::EstimateTrtRtx(const BYTE* srcData, int srcW, int srcH,
                                         int srcStride, bool isBGR, bool flipDepth,
                                         float smoothAlpha, DepthResult& result) {
    auto& s = *m_trtRtx;

    // Preprocess: BGRA → [1,3,H,W] float tensor (same normalisation as ORT path)
    std::vector<float> inputTensor;
    int mw, mh;
    PreprocessFrame(srcData, srcW, srcH, srcStride, isBGR, inputTensor, mw, mh);

    // Upload to device
    cudaError_t cerr = cudaMemcpyAsync(
        s.d_input, inputTensor.data(), s.inputBytes,
        cudaMemcpyHostToDevice, s.stream);
    if (cerr != cudaSuccess) {
        LOG_ERR("TRT-RTX: cudaMemcpyAsync(HtoD) failed (", cerr, ")");
        return E_FAIL;
    }

    // Run inference using executeV2 (synchronous, old-style bindings array)
    // Bindings array: index 0 = input, index 1 = output (as per nbBindings order).
    std::vector<void*> bindings((size_t)s.nbBindings, nullptr);
    if (s.inputIdx  >= 0 && s.inputIdx  < s.nbBindings) bindings[s.inputIdx]  = s.d_input;
    if (s.outputIdx >= 0 && s.outputIdx < s.nbBindings) bindings[s.outputIdx] = s.d_output;

    if (!s.context->executeV2(bindings.data())) {
        LOG_ERR("TRT-RTX: executeV2 failed.");
        return E_FAIL;
    }

    // Download result
    cerr = cudaMemcpy(s.h_output, s.d_output, s.outputBytes, cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) {
        LOG_ERR("TRT-RTX: cudaMemcpy(DtoH) failed (", cerr, ")");
        return E_FAIL;
    }

    // Postprocess: same pipeline as ORT path
    std::vector<float> out(srcW * srcH);
    PostprocessDepth(s.h_output, mw, mh, srcW, srcH, flipDepth, out);

    if (!m_da3StreamMode && smoothAlpha > 0.f && smoothAlpha < 1.f)
        TemporalSmooth(out, smoothAlpha);

    result.data   = std::move(out);
    result.width  = srcW;
    result.height = srcH;
    return S_OK;
}

#endif // ORT_ENABLE_TRTRTX



// ── BuildSessionOptions ───────────────────────────────────────────────────────

// Returns the directory that contains this DLL (.ax file).
// __declspec(noinline) prevents ICF / inlining which would corrupt the
// address used by GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS.
__declspec(noinline) static std::wstring GetDllDir() {
    wchar_t path[MAX_PATH] = {};
    HMODULE hm = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&GetDllDir), &hm) || !hm) {
        // Fallback: look up the registered InprocServer32 path for our CLSID.
        // This handles the case where PotPlayer copies the .ax to its own
        // directory but the companion DLLs live in the original install path.
        return {};  // caller handles empty case
    }
    GetModuleFileNameW(hm, path, MAX_PATH);
    wchar_t* sl = wcsrchr(path, L'\\');
    if (!sl) sl = wcsrchr(path, L'/');
    if (sl) *sl = L'\0';
    return path;
}

// Read the InprocServer32 path for our filter CLSID from the registry.
// PotPlayer (and some other hosts) copy the .ax to their own directory when
// registering it as an external filter. The registry retains the original
// path, which is where onnxruntime.dll and the model files actually live.
// Only needed by ProviderDllLoadable, which is itself GPU-only.
#if defined(ORT_ENABLE_CUDA) || defined(ORT_ENABLE_TENSORRT)
static std::wstring GetRegisteredDllDir() {
    // CLSID_3Deflatten = {4D455F32-1A2B-4C3D-8E4F-5A6B7C8D9E0F}
    const wchar_t* key =
        L"CLSID\\{4D455F32-1A2B-4C3D-8E4F-5A6B7C8D9E0F}\\InprocServer32";
    wchar_t regPath[MAX_PATH] = {};
    DWORD cb = sizeof(regPath);
    if (RegGetValueW(HKEY_CLASSES_ROOT, key, nullptr,
                     RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY,
                     nullptr, regPath, &cb) != ERROR_SUCCESS) {
        RegGetValueW(HKEY_CLASSES_ROOT, key, nullptr,
                     RRF_RT_REG_SZ | RRF_SUBKEY_WOW6432KEY,
                     nullptr, regPath, &cb);
    }
    if (!regPath[0]) return {};
    wchar_t* sl = wcsrchr(regPath, L'\\');
    if (sl) *sl = L'\0';
    return regPath;
}
#endif // ORT_ENABLE_CUDA || ORT_ENABLE_TENSORRT

// Probe whether a DLL can be fully loaded with all its dependencies resolved.
// LOAD_LIBRARY_AS_DATAFILE succeeds even when deps are missing, so we need a
// real load.  IMPORTANT: LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR requires an ABSOLUTE
// path -- using it with a bare DLL name causes ERROR_INVALID_PARAMETER (87).
// For bare names we include USER_DIRS so directories registered via
// AddDllDirectory() (including our own Win64/ folder) are searched too.
static DWORD DllLoadable(const wchar_t* path) {
    bool isAbsPath = (wcschr(path, L'\\') || wcschr(path, L'/'));
    DWORD flags = isAbsPath
        ? (LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
           LOAD_LIBRARY_SEARCH_USER_DIRS    |
           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
        : (LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
           LOAD_LIBRARY_SEARCH_USER_DIRS);   // includes AddDllDirectory paths
    HMODULE h = LoadLibraryExW(path, nullptr, flags);
    if (h) { FreeLibrary(h); return 0; }
    return GetLastError();
}

// Probe a single named DLL, log its status as INFO.  Used for the full
// dependency scan logged when CUDA/TRT init begins.
// Only referenced inside LogCudaDependencies, which is GPU-only.
#if defined(ORT_ENABLE_CUDA) || defined(ORT_ENABLE_TENSORRT)
static bool ProbeDep(const wchar_t* name, const wchar_t* purpose) {
    DWORD e = DllLoadable(name);
    if (e == 0) {
        LOG_INFO("  [OK]     ", std::wstring(name), "  (", std::wstring(purpose), ")");
        return true;
    }
    // Error 2 / 126 / 127 / 193 are all "not found or wrong arch"
    LOG_WARN("  [MISSING] ", std::wstring(name), "  (", std::wstring(purpose),
             ")  error=", e);
    return false;
}

// Log every DLL that the CUDA and TRT EPs depend on, with OK/MISSING status.
// Called once before attempting EP init so the user always sees the full
// dependency picture, not just the first missing piece.
static void LogCudaDependencies(bool includeTrt) {
    LOG_INFO("--- CUDA/TRT dependency scan ---");
#if ORT_CUDA_MAJOR == 13
    LOG_INFO("  ORT 1.24.3 gpu_cuda13 build requirements:");
    LOG_INFO("    CUDA 13.0  (cudart64_13.dll)");
    LOG_INFO("    cuDNN 9.x  (cudnn64_9.dll + 7 split DLLs)");
    if (includeTrt)
        LOG_INFO("    TensorRT 10.13.3.9 (CUDA 13.0 build)");
    LOG_INFO("  NOTE: ORT 1.24.3 gpu_cuda13 was compiled against CUDA 13.0.");
    LOG_INFO("        Using CUDA 13.1+ DLLs at runtime causes error=1114 (version mismatch).");
    LOG_INFO("  NOTE: Driver 572+ required for CUDA 13.0.");
#elif ORT_CUDA_MAJOR == 12
    LOG_INFO("  Unified build requirements (CUDA 12.9.1 / TRT 10.16.0.72):");
    LOG_INFO("    CUDA 12.x  (cudart64_12.dll)");
    LOG_INFO("    cuDNN 9.x  (cudnn64_9.dll + 7 split DLLs)");
    if (includeTrt)
        LOG_INFO("    TensorRT 10.16.0.72 (CUDA 12.9 build)");
    LOG_INFO("  NOTE: Run collect_runtime_dlls_cuda12.py --trt-zip <trt.zip> to bundle DLLs.");
    LOG_INFO("  NOTE: Driver 525+ required for CUDA 12.x.");
#else
    LOG_INFO("  ORT 1.18.1 GPU build requirements:");
    LOG_INFO("    CUDA 11.x  (cudart64_110.dll)");
    LOG_INFO("    cuDNN 8.x  (cudnn64_8.dll + split infer/train DLLs)");
    if (includeTrt)
        LOG_INFO("    TensorRT 10.0.0.6 (CUDA 11.8 build)");
    LOG_INFO("  NOTE: ORT 1.18.1 requires CUDA 11.8 exactly.");
    LOG_INFO("        Using CUDA 11.7 DLLs causes error=1114 (runtime version check).");
    LOG_INFO("  NOTE: nvJitLink is NOT required for CUDA 11 / ORT 1.18.x.");
    LOG_INFO("  NOTE: Driver 452+ required for CUDA 11.x.");
#endif
    LOG_INFO("");

    // NVIDIA driver kernel proxy
    ProbeDep(L"nvcuda.dll", L"NVIDIA driver kernel proxy -- must be in System32");

#if ORT_CUDA_MAJOR == 13
    // ── CUDA 13.0 ─────────────────────────────────────────────────────────
    bool hasCuda = ProbeDep(L"cudart64_13.dll", L"CUDA 13.x runtime");
    if (!hasCuda) LOG_WARN("  -> cudart64_13.dll not found. Run collect_runtime_dlls_cuda13.py.");
    ProbeDep(L"cublas64_13.dll",    L"cuBLAS 13 -- in CUDA Toolkit bin");
    ProbeDep(L"cublasLt64_13.dll",  L"cuBLAS-Lt 13 -- in CUDA Toolkit bin");
    ProbeDep(L"cufft64_12.dll",     L"cuFFT 12 -- in CUDA Toolkit bin");
    bool hasJitLink = ProbeDep(L"nvJitLink_130_0.dll", L"CUDA 13 JIT-Link (required by ORT CUDA EP)");
    if (!hasJitLink) LOG_WARN("  nvJitLink_130_0.dll missing -- CUDA EP will fail even if other DLLs are present.");
    ProbeDep(L"cusolver64_11.dll",  L"cuSolver 11 -- loaded by ORT CUDA EP at startup");
    ProbeDep(L"curand64_10.dll",    L"cuRand 10 -- loaded by ORT CUDA EP at startup");
    bool hasCudnn = ProbeDep(L"cudnn64_9.dll", L"cuDNN 9.x main library");
    if (!hasCudnn) LOG_WARN("  -> cudnn64_9.dll not found. Run collect_runtime_dlls_cuda13.py.");
#elif ORT_CUDA_MAJOR == 12
    // ── CUDA 12.x ─────────────────────────────────────────────────────────
    bool hasCuda = ProbeDep(L"cudart64_12.dll", L"CUDA 12.x runtime");
    if (!hasCuda) LOG_WARN("  -> cudart64_12.dll not found. Run collect_runtime_dlls_cuda12.py.");
    ProbeDep(L"cublas64_12.dll",    L"cuBLAS 12 -- in CUDA Toolkit bin");
    ProbeDep(L"cublasLt64_12.dll",  L"cuBLAS-Lt 12 -- in CUDA Toolkit bin");
    ProbeDep(L"cufft64_12.dll",     L"cuFFT 12 -- in CUDA Toolkit bin");
    bool hasJitLink = ProbeDep(L"nvJitLink_120_0.dll", L"CUDA 12 JIT-Link (required by ORT CUDA EP)");
    if (!hasJitLink) LOG_WARN("  nvJitLink_120_0.dll missing -- CUDA EP will fail even if other DLLs are present.");
    ProbeDep(L"cusolver64_11.dll",  L"cuSolver 11 -- loaded by ORT CUDA EP at startup");
    ProbeDep(L"curand64_10.dll",    L"cuRand 10 -- loaded by ORT CUDA EP at startup");
    bool hasCudnn = ProbeDep(L"cudnn64_9.dll", L"cuDNN 9.x main library");
    if (!hasCudnn) LOG_WARN("  -> cudnn64_9.dll not found. Run collect_runtime_dlls_cuda12.py.");
#else
    // ── CUDA 11.x ─────────────────────────────────────────────────────────
    bool hasCuda = ProbeDep(L"cudart64_110.dll", L"CUDA 11.x runtime");
    if (!hasCuda) LOG_WARN("  -> cudart64_110.dll not found. Run collect_runtime_dlls.py.");
    ProbeDep(L"cublas64_11.dll",   L"cuBLAS 11 -- in CUDA Toolkit bin");
    ProbeDep(L"cublasLt64_11.dll", L"cuBLAS-Lt 11 -- in CUDA Toolkit bin");
    ProbeDep(L"cufft64_10.dll",    L"cuFFT 10 -- in CUDA Toolkit bin");
    ProbeDep(L"cusolver64_11.dll", L"cuSolver 11 -- loaded by ORT CUDA EP at startup");
    ProbeDep(L"curand64_10.dll",   L"cuRand 10 -- loaded by ORT CUDA EP at startup");
    bool hasCudnn = ProbeDep(L"cudnn64_8.dll", L"cuDNN 8.x main library");
    if (!hasCudnn) LOG_WARN("  -> cudnn64_8.dll not found. Run collect_runtime_dlls.py.");
#endif

    if (includeTrt) {
        LOG_INFO("");
#if ORT_CUDA_MAJOR >= 12
        // TRT 10.3+ uses _10 suffix on all main DLLs
        LOG_INFO("  TensorRT 10.x libraries (CUDA 12/13 build):");
        LOG_INFO("  NOTE: zlibwapi.dll is NOT required by TRT 10.3+.");
        ProbeDep(L"nvinfer_10.dll",          L"TRT 10.x inference engine");
        ProbeDep(L"nvonnxparser_10.dll",     L"TRT 10.x ONNX parser");
        ProbeDep(L"nvinfer_dispatch_10.dll", L"TRT 10.x dispatch runtime");
        ProbeDep(L"nvinfer_lean_10.dll",     L"TRT 10.x lean runtime");
        ProbeDep(L"nvinfer_plugin_10.dll",   L"TRT 10.x plugins");
#else
        LOG_INFO("  TensorRT 10.0.x libraries (CUDA 11.8 build):");
        LOG_INFO("  NOTE: TRT 10.0.x uses plain names (nvinfer.dll, NOT nvinfer_10.dll).");
        LOG_INFO("        The _10 suffix was introduced in TRT 10.3+.");
        ProbeDep(L"nvinfer.dll",        L"TRT 10.0 inference engine");
        ProbeDep(L"nvonnxparser.dll",   L"TRT 10.0 ONNX parser");
        ProbeDep(L"nvinfer_plugin.dll", L"TRT 10.0 plugins");
#endif
    }
    LOG_INFO("--- end dependency scan ---");
}
#endif // ORT_ENABLE_CUDA || ORT_ENABLE_TENSORRT

#if defined(ORT_ENABLE_CUDA) || defined(ORT_ENABLE_TENSORRT)
// Probe for nvcuda.dll + CUDA runtime. Returns false if absent.
static bool CudaDriverPresent() {
    DWORD e = DllLoadable(L"nvcuda.dll");
    if (e != 0) {
        LOG_WARN("nvcuda.dll not loadable (error ", e, ") – no NVIDIA driver detected.");
        return false;
    }
#if ORT_CUDA_MAJOR == 13
    if (DllLoadable(L"cudart64_13.dll") != 0) {
        LOG_WARN("cudart64_13.dll not found. Run collect_runtime_dlls_cuda13.py.");
        return false;
    }
#elif ORT_CUDA_MAJOR == 12
    if (DllLoadable(L"cudart64_12.dll") != 0) {
        LOG_WARN("cudart64_12.dll not found. Run collect_runtime_dlls_cuda12.py.");
        return false;
    }
#else
    if (DllLoadable(L"cudart64_110.dll") != 0) {
        LOG_WARN("cudart64_110.dll not found. Run collect_runtime_dlls.py.");
        return false;
    }
#endif
    return true;
}

// ---------------------------------------------------------------------------
// PE import scanner: maps a DLL as a data file and walks its import directory
// to report exactly which DLLs it needs and whether each one is loadable.
// This is the same information Dependencies.exe / depends.exe shows, but
// embedded directly in the log so no external tool is needed.
//
// C2712 rule: __try/__except cannot be in a function that requires object
// unwinding (i.e. has C++ objects with destructors).  We solve this by
// splitting into two functions:
//   ScanPeImports_SEH  -- contains ONLY __try/__except, no C++ objects at all.
//                         Uses a fixed-size C array to return results.
//   ScanAndLogMissingImports -- handles all C++ (std::wstring, LOG_*) and
//                         calls ScanPeImports_SEH.
// ---------------------------------------------------------------------------

// Max imports we track (practically no DLL has more than 64 direct deps)
static constexpr int PE_SCAN_MAX = 64;

struct PeScanResult {
    enum Status : BYTE { EMPTY = 0, OK_DEP, MISSING_DEP, BAD_PE, EXCEPTION };
    char     names[PE_SCAN_MAX][64];  // import DLL names (truncated to 63 chars)
    DWORD    errors[PE_SCAN_MAX];     // 0 = loadable, non-zero = error code
    int      count;                   // number of entries filled
    Status   status;                  // overall result
};

// RVA-to-file-offset helper: plain static function so it can be called from
// inside a __try block without triggering C2712 (no destructor, no lambda).
static DWORD RvaToFileOffset(const IMAGE_SECTION_HEADER* sec, WORD numSec, DWORD rva) {
    for (WORD i = 0; i < numSec; ++i) {
        if (rva >= sec[i].VirtualAddress &&
            rva <  sec[i].VirtualAddress + sec[i].SizeOfRawData)
            return rva - sec[i].VirtualAddress + sec[i].PointerToRawData;
    }
    return 0;
}

// Inner function: NO C++ objects, safe for __try/__except.
static void ScanPeImports_SEH(const BYTE* base, PeScanResult* out) {
    out->count  = 0;
    out->status = PeScanResult::BAD_PE;
    __try {
        const IMAGE_DOS_HEADER* dos = (const IMAGE_DOS_HEADER*)base;
        if (dos->e_magic != IMAGE_DOS_SIGNATURE) return;
        const IMAGE_NT_HEADERS* nt  = (const IMAGE_NT_HEADERS*)(base + dos->e_lfanew);
        if (nt->Signature != IMAGE_NT_SIGNATURE) return;

        WORD numSec = nt->FileHeader.NumberOfSections;
        const IMAGE_SECTION_HEADER* sec = IMAGE_FIRST_SECTION(nt);

        DWORD importRVA =
            nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress;
        if (!importRVA) { out->status = PeScanResult::OK_DEP; return; }
        DWORD importOff = RvaToFileOffset(sec, numSec, importRVA);
        if (!importOff) { out->status = PeScanResult::OK_DEP; return; }

        const IMAGE_IMPORT_DESCRIPTOR* desc =
            (const IMAGE_IMPORT_DESCRIPTOR*)(base + importOff);
        for (; desc->Name && out->count < PE_SCAN_MAX; ++desc) {
            DWORD nameOff = RvaToFileOffset(sec, numSec, desc->Name);
            if (!nameOff) continue;
            const char* src = (const char*)(base + nameOff);
            int j = 0;
            for (; j < 63 && src[j]; ++j)
                out->names[out->count][j] = src[j];
            out->names[out->count][j] = '\0';
            ++out->count;
        }
        out->status = PeScanResult::OK_DEP;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        out->status = PeScanResult::EXCEPTION;
    }
}

// Outer function: all C++ objects live here; no __try/__except.
static void ScanAndLogMissingImports(const std::wstring& dllPath) {
    HANDLE hFile = CreateFileW(dllPath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, 0, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        LOG_WARN("  [dep-scan] Cannot open: ", dllPath, " (error=", GetLastError(), ")");
        return;
    }
    HANDLE hMap = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(hFile);
    if (!hMap) { LOG_WARN("  [dep-scan] CreateFileMapping failed"); return; }
    const BYTE* base = (const BYTE*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    if (!base) { LOG_WARN("  [dep-scan] MapViewOfFile failed"); return; }

    LOG_INFO("  [dep-scan] Direct imports of: ", dllPath);

    PeScanResult res{};
    ScanPeImports_SEH(base, &res);
    UnmapViewOfFile(base);

    if (res.status == PeScanResult::EXCEPTION) {
        LOG_WARN("  [dep-scan] Access violation reading PE headers -- file may be corrupt.");
        return;
    }
    if (res.status == PeScanResult::BAD_PE) {
        LOG_WARN("  [dep-scan] Not a valid PE/DLL file.");
        return;
    }
    if (res.count == 0) {
        LOG_INFO("  [dep-scan] No imports found (or import directory empty).");
        return;
    }

    // Now check each import name (C++ objects are fine here)
    int okCnt = 0, missCnt = 0;
    for (int i = 0; i < res.count; ++i) {
        const char* name = res.names[i];
        wchar_t wName[64];
        for (int j = 0; j < 64; ++j) wName[j] = (wchar_t)(unsigned char)name[j];
        DWORD e = DllLoadable(wName);
        if (e == 0) {
            LOG_INFO("  [dep-scan]   [OK]      ", name);
            ++okCnt;
        } else {
            LOG_WARN("  [dep-scan]   [MISSING] ", name, "  (error=", e, ")");
            ++missCnt;
        }
    }
    LOG_INFO("  [dep-scan] Summary: ", okCnt, " OK,  ", missCnt, " MISSING");
}

// Probe a provider DLL with full-load check, log result with detail on error 126.
static bool ProviderDllLoadable(const char* name, const std::wstring& path) {
    DWORD e = DllLoadable(path.c_str());
    if (e == 0) return true;
    std::string narrow(path.begin(), path.end());
    if (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) {
        LOG_WARN(name, " not found at '", narrow, "'");
        std::wstring regDir = GetRegisteredDllDir();
        std::wstring dllDir = GetDllDir();
        if (!regDir.empty() && _wcsicmp(regDir.c_str(), dllDir.c_str()) != 0) {
            LOG_WARN("  DLL dir = ", dllDir);
            LOG_WARN("  Registry install dir = ", regDir);
            LOG_WARN("  Hint: copy onnxruntime*.dll to the registry dir,");
            LOG_WARN("  OR move the .ax back to its original install directory.");
        }
    } else if (e == 126) {
        LOG_WARN(name, " exists but failed to load (error 126 = missing dependency).");
        // Scan the PE import table to identify the exact missing DLL.
        // This is the same info Dependencies.exe shows -- no external tool needed.
        ScanAndLogMissingImports(path);
        bool isTrt = (std::wstring(path).find(L"tensorrt") != std::wstring::npos ||
                      std::wstring(path).find(L"nvinfer")   != std::wstring::npos);
        if (isTrt) {
            LOG_WARN("  TIP: Copy ALL DLLs from TensorRT lib\\ folder next to the .ax,");
#if ORT_CUDA_MAJOR == 13
            LOG_WARN("       not just nvinfer_10.dll -- TRT has many sub-dependencies.");
#else
            LOG_WARN("       not just nvinfer.dll -- TRT has many sub-dependencies.");
#endif
        }
    } else if (e == 1114) {
        // ERROR_DLL_INIT_FAILED: all dependencies loaded but DllMain returned FALSE.
        // All dep probes [OK] means this is NOT a missing-file issue.
        // Run the import scanner anyway to confirm nothing slipped through.
        bool isTrt = (std::wstring(path).find(L"tensorrt") != std::wstring::npos);
        LOG_WARN(name, " load failed error=1114 (DLL init failed -- all deps present).");
        LOG_WARN("  Running PE import scan to confirm no hidden missing dep:");
        ScanAndLogMissingImports(path);
        if (isTrt) {
            LOG_WARN("  TRT EP DllMain failed.  If CUDA EP also failed, fix CUDA EP first.");
        } else {
#if ORT_CUDA_MAJOR == 13
            LOG_WARN("  CUDA EP: DllMain failed. Possible causes:");
            LOG_WARN("    (A) CUDA minor version mismatch -- ORT 1.24.3 requires CUDA 13.0.");
            LOG_WARN("        Run collect_runtime_dlls_cuda13.py to rebundle correct DLLs.");
            LOG_WARN("    (B) Driver too old -- CUDA 13.0 requires driver 572+.");
            LOG_WARN("        Verify: nvidia-smi shows Driver Version 572+");
#else
            LOG_WARN("  CUDA EP: DllMain failed. Possible causes:");
            LOG_WARN("    (A) Wrong CUDA runtime version. ORT 1.18.1 requires CUDA 11.8.");
            LOG_WARN("        CUDA 11.7 (or older) causes this exact error on drivers 520+.");
            LOG_WARN("        Fix: delete Win64_GPU DLLs and re-run collect_runtime_dlls.py.");
            LOG_WARN("        This downloads CUDA 11.8 DLLs automatically.");
            LOG_WARN("        NOTE: If PyTorch GPU works on this machine, its bundled CUDA 11.8");
            LOG_WARN("        DLLs confirm CUDA works -- you just need the 11.8 DLLs for us.");
            LOG_WARN("    (B) Driver too old -- CUDA 11.8 requires driver 520+.");
            LOG_WARN("        Verify: nvidia-smi shows Driver Version 520+");
#endif
        }
    } else {
        LOG_WARN(name, " load failed error=", e, " path='", narrow, "'");
    }
    return false;
}

#endif // ORT_ENABLE_CUDA || ORT_ENABLE_TENSORRT

#ifdef ORT_ENABLE_TENSORRT
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
#endif // ORT_ENABLE_TENSORRT

void DepthEstimator::BuildSessionOptions(GPUProvider provider,
                                          std::wstring& outInfo) {
    m_sessionOpts = Ort::SessionOptions();
    m_sessionOpts.SetIntraOpNumThreads(4);
    m_sessionOpts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    m_sessionOpts.EnableMemPattern();
    m_sessionOpts.EnableCpuMemArena();

    // ── Diagnostic: log which onnxruntime.dll is actually loaded ─────────────
    // "DML execution provider is not supported in this build" means the wrong
    // onnxruntime.dll was loaded (CPU-only or CUDA build, not DirectML NuGet).
    // Log the actual module path so the user can see which one won the race.
    {
        HMODULE hOrt = nullptr;
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               L"onnxruntime", &hOrt) && hOrt) {
            wchar_t ortPath[MAX_PATH]{};
            GetModuleFileNameW(hOrt, ortPath, MAX_PATH);
            LOG_INFO("ORT: loaded onnxruntime.dll from: ", std::wstring(ortPath));
        } else {
            LOG_WARN("ORT: GetModuleHandleEx(onnxruntime) failed — DLL not yet in process?");
        }

        // List all EPs that this ORT build actually supports.
        // If "DmlExecutionProvider" is missing here, the wrong ORT DLL is loaded.
        auto providers = Ort::GetAvailableProviders();
        std::string epList;
        for (auto& p : providers) { if (!epList.empty()) epList += ", "; epList += p; }
        LOG_INFO("ORT: available execution providers: [", epList, "]");
        bool hasDml = false;
        for (auto& p : providers) if (p == "DmlExecutionProvider") { hasDml = true; break; }
        if (!hasDml) {
            LOG_WARN("ORT: 'DmlExecutionProvider' NOT in available providers list.");
            LOG_WARN("  This means the loaded onnxruntime.dll was built WITHOUT DirectML.");
            LOG_WARN("  The DirectML build (.ax in Win64/) requires the NuGet onnxruntime.dll");
            LOG_WARN("  from Microsoft.ML.OnnxRuntime.DirectML — NOT the GPU zip build.");
            LOG_WARN("  Check that Win64\\onnxruntime.dll came from the NuGet package.");
            LOG_WARN("  If Win64_GPU\\ or Win64_GPU11\\ also exist, PotPlayer may have");
            LOG_WARN("  loaded the wrong onnxruntime.dll from a different registered filter.");
            LOG_WARN("  Fix: register ONLY the Win64 .ax, or ensure Win64\\ is first in");
            LOG_WARN("  the DLL search path (checked before Win64_GPU\\ and System32).");
        }
    }

    // Returns true if the EP was successfully appended.
    // On failure (DLL absent, CUDA not installed, ORT exception) returns false.
    auto tryEP = [&](GPUProvider ep) -> bool {

        // ── TensorRT RTX ─────────────────────────────────────────────────────
        // Requires ORT built from source with --use_nv_tensorrt_rtx.
        // The EP is compiled into onnxruntime.dll directly (no side DLL).
        // Provider string is kNvTensorRTRTXExecutionProvider, which ORT defines as
        // "NvTensorRTRTXExecutionProvider" — must match exactly what GetAvailableProviders()
        // enumerates.  AppendExecutionProvider uses the same registry key.
        if (ep == GPUProvider::TensorRTRtx) {
#ifndef ORT_ENABLE_TRTRTX
            LOG_INFO("TRT-RTX EP: not compiled in (build needs ORT built with --use_nv_tensorrt_rtx)");
            return false;
#else
            try {
                m_sessionOpts.AppendExecutionProvider("NvTensorRTRTXExecutionProvider", {});
                outInfo = L"TensorRT-RTX (built-in EP)";
                LOG_INFO("Execution provider: TensorRT-RTX (NvTensorRTRTXExecutionProvider)");
                return true;
            } catch (const Ort::Exception& e) {
                LOG_WARN("TRT-RTX EP init failed: ", e.what());
                LOG_WARN("  Ensure onnxruntime.dll was built with --use_nv_tensorrt_rtx");
                LOG_WARN("  and that TensorRT-RTX runtime libs are present.");
                return false;
            }
#endif
        }

        // ── TensorRT ─────────────────────────────────────────────────────────
        if (ep == GPUProvider::TensorRT) {
#ifndef ORT_ENABLE_TENSORRT
            LOG_INFO("TensorRT EP: not compiled in (build uses DirectML backend)");
            return false;
#else
            LogCudaDependencies(/*includeTrt=*/true);
            if (!CudaDriverPresent()) return false;

            // onnxruntime_providers_tensorrt.dll must be next to the .ax
            std::wstring trtDll = GetDllDir() + L"\\onnxruntime_providers_tensorrt.dll";
            if (!ProviderDllLoadable("onnxruntime_providers_tensorrt.dll", trtDll))
                return false;

            try {
                m_trtCacheDir = TrtEngineCacheDir();

                // Use the V2 provider options API which works across all ORT versions
                // that support TRT.  The old OrtTensorRTProviderOptions struct API is
                // deprecated in ORT built from source (≥1.19).
                const OrtApi& ortApi = Ort::GetApi();
                OrtTensorRTProviderOptionsV2* trtOpts = nullptr;
                Ort::ThrowOnError(ortApi.CreateTensorRTProviderOptions(&trtOpts));
                // Copy the function pointer to a local — decltype on a struct
                // field pointer gives void(*const*)() (ptr-to-ptr), but we need
                // the plain function pointer type as the unique_ptr deleter.
                auto trtRelease = ortApi.ReleaseTensorRTProviderOptions;
                std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(trtRelease)>
                    trtOptsGuard(trtOpts, trtRelease);

                const char* keys[] = {
                    "device_id",
                    "trt_fp16_enable",
                    "trt_engine_cache_enable",
                    "trt_engine_cache_path",
                    "trt_dump_subgraphs",
                    nullptr
                };
                const char* vals[] = {
                    "0",
                    "1",
                    "1",
                    m_trtCacheDir.c_str(),
                    "0",
                    nullptr
                };
                int nOpts = 5;
                Ort::ThrowOnError(ortApi.UpdateTensorRTProviderOptions(
                    trtOpts, keys, vals, nOpts));
                Ort::ThrowOnError(ortApi.SessionOptionsAppendExecutionProvider_TensorRT_V2(
                    m_sessionOpts, trtOpts));

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
                LOG_WARN("  Common causes:");
                LOG_WARN("    1. TRT version does not match the ORT build's CUDA version.");
                LOG_WARN("       Check zip filename for the cuda-XX.N suffix.");
#if ORT_CUDA_MAJOR == 13
                LOG_WARN("       TRT 10.13.x (cuda-13.0 build) is required.");
                LOG_WARN("    2. nvinfer_10.dll / nvonnxparser_10.dll missing or their deps missing.");
#elif ORT_CUDA_MAJOR == 12
                LOG_WARN("       TRT 10.16.x (cuda-12.9 build) is required.");
                LOG_WARN("    2. nvinfer_10.dll / nvonnxparser_10.dll missing or their deps missing.");
#else
                LOG_WARN("       TRT 10.0.x (cuda-11.8 build) is required.");
                LOG_WARN("    2. nvinfer.dll / nvonnxparser.dll missing or their deps missing.");
                LOG_WARN("       NOTE: TRT 10.0.x uses plain names (no _10 suffix).");
#endif
                LOG_WARN("    3. Ensure all TRT lib\\ DLLs are next to the .ax file,");
                LOG_WARN("       or set TRT_LIB_PATH before launching the host app.");
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
            LogCudaDependencies(/*includeTrt=*/false);
            if (!CudaDriverPresent()) return false;

            std::wstring cudaDll = GetDllDir() + L"\\onnxruntime_providers_cuda.dll";
            if (!ProviderDllLoadable("onnxruntime_providers_cuda.dll", cudaDll))
                return false;

            try {
                // Use the V2 provider options API (the old OrtCUDAProviderOptions
                // struct API is deprecated in ORT source builds ≥1.19).
                const OrtApi& ortApi = Ort::GetApi();
                OrtCUDAProviderOptionsV2* cudaOpts = nullptr;
                Ort::ThrowOnError(ortApi.CreateCUDAProviderOptions(&cudaOpts));
                auto cudaRelease = ortApi.ReleaseCUDAProviderOptions;
                std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(cudaRelease)>
                    cudaOptsGuard(cudaOpts, cudaRelease);

                const char* keys[] = {"device_id", nullptr};
                const char* vals[] = {"0",          nullptr};
                Ort::ThrowOnError(ortApi.UpdateCUDAProviderOptions(
                    cudaOpts, keys, vals, 1));
                Ort::ThrowOnError(ortApi.SessionOptionsAppendExecutionProvider_CUDA_V2(
                    m_sessionOpts, cudaOpts));

                outInfo = L"NVIDIA CUDA";
                LOG_INFO("Execution provider: CUDA");
                return true;
            } catch (const Ort::Exception& e) {
                LOG_WARN("CUDA EP init failed: ", e.what());
                LOG_WARN("  Ensure all CUDA + cuDNN DLLs are next to the .ax file.");
                LOG_WARN("  Run collect_runtime_dlls_cuda12.py --trt-zip <file> to bundle them.");
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
            // ── Dependency check: directml.dll ────────────────────────────────
            // Now uses LOAD_LIBRARY_SEARCH_USER_DIRS so our Win64\ dir is searched.
            // Error 1114 (DLL_INIT_FAILED) from our probe doesn't mean the DLL is wrong —
            // directml.dll's DllMain needs the DX12 device context that ORT creates later.
            // We treat 1114 as a warning and proceed; ORT's own load will succeed.
            DWORD _dmlErr = DllLoadable(L"directml.dll");
            if (_dmlErr != 0 && _dmlErr != 1114) {
                LOG_WARN("directml.dll not loadable (error=", _dmlErr, ")");
                LOG_WARN("  Possible causes:");
                LOG_WARN("    error=2  (not found): place directml.dll from Microsoft.AI.DirectML");
                LOG_WARN("             NuGet next to the .ax file, or update Windows 10 1903+");
                LOG_WARN("    error=193 (bad image): wrong architecture (x86 vs x64)");
                LOG_WARN("    error=14001 (side-by-side): VC++ runtime mismatch");
                return false;
            }
            if (_dmlErr == 1114)
                LOG_WARN("directml.dll probe returned 1114 (DLL_INIT_FAILED) — proceeding anyway;");

            // Log directml.dll version and path using module handle (more reliable than SearchPath)
            {
                // Try to find it via the module handle first (most accurate)
                HMODULE hDml = GetModuleHandleW(L"directml");
                if (!hDml) hDml = GetModuleHandleW(L"directml.dll");
                if (hDml) {
                    wchar_t dmlPath[MAX_PATH]{};
                    GetModuleFileNameW(hDml, dmlPath, MAX_PATH);
                    LOG_INFO("  directml.dll (in process) path: ", std::wstring(dmlPath));
                } else {
                    // Fall back to SearchPathW (searches all registered DLL dirs)
                    wchar_t dmlPath[MAX_PATH]{};
                    if (SearchPathW(nullptr, L"directml.dll", nullptr, MAX_PATH, dmlPath, nullptr) > 0)
                        LOG_INFO("  directml.dll (SearchPath) path: ", std::wstring(dmlPath));
                    else
                        LOG_WARN("  directml.dll not found via SearchPath");
                }

                // d3d12.dll is required
                HMODULE hD3D12 = LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
                if (!hD3D12) {
                    LOG_WARN("  d3d12.dll not found – DirectML requires DirectX 12.");
                    return false;
                }
                FreeLibrary(hD3D12);
            }

            try {
                m_sessionOpts.AppendExecutionProvider("DML", {{"device_id", "0"}});
                outInfo = L"DirectML (DX12 GPU)";
                LOG_INFO("Execution provider: DirectML");
                LOG_INFO("  NOTE: DirectML compiles GPU shaders on first inference.");
                LOG_INFO("  First frame may take 5-30 s; subsequent frames are fast.");
                LOG_INFO("  Expected throughput for DA V2 Small:");
                LOG_INFO("    High-end GPU (RTX 3070+, RX 6700+): ~100-300 ms/frame");
                LOG_INFO("    Mid-range GPU (GTX 1060-1080, RX 580): ~300-600 ms/frame");
                LOG_INFO("    Low-end / integrated GPU: 600ms-2s/frame");
                return true;
            } catch (const Ort::Exception& e) {
                LOG_WARN("DirectML EP init failed: ", e.what());
                LOG_WARN("  Root cause: the onnxruntime.dll that was loaded does NOT have DML");
                LOG_WARN("  compiled in.  The available providers list logged above shows what IS");
                LOG_WARN("  present.  If 'DmlExecutionProvider' is missing there, a CPU-only or");
                LOG_WARN("  CUDA onnxruntime.dll won the DLL search race over the DML NuGet one.");
                LOG_WARN("  Common scenarios:");
                LOG_WARN("    (A) Win64_GPU\\ or Win64_GPU11\\ is also registered as a filter and");
                LOG_WARN("        its onnxruntime.dll was loaded first via the DLL search path.");
                LOG_WARN("        Fix: un-register the GPU build, or move its folder out of the");
                LOG_WARN("        DLL search path, or rename its onnxruntime.dll temporarily.");
                LOG_WARN("    (B) PotPlayer copied 3Deflatten_x64.ax to its own folder and");
                LOG_WARN("        loaded onnxruntime.dll from there (not Win64\\).");
                LOG_WARN("        Fix: set DEFLATTEN_MODEL_PATH env var and use 'Register' not");
                LOG_WARN("        PotPlayer's 'Add External Filter' copy-mode.");
                LOG_WARN("    (C) Win64\\onnxruntime.dll was replaced or is missing.");
                LOG_WARN("        Fix: rebuild or re-copy the NuGet onnxruntime.dll into Win64\\.");
                return false;
            }
#endif // ORT_ENABLE_DML
        }

        return false; // unknown EP
    };

    // ── Explicit provider selection ───────────────────────────────────────────
    if (provider == GPUProvider::TensorRTRtx) {
        if (tryEP(GPUProvider::TensorRTRtx)) return;
        LOG_INFO("TRT-RTX requested but unavailable – trying TensorRT");
        provider = GPUProvider::TensorRT;
    }
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
    if (tryEP(GPUProvider::TensorRTRtx)) return;
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
    if (!m_loaded) {
        LOG_ERR("Estimate called but model not loaded");
        return E_FAIL;
    }
#ifdef ORT_ENABLE_TRTRTX
    if (m_trtRtx)
        return EstimateTrtRtx(srcData, srcWidth, srcHeight, srcStride,
                              isBGR, flipDepth, smoothAlpha, result);
#endif
    if (!m_session) {
        LOG_ERR("Estimate: no ORT session and no native TRT session");
        return E_FAIL;
    }
    // Recurrent-context streaming (future models with context_in/out tensors)
    if (m_streaming)
        return EstimateStreaming(srcData, srcWidth, srcHeight, srcStride,
                                 isBGR, flipDepth, smoothAlpha, result);

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

        // Normalise to model-output resolution first (before upscaling)
        // so DA3-Streaming accumulation works at model resolution.
        int accumW = rawW, accumH = rawH;
        std::vector<float> depth(accumW * accumH);
        {
            // Min-max normalise at model res (no flip yet — flip after accumulation)
            float mn = rawDepth[0], mx = rawDepth[0];
            for (int i = 1; i < accumW * accumH; ++i) {
                mn = std::min(mn, rawDepth[i]);
                mx = std::max(mx, rawDepth[i]);
            }
            float range = (mx - mn) > 1e-6f ? (mx - mn) : 1e-6f;
            for (int i = 0; i < accumW * accumH; ++i)
                depth[i] = (rawDepth[i] - mn) / range;
        }

        // DA3-Streaming: accumulate into sliding window and affine-align
        if (m_da3StreamMode) {
            m_streamW = accumW; m_streamH = accumH;
            DA3StreamAccumulate(depth);
        }

        // Resize to source resolution
        std::vector<float> out(srcWidth * srcHeight);
        BilinearResize(depth.data(), accumW, accumH,
                       out.data(), srcWidth, srcHeight);

        // Apply flip after accumulation
        if (flipDepth)
            for (float& v : out) v = 1.f - v;

        // Temporal smoothing.  Disabled when DA3-Streaming is active because:
        // (a) the affine alignment already provides temporal consistency, and
        // (b) feeding smoothed output back into the anchor (via the ring buffer)
        //     creates a positive feedback loop that drives depth toward 1.0 (white).
        if (!m_da3StreamMode && smoothAlpha > 0.f && smoothAlpha < 1.f)
            TemporalSmooth(out, smoothAlpha);

        result.data   = std::move(out);
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
        // Fit the longer side to 1022 (Depth Anything V2 max) and scale the
        // shorter side proportionally, then snap BOTH to the nearest multiple
        // of 14 (patch size).  Without this both dimensions were clamped to
        // 1022 independently, turning 16:9 footage into a 1:1 square tensor
        // and distorting all geometry in the depth output.
        const int maxDim = 1022;
        const int base   = 14;
        float scale = std::min((float)maxDim / w, (float)maxDim / h);
        // Round to nearest multiple of 14, minimum 14
        mw = std::max(base, (int)std::round(w * scale / base) * base);
        mh = std::max(base, (int)std::round(h * scale / base) * base);
        // Clamp in case rounding pushed past maxDim
        mw = std::min(mw, maxDim);
        mh = std::min(mh, maxDim);
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
// alpha = 0.0 → pass current frame through unchanged (no smoothing)
// alpha = 0.9 → heavily weighted towards previous frame (strong smoothing)
// The old code had the alpha convention inverted (0.4 meant 60% old, 40% new).
void DepthEstimator::TemporalSmooth(std::vector<float>& cur, float alpha) {
    int n = (int)cur.size();
    if ((int)m_prevDepth.size() != n) {
        m_prevDepth = cur;
        return;
    }
    for (int i = 0; i < n; ++i)
        cur[i] = (1.f - alpha) * cur[i] + alpha * m_prevDepth[i];
    m_prevDepth = cur;
}

// ── DA3-Streaming support ─────────────────────────────────────────────────────
//
// DA3-Streaming (ByteDance Seed, Depth-Anything-3) is a recurrent model that
// maintains a context tensor across frames to improve temporal consistency and
// reduce flickering on video.  The model schema is:
//
//   Inputs:  img         [1, 3, H, W]          (the current frame)
//            context_in  [1, C, H/4, W/4]      (temporal state from prev frame)
//   Outputs: depth       [1, 1, H, W]          (depth map)
//            context_out [1, C, H/4, W/4]      (temporal state for next frame)
//
// On the first call context_in is a zero tensor.  After each successful
// inference context_out is stored in m_ctxTensor and fed back as context_in
// on the next call.  ResetStreamingContext() zeroes the tensor (e.g. on seek).
//
// Reference: https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md

bool DepthEstimator::DetectStreamingModel() {
    if (!m_session) return false;
    size_t nIn  = m_session->GetInputCount();
    size_t nOut = m_session->GetOutputCount();
    if (nIn < 2 || nOut < 2) return false;

    Ort::AllocatorWithDefaultOptions alloc;

    // Collect all input/output names
    std::vector<std::string> inNames, outNames;
    for (size_t i = 0; i < nIn;  ++i)
        inNames.push_back(m_session->GetInputNameAllocated(i, alloc).get());
    for (size_t i = 0; i < nOut; ++i)
        outNames.push_back(m_session->GetOutputNameAllocated(i, alloc).get());

    // Look for context_in / context_out by name substring
    // (the reference implementation uses exactly these names)
    for (auto& n : inNames)  if (n.find("context") != std::string::npos) m_ctxInName  = n;
    for (auto& n : outNames) if (n.find("context") != std::string::npos) m_ctxOutName = n;

    if (m_ctxInName.empty() || m_ctxOutName.empty()) return false;

    // Find the image input (not context)
    for (auto& n : inNames)  if (n != m_ctxInName)  m_inputName  = n;
    // Find the depth output (not context)
    for (auto& n : outNames) if (n != m_ctxOutName) m_outputName = n;

    // Read context shape from the input type info for context_in
    for (size_t i = 0; i < nIn; ++i) {
        std::string name = m_session->GetInputNameAllocated(i, alloc).get();
        if (name != m_ctxInName) continue;
        auto shape = m_session->GetInputTypeInfo(i)
                         .GetTensorTypeAndShapeInfo()
                         .GetShape();
        // Expected shape: [1, C, H/4, W/4]  (dims may be -1 for dynamic)
        if (shape.size() == 4) {
            m_ctxC = shape[1];   // may be -1 if dynamic; clamped below
            m_ctxH = shape[2];
            m_ctxW = shape[3];
        }
        break;
    }
    return true;
}

void DepthEstimator::InitStreamingContext(int modelW, int modelH) {
    // If the model reported concrete context dims, use them.
    // If any dim was dynamic (-1), derive from model input size (H/4, W/4).
    int64_t ctxC = (m_ctxC > 0) ? m_ctxC : 256;   // DA3 default channels
    int64_t ctxH = (m_ctxH > 0) ? m_ctxH : (int64_t)(modelH / 4);
    int64_t ctxW = (m_ctxW > 0) ? m_ctxW : (int64_t)(modelW / 4);

    m_ctxC = ctxC; m_ctxH = ctxH; m_ctxW = ctxW;
    size_t sz = (size_t)(ctxC * ctxH * ctxW);
    m_ctxTensor.assign(sz, 0.0f);   // zero = "no prior context"
    m_ctxReady = true;
    LOG_INFO("  [streaming] context shape: [1,", ctxC, ",", ctxH, ",", ctxW,
             "]  elems=", sz);
}


HRESULT DepthEstimator::EstimateStreaming(const BYTE* srcData,
                                           int srcW, int srcH, int srcStride,
                                           bool isBGR, bool flipDepth,
                                           float smoothAlpha,
                                           DepthResult& result) {
    try {
        std::vector<float> imgTensor;
        int mw, mh;
        PreprocessFrame(srcData, srcW, srcH, srcStride, isBGR, imgTensor, mw, mh);

        if (!m_ctxReady)
            InitStreamingContext(mw, mh);

        if (m_estimateCount == 0)
            LOG_INFO("First Streaming Estimate:"
                     " src=", srcW, "x", srcH,
                     " model=", mw, "x", mh,
                     " ctx=[1,", m_ctxC, ",", m_ctxH, ",", m_ctxW, "]");
        ++m_estimateCount;

        auto memInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        // Image tensor
        std::array<int64_t, 4> imgShape{1, 3, (int64_t)mh, (int64_t)mw};
        auto imgVal = Ort::Value::CreateTensor<float>(
            memInfo, imgTensor.data(), imgTensor.size(),
            imgShape.data(), imgShape.size());

        // Context tensor (zero on first frame, carried forward thereafter)
        std::array<int64_t, 4> ctxShape{1, m_ctxC, m_ctxH, m_ctxW};
        auto ctxVal = Ort::Value::CreateTensor<float>(
            memInfo, m_ctxTensor.data(), m_ctxTensor.size(),
            ctxShape.data(), ctxShape.size());

        // Run with both inputs, expect both outputs
        std::vector<const char*> inNames  = {m_inputName.c_str(),  m_ctxInName.c_str()};
        std::vector<const char*> outNames = {m_outputName.c_str(), m_ctxOutName.c_str()};
        std::vector<Ort::Value>  inVals;
        inVals.push_back(std::move(imgVal));
        inVals.push_back(std::move(ctxVal));

        auto outputs = m_session->Run(
            Ort::RunOptions{nullptr},
            inNames.data(),  inVals.data(),  inVals.size(),
            outNames.data(), outNames.size());

        // Depth output (index 0)
        const float* rawDepth = outputs[0].GetTensorData<float>();
        auto rawShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int rawH = (int)rawShape[rawShape.size() - 2];
        int rawW = (int)rawShape[rawShape.size() - 1];
        if (m_estimateCount == 1)
            LOG_INFO("First streaming output: depth=", rawW, "x", rawH,
                     " -> resample to ", srcW, "x", srcH);

        // Context output (index 1) — copy back to m_ctxTensor for next frame
        const float* ctxOut = outputs[1].GetTensorData<float>();
        auto ctxOutShape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        size_t ctxElems = 1;
        for (auto d : ctxOutShape) ctxElems *= (size_t)(d > 0 ? d : 1);

        // If context shape changed (shouldn't happen, but guard anyway)
        if (ctxElems != m_ctxTensor.size()) {
            LOG_WARN("Streaming: context output size mismatch (expected=",
                     m_ctxTensor.size(), " got=", ctxElems, ") – reinit");
            m_ctxTensor.assign(ctxElems, 0.0f);
        }
        std::copy(ctxOut, ctxOut + ctxElems, m_ctxTensor.begin());

        // Postprocess depth → [0,1], resample to source resolution
        std::vector<float> depth(srcW * srcH);
        PostprocessDepth(rawDepth, rawW, rawH, srcW, srcH, flipDepth, depth);

        // Temporal smoothing is redundant when streaming context is active
        // (the model itself provides temporal consistency), but we still
        // honour it at low alpha values for any residual flicker.
        if (smoothAlpha > 0.f && smoothAlpha < 0.5f)
            TemporalSmooth(depth, smoothAlpha);

        result.data   = std::move(depth);
        result.width  = srcW;
        result.height = srcH;
        return S_OK;

    } catch (const Ort::Exception& e) {
        LOG_ERR("ORT EstimateStreaming exception: ", e.what());
        return E_FAIL;
    }
}

// ── DA3-Streaming: sliding-window temporal alignment ─────────────────────────
//
// AffineAlignTo: least-squares affine fit of `depth` values onto `anchor`.
//
// For each frame we want to find (scale, shift) such that:
//   aligned[i] = scale * depth[i] + shift   ≈ anchor[i]
//
// Solved analytically (closed form):
//   Sigma_xy = cov(anchor, depth)
//   Sigma_xx = var(depth)
//   scale    = Sigma_xy / Sigma_xx     (or 1.0 if degenerate)
//   shift    = mean(anchor) - scale * mean(depth)
//
// This is identical to the core alignment step in DA3-Streaming (da3_streaming/
// consistency.py) which uses np.polyfit(frame, anchor, 1).
//
void DepthEstimator::AffineAlignTo(const std::vector<float>& anchor,
                                    std::vector<float>&        depth) {
    int n = (int)depth.size();
    if (n == 0 || (int)anchor.size() != n) return;

    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (int i = 0; i < n; ++i) {
        double x = depth[i], y = anchor[i];
        sum_x  += x;
        sum_y  += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    double mean_x = sum_x / n;
    double mean_y = sum_y / n;
    double cov_xy = sum_xy / n - mean_x * mean_y;
    double var_x  = sum_xx / n - mean_x * mean_x;

    // Degenerate case: flat or near-flat input (uniform scene, black frame)
    if (var_x < 1e-8) return;

    double scale = cov_xy / var_x;
    double shift = mean_y - scale * mean_x;

    // Clamp scale to [0.25, 4.0] to prevent catastrophic divergence on cuts.
    // Also clamp shift: after scaling, the combined (scale*x + shift) for a
    // value in [0,1] must stay roughly in [0,1], so |shift| <= 2.0 is safe.
    scale = std::max(0.25, std::min(4.0, scale));
    shift = std::max(-2.0, std::min(2.0, shift));

    for (int i = 0; i < n; ++i) {
        float v = (float)(scale * depth[i] + shift);
        depth[i] = std::max(0.f, std::min(1.f, v));
    }
}

void DepthEstimator::DA3StreamAccumulate(std::vector<float>& depth) {
    int n = (int)depth.size();
    if (n == 0) return;
    ++m_streamFrameCount;

    // First frame: initialise anchor, no alignment needed
    if (m_streamAnchor.empty()) {
        m_streamAnchor = depth;
        m_streamBuf.push_back(depth);
        return;
    }

    // Affine-align to anchor (removes per-frame scale/shift jitter)
    AffineAlignTo(m_streamAnchor, depth);

    // Push aligned frame into ring buffer, trim to window
    m_streamBuf.push_back(depth);
    while ((int)m_streamBuf.size() > STREAM_WINDOW)
        m_streamBuf.pop_front();

    // Periodically recompute anchor as pixel-wise mean of the window.
    // NOTE: we do NOT blend depth toward the anchor here — that caused convergence
    // to a flat 0.5 map over time.  Temporal smoothing is handled separately by
    // TemporalSmooth() in Estimate() after this function returns.
    if (m_streamFrameCount % ANCHOR_RESET_FRAMES == 0) {
        std::vector<float> newAnchor(n, 0.f);
        for (auto& buf : m_streamBuf)
            for (int i = 0; i < n && i < (int)buf.size(); ++i)
                newAnchor[i] += buf[i];
        float inv = 1.f / (float)m_streamBuf.size();
        for (float& v : newAnchor) v *= inv;

        // Renormalise anchor to [0,1] so accumulated clipping bias from
        // AffineAlignTo's clamping can't cause the anchor to drift toward
        // 1.0 over successive anchor resets (the white-out feedback loop).
        float amn = newAnchor[0], amx = newAnchor[0];
        for (float v : newAnchor) { amn = std::min(amn, v); amx = std::max(amx, v); }
        float arange = (amx - amn) > 1e-6f ? (amx - amn) : 1e-6f;
        for (float& v : newAnchor) v = (v - amn) / arange;

        m_streamAnchor = std::move(newAnchor);
    }
    // depth is returned as-is (affine-aligned); TemporalSmooth handles flicker
}

void DepthEstimator::ResetStreamingContext() {
    // Reset recurrent-context state (future models)
    if (m_streaming && m_ctxReady) {
        std::fill(m_ctxTensor.begin(), m_ctxTensor.end(), 0.0f);
        LOG_INFO("Recurrent streaming context reset (seek/stop)");
    }
    // Reset DA3-Streaming sliding-window state
    if (m_da3StreamMode) {
        m_streamBuf.clear();
        m_streamAnchor.clear();
        m_streamFrameCount = 0;
        m_prevDepth.clear();
        LOG_INFO("DA3-Streaming window reset (seek/stop)");
    }
}
