// SPDX-License-Identifier: GPL-3.0-or-later
#include "depth_estimator.h"
#include "logger.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <shlobj.h>
#include <winreg.h>

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
// We use it only when path contains a directory separator.
static DWORD DllLoadable(const wchar_t* path) {
    bool isAbsPath = (wcschr(path, L'\\') || wcschr(path, L'/'));
    DWORD flags = isAbsPath
        ? (LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
           LOAD_LIBRARY_SEARCH_USER_DIRS    |
           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
        : 0;  // bare name: use default Windows search order
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
    LOG_INFO("  ORT 1.24.x GPU build requirements:");
    LOG_INFO("    CUDA 13.x runtime  (install: https://developer.nvidia.com/cuda-downloads)");
    LOG_INFO("    cuDNN 9.x          (install: https://developer.nvidia.com/cudnn)");
    if (includeTrt)
        LOG_INFO("    TensorRT 10.x      (install: https://developer.nvidia.com/tensorrt)");
    LOG_INFO("  NOTE: ORT 1.24.x GPU build requires CUDA 13.x (cudart64_13.dll).");
    LOG_INFO("        CUDA 12.x ships cudart64_12.dll and is NOT compatible with this build.");
    LOG_INFO("  NOTE: Driver 545+ required for CUDA 13.");
    LOG_INFO("");

    // NVIDIA driver (kernel proxy -- loaded by nvcuda.dll consumers)
    ProbeDep(L"nvcuda.dll",       L"NVIDIA driver kernel proxy -- must be in System32");

    // CUDA 13 runtime -- required by ORT 1.24.x GPU build
    bool hasCuda13 = ProbeDep(L"cudart64_13.dll", L"CUDA 13.x runtime");
    if (!hasCuda13) {
        if (DllLoadable(L"cudart64_12.dll") == 0)
            LOG_WARN("  -> CUDA 12.x detected. ORT 1.24.x requires CUDA 13.x. "
                     "Install CUDA 13: https://developer.nvidia.com/cuda-downloads");
        else
            LOG_WARN("  -> No CUDA 13 runtime found at all. "
                     "Install CUDA 13: https://developer.nvidia.com/cuda-downloads");
    }

    // cuBLAS 13
    ProbeDep(L"cublas64_13.dll",   L"cuBLAS 13 -- in CUDA Toolkit bin");
    ProbeDep(L"cublasLt64_13.dll", L"cuBLAS-Lt 13 -- in CUDA Toolkit bin");

    // cuDNN 9 -- often the missing piece after CUDA itself
    bool hasCudnn9 = ProbeDep(L"cudnn64_9.dll", L"cuDNN 9.x main library");
    if (!hasCudnn9) {
        // cuDNN 9 also ships as split libraries on some installs
        ProbeDep(L"cudnn_ops64_9.dll", L"cuDNN 9.x ops -- alternative layout");
    }

    // cuFFT (12 = part of CUDA 13 toolkit)
    ProbeDep(L"cufft64_12.dll",    L"cuFFT 12 -- in CUDA Toolkit bin");

    // nvJitLink: CUDA JIT-linking library required by ORT 1.22+ CUDA EP.
    // In CUDA 13.x this is nvJitLink_130_0.dll in the CUDA bin directory.
    // NOTE: if all other CUDA DLLs are [OK] but onnxruntime_providers_cuda.dll
    // still fails with error 126, this is almost always the missing piece.
    bool hasJitLink = ProbeDep(L"nvJitLink_130_0.dll",
                               L"CUDA 13 JIT-Link (required by ORT CUDA EP)");
    if (!hasJitLink) {
        // Older naming convention (CUDA 12 used this format)
        hasJitLink = ProbeDep(L"nvJitLink_120_0.dll",
                              L"CUDA 12 JIT-Link (fallback)");
        if (!hasJitLink)
            LOG_WARN("  nvJitLink not found -- onnxruntime_providers_cuda.dll WILL fail");
            LOG_WARN("  with error 126 even if all other CUDA DLLs are present.");
            LOG_WARN("  nvJitLink_130_0.dll should be in the CUDA 13 bin\\ directory.");
            LOG_WARN("  If using CUDA 13 from bin\\x64\\, also add the parent bin\\ dir.");
    }

    if (includeTrt) {
        LOG_INFO("");
        LOG_INFO("  TensorRT 10.x libraries (install from https://developer.nvidia.com/tensorrt):");
        ProbeDep(L"nvinfer_10.dll",              L"TensorRT 10 inference engine");
        ProbeDep(L"nvonnxparser_10.dll",         L"TensorRT 10 ONNX parser");
        ProbeDep(L"nvinfer_builder_resource_10.dll",
                                                 L"TRT 10 builder resource (sub-dep of nvinfer)");
        // zlibwapi.dll: required by TensorRT, NOT bundled with CUDA.
        // TRT 10.x Windows zips may or may not include it in lib\.
        bool hasZlib = ProbeDep(L"zlibwapi.dll", L"zlib (required by TRT 10)");
        if (!hasZlib) {
            LOG_WARN("  zlibwapi.dll NOT found -- TensorRT WILL fail with error 126.");
            LOG_WARN("  Copy zlibwapi.dll from the TRT zip lib\\ folder, or download:");
            LOG_WARN("  https://www.dll-files.com/zlibwapi.dll.html");
            LOG_WARN("  Place it next to 3Deflatten_x64.ax or in TRT_LIB_PATH.");
        }
    }
    LOG_INFO("--- end dependency scan ---");
}
#endif // ORT_ENABLE_CUDA || ORT_ENABLE_TENSORRT

#if defined(ORT_ENABLE_CUDA) || defined(ORT_ENABLE_TENSORRT)
// Probe for nvcuda.dll + CUDA 12 runtime.  Returns false and logs clearly if
// either is absent or the wrong version.
static bool CudaDriverPresent() {
    DWORD e = DllLoadable(L"nvcuda.dll");
    if (e != 0) {
        LOG_WARN("nvcuda.dll not loadable (error ", e, ") – no NVIDIA driver detected.");
        if (e == 87)
            LOG_WARN("  (Error 87 = invalid parameter -- this is a 3Deflatten internal bug; "
                     "please report it.)");
        return false;
    }
    if (DllLoadable(L"cudart64_13.dll") != 0) {
        if (DllLoadable(L"cudart64_12.dll") == 0)
            LOG_WARN("CUDA 12.x detected. ORT 1.24.x requires CUDA 13.x. "
                     "Install CUDA 13: https://developer.nvidia.com/cuda-downloads");
        else
            LOG_WARN("No CUDA 13 runtime found. "
                     "Install CUDA 13: https://developer.nvidia.com/cuda-downloads");
        return false;
    }
    return true;
}

// Probe a provider DLL with full-load check, log result with detail on error 126.
static bool ProviderDllLoadable(const char* name, const std::wstring& path) {
    DWORD e = DllLoadable(path.c_str());
    if (e == 0) return true;
    std::string narrow(path.begin(), path.end());
    if (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) {
        LOG_WARN(name, " not found at '", narrow, "'");
        // Suggest the registered install dir if it differs from current DLL dir
        std::wstring regDir = GetRegisteredDllDir();
        std::wstring dllDir = GetDllDir();
        if (!regDir.empty() && _wcsicmp(regDir.c_str(), dllDir.c_str()) != 0) {
            LOG_WARN("  DLL dir = ", dllDir);
            LOG_WARN("  Registry install dir = ", regDir);
            LOG_WARN("  Hint: copy onnxruntime*.dll and directml.dll to the registry dir,");
            LOG_WARN("  OR move the .ax file back to its original install directory.");
        }
    } else if (e == 126) {
        LOG_WARN(name, " exists but failed to load (error 126 = missing dependencies).");
        bool isTrt = (std::wstring(path).find(L"tensorrt") != std::wstring::npos);
        LOG_WARN("  Running dependency scan to identify what is missing:");
        LogCudaDependencies(isTrt);
        if (isTrt) {
            LOG_WARN("  TIP: Copy ALL DLLs from TensorRT lib\\ folder next to the .ax,");
            LOG_WARN("       not just nvinfer_10.dll -- TRT has sub-dependencies too.");
            LOG_WARN("       Or ensure TRT_LIB_PATH was set before launching the host app.");
        }
    } else if (e == 1114) {
        // ERROR_DLL_INIT_FAILED: DLL loaded but its DllMain threw an exception
        // or returned FALSE.  For CUDA/TRT providers this has two common causes:
        //
        //   A) CUDA version mismatch -- ORT 1.24.x GPU requires CUDA 13.x.
        //      Even if cudart64_13.dll is present, CUDA 13.x needs driver 572+.
        //      Installing CUDA 12.x later DOWNGRADES the driver to 561.x which
        //      breaks CUDA 13 even though the toolkit files remain on disk.
        //
        //   B) TensorRT version mismatch -- TRT 10.7.x is built against CUDA 12.6
        //      and will NOT work with ORT 1.24.x (CUDA 13). Use TRT 10.15+ for CUDA 13.
        //
        bool isTrt = (std::wstring(path).find(L"tensorrt") != std::wstring::npos);
        LOG_WARN(name, " load failed error=1114 (DLL init failed).");
        if (isTrt) {
            LOG_WARN("  For TensorRT: TRT 10.7.x is built against CUDA 12.6 and is");
            LOG_WARN("    INCOMPATIBLE with ORT 1.24.x (which requires CUDA 13).");
            LOG_WARN("  Use TRT 10.15.x or newer (the CUDA 13 build):");
            LOG_WARN("    https://developer.nvidia.com/tensorrt");
            LOG_WARN("  Check your TRT version: look at the nvinfer_10.dll filename or");
            LOG_WARN("    the folder name (TensorRT-10.7.x = CUDA 12, TensorRT-10.15.x = CUDA 13).");
        } else {
            LOG_WARN("  Most likely cause: NVIDIA driver is too old for CUDA 13.");
            LOG_WARN("  CUDA 13.1 requires driver 572.xx or newer.");
            LOG_WARN("  Installing CUDA 12.x DOWNGRADES the driver (e.g. to 561.17),");
            LOG_WARN("  which breaks CUDA 13 even if cudart64_13.dll is still on disk.");
            LOG_WARN("  FIX: Update NVIDIA driver to 572+ (do NOT install CUDA 12.x):");
            LOG_WARN("    https://www.nvidia.com/drivers  (select Game Ready or Studio driver)");
            LOG_WARN("  Or reinstall CUDA 13.1 (bundles driver 572.xx):");
            LOG_WARN("    https://developer.nvidia.com/cuda-downloads");
            LOG_WARN("  Verify current driver with: nvidia-smi  (look for 'Driver Version')");
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

    // Returns true if the EP was successfully appended.
    // On failure (DLL absent, CUDA not installed, ORT exception) returns false.
    auto tryEP = [&](GPUProvider ep) -> bool {

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
                LOG_WARN("  Common causes:");
                LOG_WARN("    1. TRT version does not match CUDA version.");
                LOG_WARN("       Check zip filename for the cuda-XX.N suffix.");
                LOG_WARN("       TRT 10.x must match the CUDA 13.x minor you have installed.");
                LOG_WARN("    2. Replaced onnxruntime*.dll with a version != " ORT_VER_STR ".");
                LOG_WARN("       The .ax is ABI-linked to ORT " ORT_VER_STR ". Other versions crash.");
                LOG_WARN("    3. nvinfer_10.dll / nvonnxparser_10.dll missing or their deps missing.");
                LOG_WARN("       Ensure TRT_LIB_PATH was set before launching the host app,");
                LOG_WARN("       or copy all DLLs from TensorRT lib\\ next to the .ax file.");
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
                LOG_INFO("  NOTE: DirectML compiles GPU shaders on first inference.");
                LOG_INFO("  First frame may take 5-30 s; subsequent frames are fast.");
                LOG_INFO("  Expected throughput for DA V2 Small (depth_anything_v2_small.onnx):");
                LOG_INFO("    High-end GPU (RTX 3070+, RX 6700+): ~100-300 ms/frame");
                LOG_INFO("    Mid-range GPU (GTX 1060-1080, RX 580, RTX 3060): ~300-600 ms/frame");
                LOG_INFO("    Low-end / integrated GPU: 600ms-2s/frame");
                LOG_INFO("  If performance is lower than expected, check GPU utilization in");
                LOG_INFO("  Task Manager -> Performance -> GPU. Low utilization (<50%) with");
                LOG_INFO("  slow inference is normal for smaller models on high-end GPUs.");
                LOG_INFO("  The GPU name is logged at startup (search log for 'GPU:').");
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
