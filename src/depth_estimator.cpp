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
#if ORT_CUDA_MAJOR == 13
    LOG_INFO("  ORT 1.24.3 gpu_cuda13 build requirements:");
    LOG_INFO("    CUDA 13.0  (cudart64_13.dll)");
    LOG_INFO("    cuDNN 9.x  (cudnn64_9.dll + 7 split DLLs)");
    if (includeTrt)
        LOG_INFO("    TensorRT 10.13.3.9 (CUDA 13.0 build)");
    LOG_INFO("  NOTE: ORT 1.24.3 gpu_cuda13 was compiled against CUDA 13.0.");
    LOG_INFO("        Using CUDA 13.1+ DLLs at runtime causes error=1114 (version mismatch).");
    LOG_INFO("  NOTE: Driver 572+ required for CUDA 13.0.");
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
    // nvJitLink: required by ORT 1.22+ CUDA EP (JIT kernel compilation)
    bool hasJitLink = ProbeDep(L"nvJitLink_130_0.dll", L"CUDA 13 JIT-Link (required by ORT CUDA EP)");
    if (!hasJitLink) LOG_WARN("  nvJitLink_130_0.dll missing -- CUDA EP will fail even if other DLLs are present.");
    // cuSolver + cuRand: loaded by ORT CUDA EP at init time (not lazy)
    ProbeDep(L"cusolver64_11.dll",  L"cuSolver 11 -- loaded by ORT CUDA EP at startup");
    ProbeDep(L"curand64_10.dll",    L"cuRand 10 -- loaded by ORT CUDA EP at startup");
    // cuDNN 9 (split library layout)
    bool hasCudnn = ProbeDep(L"cudnn64_9.dll", L"cuDNN 9.x main library");
    if (!hasCudnn) LOG_WARN("  -> cudnn64_9.dll not found. Run collect_runtime_dlls_cuda13.py.");
#else
    // ── CUDA 11.x ─────────────────────────────────────────────────────────
    bool hasCuda = ProbeDep(L"cudart64_110.dll", L"CUDA 11.x runtime");
    if (!hasCuda) LOG_WARN("  -> cudart64_110.dll not found. Run collect_runtime_dlls.py.");
    ProbeDep(L"cublas64_11.dll",   L"cuBLAS 11 -- in CUDA Toolkit bin");
    ProbeDep(L"cublasLt64_11.dll", L"cuBLAS-Lt 11 -- in CUDA Toolkit bin");
    ProbeDep(L"cufft64_10.dll",    L"cuFFT 10 -- in CUDA Toolkit bin");
    // cuSolver + cuRand: loaded by ORT CUDA EP at init time
    ProbeDep(L"cusolver64_11.dll", L"cuSolver 11 -- loaded by ORT CUDA EP at startup");
    ProbeDep(L"curand64_10.dll",   L"cuRand 10 -- loaded by ORT CUDA EP at startup");
    // cuDNN 8 (monolithic + split infer/train DLLs)
    bool hasCudnn = ProbeDep(L"cudnn64_8.dll", L"cuDNN 8.x main library");
    if (!hasCudnn) LOG_WARN("  -> cudnn64_8.dll not found. Run collect_runtime_dlls.py.");
#endif

    if (includeTrt) {
        LOG_INFO("");
#if ORT_CUDA_MAJOR == 13
        LOG_INFO("  TensorRT 10.13.x libraries (CUDA 13.0 build):");
        LOG_INFO("  NOTE: zlibwapi.dll is NOT required by TRT 10.13+.");
        ProbeDep(L"nvinfer_10.dll",          L"TRT 10.13+ inference engine");
        ProbeDep(L"nvonnxparser_10.dll",     L"TRT 10.13+ ONNX parser");
        ProbeDep(L"nvinfer_dispatch_10.dll", L"TRT 10.13+ dispatch runtime");
        ProbeDep(L"nvinfer_lean_10.dll",     L"TRT 10.13+ lean runtime");
        ProbeDep(L"nvinfer_plugin_10.dll",   L"TRT 10.13+ plugins");
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
#if ORT_CUDA_MAJOR == 13
                LOG_WARN("       TRT 10.13.x (cuda-13.0 build) is required.");
                LOG_WARN("    3. nvinfer_10.dll / nvonnxparser_10.dll missing or their deps missing.");
#else
                LOG_WARN("       TRT 10.0.x (cuda-11.8 build) is required.");
                LOG_WARN("    3. nvinfer.dll / nvonnxparser.dll missing or their deps missing.");
                LOG_WARN("       NOTE: TRT 10.0.x uses plain names (no _10 suffix).");
#endif
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
