// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten -- DLL factory table, dependency-path setup, and logger init
//
// DllMain, DllGetClassObject, and DllCanUnloadNow are provided by
// dllentry.cpp from the DirectShow baseclasses.  We must NOT define DllMain.
//
// DEPENDENCY LOADING ORDER (critical for COM / regsvr32 use)
// ──────────────────────────────────────────────────────────
// onnxruntime.dll (and directml.dll, provider DLLs) are /DELAYLOAD-ed.
// Windows therefore does NOT resolve them when the .ax is first loaded by
// COM.  The static DllInit object below runs BEFORE any ORT symbol is
// touched and calls SetDefaultDllDirectories + AddDllDirectory to teach
// Windows where our DLLs and the CUDA/TRT runtime DLLs live.
//
// Without this, COM fails to load the .ax as a registered filter
// (CO_E_ERRORINDLL / error 126) because neither onnxruntime.dll nor
// cudart64_12.dll are on the system PATH.

#include <windows.h>
#include <winreg.h>
#include <commctrl.h>
#pragma comment(lib, "comctl32.lib")
#include <streams.h>
#include "filter.h"
#include "prop_page.h"
#include "guids.h"
#include "logger.h"
#include <string>

// ── Helpers ───────────────────────────────────────────────────────────────────

// __declspec(noinline) is mandatory: if the compiler inlines or COMDAT-folds
// this function the FROM_ADDRESS probe would return the wrong module.
__declspec(noinline) static std::wstring GetThisDllDir() {
    wchar_t path[MAX_PATH] = {};
    HMODULE hm = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&GetThisDllDir), &hm) || !hm)
        return {};
    GetModuleFileNameW(hm, path, MAX_PATH);
    wchar_t* sl = wcsrchr(path, L'\\');
    if (sl) *sl = L'\0';
    return path;
}

// Read the InprocServer32 registry path for our filter CLSID.
// Some host apps copy the .ax to their own directory; the registry retains
// the original path where onnxruntime.dll and the model actually live.
static std::wstring GetRegisteredInstallDir() {
    const wchar_t* key =
        L"CLSID\\{4D455F32-1A2B-4C3D-8E4F-5A6B7C8D9E0F}\\InprocServer32";
    wchar_t regPath[MAX_PATH] = {};
    DWORD cb = sizeof(regPath);
    if (RegGetValueW(HKEY_CLASSES_ROOT, key, nullptr,
                     RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY,
                     nullptr, regPath, &cb) != ERROR_SUCCESS) {
        cb = sizeof(regPath);
        RegGetValueW(HKEY_CLASSES_ROOT, key, nullptr,
                     RRF_RT_REG_SZ | RRF_SUBKEY_WOW6432KEY,
                     nullptr, regPath, &cb);
    }
    if (!regPath[0]) return {};
    wchar_t* sl = wcsrchr(regPath, L'\\');
    if (sl) *sl = L'\0';
    return regPath;
}

// Read a REG_SZ value; empty string on failure.
static std::wstring RegReadSz(HKEY root, const wchar_t* subkey,
                               const wchar_t* value) {
    wchar_t buf[512] = {};
    DWORD cb = sizeof(buf);
    if (RegGetValueW(root, subkey, value, RRF_RT_REG_SZ,
                     nullptr, buf, &cb) == ERROR_SUCCESS)
        return buf;
    return {};
}

// Add dir to the DLL search path if it exists and contains the probe DLL.
// Logs what it finds (or skips) so the user can see why CUDA was/wasn't found.
static void TryAddDir(const std::wstring& dir, const wchar_t* probeDll,
                      const char* label) {
    if (dir.empty()) return;
    std::wstring probe = dir + L"\\" + probeDll;
    if (GetFileAttributesW(probe.c_str()) == INVALID_FILE_ATTRIBUTES) {
        LOG_INFO("  skip ", std::string(label), ": ",
                 std::string(dir.begin(), dir.end()),
                 " (", std::wstring(probeDll), " absent)");
        return;
    }
    AddDllDirectory(dir.c_str());
    LOG_INFO("  added ", std::string(label), ": ",
             std::string(dir.begin(), dir.end()));
}

// Recursively walk baseDir (up to maxDepth levels) looking for any
// subdirectory that directly contains probeDll, then calls TryAddDir on it.
// Stops after the first match to avoid registering conflicting versions.
// Returns true if a directory was added.
static bool RecursiveFindAndAdd(const std::wstring& baseDir,
                                const wchar_t* probeDll,
                                const char* label,
                                int maxDepth = 6) {
    if (maxDepth <= 0 || baseDir.empty()) return false;

    // Check if the DLL lives directly in baseDir
    std::wstring probe = baseDir + L"\\" + probeDll;
    if (GetFileAttributesW(probe.c_str()) != INVALID_FILE_ATTRIBUTES) {
        TryAddDir(baseDir, probeDll, label);
        return true;
    }

    // Enumerate sub-directories and recurse
    WIN32_FIND_DATAW fd{};
    HANDLE h = FindFirstFileW((baseDir + L"\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return false;
    bool found = false;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
        if (wcscmp(fd.cFileName, L".") == 0 ||
            wcscmp(fd.cFileName, L"..") == 0) continue;
        found = RecursiveFindAndAdd(baseDir + L"\\" + fd.cFileName,
                                    probeDll, label, maxDepth - 1);
    } while (!found && FindNextFileW(h, &fd));
    FindClose(h);
    return found;
}

// Discover CUDA 12.x, cuDNN 9.x, and TensorRT 10.x install directories and
// register them as DLL search paths so ORT can find provider DLLs without
// requiring the user to modify PATH.
//
// Discovery order for each component:
//   1. Environment variable set by the component's own installer
//   2. Registry key written by the installer
//   3. Hard-coded default path as last resort
//
// Call AFTER SetDefaultDllDirectories and AFTER logger is live.
static void RegisterGpuRuntimeDirs() {
    LOG_INFO("--- GPU runtime path discovery ---");
    LOG_INFO("  (searching for CUDA 12.x, cuDNN 9.x, TensorRT 10.x)");

    // ── CUDA Toolkit bin ─────────────────────────────────────────────────────
    // CUDA 12.6 installer sets CUDA_PATH_V12_6 and CUDA_PATH.
    // Multiple CUDA versions can coexist; we pick the highest 12.x present.
    bool foundCuda = false;
    const wchar_t* cudaEnvVars[] = {
        L"CUDA_PATH_V12_6", L"CUDA_PATH_V12_5", L"CUDA_PATH_V12_4",
        L"CUDA_PATH_V12_3", L"CUDA_PATH_V12_2", L"CUDA_PATH_V12_1",
        L"CUDA_PATH_V12_0", L"CUDA_PATH",
        nullptr
    };
    for (int i = 0; cudaEnvVars[i] && !foundCuda; ++i) {
        wchar_t val[MAX_PATH] = {};
        if (GetEnvironmentVariableW(cudaEnvVars[i], val, MAX_PATH) && val[0]) {
            std::wstring binDir = std::wstring(val) + L"\\bin";
            // Validate it's actually a CUDA 12 install
            std::wstring probe12 = binDir + L"\\cudart64_12.dll";
            if (GetFileAttributesW(probe12.c_str()) != INVALID_FILE_ATTRIBUTES) {
                TryAddDir(binDir, L"cudart64_12.dll", "CUDA 12 bin (env var)");
                foundCuda = true;
            }
        }
    }
    if (!foundCuda) {
        // Registry fallback: enumerate all subkeys under the CUDA key so we
        // don't need to hard-code version strings (works for 12.7, 12.8, etc.)
        const wchar_t* regBase =
            L"SOFTWARE\\NVIDIA Corporation\\GPU Computing Toolkit\\CUDA";
        HKEY hBase = nullptr;
        if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, regBase, 0,
                          KEY_READ | KEY_ENUMERATE_SUB_KEYS, &hBase) == ERROR_SUCCESS) {
            wchar_t subName[64] = {};
            for (DWORD idx = 0; !foundCuda; ++idx) {
                DWORD nameLen = ARRAYSIZE(subName);
                if (RegEnumKeyExW(hBase, idx, subName, &nameLen,
                                  nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS)
                    break;
                // Only consider v12.x subkeys
                if (wcsncmp(subName, L"v12.", 4) != 0) continue;
                std::wstring inst = RegReadSz(hBase, subName, L"InstallDir");
                if (inst.empty()) continue;
                std::wstring binDir = inst + L"\\bin";
                std::wstring probe12 = binDir + L"\\cudart64_12.dll";
                if (GetFileAttributesW(probe12.c_str()) != INVALID_FILE_ATTRIBUTES) {
                    TryAddDir(binDir, L"cudart64_12.dll", "CUDA 12 bin (registry)");
                    foundCuda = true;
                }
            }
            RegCloseKey(hBase);
        }
    }
    if (!foundCuda) {
        // Recursive scan of the CUDA Toolkit base directory.
        // Works for any 12.x version regardless of the exact version subfolder.
        foundCuda = RecursiveFindAndAdd(
            L"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
            L"cudart64_12.dll", "CUDA 12 bin (default scan)");
    }
    if (!foundCuda) {
        LOG_WARN("  CUDA 12.x not found. ORT GPU EP requires CUDA 12.x (not 13.x).");
        LOG_WARN("  Install: https://developer.nvidia.com/cuda-12-6-0-download-archive");
        LOG_WARN("  CUDA 13.x (cudart64_13.dll) is NOT compatible with ORT 1.21.");
    }

    // ── cuDNN 9.x ────────────────────────────────────────────────────────────
    // The cuDNN standalone installer nests DLLs in a version-specific path like
    //   bin\<cuda-ver>\<arch>\    e.g. bin\12.9\x64\  (v9.19, Mar 2026)
    // Rather than hard-coding these version strings we recursively scan the
    // install root for cudnn64_9.dll at any depth.
    {
        bool foundCuDnn = false;
        wchar_t val[MAX_PATH] = {};

        // 1. CUDNN_PATH env var (set by standalone installer)
        if (!foundCuDnn &&
            GetEnvironmentVariableW(L"CUDNN_PATH", val, MAX_PATH) && val[0])
            foundCuDnn = RecursiveFindAndAdd(val, L"cudnn64_9.dll",
                                             "cuDNN 9 (CUDNN_PATH)");

        // 2. Registry key written by standalone installer
        if (!foundCuDnn) {
            std::wstring inst = RegReadSz(HKEY_LOCAL_MACHINE,
                L"SOFTWARE\\NVIDIA Corporation\\cuDNN", L"InstallPath");
            if (!inst.empty())
                foundCuDnn = RecursiveFindAndAdd(inst, L"cudnn64_9.dll",
                                                  "cuDNN 9 (registry)");
        }

        // 3. Recursive scan of default base dir -- handles any version layout
        if (!foundCuDnn)
            foundCuDnn = RecursiveFindAndAdd(
                L"C:\\Program Files\\NVIDIA\\CUDNN",
                L"cudnn64_9.dll", "cuDNN 9 (default scan)");

        // 4. If still not found: cuDNN may be co-installed into the CUDA Toolkit
        //    directory, in which case cudnn64_9.dll already lives in the CUDA bin
        //    folder registered above -- no extra step needed.
        if (!foundCuDnn) {
            LOG_INFO("  cuDNN 9 not found via env/registry/default path.");
            LOG_INFO("  If installed elsewhere: set CUDNN_PATH=<install root>.");
            LOG_INFO("  Download: https://developer.nvidia.com/cudnn");
        }
    }

    // ── TensorRT 10.x ────────────────────────────────────────────────────────
    // TRT is distributed as a zip.  DLLs live in <TRT_root>/lib/  (NOT bin/).
    // Set TRT_LIB_PATH=<TRT_root>\lib  OR  TENSORRT_DIR=<TRT_root>
    // IMPORTANT: TRT is compiled against a specific CUDA minor version.
    //   TRT 10.7.x  -> CUDA 12.6    TRT 10.9.x  -> CUDA 12.8
    //   TRT 10.15.x -> CUDA 12.9
    // Your TRT and CUDA Toolkit versions must match.  Mismatches cause
    // nvinfer_10.dll to fail to load even if all DLL files are present.
    {
        wchar_t val[MAX_PATH] = {};
        bool foundTrt = false;
        if (GetEnvironmentVariableW(L"TRT_LIB_PATH", val, MAX_PATH) && val[0]) {
            // TRT_LIB_PATH should point to the lib/ folder (contains nvinfer_10.dll)
            TryAddDir(val, L"nvinfer_10.dll", "TRT 10 lib (TRT_LIB_PATH)");
            foundTrt = true;
        }
        if (!foundTrt &&
            GetEnvironmentVariableW(L"TENSORRT_DIR", val, MAX_PATH) && val[0]) {
            TryAddDir(std::wstring(val) + L"\\lib",
                      L"nvinfer_10.dll", "TRT 10 lib (TENSORRT_DIR/lib)");
            foundTrt = true;
        }
        if (!foundTrt) {
            std::wstring inst = RegReadSz(HKEY_LOCAL_MACHINE,
                L"SOFTWARE\\NVIDIA Corporation\\TensorRT", L"InstallPath");
            if (!inst.empty())
                foundTrt = RecursiveFindAndAdd(inst, L"nvinfer_10.dll",
                                               "TRT 10 (registry)");
        }
        if (!foundTrt) {
            LOG_INFO("  TensorRT 10 not found via env/registry.");
            LOG_INFO("  To use TRT: extract TensorRT-10.x.zip, set:");
            LOG_INFO("    TRT_LIB_PATH=C:\\path\\to\\TensorRT-10.x.y.z\\lib");
            LOG_INFO("  IMPORTANT: match TRT version to your CUDA version:");
            LOG_INFO("    TRT 10.7.x -> CUDA 12.6    TRT 10.15.x -> CUDA 12.9");
            LOG_INFO("  Download: https://developer.nvidia.com/tensorrt");
        }
    }

    LOG_INFO("--- end GPU runtime path discovery ---");
}

// ── Static initialiser: runs during DLL_PROCESS_ATTACH ───────────────────────
namespace {
struct DllInit {
    DllInit() {
        // ── Step 1: add our own .ax directory and the registered install dir ──
        // Must be the first thing we do so the delay-loaded onnxruntime.dll
        // is found when its thunk fires on the first ORT call.
        std::wstring dllDir     = GetThisDllDir();
        std::wstring installDir = GetRegisteredInstallDir();

        // Step 2 calls SetDefaultDllDirectories.  AddDllDirectory before that
        // call uses the old search order, which is fine for our own dir.
        if (!dllDir.empty())
            AddDllDirectory(dllDir.c_str());
        if (!installDir.empty() &&
            _wcsicmp(installDir.c_str(), dllDir.c_str()) != 0)
            AddDllDirectory(installDir.c_str());

        // ── Step 2: switch to safe DLL search order ───────────────────────────
        // This restricts LoadLibrary to System32, Windows dir, and directories
        // explicitly added via AddDllDirectory.  Must be called before the
        // CUDA env-var early pass below so those dirs are also in effect.
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                                  LOAD_LIBRARY_SEARCH_USER_DIRS);

        // ── Step 3: early silent CUDA path registration ───────────────────────
        // The full logged pass happens in step 6 after the logger is live.
        // This early pass ensures CUDA is on the search path even if ORT
        // delay-load fires before the logger initialises (unlikely but safe).
        {
            const wchar_t* vars[] = {
                L"CUDA_PATH_V12_6",L"CUDA_PATH_V12_5",L"CUDA_PATH_V12_4",
                L"CUDA_PATH_V12_3",L"CUDA_PATH_V12_2",L"CUDA_PATH_V12_1",
                L"CUDA_PATH_V12_0",L"CUDA_PATH", nullptr
            };
            for (int i = 0; vars[i]; ++i) {
                wchar_t v[MAX_PATH] = {};
                if (GetEnvironmentVariableW(vars[i], v, MAX_PATH) && v[0]) {
                    std::wstring bin = std::wstring(v) + L"\\bin";
                    std::wstring chk = bin + L"\\cudart64_12.dll";
                    if (GetFileAttributesW(chk.c_str()) != INVALID_FILE_ATTRIBUTES) {
                        AddDllDirectory(bin.c_str());
                        break;
                    }
                }
            }
            // TRT: register both TRT_LIB_PATH (has nvinfer_10.dll) and TENSORRT_DIR/lib
            {
                wchar_t trtVal[MAX_PATH] = {};
                if (GetEnvironmentVariableW(L"TRT_LIB_PATH", trtVal, MAX_PATH) && trtVal[0]) {
                    AddDllDirectory(trtVal);
                    // Also add sibling bin/ dir in case TRT has split DLL locations
                    std::wstring trtParent = trtVal;
                    auto sl = trtParent.find_last_of(L"\\/");
                    if (sl != std::wstring::npos) {
                        std::wstring binSib = trtParent.substr(0, sl) + L"\\bin";
                        if (GetFileAttributesW(binSib.c_str()) != INVALID_FILE_ATTRIBUTES)
                            AddDllDirectory(binSib.c_str());
                    }
                }
                wchar_t trtDir[MAX_PATH] = {};
                if (GetEnvironmentVariableW(L"TENSORRT_DIR", trtDir, MAX_PATH) && trtDir[0]) {
                    AddDllDirectory((std::wstring(trtDir) + L"\\lib").c_str());
                    AddDllDirectory((std::wstring(trtDir) + L"\\bin").c_str());
                }
            }
        }

        // ── Step 4: register common controls (trackbar / slider classes) ──────
        INITCOMMONCONTROLSEX icc{};
        icc.dwSize = sizeof(icc);
        icc.dwICC  = ICC_BAR_CLASSES | ICC_STANDARD_CLASSES;
        InitCommonControlsEx(&icc);

        // ── Step 5: init logger ───────────────────────────────────────────────
        wchar_t exePath[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        Logger::Instance().Init(exePath);

        if (Logger::Instance().IsEnabled()) {
            wchar_t logPath[MAX_PATH] = {};
            GetEnvironmentVariableW(L"DEFLATTEN_LOG_FILE", logPath, MAX_PATH);
            LOG_INFO("================================================");
            LOG_INFO("3Deflatten v1.0.0  build: " __DATE__ " " __TIME__);
            LOG_INFO("Log file  : ", std::wstring(logPath));
            LOG_INFO("Host EXE  : ", std::wstring(exePath));
            // Log the full path to the .ax file for easy diagnosis
            {
                wchar_t axPath[MAX_PATH] = {};
                HMODULE hm = nullptr;
                GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCWSTR>(&GetThisDllDir), &hm);
                if (hm) GetModuleFileNameW(hm, axPath, MAX_PATH);
                LOG_INFO("DLL path  : ", std::wstring(axPath));
                LOG_INFO("DLL dir   : ", dllDir);
            }
            if (!installDir.empty() &&
                _wcsicmp(installDir.c_str(), dllDir.c_str()) != 0)
                LOG_INFO("Install dir: ", installDir,
                         "  (from registry -- also on DLL search path)");
            LOG_INFO("================================================");
            LOG_INFO("Set DEFLATTEN_LOG_FILE=<path> to enable logging");
            LOG_INFO("Set DEFLATTEN_MODEL_PATH=<path.onnx> to force model");
        }

        // ── Step 6: full GPU runtime path discovery (with logging) ───────────
        RegisterGpuRuntimeDirs();
    }
} g_dllInit;
} // namespace

// ── Filter factory table ──────────────────────────────────────────────────────
CFactoryTemplate g_Templates[] = {
    {
        L"3Deflatten (2D to 3D AI Depth)",
        &CLSID_3Deflatten,
        C3DeflattenFilter::CreateInstance,
        nullptr,
        &sudFilter
    },
    {
        L"3Deflatten Property Page",
        &CLSID_3DeflattenProp,
        C3DeflattenProp::CreateInstance,
        nullptr,
        nullptr
    }
};
int g_cTemplates = ARRAYSIZE(g_Templates);

STDAPI DllRegisterServer()   { return AMovieDllRegisterServer2(TRUE);  }
STDAPI DllUnregisterServer() { return AMovieDllRegisterServer2(FALSE); }
