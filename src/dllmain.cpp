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
#include <functional>
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


// ─────────────────────────────────────────────────────────────────────────────
// AddCudaDir: add CUDA bin directory AND its parent + nvJitLink subdirectory.
//
// CUDA 13.x on Windows may store DLLs under  bin\x64\  while nvJitLink
// (required by ORT CUDA EP) lives in  bin\  or a  nvJitLink\  sub-tree.
// Adding multiple levels ensures all transitive dependencies can be loaded.
// ─────────────────────────────────────────────────────────────────────────────
static void AddCudaDir(const std::wstring& cudaDllDir, const wchar_t* probeDll,
                       const char* label) {
    if (cudaDllDir.empty()) return;
    TryAddDir(cudaDllDir, probeDll, label);
    // Add parent directory (e.g. bin\ when we found bin\x64\) --
    // nvJitLink_130_0.dll and nvrtc64_130_0.dll may live there.
    auto sl = cudaDllDir.find_last_of(L"\\/");
    if (sl != std::wstring::npos) {
        std::wstring parent = cudaDllDir.substr(0, sl);
        if (_wcsicmp(parent.c_str(), cudaDllDir.c_str()) != 0 &&
            GetFileAttributesW(parent.c_str()) != INVALID_FILE_ATTRIBUTES) {
            AddDllDirectory(parent.c_str());
            LOG_INFO("  added CUDA parent dir: ", parent);
        }
        // nvJitLink sub-directories exist only in CUDA 12+; not needed for CUDA 11.
    }
}

// Discover CUDA 11.x, cuDNN 8.x, and TensorRT 10.x install directories and
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
#if ORT_CUDA_MAJOR == 13
    LOG_INFO("  (searching for CUDA 13.x, cuDNN 9.x, TensorRT 10.x)");
    constexpr wchar_t CUDA_RT_DLL[] = L"cudart64_13.dll";
    constexpr wchar_t CUDNN_DLL[]   = L"cudnn64_9.dll";
    constexpr char    CUDA_LABEL[]  = "CUDA 13";
    constexpr char    CUDNN_LABEL[] = "cuDNN 9";
    constexpr char    CUDA_COLLECT[]= "collect_runtime_dlls_cuda13.py";
    constexpr char    CUDNN_COLLECT[]="collect_runtime_dlls_cuda13.py";
    const wchar_t* cudaEnvVars[] = {
        L"CUDA_PATH_V13_0", L"CUDA_PATH_V13_1", L"CUDA_PATH_V13_2", L"CUDA_PATH",
        nullptr
    };
    auto versionMatch = [](const wchar_t* s) { return wcsncmp(s, L"v13.", 4) == 0; };
#else
    LOG_INFO("  (searching for CUDA 11.x, cuDNN 8.x, TensorRT 10.x)");
    constexpr wchar_t CUDA_RT_DLL[] = L"cudart64_11.dll";
    constexpr wchar_t CUDNN_DLL[]   = L"cudnn64_8.dll";
    constexpr char    CUDA_LABEL[]  = "CUDA 11";
    constexpr char    CUDNN_LABEL[] = "cuDNN 8";
    constexpr char    CUDA_COLLECT[]= "collect_runtime_dlls.py";
    constexpr char    CUDNN_COLLECT[]="collect_runtime_dlls.py";
    const wchar_t* cudaEnvVars[] = {
        L"CUDA_PATH_V11_8", L"CUDA_PATH_V11_7", L"CUDA_PATH_V11_6",
        L"CUDA_PATH_V11_5", L"CUDA_PATH_V11_4", L"CUDA_PATH",
        nullptr
    };
    auto versionMatch = [](const wchar_t* s) { return wcsncmp(s, L"v11.", 4) == 0; };
#endif

    // ── CUDA Toolkit bin ─────────────────────────────────────────────────────
    bool foundCuda = false;
    for (int i = 0; cudaEnvVars[i] && !foundCuda; ++i) {
        wchar_t val[MAX_PATH] = {};
        if (GetEnvironmentVariableW(cudaEnvVars[i], val, MAX_PATH) && val[0]) {
            std::wstring binDir = std::wstring(val) + L"\\bin";
            std::wstring probe  = binDir + L"\\" + CUDA_RT_DLL;
            if (GetFileAttributesW(probe.c_str()) != INVALID_FILE_ATTRIBUTES) {
                AddCudaDir(binDir, CUDA_RT_DLL, "CUDA bin (env var)");
                foundCuda = true;
            }
        }
    }
    if (!foundCuda) {
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
                if (!versionMatch(subName)) continue;
                std::wstring inst = RegReadSz(hBase, subName, L"InstallDir");
                if (inst.empty()) continue;
                std::wstring binDir = inst + L"\\bin";
                std::wstring probe  = binDir + L"\\" + CUDA_RT_DLL;
                if (GetFileAttributesW(probe.c_str()) != INVALID_FILE_ATTRIBUTES) {
                    AddCudaDir(binDir, CUDA_RT_DLL, "CUDA bin (registry)");
                    foundCuda = true;
                }
            }
            RegCloseKey(hBase);
        }
    }
    if (!foundCuda) {
        std::wstring cudaBase =
            L"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA";
        WIN32_FIND_DATAW fd{};
        std::function<std::wstring(const std::wstring&, int)> findDir;
        findDir = [&](const std::wstring& dir, int depth) -> std::wstring {
            if (depth <= 0 || dir.empty()) return {};
            if (GetFileAttributesW((dir + L"\\" + CUDA_RT_DLL).c_str()) !=
                INVALID_FILE_ATTRIBUTES) return dir;
            HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd);
            if (h == INVALID_HANDLE_VALUE) return {};
            std::wstring r;
            do {
                if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
                if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
                r = findDir(dir + L"\\" + fd.cFileName, depth - 1);
            } while (r.empty() && FindNextFileW(h, &fd));
            FindClose(h);
            return r;
        };
        std::wstring dir = findDir(cudaBase, 5);
        if (!dir.empty()) {
            AddCudaDir(dir, CUDA_RT_DLL, "CUDA bin (default scan)");
            foundCuda = true;
        }
    }
    if (!foundCuda) {
        HMODULE hBundled = LoadLibraryExW(CUDA_RT_DLL, nullptr,
                               LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                               LOAD_LIBRARY_SEARCH_USER_DIRS);
        if (hBundled) {
            FreeLibrary(hBundled);
            LOG_INFO("  ", CUDA_LABEL, " not found via system install, but ", CUDA_RT_DLL,
                     " is in the bundled DLLs folder -- OK.");
        } else {
            LOG_WARN("  ", CUDA_LABEL, " not found (no system install and not bundled).");
            LOG_WARN("  Run ", CUDA_COLLECT, " to bundle CUDA DLLs.");
        }
    }

    // ── cuDNN ─────────────────────────────────────────────────────────────────
    {
        bool foundCuDnn = false;
        wchar_t val[MAX_PATH] = {};

        if (!foundCuDnn &&
            GetEnvironmentVariableW(L"CUDNN_PATH", val, MAX_PATH) && val[0])
            foundCuDnn = RecursiveFindAndAdd(val, CUDNN_DLL,
                                             "cuDNN (CUDNN_PATH)");
        if (!foundCuDnn) {
            std::wstring inst = RegReadSz(HKEY_LOCAL_MACHINE,
                L"SOFTWARE\\NVIDIA Corporation\\cuDNN", L"InstallPath");
            if (!inst.empty())
                foundCuDnn = RecursiveFindAndAdd(inst, CUDNN_DLL,
                                                  "cuDNN (registry)");
        }
        if (!foundCuDnn)
            foundCuDnn = RecursiveFindAndAdd(
                L"C:\\Program Files\\NVIDIA\\CUDNN",
                CUDNN_DLL, "cuDNN (default scan)");

        if (!foundCuDnn) {
            HMODULE hBundled = LoadLibraryExW(CUDNN_DLL, nullptr,
                                   LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                                   LOAD_LIBRARY_SEARCH_USER_DIRS);
            if (hBundled) {
                FreeLibrary(hBundled);
                LOG_INFO("  ", CUDNN_LABEL, " not found via system install, but ", CUDNN_DLL,
                         " is in the bundled DLLs folder -- OK.");
            } else {
                LOG_INFO("  ", CUDNN_LABEL, " not found via env/registry/default path.");
                LOG_INFO("  Run ", CUDNN_COLLECT, " to bundle cuDNN DLLs, or");
                LOG_INFO("  if installed elsewhere: set CUDNN_PATH=<install root>.");
            }
        }
    } // end cuDNN block

    // ── TensorRT 10.x ────────────────────────────────────────────────────────
    // TRT 10.13+ layout: nvinfer_10.dll in lib\ alongside all other TRT DLLs.
    // zlibwapi.dll is NOT required by TRT 10.13+ (dependency was removed).
    {
        wchar_t val[MAX_PATH] = {};
        bool foundTrt = false;

        // Helper: add nvinfer directory + siblings to maximise sub-dep coverage
        auto addTrtRoot = [&](const std::wstring& nvinferDir, const char* label) {
            TryAddDir(nvinferDir, L"nvinfer_10.dll", label);
            // Walk up to TRT root and add sibling directories
            auto sl2 = nvinferDir.find_last_of(L"\\/");
            if (sl2 != std::wstring::npos) {
                std::wstring trtRoot = nvinferDir.substr(0, sl2);
                for (auto* sub : {L"lib", L"bin", L""}) {
                    std::wstring sib = sub[0] ? trtRoot + L"\\" + sub : trtRoot;
                    if (sib != nvinferDir &&
                        GetFileAttributesW(sib.c_str()) != INVALID_FILE_ATTRIBUTES) {
                        AddDllDirectory(sib.c_str());
                        LOG_INFO("  added TRT dir: ", sib);
                    }
                }
            }
        };

        if (GetEnvironmentVariableW(L"TRT_LIB_PATH", val, MAX_PATH) && val[0]) {
            std::wstring dir = val;
            std::wstring probe = dir + L"\\nvinfer_10.dll";
            if (GetFileAttributesW(probe.c_str()) != INVALID_FILE_ATTRIBUTES) {
                addTrtRoot(dir, "TRT 10 (TRT_LIB_PATH)");
                foundTrt = true;
            } else {
                LOG_WARN("  TRT_LIB_PATH set but nvinfer_10.dll not found in: ", dir);
            }
        }
        if (!foundTrt &&
            GetEnvironmentVariableW(L"TENSORRT_DIR", val, MAX_PATH) && val[0]) {
            std::wstring libDir = std::wstring(val) + L"\\lib";
            std::wstring probe  = libDir + L"\\nvinfer_10.dll";
            if (GetFileAttributesW(probe.c_str()) != INVALID_FILE_ATTRIBUTES) {
                addTrtRoot(libDir, "TRT 10 (TENSORRT_DIR/lib)");
                foundTrt = true;
            }
        }
        if (!foundTrt) {
            std::wstring inst = RegReadSz(HKEY_LOCAL_MACHINE,
                L"SOFTWARE\\NVIDIA Corporation\\TensorRT", L"InstallPath");
            if (!inst.empty())
                foundTrt = RecursiveFindAndAdd(inst, L"nvinfer_10.dll",
                                               "TRT 10 (registry)");
        }
        // Default scan: TensorRT zips typically extract under C:\Program Files\NVIDIA
        if (!foundTrt) {
            // Use lambda to find nvinfer_10.dll then call addTrtRoot
            std::wstring base = L"C:\\Program Files\\NVIDIA";
            WIN32_FIND_DATAW fd2{};
            std::function<std::wstring(const std::wstring&, int)> findTrt;
            findTrt = [&](const std::wstring& dir, int depth) -> std::wstring {
                if (depth <= 0 || dir.empty()) return {};
                std::wstring p = dir + L"\\nvinfer_10.dll";
                if (GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES) return dir;
                HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd2);
                if (h == INVALID_HANDLE_VALUE) return {};
                std::wstring r;
                do {
                    if (!(fd2.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
                    if (!wcscmp(fd2.cFileName, L".") || !wcscmp(fd2.cFileName, L"..")) continue;
                    r = findTrt(dir + L"\\" + fd2.cFileName, depth - 1);
                } while (r.empty() && FindNextFileW(h, &fd2));
                FindClose(h);
                return r;
            };
            std::wstring trtDir = findTrt(base, 5);
            if (!trtDir.empty()) {
                addTrtRoot(trtDir, "TRT 10 (default scan)");
                foundTrt = true;
            }
        }
        if (!foundTrt) {
            // Check if nvinfer_10.dll is bundled next to the .ax
            HMODULE hTrt = LoadLibraryExW(L"nvinfer_10.dll", nullptr,
                               LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                               LOAD_LIBRARY_SEARCH_USER_DIRS);
            if (hTrt) {
                FreeLibrary(hTrt);
                LOG_INFO("  TRT not found via system install, but nvinfer_10.dll");
                LOG_INFO("  is present in the bundled DLLs folder -- OK.");
            } else {
                LOG_INFO("  TensorRT 10 not found via env/registry/default scan.");
#if ORT_CUDA_MAJOR == 13
                LOG_INFO("  Run collect_runtime_dlls_cuda13.py to bundle TRT 10.13.3 DLLs, or");
                LOG_INFO("  extract TensorRT-10.13.3.9.Windows.win10.cuda-13.0.zip and set:");
                LOG_INFO("    TRT_LIB_PATH=C:\\path\\to\\TensorRT-10.13.3.9\\lib");
#else
                LOG_INFO("  Run collect_runtime_dlls.py to bundle TRT 10.13.0 DLLs, or");
                LOG_INFO("  extract TensorRT-10.13.0.35.Windows.win10.cuda-11.8.zip and set:");
                LOG_INFO("    TRT_LIB_PATH=C:\\path\\to\\TensorRT-10.13.0.35\\lib");
#endif
                LOG_INFO("  NOTE: env vars only take effect after restarting the host app.");
            }
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
                L"CUDA_PATH_V11_8",L"CUDA_PATH_V11_7",L"CUDA_PATH_V11_6",
                L"CUDA_PATH_V11_5",L"CUDA_PATH", nullptr
            };
            for (int i = 0; vars[i]; ++i) {
                wchar_t v[MAX_PATH] = {};
                if (GetEnvironmentVariableW(vars[i], v, MAX_PATH) && v[0]) {
                    std::wstring bin = std::wstring(v) + L"\\bin";
                    std::wstring chk = bin + L"\\cudart64_11.dll";
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
