// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten -- DLL factory table, dependency-path setup, and logger init
//
// DllMain, DllGetClassObject, and DllCanUnloadNow are provided by
// dllentry.cpp from the DirectShow baseclasses.  We must NOT define DllMain.
//
// DEPENDENCY LOADING ORDER (critical for COM / regsvr32 use)
// ──────────────────────────────────────────────────────────
// onnxruntime.dll (and directml.dll, provider DLLs) are /DELAYLOAD-ed.
// This means Windows does NOT resolve them when the .ax is first loaded by
// COM -- they are resolved on first call.  The static DllInit object below
// therefore runs BEFORE any ORT symbol is touched and can safely call
// AddDllDirectory() to teach Windows where to find those DLLs.
//
// Without delay-loading, COM would fail to load the .ax as a registered
// filter (CO_E_ERRORINDLL / error 126) because onnxruntime.dll is not on
// the system PATH -- it lives in the same folder as the .ax file.

#include <windows.h>
#include <commctrl.h>
#pragma comment(lib, "comctl32.lib")
#include <streams.h>
#include "filter.h"
#include "prop_page.h"
#include "guids.h"
#include "logger.h"

// ── DLL self-location helper ──────────────────────────────────────────────────
static std::wstring GetThisDllDir() {
    wchar_t path[MAX_PATH] = {};
    HMODULE hm = nullptr;
    // Use the address of this function itself -- works regardless of module name.
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&GetThisDllDir), &hm);
    if (hm) GetModuleFileNameW(hm, path, MAX_PATH);
    wchar_t* sl = wcsrchr(path, L'\\');
    if (sl) *sl = L'\0';
    return path;
}

// ── Static initialiser: runs during DLL_PROCESS_ATTACH ───────────────────────
namespace {
struct DllInit {
    DllInit() {
        // ── Step 1: add our own directory to the DLL search path ─────────────
        // Must happen BEFORE any delay-loaded symbol (onnxruntime.dll,
        // directml.dll, onnxruntime_providers_*.dll) is first called.
        // This is what allows the filter to load correctly when COM instantiates
        // it from the registry, where our directory is NOT on PATH.
        std::wstring dllDir = GetThisDllDir();
        if (!dllDir.empty()) {
            // AddDllDirectory registers an additional search directory.
            // SetDefaultDllDirectories enables the "safe" search order which
            // includes all directories added via AddDllDirectory.
            AddDllDirectory(dllDir.c_str());
            SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                                      LOAD_LIBRARY_SEARCH_USER_DIRS);
        }

        // ── Step 2: register common controls (trackbar / slider classes) ─────
        // Without this the property page dialog renders blank because
        // "msctls_trackbar32" is an unknown window class.
        INITCOMMONCONTROLSEX icc{};
        icc.dwSize = sizeof(icc);
        icc.dwICC  = ICC_BAR_CLASSES | ICC_STANDARD_CLASSES;
        InitCommonControlsEx(&icc);

        // ── Step 3: init logger ───────────────────────────────────────────────
        // GetModuleFileName(nullptr) returns the HOST exe path, which is what
        // Logger::Init uses to find the DEFLATTEN_LOG_FILE env var fallback.
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
            LOG_INFO("DLL dir   : ", dllDir);
            LOG_INFO("================================================");
            LOG_INFO("Set DEFLATTEN_LOG_FILE=<path> to enable logging");
            LOG_INFO("Set DEFLATTEN_MODEL_PATH=<path.onnx> to force model");
        }
    }
} g_dllInit;
} // namespace

// ── Filter factory table (read by dllentry.cpp / AMovieSetupRegisterFilter) ──
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

// ── Standard DirectShow DLL exports (forwarded from dllentry.cpp) ─────────────
STDAPI DllRegisterServer()   { return AMovieDllRegisterServer2(TRUE);  }
STDAPI DllUnregisterServer() { return AMovieDllRegisterServer2(FALSE); }
