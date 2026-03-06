// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten -- DLL factory table and logger initialisation
//
// DllMain, DllGetClassObject, and DllCanUnloadNow are all provided by
// dllentry.cpp from the DirectShow baseclasses (compiled into the filter
// target via CMakeLists.txt).  We must NOT define DllMain here.
//
// Logger init is done via a static-storage object whose constructor runs
// before any DirectShow factory code, giving us logging from the very start.
#include <streams.h>
#include "filter.h"
#include "prop_page.h"
#include "guids.h"
#include "logger.h"

// ── Logger init ───────────────────────────────────────────────────────────────
namespace {
struct LoggerInit {
    LoggerInit() {
        wchar_t path[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, path, MAX_PATH);
        Logger::Instance().Init(path);
        if (Logger::Instance().IsEnabled()) {
            wchar_t logPath[MAX_PATH] = {};
            GetEnvironmentVariableW(L"DEFLATTEN_LOG_FILE", logPath, MAX_PATH);
            LOG_INFO("================================================");
            LOG_INFO("3Deflatten v1.0.0  build: " __DATE__ " " __TIME__);
            LOG_INFO("Log file  : ", std::wstring(logPath));
            LOG_INFO("Host EXE  : ", std::wstring(path));
            LOG_INFO("================================================");
            LOG_INFO("Set DEFLATTEN_LOG_FILE=<path> to enable logging");
            LOG_INFO("Set DEFLATTEN_MODEL_PATH=<path.onnx> to force model");
        }
    }
} g_loggerInit;
} // namespace

// ── Filter factory table (read by dllentry.cpp) ───────────────────────────────
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
