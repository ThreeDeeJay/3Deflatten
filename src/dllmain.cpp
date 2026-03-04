// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten – DLL entry point & COM factory registration
#include <streams.h>
#include "filter.h"
#include "prop_page.h"
#include "guids.h"
#include "logger.h"

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

// ── Standard DirectShow DLL exports ──────────────────────────────────────────
STDAPI DllRegisterServer()   { return AMovieDllRegisterServer2(TRUE);  }
STDAPI DllUnregisterServer() { return AMovieDllRegisterServer2(FALSE); }

extern "C" BOOL WINAPI DllEntryPoint(HINSTANCE, ULONG, LPVOID);

BOOL WINAPI DllMain(HINSTANCE hInst, DWORD reason, LPVOID pv) {
    if (reason == DLL_PROCESS_ATTACH) {
        wchar_t path[MAX_PATH] = {};
        GetModuleFileNameW(hInst, path, MAX_PATH);
        Logger::Instance().Init(path);
        LOG_INFO("3Deflatten v1.0.0 DLL loaded");
        LOG_INFO("DLL path: ", std::wstring(path));
        LOG_INFO("Build: " __DATE__ " " __TIME__);
    }
    if (reason == DLL_PROCESS_DETACH) {
        LOG_INFO("3Deflatten DLL unloaded");
    }
    return DllEntryPoint(hInst, reason, pv);
}
