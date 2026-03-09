// SPDX-License-Identifier: GPL-3.0-or-later
#include "prop_page.h"
#include "logger.h"
#include <commctrl.h>
#include <shlwapi.h>
#include <cstdio>

// ── Slider mappings ───────────────────────────────────────────────────────────
static constexpr int CONV_TICKS   = 1000;  // -> [0.000 .. 1.000]
static constexpr int SEP_TICKS    = 1000;  // -> [0.000 .. 0.200]
static constexpr int SMOOTH_TICKS = 100;   // -> [0.00  .. 1.00 ]
static constexpr float SEP_MAX    = 0.20f;

static int   ConvToSlider(float f)  { return (int)(f * CONV_TICKS + 0.5f); }
static float SliderToConv(int v)    { return (float)v / CONV_TICKS; }
static int   SepToSlider(float f)   { return (int)(f / SEP_MAX * SEP_TICKS + 0.5f); }
static float SliderToSep(int v)     { return (float)v / SEP_TICKS * SEP_MAX; }
static int   SmoothToSlider(float f){ return (int)(f * SMOOTH_TICKS + 0.5f); }
static float SliderToSmooth(int v)  { return (float)v / SMOOTH_TICKS; }

// ── CreateInstance ────────────────────────────────────────────────────────────
// g_hInst is the DirectShow baseclasses global used by CBasePropertyPage::
// OnActivate to call CreateDialogParam.  If it somehow points to a different
// module (stale value from another filter, or set before our DllMain ran),
// the dialog resource 101 won't be found and the property page will be blank.
// Patching it here ensures it is always our .ax's HMODULE.
extern HINSTANCE g_hInst;  // from baseclasses dllentry.cpp

CUnknown* WINAPI C3DeflattenProp::CreateInstance(LPUNKNOWN pUnk, HRESULT* phr) {
    HMODULE hSelf = nullptr;
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&C3DeflattenProp::CreateInstance), &hSelf);
    if (hSelf) {
        g_hInst = hSelf;
        LOG_DBG("PropPage::CreateInstance -- g_hInst patched to our .ax module");
    }
    return new C3DeflattenProp(pUnk, phr);
}

C3DeflattenProp::C3DeflattenProp(LPUNKNOWN pUnk, HRESULT* phr)
    : CBasePropertyPage(L"3Deflatten", pUnk, IDD_PROP_PAGE, IDS_PROP_TITLE)
{
    if (phr) *phr = S_OK;
}

// ── Connect / Disconnect ──────────────────────────────────────────────────────
HRESULT C3DeflattenProp::OnConnect(IUnknown* pUnk) {
    HRESULT hr = pUnk->QueryInterface(IID_I3Deflatten, (void**)&m_pFilter);
    if (FAILED(hr)) { LOG_WARN("PropPage::OnConnect QI failed"); return hr; }
    m_pFilter->GetConfig(&m_cfg);
    m_pFilter->GetModelPath(m_modelPath, MAX_PATH);
    LOG_DBG("PropPage::OnConnect ok");
    return S_OK;
}

HRESULT C3DeflattenProp::OnDisconnect() {
    if (m_pFilter) { m_pFilter->Release(); m_pFilter = nullptr; }
    return S_OK;
}

HRESULT C3DeflattenProp::OnApplyChanges() {
    if (!m_pFilter || !m_hwnd) return E_UNEXPECTED;
    ReadControls(m_hwnd);
    PushConfig();
    return S_OK;
}

// ── Message handler ───────────────────────────────────────────────────────────
INT_PTR C3DeflattenProp::OnReceiveMessage(HWND hwnd, UINT msg,
                                           WPARAM wParam, LPARAM lParam) {
    switch (msg) {

    case WM_INITDIALOG: {
        // Note: CBasePropertyPage's DialogProc sets m_hwnd = hwnd BEFORE
        // calling us, so m_hwnd is valid here.  Use hwnd directly for safety.

        // Slider ranges
        SendDlgItemMessage(hwnd, IDC_CONV_SLIDER,   TBM_SETRANGE, TRUE, MAKELPARAM(0, CONV_TICKS));
        SendDlgItemMessage(hwnd, IDC_SEP_SLIDER,    TBM_SETRANGE, TRUE, MAKELPARAM(0, SEP_TICKS));
        SendDlgItemMessage(hwnd, IDC_SMOOTH_SLIDER, TBM_SETRANGE, TRUE, MAKELPARAM(0, SMOOTH_TICKS));

        // Output Mode
        SendDlgItemMessage(hwnd, IDC_MODE_COMBO, CB_ADDSTRING, 0, (LPARAM)L"Side-by-Side (SBS)");
        SendDlgItemMessage(hwnd, IDC_MODE_COMBO, CB_ADDSTRING, 0, (LPARAM)L"Top-and-Bottom (TAB)");

        // GPU Provider – 5 options
        SendDlgItemMessage(hwnd, IDC_GPU_COMBO, CB_ADDSTRING, 0, (LPARAM)L"Auto (best available)");
        SendDlgItemMessage(hwnd, IDC_GPU_COMBO, CB_ADDSTRING, 0, (LPARAM)L"TensorRT  (NVIDIA, fastest – slow 1st run)");
        SendDlgItemMessage(hwnd, IDC_GPU_COMBO, CB_ADDSTRING, 0, (LPARAM)L"CUDA  (NVIDIA, fast)");
        SendDlgItemMessage(hwnd, IDC_GPU_COMBO, CB_ADDSTRING, 0, (LPARAM)L"DirectML  (any DX12 GPU)");
        SendDlgItemMessage(hwnd, IDC_GPU_COMBO, CB_ADDSTRING, 0, (LPARAM)L"CPU  (slow, always works)");

        PopulateModelCombo(hwnd);
        PopulateControls(hwnd);
        RefreshStatus(hwnd);
        return TRUE;
    }

    case WM_HSCROLL: {
        HWND hCtl = (HWND)lParam;
        int  id   = GetDlgCtrlID(hCtl);
        if (id==IDC_CONV_SLIDER || id==IDC_SEP_SLIDER || id==IDC_SMOOTH_SLIDER) {
            ReadControls(hwnd);
            UpdateValueLabels(hwnd);
            PushConfig();   // real-time update while scrubbing
            SetDirty();
        }
        break;
    }

    case WM_COMMAND: {
        int ctl  = LOWORD(wParam);
        int note = HIWORD(wParam);

        if (ctl==IDC_MODE_COMBO && note==CBN_SELCHANGE) {
            ReadControls(hwnd); PushConfig(); SetDirty(); break;
        }
        if (ctl==IDC_FLIP_CHECK && note==BN_CLICKED) {
            ReadControls(hwnd); PushConfig(); SetDirty(); break;
        }
        if (ctl==IDC_GPU_COMBO && note==CBN_SELCHANGE) {
            // Don't reload now; user must press Reload.
            ReadControls(hwnd); SetDirty();
            SetDlgItemTextW(hwnd, IDC_GPU_INFO,
                L"Press 'Reload' to apply the new provider.");
            break;
        }
        if (ctl==IDC_MODEL_COMBO && note==CBN_SELCHANGE) {
            int idx = (int)SendDlgItemMessage(hwnd, IDC_MODEL_COMBO,
                                               CB_GETCURSEL, 0, 0);
            if (idx >= 0 && idx < (int)m_onnxFiles.size()) {
                wcsncpy_s(m_modelPath, m_onnxFiles[idx].c_str(), _TRUNCATE);
                if (m_pFilter) m_pFilter->SetModelPath(m_modelPath);
            }
            SetDirty();
            SetDlgItemTextW(hwnd, IDC_GPU_INFO,
                L"Press 'Reload' to load the selected model.");
            break;
        }
        if (ctl==IDC_RELOAD_BTN && m_pFilter) {
            ReadControls(hwnd);
            m_pFilter->SetConfig(&m_cfg);
            m_pFilter->SetModelPath(m_modelPath);
            SetDlgItemTextW(hwnd, IDC_GPU_INFO, L"Loading model…");
            UpdateWindow(hwnd);
            HRESULT hr = m_pFilter->ReloadModel();
            RefreshStatus(hwnd);
            if (FAILED(hr))
                SetDlgItemTextW(hwnd, IDC_GPU_INFO,
                    L"Reload FAILED – check the log file.");
            break;
        }
        break;
    }

    } // switch
    return CBasePropertyPage::OnReceiveMessage(hwnd, msg, wParam, lParam);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
void C3DeflattenProp::SetDirty() {
    m_bDirty = TRUE;
    if (m_pPageSite) m_pPageSite->OnStatusChange(PROPPAGESTATUS_DIRTY);
}

void C3DeflattenProp::PushConfig() {
    if (m_pFilter) m_pFilter->SetConfig(&m_cfg);
}

void C3DeflattenProp::PopulateControls(HWND hwnd) {
    SendDlgItemMessage(hwnd, IDC_CONV_SLIDER,   TBM_SETPOS, TRUE, ConvToSlider(m_cfg.convergence));
    SendDlgItemMessage(hwnd, IDC_SEP_SLIDER,    TBM_SETPOS, TRUE, SepToSlider(m_cfg.separation));
    SendDlgItemMessage(hwnd, IDC_SMOOTH_SLIDER, TBM_SETPOS, TRUE, SmoothToSlider(m_cfg.depthSmooth));
    SendDlgItemMessage(hwnd, IDC_MODE_COMBO, CB_SETCURSEL, (int)m_cfg.outputMode, 0);
    SendDlgItemMessage(hwnd, IDC_GPU_COMBO,  CB_SETCURSEL, (int)m_cfg.gpuProvider, 0);
    SendDlgItemMessage(hwnd, IDC_FLIP_CHECK, BM_SETCHECK,
                       m_cfg.flipDepth ? BST_CHECKED : BST_UNCHECKED, 0);
    UpdateValueLabels(hwnd);
}

void C3DeflattenProp::ReadControls(HWND hwnd) {
    m_cfg.convergence = SliderToConv((int)SendDlgItemMessage(hwnd, IDC_CONV_SLIDER,   TBM_GETPOS, 0, 0));
    m_cfg.separation  = SliderToSep ((int)SendDlgItemMessage(hwnd, IDC_SEP_SLIDER,    TBM_GETPOS, 0, 0));
    m_cfg.depthSmooth = SliderToSmooth((int)SendDlgItemMessage(hwnd, IDC_SMOOTH_SLIDER, TBM_GETPOS, 0, 0));
    m_cfg.outputMode  = (OutputMode) SendDlgItemMessage(hwnd, IDC_MODE_COMBO, CB_GETCURSEL, 0, 0);
    m_cfg.gpuProvider = (GPUProvider)SendDlgItemMessage(hwnd, IDC_GPU_COMBO,  CB_GETCURSEL, 0, 0);
    m_cfg.flipDepth   = (SendDlgItemMessage(hwnd, IDC_FLIP_CHECK, BM_GETCHECK, 0, 0) == BST_CHECKED)
                        ? TRUE : FALSE;
}

void C3DeflattenProp::UpdateValueLabels(HWND hwnd) {
    wchar_t buf[32];
    swprintf_s(buf, L"%.3f", m_cfg.convergence);
    SetDlgItemTextW(hwnd, IDC_CONV_LABEL, buf);
    swprintf_s(buf, L"%.3f", m_cfg.separation);
    SetDlgItemTextW(hwnd, IDC_SEP_LABEL, buf);
    swprintf_s(buf, L"%.2f", m_cfg.depthSmooth);
    SetDlgItemTextW(hwnd, IDC_SMOOTH_LABEL, buf);
}

void C3DeflattenProp::RefreshStatus(HWND hwnd) {
    wchar_t info[512]{};
    if (m_pFilter) m_pFilter->GetGPUInfo(info, 512);
    if (info[0] == L'\0')
        wcscpy_s(info, L"Model not loaded. Press 'Reload' to start.");
    SetDlgItemTextW(hwnd, IDC_GPU_INFO, info);
}

// Return the directory that contains this DLL (the .ax file).
// Uses the function's own address so it works regardless of module name.
/*static*/ std::wstring C3DeflattenProp::GetDllDir() {
    HMODULE hm = nullptr;
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&C3DeflattenProp::GetDllDir),
        &hm);
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(hm, path, MAX_PATH);
    // Strip filename to get directory
    wchar_t* slash = wcsrchr(path, L'\\');
    if (!slash) slash = wcsrchr(path, L'/');
    if (slash) *slash = L'\0';
    return path;
}

void C3DeflattenProp::PopulateModelCombo(HWND hwnd) {
    SendDlgItemMessage(hwnd, IDC_MODEL_COMBO, CB_RESETCONTENT, 0, 0);
    m_onnxFiles.clear();

    // Search directories: DLL directory, then current model's directory
    std::vector<std::wstring> searchDirs;
    searchDirs.push_back(GetDllDir());
    if (m_modelPath[0]) {
        wchar_t parentDir[MAX_PATH]{};
        wcsncpy_s(parentDir, m_modelPath, _TRUNCATE);
        wchar_t* sl = wcsrchr(parentDir, L'\\');
        if (!sl) sl = wcsrchr(parentDir, L'/');
        if (sl) *sl = L'\0';
        if (!parentDir[0] || _wcsicmp(parentDir, searchDirs[0].c_str()) != 0)
            searchDirs.push_back(parentDir);
    }

    // Also add parent of DLL dir (install root)
    std::wstring dllParent = GetDllDir();
    auto pslash = dllParent.find_last_of(L"\\/");
    if (pslash != std::wstring::npos) {
        dllParent.resize(pslash);
        searchDirs.push_back(dllParent);
    }

    int selIdx = 0;
    for (auto& dir : searchDirs) {
        if (dir.empty()) continue;
        wchar_t pattern[MAX_PATH]{};
        swprintf_s(pattern, L"%s\\*.onnx", dir.c_str());
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileW(pattern, &fd);
        if (h == INVALID_HANDLE_VALUE) continue;
        do {
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
            wchar_t full[MAX_PATH]{};
            swprintf_s(full, L"%s\\%s", dir.c_str(), fd.cFileName);
            // Deduplicate
            bool dup = false;
            for (auto& e : m_onnxFiles)
                if (_wcsicmp(e.c_str(), full) == 0) { dup=true; break; }
            if (dup) continue;

            int idx = (int)SendDlgItemMessage(hwnd, IDC_MODEL_COMBO,
                                               CB_ADDSTRING, 0, (LPARAM)fd.cFileName);
            m_onnxFiles.push_back(full);
            if (_wcsicmp(full, m_modelPath) == 0) selIdx = idx;
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }

    if (m_onnxFiles.empty()) {
        SendDlgItemMessage(hwnd, IDC_MODEL_COMBO, CB_ADDSTRING, 0,
                           (LPARAM)L"(no .onnx files found)");
        m_onnxFiles.push_back(L"");
    }
    SendDlgItemMessage(hwnd, IDC_MODEL_COMBO, CB_SETCURSEL, selIdx, 0);
    LOG_DBG("PropPage: found ", m_onnxFiles.size(), " .onnx file(s)");
}
