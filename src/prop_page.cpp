// SPDX-License-Identifier: GPL-3.0-or-later
#include "prop_page.h"
#include "logger.h"
#include <commctrl.h>
#include <commdlg.h>
#include <cstdio>

// Slider integer ranges (mapped to float internally)
static constexpr int CONV_TICKS   = 1000;  // 0.000 – 1.000
static constexpr int SEP_TICKS    = 1000;  // 0.000 – 0.200
static constexpr int SMOOTH_TICKS = 100;   // 0.00  – 1.00
static constexpr float SEP_MAX_F  = 0.20f;

static inline float SliderToConv(int v)
    { return (float)v / CONV_TICKS; }
static inline int   ConvToSlider(float f)
    { return (int)(f * CONV_TICKS + 0.5f); }

static inline float SliderToSep(int v)
    { return (float)v / SEP_TICKS * SEP_MAX_F; }
static inline int   SepToSlider(float f)
    { return (int)(f / SEP_MAX_F * SEP_TICKS + 0.5f); }

static inline float SliderToSmooth(int v)
    { return (float)v / SMOOTH_TICKS; }
static inline int   SmoothToSlider(float f)
    { return (int)(f * SMOOTH_TICKS + 0.5f); }

// ── CreateInstance ────────────────────────────────────────────────────────────
CUnknown* WINAPI C3DeflattenProp::CreateInstance(LPUNKNOWN pUnk, HRESULT* phr) {
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
    LOG_DBG("PropPage::OnConnect ok  model='",
            std::wstring(m_modelPath), "'");
    return S_OK;
}

HRESULT C3DeflattenProp::OnDisconnect() {
    if (m_pFilter) { m_pFilter->Release(); m_pFilter = nullptr; }
    return S_OK;
}

// ── Apply ─────────────────────────────────────────────────────────────────────
HRESULT C3DeflattenProp::OnApplyChanges() {
    if (!m_pFilter) return E_UNEXPECTED;
    ReadControls();
    PushToFilter();
    return S_OK;
}

// ── Message handler ───────────────────────────────────────────────────────────
INT_PTR C3DeflattenProp::OnReceiveMessage(HWND hwnd, UINT msg,
                                           WPARAM wParam, LPARAM lParam) {
    switch (msg) {

    // ── Init ──────────────────────────────────────────────────────────────────
    case WM_INITDIALOG: {
        // Slider ranges
        SendDlgItemMessage(hwnd, IDC_CONV_SLIDER,   TBM_SETRANGE, TRUE,
                           MAKELPARAM(0, CONV_TICKS));
        SendDlgItemMessage(hwnd, IDC_SEP_SLIDER,    TBM_SETRANGE, TRUE,
                           MAKELPARAM(0, SEP_TICKS));
        SendDlgItemMessage(hwnd, IDC_SMOOTH_SLIDER, TBM_SETRANGE, TRUE,
                           MAKELPARAM(0, SMOOTH_TICKS));

        // Output Mode combo
        SendDlgItemMessage(hwnd, IDC_MODE_COMBO, CB_ADDSTRING, 0,
                           (LPARAM)L"Side-by-Side (SBS)");
        SendDlgItemMessage(hwnd, IDC_MODE_COMBO, CB_ADDSTRING, 0,
                           (LPARAM)L"Top-and-Bottom (TAB)");

        // GPU Provider combo
        SendDlgItemMessage(hwnd, IDC_GPU_COMBO, CB_ADDSTRING, 0,
                           (LPARAM)L"Auto (best available)");
        SendDlgItemMessage(hwnd, IDC_GPU_COMBO, CB_ADDSTRING, 0,
                           (LPARAM)L"CUDA (NVIDIA)");
        SendDlgItemMessage(hwnd, IDC_GPU_COMBO, CB_ADDSTRING, 0,
                           (LPARAM)L"DirectML (DX12 GPU)");
        SendDlgItemMessage(hwnd, IDC_GPU_COMBO, CB_ADDSTRING, 0,
                           (LPARAM)L"CPU");

        PopulateControls();
        RefreshStatus();
        return TRUE;
    }

    // ── Slider moved ──────────────────────────────────────────────────────────
    case WM_HSCROLL: {
        HWND hSlider = (HWND)lParam;
        int  id      = GetDlgCtrlID(hSlider);
        if (id == IDC_CONV_SLIDER ||
            id == IDC_SEP_SLIDER  ||
            id == IDC_SMOOTH_SLIDER) {
            ReadControls();
            UpdateValueLabels();
            PushToFilter();   // real-time
            SetDirty();
        }
        break;
    }

    // ── Combo / checkbox changed ──────────────────────────────────────────────
    case WM_COMMAND: {
        int ctl  = LOWORD(wParam);
        int note = HIWORD(wParam);

        // Output mode or flip depth: take effect immediately
        if ((ctl == IDC_MODE_COMBO  && note == CBN_SELCHANGE) ||
            (ctl == IDC_FLIP_CHECK  && (note == BN_CLICKED))) {
            ReadControls();
            PushToFilter();
            SetDirty();
            break;
        }

        // GPU provider changed: mark dirty but don't reload yet –
        // user must click Reload Model to apply (avoids mid-stream reload).
        if (ctl == IDC_GPU_COMBO && note == CBN_SELCHANGE) {
            ReadControls();
            SetDirty();
            SetDlgItemTextW(hwnd, IDC_GPU_INFO,
                            L"Click 'Reload Model' to apply new provider.");
            break;
        }

        // Browse for model file
        if (ctl == IDC_BROWSE_BTN) {
            OPENFILENAMEW ofn{};
            wchar_t path[MAX_PATH]{};
            ofn.lStructSize  = sizeof(ofn);
            ofn.hwndOwner    = hwnd;
            ofn.lpstrFilter  = L"ONNX Models\0*.onnx\0All Files\0*.*\0";
            ofn.lpstrFile    = path;
            ofn.nMaxFile     = MAX_PATH;
            ofn.Flags        = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
            ofn.lpstrTitle   = L"Select ONNX depth model";
            if (GetOpenFileNameW(&ofn)) {
                wcsncpy_s(m_modelPath, path, _TRUNCATE);
                SetDlgItemTextW(hwnd, IDC_MODEL_PATH, m_modelPath);
                if (m_pFilter) m_pFilter->SetModelPath(m_modelPath);
                SetDirty();
                SetDlgItemTextW(hwnd, IDC_GPU_INFO,
                                L"Click 'Reload Model' to load the new file.");
            }
            break;
        }

        // Reload model with current path + provider
        if (ctl == IDC_RELOAD_BTN && m_pFilter) {
            ReadControls();
            m_pFilter->SetConfig(&m_cfg);
            m_pFilter->SetModelPath(m_modelPath);
            SetDlgItemTextW(hwnd, IDC_GPU_INFO, L"Loading…");
            UpdateWindow(hwnd);
            HRESULT hr = m_pFilter->ReloadModel();
            RefreshStatus();
            if (FAILED(hr))
                SetDlgItemTextW(hwnd, IDC_GPU_INFO, L"Reload FAILED – check log.");
            break;
        }
        break;
    }
    } // switch

    return CBasePropertyPage::OnReceiveMessage(hwnd, msg, wParam, lParam);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
void C3DeflattenProp::PushToFilter() {
    if (!m_pFilter) return;
    m_pFilter->SetConfig(&m_cfg);
    // Model path is already pushed on Browse; no need to resend here.
}

void C3DeflattenProp::SetDirty() {
    m_bDirty = TRUE;
    if (m_pPageSite) m_pPageSite->OnStatusChange(PROPPAGESTATUS_DIRTY);
}

void C3DeflattenProp::PopulateControls() {
    if (!m_hwnd) return;

    SendDlgItemMessage(m_hwnd, IDC_CONV_SLIDER,   TBM_SETPOS, TRUE,
                       ConvToSlider(m_cfg.convergence));
    SendDlgItemMessage(m_hwnd, IDC_SEP_SLIDER,    TBM_SETPOS, TRUE,
                       SepToSlider(m_cfg.separation));
    SendDlgItemMessage(m_hwnd, IDC_SMOOTH_SLIDER, TBM_SETPOS, TRUE,
                       SmoothToSlider(m_cfg.depthSmooth));

    SendDlgItemMessage(m_hwnd, IDC_MODE_COMBO, CB_SETCURSEL,
                       (int)m_cfg.outputMode, 0);
    SendDlgItemMessage(m_hwnd, IDC_GPU_COMBO,  CB_SETCURSEL,
                       (int)m_cfg.gpuProvider, 0);
    SendDlgItemMessage(m_hwnd, IDC_FLIP_CHECK, BM_SETCHECK,
                       m_cfg.flipDepth ? BST_CHECKED : BST_UNCHECKED, 0);

    SetDlgItemTextW(m_hwnd, IDC_MODEL_PATH, m_modelPath);
    UpdateValueLabels();
}

void C3DeflattenProp::UpdateValueLabels() {
    if (!m_hwnd) return;
    wchar_t buf[32];
    swprintf_s(buf, L"%.3f", m_cfg.convergence);
    SetDlgItemTextW(m_hwnd, IDC_CONV_LABEL, buf);
    swprintf_s(buf, L"%.3f", m_cfg.separation);
    SetDlgItemTextW(m_hwnd, IDC_SEP_LABEL, buf);
    swprintf_s(buf, L"%.2f", m_cfg.depthSmooth);
    SetDlgItemTextW(m_hwnd, IDC_SMOOTH_LABEL, buf);
}

void C3DeflattenProp::ReadControls() {
    if (!m_hwnd) return;
    m_cfg.convergence = SliderToConv((int)SendDlgItemMessage(
        m_hwnd, IDC_CONV_SLIDER, TBM_GETPOS, 0, 0));
    m_cfg.separation  = SliderToSep((int)SendDlgItemMessage(
        m_hwnd, IDC_SEP_SLIDER, TBM_GETPOS, 0, 0));
    m_cfg.depthSmooth = SliderToSmooth((int)SendDlgItemMessage(
        m_hwnd, IDC_SMOOTH_SLIDER, TBM_GETPOS, 0, 0));
    m_cfg.outputMode  = (OutputMode)SendDlgItemMessage(
        m_hwnd, IDC_MODE_COMBO, CB_GETCURSEL, 0, 0);
    m_cfg.gpuProvider = (GPUProvider)SendDlgItemMessage(
        m_hwnd, IDC_GPU_COMBO, CB_GETCURSEL, 0, 0);
    m_cfg.flipDepth   = (SendDlgItemMessage(
        m_hwnd, IDC_FLIP_CHECK, BM_GETCHECK, 0, 0) == BST_CHECKED) ? TRUE : FALSE;
    GetDlgItemTextW(m_hwnd, IDC_MODEL_PATH, m_modelPath, MAX_PATH);
}

void C3DeflattenProp::RefreshStatus() {
    if (!m_hwnd || !m_pFilter) return;
    wchar_t info[512]{};
    m_pFilter->GetGPUInfo(info, 512);
    if (info[0] == L'\0')
        wcscpy_s(info, L"Model not loaded yet. Click 'Reload Model' to start.");
    SetDlgItemTextW(m_hwnd, IDC_GPU_INFO, info);
}
