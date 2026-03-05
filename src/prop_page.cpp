// SPDX-License-Identifier: GPL-3.0-or-later
#include "prop_page.h"
#include "logger.h"
#include <commctrl.h>
#include <commdlg.h>
#include <cstdio>

static constexpr int CONV_MIN=0, CONV_MAX=100;
static constexpr int SEP_MIN=0,  SEP_MAX=200;

CUnknown* WINAPI C3DeflattenProp::CreateInstance(LPUNKNOWN pUnk,
                                                  HRESULT* phr) {
    return new C3DeflattenProp(pUnk, phr);
}

C3DeflattenProp::C3DeflattenProp(LPUNKNOWN pUnk, HRESULT* phr)
    : CBasePropertyPage(L"3Deflatten Settings",
                        pUnk, IDD_PROP_PAGE, IDS_PROP_TITLE) {
    if (phr) *phr = S_OK;
}

HRESULT C3DeflattenProp::OnConnect(IUnknown* pUnk) {
    HRESULT hr = pUnk->QueryInterface(IID_I3Deflatten, (void**)&m_pFilter);
    if (FAILED(hr)) return hr;
    m_pFilter->GetConfig(&m_cfg);
    return S_OK;
}

HRESULT C3DeflattenProp::OnDisconnect() {
    if (m_pFilter) { m_pFilter->Release(); m_pFilter = nullptr; }
    return S_OK;
}

HRESULT C3DeflattenProp::OnApplyChanges() {
    if (!m_pFilter) return E_UNEXPECTED;
    ReadControls();
    return m_pFilter->SetConfig(&m_cfg);
}

INT_PTR C3DeflattenProp::OnReceiveMessage(HWND hwnd, UINT msg,
                                           WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_INITDIALOG: {
        SendDlgItemMessage(hwnd,IDC_CONV_SLIDER,TBM_SETRANGE,TRUE,
                           MAKELPARAM(CONV_MIN,CONV_MAX));
        SendDlgItemMessage(hwnd,IDC_SEP_SLIDER,TBM_SETRANGE,TRUE,
                           MAKELPARAM(SEP_MIN,SEP_MAX));

        SendDlgItemMessage(hwnd,IDC_MODE_COMBO,CB_ADDSTRING,0,
                           (LPARAM)L"Side-by-Side (SBS)");
        SendDlgItemMessage(hwnd,IDC_MODE_COMBO,CB_ADDSTRING,0,
                           (LPARAM)L"Top-and-Bottom (TAB)");

        SendDlgItemMessage(hwnd,IDC_GPU_COMBO,CB_ADDSTRING,0,(LPARAM)L"Auto");
        SendDlgItemMessage(hwnd,IDC_GPU_COMBO,CB_ADDSTRING,0,(LPARAM)L"CUDA (NVIDIA)");
        SendDlgItemMessage(hwnd,IDC_GPU_COMBO,CB_ADDSTRING,0,(LPARAM)L"DirectML");
        SendDlgItemMessage(hwnd,IDC_GPU_COMBO,CB_ADDSTRING,0,(LPARAM)L"CPU");

        if (m_pFilter) {
            wchar_t info[256]={}, model[MAX_PATH]={};
            m_pFilter->GetGPUInfo(info, 256);
            m_pFilter->GetModelPath(model, MAX_PATH);
            SetDlgItemTextW(hwnd, IDC_GPU_INFO,   info);
            SetDlgItemTextW(hwnd, IDC_MODEL_PATH, model);
        }
        UpdateControls();
        return TRUE;
    }
    case WM_HSCROLL:
    case WM_VSCROLL: {
        int id = GetDlgCtrlID((HWND)lParam);
        if (id==IDC_CONV_SLIDER || id==IDC_SEP_SLIDER) {
            ReadControls();
            wchar_t buf[32];
            if (id==IDC_CONV_SLIDER) swprintf_s(buf,L"%.2f", m_cfg.convergence);
            else                     swprintf_s(buf,L"%.4f", m_cfg.separation);
            SetDlgItemTextW(hwnd,
                id==IDC_CONV_SLIDER ? IDC_CONV_LABEL : IDC_SEP_LABEL, buf);
            m_bDirty = TRUE;
            if (m_pPageSite) {
                m_pPageSite->OnStatusChange(PROPPAGESTATUS_DIRTY);
            };
        }
        break;
    }
    case WM_COMMAND: {
        int ctl = LOWORD(wParam);
        if (ctl==IDC_MODE_COMBO||ctl==IDC_GPU_COMBO||ctl==IDC_FLIP_CHECK) {
            ReadControls(); m_bDirty = TRUE;
            if (m_pPageSite) {
                m_pPageSite->OnStatusChange(PROPPAGESTATUS_DIRTY);
            };
        }
        if (ctl==IDC_RELOAD_BTN && m_pFilter) {
            HRESULT hr = m_pFilter->ReloadModel();
            wchar_t info[256]={};
            m_pFilter->GetGPUInfo(info,256);
            if (FAILED(hr)) wcscpy_s(info,L"Model reload FAILED");
            SetDlgItemTextW(hwnd, IDC_GPU_INFO, info);
        }
        if (ctl==IDC_BROWSE_BTN) {
            OPENFILENAMEW ofn{};
            wchar_t path[MAX_PATH]={};
            ofn.lStructSize=sizeof(ofn);
            ofn.hwndOwner=hwnd;
            ofn.lpstrFilter=L"ONNX Models\0*.onnx\0All Files\0*.*\0";
            ofn.lpstrFile=path;
            ofn.nMaxFile=MAX_PATH;
            ofn.Flags=OFN_FILEMUSTEXIST;
            if (GetOpenFileNameW(&ofn)) {
                SetDlgItemTextW(hwnd,IDC_MODEL_PATH,path);
                if (m_pFilter) m_pFilter->SetModelPath(path);
                m_bDirty = TRUE;
            if (m_pPageSite) {
                m_pPageSite->OnStatusChange(PROPPAGESTATUS_DIRTY);
            };
            }
        }
        break;
    }
    }
    return CBasePropertyPage::OnReceiveMessage(hwnd,msg,wParam,lParam);
}

void C3DeflattenProp::UpdateControls() {
    if (!m_hwnd) return;
    SendDlgItemMessage(m_hwnd,IDC_CONV_SLIDER,TBM_SETPOS,TRUE,
                        (LPARAM)(m_cfg.convergence*CONV_MAX));
    SendDlgItemMessage(m_hwnd,IDC_SEP_SLIDER,TBM_SETPOS,TRUE,
                        (LPARAM)(m_cfg.separation/0.1f*SEP_MAX));
    SendDlgItemMessage(m_hwnd,IDC_MODE_COMBO,CB_SETCURSEL,(int)m_cfg.outputMode,0);
    SendDlgItemMessage(m_hwnd,IDC_GPU_COMBO, CB_SETCURSEL,(int)m_cfg.gpuProvider,0);
    SendDlgItemMessage(m_hwnd,IDC_FLIP_CHECK,BM_SETCHECK,
                        m_cfg.flipDepth?BST_CHECKED:BST_UNCHECKED,0);
    wchar_t buf[32];
    swprintf_s(buf,L"%.2f", m_cfg.convergence);
    SetDlgItemTextW(m_hwnd,IDC_CONV_LABEL,buf);
    swprintf_s(buf,L"%.4f", m_cfg.separation);
    SetDlgItemTextW(m_hwnd,IDC_SEP_LABEL,buf);
}

void C3DeflattenProp::ReadControls() {
    if (!m_hwnd) return;
    m_cfg.convergence = (float)SendDlgItemMessage(
        m_hwnd,IDC_CONV_SLIDER,TBM_GETPOS,0,0) / (float)CONV_MAX;
    m_cfg.separation  = (float)SendDlgItemMessage(
        m_hwnd,IDC_SEP_SLIDER,TBM_GETPOS,0,0) / (float)SEP_MAX * 0.1f;
    m_cfg.outputMode  = (OutputMode)SendDlgItemMessage(
        m_hwnd,IDC_MODE_COMBO,CB_GETCURSEL,0,0);
    m_cfg.gpuProvider = (GPUProvider)SendDlgItemMessage(
        m_hwnd,IDC_GPU_COMBO,CB_GETCURSEL,0,0);
    m_cfg.flipDepth   = (SendDlgItemMessage(
        m_hwnd,IDC_FLIP_CHECK,BM_GETCHECK,0,0)==BST_CHECKED);
}
