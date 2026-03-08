// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <streams.h>
#include <vector>
#include <string>
#include "ideflatten.h"
#include "guids.h"
#include "resource.h"

class C3DeflattenProp : public CBasePropertyPage {
public:
    static CUnknown* WINAPI CreateInstance(LPUNKNOWN pUnk, HRESULT* phr);

    HRESULT OnConnect(IUnknown* pUnk)  override;
    HRESULT OnDisconnect()             override;
    HRESULT OnApplyChanges()           override;
    INT_PTR OnReceiveMessage(HWND hwnd, UINT msg,
                              WPARAM wParam, LPARAM lParam) override;

private:
    C3DeflattenProp(LPUNKNOWN pUnk, HRESULT* phr);

    void PopulateControls(HWND hwnd);
    void ReadControls(HWND hwnd);
    void UpdateValueLabels(HWND hwnd);
    void RefreshStatus(HWND hwnd);
    void PopulateModelCombo(HWND hwnd);
    void SetDirty();

    // Push current m_cfg to the live filter immediately.
    void PushConfig();

    static std::wstring GetDllDir();

    I3Deflatten*               m_pFilter = nullptr;
    DeflattenConfig            m_cfg{};
    wchar_t                    m_modelPath[MAX_PATH]{};
    std::vector<std::wstring>  m_onnxFiles;  // full paths, indexed by combo pos
};
