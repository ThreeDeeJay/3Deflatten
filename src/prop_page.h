// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <streams.h>
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

    // Push current m_cfg + m_modelPath to the filter immediately.
    // Called on every control interaction so changes are live.
    void PushToFilter();
    void SetDirty();        // marks page dirty and notifies host

    // Populate all controls from m_cfg / m_modelPath.
    void PopulateControls();

    // Read all controls back into m_cfg / m_modelPath.
    void ReadControls();

    // Update just the value labels next to sliders.
    void UpdateValueLabels();

    // Update the status line from the filter.
    void RefreshStatus();

    I3Deflatten*    m_pFilter    = nullptr;
    DeflattenConfig m_cfg{};
    wchar_t         m_modelPath[MAX_PATH]{};
};
