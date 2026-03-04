// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten – Win32 property page
#pragma once
#include <streams.h>
#include "ideflatten.h"
#include "guids.h"
#include "resource.h"

class C3DeflattenProp : public CBasePropertyPage {
public:
    static CUnknown* WINAPI CreateInstance(LPUNKNOWN pUnk, HRESULT* phr);

    HRESULT OnConnect(IUnknown* pUnk) override;
    HRESULT OnDisconnect()            override;
    HRESULT OnApplyChanges()          override;
    INT_PTR OnReceiveMessage(HWND hwnd, UINT msg,
                              WPARAM wParam, LPARAM lParam) override;

private:
    C3DeflattenProp(LPUNKNOWN pUnk, HRESULT* phr);
    void UpdateControls();
    void ReadControls();

    I3Deflatten*    m_pFilter = nullptr;
    DeflattenConfig m_cfg{};
};
