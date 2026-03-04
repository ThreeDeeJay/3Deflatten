// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten – public COM configuration interface
#pragma once
#include <unknwn.h>

enum class OutputMode : int {
    SideBySide   = 0,   // full-res SBS  (output width  = 2x input)
    TopAndBottom = 1,   // full-res TAB  (output height = 2x input)
};

enum class GPUProvider : int {
    Auto      = 0,   // try CUDA -> DirectML -> CPU
    CUDA      = 1,
    DirectML  = 2,
    CPU       = 3,
};

struct DeflattenConfig {
    float       convergence;   // [0,1]   depth plane at screen depth (default 0.5)
    float       separation;    // [0,0.1] stereo strength             (default 0.03)
    OutputMode  outputMode;    // SBS or TAB
    GPUProvider gpuProvider;   // inference EP
    float       depthSmooth;   // [0,1]   temporal smoothing alpha    (default 0.4)
    BOOL        flipDepth;     // invert depth map polarity
};

MIDL_INTERFACE("4D455F32-1A2B-4C3D-8E4F-5A6B7C8D9E0F")
I3Deflatten : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE GetConfig(DeflattenConfig* pCfg)       = 0;
    virtual HRESULT STDMETHODCALLTYPE SetConfig(const DeflattenConfig* pCfg) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetModelPath(LPWSTR buf, UINT cch)     = 0;
    virtual HRESULT STDMETHODCALLTYPE SetModelPath(LPCWSTR path)             = 0;
    virtual HRESULT STDMETHODCALLTYPE GetGPUInfo(LPWSTR buf, UINT cch)       = 0;
    virtual HRESULT STDMETHODCALLTYPE ReloadModel()                          = 0;
};
