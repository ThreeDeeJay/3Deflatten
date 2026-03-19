// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten – public COM configuration interface
#pragma once
#include <unknwn.h>

enum class OutputMode : int {
    SideBySide   = 0,   // full-res SBS  (output width  = 2x input)
    TopAndBottom = 1,   // full-res TAB  (output height = 2x input)
};

// Occlusion-gap infill algorithm applied in the stereo warp shader.
// When a shifted pixel lands on a foreground area (occlusion gap), the gap
// must be filled.  Three strategies:
//   Inner: search BEHIND the foreground edge (background just inside the gap)
//   Outer: search BEYOND the foreground edge (extends the far-edge pixel outward)
//   Blend: confidence-weighted mix of inner + outer for smoother seams
enum class InfillMode : int {
    Inner = 0,   // walk inward toward background behind the occluding edge
    Outer = 1,   // walk outward, smear the far-edge pixel into the gap
    Blend = 2,   // blend inner bg-search + outer smear by search confidence
};

enum class GPUProvider : int {
    Auto      = 0,   // best available: TensorRT -> CUDA -> DirectML -> CPU
    TensorRT  = 1,   // NVIDIA TensorRT  (fastest;  requires CUDA + TRT SDK at runtime)
    CUDA      = 2,   // NVIDIA CUDA      (fast;     requires CUDA at runtime)
    DirectML  = 3,   // DX12 / DirectML  (good;     any modern GPU, Windows 10 1903+)
    CPU       = 4,   // CPU only         (slow;     always works)
};

struct DeflattenConfig {
    float       convergence;   // [0,1]   depth plane at screen depth (default 0.5)
    float       separation;    // [0,0.1] stereo strength             (default 0.03)
    OutputMode  outputMode;    // SBS or TAB
    GPUProvider gpuProvider;   // inference EP
    float       depthSmooth;   // [0,1]   temporal smoothing alpha    (default 0.4)
    BOOL        flipDepth;     // invert depth map polarity
    InfillMode  infillMode;    // occlusion gap fill algorithm        (default Inner)
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
