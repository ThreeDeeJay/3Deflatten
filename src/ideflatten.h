// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten – public COM configuration interface
#pragma once
#include <unknwn.h>

enum class InferenceRuntime : int {
    OnnxRuntime    = 0,  // use ORT with the Provider combo
    TensorRTRtx    = 1,  // native TRT-RTX API via TensorRT-RTX SDK
    TensorRTNative = 2,  // native standard TensorRT API via TensorRT 10.x SDK
};

enum class OutputMode : int {
    SideBySide   = 0,   // full-res SBS  (output width  = 2x input)
    TopAndBottom = 1,   // full-res TAB  (output height = 2x input)
};

// Occlusion-gap infill algorithm applied in the stereo warp shader.
//
// When a pixel is shifted into an area occluded by foreground, the gap is
// filled from a nearby background pixel.  The gap search always runs in
// SOURCE-SPACE starting from srcUV (where the warp landed on foreground).
//
//  Inner   : walk +eyeSign from srcUV through the foreground to the HIDDEN
//             background on the far side of the occluder.
//  Outer   : walk -eyeSign from srcUV to the VISIBLE background on the same
//             side as the gap; take the LAST (outermost) match.
//  Blend   : confidence-weighted mix of Inner + Outer.
//  EdgeClamp: same as Outer but takes the FIRST (nearest) match — reproduces
//             the SuperDepth3D edge-clamp behaviour.
//  Inpaint : bilateral-weighted blend from both sides, depth-guided —
//             real-time approximation of 3D Photo Inpainting (Shih et al.)
enum class InfillMode : int {
    Inner     = 0,
    Outer     = 1,
    Blend     = 2,
    EdgeClamp = 3,   // SuperDepth3D-style: nearest outer-edge pixel
    Inpaint   = 4,   // 3D Photo Inpainting approx: bilateral depth-guided blend
};

enum class GPUProvider : int {
    Auto         = 0,   // best available: TRT-RTX -> TensorRT -> CUDA -> DirectML -> CPU
    TensorRT     = 1,   // NVIDIA TensorRT  (fast;    requires CUDA + TRT SDK at runtime)
    CUDA         = 2,   // NVIDIA CUDA      (fast;     requires CUDA at runtime)
    DirectML     = 3,   // DX12 / DirectML  (good;     any modern GPU, Windows 10 1903+)
    CPU          = 4,   // CPU only         (slow;     always works)
    TensorRTRtx  = 5,   // NVIDIA TRT-RTX EP (fastest; requires ORT built with --use_nv_tensorrt_rtx)
};

struct DeflattenConfig {
    float       convergence;   // [0,1]   depth plane at screen depth (default 0.25)
    float       separation;    // [0,0.1] stereo strength             (default 0.05)
    OutputMode  outputMode;    // SBS or TAB
    GPUProvider gpuProvider;   // inference EP
    float       depthSmooth;   // [0,1]   temporal smoothing alpha    (default 0.0)
    BOOL        flipDepth;     // invert depth map polarity
    InfillMode  infillMode;    // occlusion gap fill algorithm        (default Outer)
    BOOL        showDepth;     // overlay depth map on both views (toggleable via hotkey)
    int         depthViewKey;  // VK code to toggle showDepth (default VK_RSHIFT = 161)
    InferenceRuntime inferenceRuntime; // OnnxRuntime or TensorRTRtx native
    int         depthMaxDim;   // max depth tensor side (0=auto 1022 for dynamic, ignored for fixed)
    int         meshDiv;       // mesh vertex grid divisor: 1=full 2=half(default) 4=quarter
    int         depthDilate;   // foreground edge dilation radius in pixels (0=off, default 4)
    float       depthEdgeThresh; // depth discontinuity threshold for dilation [0,1] (default 0.20)
    BOOL        depthJBU;      // joint bilateral upscaling using RGB guide (default FALSE)
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
