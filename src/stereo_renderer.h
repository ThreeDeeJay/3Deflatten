// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten – DirectX 11 stereo compositor
#pragma once
#include <windows.h>
#include <d3d11.h>
#include <wrl/client.h>
#include <vector>
#include "ideflatten.h"
#include "depth_estimator.h"

using Microsoft::WRL::ComPtr;

class StereoRenderer {
public:
    StereoRenderer();
    ~StereoRenderer();

    HRESULT Init(bool forceNoGPU = false);
    bool    IsGPUAvailable() const { return m_gpuOK; }

    // Composite left+right views into an SBS or TAB output frame.
    // Input:  BGRA32 frame (srcW x srcH)
    // Depth:  float [0,1]  (srcW x srcH)
    // Output: BGRA32 buffer
    //   SBS: (2*srcW x srcH)
    //   TAB: (srcW x 2*srcH)
    HRESULT Render(const BYTE*           srcFrame,
                   int                   srcW,
                   int                   srcH,
                   int                   srcStride,
                   const float*          depthMap,
                   const DeflattenConfig& cfg,
                   float                 motionDx,   // source-pixel offset since depth was computed
                   float                 motionDy,
                   BYTE*                 dstFrame,
                   int                   dstStride);

private:
    HRESULT InitGPU();
    HRESULT CreateShaders();
    HRESULT EnsureTextures(int srcW, int srcH, OutputMode mode);

    void RenderGPU(const BYTE* srcFrame, int srcW, int srcH, int srcStride,
                   const float* depthMap, const DeflattenConfig& cfg,
                   float motionDx, float motionDy,
                   BYTE* dstFrame, int dstStride);

    void RenderCPU(const BYTE* srcFrame, int srcW, int srcH, int srcStride,
                   const float* depthMap, const DeflattenConfig& cfg,
                   BYTE* dstFrame, int dstStride);

    // ── DX11 objects ─────────────────────────────────────────────────────────
    ComPtr<ID3D11Device>           m_dev;
    ComPtr<ID3D11DeviceContext>    m_ctx;
    ComPtr<ID3D11VertexShader>     m_vs;
    ComPtr<ID3D11PixelShader>      m_ps;
    ComPtr<ID3D11InputLayout>      m_il;
    ComPtr<ID3D11Buffer>           m_vb;
    ComPtr<ID3D11Buffer>           m_cb;
    ComPtr<ID3D11SamplerState>     m_sampler;
    ComPtr<ID3D11RasterizerState>  m_raster;

    ComPtr<ID3D11Texture2D>           m_srcTex;
    ComPtr<ID3D11ShaderResourceView>  m_srcSRV;
    ComPtr<ID3D11Texture2D>           m_depthTex;
    ComPtr<ID3D11ShaderResourceView>  m_depthSRV;
    ComPtr<ID3D11Texture2D>           m_rtTex;
    ComPtr<ID3D11RenderTargetView>    m_rtv;
    // Triple-buffered staging: frame N submits CopyResource → staging[N%3],
    // reads staging[(N-2)%3] which is guaranteed GPU-complete (2 full frames old).
    // This eliminates all Map(READ) stalls without adding perceptible visual lag.
    ComPtr<ID3D11Texture2D>           m_stagingTex[3];
    int                               m_stagingFrame = 0;

    int        m_lastSrcW = 0, m_lastSrcH = 0;
    OutputMode m_lastMode = OutputMode::SideBySide;

    bool m_gpuOK       = false;
    int  m_renderCount = 0;

    // Constant buffer layout (must match stereo_warp.hlsl cbuffer CBStereo)
    // 48 bytes = 3 × 16-byte D3D11 cbuffer rows.
    struct alignas(16) CBStereo {
        float convergence;
        float separation;
        float flipDepth;      // unused (depth pre-flipped on CPU)
        int   outputMode;     // 0=SBS 1=TAB
        float texelW;         // 1.0f / srcWidth
        float texelH;         // 1.0f / srcHeight
        int   infillMode;     // 0..4
        float depthOffsetU;   // UV offset applied to ALL depth samples (motion compensation)
        // -- row 3 --
        float depthOffsetV;
        float pad0, pad1, pad2;
    };
};
