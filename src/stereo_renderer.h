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
    HRESULT EnsureTextures(int srcW, int srcH, OutputMode mode, int meshDiv);

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

    int        m_lastSrcW    = 0, m_lastSrcH = 0;
    OutputMode m_lastMode    = OutputMode::SideBySide;
    int        m_lastMeshDiv = 0;  // track meshDiv so EnsureTextures rebuilds on change

    bool m_gpuOK       = false;
    int  m_renderCount = 0;

    // Constant buffer layout — 48 bytes = 3 × 16-byte rows.
    // MUST match the cbuffer CBStereo in all embedded HLSL.
    struct alignas(16) CBStereo {
        float convergence;
        float separation;
        float flipDepth;      // unused (depth pre-flipped on CPU)
        int   outputMode;     // 0=SBS 1=TAB
        // row 2
        float texelW;         // 1.0f / srcWidth
        float texelH;         // 1.0f / srcHeight
        int   infillMode;     // 0..4
        float depthOffsetU;   // motion-comp UV offset (pixels/srcW)
        // row 3
        float depthOffsetV;   // motion-comp UV offset (pixels/srcH)
        float discThresh;     // depth jump > this → cut mesh edge (default 0.05)
        float eyeSign;        // +1=left/top eye, −1=right/bottom (mesh pass)
        float pad1;
    };

    // ── Mesh reprojection + GS culling + UV-warp hole-fill (two-pass) ────────
    // Pass 1 — mesh + GS:
    //   MeshVS computes per-vertex disparity-shifted screen position.
    //   MeshGS (geometry shader) inspects each triangle's three depth values;
    //   if the max pairwise difference exceeds g_discThresh the triangle spans
    //   a foreground/background boundary and is culled — emitting nothing.
    //   The gap is left at the DSV's cleared depth (1.0) and RTV's cleared
    //   colour (black), ready for pass 2.
    // Pass 2 — UV-warp hole-fill:
    //   A full-screen quad with depth-test EQUAL 1.0 (m_dsStateHoleFill, no
    //   depth write).  PS only runs for pixels still at depth 1.0 (gaps the
    //   GS cut).  PS_StereoWarp with the selected g_infillMode searches for
    //   the correct background colour in source-space and writes it, filling
    //   each hole exactly once with no overdraw on covered pixels.
    // Reference: Shih et al. CVPR 2020 §3 for the mesh + edge-cutting concept.
    ComPtr<ID3D11VertexShader>      m_meshVS;
    ComPtr<ID3D11GeometryShader>    m_meshGS;   // culls edge-straddling triangles
    ComPtr<ID3D11PixelShader>       m_meshPS;
    ComPtr<ID3D11InputLayout>       m_meshIL;
    ComPtr<ID3D11Buffer>            m_meshVB;   // float2 uv per vertex, meshW*meshH verts
    ComPtr<ID3D11Buffer>            m_meshIB;   // uint32 indices, (meshW-1)*(meshH-1)*6
    ComPtr<ID3D11Texture2D>         m_dsTex;    // depth-stencil for mesh z-test
    ComPtr<ID3D11DepthStencilView>  m_dsv;
    ComPtr<ID3D11DepthStencilState> m_dsState;         // LESS_EQUAL, depth write (mesh pass)
    ComPtr<ID3D11DepthStencilState> m_dsStateHoleFill; // EQUAL, no write (hole-fill pass)
    ComPtr<ID3D11RasterizerState>   m_meshRaster;      // no backface cull
    int                             m_meshW = 0, m_meshH = 0;
};
