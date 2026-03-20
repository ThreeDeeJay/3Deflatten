// SPDX-License-Identifier: GPL-3.0-or-later
#include "stereo_renderer.h"
#include "logger.h"
#include <d3dcompiler.h>
#include <dxgi.h>
#include <algorithm>
#include <chrono>
#include <cstring>

#ifdef USE_PRECOMPILED_SHADERS
#include "stereo_warp_vs.h"
#include "stereo_warp_ps.h"
#endif

// ── Embedded HLSL source (used when FXC is not available at build time) ──────
//
// Gap-fill searches run in SOURCE-SPACE starting from srcUV.
// For left eye (eyeSign=+1): srcUV = eyeUV + positive_disparity → srcUV is
// inside the foreground when a gap is detected.
//   +eyeSign from srcUV → through foreground to hidden bg   (Inner)
//   -eyeSign from srcUV → back to visible outer bg           (Outer / EdgeClamp)
static const char* kShaderSrc = R"HLSL(
cbuffer CBStereo : register(b0) {
    float  g_convergence;
    float  g_separation;
    float  g_flipDepth;
    int    g_outputMode;
    float  g_texelW;
    float  g_texelH;
    int    g_infillMode;  // 0=Inner 1=Outer 2=Blend 3=EdgeClamp 4=Inpaint
    float  g_pad;
};
Texture2D<float4> g_srcTex   : register(t0);
Texture2D<float>  g_depthTex : register(t1);
SamplerState      g_sampler  : register(s0);

struct VS_IN  { float2 pos : POSITION; float2 uv : TEXCOORD; };
struct VS_OUT { float4 pos : SV_POSITION; float2 uv : TEXCOORD; };

VS_OUT VS_FullScreen(VS_IN v) {
    VS_OUT o; o.pos = float4(v.pos, 0, 1); o.uv = v.uv; return o;
}

// Walk in source space from 'origin' in 'dir'; return FIRST depth-match sample.
// hitStep: 0 if not found within maxSteps.
float4 SrcSearchFirst(float2 origin, float2 dir, float dC, int maxSteps, out int hitStep) {
    hitStep = 0;
    [loop]
    for (int s = 1; s <= maxSteps; ++s) {
        float2 p = origin + dir * s;
        if (p.x <= 0.0 || p.x >= 1.0) {
            hitStep = s;
            return g_srcTex.SampleLevel(g_sampler, saturate(p), 0);
        }
        float d = g_depthTex.SampleLevel(g_sampler, p, 0).r;
        if (abs(d - dC) < 0.08) { hitStep = s; return g_srcTex.SampleLevel(g_sampler, p, 0); }
    }
    return g_srcTex.SampleLevel(g_sampler, origin, 0);
}

// Walk in source space; keep updating on every depth-match → returns LAST match.
float4 SrcSearchLast(float2 origin, float2 dir, float dC, int maxSteps) {
    float4 result = g_srcTex.SampleLevel(g_sampler, origin, 0);
    [loop]
    for (int s = 1; s <= maxSteps; ++s) {
        float2 p = origin + dir * s;
        if (p.x <= 0.0 || p.x >= 1.0) { return g_srcTex.SampleLevel(g_sampler, saturate(p), 0); }
        float d = g_depthTex.SampleLevel(g_sampler, p, 0).r;
        if (abs(d - dC) < 0.08) result = g_srcTex.SampleLevel(g_sampler, p, 0);
    }
    return result;
}

float4 PS_StereoWarp(VS_OUT i) : SV_TARGET {
    float2 uv = i.uv;
    bool isLeft; float2 eyeUV;
    if (g_outputMode == 0) {
        isLeft = (uv.x < 0.5);
        eyeUV  = float2(isLeft ? uv.x * 2.0 : (uv.x - 0.5) * 2.0, uv.y);
    } else {
        isLeft = (uv.y < 0.5);
        eyeUV  = float2(uv.x, isLeft ? uv.y * 2.0 : (uv.y - 0.5) * 2.0);
    }
    float eyeSign = isLeft ? 1.0 : -1.0;

    float dC = g_depthTex.SampleLevel(g_sampler, eyeUV, 0).r;
    float dL = g_depthTex.SampleLevel(g_sampler, eyeUV + float2(-g_texelW * 3.0, 0), 0).r;
    float dR = g_depthTex.SampleLevel(g_sampler, eyeUV + float2(+g_texelW * 3.0, 0), 0).r;
    float depth = max(dC, max(dL, dR));

    float  disparity = g_separation * (depth - g_convergence);
    float2 srcUV     = saturate(eyeUV + float2(eyeSign * disparity, 0.0));

    float sampledDepth = g_depthTex.SampleLevel(g_sampler, srcUV, 0).r;
    float depthJump    = sampledDepth - dC;

    [branch]
    if (depthJump > 0.10) {
        float  blend   = saturate((depthJump - 0.10) * 10.0);
        float4 rawSmp  = g_srcTex.SampleLevel(g_sampler, srcUV, 0);
        // Source-space step vectors:
        //   innerDir (+eyeSign): through foreground to the hidden background
        //   outerDir (-eyeSign): back to the visible outer background
        float2 innerDir = float2( eyeSign * g_texelW * 2.0, 0);
        float2 outerDir = float2(-eyeSign * g_texelW * 2.0, 0);

        if (g_infillMode == 0) {
            // Inner: hidden background behind near edge (+eyeSign through fg)
            // Fallback = outermost visible bg (not rawSample) for wide gaps
            int hitStep;
            float4 inner = SrcSearchFirst(srcUV, innerDir, dC, 32, hitStep);
            if (hitStep == 0) inner = SrcSearchLast(srcUV, outerDir, dC, 32);
            return lerp(rawSmp, inner, blend);

        } else if (g_infillMode == 1) {
            // Outer: extend visible background from gap's outer edge
            // -eyeSign from srcUV; LAST match = outermost visible bg
            float4 outer = SrcSearchLast(srcUV, outerDir, dC, 32);
            return lerp(rawSmp, outer, blend);

        } else if (g_infillMode == 2) {
            // Blend: confidence-weighted mix of Inner + Outer
            int innerHit;
            float4 inner = SrcSearchFirst(srcUV, innerDir, dC, 32, innerHit);
            float4 outer = SrcSearchLast (srcUV, outerDir, dC, 32);
            float conf = (innerHit > 0) ? saturate(1.0 - (float)innerHit / 32.0) : 0.0;
            return lerp(rawSmp, lerp(outer, inner, conf), blend);

        } else if (g_infillMode == 3) {
            // EdgeClamp (SuperDepth3D-style): FIRST visible outer bg pixel
            // = texture edge-clamp in the warp direction, same effect as
            //   SuperDepth3D "Offset Based" / VM0 Normal infill.
            int hitStep;
            float4 edge = SrcSearchFirst(srcUV, outerDir, dC, 32, hitStep);
            return lerp(rawSmp, edge, blend);

        } else {
            // Inpaint: bilateral-weighted blend from both directions
            // Approximates 3D Photo Inpainting (Shih et al. CVPR 2020):
            // context-aware fill using depth-weighted samples on both sides.
            float4 fillColor   = float4(0, 0, 0, 0);
            float  totalWeight = 0.001;
            [loop]
            for (int s = 1; s <= 32; ++s) {
                float sf = (float)s;
                float2 pIn  = srcUV + innerDir * sf;
                float2 pOut = srcUV + outerDir * sf;
                if (pIn.x > 0.0 && pIn.x < 1.0) {
                    float dIn = g_depthTex.SampleLevel(g_sampler, pIn, 0).r;
                    float wIn = exp(-sf * 0.12) * max(0.0, 1.0 - abs(dIn - dC) * 10.0);
                    fillColor += wIn * g_srcTex.SampleLevel(g_sampler, pIn, 0);
                    totalWeight += wIn;
                }
                if (pOut.x > 0.0 && pOut.x < 1.0) {
                    float dOut = g_depthTex.SampleLevel(g_sampler, pOut, 0).r;
                    float wOut = exp(-sf * 0.12) * max(0.0, 1.0 - abs(dOut - dC) * 10.0);
                    fillColor += wOut * g_srcTex.SampleLevel(g_sampler, pOut, 0);
                    totalWeight += wOut;
                }
            }
            return lerp(rawSmp, fillColor / totalWeight, blend);
        }
    }
    return g_srcTex.SampleLevel(g_sampler, srcUV, 0);
}
)HLSL";

// ── Full-screen quad ──────────────────────────────────────────────────────────
struct Vertex { float x, y, u, v; };
static const Vertex kQuad[] = {
    {-1,+1, 0,0}, {+1,+1, 1,0},
    {-1,-1, 0,1}, {+1,-1, 1,1},
};

StereoRenderer::StereoRenderer()  = default;
StereoRenderer::~StereoRenderer() = default;

HRESULT StereoRenderer::Init(bool forceNoGPU) {
    // Always reset texture dimensions so EnsureTextures() recreates everything
    // on the new device.  Without this, a second Init() (e.g. after seek/pause)
    // would reuse textures from the OLD device with the NEW context → undefined behaviour.
    m_lastSrcW = 0; m_lastSrcH = 0;
    m_stagingFrame = 0;
    if (forceNoGPU) { m_gpuOK = false; return S_OK; }
    HRESULT hr = InitGPU();
    if (FAILED(hr)) {
        LOG_WARN("DX11 init failed (hr=0x", std::hex, (unsigned)hr,
                 std::dec, ") – falling back to CPU compositor");
        m_gpuOK = false;
    }
    return S_OK;
}

HRESULT StereoRenderer::InitGPU() {
    D3D_FEATURE_LEVEL fl;
    D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        0, levels, ARRAYSIZE(levels),
        D3D11_SDK_VERSION, &m_dev, &fl, &m_ctx);
    if (FAILED(hr)) {
        LOG_ERR("D3D11CreateDevice failed hr=0x", std::hex, (unsigned)hr, std::dec);
        return hr;
    }
    LOG_INFO("D3D11 device created  feature_level=0x", std::hex, (unsigned)fl, std::dec);

    // Log GPU adapter name and VRAM for diagnostics (helps explain performance)
    {
        IDXGIDevice* dxgiDev = nullptr;
        if (SUCCEEDED(m_dev->QueryInterface(__uuidof(IDXGIDevice),
                                            reinterpret_cast<void**>(&dxgiDev)))) {
            IDXGIAdapter* adapter = nullptr;
            if (SUCCEEDED(dxgiDev->GetAdapter(&adapter))) {
                DXGI_ADAPTER_DESC desc{};
                if (SUCCEEDED(adapter->GetDesc(&desc))) {
                    size_t vramMB = desc.DedicatedVideoMemory / (1024 * 1024);
                    LOG_INFO("GPU: ", std::wstring(desc.Description),
                             "  VRAM=", vramMB, " MB");
                }
                adapter->Release();
            }
            dxgiDev->Release();
        }
    }

    hr = CreateShaders();
    if (FAILED(hr)) {
        LOG_ERR("CreateShaders failed hr=0x", std::hex, (unsigned)hr, std::dec);
        return hr;
    }
    LOG_INFO("Shaders compiled/loaded OK");

    // Vertex buffer
    D3D11_BUFFER_DESC vbd{};
    vbd.Usage     = D3D11_USAGE_IMMUTABLE;
    vbd.ByteWidth = sizeof(kQuad);
    vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA vd{ kQuad, 0, 0 };
    hr = m_dev->CreateBuffer(&vbd, &vd, &m_vb);
    if (FAILED(hr)) return hr;

    // Constant buffer
    D3D11_BUFFER_DESC cbd{};
    cbd.Usage          = D3D11_USAGE_DYNAMIC;
    cbd.ByteWidth      = sizeof(CBStereo);
    cbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hr = m_dev->CreateBuffer(&cbd, nullptr, &m_cb);
    if (FAILED(hr)) return hr;

    // Sampler
    D3D11_SAMPLER_DESC sd{};
    sd.Filter   = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD   = D3D11_FLOAT32_MAX;
    hr = m_dev->CreateSamplerState(&sd, &m_sampler);
    if (FAILED(hr)) return hr;

    // Rasterizer
    D3D11_RASTERIZER_DESC rd{};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    hr = m_dev->CreateRasterizerState(&rd, &m_raster);
    if (FAILED(hr)) return hr;

    m_gpuOK = true;
    LOG_INFO("StereoRenderer: DX11 GPU init OK");
    return S_OK;
}

HRESULT StereoRenderer::CreateShaders() {
    using Clock = std::chrono::steady_clock;
#ifdef USE_PRECOMPILED_SHADERS
    LOG_INFO("Loading precompiled shaders (FXC offline compilation)...");
    auto t0 = Clock::now();
    HRESULT hr = m_dev->CreateVertexShader(
        g_vsStereoWarp, sizeof(g_vsStereoWarp), nullptr, &m_vs);
    if (FAILED(hr)) return hr;
    hr = m_dev->CreatePixelShader(
        g_psStereoWarp, sizeof(g_psStereoWarp), nullptr, &m_ps);
    if (FAILED(hr)) return hr;
    D3D11_INPUT_ELEMENT_DESC ied[] = {
        {"POSITION",0,DXGI_FORMAT_R32G32_FLOAT,0, 0,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0, 8,D3D11_INPUT_PER_VERTEX_DATA,0},
    };
    hr = m_dev->CreateInputLayout(
        ied, ARRAYSIZE(ied), g_vsStereoWarp, sizeof(g_vsStereoWarp), &m_il);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
    LOG_INFO("Precompiled shaders loaded in ", ms, " ms");
    return hr;
#else
    LOG_INFO("Compiling shaders at runtime via D3DCompile (no precompiled .h found)...");
    LOG_INFO("  This may take a few seconds on first run.");
    auto t0 = Clock::now();
    ComPtr<ID3DBlob> vsBlob, psBlob, err;
    HRESULT hr = D3DCompile(kShaderSrc, strlen(kShaderSrc),
        "stereo_warp.hlsl", nullptr, nullptr,
        "VS_FullScreen", "vs_5_0", 0, 0, &vsBlob, &err);
    if (FAILED(hr)) {
        if (err) LOG_ERR("VS compile: ", (const char*)err->GetBufferPointer());
        return hr;
    }
    hr = D3DCompile(kShaderSrc, strlen(kShaderSrc),
        "stereo_warp.hlsl", nullptr, nullptr,
        "PS_StereoWarp", "ps_5_0", 0, 0, &psBlob, &err);
    if (FAILED(hr)) {
        if (err) LOG_ERR("PS compile: ", (const char*)err->GetBufferPointer());
        return hr;
    }
    hr = m_dev->CreateVertexShader(
        vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_vs);
    if (FAILED(hr)) return hr;
    hr = m_dev->CreatePixelShader(
        psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &m_ps);
    if (FAILED(hr)) return hr;
    D3D11_INPUT_ELEMENT_DESC ied[] = {
        {"POSITION",0,DXGI_FORMAT_R32G32_FLOAT,0, 0,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0, 8,D3D11_INPUT_PER_VERTEX_DATA,0},
    };
    hr = m_dev->CreateInputLayout(
        ied, ARRAYSIZE(ied),
        vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &m_il);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
    LOG_INFO("Shaders compiled in ", ms, " ms");
    return hr;
#endif
}

HRESULT StereoRenderer::EnsureTextures(int srcW, int srcH, OutputMode mode) {
    if (m_lastSrcW==srcW && m_lastSrcH==srcH && m_lastMode==mode) return S_OK;

    int dstW2 = (mode == OutputMode::SideBySide)  ? srcW*2 : srcW;
    int dstH2 = (mode == OutputMode::TopAndBottom) ? srcH*2 : srcH;
    LOG_INFO("EnsureTextures: src=", srcW, "x", srcH,
             " dst=", dstW2, "x", dstH2,
             " mode=", mode==OutputMode::SideBySide?"SBS":"TAB");

    m_srcTex.Reset();   m_srcSRV.Reset();
    m_depthTex.Reset(); m_depthSRV.Reset();
    m_rtTex.Reset();    m_rtv.Reset();
    m_stagingTex[0].Reset();
    m_stagingTex[1].Reset();
    m_stagingTex[2].Reset();
    m_stagingFrame = 0;

    int dstW = (mode == OutputMode::SideBySide)  ? srcW*2 : srcW;
    int dstH = (mode == OutputMode::TopAndBottom) ? srcH*2 : srcH;

    auto makeTex = [&](int w, int h, DXGI_FORMAT fmt, UINT bind,
                        ComPtr<ID3D11Texture2D>& tex) -> HRESULT {
        D3D11_TEXTURE2D_DESC td{};
        td.Width=w; td.Height=h; td.MipLevels=1; td.ArraySize=1;
        td.Format=fmt; td.SampleDesc.Count=1;
        td.Usage=D3D11_USAGE_DEFAULT; td.BindFlags=bind;
        return m_dev->CreateTexture2D(&td, nullptr, &tex);
    };

    HRESULT hr;
    // DYNAMIC textures allow Map(WRITE_DISCARD) for zero-copy CPU→GPU upload,
    // avoiding the internal staging allocation that UpdateSubresource uses.
    auto makeDynTex = [&](int w, int h, DXGI_FORMAT fmt, UINT bind,
                           ComPtr<ID3D11Texture2D>& tex) -> HRESULT {
        D3D11_TEXTURE2D_DESC td{};
        td.Width=w; td.Height=h; td.MipLevels=1; td.ArraySize=1;
        td.Format=fmt; td.SampleDesc.Count=1;
        td.Usage=D3D11_USAGE_DYNAMIC;
        td.BindFlags=bind;
        td.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
        return m_dev->CreateTexture2D(&td, nullptr, &tex);
    };
    auto makeDefaultTex = [&](int w, int h, DXGI_FORMAT fmt, UINT bind,
                               ComPtr<ID3D11Texture2D>& tex) -> HRESULT {
        D3D11_TEXTURE2D_DESC td{};
        td.Width=w; td.Height=h; td.MipLevels=1; td.ArraySize=1;
        td.Format=fmt; td.SampleDesc.Count=1;
        td.Usage=D3D11_USAGE_DEFAULT; td.BindFlags=bind;
        return m_dev->CreateTexture2D(&td, nullptr, &tex);
    };

    hr = makeDynTex(srcW,srcH, DXGI_FORMAT_B8G8R8A8_UNORM,
                    D3D11_BIND_SHADER_RESOURCE, m_srcTex);
    if (FAILED(hr)) return hr;
    hr = m_dev->CreateShaderResourceView(m_srcTex.Get(), nullptr, &m_srcSRV);
    if (FAILED(hr)) return hr;

    hr = makeDynTex(srcW,srcH, DXGI_FORMAT_R32_FLOAT,
                    D3D11_BIND_SHADER_RESOURCE, m_depthTex);
    if (FAILED(hr)) return hr;
    hr = m_dev->CreateShaderResourceView(m_depthTex.Get(), nullptr, &m_depthSRV);
    if (FAILED(hr)) return hr;

    hr = makeDefaultTex(dstW,dstH, DXGI_FORMAT_B8G8R8A8_UNORM,
                        D3D11_BIND_RENDER_TARGET, m_rtTex);
    if (FAILED(hr)) return hr;
    hr = m_dev->CreateRenderTargetView(m_rtTex.Get(), nullptr, &m_rtv);
    if (FAILED(hr)) return hr;

    // Three staging textures for readback — always read staging[(N-2)%3],
    // which is two full frames old and guaranteed GPU-complete.
    D3D11_TEXTURE2D_DESC sd{};
    sd.Width=dstW; sd.Height=dstH; sd.MipLevels=1; sd.ArraySize=1;
    sd.Format=DXGI_FORMAT_B8G8R8A8_UNORM; sd.SampleDesc.Count=1;
    sd.Usage=D3D11_USAGE_STAGING; sd.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    for (int i = 0; i < 3; ++i) {
        hr = m_dev->CreateTexture2D(&sd, nullptr, &m_stagingTex[i]);
        if (FAILED(hr)) return hr;
    }

    m_lastSrcW=srcW; m_lastSrcH=srcH; m_lastMode=mode;
    LOG_DBG("StereoRenderer: textures ", srcW,"x",srcH,
            " -> ", dstW,"x",dstH);
    return S_OK;
}

HRESULT StereoRenderer::Render(const BYTE* srcFrame, int srcW, int srcH,
                                int srcStride,
                                const float* depthMap,
                                const DeflattenConfig& cfg,
                                BYTE* dstFrame, int dstStride) {
    if (m_renderCount == 0)
        LOG_INFO("First Render: src=", srcW, "x", srcH,
                 " path=", m_gpuOK ? "GPU (DX11)" : "CPU (software)",
                 " conv=", cfg.convergence, " sep=", cfg.separation,
                 " mode=", cfg.outputMode==OutputMode::SideBySide?"SBS":"TAB");
    ++m_renderCount;

    if (m_gpuOK)
        RenderGPU(srcFrame,srcW,srcH,srcStride,depthMap,cfg,dstFrame,dstStride);
    else
        RenderCPU(srcFrame,srcW,srcH,srcStride,depthMap,cfg,dstFrame,dstStride);
    return S_OK;
}

void StereoRenderer::RenderGPU(const BYTE* srcFrame, int srcW, int srcH,
                                 int srcStride,
                                 const float* depthMap,
                                 const DeflattenConfig& cfg,
                                 BYTE* dstFrame, int dstStride) {
    HRESULT hrET = EnsureTextures(srcW, srcH, cfg.outputMode);
    if (FAILED(hrET)) {
        LOG_ERR("EnsureTextures failed hr=0x", std::hex, (unsigned)hrET, std::dec,
                " -- CPU fallback");
        RenderCPU(srcFrame,srcW,srcH,srcStride,depthMap,cfg,dstFrame,dstStride);
        return;
    }

    int dstW = (cfg.outputMode==OutputMode::SideBySide)  ? srcW*2 : srcW;
    int dstH = (cfg.outputMode==OutputMode::TopAndBottom) ? srcH*2 : srcH;

    // Upload source frame via Map(WRITE_DISCARD) — avoids internal D3D11 staging alloc
    {
        D3D11_MAPPED_SUBRESOURCE ms{};
        if (SUCCEEDED(m_ctx->Map(m_srcTex.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
            for (int y = 0; y < srcH; ++y)
                memcpy((BYTE*)ms.pData + y * ms.RowPitch,
                       srcFrame + y * srcStride, srcW * 4);
            m_ctx->Unmap(m_srcTex.Get(), 0);
        }
    }
    {
        D3D11_MAPPED_SUBRESOURCE ms{};
        if (SUCCEEDED(m_ctx->Map(m_depthTex.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
            for (int y = 0; y < srcH; ++y)
                memcpy((BYTE*)ms.pData + y * ms.RowPitch,
                       depthMap + y * srcW, srcW * sizeof(float));
            m_ctx->Unmap(m_depthTex.Get(), 0);
        }
    }

    D3D11_MAPPED_SUBRESOURCE mapped;
    if (SUCCEEDED(m_ctx->Map(m_cb.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped))) {
        auto* cb = static_cast<CBStereo*>(mapped.pData);
        cb->convergence = cfg.convergence;
        cb->separation  = cfg.separation;
        cb->flipDepth   = cfg.flipDepth ? 1.f : 0.f;
        cb->outputMode  = (int)cfg.outputMode;
        cb->texelW      = srcW > 0 ? 1.0f / (float)srcW : 0.0f;
        cb->texelH      = srcH > 0 ? 1.0f / (float)srcH : 0.0f;
        cb->infillMode  = (int)cfg.infillMode;
        cb->pad         = 0.0f;
        m_ctx->Unmap(m_cb.Get(), 0);
    }

    D3D11_VIEWPORT vp{0,0,(float)dstW,(float)dstH,0,1};
    m_ctx->RSSetViewports(1,&vp);
    m_ctx->RSSetState(m_raster.Get());
    m_ctx->OMSetRenderTargets(1,m_rtv.GetAddressOf(),nullptr);

    UINT stride=sizeof(Vertex), offset=0;
    m_ctx->IASetInputLayout(m_il.Get());
    m_ctx->IASetVertexBuffers(0,1,m_vb.GetAddressOf(),&stride,&offset);
    m_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    m_ctx->VSSetShader(m_vs.Get(),nullptr,0);
    m_ctx->PSSetShader(m_ps.Get(),nullptr,0);
    m_ctx->PSSetConstantBuffers(0,1,m_cb.GetAddressOf());
    ID3D11ShaderResourceView* srvs[]={m_srcSRV.Get(),m_depthSRV.Get()};
    m_ctx->PSSetShaderResources(0,2,srvs);
    m_ctx->PSSetSamplers(0,1,m_sampler.GetAddressOf());
    m_ctx->Draw(4,0);

    // ── Triple-buffered staging readback — guaranteed zero Map(READ) stall ───
    // Frame N: CopyResource → staging[N%3]      (async, returns immediately)
    //          Map          → staging[(N-2)%3]   (2 frames old — always GPU-complete)
    // The 2-frame readback lag is 2/24 s ≈ 83 ms — imperceptible.
    // First two frames: fall back to mapping the just-submitted buffer (stall once).
    int cur  = m_stagingFrame % 3;
    int read = (m_stagingFrame >= 2) ? (m_stagingFrame - 2) % 3 : cur;
    m_ctx->CopyResource(m_stagingTex[cur].Get(), m_rtTex.Get());

    D3D11_MAPPED_SUBRESOURCE ms;
    if (SUCCEEDED(m_ctx->Map(m_stagingTex[read].Get(), 0, D3D11_MAP_READ, 0, &ms))) {
        for (int y=0; y<dstH; ++y)
            memcpy(dstFrame + y*dstStride,
                   (const BYTE*)ms.pData + y*ms.RowPitch,
                   dstW*4);
        m_ctx->Unmap(m_stagingTex[read].Get(), 0);
    }
    ++m_stagingFrame;
}

void StereoRenderer::RenderCPU(const BYTE* src, int srcW, int srcH,
                                 int srcStride,
                                 const float* depth,
                                 const DeflattenConfig& cfg,
                                 BYTE* dst, int dstStride) {
    const bool isSBS = (cfg.outputMode == OutputMode::SideBySide);

    for (int eye=0; eye<2; ++eye) {
        float eyeSign = (eye==0) ? 1.f : -1.f;
        for (int y=0; y<srcH; ++y) {
            for (int x=0; x<srcW; ++x) {
                float d     = depth[y*srcW+x];
                float disp  = cfg.separation * (d - cfg.convergence);
                float srcXf = (float)x + eyeSign*disp*srcW;
                int   sx0   = std::max(0, std::min((int)srcXf, srcW-1));
                int   sx1   = std::min(sx0+1, srcW-1);
                float tx    = srcXf - (float)sx0;

                const BYTE* p0 = src + y*srcStride + sx0*4;
                const BYTE* p1 = src + y*srcStride + sx1*4;

                BYTE px[4];
                for (int c=0; c<4; ++c)
                    px[c] = (BYTE)((float)p0[c]*(1-tx) + (float)p1[c]*tx);

                int outX, outY;
                if (isSBS) { outX = (eye==0)?x:(srcW+x); outY = y; }
                else        { outX = x; outY = (eye==0)?y:(srcH+y); }

                memcpy(dst + outY*dstStride + outX*4, px, 4);
            }
        }
    }
}
