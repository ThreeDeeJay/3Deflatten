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
    float  g_depthOffsetU;
    // row 3
    float  g_depthOffsetV;
    float  g_discThresh;  // used by both the mesh GS edge-cull AND the warp
                           // pass's background-search (see PS_StereoWarp)
    float  g_eyeSign;     // unused in warp pass
    float  g_pad1;
};
Texture2D<float4> g_srcTex   : register(t0);
Texture2D<float>  g_depthTex : register(t1);
SamplerState      g_sampler  : register(s0);

struct VS_IN  { float2 pos : POSITION; float2 uv : TEXCOORD; };
struct VS_OUT { float4 pos : SV_POSITION; float2 uv : TEXCOORD; };

VS_OUT VS_FullScreen(VS_IN v) {
    VS_OUT o;
    // z = w = 1.0 → NDC depth = 1.0 (farthest).  After the mesh renders with
    // LESS depth, UV-warp uses EQUAL 1.0 to fill only uncovered holes.
    o.pos = float4(v.pos, 1, 1);
    o.uv = v.uv;
    return o;
}

// Depth UV: shift eyeUV back to where the pixel was when the depth was computed.
// If the scene/camera moved by (motionDx, motionDy) source pixels since inference,
// the depth for current pixel (u,v) lives at (u - offsetU, v - offsetV) in the
// stored depth map.
float2 DepthUV(float2 eyeUV) {
    return saturate(eyeUV - float2(g_depthOffsetU, g_depthOffsetV));
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

    // ── Find the true background reference ──────────────────────────────────
    // This PS now ONLY runs on pixels the mesh pass left at depth==1.0 (gaps
    // cut by MeshGS — see the EQUAL-depth hole-fill pass in RenderGPU).
    // Sampling depth directly at eyeUV (the old model, designed for a
    // standalone per-pixel backward-warp with no mesh) is wrong here: a gap
    // is many texels wide and its output positions can numerically coincide
    // with the FOREGROUND's own source extent, making "depth at this pixel"
    // read as foreground instead of the background that's actually supposed
    // to show through. That produced a fill with no real disparity — flat,
    // "stuck to the screen" looking.
    //
    // Fix: walk in the established "outer" direction (-eyeSign, away from
    // where the foreground shifted to) from eyeUV, tracking the highest
    // depth seen so far (the foreground's peak as we cross it). The moment
    // depth falls more than g_discThresh below that running peak, we've
    // stepped past the foreground's edge into real background — use THAT
    // position's depth as the reference for the infill searches below,
    // instead of the unreliable eyeUV sample.
    float2 bgSearchDir = float2(-eyeSign * g_texelW, 0.0);
    float  runningPeak = g_depthTex.SampleLevel(g_sampler, DepthUV(eyeUV), 0).r;
    float2 srcUV  = eyeUV;
    float  depth  = runningPeak;
    [loop]
    for (int s = 1; s <= 96; ++s) {
        float2 p = eyeUV + bgSearchDir * s;
        if (p.x <= 0.0 || p.x >= 1.0) { srcUV = saturate(p); depth = runningPeak; break; }
        float d = g_depthTex.SampleLevel(g_sampler, p, 0).r;
        runningPeak = max(runningPeak, d);
        if (runningPeak - d > g_discThresh) { srcUV = p; depth = d; break; }
    }

    // Source-space step vectors for the infill searches below:
    //   innerDir (+eyeSign): back toward the foreground
    //   outerDir (-eyeSign): further into the background
    float2 innerDir = float2( eyeSign * g_texelW * 2.0, 0);
    float2 outerDir = float2(-eyeSign * g_texelW * 2.0, 0);

    if (g_infillMode == 0) {
        // Inner: hidden background behind near edge (+eyeSign through fg)
        // Fallback = outermost visible bg for wide gaps
        int hitStep;
        float4 inner = SrcSearchFirst(srcUV, innerDir, depth, 32, hitStep);
        if (hitStep == 0) inner = SrcSearchLast(srcUV, outerDir, depth, 32);
        return inner;

    } else if (g_infillMode == 1) {
        // Outer: extend visible background from gap's outer edge
        return SrcSearchLast(srcUV, outerDir, depth, 32);

    } else if (g_infillMode == 2) {
        // Blend: confidence-weighted mix of Inner + Outer
        int innerHit;
        float4 inner = SrcSearchFirst(srcUV, innerDir, depth, 32, innerHit);
        float4 outer = SrcSearchLast (srcUV, outerDir, depth, 32);
        float conf = (innerHit > 0) ? saturate(1.0 - (float)innerHit / 32.0) : 0.0;
        return lerp(outer, inner, conf);

    } else if (g_infillMode == 3) {
        // EdgeClamp: FIRST visible outer bg pixel
        int hitStep;
        return SrcSearchFirst(srcUV, outerDir, depth, 32, hitStep);

    } else {
        // Inpaint: bilateral-weighted blend from both directions
        float4 fillColor   = float4(0, 0, 0, 0);
        float  totalWeight = 0.001;
        [loop]
        for (int s = 1; s <= 32; ++s) {
            float sf = (float)s;
            float2 pIn  = srcUV + innerDir * sf;
            float2 pOut = srcUV + outerDir * sf;
            if (pIn.x > 0.0 && pIn.x < 1.0) {
                float dIn = g_depthTex.SampleLevel(g_sampler, pIn, 0).r;
                float wIn = exp(-sf * 0.12) * max(0.0, 1.0 - abs(dIn - depth) * 10.0);
                fillColor += wIn * g_srcTex.SampleLevel(g_sampler, pIn, 0);
                totalWeight += wIn;
            }
            if (pOut.x > 0.0 && pOut.x < 1.0) {
                float dOut = g_depthTex.SampleLevel(g_sampler, pOut, 0).r;
                float wOut = exp(-sf * 0.12) * max(0.0, 1.0 - abs(dOut - depth) * 10.0);
                fillColor += wOut * g_srcTex.SampleLevel(g_sampler, pOut, 0);
                totalWeight += wOut;
            }
        }
        return fillColor / totalWeight;
    }
}
)HLSL";

// ── Mesh vertex + geometry shaders ───────────────────────────────────────────
// MeshVS: each vertex carries its source UV (u,v).  The VS reads depth at
// that UV, computes a horizontal disparity shift, and places the vertex at
// the correct shifted screen position for the current eye.  The PS samples
// srcTex at the original UV so RGB always exactly matches the mesh geometry.
//
// MeshGS: geometry shader that culls triangles whose three vertices span a
// depth discontinuity (max pairwise depth difference > g_discThresh).
// Culled triangles leave their pixels at the DSV-cleared depth (1.0) and
// RTV-cleared colour (black); pass 2 (UV-warp hole-fill) then fills those
// gaps using the selected infill algorithm.
static const char* kMeshVSSrc = R"HLSL(
cbuffer CBStereo : register(b0) {
    float  g_convergence;
    float  g_separation;
    float  g_flipDepth;
    int    g_outputMode;
    float  g_texelW;
    float  g_texelH;
    int    g_infillMode;
    float  g_depthOffsetU;
    float  g_depthOffsetV;
    float  g_discThresh;
    float  g_eyeSign;
    float  g_pad1;
};
Texture2D<float> g_depthTex : register(t1);
SamplerState     g_sampler  : register(s0);

struct MeshOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
    float  smoothDepth : TEXCOORD1;   // spatially-averaged — see MeshGS comment
    float  skirtBlend  : TEXCOORD2;   // 0=real mesh; 1=skirt corner (inpaint in PS)
};

MeshOut MeshVS(float2 uv : TEXCOORD) {
    MeshOut o;

    // Read depth with motion compensation
    float2 depUV = saturate(uv - float2(g_depthOffsetU, g_depthOffsetV));
    float  depth = g_depthTex.SampleLevel(g_sampler, depUV, 0).r;

    // Spatially-averaged depth used ONLY by MeshGS's cut decision below —
    // NOT for disparity/geometry, which still uses the single-sample
    // `depth` for full edge sharpness. Depth models can be noisy/unstable
    // in flat or texture-less regions (sky, blown-out highlights), and a
    // single pair of adjacent vertices can exceed g_discThresh by chance no
    // matter how high the threshold is set. Averaging a 3x3 neighbourhood
    // (spaced a few texels apart to also cover ViT-patch-scale artifacts,
    // not just single-pixel dither) suppresses that noise before the
    // comparison; genuine object-boundary jumps are much larger and
    // consistent across the neighbourhood, so they survive averaging intact.
    float smoothDepth = 0.0;
    [unroll]
    for (int sy = -1; sy <= 1; ++sy) {
        [unroll]
        for (int sx = -1; sx <= 1; ++sx) {
            float2 p = saturate(depUV + float2(sx * g_texelW * 4.0, sy * g_texelH * 4.0));
            smoothDepth += g_depthTex.SampleLevel(g_sampler, p, 0).r;
        }
    }
    smoothDepth *= (1.0 / 9.0);

    // Horizontal disparity: positive = near (shift outward from screen centre)
    float disparity = g_separation * (depth - g_convergence);

    // Shifted screen X for this eye.
    // eyeSign +1 = left eye: foreground shifts right (+) relative to background
    // eyeSign -1 = right eye: foreground shifts left  (-)
    float eyeX = uv.x + g_eyeSign * disparity;

    // Map (eyeX, uv.y) → NDC in the correct half of the SBS / TAB output.
    float ndcX, ndcY;
    if (g_outputMode == 0) {
        // SBS: left eye  → NDC x ∈ [-1, 0]  (eyeSign +1)
        //      right eye → NDC x ∈ [ 0,+1]  (eyeSign -1)
        float halfOfs = (g_eyeSign > 0.0) ? -1.0 : 0.0;
        ndcX = eyeX * 1.0 + halfOfs;
        ndcY = 1.0 - 2.0 * uv.y;
    } else {
        // TAB: top eye    → NDC y ∈ [ 1, 0]  (eyeSign +1)
        //      bottom eye → NDC y ∈ [ 0,-1]  (eyeSign -1)
        float halfOfs = (g_eyeSign > 0.0) ? 0.0 : -1.0;
        ndcX = 2.0 * eyeX - 1.0;
        ndcY = (1.0 - uv.y) + halfOfs;
    }

    // NDC Z: 1-depth so near (depth=1) → z=0 wins LESS z-test over far (depth=0) → z=1
    o.pos = float4(ndcX, ndcY, 1.0 - depth, 1.0);
    o.uv  = uv;   // UNCHANGED — PS always samples the original source pixel
    o.smoothDepth = smoothDepth;
    o.skirtBlend  = 0.0;
    return o;
}

// ── Mesh Geometry Shader — depth-discontinuity triangle culling ───────────
// Each triangle from MeshVS is inspected here before rasterization.
// If any pair of the three vertices has a SMOOTHED depth difference
// exceeding g_discThresh, the triangle straddles a foreground/background
// boundary (a silhouette edge). Emitting nothing for such triangles creates
// a clean gap at the edge instead of a stretched/smeared polygon. The gap
// pixels retain the DSV-cleared depth 1.0 and are filled in pass 2 by
// PS_StereoWarp (UV-warp hole-fill) using the selected infill algorithm.
// Triangles that do NOT straddle a discontinuity are passed through as-is.
// Uses smoothDepth (MeshVS's 3x3-averaged sample), not the raw per-vertex
// depth, so isolated noisy depth estimates in flat regions (sky, etc.)
// don't trigger spurious cuts regardless of how g_discThresh is set.
[maxvertexcount(3)]
void MeshGS(triangle MeshOut v[3], inout TriangleStream<MeshOut> stream) {
    float d0=v[0].smoothDepth, d1=v[1].smoothDepth, d2=v[2].smoothDepth;
    float maxDiff = max(max(abs(d0-d1),abs(d1-d2)),abs(d0-d2));
    if (maxDiff <= g_discThresh) {
        // No discontinuity — pass through as normal mesh geometry.
        stream.Append(v[0]); stream.Append(v[1]); stream.Append(v[2]);
        stream.RestartStrip(); return;
    }
    // Build a background-extension skirt instead of discarding the triangle.
    // Anchor = vertex with the lowest smoothDepth (background/far side).
    // The other two keep their screen XY (the silhouette edge positions) but
    // inherit the anchor's depth/Z/UV — fills the disocclusion gap by
    // construction with zero residual. skirtBlend=1 at relocated corners so
    // MeshPS runs an inpaint search there instead of a straight sample.
    int bgIdx = (d0<=d1) ? ((d0<=d2)?0:2) : ((d1<=d2)?1:2);
    int o1=(bgIdx+1)%3, o2=(bgIdx+2)%3;
    MeshOut anchor=v[bgIdx]; anchor.skirtBlend=0.0;
    MeshOut ext1=v[bgIdx];   ext1.pos.xy=v[o1].pos.xy; ext1.skirtBlend=1.0;
    MeshOut ext2=v[bgIdx];   ext2.pos.xy=v[o2].pos.xy; ext2.skirtBlend=1.0;
    stream.Append(anchor); stream.Append(ext1); stream.Append(ext2);
    stream.RestartStrip();
}
)HLSL";

// Mesh PS: sample source colour at the original vertex UV.
// Since the VS placed the geometry at the correct shifted screen position
// and uv was not modified, the RGB texture is always perfectly aligned with
// the mesh geometry regardless of separation or depth value.
static const char* kMeshPSSrc = R"HLSL(
cbuffer CBStereo : register(b0) {
    float g_convergence; float g_separation; float g_flipDepth; int g_outputMode;
    float g_texelW; float g_texelH; int g_infillMode; float g_depthOffsetU;
    float g_depthOffsetV; float g_discThresh; float g_eyeSign; float g_pad1;
};
Texture2D<float4> g_srcTex   : register(t0);
Texture2D<float>  g_depthTex : register(t1);
SamplerState      g_sampler  : register(s0);

float4 MeshPS(float4 pos : SV_Position, float2 uv : TEXCOORD0,
              float smoothDepth : TEXCOORD1, float skirtBlend : TEXCOORD2) : SV_TARGET {
    float4 base = g_srcTex.SampleLevel(g_sampler, uv, 0);
    if (skirtBlend < 0.01) return base;
    float refD = g_depthTex.SampleLevel(g_sampler, uv, 0).r;
    float4 fill = float4(0,0,0,0); float wT = 0.001;
    [loop] for (int s=1; s<=24; ++s) {
        float sf=(float)s;
        float2 p1=uv+float2(g_texelW*sf,0), p2=uv-float2(g_texelW*sf,0);
        if (p1.x>0&&p1.x<1) { float d=g_depthTex.SampleLevel(g_sampler,p1,0).r;
            float w=exp(-sf*0.12)*max(0.0,1.0-abs(d-refD)*10.0);
            fill+=w*g_srcTex.SampleLevel(g_sampler,p1,0); wT+=w; }
        if (p2.x>0&&p2.x<1) { float d=g_depthTex.SampleLevel(g_sampler,p2,0).r;
            float w=exp(-sf*0.12)*max(0.0,1.0-abs(d-refD)*10.0);
            fill+=w*g_srcTex.SampleLevel(g_sampler,p2,0); wT+=w; }
    }
    return lerp(base, fill/wT, skirtBlend);
}
)HLSL";
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

    // ── Compile mesh shaders ──────────────────────────────────────────────────
    {
        ComPtr<ID3DBlob> vsBlob2, psBlob2, err2;
        hr = D3DCompile(kMeshVSSrc, strlen(kMeshVSSrc),
            "mesh_vs.hlsl", nullptr, nullptr,
            "MeshVS", "vs_5_0", 0, 0, &vsBlob2, &err2);
        if (FAILED(hr)) {
            if (err2) LOG_ERR("MeshVS compile: ", (const char*)err2->GetBufferPointer());
            LOG_WARN("Mesh VS failed to compile — mesh pass disabled.");
        } else {
            hr = D3DCompile(kMeshPSSrc, strlen(kMeshPSSrc),
                "mesh_ps.hlsl", nullptr, nullptr,
                "MeshPS", "ps_5_0", 0, 0, &psBlob2, &err2);
            if (FAILED(hr)) {
                if (err2) LOG_ERR("MeshPS compile: ", (const char*)err2->GetBufferPointer());
            } else {
                m_dev->CreateVertexShader(
                    vsBlob2->GetBufferPointer(), vsBlob2->GetBufferSize(), nullptr, &m_meshVS);
                m_dev->CreatePixelShader(
                    psBlob2->GetBufferPointer(), psBlob2->GetBufferSize(), nullptr, &m_meshPS);
                // Mesh input: float2 uv only (position computed in VS from depth texture)
                D3D11_INPUT_ELEMENT_DESC mied[] = {
                    {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
                };
                m_dev->CreateInputLayout(
                    mied, 1,
                    vsBlob2->GetBufferPointer(), vsBlob2->GetBufferSize(), &m_meshIL);

                // Compile GS from the same source string (MeshGS entry point).
                // The GS culls triangles spanning depth discontinuities; without
                // it the mesh stretches at silhouette edges and infill has no gap
                // to fill.  Non-fatal if it fails — log but continue.
                ComPtr<ID3DBlob> gsBlob, err3;
                HRESULT hrGS = D3DCompile(kMeshVSSrc, strlen(kMeshVSSrc),
                    "mesh_gs.hlsl", nullptr, nullptr,
                    "MeshGS", "gs_5_0", 0, 0, &gsBlob, &err3);
                if (FAILED(hrGS)) {
                    if (err3) LOG_ERR("MeshGS compile: ", (const char*)err3->GetBufferPointer());
                    LOG_WARN("Mesh GS failed to compile — edge culling + infill disabled.");
                } else {
                    m_dev->CreateGeometryShader(
                        gsBlob->GetBufferPointer(), gsBlob->GetBufferSize(), nullptr, &m_meshGS);
                    LOG_INFO("Mesh shaders compiled OK (VS + GS + PS)");
                }
            }
        }
    }

    // Depth-stencil state: LESS test with write — foreground occludes background
    {
        D3D11_DEPTH_STENCIL_DESC dsd{};
        dsd.DepthEnable    = TRUE;
        dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        dsd.DepthFunc      = D3D11_COMPARISON_LESS_EQUAL;
        m_dev->CreateDepthStencilState(&dsd, &m_dsState);
    }

    // Hole-fill DS state: GREATER test (against the CLEAR value, see below),
    // no write — UV-warp only draws pixels the mesh left completely
    // untouched.
    //
    // Why GREATER instead of EQUAL: mesh z = 1.0 - depth, and depth is
    // clamped to [0,1], so the FARTHEST legitimate geometry (depth -> 0)
    // writes z -> 1.0 — the SAME value the depth buffer is cleared to. An
    // EQUAL-vs-1.0 test cannot tell "legitimately-rendered far/sky mesh
    // triangle" apart from "an actual gap": both read back as exactly 1.0,
    // so the old EQUAL test incorrectly re-filled correctly-rendered far
    // geometry too — the "far depth artifacts".
    //
    // Fix: clear the depth buffer to 2.0 instead of 1.0 (clearly outside the
    // viewport's [0,1] depth range, so no rasterized fragment can ever
    // legitimately equal or exceed it — see RenderGPU's ClearDepthStencilView
    // call). GREATER-vs-the-buffer then unambiguously distinguishes "still
    // at the impossible clear value" (a true gap) from "covered by ANY mesh
    // triangle, including ones at the farthest depth" (buffer <= 1.0,
    // clamped by the viewport, so never > a 1.0-z hole-fill fragment... wait,
    // see below: the comparison is buffer > incoming, with incoming fixed at
    // 1.0 from VS_FullScreen, so buffer must be the clear value 2.0 to pass).
    {
        D3D11_DEPTH_STENCIL_DESC dsd{};
        dsd.DepthEnable    = TRUE;
        dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        dsd.DepthFunc      = D3D11_COMPARISON_GREATER;
        m_dev->CreateDepthStencilState(&dsd, &m_dsStateHoleFill);
    }

    // Mesh rasterizer: no backface culling (triangle winding varies with depth)
    {
        D3D11_RASTERIZER_DESC mrd{};
        mrd.FillMode        = D3D11_FILL_SOLID;
        mrd.CullMode        = D3D11_CULL_NONE;
        mrd.DepthClipEnable = TRUE;
        m_dev->CreateRasterizerState(&mrd, &m_meshRaster);
    }

    return S_OK;
#endif
}

HRESULT StereoRenderer::EnsureTextures(int srcW, int srcH, OutputMode mode, int meshDiv) {
    if (m_lastSrcW==srcW && m_lastSrcH==srcH && m_lastMode==mode && m_lastMeshDiv==meshDiv)
        return S_OK;

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
    // Reset mesh resources so they are rebuilt at the new resolution
    m_meshVB.Reset(); m_meshIB.Reset();
    m_dsTex.Reset();  m_dsv.Reset();
    m_meshW = 0; m_meshH = 0;

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

    m_lastSrcW=srcW; m_lastSrcH=srcH; m_lastMode=mode; m_lastMeshDiv=meshDiv;
    LOG_DBG("StereoRenderer: textures ", srcW,"x",srcH,
            " -> ", dstW,"x",dstH);

    // ── Mesh VB / IB at half source resolution ────────────────────────────────
    // Using srcW/2 × srcH/2 vertices balances edge-cutting accuracy with GPU cost.
    // Each vertex stores (u, v) in [0,1]; position is computed in the VS from depth.
    if (m_meshVS) {   // only build if mesh shaders compiled successfully
        // meshDiv: 1=full source resolution, 2=half(default), 4=quarter.
        // Clamped so the grid is at least 2×2.
        int div = std::max(1, meshDiv);
        int mW = std::max(2, srcW / div);
        int mH = std::max(2, srcH / div);

        // Vertex buffer: mW*mH float2 UVs
        std::vector<float> vdata;
        vdata.reserve((size_t)mW * mH * 2);
        for (int y = 0; y < mH; ++y)
            for (int x = 0; x < mW; ++x) {
                vdata.push_back((x + 0.5f) / mW);   // u centre of texel
                vdata.push_back((y + 0.5f) / mH);   // v
            }
        D3D11_BUFFER_DESC vbd{};
        vbd.ByteWidth      = (UINT)(vdata.size() * sizeof(float));
        vbd.Usage          = D3D11_USAGE_IMMUTABLE;
        vbd.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
        D3D11_SUBRESOURCE_DATA vinitd{ vdata.data(), 0, 0 };
        m_dev->CreateBuffer(&vbd, &vinitd, &m_meshVB);

        // Index buffer: two triangles (6 indices) per quad
        std::vector<uint32_t> idata;
        idata.reserve((size_t)(mW-1) * (mH-1) * 6);
        for (int y = 0; y < mH-1; ++y)
            for (int x = 0; x < mW-1; ++x) {
                uint32_t tl = (uint32_t)(y * mW + x);
                uint32_t tr = tl + 1;
                uint32_t bl = tl + (uint32_t)mW;
                uint32_t br = bl + 1;
                idata.push_back(tl); idata.push_back(bl); idata.push_back(tr);
                idata.push_back(tr); idata.push_back(bl); idata.push_back(br);
            }
        D3D11_BUFFER_DESC ibd{};
        ibd.ByteWidth  = (UINT)(idata.size() * sizeof(uint32_t));
        ibd.Usage      = D3D11_USAGE_IMMUTABLE;
        ibd.BindFlags  = D3D11_BIND_INDEX_BUFFER;
        D3D11_SUBRESOURCE_DATA iinitd{ idata.data(), 0, 0 };
        m_dev->CreateBuffer(&ibd, &iinitd, &m_meshIB);

        m_meshW = mW; m_meshH = mH;
        LOG_DBG("StereoRenderer: mesh ", mW, "x", mH,
                " (", (int)idata.size(), " indices)");
    }

    // ── Depth-stencil texture (same size as output) ───────────────────────────
    {
        D3D11_TEXTURE2D_DESC dsd{};
        dsd.Width=dstW; dsd.Height=dstH; dsd.MipLevels=1; dsd.ArraySize=1;
        dsd.Format=DXGI_FORMAT_D32_FLOAT; dsd.SampleDesc.Count=1;
        dsd.Usage=D3D11_USAGE_DEFAULT;
        dsd.BindFlags=D3D11_BIND_DEPTH_STENCIL;
        ComPtr<ID3D11Texture2D> dsTex;
        if (SUCCEEDED(m_dev->CreateTexture2D(&dsd, nullptr, &dsTex))) {
            D3D11_DEPTH_STENCIL_VIEW_DESC dvd{};
            dvd.Format=DXGI_FORMAT_D32_FLOAT;
            dvd.ViewDimension=D3D11_DSV_DIMENSION_TEXTURE2D;
            m_dev->CreateDepthStencilView(dsTex.Get(), &dvd, &m_dsv);
            m_dsTex = std::move(dsTex);
        }
    }

    return S_OK;
}

HRESULT StereoRenderer::Render(const BYTE* srcFrame, int srcW, int srcH,
                                int srcStride,
                                const float* depthMap,
                                const DeflattenConfig& cfg,
                                float motionDx, float motionDy,
                                BYTE* dstFrame, int dstStride) {
    if (m_renderCount == 0)
        LOG_INFO("First Render: src=", srcW, "x", srcH,
                 " path=", m_gpuOK ? "GPU (DX11)" : "CPU (software)",
                 " conv=", cfg.convergence, " sep=", cfg.separation,
                 " mode=", cfg.outputMode==OutputMode::SideBySide?"SBS":"TAB");
    ++m_renderCount;

    if (m_gpuOK)
        RenderGPU(srcFrame,srcW,srcH,srcStride,depthMap,cfg,motionDx,motionDy,dstFrame,dstStride);
    else
        RenderCPU(srcFrame,srcW,srcH,srcStride,depthMap,cfg,dstFrame,dstStride);
    return S_OK;
}

void StereoRenderer::RenderGPU(const BYTE* srcFrame, int srcW, int srcH,
                                 int srcStride,
                                 const float* depthMap,
                                 const DeflattenConfig& cfg,
                                 float motionDx, float motionDy,
                                 BYTE* dstFrame, int dstStride) {
    HRESULT hrET = EnsureTextures(srcW, srcH, cfg.outputMode, cfg.meshDiv);
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
    CBStereo cbBase{};
    if (SUCCEEDED(m_ctx->Map(m_cb.Get(),0,D3D11_MAP_WRITE_DISCARD,0,&mapped))) {
        auto* cb = static_cast<CBStereo*>(mapped.pData);
        cb->convergence   = cfg.convergence;
        cb->separation    = cfg.separation;
        cb->flipDepth     = cfg.flipDepth ? 1.f : 0.f;
        cb->outputMode    = (int)cfg.outputMode;
        cb->texelW        = srcW > 0 ? 1.0f / (float)srcW : 0.0f;
        cb->texelH        = srcH > 0 ? 1.0f / (float)srcH : 0.0f;
        cb->infillMode    = (int)cfg.infillMode;
        cb->depthOffsetU  = srcW > 0 ? motionDx / (float)srcW : 0.f;
        cb->depthOffsetV  = srcH > 0 ? motionDy / (float)srcH : 0.f;
        cb->discThresh    = cfg.discThresh;
        cb->eyeSign       = 0.f;
        cb->pad1          = 0.f;
        cbBase = *cb;   // save for mesh eye-sign updates
        m_ctx->Unmap(m_cb.Get(), 0);
    }

    D3D11_VIEWPORT vp{0,0,(float)dstW,(float)dstH,0,1};
    m_ctx->RSSetViewports(1,&vp);

    // ── Clear output to black and depth to 1.0 ───────────────────────────────
    // Uncovered pixels (gaps cut by MeshGS at depth discontinuities) retain
    // depth 1.0 and colour black.  Pass 2 (UV-warp hole-fill) fills them.
    const float black[4] = {0,0,0,1};
    m_ctx->ClearRenderTargetView(m_rtv.Get(), black);
    // 2.0 = outside [0,1] so GREATER-vs-2.0 identifies true gaps unambiguously
    if (m_dsv) m_ctx->ClearDepthStencilView(m_dsv.Get(), D3D11_CLEAR_DEPTH, 2.0f, 0);

    // ── Pass 1: mesh + GS (two DrawIndexed calls for left+right eye) ─────────
    // MeshVS shifts each vertex by its per-depth disparity.
    // MeshGS culls triangles whose vertices straddle a depth discontinuity,
    // leaving those pixels at cleared depth 1.0 for pass 2 to fill.
    // MeshPS samples srcTex at the original (unshifted) UV.
    if (!m_meshVS || !m_meshGS || !m_meshIL || !m_meshVB || !m_meshIB) {
        LOG_WARN("Mesh resources missing (VS/GS/IL/VB/IB) — no output this frame.");
        return;
    }

    m_ctx->OMSetDepthStencilState(m_dsState.Get(), 0);
    m_ctx->OMSetRenderTargets(1, m_rtv.GetAddressOf(), m_dsv.Get());
    m_ctx->RSSetState(m_meshRaster.Get());
    m_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    m_ctx->IASetInputLayout(m_meshIL.Get());
    UINT mstride = sizeof(float) * 2, moff = 0;
    m_ctx->IASetVertexBuffers(0, 1, m_meshVB.GetAddressOf(), &mstride, &moff);
    m_ctx->IASetIndexBuffer(m_meshIB.Get(), DXGI_FORMAT_R32_UINT, 0);
    m_ctx->VSSetShader(m_meshVS.Get(), nullptr, 0);
    m_ctx->GSSetShader(m_meshGS.Get(), nullptr, 0);
    m_ctx->PSSetShader(m_meshPS.Get(), nullptr, 0);
    m_ctx->VSSetConstantBuffers(0, 1, m_cb.GetAddressOf());
    m_ctx->GSSetConstantBuffers(0, 1, m_cb.GetAddressOf());
    m_ctx->PSSetConstantBuffers(0, 1, m_cb.GetAddressOf());
    m_ctx->VSSetShaderResources(1, 1, m_depthSRV.GetAddressOf());
    m_ctx->VSSetSamplers(0, 1, m_sampler.GetAddressOf());
    m_ctx->PSSetShaderResources(0, 1, m_srcSRV.GetAddressOf());
    m_ctx->PSSetShaderResources(1, 1, m_depthSRV.GetAddressOf());
    m_ctx->PSSetSamplers(0, 1, m_sampler.GetAddressOf());

    int nIdx = (m_meshW - 1) * (m_meshH - 1) * 6;

    auto drawEye = [&](float eyeSign) {
        D3D11_MAPPED_SUBRESOURCE mc{};
        if (SUCCEEDED(m_ctx->Map(m_cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mc))) {
            auto* c = static_cast<CBStereo*>(mc.pData);
            *c = cbBase;
            c->eyeSign = eyeSign;
            m_ctx->Unmap(m_cb.Get(), 0);
        }
        m_ctx->DrawIndexed((UINT)nIdx, 0, 0);
    };

    drawEye(+1.f);   // left eye  (SBS: left half;  TAB: top half)
    drawEye(-1.f);   // right eye (SBS: right half; TAB: bottom half)

    // Unbind GS — required before pass 2 which has no GS stage.
    m_ctx->GSSetShader(nullptr, nullptr, 0);
    m_ctx->GSSetShader(nullptr, nullptr, 0);
    m_ctx->VSSetShaderResources(1, 0, nullptr);
    { ID3D11ShaderResourceView* n2[2]={nullptr,nullptr}; m_ctx->PSSetShaderResources(0,2,n2); }
    m_ctx->OMSetDepthStencilState(nullptr, 0);

    // ── Pass 2: UV-warp hole-fill ─────────────────────────────────────────────
    // Draw a full-screen quad with depth-test EQUAL 1.0, no depth write.
    // The PS (PS_StereoWarp) only executes for pixels still at depth 1.0 —
    // exactly the gaps MeshGS left by culling discontinuity-straddling
    // triangles. For each such pixel the PS computes the warp disparity, detects
    // that it lands on foreground (depthJump > 0.10), and runs the selected
    // g_infillMode search (Inner/Outer/Blend/EdgeClamp/Inpaint) to find and
    // write the correct background colour. Covered pixels (mesh depth < 1.0)
    // fail the EQUAL test and are skipped with no PS invocation (essentially
    // free early-Z rejection on FL11.0 hardware like the RTX 2080 Ti).
    // PS_StereoWarp determines which eye each pixel belongs to from its UV
    // (uv.x < 0.5 for SBS, uv.y < 0.5 for TAB), so one Draw(4) covers both.
    {
        m_ctx->OMSetDepthStencilState(m_dsStateHoleFill.Get(), 0);
        m_ctx->OMSetRenderTargets(1, m_rtv.GetAddressOf(), m_dsv.Get());
        m_ctx->RSSetState(m_raster.Get());
        m_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        m_ctx->IASetInputLayout(m_il.Get());
        UINT wstride = sizeof(Vertex), woff = 0;
        m_ctx->IASetVertexBuffers(0, 1, m_vb.GetAddressOf(), &wstride, &woff);
        m_ctx->VSSetShader(m_vs.Get(), nullptr, 0);
        m_ctx->PSSetShader(m_ps.Get(), nullptr, 0);
        // Reset CB: eyeSign=0 (PS_StereoWarp derives eye from UV, not eyeSign).
        D3D11_MAPPED_SUBRESOURCE mcw{};
        if (SUCCEEDED(m_ctx->Map(m_cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mcw))) {
            *static_cast<CBStereo*>(mcw.pData) = cbBase;  // cbBase.eyeSign == 0
            m_ctx->Unmap(m_cb.Get(), 0);
        }
        m_ctx->VSSetConstantBuffers(0, 1, m_cb.GetAddressOf());
        m_ctx->PSSetConstantBuffers(0, 1, m_cb.GetAddressOf());
        // PS_StereoWarp uses t0=srcTex, t1=depthTex
        m_ctx->PSSetShaderResources(0, 1, m_srcSRV.GetAddressOf());
        m_ctx->PSSetShaderResources(1, 1, m_depthSRV.GetAddressOf());
        m_ctx->PSSetSamplers(0, 1, m_sampler.GetAddressOf());
        m_ctx->Draw(4, 0);
        // Unbind PS SRVs to avoid binding hazards on the next mesh pass.
        ID3D11ShaderResourceView* nullSRVs[2] = {nullptr, nullptr};
        m_ctx->PSSetShaderResources(0, 2, nullSRVs);
        m_ctx->OMSetDepthStencilState(nullptr, 0);
    }

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
