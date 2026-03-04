// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten – HLSL stereo warp shader
// Entry points: VS_FullScreen (vs_5_0)  PS_StereoWarp (ps_5_0)

// ── Constant buffer ───────────────────────────────────────────────────────────
cbuffer CBStereo : register(b0)
{
    float  g_convergence;   // [0,1]   depth at screen plane
    float  g_separation;    // [0,0.1] disparity scale
    float  g_flipDepth;     // reserved (depth pre-flipped on CPU)
    int    g_outputMode;    // 0 = SBS,  1 = TAB
};

// ── Texture / sampler bindings ────────────────────────────────────────────────
Texture2D<float4> g_srcTex   : register(t0);   // BGRA source frame
Texture2D<float>  g_depthTex : register(t1);   // float [0,1] depth map
SamplerState      g_sampler  : register(s0);   // bilinear + clamp

// ── Vertex shader – full-screen quad (triangle strip) ────────────────────────
struct VS_IN  { float2 pos : POSITION; float2 uv : TEXCOORD; };
struct VS_OUT { float4 pos : SV_POSITION; float2 uv : TEXCOORD; };

VS_OUT VS_FullScreen(VS_IN v)
{
    VS_OUT o;
    o.pos = float4(v.pos, 0.0, 1.0);
    o.uv  = v.uv;
    return o;
}

// ── Pixel shader ──────────────────────────────────────────────────────────────
//
//  SBS layout (g_outputMode == 0):
//    Left  eye  ->  output UV.x in [0.00, 0.50)
//    Right eye  ->  output UV.x in [0.50, 1.00)
//
//  TAB layout (g_outputMode == 1):
//    Left  eye  ->  output UV.y in [0.00, 0.50)
//    Right eye  ->  output UV.y in [0.50, 1.00)
//
//  Disparity model (inverse warp):
//    d        = g_separation * (depth - g_convergence)
//    left  UV = eyeUV + ( d, 0)   (sample slightly to the right)
//    right UV = eyeUV + (-d, 0)   (sample slightly to the left)
//
float4 PS_StereoWarp(VS_OUT i) : SV_TARGET
{
    float2 uv = i.uv;

    bool   isLeft;
    float2 eyeUV;

    if (g_outputMode == 0)          // Side-by-Side
    {
        isLeft = (uv.x < 0.5);
        eyeUV  = float2(isLeft ? uv.x * 2.0 : (uv.x - 0.5) * 2.0, uv.y);
    }
    else                            // Top-and-Bottom
    {
        isLeft = (uv.y < 0.5);
        eyeUV  = float2(uv.x, isLeft ? uv.y * 2.0 : (uv.y - 0.5) * 2.0);
    }

    float  depth      = g_depthTex.Sample(g_sampler, eyeUV).r;
    float  disparity  = g_separation * (depth - g_convergence);
    float  eyeSign    = isLeft ? 1.0 : -1.0;
    float2 srcUV      = saturate(eyeUV + float2(eyeSign * disparity, 0.0));

    return g_srcTex.Sample(g_sampler, srcUV);
}
