// SPDX-License-Identifier: GPL-3.0-or-later
// Stereo warp shader for 3Deflatten
// Produces Side-by-Side or Top-and-Bottom stereoscopic output from a mono
// source frame + a grayscale depth map.
//
// Edge handling strategy
// ──────────────────────
// Simple backward warping creates an "occlusion gap" at depth discontinuities:
// the background pixels that were hidden behind a near object in the original
// view have no source to sample from, so they end up repeating the near-object
// edge ("halo" / "ghosting").
//
// Two complementary techniques are applied:
//  1. Depth DILATION  – the depth value at each pixel is replaced by the
//     maximum over a small horizontal neighbourhood. This causes near-field
//     regions to grow outward, pre-filling a fraction of the gap before it
//     even appears. Cheap: 2 extra texture reads per pixel.
//  2. Background SEARCH – when the warped sample crosses a depth
//     discontinuity (the pixel at srcUV is significantly nearer than the
//     pixel at eyeUV), we walk horizontally in the opposite direction until
//     we find a texel at similar depth, then warp that instead.

cbuffer CBStereo : register(b0) {
    float  g_convergence;   // [0,1] depth at screen plane
    float  g_separation;    // stereo strength
    float  g_flipDepth;     // unused (depth pre-flipped on CPU before upload)
    int    g_outputMode;    // 0=SBS, 1=TAB
    float  g_texelW;        // 1.0 / srcWidth
    float  g_texelH;        // 1.0 / srcHeight  (currently unused)
    float2 g_pad;
};

Texture2D<float4> g_srcTex   : register(t0);
Texture2D<float>  g_depthTex : register(t1);
SamplerState      g_sampler  : register(s0);  // bilinear clamp

struct VS_IN  { float2 pos : POSITION; float2 uv : TEXCOORD; };
struct VS_OUT { float4 pos : SV_POSITION; float2 uv : TEXCOORD; };

VS_OUT VS_FullScreen(VS_IN v) {
    VS_OUT o;
    o.pos = float4(v.pos, 0, 1);
    o.uv  = v.uv;
    return o;
}

float4 PS_StereoWarp(VS_OUT i) : SV_TARGET {
    float2 uv = i.uv;

    // ── Split into eye-space UV ───────────────────────────────────────────────
    bool   isLeft;
    float2 eyeUV;
    if (g_outputMode == 0) {            // Side-by-Side
        isLeft = (uv.x < 0.5);
        eyeUV  = float2(isLeft ? uv.x * 2.0 : (uv.x - 0.5) * 2.0, uv.y);
    } else {                            // Top-and-Bottom
        isLeft = (uv.y < 0.5);
        eyeUV  = float2(uv.x, isLeft ? uv.y * 2.0 : (uv.y - 0.5) * 2.0);
    }
    float eyeSign = isLeft ? 1.0 : -1.0;

    // ── Depth with edge dilation ──────────────────────────────────────────────
    // max() over a ±3-texel neighbourhood causes near objects to bleed
    // outward, reducing the exposed-background width at depth edges.
    float dC = g_depthTex.SampleLevel(g_sampler, eyeUV, 0).r;
    float dL = g_depthTex.SampleLevel(g_sampler, eyeUV + float2(-g_texelW * 3.0, 0), 0).r;
    float dR = g_depthTex.SampleLevel(g_sampler, eyeUV + float2(+g_texelW * 3.0, 0), 0).r;
    float depth = max(dC, max(dL, dR));

    // ── Primary warp ─────────────────────────────────────────────────────────
    float  disparity = g_separation * (depth - g_convergence);
    float2 srcUV     = saturate(eyeUV + float2(eyeSign * disparity, 0.0));

    // ── Occlusion-gap background infill ──────────────────────────────────────
    // If the sampled depth is significantly closer than our pixel's depth
    // we've landed on a foreground surface – this is an occlusion gap.
    // Walk away from the warp direction in eye-space to find a background
    // pixel at similar depth, then warp that instead.
    float sampledDepth = g_depthTex.SampleLevel(g_sampler, srcUV, 0).r;
    float depthJump    = sampledDepth - dC;   // positive → sampled point is nearer

    [branch]
    if (depthJump > 0.10) {
        float2 searchDir = float2(-eyeSign * g_texelW * 3.0, 0);
        float4 fillColor = g_srcTex.SampleLevel(g_sampler, srcUV, 0);  // safe fallback

        [loop]
        for (int s = 1; s <= 16; ++s) {
            float2 cUV    = saturate(eyeUV + searchDir * (float)s);
            float  cDepth = g_depthTex.SampleLevel(g_sampler, cUV, 0).r;
            if (abs(cDepth - dC) < 0.08) {
                // Found a background pixel – warp it with its own disparity
                float cDisp = g_separation * (cDepth - g_convergence);
                float2 cSrcUV = saturate(cUV + float2(eyeSign * cDisp, 0.0));
                fillColor = g_srcTex.SampleLevel(g_sampler, cSrcUV, 0);
                break;
            }
        }

        // Hard-blend: transition over a narrow depth-jump range
        float blend = saturate((depthJump - 0.10) * 10.0);
        return lerp(g_srcTex.SampleLevel(g_sampler, srcUV, 0), fillColor, blend);
    }

    return g_srcTex.SampleLevel(g_sampler, srcUV, 0);
}
