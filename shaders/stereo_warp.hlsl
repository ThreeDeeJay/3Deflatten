// SPDX-License-Identifier: GPL-3.0-or-later
// Stereo warp shader for 3Deflatten

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
