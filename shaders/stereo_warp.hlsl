// SPDX-License-Identifier: GPL-3.0-or-later
// Stereo warp shader for 3Deflatten
cbuffer CBStereo : register(b0) {
    float  g_convergence;
    float  g_separation;
    float  g_flipDepth;
    int    g_outputMode;
    float  g_texelW;
    float  g_texelH;
    int    g_infillMode;   // 0=Inner  1=Outer  2=Blend
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
float4 PS_StereoWarp(VS_OUT i) : SV_TARGET {
    float2 uv = i.uv;
    bool isLeft; float2 eyeUV;
    if (g_outputMode == 0) { isLeft=(uv.x<0.5); eyeUV=float2(isLeft?uv.x*2.0:(uv.x-0.5)*2.0,uv.y); }
    else                   { isLeft=(uv.y<0.5); eyeUV=float2(uv.x,isLeft?uv.y*2.0:(uv.y-0.5)*2.0); }
    float eyeSign = isLeft ? 1.0 : -1.0;
    float dC=g_depthTex.SampleLevel(g_sampler,eyeUV,0).r;
    float dL=g_depthTex.SampleLevel(g_sampler,eyeUV+float2(-g_texelW*3,0),0).r;
    float dR=g_depthTex.SampleLevel(g_sampler,eyeUV+float2(+g_texelW*3,0),0).r;
    float depth=max(dC,max(dL,dR));
    float disparity=g_separation*(depth-g_convergence);
    float2 srcUV=saturate(eyeUV+float2(eyeSign*disparity,0));
    float sampledDepth=g_depthTex.SampleLevel(g_sampler,srcUV,0).r;
    float depthJump=sampledDepth-dC;
    [branch]
    if (depthJump>0.10) {
        float blend=saturate((depthJump-0.10)*10.0);
        float4 rawSample=g_srcTex.SampleLevel(g_sampler,srcUV,0);
        if (g_infillMode==0) {
            float2 searchDir=float2(-eyeSign*g_texelW*3,0);
            float4 fillColor=rawSample;
            [loop] for (int s=1;s<=16;++s) {
                float2 cUV=saturate(eyeUV+searchDir*(float)s);
                float cDepth=g_depthTex.SampleLevel(g_sampler,cUV,0).r;
                if (abs(cDepth-dC)<0.08) {
                    fillColor=g_srcTex.SampleLevel(g_sampler,saturate(cUV+float2(eyeSign*g_separation*(cDepth-g_convergence),0)),0);
                    break;
                }
            }
            return lerp(rawSample,fillColor,blend);
        } else if (g_infillMode==1) {
            float2 outerDir=float2(eyeSign*g_texelW*3,0);
            float4 fillColor=rawSample;
            [loop] for (int s=1;s<=16;++s) {
                float2 cUV=eyeUV+outerDir*(float)s;
                if (cUV.x<0||cUV.x>1) {
                    cUV=saturate(cUV);
                    fillColor=g_srcTex.SampleLevel(g_sampler,saturate(cUV+float2(eyeSign*g_separation*(g_depthTex.SampleLevel(g_sampler,cUV,0).r-g_convergence),0)),0);
                    break;
                }
                float cDepth=g_depthTex.SampleLevel(g_sampler,cUV,0).r;
                if (abs(cDepth-dC)<0.08)
                    fillColor=g_srcTex.SampleLevel(g_sampler,saturate(cUV+float2(eyeSign*g_separation*(cDepth-g_convergence),0)),0);
            }
            return lerp(rawSample,fillColor,blend);
        } else {
            float2 innerDir=float2(-eyeSign*g_texelW*3,0);
            float4 innerFill=rawSample; float innerConf=0;
            [loop] for (int si=1;si<=16;++si) {
                float2 cUV=saturate(eyeUV+innerDir*(float)si);
                float cDepth=g_depthTex.SampleLevel(g_sampler,cUV,0).r;
                if (abs(cDepth-dC)<0.08) {
                    innerFill=g_srcTex.SampleLevel(g_sampler,saturate(cUV+float2(eyeSign*g_separation*(cDepth-g_convergence),0)),0);
                    innerConf=1.0-(float)si/16.0; break;
                }
            }
            float2 outerDir=float2(eyeSign*g_texelW*3,0);
            float4 outerFill=rawSample;
            [loop] for (int so=1;so<=16;++so) {
                float2 cUV=saturate(eyeUV+outerDir*(float)so);
                float cDepth=g_depthTex.SampleLevel(g_sampler,cUV,0).r;
                if (abs(cDepth-dC)<0.08)
                    outerFill=g_srcTex.SampleLevel(g_sampler,saturate(cUV+float2(eyeSign*g_separation*(cDepth-g_convergence),0)),0);
            }
            return lerp(rawSample,lerp(outerFill,innerFill,innerConf),blend);
        }
    }
    return g_srcTex.SampleLevel(g_sampler,srcUV,0);
}
