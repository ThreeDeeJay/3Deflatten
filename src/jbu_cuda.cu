// SPDX-License-Identifier: GPL-3.0-or-later
// jbu_cuda.cu — CUDA guided depth upscaling: JBU + Weighted Mode Filtering,
//               plus separable max-dilation.
//
// Both k_jbu and k_wmf sample the guide image directly at full resolution
// for every low-res neighbour (mapped to that neighbour's HR-cell centre)
// rather than pre-averaging into a blurred low-res guide buffer — this is
// what makes edges inherit the full sharpness of the RGB image instead of
// a softened approximation of it.
//
// Colour distance uses full RGB (luma+chroma), not luma alone, so edges
// that differ in hue but not brightness (e.g. red object on orange
// background) are still detected and respected.
#include <cuda_runtime.h>
#include "jbu_cuda.h"

// ── Joint Bilateral Upsampling ────────────────────────────────────────────────
// weight(p,q) = exp(-|Δspatial|² / 2σs²) × exp(-|ΔRGB|² / 2σc²)
// output(p)   = Σ weight·depth(q) / Σ weight
//
// This is a continuous weighted blend: even far-from-mode (wrong-side-of-
// edge) samples contribute a small nonzero amount, which is the structural
// reason JBU can show faint glow/feathering at edges even with a sharp
// guide.  See k_wmf below for the mode-based alternative that excludes
// those samples entirely instead of merely down-weighting them.
__global__ void k_jbu(
    const float*          __restrict__ dlr,
    const unsigned char*  __restrict__ g, int hrW, int hrH, int hrS,
    int lrW, int lrH,
    float* __restrict__ dhr,
    float inv2ss, float inv2cs, int radius)
{
    int ox = blockIdx.x*blockDim.x + threadIdx.x;
    int oy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ox >= hrW || oy >= hrH) return;

    const unsigned char* gp = g + oy*hrS + ox*4;
    float rP = gp[2] * (1.f/255.f);
    float gP = gp[1] * (1.f/255.f);
    float bP = gp[0] * (1.f/255.f);

    float lu = (ox + 0.5f) * lrW / (float)hrW - 0.5f;
    float lv = (oy + 0.5f) * lrH / (float)hrH - 0.5f;
    int lu0 = (int)lu, lv0 = (int)lv;

    float wS = 0.f, dS = 0.f;
    for (int ky = lv0 - radius; ky <= lv0 + radius + 1; ++ky) {
        int ky2 = max(0, min(ky, lrH - 1));
        float dv = lv - ky;
        for (int kx = lu0 - radius; kx <= lu0 + radius + 1; ++kx) {
            int kx2 = max(0, min(kx, lrW - 1));
            float du = lu - kx;
            float ws = expf(-(du*du + dv*dv) * inv2ss);

            // Full-res guide sample at this LR neighbour's HR-cell centre
            // (sharp — never a blurred low-res guide average).
            int hx = min(hrW - 1, (int)((kx2 + 0.5f) * hrW / lrW));
            int hy = min(hrH - 1, (int)((ky2 + 0.5f) * hrH / lrH));
            const unsigned char* gq = g + hy*hrS + hx*4;
            float dr = rP - gq[2]*(1.f/255.f);
            float dg = gP - gq[1]*(1.f/255.f);
            float db = bP - gq[0]*(1.f/255.f);
            float dc2 = dr*dr + dg*dg + db*db;     // full-RGB squared distance

            float w = ws * expf(-dc2 * inv2cs);
            wS += w;
            dS += w * dlr[ky2*lrW + kx2];
        }
    }
    int nr = max(0, min(lv0, lrH-1)) * lrW + max(0, min(lu0, lrW-1));
    dhr[oy*hrW + ox] = (wS > 1e-8f) ? dS / wS : dlr[nr];
}

int jbu_cuda(const float*          dlr, int lrW, int lrH,
             const unsigned char*  g,   int hrW, int hrH, int hrS,
             float*                dhr,
             float ss, float sc, int radius,
             float* /*guide_lr_dev, unused*/, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    dim3 b(16, 16);
    float i2ss = 1.f / (2.f * ss * ss);
    float i2cs = 1.f / (2.f * sc * sc);
    k_jbu<<<dim3((hrW+15)/16, (hrH+15)/16), b, 0, st>>>(
        dlr, g, hrW, hrH, hrS, lrW, lrH, dhr, i2ss, i2cs, radius);
    return (int)cudaGetLastError();
}

// ── Weighted Mode Filtering ───────────────────────────────────────────────────
// Min, Lu & Do, "Depth Video Enhancement Based on Weighted Mode Filtering",
// IEEE TIP 2012.
//
// Pass 1: build a weighted histogram over depth bins using the SAME
//         spatial+colour weight as JBU.
// Pass 2: find the bin with the most accumulated weight (the dominant
//         "mode" depth level in this neighbourhood), then recompute a
//         weighted average using ONLY the samples whose depth value falls
//         within that winning bin (±1 bin for sub-bin smoothness).
//
// The key structural difference from JBU: a sample on the wrong side of an
// edge doesn't just get a small weight — once the mode is chosen, it is
// excluded from the final average completely (weight forced to zero by the
// bin-membership test). There is no residual blend to glow.
#define WMF_BINS 16

__global__ void k_wmf(
    const float*          __restrict__ dlr,
    const unsigned char*  __restrict__ g, int hrW, int hrH, int hrS,
    int lrW, int lrH,
    float* __restrict__ dhr,
    float inv2ss, float inv2cs, int radius)
{
    int ox = blockIdx.x*blockDim.x + threadIdx.x;
    int oy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ox >= hrW || oy >= hrH) return;

    const unsigned char* gp = g + oy*hrS + ox*4;
    float rP = gp[2] * (1.f/255.f);
    float gP = gp[1] * (1.f/255.f);
    float bP = gp[0] * (1.f/255.f);

    float lu = (ox + 0.5f) * lrW / (float)hrW - 0.5f;
    float lv = (oy + 0.5f) * lrH / (float)hrH - 0.5f;
    int lu0 = (int)lu, lv0 = (int)lv;

    // Pass 1: weighted histogram (register array — WMF_BINS is small by design)
    float hist[WMF_BINS];
#pragma unroll
    for (int i = 0; i < WMF_BINS; ++i) hist[i] = 0.f;

    for (int ky = lv0 - radius; ky <= lv0 + radius + 1; ++ky) {
        int ky2 = max(0, min(ky, lrH - 1));
        float dv = lv - ky;
        for (int kx = lu0 - radius; kx <= lu0 + radius + 1; ++kx) {
            int kx2 = max(0, min(kx, lrW - 1));
            float du = lu - kx;
            float ws = expf(-(du*du + dv*dv) * inv2ss);

            int hx = min(hrW - 1, (int)((kx2 + 0.5f) * hrW / lrW));
            int hy = min(hrH - 1, (int)((ky2 + 0.5f) * hrH / lrH));
            const unsigned char* gq = g + hy*hrS + hx*4;
            float dr = rP - gq[2]*(1.f/255.f);
            float dg = gP - gq[1]*(1.f/255.f);
            float db = bP - gq[0]*(1.f/255.f);
            float dc2 = dr*dr + dg*dg + db*db;
            float w = ws * expf(-dc2 * inv2cs);

            float dval = dlr[ky2*lrW + kx2];
            int bin = (int)(dval * (WMF_BINS - 1) + 0.5f);
            bin = max(0, min(WMF_BINS - 1, bin));
            hist[bin] += w;
        }
    }

    // Find the dominant bin (the mode)
    int bestBin = 0;
    float bestW = hist[0];
#pragma unroll
    for (int i = 1; i < WMF_BINS; ++i)
        if (hist[i] > bestW) { bestW = hist[i]; bestBin = i; }

    const float binWidth   = 1.0f / (WMF_BINS - 1);
    const float modeCenter = bestBin * binWidth;

    // Pass 2: refine using ONLY samples within the winning bin (±1 for
    // sub-bin smoothness). Samples outside this range get excluded
    // entirely — not down-weighted — which is what eliminates the glow.
    float wS = 0.f, dS = 0.f;
    for (int ky = lv0 - radius; ky <= lv0 + radius + 1; ++ky) {
        int ky2 = max(0, min(ky, lrH - 1));
        float dv = lv - ky;
        for (int kx = lu0 - radius; kx <= lu0 + radius + 1; ++kx) {
            int kx2 = max(0, min(kx, lrW - 1));
            float dval = dlr[ky2*lrW + kx2];
            if (fabsf(dval - modeCenter) > 1.5f * binWidth) continue;

            float du = lu - kx;
            float ws = expf(-(du*du + dv*dv) * inv2ss);

            int hx = min(hrW - 1, (int)((kx2 + 0.5f) * hrW / lrW));
            int hy = min(hrH - 1, (int)((ky2 + 0.5f) * hrH / lrH));
            const unsigned char* gq = g + hy*hrS + hx*4;
            float dr = rP - gq[2]*(1.f/255.f);
            float dg = gP - gq[1]*(1.f/255.f);
            float db = bP - gq[0]*(1.f/255.f);
            float dc2 = dr*dr + dg*dg + db*db;
            float w = ws * expf(-dc2 * inv2cs);

            wS += w;
            dS += w * dval;
        }
    }
    dhr[oy*hrW + ox] = (wS > 1e-8f) ? dS / wS : modeCenter;
}

int wmf_cuda(const float*          dlr, int lrW, int lrH,
             const unsigned char*  g,   int hrW, int hrH, int hrS,
             float*                dhr,
             float ss, float sc, int radius,
             float* /*guide_lr_dev, unused*/, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    dim3 b(16, 16);
    float i2ss = 1.f / (2.f * ss * ss);
    float i2cs = 1.f / (2.f * sc * sc);
    k_wmf<<<dim3((hrW+15)/16, (hrH+15)/16), b, 0, st>>>(
        dlr, g, hrW, hrH, hrS, lrW, lrH, dhr, i2ss, i2cs, radius);
    return (int)cudaGetLastError();
}

// ── Separable max-dilation (unchanged algorithm) ─────────────────────────────
__global__ void k_dilate_h(const float* __restrict__ in, float* __restrict__ out,
                             int w, int h, int radius, float edgeThresh)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= w || y >= h) return;
    float center = in[y*w+x], best = center;
    for (int xi = max(0,x-radius); xi <= min(w-1,x+radius); ++xi) {
        float v = in[y*w+xi];
        if (v > best && (v-center) >= edgeThresh) best = v;
    }
    out[y*w+x] = best;
}

__global__ void k_dilate_v(const float* __restrict__ in, float* __restrict__ out,
                             int w, int h, int radius, float edgeThresh)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= w || y >= h) return;
    float center = in[y*w+x], best = center;
    for (int yi = max(0,y-radius); yi <= min(h-1,y+radius); ++yi) {
        float v = in[yi*w+x];
        if (v > best && (v-center) >= edgeThresh) best = v;
    }
    out[y*w+x] = best;
}

int gpu_dilate(const float* src, float* tmp, float* dst,
               int w, int h, int radius, float edgeThresh, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    dim3 block(256), grid((w+255)/256, h);
    k_dilate_h<<<grid,block,0,st>>>(src, tmp, w, h, radius, edgeThresh);
    k_dilate_v<<<grid,block,0,st>>>(tmp, dst, w, h, radius, edgeThresh);
    return (int)cudaGetLastError();
}