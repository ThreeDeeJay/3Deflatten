// SPDX-License-Identifier: GPL-3.0-or-later
// jbu_cuda.cu — CUDA Joint Bilateral Upsampling + separable max-dilation
//
// v4 changes vs v3:
//   1. k_smooth_lr: pre-smooth the raw LR depth with a Gaussian (sigma=0.7,
//      radius=1) before upsampling.  Removes sub-pixel jaggedness at model-
//      resolution object edges without blurring real depth boundaries.
//      The smoothed result is written into guide_lr_dev (repurposed scratch;
//      same size = lrW*lrH floats, no new allocation needed).
//   2. k_jbu: uses full RGB colour distance for the bilateral range weight
//      instead of luma-only.  Catches colour edges where luminance is similar
//      (e.g. red foreground on orange background) and prevents cross-edge
//      bleed in those cases.  Guide is still sampled at full HR resolution
//      (no blurry LR guide).
//
// Call-site: keep the same jbu_cuda signature; pass s.d_guideLR[writeBuf]
// as guide_lr_dev (it is written by k_smooth_lr and read by k_jbu).
// Update parameters: ss=1.5f, sc=0.08f, radius=3.
#include <cuda_runtime.h>
#include "jbu_cuda.h"

// ── Step 1: pre-smooth LR depth (tiny Gaussian, radius=1, sigma≈0.7) ─────────
// Cleans up the blocky per-pixel steps at LR object edges without changing
// the overall depth structure.  Output goes into a scratch buffer (guide_lr_dev).
__global__ void k_smooth_lr(const float* __restrict__ in,
                              float*       __restrict__ out,
                              int W, int H, float inv2s)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float wSum = 0.f, dSum = 0.f;
    for (int ky = max(0, y-1); ky <= min(H-1, y+1); ++ky) {
        float dvy = (float)(y - ky);
        for (int kx = max(0, x-1); kx <= min(W-1, x+1); ++kx) {
            float dvx = (float)(x - kx);
            float wt  = expf(-(dvx*dvx + dvy*dvy) * inv2s);
            wSum += wt;
            dSum += wt * in[ky*W + kx];
        }
    }
    out[y*W + x] = dSum / wSum;
}

// ── Step 2: JBU — HR output pixel by pixel ────────────────────────────────────
// For each HR output pixel (ox,oy):
//   • gP = full RGB at (ox,oy) in the original full-res guide
//   • For each LR neighbour (kx2,ky2):
//       spatial weight  ws = exp(−|ΔLR|² / 2σs²)
//       gQ = full RGB at the HR centre of that LR pixel  (sharp, not blurry)
//       colour weight   wc = exp(−|ΔRGBsq| / 2σc²)   where ΔRGBsq = ΔR²+ΔG²+ΔB²
// Using full RGB rather than luma catches colour-only edges (similar luminance,
// different hue) that a luma-only filter would bleed across.
__global__ void k_jbu(
    const float*         __restrict__ dlr,  // smoothed LR depth (guide_lr_dev)
    const unsigned char* __restrict__ g,    // full-res BGRA guide
    int hrW, int hrH, int hrS,
    int lrW, int lrH,
    float* __restrict__ dhr,
    float inv2ss, float inv2cs, int radius)
{
    int ox = blockIdx.x*blockDim.x + threadIdx.x;
    int oy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ox >= hrW || oy >= hrH) return;

    // Full RGB at this HR output pixel (query)
    const unsigned char* gp = g + oy*hrS + ox*4;
    float rP = gp[2] * (1.f/255.f);
    float gGP = gp[1] * (1.f/255.f);
    float bP  = gp[0] * (1.f/255.f);

    // Corresponding continuous LR position
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

            // Spatial weight in LR space
            float ws = expf(-(du*du + dv*dv) * inv2ss);

            // Full RGB at the HR centre of this LR pixel (no blurry LR guide)
            int hx = min(hrW-1, (int)((kx2 + 0.5f) * hrW / lrW));
            int hy = min(hrH-1, (int)((ky2 + 0.5f) * hrH / lrH));
            const unsigned char* gq = g + hy*hrS + hx*4;
            float dr  = rP  - gq[2]*(1.f/255.f);
            float dg  = gGP - gq[1]*(1.f/255.f);
            float db  = bP  - gq[0]*(1.f/255.f);
            float dc2 = dr*dr + dg*dg + db*db;   // squared RGB distance

            float wt = ws * expf(-dc2 * inv2cs);
            wS += wt;
            dS += wt * dlr[ky2*lrW + kx2];
        }
    }
    int nr = max(0, min(lv0, lrH-1)) * lrW + max(0, min(lu0, lrW-1));
    dhr[oy*hrW + ox] = (wS > 1e-8f) ? dS / wS : dlr[nr];
}

// ── Public API ────────────────────────────────────────────────────────────────
// guide_lr_dev  — REPURPOSED as the smoothed-LR-depth scratch buffer.
//                 Must be pre-allocated at lrW*lrH floats (same as before).
//                 It is written by k_smooth_lr and then read by k_jbu.
// Recommended call-site parameters: ss=1.5f, sc=0.08f, radius=3
int jbu_cuda(const float*         dlr,
             int lrW, int lrH,
             const unsigned char* g,
             int hrW, int hrH, int hrS,
             float*               dhr,
             float ss, float sc, int radius,
             float*               guide_lr_dev,   // smoothed-LR scratch (lrW*lrH)
             void*                stream)
{
    cudaStream_t st  = (cudaStream_t)stream;
    dim3         blk(16, 16);
    dim3         gridLR((lrW+15)/16, (lrH+15)/16);
    dim3         gridHR((hrW+15)/16, (hrH+15)/16);

    // Step 1: pre-smooth the raw LR depth → guide_lr_dev
    const float smooth_inv2s = 1.f / (2.f * 0.7f * 0.7f);  // sigma=0.7
    k_smooth_lr<<<gridLR, blk, 0, st>>>(dlr, guide_lr_dev, lrW, lrH, smooth_inv2s);

    // Step 2: JBU on the smoothed LR depth, guided by full-res RGB
    const float i2ss = 1.f / (2.f * ss * ss);
    const float i2cs = 1.f / (2.f * sc * sc);
    k_jbu<<<gridHR, blk, 0, st>>>(
        guide_lr_dev, g, hrW, hrH, hrS, lrW, lrH, dhr, i2ss, i2cs, radius);

    return (int)cudaGetLastError();
}

// ── Separable max-dilation (unchanged) ───────────────────────────────────────
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
    k_dilate_h<<<grid, block, 0, st>>>(src, tmp, w, h, radius, edgeThresh);
    k_dilate_v<<<grid, block, 0, st>>>(tmp, dst, w, h, radius, edgeThresh);
    return (int)cudaGetLastError();
}