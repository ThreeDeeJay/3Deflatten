// SPDX-License-Identifier: GPL-3.0-or-later
// jbu_cuda.cu — CUDA Joint Bilateral Upsampling + separable max-dilation
//
// v3 change: k_jbu now compares guide luma in full-res (HR) space for both
// the query pixel AND each low-res neighbor sample, eliminating the blurry
// downsampled guide that caused edge glow/feathering at depth boundaries.
// k_guide_lr is removed; guide_lr_dev is accepted but unused (API compat).
#include <cuda_runtime.h>
#include "jbu_cuda.h"

// ── JBU ──────────────────────────────────────────────────────────────────────
// For each HR output pixel (ox,oy):
//   gP  = guide luma at (ox,oy)                                  [HR, sharp]
//   gQ  = guide luma at the HR centre of LR neighbor (kx2,ky2)   [HR, sharp]
// Both luma values come from the original full-res guide, so the bilateral
// color weights inherit the full sharpness of the RGB image.
__global__ void k_jbu(
    const float* __restrict__ dlr,
    const unsigned char* __restrict__ g, int hrW, int hrH, int hrS,
    int lrW, int lrH, float* __restrict__ dhr,
    float inv2ss, float inv2cs, int radius)
{
    int ox = blockIdx.x*blockDim.x + threadIdx.x;
    int oy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ox >= hrW || oy >= hrH) return;

    // Guide luma at this HR output pixel (query)
    const unsigned char* gp = g + oy*hrS + ox*4;
    float gP = (29*gp[0] + 150*gp[1] + 77*gp[2]) * (1.f/65280.f);

    // Corresponding continuous LR position
    float lu = (ox + 0.5f) * lrW / (float)hrW - 0.5f;
    float lv = (oy + 0.5f) * lrH / (float)hrH - 0.5f;
    int lu0 = (int)lu, lv0 = (int)lv;

    float wS = 0.f, dS = 0.f;
    for (int ky = lv0 - radius; ky <= lv0 + radius + 1; ++ky) {
        int ky2 = max(0, min(ky, lrH - 1));
        float dv  = lv - ky;
        for (int kx = lu0 - radius; kx <= lu0 + radius + 1; ++kx) {
            int kx2 = max(0, min(kx, lrW - 1));
            float du = lu - kx;

            // Spatial weight (in LR space)
            float ws = expf(-(du*du + dv*dv) * inv2ss);

            // Guide luma at the HR centre of this LR pixel (sharp — no blurry LR guide)
            int hx = min(hrW - 1, (int)((kx2 + 0.5f) * hrW / lrW));
            int hy = min(hrH - 1, (int)((ky2 + 0.5f) * hrH / lrH));
            const unsigned char* gq = g + hy*hrS + hx*4;
            float gQ = (29*gq[0] + 150*gq[1] + 77*gq[2]) * (1.f/65280.f);

            // Color weight — tight sigma suppresses cross-edge bleeding
            float dc = gP - gQ;
            float w  = ws * expf(-dc*dc * inv2cs);
            wS += w;
            dS += w * dlr[ky2*lrW + kx2];
        }
    }
    int nr = max(0, min(lv0, lrH-1)) * lrW + max(0, min(lu0, lrW-1));
    dhr[oy*hrW + ox] = (wS > 1e-8f) ? dS / wS : dlr[nr];
}

// guide_lr_dev: accepted for API compatibility, not used (HR guide used directly).
int jbu_cuda(const float* dlr, int lrW, int lrH,
             const unsigned char* g, int hrW, int hrH, int hrS,
             float* dhr, float ss, float sc, int radius,
             float* /*guide_lr_dev*/, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    dim3 b(16, 16);
    float i2ss = 1.f / (2.f * ss * ss);
    float i2cs = 1.f / (2.f * sc * sc);
    k_jbu<<<dim3((hrW+15)/16, (hrH+15)/16), b, 0, st>>>(
        dlr, g, hrW, hrH, hrS, lrW, lrH, dhr, i2ss, i2cs, radius);
    return (int)cudaGetLastError();
}

// ── Separable max-dilation ────────────────────────────────────────────────────
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