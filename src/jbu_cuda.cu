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
    float inv2ss, float inv2cs, int radius, float dilateBias, bool flipped)
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

    // dilateBias > 0: among bins at least as supported as bestW*(1-bias),
    // prefer the bin closer to the foreground class instead of strictly the
    // best. This natively grows the foreground class within WMF's own
    // RGB-guided neighbourhood — a cleaner alternative to a separate
    // box-shaped max-dilate stacked on top of WMF's already-sharp output.
    //
    // Direction depends on `flipped`: this runs BEFORE flipDepth is applied
    // (which happens later, on CPU, in the collect phase) — pre-flip,
    // "foreground" is normally the HIGHEST depth bin, but the LOWEST bin
    // when the data's polarity will be inverted afterward. Biasing toward
    // the wrong end here is the same bug as gpu_dilate's flip handling —
    // it makes the foreground class visually SHRINK once flipped instead
    // of expanding.
    if (dilateBias > 0.f) {
        float thresh = bestW * (1.f - dilateBias);
        if (!flipped) {
            for (int i = WMF_BINS - 1; i > bestBin; --i) {
                if (hist[i] >= thresh) { bestBin = i; break; }
            }
        } else {
            for (int i = 0; i < bestBin; ++i) {
                if (hist[i] >= thresh) { bestBin = i; break; }
            }
        }
    }

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
             float ss, float sc, int radius, float dilateBias, bool flipped,
             float* /*guide_lr_dev, unused*/, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    dim3 b(16, 16);
    float i2ss = 1.f / (2.f * ss * ss);
    float i2cs = 1.f / (2.f * sc * sc);
    k_wmf<<<dim3((hrW+15)/16, (hrH+15)/16), b, 0, st>>>(
        dlr, g, hrW, hrH, hrS, lrW, lrH, dhr, i2ss, i2cs, radius, dilateBias, flipped);
    return (int)cudaGetLastError();
}

// ── Plain bilinear upscale (no guide) ────────────────────────────────────────
// Used for the "off" / Bilinear mode so the GPU stays busy and the pipeline
// stays GPU-resident regardless of which algorithm is selected. Previously
// Bilinear mode skipped all GPU upscale kernels and fell back to a CPU
// resize in the collect phase — which left the GPU idle for that portion of
// the frame and was, paradoxically, SLOWER than running JBU/WMF on the GPU
// (a single-threaded CPU resize of a full HD frame is slower than a small,
// massively-parallel GPU kernel).
__global__ void k_bilinear(
    const float* __restrict__ dlr, int lrW, int lrH,
    float* __restrict__ dhr, int hrW, int hrH)
{
    int ox = blockIdx.x*blockDim.x + threadIdx.x;
    int oy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ox >= hrW || oy >= hrH) return;

    float fu = (ox + 0.5f) * lrW / (float)hrW - 0.5f;
    float fv = (oy + 0.5f) * lrH / (float)hrH - 0.5f;
    int lx0 = (int)floorf(fu);
    int ly0 = (int)floorf(fv);
    float tx = (fu < 0.f) ? 0.f : fu - lx0;
    float ty = (fv < 0.f) ? 0.f : fv - ly0;
    lx0 = max(0, min(lx0, lrW - 1));
    ly0 = max(0, min(ly0, lrH - 1));
    int lx1 = min(lx0 + 1, lrW - 1);
    int ly1 = min(ly0 + 1, lrH - 1);

    float v00 = dlr[ly0*lrW + lx0];
    float v10 = dlr[ly0*lrW + lx1];
    float v01 = dlr[ly1*lrW + lx0];
    float v11 = dlr[ly1*lrW + lx1];
    dhr[oy*hrW + ox] = v00*(1-tx)*(1-ty) + v10*tx*(1-ty)
                      + v01*(1-tx)*ty    + v11*tx*ty;
}

int bilinear_cuda(const float* dlr, int lrW, int lrH,
                   float* dhr, int hrW, int hrH, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    dim3 b(16, 16);
    k_bilinear<<<dim3((hrW+15)/16, (hrH+15)/16), b, 0, st>>>(
        dlr, lrW, lrH, dhr, hrW, hrH);
    return (int)cudaGetLastError();
}

// ── Min/max normalisation (GPU-side preprocessing required by WMF) ──────────
// Float atomicMin/Max via CAS loop — correct for negative values too (unlike
// the common int-bitcast trick, which only preserves ordering within a
// single sign). Contention is negligible: only one CAS per BLOCK, not per
// thread.
__device__ __forceinline__ void atomicMinFloat(float* addr, float val) {
    int* iaddr = (int*)addr;
    int old = *iaddr, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) break;
        old = atomicCAS(iaddr, assumed, __float_as_int(val));
    } while (assumed != old);
}
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    int* iaddr = (int*)addr;
    int old = *iaddr, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(iaddr, assumed, __float_as_int(val));
    } while (assumed != old);
}

__global__ void k_minmax_init(float* mm) {
    mm[0] =  3.0e38f;   // running min
    mm[1] = -3.0e38f;   // running max
}

__global__ void k_minmax_reduce(const float* __restrict__ data, int n, float* mm) {
    __shared__ float smin[256];
    __shared__ float smax[256];
    int tid = threadIdx.x;
    float lmin =  3.0e38f, lmax = -3.0e38f;
    for (int i = blockIdx.x*blockDim.x + tid; i < n; i += blockDim.x*gridDim.x) {
        float v = data[i];
        lmin = fminf(lmin, v);
        lmax = fmaxf(lmax, v);
    }
    smin[tid] = lmin; smax[tid] = lmax;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smin[tid] = fminf(smin[tid], smin[tid + s]);
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicMinFloat(&mm[0], smin[0]);
        atomicMaxFloat(&mm[1], smax[0]);
    }
}

__global__ void k_normalize(const float* __restrict__ raw, int n,
                              const float* __restrict__ mm,
                              float* __restrict__ out) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float mn = mm[0], mx = mm[1];
    float range = (mx - mn) > 1e-6f ? (mx - mn) : 1e-6f;
    out[i] = (raw[i] - mn) / range;
}

int normalize_depth_cuda(const float* raw, int n,
                          float* mm_scratch, float* out, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    k_minmax_init<<<1, 1, 0, st>>>(mm_scratch);
    int blocks = (n + 255) / 256;
    if (blocks > 256) blocks = 256;   // grid-stride loop covers the rest
    k_minmax_reduce<<<blocks, 256, 0, st>>>(raw, n, mm_scratch);
    int nblk = (n + 255) / 256;
    k_normalize<<<nblk, 256, 0, st>>>(raw, n, mm_scratch, out);
    return (int)cudaGetLastError();
}

// ── Separable max-dilation ────────────────────────────────────────────────
// dirSign: +1 expands HIGH values (normal "depth=1=near" convention),
// -1 expands LOW values instead. Must reflect the polarity the data will
// have AFTER flipDepth is applied — see gpu_dilate()'s declaration comment.
__global__ void k_dilate_h(const float* __restrict__ in, float* __restrict__ out,
                             int w, int h, int radius, float edgeThresh, float dirSign)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= w || y >= h) return;
    float center = in[y*w+x], best = center;
    for (int xi = max(0,x-radius); xi <= min(w-1,x+radius); ++xi) {
        float v = in[y*w+xi];
        if (dirSign*v > dirSign*best && dirSign*(v-center) >= edgeThresh) best = v;
    }
    out[y*w+x] = best;
}

__global__ void k_dilate_v(const float* __restrict__ in, float* __restrict__ out,
                             int w, int h, int radius, float edgeThresh, float dirSign)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= w || y >= h) return;
    float center = in[y*w+x], best = center;
    for (int yi = max(0,y-radius); yi <= min(h-1,y+radius); ++yi) {
        float v = in[yi*w+x];
        if (dirSign*v > dirSign*best && dirSign*(v-center) >= edgeThresh) best = v;
    }
    out[y*w+x] = best;
}

int gpu_dilate(const float* src, float* tmp, float* dst,
               int w, int h, int radius, float edgeThresh,
               bool flipped, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    float dirSign = flipped ? -1.0f : 1.0f;
    dim3 block(256), grid((w+255)/256, h);
    k_dilate_h<<<grid,block,0,st>>>(src, tmp, w, h, radius, edgeThresh, dirSign);
    k_dilate_v<<<grid,block,0,st>>>(tmp, dst, w, h, radius, edgeThresh, dirSign);
    return (int)cudaGetLastError();
}
