// SPDX-License-Identifier: GPL-3.0-or-later
// jbu_cuda.cu — CUDA guided depth upscaling: JBU + Weighted Mode Filtering,
//               bilinear, normalize, and dilation variants.
//
// Both k_jbu and k_wmf sample the guide image directly at full resolution
// for every low-res neighbour (mapped to that neighbour's HR-cell centre)
// rather than pre-averaging into a blurred low-res guide buffer.
// Colour distance uses full RGB (luma+chroma), not luma alone.
#include <cuda_runtime.h>
#include "jbu_cuda.h"

// ── JBU ──────────────────────────────────────────────────────────────────────
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
            int hx = min(hrW - 1, (int)((kx2 + 0.5f) * hrW / lrW));
            int hy = min(hrH - 1, (int)((ky2 + 0.5f) * hrH / lrH));
            const unsigned char* gq = g + hy*hrS + hx*4;
            float dr = rP - gq[2]*(1.f/255.f);
            float dg = gP - gq[1]*(1.f/255.f);
            float db = bP - gq[0]*(1.f/255.f);
            float dc2 = dr*dr + dg*dg + db*db;
            float w = ws * expf(-dc2 * inv2cs);
            wS += w;
            dS += w * dlr[ky2*lrW + kx2];
        }
    }
    int nr = max(0, min(lv0, lrH-1)) * lrW + max(0, min(lu0, lrW-1));
    dhr[oy*hrW + ox] = (wS > 1e-8f) ? dS / wS : dlr[nr];
}

int jbu_cuda(const float* dlr, int lrW, int lrH,
             const unsigned char* g, int hrW, int hrH, int hrS,
             float* dhr, float ss, float sc, int radius,
             float*, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    dim3 b(16, 16);
    k_jbu<<<dim3((hrW+15)/16,(hrH+15)/16),b,0,st>>>(
        dlr, g, hrW, hrH, hrS, lrW, lrH, dhr,
        1.f/(2.f*ss*ss), 1.f/(2.f*sc*sc), radius);
    return (int)cudaGetLastError();
}

// ── WMF ──────────────────────────────────────────────────────────────────────
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
            int bin = max(0, min(WMF_BINS-1, (int)(dval*(WMF_BINS-1)+0.5f)));
            hist[bin] += w;
        }
    }

    int bestBin = 0; float bestW = hist[0];
#pragma unroll
    for (int i = 1; i < WMF_BINS; ++i)
        if (hist[i] > bestW) { bestW = hist[i]; bestBin = i; }

    const float binWidth   = 1.0f / (WMF_BINS - 1);
    const float modeCenter = bestBin * binWidth;

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
            wS += w; dS += w * dval;
        }
    }
    dhr[oy*hrW + ox] = (wS > 1e-8f) ? dS / wS : modeCenter;
}

int wmf_cuda(const float* dlr, int lrW, int lrH,
             const unsigned char* g, int hrW, int hrH, int hrS,
             float* dhr, float ss, float sc, int radius,
             float*, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    dim3 b(16, 16);
    k_wmf<<<dim3((hrW+15)/16,(hrH+15)/16),b,0,st>>>(
        dlr, g, hrW, hrH, hrS, lrW, lrH, dhr,
        1.f/(2.f*ss*ss), 1.f/(2.f*sc*sc), radius);
    return (int)cudaGetLastError();
}

// ── Bilinear ──────────────────────────────────────────────────────────────────
__global__ void k_bilinear(
    const float* __restrict__ dlr, int lrW, int lrH,
    float* __restrict__ dhr, int hrW, int hrH)
{
    int ox = blockIdx.x*blockDim.x + threadIdx.x;
    int oy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ox >= hrW || oy >= hrH) return;
    float fu = (ox+0.5f)*lrW/(float)hrW - 0.5f;
    float fv = (oy+0.5f)*lrH/(float)hrH - 0.5f;
    int lx0 = max(0, min((int)floorf(fu), lrW-1));
    int ly0 = max(0, min((int)floorf(fv), lrH-1));
    float tx = fmaxf(0.f, fu - lx0);
    float ty = fmaxf(0.f, fv - ly0);
    int lx1 = min(lx0+1, lrW-1), ly1 = min(ly0+1, lrH-1);
    dhr[oy*hrW+ox] = dlr[ly0*lrW+lx0]*(1-tx)*(1-ty)
                   + dlr[ly0*lrW+lx1]*tx*(1-ty)
                   + dlr[ly1*lrW+lx0]*(1-tx)*ty
                   + dlr[ly1*lrW+lx1]*tx*ty;
}

int bilinear_cuda(const float* dlr, int lrW, int lrH,
                   float* dhr, int hrW, int hrH, void* stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    k_bilinear<<<dim3((hrW+15)/16,(hrH+15)/16),dim3(16,16),0,st>>>(
        dlr, lrW, lrH, dhr, hrW, hrH);
    return (int)cudaGetLastError();
}

// ── Normalize ─────────────────────────────────────────────────────────────────
__device__ __forceinline__ void atomicMinFloat(float* addr, float val) {
    int* ia = (int*)addr; int old = *ia, assumed;
    do { assumed = old; if (__int_as_float(assumed) <= val) break;
         old = atomicCAS(ia, assumed, __float_as_int(val)); } while (assumed != old);
}
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    int* ia = (int*)addr; int old = *ia, assumed;
    do { assumed = old; if (__int_as_float(assumed) >= val) break;
         old = atomicCAS(ia, assumed, __float_as_int(val)); } while (assumed != old);
}
__global__ void k_minmax_init(float* mm) { mm[0]=3e38f; mm[1]=-3e38f; }
__global__ void k_minmax_reduce(const float* __restrict__ data, int n, float* mm) {
    __shared__ float smin[256], smax[256];
    int tid = threadIdx.x;
    float lmin=3e38f, lmax=-3e38f;
    for (int i = blockIdx.x*256+tid; i < n; i += blockDim.x*gridDim.x)
        { float v=data[i]; lmin=fminf(lmin,v); lmax=fmaxf(lmax,v); }
    smin[tid]=lmin; smax[tid]=lmax; __syncthreads();
    for (int s=128; s>0; s>>=1) {
        if (tid<s) { smin[tid]=fminf(smin[tid],smin[tid+s]);
                     smax[tid]=fmaxf(smax[tid],smax[tid+s]); }
        __syncthreads();
    }
    if (tid==0) { atomicMinFloat(&mm[0],smin[0]); atomicMaxFloat(&mm[1],smax[0]); }
}
__global__ void k_normalize(const float* __restrict__ raw, int n,
                              const float* __restrict__ mm, float* __restrict__ out) {
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    float range = (mm[1]-mm[0]) > 1e-6f ? (mm[1]-mm[0]) : 1e-6f;
    out[i] = (raw[i]-mm[0]) / range;
}
int normalize_depth_cuda(const float* raw, int n, float* mm, float* out, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    k_minmax_init<<<1,1,0,st>>>(mm);
    int blk = min(256,(n+255)/256);
    k_minmax_reduce<<<blk,256,0,st>>>(raw,n,mm);
    k_normalize<<<(n+255)/256,256,0,st>>>(raw,n,mm,out);
    return (int)cudaGetLastError();
}

// ── Max-dilation (Bilinear / JBU) ────────────────────────────────────────────
__global__ void k_dilate_h(const float* __restrict__ in, float* __restrict__ out,
                             int w, int h, int radius, float edgeThresh, float dirSign)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y;
    if (x>=w||y>=h) return;
    float center=in[y*w+x], best=center;
    for (int xi=max(0,x-radius); xi<=min(w-1,x+radius); ++xi) {
        float v=in[y*w+xi];
        if (dirSign*v>dirSign*best && dirSign*(v-center)>=edgeThresh) best=v;
    }
    out[y*w+x]=best;
}
__global__ void k_dilate_v(const float* __restrict__ in, float* __restrict__ out,
                             int w, int h, int radius, float edgeThresh, float dirSign)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y;
    if (x>=w||y>=h) return;
    float center=in[y*w+x], best=center;
    for (int yi=max(0,y-radius); yi<=min(h-1,y+radius); ++yi) {
        float v=in[yi*w+x];
        if (dirSign*v>dirSign*best && dirSign*(v-center)>=edgeThresh) best=v;
    }
    out[y*w+x]=best;
}
int gpu_dilate(const float* src, float* tmp, float* dst,
               int w, int h, int radius, float edgeThresh,
               bool flipped, void* stream)
{
    cudaStream_t st=(cudaStream_t)stream;
    float ds=flipped?-1.f:1.f;
    dim3 block(256),grid((w+255)/256,h);
    k_dilate_h<<<grid,block,0,st>>>(src,tmp,w,h,radius,edgeThresh,ds);
    k_dilate_v<<<grid,block,0,st>>>(tmp,dst,w,h,radius,edgeThresh,ds);
    return (int)cudaGetLastError();
}

// ── WMF boundary-shift dilation ───────────────────────────────────────────────
// Nearest-neighbour search outward; adopts the first qualifying neighbour's
// ACTUAL value rather than the window max — shifts the edge boundary by
// ~radius pixels without pulling in unrelated objects' depth values.
__global__ void k_wmf_dilate_h(const float* __restrict__ in, float* __restrict__ out,
                                 int w, int h, int radius, float edgeThresh, float dirSign)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y;
    if (x>=w||y>=h) return;
    float center=in[y*w+x], result=center;
    for (int d=1; d<=radius; ++d) {
        int xl=x-d, xr=x+d;
        bool gotL=false, gotR=false; float vl=0.f, vr=0.f;
        if (xl>=0) { vl=in[y*w+xl]; gotL=dirSign*(vl-center)>=edgeThresh; }
        if (xr< w) { vr=in[y*w+xr]; gotR=dirSign*(vr-center)>=edgeThresh; }
        if (gotL&&gotR) { result=(dirSign*vl>=dirSign*vr)?vl:vr; break; }
        else if (gotL)  { result=vl; break; }
        else if (gotR)  { result=vr; break; }
    }
    out[y*w+x]=result;
}
__global__ void k_wmf_dilate_v(const float* __restrict__ in, float* __restrict__ out,
                                 int w, int h, int radius, float edgeThresh, float dirSign)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y;
    if (x>=w||y>=h) return;
    float center=in[y*w+x], result=center;
    for (int d=1; d<=radius; ++d) {
        int yt=y-d, yb=y+d;
        bool gotT=false, gotB=false; float vt=0.f, vb=0.f;
        if (yt>=0) { vt=in[yt*w+x]; gotT=dirSign*(vt-center)>=edgeThresh; }
        if (yb< h) { vb=in[yb*w+x]; gotB=dirSign*(vb-center)>=edgeThresh; }
        if (gotT&&gotB) { result=(dirSign*vt>=dirSign*vb)?vt:vb; break; }
        else if (gotT)  { result=vt; break; }
        else if (gotB)  { result=vb; break; }
    }
    out[y*w+x]=result;
}
int wmf_dilate_cuda(const float* src, float* tmp, float* dst,
                     int w, int h, int radius, float edgeThresh,
                     bool flipped, void* stream)
{
    cudaStream_t st=(cudaStream_t)stream;
    float ds=flipped?-1.f:1.f;
    dim3 block(256),grid((w+255)/256,h);
    k_wmf_dilate_h<<<grid,block,0,st>>>(src,tmp,w,h,radius,edgeThresh,ds);
    k_wmf_dilate_v<<<grid,block,0,st>>>(tmp,dst,w,h,radius,edgeThresh,ds);
    return (int)cudaGetLastError();
}
