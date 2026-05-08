// SPDX-License-Identifier: GPL-3.0-or-later
// jbu_cuda.cu — CUDA Joint Bilateral Upsampling
//
// Change from original: the low-res guide luma buffer (glr) is now supplied
// by the caller (pre-allocated in TrtRtxSession) instead of being allocated
// here per call.  The old per-call cudaMalloc caused cudaError=1 whenever
// a sticky error existed in the CUDA context from a preceding executeV2()
// async launch, which silently fell through to the 1 FPS CPU fallback.
#include <cuda_runtime.h>
#include "jbu_cuda.h"

// Pass 1: downsample guide BGRA to low-res luma [0,1] at model-grid centres
__global__ void k_guide_lr(
    const unsigned char* __restrict__ g, int hrW, int hrH, int hrS,
    float* __restrict__ glr, int lrW, int lrH)
{
    int kx = blockIdx.x*blockDim.x + threadIdx.x;
    int ky = blockIdx.y*blockDim.y + threadIdx.y;
    if (kx >= lrW || ky >= lrH) return;
    float fx = (kx+.5f)*hrW/(float)lrW - .5f;
    float fy = (ky+.5f)*hrH/(float)lrH - .5f;
    int x0=max(0,(int)fx), x1=min(hrW-1,x0+1);
    int y0=max(0,(int)fy), y1=min(hrH-1,y0+1);
    float tx=fx-x0, ty=fy-y0;
    auto L = [&](int gx, int gy) -> float {
        const unsigned char* p = g + gy*hrS + gx*4;
        return (29*p[0] + 150*p[1] + 77*p[2]) * (1.f/65280.f);
    };
    glr[ky*lrW+kx] = L(x0,y0)*(1-tx)*(1-ty) + L(x1,y0)*tx*(1-ty)
                   + L(x0,y1)*(1-tx)*   ty   + L(x1,y1)*tx*   ty;
}

// Pass 2: bilateral upsample — one thread per output pixel
__global__ void k_jbu(
    const float* __restrict__ dlr, const float* __restrict__ glr,
    const unsigned char* __restrict__ g, int hrW, int hrH, int hrS,
    int lrW, int lrH, float* __restrict__ dhr,
    float inv2ss, float inv2cs, int radius)
{
    int ox = blockIdx.x*blockDim.x + threadIdx.x;
    int oy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ox >= hrW || oy >= hrH) return;
    const unsigned char* gp = g + oy*hrS + ox*4;
    float gP = (29*gp[0] + 150*gp[1] + 77*gp[2]) * (1.f/65280.f);
    float lu = (ox+.5f)*lrW/(float)hrW - .5f;
    float lv = (oy+.5f)*lrH/(float)hrH - .5f;
    int lu0=(int)lu, lv0=(int)lv;
    float wS=0.f, dS=0.f;
    for (int ky = lv0-radius; ky <= lv0+radius+1; ++ky) {
        int ky2 = max(0, min(ky, lrH-1));
        float dv = lv - ky;
        for (int kx = lu0-radius; kx <= lu0+radius+1; ++kx) {
            int kx2 = max(0, min(kx, lrW-1));
            float du = lu - kx;
            float ws = expf(-(du*du + dv*dv)*inv2ss);
            float dc = gP - glr[ky2*lrW + kx2];
            float w  = ws * expf(-dc*dc*inv2cs);
            wS += w;
            dS += w * dlr[ky2*lrW + kx2];
        }
    }
    int nr = max(0,min(lv0,lrH-1))*lrW + max(0,min(lu0,lrW-1));
    dhr[oy*hrW + ox] = (wS > 1e-8f) ? dS/wS : dlr[nr];
}

// guide_lr_dev: pre-allocated device buffer (lrW*lrH floats), owned by caller.
// No cudaMalloc/cudaFree here — that was the root cause of cudaError=1.
int jbu_cuda(const float* dlr, int lrW, int lrH,
             const unsigned char* g, int hrW, int hrH, int hrS,
             float* dhr, float ss, float sc, int radius,
             float* glr,    // caller-allocated device buffer, lrW*lrH floats
             void*  stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    dim3 b(16, 16);
    k_guide_lr<<<dim3((lrW+15)/16,(lrH+15)/16), b, 0, st>>>(
        g, hrW, hrH, hrS, glr, lrW, lrH);
    float i2ss = 1.f/(2.f*ss*ss), i2cs = 1.f/(2.f*sc*sc);
    k_jbu<<<dim3((hrW+15)/16,(hrH+15)/16), b, 0, st>>>(
        dlr, glr, g, hrW, hrH, hrS, lrW, lrH, dhr, i2ss, i2cs, radius);
    // Caller owns sync + error handling
    return (int)cudaGetLastError();
}
