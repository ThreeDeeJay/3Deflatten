// SPDX-License-Identifier: GPL-3.0-or-later
// fruc_interp.cu — CUDA kernels for depth ↔ ARGB conversion used by FRUCDepthInterp.
//
// Depth is encoded as 16-bit across R (high) + G (low) channels of an ARGB
// surface so FRUC's NVOFA-based optical flow can track depth gradients with
// sub-percent precision.  B is zeroed, A is 0xFF (opaque).
#include <cuda_runtime.h>
#include <cstdint>

__global__ void k_depth_to_argb(const float* __restrict__ d,
                                  uint8_t* __restrict__ argb, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    uint16_t u = (uint16_t)(__saturatef(d[i])*65535.f+.5f);
    argb[i*4+0] = 0;         // B (unused)
    argb[i*4+1] = u & 0xFF;  // G = low byte
    argb[i*4+2] = u >> 8;    // R = high byte
    argb[i*4+3] = 0xFF;      // A
}
__global__ void k_argb_to_depth(const uint8_t* __restrict__ argb,
                                  float* __restrict__ d, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    uint16_t u = ((uint16_t)argb[i*4+2]<<8) | argb[i*4+1]; // R<<8 | G
    d[i] = u/65535.f;
}
void depth_to_argb_cuda(const float* d_d, uint8_t* d_a, int n, cudaStream_t s) {
    k_depth_to_argb<<<(n+255)/256,256,0,s>>>(d_d,d_a,n);
}
void argb_to_depth_cuda(const uint8_t* d_a, float* d_d, int n, cudaStream_t s) {
    k_argb_to_depth<<<(n+255)/256,256,0,s>>>(d_a,d_d,n);
}
