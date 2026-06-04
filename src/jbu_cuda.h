// SPDX-License-Identifier: GPL-3.0-or-later
// jbu_cuda.h — CUDA-accelerated Joint Bilateral Upsampling + GPU dilation
#pragma once
#include <cstdint>

// Joint Bilateral Upsampling.
// guide_lr_dev: pre-allocated device float buffer (lrW*lrH), caller-owned.
int jbu_cuda(const float*         depth_lr,
             int lrW, int lrH,
             const unsigned char* guide_bgra,
             int hrW, int hrH, int hrStride,
             float*               depth_hr,
             float sigma_s, float sigma_c, int radius,
             float*               guide_lr_dev,
             void*                stream);

// Separable morphological max-dilation on the GPU.
//   src/tmp/dst : device float[w*h]
//   edgeThresh  : only propagate values where delta >= threshold
// Returns cudaGetLastError() (0 = success).
int gpu_dilate(const float* src, float* tmp, float* dst,
               int w, int h, int radius, float edgeThresh,
               void* stream);