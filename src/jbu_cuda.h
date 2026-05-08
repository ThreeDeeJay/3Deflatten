// SPDX-License-Identifier: GPL-3.0-or-later
// jbu_cuda.h — CUDA-accelerated Joint Bilateral Upsampling interface
#pragma once
#include <cstdint>

// guide_lr_dev: pre-allocated device float buffer of lrW*lrH elements.
//   Avoids per-frame cudaMalloc/cudaFree inside the kernel wrapper, which
//   caused cudaError=1 (cudaErrorInvalidValue) whenever a sticky error was
//   left in the CUDA context by a preceding executeV2() async launch.
//   The caller must keep it alive until cudaStreamSynchronize() returns.
int jbu_cuda(const float*         depth_lr,
             int lrW, int lrH,
             const unsigned char* guide_bgra,
             int hrW, int hrH, int hrStride,
             float*               depth_hr,
             float sigma_s, float sigma_c, int radius,
             float*               guide_lr_dev,   // pre-allocated, lrW*lrH floats
             void*                stream);
