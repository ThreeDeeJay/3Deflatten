// SPDX-License-Identifier: GPL-3.0-or-later
// jbu_cuda.h — CUDA-accelerated Joint Bilateral Upsampling interface
#pragma once
#include <cstdint>
int jbu_cuda(const float* depth_lr, int lrW, int lrH,
             const unsigned char* guide_bgra, int hrW, int hrH, int hrStride,
             float* depth_hr, float sigma_s, float sigma_c, int radius, void* stream);
