// SPDX-License-Identifier: GPL-3.0-or-later
// jbu_cuda.h — CUDA-accelerated guided depth upscaling (JBU + WMF) + GPU dilation
#pragma once
#include <cstdint>

// Joint Bilateral Upsampling.  Weighted average of nearby low-res depth
// samples, weighted by spatial distance and full RGB (luma+chroma) colour
// distance against the guide, sampling the guide directly at full resolution
// (not a blurred low-res guide average).
// guide_lr_dev: pre-allocated device float buffer (lrW*lrH), caller-owned,
//               currently unused (kept for buffer-reuse compatibility).
int jbu_cuda(const float*         depth_lr,
             int lrW, int lrH,
             const unsigned char* guide_bgra,
             int hrW, int hrH, int hrStride,
             float*               depth_hr,
             float sigma_s, float sigma_c, int radius,
             float*               guide_lr_dev,
             void*                stream);

// Weighted Mode Filtering (Min, Lu & Do, IEEE TIP 2012).
// Sharper alternative to JBU: builds a weighted histogram of nearby low-res
// depth samples, finds the dominant bin (the "mode"), then averages only the
// samples that fall within it. Wrong-side-of-an-edge samples are excluded
// entirely rather than merely down-weighted, so there is no blend left to
// glow. sigma_s/sigma_c/radius here are fixed quality/sharpness parameters,
// NOT a dilation control — see wmf_dilate_cuda() below for actual dilation.
// guide_lr_dev is likewise unused.
int wmf_cuda(const float*         depth_lr,
             int lrW, int lrH,
             const unsigned char* guide_bgra,
             int hrW, int hrH, int hrStride,
             float*               depth_hr,
             float sigma_s, float sigma_c, int radius,
             float*               guide_lr_dev,
             void*                stream);

// Plain GPU bilinear upscale — no guide needed. Keeps the pipeline fully
// GPU-resident for the "Bilinear" mode instead of falling back to a slow
// single-threaded CPU resize that was paradoxically slower than JBU/WMF.
int bilinear_cuda(const float* depth_lr, int lrW, int lrH,
                   float* depth_hr, int hrW, int hrH,
                   void* stream);

// Min/max-normalises `n` raw depth values into [0,1], writing the result to
// `out`. Required before wmf_cuda(): WMF's histogram binning assumes input
// depth is already in [0,1], but raw TensorRT model output is not.
// mm_scratch: pre-allocated 2-float device buffer (caller-owned, reused).
int normalize_depth_cuda(const float* raw, int n,
                          float* mm_scratch, float* out,
                          void* stream);

// Separable morphological max-dilation on the GPU (for Bilinear / JBU output).
//   src/tmp/dst : device float[w*h]
//   edgeThresh  : only propagate values where delta >= threshold
//   flipped     : false = expand HIGH values (normal "depth=1=near");
//                 true  = expand LOW values. Must reflect the PRE-flip
//                 polarity since this runs before the CPU-side flip is
//                 applied in the collect phase.
// Returns cudaGetLastError() (0 = success).
int gpu_dilate(const float* src, float* tmp, float* dst,
               int w, int h, int radius, float edgeThresh,
               bool flipped, void* stream);

// WMF-specific dilation: separable two-pass NEAREST-NEIGHBOUR boundary
// shift. For each background pixel, scans outward by increasing distance
// for the nearest qualifying foreground neighbour and ADOPTS its real value
// (rather than taking the max across the whole window). This shifts the
// detected edge boundary outward by close to exactly `radius` pixels while
// keeping a single nearby object's true depth value — "shifting the
// boundary" rather than "naively adding whichever is numerically highest".
// Use this (not gpu_dilate) when upscaleMode == WeightedMode.
//   src/tmp/dst : device float[w*h], full source resolution
//   edgeThresh  : only adopt a neighbour where |delta| >= threshold
//   flipped     : same meaning/requirement as gpu_dilate's `flipped`.
// Returns cudaGetLastError() (0 = success).
int wmf_dilate_cuda(const float* src, float* tmp, float* dst,
                     int w, int h, int radius, float edgeThresh,
                     bool flipped, void* stream);