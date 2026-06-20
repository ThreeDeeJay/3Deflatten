// SPDX-License-Identifier: GPL-3.0-or-later
// jbu_cuda.h — CUDA-accelerated guided depth upscaling (JBU + WMF) + GPU dilation
#pragma once
#include <cstdint>

// Joint Bilateral Upsampling.  Weighted average of nearby low-res depth
// samples, weighted by spatial distance and full RGB (luma+chroma) colour
// distance against the guide.  Sharper than plain bilinear, but — being a
// continuous weighted blend — can still show faint glow/feathering at edges,
// since cross-edge samples always contribute *some* nonzero weight.
// guide_lr_dev: pre-allocated device float buffer (lrW*lrH), caller-owned.
//               Currently unused by the implementation (kept for ABI/buffer-
//               reuse compatibility with the caller's allocation lifecycle).
int jbu_cuda(const float*         depth_lr,
             int lrW, int lrH,
             const unsigned char* guide_bgra,
             int hrW, int hrH, int hrStride,
             float*               depth_hr,
             float sigma_s, float sigma_c, int radius,
             float*               guide_lr_dev,
             void*                stream);

// Weighted Mode Filtering (Min, Lu & Do, "Depth Video Enhancement Based on
// Weighted Mode Filtering", IEEE Trans. Image Processing 2012).
// Sharper alternative to JBU: builds a weighted histogram of nearby low-res
// depth samples (same spatial + full-RGB colour weighting as JBU), finds the
// dominant bin (the "mode"), then averages only the samples that fall within
// it. Wrong-side-of-an-edge samples are excluded entirely rather than merely
// down-weighted, so there is no blend left to glow — the output snaps to a
// hard transition aligned with the guide's RGB edge.
// dilateBias [0,1]: among histogram bins whose weight is within
//   (dilateBias*100)% of the winning bin's weight, prefer the HIGHEST-depth
//   (nearest) one instead of strictly the single best-supported one. This
//   grows the foreground class natively, within WMF's own RGB-guided
//   neighbourhood, instead of stacking a separate box-shaped max-dilate on
//   top of WMF's already-sharp output (which tends to look blockier).
//   0 = no bias (original strict-mode behaviour).
// Same signature shape as jbu_cuda() (plus dilateBias) so call sites can
// switch between the two with minimal changes. guide_lr_dev is unused.
int wmf_cuda(const float*         depth_lr,
             int lrW, int lrH,
             const unsigned char* guide_bgra,
             int hrW, int hrH, int hrStride,
             float*               depth_hr,
             float sigma_s, float sigma_c, int radius,
             float                dilateBias,
             float*               guide_lr_dev,
             void*                stream);

// Plain GPU bilinear upscale — no guide needed. Use this (not a CPU resize)
// for the "off" / Bilinear case: running the upscale on GPU keeps the whole
// pipeline GPU-resident regardless of which algorithm is selected, instead
// of leaving the GPU idle while the CPU does a slow full-resolution resize.
int bilinear_cuda(const float* depth_lr, int lrW, int lrH,
                   float* depth_hr, int hrW, int hrH,
                   void* stream);

// Min/max-normalises `n` raw depth values into [0,1], writing the result to
// `out`. Required before wmf_cuda(): WMF's histogram binning assumes input
// depth is already in [0,1], but raw TensorRT model output is not bounded to
// that range. JBU does not need this (it is a pure linear weighted average,
// so it stays correct under any input scale).
// mm_scratch: pre-allocated 2-float device buffer (caller-owned, reused
//             across calls — see TrtRtxSession::d_minmax).
int normalize_depth_cuda(const float* raw, int n,
                          float* mm_scratch, float* out,
                          void* stream);

// Separable morphological max-dilation on the GPU.
//   src/tmp/dst : device float[w*h]
//   edgeThresh  : only propagate values where delta >= threshold
//   flipped     : false = expand HIGH values (the normal "depth=1=near"
//                 convention); true = expand LOW values instead.
//                 IMPORTANT: this must reflect the polarity the data will
//                 have AFTER any flipDepth correction, since dilation only
//                 makes sense relative to "which direction is near". This
//                 kernel runs BEFORE the (CPU-side, post-readback) flip is
//                 applied, so the caller must pass flipDepth here directly —
//                 dilating high-then-flipping silently inverts the effect
//                 (foreground appears to shrink instead of expand).
// Returns cudaGetLastError() (0 = success).
int gpu_dilate(const float* src, float* tmp, float* dst,
               int w, int h, int radius, float edgeThresh,
               bool flipped, void* stream);