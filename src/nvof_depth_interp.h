// SPDX-License-Identifier: GPL-3.0-or-later
// nvof_depth_interp.h — NvOF hardware optical flow depth interpolation.
// All CUDA types hidden behind void*; safe to include from non-CUDA .cpp files.
// nvofapi64.dll loaded at runtime from Win64 folder — no static imports.
#pragma once
#include <cstdint>
#include <string>

struct NvOFState; // opaque; defined in nvof_depth_interp.cu

// Create NvOF state for depth maps of size w×h.
// dllDir = Win64 folder path (nvofapi64.dll is loaded from there).
// Returns nullptr if NvOF is unavailable on this GPU/driver.
NvOFState* nvof_create(int w, int h, int maxInterp, const std::wstring& dllDir);
void       nvof_destroy(NvOFState* st);
bool       nvof_available(NvOFState* st);

// (1) Prepare both guide frames and LR depths for this pipeline slot.
// guideBGRA: source BGRA [srcW×srcH] already on device (d_guideBGRA[writeBuf]).
// dOutSlice: raw TRT output [mw×mh] on device (unnormalized inverse depth).
// Downsamples guideBGRA → GRAYSCALE8 at [mw×mh], normalises dOutSlice → [0,1].
// Call on ofStream after inferDone event (both inputs are ready at that point).
void nvof_prepare_slot(NvOFState* st, int slot,
                       const uint8_t* d_guideBGRA, int srcW, int srcH, int srcStride,
                       const float* d_outSlice, float* d_minmax_scratch,
                       int mw, int mh, void* stream);

// (2) Execute NvOF between slot prevSlot and slot currSlot (0-based ping-pong).
// Stores flow in internal buffer.  Call after both slots have been prepared.
bool nvof_execute(NvOFState* st, int prevSlot, int currSlot, void* stream);

// (3) Warp and blend to generate interpolated depth at t ∈ (0,1).
//   t = k / skipEvery  for k = 1 .. skipEvery-1
// d_out must be float[mw*mh] (device).  Call on same stream, after nvof_execute.
void nvof_warp(NvOFState* st, int prevSlot, int currSlot,
               float* d_out, float t, void* stream);

// Dimensions of the LR depth managed by this state
void nvof_dims(NvOFState* st, int* w, int* h);
