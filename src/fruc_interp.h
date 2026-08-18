// SPDX-License-Identifier: GPL-3.0-or-later
// fruc_interp.h — NvFRUC-based depth map interpolation for 3Deflatten.
//
// Dynamically loads NvFRUC.dll at runtime; no compile-time NvFRUC SDK dependency.
// Falls back silently (no interpolation) if the DLL is not present or the GPU
// does not have hardware optical flow cores (Maxwell or older).
//
// Architecture
// ─────────────
// When m_skipEvery > 1 (inference slower than source FPS), the depth worker
// creates S-1 = m_skipEvery-1 FRUC instances, one per interpolated frame.
// Each instance runs on NVOFA dedicated silicon, which is independent from the
// CUDA cores used by TRT inference — so interpolation and the next inference
// run simultaneously on different hardware.
//
// Depth encoding
// ──────────────
// FRUC requires NV12 or ARGB surfaces.  We use ARGB with 16-bit precision:
//   • B = 0, G = low8(depth_u16), R = high8(depth_u16), A = 0xFF
//   • depth_u16 = round(clamp(depth, 0,1) * 65535)
// This encodes depth losslessly at ~0.0015 % precision, far better than 8-bit
// (0.4 %), without colour information that could confuse motion estimation.
//
// Future reuse for RGB interpolation
// ────────────────────────────────────
// Swap the source frames from LR-depth ARGB to RGB ARGB — the FRUC pipeline
// and the bidirectional warping are identical.
#pragma once
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

struct NvFRUC_Slot; // defined in fruc_interp.cpp

class FRUCDepthInterp {
public:
    FRUCDepthInterp();
    ~FRUCDepthInterp();

    // Initialise for given depth-map dimensions.
    // maxInterp: maximum number of simultaneously needed interpolated frames
    //            (= max expected m_skipEvery - 1; typically 1-7).
    // Returns true if NvFRUC.dll was found and the GPU supports NvOFA.
    bool Init(int depthW, int depthH, int maxInterp = 7);

    // Reconfigure slot count (cheap if numInterp ≤ already allocated).
    // Called whenever m_skipEvery changes.
    bool SetInterpCount(int numInterp);

    // Interpolate numInterp frames between prevDepth[0..W*H-1] and currDepth.
    // Both inputs must be in [0,1].  Blocking — intended to be called from a
    // detached worker thread so inference continues on the inference stream.
    // Returns numInterp depth maps in temporal order (t = 1/S, 2/S, …, (S-1)/S).
    std::vector<std::vector<float>> Interpolate(
        const float* prevDepth, const float* currDepth, int numInterp);

    bool IsAvailable() const { return m_ready; }
    int  W()           const { return m_w; }
    int  H()           const { return m_h; }

private:
    bool LoadFRUC();
    bool AllocateSlots(int count);
    void FreeSlots();
    void FreeShared();

    bool m_ready = false;
    int  m_w = 0, m_h = 0;
    int  m_activeSlots = 0;

    // DLL function pointers (cast to the real types in .cpp)
    void* m_hDLL       = nullptr;
    void* m_fnCreate   = nullptr;
    void* m_fnRegRes   = nullptr;
    void* m_fnProcess  = nullptr;
    void* m_fnUnregRes = nullptr;
    void* m_fnDestroy  = nullptr;

    // One slot per interpolated frame
    std::vector<NvFRUC_Slot*> m_slots;

    // Shared ARGB device buffers (uploaded once per batch, read by every slot)
    uint8_t* m_d_prevARGB = nullptr; // device: W*H*4
    uint8_t* m_d_currARGB = nullptr;
    size_t   m_argbBytes  = 0;

    cudaStream_t m_uploadStream = nullptr;
};

// CUDA kernel wrappers (compiled in fruc_interp.cu)
void depth_to_argb_cuda(const float*   d_depth, uint8_t* d_argb, int n, cudaStream_t s);
void argb_to_depth_cuda(const uint8_t* d_argb,  float*   d_depth, int n, cudaStream_t s);
