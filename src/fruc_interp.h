// SPDX-License-Identifier: GPL-3.0-or-later
// fruc_interp.h — NvOFFRUC depth-map frame interpolation.
//
// Uses NvOFFRUC.dll (NvOFFRUC SDK) loaded at runtime from the filter's own
// directory — no compile-time dependency on any NVIDIA SDK.  All CUDA memory
// management uses the CUDA Driver API (nvcuda.dll) loaded dynamically, so
// there are zero static DLL imports for CUDA in this module; it cannot cause
// a DLL-loader stall even when cudart is not on PATH.
//
// Lifetime: created lazily on the depth-worker thread after the first
// successful inference (CUDA context already exists by then).  Destroyed when
// the filter is torn down.  Safe to call Interpolate() from any thread.
#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <memory>

class FRUCDepthInterp {
public:
    FRUCDepthInterp();
    ~FRUCDepthInterp();

    // Init for given LR depth dimensions.  Loads NvOFFRUC.dll + nvcuda.dll
    // from dllDir (Win64 folder).  maxInterp = max m_skipEvery-1 (typically ≤7).
    // Returns true only if both DLLs load and GPU supports NVOFA.
    bool Init(int w, int h, int maxInterp, const std::wstring& dllDir);

    // Reconfigure slot count when m_skipEvery changes (cheap if numInterp ≤ alloc'd).
    void SetInterpCount(int numInterp);

    // Generate numInterp frames between prevDepth[w*h] and currDepth[w*h] (values
    // in [0,1]).  Blocking: intended for a detached worker thread so inference
    // continues on NVOFA hardware in parallel with CUDA cores.
    // Returns numInterp float[w*h] maps in temporal order.
    std::vector<std::vector<float>> Interpolate(
        const float* prevDepth, const float* currDepth, int numInterp);

    bool IsAvailable() const { return m_ready; }
    int  W() const { return m_w; }
    int  H() const { return m_h; }

private:
    // Internal helpers (implemented in fruc_interp.cpp)
    bool LoadNvOFFRUC(const std::wstring& dir);
    bool LoadCUDA(const std::wstring& dir);
    bool AllocBuffers();
    void FreeBuffers();
    bool CreateSlot(int idx);
    void DestroySlot(int idx);

    // ── CUDA Driver API function pointers (from nvcuda.dll) ─────────────────
    void* m_hCUDA = nullptr;   // HMODULE nvcuda.dll
    typedef unsigned long long CUdeviceptr_t;
    typedef int (*PFN_cuInit)(unsigned int);
    typedef int (*PFN_cuMemAlloc)(CUdeviceptr_t*, size_t);
    typedef int (*PFN_cuMemFree)(CUdeviceptr_t);
    typedef int (*PFN_cuMemcpyHtoD)(CUdeviceptr_t, const void*, size_t);
    typedef int (*PFN_cuMemcpyDtoH)(void*, CUdeviceptr_t, size_t);
    PFN_cuInit      m_cuInit      = nullptr;
    PFN_cuMemAlloc  m_cuMemAlloc  = nullptr;
    PFN_cuMemFree   m_cuMemFree   = nullptr;
    PFN_cuMemcpyHtoD m_cuHtoD     = nullptr;
    PFN_cuMemcpyDtoH m_cuDtoH     = nullptr;

    // ── NvOFFRUC function pointers (from NvOFFRUC.dll) ──────────────────────
    void* m_hFRUC = nullptr;   // HMODULE NvOFFRUC.dll
    typedef void* NvOFFRUCHandle;
    typedef int (*PFN_Create)(void*, NvOFFRUCHandle*);
    typedef int (*PFN_RegRes)(NvOFFRUCHandle, void*);
    typedef int (*PFN_Process)(NvOFFRUCHandle, void*, void*);
    typedef int (*PFN_UnregRes)(NvOFFRUCHandle, void*);
    typedef int (*PFN_Destroy)(NvOFFRUCHandle);
    PFN_Create   m_fnCreate   = nullptr;
    PFN_RegRes   m_fnRegRes   = nullptr;
    PFN_Process  m_fnProcess  = nullptr;
    PFN_UnregRes m_fnUnregRes = nullptr;
    PFN_Destroy  m_fnDestroy  = nullptr;

    // ── Per-interpolated-frame slot ──────────────────────────────────────────
    struct Slot {
        NvOFFRUCHandle hFRUC   = nullptr;
        CUdeviceptr_t  d_out   = 0;      // ARGB output device buffer
        void*          h_out   = nullptr; // host readback (malloc)
        bool           regOk   = false;
        void*          res[3]  = {};      // ptrs registered with FRUC: [out, prev, curr]
    };

    // Shared ARGB device buffers (written once per batch, read by all slots)
    CUdeviceptr_t  m_d_prev = 0;
    CUdeviceptr_t  m_d_curr = 0;
    size_t         m_argbBytes = 0;

    std::vector<Slot> m_slots;
    int  m_w = 0, m_h = 0;
    bool m_ready = false;
};

// CUDA kernel wrappers — defined in fruc_interp.cu (compiled with nvcc).
// These are the ONLY symbols that link against the CUDA runtime; everything
// else in fruc_interp.cpp uses the Driver API loaded at runtime.
void fruc_depth_to_argb(const float*   d_depth,
                         uint8_t* d_argb, int n, void* cuStream);
void fruc_argb_to_depth(const uint8_t* d_argb,
                         float*   d_depth, int n, void* cuStream);