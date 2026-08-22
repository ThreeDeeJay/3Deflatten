// SPDX-License-Identifier: GPL-3.0-or-later
// fruc_interp.cpp — FRUCDepthInterp implementation.
#include "fruc_interp.h"
#include "logger.h"
#include <cstring>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#define LOAD_LIB(n)    LoadLibraryA(n)
#define GET_PROC(h,n)  (void*)GetProcAddress((HMODULE)(h),n)
#define FREE_LIB(h)    FreeLibrary((HMODULE)(h))
#else
#include <dlfcn.h>
#define LOAD_LIB(n)    dlopen(n,RTLD_LAZY)
#define GET_PROC(h,n)  dlsym(h,n)
#define FREE_LIB(h)    dlclose(h)
#endif

// ── NvFRUC types (no SDK header dependency — loaded at runtime) ───────────────
typedef void* NvFRUCHandle;
#define NvFRUC_SUCCESS 0
#define NvFRUC_MAX_RES 5
enum eNvFRUCResType    { eNvFRUC_CUDA=0, eNvFRUC_D3D11=1 };
enum eNvFRUCSurfFmt    { eNvFRUC_NV12=0, eNvFRUC_ARGB=1   };
enum eNvFRUCCUDAResType{ eNvFRUC_DEVPTR=0, eNvFRUC_ARRAY=1 };
struct NvFRUC_CREATE_PARAM {
    void* pDevice; uint32_t uiHeight,uiWidth;
    eNvFRUCResType eResourceType; eNvFRUCSurfFmt eSurfaceFormat;
    eNvFRUCCUDAResType eCUDAResourceType;
};
struct NvFRUC_REG_PARAM {
    void* pArrResource[NvFRUC_MAX_RES]; uint32_t uiCount; void* pFence;
};
struct NvFRUCFrameData { void* pFrame; long long nTimeStamp; uint32_t bRepeat,uSync; };
struct NvFRUC_PROC_IN  { NvFRUCFrameData stFrameDataInput; };
struct NvFRUC_PROC_OUT { void* pFrame; long long nTimeStamp; uint32_t bRepeat; };
struct NvFRUC_UNREG_PARAM { void* pArrResource[NvFRUC_MAX_RES]; uint32_t uiCount; };
using FnCreate   = int(*)(NvFRUC_CREATE_PARAM*,NvFRUCHandle*);
using FnRegRes   = int(*)(NvFRUCHandle,NvFRUC_REG_PARAM*);
using FnProcess  = int(*)(NvFRUCHandle,NvFRUC_PROC_IN*,NvFRUC_PROC_OUT*);
using FnUnregRes = int(*)(NvFRUCHandle,NvFRUC_UNREG_PARAM*);
using FnDestroy  = int(*)(NvFRUCHandle);

// ── Per-slot state ────────────────────────────────────────────────────────────
struct NvFRUC_Slot {
    NvFRUCHandle  hFRUC      = nullptr;
    uint8_t*      d_outARGB  = nullptr; // device: W*H*4 ARGB
    uint8_t*      h_outARGB  = nullptr; // pinned host readback
    float*        h_depth    = nullptr; // decoded float depth [W*H]
    bool          regOk      = false;
    // Pointers registered with FRUC (must outlive the handle)
    void*         res[3]     = {};      // &d_prevARGB, &d_currARGB, &d_outARGB
};

// ── Helpers ───────────────────────────────────────────────────────────────────
static inline FnCreate   fnC(void* p){ return (FnCreate)p;   }
static inline FnRegRes   fnR(void* p){ return (FnRegRes)p;   }
static inline FnProcess  fnP(void* p){ return (FnProcess)p;  }
static inline FnUnregRes fnU(void* p){ return (FnUnregRes)p; }
static inline FnDestroy  fnD(void* p){ return (FnDestroy)p;  }

// ── FRUCDepthInterp ───────────────────────────────────────────────────────────
FRUCDepthInterp::FRUCDepthInterp() = default;
FRUCDepthInterp::~FRUCDepthInterp() { FreeSlots(); FreeShared(); if(m_hDLL) FREE_LIB(m_hDLL); }

bool FRUCDepthInterp::LoadFRUC() {
#ifdef _WIN32
    const char* name = "NvFRUC.dll";
#else
    const char* name = "libNvFRUC.so";
#endif
    m_hDLL = LOAD_LIB(name);
    if (!m_hDLL) { LOG_INFO("FRUCDepthInterp: ",name," not found — depth interpolation disabled"); return false; }
    m_fnCreate   = GET_PROC(m_hDLL,"NvFRUCCreate");
    m_fnRegRes   = GET_PROC(m_hDLL,"NvFRUCRegisterResource");
    m_fnProcess  = GET_PROC(m_hDLL,"NvFRUCProcess");
    m_fnUnregRes = GET_PROC(m_hDLL,"NvFRUCUnregisterResource");
    m_fnDestroy  = GET_PROC(m_hDLL,"NvFRUCDestroy");
    if (!m_fnCreate||!m_fnRegRes||!m_fnProcess||!m_fnUnregRes||!m_fnDestroy) {
        LOG_WARN("FRUCDepthInterp: DLL entry points missing"); FREE_LIB(m_hDLL); m_hDLL=nullptr; return false;
    }
    return true;
}

bool FRUCDepthInterp::Init(int w, int h, int maxInterp) {
    if (!LoadFRUC()) return false;
    m_w = w; m_h = h;
    m_argbBytes = (size_t)w*h*4;
    cudaStreamCreate(&m_uploadStream);
    cudaMalloc(&m_d_prevARGB, m_argbBytes);
    cudaMalloc(&m_d_currARGB, m_argbBytes);
    if (!AllocateSlots(maxInterp)) return false;
    m_ready = true;
    LOG_INFO("FRUCDepthInterp: ready at ",w,"x",h," maxInterp=",maxInterp);
    return true;
}

bool FRUCDepthInterp::AllocateSlots(int count) {
    // Destroy any existing slots beyond 'count'
    for (int k=(int)m_slots.size()-1; k>=count; --k) {
        auto* sl = m_slots[k];
        if (sl->regOk && sl->hFRUC) {
            NvFRUC_UNREG_PARAM up{}; memcpy(up.pArrResource,sl->res,3*sizeof(void*)); up.uiCount=3;
            fnU(m_fnUnregRes)(sl->hFRUC,&up);
        }
        if (sl->hFRUC) fnD(m_fnDestroy)(sl->hFRUC);
        if (sl->d_outARGB) cudaFree(sl->d_outARGB);
        if (sl->h_outARGB) cudaFreeHost(sl->h_outARGB);
        if (sl->h_depth)   cudaFreeHost(sl->h_depth);
        delete sl; m_slots.pop_back();
    }
    // Create new slots up to 'count'
    while ((int)m_slots.size() < count) {
        auto* sl = new NvFRUC_Slot{};
        // Create FRUC instance for this slot
        NvFRUC_CREATE_PARAM cp{}; cp.uiWidth=(uint32_t)m_w; cp.uiHeight=(uint32_t)m_h;
        cp.eResourceType=eNvFRUC_CUDA; cp.eSurfaceFormat=eNvFRUC_ARGB; cp.eCUDAResourceType=eNvFRUC_DEVPTR;
        if (fnC(m_fnCreate)(&cp,&sl->hFRUC) != NvFRUC_SUCCESS) {
            LOG_WARN("FRUCDepthInterp: NvFRUCCreate failed at slot ",(int)m_slots.size());
            delete sl; return false;
        }
        // Allocate output ARGB device buffer + host buffers
        cudaMalloc(&sl->d_outARGB, m_argbBytes);
        cudaMallocHost(&sl->h_outARGB, m_argbBytes);
        cudaMallocHost(&sl->h_depth, (size_t)m_w*m_h*sizeof(float));
        // Register resources: [0]=prevARGB, [1]=currARGB, [2]=outARGB per slot
        sl->res[0]=&m_d_prevARGB; sl->res[1]=&m_d_currARGB; sl->res[2]=&sl->d_outARGB;
        NvFRUC_REG_PARAM rp{}; rp.uiCount=3;
        rp.pArrResource[0]=sl->res[0]; rp.pArrResource[1]=sl->res[1]; rp.pArrResource[2]=sl->res[2];
        sl->regOk = (fnR(m_fnRegRes)(sl->hFRUC,&rp) == NvFRUC_SUCCESS);
        if (!sl->regOk) LOG_WARN("FRUCDepthInterp: NvFRUCRegisterResource failed for slot ",(int)m_slots.size());
        m_slots.push_back(sl);
    }
    m_activeSlots = count;
    return true;
}

bool FRUCDepthInterp::SetInterpCount(int n) {
    if (!m_ready) return false;
    if (n > (int)m_slots.size()) return AllocateSlots(n);
    m_activeSlots = n;
    return true;
}

void FRUCDepthInterp::FreeSlots() {
    for (auto* sl : m_slots) {
        if (sl->regOk && sl->hFRUC) {
            NvFRUC_UNREG_PARAM up{}; memcpy(up.pArrResource,sl->res,3*sizeof(void*)); up.uiCount=3;
            fnU(m_fnUnregRes)(sl->hFRUC,&up);
        }
        if (sl->hFRUC)    fnD(m_fnDestroy)(sl->hFRUC);
        if (sl->d_outARGB)cudaFree(sl->d_outARGB);
        if (sl->h_outARGB)cudaFreeHost(sl->h_outARGB);
        if (sl->h_depth)  cudaFreeHost(sl->h_depth);
        delete sl;
    }
    m_slots.clear(); m_activeSlots = 0;
}
void FRUCDepthInterp::FreeShared() {
    if (m_d_prevARGB) { cudaFree(m_d_prevARGB); m_d_prevARGB=nullptr; }
    if (m_d_currARGB) { cudaFree(m_d_currARGB); m_d_currARGB=nullptr; }
    if (m_uploadStream){ cudaStreamDestroy(m_uploadStream); m_uploadStream=nullptr; }
    m_ready = false;
}

std::vector<std::vector<float>> FRUCDepthInterp::Interpolate(
    const float* prevDepth, const float* currDepth, int numInterp)
{
    if (!m_ready || numInterp<=0) return {};
    numInterp = std::min(numInterp, (int)m_slots.size());
    const int nPx = m_w * m_h;

    // Upload both depth maps as ARGB to device (shared across all slots)
    // Uses a dedicated upload stream so this can overlap with CPU-side FRUC setup
    depth_to_argb_cuda(nullptr, nullptr, 0, m_uploadStream); // warm-up noop
    // Actually upload via CPU host→device (depth is on CPU from depth_estimator)
    // Step 1: depth float[] → device float[], then kernel → d_prevARGB/d_currARGB
    float* d_tmp=nullptr; cudaMalloc(&d_tmp, nPx*sizeof(float));
    cudaMemcpyAsync(d_tmp, prevDepth, nPx*sizeof(float), cudaMemcpyHostToDevice, m_uploadStream);
    depth_to_argb_cuda(d_tmp, m_d_prevARGB, nPx, m_uploadStream);
    cudaMemcpyAsync(d_tmp, currDepth, nPx*sizeof(float), cudaMemcpyHostToDevice, m_uploadStream);
    depth_to_argb_cuda(d_tmp, m_d_currARGB, nPx, m_uploadStream);
    cudaStreamSynchronize(m_uploadStream);
    cudaFree(d_tmp);

    std::vector<std::vector<float>> results;
    results.reserve(numInterp);

    // For each interpolated frame k (1..numInterp), run one FRUC instance.
    // NvFRUCProcess is blocking but NVOFA hardware runs independently from CUDA
    // cores, so inference on the inference stream runs in parallel.
    for (int k=0; k<numInterp; ++k) {
        auto* sl = m_slots[k];
        if (!sl->regOk) { results.push_back({}); continue; }
        long long S = numInterp+1; // total slots: S-1 interp + 2 endpoints = S+1, timestamps 0..S

        // Prime: feed prevARGB as "frame 0" so FRUC caches it as the previous frame.
        // The output (frame 0 itself) is written to d_outARGB but ignored.
        NvFRUC_PROC_IN  in{}; in.stFrameDataInput.pFrame=sl->res[0]; in.stFrameDataInput.nTimeStamp=0;
        NvFRUC_PROC_OUT out{}; out.pFrame=sl->res[2]; out.nTimeStamp=0;
        if (fnP(m_fnProcess)(sl->hFRUC,&in,&out)!=NvFRUC_SUCCESS)
            { results.push_back({}); continue; }

        // Interpolate: feed currARGB as "frame S" → FRUC generates frame at t = k+1
        in.stFrameDataInput.pFrame=sl->res[1]; in.stFrameDataInput.nTimeStamp=S;
        out.pFrame=sl->res[2]; out.nTimeStamp=(long long)(k+1); // 1..numInterp
        if (fnP(m_fnProcess)(sl->hFRUC,&in,&out)!=NvFRUC_SUCCESS)
            { results.push_back({}); continue; }

        // Readback + decode: d_outARGB → host → float depth
        cudaMemcpy(sl->h_outARGB, sl->d_outARGB, m_argbBytes, cudaMemcpyDeviceToHost);
        // Decode 16-bit R+G back to float on CPU (cheap at LR resolution)
        std::vector<float> depth(nPx);
        for (int i=0; i<nPx; ++i) {
            uint16_t u = ((uint16_t)sl->h_outARGB[i*4+2]<<8) | sl->h_outARGB[i*4+1];
            depth[i] = u/65535.f;
        }
        results.push_back(std::move(depth));
    }
    return results;
}
