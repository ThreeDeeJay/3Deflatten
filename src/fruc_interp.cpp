// SPDX-License-Identifier: GPL-3.0-or-later
// fruc_interp.cpp  — FRUCDepthInterp implementation.
// Uses CUDA Driver API (nvcuda.dll) and NvOFFRUC.dll loaded at runtime via
// GetProcAddress.  Zero static DLL imports for CUDA — no loader-time stall.
#include "fruc_interp.h"
#include "logger.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>

// ── NvOFFRUC structures (from NvOFFRUC.h, inlined to avoid SDK dependency) ────
#define NvOFFRUC_SUCCESS    0
#define NvOFFRUC_MAX_RES    5

typedef struct {
    void* pDevice;
    unsigned int uiHeight, uiWidth;
    int  eResourceType;    // 0=CUDA
    int  eSurfaceFormat;   // 1=ARGB
    int  eCUDAResourceType;// 0=cuDevicePtr
} NvOFFRUC_CREATE_PARAM;

typedef struct {
    void* pArrResource[NvOFFRUC_MAX_RES];
    unsigned int uiCount;
    void* pFence;
} NvOFFRUC_REG_PARAM;

typedef struct {
    void*     pFrame;        // pointer to CUdeviceptr (input frame)
    long long nTimeStamp;
    unsigned int bRepeat, uSync;
} NvOFFRUCFrameData;

typedef struct { NvOFFRUCFrameData stFrameDataInput; } NvOFFRUC_PROC_IN;
typedef struct {
    void*     pFrame;        // pointer to CUdeviceptr (output frame)
    long long nTimeStamp;
    unsigned int bRepeat;
} NvOFFRUC_PROC_OUT;

typedef struct {
    void* pArrResource[NvOFFRUC_MAX_RES];
    unsigned int uiCount;
} NvOFFRUC_UNREG_PARAM;

// Proc-name constants (must match NvOFFRUC.h exactly)
#define CREATE_PROC_NAME    "NvOFFRUCCreate"
#define REGRES_PROC_NAME    "NvOFFRUCRegisterResource"
#define PROCESS_PROC_NAME   "NvOFFRUCProcess"
#define UNREGRES_PROC_NAME  "NvOFFRUCUnregisterResource"
#define DESTROY_PROC_NAME   "NvOFFRUCDestroy"

// ── FRUCDepthInterp ───────────────────────────────────────────────────────────
FRUCDepthInterp::FRUCDepthInterp()  = default;
FRUCDepthInterp::~FRUCDepthInterp() {
    for (int i=0; i<(int)m_slots.size(); i++) DestroySlot(i);
    FreeBuffers();
    if (m_hFRUC) FreeLibrary((HMODULE)m_hFRUC);
    if (m_hCUDA) FreeLibrary((HMODULE)m_hCUDA);
}

static void* LoadFrom(const std::wstring& dir, const wchar_t* name) {
    std::wstring full = dir + L"\\" + name;
    HMODULE h = LoadLibraryExW(full.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
        LOAD_LIBRARY_SEARCH_USER_DIRS);
    if (!h) h = LoadLibraryExW(name, nullptr,
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    return (void*)h;
}

bool FRUCDepthInterp::LoadCUDA(const std::wstring& dir) {
    m_hCUDA = LoadFrom(dir, L"nvcuda.dll");
    if (!m_hCUDA) {
        LOG_WARN("FRUCDepthInterp: nvcuda.dll not found — FRUC disabled");
        return false;
    }
    auto gp = [&](const char* n)->void*{ return (void*)GetProcAddress((HMODULE)m_hCUDA,n); };
    m_cuInit     = (PFN_cuInit)     gp("cuInit");
    m_cuMemAlloc = (PFN_cuMemAlloc) gp("cuMemAlloc_v2");
    m_cuMemFree  = (PFN_cuMemFree)  gp("cuMemFree_v2");
    m_cuHtoD     = (PFN_cuMemcpyHtoD)gp("cuMemcpyHtoD_v2");
    m_cuDtoH     = (PFN_cuMemcpyDtoH)gp("cuMemcpyDtoH_v2");
    if (!m_cuInit||!m_cuMemAlloc||!m_cuMemFree||!m_cuHtoD||!m_cuDtoH) {
        LOG_WARN("FRUCDepthInterp: nvcuda.dll entry points missing"); return false;
    }
    return true;
}

bool FRUCDepthInterp::LoadNvOFFRUC(const std::wstring& dir) {
    m_hFRUC = LoadFrom(dir, L"NvOFFRUC.dll");
    if (!m_hFRUC) {
        LOG_INFO("FRUCDepthInterp: NvOFFRUC.dll not found — depth interpolation disabled");
        return false;
    }
    auto gp=[&](const char* n)->void*{ return (void*)GetProcAddress((HMODULE)m_hFRUC,n); };
    m_fnCreate   = (PFN_Create)  gp(CREATE_PROC_NAME);
    m_fnRegRes   = (PFN_RegRes)  gp(REGRES_PROC_NAME);
    m_fnProcess  = (PFN_Process) gp(PROCESS_PROC_NAME);
    m_fnUnregRes = (PFN_UnregRes)gp(UNREGRES_PROC_NAME);
    m_fnDestroy  = (PFN_Destroy) gp(DESTROY_PROC_NAME);
    if (!m_fnCreate||!m_fnRegRes||!m_fnProcess||!m_fnUnregRes||!m_fnDestroy) {
        LOG_WARN("FRUCDepthInterp: NvOFFRUC.dll exports missing"); return false;
    }
    return true;
}

bool FRUCDepthInterp::AllocBuffers() {
    m_argbBytes = (size_t)m_w * m_h * 4;
    if (m_cuMemAlloc(&m_d_prev, m_argbBytes) != 0 ||
        m_cuMemAlloc(&m_d_curr, m_argbBytes) != 0) {
        LOG_WARN("FRUCDepthInterp: cuMemAlloc failed"); return false;
    }
    return true;
}
void FRUCDepthInterp::FreeBuffers() {
    if (m_d_prev && m_cuMemFree) m_cuMemFree(m_d_prev);
    if (m_d_curr && m_cuMemFree) m_cuMemFree(m_d_curr);
    m_d_prev = m_d_curr = 0;
}

bool FRUCDepthInterp::CreateSlot(int idx) {
    if (idx >= (int)m_slots.size()) m_slots.resize(idx+1);
    Slot& sl = m_slots[idx];

    NvOFFRUC_CREATE_PARAM cp{};
    cp.uiWidth=(unsigned)m_w; cp.uiHeight=(unsigned)m_h;
    cp.eResourceType=0; cp.eSurfaceFormat=1; cp.eCUDAResourceType=0;
    if (m_fnCreate(&cp, &sl.hFRUC) != NvOFFRUC_SUCCESS || !sl.hFRUC) {
        LOG_WARN("FRUCDepthInterp: NvOFFRUCCreate failed slot=",idx); return false;
    }

    if (m_cuMemAlloc(&sl.d_out, m_argbBytes) != 0) {
        LOG_WARN("FRUCDepthInterp: cuMemAlloc out failed slot=",idx); return false;
    }
    sl.h_out = malloc(m_argbBytes);
    if (!sl.h_out) return false;

    // Resource layout: [0]=output, [1]=prev input, [2]=curr input
    // Each element points to the CUdeviceptr variable (not the ptr itself)
    sl.res[0] = &sl.d_out;
    sl.res[1] = &m_d_prev;
    sl.res[2] = &m_d_curr;

    NvOFFRUC_REG_PARAM rp{}; rp.uiCount = 3;
    rp.pArrResource[0]=sl.res[0];
    rp.pArrResource[1]=sl.res[1];
    rp.pArrResource[2]=sl.res[2];
    sl.regOk = (m_fnRegRes(sl.hFRUC, &rp) == NvOFFRUC_SUCCESS);
    if (!sl.regOk) LOG_WARN("FRUCDepthInterp: RegisterResource failed slot=",idx);
    return sl.regOk;
}

void FRUCDepthInterp::DestroySlot(int idx) {
    if (idx>=(int)m_slots.size()) return;
    Slot& sl = m_slots[idx];
    if (sl.regOk && sl.hFRUC) {
        NvOFFRUC_UNREG_PARAM up{}; up.uiCount=3;
        up.pArrResource[0]=sl.res[0];
        up.pArrResource[1]=sl.res[1];
        up.pArrResource[2]=sl.res[2];
        m_fnUnregRes(sl.hFRUC, &up);
    }
    if (sl.hFRUC)   m_fnDestroy(sl.hFRUC);
    if (sl.d_out && m_cuMemFree) m_cuMemFree(sl.d_out);
    if (sl.h_out)   free(sl.h_out);
    sl = Slot{};
}

bool FRUCDepthInterp::Init(int w, int h, int maxInterp,
                             const std::wstring& dllDir) {
    m_w = w; m_h = h;
    if (!LoadCUDA(dllDir))  return false;
    if (!LoadNvOFFRUC(dllDir)) return false;
    if (!AllocBuffers())    return false;
    for (int i=0; i<maxInterp; i++) {
        if (!CreateSlot(i)) { LOG_WARN("FRUCDepthInterp: slot ",i," failed"); break; }
    }
    m_ready = !m_slots.empty();
    LOG_INFO("FRUCDepthInterp: ",m_ready?"ready":"no slots"," at ",w,"x",h,
             " maxInterp=",maxInterp," dir=",dllDir);
    return m_ready;
}

void FRUCDepthInterp::SetInterpCount(int n) {
    if (!m_ready) return;
    while ((int)m_slots.size() < n) {
        int i = (int)m_slots.size();
        m_slots.resize(i+1);
        CreateSlot(i);
    }
}

std::vector<std::vector<float>> FRUCDepthInterp::Interpolate(
    const float* prevDepth, const float* currDepth, int numInterp)
{
    if (!m_ready || numInterp<=0) return {};
    numInterp = std::min(numInterp, (int)m_slots.size());
    int nPx = m_w * m_h;

    // Upload depth as ARGB to shared device buffers (16-bit precision via R+G)
    // We use a temporary host ARGB buffer then copy via Driver API
    std::vector<uint8_t> tmpPrev(m_argbBytes), tmpCurr(m_argbBytes);
    for (int i=0; i<nPx; i++) {
        uint16_t up=(uint16_t)(std::max(0.f,std::min(1.f,prevDepth[i]))*65535.f+.5f);
        tmpPrev[i*4+0]=0; tmpPrev[i*4+1]=up&0xFF; tmpPrev[i*4+2]=up>>8; tmpPrev[i*4+3]=0xFF;
        uint16_t uc=(uint16_t)(std::max(0.f,std::min(1.f,currDepth[i]))*65535.f+.5f);
        tmpCurr[i*4+0]=0; tmpCurr[i*4+1]=uc&0xFF; tmpCurr[i*4+2]=uc>>8; tmpCurr[i*4+3]=0xFF;
    }
    m_cuHtoD(m_d_prev, tmpPrev.data(), m_argbBytes);
    m_cuHtoD(m_d_curr, tmpCurr.data(), m_argbBytes);

    std::vector<std::vector<float>> results;
    results.reserve(numInterp);
    long long S = (long long)(numInterp + 1); // denominator for timestamps

    for (int k=0; k<numInterp; k++) {
        Slot& sl = m_slots[k];
        if (!sl.regOk || !sl.hFRUC) { results.push_back({}); continue; }

        // Prime: feed prev as "frame 0" so FRUC caches it as previous
        NvOFFRUC_PROC_IN  in{};  in.stFrameDataInput.pFrame=sl.res[1];
        in.stFrameDataInput.nTimeStamp=0;
        NvOFFRUC_PROC_OUT out{}; out.pFrame=sl.res[0]; out.nTimeStamp=0;
        if (m_fnProcess(sl.hFRUC,&in,&out)!=NvOFFRUC_SUCCESS)
            { results.push_back({}); continue; }

        // Interpolate: feed curr as "frame S", request output at k+1
        in.stFrameDataInput.pFrame=sl.res[2];
        in.stFrameDataInput.nTimeStamp=S;
        out.pFrame=sl.res[0]; out.nTimeStamp=(long long)(k+1);
        if (m_fnProcess(sl.hFRUC,&in,&out)!=NvOFFRUC_SUCCESS)
            { results.push_back({}); continue; }

        // Readback and decode
        m_cuDtoH(sl.h_out, sl.d_out, m_argbBytes);
        auto* argb = (const uint8_t*)sl.h_out;
        std::vector<float> depth(nPx);
        for (int i=0; i<nPx; i++) {
            uint16_t u=((uint16_t)argb[i*4+2]<<8)|(uint16_t)argb[i*4+1];
            depth[i]=u/65535.f;
        }
        results.push_back(std::move(depth));
    }
    return results;
}