// SPDX-License-Identifier: GPL-3.0-or-later
// nvof_depth_interp.cu — compiled with nvcc.
// Contains: bgra→gray8 downsample kernel, normalise kernel, bidirectional
// warp+blend kernel, and all NvOF DLL management via dynamic loading.
#include "nvof_depth_interp.h"
#include "logger.h"
#include <cuda_runtime.h>
#include <cuda.h>  // Driver API: CUcontext, CUdeviceptr, cuCtxGetCurrent
#include <algorithm>
#include <cmath>
#include <cstring>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define LOAD_LIB(n)   LoadLibraryExW(n, nullptr, \
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS| \
    LOAD_LIBRARY_SEARCH_USER_DIRS)
#define FREE_LIB(h)   FreeLibrary((HMODULE)(h))
#define GET_PROC(h,n) GetProcAddress((HMODULE)(h),(n))
#else
#include <dlfcn.h>
#define LOAD_LIB(n)   dlopen((n),RTLD_LAZY)
#define FREE_LIB(h)   dlclose(h)
#define GET_PROC(h,n) dlsym((h),(n))
#endif

// ── NvOF API types (inlined — no SDK headers needed) ─────────────────────────
typedef int NV_OF_STATUS;
#define NV_OF_SUCCESS 0
#define NV_OF_API_VERSION ((1u<<8)|4u)  // SDK v4.0
typedef unsigned int NV_OF_BOOL;
typedef enum {
    NV_OF_PERF_LEVEL_SLOW=5, NV_OF_PERF_LEVEL_MEDIUM=10
} NV_OF_PERF_LEVEL;
typedef enum { NV_OF_OUTPUT_VECTOR_GRID_SIZE_1=1 } NV_OF_OUT_GRID;
typedef enum {
    NV_OF_BUFFER_FORMAT_GRAYSCALE8=1,
    NV_OF_BUFFER_FORMAT_SHORT2=5
} NV_OF_BUFFER_FORMAT;
typedef enum { NV_OF_BUFFER_USAGE_INPUT=1, NV_OF_BUFFER_USAGE_OUTPUT=2 } NV_OF_BUFFER_USAGE;
typedef enum { NV_OF_MODE_OPTICALFLOW=1 } NV_OF_MODE;
typedef enum { NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR=1 } NV_OF_CUDA_BUF_TYPE;

struct NV_OF_INIT_PARAMS {
    unsigned int width,height;
    NV_OF_BOOL enableOutputCost,enableExternalHints,enableRoi;
    NV_OF_PERF_LEVEL perfLevel;
    NV_OF_BOOL enableTemporalHints;
    NV_OF_OUT_GRID outputGridSize;
    NV_OF_BUFFER_FORMAT inputBufferFormat;
    NV_OF_MODE mode;
};
struct NV_OF_BUFFER_DESCRIPTOR { unsigned int width,height; NV_OF_BUFFER_USAGE usage; NV_OF_BUFFER_FORMAT format; };
struct NV_OF_EXEC_IN {
    void* inputFrame;       // NvOFGPUBufferHandle
    void* referenceFrame;
    void* externalHints;    // NULL
    NV_OF_BOOL disableTemporalHints;
    unsigned int numRois; void* roiData; // NULL,NULL
    cudaStream_t inputStream, outputStream;
};
struct NV_OF_EXEC_OUT { void* outputBuffer; void* outputCostBuffer; };

struct NV_OF_CUDA_FN {
    NV_OF_STATUS (*nvCreateOpticalFlowCuda)(CUcontext,void**);
    NV_OF_STATUS (*nvOFInit)(void*,const NV_OF_INIT_PARAMS*);
    NV_OF_STATUS (*nvOFCreateGPUBufferCuda)(void*,const NV_OF_BUFFER_DESCRIPTOR*,unsigned,void**);
    NV_OF_STATUS (*nvOFGetCudaResourceCuda)(void*,unsigned,void*);
    NV_OF_STATUS (*nvOFExecute)(void*,const NV_OF_EXEC_IN*,NV_OF_EXEC_OUT*);
    NV_OF_STATUS (*nvOFDestroyGPUBufferCuda)(void*);
    NV_OF_STATUS (*nvOFDestroy)(void*);
    void* reserved[16];
};

// ── Kernels ────────────────────────────────────────────────────────────────────
__global__ void k_bgra_to_gray8(
    const uint8_t* __restrict__ src, int srcW, int srcH, int srcStride,
    uint8_t* __restrict__ dst, int dW, int dH)
{
    int dx=blockIdx.x*blockDim.x+threadIdx.x, dy=blockIdx.y*blockDim.y+threadIdx.y;
    if(dx>=dW||dy>=dH) return;
    float fx=(dx+.5f)*srcW/(float)dW-.5f, fy=(dy+.5f)*srcH/(float)dH-.5f;
    int x0=max(0,min((int)fx,srcW-1)),x1=min(x0+1,srcW-1);
    int y0=max(0,min((int)fy,srcH-1)),y1=min(y0+1,srcH-1);
    float tx=fx-x0,ty=fy-y0;
    auto L=[&](int x,int y)->float{
        const uint8_t* p=src+y*srcStride+x*4;
        return (29*p[0]+150*p[1]+77*p[2])*(1.f/65280.f);
    };
    float v=L(x0,y0)*(1-tx)*(1-ty)+L(x1,y0)*tx*(1-ty)+
            L(x0,y1)*(1-tx)*ty   +L(x1,y1)*tx*ty;
    dst[dy*dW+dx]=(uint8_t)(v*255.f+.5f);
}

__global__ void k_minmax(const float* __restrict__ d, float* mn, float* mx, int n) {
    // Simple block-level reduction (caller uses single block for LR depth)
    extern __shared__ float smem[];
    int tid=threadIdx.x, bid=blockIdx.x, bs=blockDim.x;
    float lmn=1e30f,lmx=-1e30f;
    for(int i=bid*bs+tid;i<n;i+=gridDim.x*bs){float v=d[i];lmn=min(lmn,v);lmx=max(lmx,v);}
    smem[tid]=lmn; smem[tid+bs]=lmx; __syncthreads();
    for(int s=bs>>1;s>0;s>>=1){
        if(tid<s){smem[tid]=min(smem[tid],smem[tid+s]);smem[tid+bs]=max(smem[tid+bs],smem[tid+s+bs]);}
        __syncthreads();
    }
    if(tid==0){atomicMinf(mn,smem[0]);atomicMaxf(mx,smem[bs]);}
}

__device__ float atomicMinf_impl(float* addr,float val){
    int* p=(int*)addr; int assumed,old=*p;
    do{assumed=old;old=atomicCAS(p,assumed,__float_as_int(min(val,__int_as_float(assumed))));}
    while(assumed!=old); return __int_as_float(old);
}
__device__ float atomicMaxf_impl(float* addr,float val){
    int* p=(int*)addr; int assumed,old=*p;
    do{assumed=old;old=atomicCAS(p,assumed,__float_as_int(max(val,__int_as_float(assumed))));}
    while(assumed!=old); return __int_as_float(old);
}
// Simpler: just use our own normalize that does min-max scan on device
__global__ void k_normalize(const float* __restrict__ in, float* __restrict__ out,
                             float mn, float mx, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float r=mx-mn; if(r<1e-6f)r=1e-6f;
    out[i]=__saturatef((in[i]-mn)/r);
}

__global__ void k_scanminmax(const float* __restrict__ d,float* mn,float* mx,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float v=d[i];
    atomicMinf_impl(mn,v); atomicMaxf_impl(mx,v);
}

__global__ void k_warp_blend(
    const float* __restrict__ prev, const float* __restrict__ curr,
    const short2* __restrict__ flow, float* __restrict__ out,
    int w, int h, float t)
{
    int px=blockIdx.x*blockDim.x+threadIdx.x, py=blockIdx.y*blockDim.y+threadIdx.y;
    if(px>=w||py>=h) return;
    short2 f=flow[py*w+px];
    float fx=f.x/64.f, fy=f.y/64.f;
    // Backward from interpolated position: sample prev at (px - t*fx, py - t*fy)
    //                                      sample curr at (px + (1-t)*fx, py + (1-t)*fy)
    auto bsamp=[&](const float* img,float sx,float sy)->float{
        sx=fmaxf(0.f,fminf((float)(w-1),sx)); sy=fmaxf(0.f,fminf((float)(h-1),sy));
        int x0=(int)sx,y0=(int)sy;
        int x1=min(x0+1,w-1),y1=min(y0+1,h-1);
        float tx=sx-x0,ty=sy-y0;
        return (1-tx)*(1-ty)*img[y0*w+x0]+tx*(1-ty)*img[y0*w+x1]+
               (1-tx)*ty*img[y1*w+x0]+tx*ty*img[y1*w+x1];
    };
    float vf=bsamp(prev,px-t*fx,py-t*fy);
    float vb=bsamp(curr,px+(1.f-t)*fx,py+(1.f-t)*fy);
    out[py*w+px]=(1.f-t)*vf+t*vb;
}

// ── NvOFState ─────────────────────────────────────────────────────────────────
#define MAX_SLOTS 2   // ping-pong between writeBuf and readBuf
#define MAX_INTERP 8
struct NvOFState {
    int w=0,h=0; bool ok=false; int maxInterp=0;
    void* hDLL=nullptr;
    NV_OF_CUDA_FN fn{};
    void* hOF=nullptr;
    void* hGuide[MAX_SLOTS]={};   // NvOFGPUBufferHandle GRAYSCALE8
    void* hFlow=nullptr;           // NvOFGPUBufferHandle SHORT2
    CUdeviceptr d_guide[MAX_SLOTS]={};
    CUdeviceptr d_flow=0;
    // Normalised LR depth (device)
    float* d_depthLR[MAX_SLOTS]={};
    // Warp outputs
    float* d_warpOut[MAX_INTERP]={};
    float* h_warpOut[MAX_INTERP]={};
    // Temp min/max for normalisation
    float* d_minmax=nullptr; // [2]: [0]=min [1]=max
};

NvOFState* nvof_create(int w, int h, int maxInterp, const std::wstring& dllDir) {
    auto* st = new NvOFState;
    st->w=w; st->h=h; st->maxInterp=std::min(maxInterp,MAX_INTERP);

    // Load nvofapi64.dll from Win64 folder
    std::wstring path = dllDir + L"\\nvofapi64.dll";
    st->hDLL = (void*)LOAD_LIB(path.c_str());
    if (!st->hDLL) {
        path = L"nvofapi64.dll"; // system fallback
        st->hDLL = (void*)LOAD_LIB(path.c_str());
    }
    if (!st->hDLL) {
        LOG_INFO("NvOF: nvofapi64.dll not found — depth interpolation disabled");
        delete st; return nullptr;
    }

    // Get function table
    using PFN = NV_OF_STATUS(*)(unsigned, NV_OF_CUDA_FN*);
    auto entry = (PFN)GET_PROC(st->hDLL,"NvOFAPICreateInstanceCuda");
    if (!entry || entry(NV_OF_API_VERSION, &st->fn) != NV_OF_SUCCESS) {
        LOG_WARN("NvOF: NvOFAPICreateInstanceCuda failed"); FREE_LIB(st->hDLL); delete st; return nullptr;
    }

    // Get current CUDA context (TRT has already created it)
    CUcontext ctx=nullptr; cuCtxGetCurrent(&ctx);
    if (!ctx) { LOG_WARN("NvOF: no CUDA context"); FREE_LIB(st->hDLL); delete st; return nullptr; }

    // Create NvOF handle
    if (st->fn.nvCreateOpticalFlowCuda(ctx, &st->hOF) != NV_OF_SUCCESS) {
        LOG_WARN("NvOF: nvCreateOpticalFlowCuda failed"); FREE_LIB(st->hDLL); delete st; return nullptr;
    }

    // Init: GRAYSCALE8 input, SHORT2 output, per-pixel grid, best quality
    NV_OF_INIT_PARAMS ip{}; ip.width=(unsigned)w; ip.height=(unsigned)h;
    ip.perfLevel=NV_OF_PERF_LEVEL_SLOW;
    ip.enableTemporalHints=1; ip.outputGridSize=NV_OF_OUTPUT_VECTOR_GRID_SIZE_1;
    ip.inputBufferFormat=NV_OF_BUFFER_FORMAT_GRAYSCALE8; ip.mode=NV_OF_MODE_OPTICALFLOW;
    if (st->fn.nvOFInit(st->hOF, &ip) != NV_OF_SUCCESS) {
        LOG_WARN("NvOF: nvOFInit failed (GPU may not have NvOF hardware)");
        st->fn.nvOFDestroy(st->hOF); FREE_LIB(st->hDLL); delete st; return nullptr;
    }

    // Create guide input buffers (GRAYSCALE8)
    NV_OF_BUFFER_DESCRIPTOR bd{};
    bd.width=(unsigned)w; bd.height=(unsigned)h;
    bd.usage=NV_OF_BUFFER_USAGE_INPUT; bd.format=NV_OF_BUFFER_FORMAT_GRAYSCALE8;
    for (int s=0;s<MAX_SLOTS;s++) {
        if (st->fn.nvOFCreateGPUBufferCuda(st->hOF,&bd,(unsigned)NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,&st->hGuide[s])!=NV_OF_SUCCESS)
            { LOG_WARN("NvOF: guide buf create failed"); goto fail; }
        st->fn.nvOFGetCudaResourceCuda(st->hGuide[s],(unsigned)NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,&st->d_guide[s]);
    }
    // Create flow output buffer (SHORT2)
    bd.usage=NV_OF_BUFFER_USAGE_OUTPUT; bd.format=NV_OF_BUFFER_FORMAT_SHORT2;
    if (st->fn.nvOFCreateGPUBufferCuda(st->hOF,&bd,(unsigned)NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,&st->hFlow)!=NV_OF_SUCCESS)
        { LOG_WARN("NvOF: flow buf create failed"); goto fail; }
    st->fn.nvOFGetCudaResourceCuda(st->hFlow,(unsigned)NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,&st->d_flow);

    // Allocate normalised LR depth + warp buffers
    for (int s=0;s<MAX_SLOTS;s++) cudaMalloc(&st->d_depthLR[s],(size_t)w*h*sizeof(float));
    for (int i=0;i<st->maxInterp;i++) {
        cudaMalloc(&st->d_warpOut[i],(size_t)w*h*sizeof(float));
        cudaMallocHost(&st->h_warpOut[i],(size_t)w*h*sizeof(float));
    }
    cudaMalloc(&st->d_minmax, 2*sizeof(float));

    st->ok=true;
    LOG_INFO("NvOF: ready at ",w,"x",h," maxInterp=",st->maxInterp);
    return st;
fail:
    nvof_destroy(st); return nullptr;
}

void nvof_destroy(NvOFState* st) {
    if (!st) return;
    for (int s=0;s<MAX_SLOTS;s++) {
        if (st->hGuide[s]) st->fn.nvOFDestroyGPUBufferCuda(st->hGuide[s]);
        if (st->d_depthLR[s]) cudaFree(st->d_depthLR[s]);
    }
    if (st->hFlow) st->fn.nvOFDestroyGPUBufferCuda(st->hFlow);
    for (int i=0;i<MAX_INTERP;i++){
        if (st->d_warpOut[i]) cudaFree(st->d_warpOut[i]);
        if (st->h_warpOut[i]) cudaFreeHost(st->h_warpOut[i]);
    }
    if (st->d_minmax) cudaFree(st->d_minmax);
    if (st->hOF) st->fn.nvOFDestroy(st->hOF);
    if (st->hDLL) FREE_LIB(st->hDLL);
    delete st;
}
bool nvof_available(NvOFState* st) { return st && st->ok; }
void nvof_dims(NvOFState* st, int* w, int* h) { if(st){*w=st->w;*h=st->h;} }

void nvof_prepare_slot(NvOFState* st, int slot,
                       const uint8_t* d_guideBGRA, int srcW, int srcH, int srcStride,
                       const float* d_outSlice, float* d_minmax_scratch,
                       int mw, int mh, void* stream)
{
    if (!st||!st->ok) return;
    cudaStream_t s=(cudaStream_t)stream;
    // 1. Downsample BGRA → GRAYSCALE8 at mw×mh
    dim3 blk(16,16), grd((mw+15)/16,(mh+15)/16);
    k_bgra_to_gray8<<<grd,blk,0,s>>>(d_guideBGRA,srcW,srcH,srcStride,
                                      (uint8_t*)st->d_guide[slot],mw,mh);
    // 2. Normalise raw TRT output → d_depthLR[slot]
    int n=mw*mh;
    float init_mn=1e30f, init_mx=-1e30f;
    cudaMemcpyAsync(st->d_minmax, &init_mn, sizeof(float), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(st->d_minmax+1, &init_mx, sizeof(float), cudaMemcpyHostToDevice, s);
    k_scanminmax<<<(n+255)/256,256,0,s>>>(d_outSlice,st->d_minmax,st->d_minmax+1,n);
    float mn,mx;
    cudaMemcpyAsync(&mn,st->d_minmax,  sizeof(float),cudaMemcpyDeviceToHost,s);
    cudaMemcpyAsync(&mx,st->d_minmax+1,sizeof(float),cudaMemcpyDeviceToHost,s);
    cudaStreamSynchronize(s); // cheap: just 2 floats
    k_normalize<<<(n+255)/256,256,0,s>>>(d_outSlice,st->d_depthLR[slot],mn,mx,n);
}

bool nvof_execute(NvOFState* st, int prevSlot, int currSlot, void* stream) {
    if (!st||!st->ok) return false;
    NV_OF_EXEC_IN in{}; NV_OF_EXEC_OUT out{};
    in.inputFrame    = st->hGuide[currSlot];
    in.referenceFrame= st->hGuide[prevSlot];
    in.disableTemporalHints=0;
    in.inputStream = in.outputStream = (cudaStream_t)stream;
    out.outputBuffer = st->hFlow;
    auto r = st->fn.nvOFExecute(st->hOF, &in, &out);
    if (r != NV_OF_SUCCESS) { LOG_WARN("NvOF: execute failed r=",r); return false; }
    return true;
}

void nvof_warp(NvOFState* st, int prevSlot, int currSlot,
               float* d_out, float t, void* stream)
{
    if (!st||!st->ok) return;
    cudaStream_t s=(cudaStream_t)stream;
    dim3 blk(16,16), grd((st->w+15)/16,(st->h+15)/16);
    k_warp_blend<<<grd,blk,0,s>>>(st->d_depthLR[prevSlot], st->d_depthLR[currSlot],
                                   (const short2*)(void*)st->d_flow, d_out,
                                   st->w, st->h, t);
}
