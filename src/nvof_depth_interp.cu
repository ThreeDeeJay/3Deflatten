// SPDX-License-Identifier: GPL-3.0-or-later
// nvof_depth_interp.cu — compiled with nvcc.
//
// NVIDIA Optical Flow is loaded dynamically at runtime.  The NvOF ABI below
// matches the NVIDIA Optical Flow SDK 5.x CUDA API layout.  We intentionally
// do not link against nvofapi64.lib so the application can run on systems
// where NvOF is unavailable.
//
// Important:
//   - NvOF API versions are encoded as (major << 4) | minor.
//     SDK 5.0 = 0x50.
//   - The driver-reported maximum API version is used directly.
//   - The CUDA function table layout MUST exactly match NVIDIA's ABI.
//   - CUDA streams are supplied through nvOFSetIOCudaStreams(), not through
//     NV_OF_EXECUTE_INPUT_PARAMS.
#include "nvof_depth_interp.h"
#include "logger.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define LOAD_LIB(n) \
    LoadLibraryExW( \
        (n), \
        nullptr, \
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | \
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | \
        LOAD_LIBRARY_SEARCH_USER_DIRS)
#define LOAD_SYSTEM_LIB(n) \
    LoadLibraryExW( \
        (n), \
        nullptr, \
        LOAD_LIBRARY_SEARCH_SYSTEM32)
#define FREE_LIB(h) \
    FreeLibrary((HMODULE)(h))
#define GET_PROC(h, n) \
    GetProcAddress((HMODULE)(h), (n))
#else
#include <dlfcn.h>
#define LOAD_LIB(n)   dlopen((n), RTLD_LAZY)
#define LOAD_SYSTEM_LIB(n) dlopen((n), RTLD_LAZY)
#define FREE_LIB(h)   dlclose(h)
#define GET_PROC(h,n) dlsym((h),(n))
#endif
// =============================================================================
// NVIDIA NvOF ABI compatibility definitions
// =============================================================================
//
// These definitions mirror NVIDIA Optical Flow SDK 5.x headers:
//
//   nvOpticalFlowCommon.h
//   nvOpticalFlowCuda.h
//
// Do not reorder these structures or function-table entries.
//
// NvOF API version encoding:
//     major = upper bits
//     minor = low 4 bits
//
// Therefore:
//     4.0 -> 0x40
//     5.0 -> 0x50
//
// The driver is queried at runtime and its maximum supported version is used,
// so this constant is only a fallback when the version-query symbol itself is
// unavailable.
typedef int32_t NV_OF_STATUS;
#ifndef NV_OF_SUCCESS
#define NV_OF_SUCCESS 0
#endif
#ifndef NV_OF_API_VERSION
#define NV_OF_API_VERSION 0x50u
#endif
typedef uint32_t NV_OF_BOOL;
#ifndef NV_OF_FALSE
#define NV_OF_FALSE 0u
#endif
#ifndef NV_OF_TRUE
#define NV_OF_TRUE 1u
#endif
// -----------------------------------------------------------------------------
// Performance level
// -----------------------------------------------------------------------------
typedef enum _NV_OF_PERF_LEVEL
{
    NV_OF_PERF_LEVEL_UNDEFINED = 0,
    NV_OF_PERF_LEVEL_SLOW      = 5,
    NV_OF_PERF_LEVEL_MEDIUM    = 10,
    NV_OF_PERF_LEVEL_FAST      = 20
} NV_OF_PERF_LEVEL;
// -----------------------------------------------------------------------------
// Output vector grid size
// -----------------------------------------------------------------------------
typedef enum _NV_OF_OUTPUT_VECTOR_GRID_SIZE
{
    NV_OF_OUTPUT_VECTOR_GRID_SIZE_UNDEFINED = 0,
    NV_OF_OUTPUT_VECTOR_GRID_SIZE_1         = 1,
    NV_OF_OUTPUT_VECTOR_GRID_SIZE_2         = 2,
    NV_OF_OUTPUT_VECTOR_GRID_SIZE_4         = 4
} NV_OF_OUTPUT_VECTOR_GRID_SIZE;
// -----------------------------------------------------------------------------
// Hint vector grid size
// -----------------------------------------------------------------------------
typedef enum _NV_OF_HINT_VECTOR_GRID_SIZE
{
    NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED = 0,
    NV_OF_HINT_VECTOR_GRID_SIZE_1         = 1,
    NV_OF_HINT_VECTOR_GRID_SIZE_2         = 2,
    NV_OF_HINT_VECTOR_GRID_SIZE_4         = 4
} NV_OF_HINT_VECTOR_GRID_SIZE;
// -----------------------------------------------------------------------------
// Buffer formats
// -----------------------------------------------------------------------------
typedef enum _NV_OF_BUFFER_FORMAT
{
    NV_OF_BUFFER_FORMAT_UNDEFINED   = 0,
    NV_OF_BUFFER_FORMAT_GRAYSCALE8  = 1,
    NV_OF_BUFFER_FORMAT_NV12        = 2,
    NV_OF_BUFFER_FORMAT_ABGR8       = 3,
    NV_OF_BUFFER_FORMAT_SHORT       = 4,
    NV_OF_BUFFER_FORMAT_SHORT2      = 5,
    NV_OF_BUFFER_FORMAT_UINT        = 6,
    NV_OF_BUFFER_FORMAT_UINT8       = 7
} NV_OF_BUFFER_FORMAT;
// -----------------------------------------------------------------------------
// Buffer usage
// -----------------------------------------------------------------------------
typedef enum _NV_OF_BUFFER_USAGE
{
    NV_OF_BUFFER_USAGE_UNDEFINED = 0,
    NV_OF_BUFFER_USAGE_INPUT     = 1,
    NV_OF_BUFFER_USAGE_OUTPUT    = 2,
    NV_OF_BUFFER_USAGE_HINT      = 3,
    NV_OF_BUFFER_USAGE_COST      = 4
} NV_OF_BUFFER_USAGE;
// -----------------------------------------------------------------------------
// NvOF mode
// -----------------------------------------------------------------------------
typedef enum _NV_OF_MODE
{
    NV_OF_MODE_UNDEFINED      = 0,
    NV_OF_MODE_OPTICALFLOW    = 1,
    NV_OF_MODE_STEREODISPARITY = 2
} NV_OF_MODE;
// -----------------------------------------------------------------------------
// CUDA buffer type
// -----------------------------------------------------------------------------
typedef enum _NV_OF_CUDA_BUFFER_TYPE
{
    NV_OF_CUDA_BUFFER_TYPE_UNDEFINED  = 0,
    NV_OF_CUDA_BUFFER_TYPE_CUARRAY    = 1,
    NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR = 2,
    NV_OF_CUDA_BUFFER_TYPE_MAX        = 3
} NV_OF_CUDA_BUFFER_TYPE;
// -----------------------------------------------------------------------------
// Stereo disparity range
// -----------------------------------------------------------------------------
typedef enum _NV_OF_STEREO_DISPARITY_RANGE
{
    NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED = 0,
    NV_OF_STEREO_DISPARITY_RANGE_32        = 32,
    NV_OF_STEREO_DISPARITY_RANGE_64        = 64,
    NV_OF_STEREO_DISPARITY_RANGE_128       = 128,
    NV_OF_STEREO_DISPARITY_RANGE_256       = 256
} NV_OF_STEREO_DISPARITY_RANGE;
// -----------------------------------------------------------------------------
// Opaque NvOF handles
// -----------------------------------------------------------------------------
typedef void* NvOFHandle;
typedef void* NvOFGPUBufferHandle;
typedef void* NvOFPrivDataHandle;
// -----------------------------------------------------------------------------
// SDK 5.x initialization structure
// -----------------------------------------------------------------------------
typedef struct _NV_OF_INIT_PARAMS
{
    uint32_t width;
    uint32_t height;
    NV_OF_OUTPUT_VECTOR_GRID_SIZE outGridSize;
    NV_OF_HINT_VECTOR_GRID_SIZE hintGridSize;
    NV_OF_MODE mode;
    NV_OF_PERF_LEVEL perfLevel;
    NV_OF_BOOL enableExternalHints;
    NV_OF_BOOL enableOutputCost;
    NvOFPrivDataHandle hPrivData;
    NV_OF_STEREO_DISPARITY_RANGE disparityRange;
    NV_OF_BOOL enableRoi;
} NV_OF_INIT_PARAMS;
// -----------------------------------------------------------------------------
// GPU buffer descriptor
// -----------------------------------------------------------------------------
typedef struct _NV_OF_BUFFER_DESCRIPTOR
{
    uint32_t width;
    uint32_t height;
    NV_OF_BUFFER_USAGE bufferUsage;
    NV_OF_BUFFER_FORMAT bufferFormat;
} NV_OF_BUFFER_DESCRIPTOR;
// -----------------------------------------------------------------------------
// Execute input
// -----------------------------------------------------------------------------
typedef struct _NV_OF_EXECUTE_INPUT_PARAMS
{
    NvOFGPUBufferHandle inputFrame;
    NvOFGPUBufferHandle referenceFrame;
    NvOFGPUBufferHandle externalHints;
    NV_OF_BOOL disableTemporalHints;
    uint32_t padding;
    NvOFPrivDataHandle hPrivData;
    uint32_t padding2;
    uint32_t numRois;
    void* roiData;
} NV_OF_EXECUTE_INPUT_PARAMS;
// -----------------------------------------------------------------------------
// Execute output
// -----------------------------------------------------------------------------
typedef struct _NV_OF_EXECUTE_OUTPUT_PARAMS
{
    NvOFGPUBufferHandle outputBuffer;
    NvOFGPUBufferHandle outputCostBuffer;
    NvOFPrivDataHandle hPrivData;
} NV_OF_EXECUTE_OUTPUT_PARAMS;
// -----------------------------------------------------------------------------
// Function pointer calling convention
// -----------------------------------------------------------------------------
//
// NVIDIA defines NVOFAPI as __stdcall on Windows.
//
// The CUDA API function table contains function pointers with the same ABI.
// Using the correct calling convention is important on Windows.
#ifdef _WIN32
#define NVOF_CALL __stdcall
#else
#define NVOF_CALL
#endif
typedef NV_OF_STATUS (NVOF_CALL *PFNNVCREATEOPTICALFLOWCUDA)(
    CUcontext,
    NvOFHandle*);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFINIT)(
    NvOFHandle,
    const NV_OF_INIT_PARAMS*);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFCREATEGPUBUFFERCUDA)(
    NvOFHandle,
    const NV_OF_BUFFER_DESCRIPTOR*,
    NV_OF_CUDA_BUFFER_TYPE,
    NvOFGPUBufferHandle*);
typedef CUarray (NVOF_CALL *PFNNVOFGPUBUFFERGETCUARRAY)(
    NvOFGPUBufferHandle);
typedef CUdeviceptr (NVOF_CALL *PFNNVOFGPUBUFFERGETCUDEVICEPTR)(
    NvOFGPUBufferHandle);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFGPUBUFFERGETSTRIDEINFO)(
    NvOFGPUBufferHandle,
    uint32_t*,
    uint32_t*);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFSETIOCUDASTREAMS)(
    NvOFHandle,
    CUstream,
    CUstream);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFEXECUTE)(
    NvOFHandle,
    const NV_OF_EXECUTE_INPUT_PARAMS*,
    NV_OF_EXECUTE_OUTPUT_PARAMS*);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFDESTROYGPUBUFFERCUDA)(
    NvOFGPUBufferHandle);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFDESTROY)(
    NvOFHandle);
typedef NV_OF_STATUS (NVOF_CALL *PFNVOFGETLASTERROR)(
    NvOFHandle,
    char**);
typedef NV_OF_STATUS (NVOF_CALL *PFNVOFGETCAPS)(
    NvOFHandle,
    uint32_t,
    void*);
// -----------------------------------------------------------------------------
// CUDA function table
// -----------------------------------------------------------------------------
//
// This order is ABI-critical.
//
// NVIDIA SDK 5.x:
//
//   0  nvCreateOpticalFlowCuda
//   1  nvOFInit
//   2  nvOFCreateGPUBufferCuda
//   3  nvOFGPUBufferGetCUarray
//   4  nvOFGPUBufferGetCUdeviceptr
//   5  nvOFGPUBufferGetStrideInfo
//   6  nvOFSetIOCudaStreams
//   7  nvOFExecute
//   8  nvOFDestroyGPUBufferCuda
//   9  nvOFDestroy
//   10 nvOFGetLastError
//   11 nvOFGetCaps
typedef struct _NV_OF_CUDA_API_FUNCTION_LIST
{
    PFNNVCREATEOPTICALFLOWCUDA       nvCreateOpticalFlowCuda;
    PFNNVOFINIT                      nvOFInit;
    PFNNVOFCREATEGPUBUFFERCUDA       nvOFCreateGPUBufferCuda;
    PFNNVOFGPUBUFFERGETCUARRAY       nvOFGPUBufferGetCUarray;
    PFNNVOFGPUBUFFERGETCUDEVICEPTR   nvOFGPUBufferGetCUdeviceptr;
    PFNNVOFGPUBUFFERGETSTRIDEINFO     nvOFGPUBufferGetStrideInfo;
    PFNNVOFSETIOCUDASTREAMS           nvOFSetIOCudaStreams;
    PFNNVOFEXECUTE                    nvOFExecute;
    PFNNVOFDESTROYGPUBUFFERCUDA       nvOFDestroyGPUBufferCuda;
    PFNNVOFDESTROY                    nvOFDestroy;
    PFNVOFGETLASTERROR                nvOFGetLastError;
    PFNVOFGETCAPS                     nvOFGetCaps;
} NV_OF_CUDA_API_FUNCTION_LIST;
// =============================================================================
// Driver entry points
// =============================================================================
typedef NV_OF_STATUS (NVOF_CALL *PFN_NV_OF_GET_MAX_SUPPORTED_API_VERSION)(
    uint32_t*);
typedef NV_OF_STATUS (NVOF_CALL *PFN_NV_OF_API_CREATE_INSTANCE_CUDA)(
    uint32_t,
    NV_OF_CUDA_API_FUNCTION_LIST*);
// =============================================================================
// Kernels
// =============================================================================
__global__ void k_bgra_to_gray8(
    const uint8_t* __restrict__ src,
    int srcW,
    int srcH,
    int srcStride,
    uint8_t* __restrict__ dst,
    int dW,
    int dH)
{
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dW || dy >= dH)
        return;
    float fx =
        (dx + 0.5f) * srcW / (float)dW - 0.5f;
    float fy =
        (dy + 0.5f) * srcH / (float)dH - 0.5f;
    int x0 = max(0, min((int)fx, srcW - 1));
    int x1 = min(x0 + 1, srcW - 1);
    int y0 = max(0, min((int)fy, srcH - 1));
    int y1 = min(y0 + 1, srcH - 1);
    float tx = fx - x0;
    float ty = fy - y0;
    auto L = [&](int x, int y) -> float
    {
        const uint8_t* p =
            src + y * srcStride + x * 4;
        // BGRA
        return
            (29.0f * p[0] +
             150.0f * p[1] +
             77.0f * p[2]) *
            (1.0f / 256.0f);
    };
    float v =
        L(x0, y0) * (1.0f - tx) * (1.0f - ty) +
        L(x1, y0) * tx            * (1.0f - ty) +
        L(x0, y1) * (1.0f - tx) * ty +
        L(x1, y1) * tx            * ty;
    dst[dy * dW + dx] =
        (uint8_t)(v + 0.5f);
}
// -----------------------------------------------------------------------------
// Float atomic helpers
// -----------------------------------------------------------------------------
__device__ float atomicMinf_impl(float* addr, float val)
{
    int* p = (int*)addr;
    int assumed;
    int old = *p;
    do
    {
        assumed = old;
        old = atomicCAS(
            p,
            assumed,
            __float_as_int(
                min(
                    val,
                    __int_as_float(assumed)
                )
            )
        );
    } while (assumed != old);
    return __int_as_float(old);
}
__device__ float atomicMaxf_impl(float* addr, float val)
{
    int* p = (int*)addr;
    int assumed;
    int old = *p;
    do
    {
        assumed = old;
        old = atomicCAS(
            p,
            assumed,
            __float_as_int(
                max(
                    val,
                    __int_as_float(assumed)
                )
            )
        );
    } while (assumed != old);
    return __int_as_float(old);
}
// -----------------------------------------------------------------------------
// Min/max scan
// -----------------------------------------------------------------------------
__global__ void k_scanminmax(
    const float* __restrict__ d,
    float* mn,
    float* mx,
    int n)
{
    const int i =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float v = d[i];
    atomicMinf_impl(mn, v);
    atomicMaxf_impl(mx, v);
}
// -----------------------------------------------------------------------------
// Normalize
// -----------------------------------------------------------------------------
__global__ void k_normalize(
    const float* __restrict__ in,
    float* __restrict__ out,
    float mn,
    float mx,
    int n)
{
    const int i =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float r = mx - mn;
    if (r < 1e-6f)
        r = 1e-6f;
    out[i] =
        __saturatef((in[i] - mn) / r);
}
// -----------------------------------------------------------------------------
// Bidirectional flow warp/blend
// -----------------------------------------------------------------------------
//
// NvOF SHORT2 optical-flow vectors use S10.5 fixed point:
//
//     integer value / 32.0 = pixels
//
// The previous implementation divided by 64, which was incorrect.
__global__ void k_warp_blend(
    const float* __restrict__ prev,
    const float* __restrict__ curr,
    const short2* __restrict__ flow,
    float* __restrict__ out,
    int w,
    int h,
    float t)
{
    const int px =
        blockIdx.x * blockDim.x + threadIdx.x;
    const int py =
        blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h)
        return;
    const short2 f =
        flow[py * w + px];
    const float fx =
        f.x / 32.0f;
    const float fy =
        f.y / 32.0f;
    auto bsamp =
        [&](const float* img, float sx, float sy) -> float
    {
        sx = fmaxf(
            0.0f,
            fminf((float)(w - 1), sx)
        );
        sy = fmaxf(
            0.0f,
            fminf((float)(h - 1), sy)
        );
        const int x0 = (int)sx;
        const int y0 = (int)sy;
        const int x1 = min(x0 + 1, w - 1);
        const int y1 = min(y0 + 1, h - 1);
        const float tx = sx - x0;
        const float ty = sy - y0;
        return
            (1.0f - tx) * (1.0f - ty) *
                img[y0 * w + x0] +
            tx * (1.0f - ty) *
                img[y0 * w + x1] +
            (1.0f - tx) * ty *
                img[y1 * w + x0] +
            tx * ty *
                img[y1 * w + x1];
    };
    // Previous frame sampled backward.
    const float vf =
        bsamp(
            prev,
            px - t * fx,
            py - t * fy
        );
    // Current frame sampled backward from the other side.
    const float vb =
        bsamp(
            curr,
            px + (1.0f - t) * fx,
            py + (1.0f - t) * fy
        );
    out[py * w + px] =
        (1.0f - t) * vf +
        t * vb;
}
// =============================================================================
// NvOF state
// =============================================================================
#define MAX_SLOTS 2
#define MAX_INTERP 8
struct NvOFState
{
    int w = 0;
    int h = 0;
    bool ok = false;
    int maxInterp = 0;
    void* hDLL = nullptr;
    NV_OF_CUDA_API_FUNCTION_LIST fn{};
    NvOFHandle hOF = nullptr;
    NvOFGPUBufferHandle hGuide[MAX_SLOTS] = {};
    NvOFGPUBufferHandle hFlow = nullptr;
    CUdeviceptr d_guide[MAX_SLOTS] = {};
    CUdeviceptr d_flow = 0;
    float* d_depthLR[MAX_SLOTS] = {};
    float* d_warpOut[MAX_INTERP] = {};
    float* h_warpOut[MAX_INTERP] = {};
    float* d_minmax = nullptr;
    // Primary context retained by this object.
    CUdevice cudaDevice = 0;
    CUcontext cudaContext = nullptr;
    bool primaryContextRetained = false;
};
// =============================================================================
// NvOF error helper
// =============================================================================
static void nvof_log_last_error(NvOFState* st, const char* operation)
{
    if (!st || !st->hOF || !st->fn.nvOFGetLastError)
        return;
    char* msg = nullptr;
    const NV_OF_STATUS r =
        st->fn.nvOFGetLastError(st->hOF, &msg);
    if (r == NV_OF_SUCCESS && msg)
    {
        LOG_WARN(
            "NvOF: ",
            operation,
            " last error: ",
            msg
        );
    }
}
// =============================================================================
// NvOF creation
// =============================================================================
NvOFState* nvof_create(
    int w,
    int h,
    int maxInterp,
    const std::wstring& dllDir)
{
    if (w <= 0 || h <= 0)
    {
        LOG_WARN(
            "NvOF: invalid dimensions ",
            w,
            "x",
            h
        );
        return nullptr;
    }
    auto* st = new NvOFState;
    st->w = w;
    st->h = h;
    st->maxInterp =
        std::min(maxInterp, MAX_INTERP);
#ifdef _WIN32
    // -------------------------------------------------------------------------
    // Load the driver-installed NvOF DLL first.
    // -------------------------------------------------------------------------
    //
    // The driver copy is preferred because it is guaranteed to correspond to
    // the installed display driver.
    //
    // Fall back to the application's Win64 folder if necessary.
    st->hDLL =
        (void*)LOAD_SYSTEM_LIB(L"nvofapi64.dll");
    if (!st->hDLL)
    {
        const std::wstring p =
            dllDir + L"\\nvofapi64.dll";
        st->hDLL =
            (void*)LOAD_LIB(p.c_str());
    }
#else
    (void)dllDir;
    st->hDLL =
        LOAD_LIB("libnvofapi.so");
#endif
    if (!st->hDLL)
    {
        LOG_INFO(
            "NvOF: nvofapi64.dll not found — "
            "depth interpolation disabled"
        );
        delete st;
        return nullptr;
    }
    // -------------------------------------------------------------------------
    // Query driver-supported API version.
    // -------------------------------------------------------------------------
    //
    // This is the important part for version compatibility:
    //
    //     NvOFGetMaxSupportedApiVersion()
    //
    // returns the maximum API version supported by the installed driver.
    //
    // We use that value directly rather than attempting to force SDK 5.0.
    //
    // If the old driver does not expose this function, fall back to 0x50.
    auto fnMaxVer =
        reinterpret_cast<PFN_NV_OF_GET_MAX_SUPPORTED_API_VERSION>(
            GET_PROC(
                st->hDLL,
                "NvOFGetMaxSupportedApiVersion"
            )
        );
    uint32_t apiVer = NV_OF_API_VERSION;
    if (fnMaxVer)
    {
        uint32_t maxVer = 0;
        const NV_OF_STATUS vr =
            fnMaxVer(&maxVer);
        if (vr == NV_OF_SUCCESS && maxVer != 0)
        {
            apiVer = maxVer;
            LOG_INFO(
                "NvOF: driver max API version = 0x",
                std::hex,
                apiVer,
                std::dec
            );
        }
        else
        {
            LOG_WARN(
                "NvOF: NvOFGetMaxSupportedApiVersion failed "
                "r=",
                vr,
                "; falling back to 0x",
                std::hex,
                apiVer,
                std::dec
            );
        }
    }
    else
    {
        LOG_WARN(
            "NvOF: driver does not expose "
            "NvOFGetMaxSupportedApiVersion; "
            "falling back to 0x",
            std::hex,
            apiVer,
            std::dec
        );
    }
    // -------------------------------------------------------------------------
    // Create CUDA NvOF function table.
    // -------------------------------------------------------------------------
    auto entry =
        reinterpret_cast<PFN_NV_OF_API_CREATE_INSTANCE_CUDA>(
            GET_PROC(
                st->hDLL,
                "NvOFAPICreateInstanceCuda"
            )
        );
    if (!entry)
    {
        LOG_WARN(
            "NvOF: NvOFAPICreateInstanceCuda not found"
        );
        goto fail;
    }
    {
        const NV_OF_STATUS r =
            entry(
                apiVer,
                &st->fn
            );
        if (r != NV_OF_SUCCESS)
        {
            LOG_WARN(
                "NvOF: NvOFAPICreateInstanceCuda failed r=",
                r,
                " api=0x",
                std::hex,
                apiVer,
                std::dec
            );
            goto fail;
        }
    }
    // -------------------------------------------------------------------------
    // Get the CUDA device used by the current CUDA runtime.
    // -------------------------------------------------------------------------
    {
        int cudaDev = 0;
        cudaError_t cr =
            cudaGetDevice(&cudaDev);
        if (cr != cudaSuccess)
        {
            LOG_WARN(
                "NvOF: cudaGetDevice failed: ",
                cudaGetErrorString(cr)
            );
            goto fail;
        }
        st->cudaDevice =
            (CUdevice)cudaDev;
        CUresult cur =
            cuDevicePrimaryCtxRetain(
                &st->cudaContext,
                st->cudaDevice
            );
        if (cur != CUDA_SUCCESS ||
            !st->cudaContext)
        {
            LOG_WARN(
                "NvOF: cuDevicePrimaryCtxRetain failed r=",
                (int)cur
            );
            goto fail;
        }
        st->primaryContextRetained = true;
    }
    // -------------------------------------------------------------------------
    // Make sure the primary context is current on this thread.
    // -------------------------------------------------------------------------
    //
    // NvOF calls operate in the CUDA context supplied to
    // nvCreateOpticalFlowCuda().  CUDA/NvOF operations on this worker thread
    // also need the appropriate context current.
    {
        CUresult cur =
            cuCtxSetCurrent(
                st->cudaContext
            );
        if (cur != CUDA_SUCCESS)
        {
            LOG_WARN(
                "NvOF: cuCtxSetCurrent failed r=",
                (int)cur
            );
            goto fail;
        }
    }
    // -------------------------------------------------------------------------
    // Create NvOF handle.
    // -------------------------------------------------------------------------
    {
        const NV_OF_STATUS r =
            st->fn.nvCreateOpticalFlowCuda(
                st->cudaContext,
                &st->hOF
            );
        if (r != NV_OF_SUCCESS ||
            !st->hOF)
        {
            LOG_WARN(
                "NvOF: nvCreateOpticalFlowCuda failed r=",
                r
            );
            goto fail;
        }
    }
    // -------------------------------------------------------------------------
    // Initialize NvOF.
    // -------------------------------------------------------------------------
    //
    // These fields are the SDK 5.x NV_OF_INIT_PARAMS layout.
    //
    // Temporal hints are supported through EXECUTE_INPUT_PARAMS; they are not
    // an initialization field in SDK 5.x.
    {
        NV_OF_INIT_PARAMS ip{};
        ip.width =
            (uint32_t)w;
        ip.height =
            (uint32_t)h;
        ip.outGridSize =
            NV_OF_OUTPUT_VECTOR_GRID_SIZE_1;
        ip.hintGridSize =
            NV_OF_HINT_VECTOR_GRID_SIZE_1;
        ip.mode =
            NV_OF_MODE_OPTICALFLOW;
        ip.perfLevel =
            NV_OF_PERF_LEVEL_SLOW;
        ip.enableExternalHints =
            NV_OF_FALSE;
        ip.enableOutputCost =
            NV_OF_FALSE;
        ip.hPrivData =
            nullptr;
        // Undefined is the correct value for optical-flow mode.
        ip.disparityRange =
            NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
        ip.enableRoi =
            NV_OF_FALSE;
        const NV_OF_STATUS r =
            st->fn.nvOFInit(
                st->hOF,
                &ip
            );
        if (r != NV_OF_SUCCESS)
        {
            LOG_WARN(
                "NvOF: nvOFInit failed r=",
                r
            );
            nvof_log_last_error(
                st,
                "nvOFInit"
            );
            goto fail;
        }
    }
    // -------------------------------------------------------------------------
    // Create GRAYSCALE8 guide buffers.
    // -------------------------------------------------------------------------
    {
        NV_OF_BUFFER_DESCRIPTOR bd{};
        bd.width =
            (uint32_t)w;
        bd.height =
            (uint32_t)h;
        bd.bufferUsage =
            NV_OF_BUFFER_USAGE_INPUT;
        bd.bufferFormat =
            NV_OF_BUFFER_FORMAT_GRAYSCALE8;
        for (int s = 0; s < MAX_SLOTS; ++s)
        {
            const NV_OF_STATUS r =
                st->fn.nvOFCreateGPUBufferCuda(
                    st->hOF,
                    &bd,
                    NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
                    &st->hGuide[s]
                );
            if (r != NV_OF_SUCCESS ||
                !st->hGuide[s])
            {
                LOG_WARN(
                    "NvOF: guide buffer ",
                    s,
                    " create failed r=",
                    r
                );
                goto fail;
            }
            st->d_guide[s] =
                st->fn.nvOFGPUBufferGetCUdeviceptr(
                    st->hGuide[s]
                );
            if (!st->d_guide[s])
            {
                LOG_WARN(
                    "NvOF: guide buffer ",
                    s,
                    " returned null CUdeviceptr"
                );
                goto fail;
            }
        }
    }
    // -------------------------------------------------------------------------
    // Create SHORT2 flow output buffer.
    // -------------------------------------------------------------------------
    {
        NV_OF_BUFFER_DESCRIPTOR bd{};
        bd.width =
            (uint32_t)w;
        bd.height =
            (uint32_t)h;
        bd.bufferUsage =
            NV_OF_BUFFER_USAGE_OUTPUT;
        bd.bufferFormat =
            NV_OF_BUFFER_FORMAT_SHORT2;
        const NV_OF_STATUS r =
            st->fn.nvOFCreateGPUBufferCuda(
                st->hOF,
                &bd,
                NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
                &st->hFlow
            );
        if (r != NV_OF_SUCCESS ||
            !st->hFlow)
        {
            LOG_WARN(
                "NvOF: flow buffer create failed r=",
                r
            );
            goto fail;
        }
        st->d_flow =
            st->fn.nvOFGPUBufferGetCUdeviceptr(
                st->hFlow
            );
        if (!st->d_flow)
        {
            LOG_WARN(
                "NvOF: flow buffer returned null CUdeviceptr"
            );
            goto fail;
        }
    }
    // -------------------------------------------------------------------------
    // Allocate LR depth buffers and warp outputs.
    // -------------------------------------------------------------------------
    for (int s = 0; s < MAX_SLOTS; ++s)
    {
        cudaError_t r =
            cudaMalloc(
                &st->d_depthLR[s],
                (size_t)w * (size_t)h *
                    sizeof(float)
            );
        if (r != cudaSuccess)
        {
            LOG_WARN(
                "NvOF: cudaMalloc depthLR failed: ",
                cudaGetErrorString(r)
            );
            goto fail;
        }
    }
    for (int i = 0;
         i < st->maxInterp;
         ++i)
    {
        cudaError_t r =
            cudaMalloc(
                &st->d_warpOut[i],
                (size_t)w * (size_t)h *
                    sizeof(float)
            );
        if (r != cudaSuccess)
        {
            LOG_WARN(
                "NvOF: cudaMalloc warpOut failed: ",
                cudaGetErrorString(r)
            );
            goto fail;
        }
        r =
            cudaMallocHost(
                &st->h_warpOut[i],
                (size_t)w * (size_t)h *
                    sizeof(float)
            );
        if (r != cudaSuccess)
        {
            LOG_WARN(
                "NvOF: cudaMallocHost warpOut failed: ",
                cudaGetErrorString(r)
            );
            goto fail;
        }
    }
    {
        cudaError_t r =
            cudaMalloc(
                &st->d_minmax,
                2 * sizeof(float)
            );
        if (r != cudaSuccess)
        {
            LOG_WARN(
                "NvOF: cudaMalloc minmax failed: ",
                cudaGetErrorString(r)
            );
            goto fail;
        }
    }
    st->ok = true;
    LOG_INFO(
        "NvOF: ready at ",
        w,
        "x",
        h,
        " maxInterp=",
        st->maxInterp,
        " api=0x",
        std::hex,
        apiVer,
        std::dec
    );
    return st;
fail:
    nvof_destroy(st);
    return nullptr;
}
// =============================================================================
// NvOF destruction
// =============================================================================
void nvof_destroy(NvOFState* st)
{
    if (!st)
        return;
    // Ensure all work launched through the runtime on this thread has
    // completed before destroying NvOF-owned resources.
    //
    // This is especially useful because NvOF and CUDA resources are shared
    // with the inference pipeline.
    cudaDeviceSynchronize();
    for (int s = 0;
         s < MAX_SLOTS;
         ++s)
    {
        if (st->hGuide[s] &&
            st->fn.nvOFDestroyGPUBufferCuda)
        {
            st->fn.nvOFDestroyGPUBufferCuda(
                st->hGuide[s]
            );
            st->hGuide[s] = nullptr;
        }
        if (st->d_depthLR[s])
        {
            cudaFree(
                st->d_depthLR[s]
            );
            st->d_depthLR[s] = nullptr;
        }
    }
    if (st->hFlow &&
        st->fn.nvOFDestroyGPUBufferCuda)
    {
        st->fn.nvOFDestroyGPUBufferCuda(
            st->hFlow
        );
        st->hFlow = nullptr;
    }
    for (int i = 0;
         i < MAX_INTERP;
         ++i)
    {
        if (st->d_warpOut[i])
        {
            cudaFree(
                st->d_warpOut[i]
            );
            st->d_warpOut[i] = nullptr;
        }
        if (st->h_warpOut[i])
        {
            cudaFreeHost(
                st->h_warpOut[i]
            );
            st->h_warpOut[i] = nullptr;
        }
    }
    if (st->d_minmax)
    {
        cudaFree(
            st->d_minmax
        );
        st->d_minmax = nullptr;
    }
    if (st->hOF &&
        st->fn.nvOFDestroy)
    {
        st->fn.nvOFDestroy(
            st->hOF
        );
        st->hOF = nullptr;
    }
    // We retained the primary context during creation, so release exactly
    // once here.
    if (st->primaryContextRetained)
    {
        cuDevicePrimaryCtxRelease(
            st->cudaDevice
        );
        st->primaryContextRetained = false;
        st->cudaContext = nullptr;
    }
    if (st->hDLL)
    {
        FREE_LIB(
            st->hDLL
        );
        st->hDLL = nullptr;
    }
    delete st;
}
// =============================================================================
// Status
// =============================================================================
bool nvof_available(NvOFState* st)
{
    return st && st->ok;
}
void nvof_dims(
    NvOFState* st,
    int* w,
    int* h)
{
    if (!st)
        return;
    if (w)
        *w = st->w;
    if (h)
        *h = st->h;
}
// =============================================================================
// Prepare slot
// =============================================================================
void nvof_prepare_slot(
    NvOFState* st,
    int slot,
    const uint8_t* d_guideBGRA,
    int srcW,
    int srcH,
    int srcStride,
    const float* d_outSlice,
    float* /*d_minmax_scratch*/,
    int mw,
    int mh,
    void* stream)
{
    if (!st ||
        !st->ok ||
        slot < 0 ||
        slot >= MAX_SLOTS)
    {
        return;
    }
    cudaStream_t s =
        (cudaStream_t)stream;
    // -------------------------------------------------------------------------
    // 1. Downsample BGRA -> GRAYSCALE8
    // -------------------------------------------------------------------------
    {
        dim3 blk(16, 16);
        dim3 grd(
            (mw + 15) / 16,
            (mh + 15) / 16
        );
        k_bgra_to_gray8<<<
            grd,
            blk,
            0,
            s
        >>>(
            d_guideBGRA,
            srcW,
            srcH,
            srcStride,
            reinterpret_cast<uint8_t*>(
                st->d_guide[slot]
            ),
            mw,
            mh
        );
    }
    // -------------------------------------------------------------------------
    // 2. Normalize raw TRT output -> [0,1]
    // -------------------------------------------------------------------------
    const int n =
        mw * mh;
    float init_mn =
        1e30f;
    float init_mx =
        -1e30f;
    cudaMemcpyAsync(
        st->d_minmax,
        &init_mn,
        sizeof(float),
        cudaMemcpyHostToDevice,
        s
    );
    cudaMemcpyAsync(
        st->d_minmax + 1,
        &init_mx,
        sizeof(float),
        cudaMemcpyHostToDevice,
        s
    );
    k_scanminmax<<<
        (n + 255) / 256,
        256,
        0,
        s
    >>>(
        d_outSlice,
        st->d_minmax,
        st->d_minmax + 1,
        n
    );
    float mn = 0.0f;
    float mx = 1.0f;
    cudaMemcpyAsync(
        &mn,
        st->d_minmax,
        sizeof(float),
        cudaMemcpyDeviceToHost,
        s
    );
    cudaMemcpyAsync(
        &mx,
        st->d_minmax + 1,
        sizeof(float),
        cudaMemcpyDeviceToHost,
        s
    );
    // The host values are needed before launching normalize.
    cudaStreamSynchronize(s);
    k_normalize<<<
        (n + 255) / 256,
        256,
        0,
        s
    >>>(
        d_outSlice,
        st->d_depthLR[slot],
        mn,
        mx,
        n
    );
}
// =============================================================================
// Execute NvOF
// =============================================================================
bool nvof_execute(
    NvOFState* st,
    int prevSlot,
    int currSlot,
    void* stream)
{
    if (!st ||
        !st->ok ||
        prevSlot < 0 ||
        prevSlot >= MAX_SLOTS ||
        currSlot < 0 ||
        currSlot >= MAX_SLOTS)
    {
        return false;
    }
    CUstream cuStream =
        (CUstream)stream;
    // -------------------------------------------------------------------------
    // Set NvOF input/output CUDA streams.
    // -------------------------------------------------------------------------
    //
    // SDK 5.x does NOT put CUDA streams inside the execute input structure.
    // They are configured through this function.
    if (!st->fn.nvOFSetIOCudaStreams)
    {
        LOG_WARN(
            "NvOF: nvOFSetIOCudaStreams missing from function table"
        );
        return false;
    }
    {
        const NV_OF_STATUS r =
            st->fn.nvOFSetIOCudaStreams(
                st->hOF,
                cuStream,
                cuStream
            );
        if (r != NV_OF_SUCCESS)
        {
            LOG_WARN(
                "NvOF: nvOFSetIOCudaStreams failed r=",
                r
            );
            nvof_log_last_error(
                st,
                "nvOFSetIOCudaStreams"
            );
            return false;
        }
    }
    // -------------------------------------------------------------------------
    // Execute
    // -------------------------------------------------------------------------
    NV_OF_EXECUTE_INPUT_PARAMS in{};
    in.inputFrame =
        st->hGuide[currSlot];
    in.referenceFrame =
        st->hGuide[prevSlot];
    in.externalHints =
        nullptr;
    in.disableTemporalHints =
        NV_OF_FALSE;
    in.padding =
        0;
    in.hPrivData =
        nullptr;
    in.padding2 =
        0;
    in.numRois =
        0;
    in.roiData =
        nullptr;
    NV_OF_EXECUTE_OUTPUT_PARAMS out{};
    out.outputBuffer =
        st->hFlow;
    out.outputCostBuffer =
        nullptr;
    out.hPrivData =
        nullptr;
    const NV_OF_STATUS r =
        st->fn.nvOFExecute(
            st->hOF,
            &in,
            &out
        );
    if (r != NV_OF_SUCCESS)
    {
        LOG_WARN(
            "NvOF: execute failed r=",
            r
        );
        nvof_log_last_error(
            st,
            "nvOFExecute"
        );
        return false;
    }
    return true;
}
// =============================================================================
// Warp
// =============================================================================
void nvof_warp(
    NvOFState* st,
    int prevSlot,
    int currSlot,
    float* d_out,
    float t,
    void* stream)
{
    if (!st ||
        !st->ok ||
        !d_out)
    {
        return;
    }
    if (prevSlot < 0 ||
        prevSlot >= MAX_SLOTS ||
        currSlot < 0 ||
        currSlot >= MAX_SLOTS)
    {
        return;
    }
    cudaStream_t s =
        (cudaStream_t)stream;
    dim3 blk(16, 16);
    dim3 grd(
        (st->w + 15) / 16,
        (st->h + 15) / 16
    );
    k_warp_blend<<<
        grd,
        blk,
        0,
        s
    >>>(
        st->d_depthLR[prevSlot],
        st->d_depthLR[currSlot],
        reinterpret_cast<const short2*>(
            st->d_flow
        ),
        d_out,
        st->w,
        st->h,
        t
    );
}