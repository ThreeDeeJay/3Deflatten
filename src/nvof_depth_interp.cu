// SPDX-License-Identifier: GPL-3.0-or-later
// nvof_depth_interp.cu — compiled with nvcc.
//
// NVIDIA Optical Flow is loaded dynamically at runtime.
// No static link against nvofapi64.lib is required.
//
// This file intentionally mirrors the CUDA NvOF ABI from NVIDIA's
// Optical Flow SDK headers.  The function-table order and structure
// layouts are ABI-critical.
#include "nvof_depth_interp.h"
#include "logger.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define LOAD_LIB(n)                                                     \
    LoadLibraryExW(                                                     \
        (n),                                                            \
        nullptr,                                                        \
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |                              \
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |                             \
        LOAD_LIBRARY_SEARCH_USER_DIRS)
#define LOAD_SYSTEM_LIB(n)                                              \
    LoadLibraryExW(                                                     \
        (n),                                                            \
        nullptr,                                                        \
        LOAD_LIBRARY_SEARCH_SYSTEM32)
#define FREE_LIB(h)                                                     \
    FreeLibrary(reinterpret_cast<HMODULE>(h))
#define GET_PROC(h, n)                                                  \
    GetProcAddress(reinterpret_cast<HMODULE>(h), (n))
#else
#include <dlfcn.h>
#define LOAD_LIB(n)          dlopen((n), RTLD_LAZY)
#define LOAD_SYSTEM_LIB(n)   dlopen((n), RTLD_LAZY)
#define FREE_LIB(h)          dlclose((h))
#define GET_PROC(h, n)       dlsym((h), (n))
#endif
// =============================================================================
// NVIDIA NvOF ABI definitions
// =============================================================================
//
// These definitions intentionally match NVIDIA's public:
//   nvOpticalFlowCommon.h
//   nvOpticalFlowCuda.h
//
// Do not reorder these structures or function-table entries.
//
typedef int32_t NV_OF_STATUS;
enum
{
    NV_OF_SUCCESS                   = 0,
    NV_OF_ERR_OF_NOT_AVAILABLE      = 1,
    NV_OF_ERR_UNSUPPORTED_DEVICE    = 2,
    NV_OF_ERR_DEVICE_DOES_NOT_EXIST = 3,
    NV_OF_ERR_INVALID_PTR           = 4,
    NV_OF_ERR_INVALID_PARAM         = 5,
    NV_OF_ERR_INVALID_CALL          = 6,
    NV_OF_ERR_INVALID_VERSION       = 7,
    NV_OF_ERR_OUT_OF_MEMORY         = 8,
    NV_OF_ERR_NOT_INITIALIZED       = 9,
    NV_OF_ERR_UNSUPPORTED_FEATURE   = 10,
    NV_OF_ERR_GENERIC               = 11
};
typedef enum _NV_OF_BOOL
{
    NV_OF_FALSE = 0,
    NV_OF_TRUE  = 1
} NV_OF_BOOL;
// -----------------------------------------------------------------------------
// Capabilities
// -----------------------------------------------------------------------------
typedef enum _NV_OF_CAPS
{
    NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES = 0,
    NV_OF_CAPS_SUPPORTED_HINT_GRID_SIZES   = 1,
    NV_OF_CAPS_SUPPORT_HINT_WITH_OF_MODE   = 2,
    NV_OF_CAPS_SUPPORT_HINT_WITH_ST_MODE   = 3,
    NV_OF_CAPS_WIDTH_MIN                   = 4,
    NV_OF_CAPS_HEIGHT_MIN                  = 5,
    NV_OF_CAPS_WIDTH_MAX                   = 6,
    NV_OF_CAPS_HEIGHT_MAX                  = 7,
    NV_OF_CAPS_SUPPORT_ROI                 = 8,
    NV_OF_CAPS_SUPPORT_ROI_MAX_NUM        = 9,
    NV_OF_CAPS_SUPPORT_MAX                 = 10
} NV_OF_CAPS;
// -----------------------------------------------------------------------------
// Performance
// -----------------------------------------------------------------------------
typedef enum _NV_OF_PERF_LEVEL
{
    NV_OF_PERF_LEVEL_UNDEFINED = 0,
    NV_OF_PERF_LEVEL_SLOW      = 5,
    NV_OF_PERF_LEVEL_MEDIUM    = 10,
    NV_OF_PERF_LEVEL_FAST      = 20
} NV_OF_PERF_LEVEL;
// -----------------------------------------------------------------------------
// Output vector grid
// -----------------------------------------------------------------------------
typedef enum _NV_OF_OUTPUT_VECTOR_GRID_SIZE
{
    NV_OF_OUTPUT_VECTOR_GRID_SIZE_UNDEFINED = 0,
    NV_OF_OUTPUT_VECTOR_GRID_SIZE_1         = 1,
    NV_OF_OUTPUT_VECTOR_GRID_SIZE_2         = 2,
    NV_OF_OUTPUT_VECTOR_GRID_SIZE_4         = 4
} NV_OF_OUTPUT_VECTOR_GRID_SIZE;
// -----------------------------------------------------------------------------
// Hint vector grid
// -----------------------------------------------------------------------------
typedef enum _NV_OF_HINT_VECTOR_GRID_SIZE
{
    NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED = 0,
    NV_OF_HINT_VECTOR_GRID_SIZE_1         = 1,
    NV_OF_HINT_VECTOR_GRID_SIZE_2         = 2,
    NV_OF_HINT_VECTOR_GRID_SIZE_4         = 4,
    NV_OF_HINT_VECTOR_GRID_SIZE_8         = 8
} NV_OF_HINT_VECTOR_GRID_SIZE;
// -----------------------------------------------------------------------------
// Buffer format
// -----------------------------------------------------------------------------
typedef enum _NV_OF_BUFFER_FORMAT
{
    NV_OF_BUFFER_FORMAT_UNDEFINED  = 0,
    NV_OF_BUFFER_FORMAT_GRAYSCALE8 = 1,
    NV_OF_BUFFER_FORMAT_NV12       = 2,
    NV_OF_BUFFER_FORMAT_ABGR8      = 3,
    NV_OF_BUFFER_FORMAT_SHORT      = 4,
    NV_OF_BUFFER_FORMAT_SHORT2     = 5,
    NV_OF_BUFFER_FORMAT_UINT       = 6,
    NV_OF_BUFFER_FORMAT_UINT8      = 7
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
// Operating mode
// -----------------------------------------------------------------------------
typedef enum _NV_OF_MODE
{
    NV_OF_MODE_UNDEFINED       = 0,
    NV_OF_MODE_OPTICALFLOW     = 1,
    NV_OF_MODE_STEREODISPARITY = 2
} NV_OF_MODE;
// -----------------------------------------------------------------------------
// CUDA buffer type
// -----------------------------------------------------------------------------
typedef enum _NV_OF_CUDA_BUFFER_TYPE
{
    NV_OF_CUDA_BUFFER_TYPE_UNDEFINED   = 0,
    NV_OF_CUDA_BUFFER_TYPE_CUARRAY     = 1,
    NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR = 2
} NV_OF_CUDA_BUFFER_TYPE;
// -----------------------------------------------------------------------------
// Stereo disparity range
// -----------------------------------------------------------------------------
typedef enum _NV_OF_STEREO_DISPARITY_RANGE
{
    NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED = 0,
    NV_OF_STEREO_DISPARITY_RANGE_128       = 128,
    NV_OF_STEREO_DISPARITY_RANGE_256       = 256
} NV_OF_STEREO_DISPARITY_RANGE;
// -----------------------------------------------------------------------------
// Opaque handles
// -----------------------------------------------------------------------------
typedef void* NvOFHandle;
typedef void* NvOFGPUBufferHandle;
typedef void* NvOFPrivDataHandle;
// -----------------------------------------------------------------------------
// Init parameters
// -----------------------------------------------------------------------------
typedef struct _NV_OF_INIT_PARAMS
{
    uint32_t width;
    uint32_t height;
    NV_OF_OUTPUT_VECTOR_GRID_SIZE outGridSize;
    NV_OF_HINT_VECTOR_GRID_SIZE   hintGridSize;
    NV_OF_MODE       mode;
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
    NV_OF_BUFFER_USAGE  bufferUsage;
    NV_OF_BUFFER_FORMAT bufferFormat;
} NV_OF_BUFFER_DESCRIPTOR;
// -----------------------------------------------------------------------------
// ROI rectangle
// -----------------------------------------------------------------------------
typedef struct _NV_OF_ROI_RECT
{
    uint32_t start_x;
    uint32_t start_y;
    uint32_t width;
    uint32_t height;
} NV_OF_ROI_RECT;
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
    NV_OF_ROI_RECT* roiData;
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
// CUDA buffer stride information
// -----------------------------------------------------------------------------
#define NVOF_MAX_NUM_PLANES 3
typedef struct _NV_OF_BUFFER_STRIDE
{
    uint32_t strideXInBytes;
    uint32_t strideYInBytes;
} NV_OF_BUFFER_STRIDE;
typedef struct _NV_OF_CUDA_BUFFER_STRIDE_INFO
{
    NV_OF_BUFFER_STRIDE strideInfo[NVOF_MAX_NUM_PLANES];
    uint32_t numPlanes;
} NV_OF_CUDA_BUFFER_STRIDE_INFO;
// -----------------------------------------------------------------------------
// Calling convention
// -----------------------------------------------------------------------------
#ifdef _WIN32
#define NVOF_CALL __stdcall
#else
#define NVOF_CALL
#endif
// -----------------------------------------------------------------------------
// Function pointers
// -----------------------------------------------------------------------------
typedef NV_OF_STATUS (NVOF_CALL *PFNNVCREATEOPTICALFLOWCUDA)(
    CUcontext,
    NvOFHandle*
);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFINIT)(
    NvOFHandle,
    const NV_OF_INIT_PARAMS*
);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFCREATEGPUBUFFERCUDA)(
    NvOFHandle,
    const NV_OF_BUFFER_DESCRIPTOR*,
    NV_OF_CUDA_BUFFER_TYPE,
    NvOFGPUBufferHandle*
);
typedef CUarray (NVOF_CALL *PFNNVOFGPUBUFFERGETCUARRAY)(
    NvOFGPUBufferHandle
);
typedef CUdeviceptr (NVOF_CALL *PFNNVOFGPUBUFFERGETCUDEVICEPTR)(
    NvOFGPUBufferHandle
);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFGPUBUFFERGETSTRIDEINFO)(
    NvOFGPUBufferHandle,
    NV_OF_CUDA_BUFFER_STRIDE_INFO*
);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFSETIOCUDASTREAMS)(
    NvOFHandle,
    CUstream,
    CUstream
);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFEXECUTE)(
    NvOFHandle,
    const NV_OF_EXECUTE_INPUT_PARAMS*,
    NV_OF_EXECUTE_OUTPUT_PARAMS*
);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFDESTROYGPUBUFFERCUDA)(
    NvOFGPUBufferHandle
);
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFDESTROY)(
    NvOFHandle
);
// IMPORTANT:
// NVIDIA's actual signature is:
//
//   nvOFGetLastError(hOf, char lastError[], uint32_t *size)
//
// NOT char**.
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFGETLASTERROR)(
    NvOFHandle,
    char[],
    uint32_t*
);
// IMPORTANT:
// NVIDIA's actual signature is:
//
//   nvOFGetCaps(hOf, capsParam, capsVal, size)
//
// Both capsVal and size are uint32_t*.
typedef NV_OF_STATUS (NVOF_CALL *PFNNVOFGETCAPS)(
    NvOFHandle,
    NV_OF_CAPS,
    uint32_t*,
    uint32_t*
);
// -----------------------------------------------------------------------------
// CUDA API function table
// -----------------------------------------------------------------------------
typedef struct _NV_OF_CUDA_API_FUNCTION_LIST
{
    PFNNVCREATEOPTICALFLOWCUDA     nvCreateOpticalFlowCuda;
    PFNNVOFINIT                    nvOFInit;
    PFNNVOFCREATEGPUBUFFERCUDA     nvOFCreateGPUBufferCuda;
    PFNNVOFGPUBUFFERGETCUARRAY     nvOFGPUBufferGetCUarray;
    PFNNVOFGPUBUFFERGETCUDEVICEPTR nvOFGPUBufferGetCUdeviceptr;
    PFNNVOFGPUBUFFERGETSTRIDEINFO  nvOFGPUBufferGetStrideInfo;
    PFNNVOFSETIOCUDASTREAMS         nvOFSetIOCudaStreams;
    PFNNVOFEXECUTE                  nvOFExecute;
    PFNNVOFDESTROYGPUBUFFERCUDA     nvOFDestroyGPUBufferCuda;
    PFNNVOFDESTROY                  nvOFDestroy;
    PFNNVOFGETLASTERROR             nvOFGetLastError;
    PFNNVOFGETCAPS                  nvOFGetCaps;
} NV_OF_CUDA_API_FUNCTION_LIST;
// -----------------------------------------------------------------------------
// DLL entry points
// -----------------------------------------------------------------------------
typedef NV_OF_STATUS (NVOF_CALL *PFN_NV_OF_GET_MAX_SUPPORTED_API_VERSION)(
    uint32_t*
);
typedef NV_OF_STATUS (NVOF_CALL *PFN_NV_OF_API_CREATE_INSTANCE_CUDA)(
    uint32_t,
    NV_OF_CUDA_API_FUNCTION_LIST*
);
// =============================================================================
// CUDA kernels
// =============================================================================
__global__ void k_bgra_to_gray8(
    const uint8_t* __restrict__ src,
    int srcW,
    int srcH,
    int srcStride,
    uint8_t* __restrict__ dst,
    int dstW,
    int dstH)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH)
        return;
    const float fx =
        (x + 0.5f) * srcW / static_cast<float>(dstW) - 0.5f;
    const float fy =
        (y + 0.5f) * srcH / static_cast<float>(dstH) - 0.5f;
    int x0 = static_cast<int>(floorf(fx));
    int y0 = static_cast<int>(floorf(fy));
    x0 = max(0, min(x0, srcW - 1));
    y0 = max(0, min(y0, srcH - 1));
    const int x1 = min(x0 + 1, srcW - 1);
    const int y1 = min(y0 + 1, srcH - 1);
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const uint8_t* p00 =
        src + y0 * srcStride + x0 * 4;
    const uint8_t* p10 =
        src + y0 * srcStride + x1 * 4;
    const uint8_t* p01 =
        src + y1 * srcStride + x0 * 4;
    const uint8_t* p11 =
        src + y1 * srcStride + x1 * 4;
    // BGRA -> luminance.
    const float l00 =
        (29.0f * p00[0] +
         150.0f * p00[1] +
         77.0f * p00[2]) / 256.0f;
    const float l10 =
        (29.0f * p10[0] +
         150.0f * p10[1] +
         77.0f * p10[2]) / 256.0f;
    const float l01 =
        (29.0f * p01[0] +
         150.0f * p01[1] +
         77.0f * p01[2]) / 256.0f;
    const float l11 =
        (29.0f * p11[0] +
         150.0f * p11[1] +
         77.0f * p11[2]) / 256.0f;
    const float top = l00 + (l10 - l00) * tx;
    const float bot = l01 + (l11 - l01) * tx;
    float value = top + (bot - top) * ty;
    value = fminf(255.0f, fmaxf(0.0f, value));
    dst[y * dstW + x] =
        static_cast<uint8_t>(value + 0.5f);
}
// -----------------------------------------------------------------------------
// Float atomic min/max
// -----------------------------------------------------------------------------
__device__ float atomicMinFloat(float* address, float value)
{
    int* addressAsInt = reinterpret_cast<int*>(address);
    int old = *addressAsInt;
    while (true)
    {
        const int assumed = old;
        const float current =
            __int_as_float(assumed);
        if (current <= value)
            return current;
        old = atomicCAS(
            addressAsInt,
            assumed,
            __float_as_int(value)
        );
        if (old == assumed)
            return value;
    }
}
__device__ float atomicMaxFloat(float* address, float value)
{
    int* addressAsInt = reinterpret_cast<int*>(address);
    int old = *addressAsInt;
    while (true)
    {
        const int assumed = old;
        const float current =
            __int_as_float(assumed);
        if (current >= value)
            return current;
        old = atomicCAS(
            addressAsInt,
            assumed,
            __float_as_int(value)
        );
        if (old == assumed)
            return value;
    }
}
// -----------------------------------------------------------------------------
// Scan depth for min/max
// -----------------------------------------------------------------------------
__global__ void k_scan_minmax(
    const float* __restrict__ input,
    float* minmax,
    int count)
{
    const int i =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const float value = input[i];
    if (isfinite(value))
    {
        atomicMinFloat(&minmax[0], value);
        atomicMaxFloat(&minmax[1], value);
    }
}
// -----------------------------------------------------------------------------
// Normalize depth into [0,1].
//
// minmax[0] = minimum
// minmax[1] = maximum
// -----------------------------------------------------------------------------
__global__ void k_normalize_depth(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ minmax,
    int count)
{
    const int i =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const float mn = minmax[0];
    const float mx = minmax[1];
    const float value = input[i];
    if (!isfinite(value))
    {
        output[i] = 0.0f;
        return;
    }
    const float range = mx - mn;
    if (!(range > 1.0e-8f) || !isfinite(range))
    {
        output[i] = 0.5f;
        return;
    }
    float normalized =
        (value - mn) / range;
    normalized =
        fminf(1.0f, fmaxf(0.0f, normalized));
    output[i] = normalized;
}
// -----------------------------------------------------------------------------
// Convert NvOF S10.5 SHORT2 flow to full-resolution float2.
//
// NvOF output is stored at:
//
//   ceil(width / grid) x ceil(height / grid)
//
// Each vector is:
//
//   int16 / 32.0
//
// because the low five bits are fractional bits.
//
// Bilinear interpolation is used when grid > 1.
// -----------------------------------------------------------------------------
__global__ void k_expand_flow(
    const uint8_t* __restrict__ flowBytes,
    size_t flowStride,
    int flowW,
    int flowH,
    int grid,
    float2* __restrict__ fullFlow,
    int fullW,
    int fullH)
{
    const int x =
        blockIdx.x * blockDim.x + threadIdx.x;
    const int y =
        blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= fullW || y >= fullH)
        return;
    const float gx =
        static_cast<float>(x) /
        static_cast<float>(grid);
    const float gy =
        static_cast<float>(y) /
        static_cast<float>(grid);
    const int x0 =
        max(0, min(static_cast<int>(floorf(gx)), flowW - 1));
    const int y0 =
        max(0, min(static_cast<int>(floorf(gy)), flowH - 1));
    const int x1 =
        min(x0 + 1, flowW - 1);
    const int y1 =
        min(y0 + 1, flowH - 1);
    const float tx =
        gx - static_cast<float>(x0);
    const float ty =
        gy - static_cast<float>(y0);
    auto readFlow =
        [&](int fx, int fy) -> float2
    {
        const uint8_t* row =
            flowBytes + static_cast<size_t>(fy) * flowStride;
        const int16_t* v =
            reinterpret_cast<const int16_t*>(
                row + static_cast<size_t>(fx) * sizeof(int16_t) * 2
            );
        return make_float2(
            static_cast<float>(v[0]) / 32.0f,
            static_cast<float>(v[1]) / 32.0f
        );
    };
    const float2 f00 = readFlow(x0, y0);
    const float2 f10 = readFlow(x1, y0);
    const float2 f01 = readFlow(x0, y1);
    const float2 f11 = readFlow(x1, y1);
    const float2 top = make_float2(
        f00.x + (f10.x - f00.x) * tx,
        f00.y + (f10.y - f00.y) * tx
    );
    const float2 bottom = make_float2(
        f01.x + (f11.x - f01.x) * tx,
        f01.y + (f11.y - f01.y) * tx
    );
    fullFlow[y * fullW + x] =
        make_float2(
            top.x + (bottom.x - top.x) * ty,
            top.y + (bottom.y - top.y) * ty
        );
}
// -----------------------------------------------------------------------------
// Bilinear scalar sampling
// -----------------------------------------------------------------------------
__device__ float sample_float_bilinear(
    const float* data,
    int width,
    int height,
    float x,
    float y)
{
    x = fminf(
        static_cast<float>(width - 1),
        fmaxf(0.0f, x)
    );
    y = fminf(
        static_cast<float>(height - 1),
        fmaxf(0.0f, y)
    );
    const int x0 =
        static_cast<int>(floorf(x));
    const int y0 =
        static_cast<int>(floorf(y));
    const int x1 =
        min(x0 + 1, width - 1);
    const int y1 =
        min(y0 + 1, height - 1);
    const float tx =
        x - static_cast<float>(x0);
    const float ty =
        y - static_cast<float>(y0);
    const float a =
        data[y0 * width + x0];
    const float b =
        data[y0 * width + x1];
    const float c =
        data[y1 * width + x0];
    const float d =
        data[y1 * width + x1];
    const float top =
        a + (b - a) * tx;
    const float bottom =
        c + (d - c) * tx;
    return top + (bottom - top) * ty;
}
// -----------------------------------------------------------------------------
// Warp previous/current depth using current->reference NvOF flow.
//
// NvOF's forward flow is the movement from input frame to reference frame.
// We execute:
//
//     inputFrame     = current
//     referenceFrame = previous
//
// Therefore, for an intermediate time t:
//
//     previous sample = p + t * flow
//     current  sample = p - (1-t) * flow
//
// -----------------------------------------------------------------------------
__global__ void k_warp_blend(
    const float* __restrict__ prevDepth,
    const float* __restrict__ currDepth,
    const float2* __restrict__ fullFlow,
    float* __restrict__ output,
    int width,
    int height,
    float t)
{
    const int x =
        blockIdx.x * blockDim.x + threadIdx.x;
    const int y =
        blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    const int index =
        y * width + x;
    const float2 flow =
        fullFlow[index];
    const float prevX =
        static_cast<float>(x) +
        t * flow.x;
    const float prevY =
        static_cast<float>(y) +
        t * flow.y;
    const float currX =
        static_cast<float>(x) -
        (1.0f - t) * flow.x;
    const float currY =
        static_cast<float>(y) -
        (1.0f - t) * flow.y;
    const float prev =
        sample_float_bilinear(
            prevDepth,
            width,
            height,
            prevX,
            prevY
        );
    const float curr =
        sample_float_bilinear(
            currDepth,
            width,
            height,
            currX,
            currY
        );
    output[index] =
        (1.0f - t) * prev +
        t * curr;
}
// =============================================================================
// NvOF state
// =============================================================================
struct NvOFState
{
    int width = 0;
    int height = 0;
    int maxInterp = 0;
    int gridSize = 1;
    int flowWidth = 0;
    int flowHeight = 0;
    bool initialized = false;
    void* library = nullptr;
    CUcontext previousContext = nullptr;
    CUcontext ofContext = nullptr;
    bool ownsPrimaryContext = false;
    NvOFHandle hOF = nullptr;
    NV_OF_CUDA_API_FUNCTION_LIST fn{};
    // Per-slot input frames.
    NvOFGPUBufferHandle guide[2] = {
        nullptr,
        nullptr
    };
    // NvOF output flow.
    NvOFGPUBufferHandle flow = nullptr;
    // NvOF output stride.
    uint32_t flowStrideX = 0;
    uint32_t flowStrideY = 0;
    // Full LR-resolution flow after grid expansion.
    float2* d_flowFull = nullptr;
    // Normalized depth per pipeline slot.
    float* d_depth[2] = {
        nullptr,
        nullptr
    };
    // Scratch:
    //
    // [0] = min
    // [1] = max
    //
    float* d_minmax = nullptr;
    // CUDA stream used by the last NvOF execution.
    CUstream lastStream = nullptr;
};
// =============================================================================
// Error helpers
// =============================================================================
static const char* nvof_status_string(
    NV_OF_STATUS status)
{
    switch (status)
    {
        case NV_OF_SUCCESS:
            return "NV_OF_SUCCESS";
        case NV_OF_ERR_OF_NOT_AVAILABLE:
            return "NV_OF_ERR_OF_NOT_AVAILABLE";
        case NV_OF_ERR_UNSUPPORTED_DEVICE:
            return "NV_OF_ERR_UNSUPPORTED_DEVICE";
        case NV_OF_ERR_DEVICE_DOES_NOT_EXIST:
            return "NV_OF_ERR_DEVICE_DOES_NOT_EXIST";
        case NV_OF_ERR_INVALID_PTR:
            return "NV_OF_ERR_INVALID_PTR";
        case NV_OF_ERR_INVALID_PARAM:
            return "NV_OF_ERR_INVALID_PARAM";
        case NV_OF_ERR_INVALID_CALL:
            return "NV_OF_ERR_INVALID_CALL";
        case NV_OF_ERR_INVALID_VERSION:
            return "NV_OF_ERR_INVALID_VERSION";
        case NV_OF_ERR_OUT_OF_MEMORY:
            return "NV_OF_ERR_OUT_OF_MEMORY";
        case NV_OF_ERR_NOT_INITIALIZED:
            return "NV_OF_ERR_NOT_INITIALIZED";
        case NV_OF_ERR_UNSUPPORTED_FEATURE:
            return "NV_OF_ERR_UNSUPPORTED_FEATURE";
        case NV_OF_ERR_GENERIC:
            return "NV_OF_ERR_GENERIC";
        default:
            return "NV_OF_ERR_UNKNOWN";
    }
}
static void nvof_log_last_error(
    NvOFState* st)
{
    if (!st ||
        !st->hOF ||
        !st->fn.nvOFGetLastError)
    {
        return;
    }
    char message[256] = {};
    uint32_t size =
        static_cast<uint32_t>(sizeof(message));
    const NV_OF_STATUS status =
        st->fn.nvOFGetLastError(
            st->hOF,
            message,
            &size
        );
    if (status == NV_OF_SUCCESS &&
        message[0] != '\0')
    {
        LOG_WARN(
            "NvOF: driver error: %s",
            message
        );
    }
}
// =============================================================================
// CUDA error logging
// =============================================================================
static bool nvof_cuda_ok(
    cudaError_t status,
    const char* operation)
{
    if (status == cudaSuccess)
        return true;
    LOG_WARN(
        "NvOF: CUDA failure in %s: %s",
        operation,
        cudaGetErrorString(status)
    );
    return false;
}
static bool nvof_cu_ok(
    CUresult status,
    const char* operation)
{
    if (status == CUDA_SUCCESS)
        return true;
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &text);
    LOG_WARN(
        "NvOF: CUDA driver failure in %s: %s (%s)",
        operation,
        name ? name : "unknown",
        text ? text : "unknown"
    );
    return false;
}
// =============================================================================
// Capability query helpers
// =============================================================================
static bool nvof_get_caps(
    NvOFState* st,
    NV_OF_CAPS capability,
    std::vector<uint32_t>& values)
{
    values.clear();
    if (!st ||
        !st->hOF ||
        !st->fn.nvOFGetCaps)
    {
        return false;
    }
    uint32_t count = 0;
    NV_OF_STATUS status =
        st->fn.nvOFGetCaps(
            st->hOF,
            capability,
            nullptr,
            &count
        );
    if (status != NV_OF_SUCCESS)
    {
        LOG_WARN(
            "NvOF: nvOFGetCaps(size) cap=%u failed: %d (%s)",
            static_cast<unsigned>(capability),
            static_cast<int>(status),
            nvof_status_string(status)
        );
        return false;
    }
    if (count == 0)
        return true;
    values.resize(count);
    status =
        st->fn.nvOFGetCaps(
            st->hOF,
            capability,
            values.data(),
            &count
        );
    if (status != NV_OF_SUCCESS)
    {
        LOG_WARN(
            "NvOF: nvOFGetCaps(values) cap=%u failed: %d (%s)",
            static_cast<unsigned>(capability),
            static_cast<int>(status),
            nvof_status_string(status)
        );
        values.clear();
        return false;
    }
    values.resize(count);
    return true;
}
static bool nvof_get_scalar_cap(
    NvOFState* st,
    NV_OF_CAPS capability,
    uint32_t& value)
{
    std::vector<uint32_t> values;
    if (!nvof_get_caps(
            st,
            capability,
            values))
    {
        return false;
    }
    if (values.empty())
        return false;
    value = values[0];
    return true;
}
// =============================================================================
// Query and select output grid
// =============================================================================
static bool nvof_query_capabilities(
    NvOFState* st)
{
    std::vector<uint32_t> grids;
    if (!nvof_get_caps(
            st,
            NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES,
            grids))
    {
        LOG_WARN(
            "NvOF: failed to query supported output grid sizes"
        );
        return false;
    }
    if (grids.empty())
    {
        LOG_WARN(
            "NvOF: driver returned no supported output grids"
        );
        return false;
    }
    bool supports1 = false;
    bool supports2 = false;
    bool supports4 = false;
    for (uint32_t grid : grids)
    {
        LOG_INFO(
            "NvOF: supported output grid = %u",
            grid
        );
        if (grid == 1)
            supports1 = true;
        if (grid == 2)
            supports2 = true;
        if (grid == 4)
            supports4 = true;
    }
    // Prefer the highest-resolution flow representation.
    if (supports1)
        st->gridSize = 1;
    else if (supports2)
        st->gridSize = 2;
    else if (supports4)
        st->gridSize = 4;
    else
    {
        LOG_WARN(
            "NvOF: no usable output grid size (1/2/4)"
        );
        return false;
    }
    uint32_t widthMin = 0;
    uint32_t heightMin = 0;
    uint32_t widthMax = 0;
    uint32_t heightMax = 0;
    if (nvof_get_scalar_cap(
            st,
            NV_OF_CAPS_WIDTH_MIN,
            widthMin))
    {
        LOG_INFO(
            "NvOF: width range minimum = %u",
            widthMin
        );
        if (st->width < static_cast<int>(widthMin))
        {
            LOG_WARN(
                "NvOF: width %d is below driver minimum %u",
                st->width,
                widthMin
            );
            return false;
        }
    }
    if (nvof_get_scalar_cap(
            st,
            NV_OF_CAPS_HEIGHT_MIN,
            heightMin))
    {
        LOG_INFO(
            "NvOF: height range minimum = %u",
            heightMin
        );
        if (st->height < static_cast<int>(heightMin))
        {
            LOG_WARN(
                "NvOF: height %d is below driver minimum %u",
                st->height,
                heightMin
            );
            return false;
        }
    }
    if (nvof_get_scalar_cap(
            st,
            NV_OF_CAPS_WIDTH_MAX,
            widthMax))
    {
        LOG_INFO(
            "NvOF: width range maximum = %u",
            widthMax
        );
        if (st->width > static_cast<int>(widthMax))
        {
            LOG_WARN(
                "NvOF: width %d exceeds driver maximum %u",
                st->width,
                widthMax
            );
            return false;
        }
    }
    if (nvof_get_scalar_cap(
            st,
            NV_OF_CAPS_HEIGHT_MAX,
            heightMax))
    {
        LOG_INFO(
            "NvOF: height range maximum = %u",
            heightMax
        );
        if (st->height > static_cast<int>(heightMax))
        {
            LOG_WARN(
                "NvOF: height %d exceeds driver maximum %u",
                st->height,
                heightMax
            );
            return false;
        }
    }
    std::vector<uint32_t> hintGrids;
    if (nvof_get_caps(
            st,
            NV_OF_CAPS_SUPPORTED_HINT_GRID_SIZES,
            hintGrids))
    {
        for (uint32_t grid : hintGrids)
        {
            LOG_INFO(
                "NvOF: supported hint grid = %u",
                grid
            );
        }
    }
    uint32_t hintSupport = 0;
    if (nvof_get_scalar_cap(
            st,
            NV_OF_CAPS_SUPPORT_HINT_WITH_OF_MODE,
            hintSupport))
    {
        LOG_INFO(
            "NvOF: external OF hints supported = %u",
            hintSupport
        );
    }
    uint32_t roiSupport = 0;
    if (nvof_get_scalar_cap(
            st,
            NV_OF_CAPS_SUPPORT_ROI,
            roiSupport))
    {
        LOG_INFO(
            "NvOF: ROI support = %u",
            roiSupport
        );
    }
    st->flowWidth =
        (st->width + st->gridSize - 1) /
        st->gridSize;
    st->flowHeight =
        (st->height + st->gridSize - 1) /
        st->gridSize;
    LOG_INFO(
        "NvOF: selected output grid=%d, flow=%dx%d for input=%dx%d",
        st->gridSize,
        st->flowWidth,
        st->flowHeight,
        st->width,
        st->height
    );
    return true;
}
// =============================================================================
// Load dynamic library
// =============================================================================
static void* nvof_load_library(
    const std::wstring& dllDir)
{
#ifdef _WIN32
    // Prefer the driver-installed System32 copy.
    //
    // This avoids accidentally loading an incompatible SDK DLL from
    // an application directory when the NVIDIA driver already provides
    // the correct runtime.
    void* library =
        reinterpret_cast<void*>(
            LOAD_SYSTEM_LIB(L"nvofapi64.dll")
        );
    if (library)
    {
        LOG_INFO(
            "NvOF: loaded nvofapi64.dll from System32"
        );
        return library;
    }
    if (!dllDir.empty())
    {
        std::wstring path = dllDir;
        if (!path.empty() &&
            path.back() != L'\\' &&
            path.back() != L'/')
        {
            path += L'\\';
        }
        path += L"nvofapi64.dll";
        library =
            reinterpret_cast<void*>(
                LOAD_LIB(path.c_str())
            );
        if (library)
        {
            LOG_INFO(
                "NvOF: loaded nvofapi64.dll from application DLL directory"
            );
            return library;
        }
    }
    LOG_WARN(
        "NvOF: unable to load nvofapi64.dll"
    );
    return nullptr;
#else
    (void)dllDir;
    void* library =
        LOAD_SYSTEM_LIB("libnvidia-opticalflow.so.1");
    if (!library)
        library =
            LOAD_SYSTEM_LIB("libnvidia-opticalflow.so");
    if (library)
    {
        LOG_INFO(
            "NvOF: loaded libnvidia-opticalflow.so"
        );
        return library;
    }
    LOG_WARN(
        "NvOF: unable to load NVIDIA optical flow library"
    );
    return nullptr;
#endif
}
// =============================================================================
// Create NvOF state
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
            "NvOF: invalid dimensions %dx%d",
            w,
            h
        );
        return nullptr;
    }
    NvOFState* st =
        new NvOFState();
    st->width = w;
    st->height = h;
    st->maxInterp = maxInterp;
    // -------------------------------------------------------------------------
    // CUDA context
    // -------------------------------------------------------------------------
    CUcontext currentContext = nullptr;
    if (!nvof_cu_ok(
            cuCtxGetCurrent(&currentContext),
            "cuCtxGetCurrent"))
    {
        delete st;
        return nullptr;
    }
    st->previousContext = currentContext;
    if (currentContext)
    {
        // Prefer the caller's existing context.
        //
        // This is important for TensorRT/CUDA interoperability:
        // creating a second independent CUDA context can cause unnecessary
        // resource duplication and synchronization problems.
        st->ofContext = currentContext;
        LOG_INFO(
            "NvOF: using existing CUDA context"
        );
    }
    else
    {
        int device = 0;
        if (!nvof_cuda_ok(
                cudaGetDevice(&device),
                "cudaGetDevice"))
        {
            delete st;
            return nullptr;
        }
        CUdevice cuDevice = 0;
        if (!nvof_cu_ok(
                cuDeviceGet(&cuDevice, device),
                "cuDeviceGet"))
        {
            delete st;
            return nullptr;
        }
        if (!nvof_cu_ok(
                cuDevicePrimaryCtxRetain(
                    &st->ofContext,
                    cuDevice
                ),
                "cuDevicePrimaryCtxRetain"))
        {
            delete st;
            return nullptr;
        }
        st->ownsPrimaryContext = true;
        if (!nvof_cu_ok(
                cuCtxSetCurrent(st->ofContext),
                "cuCtxSetCurrent"))
        {
            cuDevicePrimaryCtxRelease(cuDevice);
            st->ofContext = nullptr;
            st->ownsPrimaryContext = false;
            delete st;
            return nullptr;
        }
        LOG_INFO(
            "NvOF: retained CUDA primary context"
        );
    }
    // -------------------------------------------------------------------------
    // Dynamic NvOF library
    // -------------------------------------------------------------------------
    st->library =
        nvof_load_library(dllDir);
    if (!st->library)
    {
        nvof_destroy(st);
        return nullptr;
    }
    auto getMaxApiVersion =
        reinterpret_cast<PFN_NV_OF_GET_MAX_SUPPORTED_API_VERSION>(
            GET_PROC(
                st->library,
                "NvOFGetMaxSupportedApiVersion"
            )
        );
    auto createInstance =
        reinterpret_cast<PFN_NV_OF_API_CREATE_INSTANCE_CUDA>(
            GET_PROC(
                st->library,
                "NvOFAPICreateInstanceCuda"
            )
        );
    if (!createInstance)
    {
        LOG_WARN(
            "NvOF: NvOFAPICreateInstanceCuda export not found"
        );
        nvof_destroy(st);
        return nullptr;
    }
    // -------------------------------------------------------------------------
    // Query driver API version
    // -------------------------------------------------------------------------
    uint32_t apiVersion = 0;
    if (getMaxApiVersion)
    {
        const NV_OF_STATUS status =
            getMaxApiVersion(&apiVersion);
        if (status != NV_OF_SUCCESS)
        {
            LOG_WARN(
                "NvOF: NvOFGetMaxSupportedApiVersion failed: %d (%s)",
                static_cast<int>(status),
                nvof_status_string(status)
            );
            nvof_destroy(st);
            return nullptr;
        }
        LOG_INFO(
            "NvOF: driver max API version = 0x%x",
            apiVersion
        );
    }
    else
    {
        // SDK-compatible fallback.
        //
        // The runtime version query is preferred whenever available.
        apiVersion = 0x50u;
        LOG_WARN(
            "NvOF: NvOFGetMaxSupportedApiVersion unavailable; "
            "falling back to API version 0x%x",
            apiVersion
        );
    }
    // -------------------------------------------------------------------------
    // Populate NvOF function table
    // -------------------------------------------------------------------------
    std::memset(
        &st->fn,
        0,
        sizeof(st->fn)
    );
    NV_OF_STATUS status =
        createInstance(
            apiVersion,
            &st->fn
        );
    if (status != NV_OF_SUCCESS)
    {
        LOG_WARN(
            "NvOF: NvOFAPICreateInstanceCuda failed: %d (%s)",
            static_cast<int>(status),
            nvof_status_string(status)
        );
        nvof_destroy(st);
        return nullptr;
    }
    if (!st->fn.nvCreateOpticalFlowCuda ||
        !st->fn.nvOFInit ||
        !st->fn.nvOFCreateGPUBufferCuda ||
        !st->fn.nvOFGPUBufferGetCUdeviceptr ||
        !st->fn.nvOFGPUBufferGetStrideInfo ||
        !st->fn.nvOFSetIOCudaStreams ||
        !st->fn.nvOFExecute ||
        !st->fn.nvOFDestroyGPUBufferCuda ||
        !st->fn.nvOFDestroy ||
        !st->fn.nvOFGetLastError ||
        !st->fn.nvOFGetCaps)
    {
        LOG_WARN(
            "NvOF: driver returned incomplete CUDA API function table"
        );
        nvof_destroy(st);
        return nullptr;
    }
    // -------------------------------------------------------------------------
    // Create optical-flow instance
    // -------------------------------------------------------------------------
    status =
        st->fn.nvCreateOpticalFlowCuda(
            st->ofContext,
            &st->hOF
        );
    if (status != NV_OF_SUCCESS)
    {
        LOG_WARN(
            "NvOF: nvCreateOpticalFlowCuda failed: %d (%s)",
            static_cast<int>(status),
            nvof_status_string(status)
        );
        nvof_destroy(st);
        return nullptr;
    }
    LOG_INFO(
        "NvOF: CUDA optical-flow instance created"
    );
    // -------------------------------------------------------------------------
    // Query capabilities BEFORE NvOFInit.
    //
    // This is the critical fix for the current r=5 / INVALID_PARAM failure.
    // -------------------------------------------------------------------------
    if (!nvof_query_capabilities(st))
    {
        LOG_WARN(
            "NvOF: capability negotiation failed"
        );
        nvof_destroy(st);
        return nullptr;
    }
    // -------------------------------------------------------------------------
    // Initialize NvOF
    // -------------------------------------------------------------------------
    NV_OF_INIT_PARAMS initParams{};
    initParams.width =
        static_cast<uint32_t>(w);
    initParams.height =
        static_cast<uint32_t>(h);
    initParams.outGridSize =
        static_cast<NV_OF_OUTPUT_VECTOR_GRID_SIZE>(
            st->gridSize
        );
    // External hints are disabled.
    //
    // NVIDIA explicitly says hintGridSize is considered when external hints
    // are enabled.  Leaving this UNDEFINED avoids passing an unnecessary
    // parameter combination to the driver.
    initParams.hintGridSize =
        NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    initParams.mode =
        NV_OF_MODE_OPTICALFLOW;
    initParams.perfLevel =
        NV_OF_PERF_LEVEL_SLOW;
    initParams.enableExternalHints =
        NV_OF_FALSE;
    initParams.enableOutputCost =
        NV_OF_FALSE;
    initParams.hPrivData =
        nullptr;
    // RTX 2080 Ti is Turing.  NVIDIA specifies UNDEFINED for Turing.
    initParams.disparityRange =
        NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    initParams.enableRoi =
        NV_OF_FALSE;
    LOG_INFO(
        "NvOF: initializing width=%u height=%u grid=%u "
        "mode=OPTICALFLOW perf=SLOW hints=OFF cost=OFF roi=OFF",
        initParams.width,
        initParams.height,
        static_cast<unsigned>(initParams.outGridSize)
    );
    status =
        st->fn.nvOFInit(
            st->hOF,
            &initParams
        );
    if (status != NV_OF_SUCCESS)
    {
        LOG_WARN(
            "NvOF: nvOFInit failed r=%d (%s)",
            static_cast<int>(status),
            nvof_status_string(status)
        );
        nvof_log_last_error(st);
        nvof_destroy(st);
        return nullptr;
    }
    // -------------------------------------------------------------------------
    // Allocate NvOF input buffers
    // -------------------------------------------------------------------------
    NV_OF_BUFFER_DESCRIPTOR inputDesc{};
    inputDesc.width =
        static_cast<uint32_t>(w);
    inputDesc.height =
        static_cast<uint32_t>(h);
    inputDesc.bufferUsage =
        NV_OF_BUFFER_USAGE_INPUT;
    inputDesc.bufferFormat =
        NV_OF_BUFFER_FORMAT_GRAYSCALE8;
    for (int i = 0; i < 2; ++i)
    {
        status =
            st->fn.nvOFCreateGPUBufferCuda(
                st->hOF,
                &inputDesc,
                NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
                &st->guide[i]
            );
        if (status != NV_OF_SUCCESS)
        {
            LOG_WARN(
                "NvOF: failed to create input buffer %d: %d (%s)",
                i,
                static_cast<int>(status),
                nvof_status_string(status)
            );
            nvof_log_last_error(st);
            nvof_destroy(st);
            return nullptr;
        }
    }
    // -------------------------------------------------------------------------
    // Allocate flow output buffer.
    //
    // IMPORTANT:
    // Output dimensions are expressed in outGridSize units.
    // -------------------------------------------------------------------------
    NV_OF_BUFFER_DESCRIPTOR flowDesc{};
    flowDesc.width =
        static_cast<uint32_t>(st->flowWidth);
    flowDesc.height =
        static_cast<uint32_t>(st->flowHeight);
    flowDesc.bufferUsage =
        NV_OF_BUFFER_USAGE_OUTPUT;
    flowDesc.bufferFormat =
        NV_OF_BUFFER_FORMAT_SHORT2;
    status =
        st->fn.nvOFCreateGPUBufferCuda(
            st->hOF,
            &flowDesc,
            NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
            &st->flow
        );
    if (status != NV_OF_SUCCESS)
    {
        LOG_WARN(
            "NvOF: failed to create flow output buffer: %d (%s)",
            static_cast<int>(status),
            nvof_status_string(status)
        );
        nvof_log_last_error(st);
        nvof_destroy(st);
        return nullptr;
    }
    // -------------------------------------------------------------------------
    // Query output stride.
    // -------------------------------------------------------------------------
    NV_OF_CUDA_BUFFER_STRIDE_INFO strideInfo{};
    status =
        st->fn.nvOFGPUBufferGetStrideInfo(
            st->flow,
            &strideInfo
        );
    if (status != NV_OF_SUCCESS)
    {
        LOG_WARN(
            "NvOF: nvOFGPUBufferGetStrideInfo failed: %d (%s)",
            static_cast<int>(status),
            nvof_status_string(status)
        );
        nvof_log_last_error(st);
        nvof_destroy(st);
        return nullptr;
    }
    if (strideInfo.numPlanes < 1)
    {
        LOG_WARN(
            "NvOF: flow output reported zero planes"
        );
        nvof_destroy(st);
        return nullptr;
    }
    st->flowStrideX =
        strideInfo.strideInfo[0].strideXInBytes;
    st->flowStrideY =
        strideInfo.strideInfo[0].strideYInBytes;
    LOG_INFO(
        "NvOF: flow stride = x=%u y=%u bytes",
        st->flowStrideX,
        st->flowStrideY
    );
    // -------------------------------------------------------------------------
    // Allocate CUDA-side normalized depth and full-resolution flow.
    // -------------------------------------------------------------------------
    const size_t pixelCount =
        static_cast<size_t>(w) *
        static_cast<size_t>(h);
    if (!nvof_cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&st->d_depth[0]),
                pixelCount * sizeof(float)
            ),
            "cudaMalloc(d_depth[0])"))
    {
        nvof_destroy(st);
        return nullptr;
    }
    if (!nvof_cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&st->d_depth[1]),
                pixelCount * sizeof(float)
            ),
            "cudaMalloc(d_depth[1])"))
    {
        nvof_destroy(st);
        return nullptr;
    }
    if (!nvof_cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&st->d_minmax),
                2 * sizeof(float)
            ),
            "cudaMalloc(d_minmax)"))
    {
        nvof_destroy(st);
        return nullptr;
    }
    if (!nvof_cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&st->d_flowFull),
                pixelCount * sizeof(float2)
            ),
            "cudaMalloc(d_flowFull)"))
    {
        nvof_destroy(st);
        return nullptr;
    }
    st->initialized = true;
    LOG_INFO(
        "NvOF: initialization complete"
    );
    return st;
}
// =============================================================================
// Destroy
// =============================================================================
void nvof_destroy(
    NvOFState* st)
{
    if (!st)
        return;
    // Make sure all CUDA work that might reference NvOF buffers is finished.
    cudaDeviceSynchronize();
    if (st->hOF)
    {
        if (st->guide[0])
        {
            st->fn.nvOFDestroyGPUBufferCuda(
                st->guide[0]
            );
            st->guide[0] = nullptr;
        }
        if (st->guide[1])
        {
            st->fn.nvOFDestroyGPUBufferCuda(
                st->guide[1]
            );
            st->guide[1] = nullptr;
        }
        if (st->flow)
        {
            st->fn.nvOFDestroyGPUBufferCuda(
                st->flow
            );
            st->flow = nullptr;
        }
        st->fn.nvOFDestroy(
            st->hOF
        );
        st->hOF = nullptr;
    }
    if (st->d_depth[0])
    {
        cudaFree(st->d_depth[0]);
        st->d_depth[0] = nullptr;
    }
    if (st->d_depth[1])
    {
        cudaFree(st->d_depth[1]);
        st->d_depth[1] = nullptr;
    }
    if (st->d_minmax)
    {
        cudaFree(st->d_minmax);
        st->d_minmax = nullptr;
    }
    if (st->d_flowFull)
    {
        cudaFree(st->d_flowFull);
        st->d_flowFull = nullptr;
    }
    // Restore the context that was current before NvOF creation.
    if (st->previousContext)
    {
        cuCtxSetCurrent(st->previousContext);
    }
    else if (st->ownsPrimaryContext &&
             st->ofContext)
    {
        cuCtxSetCurrent(nullptr);
    }
    if (st->ownsPrimaryContext &&
        st->ofContext)
    {
        CUdevice device = 0;
        if (cuCtxGetDevice(&device) == CUDA_SUCCESS)
        {
            cuDevicePrimaryCtxRelease(device);
        }
    }
    st->ofContext = nullptr;
    if (st->library)
    {
        FREE_LIB(st->library);
        st->library = nullptr;
    }
    delete st;
}
// =============================================================================
// Availability
// =============================================================================
bool nvof_available(
    NvOFState* st)
{
    return
        st != nullptr &&
        st->initialized &&
        st->hOF != nullptr;
}
// =============================================================================
// Dimensions
// =============================================================================
void nvof_dims(
    NvOFState* st,
    int* w,
    int* h)
{
    if (!st)
    {
        if (w)
            *w = 0;
        if (h)
            *h = 0;
        return;
    }
    if (w)
        *w = st->width;
    if (h)
        *h = st->height;
}
// =============================================================================
// Prepare pipeline slot
// =============================================================================
void nvof_prepare_slot(
    NvOFState* st,
    int slot,
    const uint8_t* d_guideBGRA,
    int srcW,
    int srcH,
    int srcStride,
    const float* d_outSlice,
    float* d_minmax_scratch,
    int mw,
    int mh,
    void* stream)
{
    if (!st ||
        !st->initialized)
    {
        return;
    }
    if (slot < 0 || slot > 1)
    {
        LOG_WARN(
            "NvOF: invalid prepare slot %d",
            slot
        );
        return;
    }
    if (!d_guideBGRA ||
        !d_outSlice)
    {
        LOG_WARN(
            "NvOF: null prepare input"
        );
        return;
    }
    if (mw != st->width ||
        mh != st->height)
    {
        LOG_WARN(
            "NvOF: prepare dimensions %dx%d do not match "
            "NvOF state %dx%d",
            mw,
            mh,
            st->width,
            st->height
        );
        return;
    }
    cudaStream_t cudaStream =
        reinterpret_cast<cudaStream_t>(stream);
    // -------------------------------------------------------------------------
    // BGRA -> grayscale
    // -------------------------------------------------------------------------
    dim3 block(16, 16);
    dim3 grid(
        (mw + block.x - 1) / block.x,
        (mh + block.y - 1) / block.y
    );
    // The NvOF guide buffer is opaque to the CUDA runtime, so obtain its
    // CUdeviceptr and use that as the destination.
    CUdeviceptr guidePtr = 0;
    if (!st->fn.nvOFGPUBufferGetCUdeviceptr)
        return;
    guidePtr =
        st->fn.nvOFGPUBufferGetCUdeviceptr(
            st->guide[slot]
        );
    if (!guidePtr)
    {
        LOG_WARN(
            "NvOF: failed to obtain guide buffer device pointer"
        );
        return;
    }
    // Re-launch with the actual NvOF destination.
    k_bgra_to_gray8<<<
        grid,
        block,
        0,
        cudaStream
    >>>(
        d_guideBGRA,
        srcW,
        srcH,
        srcStride,
        reinterpret_cast<uint8_t*>(
            static_cast<uintptr_t>(guidePtr)
        ),
        mw,
        mh
    );
    // -------------------------------------------------------------------------
    // Depth normalization
    // -------------------------------------------------------------------------
    float* minmax =
        d_minmax_scratch
            ? d_minmax_scratch
            : st->d_minmax;
    if (!minmax)
        return;
    const size_t count =
        static_cast<size_t>(mw) *
        static_cast<size_t>(mh);
    const float initialMin =
        std::numeric_limits<float>::infinity();
    const float initialMax =
        -std::numeric_limits<float>::infinity();
    cudaMemcpyAsync(
        minmax,
        &initialMin,
        sizeof(float),
        cudaMemcpyHostToDevice,
        cudaStream
    );
    cudaMemcpyAsync(
        minmax + 1,
        &initialMax,
        sizeof(float),
        cudaMemcpyHostToDevice,
        cudaStream
    );
    const int threads = 256;
    const int blocks =
        static_cast<int>(
            (count + threads - 1) /
            threads
        );
    k_scan_minmax<<<
        blocks,
        threads,
        0,
        cudaStream
    >>>(
        d_outSlice,
        minmax,
        static_cast<int>(count)
    );
    k_normalize_depth<<<
        blocks,
        threads,
        0,
        cudaStream
    >>>(
        d_outSlice,
        st->d_depth[slot],
        minmax,
        static_cast<int>(count)
    );
    // Make CUDA launch failures visible without synchronizing the pipeline.
    const cudaError_t launchError =
        cudaPeekAtLastError();
    if (launchError != cudaSuccess)
    {
        LOG_WARN(
            "NvOF: prepare kernel launch failed: %s",
            cudaGetErrorString(launchError)
        );
    }
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
        !st->initialized ||
        !st->hOF)
    {
        return false;
    }
    if (prevSlot < 0 ||
        prevSlot > 1 ||
        currSlot < 0 ||
        currSlot > 1 ||
        prevSlot == currSlot)
    {
        LOG_WARN(
            "NvOF: invalid execute slots prev=%d curr=%d",
            prevSlot,
            currSlot
        );
        return false;
    }
    CUstream cuStream =
        reinterpret_cast<CUstream>(stream);
    // The stream is used for NvOF's internal CUDA preprocessing and
    // postprocessing.
    NV_OF_STATUS status =
        st->fn.nvOFSetIOCudaStreams(
            st->hOF,
            cuStream,
            cuStream
        );
    if (status != NV_OF_SUCCESS)
    {
        LOG_WARN(
            "NvOF: nvOFSetIOCudaStreams failed: %d (%s)",
            static_cast<int>(status),
            nvof_status_string(status)
        );
        nvof_log_last_error(st);
        return false;
    }
    NV_OF_EXECUTE_INPUT_PARAMS input{};
    // Forward flow is from input -> reference.
    //
    // We want current -> previous, so current is input and previous
    // is reference.
    input.inputFrame =
        st->guide[currSlot];
    input.referenceFrame =
        st->guide[prevSlot];
    input.externalHints =
        nullptr;
    // Keep temporal hints enabled for consecutive video frames.
    input.disableTemporalHints =
        NV_OF_FALSE;
    input.padding =
        0;
    input.hPrivData =
        nullptr;
    input.padding2 =
        0;
    input.numRois =
        0;
    input.roiData =
        nullptr;
    NV_OF_EXECUTE_OUTPUT_PARAMS output{};
    output.outputBuffer =
        st->flow;
    output.outputCostBuffer =
        nullptr;
    output.hPrivData =
        nullptr;
    status =
        st->fn.nvOFExecute(
            st->hOF,
            &input,
            &output
        );
    if (status != NV_OF_SUCCESS)
    {
        LOG_WARN(
            "NvOF: nvOFExecute failed: %d (%s)",
            static_cast<int>(status),
            nvof_status_string(status)
        );
        nvof_log_last_error(st);
        return false;
    }
    st->lastStream =
        cuStream;
    // -------------------------------------------------------------------------
    // Obtain flow output pointer.
    // -------------------------------------------------------------------------
    const CUdeviceptr flowPtr =
        st->fn.nvOFGPUBufferGetCUdeviceptr(
            st->flow
        );
    if (!flowPtr)
    {
        LOG_WARN(
            "NvOF: nvOFGPUBufferGetCUdeviceptr(flow) returned null"
        );
        return false;
    }
    // -------------------------------------------------------------------------
    // Expand grid flow to full LR resolution.
    // -------------------------------------------------------------------------
    dim3 block(16, 16);
    dim3 grid(
        (st->width + block.x - 1) / block.x,
        (st->height + block.y - 1) / block.y
    );
    k_expand_flow<<<
        grid,
        block,
        0,
        reinterpret_cast<cudaStream_t>(stream)
    >>>(
        reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(flowPtr)
        ),
        static_cast<size_t>(st->flowStrideY),
        st->flowWidth,
        st->flowHeight,
        st->gridSize,
        st->d_flowFull,
        st->width,
        st->height
    );
    const cudaError_t launchError =
        cudaPeekAtLastError();
    if (launchError != cudaSuccess)
    {
        LOG_WARN(
            "NvOF: flow expansion launch failed: %s",
            cudaGetErrorString(launchError)
        );
        return false;
    }
    return true;
}
// =============================================================================
// Warp and blend
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
        !st->initialized ||
        !d_out)
    {
        return;
    }
    if (prevSlot < 0 ||
        prevSlot > 1 ||
        currSlot < 0 ||
        currSlot > 1 ||
        prevSlot == currSlot)
    {
        return;
    }
    t =
        fminf(
            1.0f,
            fmaxf(0.0f, t)
        );
    dim3 block(16, 16);
    dim3 grid(
        (st->width + block.x - 1) / block.x,
        (st->height + block.y - 1) / block.y
    );
    k_warp_blend<<<
        grid,
        block,
        0,
        reinterpret_cast<cudaStream_t>(stream)
    >>>(
        st->d_depth[prevSlot],
        st->d_depth[currSlot],
        st->d_flowFull,
        d_out,
        st->width,
        st->height,
        t
    );
    const cudaError_t launchError =
        cudaPeekAtLastError();
    if (launchError != cudaSuccess)
    {
        LOG_WARN(
            "NvOF: warp/blend launch failed: %s",
            cudaGetErrorString(launchError)
        );
    }
}