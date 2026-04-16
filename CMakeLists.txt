cmake_minimum_required(VERSION 3.20)
project(3Deflatten VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED TRUE)

# ── Platform ──────────────────────────────────────────────────────────────────
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH_SUFFIX "x64")
    set(ORT_ARCH   "x64")
else()
    set(ARCH_SUFFIX "x86")
    set(ORT_ARCH   "x86")
endif()

set(WINSDK_LIB_DIR "$ENV{WindowsSdkDir}Lib/$ENV{WindowsSDKVersion}/um/${ARCH_SUFFIX}"
    CACHE PATH "Windows SDK UM lib directory")
set(WINSDK_INC_DIR "$ENV{WindowsSdkDir}Include/$ENV{WindowsSDKVersion}"
    CACHE PATH "Windows SDK include root")

# ── DirectShow base classes ───────────────────────────────────────────────────
FetchContent_Declare(win_classic_samples
    GIT_REPOSITORY https://github.com/microsoft/Windows-classic-samples.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
    GIT_SPARSE_CHECKOUT_PATTERNS
        "Samples/Win7Samples/multimedia/directshow/baseclasses"
    SOURCE_SUBDIR  "Samples/Win7Samples/multimedia/directshow/baseclasses"
)
FetchContent_MakeAvailable(win_classic_samples)

set(BASECLASSES_DIR
    "${win_classic_samples_SOURCE_DIR}/Samples/Win7Samples/multimedia/directshow/baseclasses")

file(GLOB STRMBASE_SOURCES "${BASECLASSES_DIR}/*.cpp")
list(FILTER STRMBASE_SOURCES EXCLUDE REGEX "dllentry\\.cpp$")

add_library(strmbase STATIC ${STRMBASE_SOURCES})
target_include_directories(strmbase PUBLIC
    "${BASECLASSES_DIR}"
    "${WINSDK_INC_DIR}/um"
    "${WINSDK_INC_DIR}/shared"
)
target_compile_definitions(strmbase PUBLIC
    WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS
    UNICODE _UNICODE _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
)
target_compile_options(strmbase PRIVATE
    /W3 /wd4996 /wd4267 /wd4244 /wd4312 /wd4311 /wd4302
    /FI"algorithm"
)

# ── ONNX Runtime backend selection ───────────────────────────────────────────
#
# Two mutually-exclusive prebuilt distributions are available:
#
#   DirectML (default)  –  Microsoft.ML.OnnxRuntime.DirectML NuGet
#       EPs available at runtime: DirectML, CPU
#       Defines: ORT_ENABLE_DML
#       Good for: AMD, Intel, NVIDIA without needing CUDA installed
#
#   GPU                 –  ORT 1.18.1 GitHub release zip (CUDA 11.8 / TRT 10.0.0.6)
#       EPs available at runtime: TensorRT (requires TRT 10.0.0.6), CUDA, CPU
#       Defines: ORT_ENABLE_TENSORRT, ORT_ENABLE_CUDA, ORT_CUDA_MAJOR=11
#       Use collect_runtime_dlls.py to bundle CUDA 11.8 + cuDNN 8 + TRT 10.0.0.6 DLLs.
#       NOTE: ORT 1.18.1 requires CUDA 11.8 exactly (11.7 causes error=1114).
#       NOTE: TRT 10.0.x uses plain DLL names (nvinfer.dll, not nvinfer_10.dll).
#       Note: x64 only.
#
#   GPU13               –  ORT 1.24.3 github release zip (CUDA 13.0 / TRT 10.13.3)
#       EPs available at runtime: TensorRT, CUDA, CPU
#       Defines: ORT_ENABLE_TENSORRT, ORT_ENABLE_CUDA, ORT_CUDA_MAJOR=13
#       Use collect_runtime_dlls_cuda13.py to bundle CUDA 13.0 + cuDNN 9 + TRT DLLs.
#       Note: x64 only.
#
#   Unified             –  ORT built from source (CUDA 12.9.1 / TRT 10.16.0.72 / DML / TRT-RTX)
#       EPs available at runtime: TRT-RTX, TensorRT, CUDA, DirectML, CPU
#       Defines: ORT_ENABLE_DML, ORT_ENABLE_CUDA, ORT_ENABLE_TENSORRT, ORT_ENABLE_TRTRTX,
#                ORT_CUDA_MAJOR=12
#       Requires: ORT_UNIFIED_DIR pointing to your custom ORT build output.
#       Build ORT with:
#           build.bat --config Release --use_dml --use_cuda --cudnn_home <path>
#                     --cuda_home <path> --use_tensorrt --tensorrt_home <trt_path>
#                     --use_nv_tensorrt_rtx --tensorrt_rtx_home <trtrtx_path>
#                     --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF
#       Output dir: Win64_Unified
#       Use collect_runtime_dlls_cuda12.py to bundle runtime DLLs.
#       Note: x64 only.
#
# Select with:  cmake -B build -DORT_BACKEND=GPU   (or GPU13, DirectML, Unified)
#
set(ORT_BACKEND "DirectML" CACHE STRING "ORT inference backend: DirectML, GPU, GPU13, or Unified")
set_property(CACHE ORT_BACKEND PROPERTY STRINGS "DirectML" "GPU" "GPU13" "Unified")
message(STATUS "ORT backend: ${ORT_BACKEND}")

# ORT version per backend
if(ORT_BACKEND STREQUAL "GPU13")
    set(ORT_VERSION "1.24.3")
elseif(ORT_BACKEND STREQUAL "GPU")
    set(ORT_VERSION "1.18.1")
elseif(ORT_BACKEND STREQUAL "Unified")
    set(ORT_VERSION "custom")  # user-built from source
else()
    set(ORT_VERSION "1.24.3")   # DirectML / CPU NuGet
endif()

# ── Helper macro: resolve ORT paths from an extracted NuGet/zip directory ────
# NuGet packages (.nupkg are zips) are extracted by FetchContent directly into
# SOURCE_DIR with no wrapper folder.  Confirmed layout for ORT 1.21-1.24.x:
#   SOURCE_DIR/build/native/include/          headers
#   SOURCE_DIR/runtimes/win-x64/native/       onnxruntime.lib + onnxruntime.dll
# Older packages used build/native/lib/<arch>/ for the import lib; we probe
# both so the macro works across package versions.
macro(ort_resolve_nuget_paths SOURCE_VAR ARCH_RID)
    set(_ORT_SRC "${${SOURCE_VAR}}")
    set(ORT_INC "${_ORT_SRC}/build/native/include")

    # ── Import lib ────────────────────────────────────────────────────────────
    # Microsoft.ML.OnnxRuntime.DirectML (and the plain CPU NuGet) ship the
    # import lib at:  runtimes/win-<arch>/native/onnxruntime.lib
    # NOT under build/native/lib/ (that path does not exist in these packages).
    # Probe in priority order so we work across past and future package layouts:
    #   1. runtimes/win-<arch>/native/  (DirectML / CPU NuGet -- confirmed 1.21)
    #   2. build/native/lib/<arch>/     (older package layout, kept as fallback)
    #   3. GLOB_RECURSE filtered by arch (last resort)
    set(_ORT_LIB_RUNTIME "${_ORT_SRC}/runtimes/win-${ARCH_RID}/native/onnxruntime.lib")
    set(_ORT_LIB_BUILD   "${_ORT_SRC}/build/native/lib/${ARCH_RID}/onnxruntime.lib")
    if(EXISTS "${_ORT_LIB_RUNTIME}")
        set(ORT_LIB "${_ORT_LIB_RUNTIME}")
        message(STATUS "ORT import lib (runtimes/): ${ORT_LIB}")
    elseif(EXISTS "${_ORT_LIB_BUILD}")
        set(ORT_LIB "${_ORT_LIB_BUILD}")
        message(STATUS "ORT import lib (build/native/lib/): ${ORT_LIB}")
    else()
        file(GLOB_RECURSE _ORT_LIB_CANDIDATES "${_ORT_SRC}/*.lib")
        list(FILTER _ORT_LIB_CANDIDATES INCLUDE REGEX "onnxruntime\.lib$")
        list(FILTER _ORT_LIB_CANDIDATES INCLUDE REGEX "${ARCH_RID}")
        if(_ORT_LIB_CANDIDATES)
            list(GET _ORT_LIB_CANDIDATES 0 ORT_LIB)
            message(STATUS "ORT import lib (glob/${ARCH_RID}): ${ORT_LIB}")
        else()
            set(ORT_LIB "${_ORT_LIB_RUNTIME}")  # will be caught by existence check
        endif()
    endif()

    # ── DLL ───────────────────────────────────────────────────────────────────
    set(ORT_DLL "${_ORT_SRC}/runtimes/win-${ARCH_RID}/native/onnxruntime.dll")
    if(NOT EXISTS "${ORT_DLL}")
        file(GLOB_RECURSE _ORT_DLL_CANDIDATES "${_ORT_SRC}/runtimes/*.dll")
        list(FILTER _ORT_DLL_CANDIDATES INCLUDE REGEX "onnxruntime\.dll$")
        list(FILTER _ORT_DLL_CANDIDATES EXCLUDE REGEX "providers")
        list(FILTER _ORT_DLL_CANDIDATES INCLUDE REGEX "win-${ARCH_RID}")
        if(_ORT_DLL_CANDIDATES)
            list(GET _ORT_DLL_CANDIDATES 0 ORT_DLL)
        endif()
    endif()
endmacro()

if(ORT_BACKEND STREQUAL "GPU" OR ORT_BACKEND STREQUAL "GPU13")
    if(NOT ARCH_SUFFIX STREQUAL "x64")
        message(FATAL_ERROR "ORT_BACKEND=${ORT_BACKEND} is only supported for x64 builds. "
                            "Use ORT_BACKEND=DirectML for x86.")
    endif()

    if(ORT_BACKEND STREQUAL "GPU13")
        # ── GPU13 backend: ORT 1.24.3 / CUDA 13.0 / TRT 10.13.3 ─────────────
        # Runtime DLLs required (from collect_runtime_dlls_cuda13.py):
        #   cudart64_13, cublas64_13, cublasLt64_13, cufft64_12, nvJitLink_130_0,
        #   cusolver64_11, curand64_10, cudnn64_9 (+ 7 split libs), TRT 10.13.3 DLLs.
        set(ORT_GPU_ZIP_VARIANT "gpu_cuda13")
        set(ORT_CUDA_MAJOR_VAL  13)
        set(ORT_CI_CACHE_KEY    "ort-gpu-cuda13-${ORT_VERSION}")
    else()
        # ── GPU backend: ORT 1.18.1 / CUDA 11.8 / TRT 10.0.0.6 ─────────────
        # Runtime DLLs required (from collect_runtime_dlls.py):
        #   cudart64_110, cublas64_11, cublasLt64_11, cufft64_10,
        #   cusolver64_11, curand64_10, cudnn64_8 (+ split libs),
        #   TRT 10.0.0.6 DLLs: nvinfer.dll (no _10 suffix -- that's TRT 10.3+).
        set(ORT_GPU_ZIP_VARIANT "gpu")
        set(ORT_CUDA_MAJOR_VAL  11)
        set(ORT_CI_CACHE_KEY    "ort-gpu-cuda11-${ORT_VERSION}")
    endif()

    FetchContent_Declare(ort_gpu
        URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-${ORT_GPU_ZIP_VARIANT}-${ORT_VERSION}.zip"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(ort_gpu)

    # The GitHub release zip may extract to a nested subdirectory OR directly
    # into SOURCE_DIR depending on the archive structure and CMake version.
    set(_ORT_NESTED "${ort_gpu_SOURCE_DIR}/onnxruntime-win-x64-${ORT_GPU_ZIP_VARIANT}-${ORT_VERSION}")
    if(EXISTS "${_ORT_NESTED}/lib/onnxruntime.lib")
        set(ORT_ZIP_ROOT "${_ORT_NESTED}")
        message(STATUS "ORT GPU zip: nested layout detected (${ORT_ZIP_ROOT})")
    elseif(EXISTS "${ort_gpu_SOURCE_DIR}/lib/onnxruntime.lib")
        set(ORT_ZIP_ROOT "${ort_gpu_SOURCE_DIR}")
        message(STATUS "ORT GPU zip: flat layout detected (${ORT_ZIP_ROOT})")
    else()
        file(GLOB_RECURSE _ORT_LIB_CANDIDATES
             "${ort_gpu_SOURCE_DIR}/*/onnxruntime.lib")
        list(GET _ORT_LIB_CANDIDATES 0 _ORT_LIB_FOUND)
        if(_ORT_LIB_FOUND)
            get_filename_component(ORT_ZIP_ROOT "${_ORT_LIB_FOUND}" DIRECTORY)
            get_filename_component(ORT_ZIP_ROOT "${ORT_ZIP_ROOT}" DIRECTORY)
            message(STATUS "ORT GPU zip: found via glob at ${ORT_ZIP_ROOT}")
        else()
            set(ORT_ZIP_ROOT "${_ORT_NESTED}")
        endif()
    endif()

    set(ORT_INC  "${ORT_ZIP_ROOT}/include")
    set(ORT_LIB  "${ORT_ZIP_ROOT}/lib/onnxruntime.lib")
    set(ORT_DLL  "${ORT_ZIP_ROOT}/lib/onnxruntime.dll")
    set(ORT_EXTRA_DLLS
        "${ORT_ZIP_ROOT}/lib/onnxruntime_providers_shared.dll"
        "${ORT_ZIP_ROOT}/lib/onnxruntime_providers_cuda.dll"
        "${ORT_ZIP_ROOT}/lib/onnxruntime_providers_tensorrt.dll"
    )
    set(ORT_COMPILE_DEFS ORT_ENABLE_TENSORRT ORT_ENABLE_CUDA "ORT_CUDA_MAJOR=${ORT_CUDA_MAJOR_VAL}")

elseif(ORT_BACKEND STREQUAL "Unified")
    # ── Unified backend: user-supplied ORT built from source ──────────────────
    # Supports TRT-RTX + TensorRT + CUDA + DirectML + CPU in a single .ax.
    # Set ORT_UNIFIED_DIR to the root of your ORT build output, e.g.:
    #   cmake -B build -DORT_BACKEND=Unified -DORT_UNIFIED_DIR=C:\ort_build\Release
    # The directory must contain:
    #   include/             ORT headers
    #   lib/onnxruntime.lib  import lib
    #   lib/onnxruntime.dll  (or bin/onnxruntime.dll for some build configs)
    if(NOT ARCH_SUFFIX STREQUAL "x64")
        message(FATAL_ERROR "ORT_BACKEND=Unified is x64 only.")
    endif()
    set(ORT_UNIFIED_DIR "" CACHE PATH
        "Root of ORT source build output (for ORT_BACKEND=Unified)")
    if(NOT ORT_UNIFIED_DIR OR NOT EXISTS "${ORT_UNIFIED_DIR}")
        message(FATAL_ERROR
            "ORT_BACKEND=Unified requires ORT_UNIFIED_DIR to point to your ORT build.\n"
            "  cmake -B build -DORT_BACKEND=Unified -DORT_UNIFIED_DIR=<ort_build_root>\n"
            "Build ORT with:\n"
            "  build.bat --config Release --use_dml --use_cuda --cudnn_home <path>\n"
            "            --cuda_home <path> --use_tensorrt --tensorrt_home <trt_path>\n"
            "            --use_nv_tensorrt_rtx --tensorrt_rtx_home <trtrtx_path>\n"
            "            --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF")
    endif()

    set(ORT_INC "${ORT_UNIFIED_DIR}/include")
    # Probe lib/ then bin/ for .lib (source build can put it in either)
    if(EXISTS "${ORT_UNIFIED_DIR}/lib/onnxruntime.lib")
        set(ORT_LIB "${ORT_UNIFIED_DIR}/lib/onnxruntime.lib")
    elseif(EXISTS "${ORT_UNIFIED_DIR}/Release/onnxruntime.lib")
        set(ORT_LIB "${ORT_UNIFIED_DIR}/Release/onnxruntime.lib")
    else()
        file(GLOB_RECURSE _U_LIB "${ORT_UNIFIED_DIR}/*onnxruntime.lib")
        list(FILTER _U_LIB EXCLUDE REGEX "_providers_")
        if(_U_LIB)
            list(GET _U_LIB 0 ORT_LIB)
        else()
            set(ORT_LIB "${ORT_UNIFIED_DIR}/lib/onnxruntime.lib")  # will be caught below
        endif()
    endif()
    # Same probe for DLL
    if(EXISTS "${ORT_UNIFIED_DIR}/lib/onnxruntime.dll")
        set(ORT_DLL "${ORT_UNIFIED_DIR}/lib/onnxruntime.dll")
    elseif(EXISTS "${ORT_UNIFIED_DIR}/Release/onnxruntime.dll")
        set(ORT_DLL "${ORT_UNIFIED_DIR}/Release/onnxruntime.dll")
    else()
        file(GLOB_RECURSE _U_DLL "${ORT_UNIFIED_DIR}/*onnxruntime.dll")
        list(FILTER _U_DLL EXCLUDE REGEX "_providers_")
        if(_U_DLL)
            list(GET _U_DLL 0 ORT_DLL)
        else()
            set(ORT_DLL "${ORT_UNIFIED_DIR}/lib/onnxruntime.dll")
        endif()
    endif()

    # Collect provider DLLs (shared, cuda, tensorrt) if present
    set(ORT_EXTRA_DLLS "")
    foreach(_prov shared cuda tensorrt)
        foreach(_search
                "${ORT_UNIFIED_DIR}/lib/onnxruntime_providers_${_prov}.dll"
                "${ORT_UNIFIED_DIR}/Release/onnxruntime_providers_${_prov}.dll")
            if(EXISTS "${_search}")
                list(APPEND ORT_EXTRA_DLLS "${_search}")
                break()
            endif()
        endforeach()
    endforeach()

    # All four EPs + CUDA 12
    set(ORT_COMPILE_DEFS
        ORT_ENABLE_DML
        ORT_ENABLE_CUDA
        ORT_ENABLE_TENSORRT
        ORT_ENABLE_TRTRTX
        "ORT_CUDA_MAJOR=12")

    # ── CUDA Toolkit (provides cuda_runtime.h and cudart import lib) ───────────
    # Required for the TRT-RTX native path.  Install CUDA 12.x and either set
    # CUDAToolkit_ROOT or let CMake find it via the system registry.
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}  inc=${CUDAToolkit_INCLUDE_DIRS}")

    # ── TRT-RTX SDK (NvInfer.h, NvOnnxParser.h, nvinfer_10.lib) ──────────────
    # Set -DORT_TRTRTX_HOME=<path> to the root of the TensorRT-RTX SDK zip, e.g.:
    #   TensorRT-RTX-1.4.0.76-Windows-amd64-cuda-12.9-Release-external/
    set(ORT_TRTRTX_HOME "" CACHE PATH
        "TRT-RTX SDK root (for ORT_ENABLE_TRTRTX; contains include/ and lib/)")
    if(ORT_TRTRTX_HOME AND EXISTS "${ORT_TRTRTX_HOME}")
        message(STATUS "TRT-RTX SDK: ${ORT_TRTRTX_HOME}")
        set(_TRTRTX_INC "${ORT_TRTRTX_HOME}/include")
        # The import lib may be nvinfer_10.lib or nvinfer.lib depending on the SDK version
        foreach(_nvinfer_name "nvinfer_10.lib" "nvinfer.lib")
            if(EXISTS "${ORT_TRTRTX_HOME}/lib/${_nvinfer_name}")
                set(_TRTRTX_INFER_LIB "${ORT_TRTRTX_HOME}/lib/${_nvinfer_name}")
                break()
            endif()
        endforeach()
        foreach(_nvonnx_name "nvonnxparser_10.lib" "nvonnxparser.lib")
            if(EXISTS "${ORT_TRTRTX_HOME}/lib/${_nvonnx_name}")
                set(_TRTRTX_PARSER_LIB "${ORT_TRTRTX_HOME}/lib/${_nvonnx_name}")
                break()
            endif()
        endforeach()
        if(NOT _TRTRTX_INFER_LIB)
            message(WARNING "TRT-RTX: nvinfer*.lib not found in ${ORT_TRTRTX_HOME}/lib/")
        else()
            message(STATUS "TRT-RTX infer lib : ${_TRTRTX_INFER_LIB}")
        endif()
        if(NOT _TRTRTX_PARSER_LIB)
            message(WARNING "TRT-RTX: nvonnxparser*.lib not found in ${ORT_TRTRTX_HOME}/lib/")
        endif()
    else()
        message(WARNING
            "ORT_TRTRTX_HOME not set or not found.\n"
            "  The TRT-RTX native inference path will fail to compile.\n"
            "  Set -DORT_TRTRTX_HOME=<path_to_trt_rtx_sdk>")
    endif()

    message(STATUS "ORT Unified build:")
    message(STATUS "  ORT_INC: ${ORT_INC}")
    message(STATUS "  ORT_LIB: ${ORT_LIB}")
    message(STATUS "  ORT_DLL: ${ORT_DLL}")

elseif(ARCH_SUFFIX STREQUAL "x64")
    # ── DirectML backend, x64 (DirectML + CPU) ────────────────────────────────
    # ORT_NUGET_DIR can be set externally (e.g. by CI using nuget.exe install)
    # for reliable package resolution.  If not set we fall back to FetchContent.
    if(DEFINED ORT_NUGET_DIR AND EXISTS "${ORT_NUGET_DIR}")
        message(STATUS "ORT DirectML NuGet: using externally provided dir ${ORT_NUGET_DIR}")
        set(_ORT_NUGET_SRC "${ORT_NUGET_DIR}")
    else()
        message(STATUS "ORT DirectML NuGet: downloading via FetchContent")
        FetchContent_Declare(ort_nuget
            URL "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/${ORT_VERSION}/microsoft.ml.onnxruntime.directml.${ORT_VERSION}.nupkg"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(ort_nuget)
        set(_ORT_NUGET_SRC "${ort_nuget_SOURCE_DIR}")
    endif()
    set(ort_nuget_SOURCE_DIR "${_ORT_NUGET_SRC}")
    ort_resolve_nuget_paths(ort_nuget_SOURCE_DIR "x64")

    # ── Microsoft.AI.DirectML NuGet (latest standalone DirectML runtime) ──────
    # The ORT NuGet bundles an older directml.dll.  We download the latest
    # Microsoft.AI.DirectML package separately so users get the best perf.
    # If DML_NUGET_DIR is set externally (CI), use that; otherwise FetchContent.
    set(DML_NUGET_VERSION "1.15.4")
    if(DEFINED DML_NUGET_DIR AND EXISTS "${DML_NUGET_DIR}")
        message(STATUS "DirectML NuGet: using externally provided dir ${DML_NUGET_DIR}")
        set(_DML_NUGET_SRC "${DML_NUGET_DIR}")
    else()
        message(STATUS "DirectML NuGet: downloading Microsoft.AI.DirectML ${DML_NUGET_VERSION} via FetchContent")
        FetchContent_Declare(dml_nuget
            URL "https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/${DML_NUGET_VERSION}/microsoft.ai.directml.${DML_NUGET_VERSION}.nupkg"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(dml_nuget)
        set(_DML_NUGET_SRC "${dml_nuget_SOURCE_DIR}")
    endif()
    # DML NuGet layout: bin/x64-win/DirectML.dll  (note: capital D, capital ML)
    set(_DML_DLL_CANDIDATES
        "${_DML_NUGET_SRC}/bin/x64-win/DirectML.dll"
        "${_DML_NUGET_SRC}/runtimes/win-x64/native/DirectML.dll"
        "${_DML_NUGET_SRC}/bin/x64-win/directml.dll"
        "${_DML_NUGET_SRC}/runtimes/win-x64/native/directml.dll"
    )
    set(_DML_DLL "")
    foreach(_c ${_DML_DLL_CANDIDATES})
        if(EXISTS "${_c}")
            set(_DML_DLL "${_c}")
            message(STATUS "DirectML.dll found at ${_DML_DLL}")
            break()
        endif()
    endforeach()
    if(NOT _DML_DLL)
        file(GLOB_RECURSE _DML_CANDIDATES "${_DML_NUGET_SRC}/*.dll")
        list(FILTER _DML_CANDIDATES INCLUDE REGEX "[Dd]irect[Mm][Ll]\\.dll$")
        if(_DML_CANDIDATES)
            list(GET _DML_CANDIDATES 0 _DML_DLL)
            message(STATUS "DirectML.dll (glob fallback): ${_DML_DLL}")
        else()
            message(WARNING "DirectML.dll not found in NuGet. Users must place it manually.")
        endif()
    endif()

    set(ORT_EXTRA_DLLS "")
    foreach(_extra
            "${_ORT_NUGET_SRC}/runtimes/win-x64/native/onnxruntime_providers_shared.dll")
        if(EXISTS "${_extra}")
            list(APPEND ORT_EXTRA_DLLS "${_extra}")
        endif()
    endforeach()
    # DirectML.dll comes from Microsoft.AI.DirectML (latest/fastest version)
    if(_DML_DLL)
        list(APPEND ORT_EXTRA_DLLS "${_DML_DLL}")
    endif()
    set(ORT_COMPILE_DEFS ORT_ENABLE_DML)

else()
    # ── CPU-only backend, x86 ─────────────────────────────────────────────────
    # DirectML has no x86 build. Use the plain Microsoft.ML.OnnxRuntime NuGet.
    message(STATUS "x86 build: DirectML unavailable, using CPU-only ORT NuGet")
    if(DEFINED ORT_NUGET_DIR AND EXISTS "${ORT_NUGET_DIR}")
        message(STATUS "ORT CPU NuGet: using externally provided dir ${ORT_NUGET_DIR}")
        set(_ORT_CPU_SRC "${ORT_NUGET_DIR}")
    else()
        message(STATUS "ORT CPU NuGet: downloading via FetchContent")
        FetchContent_Declare(ort_cpu_nuget
            URL "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${ORT_VERSION}/microsoft.ml.onnxruntime.${ORT_VERSION}.nupkg"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(ort_cpu_nuget)
        set(_ORT_CPU_SRC "${ort_cpu_nuget_SOURCE_DIR}")
    endif()
    set(ort_cpu_nuget_SOURCE_DIR "${_ORT_CPU_SRC}")
    ort_resolve_nuget_paths(ort_cpu_nuget_SOURCE_DIR "x86")
    set(ORT_EXTRA_DLLS "")
    set(ORT_COMPILE_DEFS "")   # no GPU EPs for x86

endif()

if(NOT EXISTS "${ORT_LIB}")
    message(FATAL_ERROR
        "ORT import lib not found: ${ORT_LIB}\n"
        "Check that FetchContent downloaded the package correctly.")
endif()
message(STATUS "ORT include: ${ORT_INC}")
message(STATUS "ORT lib:     ${ORT_LIB}")

# ── Shaders ───────────────────────────────────────────────────────────────────
find_program(FXC fxc
    HINTS
        "$ENV{WindowsSdkDir}bin/$ENV{WindowsSDKVersion}/${ARCH_SUFFIX}"
        "$ENV{WindowsSdkDir}bin/$ENV{WindowsSDKVersion}/x64"
        "$ENV{WindowsSdkDir}bin/x64")

if(FXC)
    add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/stereo_warp_ps.h"
        COMMAND "${FXC}" /T ps_5_0 /E PS_StereoWarp
                /Fh "${CMAKE_BINARY_DIR}/stereo_warp_ps.h" /Vn g_psStereoWarp
                "${CMAKE_SOURCE_DIR}/shaders/stereo_warp.hlsl"
        DEPENDS "${CMAKE_SOURCE_DIR}/shaders/stereo_warp.hlsl"
        COMMENT "FXC: PS_StereoWarp")
    add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/stereo_warp_vs.h"
        COMMAND "${FXC}" /T vs_5_0 /E VS_FullScreen
                /Fh "${CMAKE_BINARY_DIR}/stereo_warp_vs.h" /Vn g_vsStereoWarp
                "${CMAKE_SOURCE_DIR}/shaders/stereo_warp.hlsl"
        DEPENDS "${CMAKE_SOURCE_DIR}/shaders/stereo_warp.hlsl"
        COMMENT "FXC: VS_FullScreen")
    set(SHADER_OUTPUTS
        "${CMAKE_BINARY_DIR}/stereo_warp_ps.h"
        "${CMAKE_BINARY_DIR}/stereo_warp_vs.h")
    add_compile_definitions(USE_PRECOMPILED_SHADERS)
else()
    message(STATUS "fxc not found – shaders compiled at runtime via D3DCompile")
    set(SHADER_OUTPUTS "")
endif()

# ── Filter DLL ────────────────────────────────────────────────────────────────
set(SOURCES
    src/dllmain.cpp
    src/filter.cpp
    "${BASECLASSES_DIR}/dllentry.cpp"
    src/depth_estimator.cpp
    src/stereo_renderer.cpp
    src/logger.cpp
    src/prop_page.cpp
    src/3Deflatten.rc
    ${SHADER_OUTPUTS}
)
set(HEADERS
    src/guids.h src/ideflatten.h src/filter.h
    src/depth_estimator.h src/stereo_renderer.h
    src/logger.h src/prop_page.h src/resource.h
)

add_library(3Deflatten SHARED ${SOURCES} ${HEADERS})

# Output directory name encodes the backend so all builds can coexist
if(ORT_BACKEND STREQUAL "GPU13")
    set(AX_SUBDIR "Win64_GPU")       # same folder as before -- the "production" GPU build
elseif(ORT_BACKEND STREQUAL "GPU")
    set(AX_SUBDIR "Win64_GPU11")     # distinct folder for the CUDA 11 test build
elseif(ORT_BACKEND STREQUAL "Unified")
    set(AX_SUBDIR "Win64_Unified")   # all EPs: TRT-RTX + TRT + CUDA + DML + CPU
else()
    if(ARCH_SUFFIX STREQUAL "x64")
        set(AX_SUBDIR "Win64")
    else()
        set(AX_SUBDIR "Win32")
    endif()
endif()
set(AX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/${AX_SUBDIR}")

set_target_properties(3Deflatten PROPERTIES
    OUTPUT_NAME          "3Deflatten_${ARCH_SUFFIX}"
    PREFIX               ""
    SUFFIX               ".ax"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${AX_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${AX_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${AX_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${AX_OUTPUT_DIR}"
)

target_include_directories(3Deflatten PRIVATE
    src
    "${ORT_INC}"
    "${CMAKE_BINARY_DIR}"
    "${WINSDK_INC_DIR}/um"
    "${WINSDK_INC_DIR}/shared"
)

# Unified-backend extra includes (CUDA Toolkit + TRT-RTX headers)
if(ORT_BACKEND STREQUAL "Unified")
    if(CUDAToolkit_INCLUDE_DIRS)
        target_include_directories(3Deflatten PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    endif()
    if(_TRTRTX_INC AND EXISTS "${_TRTRTX_INC}")
        target_include_directories(3Deflatten PRIVATE "${_TRTRTX_INC}")
    endif()
endif()

target_compile_definitions(3Deflatten PRIVATE
    _USRDLL WIN32_LEAN_AND_MEAN NOMINMAX
    _CRT_SECURE_NO_WARNINGS UNICODE _UNICODE
    ORT_VER_STR="${ORT_VERSION}"
    ${ORT_COMPILE_DEFS}
)
target_compile_options(3Deflatten PRIVATE /W4 /wd4996 /wd4267 /wd4244)

target_link_libraries(3Deflatten PRIVATE
    strmbase
    "${ORT_LIB}"
    d3d11.lib dxgi.lib d3dcompiler.lib
    winmm.lib strmiids.lib ole32.lib oleaut32.lib uuid.lib
    advapi32.lib comdlg32.lib comctl32.lib shlwapi.lib shell32.lib
    delayimp.lib          # required for /DELAYLOAD
)

# Unified-backend: link CUDA runtime (headers + cudart needed at compile time).
# nvinfer_10.lib and nvonnxparser_10.lib are intentionally NOT linked here —
# their factory functions are loaded via GetProcAddress at runtime, making the
# build independent of .lib symbol-name differences between TRT 10.x and TRT-RTX 1.4.
if(ORT_BACKEND STREQUAL "Unified")
    if(TARGET CUDA::cudart)
        target_link_libraries(3Deflatten PRIVATE CUDA::cudart)
    endif()
endif()

# ── Delay-load ORT ───────────────────────────────────────────────────────────
# With delay-loading, onnxruntime.dll is NOT resolved at .ax load time; it is
# resolved on first call.  This lets DllInit (dllmain.cpp) call
# AddDllDirectory() first, which is what allows the filter to work as a
# COM-registered filter even when onnxruntime.dll is not on the system PATH.
# Without this, regsvr32 / COM would fail with CO_E_ERRORINDLL (error 126).
#
# NOTE: we only DELAYLOAD onnxruntime.dll because that is the only ORT DLL
# our .ax actually imports from directly.  The provider DLLs
# (providers_cuda, providers_tensorrt, providers_shared, directml) are loaded
# at runtime by ORT itself via LoadLibrary and have no direct import entries
# in our .ax.  ORT's own import lib injects /DELAYLOAD pragmas for those DLLs;
# since we have no imports from them, the linker emits LNK4199 warnings.
# We suppress 4199 (and 4070 for the LIBRARY name mismatch from the .def).
# These are purely cosmetic -- the .ax works correctly.
set_target_properties(3Deflatten PROPERTIES
    LINK_FLAGS "/DEF:\"${CMAKE_SOURCE_DIR}/src/3Deflatten.def\" /DELAYLOAD:onnxruntime.dll /ignore:4070 /ignore:4199"
)

# ── Copy runtime DLLs next to the .ax ─────────────────────────────────────────
add_custom_command(TARGET 3Deflatten POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${AX_OUTPUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DLL}" "${AX_OUTPUT_DIR}/onnxruntime.dll"
    COMMENT "Copying onnxruntime.dll (${ORT_BACKEND} backend)"
)
foreach(extra_dll ${ORT_EXTRA_DLLS})
    if(EXISTS "${extra_dll}")
        get_filename_component(dll_name "${extra_dll}" NAME)
        add_custom_command(TARGET 3Deflatten POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${extra_dll}" "${AX_OUTPUT_DIR}/${dll_name}"
        )
    endif()
endforeach()

# ── RC compiler flags ─────────────────────────────────────────────────────────
set_source_files_properties(src/3Deflatten.rc PROPERTIES
    COMPILE_FLAGS "/I \"${WINSDK_INC_DIR}/um\" /I \"${CMAKE_SOURCE_DIR}/src\""
)

install(TARGETS 3Deflatten DESTINATION bin)
