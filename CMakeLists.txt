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
FetchContent_Populate(win_classic_samples)

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
#   GPU                 –  ORT official GitHub release zip (CUDA + TRT)
#       EPs available at runtime: TensorRT, CUDA, CPU
#       Defines: ORT_ENABLE_TENSORRT, ORT_ENABLE_CUDA
#       Good for: NVIDIA with CUDA Toolkit + (optionally) TensorRT SDK
#       Note: x64 only. CUDA/TRT DLLs NOT bundled; user must have them installed.
#
# Select with:  cmake -B build -DORT_BACKEND=GPU   (or DirectML, the default)
#
set(ORT_VERSION "1.21.0")
set(ORT_BACKEND "DirectML" CACHE STRING "ORT inference backend: DirectML or GPU")
set_property(CACHE ORT_BACKEND PROPERTY STRINGS "DirectML" "GPU")
message(STATUS "ORT backend: ${ORT_BACKEND}")

# ── Helper macro: resolve ORT paths from an extracted NuGet/zip directory ────
# NuGet packages (.nupkg are zips) are extracted by FetchContent directly into
# SOURCE_DIR with no wrapper folder, so the layout is always:
#   SOURCE_DIR/build/native/include/
#   SOURCE_DIR/build/native/lib/<arch>/onnxruntime.lib
#   SOURCE_DIR/runtimes/win-<arch>/native/onnxruntime.dll
# However the lib sub-arch folder name has varied across package versions
# (x64, x86, arm64).  We glob to be safe rather than hardcoding it.
macro(ort_resolve_nuget_paths SOURCE_VAR ARCH_RID)
    set(_ORT_SRC "${${SOURCE_VAR}}")
    # headers are stable
    set(ORT_INC "${_ORT_SRC}/build/native/include")
    # find the import lib by glob
    file(GLOB_RECURSE _ORT_LIB_CANDIDATES "${_ORT_SRC}/build/native/lib/*.lib")
    if(NOT _ORT_LIB_CANDIDATES)
        # some versions put the lib directly under runtimes/
        file(GLOB_RECURSE _ORT_LIB_CANDIDATES "${_ORT_SRC}/runtimes/*.lib")
    endif()
    list(FILTER _ORT_LIB_CANDIDATES INCLUDE REGEX "onnxruntime\.lib$")
    if(_ORT_LIB_CANDIDATES)
        list(GET _ORT_LIB_CANDIDATES 0 ORT_LIB)
        message(STATUS "ORT import lib: ${ORT_LIB}")
    else()
        # leave ORT_LIB pointing at the expected path so the fatal check below fires
        set(ORT_LIB "${_ORT_SRC}/build/native/lib/${ARCH_RID}/onnxruntime.lib")
    endif()
    # DLL
    set(ORT_DLL "${_ORT_SRC}/runtimes/win-${ARCH_RID}/native/onnxruntime.dll")
    if(NOT EXISTS "${ORT_DLL}")
        # flat layout fallback
        file(GLOB_RECURSE _ORT_DLL_CANDIDATES "${_ORT_SRC}/runtimes/*.dll")
        list(FILTER _ORT_DLL_CANDIDATES INCLUDE REGEX "onnxruntime\.dll$")
        list(FILTER _ORT_DLL_CANDIDATES EXCLUDE REGEX "providers")
        if(_ORT_DLL_CANDIDATES)
            list(GET _ORT_DLL_CANDIDATES 0 ORT_DLL)
        endif()
    endif()
endmacro()

if(ORT_BACKEND STREQUAL "GPU")
    # ── GPU backend (TensorRT + CUDA + CPU) ───────────────────────────────────
    if(NOT ARCH_SUFFIX STREQUAL "x64")
        message(FATAL_ERROR "ORT_BACKEND=GPU is only supported for x64 builds. "
                            "Use ORT_BACKEND=DirectML for x86.")
    endif()

    FetchContent_Declare(ort_gpu
        URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-gpu-${ORT_VERSION}.zip"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(ort_gpu)

    # The GitHub release zip may extract to a nested subdirectory OR directly
    # into SOURCE_DIR depending on the archive structure and CMake version.
    # Probe both layouts so the build is robust against version changes:
    #   Layout A (nested):  SOURCE_DIR/onnxruntime-win-x64-gpu-<ver>/lib/
    #   Layout B (flat):    SOURCE_DIR/lib/
    set(_ORT_NESTED "${ort_gpu_SOURCE_DIR}/onnxruntime-win-x64-gpu-${ORT_VERSION}")
    if(EXISTS "${_ORT_NESTED}/lib/onnxruntime.lib")
        set(ORT_ZIP_ROOT "${_ORT_NESTED}")
        message(STATUS "ORT GPU zip: nested layout detected (${ORT_ZIP_ROOT})")
    elseif(EXISTS "${ort_gpu_SOURCE_DIR}/lib/onnxruntime.lib")
        set(ORT_ZIP_ROOT "${ort_gpu_SOURCE_DIR}")
        message(STATUS "ORT GPU zip: flat layout detected (${ORT_ZIP_ROOT})")
    else()
        # Last resort: glob for onnxruntime.lib anywhere under SOURCE_DIR
        file(GLOB_RECURSE _ORT_LIB_CANDIDATES
             "${ort_gpu_SOURCE_DIR}/*/onnxruntime.lib")
        list(GET _ORT_LIB_CANDIDATES 0 _ORT_LIB_FOUND)
        if(_ORT_LIB_FOUND)
            get_filename_component(ORT_ZIP_ROOT "${_ORT_LIB_FOUND}" DIRECTORY)
            get_filename_component(ORT_ZIP_ROOT "${ORT_ZIP_ROOT}" DIRECTORY)
            message(STATUS "ORT GPU zip: found via glob at ${ORT_ZIP_ROOT}")
        else()
            # Will be caught by the existence check below with a clear message
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
    set(ORT_COMPILE_DEFS ORT_ENABLE_TENSORRT ORT_ENABLE_CUDA)

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
    set(ORT_EXTRA_DLLS "")
    foreach(_extra
            "${_ORT_NUGET_SRC}/runtimes/win-x64/native/directml.dll"
            "${_ORT_NUGET_SRC}/runtimes/win-x64/native/onnxruntime_providers_shared.dll")
        if(EXISTS "${_extra}")
            list(APPEND ORT_EXTRA_DLLS "${_extra}")
        endif()
    endforeach()
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

# Output directory name encodes the backend so both builds can coexist
if(ORT_BACKEND STREQUAL "GPU")
    set(AX_SUBDIR "Win64_GPU")
else()
    set(AX_SUBDIR "${ARCH_SUFFIX STREQUAL x64 ? \"Win64\" : \"Win32\"}")
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

target_compile_definitions(3Deflatten PRIVATE
    _USRDLL WIN32_LEAN_AND_MEAN NOMINMAX
    _CRT_SECURE_NO_WARNINGS UNICODE _UNICODE
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

# ── Delay-load ORT and GPU-provider DLLs ─────────────────────────────────────
# With delay-loading these DLLs are NOT resolved at .ax load time; they are
# resolved on first call.  This lets DllInit (dllmain.cpp) call
# AddDllDirectory() first, which is what allows the filter to work as a
# COM-registered filter even when onnxruntime.dll is not on the system PATH.
# Without this, regsvr32 / COM would fail with CO_E_ERRORINDLL (error 126).
set(_DELAYLOAD_FLAGS
    "/DELAYLOAD:onnxruntime.dll"
    "/DELAYLOAD:onnxruntime_providers_shared.dll"
    "/DELAYLOAD:onnxruntime_providers_cuda.dll"
    "/DELAYLOAD:onnxruntime_providers_tensorrt.dll"
    "/DELAYLOAD:directml.dll"
)
string(JOIN " " _DELAYLOAD_STR ${_DELAYLOAD_FLAGS})

set_target_properties(3Deflatten PROPERTIES
    LINK_FLAGS "/DEF:\"${CMAKE_SOURCE_DIR}/src/3Deflatten.def\" ${_DELAYLOAD_STR}"
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
