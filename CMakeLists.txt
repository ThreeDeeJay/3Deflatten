cmake_minimum_required(VERSION 3.20)
project(3Deflatten VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED TRUE)  # don't re-download if already present

# ── Platform target ───────────────────────────────────────────────────────────
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH_SUFFIX "x64")
    set(ORT_ARCH   "x64")
else()
    set(ARCH_SUFFIX "x86")
    set(ORT_ARCH   "x86")
endif()

# ── Windows SDK include/lib dirs ──────────────────────────────────────────────
set(WINSDK_LIB_DIR
    "$ENV{WindowsSdkDir}Lib/$ENV{WindowsSDKVersion}/um/${ARCH_SUFFIX}"
    CACHE PATH "Windows SDK UM lib directory")
set(WINSDK_INC_DIR
    "$ENV{WindowsSdkDir}Include/$ENV{WindowsSDKVersion}"
    CACHE PATH "Windows SDK include root")

# ── DirectShow base classes (strmbase) – built from source ───────────────────
FetchContent_Declare(
    win_classic_samples
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
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
    UNICODE _UNICODE
    _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
)
target_compile_options(strmbase PRIVATE
    /W3 /wd4996 /wd4267 /wd4244 /wd4312 /wd4311 /wd4302
    /FI"algorithm"
)

# ── ONNX Runtime (official NuGet – includes DirectML EP + bundled DML dll) ───
# The vcpkg onnxruntime port does not compile the DirectML or CUDA EPs.
# Microsoft's NuGet package 'Microsoft.ML.OnnxRuntime.DirectML' ships DML
# support built-in, along with a bundled directml.dll.  We download it via
# FetchContent (NuGet packages are just ZIP archives).
set(ORT_DML_VERSION "1.21.0")
FetchContent_Declare(ort_nuget
    URL "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/${ORT_DML_VERSION}/microsoft.ml.onnxruntime.directml.${ORT_DML_VERSION}.nupkg"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(ort_nuget)

# NuGet package layout:
#   build/native/include/          – ORT C/C++ headers
#   build/native/lib/x64/          – import lib
#   runtimes/win-x64/native/       – DLLs (onnxruntime.dll, directml.dll)
set(ORT_INC  "${ort_nuget_SOURCE_DIR}/build/native/include")
set(ORT_LIB  "${ort_nuget_SOURCE_DIR}/build/native/lib/${ORT_ARCH}/onnxruntime.lib")
set(ORT_DLL  "${ort_nuget_SOURCE_DIR}/runtimes/win-${ORT_ARCH}/native/onnxruntime.dll")
set(DML_DLL  "${ort_nuget_SOURCE_DIR}/runtimes/win-${ORT_ARCH}/native/directml.dll")
set(ORT_PROVIDERS_SHARED
    "${ort_nuget_SOURCE_DIR}/runtimes/win-${ORT_ARCH}/native/onnxruntime_providers_shared.dll")

##if(NOT EXISTS "${ORT_LIB}")
##    message(FATAL_ERROR "ORT import lib not found at ${ORT_LIB} -- check NuGet download")
##endif()

# ── Optional CUDA EP ──────────────────────────────────────────────────────────
# To enable: cmake -DORT_CUDA=ON  (also requires CUDA Toolkit installed).
# With the DML NuGet you don't get CUDA; for CUDA you'd need either vcpkg's
# onnxruntime[cuda] or the Microsoft.ML.OnnxRuntime.Gpu NuGet instead.
option(ORT_CUDA "Enable NVIDIA CUDA execution provider" OFF)
if(ORT_CUDA)
    add_compile_definitions(ORT_ENABLE_CUDA)
    message(STATUS "CUDA EP: enabled (requires CUDA Toolkit)")
endif()

# ── Shader compilation ────────────────────────────────────────────────────────
find_program(FXC fxc
    HINTS
        "$ENV{WindowsSdkDir}bin/$ENV{WindowsSDKVersion}/${ARCH_SUFFIX}"
        "$ENV{WindowsSdkDir}bin/$ENV{WindowsSDKVersion}/x64"
        "$ENV{WindowsSdkDir}bin/x64")

if(FXC)
    add_custom_command(
        OUTPUT  "${CMAKE_BINARY_DIR}/stereo_warp_ps.h"
        COMMAND "${FXC}"
                /T ps_5_0 /E PS_StereoWarp
                /Fh "${CMAKE_BINARY_DIR}/stereo_warp_ps.h"
                /Vn g_psStereoWarp
                "${CMAKE_SOURCE_DIR}/shaders/stereo_warp.hlsl"
        DEPENDS "${CMAKE_SOURCE_DIR}/shaders/stereo_warp.hlsl"
        COMMENT "Compiling stereo_warp.hlsl (PS_StereoWarp)")
    add_custom_command(
        OUTPUT  "${CMAKE_BINARY_DIR}/stereo_warp_vs.h"
        COMMAND "${FXC}"
                /T vs_5_0 /E VS_FullScreen
                /Fh "${CMAKE_BINARY_DIR}/stereo_warp_vs.h"
                /Vn g_vsStereoWarp
                "${CMAKE_SOURCE_DIR}/shaders/stereo_warp.hlsl"
        DEPENDS "${CMAKE_SOURCE_DIR}/shaders/stereo_warp.hlsl"
        COMMENT "Compiling stereo_warp.hlsl (VS_FullScreen)")
    set(SHADER_OUTPUTS
        "${CMAKE_BINARY_DIR}/stereo_warp_ps.h"
        "${CMAKE_BINARY_DIR}/stereo_warp_vs.h")
    add_compile_definitions(USE_PRECOMPILED_SHADERS)
else()
    message(STATUS "fxc not found – shaders compiled at runtime via D3DCompile")
    set(SHADER_OUTPUTS "")
endif()

# ── Filter DLL sources ────────────────────────────────────────────────────────
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

set_target_properties(3Deflatten PROPERTIES
    OUTPUT_NAME "3Deflatten_${ARCH_SUFFIX}"
    PREFIX      ""
    SUFFIX      ".ax"
)

target_include_directories(3Deflatten PRIVATE
    src
    "${ORT_INC}"                        # official ORT NuGet headers
    "${CMAKE_BINARY_DIR}"               # compiled shader headers
    "${WINSDK_INC_DIR}/um"
    "${WINSDK_INC_DIR}/shared"
)

target_compile_definitions(3Deflatten PRIVATE
    _USRDLL WIN32_LEAN_AND_MEAN NOMINMAX
    _CRT_SECURE_NO_WARNINGS UNICODE _UNICODE
)
target_compile_options(3Deflatten PRIVATE /W4 /wd4996 /wd4267 /wd4244)

target_link_libraries(3Deflatten PRIVATE
    strmbase
    "${ORT_LIB}"
    d3d11.lib dxgi.lib d3dcompiler.lib
    winmm.lib strmiids.lib ole32.lib oleaut32.lib uuid.lib
    advapi32.lib comdlg32.lib comctl32.lib shlwapi.lib
)

set_target_properties(3Deflatten PROPERTIES
    LINK_FLAGS "/DEF:\"${CMAKE_SOURCE_DIR}/src/3Deflatten.def\""
)

# Copy ORT and DirectML DLLs next to the .ax so they're found at runtime.
add_custom_command(TARGET 3Deflatten POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${AX_OUTPUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DLL}" "${AX_OUTPUT_DIR}/onnxruntime.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DML_DLL}" "${AX_OUTPUT_DIR}/directml.dll"
    COMMENT "Copying ORT + DirectML DLLs to output"
)
if(EXISTS "${ORT_PROVIDERS_SHARED}")
    add_custom_command(TARGET 3Deflatten POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_PROVIDERS_SHARED}" "${AX_OUTPUT_DIR}/onnxruntime_providers_shared.dll"
    )
endif()

# ── RC compiler include path (needed for commctrl constants in the dialog) ────
set_source_files_properties(src/3Deflatten.rc PROPERTIES
    COMPILE_FLAGS "/I \"${WINSDK_INC_DIR}/um\" /I \"${CMAKE_SOURCE_DIR}/src\""
)

install(TARGETS 3Deflatten DESTINATION bin)
