cmake_minimum_required(VERSION 3.20)
project(3Deflatten VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ── Platform target ───────────────────────────────────────────────────────────
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH_SUFFIX "x64")
else()
    set(ARCH_SUFFIX "x86")
endif()

# ── Windows SDK include/lib dirs ──────────────────────────────────────────────
# Needed for d3d11.lib, dxgi.lib, strmiids.lib, etc.
set(WINSDK_LIB_DIR
    "$ENV{WindowsSdkDir}Lib/$ENV{WindowsSDKVersion}/um/${ARCH_SUFFIX}"
    CACHE PATH "Windows SDK UM lib directory")

set(WINSDK_INC_DIR
    "$ENV{WindowsSdkDir}Include/$ENV{WindowsSDKVersion}"
    CACHE PATH "Windows SDK include root")

# ── DirectShow base classes (strmbase) – built from source ───────────────────
# Fetches the Microsoft Windows-classic-samples repo (shallow, sparse).
# The baseclasses folder is at:
#   Samples/Win7Samples/multimedia/directshow/baseclasses/
FetchContent_Declare(
    win_classic_samples
    GIT_REPOSITORY https://github.com/microsoft/Windows-classic-samples.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
    GIT_SPARSE_CHECKOUT_PATTERNS
        "Samples/Win7Samples/multimedia/directshow/baseclasses"
    SOURCE_SUBDIR  "Samples/Win7Samples/multimedia/directshow/baseclasses"
)
# We only need to populate (download), not add_subdirectory – it has no CMakeLists.
FetchContent_Populate(win_classic_samples)

set(BASECLASSES_DIR
    "${win_classic_samples_SOURCE_DIR}/Samples/Win7Samples/multimedia/directshow/baseclasses")

# Compile the DirectShow base classes as a static library target.
file(GLOB STRMBASE_SOURCES "${BASECLASSES_DIR}/*.cpp")
# Remove dllentry.cpp – it defines DllMain which conflicts with ours.
list(FILTER STRMBASE_SOURCES EXCLUDE REGEX "dllentry\\.cpp$")

add_library(strmbase STATIC ${STRMBASE_SOURCES})

target_include_directories(strmbase PUBLIC
    "${BASECLASSES_DIR}"
    "${WINSDK_INC_DIR}/um"
    "${WINSDK_INC_DIR}/shared"
)

target_compile_definitions(strmbase PUBLIC
    WIN32_LEAN_AND_MEAN
    # Do NOT define NOMINMAX for strmbase: the baseclasses rely on the Windows
    # min/max macros and do not include <algorithm> themselves.
    # NOMINMAX is defined only on our own filter sources below.
    _CRT_SECURE_NO_WARNINGS
    UNICODE
    _UNICODE
    _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
)

# /FI"algorithm" force-includes <algorithm> so std::min/std::max are always
# available even in translation units that don't include it explicitly.
# Also suppress the many warnings the old baseclasses generate under MSVC.
target_compile_options(strmbase PRIVATE
    /W3
    /wd4996 /wd4267 /wd4244 /wd4312 /wd4311 /wd4302
    /FI"algorithm"
)

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
find_package(onnxruntime CONFIG REQUIRED)

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

# ── Filter DLL source files ───────────────────────────────────────────────────
set(SOURCES
    src/dllmain.cpp
    src/filter.cpp
    src/depth_estimator.cpp
    src/stereo_renderer.cpp
    src/logger.cpp
    src/prop_page.cpp
    src/3Deflatten.rc
    ${SHADER_OUTPUTS}
)

set(HEADERS
    src/guids.h
    src/ideflatten.h
    src/filter.h
    src/depth_estimator.h
    src/stereo_renderer.h
    src/logger.h
    src/prop_page.h
    src/resource.h
)

# ── Filter DLL ────────────────────────────────────────────────────────────────
add_library(3Deflatten SHARED ${SOURCES} ${HEADERS})

set_target_properties(3Deflatten PROPERTIES
    OUTPUT_NAME "3Deflatten_${ARCH_SUFFIX}"
    PREFIX      ""
    SUFFIX      ".ax"
)

target_include_directories(3Deflatten PRIVATE
    src
    "${CMAKE_BINARY_DIR}"               # for compiled shader headers
    "${WINSDK_INC_DIR}/um"
    "${WINSDK_INC_DIR}/shared"
)

target_compile_definitions(3Deflatten PRIVATE
    _USRDLL
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
    UNICODE
    _UNICODE
)

target_compile_options(3Deflatten PRIVATE /W4 /wd4996 /wd4267 /wd4244)

target_link_libraries(3Deflatten PRIVATE
    strmbase                    # built from source above
    onnxruntime::onnxruntime
    d3d11.lib
    dxgi.lib
    d3dcompiler.lib
    winmm.lib
    strmiids.lib
    ole32.lib
    oleaut32.lib
    uuid.lib
    advapi32.lib
    comdlg32.lib
    comctl32.lib
)

set_target_properties(3Deflatten PROPERTIES
    LINK_FLAGS "/DEF:\"${CMAKE_SOURCE_DIR}/src/3Deflatten.def\""
)

install(TARGETS 3Deflatten DESTINATION bin)
