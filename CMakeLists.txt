cmake_minimum_required(VERSION 3.20)
project(3Deflatten VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Platform target ──────────────────────────────────────────────────────────
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH_SUFFIX "x64")
else()
    set(ARCH_SUFFIX "x86")
endif()

# ── Windows SDK / DirectShow base classes ────────────────────────────────────
# Requires Windows 10 SDK (10.0.19041.0 or newer).
# strmbase.lib ships with the WDK/SDK samples – build it first or provide path.
set(WINSDK_LIB_DIR
    "$ENV{WindowsSdkDir}Lib/$ENV{WindowsSDKVersion}/um/${ARCH_SUFFIX}"
    CACHE PATH "Windows SDK lib directory")

find_library(STRMBASE_LIB
    NAMES strmbase strmbasd
    HINTS
        "${WINSDK_LIB_DIR}"
        "$ENV{DXSDK_DIR}Lib/${ARCH_SUFFIX}"
        "${CMAKE_SOURCE_DIR}/lib/${ARCH_SUFFIX}"
    REQUIRED)

# ── ONNX Runtime ─────────────────────────────────────────────────────────────
find_package(onnxruntime CONFIG REQUIRED)

# ── DirectX 11 ───────────────────────────────────────────────────────────────
find_library(D3D11_LIB d3d11 HINTS "${WINSDK_LIB_DIR}" REQUIRED)
find_library(DXGI_LIB  dxgi  HINTS "${WINSDK_LIB_DIR}" REQUIRED)
find_library(D3DCOMPILER_LIB d3dcompiler HINTS "${WINSDK_LIB_DIR}" REQUIRED)

# ── Shader compilation ───────────────────────────────────────────────────────
find_program(FXC fxc
    HINTS "$ENV{WindowsSdkDir}bin/$ENV{WindowsSDKVersion}/${ARCH_SUFFIX}"
          "$ENV{WindowsSdkDir}bin/x64")

if(FXC)
    add_custom_command(
        OUTPUT  "${CMAKE_BINARY_DIR}/stereo_warp_ps.h"
        COMMAND "${FXC}"
                /T ps_5_0 /E PS_StereoWarp /Fh
                "${CMAKE_BINARY_DIR}/stereo_warp_ps.h"
                /Vn g_psStereoWarp
                "${CMAKE_SOURCE_DIR}/shaders/stereo_warp.hlsl"
        DEPENDS "${CMAKE_SOURCE_DIR}/shaders/stereo_warp.hlsl"
        COMMENT "Compiling stereo_warp.hlsl (PS)")

    add_custom_command(
        OUTPUT  "${CMAKE_BINARY_DIR}/stereo_warp_vs.h"
        COMMAND "${FXC}"
                /T vs_5_0 /E VS_FullScreen /Fh
                "${CMAKE_BINARY_DIR}/stereo_warp_vs.h"
                /Vn g_vsStereoWarp
                "${CMAKE_SOURCE_DIR}/shaders/stereo_warp.hlsl"
        DEPENDS "${CMAKE_SOURCE_DIR}/shaders/stereo_warp.hlsl"
        COMMENT "Compiling stereo_warp.hlsl (VS)")

    set(SHADER_OUTPUTS
        "${CMAKE_BINARY_DIR}/stereo_warp_ps.h"
        "${CMAKE_BINARY_DIR}/stereo_warp_vs.h")
    add_compile_definitions(USE_PRECOMPILED_SHADERS)
else()
    message(WARNING "fxc not found – shaders will be compiled at runtime via D3DCompile")
    set(SHADER_OUTPUTS "")
endif()

# ── Source files ─────────────────────────────────────────────────────────────
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

# ── Filter DLL ───────────────────────────────────────────────────────────────
add_library(3Deflatten SHARED ${SOURCES} ${HEADERS})

set_target_properties(3Deflatten PROPERTIES
    OUTPUT_NAME "3Deflatten_${ARCH_SUFFIX}"
    PREFIX      ""
    SUFFIX      ".ax"
)

target_include_directories(3Deflatten PRIVATE
    src
    "${CMAKE_BINARY_DIR}"
    "${WINSDK_LIB_DIR}/../../Include"
)

target_compile_definitions(3Deflatten PRIVATE
    _USRDLL
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
    UNICODE
    _UNICODE
)

target_link_libraries(3Deflatten PRIVATE
    ${STRMBASE_LIB}
    onnxruntime::onnxruntime
    ${D3D11_LIB}
    ${DXGI_LIB}
    ${D3DCOMPILER_LIB}
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
