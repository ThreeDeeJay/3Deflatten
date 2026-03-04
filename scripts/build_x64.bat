@echo off
setlocal
echo ============================================================
echo  3Deflatten ^| Build x64 (Release)
echo ============================================================
echo.
echo Prerequisites:
echo   1. Visual Studio 2022 with C++ workload
echo   2. CMake 3.20+  (https://cmake.org)
echo   3. Git           (needed by FetchContent for strmbase)
echo   4. vcpkg with VCPKG_ROOT set:
echo        vcpkg install onnxruntime:x64-windows directxtk:x64-windows
echo.
echo   NOTE: strmbase (DirectShow base classes) is fetched and built
echo   automatically by CMake -- no manual setup required.
echo.
echo   For NVIDIA CUDA acceleration (optional, faster AI inference):
echo     1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
echo     2. vcpkg install onnxruntime[cuda]:x64-windows
echo     3. Re-run this script
echo.

if "%VCPKG_ROOT%"=="" (
    echo ERROR: VCPKG_ROOT is not set.
    echo   Set it to your vcpkg installation directory.
    exit /b 1
)

set BUILD_DIR=build_x64

cmake -B "%BUILD_DIR%" -A x64 ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
    -DVCPKG_TARGET_TRIPLET=x64-windows ^
    -DCMAKE_BUILD_TYPE=Release ^
    .
if errorlevel 1 (
    echo [FAIL] CMake configure step failed.
    exit /b 1
)

cmake --build "%BUILD_DIR%" --config Release --parallel
if errorlevel 1 (
    echo [FAIL] Build step failed.
    exit /b 1
)

echo.
echo [OK] Build succeeded.
echo      Output: %BUILD_DIR%\Release\3Deflatten_x64.ax
echo.
echo To register (run as Administrator):
echo   regsvr32 "%BUILD_DIR%\Release\3Deflatten_x64.ax"
