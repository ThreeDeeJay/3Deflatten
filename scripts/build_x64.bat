@echo off
setlocal
echo ============================================================
echo  3Deflatten ^| Build x64 (Release)
echo ============================================================
echo.
echo NOTE: vcpkg dependency required before first build:
echo   vcpkg install onnxruntime:x64-windows directxtk:x64-windows
echo.
echo   For NVIDIA CUDA acceleration (faster than DirectML for AI inference):
echo     1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
echo     2. vcpkg install onnxruntime[cuda]:x64-windows directxtk:x64-windows
echo     3. Add -DCMAKE_C_FLAGS="/DORT_ENABLE_CUDA" to the cmake call below
echo.
echo   Without CUDA: filter auto-selects DirectML (DX12) then CPU.
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
