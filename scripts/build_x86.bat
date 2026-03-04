@echo off
setlocal
echo ============================================================
echo  3Deflatten ^| Build x86 / Win32 (Release)
echo ============================================================
echo.
echo NOTE: vcpkg dependencies required before first build:
echo   vcpkg install onnxruntime:x86-windows directxtk:x86-windows
echo.
echo   Optional - for NVIDIA GPU acceleration, install CUDA Toolkit first:
echo   https://developer.nvidia.com/cuda-downloads
echo   Then reinstall: vcpkg install onnxruntime[cuda]:x86-windows
echo.

if "%VCPKG_ROOT%"=="" (
    echo ERROR: VCPKG_ROOT is not set.
    echo   Set it to your vcpkg installation directory.
    exit /b 1
)

set BUILD_DIR=build_x86

cmake -B "%BUILD_DIR%" -A Win32 ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
    -DVCPKG_TARGET_TRIPLET=x86-windows ^
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
echo      Output: %BUILD_DIR%\Release\3Deflatten_x86.ax
echo.
echo To register on 64-bit Windows (run as Administrator):
echo   %%SystemRoot%%\SysWOW64\regsvr32 "%BUILD_DIR%\Release\3Deflatten_x86.ax"
