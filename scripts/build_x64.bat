@echo off
setlocal
echo ============================================================
echo  3Deflatten ^| Build x64 (Release)
echo ============================================================
echo.
echo NOTE: vcpkg dependencies required before first build:
echo   vcpkg install onnxruntime[cuda]:x64-windows directxtk:x64-windows
echo   (DirectML support is included in onnxruntime automatically on Windows)
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
