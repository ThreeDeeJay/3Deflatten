@echo off
:: Run this script as Administrator
setlocal

:: Always run relative to the script's own directory
cd /d "%~dp0"

set AX64=Win64\3Deflatten_x64.ax
set AX86=Win32\3Deflatten_x86.ax

echo ============================================================
echo  3Deflatten ^| Register DirectShow Filters
echo  (Requires Administrator privileges)
echo ============================================================
echo.

net session >nul 2>&1
if errorlevel 1 (
    echo ERROR: Please run this script as Administrator.
    pause
    exit /b 1
)

if exist "%AX64%" (
    regsvr32 /s "%AX64%"
    if errorlevel 1 (
        echo [WARN] x64 registration failed.
    ) else (
        echo [OK]   Registered x64: %AX64%
    )
) else (
    echo [SKIP] x64 build not found: %AX64%
)

if exist "%AX86%" (
    %SystemRoot%\SysWOW64\regsvr32 /s "%AX86%"
    if errorlevel 1 (
        echo [WARN] x86 registration failed.
    ) else (
        echo [OK]   Registered x86: %AX86%
    )
) else (
    echo [SKIP] x86 build not found: %AX86%
)

echo.
echo ============================================================
echo  Optional environment variables:
echo ============================================================
echo  DEFLATTEN_LOG_FILE    = C:\path\to\deflatten.log
echo  DEFLATTEN_MODEL_PATH  = C:\path\to\model.onnx
echo ============================================================
echo.
pause
