@echo off
:: Run this script as Administrator
setlocal

:: Always run relative to the script's own directory
cd /d "%~dp0"

set AX64=Win64\3Deflatten_x64.ax
set AX64G=Win64_GPU\3Deflatten_x64.ax
set AX86=Win32\3Deflatten_x86.ax

echo ============================================================
echo  3Deflatten ^| Unregister DirectShow Filters
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
    regsvr32 /s /u "%AX64%"
    if errorlevel 1 (
        echo [WARN] x64 DirectML unregistration failed.
    ) else (
        echo [OK]   Unregistered x64 DirectML: %AX64%
    )
) else (
    echo [SKIP] x64 DirectML build not found: %AX64%
)

if exist "%AX64G%" (
    regsvr32 /s /u "%AX64G%"
    if errorlevel 1 (
        echo [WARN] x64 GPU unregistration failed.
    ) else (
        echo [OK]   Unregistered x64 GPU:      %AX64G%
    )
) else (
    echo [SKIP] x64 GPU build not found: %AX64G%
)

if exist "%AX86%" (
    %SystemRoot%\SysWOW64\regsvr32 /s /u "%AX86%"
    if errorlevel 1 (
        echo [WARN] x86 unregistration failed.
    ) else (
        echo [OK]   Unregistered x86:          %AX86%
    )
) else (
    echo [SKIP] x86 build not found: %AX86%
)

echo.
pause
