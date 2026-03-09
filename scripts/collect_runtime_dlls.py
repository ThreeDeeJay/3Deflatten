#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
r"""
Collect CUDA 13.x / cuDNN 9 / nvJitLink runtime DLLs into the Win64_GPU
output directory so that the GPU build of 3Deflatten works without a
system-wide CUDA install.

ORT 1.24.x pre-built GPU Windows binary requires CUDA 13.x at runtime.
  - CUDA 12.x (cudart64_12.dll) is NOT compatible with ORT 1.24.x GPU.
  - Install CUDA 13.x: https://developer.nvidia.com/cuda-downloads

DLLs are obtained from the official NVIDIA pip wheels (redistributable).
If a pip wheel is unavailable, the script falls back to the system CUDA
install (useful when CUDA 13 is installed locally but pip wheels are absent).

TensorRT DLLs are copied from TRT_LIB_PATH if that env var is set.
They are NOT auto-downloaded (too large, ~2 GB).  After copying you should
also copy zlibwapi.dll from the TRT lib\ folder (or get it separately --
it is required by TRT but not included in CUDA).

Usage (local, from the release root):
    python collect_runtime_dlls.py
    python collect_runtime_dlls.py --output path\to\Win64_GPU

Usage (CI, from repo root):
    python scripts/collect_runtime_dlls.py --output build\Win64_GPU --no-confirm

Required CUDA 13 DLLs (bundled by this script):
    cudart64_13.dll        - CUDA 13 runtime
    cublas64_13.dll        - cuBLAS 13
    cublasLt64_13.dll      - cuBLAS-Lt 13
    cufft64_12.dll         - cuFFT  (API ver 12, ships in CUDA 13 toolkit)
    nvJitLink_130_0.dll    - CUDA 13 JIT-linker (required by ORT CUDA EP)
    cudnn64_9.dll          - cuDNN 9

Optional TRT DLLs (copied from TRT_LIB_PATH if set):
    nvinfer_10.dll              - TensorRT 10 inference engine
    nvonnxparser_10.dll         - TensorRT 10 ONNX parser
    nvinfer_builder_resource_10.dll  - TRT 10 builder resource
    nvinfer_plugin_10.dll       - TRT 10 built-in plugins
    zlibwapi.dll                - zlib (required by TRT, may be in lib\)
    (all other lib\*.dll)       - copy ALL of them
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

# Force UTF-8 output (Windows console may default to cp1252)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# This script is copied to the release root by the CI package job.
# Win64_GPU lives as a sibling of this script in the release root.
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "Win64_GPU"

# pip package -> list of DLL names it should provide on Windows
# ORT 1.24.x GPU binary requires CUDA 13.x DLL names.
PACKAGES_CU13 = {
    "nvidia-cuda-runtime-cu13":  ["cudart64_13.dll"],
    "nvidia-cublas-cu13":        ["cublas64_13.dll", "cublasLt64_13.dll"],
    "nvidia-cufft-cu13":         ["cufft64_12.dll"],   # cuFFT API ver 12, ships in CUDA 13
    "nvidia-nvjitlink-cu13":     ["nvJitLink_130_0.dll"],  # required by ORT CUDA EP
    "nvidia-cudnn-cu13":         ["cudnn64_9.dll"],
}

# Typical install sub-paths inside site-packages\nvidia\<n>\
SEARCH_SUBDIRS = ["bin", "lib", ""]


def find_dll_in_package(site_packages, pkg_pip_name, dll_name):
    """Search for dll_name inside the nvidia pip package directory."""
    short = (pkg_pip_name
             .replace("nvidia-", "")
             .replace("-cu13", "")
             .replace("-cu12", "")
             .replace("-", "_"))
    candidates = [
        site_packages / "nvidia" / short,
        site_packages / "nvidia" / short.split("_cu")[0],
    ]
    for base in candidates:
        if not base.exists():
            continue
        for sub in SEARCH_SUBDIRS:
            p = base / sub / dll_name if sub else base / dll_name
            if p.exists():
                return p
        for hit in base.rglob(dll_name):
            return hit
    for hit in site_packages.rglob(dll_name):
        return hit
    return None


def install_packages(packages, target_dir):
    """pip install packages into target_dir.  Returns True on success."""
    print(f"  Installing pip packages into temp dir: {target_dir}")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--quiet",
        "--target", str(target_dir),
        "--only-binary=:all:",
    ] + packages
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] pip install failed: {e}")
        return False


def find_in_system_cuda(dll_name):
    """Search for dll_name in system CUDA / cuDNN install."""
    cuda_vars = [
        "CUDA_PATH_V13_2", "CUDA_PATH_V13_1", "CUDA_PATH_V13_0",
        "CUDA_PATH_V12_9", "CUDA_PATH_V12_8", "CUDA_PATH",
    ]
    search_roots = []
    for var in cuda_vars:
        v = os.environ.get(var, "").strip()
        if v:
            search_roots.append(pathlib.Path(v))
            break
    search_roots.append(
        pathlib.Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"))
    for root in search_roots:
        if not root.exists():
            continue
        for hit in root.rglob(dll_name):
            return hit
    # cuDNN fallback
    cudnn_env = os.environ.get("CUDNN_PATH", "").strip()
    if cudnn_env:
        for hit in pathlib.Path(cudnn_env).rglob(dll_name):
            return hit
    cudnn_default = pathlib.Path(r"C:\Program Files\NVIDIA\CUDNN")
    if cudnn_default.exists():
        for hit in cudnn_default.rglob(dll_name):
            return hit
    return None


def copy_trt_dlls(output_dir):
    """Copy ALL TRT DLLs from TRT_LIB_PATH env var. Returns list of copied names."""
    trt_lib = os.environ.get("TRT_LIB_PATH", "").strip()
    if not trt_lib:
        return []
    trt_path = pathlib.Path(trt_lib)
    if not trt_path.exists():
        print(f"  [SKIP] TRT_LIB_PATH={trt_lib} does not exist.")
        return []

    # Copy ALL DLLs from TRT_LIB_PATH (nvinfer*, nvonnxparser*, zlibwapi, etc.)
    all_dlls = sorted(set(
        list(trt_path.rglob("*.dll")) + list(trt_path.rglob("*.DLL"))
    ))
    if not all_dlls:
        print(f"  [SKIP] No DLLs found in TRT_LIB_PATH={trt_lib}")
        return []

    print(f"\n  Copying TRT DLLs from TRT_LIB_PATH={trt_lib}:")
    copied = []
    for src in all_dlls:
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        size_kb = src.stat().st_size // 1024
        print(f"  [OK] {src.name}  ({size_kb:,} KB)  ->  {dst}")
        copied.append(src.name)

    # Warn about key TRT DLLs that are missing
    key_trt_dlls = [
        "nvinfer_10.dll", "nvonnxparser_10.dll",
        "nvinfer_builder_resource_10.dll", "zlibwapi.dll",
    ]
    copied_lower = {c.lower() for c in copied}
    for dll in key_trt_dlls:
        if dll.lower() not in copied_lower:
            print(f"  [WARN] {dll} not found in TRT_LIB_PATH={trt_lib}")
            if dll == "zlibwapi.dll":
                print("         zlibwapi.dll is required by TensorRT.")
                print("         Download: https://www.dll-files.com/zlibwapi.dll.html")
                print("         Place it in the Win64_GPU folder or TRT_LIB_PATH.")
    return copied


def collect(output_dir, no_confirm=False):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n3Deflatten - Collect CUDA 13 Runtime DLLs for GPU Build")
    print("=" * 60)
    print(f"Output: {output_dir}\n")
    print("NOTE: ORT 1.24.x GPU binary requires CUDA 13.x.")
    print("      CUDA 12.x (cudart64_12.dll) is NOT compatible.\n")
    print("CUDA 13 packages to collect:")
    for pkg, dlls in PACKAGES_CU13.items():
        print(f"  {pkg}  ->  {', '.join(dlls)}")
    print()

    if not no_confirm:
        ans = input("Proceed? [Y/n] ").strip().lower()
        if ans not in ("", "y", "yes"):
            print("Cancelled.")
            return False

    missing = []

    with tempfile.TemporaryDirectory(prefix="3df_cuda_") as tmp:
        tmp_path = pathlib.Path(tmp)
        pip_ok = install_packages(list(PACKAGES_CU13.keys()), tmp_path)

        for pkg, dlls in PACKAGES_CU13.items():
            for dll in dlls:
                src = None

                # 1. Try pip wheel (preferred: redistributable, version-pinned)
                if pip_ok:
                    src = find_dll_in_package(tmp_path, pkg, dll)

                # 2. Fall back to system CUDA / cuDNN install
                if src is None:
                    print(f"  [FALLBACK] {dll}: not in pip wheel, "
                          "trying system CUDA...")
                    src = find_in_system_cuda(dll)

                # 3. Already on sys.prefix (conda with CUDA packages)
                if src is None:
                    for sp in pathlib.Path(sys.prefix).rglob(dll):
                        src = sp
                        break

                if src is None:
                    missing.append((pkg, dll))
                    print(f"  [MISSING] {dll}  (from {pkg})")
                else:
                    dst = output_dir / dll
                    shutil.copy2(src, dst)
                    size_kb = src.stat().st_size // 1024
                    print(f"  [OK] {dll}  ({size_kb:,} KB)  source={src}")

    # TRT DLLs (from TRT_LIB_PATH if set)
    trt_copied = copy_trt_dlls(output_dir)
    if not trt_copied:
        print("\n  [INFO] TRT DLLs not copied (TRT_LIB_PATH not set).")
        print("         To enable TensorRT EP, set TRT_LIB_PATH=<TRT root>\\lib")
        print("         and re-run this script, or copy ALL DLLs from lib\\ manually.")
        print("         Also copy zlibwapi.dll (required by TRT, in TRT lib\\).")

    if missing:
        print(f"\n[WARNING] {len(missing)} CUDA DLL(s) not collected:")
        for pkg, dll in missing:
            print(f"  {dll}  (expected from {pkg})")
        print("The GPU build will fall back to CPU if these DLLs are absent.")
        print("Install CUDA 13.x: https://developer.nvidia.com/cuda-downloads")
    else:
        print(f"\n[OK] CUDA 13 DLLs collected -> {output_dir}")
        print("These DLLs are redistributable under the CUDA EULA and cuDNN EULA.")

    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Collect CUDA 13 / cuDNN 9 runtime DLLs for 3Deflatten GPU build",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Directory to copy DLLs into (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Skip confirmation prompt (for CI use)",
    )
    args = parser.parse_args()

    ok = collect(pathlib.Path(args.output).resolve(), no_confirm=args.no_confirm)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
