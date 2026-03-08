#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
r"""
Collect CUDA 12.x / cuDNN 9 runtime DLLs into a target directory so that
the GPU build of 3Deflatten works without a system-wide CUDA/cuDNN install.

DLLs are obtained from the official NVIDIA pip wheels (redistributable).
The wheels are downloaded to a temporary location; only the .dll files are
copied to the target directory.

Usage (local, run once after building)
---------------------------------------
  python scripts\collect_runtime_dlls.py
  python scripts\collect_runtime_dlls.py --output build\Win64_GPU

Usage (CI / automated)
------------------------
  python scripts/collect_runtime_dlls.py --output $OUTPUT_DIR --no-confirm

Required DLLs (bundled by this script)
----------------------------------------
  cudart64_12.dll        – CUDA 12 runtime
  cublas64_12.dll        – cuBLAS
  cublasLt64_12.dll      – cuBLAS Lt
  cufft64_11.dll         – cuFFT  (ORT 1.21 still uses the CUDA 11 ABI name)
  cudnn64_9.dll          – cuDNN 9

Note: onnxruntime_providers_cuda.dll / onnxruntime_providers_tensorrt.dll
are already included in the ORT GPU zip that CMake downloads at build time.
TensorRT nvinfer_10.dll / nvonnxparser_10.dll are NOT bundled here because
they are large (~2 GB) and redistributable terms are more restrictive.
Users who want TRT acceleration must install TensorRT 10.x separately.
"""

import argparse
import glob
import pathlib
import shutil
import subprocess
import sys
import tempfile

# pip package → list of DLL names it should provide on Windows
PACKAGES = {
    "nvidia-cuda-runtime-cu12":  ["cudart64_12.dll"],
    "nvidia-cublas-cu12":        ["cublas64_12.dll", "cublasLt64_12.dll"],
    "nvidia-cufft-cu12":         ["cufft64_11.dll"],
    "nvidia-cudnn-cu12":         ["cudnn64_9.dll"],
}

# Typical install sub-paths inside site-packages\nvidia\<name>\
SEARCH_SUBDIRS = ["bin", "lib", ""]


def find_dll_in_package(site_packages: pathlib.Path,
                        pkg_pip_name: str,
                        dll_name: str) -> pathlib.Path | None:
    """Search for dll_name inside the nvidia pip package directory."""
    # pip installs nvidia packages as  site-packages/nvidia/<short_name>/
    # e.g. nvidia-cuda-runtime-cu12 → nvidia/cuda_runtime/
    short = pkg_pip_name.replace("nvidia-", "").replace("-cu12", "").replace("-", "_")
    candidates = [
        site_packages / "nvidia" / short,
        site_packages / "nvidia" / short.replace("_cu12", ""),
    ]
    for base in candidates:
        if not base.exists():
            continue
        for sub in SEARCH_SUBDIRS:
            p = base / sub / dll_name if sub else base / dll_name
            if p.exists():
                return p
        # Recursive fallback
        for hit in base.rglob(dll_name):
            return hit
    # Broader search across all nvidia packages
    for hit in site_packages.rglob(dll_name):
        return hit
    return None


def install_packages(packages: list[str], target_dir: pathlib.Path) -> None:
    """pip install packages into target_dir (isolated, doesn't touch system)."""
    print(f"Installing pip packages into temp dir: {target_dir}")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--quiet",
        "--target", str(target_dir),
        "--only-binary=:all:",   # wheels only, no source builds
    ] + packages
    subprocess.check_call(cmd)


def collect(output_dir: pathlib.Path, no_confirm: bool = False) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n3Deflatten – Collect CUDA Runtime DLLs")
    print("=" * 50)
    print(f"Output: {output_dir}\n")
    print("Packages to install:")
    for pkg in PACKAGES:
        print(f"  {pkg}")
    print()

    if not no_confirm:
        ans = input("Proceed? [Y/n] ").strip().lower()
        if ans not in ("", "y", "yes"):
            print("Cancelled.")
            return False

    with tempfile.TemporaryDirectory(prefix="3df_cuda_") as tmp:
        tmp_path = pathlib.Path(tmp)

        try:
            install_packages(list(PACKAGES.keys()), tmp_path)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] pip install failed: {e}")
            print("Make sure you are connected to the internet.")
            return False

        missing = []
        for pkg, dlls in PACKAGES.items():
            for dll in dlls:
                src = find_dll_in_package(tmp_path, pkg, dll)
                if src is None:
                    # Try system path as well
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
                    print(f"  [OK] {dll}  ({size_kb:,} KB)  →  {dst}")

    if missing:
        print(f"\n[WARNING] {len(missing)} DLL(s) not found:")
        for pkg, dll in missing:
            print(f"  {dll}  (expected from {pkg})")
        print("The GPU build will still work if these DLLs are on the system")
        print("or if 3Deflatten's auto-discovery finds them at runtime.")
    else:
        print(f"\n[OK] All DLLs collected → {output_dir}")
        print("\nThese DLLs are redistributable under the CUDA EULA and cuDNN EULA.")
        print("Distribute them alongside 3Deflatten_x64.ax in your release package.")

    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Collect CUDA 12 / cuDNN 9 runtime DLLs for 3Deflatten GPU build",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", default=".",
        help="Directory to copy DLLs into (default: current directory)",
    )
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Skip confirmation prompt (for CI use)",
    )
    args = parser.parse_args()

    ok = collect(pathlib.Path(args.output).resolve(),
                 no_confirm=args.no_confirm)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
