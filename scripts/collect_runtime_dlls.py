#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
r"""
Collect CUDA 13.x / cuDNN 9 runtime DLLs into a target directory so that
the GPU build of 3Deflatten works without a system-wide CUDA/cuDNN install.

ORT 1.24.x GPU build requires CUDA 13.x at runtime.

Required DLLs (bundled by this script)
----------------------------------------
  cudart64_13.dll        - CUDA 13 runtime
  cublas64_13.dll        - cuBLAS 13
  cublasLt64_13.dll      - cuBLAS-Lt 13
  cufft64_12.dll         - cuFFT 12 (ABI name used in CUDA 13 toolkit)
  cudnn64_9.dll          - cuDNN 9 (works with CUDA 13)

Usage:
  python scripts\collect_runtime_dlls.py
  python scripts\collect_runtime_dlls.py --output build\Win64_GPU
  python scripts/collect_runtime_dlls.py --output $OUTPUT_DIR --no-confirm
"""

import argparse
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

# pip package -> list of DLL names it should provide on Windows
PACKAGES = {
    "nvidia-cuda-runtime-cu13":  ["cudart64_13.dll"],
    "nvidia-cublas-cu13":        ["cublas64_13.dll", "cublasLt64_13.dll"],
    "nvidia-cufft-cu13":         ["cufft64_12.dll"],
    "nvidia-cudnn-cu13":         ["cudnn64_9.dll"],
}

# Typical install sub-paths inside site-packages\nvidia\<n>\
SEARCH_SUBDIRS = ["bin", "lib", ""]


def find_dll_in_package(site_packages: pathlib.Path,
                        pkg_pip_name: str,
                        dll_name: str) -> pathlib.Path | None:
    short = (pkg_pip_name
             .replace("nvidia-", "")
             .replace("-cu13", "")
             .replace("-cu12", "")
             .replace("-", "_"))
    candidates = [
        site_packages / "nvidia" / short,
        site_packages / "nvidia" / short.replace("_cu13", ""),
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


def install_packages(packages: list[str], target_dir: pathlib.Path) -> None:
    print(f"Installing pip packages into temp dir: {target_dir}")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--quiet",
        "--target", str(target_dir),
        "--only-binary=:all:",
    ] + packages
    subprocess.check_call(cmd)


def collect(output_dir: pathlib.Path, no_confirm: bool = False) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n3Deflatten - Collect CUDA 13 Runtime DLLs")
    print("=" * 50)
    print(f"Output: {output_dir}\n")
    print("Packages to install (CUDA 13 / cuDNN 9):")
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
                    print(f"  [OK] {dll}  ({size_kb:,} KB)  ->  {dst}")

    if missing:
        print(f"\n[WARNING] {len(missing)} DLL(s) not found:")
        for pkg, dll in missing:
            print(f"  {dll}  (expected from {pkg})")
        print("The GPU build will still work if these DLLs are on the system PATH.")
    else:
        print(f"\n[OK] All DLLs collected -> {output_dir}")
        print("\nThese DLLs are redistributable under the CUDA EULA and cuDNN EULA.")

    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Collect CUDA 13 / cuDNN 9 runtime DLLs for 3Deflatten GPU build",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", default=".",
        help="Directory to copy DLLs into (default: current directory)")
    parser.add_argument("--no-confirm", action="store_true",
        help="Skip confirmation prompt (for CI use)")
    args = parser.parse_args()

    ok = collect(pathlib.Path(args.output).resolve(), no_confirm=args.no_confirm)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
