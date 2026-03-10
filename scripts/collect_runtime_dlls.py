#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
r"""
Collect ALL runtime DLLs required for the Win64_GPU build of 3Deflatten
so that users do NOT need system-wide CUDA/cuDNN/TensorRT installs.

ORT 1.24.3 gpu_cuda13 build requires CUDA 13.x at runtime.
All DLLs listed below are redistributable under their respective EULAs.

── What this script collects ────────────────────────────────────────────
  From CUDA 13 pip wheels (nvidia-*-cu13):
    cudart64_13.dll          CUDA 13 runtime
    cublas64_13.dll          cuBLAS 13
    cublasLt64_13.dll        cuBLAS-Lt 13
    cufft64_12.dll           cuFFT  (API ver 12, ships in CUDA 13 toolkit)
    nvJitLink_130_0.dll      CUDA JIT-linker (required by ORT CUDA EP)

  From cuDNN 9 pip wheel (nvidia-cudnn-cu13):
    cudnn64_9.dll                            main library
    cudnn_ops64_9.dll                        ops
    cudnn_adv64_9.dll                        advanced
    cudnn_cnn64_9.dll                        CNN
    cudnn_graph64_9.dll                      graph
    cudnn_engines_runtime_compiled64_9.dll   engines
    cudnn_engines_precompiled64_9.dll        precompiled engines
    cudnn_heuristic64_9.dll                  heuristics

  From TensorRT 10.15.x lib\ (set TRT_LIB_PATH env var, or pass --trt-lib):
    nvinfer_10.dll
    nvonnxparser_10.dll
    nvinfer_builder_resource_10.dll
    nvinfer_plugin_10.dll
    zlibwapi.dll
    (all other *.dll in lib\)

── NOT bundled (always in System32, never redistributed) ────────────────
    nvcuda.dll  -- NVIDIA driver component

── Usage ────────────────────────────────────────────────────────────────
  Local (from the release root):
    python collect_runtime_dlls.py
    python collect_runtime_dlls.py --trt-lib "C:\TRT\lib"

  CI (from repo root):
    python scripts/collect_runtime_dlls.py \
        --output build\Win64_GPU \
        --trt-lib <path> \
        --no-confirm
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

# Force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "Win64_GPU"

# ── pip packages → DLL names ─────────────────────────────────────────────────
CUDA_PACKAGES = {
    "nvidia-cuda-runtime-cu13": ["cudart64_13.dll"],
    "nvidia-cublas-cu13":       ["cublas64_13.dll", "cublasLt64_13.dll"],
    "nvidia-cufft-cu13":        ["cufft64_12.dll"],
    "nvidia-nvjitlink-cu13":    ["nvJitLink_130_0.dll"],
}

CUDNN_PACKAGES = {
    "nvidia-cudnn-cu13": [
        "cudnn64_9.dll",
        "cudnn_ops64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn_graph64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_heuristic64_9.dll",
    ],
}

# Key TRT DLLs we require (we copy ALL lib\ DLLs, but warn if these are absent)
TRT_KEY_DLLS = [
    "nvinfer_10.dll",
    "nvonnxparser_10.dll",
    "nvinfer_builder_resource_10.dll",
    "nvinfer_plugin_10.dll",
    "zlibwapi.dll",
]

SEARCH_SUBDIRS = ["bin", "lib", ""]


# ── Helpers ───────────────────────────────────────────────────────────────────

def install_packages(packages, target_dir):
    print(f"  pip install {' '.join(packages)}")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--quiet", "--target", str(target_dir), "--only-binary=:all:",
    ] + packages
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] pip install failed: {e}")
        return False


def find_in_pip_target(target_dir, pkg_name, dll_name):
    """Search for dll_name inside an nvidia pip package unpacked into target_dir."""
    short = (pkg_name
             .replace("nvidia-", "")
             .replace("-cu13", "").replace("-cu12", "")
             .replace("-", "_"))
    bases = [
        target_dir / "nvidia" / short,
        target_dir / "nvidia" / short.split("_cu")[0],
    ]
    for base in bases:
        if not base.exists():
            continue
        for sub in SEARCH_SUBDIRS:
            p = (base / sub / dll_name) if sub else (base / dll_name)
            if p.exists():
                return p
        for hit in base.rglob(dll_name):
            return hit
    for hit in target_dir.rglob(dll_name):
        return hit
    return None


def find_in_system(dll_name):
    """Fall back to system CUDA/cuDNN install."""
    cuda_vars = [
        "CUDA_PATH_V13_2", "CUDA_PATH_V13_1", "CUDA_PATH_V13_0",
        "CUDA_PATH_V12_9", "CUDA_PATH",
    ]
    roots = []
    for var in cuda_vars:
        v = os.environ.get(var, "").strip()
        if v:
            roots.append(pathlib.Path(v))
            break
    roots.append(pathlib.Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"))
    cudnn_env = os.environ.get("CUDNN_PATH", "").strip()
    if cudnn_env:
        roots.append(pathlib.Path(cudnn_env))
    roots.append(pathlib.Path(r"C:\Program Files\NVIDIA\CUDNN"))
    for root in roots:
        if root.exists():
            for hit in root.rglob(dll_name):
                return hit
    return None


def collect_dll(pkg_name, dll_name, target_dir, pip_ok, output_dir):
    """Try pip → system fallback. Returns True if found and copied."""
    src = None
    if pip_ok:
        src = find_in_pip_target(target_dir, pkg_name, dll_name)
    if src is None:
        src = find_in_system(dll_name)
    if src is None:
        for hit in pathlib.Path(sys.prefix).rglob(dll_name):
            src = hit
            break
    if src is None:
        return False
    dst = output_dir / dll_name
    shutil.copy2(src, dst)
    kb = src.stat().st_size // 1024
    print(f"  [OK] {dll_name:<52} {kb:>6,} KB  ← {src.parent}")
    return True


def copy_trt_dlls(trt_lib_path, output_dir):
    """Copy ALL DLLs from TRT lib\ into output_dir. Returns list of copied names."""
    if not trt_lib_path:
        return []
    trt = pathlib.Path(trt_lib_path)
    if not trt.exists():
        print(f"  [SKIP] TRT lib path not found: {trt}")
        return []

    all_dlls = sorted(set(list(trt.rglob("*.dll")) + list(trt.rglob("*.DLL"))))
    if not all_dlls:
        print(f"  [SKIP] No DLLs found in {trt}")
        return []

    print(f"\n  Copying ALL TRT DLLs from {trt}:")
    copied = []
    for src in all_dlls:
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        kb = src.stat().st_size // 1024
        print(f"  [OK] {src.name:<52} {kb:>6,} KB")
        copied.append(src.name)

    missing_key = [d for d in TRT_KEY_DLLS
                   if d.lower() not in {c.lower() for c in copied}]
    if missing_key:
        print()
        for dll in missing_key:
            print(f"  [WARN] Key TRT DLL missing: {dll}")
            if dll == "zlibwapi.dll":
                print("         zlibwapi.dll is required by TRT.  It should be in the TRT lib\\ folder.")
                print("         Download: https://www.dll-files.com/zlibwapi.dll.html")
    return copied


# ── Main ──────────────────────────────────────────────────────────────────────

def collect(output_dir, trt_lib_path=None, no_confirm=False):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pkgs = {**CUDA_PACKAGES, **CUDNN_PACKAGES}

    print("\n3Deflatten — Collect GPU Runtime DLLs (CUDA 13 + cuDNN 9 + TRT 10.15)")
    print("=" * 70)
    print(f"Output: {output_dir}\n")
    print("ORT 1.24.3 gpu_cuda13 requires CUDA 13.x at runtime.")
    print("All CUDA/cuDNN DLLs below are redistributable under NVIDIA EULAs.\n")
    print("CUDA 13 packages:")
    for pkg, dlls in CUDA_PACKAGES.items():
        print(f"  {pkg}  →  {', '.join(dlls)}")
    print("cuDNN 9 package:")
    for pkg, dlls in CUDNN_PACKAGES.items():
        print(f"  {pkg}  →  {len(dlls)} DLLs")
    if trt_lib_path:
        print(f"TRT lib path: {trt_lib_path}")
    else:
        print("TRT: not provided (pass --trt-lib or set TRT_LIB_PATH env var)")
    print()

    if not no_confirm:
        ans = input("Proceed? [Y/n] ").strip().lower()
        if ans not in ("", "y", "yes"):
            print("Cancelled.")
            return False

    missing = []

    with tempfile.TemporaryDirectory(prefix="3df_gpu_") as tmp:
        tmp_path = pathlib.Path(tmp)
        pip_ok = install_packages(list(all_pkgs.keys()), tmp_path)

        print("\n  Collecting CUDA 13 DLLs:")
        for pkg, dlls in all_pkgs.items():
            for dll in dlls:
                if not collect_dll(pkg, dll, tmp_path, pip_ok, output_dir):
                    print(f"  [MISSING] {dll}  (from {pkg})")
                    missing.append((pkg, dll))

    # TRT DLLs
    trt_path = trt_lib_path or os.environ.get("TRT_LIB_PATH", "").strip()
    trt_copied = copy_trt_dlls(trt_path, output_dir)
    if not trt_copied:
        print("\n  [INFO] TRT DLLs not copied — no TRT lib path provided.")
        print("         Pass --trt-lib <TRT_root>\\lib  or set TRT_LIB_PATH.")
        print("         Required TRT version: 10.15.x (CUDA 13 build).")
        print("         Download: https://developer.nvidia.com/tensorrt")

    print()
    if missing:
        print(f"[WARN] {len(missing)} CUDA/cuDNN DLL(s) not collected:")
        for pkg, dll in missing:
            print(f"  {dll}  (from {pkg})")
        print("Install CUDA 13: https://developer.nvidia.com/cuda-downloads")
        print("These DLLs are required for CUDA EP; without them ORT falls back to CPU.")
    else:
        print(f"[OK] All CUDA 13 + cuDNN 9 DLLs collected → {output_dir}")

    if trt_copied:
        print(f"[OK] TRT DLLs copied ({len(trt_copied)} files)")

    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Collect CUDA 13 / cuDNN 9 / TRT 10.15 DLLs for 3Deflatten GPU build",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--trt-lib", default=None,
                        help=r"Path to TensorRT lib\ folder (e.g. C:\TRT\TensorRT-10.15.x\lib)")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip confirmation prompt (CI use)")
    args = parser.parse_args()

    ok = collect(
        pathlib.Path(args.output).resolve(),
        trt_lib_path=args.trt_lib,
        no_confirm=args.no_confirm,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
