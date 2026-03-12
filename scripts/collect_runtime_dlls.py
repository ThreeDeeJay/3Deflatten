#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Collect ALL runtime DLLs required for the Win64_GPU build of 3Deflatten.

ORT 1.18.1 GPU build was compiled against CUDA 11.8.  Using CUDA 11.7 DLLs
causes error=1114 because the CUDA runtime checks its own version at DllMain
init time (11.7 < 11.8 required → DllMain returns FALSE).

Sources (official NVIDIA URLs):
  CUDA 11.8.0   : https://developer.download.nvidia.com/compute/cuda/11.8.0/
                   local_installers/cuda_11.8.0_522.06_windows.exe
  cuDNN 8.9.3   : https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/
                   windows-x86_64/cudnn-windows-x86_64-8.9.3.28_cuda11-archive.zip
  TRT 10.0.0.6  : https://developer.nvidia.com/downloads/compute/machine-learning/
                   tensorrt/10.0.0/zip/TensorRT-10.0.0.6.Windows10.win10.cuda-11.8.zip

DLLs collected:
  CUDA 11:  cudart64_110, cublas64_11, cublasLt64_11, cufft64_10,
            cusolver64_11, curand64_10
  cuDNN 8:  cudnn64_8, cudnn_ops_infer64_8, cudnn_ops_train64_8,
            cudnn_cnn_infer64_8, cudnn_cnn_train64_8,
            cudnn_adv_infer64_8, cudnn_adv_train64_8
  TRT 10.0: all *.dll from TRT lib folder (nvinfer.dll, nvonnxparser.dll, ...)
            NOTE: TRT 10.0.x uses plain names -- no _10 suffix (that's TRT 10.3+).

NOT bundled (always present as a driver component in System32):
  nvcuda.dll

Usage:
  python collect_runtime_dlls.py                     # download everything
  python collect_runtime_dlls.py --trt-zip <file>    # use local TRT zip
  python collect_runtime_dlls.py --no-confirm        # CI / unattended

CUDA 11.x: the .exe installer is an NSIS archive extracted via 7z.exe.
cuDNN 8.x: the redist package is a plain .zip extracted with Python's zipfile.
TRT:       the .zip is extracted with Python's zipfile.
7z.exe is found in common install paths, or 7zr.exe is auto-downloaded from
7-zip.org if not installed.
Downloads are cached in %%LOCALAPPDATA%%/3Deflatten/dlcache (~4 GB on first run).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile

# Force UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
# After packaging, this script lives at the release root next to Win64_GPU/.
# When run from scripts/ during development, go up one level.
_candidate     = SCRIPT_DIR / "Win64_GPU"
DEFAULT_OUTPUT = _candidate if _candidate.exists() else SCRIPT_DIR.parent / "Win64_GPU"

# CUDA 11.8 (NSIS .exe installer → extract with 7z.exe)
# ORT 1.18.1 GPU build was compiled against CUDA 11.8. Using 11.7 DLLs causes
# error=1114 because the CUDA runtime version-checks itself: 11.7 < 11.8 → fail.
CUDA_URL  = ("https://developer.download.nvidia.com/compute/cuda/11.8.0/"
             "local_installers/cuda_11.8.0_522.06_windows.exe")
# cuDNN 8.9.3 for CUDA 11 (plain .zip → extract with zipfile)
CUDNN_URL = ("https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/"
             "windows-x86_64/cudnn-windows-x86_64-8.9.3.28_cuda11-archive.zip")
# TRT 10.0.0.6 for CUDA 11.8 (.zip → extract with zipfile)
# NOTE: TRT 10.0.x uses plain DLL names (nvinfer.dll, not nvinfer_10.dll).
#       The _10 suffix was introduced in TRT 10.3+.
TRT_URL   = ("https://developer.nvidia.com/downloads/compute/machine-learning/"
             "tensorrt/10.0.0/zip/TensorRT-10.0.0.6.Windows10.win10.cuda-11.8.zip")

# DLLs to copy from the CUDA 11.8 installer.
# NOTE: cudart uses "110" suffix (= CUDA 11.0 API level, same across all 11.x).
#       Other CUDA 11 DLLs use "_11" (just the major version).
#       nvJitLink was introduced in CUDA 12 -- not present in CUDA 11.
# cusolver64_11 and curand64_10 are loaded by ORT CUDA EP at init time (not lazy).
CUDA_DLLS = [
    "cudart64_110.dll",    # runtime  -- "110" = API 11.0, used for all CUDA 11.x
    "cublas64_11.dll",
    "cublasLt64_11.dll",
    "cufft64_10.dll",      # cuFFT API version stays at 10 inside CUDA 11 toolkit
    "cusolver64_11.dll",   # cuSolver -- loaded by ORT CUDA EP at startup
    "curand64_10.dll",     # cuRand   -- loaded by ORT CUDA EP at startup
]

# DLLs to copy from the cuDNN 8.x archive
# cuDNN 8 uses a monolithic cudnn64_8.dll plus split infer/train DLLs.
CUDNN_DLLS = [
    "cudnn64_8.dll",
    "cudnn_ops_infer64_8.dll",
    "cudnn_ops_train64_8.dll",
    "cudnn_cnn_infer64_8.dll",
    "cudnn_cnn_train64_8.dll",
    "cudnn_adv_infer64_8.dll",
    "cudnn_adv_train64_8.dll",
]

# Key TRT DLLs -- we copy ALL *.dll from the TRT archive but warn if absent.
# TRT 10.0.x DLL names have NO version suffix (nvinfer.dll, not nvinfer_10.dll).
# The _10 suffix was introduced in TRT 10.3+.
TRT_REQUIRED_DLLS = [
    "nvinfer.dll",
    "nvonnxparser.dll",
]

# Download cache (avoids re-downloading ~5 GB on repeated runs)
_appdata  = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
CACHE_DIR = pathlib.Path(_appdata) / "3Deflatten" / "dlcache"


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _progress_hook(count, block_size, total_size):
    done = count * block_size
    if total_size > 0:
        pct     = min(100, done * 100 // total_size)
        mb_done = done / 1_048_576
        mb_tot  = total_size / 1_048_576
        bar     = "#" * (pct // 5)
        print(f"\r  [{bar:<20}] {pct:3d}%  {mb_done:7.1f}/{mb_tot:.1f} MB",
              end="", flush=True)
    else:
        mb = done / 1_048_576
        print(f"\r  {mb:.1f} MB downloaded...", end="", flush=True)


def download(url: str, cache_dir: pathlib.Path) -> pathlib.Path:
    """Download *url* into *cache_dir* (skip if already cached).  Returns path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = url.split("/")[-1]
    out   = cache_dir / fname
    if out.exists():
        mb = out.stat().st_size / 1_048_576
        print(f"  [CACHED] {fname}  ({mb:.0f} MB)")
        return out
    print(f"  Downloading {fname} ...")
    print(f"  Source: {url}")
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, str(tmp), reporthook=_progress_hook)
        print()
        tmp.rename(out)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {fname}: {exc}") from exc
    mb = out.stat().st_size / 1_048_576
    print(f"  Saved {fname}  ({mb:.0f} MB)")
    return out


# ---------------------------------------------------------------------------
# 7-Zip helper  (CUDA and cuDNN installers are NSIS archives; py7zr cannot
# handle them, but the real 7z.exe can.  We try common install paths first,
# then download the tiny 7zr.exe standalone binary from 7-zip.org if needed.)
# ---------------------------------------------------------------------------

# 7zr.exe standalone (no DLL deps) from the official 7-zip.org sourceforge mirror
_7ZR_URL    = "https://www.7-zip.org/a/7zr.exe"
_7ZR_CACHE  = CACHE_DIR / "7zr.exe"

# Common 7-Zip install locations on Windows
_7Z_CANDIDATES = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
]


def _find_7z_exe() -> pathlib.Path | None:
    """Return path to 7z.exe / 7zr.exe, downloading 7zr.exe if necessary."""
    # 1. Check common install paths
    for p in _7Z_CANDIDATES:
        if pathlib.Path(p).exists():
            return pathlib.Path(p)
    # 2. Check PATH
    found = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    if found:
        return pathlib.Path(found)
    # 3. Cached 7zr.exe from a previous run
    if _7ZR_CACHE.exists():
        return _7ZR_CACHE
    # 4. Download 7zr.exe (~1.5 MB standalone, no install needed)
    print("  7-Zip not found on this system.  Downloading 7zr.exe (~1.5 MB)...")
    print(f"  Source: {_7ZR_URL}")
    _7ZR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _7ZR_CACHE.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(_7ZR_URL, str(tmp), reporthook=_progress_hook)
        print()
        tmp.rename(_7ZR_CACHE)
        print(f"  Saved 7zr.exe to {_7ZR_CACHE}")
        return _7ZR_CACHE
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download 7zr.exe: {exc}") from exc


def extract_with_7z(archive: pathlib.Path, out_dir: pathlib.Path):
    """Extract *archive* (any format 7-Zip supports) to *out_dir* using 7z.exe."""
    import subprocess
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = _find_7z_exe()
    print(f"  Extracting {archive.name}  (using {exe.name}, may take a few minutes)...")
    result = subprocess.run(
        [str(exe), "x", str(archive), f"-o{out_dir}", "-y", "-bso0", "-bsp1"],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"7-Zip extraction failed (exit {result.returncode}) for {archive.name}"
        )
    print(f"  Extracted to {out_dir}")


def extract_zip_file(archive: pathlib.Path, out_dir: pathlib.Path):
    """Extract a standard zip file using Python's built-in zipfile module."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive.name}  (this may take a minute)...")
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(str(out_dir))
    print(f"  Extracted to {out_dir}")


# ---------------------------------------------------------------------------
# DLL collection helpers
# ---------------------------------------------------------------------------

def _find_dll(search_root: pathlib.Path, dll_name: str) -> pathlib.Path | None:
    """Recursively find the first file matching *dll_name* (case-insensitive)."""
    target = dll_name.lower()
    for p in search_root.rglob("*"):
        if p.is_file() and p.name.lower() == target:
            return p
    return None


def collect_named_dlls(search_root: pathlib.Path,
                       dll_names: list,
                       output_dir: pathlib.Path,
                       label: str) -> list:
    """Find each DLL name in *search_root* and copy to *output_dir*.
    Returns list of names that could not be found."""
    missing = []
    print(f"\n  {label}:")
    for dll in dll_names:
        src = _find_dll(search_root, dll)
        if src is None:
            print(f"  [MISSING] {dll}")
            missing.append(dll)
            continue
        dst = output_dir / dll
        shutil.copy2(str(src), str(dst))
        kb = src.stat().st_size // 1024
        print(f"  [OK] {dll:<56} {kb:>6,} KB")
    return missing


def collect_all_trt_dlls(search_root: pathlib.Path,
                          output_dir: pathlib.Path) -> list:
    """Copy every *.dll found under *search_root* to *output_dir*.
    Returns list of key DLL names that were absent."""
    all_dlls = list(search_root.rglob("*.dll"))
    if not all_dlls:
        print("  [WARN] No DLLs found in TRT archive.")
        return TRT_REQUIRED_DLLS

    # De-duplicate by name, preferring the shallowest copy
    seen: dict = {}
    for p in all_dlls:
        key = p.name.lower()
        if key not in seen or len(p.parts) < len(seen[key].parts):
            seen[key] = p

    print(f"\n  TensorRT ({len(seen)} DLLs):")
    for name_lower in sorted(seen):
        src = seen[name_lower]
        dst = output_dir / src.name
        shutil.copy2(str(src), str(dst))
        kb = src.stat().st_size // 1024
        print(f"  [OK] {src.name:<56} {kb:>6,} KB")

    copied_lower = set(seen.keys())
    return [d for d in TRT_REQUIRED_DLLS if d.lower() not in copied_lower]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(output_dir: pathlib.Path,
        trt_zip_path,
        no_confirm: bool) -> bool:

    output_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("3Deflatten -- Collect GPU Runtime DLLs (CUDA 11.8 + cuDNN 8 + TRT 10.0)")
    print("=" * 70)
    print(f"Output : {output_dir}")
    print(f"Cache  : {CACHE_DIR}")
    print()
    print("ORT 1.18.1 was compiled against CUDA 11.8 / TRT 10.0.0.6.")
    print("Using CUDA 11.7 DLLs causes error=1114 (CUDA runtime version check fails).")
    print("First-run downloads (~4 GB total; cached for subsequent runs):")
    print(f"  CUDA 11.8.0  installer  : {CUDA_URL.split('/')[-1]}")
    print(f"  cuDNN 8.9.3  zip        : {CUDNN_URL.split('/')[-1]}")
    if trt_zip_path:
        print(f"  TRT 10.0.0.6 zip (local): {trt_zip_path}")
    else:
        print(f"  TRT 10.0.0.6 zip        : {TRT_URL.split('/')[-1]}")
    print()
    print("Only the required DLLs will be copied; all are NVIDIA-redistributable.")
    print()

    if not no_confirm:
        ans = input("Proceed? [Y/n] ").strip().lower()
        if ans not in ("", "y", "yes"):
            print("Cancelled.")
            return False

    all_missing = []

    # ── CUDA 11.8 (NSIS installer → 7z.exe) ─────────────────────────────────
    print("\n[1/3]  CUDA 11.8.0")
    cuda_archive = download(CUDA_URL, CACHE_DIR)
    cuda_ext     = CACHE_DIR / "cuda_extracted"
    if not cuda_ext.exists():
        extract_with_7z(cuda_archive, cuda_ext)
    missing = collect_named_dlls(cuda_ext, CUDA_DLLS, output_dir,
                                 "CUDA 11 DLLs")
    all_missing.extend(missing)

    # ── cuDNN 8.9.3 (plain zip → zipfile) ────────────────────────────────────
    print("\n[2/3]  cuDNN 8.9.3")
    cudnn_archive = download(CUDNN_URL, CACHE_DIR)
    cudnn_ext     = CACHE_DIR / "cudnn_extracted"
    if not cudnn_ext.exists():
        extract_zip_file(cudnn_archive, cudnn_ext)
    missing = collect_named_dlls(cudnn_ext, CUDNN_DLLS, output_dir,
                                 "cuDNN 8 DLLs")
    all_missing.extend(missing)

    # ── TensorRT 10.0.0.6 (plain zip → zipfile) ──────────────────────────────
    print("\n[3/3]  TensorRT 10.0.0.6")
    if trt_zip_path:
        trt_archive = pathlib.Path(trt_zip_path)
        if not trt_archive.exists():
            print(f"  [ERROR] TRT zip not found: {trt_archive}")
            return False
    else:
        trt_archive = download(TRT_URL, CACHE_DIR)
    trt_ext = CACHE_DIR / "trt_extracted"
    if not trt_ext.exists():
        extract_zip_file(trt_archive, trt_ext)
    trt_missing = collect_all_trt_dlls(trt_ext, output_dir)
    if trt_missing:
        print()
        for dll in trt_missing:
            print(f"  [WARN] Key TRT DLL not found in archive: {dll}")
        all_missing.extend(trt_missing)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    if all_missing:
        print(f"[WARN] {len(all_missing)} DLL(s) not found in the extracted archives:")
        for d in all_missing:
            print(f"  {d}")
        print()
        print("The archive layout may differ from what was expected.")
        print("Inspect the extracted contents at:")
        print(f"  {CACHE_DIR}")
        print("To clear the cache and retry: delete that folder.")
        return False

    count = sum(1 for _ in output_dir.glob("*.dll"))
    print(f"[OK] All DLLs collected -- {count} *.dll files in {output_dir}")
    print()
    print("Run  python Setup.py --check  to verify detection.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and bundle CUDA 11.8 / cuDNN 8 / TRT 10.0.0.6 DLLs "
                    "for the 3Deflatten GPU build (ORT 1.18.1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument(
        "--trt-zip", default=None, metavar="PATH",
        help="Path to a locally downloaded TRT zip (skips TRT download)")
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Skip the confirmation prompt (CI / unattended use)")
    args = parser.parse_args()

    ok = run(
        output_dir=pathlib.Path(args.output).resolve(),
        trt_zip_path=args.trt_zip,
        no_confirm=args.no_confirm,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
