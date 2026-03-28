"""
3Deflatten -- Collect GPU Runtime DLLs for the unified build
           (CUDA 12.9.1 / cuDNN 9.20.0 / TRT 10.16.0.72).

This script downloads the matching NVIDIA installers / zip and extracts ONLY
the DLLs needed, so users do NOT need system-wide installs.

Sources (official NVIDIA URLs):
  CUDA 12.9.1   : https://developer.download.nvidia.com/compute/cuda/12.9.1/
                   local_installers/cuda_12.9.1_<ver>_windows.exe
  cuDNN 9.20.0  : https://developer.download.nvidia.com/compute/cudnn/9.20.0/
                   local_installers/cudnn_9.20.0_windows.exe
  TRT 10.16.0.72: user must supply the zip (NVIDIA auth-wall prevents direct
                   download; get it from https://developer.nvidia.com/tensorrt)

DLLs collected:
  CUDA 12:  cudart64_12, cublas64_12, cublasLt64_12, cufft64_12,
            nvJitLink_120_0, cusolver64_11, curand64_10
  cuDNN 9:  cudnn64_9, cudnn_ops64_9, cudnn_adv64_9, cudnn_cnn64_9,
            cudnn_graph64_9, cudnn_engines_runtime_compiled64_9,
            cudnn_engines_precompiled64_9, cudnn_heuristic64_9
  TRT 10.16: all *.dll from TRT lib folder (nvinfer_10.dll, nvonnxparser_10.dll, ...)

NOT bundled (always present as a driver component in System32):
  nvcuda.dll

Usage:
  python collect_runtime_dlls_cuda12.py --trt-zip TensorRT-10.16.0.72.Windows.win10.cuda-12.9.zip
  python collect_runtime_dlls_cuda12.py --trt-zip <file> --no-confirm
  python collect_runtime_dlls_cuda12.py --trt-zip <file> --output <dir>

The CUDA and cuDNN .exe installers are NSIS archives extracted via 7z.exe (found
in common install paths, or 7zr.exe is auto-downloaded from 7-zip.org).
Downloads are cached in %LOCALAPPDATA%/3Deflatten/dlcache-cuda12 to avoid
re-downloading on repeated runs (~5 GB total on first run).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
import urllib.request
import zipfile

# ── Download URLs ─────────────────────────────────────────────────────────────
# CUDA 12.9.1 — check for latest installer filename on
#   https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/
CUDA_URL  = ("https://developer.download.nvidia.com/compute/cuda/12.9.1/"
             "local_installers/cuda_12.9.1_576.02_windows.exe")

# cuDNN 9.20.0 for CUDA 12
CUDNN_URL = ("https://developer.download.nvidia.com/compute/cudnn/9.20.0/"
             "local_installers/cudnn_9.20.0_windows.exe")

# TRT 10.16.0.72: behind NVIDIA auth wall — user must download manually and
# pass via --trt-zip.  No TRT_URL constant is defined.

# ── DLL lists ─────────────────────────────────────────────────────────────────
# CUDA 12.x DLL suffix is "12" for most libs; cuFFT stays at 12,
# nvJitLink uses "120_0" (major.minor.0 without patch).
# cusolver/curand keep their CUDA-11-era suffixes for ABI compatibility.
CUDA_DLLS = [
    "cudart64_12.dll",       # CUDA runtime
    "cublas64_12.dll",       # cuBLAS
    "cublasLt64_12.dll",     # cuBLAS-Lt (lightweight)
    "cufft64_12.dll",        # cuFFT
    "nvJitLink_120_0.dll",   # JIT linker — required by ORT CUDA EP (CUDA 12.2+)
    "cusolver64_11.dll",     # cuSolver — loaded by ORT CUDA EP at startup
    "curand64_10.dll",       # cuRand   — loaded by ORT CUDA EP at startup
]

# cuDNN 9 split-library layout (same as in the cuda13 script).
CUDNN_DLLS = [
    "cudnn64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_heuristic64_9.dll",
]

# TRT 10.3+ uses _10 suffix on the main DLLs.
TRT_REQUIRED_DLLS = [
    "nvinfer_10.dll",
    "nvonnxparser_10.dll",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = pathlib.Path(__file__).resolve().parent
_candidate     = SCRIPT_DIR / "Win64_GPU"
DEFAULT_OUTPUT = _candidate if _candidate.exists() else SCRIPT_DIR.parent / "Win64_GPU"

_appdata   = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
CACHE_DIR  = pathlib.Path(_appdata) / "3Deflatten" / "dlcache-cuda12"

_7ZR_URL   = "https://www.7-zip.org/a/7zr.exe"
_7ZR_CACHE = CACHE_DIR / "7zr.exe"

_7Z_CANDIDATES = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
]


# ---------------------------------------------------------------------------
# Helpers (identical to collect_runtime_dlls_cuda13.py)
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


def download(url: str, cache_dir: pathlib.Path) -> pathlib.Path:
    filename = url.split("/")[-1]
    dest     = cache_dir / filename
    if dest.exists():
        print(f"  [CACHED] {filename}  ({dest.stat().st_size // 1_048_576} MB)")
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {filename} ...")
    print(f"  Source: {url}")
    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp), reporthook=_progress_hook)
        print()
        tmp.rename(dest)
        print(f"  Saved {filename}  ({dest.stat().st_size // 1_048_576} MB)")
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {exc}") from exc
    return dest


def _find_7z_exe() -> pathlib.Path | None:
    for p in _7Z_CANDIDATES:
        if pathlib.Path(p).exists():
            return pathlib.Path(p)
    found = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    if found:
        return pathlib.Path(found)
    if _7ZR_CACHE.exists():
        return _7ZR_CACHE
    print(f"  7-Zip not found. Downloading 7zr.exe (~1.5 MB) from {_7ZR_URL}...")
    _7ZR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _7ZR_CACHE.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(_7ZR_URL, str(tmp), reporthook=_progress_hook)
        print()
        tmp.rename(_7ZR_CACHE)
        return _7ZR_CACHE
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download 7zr.exe: {exc}") from exc


def extract_with_7z(archive: pathlib.Path, out_dir: pathlib.Path):
    import subprocess
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = _find_7z_exe()
    print(f"  Extracting {archive.name}  (using {exe.name}, may take a few minutes)...")
    result = subprocess.run(
        [str(exe), "x", str(archive), f"-o{out_dir}", "-y", "-bso0", "-bsp1"],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"7-Zip extraction failed (exit {result.returncode})")
    print(f"  Extracted to {out_dir}")


def extract_zip_file(archive: pathlib.Path, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive.name}  (this may take a minute)...")
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(str(out_dir))
    print(f"  Extracted to {out_dir}")


def _find_dll(search_root: pathlib.Path, dll_name: str) -> pathlib.Path | None:
    target = dll_name.lower()
    for p in search_root.rglob("*"):
        if p.is_file() and p.name.lower() == target:
            return p
    return None


def collect_named_dlls(src_root: pathlib.Path, dll_names: list[str],
                       out_dir: pathlib.Path, label: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    print(f"\n  {label}:")
    for name in dll_names:
        found = _find_dll(src_root, name)
        if found:
            dest = out_dir / name
            shutil.copy2(found, dest)
            kb = dest.stat().st_size // 1024
            print(f"  [OK] {name:<55} {kb:>10,} KB")
        else:
            print(f"  [MISSING] {name}")
            missing.append(name)
    return missing


def collect_all_trt_dlls(trt_root: pathlib.Path, out_dir: pathlib.Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dlls = [p for p in trt_root.rglob("*.dll")]
    lib_dlls = [p for p in all_dlls if "lib" in {part.lower() for part in p.parts}]
    if lib_dlls:
        all_dlls = lib_dlls
    n_copied = 0
    print(f"\n  TensorRT ({len(all_dlls)} DLLs):")
    for src in sorted(all_dlls, key=lambda p: p.name.lower()):
        dest = out_dir / src.name
        shutil.copy2(src, dest)
        kb = dest.stat().st_size // 1024
        print(f"  [OK] {src.name:<55} {kb:>10,} KB")
        n_copied += 1
    present = {p.name.lower() for p in all_dlls}
    missing = [d for d in TRT_REQUIRED_DLLS if d.lower() not in present]
    if missing:
        print()
        for d in missing:
            print(f"  [WARN] Key TRT DLL not found in archive: {d}")
    return missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(output_dir: pathlib.Path,
        trt_zip_path: str | None,
        no_confirm: bool) -> bool:

    print()
    print("3Deflatten -- Collect GPU Runtime DLLs "
          "(CUDA 12.9.1 + cuDNN 9.20.0 + TRT 10.16.0.72 / unified build)")
    print("=" * 85)
    print(f"Output : {output_dir}")
    print(f"Cache  : {CACHE_DIR}")
    print()
    print("Unified ORT build targets CUDA 12.9.1 / TRT 10.16.0.72.")
    print("First-run downloads (~5 GB total; cached in dlcache-cuda12/):")
    print(f"  CUDA 12.9.1  installer : {CUDA_URL.split('/')[-1]}")
    print(f"  cuDNN 9.20.0 installer : {CUDNN_URL.split('/')[-1]}")
    if trt_zip_path:
        print(f"  TRT 10.16.0.72 zip    : {trt_zip_path}")
    else:
        print("  TRT 10.16.0.72        : --trt-zip <path> REQUIRED")
        print("    Download from: https://developer.nvidia.com/tensorrt")
        print("    Expected file: TensorRT-10.16.0.72.Windows.win10.cuda-12.9.zip")
        print()
        print("  Re-run with:  python collect_runtime_dlls_cuda12.py --trt-zip <file>")
        return False
    print()
    print("Only the required DLLs will be copied; all are NVIDIA-redistributable.")
    print()

    if not no_confirm:
        ans = input("Proceed? [Y/n] ").strip().lower()
        if ans not in ("", "y", "yes"):
            print("Cancelled.")
            return False

    all_missing = []

    # ── CUDA 12.9.1 ───────────────────────────────────────────────────────────
    print("\n[1/3]  CUDA 12.9.1")
    cuda_archive = download(CUDA_URL, CACHE_DIR)
    cuda_ext     = CACHE_DIR / "cuda_extracted"
    if not cuda_ext.exists():
        extract_with_7z(cuda_archive, cuda_ext)
    missing = collect_named_dlls(cuda_ext, CUDA_DLLS, output_dir, "CUDA 12 DLLs")
    all_missing.extend(missing)

    # ── cuDNN 9.20.0 ──────────────────────────────────────────────────────────
    print("\n[2/3]  cuDNN 9.20.0")
    cudnn_archive = download(CUDNN_URL, CACHE_DIR)
    cudnn_ext     = CACHE_DIR / "cudnn_extracted"
    if not cudnn_ext.exists():
        extract_with_7z(cudnn_archive, cudnn_ext)
    missing = collect_named_dlls(cudnn_ext, CUDNN_DLLS, output_dir, "cuDNN 9 DLLs")
    all_missing.extend(missing)

    # ── TensorRT 10.16.0.72 ───────────────────────────────────────────────────
    print("\n[3/3]  TensorRT 10.16.0.72")
    trt_archive = pathlib.Path(trt_zip_path)
    if not trt_archive.exists():
        print(f"  [ERROR] TRT zip not found: {trt_archive}")
        return False
    trt_ext = CACHE_DIR / "trt_extracted"
    if not trt_ext.exists():
        extract_zip_file(trt_archive, trt_ext)
    trt_missing = collect_all_trt_dlls(trt_ext, output_dir)
    if trt_missing:
        print()
        print(f"  [WARN] {len(trt_missing)} key TRT DLL(s) not found: "
              + ", ".join(trt_missing))

    # ── Summary ───────────────────────────────────────────────────────────────
    n_dlls = sum(1 for p in output_dir.glob("*.dll"))
    print()
    print("=" * 85)
    if all_missing:
        print(f"[WARN] {len(all_missing)} DLL(s) not found in the extracted archives:")
        for d in all_missing:
            print(f"  {d}")
        print(f"\nThe archive layout may differ from expected. Inspect:")
        print(f"  {CACHE_DIR}")
    else:
        print(f"[OK] All DLLs collected -- {n_dlls} *.dll files in {output_dir}")
        print()
        print("Run  python Setup.py --check  to verify detection.")
    return len(all_missing) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Download and bundle CUDA 12.9.1 / cuDNN 9.20.0 / TRT 10.16.0.72 DLLs "
                    "for the 3Deflatten unified build",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument(
        "--trt-zip", default=None, metavar="PATH", required=True,
        help="Path to locally downloaded TensorRT-10.16.0.72 zip "
             "(required — TRT is behind NVIDIA auth wall)")
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Skip the 'Proceed?' prompt (for CI / scripted use)")
    args = parser.parse_args()

    ok = run(
        output_dir=pathlib.Path(args.output).resolve(),
        trt_zip_path=args.trt_zip,
        no_confirm=args.no_confirm,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
