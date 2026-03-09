#!/usr/bin/env python3
"""
3Deflatten Setup & Diagnostics
================================
Checks GPU runtime support, guides you through installing missing dependencies,
and downloads ONNX depth estimation models.

Usage:
    python Setup.py              -- interactive menu
    python Setup.py --check      -- diagnostics only, no prompts
    python Setup.py --models     -- model download only
    python Setup.py --model <id> -- download specific model (0-6)
"""
import sys
import os
import io
import ctypes
import winreg
import pathlib
import argparse
import urllib.request
import hashlib
import subprocess

# Force UTF-8 output (Windows console may default to cp1252)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BANNER = """
=================================================================
  3Deflatten  --  Setup & Diagnostics
================================================================="""

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------
MODELS = [
    # (id, display_name, filename, url, sha256_or_None, size_mb)
    (0, "Depth Anything V2 Small  (fast,  ~80 MB)",
        "depth_anything_v2_small.onnx",
        "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
        None, 80),
    (1, "Depth Anything V2 Base   (balanced, ~390 MB)",
        "depth_anything_v2_base.onnx",
        "https://huggingface.co/onnx-community/depth-anything-v2-base/resolve/main/onnx/model.onnx",
        None, 390),
    (2, "Depth Anything V2 Large  (quality, ~1.3 GB)",
        "depth_anything_v2_large.onnx",
        "https://huggingface.co/onnx-community/depth-anything-v2-large/resolve/main/onnx/model.onnx",
        None, 1300),
    (3, "DA V2 Metric Indoor Small  (metric depth, ~80 MB)",
        "depth_anything_v2_metric_hypersim_vits.onnx",
        "https://huggingface.co/onnx-community/depth-anything-v2-metric-hypersim-vits/resolve/main/onnx/model.onnx",
        None, 80),
    (4, "DA V2 Metric Outdoor Small (metric depth, ~80 MB)",
        "depth_anything_v2_metric_vkitti_vits.onnx",
        "https://huggingface.co/onnx-community/depth-anything-v2-metric-vkitti-vits/resolve/main/onnx/model.onnx",
        None, 80),
    (5, "Depth Anything V1 Small    (~100 MB)",
        "depth_anything_v1_small.onnx",
        "https://huggingface.co/onnx-community/depth-anything-small-hf/resolve/main/onnx/model.onnx",
        None, 100),
    (6, "MiDaS v2.1 Small           (~50 MB, legacy)",
        "midas_v21_small_256.onnx",
        "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.onnx",
        None, 50),
]


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def find_dll_on_path(name: str) -> str | None:
    """Search DLL search path for name; return full path or None."""
    import ctypes.util
    # Try LoadLibraryExW with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x1000
    h = ctypes.windll.kernel32.LoadLibraryExW(name, None, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
    if h:
        buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.kernel32.GetModuleFileNameW(h, buf, 512)
        ctypes.windll.kernel32.FreeLibrary(h)
        return buf.value
    return None


def reg_read(root, subkey, value_name="") -> str | None:
    try:
        with winreg.OpenKey(root, subkey, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as k:
            v, _ = winreg.QueryValueEx(k, value_name)
            return str(v)
    except OSError:
        return None


def env(name: str) -> str | None:
    v = os.environ.get(name, "")
    return v if v else None


def find_cuda_bin() -> list[str]:
    """Return list of CUDA bin directories found on this machine."""
    dirs = []
    # Environment variables set by CUDA installer
    for var, version in [
        ("CUDA_PATH_V12_9", "12.9"), ("CUDA_PATH_V12_8", "12.8"),
        ("CUDA_PATH_V12_7", "12.7"), ("CUDA_PATH_V12_6", "12.6"),
        ("CUDA_PATH_V12_5", "12.5"), ("CUDA_PATH_V12_4", "12.4"),
        ("CUDA_PATH_V12_3", "12.3"), ("CUDA_PATH_V12_2", "12.2"),
        ("CUDA_PATH_V12_1", "12.1"), ("CUDA_PATH_V12_0", "12.0"),
        ("CUDA_PATH_V13_0", "13.0"),
        ("CUDA_PATH", ""),
    ]:
        p = env(var)
        if p:
            b = pathlib.Path(p) / "bin"
            if b.exists():
                dirs.append((version or "?", str(b)))
    return dirs


def check_cuda() -> dict:
    result = {"ok": False, "version": None, "path": None, "notes": []}

    # Probe for cudart64_12.dll (required by ORT 1.21 GPU build)
    p = find_dll_on_path("cudart64_12.dll")
    if p:
        result["ok"] = True
        result["version"] = "12.x"
        result["path"] = p
        return result

    # Not found -- check what IS installed
    if find_dll_on_path("cudart64_13.dll"):
        result["notes"].append(
            "CUDA 13.x detected. ORT 1.21 requires CUDA 12.x (cudart64_12.dll).\n"
            "  Install CUDA 12.6 alongside 13.x -- both can coexist.\n"
            "  Download: https://developer.nvidia.com/cuda-12-6-0-download-archive"
        )
    elif find_dll_on_path("cudart64_110.dll"):
        result["notes"].append(
            "CUDA 11.x detected. ORT 1.21 requires CUDA 12.x.\n"
            "  Download: https://developer.nvidia.com/cuda-12-6-0-download-archive"
        )
    else:
        # Check if CUDA path env vars exist but bin isn't on PATH
        cuda_dirs = find_cuda_bin()
        cuda12 = [d for v, d in cuda_dirs if v.startswith("12.")]
        if cuda12:
            result["notes"].append(
                f"CUDA 12 bin found at {cuda12[0]} but NOT on the DLL search path.\n"
                "  The filter adds this automatically. If inference still fails,\n"
                "  check that CUDA_PATH_V12_x env var is set by the CUDA installer."
            )
            result["ok"] = True
            result["version"] = "12.x"
            result["path"] = cuda12[0]
        else:
            result["notes"].append(
                "No CUDA found. Download CUDA 12.6:\n"
                "  https://developer.nvidia.com/cuda-12-6-0-download-archive"
            )
    return result


def check_cudnn() -> dict:
    result = {"ok": False, "version": None, "path": None, "notes": []}
    p = find_dll_on_path("cudnn64_9.dll")
    if p:
        result["ok"] = True
        result["version"] = "9.x"
        result["path"] = p
        return result
    # Check if it's installed but not on PATH
    cudnn_path = env("CUDNN_PATH")
    if not cudnn_path:
        cudnn_path = reg_read(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\NVIDIA Corporation\cuDNN", "InstallPath")
    if cudnn_path:
        # Recursive scan for cudnn64_9.dll
        for root, _, files in os.walk(cudnn_path):
            if "cudnn64_9.dll" in files:
                result["ok"] = True
                result["version"] = "9.x"
                result["path"] = os.path.join(root, "cudnn64_9.dll")
                result["notes"].append(
                    f"cuDNN 9 found at {result['path']} but NOT on DLL search path.\n"
                    "  Set CUDNN_PATH=" + cudnn_path + " so the filter can find it."
                )
                return result
    result["notes"].append(
        "cuDNN 9.x not found. Download:\n"
        "  https://developer.nvidia.com/cudnn\n"
        "  Install the Windows installer -- it sets CUDNN_PATH automatically."
    )
    return result


def check_tensorrt() -> dict:
    result = {"ok": False, "version": None, "path": None, "notes": []}
    p = find_dll_on_path("nvinfer_10.dll")
    if p:
        result["ok"] = True
        result["path"] = p
        # Try to extract version from the DLL path (e.g. TensorRT-10.7.0.23)
        parts = pathlib.Path(p).parts
        for part in parts:
            if part.lower().startswith("tensorrt-"):
                result["version"] = part
                break
        return result

    # Check TRT_LIB_PATH
    trt_lib = env("TRT_LIB_PATH")
    trt_dir = env("TENSORRT_DIR")
    if trt_lib:
        candidate = pathlib.Path(trt_lib) / "nvinfer_10.dll"
        if candidate.exists():
            result["ok"] = True
            result["path"] = str(candidate)
            result["notes"].append(
                "nvinfer_10.dll found via TRT_LIB_PATH but NOT on DLL search path.\n"
                "  The filter reads TRT_LIB_PATH automatically; ensure it is set\n"
                "  as a SYSTEM (not just user) environment variable, or copy\n"
                "  nvinfer_10.dll + nvonnxparser_10.dll next to the .ax file."
            )
            return result
        result["notes"].append(
            f"TRT_LIB_PATH={trt_lib} but nvinfer_10.dll not found there.\n"
            "  TRT DLLs live in the lib\\ subfolder of the TensorRT zip.\n"
            "  Set TRT_LIB_PATH=C:\\path\\to\\TensorRT-10.x.y.z\\lib"
        )
    if trt_dir:
        lib_dir = pathlib.Path(trt_dir) / "lib"
        candidate = lib_dir / "nvinfer_10.dll"
        if candidate.exists():
            result["ok"] = True
            result["path"] = str(candidate)
            return result

    result["notes"].append(
        "TensorRT 10.x not found.\n"
        "  Download TensorRT 10 for your CUDA version from:\n"
        "  https://developer.nvidia.com/tensorrt\n"
        "\n"
        "  IMPORTANT: TRT version must match your CUDA version:\n"
        "    TRT 10.7.x  -> CUDA 12.6\n"
        "    TRT 10.9.x  -> CUDA 12.8\n"
        "    TRT 10.15.x -> CUDA 12.9\n"
        "\n"
        "  After downloading, set:\n"
        "    TRT_LIB_PATH=C:\\path\\to\\TensorRT-10.x.y.z\\lib\n"
        "  Or copy nvinfer_10.dll + nvonnxparser_10.dll next to the .ax file."
    )
    return result


def check_directml() -> dict:
    result = {"ok": False, "version": None, "notes": []}
    import platform
    ver = platform.version()  # e.g. "10.0.19041"
    try:
        build = int(platform.version().split(".")[-1])
    except Exception:
        build = 0

    if build >= 17763:  # Windows 10 1809+
        result["ok"] = True
        result["version"] = f"Windows build {build}"
    else:
        result["notes"].append(
            "DirectML requires Windows 10 1809 (build 17763) or later.\n"
            f"  Your build: {build}"
        )
        return result

    # Check directml.dll is present next to the filter or on PATH
    p = find_dll_on_path("directml.dll")
    if not p:
        # Check next to this script's parent (install root)
        script_dir = pathlib.Path(__file__).parent.parent
        for d in [script_dir, script_dir / "Win64"]:
            candidate = d / "directml.dll"
            if candidate.exists():
                p = str(candidate)
                break
    if p:
        result["path"] = p
    else:
        result["notes"].append(
            "directml.dll not found. It should be next to 3Deflatten_x64.ax.\n"
            "  Download the x64 DirectML build of 3Deflatten which bundles it."
        )
    return result


def check_ort_dll() -> dict:
    result = {"ok": False, "path": None, "notes": []}
    p = find_dll_on_path("onnxruntime.dll")
    if p:
        result["ok"] = True
        result["path"] = p
        # Warn if it looks like a non-1.21 version might have been manually swapped
        if "1.21" not in p:
            result["notes"].append(
                "WARNING: onnxruntime.dll found at a path that does not include '1.21'.\n"
                "  The filter is ABI-linked to ORT 1.21.0. Using a different version\n"
                "  will likely cause crashes or silent fallback to CPU.\n"
                "  Path: " + p
            )
    else:
        result["notes"].append(
            "onnxruntime.dll not found on DLL search path.\n"
            "  It should be next to the .ax file. Re-install 3Deflatten."
        )
    return result


def check_filter_registration() -> dict:
    result = {"registered": False, "path": None}
    clsid = r"CLSID\{4D455F30-1A2B-4C3D-8E4F-5A6B7C8D9E0F}\InprocServer32"
    p = reg_read(winreg.HKEY_CLASSES_ROOT, clsid)
    if p:
        result["registered"] = True
        result["path"] = p
    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

OK    = "[  OK  ]"
WARN  = "[ WARN ]"
FAIL  = "[ FAIL ]"
INFO  = "[ INFO ]"

def status(flag: bool, warn_ok=False):
    if flag:
        return OK
    return WARN if warn_ok else FAIL


def run_diagnostics(verbose=True) -> dict:
    results = {}

    print("\n--- GPU Runtime Diagnostics ---\n")

    # CUDA
    r = check_cuda()
    results["cuda"] = r
    sym = status(r["ok"])
    if r["ok"]:
        print(f"{sym}  CUDA 12.x   -- {r['path']}")
    else:
        print(f"{sym}  CUDA 12.x   -- NOT FOUND")
    for n in r["notes"]:
        for line in n.splitlines():
            print(f"           {line}")

    # cuDNN
    r = check_cudnn()
    results["cudnn"] = r
    sym = status(r["ok"], warn_ok=True)
    if r["ok"]:
        print(f"{sym}  cuDNN 9.x   -- {r['path']}")
    else:
        print(f"{sym}  cuDNN 9.x   -- NOT FOUND  (required for CUDA/TRT EPs)")
    for n in r["notes"]:
        for line in n.splitlines():
            print(f"           {line}")

    # TensorRT
    r = check_tensorrt()
    results["trt"] = r
    sym = status(r["ok"], warn_ok=True)
    if r["ok"]:
        label = r["version"] or "found"
        print(f"{sym}  TensorRT 10 -- {r['path']}")
    else:
        print(f"{sym}  TensorRT 10 -- NOT FOUND  (optional; fastest EP)")
    for n in r["notes"]:
        for line in n.splitlines():
            print(f"           {line}")

    # DirectML
    r = check_directml()
    results["dml"] = r
    sym = status(r["ok"])
    if r["ok"]:
        path_str = r.get("path") or "(built into Windows)"
        print(f"{sym}  DirectML    -- {path_str}")
    else:
        print(f"{sym}  DirectML    -- NOT AVAILABLE")
    for n in r["notes"]:
        for line in n.splitlines():
            print(f"           {line}")

    # onnxruntime.dll
    r = check_ort_dll()
    results["ort"] = r
    sym = status(r["ok"])
    if r["ok"]:
        print(f"{sym}  onnxruntime -- {r['path']}")
    else:
        print(f"{sym}  onnxruntime -- NOT FOUND")
    for n in r["notes"]:
        for line in n.splitlines():
            print(f"           {line}")

    # Filter registration
    r = check_filter_registration()
    results["reg"] = r
    sym = OK if r["registered"] else WARN
    if r["registered"]:
        print(f"{sym}  Filter reg  -- {r['path']}")
    else:
        print(f"{sym}  Filter reg  -- NOT REGISTERED")
        print(f"           Run:  regsvr32 \"<path>\\3Deflatten_x64.ax\"")

    print()

    # Summary
    print("--- Recommended execution provider ---")
    if results["cuda"]["ok"] and results["trt"]["ok"] and results["cudnn"]["ok"]:
        print("  TensorRT   (fastest -- use gpuProvider=1 or Auto)")
        print("  NOTE: first inference compiles TRT engines (30-120 s).")
    elif results["cuda"]["ok"] and results["cudnn"]["ok"]:
        print("  CUDA       (fast -- use gpuProvider=2 or Auto)")
    elif results["dml"]["ok"]:
        print("  DirectML   (good for any DX12 GPU -- gpuProvider=3 or Auto)")
        print("  NOTE: first frame triggers shader compilation (5-30 s).")
        print("  Subsequent frames on RTX 2080 Ti: ~100-400 ms for DA V2 Small.")
    else:
        print("  CPU        (slow -- no GPU runtime detected)")
    print()

    return results


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def default_output_dir() -> pathlib.Path:
    script_parent = pathlib.Path(__file__).parent.parent
    for d in [script_parent / "Win64", script_parent / "Win64_GPU",
              script_parent / "Win32", script_parent]:
        if d.is_dir():
            return d
    return script_parent


def download_model(model_id: int, output_dir: pathlib.Path) -> bool:
    if model_id < 0 or model_id >= len(MODELS):
        print(f"Unknown model id {model_id}")
        return False

    mid, label, fname, url, sha, size_mb = MODELS[model_id]
    dst = output_dir / fname

    if dst.exists():
        print(f"Already exists: {dst}")
        return True

    print(f"\nDownloading: {label}")
    print(f"  URL: {url}")
    print(f"  Size: ~{size_mb} MB")
    print(f"  Destination: {dst}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        def progress(block, block_size, total):
            if total > 0:
                pct = min(100, block * block_size * 100 // total)
                print(f"\r  Progress: {pct:3d}%", end="", flush=True)

        urllib.request.urlretrieve(url, dst, reporthook=progress)
        print(f"\r  Progress: 100%  -- done")
        if sha:
            h = hashlib.sha256(dst.read_bytes()).hexdigest()
            if h != sha:
                print(f"  WARNING: SHA256 mismatch! Expected {sha}, got {h}")
            else:
                print(f"  SHA256 OK")
        return True
    except Exception as e:
        print(f"\n  FAILED: {e}")
        if dst.exists():
            dst.unlink()
        return False


def model_menu(output_dir: pathlib.Path):
    print("\n--- ONNX Model Download ---")
    print(f"  Output directory: {output_dir}\n")
    for mid, label, fname, url, sha, size_mb in MODELS:
        exists = (output_dir / fname).exists()
        mark = " [downloaded]" if exists else ""
        print(f"  {mid}  {label}{mark}")
    print(f"  a  Download all models (~{sum(m[5] for m in MODELS)} MB total)")
    print(f"  q  Back / quit")
    print()
    choice = input("Enter model number (or a/q): ").strip().lower()
    if choice == "q":
        return
    if choice == "a":
        for mid, *_ in MODELS:
            download_model(mid, output_dir)
        return
    try:
        download_model(int(choice), output_dir)
    except ValueError:
        print("Invalid choice.")


# ---------------------------------------------------------------------------
# Main interactive menu
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3Deflatten Setup & Diagnostics")
    parser.add_argument("--check",  action="store_true", help="Run diagnostics only")
    parser.add_argument("--models", action="store_true", help="Model download menu only")
    parser.add_argument("--model",  type=int, default=-1, metavar="ID",
                        help="Download specific model by ID (0-6)")
    parser.add_argument("--output", default="", metavar="DIR",
                        help="Override output directory for model download")
    parser.add_argument("--list",   action="store_true", help="List available models")
    args = parser.parse_args()

    print(BANNER)

    out_dir = pathlib.Path(args.output).resolve() if args.output else default_output_dir()

    if args.list:
        for mid, label, fname, *_ in MODELS:
            print(f"  {mid}  {fname}  ({label})")
        return

    if args.model >= 0:
        download_model(args.model, out_dir)
        return

    if args.check:
        run_diagnostics()
        return

    if args.models:
        model_menu(out_dir)
        return

    # Interactive mode
    while True:
        print("\n--- Main Menu ---")
        print("  1  Run diagnostics (check CUDA / cuDNN / TensorRT / DirectML)")
        print("  2  Download depth model")
        print("  3  Show dependency download links")
        print("  q  Quit")
        choice = input("\nChoice: ").strip().lower()

        if choice == "1":
            run_diagnostics()

        elif choice == "2":
            model_menu(out_dir)

        elif choice == "3":
            print("""
--- Dependency Download Links ---

CUDA 12.6 (recommended -- required for CUDA and TensorRT EPs):
  https://developer.nvidia.com/cuda-12-6-0-download-archive

cuDNN 9.x (required for CUDA and TensorRT EPs):
  https://developer.nvidia.com/cudnn
  -> Install the Windows local installer; it sets CUDNN_PATH automatically.

TensorRT 10 (optional; fastest -- match version to your CUDA):
  https://developer.nvidia.com/tensorrt
  TRT 10.7.x  -> CUDA 12.6    TRT 10.15.x -> CUDA 12.9
  After installing, set:  TRT_LIB_PATH=<TRT root>\\lib

NOTE: ORT 1.21 (bundled with 3Deflatten) supports CUDA 12.x ONLY.
  CUDA 13.x ships cudart64_13.dll which is NOT compatible with this build.
  You can install CUDA 12.6 alongside 13.x -- multiple versions coexist.

DirectML: built into Windows 10 1809+ (no extra install needed).
  The DX12 DirectML EP is included in the x64 3Deflatten build.
""")

        elif choice in ("q", "quit", "exit"):
            break

        else:
            print("Unknown option.")


if __name__ == "__main__":
    main()
