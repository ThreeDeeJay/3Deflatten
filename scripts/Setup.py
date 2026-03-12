#!/usr/bin/env python3
"""
3Deflatten Setup & Diagnostics
================================
Checks GPU runtime support, guides you through installing missing dependencies,
and downloads ONNX depth estimation models.

Detection logic mirrors the .ax filter exactly:
  1. Environment variables (inherited at process-creation time)
  2. Registry keys written by installers
  3. Recursive filesystem scan of default install paths
  4. DLLs already bundled next to the .ax file

Usage:
    python Setup.py              -- interactive menu
    python Setup.py --check      -- diagnostics only, no prompts
    python Setup.py --models     -- model download only
    python Setup.py --model <id> -- download specific model (0-6)
"""
from __future__ import annotations
import sys, os, io, winreg, pathlib, argparse, urllib.request, hashlib

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

# ORT version string bundled in the current release
ORT_VERSION  = "1.16.3"
# ORT 1.16.3 GPU build was compiled against CUDA 11.8.
# Runtime DLLs must be CUDA 11.x (cudart64_11.dll). Driver 452+ required.
CUDA_MAJOR    = 11
CUDA_RT_DLL   = "cudart64_11.dll"
CUBLAS_DLL    = "cublas64_11.dll"
CUBLASLT_DLL  = "cublasLt64_11.dll"
CUFFT_DLL     = "cufft64_10.dll"     # cuFFT API stays at version 10 inside CUDA 11 toolkit
# nvJitLink was introduced in CUDA 12; NOT present in CUDA 11 / ORT 1.16.x
CUDNN_DLL     = "cudnn64_8.dll"      # cuDNN 8.x monolithic DLL (cuDNN 9 uses cudnn64_9)

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------
MODELS = [
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
# Filesystem helpers  (mirror the filter's RecursiveFindAndAdd logic)
# ---------------------------------------------------------------------------

# Setup.py lives at <root>/Setup.py  (same level as Win32/ Win64/ Win64_GPU/)
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

# The three release subdirectories, ordered by preference
RELEASE_DIRS = [
    SCRIPT_DIR / "Win64_GPU",
    SCRIPT_DIR / "Win64",
    SCRIPT_DIR / "Win32",
    SCRIPT_DIR,
]


def find_file_recursive(base: str | pathlib.Path,
                        filename: str,
                        max_depth: int = 6) -> pathlib.Path | None:
    """Walk *base* up to *max_depth* levels looking for *filename*.
    Matches the filter's RecursiveFindAndAdd() behaviour."""
    base = pathlib.Path(base)
    if not base.exists():
        return None
    fname_lower = filename.lower()
    for root, dirs, files in os.walk(base):
        depth = len(pathlib.Path(root).relative_to(base).parts)
        if depth >= max_depth:
            dirs.clear()
            continue
        for f in files:
            if f.lower() == fname_lower:
                return pathlib.Path(root) / f
    return None


def find_in_release_dirs(filename: str) -> pathlib.Path | None:
    """Look for *filename* in the release subfolders next to this script."""
    for d in RELEASE_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


def reg_read(root, subkey: str, value_name: str = "") -> str | None:
    for flag in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
        try:
            with winreg.OpenKey(root, subkey, 0,
                                winreg.KEY_READ | flag) as k:
                v, _ = winreg.QueryValueEx(k, value_name)
                return str(v)
        except OSError:
            pass
    return None


def reg_enum_subkeys(root, subkey: str) -> list[str]:
    try:
        with winreg.OpenKey(root, subkey, 0,
                            winreg.KEY_READ | winreg.KEY_ENUMERATE_SUB_KEYS
                            | winreg.KEY_WOW64_64KEY) as k:
            keys = []
            i = 0
            while True:
                try:
                    keys.append(winreg.EnumKey(k, i))
                    i += 1
                except OSError:
                    break
            return keys
    except OSError:
        return []


def env(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None


# ---------------------------------------------------------------------------
# CUDA detection  (matches filter's RegisterGpuRuntimeDirs CUDA block)
# ---------------------------------------------------------------------------

def check_cuda() -> dict:
    result = {"ok": False, "version": None, "path": None,
              "how": None, "notes": []}

    # ── Step 0: bundled DLLs in Win64_GPU (from collect_runtime_dlls.py) ────
    hit = find_in_release_dirs(CUDA_RT_DLL)
    if hit:
        result.update(ok=True, version="13.x", path=str(hit), how="bundled")
        return result

    # ── Step 1: env vars set by CUDA 13 installer ───────────────────────────
    cuda_env_vars = [
        ("CUDA_PATH_V13_2", "13.2"), ("CUDA_PATH_V13_1", "13.1"),
        ("CUDA_PATH_V13_0", "13.0"), ("CUDA_PATH", "?"),
    ]
    for var, ver in cuda_env_vars:
        p = env(var)
        if not p:
            continue
        # CUDA 13 on Windows stores DLLs in bin\x64\ (toolkit) or bin\
        for bin_sub in (r"bin\x64", "bin"):
            dll = pathlib.Path(p) / bin_sub / CUDA_RT_DLL
            if dll.exists():
                result.update(ok=True, version=ver, path=str(dll), how="env")
                return result

    # ── Step 2: registry ─────────────────────────────────────────────────────
    reg_base = r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA"
    for sub in reg_enum_subkeys(winreg.HKEY_LOCAL_MACHINE, reg_base):
        if not sub.startswith("v13."):
            continue
        inst = reg_read(winreg.HKEY_LOCAL_MACHINE,
                        f"{reg_base}\\{sub}", "InstallDir")
        if not inst:
            continue
        for bin_sub in (r"bin\x64", "bin"):
            dll = pathlib.Path(inst) / bin_sub / CUDA_RT_DLL
            if dll.exists():
                result.update(ok=True, version=sub.lstrip("v"),
                              path=str(dll), how="registry")
                return result

    # ── Step 3: default filesystem scan ──────────────────────────────────────
    cuda_root = pathlib.Path(
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    hit = find_file_recursive(cuda_root, CUDA_RT_DLL)
    if hit:
        result.update(ok=True, path=str(hit), how="filesystem")
        for part in hit.parts:
            if part.lower().startswith("v") and "." in part:
                result["version"] = part.lstrip("vV")
                break
        return result

    # ── Not found ─────────────────────────────────────────────────────────────
    wrong13 = find_file_recursive(cuda_root, "cudart64_13.dll")
    if wrong13:
        result["notes"].append(
            f"CUDA 13.x found at {wrong13}, but ORT {ORT_VERSION} requires CUDA 11.x.\n"
            f"  Delete Win64_GPU DLLs and re-run collect_runtime_dlls.py to get CUDA 11."
        )
    else:
        result["notes"].append(
            "CUDA 11.x not found.\n"
            "  Option 1 (recommended): run  python collect_runtime_dlls.py\n"
            "    This bundles CUDA 11.7 DLLs in Win64_GPU so no system install is needed.\n"
            "  Option 2: install CUDA 11.7:\n"
            "    https://developer.download.nvidia.com/compute/cuda/11.7.0/"
            "local_installers/cuda_11.7.0_516.01_windows.exe"
        )
    return result


# ---------------------------------------------------------------------------
# cuDNN detection  (matches filter's CUDNN block)
# ---------------------------------------------------------------------------

def check_cudnn() -> dict:
    result = {"ok": False, "version": None, "path": None,
              "how": None, "notes": []}

    # ── Step 0: bundled DLLs in Win64_GPU (from collect_runtime_dlls.py) ────
    hit = find_in_release_dirs("cudnn64_8.dll")
    if hit:
        result.update(ok=True, version="8.x", path=str(hit), how="bundled")
        return result

    # ── Step 1: CUDNN_PATH env var ────────────────────────────────────────────
    cudnn_env = env("CUDNN_PATH")
    if cudnn_env:
        hit = find_file_recursive(cudnn_env, "cudnn64_8.dll")
        if hit:
            result.update(ok=True, version="8.x", path=str(hit), how="env")
            return result
        else:
            result["notes"].append(
                f"CUDNN_PATH={cudnn_env} is set but cudnn64_8.dll was not "
                f"found inside it."
            )

    # ── Step 2: registry ─────────────────────────────────────────────────────
    inst = reg_read(winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\NVIDIA Corporation\cuDNN", "InstallPath")
    if inst:
        hit = find_file_recursive(inst, "cudnn64_8.dll")
        if hit:
            result.update(ok=True, version="8.x",
                          path=str(hit), how="registry")
            return result

    # ── Step 3: default filesystem scan ──────────────────────────────────────
    cudnn_root = pathlib.Path(r"C:\Program Files\NVIDIA\CUDNN")
    hit = find_file_recursive(cudnn_root, "cudnn64_8.dll")
    if hit:
        result.update(ok=True, version="8.x",
                      path=str(hit), how="filesystem")
        return result

    # ── Not found ─────────────────────────────────────────────────────────────
    result["notes"].append(
        "cuDNN 8.x not found.\n"
        "  Option 1 (recommended): run  python collect_runtime_dlls.py\n"
        "    This bundles cuDNN 8 DLLs in Win64_GPU so no system install is needed.\n"
        "  Option 2: download cuDNN 8.9.3 for CUDA 11:\n"
        "    https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/"
        "windows-x86_64/cudnn-windows-x86_64-8.9.3.28_cuda11-archive.zip\n"
        "  The standalone installer does NOT set CUDNN_PATH automatically.\n"
        r"  After installing, set: CUDNN_PATH=C:\Program Files\NVIDIA\CUDNN\v9.19\bin\13.1\x64"
    )
    return result


# ---------------------------------------------------------------------------
# TensorRT detection  (matches filter's TRT block)
# ---------------------------------------------------------------------------

def check_tensorrt() -> dict:
    result = {"ok": False, "version": None, "path": None,
              "how": None, "notes": []}

    # ── Step 0: bundled DLLs in Win64_GPU (from collect_runtime_dlls.py) ────
    hit = find_in_release_dirs("nvinfer_10.dll")
    if hit:
        result.update(ok=True, path=str(hit), how="bundled")
        return result

    # ── Step 1: TRT_LIB_PATH (should point to lib\ folder) ───────────────────
    trt_lib = env("TRT_LIB_PATH")
    if trt_lib:
        hit = find_file_recursive(trt_lib, "nvinfer_10.dll", max_depth=2)
        if hit:
            result.update(ok=True, path=str(hit), how="env:TRT_LIB_PATH")
            for part in hit.parts:
                if part.lower().startswith("tensorrt-"):
                    result["version"] = part
                    break
            return result
        result["notes"].append(
            f"TRT_LIB_PATH={trt_lib} set but nvinfer_10.dll not found there.\n"
            r"  TRT_LIB_PATH should point to the lib\ folder inside the TRT zip:"
            "\n"
            r"    TRT_LIB_PATH=C:\Program Files\NVIDIA\TensorRT-10.x.y.z\lib"
        )

    # ── Step 2: TENSORRT_DIR ──────────────────────────────────────────────────
    trt_dir = env("TENSORRT_DIR")
    if trt_dir:
        lib_dir = pathlib.Path(trt_dir) / "lib"
        hit = find_file_recursive(lib_dir, "nvinfer_10.dll", max_depth=2)
        if hit:
            result.update(ok=True, path=str(hit), how="env:TENSORRT_DIR")
            return result

    # ── Step 3: registry ─────────────────────────────────────────────────────
    inst = reg_read(winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\NVIDIA Corporation\TensorRT", "InstallPath")
    if inst:
        hit = find_file_recursive(inst, "nvinfer_10.dll")
        if hit:
            result.update(ok=True, path=str(hit), how="registry")
            return result

    # ── Step 4: default scan under C:\Program Files\NVIDIA ───────────────────
    hit = find_file_recursive(r"C:\Program Files\NVIDIA", "nvinfer_10.dll")
    if hit:
        result.update(ok=True, path=str(hit), how="filesystem")
        for part in hit.parts:
            if part.lower().startswith("tensorrt-"):
                result["version"] = part
                break
        return result

    # ── Not found ─────────────────────────────────────────────────────────────
    result["notes"].append(
        "TensorRT 10.13.x not found (optional; required only for TRT EP).\n"
        "  Option 1 (recommended): run  python collect_runtime_dlls.py\n"
        "    This bundles TRT DLLs in Win64_GPU -- no system install needed.\n"
        "  Option 2: download TRT 10.13.0.35 (CUDA 11.8 build):\n"
        "    https://developer.nvidia.com/downloads/compute/machine-learning/"
        "tensorrt/10.13.0/zip/TensorRT-10.13.0.35.Windows.win10.cuda-11.8.zip\n"
        r"  After extracting, set: TRT_LIB_PATH=<TRT root>\lib"
        "\n"
        "  NOTE: TRT 10.7.x and earlier are CUDA 12 builds -- do NOT use those."
    )
    return result


# ---------------------------------------------------------------------------
# DirectML detection
# ---------------------------------------------------------------------------

def check_directml() -> dict:
    result = {"ok": False, "path": None, "notes": []}
    import platform
    try:
        build = int(platform.version().split(".")[-1])
    except Exception:
        build = 0

    if build < 17763:
        result["notes"].append(
            "DirectML requires Windows 10 1809 (build 17763) or later.\n"
            f"  Your build: {build}"
        )
        return result

    result["ok"] = True   # Windows is new enough

    # Look for directml.dll in the release folders
    hit = find_in_release_dirs("directml.dll")
    if not hit:
        # System-wide (Windows ships it in system32 on modern builds)
        try:
            import ctypes
            h = ctypes.windll.kernel32.LoadLibraryExW(
                "directml.dll", None, 0x1000)  # LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
            if h:
                buf = ctypes.create_unicode_buffer(512)
                ctypes.windll.kernel32.GetModuleFileNameW(h, buf, 512)
                ctypes.windll.kernel32.FreeLibrary(h)
                hit = pathlib.Path(buf.value) if buf.value else None
        except Exception:
            pass

    if hit:
        result["path"] = str(hit)
    else:
        result["notes"].append(
            "directml.dll not found next to the .ax file.\n"
            "  It should be bundled in the Win64 3Deflatten release folder."
        )
    return result


# ---------------------------------------------------------------------------
# onnxruntime.dll detection
# ---------------------------------------------------------------------------

def check_ort_dll() -> dict:
    result = {"ok": False, "path": None, "notes": []}

    # Primary: look in the release folders next to this script
    hit = find_in_release_dirs("onnxruntime.dll")
    if hit:
        result.update(ok=True, path=str(hit))
        if ORT_VERSION not in str(hit.parent):
            pass  # version not in path is fine for zip-distributed releases
        return result

    # Secondary: DLL search path (system-wide install)
    try:
        import ctypes
        h = ctypes.windll.kernel32.LoadLibraryExW(
            "onnxruntime.dll", None, 0x1000)
        if h:
            buf = ctypes.create_unicode_buffer(512)
            ctypes.windll.kernel32.GetModuleFileNameW(h, buf, 512)
            ctypes.windll.kernel32.FreeLibrary(h)
            if buf.value:
                result.update(ok=True, path=buf.value)
                return result
    except Exception:
        pass

    result["notes"].append(
        "onnxruntime.dll not found in the Win64_GPU / Win64 / Win32 folders\n"
        "  next to Setup.py, nor on the system DLL search path.\n"
        "  It should be bundled inside the 3Deflatten release zip.\n"
        "  Re-download the release package."
    )
    return result


# ---------------------------------------------------------------------------
# Filter registration
# ---------------------------------------------------------------------------

def check_filter_registration() -> dict:
    result = {"registered": False, "path": None}
    # Try both x64 and x86 CLSIDs
    for clsid in [
        r"CLSID\{4D455F32-1A2B-4C3D-8E4F-5A6B7C8D9E0F}\InprocServer32",
        r"CLSID\{4D455F30-1A2B-4C3D-8E4F-5A6B7C8D9E0F}\InprocServer32",
    ]:
        p = reg_read(winreg.HKEY_CLASSES_ROOT, clsid)
        if p:
            result.update(registered=True, path=p)
            return result
    return result


# ---------------------------------------------------------------------------
# DLL accessibility check  (mirrors filter's "will this DLL actually load?")
# ---------------------------------------------------------------------------

def _how_label(how: str | None) -> str:
    labels = {
        "env":              "found via env var",
        "env:TRT_LIB_PATH": "found via TRT_LIB_PATH",
        "env:TENSORRT_DIR": "found via TENSORRT_DIR",
        "registry":         "found via registry",
        "filesystem":       "found via default path scan",
    }
    return labels.get(how or "", "found")


def _env_restart_note(how: str | None) -> str | None:
    """Warn if the component was found only via an env var that must be
    set before launching the host application."""
    if how and how.startswith("env"):
        return (
            "NOTE: env vars are read at process-creation time.\n"
            "  If you set this AFTER launching PotPlayer/GraphEdit, the\n"
            "  filter will NOT see it.  Restart the host app to pick it up.\n"
            "  To make it permanent: set as a SYSTEM environment variable\n"
            "  (System Properties > Environment Variables > System variables)."
        )
    return None


# ---------------------------------------------------------------------------
# run_diagnostics
# ---------------------------------------------------------------------------

OK   = "[  OK  ]"
WARN = "[ WARN ]"
FAIL = "[ FAIL ]"

def _sym(flag: bool, warn_if_fail: bool = False) -> str:
    if flag:
        return OK
    return WARN if warn_if_fail else FAIL


def run_diagnostics() -> dict:
    results = {}
    print("\n--- GPU Runtime Diagnostics ---\n")
    print(f"  Detection order mirrors the filter: env vars -> registry -> "
          f"filesystem scan -> bundled DLLs\n")

    # ── CUDA ──────────────────────────────────────────────────────────────────
    r = check_cuda()
    results["cuda"] = r
    sym = _sym(r["ok"])
    ver_str = f"CUDA {r['version']}" if r["version"] else f"CUDA {CUDA_MAJOR}.x"
    if r["ok"]:
        print(f"{sym}  {ver_str:<12}-- {r['path']}")
        print(f"           ({_how_label(r['how'])})")
        note = _env_restart_note(r["how"])
        if note:
            for line in note.splitlines():
                print(f"           {line}")
    else:
        print(f"{sym}  CUDA {CUDA_MAJOR}.x     -- NOT FOUND")
    for n in r["notes"]:
        for line in n.splitlines():
            print(f"           {line}")

    # ── cuDNN ─────────────────────────────────────────────────────────────────
    r = check_cudnn()
    results["cudnn"] = r
    sym = _sym(r["ok"], warn_if_fail=True)
    if r["ok"]:
        print(f"{sym}  cuDNN 8.x    -- {r['path']}")
        print(f"           ({_how_label(r['how'])})")
        note = _env_restart_note(r["how"])
        if note:
            for line in note.splitlines():
                print(f"           {line}")
    else:
        print(f"{sym}  cuDNN 8.x    -- NOT FOUND  (required for CUDA/TRT EPs)")
    for n in r["notes"]:
        for line in n.splitlines():
            print(f"           {line}")

    # ── TensorRT ──────────────────────────────────────────────────────────────
    r = check_tensorrt()
    results["trt"] = r
    sym = _sym(r["ok"], warn_if_fail=True)
    ver_str = r["version"] or "TensorRT 10"
    if r["ok"]:
        print(f"{sym}  {ver_str:<12}-- {r['path']}")
        print(f"           ({_how_label(r['how'])})")
        note = _env_restart_note(r["how"])
        if note:
            for line in note.splitlines():
                print(f"           {line}")
    else:
        print(f"{sym}  TensorRT 10  -- NOT FOUND  (optional; fastest EP)")
    for n in r["notes"]:
        for line in n.splitlines():
            print(f"           {line}")

    # ── DirectML ──────────────────────────────────────────────────────────────
    r = check_directml()
    results["dml"] = r
    sym = _sym(r["ok"])
    if r["ok"]:
        path_str = r.get("path") or "(built into Windows)"
        print(f"{sym}  DirectML     -- {path_str}")
    else:
        print(f"{sym}  DirectML     -- NOT AVAILABLE")
    for n in r["notes"]:
        for line in n.splitlines():
            print(f"           {line}")

    # ── onnxruntime.dll ───────────────────────────────────────────────────────
    r = check_ort_dll()
    results["ort"] = r
    sym = _sym(r["ok"])
    if r["ok"]:
        print(f"{sym}  onnxruntime  -- {r['path']}")
    else:
        print(f"{sym}  onnxruntime  -- NOT FOUND")
    for n in r["notes"]:
        for line in n.splitlines():
            print(f"           {line}")

    # ── Filter registration ───────────────────────────────────────────────────
    r = check_filter_registration()
    results["reg"] = r
    sym = OK if r["registered"] else WARN
    if r["registered"]:
        print(f"{sym}  Filter reg   -- {r['path']}")
    else:
        print(f"{sym}  Filter reg   -- NOT REGISTERED")
        print(f'           Run:  regsvr32 "<path>\\3Deflatten_x64.ax"')

    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("--- Recommended execution provider ---")
    cuda_ok  = results["cuda"]["ok"]
    cudnn_ok = results["cudnn"]["ok"]
    trt_ok   = results["trt"]["ok"]
    dml_ok   = results["dml"]["ok"]

    if cuda_ok and cudnn_ok and trt_ok:
        print("  TensorRT   (fastest -- set gpuProvider=1 or leave on Auto)")
        print("  NOTE: first inference compiles TRT engines (30-120 s).")
        print("  NOTE: TRT_LIB_PATH must be set BEFORE launching the host app.")
    elif cuda_ok and cudnn_ok:
        print("  CUDA       (fast -- set gpuProvider=2 or leave on Auto)")
    elif dml_ok:
        print("  DirectML   (good for any DX12 GPU -- gpuProvider=3 or Auto)")
        print("  NOTE: first frame triggers DX shader compilation (5-30 s).")
        print("  Subsequent frames on RTX: ~100-400 ms for DA V2 Small.")
    else:
        print("  CPU        (slow -- no GPU runtime found)")

    print()
    return results


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def default_output_dir() -> pathlib.Path:
    """Return the best directory for model files: next to the .ax file.
    Setup.py lives at <root>/Setup.py, alongside Win32/ Win64/ Win64_GPU/.
    Models go directly in <root>/ so all three builds share them."""
    # Prefer the first release subfolder that exists; fall back to SCRIPT_DIR
    for d in RELEASE_DIRS:
        if d.is_dir() and d != SCRIPT_DIR:
            # Actually put models in SCRIPT_DIR (root), not a subfolder
            break
    return SCRIPT_DIR


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
        print("\r  Progress: 100%  -- done")
        if sha:
            h = hashlib.sha256(dst.read_bytes()).hexdigest()
            if h != sha:
                print(f"  WARNING: SHA256 mismatch! Expected {sha}, got {h}")
            else:
                print("  SHA256 OK")
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
    print("  q  Back / quit")
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

    out_dir = (pathlib.Path(args.output).resolve()
               if args.output else default_output_dir())

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
            print(f"""
--- Dependency Download Links ---

CUDA 11.7 (required for ORT {ORT_VERSION} GPU build):
  https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_516.01_windows.exe
  Or run  python collect_runtime_dlls.py  to bundle CUDA 11 DLLs automatically.

cuDNN 8.9.3 for CUDA 11 (required for CUDA and TensorRT EPs):
  https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-8.9.3.28_cuda11-archive.zip
  Or run  python collect_runtime_dlls.py  to bundle DLLs automatically.

TensorRT 10.13.0.35 CUDA 11.8 build (optional; fastest EP):
  https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.0/zip/TensorRT-10.13.0.35.Windows.win10.cuda-11.8.zip
  Or run  python collect_runtime_dlls.py  to bundle DLLs.
  After extracting, set: TRT_LIB_PATH=<TRT root>\\lib

DirectML: built into Windows 10 1809+ (no extra install needed).
  The x64 DirectML build of 3Deflatten bundles directml.dll.
""")

        elif choice in ("q", "quit", "exit"):
            break

        else:
            print("Unknown option.")


if __name__ == "__main__":
    main()
