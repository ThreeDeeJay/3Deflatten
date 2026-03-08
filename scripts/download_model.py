#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
r"""
Download depth-estimation ONNX models for use with 3Deflatten.

Supported models
----------------
Depth Anything V2  (LiheYoung et al., 2024)
  Small  ~25 MB  – fastest, suitable for real-time
  Base   ~98 MB  – balanced quality / speed
  Large  ~336 MB – best quality, needs a capable GPU

Depth Anything V2 Metric (absolute-scale variants)
  Small/Large × Indoor/Outdoor

Usage
-----
  python download_model.py                          # interactive menu
  python download_model.py --list                   # print model IDs
  python download_model.py v2-small                 # download by ID
  python download_model.py --all                    # download everything
  python download_model.py --output C:\path\to\dir  # custom output dir

Place the .onnx file next to 3Deflatten_x64.ax (or its parent folder) and
3Deflatten will find it automatically.  You can also point to a specific file:
  set DEFLATTEN_MODEL_PATH=C:\full\path\to\model.onnx
or edit 3Deflatten.ini next to the .ax:
  [3Deflatten]
  modelPath=C:\full\path\to\model.onnx
"""

import argparse
import pathlib
import shutil
import subprocess
import sys

# ── Model catalogue ────────────────────────────────────────────────────────────
# Fields: (id, label, hf_repo, hf_file, local_name)
MODELS = [
    # Depth Anything V2 – relative depth
    ("v2-small",
     "Depth Anything V2 Small   (~25 MB)   fastest, good for real-time",
     "onnx-community/depth-anything-v2-small",
     "onnx/model.onnx",
     "depth_anything_v2_small.onnx"),

    ("v2-base",
     "Depth Anything V2 Base    (~98 MB)   balanced quality / speed",
     "onnx-community/depth-anything-v2-base",
     "onnx/model.onnx",
     "depth_anything_v2_base.onnx"),

    ("v2-large",
     "Depth Anything V2 Large   (~336 MB)  best quality",
     "onnx-community/depth-anything-v2-large",
     "onnx/model.onnx",
     "depth_anything_v2_large.onnx"),

    # Depth Anything V2 Metric – absolute scale
    ("v2-metric-indoor-small",
     "Depth Anything V2 Metric Small Indoor   (~25 MB)",
     "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
     "onnx/model.onnx",
     "depth_anything_v2_metric_indoor_small.onnx"),

    ("v2-metric-outdoor-small",
     "Depth Anything V2 Metric Small Outdoor  (~25 MB)",
     "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
     "onnx/model.onnx",
     "depth_anything_v2_metric_outdoor_small.onnx"),

    ("v2-metric-indoor-large",
     "Depth Anything V2 Metric Large Indoor   (~336 MB)",
     "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
     "onnx/model.onnx",
     "depth_anything_v2_metric_indoor_large.onnx"),

    ("v2-metric-outdoor-large",
     "Depth Anything V2 Metric Large Outdoor  (~336 MB)",
     "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
     "onnx/model.onnx",
     "depth_anything_v2_metric_outdoor_large.onnx"),
]

MODEL_BY_ID = {m[0]: m for m in MODELS}


def ensure_hf_hub():
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError:
        print("huggingface-hub not found – installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "huggingface-hub"])
    from huggingface_hub import hf_hub_download
    return hf_hub_download


def download_one(model_id, dest_dir, hf_hub_download):
    mid, label, repo, hf_file, local_name = MODEL_BY_ID[model_id]
    dest_path = dest_dir / local_name

    if dest_path.exists():
        print(f"  Already present: {dest_path}")
        return True

    print(f"\nDownloading: {label}")
    print(f"  Repo : {repo}")
    print(f"  File : {hf_file}")
    print(f"  Dest : {dest_path}")

    try:
        local = hf_hub_download(
            repo_id=repo,
            filename=hf_file,
            local_dir=str(dest_dir),
        )
        src = pathlib.Path(local)
        if src.resolve() != dest_path.resolve():
            shutil.move(str(src), str(dest_path))

        # Remove HuggingFace metadata artefacts left in dest_dir
        for leftover in ("onnx", ".cache", ".huggingface"):
            p = dest_dir / leftover
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

        print(f"  [OK] {dest_path}")
        return True

    except Exception as exc:
        print(f"  [ERROR] {exc}")
        return False


def print_list():
    print("\nAvailable model IDs:")
    for mid, label, *_ in MODELS:
        print(f"  {mid:<30}  {label}")
    print()


def interactive_menu(dest_dir, hf_hub_download):
    print("\n3Deflatten – Model Downloader")
    print("=" * 60)
    print(f"Output directory: {dest_dir}\n")

    for i, (mid, label, *_) in enumerate(MODELS, 1):
        local_name = MODEL_BY_ID[mid][4]
        status = " [downloaded]" if (dest_dir / local_name).exists() else ""
        print(f"  {i}. {label}{status}")

    print("\n  a. Download ALL")
    print("  q. Quit\n")
    choice = input("Select (number / a / q): ").strip().lower()

    if choice == "q":
        return
    if choice == "a":
        ids = [m[0] for m in MODELS]
    else:
        try:
            idx = int(choice) - 1
            if not (0 <= idx < len(MODELS)):
                raise ValueError
            ids = [MODELS[idx][0]]
        except ValueError:
            print("Invalid selection.")
            return

    ok = all(download_one(mid, dest_dir, hf_hub_download) for mid in ids)
    if ok:
        print("\nDone.")
        print("Place the .onnx file next to 3Deflatten_x64.ax, or set")
        print("modelPath= in 3Deflatten.ini to point to a specific file.")
    else:
        print("\nOne or more downloads failed.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download depth ONNX models for 3Deflatten",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model_id", nargs="?",
                        help="Model ID to download (see --list)")
    parser.add_argument("--list",   action="store_true")
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--output", default=".",
                        help="Output directory (default: .)")
    args = parser.parse_args()

    dest_dir = pathlib.Path(args.output).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        print_list()
        return

    hf = ensure_hf_hub()

    if args.all:
        ok = all(download_one(mid, dest_dir, hf) for mid, *_ in MODELS)
        sys.exit(0 if ok else 1)

    if args.model_id:
        if args.model_id not in MODEL_BY_ID:
            print(f"Unknown model ID '{args.model_id}'.")
            print_list()
            sys.exit(1)
        ok = download_one(args.model_id, dest_dir, hf)
        sys.exit(0 if ok else 1)

    interactive_menu(dest_dir, hf)


if __name__ == "__main__":
    main()
