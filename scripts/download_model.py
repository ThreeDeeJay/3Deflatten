#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
r"""
Download Depth Anything V2 Small ONNX model to the current directory.

The model will be saved as:
    depth_anything_v2_small.onnx

Place this file next to 3Deflatten_x64.ax / 3Deflatten_x86.ax (or in the
parent folder) and 3Deflatten will find it automatically.

Source: https://huggingface.co/onnx-community/depth-anything-v2-small
"""
import sys
import pathlib
import shutil
import subprocess


REPO_ID        = "onnx-community/depth-anything-v2-small"
MODEL_FILENAME = "onnx/model.onnx"
CANONICAL_NAME = "depth_anything_v2_small.onnx"


def main() -> None:
    # Auto-install huggingface-hub if missing.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface-hub not found -- installing it now...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "huggingface-hub"])
        from huggingface_hub import hf_hub_download

    dest_dir  = pathlib.Path.cwd()
    dest_path = dest_dir / CANONICAL_NAME

    if dest_path.exists():
        print(f"Model already present:\n  {dest_path}")
        print("\nNothing to do.")
        return

    print(f"Downloading {CANONICAL_NAME} from Hugging Face ...")
    print(f"  Repo : {REPO_ID}")
    print(f"  File : {MODEL_FILENAME}")
    print(f"  Dest : {dest_path}\n")

    try:
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=str(dest_dir),
        )
        local_path = pathlib.Path(local)
        if local_path.resolve() != dest_path.resolve():
            shutil.move(str(local_path), str(dest_path))

        # hf_hub_download leaves behind an onnx/ subdirectory (repo structure
        # mirror) and a .cache/ directory (lock + metadata files).
        # Neither is needed now that the model has been moved.
        for leftover in ("onnx", ".cache"):
            p = dest_dir / leftover
            if p.exists():
                shutil.rmtree(p)

        print(f"\n[OK] Model saved to:\n  {dest_path}")
        print("\nPlace this file next to the .ax filter and 3Deflatten will")
        print("find it automatically.  You can also set the path explicitly:")
        print(f'  set DEFLATTEN_MODEL_PATH={dest_path}')

    except Exception as exc:
        print(f"\n[ERROR] Download failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
