#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Download Depth Anything V2 Small ONNX model to the default 3Deflatten search path.

Requirements:
    pip install huggingface_hub

The model will be saved to:
    %APPDATA%\3Deflatten\models\depth_anything_v2_small.onnx

3Deflatten will find it automatically on next use.

Source: https://huggingface.co/onnx-community/depth-anything-v2-small
"""
import os
import sys
import pathlib
import shutil


REPO_ID        = "onnx-community/depth-anything-v2-small"
MODEL_FILENAME = "onnx/model.onnx"
CANONICAL_NAME = "depth_anything_v2_small.onnx"


def appdata_dir() -> pathlib.Path:
    """Return %APPDATA% on Windows, ~/.config on other platforms."""
    if os.name == "nt":
        return pathlib.Path(os.environ.get("APPDATA", "~")).expanduser()
    return pathlib.Path.home() / ".config"


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("       pip install huggingface_hub")
        sys.exit(1)

    dest_dir = appdata_dir() / "3Deflatten" / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = dest_dir / CANONICAL_NAME

    if canonical_path.exists():
        print(f"Model already present:\n  {canonical_path}")
        print("\nNothing to do.")
        return

    print(f"Downloading {CANONICAL_NAME} from Hugging Face ...")
    print(f"  Repo : {REPO_ID}")
    print(f"  File : {MODEL_FILENAME}")
    print(f"  Dest : {dest_dir}\n")

    try:
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=str(dest_dir),
        )
        local_path = pathlib.Path(local)
        if local_path.name != CANONICAL_NAME:
            shutil.move(str(local_path), str(canonical_path))

        print(f"\n[OK] Model saved to:\n  {canonical_path}")
        print("\n3Deflatten will find this model automatically.")
        print("You can also set the path explicitly:")
        print(f'  set DEFLATTEN_MODEL_PATH={canonical_path}')

    except Exception as exc:
        print(f"\n[ERROR] Download failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
