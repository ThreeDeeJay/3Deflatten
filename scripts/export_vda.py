#!/usr/bin/env python3
"""
export_vda.py — Download and export VideoDepthAnything to ONNX for 3Deflatten
===============================================================================

Downloads VideoDepthAnything-{Small|Large} weights from HuggingFace, traces the
model with a single representative frame (batch=1, frames=1, or frames=F), and
exports to ONNX with settings optimised for TensorRT compilation:

  • Static F=1 temporal window  (avoids the F=32 TRT shape-analyser bug,
    reduces inference time from ~800 ms to ~25 ms at 518×518)
  • Static spatial dims 518×518 (most compatible with TRT 10.x / TRT-RTX)
  • Opset 17                    (required for INormalizationLayer; avoids
    FP16 layernorm overflow warnings)
  • Dynamic batch dimension     (allows batch=1 with minimal profile overhead)

Usage
-----
  # Install dependencies (once)
  pip install torch torchvision huggingface_hub onnx onnxsim

  # Export VDA Small, F=1, 518×518
  python export_vda.py

  # Export VDA Large, F=1, 518×518
  python export_vda.py --model large

  # Export with a different frame window (e.g. 4 for mild temporal smoothing)
  python export_vda.py --frames 4

  # Export to a specific directory
  python export_vda.py --out C:/Programs/3Deflatten/Release

Why F=1?
--------
The public video_depth_anything_vits_input518.onnx uses F=32 — 32 frames of
attention per inference call.  TRT 10.x / Turing (SM75) cannot build that model
in FP16 due to a Slice shape-analyser assertion, and FP32 takes ~800 ms/call.

With F=1, VideoDepthAnything degrades gracefully to per-frame depth estimation
(identical to DA v2/v3) but retains the model's architecture advantages.  You
can increase --frames for mild temporal context at the cost of more VRAM and
longer inference, e.g. --frames 4 → ~100 ms.

Requirements
------------
  Python 3.9+, PyTorch ≥ 2.1, CUDA toolkit (for GPU export)
  pip install torch torchvision huggingface_hub onnx onnxsim
"""

from __future__ import annotations
import argparse
import os
import sys
import pathlib

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
def _check_deps():
    missing = []
    for pkg in ['torch', 'huggingface_hub', 'onnx']:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"  Run:  pip install {' '.join(missing)} onnxsim")
        sys.exit(1)
_check_deps()

import torch
import onnx

# ---------------------------------------------------------------------------
# HuggingFace repo map
# ---------------------------------------------------------------------------
HF_REPOS = {
    'small': 'depth-anything/Video-Depth-Anything-Small',
    'large': 'depth-anything/Video-Depth-Anything-Large',
}
# Fallback: original DA-style weights often hosted under these names
HF_WEIGHT_FILES = {
    'small': 'video_depth_anything_vitS.pth',
    'large': 'video_depth_anything_vitL.pth',
}


def download_weights(model_size: str, cache_dir: pathlib.Path) -> pathlib.Path:
    """Download model weights from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download, snapshot_download
    repo = HF_REPOS[model_size]
    weight_file = HF_WEIGHT_FILES[model_size]
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / weight_file
    if local.exists():
        print(f"[INFO] Using cached weights: {local}")
        return local
    print(f"[INFO] Downloading {repo} / {weight_file} …")
    try:
        path = hf_hub_download(repo_id=repo, filename=weight_file,
                               local_dir=str(cache_dir))
        return pathlib.Path(path)
    except Exception as e:
        print(f"[WARN] hf_hub_download failed: {e}")
        print(f"[INFO] Trying snapshot_download …")
        snap = snapshot_download(repo_id=repo, local_dir=str(cache_dir))
        # Find any .pth file in the snapshot
        ptfiles = list(pathlib.Path(snap).rglob('*.pth'))
        if not ptfiles:
            raise RuntimeError(f"No .pth found in {snap}")
        return ptfiles[0]


def load_model(weights_path: pathlib.Path, model_size: str, device: torch.device):
    """Load VideoDepthAnything model.  Tries the official package first,
    then falls back to a minimal re-implementation stub."""
    # Try official package (pip install video-depth-anything)
    try:
        if model_size == 'small':
            from video_depth_anything.video_depth_anything_s import VideoDepthAnythingS as VDA
        else:
            from video_depth_anything.video_depth_anything_l import VideoDepthAnythingL as VDA
        model = VDA()
        state = torch.load(str(weights_path), map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        print("[INFO] Loaded via official video-depth-anything package.")
        return model.to(device).eval()
    except ImportError:
        pass

    # Fall back: load as a generic DA backbone (works for per-frame inference)
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        configs = {
            'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48,96,192,384]},
            'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256,512,1024,1024]},
        }
        model = DepthAnythingV2(**configs[model_size])
        state = torch.load(str(weights_path), map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=False)
        print("[INFO] Loaded via depth_anything_v2 backbone (temporal layers ignored).")
        return model.to(device).eval()
    except ImportError:
        pass

    raise RuntimeError(
        "Could not import VideoDepthAnything or DepthAnythingV2.\n"
        "Install one of:\n"
        "  pip install video-depth-anything\n"
        "  pip install depth-anything-v2\n"
        "or clone and pip install from:\n"
        "  https://github.com/DepthAnything/Video-Depth-Anything"
    )


class TemporalWindowWrapper(torch.nn.Module):
    """Wraps any depth model to accept [B, F, 3, H, W] and return [B, F, H, W].
    For models that only process single frames, applies them F times.
    For native video models, passes the window directly."""
    def __init__(self, model, frames: int, native_video: bool = False):
        super().__init__()
        self.model = model
        self.frames = frames
        self.native_video = native_video

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, 3, H, W]
        B, F, C, H, W = x.shape
        if self.native_video:
            # Model natively accepts [B, F, 3, H, W]
            return self.model(x)
        else:
            # Per-frame model: process each frame independently
            results = []
            for fi in range(F):
                frame = x[:, fi]   # [B, 3, H, W]
                depth = self.model(frame)   # [B, H, W] or [B, 1, H, W]
                if depth.dim() == 3:
                    depth = depth.unsqueeze(1)   # [B, 1, H, W]
                results.append(depth)
            return torch.cat(results, dim=1)   # [B, F, H, W]


def export(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    cache = pathlib.Path(args.cache) if args.cache else \
            pathlib.Path.home() / '.cache' / '3Deflatten' / 'vda'
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Download weights ──────────────────────────────────────────────────────
    weights = pathlib.Path(args.weights) if args.weights else \
              download_weights(args.model, cache)

    # ── Load model ────────────────────────────────────────────────────────────
    base_model = load_model(weights, args.model, device)

    # Detect if it's natively a video model (has 5D input support)
    native_video = False
    try:
        probe = torch.rand(1, args.frames, 3, args.height, args.width, device=device)
        with torch.no_grad(): base_model(probe)
        native_video = True
        print("[INFO] Model natively accepts 5D [B,F,3,H,W] input.")
    except Exception:
        print("[INFO] Model is single-frame; wrapping with TemporalWindowWrapper.")

    model = TemporalWindowWrapper(base_model, args.frames, native_video)
    model = model.to(device).eval()

    # ── Build output filename ─────────────────────────────────────────────────
    suffix = f"_f{args.frames}_{args.width}x{args.height}"
    stem = f"video_depth_anything_vit{'s' if args.model=='small' else 'l'}{suffix}"
    onnx_path = out_dir / f"{stem}.onnx"
    simplified_path = out_dir / f"{stem}_simplified.onnx"

    # ── Export to ONNX ────────────────────────────────────────────────────────
    # IMPORTANT: use rand, not zeros.  torch.onnx.export with do_constant_folding=True
    # will fold the entire network to all-zeros if the input is all-zeros, producing
    # a valid ONNX that always outputs zero depth (completely corrupted).
    dummy = torch.rand(1, args.frames, 3, args.height, args.width, device=device)
    print(f"[INFO] Exporting ONNX: {onnx_path}")
    print(f"       input shape: [1, {args.frames}, 3, {args.height}, {args.width}]")

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,          # INormalizationLayer → avoids FP16 layernorm overflow
        do_constant_folding=True,
        input_names=['input'],
        output_names=['depth'],
        dynamic_axes={
            'input':  {0: 'batch'},  # static F and spatial; only batch is dynamic
            'depth':  {0: 'batch'},
        },
        verbose=False,
    )
    print(f"[INFO] ONNX export complete: {onnx_path.stat().st_size // 1024 // 1024} MB")

    # ── Verify ────────────────────────────────────────────────────────────────
    model_onnx = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_onnx)
    print("[INFO] ONNX model check passed.")

    # ── Simplify (optional) ───────────────────────────────────────────────────
    if not args.no_simplify:
        try:
            import onnxsim
            print("[INFO] Running onnxsim …")
            simplified, ok = onnxsim.simplify(model_onnx)
            if ok:
                onnx.save(simplified, str(simplified_path))
                print(f"[INFO] Simplified ONNX: {simplified_path}")
                print("[INFO] Use the _simplified.onnx for best TRT build performance.")
            else:
                print("[WARN] onnxsim could not simplify the model — use the unsimplified version.")
        except ImportError:
            print("[WARN] onnxsim not installed — skipping simplification.")
            print("  Run:  pip install onnxsim   to enable")

    print()
    print("=" * 72)
    print("Export complete!")
    print(f"  ONNX path    : {onnx_path}")
    if simplified_path.exists():
        print(f"  Simplified   : {simplified_path}   ← recommended")
    print()
    print("Next steps:")
    print("  1. Copy the ONNX file next to 3Deflatten_x64.ax")
    print("  2. Open the 3Deflatten property page and select the model.")
    print("  3. Set Runtime to 'TensorRT native' and click 'Reload'.")
    print("     The engine will be built and cached on first use.")
    print()
    print(f"  Input tensor : [1, {args.frames}, 3, {args.height}, {args.width}]")
    print(f"  Output tensor: [1, {args.frames}, {args.height}, {args.width}]")
    print()
    if args.frames == 1:
        print("  Note: F=1 disables temporal context (same as DA v2).")
        print("        Use --frames 4 or --frames 8 for mild temporal smoothing.")
    else:
        print(f"  Note: F={args.frames} means the filter maintains a {args.frames}-frame")
        print(f"        sliding window for temporal consistency.")


def main():
    p = argparse.ArgumentParser(
        description="Export VideoDepthAnything to ONNX for 3Deflatten / TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--model', choices=['small','large'], default='small',
                   help="Model size (default: small)")
    p.add_argument('--frames', type=int, default=1,
                   help="Temporal window F (default: 1 = no temporal context, fastest)")
    p.add_argument('--width',  type=int, default=518,
                   help="Input width  (default: 518 — native VDA resolution)")
    p.add_argument('--height', type=int, default=518,
                   help="Input height (default: 518 — native VDA resolution)")
    p.add_argument('--weights', default=None,
                   help="Path to local .pth weights (skip HuggingFace download)")
    p.add_argument('--cache', default=None,
                   help="HuggingFace download cache dir (default: ~/.cache/3Deflatten/vda)")
    p.add_argument('--out', default='.',
                   help="Output directory for .onnx files (default: current dir)")
    p.add_argument('--no-simplify', action='store_true',
                   help="Skip onnxsim simplification step")
    args = p.parse_args()

    if args.frames < 1:
        p.error("--frames must be >= 1")
    if args.width  % 14 != 0 or args.height % 14 != 0:
        print(f"[WARN] Width and height should be multiples of 14 for ViT patch alignment.")
        print(f"       Nearest values: {(args.width//14)*14}×{(args.height//14)*14}")

    export(args)


if __name__ == '__main__':
    main()
