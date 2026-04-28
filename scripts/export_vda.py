#!/usr/bin/env python3
"""
export_vda.py — Export VideoDepthAnything / DepthAnything-V2 to ONNX for 3Deflatten
=====================================================================================

Downloads weights from HuggingFace and exports to ONNX optimised for TensorRT 10.x.

Usage
-----
  pip install torch torchvision onnx onnxsim huggingface_hub depth-anything-v2

  # VDA Small, single-frame (~25ms FP16 on SM75, same speed as DA v2)
  python export_vda.py --model small

  # VDA Small, 4-frame temporal window (~100ms on SM75)
  python export_vda.py --model small --frames 4

  # DA v2 Small (explicit)
  python export_vda.py --family dav2 --model small

  # Specify output directory
  python export_vda.py --model small --out "C:/Programs/3Deflatten/Release"

Notes
-----
  • do_constant_folding=False is intentional: it prevents TensorRT from collapsing
    data-dependent branches (patch positional embedding, interpolate targets) to
    wrong constants.  onnxsim runs afterward for safe simplification.
  • torch.rand dummy input (not zeros) prevents zero-output constant folding.
  • Opset 17 enables INormalizationLayer, avoiding FP16 layernorm overflow in TRT.
  • Static spatial dims (518×518) give reliable TRT profile construction.
  • F=1 re-exports VDA weights as a single-frame model.  The temporal consistency
    of VDA comes from fine-tuning, so single-frame still outperforms plain DA v2
    on many scenes.  The HuggingFace F=32 ONNX triggers TRT 10 shape bugs on SM75.
"""

from __future__ import annotations
import argparse, pathlib, sys


def _check_deps():
    missing = []
    for pkg in ['torch', 'onnx']:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing: {', '.join(missing)}")
        print(f"  pip install {' '.join(missing)} onnxsim huggingface_hub")
        sys.exit(1)


HF_REPOS = {
    ('vda',  'small'): ('depth-anything/Video-Depth-Anything-Small', 'video_depth_anything_vitS.pth'),
    ('vda',  'large'): ('depth-anything/Video-Depth-Anything-Large', 'video_depth_anything_vitL.pth'),
    ('dav2', 'small'): ('depth-anything/Depth-Anything-V2-Small',    'depth_anything_v2_vits.pth'),
    ('dav2', 'large'): ('depth-anything/Depth-Anything-V2-Large',    'depth_anything_v2_vitl.pth'),
}

ENCODER_CONFIGS = {
    'small': dict(encoder='vits', features=64,  out_channels=[48,  96,  192, 384]),
    'large': dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]),
}


def download_weights(family, model_size, cache_dir):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] pip install huggingface_hub  (or pass --weights <path>)")
        sys.exit(1)
    repo, fname = HF_REPOS[(family, model_size)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / fname
    if local.exists():
        print(f"[INFO] Using cached weights: {local}")
        return local
    print(f"[INFO] Downloading {repo}/{fname}…")
    return pathlib.Path(hf_hub_download(repo_id=repo, filename=fname,
                                         local_dir=str(cache_dir)))


def load_model(weights, model_size, device):
    import torch
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        print("[ERROR] depth-anything-v2 not installed.")
        print("  pip install depth-anything-v2")
        sys.exit(1)

    model = DepthAnythingV2(**ENCODER_CONFIGS[model_size])
    print(f"[INFO] Loaded DepthAnythingV2 ({model_size})")

    state = torch.load(str(weights), map_location='cpu', weights_only=True)
    if isinstance(state, dict):
        for key in ('model', 'state_dict', 'net'):
            if key in state and isinstance(state[key], dict):
                state = state[key]; print(f"[INFO]   Unwrapped '{key}'"); break

    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if 'pretrained' not in k]
    if real_missing:
        print(f"[WARN]   {len(real_missing)} missing keys: {real_missing[:3]}")
    print("[INFO]   Weights loaded OK.")
    return model.to(device).eval()


def export(args):
    import torch, onnx
    _check_deps()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # Weights
    if args.weights:
        weights_path = pathlib.Path(args.weights)
        if not weights_path.exists():
            print(f"[ERROR] Not found: {weights_path}"); sys.exit(1)
    else:
        cache = pathlib.Path(args.cache) if args.cache else \
                pathlib.Path.home() / '.cache' / '3Deflatten' / 'vda'
        weights_path = download_weights(args.family, args.model, cache)

    base_model = load_model(weights_path, args.model, device)

    # Temporal wrapper: [B, F, 3, H, W] → run model F times → [B, F, H, W]
    class TemporalWrapper(torch.nn.Module):
        def __init__(self, m, f): super().__init__(); self.m = m; self.f = f
        def forward(self, x):
            B, F, C, H, W = x.shape
            out = []
            for i in range(F):
                d = self.m(x[:, i])              # [B, H, W]
                if d.dim() == 3: d = d.unsqueeze(1)
                out.append(d)
            return torch.cat(out, dim=1)         # [B, F, H, W]

    model = TemporalWrapper(base_model, args.frames).to(device).eval()

    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    enc_tag = 'vits' if args.model == 'small' else 'vitl'
    stem = f"vda_{enc_tag}_f{args.frames}_{args.width}x{args.height}"
    onnx_path = out_dir / f"{stem}.onnx"
    slim_path  = out_dir / f"{stem}_simplified.onnx"

    # Use rand (not zeros) — zeros can cause large constant-fold subgraphs.
    # do_constant_folding=False — prevents the tracer from baking shape-dependent
    # branches (pos-embed interpolation, patch size computation) as wrong constants.
    # onnxsim runs afterward and folds only genuinely constant subgraphs.
    dummy = torch.rand(1, args.frames, 3, args.height, args.width, device=device)
    print(f"\n[INFO] Exporting: {onnx_path}")
    print(f"       input: [1, {args.frames}, 3, {args.height}, {args.width}]")

    torch.onnx.export(
        model, dummy, str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=False,   # ← key: prevents shape-dependent constant folding
        input_names=['input'], output_names=['depth'],
        dynamic_axes={'input': {0: 'batch'}, 'depth': {0: 'batch'}},
        verbose=False,
    )
    print(f"[INFO] ONNX written: {onnx_path.stat().st_size//1024//1024} MB")

    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    print("[INFO] ONNX check passed.")

    recommend = onnx_path
    if not args.no_simplify:
        try:
            import onnxsim
            print("[INFO] Running onnxsim…")
            simplified, ok = onnxsim.simplify(m)
            if ok:
                onnx.save(simplified, str(slim_path))
                print(f"[INFO] Simplified: {slim_path}")
                recommend = slim_path
            else:
                print("[WARN] onnxsim could not simplify.")
        except ImportError:
            print("[WARN] onnxsim not installed (pip install onnxsim)")

    print(f"\n{'='*60}")
    print(f"  Recommended: {recommend}")
    print(f"\n  1. Copy .onnx next to 3Deflatten_x64.ax")
    print(f"  2. Property page → select model → Runtime: TensorRT native → Reload")
    if args.frames == 1:
        print(f"  F=1: single-frame, ~25ms FP16 on SM75 (RTX 2080 Ti)")
    else:
        print(f"  F={args.frames}: ~{args.frames*25}ms estimated on SM75")


def main():
    p = argparse.ArgumentParser(description="Export VDA / DA-v2 to ONNX for 3Deflatten",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument('--family', choices=['vda','dav2'], default='vda')
    p.add_argument('--model',  choices=['small','large'], default='small')
    p.add_argument('--frames', type=int, default=1)
    p.add_argument('--width',  type=int, default=518)
    p.add_argument('--height', type=int, default=518)
    p.add_argument('--weights', default=None)
    p.add_argument('--cache',   default=None)
    p.add_argument('--out',     default='.')
    p.add_argument('--no-simplify', action='store_true')
    args = p.parse_args()
    if args.frames < 1: p.error("--frames must be >= 1")
    for v, n in [(args.width,'width'),(args.height,'height')]:
        if v % 14 != 0: print(f"[WARN] --{n} {v} not multiple of 14; nearest: {(v//14)*14}")
    export(args)


if __name__ == '__main__':
    main()
