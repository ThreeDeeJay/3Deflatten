#!/usr/bin/env python3
"""
export_vda.py — Export VideoDepthAnything to ONNX for 3Deflatten
=================================================================

Prerequisites
-------------
1. Clone the Video-Depth-Anything repository:
     git clone https://github.com/DepthAnything/Video-Depth-Anything

2. Download weights (free HF account required):
     https://huggingface.co/depth-anything/Video-Depth-Anything-Small
     → video_depth_anything_vitS.pth

3. pip install torch torchvision onnx onnxsim

Usage
-----
  python export_vda.py ^
      --repo    C:/path/Video-Depth-Anything ^
      --weights C:/path/video_depth_anything_vitS.pth ^
      --frames  1 ^
      --out     C:/Programs/3Deflatten/Release

  --frames 1  → single-frame, same speed as DA v2 (~25ms FP16 on RTX 2080 Ti)
  --frames 4  → 4-frame temporal window (~100ms on RTX 2080 Ti)

Architecture note
-----------------
VideoDepthAnything.forward() natively accepts [B, T, 3, H, W] and returns
[B, T, H, W].  We export it directly with a [1, F, 3, H, W] dummy input —
no wrapper needed.  F is the temporal window stored in the exported ONNX.

Root cause of previous corruption
----------------------------------
  • Wrong model class: DepthAnythingV2 (DA v2 backbone) was used instead of
    VideoDepthAnything, causing 64 missing depth-head keys and zero output.
  • Wrong wrapper: per-frame slicing broke the model's 5D input expectation.
  • int() in dpt.py: converted tensor shapes to constants during tracing,
    causing wrong spatial sizes for inputs other than the trace dummy.
"""

from __future__ import annotations
import argparse, inspect, pathlib, re, sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _die(msg):
    sys.exit(f"\n[ERROR] {msg}")


def _find_repo(hint):
    candidates = []
    if hint: candidates.append(pathlib.Path(hint))
    candidates.append(pathlib.Path(__file__).parent)
    candidates.append(pathlib.Path(__file__).parent.parent / 'Video-Depth-Anything')
    candidates.append(pathlib.Path.cwd() / 'Video-Depth-Anything')
    for p in candidates:
        if (p / 'video_depth_anything').is_dir():
            return p.resolve()
    _die(
        "Could not find Video-Depth-Anything repository.\n\n"
        "  Clone:   git clone https://github.com/DepthAnything/Video-Depth-Anything\n"
        "  Then:    python export_vda.py --repo C:/path/Video-Depth-Anything --weights ..."
    )


def _add_path(p):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _patch_interpolate(module_names):
    """
    Remove int() from F.interpolate calls in DPT/DINOv2 modules.
    int() converts a tensor to Python int during tracing, baking the spatial
    size as a constant — corrupting depth output for all non-trace-time sizes.
    """
    import importlib
    patched_any = False
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except OSError:
            continue
        new_src = re.sub(r'int\(patch_([hw]) \* 14\)', r'patch_\1 * 14', src)
        if new_src != src:
            n = len(re.findall(r'int\(patch_[hw] \* 14\)', src))
            exec(compile(new_src, getattr(mod, '__file__', mod_name), 'exec'), vars(mod))
            print(f"[PATCH] {mod_name}: removed {n} int() call(s) around patch_h/w * 14")
            patched_any = True
    if not patched_any:
        print("[INFO] No int(patch_h/w * 14) found to patch (may already be clean).")


# ---------------------------------------------------------------------------
# Encoder configs
# ---------------------------------------------------------------------------

ENCODER_CONFIGS = {
    'vits': dict(encoder='vits', features=64,  out_channels=[48,  96,  192, 384]),
    'vitb': dict(encoder='vitb', features=128, out_channels=[96,  192, 384, 768]),
    'vitl': dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]),
}

def _guess_encoder(weights_path):
    n = weights_path.stem.lower()
    if 'vitl' in n: return 'vitl'
    if 'vitb' in n: return 'vitb'
    return 'vits'


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(repo, weights_path, encoder, device):
    import torch
    _add_path(repo)

    # Import from video_depth.py — the native 5D [B,T,3,H,W] model
    try:
        from video_depth_anything.video_depth import VideoDepthAnything
        print(f"[INFO] Using video_depth_anything.video_depth.VideoDepthAnything")
    except ImportError as e:
        _die(
            f"Could not import VideoDepthAnything from {repo}/video_depth_anything/video_depth.py\n"
            f"Error: {e}\n\n"
            "Make sure you cloned the full repo:\n"
            "  git clone https://github.com/DepthAnything/Video-Depth-Anything"
        )

    model = VideoDepthAnything(**ENCODER_CONFIGS[encoder])
    print(f"[INFO] VideoDepthAnything({encoder})")

    state = torch.load(str(weights_path), map_location='cpu', weights_only=True)
    if isinstance(state, dict):
        for k in ('model', 'state_dict', 'net'):
            if k in state and isinstance(state[k], dict):
                state = state[k]; print(f"[INFO]   Unwrapped key '{k}'"); break

    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if 'pretrained' not in k]
    if real_missing:
        if len(real_missing) > 10:
            _die(
                f"{len(real_missing)} missing keys — likely encoder/size mismatch.\n"
                f"  weights : {weights_path.name}\n"
                f"  encoder : {encoder}\n"
                f"  Missing : {real_missing[:5]}\n"
                "  Try --encoder vits / vitb / vitl to match the weights."
            )
        print(f"[WARN]   {len(real_missing)} missing: {real_missing[:5]}")
    print(f"[INFO]   Weights loaded OK  ({len(state)} tensors, "
          f"{len(missing)} missing, {len(unexpected)} unexpected)")
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(args):
    import torch, onnx

    repo = _find_repo(args.repo)
    _add_path(repo)
    print(f"[INFO] Repo: {repo}")

    # Patch int() BEFORE importing the model (patches class methods in-place)
    _patch_interpolate([
        'depth_anything_v2.dpt',
        'video_depth_anything.dpt',
        'video_depth_anything.video_depth',
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    weights_path = pathlib.Path(args.weights)
    if not weights_path.exists():
        _die(
            f"Weights not found: {weights_path}\n"
            "  Download from: https://huggingface.co/depth-anything/Video-Depth-Anything-Small"
        )

    encoder = args.encoder or _guess_encoder(weights_path)
    print(f"[INFO] Encoder: {encoder}  (override with --encoder vits/vitb/vitl)")

    model = load_model(repo, weights_path, encoder, device)

    H, W, F = args.height, args.width, args.frames
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"vda_{encoder}_f{F}_{W}x{H}"
    onnx_path = out_dir / f"{stem}.onnx"
    slim_path  = out_dir / f"{stem}_simplified.onnx"

    # VideoDepthAnything natively takes [B, T, 3, H, W] and returns [B, T, H, W].
    # Pass the full 5D tensor directly — no wrapper required.
    # torch.rand prevents constant-folding the network to zero output.
    dummy = torch.rand(1, F, 3, H, W, device=device)

    print(f"\n[INFO] Exporting: {onnx_path}")
    print(f"       input : [1, {F}, 3, {H}, {W}]")
    print(f"       output: [1, {F}, {H}, {W}]  (expected)")

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,           # INormalizationLayer — avoids FP16 overflow in TRT
        do_constant_folding=True,   # safe after int() patch
        input_names=['input'],
        output_names=['depth'],
        dynamic_axes={
            'input': {0: 'batch'},  # batch dynamic; T and spatial are static
            'depth': {0: 'batch'},
        },
        verbose=False,
    )
    sz_mb = onnx_path.stat().st_size // 1024 // 1024
    print(f"[INFO] ONNX written: {sz_mb} MB")

    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    print("[INFO] ONNX model check passed.")

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
                print("[WARN] onnxsim could not simplify — use unsimplified file.")
        except ImportError:
            print("[WARN] pip install onnxsim  to enable simplification")

    print(f"\n{'='*64}")
    print(f"  Recommended : {recommend}")
    print(f"\n  1. Copy .onnx next to 3Deflatten_x64.ax")
    print(f"  2. Property page → select model → Runtime: TensorRT native → Reload")
    if F == 1:
        print(f"\n  F=1 : single-frame (~25ms FP16 on RTX 2080 Ti)")
        print(f"         Same inference speed as DA v2; better weights from VDA fine-tuning.")
    else:
        print(f"\n  F={F} : {F}-frame temporal window (~{F*25}ms estimated on RTX 2080 Ti)")
        print(f"         The model sees {F} consecutive frames simultaneously,")
        print(f"         providing built-in temporal consistency.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Export VideoDepthAnything to ONNX for 3Deflatten / TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--repo', default=None,
                   help="Path to cloned Video-Depth-Anything repo "
                        "(auto-detected if script is placed inside it)")
    p.add_argument('--weights', required=True,
                   help="Path to .pth weights  (video_depth_anything_vitS.pth etc.)")
    p.add_argument('--encoder', default=None, choices=['vits', 'vitb', 'vitl'],
                   help="Encoder size (auto-detected from weights filename if omitted)")
    p.add_argument('--frames', type=int, default=1,
                   help="Temporal window F — number of frames per inference call "
                        "(default: 1 = single-frame, fastest)")
    p.add_argument('--width',  type=int, default=518,
                   help="Input width  in pixels (default: 518, native VDA resolution)")
    p.add_argument('--height', type=int, default=518,
                   help="Input height in pixels (default: 518, native VDA resolution)")
    p.add_argument('--out', default='.',
                   help="Output directory for .onnx files (default: current dir)")
    p.add_argument('--no-simplify', action='store_true',
                   help="Skip onnxsim simplification step")
    args = p.parse_args()

    if args.frames < 1:
        p.error("--frames must be >= 1")
    for val, name in [(args.width, 'width'), (args.height, 'height')]:
        if val % 14 != 0:
            print(f"[WARN] --{name} {val} is not a multiple of 14 (ViT patch size). "
                  f"Nearest valid: {(val // 14) * 14}")
    export(args)


if __name__ == '__main__':
    main()
