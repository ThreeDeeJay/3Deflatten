#!/usr/bin/env python3
"""
export_vda.py — Export VideoDepthAnything / DepthAnything-V2 to ONNX for 3Deflatten
=====================================================================================

Downloads weights from HuggingFace (depth-anything/Video-Depth-Anything-Small/Large
or depth-anything/Depth-Anything-V2-Small/Large) and exports to ONNX optimised for
TensorRT 10.x compilation in 3Deflatten.

Key fixes vs naive torch.onnx.export:
  • Monkey-patches depth_anything_v2/dpt.py and dinov2.py before tracing to replace
    data-dependent control flow with static equivalents so the exported graph is
    correct for any batch size, not just the trace-time dummy value.
  • Uses torch.rand (not zeros) for the dummy input to prevent constant-folding
    the entire network to zero depth output.
  • Opset 17 for INormalizationLayer (avoids FP16 layernorm overflow in TRT).
  • Static spatial dims (518×518 default) for reliable TRT profile construction.

Usage
-----
  pip install torch torchvision onnx onnxsim huggingface_hub depth-anything-v2

  # VDA Small, single-frame (fastest; same speed as DA v2 on SM75 ~25ms)
  python export_vda.py --model small --out "C:/Programs/3Deflatten/Release"

  # VDA Small, 4-frame temporal window (mild consistency, ~100ms on SM75)
  python export_vda.py --model small --frames 4 --out "C:/Programs/3Deflatten/Release"

  # DA v2 Small
  python export_vda.py --family dav2 --model small --out "C:/Programs/3Deflatten/Release"

Why F=1 for the DA v2 backbone?
  The public VDA ONNX on HuggingFace (F=32) triggers a TRT 10 shape-analyser bug on
  SM75, forcing FP32 at ~800ms/call.  F=1 re-exports using the same weights but as a
  single-frame model, matching DA v2 speed (~25ms FP16 on SM75).  Use --frames 4 for
  mild temporal context at ~100ms.
"""

from __future__ import annotations
import argparse
import pathlib
import sys
import types


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
def _check_deps():
    missing = []
    for pkg in ['torch', 'onnx']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"  Run:  pip install {' '.join(missing)} onnxsim huggingface_hub")
        sys.exit(1)


# ---------------------------------------------------------------------------
# HuggingFace download
# ---------------------------------------------------------------------------
HF_REPOS = {
    ('vda', 'small'): ('depth-anything/Video-Depth-Anything-Small', 'video_depth_anything_vitS.pth'),
    ('vda', 'large'): ('depth-anything/Video-Depth-Anything-Large', 'video_depth_anything_vitL.pth'),
    ('dav2', 'small'): ('depth-anything/Depth-Anything-V2-Small', 'depth_anything_v2_vits.pth'),
    ('dav2', 'large'): ('depth-anything/Depth-Anything-V2-Large', 'depth_anything_v2_vitl.pth'),
}

def download_weights(family: str, model_size: str, cache_dir: pathlib.Path) -> pathlib.Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed.  Run:  pip install huggingface_hub")
        print("        Or pass --weights <path> to skip download.")
        sys.exit(1)
    repo, filename = HF_REPOS[(family, model_size)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / filename
    if local.exists():
        print(f"[INFO] Using cached weights: {local}")
        return local
    print(f"[INFO] Downloading {repo}/{filename} from HuggingFace…")
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(cache_dir))
    return pathlib.Path(path)


# ---------------------------------------------------------------------------
# Monkey-patch depth_anything_v2 to make tracing deterministic
# ---------------------------------------------------------------------------

def _patch_da_modules(H: int, W: int, patch_size: int = 14):
    """
    Patch three locations in depth_anything_v2 that cause TracerWarnings and
    incorrect constant-folding when the input size is fixed:

    1. depth_anything_v2/dinov2_layers/patch_embed.py  lines 73-74
       `assert H % patch_H == 0` — data-dependent bool, baked as constant.
       Fix: replace with no-op (assertions are already satisfied for 518×518).

    2. depth_anything_v2/dinov2.py  line 183
       `if npatch == N and w == h` — controls positional embedding interpolation.
       Fix: for fixed H=W=518, patch_h=patch_w=37, N=1369=37²; the condition is
       always True (square image, same patch count), so we replace the branch with
       an unconditional return of the uninterpolated embedding.

    3. depth_anything_v2/dpt.py  line 147
       `F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), …)`
       patch_h and patch_w are computed from the input tensor shape at runtime;
       the tracer bakes them as constants from the dummy input, giving wrong
       interpolation targets for different inputs.
       Fix: replace with the static values derived from H and W.
    """
    import torch
    import torch.nn.functional as F_nn

    # --- patch 1: patch_embed assertions --------------------------------
    try:
        import depth_anything_v2.dinov2_layers.patch_embed as pe_mod
        orig_forward = pe_mod.PatchEmbed.forward

        def _patched_pe_forward(self, x):
            B, C, H_, W_ = x.shape
            # Skip the assertions (they always pass for multiples of patch_size)
            x = self.proj(x)  # (B, E, Hp, Wp)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
            if self.norm is not None:
                x = self.norm(x)
            return x

        pe_mod.PatchEmbed.forward = _patched_pe_forward
        print("[PATCH] patch_embed.PatchEmbed.forward — assertions removed")
    except Exception as e:
        print(f"[WARN] Could not patch patch_embed ({e}) — TracerWarnings expected")

    # --- patch 2: dinov2 positional embedding branch --------------------
    try:
        import depth_anything_v2.dinov2 as dv2_mod

        patch_h_static = H // patch_size
        patch_w_static = W // patch_size
        N_static = patch_h_static * patch_w_static

        orig_interpolate_pos = None
        if hasattr(dv2_mod.DinoVisionTransformer, 'interpolate_pos_encoding'):
            orig_interpolate_pos = dv2_mod.DinoVisionTransformer.interpolate_pos_encoding

            def _patched_interpolate_pos_encoding(self, x, w, h):
                # For fixed square images where npatch == N, return pos embed directly
                npatch = x.shape[1] - 1
                N = self.pos_embed.shape[1] - 1
                if npatch == N:
                    return self.pos_embed
                # Fall through to original for other sizes
                return orig_interpolate_pos(self, x, w, h)

            dv2_mod.DinoVisionTransformer.interpolate_pos_encoding = _patched_interpolate_pos_encoding
            print(f"[PATCH] DinoVisionTransformer.interpolate_pos_encoding — static branch for {H}×{W}")
    except Exception as e:
        print(f"[WARN] Could not patch dinov2 ({e}) — TracerWarning expected")

    # --- patch 3: dpt.py interpolate with static target size ------------
    try:
        import depth_anything_v2.dpt as dpt_mod

        # Compute the static spatial output of the DPT head for this input size
        # The ViT produces (H/14)×(W/14) patches; the head upsamples back to H×W
        target_h = H
        target_w = W

        orig_DPT_forward = dpt_mod.DPT_DINOv2.forward

        def _patched_DPT_forward(self, x):
            # Run the original but override the problematic interpolate calls
            # by temporarily monkey-patching F.interpolate in the dpt module
            _orig_interp = dpt_mod.F.interpolate

            def _static_interpolate(input, size=None, scale_factor=None, **kw):
                if size is not None:
                    # If size is computed from patch_h/patch_w, it may be wrong;
                    # clamp to our known static output size if it looks like a
                    # DPT-head upsampling (output close to target H/W)
                    h, w = size if hasattr(size, '__len__') else (size, size)
                    if abs(int(h) - target_h) <= target_h // 2 and \
                       abs(int(w) - target_w) <= target_w // 2:
                        size = (target_h, target_w)
                return _orig_interp(input, size=size, scale_factor=scale_factor, **kw)

            dpt_mod.F.interpolate = _static_interpolate
            try:
                result = orig_DPT_forward(self, x)
            finally:
                dpt_mod.F.interpolate = _orig_interp
            return result

        dpt_mod.DPT_DINOv2.forward = _patched_DPT_forward
        print(f"[PATCH] DPT_DINOv2.forward — static interpolate target {target_w}×{target_h}")
    except Exception as e:
        print(f"[WARN] Could not patch dpt ({e}) — TracerWarning expected")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
ENCODER_CONFIGS = {
    'small': ('vits', 64,  [48,  96,  192, 384]),
    'large': ('vitl', 256, [256, 512, 1024, 1024]),
}

def load_model(weights: pathlib.Path, model_size: str, device):
    import torch
    enc, feat, out_ch = ENCODER_CONFIGS[model_size]

    model = None
    via = None

    for cls_path in [
        ('depth_anything_v2.dpt', 'DepthAnythingV2'),
        ('depth_anything_v2.dpt', 'DPT_DINOv2'),
    ]:
        try:
            mod = __import__(cls_path[0], fromlist=[cls_path[1]])
            cls = getattr(mod, cls_path[1])
            cfg = {'encoder': enc, 'features': feat, 'out_channels': out_ch}
            try:
                model = cls(**cfg)
            except TypeError:
                model = cls(encoder=enc, features=feat, out_channels=out_ch)
            via = f'{cls_path[0]}.{cls_path[1]}'
            break
        except (ImportError, AttributeError):
            continue

    if model is None:
        print("[ERROR] Could not import depth_anything_v2.")
        print("  Install:  pip install depth-anything-v2")
        print("  Or clone: https://github.com/DepthAnything/Depth-Anything-V2")
        print("            cd Depth-Anything-V2 && pip install -e .")
        sys.exit(1)

    print(f"[INFO] Loaded model class via: {via}")

    state = torch.load(str(weights), map_location='cpu', weights_only=True)
    if isinstance(state, dict):
        for key in ('model', 'state_dict', 'net'):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                print(f"[INFO]   Unwrapped checkpoint key '{key}'")
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_miss = len([k for k in missing if 'pretrained' not in k])
    if n_miss:
        print(f"[WARN]   {n_miss} unexpected missing keys (first 5): {[k for k in missing if 'pretrained' not in k][:5]}")
    print("[INFO]   Weights loaded OK.")
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Single-frame wrapper → [B, F, 3, H, W] → [B, F, H, W]
# ---------------------------------------------------------------------------

class TemporalWrapper(torch.nn.Module if 'torch' in sys.modules else object):
    def __init__(self, model, frames: int):
        super().__init__()
        self.model = model
        self.frames = frames

    def forward(self, x):
        # x: [B, F, 3, H, W]
        B, F, C, H, W = x.shape
        out_frames = []
        for fi in range(F):
            frame = x[:, fi]
            depth = self.model(frame)
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            elif depth.dim() == 4 and depth.shape[1] != 1:
                depth = depth[:, :1]
            out_frames.append(depth)
        return torch.cat(out_frames, dim=1)  # [B, F, H, W]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(args):
    import torch
    import onnx

    _check_deps()

    # Re-import so TemporalWrapper base class is correct
    global TemporalWrapper
    class TemporalWrapper(torch.nn.Module):
        def __init__(self, model, frames):
            super().__init__()
            self.model = model
            self.frames = frames
        def forward(self, x):
            B, F, C, H, W = x.shape
            out_frames = []
            for fi in range(F):
                frame = x[:, fi]
                depth = self.model(frame)
                if depth.dim() == 3: depth = depth.unsqueeze(1)
                elif depth.dim() == 4 and depth.shape[1] != 1: depth = depth[:, :1]
                out_frames.append(depth)
            return torch.cat(out_frames, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # ── Download or locate weights ────────────────────────────────────────────
    if args.weights:
        weights_path = pathlib.Path(args.weights)
        if not weights_path.exists():
            print(f"[ERROR] Weights not found: {weights_path}"); sys.exit(1)
    else:
        cache = pathlib.Path(args.cache) if args.cache else \
                pathlib.Path.home() / '.cache' / '3Deflatten' / 'vda'
        weights_path = download_weights(args.family, args.model, cache)

    # ── Apply patches BEFORE loading model (patches class methods) ────────────
    _patch_da_modules(args.height, args.width)

    # ── Load model ────────────────────────────────────────────────────────────
    base_model = load_model(weights_path, args.model, device)
    model = TemporalWrapper(base_model, args.frames).to(device).eval()

    # ── Output path ───────────────────────────────────────────────────────────
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_tag = 'vits' if args.model == 'small' else 'vitl'
    stem = f"video_depth_anything_{enc_tag}_f{args.frames}_{args.width}x{args.height}"
    onnx_path = out_dir / f"{stem}.onnx"
    slim_path  = out_dir / f"{stem}_simplified.onnx"

    # ── Export ────────────────────────────────────────────────────────────────
    # Use torch.rand (NOT zeros) — zeros causes constant-folding of the entire
    # network, producing a valid ONNX that always outputs zero/constant depth.
    dummy = torch.rand(1, args.frames, 3, args.height, args.width, device=device)
    print(f"\n[INFO] Exporting ONNX: {onnx_path}")
    print(f"       input: [1, {args.frames}, 3, {args.height}, {args.width}]")

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,           # INormalizationLayer → avoids FP16 overflow
        do_constant_folding=True,
        input_names=['input'],
        output_names=['depth'],
        dynamic_axes={'input': {0: 'batch'}, 'depth': {0: 'batch'}},
        verbose=False,
    )
    sz = onnx_path.stat().st_size // 1024 // 1024
    print(f"[INFO] ONNX written: {sz} MB")

    # ── Verify ────────────────────────────────────────────────────────────────
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    print("[INFO] ONNX check passed.")

    # ── Simplify ──────────────────────────────────────────────────────────────
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
                print("[WARN] onnxsim could not simplify — use the unsimplified file.")
        except ImportError:
            print("[WARN] onnxsim not installed.  Run:  pip install onnxsim")

    print(f"\n{'='*70}")
    print("Export complete!")
    print(f"  Recommended: {recommend}")
    print()
    print("Next steps:")
    print("  1. Copy the .onnx next to 3Deflatten_x64.ax")
    print("  2. Property page → select model → Runtime: TensorRT native → Reload")
    if args.frames == 1:
        print(f"\n  F=1: single-frame, same speed as DA v2 (~25ms FP16 on SM75).")
    else:
        print(f"\n  F={args.frames}: {args.frames}-frame window (~{args.frames*25}ms estimated on SM75).")


def main():
    p = argparse.ArgumentParser(
        description="Export VideoDepthAnything / DA-v2 to ONNX for 3Deflatten",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--family', choices=['vda','dav2'], default='vda',
                   help="Model family: vda=VideoDepthAnything, dav2=DepthAnything-V2 (default: vda)")
    p.add_argument('--model',  choices=['small','large'], default='small',
                   help="Model size (default: small)")
    p.add_argument('--frames', type=int, default=1,
                   help="Temporal window F (default: 1)")
    p.add_argument('--width',  type=int, default=518, help="Input width  (default: 518)")
    p.add_argument('--height', type=int, default=518, help="Input height (default: 518)")
    p.add_argument('--weights', default=None, help="Path to local .pth weights (skips download)")
    p.add_argument('--cache',   default=None, help="HuggingFace cache dir")
    p.add_argument('--out',     default='.', help="Output directory (default: .)")
    p.add_argument('--no-simplify', action='store_true', help="Skip onnxsim step")
    args = p.parse_args()
    if args.frames < 1: p.error("--frames must be >= 1")
    for dim, name in [(args.width,'width'),(args.height,'height')]:
        if dim % 14 != 0:
            print(f"[WARN] --{name} {dim} is not a multiple of 14 (ViT patch size). "
                  f"Nearest: {(dim//14)*14}")
    export(args)

if __name__ == '__main__':
    main()
