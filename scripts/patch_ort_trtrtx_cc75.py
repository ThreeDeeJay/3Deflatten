"""
patch_ort_trtrtx_cc75.py  —  Patch ORT's NvTensorRTRTX EP to allow Turing (CC 7.5).

The ORT NvTensorRTRTX execution provider (nv_execution_provider.cc) contains a
hardcoded check at NvExecutionProvider::NvExecutionProvider() that refuses to
initialise on GPUs with compute capability < 86 (Ampere GA10x / RTX 3000+):

    if (sm < 86 || (sm != 86 && sm != 89 && sm != 120))
        ORT_THROW("[NvTensorRTRTX EP] The execution provider only supports RTX
                  devices with compute capabilities 86, 89, 120 and above");

The TRT-RTX SDK 1.4 itself supports Turing (CC 7.5 / RTX 2000-series) — the
restriction is in ORT's wrapper only.  This script patches the check to also
accept CC 75 before you run build.bat.

Usage:
    python patch_ort_trtrtx_cc75.py --ort-src <path_to_onnxruntime_repo>

    Then build ORT normally:
        build.bat --config Release --use_dml --use_cuda ...
                  --use_nv_tensorrt_rtx --tensorrt_rtx_home <path> ...
"""
from __future__ import annotations
import argparse
import pathlib
import re
import shutil
import sys

# The file to patch, relative to the ORT repo root
TARGET_RELPATH = pathlib.Path(
    "onnxruntime/core/providers/nv_tensorrt_rtx/nv_execution_provider.cc"
)

# ── Pattern / replacement ─────────────────────────────────────────────────────
# We look for the SM check that rejects everything below 86 (or outside the
# explicit list).  The exact wording may vary slightly across ORT commits, so we
# match on the key numeric literals and the ORT_THROW call.
#
# Original (ORT ~1.21+):
#   if (sm < 86 || (sm != 86 && sm != 89 && sm != 120))
#       ORT_THROW(...compute capabilities 86, 89, 120...);
#
# Patched — also accept 75 (Turing: RTX 2060/2070/2080/2080 Ti):
#   if (sm < 75 || (sm != 75 && sm != 86 && sm != 89 && sm != 120))
#       ORT_THROW(...compute capabilities 75, 86, 89, 120...);
# ─────────────────────────────────────────────────────────────────────────────

_PATTERN = re.compile(
    r'(if\s*\(\s*sm\s*<\s*)86(\s*\|\|'       # "if (sm < 86 ||"
    r'\s*\(sm\s*!=\s*)86(\s*&&\s*sm\s*!=\s*89'  # "(sm != 86 && sm != 89"
    r'\s*&&\s*sm\s*!=\s*120\s*\)\s*\))'         # "&& sm != 120))"
)

_REPLACEMENT = r'\g<1>75\g<2>75 && sm != 86\g<3>'

_MSG_PATTERN = re.compile(
    r'(compute capabilities\s*)86,\s*89,\s*120(\s*and above)'
)
_MSG_REPLACEMENT = r'\g<1>75, 86, 89, 120\g<2>'


def patch(ort_src: pathlib.Path, dry_run: bool = False) -> bool:
    target = ort_src / TARGET_RELPATH
    if not target.exists():
        # Try alternate path (some repo checkouts omit the nv_tensorrt_rtx folder)
        alt = ort_src / "onnxruntime" / "core" / "providers" / "nv_tensorrt_rtx" / "nv_execution_provider.cc"
        if alt.exists():
            target = alt
        else:
            print(f"[ERROR] File not found: {target}")
            print(f"  Make sure --ort-src points to the root of the onnxruntime repository.")
            return False

    src = target.read_text(encoding="utf-8")

    if "sm < 75" in src and "sm != 75" in src:
        print(f"[INFO]  Already patched (CC 75 already present): {target}")
        return True

    if not _PATTERN.search(src):
        # Try a looser check in case wording changed
        if "sm < 86" not in src and "sm < 80" not in src:
            print(f"[WARN]  SM check pattern not found in {target}")
            print(f"        The ORT source may have changed.  Review the file manually.")
            print(f"        Look for an if-statement near 'compute capabilities' that")
            print(f"        rejects sm values below 86, and add 75 to the accepted set.")
            return False
        print(f"[WARN]  Standard pattern not matched — applying loose fallback substitution.")
        # Loose fallback: just replace the literal boundary
        patched = src.replace("if (sm < 86 ||", "if (sm < 75 ||")
        patched = patched.replace("sm != 86 && sm != 89 && sm != 120",
                                  "sm != 75 && sm != 86 && sm != 89 && sm != 120")
        patched = patched.replace("compute capabilities 86, 89, 120",
                                  "compute capabilities 75, 86, 89, 120")
        if patched == src:
            print(f"[ERROR] Loose fallback also found nothing to replace.")
            return False
    else:
        patched = _PATTERN.sub(_REPLACEMENT, src)
        patched = _MSG_PATTERN.sub(_MSG_REPLACEMENT, patched)

    if patched == src:
        print(f"[WARN]  No changes produced — file may already be patched or pattern changed.")
        return False

    if dry_run:
        print(f"[DRY-RUN] Would patch: {target}")
        # Show a context diff
        import difflib
        diff = list(difflib.unified_diff(
            src.splitlines(), patched.splitlines(),
            fromfile=str(target), tofile=str(target) + " (patched)",
            lineterm="", n=3,
        ))
        for line in diff[:60]:
            print(" ", line)
        return True

    # Backup
    backup = target.with_suffix(".cc.orig")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[INFO]  Backup saved: {backup}")

    target.write_text(patched, encoding="utf-8")
    print(f"[OK]    Patched: {target}")
    print(f"        CC 75 (Turing / RTX 2000-series) added to supported compute capabilities.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Patch ORT TRT-RTX EP to accept Turing (CC 7.5 / RTX 2000-series)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ort-src", required=True, metavar="PATH",
        help="Root of the onnxruntime repository (contains build.bat / build.sh)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be changed without writing any files"
    )
    args = parser.parse_args()

    ort_src = pathlib.Path(args.ort_src).resolve()
    if not ort_src.exists():
        print(f"[ERROR] --ort-src path does not exist: {ort_src}")
        sys.exit(1)

    ok = patch(ort_src, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
