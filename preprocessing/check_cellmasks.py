"""
Read-only load test of every FOV cellmask.

The --scan sweep in inspect_dataset_structure.py samples bytes, so it catches
a zero-filled copy but not a file that is truncated, half-written, or valid-
looking yet unreadable. This opens each cellmask.mat the way the pipeline does
(utilities.mat_io.load_mat, then the same variable-name list) and reports which
ones fail. Writes nothing except an optional --json report.

The pipeline skips an unloadable cellmask silently (fov_skipped += 1), so a bad
mask costs a FOV's worth of cells with no error. This is the check that surfaces
that before a run instead of after.

Usage:
    python preprocessing/check_cellmasks.py <DATA_ROOT>
    python preprocessing/check_cellmasks.py <DATA_ROOT> --fast
    python preprocessing/check_cellmasks.py <DATA_ROOT> --json cellmask_report.json
    python preprocessing/check_cellmasks.py            # uses local_config.DATA_ROOT
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

HYB_DIRNAME_CANDIDATES = ("hyb", "hyb_raw_files")
MASK_VAR_NAMES = ("maski", "cellmask", "mask", "segmentation", "seg")

# Status codes, ordered worst -> best for reporting.
STATUSES = ("MISSING", "LOAD_FAILED", "NO_MASK_VAR", "EMPTY", "OK")


def find_hyb_root(data_root):
    for name in HYB_DIRNAME_CANDIDATES:
        p = data_root / name
        if p.is_dir():
            return p
    return None


def header_kind(path):
    """Classify a .mat by its first bytes without decompressing."""
    with open(path, "rb") as fh:
        head = fh.read(128)
    if not head:
        return "empty file"
    if not head.strip(b"\x00"):
        return "zero-filled"
    if head[:4] == b"\x89HDF" or b"MATLAB 7.3" in head:
        return "v7.3 (HDF5)"
    if head[:6] == b"MATLAB":
        return "v5/v7"
    return "unrecognized header"


def check_one(path, fast):
    """Return (status, detail) for a single cellmask.mat."""
    if not path.exists():
        return "MISSING", "file not present"

    kind = header_kind(path)
    if kind in ("empty file", "zero-filled", "unrecognized header"):
        return "LOAD_FAILED", kind
    if fast:
        return "OK", f"{kind}, header only (--fast)"

    from utilities.mat_io import load_mat
    try:
        data = load_mat(path)
    except Exception as e:
        return "LOAD_FAILED", f"{type(e).__name__}: {e}"

    mask = None
    for name in MASK_VAR_NAMES:
        if name in data:
            mask = np.asarray(data[name])
            break
    if mask is None:
        for key, value in data.items():
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                mask = np.asarray(value)
                break
    if mask is None:
        return "NO_MASK_VAR", f"vars: {sorted(k for k in data if not k.startswith('__'))}"

    nz = int(np.count_nonzero(mask))
    detail = f"{kind}, shape={tuple(mask.shape)}, dtype={mask.dtype}, nonzero={nz}, labels={int(mask.max()) if nz else 0}"
    return ("OK" if nz else "EMPTY"), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", nargs="?", help="DATA_ROOT (default: local_config.DATA_ROOT)")
    ap.add_argument("--fast", action="store_true",
                    help="header check only; skips decompression, misses truncated files")
    ap.add_argument("--json", metavar="PATH", help="write the per-FOV report")
    args = ap.parse_args()

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        import local_config
        data_root = Path(local_config.DATA_ROOT)

    print(f"DATA_ROOT = {data_root}")
    hyb_root = find_hyb_root(data_root)
    if hyb_root is None:
        print(f"  no {' or '.join(HYB_DIRNAME_CANDIDATES)} directory found")
        return 1
    print(f"hyb root  = {hyb_root}")

    fov_dirs = sorted(p for p in hyb_root.iterdir() if p.is_dir())
    print(f"{len(fov_dirs)} FOV directories"
          f"{'  (--fast: header check only)' if args.fast else ''}\n")

    results = {}
    counts = {s: 0 for s in STATUSES}
    t0 = time.time()
    for i, d in enumerate(fov_dirs, 1):
        status, detail = check_one(d / "cellmask.mat", args.fast)
        results[d.name] = {"status": status, "detail": detail}
        counts[status] += 1
        if status != "OK":
            print(f"  {status:12s} {d.name}  —  {detail}")
        if i % 100 == 0 or i == len(fov_dirs):
            el = time.time() - t0
            rate = i / el if el else 0
            eta = (len(fov_dirs) - i) / rate if rate else 0
            print(f"    [{i}/{len(fov_dirs)}] {el:.0f}s elapsed"
                  f"{f', ~{eta:.0f}s left' if i < len(fov_dirs) else ''}")

    print(f"\n=== summary ({time.time() - t0:.0f}s) ===")
    for s in STATUSES:
        if counts[s]:
            print(f"  {s:12s} {counts[s]:5d} / {len(fov_dirs)}")

    bad = [n for n, r in results.items() if r["status"] in ("MISSING", "LOAD_FAILED", "NO_MASK_VAR")]
    if bad:
        print(f"\n{len(bad)} unusable — the pipeline will skip these FOVs silently:")
        for n in bad:
            print(f"  {n}")
        print("\nRe-copy these from the source, then re-run this check.")
    else:
        print("\nEvery cellmask loads.")
    if counts["EMPTY"]:
        print(f"{counts['EMPTY']} load but contain no labels — legitimate for a FOV with no cells; "
              f"cross-check against the FOVs present in filt_neurons.mat before treating as a fault.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"data_root": str(data_root), "hyb_root": str(hyb_root),
             "fast": args.fast, "counts": counts, "fovs": results}, indent=2))
        print(f"\nwrote {args.json}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
