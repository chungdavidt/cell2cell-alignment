"""
Batch Cellpose 3D segmentation — overnight runner.

Loops over a hardcoded list of TIFF paths and runs vanilla cellpose 4 (cpsam)
3D segmentation on each, saving ``<stem>_seg.npy`` next to each input.

Settings landed on after the 2026-04-16 2D parameter sweep on by94 (vanilla
defaults beat cellprob/flow/diameter/CLAHE/sharpen variations on visual QC).
Anisotropy is hardcoded to Huang-lab Z/XY = 1.81 for all four images.

Behavior:
  - Loads the cpsam model once before the loop.
  - Skips images whose ``_seg.npy`` already exists (pass ``--force`` to override).
  - Per-image try/except so one failure doesn't kill the batch.
  - Per-image log file ``<stem>_cellpose.log`` captures cellpose's own output.
  - Prints a per-image + total summary at the end.

Usage:
    source .cellpose-venv/bin/activate          # Linux/Mac
    # .cellpose-venv\Scripts\activate            # Windows
    python alignment/batch_cellpose.py               # full batch
    python alignment/batch_cellpose.py --force       # re-run even if _seg.npy exists
    python alignment/batch_cellpose.py --dry-run     # list images only, no model load
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path


IMAGES_TO_PROCESS: list[str] = [
    "C:/Users/David/lab_local/projects/cell_type/data/by94/gad2 by94 red exvivo max proj.tif",
    "C:/Users/David/lab_local/projects/cell_type/data/by94/gad2 by94 red invivo max proj.tif",
    "C:/Users/David/lab_local/projects/cell_type/data/by84/gad2-cre by84 exvivo red max proj.tif",
    "C:/Users/David/lab_local/projects/cell_type/data/by84/gad2-c re by84 invivo red max 16x.tif",
]

EVAL_PARAMS = {
    "do_3D":              True,
    "z_axis":             0,      # TIFFs load as (Z, Y, X); cellpose 4 requires this for ndim==3
    "anisotropy":         1.81,
    "stitch_threshold":   0.0,
    "cellprob_threshold": 0.0,
    "flow_threshold":     0.4,
    "diameter":           None,
    "min_size":           8,
    "niter":              200,
    "flow3D_smooth":      0,
    "batch_size":         8,
    "normalize":          True,
}


class _Tee:
    """Duplicate writes to both a real stream and a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, s):
        self.stream.write(s)
        self.log_file.write(s)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def fileno(self):
        return self.stream.fileno()


def _seg_path_for(img_path: Path) -> Path:
    return img_path.with_name(f"{img_path.stem}_seg.npy")


def _log_path_for(img_path: Path) -> Path:
    return img_path.with_name(f"{img_path.stem}_cellpose.log")


def run_one(model, img_path: Path, force: bool = False) -> dict:
    """Run cellpose 3D on one TIFF and save ``<stem>_seg.npy`` next to it.

    Returns a dict with keys: status, n_rois, elapsed_s, error.
    status in {"skipped", "success", "error"}.
    """
    seg_path = _seg_path_for(img_path)
    log_path = _log_path_for(img_path)

    if seg_path.exists() and not force:
        print(f"  SKIP (seg already exists): {seg_path.name}")
        return {"status": "skipped", "n_rois": None, "elapsed_s": 0.0, "error": None}

    if not img_path.exists():
        msg = f"input TIFF not found: {img_path}"
        print(f"  ERROR: {msg}")
        return {"status": "error", "n_rois": None, "elapsed_s": 0.0, "error": msg}

    import numpy as np
    import tifffile
    from cellpose import io as cp_io

    t_start = time.time()
    real_stdout, real_stderr = sys.stdout, sys.stderr
    try:
        with open(log_path, "w", buffering=1) as log_f:
            sys.stdout = _Tee(real_stdout, log_f)
            sys.stderr = _Tee(real_stderr, log_f)
            try:
                print(f"  loading TIFF: {img_path}")
                img = tifffile.imread(str(img_path))
                print(f"  shape: {img.shape}, dtype: {img.dtype}")

                print(f"  running model.eval (params: {EVAL_PARAMS})")
                masks, flows, _styles = model.eval(img, **EVAL_PARAMS)

                n_rois = int(masks.max()) if masks.size else 0
                print(f"  ROIs found: {n_rois}")

                print(f"  saving seg: {seg_path}")
                cp_io.masks_flows_to_seg(img, masks, flows, str(img_path))
            finally:
                sys.stdout = real_stdout
                sys.stderr = real_stderr
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - t_start
        print(f"  ERROR after {elapsed:.1f}s: {err}")
        traceback.print_exc()
        return {"status": "error", "n_rois": None, "elapsed_s": elapsed, "error": err}

    elapsed = time.time() - t_start
    print(f"  DONE in {elapsed/60:.1f} min ({n_rois} ROIs). Log: {log_path.name}")
    return {"status": "success", "n_rois": n_rois, "elapsed_s": elapsed, "error": None}


def _print_summary(results: list[tuple[Path, dict]], total_elapsed: float) -> None:
    print()
    print("=" * 72)
    print("BATCH SUMMARY")
    print("=" * 72)
    for img_path, r in results:
        status = r["status"].upper()
        rois = f"{r['n_rois']} ROIs" if r["n_rois"] is not None else "-"
        mins = f"{r['elapsed_s']/60:5.1f} min"
        err = f"  [{r['error']}]" if r["error"] else ""
        print(f"  {status:<8} {mins}  {rois:<10}  {img_path.name}{err}")
    print("-" * 72)
    print(f"  TOTAL wall-clock: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} h)")
    print("=" * 72)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run images that already have a _seg.npy (default: skip).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the four image paths + whether _seg.npy exists, then exit. No model load.",
    )
    args = parser.parse_args(argv)

    img_paths = [Path(p) for p in IMAGES_TO_PROCESS]

    if args.dry_run:
        print("DRY RUN — image plan:")
        for p in img_paths:
            seg = _seg_path_for(p)
            exists = "exists" if p.exists() else "MISSING"
            seg_state = "seg exists" if seg.exists() else "needs seg"
            print(f"  [{exists:<7}] [{seg_state:<10}] {p}")
        return 0

    print("=" * 72)
    print("BATCH CELLPOSE 3D — overnight run")
    print("=" * 72)
    print(f"Images: {len(img_paths)}")
    print(f"Force re-run: {args.force}")
    print(f"Eval params:  {EVAL_PARAMS}")
    print()
    print("Loading cpsam model...")
    t0 = time.time()
    from cellpose.models import CellposeModel
    model = CellposeModel(gpu=True, pretrained_model="cpsam")
    print(f"  loaded in {time.time() - t0:.1f}s (device={model.device})")
    print()

    results: list[tuple[Path, dict]] = []
    t_batch = time.time()
    for i, img_path in enumerate(img_paths, start=1):
        print("-" * 72)
        print(f"[{i}/{len(img_paths)}] {img_path.name}")
        r = run_one(model, img_path, force=args.force)
        results.append((img_path, r))

    _print_summary(results, time.time() - t_batch)
    n_errors = sum(1 for _, r in results if r["status"] == "error")
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
