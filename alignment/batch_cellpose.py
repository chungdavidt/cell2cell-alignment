"""
Batch Cellpose 3D segmentation of the subject's red 2P volumes.

Segments the TIFFs local_config.py names for the red 2P nodes —
``block_stack_red`` (BLOCK_STACK_PATH_RED) and ``invivo_red`` (INVIVO_PATH_RED) —
with cellpose 4 (cpsam) in 3D, and writes to ``<ANALYSIS_ROOT>/cellpose/``:

  <stem>_seg.npy        cellpose's own seg file (masks + flows); validate_mnn.py reads it
  <stem>_masks.tif      label volume (Z, Y, X), one page per plane, 0 = background
  <stem>_cellpose.json  input, node, SCOPE, anisotropy and eval params of the run
  <stem>_cellpose.log   cellpose's own output

ANALYSIS_ROOT is required: nothing is written next to the raw TIFFs.

``anisotropy`` is cellpose's parameter — the volume's Z step over its XY pitch —
derived from SCOPE (huang_lab: 1.0 / 1.1 = 0.909). A TIFF whose own XResolution
disagrees with SCOPE by more than 5% stops the run before the model loads.

MIN_PLANES drops cells spanning fewer Z planes, after stitching, before
anything is written. MODEL_PATHS swaps cpsam for a retrained model per node (cellpose_curation.py
train prints the line). EVAL_PARAMS are cellpose's defaults, which beat the cellprob/flow/diameter/
sharpen variants on visual QC in the 2026-04-16 by94 sweep. EVAL_OVERRIDES
changes them for one node only; tune with sweep_in_vivo_2d.py --node <node>
first, and judge by the overlay on the raw image, not by ROI count.

Usage:
    .cellpose-venv\\Scripts\\python.exe alignment/batch_cellpose.py --dry-run
    .cellpose-venv\\Scripts\\python.exe alignment/batch_cellpose.py                          # both nodes
    .cellpose-venv\\Scripts\\python.exe alignment/batch_cellpose.py --node block_stack_red   # one
    .cellpose-venv\\Scripts\\python.exe alignment/batch_cellpose.py --force                  # re-run existing
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from analysis_paths import TWO_P_RED_NODES, cellpose_dir, two_p_path
from scope_profiles import (
    assert_matches_metadata,
    cellpose_anisotropy,
    um_per_px_from_resolution,
)


EVAL_PARAMS = {
    "do_3D":              True,
    "z_axis":             0,      # TIFFs load as (Z, Y, X); cellpose 4 requires this for ndim==3
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

# Per-node changes to EVAL_PARAMS, e.g.
#   "block_stack_red": {"normalize": {"percentile": [1, 99.9], "tile_norm_blocksize": 100}},
EVAL_OVERRIDES: dict[str, dict] = {
    "block_stack_red": {},
    # cellpose_curation.py grid best (score_grid_20261014_121838), run the way it was
    # scored: each plane in 2D, normalized on its own, then outlines on neighbouring
    # planes joined into one cell where they overlap by IoU >= 0.5. do_3D scored F1
    # 0.396 on the training planes (score_3d_20261014_151905). 0.5 from
    # cellpose_curation.py stitch (stitch_20261014_153517): at 0.25, 18 cells spanned
    # over 50 planes; at 0.5, 5.
    "invivo_red": {
        "do_3D": False,
        "stitch_threshold": 0.5,
        "diameter": 20,
        "cellprob_threshold": -1.0,
        "flow_threshold": 0.6,
        "normalize": {"normalize": True, "norm3D": False, "invert": False, "lowhigh": None,
                      "percentile": [0.1, 99.9], "sharpen_radius": 0, "smooth_radius": 0,
                      "tile_norm_blocksize": 0, "tile_norm_smooth3D": 1},
    },
}

# Per-node: drop cells spanning fewer Z planes than this after stitching (0 = keep
# all). Most stitched outlines that match no curated cell span 1-6 planes.
MIN_PLANES: dict[str, int] = {
    "block_stack_red": 0,
    "invivo_red": 3,
}

# Per-node model: "" = cellpose's cpsam; otherwise a model file, e.g. one written
# by cellpose_curation.py train.
MODEL_PATHS: dict[str, str] = {
    "block_stack_red": "",
    "invivo_red": r"D:\lab_local\projects\cell_type\analysis\BY95\cellpose\curation_in_vivo\train_20261014_145504\models\cpsam_in_vivo_20261014_145504",
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


def output_paths(out_dir: Path, img_path: Path) -> dict:
    stem = img_path.stem
    return {
        "seg":   out_dir / f"{stem}_seg.npy",
        "masks": out_dir / f"{stem}_masks.tif",
        "json":  out_dir / f"{stem}_cellpose.json",
        "log":   out_dir / f"{stem}_cellpose.log",
    }


def tiff_xy_um_per_px(img_path: Path):
    """XY pitch from the TIFF's own XResolution, or None if it carries none."""
    import tifffile
    with tifffile.TiffFile(str(img_path)) as tif:
        page = tif.pages[0]
        x_tag = page.tags.get("XResolution")
        if x_tag is None:
            return None
        num, denom = x_tag.value
        if not (num > 0 and denom > 0):
            return None
        unit = (tif.imagej_metadata or {}).get("unit", "")
        res_unit_tag = page.tags.get("ResolutionUnit")
        return um_per_px_from_resolution(
            num / denom, unit, res_unit_tag.value if res_unit_tag else None
        )


def eval_params_for(node: str, scope: str) -> dict:
    params = dict(EVAL_PARAMS)
    params["anisotropy"] = cellpose_anisotropy(scope)
    params.update(EVAL_OVERRIDES.get(node, {}))
    return params


def write_masks_tif(path: Path, masks) -> None:
    import numpy as np
    import tifffile
    dtype = np.uint16 if masks.max() < 2**16 else np.uint32
    tifffile.imwrite(str(path), masks.astype(dtype), compression="zlib")


def drop_short_cells(masks, min_planes: int):
    """Zero cells spanning fewer than min_planes Z planes; renumber the rest 1..n."""
    import numpy as np
    from scipy import ndimage
    keep = np.zeros(int(masks.max()) + 1, dtype=bool)
    for lab, sl in enumerate(ndimage.find_objects(masks), start=1):
        if sl is not None and sl[0].stop - sl[0].start >= min_planes:
            keep[lab] = True
    kept = np.where(keep[masks], masks, 0)
    _, inv = np.unique(kept, return_inverse=True)
    return inv.reshape(masks.shape).astype(masks.dtype)


def model_for(node: str) -> str:
    return MODEL_PATHS.get(node) or "cpsam"


def run_one(model, node: str, img_path: Path, out_dir: Path, params: dict,
            scope: str, force: bool = False) -> dict:
    """Run cellpose 3D on one TIFF. status in {"skipped", "success", "error"}."""
    paths = output_paths(out_dir, img_path)

    if paths["seg"].exists() and paths["masks"].exists() and not force:
        print(f"  SKIP (exists): {paths['seg']}")
        return {"status": "skipped", "n_rois": None, "elapsed_s": 0.0, "error": None}

    import numpy as np
    import tifffile
    import cellpose
    from cellpose import io as cp_io

    t_start = time.time()
    real_stdout, real_stderr = sys.stdout, sys.stderr
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(paths["log"], "w", buffering=1) as log_f:
            sys.stdout = _Tee(real_stdout, log_f)
            sys.stderr = _Tee(real_stderr, log_f)
            try:
                print(f"  loading TIFF: {img_path}")
                img = tifffile.imread(str(img_path))
                print(f"  shape: {img.shape}, dtype: {img.dtype}")
                if img.ndim != 3:
                    raise ValueError(f"expected a (Z, Y, X) volume, got shape {img.shape}")

                print(f"  running model.eval (params: {params})")
                masks, flows, _styles = model.eval(img, **params)
                n_raw = int(masks.max()) if masks.size else 0
                print(f"  ROIs found: {n_raw}")
                min_planes = MIN_PLANES.get(node, 0)
                if min_planes > 1 and n_raw:
                    masks = drop_short_cells(masks, min_planes)
                    print(f"  dropped cells spanning < {min_planes} planes: {n_raw - int(masks.max())}")
                n_rois = int(masks.max()) if masks.size else 0
                print(f"  ROIs kept: {n_rois}")

                # cellpose names the seg file <stem>_seg.npy beside its
                # file_names argument, so point that at the output dir.
                print(f"  saving seg: {paths['seg']}")
                cp_io.masks_flows_to_seg(img, masks, flows, str(out_dir / img_path.name))
                if not paths["seg"].exists():
                    stray = img_path.with_name(f"{img_path.stem}_seg.npy")
                    if stray.exists():
                        shutil.move(str(stray), str(paths["seg"]))
                    else:
                        raise FileNotFoundError(
                            f"cellpose did not write the expected seg file: {paths['seg']}"
                        )

                print(f"  saving masks: {paths['masks']}")
                write_masks_tif(paths["masks"], masks)

                record = {
                    "node": node,
                    "input_tiff": str(img_path),
                    "shape_zyx": list(img.shape),
                    "scope": scope,
                    "eval_params": params,
                    "n_rois": n_rois,
                    "n_rois_before_min_planes": n_raw,
                    "min_planes": min_planes,
                    "cellpose_version": getattr(cellpose, "version", None),
                    "model": model_for(node),
                    "written": datetime.now().isoformat(timespec="seconds"),
                }
                with open(paths["json"], "w") as f:
                    json.dump(record, f, indent=2, default=str)
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
    print(f"  DONE in {elapsed/60:.1f} min ({n_rois} ROIs). Log: {paths['log'].name}")
    return {"status": "success", "n_rois": n_rois, "elapsed_s": elapsed, "error": None}


def _print_summary(results: list, total_elapsed: float) -> None:
    print()
    print("=" * 72)
    print("BATCH SUMMARY")
    print("=" * 72)
    for node, img_path, r in results:
        status = r["status"].upper()
        rois = f"{r['n_rois']} ROIs" if r["n_rois"] is not None else "-"
        mins = f"{r['elapsed_s']/60:5.1f} min"
        err = f"  [{r['error']}]" if r["error"] else ""
        print(f"  {status:<8} {mins}  {rois:<10}  {node:<16} {img_path.name}{err}")
    print("-" * 72)
    print(f"  TOTAL wall-clock: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} h)")
    print("=" * 72)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--node", action="append", choices=list(TWO_P_RED_NODES),
        help="Node to segment; repeatable. Default: every red 2P node set in local_config.py.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run nodes whose _seg.npy and _masks.tif already exist (default: skip).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print inputs, params and output state, then exit. No model load.",
    )
    args = parser.parse_args(argv)

    import local_config
    scope = getattr(local_config, "SCOPE", "")
    out_dir = cellpose_dir()
    if out_dir is None:
        sys.exit("ANALYSIS_ROOT is blank in local_config.py; batch_cellpose.py writes only "
                 "to <ANALYSIS_ROOT>/cellpose/.")

    explicit = args.node is not None
    jobs: list[tuple[str, Path]] = []
    for node in (args.node or list(TWO_P_RED_NODES)):
        path = two_p_path(node)
        if path is None:
            msg = f"{TWO_P_RED_NODES[node]} is blank in local_config.py"
            if explicit:
                sys.exit(f"--node {node}: {msg}.")
            print(f"  skip {node}: {msg}")
            continue
        if not path.exists():
            msg = f"{node}: TIFF not found: {path}"
            if explicit:
                sys.exit(msg)
            print(f"  skip {msg}")
            continue
        jobs.append((node, path))
    if not jobs:
        sys.exit("Nothing to segment: no red 2P TIFF from local_config.py is on disk.")

    # SCOPE errors and pitch mismatches stop here, before an hour of GPU time.
    cellpose_anisotropy(scope)
    for node, path in jobs:
        assert_matches_metadata(scope, tiff_xy_um_per_px(path), source_name=path.name)

    print(f"Output dir: {out_dir}")
    print(f"SCOPE:      {scope}  (anisotropy {cellpose_anisotropy(scope):.4f})")
    for node, path in jobs:
        paths = output_paths(out_dir, path)
        state = "done" if paths["seg"].exists() and paths["masks"].exists() else "to run"
        print(f"  [{state:<6}] {node:<16} {path}")
        print(f"           params: {eval_params_for(node, scope)}")
        print(f"           model:  {model_for(node)}")
        print(f"           min planes: {MIN_PLANES.get(node, 0)}")
    for node, _ in jobs:
        if model_for(node) != "cpsam" and not Path(model_for(node)).exists():
            sys.exit(f"MODEL_PATHS[{node!r}] not found: {model_for(node)}")
    if args.dry_run:
        return 0
    if not args.force and all(
            all(p.exists() for p in (output_paths(out_dir, path)[k] for k in ("seg", "masks")))
            for _, path in jobs):
        print("All done; --force re-runs.")
        return 0

    from cellpose.models import CellposeModel
    models: dict[str, object] = {}
    results = []
    t_batch = time.time()
    for i, (node, path) in enumerate(jobs, start=1):
        print("-" * 72)
        print(f"[{i}/{len(jobs)}] {node}: {path.name}")
        name = model_for(node)
        if name not in models:
            print(f"Loading model {name}...")
            t0 = time.time()
            models[name] = CellposeModel(gpu=True, pretrained_model=name)
            print(f"  loaded in {time.time() - t0:.1f}s (device={models[name].device})")
        r = run_one(models[name], node, path, out_dir, eval_params_for(node, scope), scope,
                    force=args.force)
        results.append((node, path, r))

    _print_summary(results, time.time() - t_batch)
    return 1 if any(r["status"] == "error" for _, _, r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
