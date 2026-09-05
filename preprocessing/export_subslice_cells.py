#!/usr/bin/env python3
"""
Export the BARseq cell <-> cellmask-label link table for each subslice.

The point of this table is that mapping a painted cell back to its genes is an
IDENTITY lookup, not a geometric one. Every row carries `row_index`, the row of
`filt_neurons.mat`'s `expmat` for that cell, so all 114 gene counts come back
exactly with `expmat[row_index, :]`. Nothing is recovered by inverting the
downsample — that inverse is lossy (a `round()` to an integer pixel, and several
full-res positions collapsing onto one downsampled pixel at DOWNSAMPLE_XY), and
it is never needed, because the original FOV, the 40x position and the stitched
position are all stored fields on the same row.

`y_node` / `x_node` are the cell's position in the subslice node's own frame —
the coordinates a castalign transform consumes — so a row pushes straight through
the graph to the 2P volumes. Z within a subslice node is 0: the node is one plane
thick, and the edge supplies its position in the stack.

`y_img` / `x_img` are the same point ROUNDED to an array index, and exist only to
read a label out of the cellmask. Never transform them: rounding costs +-0.5
downsampled px (~0.55 µm at BY95's 1.1000 µm/px) for no reason. The cell's
position was never measured from the downsampled image — it is projected onto it
from `pos`, which is stored at full resolution and never resampled. The downsample
is lossy for PIXELS, not for CELLS.

**mScarlet+ only for now**, deliberately. The eventual cell-linking chain
(project_cell_by_cell_comparison_future.md) wants every QC-passing cell, not
just marker+ ones — a non-marker cell is still a typed cell with a mask and a
position. This is the sniff-test version; widen it once the table proves out.

Unmapped cells are KEPT, with `cell_id = 0` and a `status` saying why, so the
table is a complete account of the marker+ population rather than a silently
filtered one.

QC comes from `QC_MIN_READS` / `QC_MIN_GENES` in local_config -- the lab's
cell-typing filter, reads >= 20 AND genes >= 5 for BY95 (26.4% of cells).
`--min-reads` / `--min-genes` override for a one-off. This used to hardcode 20/5
because the config held 0/0 for the ungated marker plots; the config went to the
cell-typing pair on 2026-08-26, so the two now ask the same question and the
hardcoded copy was another brain's numbers waiting to be carried over.

Writes two files and reads nothing but `filt_neurons.mat` and the downsampled
cellmasks:
    subslice_cells_qc{reads}_{genes}.csv     one row per marker+ cell
    subslice_cells_qc{reads}_{genes}.json    provenance + per-subslice frames

Usage:
    python preprocessing/export_subslice_cells.py
    python preprocessing/export_subslice_cells.py --slices 22 24
    python preprocessing/export_subslice_cells.py --min-reads 0 --min-genes 0
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy import sparse

from preprocessing_config import (
    FILT_NEURONS_PATH,
    HYB_DOWNSAMPLED_DIR,
    OUTPUT_ROOT,
    QC_MIN_READS,
    QC_MIN_GENES,
    MSCARLET_COLUMN_INDEX,
    MSCARLET_GENE_NAME,
    DOWNSAMPLE_XY,
    TARGET_XY_UM_PER_PX,
    EXVIVO_UM_PER_PX,
    SCOPE,
)
from utilities.mat_io import (
    load_filt_neurons,
    load_mat,
    load_cellmask_h5,
    get_expression_column,
    resolve_marker_column,
)

COLUMNS = [
    "slice_id",      # subslice this cell belongs to
    "row_index",     # row of expmat / filt_neurons -> all 114 genes. THE join key.
    "cell_id",       # cellmask label in the downsampled subslice; 0 = unmapped
    "status",        # mapped | off_mask | out_of_bounds
    "y_img", "x_img",    # ROUNDED node-frame pixel; the index used to read a label
    "y_node", "x_node",  # UNROUNDED node-frame coordinate -- transform THIS
    "y_um", "x_um",      # y_node/x_node in µm (* TARGET_XY_UM_PER_PX)
    "n_pixels",      # area of this label in the downsampled mask
    "mscarlet",      # rolony count, expmat[:, MSCARLET_COLUMN_INDEX]
    "total_reads",   # summed over all 114 columns
    "n_genes",       # nonzero columns
    "fov",           # source FOV, e.g. MAX_Pos1_000_001
    "pos_x", "pos_y",        # stitched position as stored (half-res; x2 = canvas)
    "pos40x_x", "pos40x_y",  # position within the source FOV, 40x frame
]


def _column(arr, i):
    if arr is None or i >= arr.shape[1]:
        return None
    return np.asarray(arr[:, i]).ravel()


def main():
    ap = argparse.ArgumentParser(
        description="Export the BARseq cell <-> cellmask-label link table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--min-reads", type=int, default=QC_MIN_READS,
                    help=f"QC total reads floor (default: QC_MIN_READS = {QC_MIN_READS})")
    ap.add_argument("--min-genes", type=int, default=QC_MIN_GENES,
                    help=f"QC distinct genes floor (default: QC_MIN_GENES = {QC_MIN_GENES})")
    ap.add_argument("--slices", "-s", type=int, nargs="+", default=None,
                    help="specific slice IDs")
    ap.add_argument("--out", default=None, help="output directory override")
    args = ap.parse_args()

    input_dir = Path(HYB_DOWNSAMPLED_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Downsampled subslices not found: {input_dir}\n"
            f"Run stitch_subslices.py then downsample_subslices_cellmask.py first."
        )
    out_dir = Path(args.out) if args.out else Path(OUTPUT_ROOT) / "subslice_cell_table"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"subslice_cells_qc{args.min_reads}_{args.min_genes}"

    print("=" * 60)
    print("EXPORT SUBSLICE CELL TABLE")
    print("=" * 60)
    print(f"QC:      reads >= {args.min_reads} AND genes >= {args.min_genes}")
    print(f"scope:   {SCOPE}  ({TARGET_XY_UM_PER_PX:.4f} µm/px, "
          f"BARseq resampled {DOWNSAMPLE_XY:.4f}x from {EXVIVO_UM_PER_PX} µm/px)")
    print(f"output:  {out_dir}\n")

    print("Loading filt_neurons...")
    fn = load_filt_neurons(FILT_NEURONS_PATH)
    expmat = fn["expmat"]
    n_cells = expmat.shape[0]

    if sparse.issparse(expmat):
        total_reads = np.asarray(expmat.sum(axis=1)).ravel()
        n_genes = np.asarray((expmat > 0).sum(axis=1)).ravel()
    else:
        total_reads = np.sum(expmat, axis=1)
        n_genes = np.sum(expmat > 0, axis=1)
    pass_qc = (total_reads >= args.min_reads) & (n_genes >= args.min_genes)

    marker_col = resolve_marker_column(fn, MSCARLET_GENE_NAME, MSCARLET_COLUMN_INDEX)
    mscarlet = np.asarray(get_expression_column(expmat, marker_col)).ravel()
    marker_qc = pass_qc & (mscarlet > 0)
    print(f"  cells: {n_cells}")
    print(f"  QC-passing: {int(pass_qc.sum())} ({100 * pass_qc.sum() / n_cells:.1f}%)")
    print(f"  mScarlet+ and QC-passing: {int(marker_qc.sum())}\n")

    slice_ids = np.asarray(fn["slice"]).ravel()
    pos = np.asarray(fn["pos"])
    pos40x = np.asarray(fn.get("pos40x")) if fn.get("pos40x") is not None else None
    fov = fn.get("fov")
    fov = None if fov is None else np.asarray(fov).ravel()

    files = sorted(input_dir.glob("slice*_subslice_CELLMASK.h5"),
                   key=lambda f: int(re.search(r"slice(\d+)_subslice", f.name).group(1)))
    use_h5 = bool(files)
    if not files:
        files = sorted(input_dir.glob("slice*_subslice_CELLMASK.mat"),
                       key=lambda f: int(re.search(r"slice(\d+)_subslice", f.name).group(1)))
    if not files:
        raise FileNotFoundError(f"No downsampled subslices found in: {input_dir}")

    want = set(args.slices) if args.slices else None
    rows, frames = [], []

    for f in files:
        slice_id = int(re.search(r"slice(\d+)_subslice", f.name).group(1))
        if want is not None and slice_id not in want:
            continue
        keep = (slice_ids == slice_id) & marker_qc
        if not keep.any():
            print(f"  slice {slice_id}: no QC-passing mScarlet+ cells, skipped")
            continue

        if use_h5:
            cellmask, meta = load_cellmask_h5(f)
            min_x_offset = int(meta.get("min_x_offset", 0))
            min_y_offset = int(meta.get("min_y_offset", 0))
        else:
            data = load_mat(f)
            if "cellmask_down" not in data:
                print(f"  WARNING: cellmask_down missing in {f.name}, skipped")
                continue
            cellmask = np.asarray(data["cellmask_down"])
            min_x_offset = int(data.get("min_x_offset", 0))
            min_y_offset = int(data.get("min_y_offset", 0))
        if cellmask is None:
            print(f"  WARNING: no cellmask in {f.name}, skipped")
            continue

        labels = cellmask if np.issubdtype(cellmask.dtype, np.integer) \
            else cellmask.astype(np.int32)
        h, w = labels.shape
        areas = np.bincount(labels.ravel())          # label -> pixel count, one pass

        # Full-res position -> canvas (x2) -> downsampled image, 0-indexed.
        # Same formula as generate_marker_cellmask_subslice.py's paint loop and
        # check_rolony_cutoff.py; all three must agree or cells land off their masks.
        #
        # Kept twice on purpose. The rounded form is an ARRAY INDEX -- it exists
        # only to read a label out of the mask. The unrounded form is the cell's
        # actual position in the node frame and is what a castalign transform
        # should consume; rounding first would inject +-0.5 px of avoidable error
        # (~0.55 µm at BY95's 1.1000 µm/px) into every downstream match.
        idx = np.where(keep)[0]
        # Round THEN subtract 1, matching the other two copies exactly. Folding
        # the -1 inside rint() would diverge at .5 ties under banker's rounding.
        x_raw = (pos[idx, 0] * 2 - (min_x_offset - 1)) / DOWNSAMPLE_XY
        y_raw = (pos[idx, 1] * 2 - (min_y_offset - 1)) / DOWNSAMPLE_XY
        x_img = np.rint(x_raw).astype(np.int64) - 1
        y_img = np.rint(y_raw).astype(np.int64) - 1
        x_node = x_raw - 1.0
        y_node = y_raw - 1.0

        in_bounds = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
        ids = np.zeros(idx.size, dtype=np.int64)
        ids[in_bounds] = labels[y_img[in_bounds], x_img[in_bounds]]

        n_mapped = 0
        for j, cell_row in enumerate(idx):
            cid = int(ids[j])
            if not in_bounds[j]:
                status = "out_of_bounds"
            elif cid == 0:
                status = "off_mask"
            else:
                status = "mapped"
                n_mapped += 1
            rows.append({
                "slice_id": slice_id,
                "row_index": int(cell_row),
                "cell_id": cid,
                "status": status,
                "y_img": int(y_img[j]), "x_img": int(x_img[j]),
                "y_node": round(float(y_node[j]), 4),
                "x_node": round(float(x_node[j]), 4),
                "y_um": round(float(y_node[j]) * TARGET_XY_UM_PER_PX, 4),
                "x_um": round(float(x_node[j]) * TARGET_XY_UM_PER_PX, 4),
                "n_pixels": int(areas[cid]) if 0 < cid < areas.size else 0,
                "mscarlet": int(mscarlet[cell_row]),
                "total_reads": int(total_reads[cell_row]),
                "n_genes": int(n_genes[cell_row]),
                "fov": "" if fov is None else str(fov[cell_row]),
                "pos_x": float(pos[cell_row, 0]), "pos_y": float(pos[cell_row, 1]),
                "pos40x_x": "" if pos40x is None else float(pos40x[cell_row, 0]),
                "pos40x_y": "" if pos40x is None else float(pos40x[cell_row, 1]),
            })

        frames.append({
            "slice_id": slice_id, "height": int(h), "width": int(w),
            "min_x_offset": min_x_offset, "min_y_offset": min_y_offset,
            "n_marker_cells": int(idx.size), "n_mapped": n_mapped,
            "n_labels_in_mask": int((areas[1:] > 0).sum()),
        })
        print(f"  slice {slice_id}: {idx.size} marker+ cells, {n_mapped} mapped "
              f"({100 * n_mapped / idx.size:.1f}%)")

    if not rows:
        raise ValueError("No cells to export.")

    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_mapped = sum(1 for r in rows if r["status"] == "mapped")
    meta_path = out_dir / f"{stem}.json"
    with open(meta_path, "w") as fh:
        json.dump({
            "written": datetime.now().isoformat(timespec="seconds"),
            "filt_neurons": str(FILT_NEURONS_PATH),
            "cellmask_dir": str(input_dir),
            "population": "QC-passing mScarlet+ cells only",
            "qc": {"min_reads": args.min_reads, "min_genes": args.min_genes,
                   "note": "the lab's cell-typing filter; per-brain, not the "
                           "pipeline config's QC_MIN_READS/QC_MIN_GENES"},
            "marker_column_index": int(marker_col),
            "scope": SCOPE,
            "exvivo_um_per_px": EXVIVO_UM_PER_PX,
            "target_xy_um_per_px": TARGET_XY_UM_PER_PX,
            "downsample_xy": DOWNSAMPLE_XY,
            "frame": "y_node/x_node are the cell's UNROUNDED position in the "
                     "subslice node's frame -- transform these. y_img/x_img are the "
                     "same point rounded to an array index, used only to read a "
                     "cellmask label; transforming them injects +-0.5 px. z within "
                     "a node is 0 (one plane thick). Join to genes with "
                     "expmat[row_index, :] -- never by inverting the geometry. "
                     "Original position and FOV are pos / pos40x / fov on the same "
                     "row: stored at full resolution, never resampled.",
            "n_rows": len(rows), "n_mapped": n_mapped,
            "subslices": frames,
        }, fh, indent=2)

    print(f"\n{len(rows)} cells across {len(frames)} subslices, "
          f"{n_mapped} mapped ({100 * n_mapped / len(rows):.1f}%)")
    print(f"  {csv_path}")
    print(f"  {meta_path}")
    print("\nJoin back to genes with expmat[row_index, :] -- 114 counts, exact.")


if __name__ == "__main__":
    main()
