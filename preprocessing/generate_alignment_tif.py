#!/usr/bin/env python3
"""
Generate the BARseq alignment TIF: only mScarlet+ cells, everything else background.

This is the image castalign fits on. It deliberately looks like a 2P mScarlet
volume — sparse bright somas on a dark field — because that is what it has to be
matched against. A dense all-cells mask field has no counterpart in the 2P, so
aligning one against the other means matching dense speckle to sparse points.

A cell is drawn only if it passes **both** gates:

    QC:      total reads >= --min-reads AND distinct genes >= --min-genes
    marker:  mScarlet rolonies >= --min-rolonies

Everything else — QC failures, non-marker cells, cells below the cutoff, the gaps
between cells — goes to background. A cell that fails either gate is not dimmed,
it is absent, exactly as a non-expressing cell is absent from a 2P mScarlet image.

**Binary on purpose.** Cells are flat 255, not graded by rolony count. castalign
rescales display intensity, so graded values are not stable between views; 0 vs
max survives any rescaling. Grading belongs in check_rolony_cutoff.py, which is
for choosing the cutoff, not for fitting.

The whole label is filled, so somas keep the shape the segmentation gave them.

## What --min-rolonies here does and does not affect

It is a **registration** parameter, not an analysis one. It changes which cells
you can see while fitting, and therefore how well-determined the fit is. It does
NOT reach anything downstream: the transform castalign stores is a mapping
between coordinate frames, so it applies to every point in the subslice grid —
cells that were drawn, cells that were not, raw mRNA pixels, background. The
cell <-> label link (export_subslice_cells.py) takes no cutoff at all.

So pick this value purely for legibility against the 2P. Pick the analysis cutoff
separately, as a filter on the exported cell table. They need not match.

Output goes to a folder named by both gates, so several coexist and nothing is
overwritten:
    <OUTPUT_ROOT>/subslice_align/qc{reads}_{genes}_ge{n}/slice{N}_subslice_ALIGN.tif

Usage:
    python preprocessing/generate_alignment_tif.py --min-rolonies 3
    python preprocessing/generate_alignment_tif.py --min-rolonies 10 --slices 22 24
    python preprocessing/generate_alignment_tif.py --min-rolonies 3 --all-cells-level 60
"""

import argparse
import re
import sys
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
    MSCARLET_COLUMN_INDEX,
    MSCARLET_GENE_NAME,
    DOWNSAMPLE_XY,
    TARGET_XY_UM_PER_PX,
    SCOPE,
)
from utilities.mat_io import (
    load_filt_neurons,
    load_mat,
    load_cellmask_h5,
    get_expression_column,
    resolve_marker_column,
)
from utilities.image_io import imwrite_tiff

FOREGROUND = 255
SPARSE_WARN = 20        # subslices with fewer visible cells than this are flagged


def qualifying_label_mask(cellmask, x_img, y_img, keep_cell, all_cells_level=0):
    """uint8 image: FOREGROUND on labels whose cell passed both gates, else background.

    Builds a label lookup and indexes the mask once, rather than comparing the
    whole image per cell the way steps 4 and 5 do.

    Returns (image, n_labels_drawn, n_off_mask, n_out_of_bounds).
    """
    labels = cellmask if np.issubdtype(cellmask.dtype, np.integer) \
        else cellmask.astype(np.int32)
    h, w = labels.shape

    in_bounds = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
    ids = np.zeros(x_img.size, dtype=np.int64)
    ids[in_bounds] = labels[y_img[in_bounds], x_img[in_bounds]]
    on_mask = ids > 0

    lut = np.zeros(int(labels.max()) + 1, dtype=np.uint8)
    if all_cells_level:
        # optional faint substrate; off by default -- the 2P has no all-cells layer
        lut[:] = np.uint8(all_cells_level)
    lut[0] = 0
    drawn = on_mask & keep_cell
    if drawn.any():
        lut[ids[drawn]] = FOREGROUND

    return (lut[labels],
            int(np.count_nonzero(lut == FOREGROUND)),
            int((~on_mask & in_bounds).sum()),
            int((~in_bounds).sum()))


def main():
    ap = argparse.ArgumentParser(
        description="Generate the BARseq alignment TIF (mScarlet+ cells only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--min-rolonies", "-n", type=int, default=1,
                    help="mScarlet rolony floor to be drawn at all (default: 1)")
    ap.add_argument("--min-reads", type=int, default=20,
                    help="QC total reads floor; the lab's value, per-brain (default: 20)")
    ap.add_argument("--min-genes", type=int, default=5,
                    help="QC distinct genes floor; the lab's value, per-brain (default: 5)")
    ap.add_argument("--all-cells-level", type=int, default=0,
                    help="draw non-qualifying cells at this level instead of background; "
                         "0 = off, which is the point of this image (default: 0)")
    ap.add_argument("--slices", "-s", type=int, nargs="+", default=None,
                    help="specific slice IDs")
    ap.add_argument("--out", default=None, help="output directory override")
    args = ap.parse_args()

    if args.min_rolonies < 1:
        ap.error("--min-rolonies must be >= 1; a cell with 0 rolonies is not marker+")
    if not 0 <= args.all_cells_level < FOREGROUND:
        ap.error(f"--all-cells-level must be in [0, {FOREGROUND})")

    input_dir = Path(HYB_DOWNSAMPLED_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Downsampled subslices not found: {input_dir}\n"
            f"Run stitch_subslices.py then downsample_subslices_cellmask.py first."
        )
    out_dir = Path(args.out) if args.out else (
        Path(OUTPUT_ROOT) / "subslice_align"
        / f"qc{args.min_reads}_{args.min_genes}_ge{args.min_rolonies}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATE ALIGNMENT TIF")
    print("=" * 60)
    print(f"drawn:   QC (reads >= {args.min_reads} AND genes >= {args.min_genes})"
          f" AND mScarlet >= {args.min_rolonies}")
    print(f"         everything else -> background"
          f"{'' if not args.all_cells_level else f' (other cells at {args.all_cells_level})'}")
    print(f"scope:   {SCOPE} ({TARGET_XY_UM_PER_PX:.4f} µm/px)")
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

    # The AND gate. Both must hold or the cell is not drawn.
    visible = pass_qc & (mscarlet >= args.min_rolonies)
    print(f"  cells: {n_cells}")
    print(f"  QC-passing: {int(pass_qc.sum())} ({100 * pass_qc.sum() / n_cells:.1f}%)")
    print(f"  QC-passing AND mScarlet >= {args.min_rolonies}: {int(visible.sum())}\n")
    if not visible.any():
        raise ValueError(f"No cell passes both gates at --min-rolonies {args.min_rolonies}.")

    slice_ids = np.asarray(fn["slice"]).ravel()
    pos = np.asarray(fn["pos"])

    files = sorted(input_dir.glob("slice*_subslice_CELLMASK.h5"),
                   key=lambda f: int(re.search(r"slice(\d+)_subslice", f.name).group(1)))
    use_h5 = bool(files)
    if not files:
        files = sorted(input_dir.glob("slice*_subslice_CELLMASK.mat"),
                       key=lambda f: int(re.search(r"slice(\d+)_subslice", f.name).group(1)))
    if not files:
        raise FileNotFoundError(f"No downsampled subslices found in: {input_dir}")

    want = set(args.slices) if args.slices else None
    written, sparse_slices = [], []

    for f in files:
        slice_id = int(re.search(r"slice(\d+)_subslice", f.name).group(1))
        if want is not None and slice_id not in want:
            continue
        in_slice = slice_ids == slice_id
        if not in_slice.any():
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

        # Round THEN subtract 1, matching generate_mscarlet_cellmask_subslice.py,
        # check_rolony_cutoff.py and export_subslice_cells.py exactly.
        idx = np.where(in_slice)[0]
        x_img = np.rint((pos[idx, 0] * 2 - (min_x_offset - 1)) / DOWNSAMPLE_XY).astype(np.int64) - 1
        y_img = np.rint((pos[idx, 1] * 2 - (min_y_offset - 1)) / DOWNSAMPLE_XY).astype(np.int64) - 1

        img, n_drawn, off_mask, oob = qualifying_label_mask(
            cellmask, x_img, y_img, visible[idx], args.all_cells_level)

        out_path = out_dir / f"slice{slice_id}_subslice_ALIGN.tif"
        imwrite_tiff(out_path, img)
        written.append((slice_id, int(visible[idx].sum()), n_drawn, off_mask, oob))
        flag = ""
        if n_drawn < SPARSE_WARN:
            sparse_slices.append(slice_id)
            flag = "   <-- too sparse to align on"
        print(f"  slice {slice_id}: {n_drawn} cells drawn, {img.shape}{flag}")

    if not written:
        raise ValueError("No subslices written.")

    print("\n" + "=" * 60)
    print(f"{'slice':>7}{'passed gates':>15}{'drawn':>8}{'off-mask':>10}{'out-of-bounds':>15}")
    for slice_id, n_pass, n_drawn, off_mask, oob in written:
        print(f"{slice_id:>7}{n_pass:>15}{n_drawn:>8}{off_mask:>10}{oob:>15}")

    tot_drawn = sum(r[2] for r in written)
    print(f"\n{len(written)} TIFs, {tot_drawn} cells drawn total "
          f"(median {int(np.median([r[2] for r in written]))} per subslice)")
    if sparse_slices:
        print(f"\n{len(sparse_slices)} subslice(s) under {SPARSE_WARN} cells: {sparse_slices}")
        print("  Too few landmarks to fit against. Lower --min-rolonies for these,")
        print("  or accept that they align on neighbouring sections' transforms.")
    print(f"\n{out_dir}")
    print("--min-rolonies here is a registration parameter only. It changes what you")
    print("can see while fitting, never where the transform sends anything.")


if __name__ == "__main__":
    main()
