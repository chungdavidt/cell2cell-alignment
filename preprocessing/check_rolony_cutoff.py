#!/usr/bin/env python3
"""
Sniff-test viewer for picking an mScarlet rolony cutoff. Writes PNG figures only.

Each subslice appears twice, side by side: the raw downsampled mScarlet TIF, and
the same subslice's cell masks with kept cells painted by rolony count. Raising
`--min-rolonies` removes cells; where the remaining red stops tracking real
signal in the raw panel is the cutoff.

Two things this does differently from
`interactive_mscarlet_threshold_cellmask_subslice.py` (step 5):

**The cutoff is in rolonies, not a normalized fraction.** Step 5 cuts on
`count / max_expr`, so a threshold value depends on the single brightest cell in
the dataset and means something different in every brain. Here `--min-rolonies 3`
is three rolonies.

**Color is a fixed ramp over [1, --saturate-at], never rebased on the cutoff.**
A 12-rolony cell is the same color in every image; raising the cutoff deletes
cells but never re-shades the survivors, so runs at different cutoffs are
directly comparable. Step 5's `count / max_expr * BOOST` renders a 1-rolony cell
darker than the grey mask on BY95, where the median mScarlet+ cell has 1 count.

QC defaults to the lab's own cell-typing filter, `reads >= 20 AND genes >= 5`
(BY95: 147,185 cells, 26.4%), NOT `preprocessing_config`'s QC_MIN_READS /
QC_MIN_GENES, which are 0/0 to match the ungated marker plots in
`Gen_mScarlet_plots.m`. Both are per-brain; override with --min-reads/--min-genes.

Prerequisites: `stitch_subslices.py` then `downsample_subslices_cellmask.py`
WITHOUT `--cellmask-only` (that flag skips the TIFs this tool reads).

Usage:
    python preprocessing/check_rolony_cutoff.py
    python preprocessing/check_rolony_cutoff.py --min-rolonies 3
    python preprocessing/check_rolony_cutoff.py --first 3 --min-rolonies 1
    python preprocessing/check_rolony_cutoff.py --slices 22 24 --saturate-at 25
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from preprocessing_config import (
    FILT_NEURONS_PATH,
    HYB_DOWNSAMPLED_DIR,
    OUTPUT_ROOT,
    MSCARLET_COLUMN_INDEX,
    MSCARLET_GENE_NAME,
    DOWNSAMPLE_XY,
)
from utilities.mat_io import (
    load_filt_neurons,
    load_mat,
    load_cellmask_h5,
    get_expression_column,
    resolve_marker_column,
)
from utilities.image_io import imread_tiff
from utilities.visualization import create_histogram

CELLMASK_BRIGHTNESS = 0.3          # grey background, before --cellmask scales it
SUBSLICES_PER_FIGURE = 3           # one row each: raw | painted
RAMP_COLORS = [(0.45, 0.0, 0.0), (1.0, 0.35, 0.0), (1.0, 0.95, 0.25)]
RAW_PERCENTILES = (1.0, 99.5)      # per-subslice contrast stretch for the raw panel
MAX_DISPLAY_PX = 2000              # long side; see block_max()


def build_ramp():
    return LinearSegmentedColormap.from_list("rolony", RAMP_COLORS)


def ramp_rgb(counts, saturate_at, cmap):
    """Integer rolony counts -> RGB. Domain is [1, saturate_at], independent of
    the cutoff, so a given count always gets the same color."""
    frac = np.clip((np.asarray(counts, float) - 1.0) / max(saturate_at - 1, 1), 0.0, 1.0)
    return cmap(frac)[..., :3]


def as_label_array(cellmask):
    """Cellmask as an integer array safe to use as a LUT index."""
    if np.issubdtype(cellmask.dtype, np.integer):
        return cellmask
    return cellmask.astype(np.int32)


def count_image(cellmask, x_img, y_img, counts):
    """Rolony count per pixel, in one pass.

    Steps 4 and 5 run `np.where(cellmask == cell_id)` once per cell, a full-image
    comparison per cell. Instead: map each centroid to its label, build a LUT
    over labels, then index the whole mask once.

    Returns (count_img, lut, ids, n_off_mask, n_out_of_bounds). `ids` is the
    label each centroid landed in, 0 where it missed a mask or fell outside the
    image — so `ids > 0` is per-cell "this cell can be filled at all", and the
    cutoff then splits those into filled and below-cutoff.
    """
    labels = as_label_array(cellmask)
    h, w = labels.shape

    in_bounds = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
    ids = np.zeros(x_img.size, dtype=np.int64)
    ids[in_bounds] = labels[y_img[in_bounds], x_img[in_bounds]]

    on_mask = ids > 0
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    if on_mask.any():
        # max() rather than last-wins: two centroids can land in one mask
        np.maximum.at(lut, ids[on_mask], counts[on_mask].astype(np.int32))
    lut[0] = 0

    return (lut[labels], lut, ids,
            int((~on_mask & in_bounds).sum()), int((~in_bounds).sum()))


def block_max(a, k):
    """Downsample by k with a max over each k x k block.

    A subslice is far larger than the panel it is drawn into, and matplotlib's
    resampling AVERAGES — an isolated 1-rolony cell would be blended toward the
    background and vanish, which is exactly the cell the cutoff decision turns
    on. Max keeps it. Trims up to k-1 pixels off the right and bottom edges.
    """
    if k <= 1:
        return a
    h, w = a.shape
    H, W = (h // k) * k, (w // k) * k
    return a[:H, :W].reshape(H // k, k, W // k, k).max(axis=(1, 3))


def display_factor(shape, max_px):
    if max_px <= 0:
        return 1
    return max(1, int(np.ceil(max(shape) / float(max_px))))


def paint(cellmask, counts_img, min_rolonies, saturate_at, cmap, cellmask_intensity,
          k=1):
    """Grey cellmask field with cells at >= min_rolonies painted by the ramp.

    Reduces by k with block_max BEFORE painting, so the reduction happens on the
    count image rather than on rendered color.
    """
    occupied = block_max((cellmask > 0).astype(np.uint8), k).astype(np.float32)
    counts_img = block_max(counts_img, k)

    grey = occupied * CELLMASK_BRIGHTNESS * cellmask_intensity
    rgb = np.repeat(grey[:, :, None], 3, axis=2)

    drawn = counts_img >= min_rolonies
    if drawn.any():
        rgb[drawn] = ramp_rgb(counts_img[drawn], saturate_at, cmap).astype(np.float32)
    return rgb, drawn


def stretch(raw, k=1):
    """Percentile stretch, or the raw panel renders black. Reduced by block_max
    to match the painted panel pixel for pixel."""
    if raw.ndim == 3:
        raw = raw[..., 0]
    raw = block_max(raw, k)
    lo, hi = np.percentile(raw, RAW_PERCENTILES)
    return np.clip((raw.astype(np.float32) - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)


def load_subslices(input_dir, filt_neurons, mscarlet, pass_qc, slice_selection):
    """One entry per subslice with a QC-passing mScarlet+ cell, in slice order."""
    files = sorted(
        input_dir.glob("slice*_subslice_CELLMASK.h5"),
        key=lambda f: int(re.search(r"slice(\d+)_subslice", f.name).group(1)),
    )
    use_h5 = bool(files)
    if not files:
        files = sorted(
            input_dir.glob("slice*_subslice_CELLMASK.mat"),
            key=lambda f: int(re.search(r"slice(\d+)_subslice", f.name).group(1)),
        )
    if not files:
        raise FileNotFoundError(f"No downsampled subslices found in: {input_dir}")

    print(f"Found {len(files)} subslices ({'H5' if use_h5 else 'MAT'} format)")

    slice_ids = np.asarray(filt_neurons["slice"]).ravel()
    pos = np.asarray(filt_neurons["pos"])
    marker_positive = pass_qc & (mscarlet > 0)

    out = []
    for f in files:
        slice_id = int(re.search(r"slice(\d+)_subslice", f.name).group(1))
        if slice_selection is not None and slice_id not in slice_selection:
            continue

        keep = (slice_ids == slice_id) & marker_positive
        if not keep.any():
            print(f"  slice {slice_id}: no QC-passing mScarlet+ cells, skipped")
            continue

        if use_h5:
            cellmask, meta = load_cellmask_h5(f)
            min_x_offset = meta.get("min_x_offset", 0)
            min_y_offset = meta.get("min_y_offset", 0)
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

        # Full-res position -> canvas (x2) -> downsampled image, 0-indexed.
        # Must match generate_mscarlet_cellmask_subslice.py exactly.
        idx = np.where(keep)[0]
        x_img = np.rint((pos[idx, 0] * 2 - (min_x_offset - 1)) / DOWNSAMPLE_XY).astype(np.int64) - 1
        y_img = np.rint((pos[idx, 1] * 2 - (min_y_offset - 1)) / DOWNSAMPLE_XY).astype(np.int64) - 1

        raw_path = f.parent / f"{f.stem.replace('_CELLMASK', '')}_MSCARLET.tif"
        out.append({
            "slice_id": slice_id,
            "cellmask": cellmask,
            "x_img": x_img,
            "y_img": y_img,
            "counts": mscarlet[idx],
            "raw_path": raw_path if raw_path.exists() else None,
        })
        print(f"  slice {slice_id}: {idx.size} QC-passing mScarlet+ cells"
              f"{'' if raw_path.exists() else '   (no MSCARLET.tif)'}")

    if not out:
        raise ValueError("No subslices to draw.")
    return out


def rolony_distribution(counts, min_rolonies, saturate_at, cmap, out_dir):
    """Cells per rolony count across the subslices drawn, with the cutoff marked.

    check_qc_metrics.py prints this over the whole brain; here it is restricted
    to the subslices actually being looked at, and carries the cutoff line, so
    the picture and the number come from the same population.
    """
    counts = np.asarray(counts, dtype=np.int64)
    vmax = int(counts.max())
    values = np.arange(1, vmax + 1)
    per_value = np.array([(counts == v).sum() for v in values])
    at_or_above = np.array([(counts >= v).sum() for v in values])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax.bar(values, per_value, color=ramp_rgb(values, saturate_at, cmap),
           edgecolor="black", linewidth=0.4)
    ax.set_ylabel("cells with exactly n rolonies", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("mScarlet rolonies per cell, subslices drawn "
                 f"({counts.size} QC-passing mScarlet+ cells)",
                 fontsize=13, fontweight="bold")

    ax2.plot(values, 100 * at_or_above / counts.size, color="#333333", marker="o",
             markersize=3)
    ax2.set_ylabel("% of marker+ cells kept\nat cutoff >= n", fontsize=11)
    ax2.set_xlabel("mScarlet rolonies per cell", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)

    kept = 100 * float((counts >= min_rolonies).sum()) / counts.size
    for a in (ax, ax2):
        a.axvline(min_rolonies - 0.5, color="#1f77b4", linestyle="--", linewidth=2)
        a.grid(True, alpha=0.3)
    ax2.annotate(f"cutoff >= {min_rolonies}\nkeeps {kept:.1f}%",
                 xy=(min_rolonies - 0.5, kept), xytext=(8, 8),
                 textcoords="offset points", fontsize=10, color="#1f77b4",
                 fontweight="bold")
    ax.axvspan(0, min_rolonies - 0.5, color="#999999", alpha=0.18)
    ax2.axvspan(0, min_rolonies - 0.5, color="#999999", alpha=0.18)

    fig.tight_layout()
    path = Path(out_dir) / f"rolony_distribution_ge{min_rolonies}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def draw_legend(ax, saturate_at, cmap):
    strip = ramp_rgb(np.arange(1, saturate_at + 1), saturate_at, cmap)[None, :, :]
    ax.imshow(strip, aspect="auto", extent=[0.5, saturate_at + 0.5, 0, 1],
              interpolation="nearest")
    ticks = [t for t in (1, 5, 10, 15, 20, 25, 30) if t <= saturate_at]
    if ticks and ticks[-1] != saturate_at:
        ticks.append(saturate_at)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}+" if t == saturate_at else str(t) for t in ticks], fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("mScarlet rolonies per cell", fontsize=9)


def main():
    ap = argparse.ArgumentParser(
        description="Rolony-cutoff sniff test: raw mScarlet TIF beside painted cell masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--min-rolonies", "-n", type=int, default=1,
                    help="cells below this rolony count are not drawn (default: 1)")
    ap.add_argument("--min-reads", type=int, default=20,
                    help="QC total reads floor; the lab's value, per-brain (default: 20)")
    ap.add_argument("--min-genes", type=int, default=5,
                    help="QC distinct genes floor; the lab's value, per-brain (default: 5)")
    ap.add_argument("--saturate-at", type=int, default=25,
                    help="rolony count where the ramp caps (default: 25)")
    ap.add_argument("--cellmask", "-cm", type=float, default=0.5,
                    help="grey cellmask brightness 0-1 (default: 0.5)")
    ap.add_argument("--slices", "-s", type=int, nargs="+", default=None,
                    help="specific slice IDs")
    ap.add_argument("--first", "-f", type=int, default=None,
                    help="only the first N subslices")
    ap.add_argument("--max-display-px", type=int, default=MAX_DISPLAY_PX,
                    help=f"reduce panels to this long side with a block max so small "
                         f"cells survive; 0 disables (default: {MAX_DISPLAY_PX})")
    ap.add_argument("--dpi", type=int, default=150, help="figure dpi (default: 150)")
    ap.add_argument("--out", default=None, help="output directory override")
    args = ap.parse_args()

    if args.min_rolonies < 1:
        ap.error("--min-rolonies must be >= 1; a cell with 0 rolonies is not marker+")
    if args.saturate_at < 2:
        ap.error("--saturate-at must be >= 2")

    input_dir = Path(HYB_DOWNSAMPLED_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Downsampled subslices not found: {input_dir}\n"
            f"Run stitch_subslices.py then downsample_subslices_cellmask.py first."
        )

    out_dir = Path(args.out) if args.out else (
        Path(OUTPUT_ROOT) / "mScarlet_rolony_cutoff"
        / f"qc{args.min_reads}_{args.min_genes}_ge{args.min_rolonies}_sat{args.saturate_at}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ROLONY CUTOFF SNIFF TEST")
    print("=" * 60)
    print(f"cutoff:     >= {args.min_rolonies} mScarlet rolonies")
    print(f"QC:         reads >= {args.min_reads} AND genes >= {args.min_genes}"
          f"  (the lab's filter, not preprocessing_config's)")
    print(f"ramp:       1 .. {args.saturate_at}+, fixed (independent of the cutoff)")
    print(f"input:      {input_dir}")
    print(f"output:     {out_dir}\n")

    print("Loading filt_neurons...")
    filt_neurons = load_filt_neurons(FILT_NEURONS_PATH)
    expmat = filt_neurons["expmat"]
    n_cells = expmat.shape[0]

    if sparse.issparse(expmat):
        total_reads = np.asarray(expmat.sum(axis=1)).ravel()
        total_genes = np.asarray((expmat > 0).sum(axis=1)).ravel()
    else:
        total_reads = np.sum(expmat, axis=1)
        total_genes = np.sum(expmat > 0, axis=1)
    pass_qc = (total_reads >= args.min_reads) & (total_genes >= args.min_genes)
    print(f"  cells: {n_cells}")
    print(f"  QC-passing: {int(pass_qc.sum())} ({100 * pass_qc.sum() / n_cells:.1f}%)")

    mscarlet_col = resolve_marker_column(filt_neurons, MSCARLET_GENE_NAME, MSCARLET_COLUMN_INDEX)
    mscarlet = np.asarray(get_expression_column(expmat, mscarlet_col)).ravel()
    marker_qc = pass_qc & (mscarlet > 0)
    print(f"  mScarlet+ and QC-passing: {int(marker_qc.sum())}")
    if marker_qc.any():
        print(f"  their counts: max {mscarlet[marker_qc].max():.0f}, "
              f"median {np.median(mscarlet[marker_qc]):.0f}, "
              f"at or above cutoff {int((mscarlet[marker_qc] >= args.min_rolonies).sum())}\n")

    subslices = load_subslices(input_dir, filt_neurons, mscarlet, pass_qc,
                               set(args.slices) if args.slices else None)
    if args.first is not None:
        subslices = subslices[: args.first]

    cmap = build_ramp()
    n_fig = (len(subslices) + SUBSLICES_PER_FIGURE - 1) // SUBSLICES_PER_FIGURE
    print(f"\nDrawing {len(subslices)} subslices into {n_fig} figures "
          f"({SUBSLICES_PER_FIGURE} per figure)...\n")

    summary = []
    for fig_i in range(n_fig):
        batch = subslices[fig_i * SUBSLICES_PER_FIGURE:(fig_i + 1) * SUBSLICES_PER_FIGURE]
        rows = len(batch)
        fig = plt.figure(figsize=(12, 4.6 * rows + 1.2))
        gs = fig.add_gridspec(rows + 1, 2, height_ratios=[4.6] * rows + [0.45])
        fig.suptitle(
            f"mScarlet >= {args.min_rolonies} rolonies  |  QC reads>={args.min_reads} "
            f"genes>={args.min_genes}  |  ramp 1-{args.saturate_at}+  |  "
            f"figure {fig_i + 1}/{n_fig}",
            fontsize=12,
        )

        for row, data in enumerate(batch):
            counts_img, lut, ids, off_mask, oob = count_image(
                data["cellmask"], data["x_img"], data["y_img"], data["counts"])
            k = display_factor(data["cellmask"].shape, args.max_display_px)
            rgb, _ = paint(data["cellmask"], counts_img, args.min_rolonies,
                           args.saturate_at, cmap, args.cellmask, k)

            mapped = ids > 0
            n_marker = int(data["counts"].size)
            n_unmapped = int((~mapped).sum())
            n_below = int((mapped & (data["counts"] < args.min_rolonies)).sum())
            n_filled = int((mapped & (data["counts"] >= args.min_rolonies)).sum())
            n_regions = int((lut >= args.min_rolonies).sum())

            ax_raw = fig.add_subplot(gs[row, 0])
            if data["raw_path"] is not None:
                ax_raw.imshow(stretch(imread_tiff(data["raw_path"]), k), cmap="gray",
                              vmin=0, vmax=1, interpolation="antialiased")
                ax_raw.set_title(f"slice {data['slice_id']} - raw mScarlet "
                                 f"(p{RAW_PERCENTILES[0]}-p{RAW_PERCENTILES[1]}, per image)",
                                 fontsize=10)
            else:
                ax_raw.text(0.5, 0.5, "no MSCARLET.tif\n(step 3 ran --cellmask-only?)",
                            ha="center", va="center", fontsize=10)
                ax_raw.set_title(f"slice {data['slice_id']} - raw mScarlet missing", fontsize=10)
            ax_raw.axis("off")

            ax_paint = fig.add_subplot(gs[row, 1])
            ax_paint.imshow(np.clip(rgb, 0, 1), interpolation="antialiased")
            ax_paint.set_title(
                f"slice {data['slice_id']} - {n_filled} of {n_marker} marker+ filled "
                f"({100 * n_filled / max(n_marker, 1):.0f}%)", fontsize=10)
            ax_paint.axis("off")

            summary.append((data["slice_id"], n_marker, n_filled, n_below,
                            n_unmapped, off_mask, oob))
            extra = f", display /{k}" if k > 1 else ""
            if n_regions != n_filled:
                extra += f", {n_filled - n_regions} share a mask with another cell"
            print(f"  slice {data['slice_id']}: {n_filled}/{n_marker} filled, "
                  f"{n_below} below cutoff, {n_unmapped} unmapped{extra}")

        draw_legend(fig.add_subplot(gs[rows, :]), args.saturate_at, cmap)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        path = out_dir / f"figure_{fig_i + 1}_of_{n_fig}.png"
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        print(f"  saved {path.name}\n")

    print("=" * 72)
    print(f"{'slice':>7}{'marker+':>10}{'filled':>9}{'%':>7}"
          f"{'below cut':>11}{'unmapped':>10}")
    for slice_id, n_marker, n_filled, n_below, n_unmapped, _, _ in summary:
        print(f"{slice_id:>7}{n_marker:>10}{n_filled:>9}"
              f"{100 * n_filled / max(n_marker, 1):>7.1f}{n_below:>11}{n_unmapped:>10}")
    tot_marker = sum(r[1] for r in summary)
    tot_filled = sum(r[2] for r in summary)
    tot_below = sum(r[3] for r in summary)
    tot_unmapped = sum(r[4] for r in summary)
    tot_off = sum(r[5] for r in summary)
    tot_oob = sum(r[6] for r in summary)
    pct = lambda v: 100 * v / max(tot_marker, 1)

    print(f"\nAt cutoff >= {args.min_rolonies} rolonies, of {tot_marker} QC-passing "
          f"mScarlet+ cells in these subslices:")
    print(f"  filled       {tot_filled:>8}  {pct(tot_filled):5.1f}%")
    print(f"  NOT filled   {tot_below + tot_unmapped:>8}  "
          f"{pct(tot_below + tot_unmapped):5.1f}%")
    print(f"    below cutoff  {tot_below:>8}  {pct(tot_below):5.1f}%   "
          f"moves with --min-rolonies")
    print(f"    unmapped      {tot_unmapped:>8}  {pct(tot_unmapped):5.1f}%   "
          f"fixed; {tot_off} on background, {tot_oob} out of bounds")
    hist = create_histogram(
        [r[0] for r in summary], [r[2] for r in summary], [r[1] for r in summary],
        float(args.min_rolonies), out_dir,
        criterion_label=f">= {args.min_rolonies} rolonies",
        filename=f"cell_count_histogram_ge{args.min_rolonies}_rolonies.png",
    )
    dist = rolony_distribution(
        np.concatenate([d["counts"] for d in subslices]),
        args.min_rolonies, args.saturate_at, cmap, out_dir)

    print("\n  Unmapped is a cellmask/centroid-mapping check, not a cutoff result -")
    print("  it is identical at every --min-rolonies. On background: cellmask holes,")
    print("  see check_cellmasks.py. Out of bounds: a wrong DOWNSAMPLE_XY or canvas")
    print("  offset would spike it.")
    print(f"\n  {Path(hist).name}   filled vs total marker+ per subslice")
    print(f"  {Path(dist).name}   cells per rolony count + % kept vs cutoff")
    print(f"\nFigures: {out_dir}")


if __name__ == "__main__":
    main()
