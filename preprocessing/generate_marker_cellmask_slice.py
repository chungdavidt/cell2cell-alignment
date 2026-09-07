#!/usr/bin/env python3
"""
Marker cellmask overlays on WHOLE SLICES, both markers, one-off.

generate_marker_cellmask_subslice.py (step 4) draws these on the subslices step 1
selected. This draws the same thing on the whole section, for a visual check that
needs the tissue step 1 cropped away. Nothing reads its output.

The colour schema is step 4's, imported rather than copied -- `ramp_rgb`, its
fixed [ROLONY_RAMP_MIN, ceiling] domain, `write_ramp_legend`, the grey cellmask
field -- so a count is the same colour here as in the subslice render, and a
retune of either reaches both. Each marker keeps its own floor and ceiling from
marker_profiles.py: mScarlet ge5_sat15, GCaMP ge3_sat10.

Input (stitch_slices.py --downsample, at the 2P pitch):

    <OUTPUT_ROOT>/HYB_slice_stitched_tif_downsampled_micronwise/slice{N}_CELLMASK.h5

plus filt_neurons.mat for positions and counts. The full-resolution set in
HYB_slice_stitched_tif/ is not read: it is 0.32 um/px, so it shares a grid with
nothing else, and a section runs ~279 Mpx.

Output:

    <OUTPUT_ROOT>/mScarlet_cellmask_slice/rolony_ge5_sat15/slice{N}_mScarlet_cellmask.tif
    <OUTPUT_ROOT>/GCaMP_cellmask_slice/rolony_ge3_sat10/slice{N}_GCaMP_cellmask.tif
                                       rolony_ramp_legend.png  (one per folder)

The output roots are hardcoded below rather than added to preprocessing_config,
the same call montage_downsampled_subslices.py and add_subslice_scale_bars.py
make: no pipeline step reads them, and ensure_output_dirs() lists no
out-of-pipeline output. The names carry no `subslice` token, so
downsample_subslices_cellmask.py's slice*_subslice_CELLMASK* glob and the graph
builder's *_subslice_ALIGN.tif glob cannot see them.

Two things differ from step 4 beyond the input folder:

Canvas offsets are read under either name. stitch_slices.py stores min_x/min_y;
downsample_subslices_cellmask.py stores min_x_offset/min_y_offset. Step 4 reads
only the second, with `.get(..., 0)`, so a whole-slice file handed to it would
shift every centroid by the canvas origin and paint the wrong blobs with a low
`Mapped N / N` as the only symptom. Neither name present is a hard error here.

Cells are painted through a lookup table indexed by cell id, not step 4's
per-cell `cellmask == cell_id` scan, which is one full-array pass per drawn cell.
A whole section at this pitch runs ~24 Mpx and draws every one of its cells.

Usage:
    python generate_marker_cellmask_slice.py --dry-run   # sizes and counts, writes nothing
    python generate_marker_cellmask_slice.py             # every section, both markers
    python generate_marker_cellmask_slice.py --slice 44
    python generate_marker_cellmask_slice.py --marker gcamp
"""

import argparse
import re
import sys
import time
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
    HYB_SLICE_DOWNSAMPLED_DIR,
    OUTPUT_ROOT,
    QC_MIN_READS,
    QC_MIN_GENES,
    CELLMASK_BRIGHTNESS,
    DOWNSAMPLE_XY,
    TARGET_XY_UM_PER_PX,
)
from marker_profiles import get_marker, marker_names
from generate_marker_cellmask_subslice import (
    ROLONY_RAMP_MIN,
    CELLMASK_SCALE,
    make_ramp,
    ramp_rgb,
    write_ramp_legend,
)
from utilities.mat_io import (
    load_filt_neurons,
    load_cellmask_h5,
    get_expression_column,
    resolve_marker_column,
)
from utilities.image_io import imwrite_tiff, get_file_size_mb


# Not in preprocessing_config: no pipeline step reads either folder.
MARKER_OUTPUT_ROOTS = {
    "mscarlet": Path(OUTPUT_ROOT) / "mScarlet_cellmask_slice",
    "gcamp": Path(OUTPUT_ROOT) / "GCaMP_cellmask_slice",
}

# The subslice files are slice{N}_subslice_CELLMASK, which this does not match,
# so pointing the script at the wrong folder finds nothing instead of rendering
# subslices into the whole-slice output.
SLICE_RE = re.compile(r'slice(\d+)_CELLMASK')


def to_uint8(rgb):
    """0-1 RGB -> uint8 exactly as step 4 writes it.

    (value * 255).astype(uint8) TRUNCATES: the grey cellmask field is
    0.25 * 0.5 * 255 = 31.875 and lands on 31, not the 32 the docstrings round
    it to. Rounding here would put every pixel of both images one level apart.
    """
    return (np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0) * 255).astype(np.uint8)


def cellmask_shape(path):
    """(height, width) from the h5 header, without loading the array."""
    import h5py

    with h5py.File(path, 'r') as f:
        return tuple(f['cellmask'].shape[:2])


def canvas_offsets(metadata, path):
    """(min_x, min_y) under whichever names the writer used.

    stitch_slices.py:477 stores min_x/min_y; downsample_subslices_cellmask.py:283
    stores min_x_offset/min_y_offset. Neither present raises: a default of 0
    displaces every centroid by the canvas origin, and the render still looks
    like a render.
    """
    for x_key, y_key in (("min_x", "min_y"), ("min_x_offset", "min_y_offset")):
        if x_key in metadata and y_key in metadata:
            return int(metadata[x_key]), int(metadata[y_key])

    raise KeyError(
        f"No canvas offset in {path.name}.\n"
        f"  Expected min_x/min_y (stitch_slices.py) or "
        f"min_x_offset/min_y_offset (downsample_subslices_cellmask.py).\n"
        f"  Present: {sorted(metadata)}")


def discover(input_dir, targets=None):
    """[(slice_id, path)] ascending, for every whole-slice cellmask on disk."""
    found = {}
    for path in input_dir.glob("slice*_CELLMASK.h5"):
        match = SLICE_RE.fullmatch(path.stem)
        if match:
            found[int(match.group(1))] = path

    if targets is not None:
        missing = sorted(set(targets) - set(found))
        if missing:
            available = sorted(found)
            raise ValueError(
                f"No whole-slice cellmask for slice(s): {missing}\n"
                f"  Looked in: {input_dir}\n"
                f"  Available: {available if available else 'none'}")
        found = {s: p for s, p in found.items() if s in targets}

    return [(s, found[s]) for s in sorted(found)]


def centroid_pixels(pos_rows, min_x_offset, min_y_offset):
    """Full-res centroids -> 0-indexed (y, x) in the downsampled canvas.

    generate_marker_cellmask_subslice.py:496-504 over an array instead of a
    loop. np.round and Python's round() are both half-to-even, so the pixel a
    cell lands on is the one step 4 picks.
    """
    x = np.round(
        (pos_rows[:, 0] * 2 - (min_x_offset - 1)) / DOWNSAMPLE_XY
    ).astype(np.int64) - 1
    y = np.round(
        (pos_rows[:, 1] * 2 - (min_y_offset - 1)) / DOWNSAMPLE_XY
    ).astype(np.int64) - 1
    return y, x


def paint(cellmask, ids, colors, grey):
    """RGB uint8: every labelled pixel grey, the listed ids their ramp colour.

    A table indexed by cell id, one pass over the canvas. stitch_subslices.py:225
    renumbers the stitched cellmask with a running offset, so its ids are dense
    and bounded by the section's own cell count -- the table is a few hundred
    kilobytes whatever the canvas is. Step 4's `cellmask == cell_id` scan costs
    one full pass per drawn cell instead.

    `ids` is assigned in the caller's cell order, so two centroids landing on one
    blob resolve to the last of them, as the loop does.
    """
    lut = np.zeros((int(cellmask.max()) + 1, 3), dtype=np.uint8)
    lut[1:] = grey
    lut[ids] = colors
    return lut[cellmask]


def build_marker_state(marker, filt_neurons, expmat, pass_qc):
    """Everything about one marker that does not change from section to section."""
    settings = get_marker(marker)
    floor = settings["floor"]
    ceiling = settings["ceiling"]
    label = settings["label"]

    column = resolve_marker_column(
        filt_neurons, settings["gene_name"], settings["column"])
    expression = get_expression_column(expmat, column)

    marker_qc_pass = pass_qc & (expression > 0)
    drawn = marker_qc_pass & (expression >= floor)

    print(f"{label} (column {column}), ramp {ROLONY_RAMP_MIN} -> {ceiling}+, "
          f"drawn at >= {floor}")
    print(f"  {label}+ cells: {int(np.sum(expression > 0))}")
    print(f"  {label}+ QC-passing: {int(np.sum(marker_qc_pass))}")
    print(f"  at >= {floor} rolonies (drawn): {int(np.sum(drawn))}")
    print(f"  at >= {ceiling} rolonies (saturated): "
          f"{int(np.sum(marker_qc_pass & (expression >= ceiling)))}")
    print(f"  max (QC-passed): {int(np.max(expression[pass_qc]))} rolonies")

    return {
        "label": label,
        "floor": floor,
        "ceiling": ceiling,
        "cmap": make_ramp(settings["ramp"]),
        "expression": expression,
        "marker_qc_pass": marker_qc_pass,
        "drawn": drawn,
        "out_dir": MARKER_OUTPUT_ROOTS[marker] / f"rolony_ge{floor}_sat{ceiling}",
    }


def generate_marker_cellmask_slice(targets=None, markers=None, dry_run=False):
    """Whole-slice marker overlays for every marker in `markers`."""
    markers = list(markers) if markers else marker_names()

    input_dir = Path(HYB_SLICE_DOWNSAMPLED_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Downsampled whole slices not found: {input_dir}\n"
            f"  Run stitch_slices.py --downsample first.")

    entries = discover(input_dir, targets)
    if not entries:
        raise FileNotFoundError(
            f"No slice*_CELLMASK.h5 in {input_dir}\n"
            f"  Run stitch_slices.py --downsample first.")

    print("=" * 40)
    print("WHOLE-SLICE MARKER CELL MASK OVERLAYS")
    print("=" * 40)
    print(f"Input:   {input_dir}")
    print(f"Markers: {', '.join(markers)}")
    print(f"Slices:  {len(entries)}")
    print(f"Pitch:   {TARGET_XY_UM_PER_PX:.4f} um/px "
          f"(downsample {DOWNSAMPLE_XY:.4f}x, both axes)")
    print(f"Cell mask brightness: {CELLMASK_BRIGHTNESS * CELLMASK_SCALE:.3f} "
          f"-> uint8 {int(to_uint8(CELLMASK_BRIGHTNESS * CELLMASK_SCALE))}\n")

    print("Loading filt_neurons...")
    filt_neurons = load_filt_neurons(FILT_NEURONS_PATH)
    expmat = filt_neurons['expmat']
    n_cells = expmat.shape[0]

    if sparse.issparse(expmat):
        total_reads = np.asarray(expmat.sum(axis=1)).flatten()
        total_genes = np.asarray((expmat > 0).sum(axis=1)).flatten()
    else:
        total_reads = np.sum(expmat, axis=1)
        total_genes = np.sum(expmat > 0, axis=1)

    pass_qc = (total_reads >= QC_MIN_READS) & (total_genes >= QC_MIN_GENES)
    print(f"  Total cells: {n_cells}")
    print(f"  QC ({QC_MIN_READS} reads / {QC_MIN_GENES} genes): "
          f"{int(np.sum(pass_qc))} pass ({100*np.sum(pass_qc)/n_cells:.1f}%)\n")

    states = [build_marker_state(m, filt_neurons, expmat, pass_qc)
              for m in markers]
    print()

    slice_ids = np.asarray(filt_neurons['slice']).flatten()
    pos = np.asarray(filt_neurons['pos'])

    if dry_run:
        # Shapes off the h5 headers, cell counts off filt_neurons: everything
        # the real run reports except what the centroid lookup finds.
        total_mb = 0.0
        for slice_id, path in entries:
            height, width = cellmask_shape(path)
            mb = height * width * 3 / 1e6
            counts = "  ".join(
                f"{s['label']} {int(np.sum(s['drawn'] & (slice_ids == slice_id)))}"
                for s in states)
            total_mb += mb * len(states)
            print(f"  Slice {slice_id}: {width} x {height} "
                  f"({height*width/1e6:.1f} Mpx), {mb:.0f} MB per marker   {counts}")
        print(f"\n{len(entries)} slices x {len(states)} markers = "
              f"{len(entries)*len(states)} TIFs, {total_mb/1000:.1f} GB")
        print("Dry run: nothing written.")
        return

    for state in states:
        state["out_dir"].mkdir(parents=True, exist_ok=True)
        legend = write_ramp_legend(state["out_dir"], state["floor"],
                                   state["cmap"], state["ceiling"],
                                   state["label"])
        print(f"{state['label']}: {state['out_dir']}")
        print(f"  legend {Path(legend).name}")
    print()

    for i, (slice_id, path) in enumerate(entries):
        print("=" * 40)
        print(f"[{i+1}/{len(entries)}] Slice {slice_id}")
        print("=" * 40)

        load_start = time.time()
        cellmask, metadata = load_cellmask_h5(path)
        min_x_offset, min_y_offset = canvas_offsets(metadata, path)
        print(f"  Cellmask: {cellmask.shape[1]} x {cellmask.shape[0]} "
              f"(canvas offset: x={min_x_offset}, y={min_y_offset}) "
              f"in {time.time() - load_start:.1f} sec")

        in_slice = slice_ids == slice_id
        print(f"  Cells in slice: {int(np.sum(in_slice))}")

        for state in states:
            label = state["label"]
            drawn_idx = np.where(state["drawn"] & in_slice)[0]
            eligible = int(np.sum(state["marker_qc_pass"] & in_slice))
            print(f"  {label}+ QC-passing: {eligible}, "
                  f"drawn at >= {state['floor']}: {len(drawn_idx)}")

            paint_start = time.time()
            if len(drawn_idx):
                y, x = centroid_pixels(pos[drawn_idx], min_x_offset, min_y_offset)
                in_bounds = ((x >= 0) & (x < cellmask.shape[1]) &
                             (y >= 0) & (y < cellmask.shape[0]))
                ids = np.zeros(len(drawn_idx), dtype=cellmask.dtype)
                ids[in_bounds] = cellmask[y[in_bounds], x[in_bounds]]

                hit = ids > 0
                colors = np.zeros((int(np.sum(hit)), 3), dtype=np.uint8)
                for row, cell_idx in enumerate(drawn_idx[hit]):
                    colors[row] = to_uint8(
                        ramp_rgb(state["expression"][cell_idx],
                                 state["cmap"], state["ceiling"]))

                overlay = paint(cellmask, ids[hit], colors,
                                to_uint8(CELLMASK_BRIGHTNESS * CELLMASK_SCALE))
                mapped = int(np.sum(hit))
                print(f"    Mapped {mapped} / {len(drawn_idx)} cells "
                      f"({100*mapped/len(drawn_idx):.1f}%), "
                      f"{int(np.sum(~in_bounds))} out of bounds, "
                      f"{int(np.sum(in_bounds & ~hit))} on background, "
                      f"in {time.time() - paint_start:.1f} sec")
            else:
                print(f"    WARNING: no cells at >= {state['floor']} rolonies, "
                      f"saving cell mask only")
                overlay = paint(cellmask,
                                np.zeros(0, dtype=cellmask.dtype),
                                np.zeros((0, 3), dtype=np.uint8),
                                to_uint8(CELLMASK_BRIGHTNESS * CELLMASK_SCALE))

            out_path = state["out_dir"] / f"slice{slice_id}_{label}_cellmask.tif"
            imwrite_tiff(out_path, overlay)
            print(f"    Saved {out_path.name} "
                  f"({get_file_size_mb(out_path):.1f} MB)")

            # Freed before the next marker paints and before the next section
            # loads: at ~24 Mpx the overlay is 71 MB and the cellmask 94 MB, and
            # holding either while its successor allocates doubles the peak.
            del overlay

        del cellmask
        print()

    print("=" * 40)
    print("WHOLE-SLICE OVERLAYS COMPLETE")
    print("=" * 40)
    for state in states:
        print(f"{state['label']}: {state['out_dir']}")
    print(f"\nSlices: {len(entries)}   Resolution: {TARGET_XY_UM_PER_PX:.4f} um/px")
    print("A count is the same colour here as in the subslice render at the "
          "same cutoff and cap; the subslice is a crop of this grid.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Marker cellmask overlays on whole slices, both markers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--marker', '-m', choices=marker_names(), default=None,
                        help='Render this marker only (default: every marker)')
    parser.add_argument('--slice', '-s', type=int, default=None,
                        help='Process this slice only')
    parser.add_argument('--slices', type=int, nargs='+', default=None,
                        help='Process these slices only')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report canvas sizes and cell counts, write nothing')
    args = parser.parse_args()

    targets = None
    if args.slice is not None or args.slices is not None:
        targets = set(args.slices or [])
        if args.slice is not None:
            targets.add(args.slice)

    generate_marker_cellmask_slice(
        targets=targets,
        markers=[args.marker] if args.marker else None,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
