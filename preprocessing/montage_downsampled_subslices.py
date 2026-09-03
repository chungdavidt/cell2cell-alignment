#!/usr/bin/env python3
"""
Tile every downsampled subslice into one montage per channel.

A contact sheet, for looking at all 62 sections at once. It stitches nothing:
step 3 already wrote each section as a single image, and this reads those files
and pastes them into a grid. `filt_neurons.mat`, `subslice_definitions.mat` and
the raw FOVs are never opened.

Input (written by downsample_subslices_cellmask.py, step 3, TARGET_XY_UM_PER_PX):

    <OUTPUT_ROOT>/HYB_subslice_stitched_tif_downsampled_micronwise/
        slice{N}_subslice_{GCAMP,DAPI,MSCARLET}.tif

Output:

    <OUTPUT_ROOT>/HYB_subslice_downsampled_montage/montage_{CHANNEL}.tif

The three montages share one geometry, so a cell in the DAPI montage is the same
cell in the MSCARLET montage. Slices run left to right in ascending order,
wrapping every --columns (10 by default): 62 sections fill 6 rows and 2 cells of
a 7th, and the remaining 8 cells stay black.

GRID
    Each section's canvas is a bounding box over its own FOVs, so no two are the
    same size and each has to be padded to fill its cell. Every cell is the same
    size here -- the widest tile by the tallest -- so the grid lines are evenly
    spaced and one section does not shift the others. The alternative, sizing
    each column to its own widest tile and each row to its own tallest, packs
    into fewer pixels but makes the layout depend on which section lands where;
    --dry-run prints what it would have cost so the choice can be revisited.

    A tile sits at the TOP-LEFT of its cell, and the label and scale bars are
    drawn at the tile's own edges rather than the cell's, so they stay against
    the tissue instead of floating in the padding.

WHAT IS BURNED IN
    Each tile carries "slice N" in its top-left and, by default, the same scale
    bars stitch_slices.py burns into the whole-slice TIFs in its bottom-left.
    Both are white at the montage's own maximum, both report how many nonzero
    pixels they covered. `draw_scale_bars`, `_render_label` and every SCALE_BAR_*
    constant are imported from stitch_slices.py rather than copied, so a retune
    there reaches here.

    At TARGET_XY_UM_PER_PX the bars are much smaller than they are on a
    full-resolution section -- 1.1 um/px puts the 250 um bar at 227 px -- which
    is why they are per tile and not one to a montage: one bar in the corner of a
    30,000 px image is invisible zoomed out and absent wherever you are zoomed
    in. --scale-bars montage draws a single set at the montage's bottom-left
    instead, --scale-bars none draws none.

PIXELS
    Raw uint16 carried through with no per-tile normalization, so one contrast
    window in a viewer compares sections against each other honestly. The
    cellmask (slice{N}_subslice_CELLMASK.h5) is not tiled: it is a uint32 label
    array where a scale bar would be a fake cell id.

The output folder is hardcoded below rather than added to preprocessing_config,
because no pipeline step reads it. Nothing can pick these files up despite the
`subslice` token in the folder name: every glob in the pipeline is a
per-directory `Path(dir).glob`, and downsample_subslices_cellmask.py,
assign_orientation.py and the graph builder all glob directories named in the
config, which this folder is not. The files themselves are named montage_*, so
even copied into one of those directories they would match no pattern.

Usage:
    python montage_downsampled_subslices.py --dry-run   # geometry + sizes, writes nothing
    python montage_downsampled_subslices.py
    python montage_downsampled_subslices.py --columns 8
    python montage_downsampled_subslices.py --slices 10 22 30   # a small test sheet
    python montage_downsampled_subslices.py --channel DAPI
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

from preprocessing_config import (
    HYB_DOWNSAMPLED_DIR,
    OUTPUT_ROOT,
    TARGET_XY_UM_PER_PX,
)
from stitch_slices import (
    draw_scale_bars,
    _render_label,
    SCALE_BAR_LENGTHS_UM,
    SCALE_BAR_MARGIN_UM,
    SCALE_BAR_THICKNESS_UM,
    SCALE_BAR_LABEL_SCALE,
)
from utilities.image_io import (
    imread_tiff,
    imwrite_tiff,
    get_tiff_info,
    get_file_size_mb,
)


# Not in preprocessing_config: no pipeline step reads it.
OUTPUT_DIR = Path(OUTPUT_ROOT) / "HYB_subslice_downsampled_montage"

CHANNELS = ("GCAMP", "DAPI", "MSCARLET")

# Step 3 resamples to the 2P in-plane pitch, so that -- not EXVIVO_UM_PER_PX --
# is what a micron is worth in these files.
UM_PER_PX = TARGET_XY_UM_PER_PX

DEFAULT_COLUMNS = 10

# Black seam between cells, in um so it holds at any pitch. 100 um is 91 px at
# 1.1 um/px. Not drawn around the outside of the grid.
MONTAGE_GUTTER_UM = 100.0

# Height of the "slice N" label, in um like the scale bar constants. 200 um is
# 182 px at 1.1 um/px, which is legible with the whole 7-row sheet on screen.
# Expect to retune this by eye.
SLICE_LABEL_HEIGHT_UM = 200.0

SLICE_RE = re.compile(r'slice(\d+)_subslice')


def discover(input_dir, targets=None):
    """[(slice_id, {channel: path})], slice order ascending.

    A slice appears if at least one of its channel TIFs is present. A channel
    missing from a slice leaves that cell black in that channel's montage only --
    the grid is measured over every file found, so the three montages keep one
    geometry whatever is missing.
    """
    found = {}
    for channel in CHANNELS:
        for path in input_dir.glob(f"slice*_subslice_{channel}.tif"):
            match = SLICE_RE.search(path.name)
            if match:
                found.setdefault(int(match.group(1)), {})[channel] = path

    if targets is not None:
        missing = sorted(set(targets) - set(found))
        if missing:
            available = sorted(found)
            raise ValueError(
                f"No downsampled subslice TIFs for slice(s): {missing}\n"
                f"  Looked in: {input_dir}\n"
                f"  Available: {available if available else 'none'}"
            )
        found = {s: c for s, c in found.items() if s in targets}

    return [(s, found[s]) for s in sorted(found)]


def read_headers(entries):
    """{slice_id: (height, width)} and the common dtype, from TIFF headers only.

    Step 3 resamples every channel of a section to one target shape computed from
    its cellmask, so a section whose channels disagree means those files were not
    written by the same run, and the montage would be tiling two geometries.
    """
    shapes = {}
    dtypes = set()
    for slice_id, channels in entries:
        per_channel = {}
        for channel, path in channels.items():
            info = get_tiff_info(path)
            per_channel[channel] = tuple(info['shape'][:2])
            dtypes.add(np.dtype(info['dtype']))
        distinct = set(per_channel.values())
        if len(distinct) > 1:
            detail = ", ".join(f"{c}={s[1]}x{s[0]}" for c, s in sorted(per_channel.items()))
            raise ValueError(
                f"Slice {slice_id}: channels have different shapes ({detail}).\n"
                f"  Step 3 resamples every channel of a section to one shape, so\n"
                f"  these files are from different runs. Re-run\n"
                f"  downsample_subslices_cellmask.py --slice {slice_id}."
            )
        shapes[slice_id] = distinct.pop()

    if len(dtypes) > 1:
        raise ValueError(
            f"Downsampled subslices have mixed dtypes: {sorted(str(d) for d in dtypes)}.\n"
            f"  One montage cannot hold both. Re-run step 3 for the odd ones out."
        )
    return shapes, dtypes.pop()


def plan_grid(shapes, columns):
    """Cell size, grid shape and the montage's pixel extent.

    Returns a dict; `packed_px` is what per-column-width / per-row-height would
    have come to, reported for comparison and not used to place anything.
    """
    slice_ids = sorted(shapes)
    rows = -(-len(slice_ids) // columns)          # ceil
    gutter = int(round(MONTAGE_GUTTER_UM / UM_PER_PX))

    cell_h = max(h for h, _ in shapes.values())
    cell_w = max(w for _, w in shapes.values())

    # Same row-major placement, sizing each column to its own widest tile and
    # each row to its own tallest.
    col_w = [0] * columns
    row_h = [0] * rows
    for i, slice_id in enumerate(slice_ids):
        h, w = shapes[slice_id]
        col_w[i % columns] = max(col_w[i % columns], w)
        row_h[i // columns] = max(row_h[i // columns], h)

    return {
        'slice_ids': slice_ids,
        'columns': columns,
        'rows': rows,
        'gutter': gutter,
        'cell_h': cell_h,
        'cell_w': cell_w,
        'height': rows * cell_h + (rows - 1) * gutter,
        'width': columns * cell_w + (columns - 1) * gutter,
        'packed_px': ((sum(row_h) + (rows - 1) * gutter)
                      * (sum(col_w) + (columns - 1) * gutter)),
    }


def cell_origin(index, grid):
    """Top-left pixel (y, x) of the cell holding the index'th slice."""
    row, col = divmod(index, grid['columns'])
    return (row * (grid['cell_h'] + grid['gutter']),
            col * (grid['cell_w'] + grid['gutter']))


def draw_slice_label(image, text, white):
    """Burn `text` into the top-left of `image`, in place.

    Returns (overwritten, drawn) like draw_scale_bars: how many of the pixels
    written were nonzero beforehand, out of how many were written. The top-left
    corner of a bounding box over FOV placements is usually empty but is not
    guaranteed to be, and unlike a scale bar this one cannot be moved to a
    quieter corner without leaving the tile.
    """
    height, width = image.shape[:2]
    margin = int(round(SCALE_BAR_MARGIN_UM / UM_PER_PX))
    label = _render_label(text, SLICE_LABEL_HEIGHT_UM / UM_PER_PX)
    label_h, label_w = label.shape

    if margin + label_w > width or margin + label_h > height:
        print(f"    Label {text!r} does not fit this tile, skipped")
        return 0, 0

    patch = image[margin:margin + label_h, margin:margin + label_w]
    overwritten = int(np.count_nonzero(patch[label]))
    patch[label] = white
    return overwritten, int(label.sum())


def report_geometry(grid, dtype):
    """The grid, the bar geometry and the file size, once, before any pixels."""
    n = len(grid['slice_ids'])
    px = grid['height'] * grid['width']
    mb = px * np.dtype(dtype).itemsize / 1e6

    print(f"Grid:   {grid['columns']} x {grid['rows']} cells for {n} slices "
          f"({grid['columns'] * grid['rows'] - n} blank)")
    print(f"Cell:   {grid['cell_w']} x {grid['cell_h']} px "
          f"({grid['cell_w'] * UM_PER_PX:.0f} x {grid['cell_h'] * UM_PER_PX:.0f} um), "
          f"gutter {grid['gutter']} px")
    print(f"Sheet:  {grid['width']} x {grid['height']} px = {px/1e6:.0f} Mpx, "
          f"{mb:.0f} MB per channel {dtype}")
    print(f"        per-column/per-row cells would be {grid['packed_px']/1e6:.0f} Mpx "
          f"({100 * grid['packed_px'] / px:.0f}% of this)")
    print(f"Scale bars at {UM_PER_PX:g} um/px:")
    thickness = max(1, int(round(SCALE_BAR_THICKNESS_UM / UM_PER_PX)))
    for length_um in sorted(SCALE_BAR_LENGTHS_UM):
        print(f"  {length_um:g} um -> {int(round(length_um / UM_PER_PX))} px")
    print(f"  thickness {SCALE_BAR_THICKNESS_UM:g} um -> {thickness} px")
    print(f"  bar label -> {int(round(SCALE_BAR_LABEL_SCALE * thickness))} px")
    print(f"  slice label {SLICE_LABEL_HEIGHT_UM:g} um -> "
          f"{int(round(SLICE_LABEL_HEIGHT_UM / UM_PER_PX))} px")
    print(f"  corner inset {SCALE_BAR_MARGIN_UM:g} um -> "
          f"{int(round(SCALE_BAR_MARGIN_UM / UM_PER_PX))} px\n")


def build_channel(channel, entries, shapes, grid, dtype, scale_bars):
    """One montage array for one channel, tiles pasted and annotations burned.

    Two passes over the slices, because the white the annotations are drawn in
    is the finished sheet's own maximum and so cannot be known until every tile
    is down.
    """
    by_slice = dict(entries)
    montage = np.zeros((grid['height'], grid['width']), dtype=dtype)

    drawn_on = []
    for index, slice_id in enumerate(grid['slice_ids']):
        path = by_slice[slice_id].get(channel)
        if path is None:
            print(f"    slice {slice_id}: no {channel} tif, cell left black")
            continue
        tile = imread_tiff(path)
        h, w = shapes[slice_id]
        if tile.shape[:2] != (h, w) or tile.ndim != 2:
            raise ValueError(
                f"{path.name} is {tile.shape} on disk but its header said "
                f"{(h, w)}; the file changed under the run.")
        y0, x0 = cell_origin(index, grid)
        montage[y0:y0 + h, x0:x0 + w] = tile
        drawn_on.append((slice_id, y0, x0, h, w))

    if not drawn_on:
        return montage, 0

    # White is the montage's own maximum, not the dtype's: one 65535 pixel in a
    # uint16 fluorescence image that really peaks at a few thousand makes every
    # autoscaling viewer darken the tissue. Taken once over the whole sheet so
    # every annotation on it is the same white.
    white = int(montage.max()) or int(np.iinfo(dtype).max)

    overwritten = drawn = 0
    for slice_id, y0, x0, h, w in drawn_on:
        tile_view = montage[y0:y0 + h, x0:x0 + w]

        o, d = draw_slice_label(tile_view, f"slice {slice_id}", white)
        overwritten += o
        drawn += d
        if scale_bars == 'tile':
            o, d = draw_scale_bars(tile_view, UM_PER_PX)
            overwritten += o
            drawn += d

    if scale_bars == 'montage':
        o, d = draw_scale_bars(montage, UM_PER_PX)
        overwritten += o
        drawn += d

    print(f"    labels and bars covered {overwritten}/{drawn} nonzero px")
    return montage, len(drawn_on)


def montage_subslices(targets=None, columns=DEFAULT_COLUMNS, channels=CHANNELS,
                      scale_bars='tile', dry_run=False):
    input_dir = Path(HYB_DOWNSAMPLED_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Downsampled subslices not found: {input_dir}\n"
            f"  Run downsample_subslices_cellmask.py (pipeline step 3) first."
        )
    if OUTPUT_DIR.resolve() == input_dir.resolve():
        raise ValueError(
            f"Output directory is the input directory: {OUTPUT_DIR}\n"
            f"  This script must never write over step 3's subslices."
        )

    entries = discover(input_dir, targets)
    if not entries:
        raise FileNotFoundError(
            f"No slice*_subslice_{{{','.join(CHANNELS)}}}.tif in {input_dir}"
        )

    shapes, dtype = read_headers(entries)
    grid = plan_grid(shapes, columns)

    print(f"Input:  {input_dir}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Slices: {len(entries)}\n")
    report_geometry(grid, dtype)

    if dry_run:
        for index, slice_id in enumerate(grid['slice_ids']):
            row, col = divmod(index, columns)
            h, w = shapes[slice_id]
            present = "".join(c[0] for c in CHANNELS if c in dict(entries)[slice_id])
            print(f"  slice {slice_id:<3} r{row} c{col}  {w} x {h}  [{present}]")
        print("\nDry run: nothing written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for channel in channels:
        print(f"{channel}")
        montage, placed = build_channel(
            channel, entries, shapes, grid, dtype, scale_bars)
        if not placed:
            print(f"    no {channel} tifs found, montage not written")
            continue
        out_path = OUTPUT_DIR / f"montage_{channel}.tif"
        imwrite_tiff(out_path, montage)
        print(f"    {placed} tiles -> {out_path.name}, "
              f"{montage.shape[1]} x {montage.shape[0]}, "
              f"{get_file_size_mb(out_path):.0f} MB")
        del montage

    print(f"\nWrote to {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Tile every downsampled subslice into one montage per channel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--columns', '-c', type=int, default=DEFAULT_COLUMNS,
                        help=f'Slices per row (default {DEFAULT_COLUMNS})')
    parser.add_argument('--slice', '-s', type=int, default=None,
                        help='Include this slice only')
    parser.add_argument('--slices', type=int, nargs='+', default=None,
                        help='Include these slices only')
    parser.add_argument('--channel', action='append', choices=CHANNELS,
                        dest='channels',
                        help='Montage this channel only; repeatable')
    parser.add_argument('--scale-bars', choices=('tile', 'montage', 'none'),
                        default='tile',
                        help='Where to burn scale bars (default: one set per tile)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report grid, sizes and bar geometry, write nothing')
    args = parser.parse_args()

    if args.columns < 1:
        parser.error("--columns must be at least 1")

    targets = None
    if args.slice is not None or args.slices is not None:
        targets = set(args.slices or [])
        if args.slice is not None:
            targets.add(args.slice)

    montage_subslices(targets=targets,
                      columns=args.columns,
                      channels=tuple(args.channels or CHANNELS),
                      scale_bars=args.scale_bars,
                      dry_run=args.dry_run)


if __name__ == '__main__':
    main()
