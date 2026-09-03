#!/usr/bin/env python3
"""
Tile the downsampled subslices into one image per ROW, per channel.

A contact sheet cut into strips. It stitches nothing: step 3 already wrote each
section as a single image, and this reads those files and pastes them into a
row. `filt_neurons.mat`, `subslice_definitions.mat` and the raw FOVs are never
opened.

Input (written by downsample_subslices_cellmask.py, step 3, TARGET_XY_UM_PER_PX):

    <OUTPUT_ROOT>/HYB_subslice_stitched_tif_downsampled_micronwise/
        slice{N}_subslice_{GCAMP,DAPI,MSCARLET}.tif

Output:

    <OUTPUT_ROOT>/HYB_subslice_downsampled_montage/
        montage_{CHANNEL}_row{i}_slice{first}-{last}.tif

Slices run left to right in ascending order, --columns per row (10 by default),
so BY95's 62 sections come out as 7 files per channel: six rows of 10 and a last
row of 2. One sheet holding all 62 was the first build and was **too unwieldy to
work with (David, 2026-09-02)** -- 45,969 x 27,146 px, 2,496 MB per channel.

ROWS ARE INDEPENDENT IMAGES, SO EACH SIZES ITS OWN CELLS
    Every section's canvas is a bounding box over its own FOVs, so no two are the
    same size and each has to be padded to fill its cell. Within one row every
    cell is the same size -- the widest tile in that row by the tallest -- but
    one row's cell has nothing to do with another's. That is the point of
    splitting: BY95's sections run 931 px (a single FOV) in the first rows to
    4,515 px in the last, and on one sheet the small ones carried the large
    ones' padding. Per row it is 407 Mpx against 1,248, and no single file is
    over ~350 MB.

    Within a row the three channels share one geometry, so a cell in the DAPI
    strip is the same cell in the MSCARLET strip. Across rows they do not, and
    do not need to.

    A tile sits at the TOP-LEFT of its cell.

NOTHING IS BURNED IN
    No scale bars, no slice numbers -- removed 2026-09-02 after the first run,
    and the reason is not that they landed on tissue. They were drawn at the
    image's own maximum, and every autoscaling viewer picks its display maximum
    by walking the histogram, so the ink pinned the display range and rendered
    the tissue black. Every tile is the same pitch as every other and the order
    is ascending slice number, which is what the labels were for; the filename
    carries the slice range.

PIXELS
    Raw uint16 carried through with no per-tile normalization, so one contrast
    window compares sections against each other honestly. The cost is that one
    bright pixel sets the window for its whole row: BY95's slice 53 holds a
    saturated 65,328 where every other section tops out at 3,780, and with a
    0-65,328 display range tissue at a mean of 10-40 is black. The per-tile max
    and mean are printed as each tile is pasted, and each row reports its own
    maximum with the slice holding it -- read those, then set the viewer's
    display range by hand (0-1000 works for BY95 DAPI). Do not press Auto.

    The cellmask (slice{N}_subslice_CELLMASK.h5) is not tiled: it is a uint32
    label array, not a picture.

The output folder is hardcoded below rather than added to preprocessing_config,
because no pipeline step reads it. Nothing can pick these files up despite the
`subslice` token in the folder name: every glob in the pipeline is a
per-directory `Path(dir).glob`, and downsample_subslices_cellmask.py,
assign_orientation.py and the graph builder all glob directories named in the
config, which this folder is not. The files themselves are named montage_*, so
even copied into one of those directories they would match no pattern.

Usage:
    python montage_downsampled_subslices.py --dry-run   # rows + sizes, writes nothing
    python montage_downsampled_subslices.py
    python montage_downsampled_subslices.py --columns 6
    python montage_downsampled_subslices.py --slices 10 22 30   # one short row
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
# 1.1 um/px. Not drawn at either end of the row.
MONTAGE_GUTTER_UM = 100.0

SLICE_RE = re.compile(r'slice(\d+)_subslice')


def discover(input_dir, targets=None):
    """[(slice_id, {channel: path})], slice order ascending.

    A slice appears if at least one of its channel TIFs is present. A channel
    missing from a slice leaves that cell black in that channel's strip only --
    a row is measured over every file found in it, so its three channels keep
    one geometry whatever is missing.
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
    written by the same run, and the strip would be tiling two geometries.
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
            f"  One image cannot hold both. Re-run step 3 for the odd ones out."
        )
    return shapes, dtypes.pop()


def plan_rows(shapes, columns):
    """One plan per row: its slices, its own cell size, its pixel extent.

    Each row is written as its own file, so its cell is the widest tile IN THAT
    ROW by the tallest. Nothing is shared between rows.
    """
    slice_ids = sorted(shapes)
    gutter = int(round(MONTAGE_GUTTER_UM / UM_PER_PX))

    rows = []
    for start in range(0, len(slice_ids), columns):
        members = slice_ids[start:start + columns]
        cell_h = max(shapes[s][0] for s in members)
        cell_w = max(shapes[s][1] for s in members)
        rows.append({
            'index': len(rows) + 1,            # 1-based, appears in the filename
            'slice_ids': members,
            'gutter': gutter,
            'cell_h': cell_h,
            'cell_w': cell_w,
            'height': cell_h,
            'width': len(members) * cell_w + (len(members) - 1) * gutter,
        })
    return rows


def cell_origin(index_in_row, row):
    """Top-left pixel (y, x) of the cell holding the index'th slice of `row`."""
    return (0, index_in_row * (row['cell_w'] + row['gutter']))


def row_stem(channel, row):
    """Filename stem: the row's position and the slice range it holds."""
    return (f"montage_{channel}_row{row['index']}"
            f"_slice{row['slice_ids'][0]}-{row['slice_ids'][-1]}")


def report_rows(rows, dtype):
    """Every row's cell, extent and file size, before any pixels."""
    itemsize = np.dtype(dtype).itemsize
    total_px = 0
    print(f"Rows:   {len(rows)} at {UM_PER_PX:g} um/px, "
          f"gutter {rows[0]['gutter']} px, dtype {dtype}")
    for row in rows:
        px = row['height'] * row['width']
        total_px += px
        print(f"  row {row['index']}  slices {row['slice_ids'][0]}-"
              f"{row['slice_ids'][-1]} ({len(row['slice_ids'])})  "
              f"cell {row['cell_w']} x {row['cell_h']}  ->  "
              f"{row['width']} x {row['height']} px, "
              f"{px * itemsize / 1e6:.0f} MB")
    print(f"  total {total_px/1e6:.0f} Mpx, {total_px * itemsize / 1e6:.0f} MB "
          f"per channel across {len(rows)} files\n")


def build_row(channel, entries, shapes, row, dtype):
    """One strip for one row of one channel, tiles pasted at their cell origins.

    Prints each tile's max and mean. Nothing is normalized -- the strip carries
    one contrast window for every section in it -- so one bright pixel sets the
    window for the row, and these are the numbers that say which.
    """
    by_slice = dict(entries)
    strip = np.zeros((row['height'], row['width']), dtype=dtype)

    placed = 0
    brightest = (0, None)
    for index, slice_id in enumerate(row['slice_ids']):
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
        y0, x0 = cell_origin(index, row)
        strip[y0:y0 + h, x0:x0 + w] = tile

        tile_max = int(tile.max())
        print(f"    slice {slice_id:<3} {w} x {h}  max {tile_max:>6}  "
              f"mean {float(tile.mean()):8.1f}")
        if tile_max > brightest[0]:
            brightest = (tile_max, slice_id)
        placed += 1

    if placed:
        print(f"    row max {brightest[0]} (slice {brightest[1]}) -- set the "
              f"viewer's display range by hand, not with Auto")
    return strip, placed


def montage_subslices(targets=None, columns=DEFAULT_COLUMNS, channels=CHANNELS,
                      dry_run=False):
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
    rows = plan_rows(shapes, columns)

    print(f"Input:  {input_dir}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Slices: {len(entries)}\n")
    report_rows(rows, dtype)

    if dry_run:
        for row in rows:
            for index, slice_id in enumerate(row['slice_ids']):
                h, w = shapes[slice_id]
                present = "".join(c[0] for c in CHANNELS
                                  if c in dict(entries)[slice_id])
                print(f"  row {row['index']} col {index}  slice {slice_id:<3} "
                      f"{w} x {h}  [{present}]")
        print("\nDry run: nothing written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    for channel in channels:
        print(f"{channel}")
        for row in rows:
            print(f"  row {row['index']} (slices {row['slice_ids'][0]}-"
                  f"{row['slice_ids'][-1]})")
            strip, placed = build_row(channel, entries, shapes, row, dtype)
            if not placed:
                print(f"    no {channel} tifs in this row, not written")
                continue
            out_path = OUTPUT_DIR / f"{row_stem(channel, row)}.tif"
            imwrite_tiff(out_path, strip)
            written += 1
            print(f"    {placed} tiles -> {out_path.name}, "
                  f"{strip.shape[1]} x {strip.shape[0]}, "
                  f"{get_file_size_mb(out_path):.0f} MB")
            del strip

    print(f"\nWrote {written} files to {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Tile the downsampled subslices into one image per row, per channel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--columns', '-c', type=int, default=DEFAULT_COLUMNS,
                        help=f'Slices per row, and so per file '
                             f'(default {DEFAULT_COLUMNS})')
    parser.add_argument('--slice', '-s', type=int, default=None,
                        help='Include this slice only')
    parser.add_argument('--slices', type=int, nargs='+', default=None,
                        help='Include these slices only')
    parser.add_argument('--channel', action='append', choices=CHANNELS,
                        dest='channels',
                        help='Montage this channel only; repeatable')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report rows and sizes, write nothing')
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
                      dry_run=args.dry_run)


if __name__ == '__main__':
    main()
