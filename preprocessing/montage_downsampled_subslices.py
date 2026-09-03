#!/usr/bin/env python3
"""
Tile downsampled sections into one image per ROW, per channel.

A contact sheet cut into strips. It stitches nothing and resamples nothing: the
sections are already single images at TARGET_XY_UM_PER_PX, and this reads them
and pastes them into a row. `filt_neurons.mat`, `subslice_definitions.mat` and
the raw FOVs are never opened.

TWO SOURCES, --source (default "slice")
    slice     the WHOLE section -- every FOV that has a cell in it, no QC
              filter, no marker filter, no connected components. From
              `stitch_slices.py --downsample --no-scale-bars`, in
              HYB_slice_stitched_tif_downsampled_micronwise/, named
              slice{N}_{CHANNEL}.tif.
    subslice  the marker-defined region the alignment is fitted on. Step 3's
              output, in HYB_subslice_stitched_tif_downsampled_micronwise/,
              named slice{N}_subslice_{CHANNEL}.tif.

    Different pictures, not two qualities of one. For eyeballing the sections
    the whole slice is the right one; for seeing what the graph actually fits
    on, it is not. Both are at the same pitch, which is the only reason one
    script covers them -- the tiling is identical and only the input folder,
    the filename shape and the output folder differ. They write to separate
    output folders, so neither can overwrite the other.

Output:

    <OUTPUT_ROOT>/HYB_{slice,subslice}_downsampled_montage/
        row{i}_slice{first}-{last}_montage_{CHANNEL}.tif

The row number leads so a listing groups a row's three channels together, and it
is zero-padded so row10 does not sort ahead of row2.

Slices run left to right in ascending order, --columns per row (5 by default,
David's call after seeing 10), so BY95's 62 sections come out as 13 files per
channel: twelve rows of 5 and a last row of 2. One sheet holding all 62 was the
first build and was **too unwieldy to work with (David, 2026-09-02)** --
45,969 x 27,146 px, 2,496 MB per channel.

ROWS ARE INDEPENDENT IMAGES, SO EACH SIZES ITS OWN CELLS
    Every section's canvas is a bounding box over its own FOVs, so no two are the
    same size and each has to be padded to fill its cell. Within one row every
    cell is the same size -- the widest tile in that row by the tallest -- but
    one row's cell has nothing to do with another's. That is the point of
    splitting: BY95's subslices run 931 px (a single FOV) in the first rows to
    4,515 px in the last, and on one sheet the small ones carried the large
    ones' padding, and the narrower the row the less of it there is. Measured on
    BY95's subslice shapes: one sheet 1,248 Mpx, 10 per row 426, 5 per row 368,
    with the largest single file falling from 2,496 MB to 349 to 174. Whole
    slices cover more tissue per section and so run larger.

    Within a row the three channels share one geometry, so a cell in the DAPI
    strip is the same cell in the MSCARLET strip. Across rows they do not, and
    do not need to.

    A tile sits at the TOP-LEFT of its cell.

NOTHING IS BURNED IN
    No scale bars, no slice numbers -- removed 2026-09-02 after the first run,
    and the reason is not that they landed on tissue. They were drawn at the
    image's own maximum, and every autoscaling viewer picks its display maximum
    by walking the histogram, so the ink pinned the display range and rendered
    the tissue black. That is also why `--source slice` reads the `--downsample
    --no-scale-bars` folder and not HYB_slice_stitched_tif/, whose TIFs have
    stitch_slices.py's bars burned into their pixels and cannot have them taken
    back out. Every tile is the same pitch as every other and the order is
    ascending slice number; the filename carries the slice range.

PIXELS
    Raw uint16 carried through with no per-tile normalization, so one contrast
    window compares sections against each other honestly. The cost is that one
    bright pixel sets the window for its whole row: BY95's slice 53 holds a
    saturated 65,328 where every other section tops out at 3,780, and with a
    0-65,328 display range tissue at a mean of 10-40 is black. The per-tile max
    and mean are printed as each tile is pasted, and each row reports its own
    maximum with the slice holding it -- read those, then set the viewer's
    display range by hand (0-1000 works for BY95 DAPI). Do not press Auto.

    The cellmask (slice{N}_..._CELLMASK.h5) is not tiled: it is a uint32 label
    array, not a picture.

The output folders are hardcoded below rather than added to preprocessing_config,
because no pipeline step reads them. Nothing can pick these files up: every glob
in the pipeline is a per-directory `Path(dir).glob` over a folder named in the
config, which neither of these is, and the filenames start with `row` and contain
no `_subslice_` at all where every pipeline pattern is anchored at
`slice*_subslice_` or `*_subslice_ALIGN`.

Usage:
    python montage_downsampled_subslices.py --dry-run   # rows + sizes, writes nothing
    python montage_downsampled_subslices.py                      # whole slices
    python montage_downsampled_subslices.py --source subslice
    python montage_downsampled_subslices.py --columns 10
    python montage_downsampled_subslices.py --slices 10 22 30    # one short row
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
    HYB_SLICE_DOWNSAMPLED_DIR,
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
CHANNELS = ("GCAMP", "DAPI", "MSCARLET")

# Step 3 resamples to the 2P in-plane pitch, so that -- not EXVIVO_UM_PER_PX --
# is what a micron is worth in these files.
UM_PER_PX = TARGET_XY_UM_PER_PX

DEFAULT_COLUMNS = 5

# Black seam between cells, in um so it holds at any pitch. 100 um is 91 px at
# 1.1 um/px. Not drawn at either end of the row.
MONTAGE_GUTTER_UM = 100.0

# What can be tiled. Both are already at TARGET_XY_UM_PER_PX, which is the only
# reason one script covers them: the tiling is identical and only the input
# folder, the filename shape and the output folder differ.
#
#   subslice  the marker-defined region the alignment is fitted on
#   slice     the whole section, every FOV that has a cell in it -- no QC
#             filter, no marker filter, no connected components
#
# They are different pictures, not two qualities of one. For eyeballing the
# sections the whole slice is the right one; for seeing what the graph sees it
# is not.
SOURCES = {
    'subslice': {
        'input': HYB_DOWNSAMPLED_DIR,
        'output': "HYB_subslice_downsampled_montage",
        'name': "slice{slice}_subslice_{channel}.tif",
        'made_by': "downsample_subslices_cellmask.py (pipeline step 3)",
    },
    'slice': {
        'input': HYB_SLICE_DOWNSAMPLED_DIR,
        'output': "HYB_slice_downsampled_montage",
        'name': "slice{slice}_{channel}.tif",
        'made_by': "stitch_slices.py --downsample --no-scale-bars",
    },
}
DEFAULT_SOURCE = 'slice'


def source_dirs(source):
    """(input_dir, output_dir) for one source. Not in preprocessing_config:
    no pipeline step reads either montage folder."""
    spec = SOURCES[source]
    return Path(spec['input']), Path(OUTPUT_ROOT) / spec['output']


def _pattern(source, channel):
    r"""(glob, compiled regex) for one channel of one source.

    The regex is anchored, so the whole-slice form cannot also pick up a
    subslice file: `slice*_DAPI.tif` as a glob would match
    `slice22_subslice_DAPI.tif`, `^slice(\d+)_DAPI\.tif$` will not.
    """
    template = SOURCES[source]['name']
    glob = template.format(slice="*", channel=channel)
    regex = ("^"
             + re.escape(template)
                 .replace(re.escape("{slice}"), r"(\d+)")
                 .replace(re.escape("{channel}"), re.escape(channel))
             + "$")
    return glob, re.compile(regex)


def discover(input_dir, source, targets=None):
    """[(slice_id, {channel: path})], slice order ascending.

    A slice appears if at least one of its channel TIFs is present. A channel
    missing from a slice leaves that cell black in that channel's strip only --
    a row is measured over every file found in it, so its three channels keep
    one geometry whatever is missing.
    """
    found = {}
    for channel in CHANNELS:
        glob, pattern = _pattern(source, channel)
        for path in input_dir.glob(glob):
            match = pattern.match(path.name)
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
    """Filename stem: the row's position first, then its slices, then the channel.

    Row first so a directory listing groups the three channels of one row
    together instead of scattering them across three channel blocks, and
    zero-padded so row10 does not sort before row2 at a small --columns.
    """
    return (f"row{row['index']:02d}"
            f"_slice{row['slice_ids'][0]}-{row['slice_ids'][-1]}"
            f"_montage_{channel}")


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
                      source=DEFAULT_SOURCE, dry_run=False):
    spec = SOURCES[source]
    input_dir, output_dir = source_dirs(source)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"No {source} images at: {input_dir}\n"
            f"  Written by {spec['made_by']}."
        )
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError(
            f"Output directory is the input directory: {output_dir}\n"
            f"  This script must never write over its own input."
        )

    entries = discover(input_dir, source, targets)
    if not entries:
        raise FileNotFoundError(
            f"No {spec['name'].format(slice='*', channel='{' + ','.join(CHANNELS) + '}')} "
            f"in {input_dir}\n  Written by {spec['made_by']}."
        )

    shapes, dtype = read_headers(entries)
    rows = plan_rows(shapes, columns)

    print(f"Source: {source} ({spec['made_by']})")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
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

    output_dir.mkdir(parents=True, exist_ok=True)

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
            out_path = output_dir / f"{row_stem(channel, row)}.tif"
            imwrite_tiff(out_path, strip)
            written += 1
            print(f"    {placed} tiles -> {out_path.name}, "
                  f"{strip.shape[1]} x {strip.shape[0]}, "
                  f"{get_file_size_mb(out_path):.0f} MB")
            del strip

    print(f"\nWrote {written} files to {output_dir}")


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
    parser.add_argument('--source', choices=sorted(SOURCES),
                        default=DEFAULT_SOURCE,
                        help=f'Which images to tile (default {DEFAULT_SOURCE}): '
                             f'"slice" whole sections, "subslice" the '
                             f'marker-defined regions the alignment fits on')
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
                      source=args.source,
                      dry_run=args.dry_run)


if __name__ == '__main__':
    main()
