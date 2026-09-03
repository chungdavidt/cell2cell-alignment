#!/usr/bin/env python3
"""
TEMPORARY. Copy step 2's stitched subslices with scale bars burned in.

Visualization only, and meant to be thrown away with its output folder. The
subslices are already on disk; this reads them, burns the same bars
`stitch_slices.py` burns into the whole-slice TIFs, and writes the copies
somewhere nothing else looks. It stitches nothing -- `filt_neurons.mat`,
`subslice_definitions.mat` and the raw FOVs are never opened.

Input (written by stitch_subslices.py, step 2, full resolution, 0.32 um/px):

    <OUTPUT_ROOT>/HYB_subslice_stitched_tif/slice{N}_subslice_{GCAMP,DAPI,MSCARLET}.tif

Output:

    <OUTPUT_ROOT>/HYB_subslice_stitched_tif_scalebar/slice{N}_subslice_{CHANNEL}.tif

The input files are read and never written: the bars are drawn into the array in
memory and only the copy is saved. `slice{N}_subslice_CELLMASK.h5` is not copied
at all -- it is a uint32 label array where a bar would be a fake cell, and
nothing here needs it.

The output folder is hardcoded below rather than added to preprocessing_config,
because this is not part of the pipeline. Nothing can pick these files up
despite the `subslice` token in their names: every glob in the pipeline is a
per-directory `Path(dir).glob`, and `downsample_subslices_cellmask.py`,
`assign_orientation.py` and the graph builder all glob directories named in the
config, which this folder is not.

`draw_scale_bars` and every SCALE_BAR_* constant are imported from
`stitch_slices.py` rather than copied, so the bars here are the bars there and a
retune reaches both.

Usage:
    python add_subslice_scale_bars.py --dry-run      # geometry per slice, writes nothing
    python add_subslice_scale_bars.py --slice 22
    python add_subslice_scale_bars.py --slices 10 22 30
    python add_subslice_scale_bars.py               # every stitched subslice
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
    HYB_STITCHED_DIR,
    OUTPUT_ROOT,
    EXVIVO_UM_PER_PX,
)
from stitch_slices import (
    draw_scale_bars,
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


# Not in preprocessing_config: temporary, and no pipeline step reads it.
OUTPUT_DIR = Path(OUTPUT_ROOT) / "HYB_subslice_stitched_tif_scalebar"

CHANNELS = ("GCAMP", "DAPI", "MSCARLET")

# Step 2 writes at full resolution, so the source pitch is the BARseq pixel
# itself -- not TARGET_XY_UM_PER_PX, which only applies after step 3.
UM_PER_PX = EXVIVO_UM_PER_PX

SLICE_RE = re.compile(r'slice(\d+)_subslice')


def discover(input_dir, targets=None):
    """[(slice_id, {channel: path})], slice order ascending.

    A slice appears if at least one of its channel TIFs is present; a channel
    missing from a slice is reported and skipped rather than raising, since a
    partial step 2 run is a normal state to look at.
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
                f"No stitched subslice TIFs for slice(s): {missing}\n"
                f"  Looked in: {input_dir}\n"
                f"  Available: {available if available else 'none'}"
            )
        found = {s: c for s, c in found.items() if s in targets}

    return [(s, found[s]) for s in sorted(found)]


def report_geometry():
    """The bar geometry in pixels at this pitch, once, before any file."""
    print(f"Scale bars at {UM_PER_PX} um/px:")
    thickness = max(1, int(round(SCALE_BAR_THICKNESS_UM / UM_PER_PX)))
    for length_um in sorted(SCALE_BAR_LENGTHS_UM):
        print(f"  {length_um:g} um -> {int(round(length_um / UM_PER_PX))} px")
    print(f"  thickness {SCALE_BAR_THICKNESS_UM:g} um -> {thickness} px")
    print(f"  label height -> {int(round(SCALE_BAR_LABEL_SCALE * thickness))} px")
    print(f"  corner inset {SCALE_BAR_MARGIN_UM:g} um -> "
          f"{int(round(SCALE_BAR_MARGIN_UM / UM_PER_PX))} px\n")


def add_subslice_scale_bars(targets=None, dry_run=False):
    input_dir = Path(HYB_STITCHED_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Stitched subslices not found: {input_dir}\n"
            f"  Run stitch_subslices.py (pipeline step 2) first."
        )
    if OUTPUT_DIR.resolve() == input_dir.resolve():
        raise ValueError(
            f"Output directory is the input directory: {OUTPUT_DIR}\n"
            f"  This script must never write over step 2's subslices."
        )

    entries = discover(input_dir, targets)
    if not entries:
        raise FileNotFoundError(
            f"No slice*_subslice_{{{','.join(CHANNELS)}}}.tif in {input_dir}"
        )

    print(f"Input:  {input_dir}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Slices: {len(entries)}\n")
    report_geometry()

    if dry_run:
        # Runs the real draw_scale_bars over a blank canvas of each file's own
        # shape, read from the TIFF header, so the "does not fit" decisions are
        # the ones the real run will make. Occlusion is always 0/N here: there
        # is nothing under the bars to cover.
        for slice_id, channels in entries:
            print(f"  Slice {slice_id}")
            for channel in CHANNELS:
                path = channels.get(channel)
                if path is None:
                    print(f"    {channel}: missing, skipped")
                    continue
                info = get_tiff_info(path)
                height, width = info['shape'][:2]
                print(f"    {channel}: {width} x {height} {info['dtype']}, "
                      f"{get_file_size_mb(path):.1f} MB")
                draw_scale_bars(np.zeros((height, width), dtype=info['dtype']),
                                UM_PER_PX)
        print("\nDry run: nothing written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    for i, (slice_id, channels) in enumerate(entries):
        print(f"[{i+1}/{len(entries)}] Slice {slice_id}")
        for channel in CHANNELS:
            path = channels.get(channel)
            if path is None:
                print(f"    {channel}: missing, skipped")
                continue
            image = imread_tiff(path)
            occluded, painted = draw_scale_bars(image, UM_PER_PX)
            out_path = OUTPUT_DIR / path.name
            imwrite_tiff(out_path, image)
            written += 1
            print(f"    {channel}: {image.shape[1]} x {image.shape[0]}, "
                  f"{get_file_size_mb(out_path):.1f} MB"
                  f"   scale bars covered {occluded}/{painted} nonzero px")

    print(f"\nWrote {written} TIFs to {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy step 2's stitched subslices with scale bars burned in",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--slice', '-s', type=int, default=None,
                        help='Process this slice only')
    parser.add_argument('--slices', type=int, nargs='+', default=None,
                        help='Process these slices only')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report sizes and bar geometry, write nothing')
    args = parser.parse_args()

    targets = None
    if args.slice is not None or args.slices is not None:
        targets = set(args.slices or [])
        if args.slice is not None:
            targets.add(args.slice)

    add_subslice_scale_bars(targets=targets, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
