#!/usr/bin/env python3
"""
Stitch a WHOLE slice — every FOV in it — into per-channel TIFs.

The pipeline's stitched images are subslices: `identify_mscarlet_subslices.py`
filters cells by QC, then by mScarlet expression, keeps the largest 8-connected
component of the surviving FOVs and adds bridge FOVs across diagonals. That list
is what the alignment is fitted on. This script exists to see the section itself,
so it applies none of those screens.

The FOV list is read straight out of `filt_neurons.mat`:

    fov_list = unique(fov[slice == N])

No QC filter, no marker filter, no connected components, no bridges.
`subslice_definitions.mat` is not read at all. A FOV appears if it has at least
one row in that slice -- that mapping exists nowhere else, since the raw
directories are named `MAX_Pos{N}_{row}_{col}` with nothing naming a slice.

Stitching is `stitch_subslices.stitch_fov_channels`, imported rather than copied
so the two cannot drift. Everything it does is inherited as-is:

  - Each FOV is placed by regressing `pos*2 = offset + pos40x` on its own rows in
    that slice, counted over raw filt_neurons rows (the QC and marker screens do
    not reach this code).
  - A FOV with fewer than 3 rows in the slice CANNOT be placed and is skipped
    with a warning, and is left out of the canvas bounds. `--dry-run` reports how
    many FOVs that will be per slice.
  - Overlap between FOVs is max-projected per pixel for the three fluorescence
    channels. The cellmask is not: the later FOV overwrites, with a running
    global offset so labels stay unique across the canvas.

Output (full resolution, no downsampling):

    <OUTPUT_ROOT>/HYB_slice_stitched_tif/slice{N}_{GCAMP,DAPI,MSCARLET}.tif
    <OUTPUT_ROOT>/HYB_slice_stitched_tif/slice{N}_CELLMASK.h5

The names carry no `subslice` token on purpose: `downsample_subslices_cellmask.py`
globs `slice*_subslice_CELLMASK*` and the graph builder globs
`*_subslice_ALIGN.tif`, so neither can pick these up.

Four full-res canvases are allocated per slice at once -- three uint16 plus one
uint32 -- over the whole section rather than a marker-defined crop. `--dry-run`
prints the canvas each slice would need before anything is written.

Usage:
    python stitch_slices.py --dry-run          # FOV counts + canvas sizes, writes nothing
    python stitch_slices.py --slice 22
    python stitch_slices.py --slices 10 22 30
    python stitch_slices.py                    # every slice
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from preprocessing_config import (
    FILT_NEURONS_PATH,
    HYB_ROOT,
    HYB_CHANNELS_DIR,
    HYB_SLICE_STITCHED_DIR,
    FOV_SIZE,
)
from stitch_subslices import stitch_fov_channels
from utilities.mat_io import load_filt_neurons, save_cellmask_h5
from utilities.image_io import imwrite_tiff, get_file_size_mb
from utilities.regression import calculate_fov_offset


# stitch_fov_channels' own gate, mirrored so --dry-run can report what it will
# skip. calculate_fov_offset fits an intercept and a slope and raises below this.
MIN_ROWS_FOR_PLACEMENT = 3


def slice_fov_lists(filt_neurons, targets=None):
    """Every FOV per slice, from filt_neurons alone.

    Returns [(slice_id, fov_list, row_counts)], slice order ascending, where
    row_counts maps a FOV name to how many rows it has in that slice -- the
    count stitch_fov_channels regresses the FOV's canvas position on.
    """
    fov_names = np.asarray(filt_neurons['fov'])
    slice_ids = np.asarray(filt_neurons['slice']).flatten()

    # A NaN slice carries a FOV name but belongs to no section, so it enters no
    # list. ~40% of BY95's rows are NaN here.
    finite = ~np.isnan(slice_ids)
    available = np.unique(slice_ids[finite]).astype(int)

    if targets is not None:
        missing = sorted(set(targets) - set(available.tolist()))
        if missing:
            raise ValueError(
                f"Slice(s) not in filt_neurons: {missing}. "
                f"Available: {available.min()}-{available.max()}"
            )
        available = np.array(sorted(targets), dtype=int)

    out = []
    for slice_id in available:
        in_slice = finite & (slice_ids == slice_id)
        names = fov_names[in_slice]
        counts = Counter(str(n) for n in names)
        out.append((int(slice_id), sorted(counts), counts))
    return out


def plan_canvas(fov_list, slice_id, filt_neurons, row_counts):
    """What stitch_fov_channels would allocate, without loading an image.

    Runs the same per-FOV regression on the same rows, so the placed/skipped
    split and the canvas match the real run. Returns (height, width, n_placed).
    """
    fov_names = np.asarray(filt_neurons['fov'])
    slice_ids = np.asarray(filt_neurons['slice']).flatten()
    pos = np.asarray(filt_neurons['pos'])
    pos40x = np.asarray(filt_neurons['pos40x'])

    # One pass over the slice's rows instead of a full-length comparison per
    # FOV -- the answer is the same, and a dry run covers every slice.
    idx = np.where(slice_ids == slice_id)[0]
    groups = {}
    for j in idx:
        groups.setdefault(str(fov_names[j]), []).append(j)

    offsets = []
    for fov_name in fov_list:
        if row_counts[fov_name] < MIN_ROWS_FOR_PLACEMENT:
            continue
        i_cell = np.asarray(groups[fov_name])
        try:
            offsets.append(calculate_fov_offset(pos[i_cell], pos40x[i_cell], scale_factor=2.0))
        except Exception:
            continue

    if not offsets:
        return 0, 0, 0

    offsets = np.array(offsets)
    width = int(offsets[:, 0].max()) + FOV_SIZE - int(offsets[:, 0].min())
    height = int(offsets[:, 1].max()) + FOV_SIZE - int(offsets[:, 1].min())
    return height, width, len(offsets)


def stitch_slices(targets=None, dry_run=False):
    print("=" * 40)
    print("STITCH WHOLE SLICES")
    print("=" * 40)
    print("Mode: DRY RUN (nothing written)" if dry_run else f"Output: {HYB_SLICE_STITCHED_DIR}")
    print()

    print("Loading filt_neurons...")
    filt_neurons = load_filt_neurons(FILT_NEURONS_PATH)
    entries = slice_fov_lists(filt_neurons, targets)
    print(f"  Slices to process: {len(entries)}\n")

    if dry_run:
        print(f"{'slice':>6}  {'FOVs':>5}  {'<3 rows':>7}  {'placed':>6}  {'canvas (h x w)':>19}  {'RAM MB':>8}")
        for slice_id, fov_list, counts in entries:
            short = [f for f in fov_list if counts[f] < MIN_ROWS_FOR_PLACEMENT]
            height, width, n_placed = plan_canvas(fov_list, slice_id, filt_neurons, counts)
            # 3 uint16 channels + 1 uint32 cellmask, allocated together.
            ram_mb = height * width * (2 * 3 + 4) / 1e6
            print(f"{slice_id:>6}  {len(fov_list):>5}  {len(short):>7}  {n_placed:>6}  "
                  f"{height:>8} x {width:<8}  {ram_mb:>8.0f}")
            if short:
                print(f"          skipped: {', '.join(short)}")
        return

    output_dir = Path(HYB_SLICE_STITCHED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    hyb_root = Path(HYB_ROOT)
    channels_root = Path(HYB_CHANNELS_DIR)

    for i, (slice_id, fov_list, counts) in enumerate(entries):
        short = [f for f in fov_list if counts[f] < MIN_ROWS_FOR_PLACEMENT]
        print("=" * 40)
        print(f"[{i+1}/{len(entries)}] Stitching slice {slice_id} (whole slice)")
        print("=" * 40)
        print(f"  FOVs in slice: {len(fov_list)}")
        print(f"  FOVs with < {MIN_ROWS_FOR_PLACEMENT} rows (will be skipped): {len(short)}\n")

        stitch_start = time.time()
        gcamp, dapi, mscarlet, cellmask, min_x, min_y, fov_offsets = stitch_fov_channels(
            fov_list, slice_id, filt_neurons, hyb_root, channels_root
        )
        print(f"  Stitching completed in {time.time() - stitch_start:.1f} seconds\n")

        print("  Saving stitched channels...")
        prefix = f"slice{slice_id}"

        for channel, image in (("GCAMP", gcamp), ("DAPI", dapi), ("MSCARLET", mscarlet)):
            path = output_dir / f"{prefix}_{channel}.tif"
            imwrite_tiff(path, image)
            print(f"    {channel}: {get_file_size_mb(path):.1f} MB")

        cellmask_file = output_dir / f"{prefix}_CELLMASK.h5"
        save_cellmask_h5(cellmask_file, cellmask, metadata={
            'fov_offsets': fov_offsets,
            'min_x': min_x,
            'min_y': min_y,
        })
        print(f"    CELLMASK: {get_file_size_mb(cellmask_file):.1f} MB\n")

    print("=" * 40)
    print("STITCHING COMPLETE")
    print("=" * 40)
    print(f"Output directory: {output_dir}")
    print(f"Slices stitched: {len(entries)}")


def main():
    parser = argparse.ArgumentParser(
        description="Stitch every FOV of a slice into per-channel TIFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--slice', '-s', type=int, default=None,
                        help='Process this slice only')
    parser.add_argument('--slices', type=int, nargs='+', default=None,
                        help='Process these slices only')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report FOV counts and canvas sizes, write nothing')
    args = parser.parse_args()

    targets = None
    if args.slice is not None or args.slices is not None:
        targets = set(args.slices or [])
        if args.slice is not None:
            targets.add(args.slice)

    stitch_slices(targets=targets, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
