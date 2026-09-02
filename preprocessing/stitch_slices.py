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
so the two cannot drift:

  - Each FOV is placed by regressing `pos*2 = offset + pos40x` on its own rows in
    that slice, counted over raw filt_neurons rows (the QC and marker screens do
    not reach this code).
  - **Every FOV is stitched** — this passes `min_rows=1`. A FOV with 1 or 2 rows
    has too few points to leave the regression's slope free, so the slope is
    pinned at 1, which the model asserts anyway (`pos*2 = offset + pos40x`), and
    the offset is the mean residual. Step 2 keeps the default `min_rows=3` and
    is unchanged. `--dry-run` reports how many FOVs per slice are placed that way,
    and the range of the slopes the regression fits where it can — the quantity
    pinning replaces.
  - Overlap between FOVs is max-projected per pixel for the three fluorescence
    channels. The cellmask is not: the later FOV overwrites, with a running
    global offset so labels stay unique across the canvas.

Output (full resolution, no downsampling):

    <OUTPUT_ROOT>/HYB_slice_stitched_tif/slice{N}_{GCAMP,DAPI,MSCARLET}.tif
    <OUTPUT_ROOT>/HYB_slice_stitched_tif/slice{N}_CELLMASK.h5

Each channel TIF carries labelled scale bars BURNED INTO ITS PIXELS in the
bottom-left corner -- see SCALE_BAR_LENGTHS_UM below. They are drawn here, after
stitch_fov_channels returns, never inside it, so they reach these images and
nothing else. The cellmask is left alone: it is a label array, not a picture.

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
    EXVIVO_UM_PER_PX,
)
from stitch_subslices import stitch_fov_channels
from utilities.mat_io import load_filt_neurons, save_cellmask_h5
from utilities.image_io import imwrite_tiff, get_file_size_mb
from utilities.regression import calculate_fov_offset


# Every FOV in the slice gets stitched, so the only FOV this cannot place is one
# with no rows at all -- and such a FOV never enters the list, which is built from
# rows. Passed to stitch_fov_channels, whose own default is 3 (step 2's behaviour).
MIN_ROWS = 1

# Below this the regression cannot leave its slope free, so calculate_fov_offset
# pins it at 1. Mirrored here only so --dry-run can count and report it.
SLOPE_FIT_MIN_ROWS = 3


# ------------------------------------------------------------------
# Scale bars
#
# Burned into the pixels of the three channel TIFs. These images exist to be
# eyeballed, and a burned bar stays at the right physical length through any
# later crop or downsample, where an annotation drawn on a figure would not.
#
# All of it lives in this file rather than utilities/ so it cannot reach
# stitch_subslices.stitch_fov_channels, and from there step 3's downsample,
# step 4's overlay and the ALIGN tif the graph is fitted on.
#
# Not drawn on the cellmask: that canvas is uint32 where every nonzero value is
# a cell id, so a bar would be a fake cell whose value collides with a real one.
# ------------------------------------------------------------------

# Bar lengths in um, stacked in the bottom-left with the longest at the bottom.
# At EXVIVO_UM_PER_PX (0.32) these are 62, 156 and 312 px, against a canvas over
# 10,000 px wide -- all three are read zoomed in on cells, not with the whole
# section on screen. Edit freely.
SCALE_BAR_LENGTHS_UM = (20.0, 50.0, 100.0)

# Inset from the left and bottom edges, and the vertical gap between one bar's
# block and the next. Physical, so they hold at any resolution.
SCALE_BAR_MARGIN_UM = 60.0
SCALE_BAR_SPACING_UM = 40.0

# One thickness for every bar, physical like the rest of these. 3 um is 9 px at
# EXVIVO_UM_PER_PX, what the 100 um bar drew when thickness was a fraction of
# each bar's own length -- that scheme is gone, so a 20 um bar and a 100 um bar
# now differ in length only.
SCALE_BAR_THICKNESS_UM = 3.0

# Label height as a multiple of the bar thickness, so every label is the same
# size too. 6.0 puts it at 55 px against a 9 px bar.
SCALE_BAR_LABEL_SCALE = 6.0


def _render_label(text, height_px):
    """Boolean mask of `text` drawn in white, roughly `height_px` tall.

    PIL's default bitmap font is ~11 px tall, a smudge on a canvas thousands of
    px wide, and a TrueType font would be a font path to configure per machine.
    Rendering at the default size and repeating each pixel an integer number of
    times is blocky and needs nothing beyond Pillow, which is already required.
    """
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    probe = ImageDraw.Draw(Image.new('L', (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    w, h = max(right - left, 1), max(bottom - top, 1)

    small = Image.new('L', (w, h), 0)
    ImageDraw.Draw(small).text((-left, -top), text, fill=255, font=font)
    mask = np.array(small) > 0

    factor = max(1, int(round(height_px / h)))
    return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)


def draw_scale_bars(image, um_per_px, lengths_um=SCALE_BAR_LENGTHS_UM):
    """Burn labelled white scale bars into the bottom-left of `image`, in place.

    Returns (overwritten, drawn): how many of the pixels the bars and labels
    cover were nonzero beforehand, out of how many were written. That is the
    occlusion report -- a bar destroys whatever was under it, and the bottom-left
    corner of a bounding box over FOV placements is *usually* empty but is not
    guaranteed to be. 0 out of N means the bars landed on blank canvas; a large
    fraction means one is sitting on tissue and the corner should change.

    White is the canvas's own maximum, not the dtype's. Once a 65535 pixel
    exists in a raw uint16 fluorescence canvas that really peaks at a few
    thousand, any autoscaling viewer stretches its display range to 65535 and
    darkens the tissue the image exists to show.
    """
    height, width = image.shape[:2]
    white = int(image.max()) or int(np.iinfo(image.dtype).max)

    margin = int(round(SCALE_BAR_MARGIN_UM / um_per_px))
    spacing = int(round(SCALE_BAR_SPACING_UM / um_per_px))
    thickness = max(1, int(round(SCALE_BAR_THICKNESS_UM / um_per_px)))
    label_height = SCALE_BAR_LABEL_SCALE * thickness

    overwritten = 0
    drawn = 0
    y = height - margin  # bottom edge of the next bar, exclusive

    for length_um in sorted(lengths_um, reverse=True):
        length_px = int(round(length_um / um_per_px))
        label = _render_label(f"{length_um:g} um", label_height)
        label_h, label_w = label.shape
        block_h = thickness + spacing // 2 + label_h

        if margin + max(length_px, label_w) > width or y - block_h < 0:
            print(f"    Scale bar {length_um:g} um does not fit this canvas, skipped")
            continue

        bar = image[y - thickness:y, margin:margin + length_px]
        overwritten += int(np.count_nonzero(bar))
        drawn += bar.size
        bar[:] = white

        label_bottom = y - thickness - spacing // 2
        text = image[label_bottom - label_h:label_bottom, margin:margin + label_w]
        overwritten += int(np.count_nonzero(text[label]))
        drawn += int(label.sum())
        text[label] = white

        y = label_bottom - label_h - spacing

    return overwritten, drawn


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

    Runs the same placement on the same rows, so the canvas and the
    regressed/pinned split match the real run.

    Returns (height, width, n_placed, n_pinned, slopes), where slopes are the
    (x, y) pairs the regression fitted and then discarded on the FOVs that had
    enough rows for it -- the check on whether pinning costs anything.
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
    slopes = []
    n_pinned = 0
    for fov_name in fov_list:
        if row_counts[fov_name] < MIN_ROWS:
            continue
        i_cell = np.asarray(groups[fov_name])
        fit_slope = row_counts[fov_name] >= SLOPE_FIT_MIN_ROWS
        try:
            x_off, y_off, slope = calculate_fov_offset(
                pos[i_cell], pos40x[i_cell], scale_factor=2.0,
                fit_slope=fit_slope, return_slope=True,
            )
        except Exception:
            continue
        offsets.append((x_off, y_off))
        if fit_slope:
            slopes.append(slope)
        else:
            n_pinned += 1

    if not offsets:
        return 0, 0, 0, 0, slopes

    offsets = np.array(offsets)
    width = int(offsets[:, 0].max()) + FOV_SIZE - int(offsets[:, 0].min())
    height = int(offsets[:, 1].max()) + FOV_SIZE - int(offsets[:, 1].min())
    return height, width, len(offsets), n_pinned, slopes


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
        print(f"{'slice':>6}  {'FOVs':>5}  {'pinned':>6}  {'placed':>6}  "
              f"{'canvas (h x w)':>19}  {'RAM MB':>8}")
        all_slopes = []
        for slice_id, fov_list, counts in entries:
            pinned = [f for f in fov_list if counts[f] < SLOPE_FIT_MIN_ROWS]
            height, width, n_placed, n_pinned, slopes = plan_canvas(
                fov_list, slice_id, filt_neurons, counts)
            all_slopes.extend(slopes)
            # 3 uint16 channels + 1 uint32 cellmask, allocated together.
            ram_mb = height * width * (2 * 3 + 4) / 1e6
            print(f"{slice_id:>6}  {len(fov_list):>5}  {n_pinned:>6}  {n_placed:>6}  "
                  f"{height:>8} x {width:<8}  {ram_mb:>8.0f}")
            if pinned:
                detail = ', '.join(f"{f} ({counts[f]} row{'s' if counts[f] != 1 else ''})"
                                   for f in pinned)
                print(f"          slope pinned: {detail}")
            missing = [f for f in fov_list if counts[f] < MIN_ROWS]
            if missing:
                print(f"          NOT PLACED: {', '.join(missing)}")

        if all_slopes:
            sx = np.array([s_[0] for s_ in all_slopes])
            sy = np.array([s_[1] for s_ in all_slopes])
            print()
            print("Slopes the regression fitted and discarded, over every FOV with "
                  f">= {SLOPE_FIT_MIN_ROWS} rows (n={len(all_slopes)}).")
            print("Pinning replaces these with exactly 1.0:")
            print(f"  x: min {sx.min():.6f}  median {np.median(sx):.6f}  max {sx.max():.6f}")
            print(f"  y: min {sy.min():.6f}  median {np.median(sy):.6f}  max {sy.max():.6f}")
        return

    output_dir = Path(HYB_SLICE_STITCHED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    hyb_root = Path(HYB_ROOT)
    channels_root = Path(HYB_CHANNELS_DIR)

    for i, (slice_id, fov_list, counts) in enumerate(entries):
        pinned = [f for f in fov_list if counts[f] < SLOPE_FIT_MIN_ROWS]
        print("=" * 40)
        print(f"[{i+1}/{len(entries)}] Stitching slice {slice_id} (whole slice)")
        print("=" * 40)
        print(f"  FOVs in slice: {len(fov_list)}")
        print(f"  FOVs placed with the slope pinned at 1 (< {SLOPE_FIT_MIN_ROWS} rows): "
              f"{len(pinned)}\n")

        stitch_start = time.time()
        gcamp, dapi, mscarlet, cellmask, min_x, min_y, fov_offsets = stitch_fov_channels(
            fov_list, slice_id, filt_neurons, hyb_root, channels_root, min_rows=MIN_ROWS
        )
        print(f"  Stitching completed in {time.time() - stitch_start:.1f} seconds\n")

        print("  Saving stitched channels...")
        prefix = f"slice{slice_id}"

        for channel, image in (("GCAMP", gcamp), ("DAPI", dapi), ("MSCARLET", mscarlet)):
            occluded, painted = draw_scale_bars(image, EXVIVO_UM_PER_PX)
            path = output_dir / f"{prefix}_{channel}.tif"
            imwrite_tiff(path, image)
            print(f"    {channel}: {get_file_size_mb(path):.1f} MB"
                  f"   scale bars covered {occluded}/{painted} nonzero px")

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
