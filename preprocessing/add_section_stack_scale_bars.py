#!/usr/bin/env python3
"""
TEMPORARY. Copy section stacks with scale bars burned into every plane.

Visualization only, like add_align_scale_bars.py. Reads a composite written by
stack_sections.py, burns the bars `stitch_slices.py` burns into the whole-slice
TIFs into every section plane of every channel, and writes the copy to a folder
of its own. The source stack and its _offsets.json are read and never written.

Input, under <OUTPUT_ROOT>/section_stacks/ (a path, or a bare filename there):

    sections_{source}_{channel}__{channel}.tif     Z C Y X ImageJ composite

Output:

    <OUTPUT_ROOT>/section_stacks_scalebar/sections_{source}_..._.tif

with the source's ImageJ metadata carried over: per-channel LUTs and display
ranges, slice labels, composite mode, XY calibration and Z spacing.

Pitch: stack_sections.py writes the stack calibrated at TARGET_XY_UM_PER_PX,
and that tag is the pitch used here. It raises if the tag is missing or
disagrees with this config's TARGET_XY_UM_PER_PX -- local_config set to a
different brain than the stack came from.

Every section shares one canvas, so the bars sit in the same place on every
plane: the canvas's bottom-left. A bar in one channel only shows when that
channel is on, so every channel gets one, at that channel's stored display
maximum: full brightness in the channel's colour without widening any range.

`draw_scale_bars` and every SCALE_BAR_* constant are imported from
`stitch_slices.py` rather than copied, so a retune reaches every bar.

Usage:
    python add_section_stack_scale_bars.py --dry-run                 # every stack in section_stacks/
    python add_section_stack_scale_bars.py sections_slice_MSCARLET__GCAMP.tif
    python add_section_stack_scale_bars.py "D:\\...\\section_stacks\\sections_slice_MSCARLET__GCAMP.tif"
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import tifffile

from preprocessing_config import (
    OUTPUT_ROOT,
    TARGET_XY_UM_PER_PX,
)
from stitch_slices import (
    draw_scale_bars,
    SCALE_BAR_LENGTHS_UM,
    SCALE_BAR_MARGIN_UM,
    SCALE_BAR_THICKNESS_UM,
    SCALE_BAR_LABEL_SCALE,
)
from utilities.image_io import get_tiff_resolution, get_file_size_mb


# Not in preprocessing_config: temporary, and no pipeline step reads it.
INPUT_DIR = Path(OUTPUT_ROOT) / "section_stacks"
OUTPUT_DIR = Path(OUTPUT_ROOT) / "section_stacks_scalebar"

# Relative tolerance between the stack's calibration tag and this config's
# pitch; the tag is a rational, so not bit-exact.
PITCH_RTOL = 1e-4

# ImageJ metadata carried from source to copy.
CARRIED_METADATA = ("spacing", "unit", "mode", "Labels", "LUTs", "Ranges")


def resolve_stacks(names):
    """Stack paths from `names` (paths or filenames in INPUT_DIR); none = all."""
    if not names:
        # *_scalebar stacks (crop_section_stack_to_subslices.py) already carry bars.
        paths = sorted(p for p in INPUT_DIR.glob("sections_*.tif")
                       if not p.stem.endswith("_scalebar"))
        if not paths:
            raise FileNotFoundError(f"No sections_*.tif in {INPUT_DIR}")
        return paths
    paths = []
    for name in names:
        path = Path(name)
        if not path.is_file():
            path = INPUT_DIR / name
        if not path.is_file():
            raise FileNotFoundError(
                f"No section stack {name!r}: not a file, and not in {INPUT_DIR}"
            )
        paths.append(path)
    return paths


def stack_pitch(path):
    """um/px from the stack's own calibration; raises if absent or off-config."""
    pitch = get_tiff_resolution(path)['xy_um_per_px']
    if pitch is None:
        raise ValueError(f"{path.name}: no XY calibration in the file")
    if abs(pitch - TARGET_XY_UM_PER_PX) > PITCH_RTOL * TARGET_XY_UM_PER_PX:
        raise ValueError(
            f"{path.name} is calibrated at {pitch} um/px, but this config's "
            f"SCOPE gives {TARGET_XY_UM_PER_PX} um/px.\n"
            f"  local_config.py is set to a different brain or scope than this stack."
        )
    return pitch


def bar_mask(height, width, um_per_px):
    """Boolean (H, W) mask of the bars and labels for this canvas."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    draw_scale_bars(canvas, um_per_px)
    return canvas > 0


def report_geometry(um_per_px):
    print(f"Scale bars at {um_per_px:.4f} um/px (the stack's calibration):")
    thickness = max(1, int(round(SCALE_BAR_THICKNESS_UM / um_per_px)))
    for length_um in sorted(SCALE_BAR_LENGTHS_UM):
        print(f"  {length_um:g} um -> {int(round(length_um / um_per_px))} px")
    print(f"  thickness {SCALE_BAR_THICKNESS_UM:g} um -> {thickness} px")
    print(f"  label height -> {int(round(SCALE_BAR_LABEL_SCALE * thickness))} px")
    print(f"  corner inset {SCALE_BAR_MARGIN_UM:g} um -> "
          f"{int(round(SCALE_BAR_MARGIN_UM / um_per_px))} px")


def add_bars_to_stack(path, dry_run=False):
    out_path = OUTPUT_DIR / path.name
    if out_path.resolve() == path.resolve():
        raise ValueError(f"Output is the input: {out_path}")

    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes, shape, dtype = series.axes, series.shape, series.dtype
        ij = dict(tif.imagej_metadata or {})
    if axes != "ZCYX":
        raise ValueError(f"{path.name}: axes {axes!r}, expected 'ZCYX' "
                         f"(a stack_sections.py composite)")
    Z, C, H, W = shape

    pitch = stack_pitch(path)
    print(f"\n{path.name}")
    print(f"  Z {Z} x C {C} x {H} x {W} {dtype}, {get_file_size_mb(path):.0f} MB")
    report_geometry(pitch)

    # Display max per channel from the stored Ranges (min0, max0, min1, ...).
    ranges = ij.get("Ranges")
    if ranges is not None and len(ranges) == 2 * C:
        levels = [float(ranges[2 * c + 1]) for c in range(C)]
    else:
        levels = [float(np.iinfo(dtype).max)] * C
        print(f"  no per-channel Ranges stored; bars at the dtype max {levels[0]:g}")
    levels = [min(v, float(np.iinfo(dtype).max)) for v in levels]
    print("  bar value per channel: " + ", ".join(f"C{c} {v:g}" for c, v in enumerate(levels)))

    mask = bar_mask(H, W, pitch)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        print("  no bar fits this canvas; skipped")
        return
    y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    sub = mask[y0:y1, x0:x1]

    src = tifffile.memmap(str(path), mode="r")
    covered = np.zeros(Z, dtype=np.int64)
    for z in range(Z):
        for c in range(C):
            covered[z] += int(np.count_nonzero(src[z, c, y0:y1, x0:x1][sub]))
    painted = int(sub.sum()) * C
    hit = np.nonzero(covered)[0]
    print(f"  bars cover nonzero px on {hit.size} of {Z} planes"
          + (": " + ", ".join(f"z{z} {covered[z]}/{painted}" for z in hit) if hit.size else ""))

    if dry_run:
        del src
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {"axes": "ZCYX"}
    metadata.update({k: ij[k] for k in CARRIED_METADATA if k in ij})
    dst = tifffile.memmap(
        str(out_path), shape=(Z, C, H, W), dtype=dtype, imagej=True,
        resolution=(1 / pitch, 1 / pitch), metadata=metadata)
    for z in range(Z):
        dst[z] = src[z]
        for c in range(C):
            dst[z, c, y0:y1, x0:x1][sub] = levels[c]
    dst.flush()
    del dst, src
    print(f"  wrote {out_path} ({get_file_size_mb(out_path):.0f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Copy section stacks with scale bars burned into every plane",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('stacks', nargs='*',
                        help='Stack paths or filenames in section_stacks/; none = all')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report geometry and occlusion, write nothing')
    args = parser.parse_args()

    paths = resolve_stacks(args.stacks)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Stacks: {len(paths)}")
    for path in paths:
        add_bars_to_stack(path, dry_run=args.dry_run)
    if args.dry_run:
        print("\nDry run: nothing written.")


if __name__ == '__main__':
    main()
