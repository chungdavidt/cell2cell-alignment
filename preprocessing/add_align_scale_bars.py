#!/usr/bin/env python3
"""
TEMPORARY. Copy one folder of subslice ALIGN tifs with scale bars burned in.

Visualization only, like add_subslice_scale_bars.py. Reads the ALIGN tifs that
generate_alignment_tif.py wrote into one render folder, burns the bars
`stitch_slices.py` burns into the whole-slice TIFs, and writes the copies to a
folder of their own. The render folder is read and never written.

Input, one render folder (a path, or its leaf name looked up under both
subslice ALIGN roots):

    <OUTPUT_ROOT>/subslice_align_{mscarlet,gcamp}/{leaf}/slice{N}_subslice_ALIGN_{leaf}.tif

Output:

    <OUTPUT_ROOT>/subslice_align_scalebar/{leaf}/slice{N}_subslice_SCALEBAR_{leaf}.tif

plus a copy of the folder's align_render.json, so the copies carry the settings
they were rendered with.

The copies must never become graph nodes: a bar would be fitted as cells. The
builder ingests only `slice*_subslice_ALIGN*.tif` (analysis_paths.ALIGN_TIF_GLOB)
and resolves a relative SUBSLICE_DIR only under the roots in
analysis_paths._preprocessing_roots (step 4's overlays, subslice_align_{mscarlet,gcamp}/,
subslice_align/). The copies match neither: `SCALEBAR` replaces `ALIGN` in the
name and the root is subslice_align_scalebar/. Pointing SUBSLICE_DIR at
this folder by absolute path makes the builder raise on a folder with no ALIGN
tifs.

Pitch: an ALIGN tif is drawn on step 3's downsampled canvas for its slice,
HYB_DOWNSAMPLED_DIR/slice{N}_subslice_CELLMASK.h5, and step 3 stored the pitch it
used in that file's metadata (TARGET_XY_UM_PER_PX = SCOPE's um/px, reached from
0.32 um/px by DOWNSAMPLE_XY). The bars use that recorded value, per slice. It
raises if the record is missing, if it disagrees with this config's
TARGET_XY_UM_PER_PX, if the ALIGN tif's shape is not that canvas's, or if the
render folder is not under this config's OUTPUT_ROOT -- each means local_config
is set to a different brain than the folder came from. At BY95's 1.1 um/px the
250 um bar is 227 px, so on the narrowest subslices (one FOV, ~270 px) it does
not fit and is skipped with a message.

Each axis was resized to round(n / DOWNSAMPLE_XY) px, so the true pitch differs
from the nominal by under half a pixel over the canvas: <0.2% on a 270 px one.

`draw_scale_bars` and every SCALE_BAR_* constant are imported from
`stitch_slices.py` rather than copied, so a retune reaches every bar.

Usage:
    python add_align_scale_bars.py mscarlet_qc0_3_ge0_sat5 --dry-run
    python add_align_scale_bars.py mscarlet_qc0_3_ge0_sat5 --slice 22
    python add_align_scale_bars.py "D:\\...\\subslice_align_mscarlet\\mscarlet_qc0_3_ge0_sat5"
"""

import argparse
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from preprocessing_config import (
    OUTPUT_ROOT,
    HYB_DOWNSAMPLED_DIR,
    SUBSLICE_ALIGN_MSCARLET_DIR,
    SUBSLICE_ALIGN_GCAMP_DIR,
    TARGET_XY_UM_PER_PX,
)
from stitch_slices import (
    draw_scale_bars,
    SCALE_BAR_LENGTHS_UM,
    SCALE_BAR_MARGIN_UM,
    SCALE_BAR_THICKNESS_UM,
    SCALE_BAR_LABEL_SCALE,
)
from analysis_paths import align_tifs, align_tif_slice
from utilities.mat_io import load_cellmask_h5
from utilities.image_io import (
    imread_tiff,
    imwrite_tiff,
    get_file_size_mb,
)


# Not in preprocessing_config: temporary, and no pipeline step reads it.
OUTPUT_PARENT = Path(OUTPUT_ROOT) / "subslice_align_scalebar"

ALIGN_ROOTS = (SUBSLICE_ALIGN_MSCARLET_DIR, SUBSLICE_ALIGN_GCAMP_DIR)

RENDER_SIDECAR = "align_render.json"

# Relative tolerance between step 3's recorded pitch and this config's.
PITCH_RTOL = 1e-6


def resolve_render_dir(render):
    """The render folder named by `render`: a path, or a leaf under ALIGN_ROOTS."""
    path = Path(render)
    if path.is_dir():
        return path
    hits = [Path(root) / render for root in ALIGN_ROOTS
            if (Path(root) / render).is_dir()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(
            f"No render folder {render!r}.\n"
            f"  Not a directory, and not a leaf under:\n"
            + "".join(f"    {root}\n" for root in ALIGN_ROOTS)
        )
    raise ValueError(f"{render!r} is a leaf under more than one root; pass the full path:\n"
                     + "".join(f"  {h}\n" for h in hits))


def step3_pitch(slice_id, shape):
    """um/px step 3 recorded for this slice's canvas; raises on any mismatch."""
    path = Path(HYB_DOWNSAMPLED_DIR) / f"slice{slice_id}_subslice_CELLMASK.h5"
    cellmask, meta = load_cellmask_h5(path)
    if cellmask.shape != tuple(shape[:2]):
        raise ValueError(
            f"Slice {slice_id}: ALIGN tif is {shape[1]} x {shape[0]} but step 3's "
            f"canvas is {cellmask.shape[1]} x {cellmask.shape[0]}:\n  {path}\n"
            f"  The render predates the current step 3 output; re-render it."
        )
    pitch = meta.get('TARGET_XY_UM_PER_PX')
    if pitch is None:
        raise ValueError(f"Slice {slice_id}: no TARGET_XY_UM_PER_PX in {path}")
    if abs(pitch - TARGET_XY_UM_PER_PX) > PITCH_RTOL * TARGET_XY_UM_PER_PX:
        raise ValueError(
            f"Slice {slice_id}: step 3 drew this canvas at {pitch} um/px, but this "
            f"config's SCOPE gives {TARGET_XY_UM_PER_PX} um/px.\n"
            f"  local_config.py is set to a different brain or scope than this render."
        )
    return float(pitch)


def report_geometry(um_per_px):
    """The bar geometry in pixels at this pitch, once, before any file."""
    print(f"Scale bars at {um_per_px:.4f} um/px (step 3's record):")
    thickness = max(1, int(round(SCALE_BAR_THICKNESS_UM / um_per_px)))
    for length_um in sorted(SCALE_BAR_LENGTHS_UM):
        print(f"  {length_um:g} um -> {int(round(length_um / um_per_px))} px")
    print(f"  thickness {SCALE_BAR_THICKNESS_UM:g} um -> {thickness} px")
    print(f"  label height -> {int(round(SCALE_BAR_LABEL_SCALE * thickness))} px")
    print(f"  corner inset {SCALE_BAR_MARGIN_UM:g} um -> "
          f"{int(round(SCALE_BAR_MARGIN_UM / um_per_px))} px\n")


def add_align_scale_bars(render, targets=None, dry_run=False):
    input_dir = resolve_render_dir(render)
    leaf = input_dir.name
    if input_dir.resolve().parent.parent != Path(OUTPUT_ROOT).resolve():
        raise ValueError(
            f"Render folder is not under this config's OUTPUT_ROOT:\n"
            f"  render folder: {input_dir}\n"
            f"  OUTPUT_ROOT:   {OUTPUT_ROOT}\n"
            f"  Set local_config.py to the brain this render belongs to."
        )
    output_dir = OUTPUT_PARENT / leaf
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError(
            f"Output directory is the input directory: {output_dir}\n"
            f"  This script must never write into a render folder."
        )

    paths = align_tifs(input_dir)
    if not paths:
        raise FileNotFoundError(f"No slice*_subslice_ALIGN*.tif in {input_dir}")
    if targets is not None:
        by_slice = {align_tif_slice(p): p for p in paths}
        missing = sorted(set(targets) - set(by_slice))
        if missing:
            raise ValueError(
                f"No ALIGN tif for slice(s): {missing}\n"
                f"  Looked in: {input_dir}\n"
                f"  Available: {sorted(by_slice)}"
            )
        paths = [by_slice[s] for s in sorted(targets)]

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Slices: {len(paths)}\n")
    report_geometry(TARGET_XY_UM_PER_PX)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        sidecar = input_dir / RENDER_SIDECAR
        if sidecar.exists():
            shutil.copy2(sidecar, output_dir / RENDER_SIDECAR)
        else:
            print(f"note: {input_dir.name} has no {RENDER_SIDECAR}; none copied\n")

    # The dry run draws on each image and discards it, so its "does not fit"
    # decisions are the real run's.
    written = 0
    for i, path in enumerate(paths):
        slice_id = align_tif_slice(path)
        image = imread_tiff(path)
        print(f"[{i+1}/{len(paths)}] Slice {slice_id}: "
              f"{image.shape[1]} x {image.shape[0]} {image.dtype}")
        um_per_px = step3_pitch(slice_id, image.shape)
        occluded, painted = draw_scale_bars(image, um_per_px)
        print(f"    scale bars covered {occluded}/{painted} nonzero px")
        if dry_run:
            continue
        out_path = output_dir / f"slice{slice_id}_subslice_SCALEBAR_{leaf}.tif"
        imwrite_tiff(out_path, image)
        written += 1
        print(f"    {get_file_size_mb(out_path):.1f} MB")

    if dry_run:
        print("\nDry run: nothing written.")
    else:
        print(f"\nWrote {written} TIFs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy one render folder of subslice ALIGN tifs with scale bars burned in",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('render',
                        help='Render folder: a path, or its leaf name '
                             '(e.g. mscarlet_qc0_3_ge0_sat5)')
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

    add_align_scale_bars(args.render, targets=targets, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
