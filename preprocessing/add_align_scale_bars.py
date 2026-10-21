#!/usr/bin/env python3
"""
Copy one folder of ALIGN tifs with scale bars burned in, into its _scalebar/.

Visualization only. Reads the ALIGN tifs that generate_alignment_tif.py wrote
into one render folder, burns the bars `stitch_slices.py` burns into the
whole-slice TIFs, and writes the copies to a `_scalebar/` subfolder of that
render folder. The ALIGN tifs themselves are read and never written.
generate_alignment_tif.py calls this after every render (unless --no-scale-bars
or --out); run it by hand for a folder rendered before that.

Input, one render folder (a path, or its leaf name looked up under the subslice
and whole-slice ALIGN roots):

    <OUTPUT_ROOT>/subslice_align_{mscarlet,gcamp}/{leaf}/slice{N}_subslice_ALIGN_{leaf}.tif
    <OUTPUT_ROOT>/slice_align_{mscarlet,gcamp}/{leaf}/slice{N}_ALIGN_{leaf}.tif

Output, beside them:

    {leaf}/_scalebar/slice{N}_subslice_SCALEBAR_{leaf}.tif     (subslice)
    {leaf}/_scalebar/slice{N}_SCALEBAR_{leaf}.tif              (whole slice)

plus a copy of the folder's align_render.json. Each copy is an ImageJ TIFF
calibrated at its slice's recorded pitch, so ImageJ opens it in um.

The copies must never become graph nodes: a bar would be fitted as cells. The
builder globs `slice*_subslice_ALIGN*.tif` (analysis_paths.ALIGN_TIF_GLOB) in the
SUBSLICE_DIR folder itself, never its subfolders, and the copies match neither
that pattern nor SLICE_ALIGN_TIF_GLOB: `SCALEBAR` replaces `ALIGN`. Pointing
SUBSLICE_DIR at a _scalebar/ folder makes the builder raise on a folder with no
ALIGN tifs.

Pitch: an ALIGN tif is drawn on the downsampled canvas of its section, and the
script that wrote that canvas stored the pitch in its metadata:

    subslice   HYB_DOWNSAMPLED_DIR/slice{N}_subslice_CELLMASK.h5   TARGET_XY_UM_PER_PX  (step 3)
    slice      HYB_SLICE_DOWNSAMPLED_DIR/slice{N}_CELLMASK.h5      um_per_px  (stitch_slices.py --downsample)

The bars use that recorded value, per slice: 0.32 um/px x DOWNSAMPLE_XY, i.e.
SCOPE's pitch. It raises if the record is missing, if it disagrees with this
config's TARGET_XY_UM_PER_PX, if the ALIGN tif's shape is not that canvas's, or
if the render folder is not under this config's OUTPUT_ROOT -- each means
local_config is set to a different brain than the folder came from. At BY95's
1.1 um/px the bars are 18/91/227 px; a bar that does not fit a canvas is skipped
with a message (none on BY95 subslices, the narrowest of which is 931 px).

Each axis was resized to round(n / DOWNSAMPLE_XY) px, so the true pitch differs
from the nominal by under half a pixel over the canvas: <0.06% on a 931 px one.

`draw_scale_bars` and every SCALE_BAR_* constant are imported from
`stitch_slices.py` rather than copied, so a retune reaches every bar.

Usage:
    python add_align_scale_bars.py mscarlet_qc0_3_ge0_sat5 --dry-run
    python add_align_scale_bars.py mscarlet_qc0_3_ge0_sat5 --slice 22
    python add_align_scale_bars.py "D:\\...\\slice_align_gcamp\\gcamp_qc0_3_ge0_sat5"
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

import h5py
import tifffile

from preprocessing_config import (
    OUTPUT_ROOT,
    HYB_DOWNSAMPLED_DIR,
    HYB_SLICE_DOWNSAMPLED_DIR,
    SUBSLICE_ALIGN_MSCARLET_DIR,
    SUBSLICE_ALIGN_GCAMP_DIR,
    SLICE_ALIGN_MSCARLET_DIR,
    SLICE_ALIGN_GCAMP_DIR,
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
from utilities.image_io import (
    imread_tiff,
    get_file_size_mb,
)


SCALEBAR_SUBDIR = "_scalebar"

ALIGN_ROOTS = (SUBSLICE_ALIGN_MSCARLET_DIR, SUBSLICE_ALIGN_GCAMP_DIR,
               SLICE_ALIGN_MSCARLET_DIR, SLICE_ALIGN_GCAMP_DIR)

RENDER_SIDECAR = "align_render.json"

# Where each source's canvas record lives, its filename, and its pitch key.
CANVAS_RECORDS = {
    "subslice": (HYB_DOWNSAMPLED_DIR, "slice{n}_subslice_CELLMASK.h5", "TARGET_XY_UM_PER_PX"),
    "slice": (HYB_SLICE_DOWNSAMPLED_DIR, "slice{n}_CELLMASK.h5", "um_per_px"),
}

# Relative tolerance between the recorded pitch and this config's.
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


def render_source(input_dir):
    """(source, ALIGN tif paths): whichever of the two filename forms the folder holds."""
    found = {s: align_tifs(input_dir, s) for s in ("subslice", "slice")}
    held = [s for s, paths in found.items() if paths]
    if not held:
        raise FileNotFoundError(f"No ALIGN tifs in {input_dir}")
    if len(held) > 1:
        raise ValueError(f"{input_dir} holds both subslice and whole-slice ALIGN tifs")
    return held[0], found[held[0]]


def recorded_pitch(slice_id, shape, source):
    """um/px recorded for this slice's canvas; raises on any mismatch."""
    folder, pattern, key = CANVAS_RECORDS[source]
    path = Path(folder) / pattern.format(n=slice_id)
    if not path.exists():
        raise FileNotFoundError(f"Slice {slice_id}: canvas record not found: {path}")
    with h5py.File(path, "r") as f:
        canvas = tuple(f["cellmask"].shape)
        meta = f.get("metadata")
        pitch = float(meta[key][()]) if meta is not None and key in meta else None
    if canvas != tuple(shape[:2]):
        raise ValueError(
            f"Slice {slice_id}: ALIGN tif is {shape[1]} x {shape[0]} but its canvas "
            f"is {canvas[1]} x {canvas[0]}:\n  {path}\n"
            f"  The render predates the current canvas; re-render it."
        )
    if pitch is None:
        raise ValueError(f"Slice {slice_id}: no {key} in {path}")
    if abs(pitch - TARGET_XY_UM_PER_PX) > PITCH_RTOL * TARGET_XY_UM_PER_PX:
        raise ValueError(
            f"Slice {slice_id}: the canvas was drawn at {pitch} um/px, but this "
            f"config's SCOPE gives {TARGET_XY_UM_PER_PX} um/px.\n"
            f"  local_config.py is set to a different brain or scope than this render."
        )
    return pitch


def report_geometry(um_per_px):
    """The bar geometry in pixels at this pitch, once, before any file."""
    print(f"Scale bars at {um_per_px:.4f} um/px:")
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
    output_dir = input_dir / SCALEBAR_SUBDIR

    source, paths = render_source(input_dir)
    if targets is not None:
        by_slice = {align_tif_slice(p, source): p for p in paths}
        missing = sorted(set(targets) - set(by_slice))
        if missing:
            raise ValueError(
                f"No ALIGN tif for slice(s): {missing}\n"
                f"  Looked in: {input_dir}\n"
                f"  Available: {sorted(by_slice)}"
            )
        paths = [by_slice[s] for s in sorted(targets)]

    print(f"Input:  {input_dir} ({source} ALIGN tifs)")
    print(f"Output: {output_dir}")
    print(f"Slices: {len(paths)}\n")
    report_geometry(TARGET_XY_UM_PER_PX)

    if not dry_run:
        output_dir.mkdir(exist_ok=True)
        sidecar = input_dir / RENDER_SIDECAR
        if sidecar.exists():
            shutil.copy2(sidecar, output_dir / RENDER_SIDECAR)
        else:
            print(f"note: {leaf} has no {RENDER_SIDECAR}; none copied\n")

    stem = "subslice_SCALEBAR" if source == "subslice" else "SCALEBAR"
    # The dry run draws on each image and discards it, so its "does not fit"
    # decisions are the real run's.
    written = 0
    for i, path in enumerate(paths):
        slice_id = align_tif_slice(path, source)
        image = imread_tiff(path)
        print(f"[{i+1}/{len(paths)}] Slice {slice_id}: "
              f"{image.shape[1]} x {image.shape[0]} {image.dtype}")
        um_per_px = recorded_pitch(slice_id, image.shape, source)
        occluded, painted = draw_scale_bars(image, um_per_px)
        print(f"    scale bars covered {occluded}/{painted} nonzero px")
        if dry_run:
            continue
        out_path = output_dir / f"slice{slice_id}_{stem}_{leaf}.tif"
        tifffile.imwrite(out_path, image, imagej=True,
                         resolution=(1 / um_per_px, 1 / um_per_px),
                         metadata={"unit": "um"})
        written += 1
        print(f"    {get_file_size_mb(out_path):.1f} MB")

    if dry_run:
        print("\nDry run: nothing written.")
    else:
        print(f"\nWrote {written} TIFs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy one render folder of ALIGN tifs with scale bars into its _scalebar/",
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
