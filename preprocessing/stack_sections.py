#!/usr/bin/env python3
"""
Stack BARseq sections on their clicked centres into one ImageJ composite TIFF.

For viewing: where the red and green signal sits through the series. Reads
<ANALYSIS_ROOT>/section_centers.json (pick_section_centers.py) and shifts every
section by whole pixels so its centre lands on one canvas pixel. Translation
only, no resampling: pixel values in the stack are the values on disk. Rotation
and flips between sections are NOT corrected.

Each --channel becomes one channel of the composite, all sharing one shift per
section (a section's raw channels and ALIGN renders sit on one grid):

    DAPI | MSCARLET | GCAMP     step 3's downsampled raw channel
                                (HYB_subslice_stitched_tif_downsampled_micronwise/)
    <render folder name>        an ALIGN folder, resolved like SUBSLICE_DIR
                                (mscarlet_qc20_5_ge3_sat15, gcamp_qc20_5_ge3_sat10,
                                or a pre-2026-10-08 qc20_5_ge5)

Colours: mScarlet red, GCaMP green, DAPI grey, anything else blue.

Output, under <OUTPUT_ROOT>/section_stacks/:

    sections_{channel}__{channel}.tif            Z C Y X, Z = sections in ascending
                                                 slice number; calibrated XY from
                                                 SCOPE, Z = --z-um (20 µm sections)
    sections_{channel}__{channel}_offsets.json   per section: z index, depth, (dy, dx)

A section pixel (y, x) is at (y + dy, x + dx) in the stack; the offsets file is
what a point cloud built from export_subslice_cells.py's y_node/x_node reads.
Written through a memory map, so a stack larger than RAM is fine; --dry-run
prints its size first.

Usage:
    python preprocessing/stack_sections.py -c mscarlet_qc20_5_ge3_sat15 -c gcamp_qc20_5_ge3_sat10 --dry-run
    python preprocessing/stack_sections.py -c mscarlet_qc20_5_ge3_sat15 -c gcamp_qc20_5_ge3_sat10 -c DAPI
    python preprocessing/stack_sections.py -c MSCARLET -c GCAMP --slices 20 21 22
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import section_centers
from analysis_paths import ALIGN_TIF_GLOB, align_tif_slice, align_tifs, resolve_subslice_dir
from scope_profiles import SECTION_THICKNESS_UM

RAW_CHANNELS = ("DAPI", "MSCARLET", "GCAMP")
STACK_DIRNAME = "section_stacks"
RAW_RANGE_PERCENTILE = 99.9     # display ceiling of a raw channel, max over sections


def render_folders(align_roots):
    """``root/folder`` for every folder under `align_roots` holding ALIGN tifs."""
    found = []
    for root in map(Path, align_roots):
        if root.is_dir():
            found += [f"{root.name}/{d.name}" for d in sorted(root.iterdir())
                      if d.is_dir() and any(d.glob(ALIGN_TIF_GLOB))]
    return found


def channel_files(name, hyb_dir, sections, align_roots):
    """``{slice_no: path}`` for one --channel, over `sections`."""
    if name.upper() in RAW_CHANNELS:
        files = {n: Path(hyb_dir) / f"slice{n}_subslice_{name.upper()}.tif" for n in sections}
        files = {n: p for n, p in files.items() if p.exists()}
    else:
        try:
            folder = resolve_subslice_dir(name)
        except ValueError:
            folder = None
        if folder is None or not any(folder.glob(ALIGN_TIF_GLOB)):
            listing = "\n".join(f"    {f}" for f in render_folders(align_roots)) or "    (none)"
            raise SystemExit(
                f"--channel {name}: not one of {'/'.join(RAW_CHANNELS)}, and no folder "
                f"of ALIGN tifs by that name.\n\nALIGN render folders (pass the part "
                f"after the /):\n{listing}")
        files = {align_tif_slice(p): p for p in align_tifs(folder)}
    missing = [n for n in sections if n not in files]
    if missing:
        raise SystemExit(f"--channel {name}: no image for slice(s) {missing}")
    return {n: files[n] for n in sections}


def channel_colour(name):
    n = name.lower()
    if "mscarlet" in n:
        return (1, 0, 0)
    if "gcamp" in n:
        return (0, 1, 0)
    if "dapi" in n:
        return (1, 1, 1)
    return (0, 0, 1)


def lut(colour):
    ramp = np.arange(256, dtype=np.uint8)
    return np.stack([ramp * c for c in colour]).astype(np.uint8)


def tiff_shape_dtype(path):
    """(H, W), dtype from the header, without reading pixels."""
    with tifffile.TiffFile(path) as t:
        s = t.series[0]
        shape = tuple(d for d in s.shape if d != 1)
        return shape, s.dtype


def read_plane(path):
    img = tifffile.imread(path)
    img = np.squeeze(img)
    if img.ndim != 2:
        raise SystemExit(f"{path.name}: expected one 2D plane, got shape {img.shape}")
    return img


def main():
    ap = argparse.ArgumentParser(
        description="Stack BARseq sections on their clicked centres (ImageJ composite)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("-c", "--channel", action="append", required=True,
                    help=f"{'/'.join(RAW_CHANNELS)} or an ALIGN render folder name; repeat")
    ap.add_argument("--slices", type=int, nargs="+", default=None, metavar="N",
                    help="only these sections (default: every centred section)")
    ap.add_argument("--z-um", type=float, default=SECTION_THICKNESS_UM,
                    help=f"spacing between consecutive sections (default {SECTION_THICKNESS_UM})")
    ap.add_argument("--centers", default=None,
                    help="record path (default: <ANALYSIS_ROOT>/section_centers.json)")
    ap.add_argument("--out", default=None, help="output .tif (default: under section_stacks/)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    ap.add_argument("--dry-run", action="store_true", help="report the layout, write nothing")
    args = ap.parse_args()

    from preprocessing_config import (
        HYB_DOWNSAMPLED_DIR, OUTPUT_ROOT, TARGET_XY_UM_PER_PX,
        SUBSLICE_ALIGN_MSCARLET_DIR, SUBSLICE_ALIGN_GCAMP_DIR, SUBSLICE_ALIGN_DIR,
    )
    align_roots = (SUBSLICE_ALIGN_MSCARLET_DIR, SUBSLICE_ALIGN_GCAMP_DIR, SUBSLICE_ALIGN_DIR)

    if len(set(c.lower() for c in args.channel)) != len(args.channel):
        ap.error("a --channel is listed twice")

    cpath = section_centers.centers_path(args.centers)
    record = section_centers.load(cpath)
    centres = section_centers.centers(record)
    grids = section_centers.shapes(record)
    if not centres:
        raise SystemExit(f"No centres in {cpath}. Run preprocessing/pick_section_centers.py first.")
    if args.slices:
        missing = sorted(set(args.slices) - set(centres))
        if missing:
            raise SystemExit(f"No centre recorded for slice(s) {missing} in {cpath}")
        centres = {n: centres[n] for n in args.slices}
    sections = sorted(centres)

    files = {c: channel_files(c, HYB_DOWNSAMPLED_DIR, sections, align_roots)
             for c in args.channel}

    # One grid per section across every channel, and the grid the click was made on.
    dtypes = set()
    bad = []
    for c, by_slice in files.items():
        for n, p in by_slice.items():
            shape, dtype = tiff_shape_dtype(p)
            dtypes.add(np.dtype(dtype))
            if shape != tuple(grids[n]):
                bad.append(f"  slice {n}  {c}: {shape[0]}x{shape[1]}, "
                           f"centre clicked on {grids[n][0]}x{grids[n][1]}  ({p.name})")
    if bad:
        raise SystemExit(
            "These images are not on the grid their centre was clicked on:\n"
            + "\n".join(bad) + "\n"
            "Step 3 was re-run at another pitch, or the folder is from another subject. "
            "Re-click those sections (pick_section_centers.py --slices N --redo).")
    if any(d.kind not in "ui" or d.itemsize > 2 for d in dtypes):
        raise SystemExit(f"Only 8/16-bit integer images stack; found {sorted(map(str, dtypes))}")
    dtype = np.dtype(np.uint8) if dtypes == {np.dtype(np.uint8)} else np.dtype(np.uint16)

    (H, W), shifts = section_centers.layout(centres, {n: grids[n] for n in sections})
    Z, C = len(sections), len(args.channel)
    nbytes = Z * C * H * W * dtype.itemsize

    gaps = [(a, b) for a, b in zip(sections, sections[1:]) if b - a != 1]
    stem = "sections_" + "__".join(args.channel)
    out = Path(args.out) if args.out else Path(OUTPUT_ROOT) / STACK_DIRNAME / f"{stem}.tif"
    offsets_path = out.with_name(out.stem + "_offsets.json")

    print("=" * 60)
    print("STACK SECTIONS")
    print("=" * 60)
    print(f"centres:  {cpath}  ({Z} section(s): {sections[0]}..{sections[-1]})")
    for c in args.channel:
        print(f"channel:  {c}")
    print(f"canvas:   {H} x {W} px  ({H * TARGET_XY_UM_PER_PX / 1000:.2f} x "
          f"{W * TARGET_XY_UM_PER_PX / 1000:.2f} mm at {TARGET_XY_UM_PER_PX:.4f} µm/px)")
    print(f"stack:    Z {Z} x C {C} x {H} x {W}  {dtype}  {nbytes / 1e9:.2f} GB")
    print(f"z:        {args.z_um} µm per section, ascending slice number")
    if gaps:
        print(f"WARNING:  section numbers not consecutive at {gaps}; "
              f"planes are still spaced {args.z_um} µm apart")
    print(f"output:   {out}")
    if args.dry_run:
        print("\ndry run — nothing written")
        return
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite, or --out elsewhere")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Display range per channel: ALIGN renders are 0-255 by construction; a raw
    # channel's ceiling is the max over sections of its RAW_RANGE_PERCENTILE.
    # Read before the stack is created because ImageJ takes ranges at write time.
    ranges = []
    for c in args.channel:
        if c.upper() in RAW_CHANNELS:
            ranges.append(max(float(np.percentile(read_plane(p)[::4, ::4], RAW_RANGE_PERCENTILE))
                              for p in files[c].values()))
        else:
            ranges.append(255.0)

    stack = tifffile.memmap(
        out, shape=(Z, C, H, W), dtype=dtype, imagej=True,
        resolution=(1 / TARGET_XY_UM_PER_PX, 1 / TARGET_XY_UM_PER_PX),
        metadata={
            "axes": "ZCYX", "spacing": args.z_um, "unit": "um", "mode": "composite",
            "Labels": [f"slice{n} {c}" for n in sections for c in args.channel],
            "LUTs": [lut(channel_colour(c)) for c in args.channel],
            "Ranges": tuple(v for r in ranges for v in (0.0, max(r, 1.0))),
        })
    for z, n in enumerate(sections):
        dy, dx = shifts[n]
        h, w = grids[n]
        for ci, c in enumerate(args.channel):
            img = read_plane(files[c][n])
            stack[z, ci, dy:dy + h, dx:dx + w] = img
        print(f"  slice {n:>3}  z {z:>3}  shift (dy {dy:>5}, dx {dx:>5})")
    stack.flush()
    del stack

    offsets = {
        "centers_file": str(cpath),
        "channels": list(args.channel),
        "canvas_shape": [H, W],
        "xy_um_per_px": TARGET_XY_UM_PER_PX,
        "z_um_per_section": args.z_um,
        "stack_from_section": "stack (y, x) = section (y + dy, x + dx); "
                              "z_um = z_index * z_um_per_section",
        "sections": [
            {"slice": n, "z_index": z, "z_um": z * args.z_um,
             "dy": shifts[n][0], "dx": shifts[n][1],
             "center": [record["sections"][str(n)]["y"], record["sections"][str(n)]["x"]],
             "shape": list(grids[n])}
            for z, n in enumerate(sections)],
    }
    offsets_path.write_text(json.dumps(offsets, indent=2) + "\n")
    print(f"\n{out}\n{offsets_path}")


if __name__ == "__main__":
    main()
