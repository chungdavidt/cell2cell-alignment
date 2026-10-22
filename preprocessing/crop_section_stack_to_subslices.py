#!/usr/bin/env python3
"""
Crop a whole-slice section stack to the box holding every section's subslice,
with scale bars burned in.

For viewing: the same planes, shifts, channels and display settings as the
stack_sections.py --source slice stack it reads, zoomed to the marker region
so the bars sit next to tissue instead of across a mostly empty canvas.

Where each subslice sits in the stack comes from the stitching records, not
from image content. Step 2 and stitch_slices.py stitch with one function in one
stage frame, and each records its canvas origin there (the smallest FOV offset
it placed, full-resolution px):

    subslice origin   HYB_DOWNSAMPLED_DIR/slice{N}_subslice_CELLMASK.h5   min_x_offset, min_y_offset
    slice origin      HYB_SLICE_DOWNSAMPLED_DIR/slice{N}_CELLMASK.h5     min_x, min_y
    section -> stack  <stack>_offsets.json                               dy, dx

    stack (y, x) = subslice (y, x) + (origin_sub - origin_slice) / DOWNSAMPLE_XY + (dy, dx)

That holds only if both runs placed their shared FOVs identically, so every
FOV in step 2's record (HYB_STITCHED_DIR/slice{N}_subslice_CELLMASK.h5) is
compared against the slice record and any difference raises. Each canvas was
resized on its own to round(n / DOWNSAMPLE_XY) px, so a footprint can be off by
about 1 px; the box takes floor/ceil to contain it.

The box is the union of every section's subslice footprint in stack
coordinates, plus --pad-um on each side, clipped to the canvas. Every plane is
cropped to it; sections with no subslice keep their pixels inside the box.

Bars as add_section_stack_scale_bars.py draws them: bottom-left of the box, on
every plane of every channel, at that channel's stored display maximum. In a
tight box they can land on tissue; the occlusion line per plane says where, and
--pad-um moves the edge out.

--exclude-slices N ... drops those sections entirely: they do not size the box
and their planes are left out of the output. The numbers go into the output
name (`_ex58`) so an excluded run never overwrites a full one. The planes either
side of a dropped section become neighbours while ImageJ's Z spacing stays one
section, so depth across the gap reads short by a section per dropped plane; the
offsets record keeps each kept section's true z_um.

Output, beside the input in <OUTPUT_ROOT>/section_stacks/:

    {stem}_subslicebox[_ex{N}...]_scalebar.tif            Z C Y X, calibrated, metadata carried
    {stem}_subslicebox[_ex{N}...]_scalebar_offsets.json   box origin in the stack, per-section
                                                          shift and subslice footprint in the box

Usage:
    python preprocessing/crop_section_stack_to_subslices.py sections_slice_MSCARLET__GCAMP.tif --dry-run
    python preprocessing/crop_section_stack_to_subslices.py sections_slice_MSCARLET__GCAMP.tif
    python preprocessing/crop_section_stack_to_subslices.py sections_slice_MSCARLET__GCAMP.tif --pad-um 100
    python preprocessing/crop_section_stack_to_subslices.py sections_slice_MSCARLET__GCAMP.tif --exclude-slices 58
"""

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h5py
import numpy as np
import tifffile

from preprocessing_config import (
    DOWNSAMPLE_XY,
    HYB_DOWNSAMPLED_DIR,
    HYB_SLICE_DOWNSAMPLED_DIR,
    HYB_STITCHED_DIR,
)
from add_section_stack_scale_bars import (
    CARRIED_METADATA,
    INPUT_DIR,
    bar_mask,
    report_geometry,
    resolve_stacks,
    stack_pitch,
)
from utilities.image_io import get_file_size_mb

BOX_TAG = "_subslicebox"
SUFFIX = "_scalebar"

# Relative tolerance on the recorded downsample factor; stored as a float.
FACTOR_RTOL = 1e-9


def read_record(path, keys, fovs=False):
    """Metadata `keys` (and fov_offsets when `fovs`) plus the cellmask shape,
    without reading the label array."""
    if not path.exists():
        raise FileNotFoundError(f"Stitching record not found: {path}")
    with h5py.File(path, "r") as f:
        meta = f["metadata"]
        missing = [k for k in keys if k not in meta]
        if missing:
            raise ValueError(f"{path.name}: no {missing} in its metadata")
        out = {k: meta[k][()] for k in keys}
        if fovs:
            if "fov_offsets" not in meta:
                raise ValueError(f"{path.name}: no fov_offsets in its metadata")
            grp = meta["fov_offsets"]
            out["fov_offsets"] = {name: np.asarray(grp[name][()], dtype=float)
                                  for name in grp}
        out["shape"] = tuple(f["cellmask"].shape)
    return out


def subslice_footprint(n, dy, dx, section_shape):
    """(y0, x0, y1, x1, n_shared_fovs): slice n's subslice in stack coordinates.

    None when the section has no step 3 subslice. Raises when the slice record
    is not the canvas the stack was built from, or the shared FOV placements or
    the downsample factors disagree.
    """
    sub_path = Path(HYB_DOWNSAMPLED_DIR) / f"slice{n}_subslice_CELLMASK.h5"
    if not sub_path.exists():
        return None
    sub = read_record(sub_path, ("min_x_offset", "min_y_offset", "DOWNSAMPLE_XY"))
    sl = read_record(Path(HYB_SLICE_DOWNSAMPLED_DIR) / f"slice{n}_CELLMASK.h5",
                     ("min_x", "min_y", "downsample_xy"), fovs=True)
    full = read_record(Path(HYB_STITCHED_DIR) / f"slice{n}_subslice_CELLMASK.h5",
                       ("min_x", "min_y"), fovs=True)

    if list(sl["shape"]) != list(section_shape):
        raise ValueError(
            f"Slice {n}: stitch_slices.py's canvas is {sl['shape']} but the stack "
            f"placed a {tuple(section_shape)} section; the whole slices were "
            f"re-stitched after the stack was built."
        )

    factors = {"step 3": float(sub["DOWNSAMPLE_XY"]),
               "stitch_slices.py": float(sl["downsample_xy"]),
               "config": float(DOWNSAMPLE_XY)}
    if max(factors.values()) - min(factors.values()) > FACTOR_RTOL * DOWNSAMPLE_XY:
        raise ValueError(f"Slice {n}: downsample factors disagree: {factors}")

    if (int(full["min_x"]), int(full["min_y"])) != (int(sub["min_x_offset"]),
                                                    int(sub["min_y_offset"])):
        raise ValueError(
            f"Slice {n}: step 2's origin ({full['min_x']}, {full['min_y']}) is not "
            f"step 3's ({sub['min_x_offset']}, {sub['min_y_offset']}); step 3 is "
            f"stale against step 2."
        )

    shared = sorted(set(full["fov_offsets"]) & set(sl["fov_offsets"]))
    if not shared:
        raise ValueError(f"Slice {n}: no FOV is in both the subslice and the slice record")
    moved = [(name, full["fov_offsets"][name], sl["fov_offsets"][name]) for name in shared
             if not np.allclose(full["fov_offsets"][name], sl["fov_offsets"][name])]
    if moved:
        listing = "\n".join(f"  {name}: subslice {a.tolist()}, slice {b.tolist()}"
                            for name, a, b in moved)
        raise ValueError(
            f"Slice {n}: {len(moved)} of {len(shared)} shared FOVs were placed "
            f"differently in the two stitches, so the origin difference does not "
            f"locate the subslice:\n{listing}"
        )

    oy = (float(sub["min_y_offset"]) - float(sl["min_y"])) / DOWNSAMPLE_XY + dy
    ox = (float(sub["min_x_offset"]) - float(sl["min_x"])) / DOWNSAMPLE_XY + dx
    h, w = sub["shape"]
    return oy, ox, oy + h, ox + w, len(shared)


def crop_stack(path, pad_um=0.0, exclude=(), dry_run=False):
    offsets_path = path.with_name(path.stem + "_offsets.json")
    if not offsets_path.exists():
        raise FileNotFoundError(f"No offsets file beside the stack: {offsets_path}")
    offsets = json.loads(offsets_path.read_text())
    if offsets.get("source") != "slice":
        raise ValueError(f"{path.name} is a --source {offsets.get('source')} stack; "
                         f"this crops whole-slice stacks only")

    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes, shape, dtype = series.axes, series.shape, series.dtype
        ij = dict(tif.imagej_metadata or {})
    if axes != "ZCYX":
        raise ValueError(f"{path.name}: axes {axes!r}, expected 'ZCYX'")
    Z, C, H, W = shape
    if [H, W] != list(offsets["canvas_shape"]):
        raise ValueError(f"{path.name} is {H} x {W} but its offsets file says "
                         f"{offsets['canvas_shape']}")
    pitch = stack_pitch(path)

    print(f"\n{path.name}")
    print(f"  Z {Z} x C {C} x {H} x {W} {dtype}, {get_file_size_mb(path):.0f} MB")

    in_stack = [sec["slice"] for sec in offsets["sections"]]
    unknown = sorted(set(exclude) - set(in_stack))
    if unknown:
        raise ValueError(f"--exclude-slices {unknown} not in this stack; it holds "
                         f"slices {in_stack[0]}..{in_stack[-1]} ({len(in_stack)})")
    kept = [sec for sec in offsets["sections"] if sec["slice"] not in exclude]
    if not kept:
        raise ValueError("--exclude-slices leaves no section")
    if exclude:
        print(f"  excluded: slice(s) {sorted(exclude)} -- not in the box, planes dropped "
              f"({len(kept)} of {Z} kept)")

    footprints = {}
    no_subslice = []
    for sec in kept:
        fp = subslice_footprint(sec["slice"], sec["dy"], sec["dx"], sec["shape"])
        if fp is None:
            no_subslice.append(sec["slice"])
        else:
            footprints[sec["slice"]] = fp
    if not footprints:
        raise ValueError("No section in this stack has a step 3 subslice")
    print(f"  subslices located: {len(footprints)} of {len(kept)} sections "
          f"(shared FOVs checked: {sum(fp[4] for fp in footprints.values())})")
    if no_subslice:
        print(f"  no subslice (plane kept, not used for the box): {no_subslice}")

    pad = int(round(pad_um / pitch))
    y0 = max(0, math.floor(min(fp[0] for fp in footprints.values())) - pad)
    x0 = max(0, math.floor(min(fp[1] for fp in footprints.values())) - pad)
    y1 = min(H, math.ceil(max(fp[2] for fp in footprints.values())) + pad)
    x1 = min(W, math.ceil(max(fp[3] for fp in footprints.values())) + pad)
    bh, bw = y1 - y0, x1 - x0
    print(f"  box: y {y0}:{y1}, x {x0}:{x1} -> {bw} x {bh} px "
          f"({bw * pitch / 1000:.2f} x {bh * pitch / 1000:.2f} mm), pad {pad_um:g} um = {pad} px; "
          f"{100 * bh * bw / (H * W):.1f}% of the canvas")
    report_geometry(pitch)

    ranges = ij.get("Ranges")
    if ranges is not None and len(ranges) == 2 * C:
        levels = [float(ranges[2 * c + 1]) for c in range(C)]
    else:
        levels = [float(np.iinfo(dtype).max)] * C
        print(f"  no per-channel Ranges stored; bars at the dtype max {levels[0]:g}")
    levels = [min(v, float(np.iinfo(dtype).max)) for v in levels]

    mask = bar_mask(bh, bw, pitch)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        print("  no bar fits this box; written without bars")
        by0 = by1 = bx0 = bx1 = 0
    else:
        by0, by1, bx0, bx1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    sub = mask[by0:by1, bx0:bx1]

    src = tifffile.memmap(str(path), mode="r")
    covered = {}
    for sec in kept:
        z = sec["z_index"]
        covered[sec["slice"]] = sum(
            int(np.count_nonzero(src[z, c, y0 + by0:y0 + by1, x0 + bx0:x0 + bx1][sub]))
            for c in range(C))
    painted = int(sub.sum()) * C
    hit = [n for n, k in covered.items() if k]
    print(f"  bars cover nonzero px on {len(hit)} of {len(kept)} planes"
          + (": " + ", ".join(f"slice {n} {covered[n]}/{painted}" for n in hit) if hit else ""))

    ex_tag = "".join(f"_ex{n}" for n in sorted(exclude))
    out_path = path.with_name(path.stem + BOX_TAG + ex_tag + SUFFIX + ".tif")
    out_offsets = out_path.with_name(out_path.stem + "_offsets.json")
    print(f"  output: {out_path}")
    if dry_run:
        del src
        return

    metadata = {"axes": "ZCYX"}
    metadata.update({k: ij[k] for k in CARRIED_METADATA if k in ij})
    # Labels run section-major, one per (section, channel); keep the kept ones.
    labels = metadata.get("Labels")
    if labels is not None and len(labels) == Z * C:
        metadata["Labels"] = [labels[sec["z_index"] * C + c] for sec in kept for c in range(C)]
    elif labels is not None:
        del metadata["Labels"]
    dst = tifffile.memmap(
        str(out_path), shape=(len(kept), C, bh, bw), dtype=dtype, imagej=True,
        resolution=(1 / pitch, 1 / pitch), metadata=metadata)
    for zo, sec in enumerate(kept):
        dst[zo] = src[sec["z_index"], :, y0:y1, x0:x1]
        for c in range(C):
            dst[zo, c, by0:by1, bx0:bx1][sub] = levels[c]
    dst.flush()
    del dst, src

    record = {
        "source_stack": path.name,
        "box_in_stack": {"y0": y0, "x0": x0, "y1": y1, "x1": x1},
        "pad_um": pad_um,
        "excluded_slices": sorted(exclude),
        "xy_um_per_px": pitch,
        "z_um_per_section": offsets.get("z_um_per_section"),
        "box_from_stack": "box (y, x) = stack (y - y0, x - x0)",
        "box_from_section": "box (y, x) = section (y + dy - y0, x + dx - x0)",
        "sections": [
            {"slice": sec["slice"], "z_index": zo, "source_z_index": sec["z_index"],
             "z_um": sec.get("z_um"), "dy": sec["dy"], "dx": sec["dx"],
             "subslice_in_box": (None if sec["slice"] not in footprints else
                                 [round(v - o, 3) for v, o in
                                  zip(footprints[sec["slice"]][:4], (y0, x0, y0, x0))])}
            for zo, sec in enumerate(kept)],
    }
    out_offsets.write_text(json.dumps(record, indent=2) + "\n")
    print(f"  wrote {out_path.name} ({get_file_size_mb(out_path):.0f} MB) + {out_offsets.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Crop a section stack to its subslices' box, with scale bars",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("stacks", nargs="+",
                        help="Whole-slice stack paths or filenames in section_stacks/")
    parser.add_argument("--pad-um", type=float, default=0.0,
                        help="margin around the subslice box, um (default 0)")
    parser.add_argument("--exclude-slices", type=int, nargs="+", default=[], metavar="N",
                        help="drop these sections: not in the box, planes left out")
    parser.add_argument("--dry-run", action="store_true",
                        help="locate subslices and report the box, write nothing")
    args = parser.parse_args()

    print(f"Stacks in: {INPUT_DIR}")
    for path in resolve_stacks(args.stacks):
        crop_stack(path, pad_um=args.pad_um, exclude=set(args.exclude_slices),
                   dry_run=args.dry_run)
    if args.dry_run:
        print("\nDry run: nothing written.")


if __name__ == "__main__":
    main()
