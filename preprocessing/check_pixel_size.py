#!/usr/bin/env python3
"""
Measure the BARseq in-plane pixel size instead of declaring it.

`EXVIVO_UM_PER_PX = 0.32` (scope_profiles.py) is the last unverified number in
the scaling chain. The 2P side is settled -- BY95 is 200.19 um over 512 px, so
0.3910 um/px and DOWNSAMPLE_XY = 1.2219 are correct. Nothing checks the BARseq
side: raw hyb TIFFs carry a 72-DPI placeholder, `is_plausible_xy` rejects it,
and `assert_matches_metadata` silently returns. The constant also has no
per-dataset knob -- it is declared "invariant across datasets" -- while BY95's
filt_neurons.mat comes from collaborators, not the rig 0.32 was confirmed on.

Two probes, neither needing image metadata:

    B  stage step   FOV grid step in pixels, from filt_neurons.mat alone
    C  soma size    label diameters in the raw per-FOV cellmasks

B is decisive. The FOV grid step is a physical stage displacement, so
step_um / step_px IS the pixel size. It needs one number you supply, and it has
a reference value to compare against (see below). C needs nothing at all but
only resolves ~20%: it catches a 2x or 2.83x and cannot separate 0.32 from
0.325.

TIFF metadata is deliberately not probed. The per-FOV file is
`alignedn2vhyb01.tif` -- aligned, Noise2Void-denoised, max-projected -- at least
three rewrites past acquisition, none of which carry calibration forward. Its
tags are placeholders by construction. If the calibration survives anywhere it
is upstream of that chain or in an acquisition sidecar, neither of which this
sees.

Read-only. Opens nothing for writing and touches no file under DATA_ROOT.

Usage:
    python preprocessing/check_pixel_size.py
    python preprocessing/check_pixel_size.py --step-um 788.8
    python preprocessing/check_pixel_size.py --slice 22 --n-fovs 8 --soma-um 12
    python preprocessing/check_pixel_size.py --only C
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
    FILT_NEURONS_PATH,
    HYB_ROOT,
    HYB_CHANNELS_DIR,
    EXVIVO_UM_PER_PX,
    TARGET_XY_UM_PER_PX,
    DOWNSAMPLE_XY,
    SCOPE,
    FOV_SIZE,
)
from utilities.mat_io import load_filt_neurons, load_mat

# Cellmask variable names, in the order load_fov_images tries them.
MASK_VAR_NAMES = ("maski", "cellmask", "mask", "segmentation", "seg")

# ---------------------------------------------------------------------------
# JH302 reference geometry, read from Ben's ORIGINAL stitcher
# (dtc-li-lab-rotation-project/scripts/ben_generate_in_situ_stack.m), not the
# _dtc edit. Ben places each FOV deterministically on a fixed grid parsed from
# the folder name -- no regression:
#
#     x_offset = (2465*(num_of_FOV_x-1-FOV_x_pos));
#     y_offset = (2465*FOV_y_pos);
#     im_tmp(y_offset+1+165:y_offset+3200-166, x_offset+1+203:x_offset+3200-143)
#         = im_hyb(166:3200-166, 204:3200-143);
#
# So the stage step is a hard constant, 2465 full-res px on a 3200 px FOV. The
# resulting 735 px overlap is the same 735 that appears in his FOV-count
# formula, `(size*10 + 710 - 735)/2465`, which also fixes the checkregistration
# preview at a 10x downsample of the full-res canvas -- consistent with his
# debug plots drawing cells at pos*2 on the canvas and pos/5 on the preview.
#
# NOTE the X grid index is mirrored (num_of_FOV_x-1-FOV_x_pos) while Y is
# direct, so under Ben's convention a +1 step in the first filename index moves
# -2465 px in x. This probe measures signed displacement and reports it, rather
# than assuming either sign.
# ---------------------------------------------------------------------------
JH302_GRID_STEP_PX = 2465
JH302_FOV_PX = 3200
JH302_CROP_ROWS = (166, 3200 - 166)     # 2869 rows kept
JH302_CROP_COLS = (204, 3200 - 143)     # 2854 cols kept

GRID_RE = re.compile(r"_(\d+)_(\d+)$")


def implied_line(label, um_per_px):
    """One summary line: an estimate, and what it does to the configured factor."""
    ratio = um_per_px / EXVIVO_UM_PER_PX
    factor = TARGET_XY_UM_PER_PX / um_per_px
    return (f"  {label:<38} {um_per_px:7.4f} um/px   "
            f"{ratio:5.3f}x configured   DOWNSAMPLE_XY -> {factor:.4f}")


def _fit(x, y):
    """Least-squares intercept and slope for y = b0 + b1*x."""
    X = np.column_stack([np.ones_like(x, dtype=float), np.asarray(x, dtype=float)])
    b = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)[0]
    return float(b[0]), float(b[1])


# ------------------------------------------------------------ B: stage step

def probe_stage_step(filt_neurons, slice_id, step_um):
    print("=" * 72)
    print("B. STAGE STEP -- FOV grid displacement, in pixels")
    print("=" * 72)

    fov_names = [str(f) for f in filt_neurons["fov"]]
    slice_ids = np.asarray(filt_neurons["slice"]).ravel()
    pos = np.asarray(filt_neurons["pos"])
    pos40x = np.asarray(filt_neurons["pos40x"])

    fov_arr = np.asarray(fov_names)

    if slice_id is None:
        finite = np.unique(slice_ids[np.isfinite(slice_ids)])
        counts = {int(s): int((slice_ids == s).sum()) for s in finite}
        if not counts:
            print("  no finite slice IDs in filt_neurons")
            return None
        slice_id = max(counts, key=counts.get)
        print(f"  no --slice given; using the most populated: slice {slice_id} "
              f"({counts[slice_id]} cells)")

    in_slice = slice_ids == slice_id
    fovs = sorted(set(fov_arr[in_slice]))
    print(f"  slice {int(slice_id)}: {len(fovs)} FOVs, {int(in_slice.sum())} cells\n")

    # --- per-FOV regression: pos*2 = intercept + slope * pos40x -------------
    offsets, slopes = {}, []
    skipped = 0
    for name in fovs:
        m = (fov_arr == name) & in_slice
        if m.sum() < 3:
            skipped += 1
            continue
        b0x, b1x = _fit(pos40x[m, 0], pos[m, 0] * 2.0)
        b0y, b1y = _fit(pos40x[m, 1], pos[m, 1] * 2.0)
        offsets[name] = (b0x, b0y)
        slopes.extend([b1x, b1y])

    if skipped:
        print(f"  {skipped} FOV(s) skipped (<3 cells)")
    if len(offsets) < 2:
        print("  fewer than 2 positionable FOVs; cannot measure a step\n")
        return None

    # --- the slope the stitcher fits and then throws away -------------------
    slopes = np.asarray(slopes)
    dev = float(np.abs(slopes - 1.0).max())
    print("  Regression slope (pos*2 vs pos40x). The stitcher fits this and keeps")
    print("  only the intercept, asserting slope == 1 without ever reporting it.")
    print(f"    median {np.median(slopes):.6f}   min {slopes.min():.6f}   "
          f"max {slopes.max():.6f}")
    print(f"    max |slope-1| = {dev:.6f}  ->  up to {dev * FOV_SIZE:.1f} px "
          f"placement error at the FOV edge")
    verdict = ("OK -- pos*2 and pos40x share a pitch" if dev < 0.005
               else "CHECK -- pos and pos40x are not in a 1:2 relationship")
    print(f"    {verdict}")
    print("    (JH302 measured this as 'scale factor = 1.0', recorded in")
    print("     crop_ex_vivo_slices_with_metadata.m)\n")

    # --- step between grid-adjacent FOVs ------------------------------------
    grid = {}
    for name in offsets:
        m = GRID_RE.search(name)
        if m:
            grid[name] = (int(m.group(1)), int(m.group(2)))

    step_px = None
    if len(grid) == len(offsets) and grid:
        print(f"  All {len(grid)} FOV names carry a grid index "
              f"(trailing _NNN_NNN). Step between grid-adjacent FOVs:\n")
        axis_steps = {}
        for a in (0, 1):
            deltas = []
            for n1, g1 in grid.items():
                for n2, g2 in grid.items():
                    if g2[a] - g1[a] == 1 and g2[1 - a] == g1[1 - a]:
                        deltas.append((offsets[n2][0] - offsets[n1][0],
                                       offsets[n2][1] - offsets[n1][1]))
            if not deltas:
                print(f"    filename index {a} +1:  no adjacent pairs in this slice")
                continue
            arr = np.asarray(deltas, dtype=float)
            mdx, mdy = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
            print(f"    filename index {a} +1  ({len(arr):3d} pairs):  "
                  f"dx {mdx:9.1f} px (sd {arr[:, 0].std():6.1f})   "
                  f"dy {mdy:9.1f} px (sd {arr[:, 1].std():6.1f})")
            axis_steps[a] = (mdx, mdy)
        # The grid axis moves along whichever image axis has the larger |median|.
        mags = [abs(v) for pair in axis_steps.values() for v in pair]
        if mags:
            step_px = float(np.median([m for m in mags if m > 50])) if any(
                m > 50 for m in mags) else None
        print("\n    (Ben mirrors the first index: x_offset = 2465*(N-1-idx),")
        print("     so a negative dx there is his convention, not an error.)")
    else:
        print(f"  only {len(grid)}/{len(offsets)} FOV names parse as a grid; "
              f"using nearest-neighbour fallback")

    # --- fallback / cross-check: axis-aligned neighbour distances -----------
    names = list(offsets)
    ox = np.array([offsets[n][0] for n in names])
    oy = np.array([offsets[n][1] for n in names])
    cand = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dx, dy = abs(ox[i] - ox[j]), abs(oy[i] - oy[j])
            if dx > 50 and dy < 50:
                cand.append(dx)
            elif dy > 50 and dx < 50:
                cand.append(dy)
    if cand:
        cand = np.asarray(cand)
        near = cand[cand <= np.percentile(cand, 40)] if cand.size > 4 else cand
        nn_step = float(np.median(near))
        print(f"\n  Nearest-neighbour step (axis-aligned pairs, n={cand.size}): "
              f"{nn_step:.1f} px")
        if step_px is None:
            step_px = nn_step

    if step_px is None:
        print("\n  could not determine a step\n")
        return None

    # --- compare against JH302 ---------------------------------------------
    overlap_px = FOV_SIZE - step_px
    print(f"\n  MEASURED STEP: {step_px:.1f} px on a {FOV_SIZE} px FOV")
    print(f"    overlap {overlap_px:.0f} px = {100 * overlap_px / FOV_SIZE:.1f}%")
    print(f"  JH302 REFERENCE: {JH302_GRID_STEP_PX} px on {JH302_FOV_PX} px "
          f"= {100 * (JH302_FOV_PX - JH302_GRID_STEP_PX) / JH302_FOV_PX:.1f}% overlap")
    print(f"    (hardcoded in Ben's ben_generate_in_situ_stack.m; at "
          f"{EXVIVO_UM_PER_PX} um/px that is a {JH302_GRID_STEP_PX * EXVIVO_UM_PER_PX:.1f} um stage step)")
    rel = step_px / JH302_GRID_STEP_PX
    print(f"    BY95 / JH302 step ratio: {rel:.4f}")
    if abs(rel - 1.0) < 0.02:
        print("    -> same stitching geometry. 0.32 um/px transfers from JH302,")
        print("       to the extent JH302's own 0.32 was ever right.")
    else:
        print("    -> DIFFERENT geometry. 0.32 was confirmed on the JH302 rig and")
        print("       does not automatically carry to this acquisition.")

    print()
    if step_um:
        print(implied_line(f"stage step {step_um:g} um / {step_px:.1f} px",
                           step_um / step_px))
    else:
        print(f"  Supply the commanded stage step with --step-um to convert this")
        print(f"  into um/px. What the configured {EXVIVO_UM_PER_PX} um/px predicts:")
        print(f"    stage step  {step_px * EXVIVO_UM_PER_PX:8.1f} um")
        print(f"    FOV width   {FOV_SIZE * EXVIVO_UM_PER_PX:8.1f} um")
        print(f"  Ask the collaborators whether either matches their acquisition.")
    print()
    return step_px


# ------------------------------------------------------------- C: soma size

def probe_soma_size(hyb_root, channels_root, n_fovs, soma_um):
    print("=" * 72)
    print("C. SOMA SIZE -- label diameters in the raw per-FOV cellmasks")
    print("=" * 72)

    hyb_root = Path(hyb_root)
    if not hyb_root.is_dir():
        print(f"  hyb root not found: {hyb_root}\n")
        return

    fov_dirs = sorted(d for d in hyb_root.iterdir() if d.is_dir())
    diams, shapes, used = [], set(), 0

    for d in fov_dirs:
        if used >= n_fovs:
            break
        path = Path(channels_root) / d.name / "cellmask.mat"
        if not path.exists():
            path = d / "cellmask.mat"
        if not path.exists():
            continue
        try:
            data = load_mat(path)
        except Exception as e:
            print(f"  {d.name}: load failed ({type(e).__name__}: {e})")
            continue

        mask = None
        for name in MASK_VAR_NAMES:
            if name in data:
                mask = np.asarray(data[name])
                break
        if mask is None or not np.issubdtype(mask.dtype, np.number):
            print(f"  {d.name}: no numeric mask variable")
            continue

        flat = mask.ravel().astype(np.int64)
        if flat.size == 0 or flat.max() <= 0:
            print(f"  {d.name}: empty mask")
            continue
        # Per-FOV labels run 1..N_cells_in_FOV, so bincount is cheap. Guard in
        # case a mask ever arrives already carrying global ids (fov*10000+label),
        # where the allocation would be by max id rather than by cell count.
        if flat.max() > 10_000_000:
            print(f"  {d.name}: max label {flat.max()} looks like a global id, "
                  f"skipped")
            continue

        areas = np.bincount(np.clip(flat, 0, None))[1:]
        areas = areas[areas > 0]
        if areas.size == 0:
            continue

        diams.append(np.sqrt(4.0 * areas / np.pi))
        shapes.add(tuple(mask.shape))
        used += 1
        print(f"  {d.name}: {areas.size:5d} labels, mask {mask.shape}, "
              f"median area {np.median(areas):7.0f} px")

    if not diams:
        print("\n  no readable cellmasks found\n")
        return

    d = np.concatenate(diams)
    q = np.percentile(d, [25, 50, 75])
    print(f"\n  {d.size} labels over {used} FOV(s)")
    print(f"  mask shape(s) seen: {sorted(shapes)}   FOV_SIZE in config: {FOV_SIZE}")
    if shapes and all(s[0] != FOV_SIZE or s[1] != FOV_SIZE for s in shapes):
        print(f"    WARNING: no mask is {FOV_SIZE}x{FOV_SIZE}. The stitcher sizes")
        print(f"    its canvas with the hardcoded FOV_SIZE but places by actual")
        print(f"    image shape, so a mismatch pads or clips the canvas.")

    print(f"\n  equivalent diameter sqrt(4A/pi):")
    print(f"    median {q[1]:6.1f} px    IQR {q[0]:.1f} - {q[2]:.1f} px")
    print(f"  at the configured {EXVIVO_UM_PER_PX} um/px that is "
          f"{q[1] * EXVIVO_UM_PER_PX:.1f} um "
          f"(IQR {q[0] * EXVIVO_UM_PER_PX:.1f} - {q[2] * EXVIVO_UM_PER_PX:.1f} um)")
    print(f"  Cortical somas run ~10-15 um, so the median should land in range.")
    print(f"  This resolves ~20%: it catches a 2x or 2.83x, not 0.32 vs 0.325.\n")

    for um in sorted({10.0, float(soma_um), 15.0}):
        print(implied_line(f"if the median soma is {um:g} um", um / q[1]))
    print()


# ------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(
        description="Measure the BARseq pixel size two ways",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--step-um", type=float, default=None,
                    help="commanded stage step between adjacent FOVs, um "
                         "(from the collaborators' acquisition settings)")
    ap.add_argument("--soma-um", type=float, default=12.0,
                    help="assumed median soma diameter for probe C (default 12)")
    ap.add_argument("--slice", type=int, default=None,
                    help="slice to measure the FOV grid on (default: most populated)")
    ap.add_argument("--n-fovs", type=int, default=4,
                    help="FOVs to read in probe C (default 4)")
    ap.add_argument("--only", default="", metavar="BC",
                    help="run only these probes, e.g. 'B' or 'C'")
    args = ap.parse_args()

    run = args.only.upper() or "BC"

    print("=" * 72)
    print("BARSEQ PIXEL SIZE CHECK")
    print("=" * 72)
    print(f"  BARseq pitch  {EXVIVO_UM_PER_PX} um/px   EXVIVO_UM_PER_PX, "
          f"scope_profiles.py")
    print(f"                -- hand-entered, no per-dataset knob, UNVERIFIED")
    print(f"  2P pitch      {TARGET_XY_UM_PER_PX:.4f} um/px   SCOPE = {SCOPE!r}")
    print(f"                -- 200.19 um / 512 px, confirmed")
    print(f"  DOWNSAMPLE_XY {DOWNSAMPLE_XY:.4f}x")
    print(f"  A {FOV_SIZE} px FOV is therefore "
          f"{FOV_SIZE * EXVIVO_UM_PER_PX:.1f} um across")
    print(f"  hyb root      {HYB_ROOT}\n")

    if "B" in run:
        fn = load_filt_neurons(FILT_NEURONS_PATH)
        probe_stage_step(fn, args.slice, args.step_um)
    if "C" in run:
        probe_soma_size(HYB_ROOT, HYB_CHANNELS_DIR, args.n_fovs, args.soma_um)

    print("=" * 72)
    print("A ratio of 2.827 anywhere above would mean SCOPE should be")
    print("huang_lab_566um -- but the 2P is confirmed at 200.19 um / 512 px,")
    print("so any ratio here belongs to the BARseq side and lands on one line:")
    print("EXVIVO_UM_PER_PX in scope_profiles.py.")
    print("=" * 72)


if __name__ == "__main__":
    main()
