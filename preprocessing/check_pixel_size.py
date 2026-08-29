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

Three probes:

    B  stage step   FOV grid step in pixels, from filt_neurons.mat alone
    C  soma size    label diameters in the raw per-FOV cellmasks
    D  the 2P side  stack geometry + acquisition metadata, from the TIFFs

B and C need no image metadata at all. B is decisive on the BARseq side: the FOV
grid step is a physical stage displacement, so step_um / step_px IS the pixel
size. It needs one number you supply, and it has a reference value to compare
against (see below). C needs nothing but only resolves ~20%: it catches a 2x or
2.83x and cannot separate 0.32 from 0.325.

D exists because a mismatch in APPARENT CELL SIZE between the two modalities is
a different symptom from a mismatch in image extent, and it indicts the 2P, not
BARseq. If both images really sit at the same pitch, a soma must occupy the same
pixel count in both. When BARseq cells look ~2.83x too big, the likely cause is
a 2P volume acquired at the 566 um zoom while SCOPE names the 200 um one. The
cheapest discriminator is not XY at all -- it is the PLANE COUNT: huang_lab is
401 planes at 1 um, huang_lab_566um is 201 at 2 um.

BARseq TIFF metadata is deliberately not probed. The per-FOV file is
`alignedn2vhyb01.tif` -- aligned, Noise2Void-denoised, max-projected -- at least
three rewrites past acquisition, none of which carry calibration forward. Its
tags are placeholders by construction. The 2P stacks are a different matter:
they come off the scope with ScanImage / Bruker headers intact, which is why D
reads them and B/C do not.

Read-only. Opens nothing for writing and touches no file under DATA_ROOT.

Usage:
    python preprocessing/check_pixel_size.py
    python preprocessing/check_pixel_size.py --step-um 788.8
    python preprocessing/check_pixel_size.py --slice 22 --n-fovs 8 --soma-um 12
    python preprocessing/check_pixel_size.py --only C
    python preprocessing/check_pixel_size.py --only D --tif "C:\\path\\to\\2p_folder"
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


# ----------------------------------------------------------------- D: the 2P

# Plane counts observed for each Huang zoom. Not a field in MICROSCOPE_PROFILES
# -- taken from the scope_profiles.py comments ("401 z levels" for huang_lab,
# "566.08 um FOV, 2 um Z" for the older setting) and from the by84/by94/by89 vs
# BY95 split. Z is what separates the two profiles most cheaply: 401 planes at
# 1 um vs 201 planes at 2 um.
PROFILE_PLANES = {"huang_lab": 401, "huang_lab_566um": 201, "li_lab": None}

# Acquisition keys worth surfacing out of a ScanImage / Bruker / MicroManager
# header. ScanImage puts "SI.hRoiManager.scanZoomFactor" and
# "SI.objectiveResolution" in ImageDescription; Bruker writes a .xml sidecar
# carrying "micronsPerPixel"; MicroManager writes "PixelSizeUm" into a .txt.
ACQ_KEY_RE = re.compile(
    r"(micron|pixelsize|pixel_size|umperpix|um_per_pix|scanzoom|zoomfactor|"
    r"objectiveresolution|resolution|stackzstep|zstepsize|framesperslice|"
    r"scanframerate|fovsize|field_?of_?view)", re.I)

SIDECAR_SUFFIXES = (".xml", ".txt", ".json", ".env", ".ini", ".cfg")


def _emit_acq_lines(text, source, limit=25):
    """Print lines of a header/sidecar that look like acquisition geometry."""
    hits = [ln.strip() for ln in text.splitlines() if ACQ_KEY_RE.search(ln)]
    if not hits:
        print(f"      {source}: no geometry-looking keys "
              f"({len(text)} chars scanned)")
        return
    print(f"      {source}: {len(hits)} geometry-looking line(s)")
    for h in hits[:limit]:
        print(f"        {h[:220]}")
    if len(hits) > limit:
        print(f"        ... {len(hits) - limit} more")


def _scan_sidecars(tif_path):
    """Report sibling files that acquisition software writes beside a stack."""
    tif_path = Path(tif_path)
    sibs = [f for f in tif_path.parent.iterdir()
            if f.is_file() and f.suffix.lower() in SIDECAR_SUFFIXES]
    if not sibs:
        print("    sidecars: none "
              f"({', '.join(SIDECAR_SUFFIXES)} in {tif_path.parent.name}/)")
        return
    print(f"    sidecars: {len(sibs)} candidate file(s)")
    for f in sorted(sibs)[:6]:
        try:
            text = f.read_text(errors="replace")
        except Exception as e:
            print(f"      {f.name}: unreadable ({type(e).__name__})")
            continue
        _emit_acq_lines(text, f.name)


def expand_tif_args(items, max_tifs, recursive=False):
    """
    Accept files or directories; a directory is searched for TIFF stacks.

    TOP LEVEL ONLY by default. The 2P stacks sit directly in the subject folder
    ("050526 BY95/BY95 invivo run two crosstalk removed red.tif"), while that
    same folder holds allen_transcriptomics/BY95/hyb_raw_files/ with 1300+
    per-FOV BARseq TIFFs underneath it. Recursing would bury the four stacks you
    want in the FOVs you don't. --recursive opts back in.
    """
    out = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            walk = p.rglob if recursive else p.glob
            found = sorted(set(list(walk("*.tif")) + list(walk("*.tiff"))))
            how = ("(recursive)" if recursive
                   else "(top level only; --recursive to descend)")
            print(f"  {p}\n    {len(found)} TIFF(s) found {how}", end="")
            if len(found) > max_tifs:
                print(f", showing the first {max_tifs} (--max-tifs to change)")
                found = found[:max_tifs]
            else:
                print()
            out.extend(found)
        else:
            out.append(p)
    return out


def probe_2p(paths):
    print("=" * 72)
    print("D. THE 2P STACK -- geometry and acquisition metadata")
    print("=" * 72)

    if not paths:
        print("  No 2P stack given.")
        print("  Pass --tif <path> (repeatable), or fill in INVIVO_PATH_RED /")
        print("  INVIVO_PATH_GREEN / BLOCK_STACK_PATH_RED / BLOCK_STACK_PATH_GREEN")
        print("  in local_config.py -- they are all blank, which is also why")
        print("  subslice_graph_builder's assert_matches_metadata has never run")
        print("  on this subject.\n")
        return

    import tifffile

    rows = []
    for p in paths:
        p = Path(p)
        print(f"\n  {p}")
        if not p.exists():
            print("    !! not found")
            rows.append((p.name, "not found", "", ""))
            continue
        try:
            with tifffile.TiffFile(str(p)) as tf:
                series = tf.series[0]
                shape, dtype = series.shape, series.dtype
                page = tf.pages[0]
                print(f"    shape {shape}  dtype {dtype}  pages {len(tf.pages)}")

                n_planes = shape[0] if len(shape) >= 3 else 1
                yx = shape[-2:]

                for tag in ("XResolution", "YResolution", "ResolutionUnit",
                            "Software", "Make", "Model", "DateTime"):
                    t = page.tags.get(tag)
                    if t is not None:
                        print(f"    {tag}: {t.value}")

                ij = tf.imagej_metadata or {}
                if ij:
                    keep = {k: v for k, v in ij.items()
                            if k in ("unit", "spacing", "slices", "frames",
                                     "channels", "Info")}
                    print(f"    imagej_metadata: "
                          f"{ {k: v for k, v in keep.items() if k != 'Info'} }")
                    if "Info" in keep:
                        _emit_acq_lines(str(keep["Info"]), "imagej Info")

                desc = page.tags.get("ImageDescription")
                if desc is not None:
                    text = str(desc.value)
                    print(f"    ImageDescription: {len(text)} chars")
                    _emit_acq_lines(text, "ImageDescription")

                ome = getattr(tf, "ome_metadata", None)
                if ome:
                    _emit_acq_lines(str(ome), "ome_metadata")

                _scan_sidecars(p)

                # ---- what the geometry implies -------------------------
                print(f"\n    GEOMETRY: {n_planes} planes, {yx[0]}x{yx[1]} in-plane")
                match = [k for k, v in PROFILE_PLANES.items()
                         if v is not None and v == n_planes]
                if match:
                    print(f"    plane count matches profile: {', '.join(match)}")
                else:
                    known = {k: v for k, v in PROFILE_PLANES.items() if v}
                    print(f"    plane count matches no profile (known: {known})")
                rows.append((p.name, str(shape), str(n_planes),
                             ", ".join(match) if match else "-"))

                z_meta = None
                if "spacing" in ij:
                    try:
                        z_meta = float(ij["spacing"])
                        print(f"    z step from ImageJ metadata: {z_meta} um")
                    except (TypeError, ValueError):
                        pass

                print(f"\n    CANDIDATE PITCHES for a {yx[1]} px frame:")
                from scope_profiles import MICROSCOPE_PROFILES
                soma_um = 12.0
                for name, prof in MICROSCOPE_PROFILES.items():
                    xy = prof["xy_um_per_px"]
                    fov = yx[1] * xy
                    px = soma_um / xy
                    flag = "  <-- SCOPE" if name == SCOPE else ""
                    print(f"      {name:<17} {xy:7.4f} um/px  FOV {fov:8.2f} um"
                          f"  z {prof['z_um_per_px']:.1f}  "
                          f"{soma_um:.0f}um soma = {px:5.1f} px{flag}")

                print(f"\n    THE DECIDING MEASUREMENT: open this stack, measure a")
                print(f"    soma across in pixels, and read it off the table above.")
                print(f"    Your BARseq somas are 38.3 px raw at "
                      f"{EXVIVO_UM_PER_PX} um/px, i.e. "
                      f"{38.3 / DOWNSAMPLE_XY:.1f} px after the resample.")
                print(f"    If 2P somas are near that, the pitches agree. If they")
                print(f"    are ~2.83x smaller, SCOPE should be huang_lab_566um.")

        except Exception as e:
            print(f"    !! could not read: {type(e).__name__}: {e}")
            rows.append((p.name, f"ERROR {type(e).__name__}", "", ""))

    if len(rows) > 1:
        print("\n" + "=" * 72)
        print("  SUMMARY -- plane count is what separates the two Huang zooms")
        print("=" * 72)
        w = max(len(r[0]) for r in rows)
        print(f"  {'file':<{w}}  {'shape':<22}{'planes':>7}  profile by planes")
        for name, shape, planes, prof in rows:
            print(f"  {name:<{w}}  {shape:<22}{planes:>7}  {prof}")
        profs = {r[3] for r in rows if r[3] and r[3] != "-"}
        print()
        if len(profs) > 1:
            print(f"  MIXED: {sorted(profs)} -- these stacks are not all the same")
            print("  zoom. One SCOPE cannot describe them; the graph currently")
            print("  stamps every 2P node with SCOPE's pitch regardless.")
        elif profs and SCOPE not in profs:
            print(f"  All stacks look like {sorted(profs)[0]}, but SCOPE = {SCOPE!r}.")
            print("  That is the 2.83x, and it is one line in local_config.py.")
        elif profs:
            print(f"  All stacks agree with SCOPE = {SCOPE!r} on plane count.")
            print("  Plane count is necessary, not sufficient -- confirm with a")
            print("  soma measurement or an acquisition header above.")
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
    ap.add_argument("--tif", action="append", default=[], metavar="PATH",
                    help="2P stack or a DIRECTORY of them, for probe D. "
                         "Repeatable. Blank -> the four 2P paths in "
                         "local_config.py, which are currently unset.")
    ap.add_argument("--max-tifs", type=int, default=20,
                    help="cap on TIFFs read per directory (default 20)")
    ap.add_argument("--recursive", action="store_true",
                    help="descend into subdirectories when --tif is a folder. "
                         "Off by default: the subject folder holds the 2P stacks "
                         "at its top level and 1300+ BARseq FOV TIFFs beneath it.")
    ap.add_argument("--only", default="", metavar="BCD",
                    help="run only these probes, e.g. 'B', 'D', 'CD'")
    args = ap.parse_args()

    run = args.only.upper() or "BCD"

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
    if "D" in run:
        import local_config
        tif_args = args.tif or [
            p for p in (getattr(local_config, name, "") for name in (
                "INVIVO_PATH_RED", "INVIVO_PATH_GREEN",
                "BLOCK_STACK_PATH_RED", "BLOCK_STACK_PATH_GREEN"))
            if p
        ]
        probe_2p(expand_tif_args(tif_args, args.max_tifs, args.recursive)
                 if tif_args else [])

    print("=" * 72)
    print("A ratio of 2.827 anywhere above would mean SCOPE should be")
    print("huang_lab_566um -- but the 2P is confirmed at 200.19 um / 512 px,")
    print("so any ratio here belongs to the BARseq side and lands on one line:")
    print("EXVIVO_UM_PER_PX in scope_profiles.py.")
    print("=" * 72)


if __name__ == "__main__":
    main()
