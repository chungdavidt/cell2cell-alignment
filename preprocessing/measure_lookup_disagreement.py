"""
How often does the geometric centroid lookup return a different cell than the id join?

check_cell_id_link.py established that filt_neurons.id names the cellmask label
exactly (id = FOV_index * 10000 + label, BY95, 20 FOVs, ~48,000 rows, 100%).
That makes `id % 10000` a ground truth, so the error rate of the lookup the
pipeline actually uses -- round the centroid, read whatever label is underneath
-- is measurable rather than arguable. This measures it.

Two parts, independent, both read-only:

PART 1, per FOV (needs cellmask.mat). Four lookup variants against the join,
so each defect is priced separately rather than as one lump:

    full_fixed    rint(pos40x) - 1            geometry alone
    full_current  rint(pos40x + 1) - 1        + the canvas off-by-one
    down_fixed    rint(pos40x / f) - 1        geometry + resample
    down_current  rint((pos40x + 1) / f) - 1  + off-by-one, = what ships today

f is DOWNSAMPLE_XY. `pos40x` is 1-indexed (measured), which is why the fixed
form carries the -1 and the current form does not; see
reference_centroid_mapping_formula.md. Column order is (x, y).

This runs on ONE FOV's own mask, not the stitched canvas. The canvas origin is a
translation, so it can shift the rounding phase but not the geometry -- these
numbers are representative of the pipeline, not byte-identical to it.

PART 2, per subslice (needs only filt_neurons + subslice definitions, no images).
stitch_subslices.py:212-226 lets a later FOV OVERWRITE an earlier one wherever
they overlap, and FOVs overlap by ~735 px of 3200. A cell whose pixels get
overwritten by a neighbouring FOV keeps its own id but sits under a label that
came from a different FOV, so the lookup returns another cell entirely. This
counts the cells exposed to that -- an UPPER BOUND, since the later FOV must
also have a non-background pixel there. Part 1 cannot see this at all.

Usage:
    python preprocessing/measure_lookup_disagreement.py               # both parts
    python preprocessing/measure_lookup_disagreement.py -n 30
    python preprocessing/measure_lookup_disagreement.py --slice 22
    python preprocessing/measure_lookup_disagreement.py --overlap-only
    python preprocessing/measure_lookup_disagreement.py --json disagreement.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from preprocessing_config import (
    FILT_NEURONS_PATH,
    HYB_ROOT,
    SUBSLICE_DEFINITIONS_FILE,
    DOWNSAMPLE_XY,
    FOV_SIZE,
)
from utilities.mat_io import load_filt_neurons, load_mat
from utilities.regression import calculate_fov_offset
from downsample_subslices_cellmask import imresize_nearest   # the pipeline's own resample

MASK_VAR_NAMES = ("maski", "cellmask", "mask", "segmentation", "seg")
ID_STRIDE = 10000

# (name, scale, plus_one). index = rint((pos40x + plus_one) / scale) - 1
VARIANTS = (
    ("full_fixed",   1.0, 0.0),
    ("full_current", 1.0, 1.0),
    ("down_fixed",   None, 0.0),     # None -> DOWNSAMPLE_XY, filled at runtime
    ("down_current", None, 1.0),
)
STATUSES = ("agree", "wrong_label", "background", "out_of_bounds")


def load_mask(fov_dir):
    path = fov_dir / "cellmask.mat"
    if not path.exists():
        return None
    data = load_mat(path)
    for name in MASK_VAR_NAMES:
        if name in data:
            mask = np.asarray(data[name])
            break
    else:
        return None
    if mask.ndim != 2 or mask.size == 0:
        return None
    if not np.issubdtype(mask.dtype, np.integer):
        mask = np.rint(mask).astype(np.int64)
    return mask


def lookup(labels, pos40x, truth, scale, plus_one):
    """Status counts for one variant. pos40x is (x, y), 1-indexed."""
    h, w = labels.shape
    x = np.rint((pos40x[:, 0] + plus_one) / scale).astype(np.int64) - 1
    y = np.rint((pos40x[:, 1] + plus_one) / scale).astype(np.int64) - 1
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)

    got = np.zeros(x.size, dtype=np.int64)
    got[inside] = labels[y[inside], x[inside]]

    status = np.full(x.size, "out_of_bounds", dtype=object)
    status[inside & (got == 0)] = "background"
    status[inside & (got != 0) & (got != truth)] = "wrong_label"
    status[inside & (got == truth)] = "agree"
    return status, got


def check_fov(fov_name, fov_dir, rows, pos40x_all, id_all):
    labels_full = load_mask(fov_dir)
    if labels_full is None:
        return None

    pos40x = np.asarray(pos40x_all[rows], dtype=np.float64)
    ids = np.asarray(id_all[rows], dtype=np.float64)
    ok = np.all(np.isfinite(pos40x), axis=1) & np.isfinite(ids)
    pos40x, ids = pos40x[ok], ids[ok].astype(np.int64)
    if ids.size < 3:
        return None

    truth = ids % ID_STRIDE
    present = set(np.unique(labels_full).tolist())
    has_label = np.array([int(t) in present for t in truth])

    # Rows whose label is missing from the mask cannot be scored either way.
    # They are reported, never folded into a rate.
    scored = has_label
    labels_down = imresize_nearest(
        labels_full,
        (round(labels_full.shape[0] / DOWNSAMPLE_XY),
         round(labels_full.shape[1] / DOWNSAMPLE_XY)),
    )

    result = {
        "fov": fov_name,
        "n_rows": int(ids.size),
        "n_scored": int(scored.sum()),
        "n_label_absent": int((~has_label).sum()),
        "variants": {},
    }
    for name, scale, plus_one in VARIANTS:
        lab = labels_full if scale is not None else labels_down
        s = 1.0 if scale is not None else DOWNSAMPLE_XY
        status, _ = lookup(lab, pos40x[scored], truth[scored], s, plus_one)
        c = Counter(status.tolist())
        n = int(scored.sum())
        result["variants"][name] = {
            **{k: int(c.get(k, 0)) for k in STATUSES},
            "disagree": int(n - c.get("agree", 0)),
            "rate": round((n - c.get("agree", 0)) / n, 6) if n else 0.0,
        }
    return result


def overlap_exposure(filt_neurons, target_slice):
    """Cells whose pixels a later FOV overwrites during stitching. Upper bound."""
    defs_path = Path(SUBSLICE_DEFINITIONS_FILE)
    if not defs_path.exists():
        return None, f"subslice definitions not found: {defs_path}"

    data = load_mat(defs_path)
    infos = data["subslice_info"]
    if isinstance(infos, np.ndarray):
        infos = list(infos.flatten())
    elif not isinstance(infos, list):
        infos = [infos]

    fov_of_row = np.asarray(filt_neurons["fov"])
    slice_ids = np.asarray(filt_neurons["slice"]).ravel()
    pos = np.asarray(filt_neurons["pos"], dtype=np.float64)
    pos40x = np.asarray(filt_neurons["pos40x"], dtype=np.float64)

    out = []
    for info in infos:
        slice_id = info["slice_id"] if isinstance(info, dict) else info.slice_id
        slice_id = int(np.asarray(slice_id).ravel()[0])
        if target_slice is not None and slice_id != target_slice:
            continue
        fov_list = info["fov_list"] if isinstance(info, dict) else info.fov_list
        if isinstance(fov_list, str):
            fov_list = [fov_list]
        fov_list = [str(f) for f in np.asarray(fov_list).ravel()]

        # Same offsets stitch_subslices.py computes, same regression.
        offsets = {}
        for name in fov_list:
            sel = np.where((fov_of_row == name) & (slice_ids == slice_id))[0]
            if sel.size < 3:
                continue
            try:
                offsets[name] = calculate_fov_offset(pos[sel], pos40x[sel], 2.0)
            except Exception:
                continue
        painters = [f for f in fov_list if f in offsets]
        if len(painters) < 2:
            continue

        in_slice = np.where(slice_ids == slice_id)[0]
        canvas = pos[in_slice] * 2.0
        own = fov_of_row[in_slice]

        # Later entries in fov_list paint over earlier ones.
        last = np.full(in_slice.size, -1, dtype=np.int64)
        for k, name in enumerate(painters):
            ox, oy = offsets[name]
            inside = ((canvas[:, 0] > ox) & (canvas[:, 0] <= ox + FOV_SIZE) &
                      (canvas[:, 1] > oy) & (canvas[:, 1] <= oy + FOV_SIZE))
            last[inside] = k

        idx = {name: k for k, name in enumerate(painters)}
        own_k = np.array([idx.get(str(f), -2) for f in own])
        placed = last >= 0
        at_risk = placed & (own_k >= 0) & (last != own_k)
        out.append({
            "slice_id": slice_id,
            "n_fovs": len(painters),
            "n_cells": int(placed.sum()),
            "n_at_risk": int(at_risk.sum()),
            "rate": round(float(at_risk.sum()) / int(placed.sum()), 6) if placed.any() else 0.0,
        })
    return out, None


def main():
    p = argparse.ArgumentParser(
        description="Measure how often the geometric centroid lookup disagrees "
                    "with the filt_neurons.id join",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("-n", type=int, default=10, help="FOVs to score in part 1 (default 10)")
    p.add_argument("--slice", type=int, default=None, help="restrict part 2 to this slice")
    p.add_argument("--min-cells", type=int, default=20, help="skip FOVs with fewer rows")
    p.add_argument("--overlap-only", action="store_true", help="skip part 1 (no images read)")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    print("=" * 40)
    print("LOOKUP DISAGREEMENT VS THE id JOIN")
    print("=" * 40)
    print(f"filt_neurons: {FILT_NEURONS_PATH}")
    print(f"DOWNSAMPLE_XY: {DOWNSAMPLE_XY:.4f}x   id stride: {ID_STRIDE}\n")

    fn = load_filt_neurons(FILT_NEURONS_PATH)
    fov_of_row = np.asarray(fn["fov"])
    pos40x = np.asarray(fn["pos40x"])
    ids = np.asarray(fn["id"]).ravel()

    results, totals = [], {name: Counter() for name, _, _ in VARIANTS}
    if not args.overlap_only:
        print("PART 1 — per-FOV lookup, four variants\n")
        counts = Counter(fov_of_row.tolist())
        checked = 0
        for name, c in counts.most_common():
            if checked >= args.n:
                break
            if c < args.min_cells:
                break
            rows = np.where(fov_of_row == name)[0]
            r = check_fov(name, Path(HYB_ROOT) / name, rows, pos40x, ids)
            if r is None:
                print(f"{name}: no readable cellmask, skipped")
                continue
            print(f"{r['fov']}   scored {r['n_scored']}/{r['n_rows']}"
                  f"   (label absent from mask: {r['n_label_absent']})")
            for vname, v in r["variants"].items():
                print(f"    {vname:<13} disagree {v['disagree']:>5} / {r['n_scored']:<5} "
                      f"({100 * v['rate']:6.3f}%)   "
                      f"wrong {v['wrong_label']}, background {v['background']}, "
                      f"oob {v['out_of_bounds']}")
            for vname, v in r["variants"].items():
                totals[vname].update({k: v[k] for k in STATUSES})
            results.append(r)
            checked += 1
        print()

        if results:
            n = sum(r["n_scored"] for r in results)
            print("-" * 40)
            print(f"TOTAL over {len(results)} FOVs, {n} scored cells")
            for vname, c in totals.items():
                dis = n - c["agree"]
                print(f"    {vname:<13} disagree {dis:>6} / {n} ({100 * dis / n:6.3f}%)")
            cur = n - totals["down_current"]["agree"]
            fix = n - totals["down_fixed"]["agree"]
            print(f"\n  What ships today is 'down_current': {100 * cur / n:.3f}%.")
            print(f"  Cost of the canvas off-by-one alone: "
                  f"{100 * (cur - fix) / n:+.3f} points.")
            print()

    print("PART 2 — cells exposed to FOV-overlap overwrite during stitching")
    print("  Upper bound: counts cells a later FOV paints over, without checking")
    print("  whether that FOV has a non-background pixel there. Part 1 cannot see this.\n")
    overlap, err = overlap_exposure(fn, args.slice)
    if err:
        print(f"  skipped — {err}")
    elif not overlap:
        print("  no subslice with 2+ placed FOVs")
    else:
        tot_c = sum(o["n_cells"] for o in overlap)
        tot_r = sum(o["n_at_risk"] for o in overlap)
        for o in overlap[:20]:
            print(f"    slice {o['slice_id']:>3}   {o['n_fovs']:>3} FOVs   "
                  f"{o['n_at_risk']:>6} / {o['n_cells']:<6} cells at risk "
                  f"({100 * o['rate']:6.3f}%)")
        if len(overlap) > 20:
            print(f"    ... {len(overlap) - 20} more subslices")
        print(f"\n  TOTAL {tot_r} / {tot_c} ({100 * tot_r / tot_c:.3f}%)" if tot_c else "")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"downsample_xy": DOWNSAMPLE_XY, "id_stride": ID_STRIDE,
             "fovs": results, "overlap": overlap}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
