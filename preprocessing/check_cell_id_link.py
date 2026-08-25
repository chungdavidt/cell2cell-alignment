"""
Read-only probe: does filt_neurons.id name a cellmask label, and is pos40x 1-indexed?

Two questions, one measurement. Both come out of archive/ben_generate_in_situ_stack.m:

  1. Lines 106-118 (commented out there) join a filt_neurons row to a label in
     maski by rebasing the id within the FOV:

         cell_ids_one_index = uint16(cell_ids - min(cell_ids)) + 1
         im_hyb(im_hyb == id) = 65535

     If that holds, cells link to masks EXACTLY and the geometric lookup in
     export_subslice_cells.py / generate_alignment_tif.py -- round the centroid,
     read whatever label is underneath -- can be replaced by a join.

  2. Line 132 places a FOV at im_tmp(y+1:y+3200, x+1:x+3200), so pos*2 lands on
     the 1-indexed row pos*2 only if pos40x is 1-indexed. The centroid formula in
     four files uses (pos*2 - (min_x_offset - 1)) / DOWNSAMPLE_XY, then rounds and
     subtracts 1, which is correct only if pos40x is 0-INDEXED. If it is
     1-indexed instead, every centroid is one full-res pixel down-and-right
     (0.32 um; 0.82 downsampled px at BY95's 1.2219x factor). See
     reference_centroid_mapping_formula.md.

Method, per FOV: take every filt_neurons row in that FOV (no slice or QC filter,
matching ben_generate_in_situ_stack.m:109), compute each mask label's area
centroid, and for each candidate id->label rule measure pos40x minus that
centroid. The MEDIAN of that difference answers question 2 (0 = pos40x is
0-indexed, +1 = 1-indexed); its SPREAD answers question 1 (a wrong join scatters,
the right one is tight). A nearest-centroid match is run alongside as a
cross-check that assumes neither answer.

Writes nothing except an optional --json report.

Usage:
    python preprocessing/check_cell_id_link.py                 # 5 FOVs, most cells first
    python preprocessing/check_cell_id_link.py -n 20
    python preprocessing/check_cell_id_link.py --fov MAX_Pos1_003_005
    python preprocessing/check_cell_id_link.py --json id_link.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import ndimage as ndi

from preprocessing_config import FILT_NEURONS_PATH, HYB_ROOT
from utilities.mat_io import load_filt_neurons, load_mat

MASK_VAR_NAMES = ("maski", "cellmask", "mask", "segmentation", "seg")

# id -> label rules to test. Named for the report; the second is Ben's.
CANDIDATES = (
    ("id", lambda ids: ids),
    ("id-min(id)+1", lambda ids: ids - ids.min() + 1),
    ("id-min(id)", lambda ids: ids - ids.min()),
)

# A cell is "consistent" with a rule if its offset sits within this many pixels
# of the rule's median offset, in both axes.
TOL_PX = 0.5


def load_mask(fov_dir):
    """Return the label array for one FOV, or None."""
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


def label_centroids(labels):
    """label -> (row, col) area centroid, 0-indexed. Matches regionprops Centroid
    except for MATLAB's 1-indexing, which is exactly what we are measuring."""
    present = np.unique(labels)
    present = present[present > 0]
    if present.size == 0:
        return present, np.empty((0, 2))
    coms = ndi.center_of_mass(np.ones(labels.shape, dtype=np.uint8), labels, present)
    return present, np.asarray(coms, dtype=np.float64)


def offsets_for(rule_labels, pos40x, lut, order):
    """pos40x minus the joined label's centroid, for rows whose label exists.

    order 'xy': pos40x[:, 0] is the column, pos40x[:, 1] is the row.
    order 'yx': the reverse. A column swap is invisible in a square FOV, so both
    are measured and the tighter one is reported.
    """
    hit = np.array([lab in lut for lab in rule_labels])
    if not hit.any():
        return hit, np.empty((0, 2))
    cen = np.array([lut[lab] for lab in rule_labels[hit]])      # (row, col)
    cen_rc = cen if order == "yx" else cen[:, ::-1]             # -> match pos40x order
    return hit, pos40x[hit] - cen_rc


def summarize(offs):
    """(median offset, fraction within TOL_PX of it in both axes)."""
    if offs.shape[0] == 0:
        return None, 0.0
    med = np.median(offs, axis=0)
    ok = np.all(np.abs(offs - med) <= TOL_PX, axis=1)
    return med, float(ok.mean())


def check_fov(fov_name, fov_dir, rows, pos40x_all, id_all):
    """Return a result dict for one FOV, or None if it could not be evaluated."""
    labels = load_mask(fov_dir)
    if labels is None:
        return None

    present, coms = label_centroids(labels)
    if present.size == 0:
        return None
    lut = {int(lab): tuple(c) for lab, c in zip(present, coms)}

    pos40x = np.asarray(pos40x_all[rows], dtype=np.float64)
    ids = np.asarray(id_all[rows])
    finite = np.all(np.isfinite(pos40x), axis=1) & np.isfinite(ids)
    pos40x, ids, rows = pos40x[finite], ids[finite].astype(np.int64), rows[finite]
    if rows.size < 3:
        return None

    result = {
        "fov": fov_name,
        "n_rows": int(rows.size),
        "n_labels": int(present.size),
        "id_min": int(ids.min()), "id_max": int(ids.max()),
        "id_unique": bool(np.unique(ids).size == ids.size),
        "id_contiguous": bool(ids.max() - ids.min() + 1 == np.unique(ids).size),
        "rules": {},
    }

    for name, fn in CANDIDATES:
        rule_labels = fn(ids)
        best = None
        for order in ("xy", "yx"):
            hit, offs = offsets_for(rule_labels, pos40x, lut, order)
            med, frac = summarize(offs)
            if med is None:
                continue
            if best is None or frac > best["consistent"]:
                best = {"order": order, "coverage": int(hit.sum()),
                        "median": [round(float(v), 3) for v in med],
                        "consistent": round(frac, 4)}
        result["rules"][name] = best or {"coverage": 0}

    # Assumption-free cross-check: match each row to its nearest centroid and read
    # off what the label - id relationship actually is.
    cen_xy = coms[:, ::-1]                                       # (col, row)
    d = pos40x[:, None, :] - cen_xy[None, :, :]
    near = np.argmin((d ** 2).sum(axis=2), axis=1)
    resid = pos40x - cen_xy[near]
    matched = present[near].astype(np.int64)
    result["nearest"] = {
        "median_resid": [round(float(v), 3) for v in np.median(resid, axis=0)],
        "max_resid": round(float(np.abs(resid).max()), 2),
        "label_minus_id_mode": int(Counter((matched - ids).tolist()).most_common(1)[0][0]),
        "label_minus_id_agree": round(float(
            (matched - ids == Counter((matched - ids).tolist()).most_common(1)[0][0]).mean()), 4),
    }
    result["unclaimed_labels"] = int(present.size - np.unique(matched).size)
    return result


def fmt(vec):
    return "(" + ", ".join(f"{float(v):+.3f}" for v in vec) + ")"


def report(r):
    print(f"{r['fov']}   rows {r['n_rows']}   mask labels {r['n_labels']}   "
          f"id {r['id_min']}..{r['id_max']} "
          f"({'unique' if r['id_unique'] else 'DUPLICATED'}, "
          f"{'contiguous' if r['id_contiguous'] else 'gapped'})")
    for name, s in r["rules"].items():
        if not s.get("coverage"):
            print(f"    {name:<14} coverage 0/{r['n_rows']}")
            continue
        print(f"    {name:<14} coverage {s['coverage']}/{r['n_rows']}   "
              f"median offset {fmt(s['median'])} px [{s['order']}]   "
              f"within {TOL_PX} px: {100 * s['consistent']:.1f}%")
    n = r["nearest"]
    print(f"    nearest-centroid  median residual {fmt(n['median_resid'])} px, "
          f"max |resid| {n['max_resid']} px;  label - id = {n['label_minus_id_mode']:+d} "
          f"for {100 * n['label_minus_id_agree']:.1f}% of rows")
    print(f"    labels claimed by no row: {r['unclaimed_labels']}")


def verdict(results):
    print("=" * 40)
    print("VERDICT")
    print("=" * 40)

    best_name, best_frac = None, -1.0
    for name, _ in CANDIDATES:
        fracs = [r["rules"][name]["consistent"] for r in results
                 if r["rules"][name].get("coverage")]
        if not fracs:
            continue
        mean = float(np.mean(fracs))
        if mean > best_frac:
            best_name, best_frac = name, mean

    joined = best_name is not None and best_frac >= 0.95
    if joined:
        print(f"JOIN: '{best_name}' holds for {100 * best_frac:.1f}% of rows.")
        print("  filt_neurons.id names a cellmask label -> export_subslice_cells.py and")
        print("  generate_alignment_tif.py can join instead of rounding a centroid.")
    else:
        print("JOIN: no id -> label rule is consistent. filt_neurons.id does not name a")
        print("  cellmask label; the geometric centroid lookup stays the only link.")

    # The indexing question is independent of the join. Read it off the joined
    # rule when there is one, else off the nearest-centroid match, which assumes
    # no link at all.
    if joined:
        meds = np.array([r["rules"][best_name]["median"] for r in results])
        source = f"via the '{best_name}' join"
    else:
        meds = np.array([r["nearest"]["median_resid"] for r in results])
        source = "via nearest-centroid matching"

    if meds.size:
        med = np.median(meds, axis=0)
        print(f"\nINDEXING ({source}): pos40x - (0-indexed centroid) = {fmt(med)} px")
        if np.allclose(med, 1.0, atol=0.15):
            print("  pos40x is 1-INDEXED (MATLAB). The canvas term in the centroid formula")
            print("  is one pixel too small: it should be (pos*2 - min_x_offset), not")
            print("  (pos*2 - (min_x_offset - 1)). Every mapped cell currently sits one")
            print("  full-res px down-and-right. Fix all four copies together -- see")
            print("  reference_centroid_mapping_formula.md.")
        elif np.allclose(med, 0.0, atol=0.15):
            print("  pos40x is 0-INDEXED. The existing formula is correct as written;")
            print("  no change to the four copies.")
        else:
            print("  Neither 0 nor 1. Do not change the formula on this -- the centroid")
            print("  convention or the column order is not what either reading assumes.")

    orders = {r["rules"][best_name]["order"] for r in results
              if joined and r["rules"][best_name].get("coverage")}
    if orders and orders != {"xy"}:
        print(f"\nCOLUMN ORDER: best fit used {orders}. 'yx' means pos40x is (row, col),")
        print("  not (x, y) -- the regression in stitch_subslices.py pairs the columns")
        print("  the other way round.")


def main():
    p = argparse.ArgumentParser(
        description="Read-only probe of the filt_neurons.id -> cellmask label link "
                    "and the pos40x indexing convention",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("-n", type=int, default=5, help="FOVs to check (default 5)")
    p.add_argument("--fov", action="append", default=None, help="check this FOV (repeatable)")
    p.add_argument("--min-cells", type=int, default=20, help="skip FOVs with fewer rows")
    p.add_argument("--json", default=None, help="write the full report here")
    args = p.parse_args()

    print("=" * 40)
    print("CELL ID -> CELLMASK LABEL LINK")
    print("=" * 40)
    print(f"filt_neurons: {FILT_NEURONS_PATH}")
    print(f"hyb:          {HYB_ROOT}\n")

    fn = load_filt_neurons(FILT_NEURONS_PATH)
    for field in ("id", "pos40x", "fov"):
        if field not in fn:
            print(f"filt_neurons has no '{field}' field — cannot run.")
            return 1

    fov_of_row = np.asarray(fn["fov"])
    pos40x = np.asarray(fn["pos40x"])
    ids = np.asarray(fn["id"]).astype(np.float64).ravel()
    if pos40x.ndim != 2 or pos40x.shape[1] != 2:
        print(f"pos40x has shape {pos40x.shape}, expected (N, 2) — cannot run.")
        return 1

    hyb_root = Path(HYB_ROOT)
    counts = Counter(fov_of_row.tolist())
    if args.fov:
        wanted = [f for f in args.fov]
    else:
        wanted = [name for name, c in counts.most_common() if c >= args.min_cells]

    results, checked = [], 0
    for name in wanted:
        if checked >= args.n and not args.fov:
            break
        rows = np.where(fov_of_row == name)[0]
        if rows.size == 0:
            print(f"{name}: no rows in filt_neurons, skipped\n")
            continue
        r = check_fov(name, hyb_root / name, rows, pos40x, ids)
        if r is None:
            print(f"{name}: no readable cellmask or too few usable rows, skipped\n")
            continue
        report(r)
        print()
        results.append(r)
        checked += 1

    if not results:
        print("No FOV could be evaluated.")
        return 1

    verdict(results)

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"filt_neurons": str(FILT_NEURONS_PATH), "hyb_root": str(HYB_ROOT),
             "tol_px": TOL_PX, "fovs": results}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
