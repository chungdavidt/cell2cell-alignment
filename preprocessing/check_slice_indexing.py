#!/usr/bin/env python3
"""
Is the slice numbering right? Everything filt_neurons.mat can say about it.

Read-only, and it opens nothing but `filt_neurons.mat` -- no images, no FOV
TIFFs, no cellmasks. Answers a question this pipeline cannot answer for itself:
`slice` is assigned by the lab's upstream BARseq processing, and every script
here only ever compares its VALUE (`slice_ids == slice_id`). Nothing indexes an
array with it, so a MATLAB/Python off-by-one is impossible; what is possible is
that the numbering does not follow the physical section order.

Four checks, each printed with the evidence rather than a verdict:

1. THE FIELD ITSELF
   Range, NaN fraction, which numbers in 1..max are absent, how many cells
   carry each. A gap in the sequence is worth knowing about before it shows up
   as a missing tile in a montage.

2. slice VS orig_slice
   filt_neurons carries BOTH. If they are identical the numbering was never
   remapped and there is nothing more to say. If they differ, the remapping is
   printed in full -- that is the single most likely place for the numbering to
   have been changed out from under the images.

3. THE 8-GROUP PARTITION
   `uniq_slice` / `slice_boundaries` partition BY95's 62 sections into 8 sets
   (the section-to-run grouping). Whether slice numbers run consecutively
   within a group, or interleave across groups, says how the numbering was
   assigned -- per run or globally.

4. STAGE LAYOUT (the real test)
   Sections are cut in order and laid onto slides in order, so consecutive
   section NUMBERS should be neighbours in stage coordinates. For each slice
   this prints the centroid and bounding box of its cells' `pos`, the step to
   the next slice number, and then flags any slice whose numeric neighbour is
   NOT its nearest neighbour in space. A correct numbering makes a tidy raster;
   a scrambled one puts slice 30 next to slice 7.

   Read the flags as questions, not errors. A section can legitimately sit far
   from its predecessor at a slide break or a new run, which is exactly what
   check 3 is there to cross-reference. What should NOT happen is many
   scattered breaks with no group boundary behind them.

Also reports FOVs shared between two slices. Two sections in one FOV is
possible when they are mounted close together; it is also what a mis-assigned
section looks like, so the count is printed either way.

Usage:
    python check_slice_indexing.py
    python check_slice_indexing.py --data-root "<path to the dataset folder>"
    python check_slice_indexing.py --max-report 20   # rows in the long tables
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

from utilities.mat_io import load_mat, load_filt_neurons


def _flat(value):
    """A 1-D float array from a filt_neurons field, or None if it is absent."""
    if value is None:
        return None
    arr = np.asarray(value).ravel()
    if arr.dtype.kind in "fc":
        return arr.astype(float)
    try:
        return arr.astype(float)
    except (TypeError, ValueError):
        return None


def report_field(slice_ids):
    """Check 1: what the slice column contains."""
    print("=" * 70)
    print("1. THE slice FIELD")
    print("=" * 70)

    finite = ~np.isnan(slice_ids)
    values = slice_ids[finite]
    ints = values.astype(int)

    non_integer = int(np.count_nonzero(values != ints))
    present = np.unique(ints)

    print(f"  rows total          {slice_ids.size}")
    print(f"  NaN                 {np.count_nonzero(~finite)} "
          f"({100 * np.count_nonzero(~finite) / slice_ids.size:.1f}%) "
          f"-- cells assigned to no section")
    print(f"  distinct sections   {present.size}")
    print(f"  range               {present.min()} .. {present.max()}")
    if non_integer:
        print(f"  NON-INTEGER VALUES  {non_integer} -- a section number should "
              f"be a whole number")

    expected = set(range(int(present.min()), int(present.max()) + 1))
    missing = sorted(expected - set(present.tolist()))
    if missing:
        print(f"  MISSING from the run: {missing}")
        print(f"    Nothing here says whether those sections were never cut, "
              f"never imaged, or\n    dropped upstream -- only that no cell "
              f"claims them.")
    else:
        print(f"  no gaps: every number from {present.min()} to {present.max()} "
              f"has cells")
    print()
    return ints, finite


def report_orig_slice(fn, slice_ids, finite, max_report):
    """Check 2: slice against orig_slice."""
    print("=" * 70)
    print("2. slice VS orig_slice")
    print("=" * 70)

    orig = _flat(fn.get("orig_slice"))
    if orig is None:
        print("  orig_slice is absent from this filt_neurons -- nothing to "
              "compare.\n")
        return
    if orig.size != slice_ids.size:
        print(f"  orig_slice has {orig.size} rows against slice's "
              f"{slice_ids.size} -- not row-aligned, cannot compare.\n")
        return

    both = finite & ~np.isnan(orig)
    a = slice_ids[both].astype(int)
    b = orig[both].astype(int)

    if np.array_equal(a, b):
        print(f"  IDENTICAL on all {both.sum()} rows where both are set.")
        print("  The numbering was never remapped: slice IS orig_slice.\n")
        return

    differ = int(np.count_nonzero(a != b))
    print(f"  THEY DIFFER on {differ} of {both.sum()} rows "
          f"({100 * differ / both.sum():.1f}%).")
    print("  So the section numbering WAS remapped upstream. The mapping:\n")

    pairs = {}
    for s, o in zip(a, b):
        pairs.setdefault(int(s), {}).setdefault(int(o), 0)
        pairs[int(s)][int(o)] += 1

    ambiguous = [s for s, o in pairs.items() if len(o) > 1]
    print(f"    {'slice':>6}  {'orig_slice':>28}")
    for s in sorted(pairs)[:max_report]:
        detail = ", ".join(f"{o} ({n} cells)" for o, n in
                           sorted(pairs[s].items(), key=lambda kv: -kv[1]))
        mark = "  <- SPLIT" if len(pairs[s]) > 1 else ""
        print(f"    {s:>6}  {detail:>28}{mark}")
    if len(pairs) > max_report:
        print(f"    ... {len(pairs) - max_report} more (--max-report to widen)")

    if ambiguous:
        print(f"\n  {len(ambiguous)} slice number(s) draw from more than one "
              f"orig_slice: {sorted(ambiguous)[:20]}")
        print("  That is a merge, not a renumber -- worth understanding before "
              "trusting either field.")
    print()


def report_groups(data_root, slice_ids, finite, max_report):
    """Check 3: the uniq_slice / slice_boundaries partition."""
    print("=" * 70)
    print("3. THE uniq_slice / slice_boundaries PARTITION")
    print("=" * 70)

    raw = load_mat(data_root / "filt_neurons.mat")
    groups = raw.get("uniq_slice")
    if groups is None:
        print("  uniq_slice is absent -- nothing to partition.\n")
        return None

    members = {}
    try:
        arr = np.asarray(groups, dtype=object).ravel()
        for i, entry in enumerate(arr):
            nums = np.asarray(entry).ravel()
            nums = nums[~np.isnan(nums.astype(float))] if nums.dtype.kind == "f" else nums
            members[i + 1] = sorted({int(v) for v in np.asarray(nums).ravel()})
    except Exception as exc:                                   # noqa: BLE001
        print(f"  uniq_slice could not be read as a list of section sets "
              f"({exc.__class__.__name__}: {exc}).")
        print(f"  Raw type {type(groups).__name__}, "
              f"shape {getattr(np.asarray(groups, dtype=object), 'shape', '?')}.\n")
        return None

    print(f"  {len(members)} group(s) over "
          f"{sum(len(v) for v in members.values())} section slots\n")
    contiguous = 0
    for gid in sorted(members)[:max_report]:
        nums = members[gid]
        if not nums:
            print(f"    group {gid:>2}: empty")
            continue
        runs = nums == list(range(nums[0], nums[-1] + 1))
        contiguous += bool(runs)
        span = f"{nums[0]}-{nums[-1]}" if runs else ", ".join(map(str, nums[:12]))
        print(f"    group {gid:>2}: {len(nums):>3} sections  "
              f"{'consecutive ' if runs else 'SCATTERED   '}{span}"
              f"{'' if runs or len(nums) <= 12 else ' ...'}")
    if len(members) > max_report:
        print(f"    ... {len(members) - max_report} more")

    print()
    if contiguous == len(members):
        print("  Every group is a consecutive block of section numbers, so the "
              "numbering runs\n  with the groups rather than across them -- "
              "what you want if the groups are runs\n  and the sections were "
              "numbered in cutting order.")
    else:
        print(f"  {len(members) - contiguous} group(s) hold NON-consecutive "
              f"section numbers. Either the\n  numbering is global and the "
              f"groups are not runs, or the two disagree.")
    print()
    return members


def report_layout(slice_ids, finite, pos, fov, groups, max_report):
    """Check 4: stage layout, the one that can actually catch a scramble."""
    print("=" * 70)
    print("4. STAGE LAYOUT")
    print("=" * 70)

    if pos is None or pos.ndim != 2 or pos.shape[1] < 2:
        print("  pos is absent or not N x 2 -- cannot place sections.\n")
        return

    ints = slice_ids.astype(int)
    order = sorted(set(ints[finite].tolist()))

    centroids = {}
    print(f"    {'slice':>6} {'cells':>7} {'FOVs':>5}  "
          f"{'centroid x':>11} {'centroid y':>11}  {'step to next':>12}")
    rows = []
    for s in order:
        sel = finite & (ints == s)
        p = pos[sel]
        cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
        centroids[s] = (cx, cy)
        n_fov = len({str(f) for f in fov[sel]}) if fov is not None else -1
        rows.append((s, int(sel.sum()), n_fov, cx, cy))

    steps = {}
    for i, s in enumerate(order[:-1]):
        nxt = order[i + 1]
        steps[s] = float(np.hypot(centroids[nxt][0] - centroids[s][0],
                                  centroids[nxt][1] - centroids[s][1]))

    for s, n, n_fov, cx, cy in rows[:max_report]:
        step = f"{steps[s]:>12.0f}" if s in steps else f"{'-':>12}"
        print(f"    {s:>6} {n:>7} {n_fov if n_fov >= 0 else '?':>5}  "
              f"{cx:>11.0f} {cy:>11.0f}  {step}")
    if len(rows) > max_report:
        print(f"    ... {len(rows) - max_report} more (--max-report to widen)")

    if len(order) < 3:
        print("\n  Too few sections to judge an ordering.\n")
        return

    # The test: is each section's numeric neighbour also its nearest neighbour?
    print()
    print("  Is each section's numeric neighbour its NEAREST neighbour in "
          "stage space?")
    coords = np.array([centroids[s] for s in order])
    flagged = []
    for i, s in enumerate(order):
        d = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
        d[i] = np.inf
        nearest = order[int(np.argmin(d))]
        neighbours = {order[i - 1] if i else None,
                      order[i + 1] if i + 1 < len(order) else None}
        if nearest not in neighbours:
            flagged.append((s, nearest, float(d.min()),
                            min(steps.get(s, np.inf),
                                steps.get(order[i - 1], np.inf) if i else np.inf)))

    if not flagged:
        print("    YES for every section. The numbering follows the physical "
              "layout.")
    else:
        print(f"    NO for {len(flagged)} of {len(order)} sections:\n")
        print(f"    {'slice':>6}  {'nearest':>8}  {'that far':>9}  "
              f"{'numeric nbr':>12}")
        for s, nearest, dist, own in flagged[:max_report]:
            print(f"    {s:>6}  {nearest:>8}  {dist:>9.0f}  "
                  f"{own if np.isfinite(own) else float('nan'):>12.0f}")
        if len(flagged) > max_report:
            print(f"    ... {len(flagged) - max_report} more")
        if groups:
            at_break = set()
            for nums in groups.values():
                if nums:
                    at_break |= {nums[0], nums[-1]}
            explained = [s for s, *_ in flagged if s in at_break]
            print(f"\n    {len(explained)} of these sit at a "
                  f"uniq_slice group boundary, where a jump is expected.")
            print(f"    {len(flagged) - len(explained)} do not, and those are "
                  f"the ones to look at.")
        print("\n    A section mounted out of order, or two sections whose "
              "cells were swapped,\n    looks exactly like this. So does a "
              "slide break. The montage settles it.")
    print()


def report_shared_fovs(slice_ids, finite, fov, max_report):
    """FOVs claimed by more than one section."""
    if fov is None:
        return
    print("=" * 70)
    print("5. FOVs SHARED BETWEEN SECTIONS")
    print("=" * 70)

    ints = slice_ids.astype(int)
    owners = {}
    for name, s in zip(fov[finite], ints[finite]):
        owners.setdefault(str(name), set()).add(int(s))

    shared = {f: sorted(s) for f, s in owners.items() if len(s) > 1}
    if not shared:
        print(f"  None. Each of the {len(owners)} FOVs belongs to exactly one "
              f"section.\n")
        return

    print(f"  {len(shared)} of {len(owners)} FOVs carry cells from more than "
          f"one section.")
    print("  Legitimate when two sections are mounted close enough to share a "
          "field; also\n  what a mis-assigned section looks like.\n")
    for f, ss in sorted(shared.items())[:max_report]:
        print(f"    {f:<28} {ss}")
    if len(shared) > max_report:
        print(f"    ... {len(shared) - max_report} more")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Check whether the slice numbering is trustworthy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data-root', default=None,
                        help='Dataset folder holding filt_neurons.mat '
                             '(default: DATA_ROOT from local_config.py)')
    parser.add_argument('--max-report', type=int, default=70,
                        help='Rows to print in the long tables (default 70)')
    args = parser.parse_args()

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        from preprocessing_config import DATA_ROOT
        data_root = Path(DATA_ROOT)

    path = data_root / "filt_neurons.mat"
    if not path.exists():
        raise SystemExit(f"filt_neurons.mat not found: {path}")

    print(f"filt_neurons: {path}\n")
    fn = load_filt_neurons(path)

    slice_ids = _flat(fn.get("slice"))
    if slice_ids is None:
        raise SystemExit("filt_neurons has no readable `slice` field.")

    pos = np.asarray(fn["pos"]) if fn.get("pos") is not None else None
    fov = fn.get("fov")
    fov = None if fov is None else np.asarray(fov).ravel()

    _, finite = report_field(slice_ids)
    report_orig_slice(fn, slice_ids, finite, args.max_report)
    groups = report_groups(data_root, slice_ids, finite, args.max_report)
    report_layout(slice_ids, finite, pos, fov, groups, args.max_report)
    report_shared_fovs(slice_ids, finite, fov, args.max_report)

    print("=" * 70)
    print("Nothing here can prove the numbering matches the tissue -- only the")
    print("images can. What it can do is say whether the numbering is")
    print("self-consistent, and where to look if it is not.")
    print("=" * 70)


if __name__ == '__main__':
    main()
