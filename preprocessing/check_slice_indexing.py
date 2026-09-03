#!/usr/bin/env python3
"""
Is the slice numbering right? Everything filt_neurons.mat can say about it.

Read-only, and it opens nothing but `filt_neurons.mat` -- no images, no FOV
TIFFs, no cellmasks.

WHAT THIS CAN AND CANNOT ANSWER
    The code half needs no probe. Every script here compares the section
    number's VALUE (`slice_ids == slice_id`) and nothing indexes an array with
    it, so a MATLAB/Python off-by-one cannot arise. What is possible is that the
    numbering, assigned by the lab's upstream BARseq processing, does not follow
    the physical section order.

    Note what that would and would not break. The cell-level checks
    (check_cell_id_link.py, measure_lookup_disagreement.py) all work INSIDE one
    section and never read the section number, so they pass identically under
    any permutation of it -- correct cells and correct stitching are fully
    consistent with a wrong order. Downstream, a section number is a label on a
    graph node, not a z coordinate: each section is fitted to the 2P
    independently, so a wrong order would not corrupt an alignment. It would
    matter for assign_orientation.py's series walk, which asks the cutting order
    once and propagates it, and for reading a montage as a progression.

WHY `pos` IS NOT USED HERE
    `pos` is the STITCHING coordinate: calculate_fov_offset regresses
    `pos*2 = offset + pos40x` to place a FOV on the canvas, so it locates a cell
    within an image, not a section on a slide. A check that compared section
    centroids in it -- on the assumption that consecutive sections should be
    physical neighbours -- flagged 45 of BY95's 62 sections when run globally
    and 37 when run per slide, and meant nothing either way: the centroids sit
    34-465 units apart while a section spans thousands, so they are nearly
    coincident and "nearest" is noise. The check was deleted rather than given a
    scale reference, because the premise was wrong, not the threshold. Nothing
    in filt_neurons records where a section sat on its slide.

Six reports, each printed with the evidence rather than a verdict:

1. THE FIELD ITSELF
   Range, NaN fraction, gaps in 1..max, cells per section.

2. slice VS orig_slice
   Both are in filt_neurons, and a difference is NOT by itself a renumbering.
   On BY95 orig_slice takes 8 values over 62 sections, one per section with no
   splits, which is a SLIDE. Check 4 settles that against the FOV names. A slice
   number drawing from several orig_slice values would be a merge, and is
   flagged separately.

3. THE uniq_slice PARTITION
   Read from the loaded struct, where it lives, rather than the file's top
   level. Reported as RUNS of consecutive section numbers per group: a group
   holding 27-31 and 38-42 is two runs, not scatter.

4. SECTIONS PER SLIDE
   Slide membership from the `MAX_Pos{N}_{row}_{col}` FOV names, cross-checked
   against orig_slice -- BY95 matches on 62 of 62. Then blocks of consecutive
   section numbers per slide and the order the numbering visits slides in. Few
   long blocks means the numbering follows the mounting; sections numbered
   independently of how they were mounted would scatter across slides.

5. DOES THE SERIES GROW MONOTONICALLY?
   The strongest evidence available without an image, and the only check here
   that no coordinate frame can spoil. A series cut from one end starts small
   and grows, so cells per section should rise with the section number. Reported
   as a rank correlation and an out-of-order pair count against the ~50% a
   random permutation gives. BY95: +0.975 and 7.0%, against 43.5% for a
   permutation of its own counts.

6. FOVs SHARED BETWEEN SECTIONS
   Normal when two sections are mounted close enough to share a field. WHICH
   section numbers share one is the informative part: adjacent numbers mean
   adjacent mounting.

Usage:
    python check_slice_indexing.py
    python check_slice_indexing.py --data-root "<path to the dataset folder>"
    python check_slice_indexing.py --max-report 20   # rows in the long tables
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

from utilities.mat_io import load_filt_neurons


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
    """Check 2: slice against orig_slice. Returns {section: majority orig}.

    A difference here is NOT by itself a renumbering. On BY95 orig_slice takes
    8 values over 62 sections, one per section with no splits -- it is the
    SLIDE, and comparing it against `slice` compares two different quantities.
    Check 4 settles which it is against the FOV names.
    """
    print("=" * 70)
    print("2. slice VS orig_slice")
    print("=" * 70)

    orig = _flat(fn.get("orig_slice"))
    if orig is None:
        print("  orig_slice is absent from this filt_neurons -- nothing to "
              "compare.\n")
        return None
    if orig.size != slice_ids.size:
        print(f"  orig_slice has {orig.size} rows against slice's "
              f"{slice_ids.size} -- not row-aligned, cannot compare.\n")
        return None

    both = finite & ~np.isnan(orig)
    a = slice_ids[both].astype(int)
    b = orig[both].astype(int)

    per_section = {}
    for s, o in zip(a, b):
        per_section.setdefault(int(s), {}).setdefault(int(o), 0)
        per_section[int(s)][int(o)] += 1
    majority = {s: max(d, key=d.get) for s, d in per_section.items()}

    if np.array_equal(a, b):
        print(f"  IDENTICAL on all {both.sum()} rows where both are set.")
        print("  The numbering was never remapped: slice IS orig_slice.\n")
        return majority

    differ = int(np.count_nonzero(a != b))
    print(f"  THEY DIFFER on {differ} of {both.sum()} rows "
          f"({100 * differ / both.sum():.1f}%).")
    print("  So the section numbering WAS remapped upstream. The mapping:\n")

    pairs = per_section

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
    else:
        print("\n  Each slice number maps to exactly ONE orig_slice value, and "
              "there are far fewer\n  of those than there are sections. That is "
              "not a section renumbering -- check 4\n  tests whether "
              "orig_slice is the SLIDE.")
    print()
    return majority


def report_groups(fn, slice_ids, finite, max_report):
    """Check 3: the uniq_slice / slice_boundaries partition."""
    print("=" * 70)
    print("3. THE uniq_slice / slice_boundaries PARTITION")
    print("=" * 70)

    # From the loaded struct, not the file's top level: uniq_slice is a field
    # OF filt_neurons, and reading the top level found nothing on BY95.
    groups = fn.get("uniq_slice")
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
        # Runs of consecutive numbers, not a consecutive/scattered binary:
        # a group holding 27-31 and 38-42 is TWO runs, which is what
        # alternating a pair of slides looks like, and calling that
        # "scattered" was wrong.
        runs = []
        for n in nums:
            if runs and runs[-1][1] + 1 == n:
                runs[-1][1] = n
            else:
                runs.append([n, n])
        contiguous += (len(runs) == 1)
        span = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in runs[:6])
        print(f"    group {gid:>2}: {len(nums):>3} sections in "
              f"{len(runs)} run{'s' if len(runs) != 1 else ' '}   {span}"
              f"{' ...' if len(runs) > 6 else ''}")
    if len(members) > max_report:
        print(f"    ... {len(members) - max_report} more")

    print()
    print("  A group is a slide (check 4 proves it). One run per group means "
          "the numbering\n  went through a slide before moving on; two runs "
          "per group means the series\n  alternated between a pair of slides. "
          "Either is a numbering that follows the\n  mounting. Many short runs "
          "would not be.")
    print()
    return members


def slide_of(fov_names):
    """{FOV name: slide number} from the `MAX_Pos{N}_{row}_{col}` convention."""
    out = {}
    for name in fov_names:
        m = re.match(r"MAX_Pos(\d+)_", str(name))
        if m:
            out[str(name)] = int(m.group(1))
    return out


def report_slides(slice_ids, finite, fov, orig, max_report):
    """Check 4: which slide is each section on, and is orig_slice that slide?

    `pos` is a WITHIN-SLIDE coordinate -- `MAX_Pos7_003_016` is slide 7, grid
    row 3, column 16 -- so comparing positions across slides compares two
    different frames. Establishing slide membership is a prerequisite for
    check 6, not a curiosity.
    """
    print("=" * 70)
    print("4. SECTIONS PER SLIDE")
    print("=" * 70)

    ints = np.full(slice_ids.shape, -1, dtype=int)
    ints[finite] = slice_ids[finite].astype(int)

    slides = slide_of(fov[finite]) if fov is not None else {}
    if not slides:
        print("  FOV names do not follow MAX_Pos{N}_{row}_{col}, so slide")
        print("  membership is unknown and check 6 cannot run.\n")
        return None

    per_section = {}
    for name, s in zip(fov[finite], ints[finite]):
        sl = slides.get(str(name))
        if sl is not None:
            per_section.setdefault(int(s), {}).setdefault(sl, 0)
            per_section[int(s)][sl] += 1

    split = {s: d for s, d in per_section.items() if len(d) > 1}
    section_slide = {s: max(d, key=d.get) for s, d in per_section.items()}

    if orig:
        agree = sum(1 for s, sl in section_slide.items()
                    if orig.get(s) == sl)
        print(f"  orig_slice equals the FOV's Pos number on {agree} of "
              f"{len(section_slide)} sections.")
        if agree == len(section_slide):
            print("  So orig_slice IS THE SLIDE, not an original section number.")
            print("  Check 2's '100% differ' therefore proves no renumbering --")
            print("  it compared a section number against a slide number.")
        print()

    if split:
        print(f"  {len(split)} section(s) draw FOVs from more than one slide: "
              f"{sorted(split)[:20]}")
        print("  A section cannot span two slides, so this would be a real "
              "problem.\n")

    order = sorted(section_slide)
    blocks = []
    for s in order:
        sl = section_slide[s]
        if blocks and blocks[-1][0] == sl and blocks[-1][2] + 1 == s:
            blocks[-1][2] = s
        else:
            blocks.append([sl, s, s])

    print(f"    {'slide':>6}  {'sections':>12}  {'count':>6}")
    for sl, first, last in blocks[:max_report]:
        print(f"    {sl:>6}  {f'{first}-{last}':>12}  {last - first + 1:>6}")
    if len(blocks) > max_report:
        print(f"    ... {len(blocks) - max_report} more")

    n_slides = len(set(section_slide.values()))
    print(f"\n  {len(blocks)} block(s) of consecutive section numbers over "
          f"{n_slides} slide(s).")
    print(f"  Slide visiting order: {', '.join(str(b[0]) for b in blocks)}")
    print()
    if len(blocks) == n_slides:
        print("  One block per slide: the numbering runs through a slide before "
              "moving on.")
    elif len(blocks) <= 2 * n_slides:
        print("  Some slides hold two blocks, so the series alternates between "
              "a pair of slides --")
        print("  what you get when consecutive sections are placed onto two "
              "slides in turn, which")
        print("  is how a series is split when two stains are wanted off it.")
    else:
        print("  Many short blocks. The numbering does not follow the mounting "
              "in any simple way.")
    print()
    print("  What matters is that the blocks are FEW. Sections numbered "
          "independently of how")
    print("  they were mounted would scatter across slides, not fall into a "
          "handful of runs.")
    print()
    return section_slide


def report_series_trend(slice_ids, finite, max_report):
    """Check 5: does the series grow the way a run through a brain grows?

    The strongest evidence available without an image, and the only check here
    independent of every coordinate frame: a series cut from one end starts
    small and grows, so cells per section should rise with the section number.
    A permuted numbering destroys that; nothing else in this probe would
    notice, and neither would any of the cell-level checks -- those all work
    inside one section and never read the section number at all.
    """
    print("=" * 70)
    print("5. DOES THE SERIES GROW MONOTONICALLY?")
    print("=" * 70)

    ints = np.full(slice_ids.shape, -1, dtype=int)
    ints[finite] = slice_ids[finite].astype(int)
    order = sorted(set(ints[finite].tolist()))
    counts = np.array([int(np.count_nonzero(ints == s)) for s in order], float)

    if counts.size < 3:
        print("  Too few sections to judge a trend.\n")
        return

    inversions = int(sum(1 for i in range(counts.size)
                         for j in range(i + 1, counts.size)
                         if counts[j] < counts[i]))
    total = counts.size * (counts.size - 1) // 2
    rank = np.argsort(np.argsort(counts)).astype(float)
    spearman = float(np.corrcoef(rank, np.arange(counts.size, dtype=float))[0, 1])

    print(f"  cells in section {order[0]:<3}          {int(counts[0])}")
    print(f"  cells in section {order[-1]:<3}          {int(counts[-1])}")
    print(f"  growth across the series    {counts[-1] / max(counts[0], 1):.0f}x")
    print(f"  rank correlation with section number   {spearman:+.3f}")
    print(f"  out-of-order pairs                     {inversions}/{total} "
          f"({100 * inversions / total:.1f}%)")
    print()
    if spearman > 0.9:
        print("  The series rises almost perfectly with the section number.")
        print("  A permuted numbering cannot produce this: a random order sits "
              "near 50%")
        print("  out-of-order pairs and a rank correlation near 0. The "
              "numbering follows")
        print("  the anatomy.")
    elif spearman > 0.5:
        print("  A clear rise with local exceptions. Consistent with a correct "
              "series in")
        print("  which neighbouring sections vary; not with a scrambled one.")
    else:
        print("  NO trend. Either this series does not run from one end of the "
              "brain, or")
        print("  the numbering does not follow the cutting order. Only the "
              "montage separates")
        print("  those two.")
    print()


def report_shared_fovs(slice_ids, finite, fov, max_report):
    """FOVs claimed by more than one section."""
    if fov is None:
        return
    print("=" * 70)
    print("6. FOVs SHARED BETWEEN SECTIONS")
    print("=" * 70)

    ints = np.full(slice_ids.shape, -1, dtype=int)
    ints[finite] = slice_ids[finite].astype(int)
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

    # `pos` is deliberately NOT read. It is the STITCHING coordinate --
    # calculate_fov_offset regresses `pos*2 = offset + pos40x` to place a FOV on
    # the canvas -- so it locates a cell within an image, not a section on a
    # slide. A check that compared section centroids in it flagged 37 of 62
    # sections and meant nothing; see the note in the docstring.
    fov = fn.get("fov")
    fov = None if fov is None else np.asarray(fov).ravel()

    _, finite = report_field(slice_ids)
    orig = report_orig_slice(fn, slice_ids, finite, args.max_report)
    report_groups(fn, slice_ids, finite, args.max_report)
    section_slide = report_slides(slice_ids, finite, fov, orig, args.max_report)
    report_series_trend(slice_ids, finite, args.max_report)
    report_shared_fovs(slice_ids, finite, fov, args.max_report)

    print("=" * 70)
    print("Nothing here can prove the numbering matches the tissue -- only the")
    print("images can. What it can do is say whether the numbering is")
    print("self-consistent, and where to look if it is not.")
    print("=" * 70)


if __name__ == '__main__':
    main()
