#!/usr/bin/env python3
"""
Bar chart of the per-slice rolony census written by count_rolonies_per_slice.py.

Reads that script's per-slice CSV -- one row per slice -- and draws one bar per
slice. Default column is `rolonies`, the total marker rolonies in the slice.

The CSV is a per-slice summary, so this is a bar per slice, not a distribution.
For the distribution of rolony counts ACROSS CELLS, re-run
count_rolonies_per_slice.py with --cells-csv and histogram its `rolonies` column.

--exclude-slices drops those sections' rows before anything is drawn or summed,
for a section whose data is bad. It is local to this script: nothing else reads
it, no config name holds it, and the CSV still contains the section. A number
with no row in the CSV is an error, so a typo cannot write a file whose name
claims an exclusion that did not happen.

Usage:
    python preprocessing/plot_rolonies_per_slice.py
    python preprocessing/plot_rolonies_per_slice.py <path to the per-slice CSV>
    python preprocessing/plot_rolonies_per_slice.py --column marker_cells
    python preprocessing/plot_rolonies_per_slice.py --exclude-slices 58
    python preprocessing/plot_rolonies_per_slice.py --out figure.png

Columns available: cells, qc, marker_cells, rolonies, max, median_pos, mean_pos,
and one ge{n} per cumulative cutoff.
"""

import argparse
import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAR_COLOR = "#CC3333"      # same red the other per-slice figures use

LABELS = {
    "cells": "Cells",
    "qc": "QC-passing cells",
    "marker_cells": "mScarlet+ cells",
    "rolonies": "mScarlet rolonies",
    "max": "Max rolonies in one cell",
    "median_pos": "Median rolonies per mScarlet+ cell",
    "mean_pos": "Mean rolonies per mScarlet+ cell",
    "rolonies_per_qc_cell": "Mean rolonies per QC-passing cell",
}

# Not in the CSV; computed from columns that are. `mean_pos` already divides by
# the marker+ cells, so the open question is only the other denominator -- every
# QC-passing cell, zeros included.
DERIVED = {
    "rolonies_per_qc_cell": lambda r: float(r["rolonies"]) / float(r["qc"])
                                      if float(r["qc"]) else 0.0,
}


def default_csv():
    """The per-slice CSV count_rolonies_per_slice.py writes, if there is one."""
    try:
        from analysis_paths import analysis_subdir

        d = analysis_subdir("preprocessing", create=False)
    except Exception:
        d = None
    if d is None:
        d = Path.cwd()
    found = sorted(Path(d).glob("rolonies_per_slice_*_qc*.csv"))
    found = [f for f in found if not f.name.endswith("_cells.csv")]
    if not found:
        raise FileNotFoundError(
            f"No per-slice rolony CSV in {d}.\n"
            f"Run: python preprocessing/count_rolonies_per_slice.py"
        )
    return found[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", nargs="?", help="per-slice CSV (default: the newest "
                                                "one beside the other preprocessing output)")
    ap.add_argument("--column", "-c", default="rolonies", help="column to plot (default: rolonies)")
    ap.add_argument("--exclude-slices", type=int, nargs="+", metavar="N",
                    help="drop these sections' rows before plotting; "
                         "the numbers land in the output filename")
    ap.add_argument("--out", help="output PNG (default: beside the CSV)")
    ap.add_argument("--no-labels", action="store_true",
                    help="omit the value printed above each bar")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    path = Path(args.csv_path) if args.csv_path else default_csv()
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows")
    if args.column not in rows[0] and args.column not in DERIVED:
        raise ValueError(f"No column {args.column!r} in {path.name}. "
                         f"Columns: {', '.join(rows[0])}. "
                         f"Derived: {', '.join(DERIVED)}")

    excluded = sorted(set(args.exclude_slices or []))
    if excluded:
        present = {int(r["slice"]) for r in rows}
        missing = [s for s in excluded if s not in present]
        if missing:
            raise SystemExit(f"--exclude-slices {missing}: no row in {path.name}. "
                             f"present: {sorted(present)}")
        rows = [r for r in rows if int(r["slice"]) not in excluded]

    rows.sort(key=lambda r: int(r["slice"]))
    slices = [int(r["slice"]) for r in rows]
    if args.column in DERIVED:
        values = [DERIVED[args.column](r) for r in rows]
    else:
        values = [float(r[args.column]) for r in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(slices)), values, color=BAR_COLOR, width=0.8)

    ax.set_xticks(range(len(slices)))
    ax.set_xticklabels([str(s) for s in slices], fontsize=7, rotation=90)
    ax.set_xlabel("Slice", fontsize=12)
    ax.set_ylabel(LABELS.get(args.column, args.column), fontsize=12)
    ax.set_title(f"{LABELS.get(args.column, args.column)} per slice  ({path.name})",
                 fontsize=13)
    if not args.no_labels:
        # Rotated and small: 62 slices side by side leave no room for horizontal
        # text. Headroom is opened below so the tallest bar's label still fits.
        # One format for the whole column, not per bar: a mean that happens to
        # land on 5.0 should still read "5.0" beside its neighbour's "4.7".
        fmt = "{:,.0f}" if all(v == int(v) for v in values) else "{:,.1f}"
        for i, v in enumerate(values):
            ax.text(i, v, fmt.format(v),
                    ha="center", va="bottom", rotation=90, fontsize=6)
        ax.set_ylim(0, max(values) * 1.30)

    ax.set_xlim(-0.7, len(slices) - 0.3)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    stem = f"{path.stem}_{args.column}"
    if excluded:
        stem += "_ex" + "_".join(str(s) for s in excluded)
    out = Path(args.out) if args.out else path.with_name(f"{stem}.png")
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)

    total = sum(values)
    if excluded:
        print(f"excluded slice{'s' if len(excluded) > 1 else ''} "
              f"{', '.join(str(s) for s in excluded)}")
    print(f"{len(slices)} slices, {args.column} total {total:,.0f}, "
          f"max {max(values):,.0f} (slice {slices[values.index(max(values))]})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
