#!/usr/bin/env python3
"""
Per-slice marker rolony census. Reads filt_neurons.mat, prints a table, and writes
that table as a CSV (--no-csv to skip; --cells-csv adds one row per marker+ cell).

For every slice in filt_neurons.mat: how many cells carry the marker, how many
rolonies they carry in total, and how those rolonies spread over cells. Counts
the WHOLE slice, not the subslice step 1 keeps -- for the subslice cut use
preview_subslices.py, whose QC+mSc column is this table's marker+ column
restricted to the largest connected FOV component.

What already covers part of this:
    preview_subslices.py      per-slice marker+ CELL counts, no rolony counts
    check_qc_metrics.py       rolony distribution pooled over the whole dataset
    export_subslice_cells.py  per-cell rolony counts, subslice FOVs only, post step 3

QC defaults to QC_MIN_READS / QC_MIN_GENES so the marker+ column matches what the
pipeline selects on. --min-reads / --min-genes override, and are required when
preprocessing_config will not import (no local_config, or a blank SCOPE) -- the
gates are per-brain and there is no safe default to fall back to.

ROLONY_FLOOR is the per-cell floor: a cell joins the marker+ count and the
rolony total only at >= that many rolonies. --min-rolonies overrides it, and 1
counts every cell carrying the marker. The ge{c} columns ignore the floor, so
what another floor would keep stays visible. A floor other than 1 lands in the
CSV name as _ge{N}.

Usage:
    python preprocessing/count_rolonies_per_slice.py
    python preprocessing/count_rolonies_per_slice.py <DATA_ROOT_or_filt_neurons.mat>
    python preprocessing/count_rolonies_per_slice.py --distribution
    python preprocessing/count_rolonies_per_slice.py --min-rolonies 1
    python preprocessing/count_rolonies_per_slice.py --marker gcamp
    python preprocessing/count_rolonies_per_slice.py --cutoffs 1 3 5 10
    python preprocessing/count_rolonies_per_slice.py --cells-csv
    python preprocessing/count_rolonies_per_slice.py --csv slices.csv
    python preprocessing/count_rolonies_per_slice.py --no-csv

The CSVs land in <ANALYSIS_ROOT>/preprocessing/ as
rolonies_per_slice_{marker}_qc{reads}_{genes}.csv and ..._cells.csv, or in the
working directory when ANALYSIS_ROOT is unset. --csv/--cells-csv take a path to
put them elsewhere.
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

import numpy as np

# Mirrors preprocessing_config; kept local so a bare filt_neurons.mat path works
# without local_config. Marker slots are index-only -- the panel's gene names for
# these columns are stale.
MARKERS = {
    "mscarlet": ("mScarlet", 113),
    "gcamp": ("GCaMP", 111),
}
DEFAULT_CUTOFFS = (1, 2, 3, 5)
# Cells below this carry no rolonies into the census. Matches step 4's
# ROLONY_FLOOR. BY95 number; re-pick per brain.
ROLONY_FLOOR = 5


def _config_gates():
    """(min_reads, min_genes, align_min_rolonies), or None with the import error."""
    try:
        import preprocessing_config as cfg
    except Exception as exc:
        return None, exc
    return (cfg.QC_MIN_READS, cfg.QC_MIN_GENES, cfg.ALIGN_MIN_ROLONIES), None


def _default_out(name):
    try:
        from analysis_paths import analysis_subdir

        d = analysis_subdir("preprocessing", create=True)
        if d is not None:
            return Path(d) / name
    except Exception:
        pass
    return Path.cwd() / name


def _load(path):
    p = Path(path)
    if p.is_dir():
        p = p / "filt_neurons.mat"
    from utilities.mat_io import load_filt_neurons

    return p, load_filt_neurons(p)


def _column(expmat, idx):
    import scipy.sparse as sp

    if idx >= expmat.shape[1]:
        return None
    col = expmat[:, idx]
    return np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()


def census(expmat, slice_ids, col, min_reads, min_genes, cutoffs,
           min_rolonies=ROLONY_FLOOR):
    """One row per slice, plus the QC mask and float slice ids."""
    reads = np.asarray(expmat.sum(axis=1)).ravel()
    genes = np.asarray((expmat > 0).sum(axis=1)).ravel()
    qc = (reads >= min_reads) & (genes >= min_genes)
    sl = np.asarray(slice_ids).ravel().astype(float)

    rows = []
    for s in np.unique(sl[~np.isnan(sl)]):
        in_slice = sl == s
        counts = col[in_slice & qc].astype(np.int64)
        pos = counts[counts >= min_rolonies]
        row = {
            "slice": int(s),
            "cells": int(in_slice.sum()),
            "qc": int((in_slice & qc).sum()),
            "marker_cells": int(pos.size),
            "rolonies": int(pos.sum()),
            "max": int(pos.max()) if pos.size else 0,
            "median_pos": float(np.median(pos)) if pos.size else 0.0,
            "mean_pos": float(pos.mean()) if pos.size else 0.0,
        }
        for c in cutoffs:   # on every QC cell, floor or no floor
            row[f"ge{c}"] = int((counts >= c).sum())
        rows.append(row)
    return rows, qc, sl


def print_table(rows, cutoffs, marker_name, min_rolonies=ROLONY_FLOOR):
    head = (f"{'slice':>6}{'cells':>8}{'QC':>8}{marker_name + '+':>10}"
            f"{'% of QC':>9}{'rolonies':>10}{'max':>6}{'median+':>9}{'mean+':>8}"
            + "".join(f"{'>=' + str(c):>8}" for c in cutoffs))
    print(f"\n=== {marker_name} rolonies per slice "
          f"(cells with >= {min_rolonies}) ===")
    print(head)
    print("-" * len(head))
    for r in rows:
        pct = 100.0 * r["marker_cells"] / r["qc"] if r["qc"] else 0.0
        print(f"{r['slice']:>6}{r['cells']:>8}{r['qc']:>8}{r['marker_cells']:>10}"
              f"{pct:>9.2f}{r['rolonies']:>10}{r['max']:>6}"
              f"{r['median_pos']:>9.0f}{r['mean_pos']:>8.1f}"
              + "".join(f"{r['ge' + str(c)]:>8}" for c in cutoffs))

    tot = {k: sum(r[k] for r in rows) for k in ("cells", "qc", "marker_cells", "rolonies")}
    pct = 100.0 * tot["marker_cells"] / tot["qc"] if tot["qc"] else 0.0
    print("-" * len(head))
    print(f"{'all':>6}{tot['cells']:>8}{tot['qc']:>8}{tot['marker_cells']:>10}"
          f"{pct:>9.2f}{tot['rolonies']:>10}"
          f"{max((r['max'] for r in rows), default=0):>6}{'':>9}{'':>8}"
          + "".join(f"{sum(r['ge' + str(c)] for r in rows):>8}" for c in cutoffs))


def print_distribution(rows_slices, col, qc, sl, max_count, marker_name):
    """Cells per exact rolony count, per slice. Counts above max_count fold into >N."""
    cols = list(range(1, max_count + 1))
    head = f"{'slice':>6}" + "".join(f"{c:>7}" for c in cols) + f"{'>' + str(max_count):>7}"
    print(f"\n=== {marker_name}+ cells by exact rolony count ===")
    print(head)
    print("-" * len(head))
    for s in rows_slices:
        counts = col[(sl == s) & qc].astype(np.int64)
        line = "".join(f"{int((counts == c).sum()):>7}" for c in cols)
        print(f"{s:>6}{line}{int((counts > max_count).sum()):>7}")


def write_csv(path, rows, cutoffs):
    fields = ["slice", "cells", "qc", "marker_cells", "rolonies", "max",
              "median_pos", "mean_pos"] + [f"ge{c}" for c in cutoffs]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


def write_cells_csv(path, fn, col, qc, sl, expmat, min_rolonies=ROLONY_FLOOR):
    """One row per marker+ QC-passing cell, so the table can be re-grouped."""
    reads = np.asarray(expmat.sum(axis=1)).ravel()
    genes = np.asarray((expmat > 0).sum(axis=1)).ravel()
    fov = np.asarray(fn["fov"]).ravel()
    ids = np.asarray(fn["id"]).ravel() if "id" in fn else None
    keep = np.where(qc & (col >= min_rolonies) & ~np.isnan(sl))[0]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_index", "id", "slice", "fov", "rolonies", "total_reads", "n_genes"])
        for i in keep:
            w.writerow([int(i), int(ids[i]) if ids is not None else "",
                        int(sl[i]), str(fov[i]),
                        int(col[i]), int(reads[i]), int(genes[i])])
    print(f"wrote {path} ({keep.size} cells)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="DATA_ROOT or filt_neurons.mat "
                                            "(default: local_config.DATA_ROOT)")
    ap.add_argument("--marker", choices=sorted(MARKERS), default="mscarlet")
    ap.add_argument("--marker-col", type=int, help="column index, overrides --marker")
    ap.add_argument("--min-reads", type=int, help="default: QC_MIN_READS")
    ap.add_argument("--min-genes", type=int, help="default: QC_MIN_GENES")
    ap.add_argument("--min-rolonies", type=int, default=ROLONY_FLOOR, metavar="N",
                    help=f"count only cells with >= N rolonies (default "
                         f"{ROLONY_FLOOR}); a floor other than 1 lands in the "
                         "CSV name as _geN")
    ap.add_argument("--cutoffs", type=int, nargs="+",
                    help=f"cumulative cells >= n columns (default: {' '.join(map(str, DEFAULT_CUTOFFS))} "
                         "plus ALIGN_MIN_ROLONIES)")
    ap.add_argument("--distribution", action="store_true",
                    help="also print cells per exact rolony count, per slice")
    ap.add_argument("--max-count", type=int, default=10,
                    help="last exact-count column in --distribution (default 10)")
    ap.add_argument("--csv", help="per-slice table path "
                                  "(default: <ANALYSIS_ROOT>/preprocessing/, else cwd)")
    ap.add_argument("--no-csv", action="store_true", help="print the table, write nothing")
    ap.add_argument("--cells-csv", nargs="?", const="", metavar="PATH",
                    help="also write one row per marker+ cell; bare flag uses the "
                         "default directory")
    args = ap.parse_args()

    gates, gate_err = _config_gates()
    min_reads = args.min_reads if args.min_reads is not None else (gates[0] if gates else None)
    min_genes = args.min_genes if args.min_genes is not None else (gates[1] if gates else None)
    if min_reads is None or min_genes is None:
        ap.error(f"preprocessing_config did not import ({gate_err}); "
                 "pass --min-reads and --min-genes for THIS brain")

    cutoffs = args.cutoffs
    if cutoffs is None:
        cutoffs = sorted(set(DEFAULT_CUTOFFS) | ({gates[2]} if gates else set()))

    path = args.path
    if path is None:
        import local_config

        path = local_config.DATA_ROOT

    marker_name, marker_col = MARKERS[args.marker]
    if args.marker_col is not None:
        marker_col = args.marker_col
        marker_name = f"col{marker_col}"

    p, fn = _load(path)
    expmat = fn["expmat"]
    col = _column(expmat, marker_col)
    if col is None:
        sys.exit(f"{marker_name}: column {marker_col} out of range "
                 f"({expmat.shape[1]} columns)")

    print(f"filt_neurons: {p}")
    print(f"cells: {expmat.shape[0]}   columns: {expmat.shape[1]}")
    print(f"marker: {marker_name} (column {marker_col})")
    src = lambda v: "CLI" if v is not None else "local_config"
    print(f"QC: reads >= {min_reads} ({src(args.min_reads)}) AND "
          f"genes >= {min_genes} ({src(args.min_genes)}), over all "
          f"{expmat.shape[1]} columns")
    print(f"floor: >= {args.min_rolonies} {marker_name} rolonies per cell")

    rows, qc, sl = census(expmat, fn["slice"], col, min_reads, min_genes, cutoffs,
                          args.min_rolonies)
    if not rows:
        sys.exit("no cell carries a slice id")
    print_table(rows, cutoffs, marker_name, args.min_rolonies)

    n_nan = int((np.isnan(sl) & qc & (col >= args.min_rolonies)).sum())
    if n_nan:
        print(f"\n{n_nan} QC-passing {marker_name}+ cells have no slice id and are "
              "in no row above")

    if args.distribution:
        print_distribution([r["slice"] for r in rows], col, qc, sl,
                           args.max_count, marker_name)

    stem = f"rolonies_per_slice_{marker_name.lower()}_qc{min_reads}_{min_genes}"
    if args.min_rolonies != 1:
        stem += f"_ge{args.min_rolonies}"
    if not args.no_csv:
        write_csv(args.csv or _default_out(f"{stem}.csv"), rows, cutoffs)
    if args.cells_csv is not None:
        write_cells_csv(args.cells_csv or _default_out(f"{stem}_cells.csv"),
                        fn, col, qc, sl, expmat, args.min_rolonies)


if __name__ == "__main__":
    main()
