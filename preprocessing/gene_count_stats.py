#!/usr/bin/env python3
"""
Numbers behind plot_gene_histograms.py: what a typical cell holds, and what a
typical gene looks like across cells. Prints tables; writes a per-gene CSV.

"Median rolony count across all the genes, per cell" reads two ways, so both are
printed:

  PER GENE   for each column, the median rolonies per cell -- over expressing
             cells (`median_pos`) and over every cell (`median_all`, which is 0
             for nearly every gene, since most cells do not express most genes).
             The RANGE across genes is then min .. max of that per-gene median,
             with the median-of-medians as the middle. This is the "how much does
             one gene read in one cell, and how far does that vary between genes"
             answer.

  PER CELL   for each cell, the spread of its own 114 counts: max, and max - min.
             `min` is 0 for any cell not expressing all 114 columns, i.e. all of
             them, so the per-cell range collapses onto the per-cell max -- both
             are printed so that is visible rather than assumed.

Genes per cell is the third block: distinct columns with a nonzero count, the
quantity `QC_MIN_GENES` gates, as median / IQR / percentiles.

Every block is printed twice, all cells then QC-passing (QC_MIN_READS /
QC_MIN_GENES over all 114 columns, the lab's definition). --min-reads /
--min-genes override and are required when preprocessing_config will not import.

Outputs land in <ANALYSIS_ROOT>/preprocessing/, else the working directory:
    gene_count_stats_qc{reads}_{genes}.csv     one row per column

Usage:
    python preprocessing/gene_count_stats.py
    python preprocessing/gene_count_stats.py <DATA_ROOT_or_filt_neurons.mat>
    python preprocessing/gene_count_stats.py --top 20
    python preprocessing/gene_count_stats.py --panel-only --no-csv
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

from plot_gene_histograms import PANEL_END, _config_gates, _default_out, _load, column_label

PCTS = (1, 5, 25, 50, 75, 95, 99)


def _describe(name, vals, unit):
    if vals.size == 0:
        print(f"{name:<28} (no cells)")
        return
    qs = np.percentile(vals, PCTS)
    print(f"{name:<28}" + "".join(f"{q:>9.1f}" for q in qs)
          + f"{vals.mean():>10.2f}{vals.max():>9.0f}   {unit}")


def per_cell_block(tag, n_genes, totals, row_max, row_range, min_genes):
    print(f"\n=== per cell, {tag} ({n_genes.size} cells) ===")
    print(f"{'':<28}" + "".join(f"{'p' + str(p):>9}" for p in PCTS)
          + f"{'mean':>10}{'max':>9}")
    _describe("different genes per cell", n_genes.astype(float), "columns with a nonzero count")
    _describe("total rolonies per cell", totals.astype(float), "summed over all columns")
    _describe("largest single gene", row_max.astype(float), "max count in that cell")
    _describe("range across genes", row_range.astype(float),
              "max - min over the cell's own columns")
    n_nonzero_min = int((row_max != row_range).sum())
    if n_nonzero_min:
        print(f"{n_nonzero_min} cells express every column, so their min is above 0")
    print(f"median different genes per cell: {np.median(n_genes):.0f}"
          f"   (QC_MIN_GENES = {min_genes})")


def per_gene_rows(expmat, columns, genes, qc, panel_end):
    """One row per column, both cell sets."""
    from utilities.mat_io import get_expression_column

    rows = []
    for idx in columns:
        col = np.asarray(get_expression_column(expmat, idx)).ravel().astype(np.int64)
        row = {"column": idx, "label": column_label(idx, genes, panel_end),
               "is_panel_gene": int(idx < panel_end)}
        for tag, vals in (("all", col), ("qc", col[qc])):
            pos = vals[vals > 0]
            row[f"cells_{tag}"] = int(vals.size)
            row[f"cells_pos_{tag}"] = int(pos.size)
            row[f"pct_cells_pos_{tag}"] = 100.0 * pos.size / vals.size if vals.size else 0.0
            row[f"median_all_{tag}"] = float(np.median(vals)) if vals.size else 0.0
            row[f"median_pos_{tag}"] = float(np.median(pos)) if pos.size else 0.0
            row[f"p25_pos_{tag}"] = float(np.percentile(pos, 25)) if pos.size else 0.0
            row[f"p75_pos_{tag}"] = float(np.percentile(pos, 75)) if pos.size else 0.0
            row[f"mean_pos_{tag}"] = float(pos.mean()) if pos.size else 0.0
            row[f"max_{tag}"] = int(vals.max()) if vals.size else 0
            row[f"rolonies_{tag}"] = int(vals.sum())
        rows.append(row)
    return rows


def per_gene_block(tag, rows, top):
    key_med, key_pos, key_max = f"median_pos_{tag}", f"cells_pos_{tag}", f"max_{tag}"
    seen = [r for r in rows if r[key_pos] > 0]
    print(f"\n=== per gene, {tag} ({len(seen)} of {len(rows)} columns detected) ===")
    if not seen:
        return

    meds = np.array([r[key_med] for r in seen])
    med_all = np.array([r[f"median_all_{tag}"] for r in seen])
    lo, hi = min(seen, key=lambda r: r[key_med]), max(seen, key=lambda r: r[key_med])
    print(f"median rolonies per EXPRESSING cell, across genes: "
          f"range {meds.min():.0f} .. {meds.max():.0f}, "
          f"median-of-medians {np.median(meds):.1f}, mean {meds.mean():.2f}")
    print(f"  lowest  {lo['label']}  median {lo[key_med]:.0f}  "
          f"in {lo[key_pos]} cells")
    print(f"  highest {hi['label']}  median {hi[key_med]:.0f}  "
          f"in {hi[key_pos]} cells")
    n_zero = int((med_all == 0).sum())
    print(f"median over ALL cells (zeros included): {n_zero} of {len(seen)} genes "
          f"sit at 0, range {med_all.min():.0f} .. {med_all.max():.0f}")

    print(f"\ntop {top} by median rolonies per expressing cell")
    print(f"{'col':>5}  {'median':>7}{'p25':>6}{'p75':>6}{'mean':>7}{'max':>7}"
          f"{'cells>0':>9}{'% cells':>9}  label")
    ranked = sorted(seen, key=lambda r: (r[key_med], r[f"mean_pos_{tag}"]), reverse=True)
    for r in ranked[:top]:
        print(f"{r['column']:>5}  {r[key_med]:>7.0f}{r[f'p25_pos_{tag}']:>6.0f}"
              f"{r[f'p75_pos_{tag}']:>6.0f}{r[f'mean_pos_{tag}']:>7.2f}{r[key_max]:>7d}"
              f"{r[key_pos]:>9d}{r[f'pct_cells_pos_{tag}']:>9.2f}  {r['label']}")


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="DATA_ROOT or filt_neurons.mat "
                                            "(default: local_config.DATA_ROOT)")
    ap.add_argument("--min-reads", type=int, help="default: QC_MIN_READS")
    ap.add_argument("--min-genes", type=int, help="default: QC_MIN_GENES")
    ap.add_argument("--panel-end", type=int, default=PANEL_END,
                    help="first non-panel column (default 106)")
    ap.add_argument("--panel-only", action="store_true",
                    help="restrict every block to the barcoded panel")
    ap.add_argument("--top", type=int, default=15, help="rows in the ranked table")
    ap.add_argument("--csv", help="per-gene CSV path")
    ap.add_argument("--no-csv", action="store_true", help="print only")
    args = ap.parse_args()

    gates, gate_err = _config_gates()
    min_reads = args.min_reads if args.min_reads is not None else (gates[0] if gates else None)
    min_genes = args.min_genes if args.min_genes is not None else (gates[1] if gates else None)
    if min_reads is None or min_genes is None:
        ap.error(f"preprocessing_config did not import ({gate_err}); "
                 "pass --min-reads and --min-genes for THIS brain")

    path = args.path
    if path is None:
        import local_config

        path = local_config.DATA_ROOT

    p, fn = _load(path)
    expmat = fn["expmat"]
    genes = fn.get("genes")
    n_cells, n_cols = expmat.shape
    panel_end = min(args.panel_end, n_cols)

    import scipy.sparse as sp

    if sp.issparse(expmat):
        expmat = expmat.tocsc()

    # QC is the lab's filter: over all columns, whatever --panel-only asks of the blocks.
    reads_all = np.asarray(expmat.sum(axis=1)).ravel()
    genes_all = np.asarray((expmat > 0).sum(axis=1)).ravel().astype(np.int64)
    qc = (reads_all >= min_reads) & (genes_all >= min_genes)

    block = expmat[:, :panel_end] if args.panel_only else expmat
    columns = list(range(panel_end if args.panel_only else n_cols))
    scope = f"columns 0..{panel_end - 1}" if args.panel_only else f"all {n_cols} columns"

    totals = np.asarray(block.sum(axis=1)).ravel().astype(np.int64)
    n_genes = np.asarray((block > 0).sum(axis=1)).ravel().astype(np.int64)
    if sp.issparse(block):
        row_max = np.asarray(block.max(axis=1).todense()).ravel().astype(np.int64)
    else:
        row_max = np.asarray(block).max(axis=1).astype(np.int64)

    # min is 0 unless a cell has a count in every column, which is why the
    # per-cell range and the per-cell max are nearly the same number.
    row_min = np.zeros_like(row_max)
    full = n_genes == block.shape[1]
    if full.any():
        sub = block.tocsr()[full].todense() if sp.issparse(block) else np.asarray(block)[full]
        row_min[full] = np.asarray(sub).min(axis=1).ravel()
    row_range = row_max - row_min

    print(f"filt_neurons: {p}")
    print(f"cells: {n_cells}   columns: {n_cols}   counting over {scope}")
    src = lambda v: "CLI" if v is not None else "local_config"
    print(f"QC: reads >= {min_reads} ({src(args.min_reads)}) AND "
          f"genes >= {min_genes} ({src(args.min_genes)}), over all {n_cols} columns")
    print(f"QC-passing: {int(qc.sum())} ({100.0 * qc.sum() / n_cells:.1f}%)")

    per_cell_block("all cells", n_genes, totals, row_max, row_range, min_genes)
    per_cell_block("QC-passing", n_genes[qc], totals[qc], row_max[qc], row_range[qc],
                   min_genes)

    rows = per_gene_rows(expmat, columns, genes, qc, panel_end)
    per_gene_block("all", rows, args.top)
    per_gene_block("qc", rows, args.top)

    if not args.no_csv:
        out = Path(args.csv) if args.csv else _default_out(
            f"gene_count_stats_qc{min_reads}_{min_genes}.csv")
        write_csv(out, rows)


if __name__ == "__main__":
    main()
