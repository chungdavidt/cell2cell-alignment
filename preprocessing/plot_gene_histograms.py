#!/usr/bin/env python3
"""
Rolony-count histogram for every gene in the panel, plus the genes-per-cell
aggregate. Writes one multi-page PDF and a per-gene summary CSV.

Two different x axes, deliberately:

**Per gene (pages 2+).** One pair of axes per expmat column: x = rolonies of THAT
gene in a cell, y = number of cells at that value. This is what
`check_qc_metrics.py` draws for mScarlet alone (column 113), drawn for all 114.
Left axes = all cells, right axes = QC-passing, so the gate's effect is visible
per gene. Both share a fixed x range [0, --max-count] with everything above it
folded into one overflow bar, so panels are comparable across genes; each title
carries that gene's true max.

**Aggregate (page 1).** x = number of DIFFERENT genes detected in a cell (columns
with a nonzero count), y = number of cells. Same two-axes split. This is the
quantity `QC_MIN_GENES` thresholds, so the gate is drawn on it as a line.
Counted over the barcoded panel (columns 0..--panel-end) by default; the QC gate
itself counts all 114 columns, `--gene-count-columns all` matches it.

Columns 106+ are unused/readout slots, not panel genes, and the panel's labels
for them are stale -- they are titled by INDEX, with 111 = GCaMP and 113 =
mScarlet named from the config and the stale label shown as `label: ...`.

QC defaults to QC_MIN_READS / QC_MIN_GENES; --min-reads / --min-genes override,
and are required when preprocessing_config will not import (no local_config, or
a blank SCOPE) -- the gates are per-brain with no safe default.

Outputs land in <ANALYSIS_ROOT>/preprocessing/, else the working directory:
    gene_histograms_qc{reads}_{genes}.pdf
    gene_histograms_qc{reads}_{genes}.csv

Usage:
    python preprocessing/plot_gene_histograms.py
    python preprocessing/plot_gene_histograms.py <DATA_ROOT_or_filt_neurons.mat>
    python preprocessing/plot_gene_histograms.py --max-count 60 --per-page 9
    python preprocessing/plot_gene_histograms.py --genes 113 111 Slc17a7
    python preprocessing/plot_gene_histograms.py --linear --no-csv
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
# without local_config. Readout slots are index-only -- their panel labels are stale.
READOUT_NAMES = {111: "GCaMP", 113: "mScarlet"}
PANEL_END = 106          # columns 0..105 are barcoded genes; 106+ unused/readout

ALL_COLOR = "0.55"
QC_COLOR = "#c0392b"
OVERFLOW_COLOR = "#2c3e50"


def _config_gates():
    """(min_reads, min_genes), or None with the import error."""
    try:
        import preprocessing_config as cfg
    except Exception as exc:
        return None, exc
    return (cfg.QC_MIN_READS, cfg.QC_MIN_GENES), None


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


def column_label(idx, genes, panel_end):
    """Title for one expmat column. Readout slots never resolve by name."""
    stale = genes[idx] if genes is not None and idx < len(genes) else None
    if idx >= panel_end:
        name = READOUT_NAMES.get(idx)
        head = f"col {idx}: {name}" if name else f"col {idx}: readout/unused"
        return f"{head}   label: {stale!r}" if stale else head
    return f"col {idx}: {stale}" if stale else f"col {idx}"


def binned(counts, max_count):
    """Cells at each rolony count 0..max_count, plus one overflow bin."""
    clipped = np.minimum(counts, max_count + 1)
    return np.bincount(clipped, minlength=max_count + 2)[: max_count + 2]


def draw_counts(ax, heights, max_count, title, log_scale, color):
    x = np.arange(max_count + 2)
    ax.bar(x[:-1], heights[:-1], width=1.0, color=color, linewidth=0)
    ax.bar([x[-1]], [heights[-1]], width=1.0, color=OVERFLOW_COLOR, linewidth=0)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    ax.set_xlim(-0.7, max_count + 1.7)
    ticks = [t for t in (0, max_count // 2, max_count) if t <= max_count]
    ax.set_xticks(ticks + [max_count + 1])
    ax.set_xticklabels([str(t) for t in ticks] + [f">{max_count}"], fontsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title(title, fontsize=7)


def gene_pages(pdf, expmat, columns, genes, qc, max_count, per_page, pairs_per_row,
               log_scale, panel_end, summary):
    """One pair of axes per column: all cells, then QC-passing."""
    import matplotlib.pyplot as plt

    from utilities.mat_io import get_expression_column

    rows_per_page = int(np.ceil(per_page / pairs_per_row))
    for start in range(0, len(columns), per_page):
        chunk = columns[start: start + per_page]
        fig, axes = plt.subplots(rows_per_page, pairs_per_row * 2,
                                 figsize=(2.6 * pairs_per_row * 2, 2.1 * rows_per_page),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")

        for k, idx in enumerate(chunk):
            r, c = divmod(k, pairs_per_row)
            ax_all, ax_qc = axes[r][2 * c], axes[r][2 * c + 1]
            col = np.asarray(get_expression_column(expmat, idx)).ravel().astype(np.int64)
            label = column_label(idx, genes, panel_end)

            for ax, mask, tag, color in ((ax_all, None, "all", ALL_COLOR),
                                         (ax_qc, qc, "QC", QC_COLOR)):
                vals = col if mask is None else col[mask]
                pos = vals[vals > 0]
                ax.axis("on")
                title = (f"{label}\n{tag}: {pos.size} cells > 0, "
                         f"max {int(vals.max()) if vals.size else 0}")
                draw_counts(ax, binned(vals, max_count), max_count, title,
                            log_scale, color)

            qc_vals = col[qc]
            qc_pos = qc_vals[qc_vals > 0]
            summary.append({
                "column": idx,
                "label": label.replace("\n", " "),
                "is_panel_gene": int(idx < panel_end),
                "cells_pos_all": int((col > 0).sum()),
                "cells_pos_qc": int(qc_pos.size),
                "rolonies_all": int(col.sum()),
                "rolonies_qc": int(qc_vals.sum()),
                "max_all": int(col.max()) if col.size else 0,
                "max_qc": int(qc_vals.max()) if qc_vals.size else 0,
                "median_pos_qc": float(np.median(qc_pos)) if qc_pos.size else 0.0,
                "mean_pos_qc": float(qc_pos.mean()) if qc_pos.size else 0.0,
            })

        fig.suptitle("rolonies per cell (x) vs cells (y) — left: all cells, "
                     f"right: QC-passing   [columns {chunk[0]}–{chunk[-1]}]", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)
        plt.close(fig)


def aggregate_page(pdf, n_genes_all, n_genes_qc, min_genes, hi, log_scale, source):
    """x = number of different genes detected in a cell, y = cells."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, tag, color in ((axes[0], n_genes_all, "all cells", ALL_COLOR),
                                 (axes[1], n_genes_qc, "QC-passing", QC_COLOR)):
        heights = np.bincount(vals, minlength=hi + 1)
        ax.bar(np.arange(heights.size), heights, width=1.0, color=color, linewidth=0)
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.5)
        ax.axvline(min_genes - 0.5, color="#2980b9", lw=1.0,
                   label=f"QC_MIN_GENES = {min_genes}")
        ax.set_xlabel(f"different genes detected per cell ({source})")
        ax.set_ylabel("cells")
        med = int(np.median(vals)) if vals.size else 0
        ax.set_title(f"{tag}: {vals.size} cells, median {med}", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def write_csv(path, summary):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {path}")


def resolve_columns(spec, genes, n_cols):
    """--genes entries: column indices or gene names."""
    if not spec:
        return list(range(n_cols))
    lowered = [g.lower() for g in (genes or [])]
    out = []
    for token in spec:
        if token.isdigit():
            idx = int(token)
        elif token.lower() in lowered:
            idx = lowered.index(token.lower())
        else:
            raise SystemExit(f"--genes {token!r}: not a column index and not in the gene list")
        if idx >= n_cols:
            raise SystemExit(f"--genes {token!r}: column {idx} of {n_cols}")
        out.append(idx)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="DATA_ROOT or filt_neurons.mat "
                                            "(default: local_config.DATA_ROOT)")
    ap.add_argument("--min-reads", type=int, help="default: QC_MIN_READS")
    ap.add_argument("--min-genes", type=int, help="default: QC_MIN_GENES")
    ap.add_argument("--max-count", type=int, default=40,
                    help="last rolony-count bar; higher counts fold into one "
                         "overflow bar (default 40)")
    ap.add_argument("--per-page", type=int, default=12, help="genes per PDF page")
    ap.add_argument("--pairs-per-row", type=int, default=3,
                    help="gene pairs across one row (default 3, i.e. 6 axes)")
    ap.add_argument("--panel-end", type=int, default=PANEL_END,
                    help="first non-panel column (default 106)")
    ap.add_argument("--gene-count-columns", choices=("panel", "all"), default="panel",
                    help="columns the aggregate counts distinct genes over "
                         "(default panel; 'all' matches the QC gate)")
    ap.add_argument("--genes", nargs="+", metavar="COL_OR_NAME",
                    help="restrict the per-gene pages to these columns")
    ap.add_argument("--linear", action="store_true", help="linear cell-count axis")
    ap.add_argument("--pdf", help="PDF path (default: <ANALYSIS_ROOT>/preprocessing/)")
    ap.add_argument("--csv", help="per-gene summary CSV path")
    ap.add_argument("--no-csv", action="store_true", help="plot only, write no CSV")
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
        expmat = expmat.tocsc()      # every page slices by column

    reads = np.asarray(expmat.sum(axis=1)).ravel()
    n_genes_all_cols = np.asarray((expmat > 0).sum(axis=1)).ravel().astype(np.int64)
    qc = (reads >= min_reads) & (n_genes_all_cols >= min_genes)

    if args.gene_count_columns == "panel":
        counted = np.asarray((expmat[:, :panel_end] > 0).sum(axis=1)).ravel().astype(np.int64)
        source = f"columns 0..{panel_end - 1}"
    else:
        counted = n_genes_all_cols
        source = f"all {n_cols} columns"

    print(f"filt_neurons: {p}")
    print(f"cells: {n_cells}   columns: {n_cols}   panel: 0..{panel_end - 1}")
    src = lambda v: "CLI" if v is not None else "local_config"
    print(f"QC: reads >= {min_reads} ({src(args.min_reads)}) AND "
          f"genes >= {min_genes} ({src(args.min_genes)}), over all {n_cols} columns")
    print(f"QC-passing: {int(qc.sum())} ({100.0 * qc.sum() / n_cells:.1f}%)")

    columns = resolve_columns(args.genes, genes, n_cols)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    stem = f"gene_histograms_qc{min_reads}_{min_genes}"
    pdf_path = Path(args.pdf) if args.pdf else _default_out(f"{stem}.pdf")
    summary = []
    with PdfPages(pdf_path) as pdf:
        aggregate_page(pdf, counted, counted[qc], min_genes, int(counted.max()),
                       not args.linear, source)
        gene_pages(pdf, expmat, columns, genes, qc, args.max_count, args.per_page,
                   args.pairs_per_row, not args.linear, panel_end, summary)
    n_pages = 1 + int(np.ceil(len(columns) / args.per_page))
    print(f"wrote {pdf_path} ({n_pages} pages, {len(columns)} genes)")

    if not args.no_csv and summary:
        write_csv(Path(args.csv) if args.csv else _default_out(f"{stem}.csv"), summary)


if __name__ == "__main__":
    main()
