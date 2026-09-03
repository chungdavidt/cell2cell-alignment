"""
Read-only audit of per-cell read/gene QC in a filt_neurons.mat.

Two questions:

1. Does the QC pass rate the lab reports for a brain reproduce under
   `reads >= 20 AND genes >= 5`? For BY95 they reported 26.4% passing with a
   median of 37 reads and 20 genes per passing cell, and compare_datasets.py
   already got 26.4% at 20/5, so all three should land. A mismatch means their
   filter is not this one.

2. Among the cells that pass, how are mScarlet counts distributed? Prints the
   exact per-count table with a cumulative >= column, and saves a histogram.
   That cumulative column is what sets a defensible mScarlet cutoff: it says how
   many marker cells survive at >=1, >=2, >=3 ... and how many slices keep any.

A third block prices a reads/genes gate on the marker+ cells subslice selection
keeps. It dates from when QC_MIN_READS/QC_MIN_GENES were 0 and step 1 selected
on `expmat[:, MSCARLET_COL] > 0` alone; they are 20/5 as of 2026-08-26, so step
1 already applies the gate and this block now prices it after the fact.

Read/gene totals are computed two ways: over all 114 expmat columns (the lab's
definition, used for the reproduction check and the QC mask) and over the
barcoded panel alone (columns 0..PANEL_END-1), which excludes unused-1..5 and
the three readout channels.

--exclude-slices drops every cell in those sections before anything is counted,
for a section whose data is bad. It applies to every block, so the reproduction
check runs on a subset of the brain and is no longer a reproduction of the
reported number -- it says so and withholds its verdict when the flag is used.
Cells with a NaN slice (assigned to no section) are never dropped, and a slice
number with no cells is an error, so a typo cannot write a file whose name
claims an exclusion that did not happen. The flag is local to this script, like
plot_gene_histograms.py's flag of the same name: no config name holds it and the
pipeline still renders the section. --compare is untouched -- the second dataset
is a different brain.

The only file written is the histogram PNG,
qc_<marker>_counts[_ex{slices}].png.

Usage:
    python preprocessing/check_qc_metrics.py <DATA_ROOT_or_filt_neurons.mat>
    python preprocessing/check_qc_metrics.py            # uses local_config.DATA_ROOT
    python preprocessing/check_qc_metrics.py --lab-reads 20 --lab-genes 5
    python preprocessing/check_qc_metrics.py --out qc.png --no-plot
    python preprocessing/check_qc_metrics.py --exclude-slices 58
    python preprocessing/check_qc_metrics.py --compare <OTHER_DATA_ROOT>
    python preprocessing/check_qc_metrics.py --no-reported          # new brain

The --reported-* defaults are BY95's numbers. They are per-brain, like
QC_MIN_READS/QC_MIN_GENES; pass the new brain's or use --no-reported.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

MSCARLET_COLUMN_INDEX = 113
GCAMP_COLUMN_INDEX = 111
PANEL_END = 106          # columns 0..105 are barcoded genes; 106+ are unused/readout

SWEEP_READS = (0, 1, 2, 5, 10, 20)
SWEEP_GENES = (0, 1, 2, 5)


def _column(expmat, idx):
    import scipy.sparse as sp

    if idx >= expmat.shape[1]:
        return None
    col = expmat[:, idx]
    return np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()


def _totals(expmat, hi=None):
    """(reads, genes) per cell, over columns [0, hi)."""
    sub = expmat if hi is None else expmat[:, :hi]
    reads = np.asarray(sub.sum(axis=1)).ravel()
    genes = np.asarray((sub > 0).sum(axis=1)).ravel()
    return reads, genes


def _pct(n, total):
    return 100.0 * n / total if total else 0.0


def _default_out(marker_name, excluded=()):
    stem = f"qc_{marker_name.lower()}_counts"
    if excluded:
        stem += "_ex" + "_".join(str(s) for s in excluded)
    name = f"{stem}.png"
    try:
        from analysis_paths import analysis_subdir

        d = analysis_subdir("preprocessing", create=True)
        if d is not None:
            return Path(d) / name
    except Exception:
        pass
    return Path.cwd() / name


def drop_slices(expmat, slice_field, excluded):
    """expmat without the rows in `excluded`, and the keep mask that cut them.

    Mirrors plot_gene_histograms.drop_slices, kept local so this script still
    runs off a bare filt_neurons.mat path. The mask comes back because `slice`
    is passed on to the per-count table and the sweep, and has to lose the same
    rows the matrix did. Cells whose slice is NaN belong to no section and are
    always kept -- NaN equals nothing, so the mask never sees them.
    """
    import scipy.sparse as sp

    if slice_field is None:
        raise SystemExit("--exclude-slices: filt_neurons.mat has no `slice` field")
    sl = np.asarray(slice_field).ravel().astype(float)
    if sl.size != expmat.shape[0]:
        raise SystemExit(f"--exclude-slices: `slice` is {sl.size} rows against "
                         f"expmat's {expmat.shape[0]}")

    present = set(np.unique(sl[~np.isnan(sl)]).astype(int).tolist())
    missing = [s for s in excluded if s not in present]
    if missing:
        raise SystemExit(f"--exclude-slices {missing}: no cells in this dataset. "
                         f"present: {sorted(present)}")

    keep = ~np.isin(sl, excluded)
    expmat = sp.csr_matrix(expmat)[keep] if sp.issparse(expmat) else np.asarray(expmat)[keep]
    return expmat, keep


def _slice_phrase(excluded):
    return (f"slice{'s' if len(excluded) > 1 else ''} "
            f"{', '.join(str(s) for s in excluded)}")


def _load(path):
    p = Path(path)
    if p.is_dir():
        p = p / "filt_neurons.mat"
    from utilities.mat_io import load_filt_neurons

    fn = load_filt_neurons(p)
    return p, fn


def _summary(expmat, hi, reads_thr, genes_thr):
    reads, genes = _totals(expmat, hi)
    keep = (reads >= reads_thr) & (genes >= genes_thr)
    n = reads.size
    return {
        "n": n,
        "n_pass": int(keep.sum()),
        "pass_pct": _pct(int(keep.sum()), n),
        "median_reads": float(np.median(reads[keep])) if keep.any() else 0.0,
        "median_genes": float(np.median(genes[keep])) if keep.any() else 0.0,
        "mean_reads": float(reads.mean()),
    }


def report_lab_filter(expmat, reads_thr, genes_thr, label, reported, excluded=()):
    print(f"\n=== reproducing the reported QC ({label}) ===")
    print(f"filter: reads >= {reads_thr} AND genes >= {genes_thr}")
    if excluded:
        print(f"{_slice_phrase(excluded)} excluded: these numbers cover part of the "
              "brain, the reported ones cover all of it, so this is not a "
              "reproduction check")
    print(f"{'columns':<20}{'pass':>9}{'pass %':>9}{'med reads':>11}{'med genes':>11}")
    out = {}
    for name, hi in (("all 114", None), (f"panel 0..{PANEL_END-1}", PANEL_END)):
        s = _summary(expmat, hi, reads_thr, genes_thr)
        out[name] = s
        print(f"{name:<20}{s['n_pass']:>9d}{s['pass_pct']:>9.1f}{s['median_reads']:>11.0f}"
              f"{s['median_genes']:>11.0f}")

    if reported is None:
        return
    s = out["all 114"]
    d_pass = abs(s["pass_pct"] - reported["pass_pct"])
    d_r = abs(s["median_reads"] - reported["median_reads"])
    d_g = abs(s["median_genes"] - reported["median_genes"])
    print(f"\nlab reported: {reported['pass_pct']}% pass, median "
          f"{reported['median_reads']} reads, {reported['median_genes']} genes")
    if excluded:
        print(f"  -> no verdict: {_slice_phrase(excluded)} excluded, "
              "the two do not cover the same cells")
    elif d_pass <= 0.1 and d_r <= 1 and d_g <= 1:
        print("  -> all three reproduce; this is their filter")
    elif d_pass <= 0.1:
        print(f"  -> pass rate reproduces, medians differ by {d_r:.0f} reads / {d_g:.0f} genes")
        print("     their medians are probably computed over a different column set or cell subset")
    else:
        print(f"  -> pass rate is off by {d_pass:.1f} points; their filter is not reads>={reads_thr} "
              f"& genes>={genes_thr}")


def report_distribution(expmat):
    print("\n=== read/gene distribution ===")
    qs = (10, 25, 50, 75, 90, 99)
    for name, hi in (("all 114", None), (f"panel 0..{PANEL_END-1}", PANEL_END)):
        reads, genes = _totals(expmat, hi)
        nz = reads > 0
        print(f"\n{name}:  zero-read {int((~nz).sum())} ({_pct(int((~nz).sum()), reads.size):.1f}%)"
              f"   mean {reads.mean():.2f}   max {reads.max():.0f}")
        print(f"{'percentile':<16}" + "".join(f"{q:>8d}" for q in qs))
        for lbl, arr in (("reads, all", reads), ("genes, all", genes),
                         ("reads, nonzero", reads[nz]), ("genes, nonzero", genes[nz])):
            if arr.size == 0:
                continue
            print(f"{lbl:<16}" + "".join(f"{np.percentile(arr, q):>8.0f}" for q in qs))


def marker_distribution(expmat, marker_col, marker_name, reads_thr, genes_thr,
                        slice_ids):
    """Per-count table of the marker among QC-passing cells. Returns (col, qc)."""
    col = _column(expmat, marker_col)
    if col is None:
        print(f"\n{marker_name}: column {marker_col} out of range")
        return None, None

    reads, genes = _totals(expmat, None)
    qc = (reads >= reads_thr) & (genes >= genes_thr)
    reads_p, genes_p = _totals(expmat, PANEL_END)
    qc_panel = (reads_p >= reads_thr) & (genes_p >= genes_thr)

    sl = np.asarray(slice_ids).ravel().astype(float)
    counts = col[qc].astype(np.int64)
    pos = counts > 0

    print(f"\n=== {marker_name} counts among QC-passing cells ===")
    print(f"QC: reads >= {reads_thr} AND genes >= {genes_thr}, over all "
          f"{expmat.shape[1]} columns")
    print(f"QC-passing cells: {int(qc.sum())} of {col.size} "
          f"({_pct(int(qc.sum()), col.size):.1f}%)")
    print(f"  same filter over the barcoded panel only: {int(qc_panel.sum())} "
          f"({_pct(int(qc_panel.sum()), col.size):.1f}%)")
    if not qc.any():
        return col, qc
    print(f"{marker_name}+ within QC-passing: {int(pos.sum())} "
          f"({_pct(int(pos.sum()), int(qc.sum())):.1f}%)   "
          f"max {counts.max()}   median-of-positive "
          f"{np.median(counts[pos]) if pos.any() else 0:.0f}")

    print(f"\n{marker_name} count   cells   % of QC   cells >= n   slices >= n")
    total = int(qc.sum())
    for v in range(0, int(counts.max()) + 1):
        n_at = int((counts == v).sum())
        n_ge = int((counts >= v).sum())
        keep = qc & (col >= v)
        ks = sl[keep & ~np.isnan(sl)]
        n_sl = int(np.unique(ks).size)
        print(f"{v:>12d}{n_at:>8d}{_pct(n_at, total):>10.2f}{n_ge:>13d}{n_sl:>14d}")

    return col, qc


def plot_marker_histogram(col, qc, marker_name, out_path, log_scale=True, note=""):
    """Histogram: rolony count per cell on x, number of cells on y.

    Log y by default -- the distribution spans orders of magnitude, so on a
    linear axis the 0-rolony bar flattens every cutoff-relevant bar to nothing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = col[qc].astype(np.int64)
    vmax = int(counts.max())
    values = np.arange(0, vmax + 1)
    n_cells = np.array([(counts == v).sum() for v in values], dtype=np.int64)

    fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 0.22 * (vmax + 1)), 8))
    for ax, lo, title in (
        (axes[0], 0, f"all QC-passing cells (n={counts.size})"),
        (axes[1], 1, f"{marker_name}+ only (n={int((counts > 0).sum())})"),
    ):
        sel = values >= lo
        ax.bar(values[sel], n_cells[sel], width=0.8, color="#b03030")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{marker_name} rolony count per cell")
        ax.set_ylabel("number of cells")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        # Per-bar counts stop being readable once the axis is crowded.
        if sel.sum() <= 60:
            for v, n in zip(values[sel], n_cells[sel]):
                if n:
                    ax.annotate(f"{n}", (v, n), textcoords="offset points",
                                xytext=(0, 2), ha="center", fontsize=6)

    title = f"{marker_name} counts per QC-passing cell"
    if note:
        title += f"\n{note}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nhistogram: {out_path}")


def report_sweep(expmat, pos, slice_ids, marker_name):
    if pos is None:
        return
    reads_p, genes_p = _totals(expmat, PANEL_END)
    n_pos = int(pos.sum())
    sl = np.asarray(slice_ids).ravel().astype(float)
    assigned = ~np.isnan(sl)

    print(f"\n=== cost of a QC gate on {marker_name}+ cells ===")
    print("kept = marker+ AND reads >= R AND genes >= G, counted over the barcoded panel")
    print(f"{'R':>4}{'G':>4}{'kept':>10}{'% of pos':>10}{'slices':>9}"
          f"{'cells/slice min':>17}{'median':>9}")
    for r in SWEEP_READS:
        for g in SWEEP_GENES:
            keep = pos & (reads_p >= r) & (genes_p >= g)
            k = int(keep.sum())
            ks = sl[keep & assigned]
            if ks.size:
                counts = np.array([int((ks == s).sum()) for s in np.unique(ks)])
                nsl, cmin, cmed = counts.size, int(counts.min()), int(np.median(counts))
            else:
                nsl = cmin = cmed = 0
            print(f"{r:>4}{g:>4}{k:>10d}{_pct(k, n_pos):>10.1f}{nsl:>9d}{cmin:>17d}{cmed:>9d}")

    base_slices = np.unique(sl[pos & assigned])
    print(f"\nbaseline (R=0, G=0): {n_pos} {marker_name}+ cells across "
          f"{base_slices.size} slices")
    print("a row that drops the slice count loses a subslice from step 1 entirely")
    print("unused/readout columns are excluded here, so marker-only cells fail every R>0 row")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="DATA_ROOT or filt_neurons.mat")
    ap.add_argument("--compare", help="second DATA_ROOT or filt_neurons.mat for the QC block only")
    ap.add_argument("--lab-reads", type=int, default=20)
    ap.add_argument("--lab-genes", type=int, default=5)
    ap.add_argument("--marker-col", type=int, default=MSCARLET_COLUMN_INDEX)
    ap.add_argument("--marker-name", default="mScarlet")
    ap.add_argument("--reported-pass", type=float, default=26.4,
                    help="pass %% the lab reported for THIS brain (BY95 default)")
    ap.add_argument("--reported-reads", type=float, default=37,
                    help="median reads/cell the lab reported for THIS brain")
    ap.add_argument("--reported-genes", type=float, default=20,
                    help="median genes/cell the lab reported for THIS brain")
    ap.add_argument("--exclude-slices", type=int, nargs="+", metavar="N",
                    help="drop every cell in these sections before counting; "
                         "the numbers land in the output filename")
    ap.add_argument("--no-reported", action="store_true",
                    help="skip the reproduction check (new brain, no reported numbers yet)")
    ap.add_argument("--out", help="histogram path (default: <ANALYSIS_ROOT>/preprocessing/"
                                  "qc_<marker>_counts.png, else cwd)")
    ap.add_argument("--no-plot", action="store_true", help="tables only, write nothing")
    ap.add_argument("--linear", action="store_true",
                    help="linear cell-count axis instead of log")
    args = ap.parse_args()

    reported = None if args.no_reported else {
        "pass_pct": args.reported_pass,
        "median_reads": args.reported_reads,
        "median_genes": args.reported_genes,
    }

    path = args.path
    if path is None:
        import local_config

        path = local_config.DATA_ROOT

    p, fn = _load(path)
    expmat = fn["expmat"]
    slice_ids = fn.get("slice")
    print(f"filt_neurons: {p}")
    print(f"cells: {expmat.shape[0]}   columns: {expmat.shape[1]}")

    excluded = sorted(set(args.exclude_slices or []))
    if excluded:
        expmat, keep = drop_slices(expmat, slice_ids, excluded)
        slice_ids = np.asarray(slice_ids).ravel()[keep]
        print(f"excluded {_slice_phrase(excluded)}: dropped {int((~keep).sum())} cells, "
              f"{expmat.shape[0]} left")

    report_lab_filter(expmat, args.lab_reads, args.lab_genes, p.parent.name, reported,
                      excluded)
    report_distribution(expmat)

    col, qc = marker_distribution(expmat, args.marker_col, args.marker_name,
                                  args.lab_reads, args.lab_genes, slice_ids)

    gcamp = _column(expmat, GCAMP_COLUMN_INDEX)
    if gcamp is not None and args.marker_col != GCAMP_COLUMN_INDEX:
        g = gcamp > 0
        print(f"\nGCaMP+ (col {GCAMP_COLUMN_INDEX}): {int(g.sum())} "
              f"({_pct(int(g.sum()), gcamp.size):.1f}%)   max {gcamp.max():.0f}")

    if col is not None and qc is not None and qc.any() and not args.no_plot:
        out = args.out or _default_out(args.marker_name, excluded)
        plot_marker_histogram(col, qc, args.marker_name, out, log_scale=not args.linear,
                              note=f"excluding {_slice_phrase(excluded)}" if excluded else "")

    report_sweep(expmat, None if col is None else (col > 0), slice_ids, args.marker_name)

    if args.compare:
        p2, fn2 = _load(args.compare)
        print(f"\n\n########## comparison: {p2} ##########")
        print(f"cells: {fn2['expmat'].shape[0]}")
        if excluded:
            print(f"--exclude-slices applies to {p.parent.name} only; "
                  f"{p2.parent.name} is whole")
        report_lab_filter(fn2["expmat"], args.lab_reads, args.lab_genes, p2.parent.name, None)
        print("\n=== side by side (all 114 columns) ===")
        a = _summary(expmat, None, args.lab_reads, args.lab_genes)
        b = _summary(fn2["expmat"], None, args.lab_reads, args.lab_genes)
        print(f"{'':<16}{p.parent.name:>16}{p2.parent.name:>16}")
        for lbl, key, fmt in (("cells", "n", "d"), ("pass", "n_pass", "d"),
                              ("pass %", "pass_pct", ".1f"),
                              ("med reads", "median_reads", ".0f"),
                              ("med genes", "median_genes", ".0f"),
                              ("mean reads", "mean_reads", ".2f")):
            print(f"{lbl:<16}{a[key]:>16{fmt}}{b[key]:>16{fmt}}")


if __name__ == "__main__":
    main()
