#!/usr/bin/env python3
"""
Checks on make_slice_figures.py's clustering that do not use its layer-marker table.

    python alignment/check_clustering.py --mat filt_neurons.mat \
        --assignments cell_cluster_assignments.csv --outdir clustering_check [--stability]

Prints each table and writes it as CSV to --outdir:
  panel_genes.csv        every expmat column: name, clustering feature or not,
                         % of clustered cells detected, mean count
  pc_loadings.csv        first 10 PCs: variance %, top 6 genes by |loading|
  cluster_qc.csv         per cluster: n, median reads, median genes detected
  cluster_top_genes.csv  per cluster, top 10 genes by mean log1p difference
                         (cluster - rest), with % detected in / out
  cluster_by_slide.csv   per cluster: most over-represented orig_slice and its
                         share vs that slide's share of all cells; sections
                         holding half the cluster

--stability reclusters at resolutions 0.3 / 0.5 / 1.0 and seeds 0 / 1 and prints
the cluster count, largest-cluster share and adjusted Rand index against
--assignments. Resolution 0.5 / seed 0 should give ARI 1.0 if the CSV came from
this code. Needs scikit-learn, leidenalg, python-igraph.
"""

import argparse
import os

import numpy as np
import pandas as pd

from make_slice_figures import NON_GENE_COLUMNS, align, feature_mask, leiden, load_mat, pc_space


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mat", required=True)
    ap.add_argument("--assignments", required=True)
    ap.add_argument("--outdir", default="clustering_check")
    ap.add_argument("--exclude-columns", type=int, nargs="*", default=NON_GENE_COLUMNS)
    ap.add_argument("--stability", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out = lambda f: os.path.join(args.outdir, f)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 200)
    pd.set_option("display.max_rows", 200)

    d = load_mat(args.mat)
    assign = pd.read_csv(args.assignments)
    genes = d["genes"]
    row_of = pd.Series(np.arange(len(d["id"])), index=d["id"])
    rows = row_of.loc[assign["cell_id"].values].values
    E = d["E"][rows]
    clusters = assign["cluster"].values
    order = sorted(set(clusters), key=lambda c: int(c[1:]))

    # ---- panel
    feat = feature_mask(genes, args.exclude_columns)
    panel = pd.DataFrame({
        "col": np.arange(len(genes)), "gene": genes, "feature": feat,
        "pct_detected": (100 * np.asarray((E > 0).mean(0)).ravel()).round(2),
        "mean_count": np.asarray(E.mean(0)).ravel().round(3),
    })
    panel.to_csv(out("panel_genes.csv"), index=False)
    print("\n== panel (%d columns, %d used as features)" % (len(genes), feat.sum()))
    print(panel.to_string(index=False))

    # ---- PCs
    keep, _, cellpc, Vt, var, fnames = pc_space(d, args.exclude_columns)
    print("\n== PCA: %d QC cells now vs %d in the CSV" % (keep.sum(), len(assign)))
    pcs = pd.DataFrame({
        "pc": np.arange(1, 11), "var_pct": (100 * var[:10]).round(2),
        "top_genes": [", ".join("%s%+.2f" % (fnames[j], Vt[i, j])
                                for j in np.argsort(-np.abs(Vt[i]))[:6]) for i in range(10)],
    })
    pcs.to_csv(out("pc_loadings.csv"), index=False)
    print(pcs.to_string(index=False))

    # ---- per-cluster depth
    reads = np.asarray(E.sum(1)).ravel()
    ngen = np.diff(E.tocsr().indptr)
    qc = pd.DataFrame({"cluster": clusters, "reads": reads, "genes": ngen}).groupby("cluster") \
        .agg(n=("reads", "size"), median_reads=("reads", "median"),
             median_genes=("genes", "median")).loc[order]
    qc.to_csv(out("cluster_qc.csv"))
    print("\n== depth per cluster (all cells: median reads %.0f, genes %.0f)"
          % (np.median(reads), np.median(ngen)))
    print(qc.to_string())

    # ---- top genes, no marker table
    Ln, gnames = align(d, assign, args.exclude_columns)
    det = Ln > 0
    tops = []
    for c in order:
        m = clusters == c
        diff = Ln[m].mean(0) - Ln[~m].mean(0)
        pin, pout = 100 * det[m].mean(0), 100 * det[~m].mean(0)
        idx = np.argsort(-diff)[:10]
        tops.append({"cluster": c, "n": int(m.sum()),
                     "top_genes": ", ".join("%s %+.2f (%.0f/%.0f%%)" % (gnames[j], diff[j], pin[j], pout[j])
                                            for j in idx)})
    tops = pd.DataFrame(tops)
    tops.to_csv(out("cluster_top_genes.csv"), index=False)
    print("\n== top genes per cluster: log1p mean difference vs rest (% detected in/out)")
    for r in tops.itertuples():
        print("%-4s n=%-6d %s" % (r.cluster, r.n, r.top_genes))

    # ---- slide / section concentration
    overall = assign["orig_slice"].value_counts(normalize=True)
    rows_ = []
    for c in order:
        a = assign[assign["cluster"] == c]
        share = a["orig_slice"].value_counts(normalize=True)
        ratio = (share / overall.loc[share.index]).sort_values(ascending=False)
        sec = a["slice"].value_counts(normalize=True).cumsum()
        rows_.append({"cluster": c, "n": len(a),
                      "top_slide": int(ratio.index[0]),
                      "slide_share_pct": round(100 * share.loc[ratio.index[0]], 1),
                      "slide_share_all_pct": round(100 * overall.loc[ratio.index[0]], 1),
                      "enrichment": round(ratio.iloc[0], 2),
                      "sections_for_half": int((sec < 0.5).sum() + 1),
                      "n_sections": a["slice"].nunique()})
    sl = pd.DataFrame(rows_)
    sl.to_csv(out("cluster_by_slide.csv"), index=False)
    print("\n== slide / section concentration (%d slides, %d sections)"
          % (assign["orig_slice"].nunique(), assign["slice"].nunique()))
    print(sl.to_string(index=False))

    # ---- stability
    if args.stability:
        from sklearn.metrics import adjusted_rand_score
        saved = assign.set_index("cell_id").loc[d["id"][keep], "cluster"].values
        print("\n== stability vs the CSV")
        print("resolution  seed  n_clusters  largest_%  ARI")
        for res in (0.3, 0.5, 1.0):
            for seed in (0, 1):
                lab = leiden(cellpc, res, seed=seed)
                print("%10.1f  %4d  %10d  %9.1f  %.3f"
                      % (res, seed, lab.max() + 1, 100 * np.bincount(lab).max() / len(lab),
                         adjusted_rand_score(saved, lab)))
    print("\nwrote CSVs to %s" % args.outdir)


if __name__ == "__main__":
    main()
