#!/usr/bin/env python3
"""
Regenerate the per-section cluster/layer figures for a BARseq-style cortical panel.

Usage
-----
    python make_slice_figures.py --mat filt_neurons.mat \
                                 --assignments cell_cluster_assignments.csv \
                                 --outdir figures

    # no assignments file on hand: recompute the 12 clusters from scratch
    # (needs scikit-learn + leidenalg + python-igraph; cluster NUMBERING may
    #  differ slightly from the saved run if library versions differ)
    python make_slice_figures.py --mat filt_neurons.mat --recluster

Inputs
------
filt_neurons.mat            MATLAB v5 struct with sparse `expmat` (cells x 114
                            barcodes), `id`, `pos`, `slice`, `orig_slice`.
cell_cluster_assignments.csv  cell_id, slice, orig_slice, pos_x, pos_y,
                            total_counts, cluster (C1-C12), umap1, umap2.

Outputs (written to --outdir)
-----------------------------
    clusters_to_layers.png            marker z-scores per cluster + layer calls
    cluster_layer_assignment.csv      layer call, score, runner-up, margin
    clusters_by_slice_spatial.png     9 largest sections, coloured by cluster
    layers_by_slice_spatial.png       9 largest sections, coloured by layer class
    layers_all_slices.png             all sections, coloured by layer class
    layers_by_slice_composition.png   layer shares per section (>=500 cells)

Requires: numpy, scipy, pandas, matplotlib
          (+ scikit-learn, leidenalg, python-igraph only for --recluster)
"""

import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- constants
GREY = "#6E6E6E"

# canonical mouse-cortex laminar markers, restricted to this panel
LAYER_MARKERS = {
    "L2/3 IT":    ["Cux2", "Calb1", "Lamp5", "Rasgrf2", "Otof", "Enpp2", "Cbln2", "C1ql3"],
    "L4/5 IT":    ["Rorb", "Scnn1a"],
    "L5 PT/ET":   ["Fezf2", "Pcp4", "Etv1", "Hcn1"],
    "L5 NP":      ["Tshz2", "Nxph1"],
    "L6 CT":      ["Foxp2", "Tle4"],
    "L6 IT Car3": ["Car3", "Oprk1"],
    "L6b":        ["Cplx3", "Ccn2"],
    "Inhibitory": ["Gad1", "Synpr"],
}

# one saturated hue per class; no two tints of the same hue
LAYER_COLOURS = {
    "L2/3 IT": "#0072B2", "L4/5 IT": "#009E73", "L5 PT/ET": "#D81B7A",
    "L5 NP": "#000000", "L6 CT": "#56B4E9", "L6 IT Car3": "#CC79A7",
    "L6b": "#E69F00", "Inhibitory": "#666666", "unresolved": "#BBBBBB",
}
CLUSTER_COLOURS = [mpl.colormaps["tab20"](i) for i in
                   [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3]]

# margin thresholds for the layer call
CONFIDENT = 0.45
TENTATIVE = 0.20
MIN_CELLS_FOR_COMPOSITION = 500
MARKS = {2: "", 1: " \u00b7", 0: " ?"}


def style():
    mpl.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


# --------------------------------------------------------------------- input
def load_mat(path, struct_name="filt_neurons"):
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=False)
    s = m[struct_name][0, 0]
    return {"E": s.expmat.tocsr(),
            "genes": np.array([g[0][0] for g in s.genes]),
            "id": s.id.ravel(),
            "pos": s.pos,
            "slice": s.slice.ravel(),
            "orig_slice": s.orig_slice.ravel()}


def recluster(d, resolution=0.5, n_pcs=25, n_neighbors=15, seed=0):
    """Reproduce the 12-cluster partition from the raw matrix."""
    import igraph as ig
    import leidenalg as la
    from sklearn.neighbors import NearestNeighbors

    E, genes = d["E"], d["genes"]
    tot = np.asarray(E.sum(1)).ravel()
    ndet = np.diff(E.indptr)
    keep = (tot >= 20) & (ndet >= 5) & (~np.isnan(d["slice"]))
    feat = np.array([not str(g).lower().startswith("unused") for g in genes]) & (genes != "Actb")

    X = E[keep][:, feat].tocsr()
    tc = np.asarray(X.sum(1)).ravel()
    Xl = (sp.diags(np.median(tc) / np.maximum(tc, 1)) @ X).tocsr()
    Xl.data = np.log1p(Xl.data)
    Ln = np.asarray(Xl.todense())
    M = (Ln - Ln.mean(0)) / Ln.std(0)
    U, S, _ = np.linalg.svd(M / np.sqrt(M.shape[0] - 1), full_matrices=False)
    cellpc = U[:, :n_pcs] * S[:n_pcs]

    nb = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(cellpc)
    src, dst = nb.kneighbors_graph(cellpc, mode="connectivity").nonzero()
    g = ig.Graph(n=cellpc.shape[0], edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    g.simplify()
    lab = np.array(la.find_partition(g, la.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution, seed=seed).membership)
    o = np.argsort(-np.bincount(lab))
    rm = np.empty(lab.max() + 1, int)
    rm[o] = np.arange(len(o))
    lab = rm[lab]

    return pd.DataFrame({
        "cell_id": d["id"][keep], "slice": d["slice"][keep],
        "orig_slice": d["orig_slice"][keep],
        "pos_x": d["pos"][keep, 0], "pos_y": d["pos"][keep, 1],
        "total_counts": tot[keep],
        "cluster": [f"C{c + 1}" for c in lab],
    })


def align(d, assign):
    """Row-align the .mat to the assignment table on cell_id; return log-norm matrix."""
    row_of = pd.Series(np.arange(len(d["id"])), index=d["id"])
    missing = ~assign["cell_id"].isin(row_of.index)
    if missing.any():
        raise ValueError("%d cell_ids in the assignments file are absent from the .mat"
                         % int(missing.sum()))
    rows = row_of.loc[assign["cell_id"].values].values
    genes = d["genes"]
    feat = np.array([not str(g).lower().startswith("unused") for g in genes]) & (genes != "Actb")
    X = d["E"][rows][:, feat].tocsr()
    tc = np.asarray(X.sum(1)).ravel()
    Xl = (sp.diags(np.median(tc) / np.maximum(tc, 1)) @ X).tocsr()
    Xl.data = np.log1p(Xl.data)
    return np.asarray(Xl.todense()), genes[feat]


# ---------------------------------------------------------------- layer call
def layer_assignment(Ln, gene_names, clusters):
    """Mean marker z-score per cluster -> best layer, runner-up, margin."""
    gi = {g: i for i, g in enumerate(gene_names)}
    lab = clusters.str.slice(1).astype(int).values - 1
    K = lab.max() + 1
    PB = np.vstack([Ln[lab == c].mean(0) for c in range(K)]).T
    Z = (PB - PB.mean(1, keepdims=True)) / PB.std(1, keepdims=True)

    present = {k: [g for g in v if g in gi] for k, v in LAYER_MARKERS.items()}
    missing = {k: [g for g in v if g not in gi] for k, v in LAYER_MARKERS.items()}
    for k, v in missing.items():
        if v:
            print("  note: %s markers not on this panel, ignored: %s" % (k, ", ".join(v)))

    score = pd.DataFrame({k: Z[[gi[g] for g in v]].mean(0) for k, v in present.items()},
                         index=[f"C{c + 1}" for c in range(K)])
    best = score.idxmax(1)
    bestv = score.max(1)
    runner = score.apply(lambda r: r.drop(r.idxmax()).idxmax(), axis=1)
    runnerv = score.apply(lambda r: r.drop(r.idxmax()).max(), axis=1)
    margin = bestv - runnerv
    tier = np.where(margin >= CONFIDENT, 2, np.where(margin >= TENTATIVE, 1, 0))

    table = pd.DataFrame({
        "cluster": score.index, "n_cells": np.bincount(lab),
        "best_layer": best.values, "score": bestv.values.round(2),
        "runner_up": runner.values, "runner_up_score": runnerv.values.round(2),
        "margin": margin.values.round(2), "tier": tier,
        "top_genes": [", ".join(gene_names[np.argsort(-Z[:, c])[:5]]) for c in range(K)],
    })
    return Z, score, table, present


# ------------------------------------------------------------------- figures
def fig_layer_call(Z, table, present, gene_names, path):
    gi = {g: i for i, g in enumerate(gene_names)}
    rows, bands = [], []
    for L, gs in present.items():
        bands.append((L, len(rows), len(rows) + len(gs)))
        rows += gs
    K = len(table)
    labels = [f"{table['cluster'][i]}  {table['best_layer'][i]}{MARKS[table['tier'][i]]}"
              for i in range(K)]

    fig = plt.figure(figsize=(9.6, 8.0))
    ax = fig.add_axes([0.20, 0.30, 0.66, 0.62])
    cax = fig.add_axes([0.30, 0.135, 0.26, 0.016])
    im = ax.imshow(Z[[gi[g] for g in rows]], cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                   aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"$\\it{{{g}}}$" for g in rows], fontsize=7)
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
    for i in range(K):
        ax.get_xticklabels()[i].set_color(["#B03030", GREY, "black"][table["tier"][i]])
    for L, a, b in bands:
        if a:
            ax.axhline(a - 0.5, color="k", lw=0.8)
        ax.text(-0.52, (a + b - 1) / 2, L, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7.5)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-2.5, 0, 2.5])
    cb.set_label("expression z-score across clusters", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    n_called = int((table["tier"] >= 1).sum())
    ax.set_title("Canonical laminar markers assign a layer to %d of %d clusters\n"
                 "black = margin \u2265%.2f,  grey \u00b7 = %.2f\u2013%.2f,  red ? = below %.2f (unresolved)"
                 % (n_called, K, CONFIDENT, TENTATIVE, CONFIDENT, TENTATIVE),
                 fontsize=9, loc="left", pad=8)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _section_panel(ax, d, colours, s_, flag=False):
    rng = np.random.default_rng(0)
    o = rng.permutation(len(d))
    ax.scatter(d["pos_x"].values[o], d["pos_y"].values[o], s=1.6,
               c=[colours[k] for k in d["_key"].values[o]], linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)


def fig_sections(assign, key, colours, legend, title, path, slices, ncol, figsize,
                 dpi=200, flag_below=None, meta=None):
    assign = assign.copy()
    assign["_key"] = assign[key]
    nrow = int(np.ceil(len(slices) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, s_ in zip(axes.ravel(), slices):
        d = assign[assign["slice"] == s_]
        ax.set_axis_on()
        _section_panel(ax, d, colours, s_)
        thin = flag_below is not None and len(d) < flag_below
        blk = "" if meta is None else "  (block %d)" % int(meta.loc[s_, "orig"])
        ax.set_title("slice %d%s   n=%s%s" % (s_, blk, format(len(d), ","), "  \u2020" if thin else ""),
                     fontsize=7 if ncol > 4 else 8.5, loc="left",
                     color="#B03030" if thin else "black")
    h = [plt.Line2D([], [], marker="o", ls="", ms=7, color=colours[k], label=v)
         for k, v in legend]
    fig.legend(handles=h, frameon=False, fontsize=8, loc="lower center",
               ncol=2 if len(h) <= 6 else 3, handletextpad=0.4, columnspacing=2.0,
               labelspacing=0.5, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(title, fontsize=9.5 if ncol <= 4 else 11, x=0.014, ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 0.945])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig_composition(assign, groups, colours, meta, path):
    n_per = assign.groupby("slice").size()
    keep = n_per[n_per >= MIN_CELLS_FOR_COMPOSITION].index
    order = meta.loc[keep].sort_values(["orig", "n"], ascending=[True, False]).index.tolist()
    comp = pd.crosstab(assign["slice"], assign["layer_group"], normalize="index")
    comp = comp.reindex(columns=groups, fill_value=0.0).loc[order]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(11.6, 4.4))
    bottom = np.zeros(len(order))
    for gname in groups:
        v = comp[gname].values * 100
        ax.bar(x, v, bottom=bottom, color=colours[gname], width=0.8, label=gname)
        bottom += v
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of section\u2019s cells (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s_) for s_ in order], rotation=90, fontsize=6.5)
    ax.set_xlabel("section, grouped by imaging block")
    for b in np.where(np.diff(meta.loc[order, "orig"].values) != 0)[0]:
        ax.axvline(b + 0.5, color="white", lw=0.9, ls=":")
    ax.legend(frameon=False, fontsize=7.5, ncol=len(groups), loc="upper center",
              bbox_to_anchor=(0.5, -0.30), handletextpad=0.3, columnspacing=1.2)
    ax.set_title("Layer-class composition across the %d sections with \u2265%d usable cells"
                 % (len(order), MIN_CELLS_FOR_COMPOSITION), fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return comp


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mat", required=True, help="filt_neurons.mat")
    ap.add_argument("--assignments", help="cell_cluster_assignments.csv")
    ap.add_argument("--recluster", action="store_true",
                    help="recompute clusters from the .mat instead of reading --assignments")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--struct", default="filt_neurons", help="name of the MATLAB struct")
    ap.add_argument("--n-sections", type=int, default=9,
                    help="how many of the largest sections to show in the detail figures")
    args = ap.parse_args()

    if not args.recluster and not args.assignments:
        ap.error("pass --assignments, or --recluster to compute them")
    os.makedirs(args.outdir, exist_ok=True)
    out = lambda f: os.path.join(args.outdir, f)
    style()

    print("loading %s" % args.mat)
    d = load_mat(args.mat, args.struct)

    if args.recluster:
        print("reclustering (Leiden, resolution 0.5, 25 PCs, 15-NN)")
        assign = recluster(d)
        assign.to_csv(out("cell_cluster_assignments.csv"), index=False)
    else:
        assign = pd.read_csv(args.assignments)
    assign["slice"] = assign["slice"].astype(int)
    print("  %s cells, %d sections, %d clusters"
          % (format(len(assign), ","), assign["slice"].nunique(), assign["cluster"].nunique()))

    Ln, gene_names = align(d, assign)
    Z, score, table, present = layer_assignment(Ln, gene_names, assign["cluster"])
    table.to_csv(out("cluster_layer_assignment.csv"), index=False)
    fig_layer_call(Z, table, present, gene_names, out("clusters_to_layers.png"))
    print(table[["cluster", "n_cells", "best_layer", "score", "runner_up", "margin"]]
          .to_string(index=False))

    # collapse to layer classes; clusters below the tentative margin stay separate
    grp_of = {table["cluster"][i]: (table["best_layer"][i] if table["tier"][i] >= 1
                                   else "unresolved") for i in range(len(table))}
    assign["layer_group"] = assign["cluster"].map(grp_of)
    gn = assign["layer_group"].value_counts()
    depth_order = list(LAYER_MARKERS.keys()) + ["unresolved"]
    groups = [g for g in depth_order if g in gn.index]
    members = {g: ", ".join(sorted([c for c, v in grp_of.items() if v == g],
                                   key=lambda s: int(s[1:]))) for g in groups}

    meta = assign.groupby("slice").agg(n=("cluster", "size"), orig=("orig_slice", "first"))
    n_per = assign.groupby("slice").size().sort_values(ascending=False)
    top = n_per.head(args.n_sections).index.tolist()
    frame_note = ("each section has its own pixel coordinate frame, so panel sizes "
                  "and orientations are not comparable")

    K = len(table)
    cluster_cols = {f"C{i + 1}": CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)] for i in range(K)}
    cluster_leg = [(table["cluster"][i],
                    "%s  %s%s  (%s)" % (table["cluster"][i], table["best_layer"][i],
                                        MARKS[table["tier"][i]],
                                        format(int(table["n_cells"][i]), ",")))
                   for i in range(K)]
    fig_sections(assign, "cluster", cluster_cols, cluster_leg,
                 "Expression clusters mapped onto the %d largest sections\n"
                 "layer labels are marker-based:  \u00b7 = weak margin,  ? = unresolved;  %s"
                 % (len(top), frame_note),
                 out("clusters_by_slice_spatial.png"), top, 3, (12.6, 12.6), dpi=220)

    layer_leg = [(g, "%s  (%s,  %s)" % (g, format(int(gn[g]), ","), members[g])) for g in groups]
    fig_sections(assign, "layer_group", LAYER_COLOURS, layer_leg,
                 "Clusters grouped into marker-based layer classes, %d largest sections\n"
                 "clusters with a layer-call margin below %.2f are kept separate as "
                 "\u201cunresolved\u201d, not merged;  %s" % (len(top), TENTATIVE, frame_note),
                 out("layers_by_slice_spatial.png"), top, 3, (12.6, 12.2), dpi=220)

    all_slices = meta.sort_values(["orig", "n"]).index.tolist()
    thin_n = int((meta["n"] < MIN_CELLS_FOR_COMPOSITION).sum())
    thin_cells = int(meta.loc[meta["n"] < MIN_CELLS_FOR_COMPOSITION, "n"].sum())
    ncol_all = int(np.ceil(np.sqrt(len(all_slices))))
    fig_sections(assign, "layer_group", LAYER_COLOURS, layer_leg,
                 "All %d sections, cells coloured by marker-based layer class\n"
                 "\u2020 = fewer than %d usable cells (%d sections, %s cells total);  %s"
                 % (len(all_slices), MIN_CELLS_FOR_COMPOSITION, thin_n,
                    format(thin_cells, ","), frame_note),
                 out("layers_all_slices.png"), all_slices, ncol_all,
                 (2.4 * ncol_all, 2.45 * int(np.ceil(len(all_slices) / ncol_all)) + 1.0),
                 dpi=170, flag_below=MIN_CELLS_FOR_COMPOSITION, meta=meta)

    comp = fig_composition(assign, groups, LAYER_COLOURS, meta,
                           out("layers_by_slice_composition.png"))
    print("\nlayer-class share per section (%%): \n%s"
          % (comp * 100).agg(["min", "max"]).round(1).T.to_string())
    print("\nwrote 6 files to %s/" % args.outdir)


if __name__ == "__main__":
    main()
