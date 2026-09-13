"""
BARseq-style 114-gene mouse cortical panel: cell clustering and gene co-expression modules.

Input  : filt_neurons.mat  (MATLAB v5; one struct `filt_neurons`)
Outputs (files written by this script):
         cell_clusters_blind.png        - section 5
         cell_cluster_summary.csv       - section 5
         cell_cluster_assignments.csv   - section 5
         gene_modules_specificity.csv   - section 6

Sections 6 and 7 additionally print the gene-module silhouette scan, the
between-method adjusted Rand indices, and the 15-NN preservation of each 2-D
view; they do not write figures. The other figures shown in the originating
session (the gene-module correlation heatmap, the module-by-cell-type heatmap,
the annotated cell UMAP, the PCA-vs-UMAP panel and the method-comparison panel)
were rendered in ad-hoc cells and are not reproduced here.

Environment: python 3.11 + numpy, scipy, pandas, matplotlib, scikit-learn,
             leidenalg, python-igraph, umap-learn

Note: in the original session the figure styling came from the `figure-style` kernel
plugin (apply_figure_style / META_GREY). Those are inlined below as rcParams so this
file is self-contained.
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
import igraph as ig
import leidenalg as la
import umap
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

MAT_PATH = "filt_neurons.mat"          # artifact af23c83d-7ef2-4721-9613-c7ff9da657ee
META_GREY = "#6E6E6E"
mpl.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.grid": False,
})

# ----------------------------------------------------------------------------- 1. load
m = sio.loadmat(MAT_PATH, squeeze_me=False, struct_as_record=False)
s = m["filt_neurons"][0, 0]

E       = s.expmat.tocsr()                       # 557,554 cells x 114 barcodes, integer counts
genes   = np.array([g[0][0] for g in s.genes])   # barcode names (109 real + 5 'unused' blanks)
pos     = s.pos                                  # per-slice imaging pixels (each slice starts at 1)
sl      = s.slice.ravel()                        # 1..62, NaN for unassigned cells
oslice  = s.orig_slice.ravel()                   # 1..8 imaging blocks
cell_id = s.id.ravel()
# s.depth is all zeros and s.angle is a constant 180 -> neither carries information

tot   = np.asarray(E.sum(1)).ravel()
ndet  = np.diff(E.indptr)
blank = np.array([g.startswith("unused") for g in genes])

# ------------------------------------------------------------------- 2. QC and features
qc   = (tot >= 20) & (ndet >= 5) & (~np.isnan(sl))      # -> 147,176 cells
feat = ~blank & (genes != "Actb")                        # -> 108 feature genes
Xq   = E[qc][:, feat].tocsr()
gf   = genes[feat]
gfi  = {g: i for i, g in enumerate(gf)}
totq = tot[qc]

# --------------------------------------------- 3. normalise, log1p, z-score, 25-PC space
tc = np.asarray(Xq.sum(1)).ravel()
Xl = (sp.diags(np.median(tc) / tc) @ Xq).tocsr()         # scale each cell to median depth
Xl.data = np.log1p(Xl.data)
Ln = np.asarray(Xl.todense())                            # log-normalised, un-scaled
M  = (Ln - Ln.mean(0)) / Ln.std(0)                       # z-score per gene

U, S, Vt = np.linalg.svd(M / np.sqrt(M.shape[0] - 1), full_matrices=False)
cellpc = U[:, :25] * S[:25]
pcv    = (S ** 2 / np.sum(S ** 2)) * 100                 # PC1 3.5%, PC2 2.2%, PC1-25 31.9%

# ------------------------------------------------------- 4. kNN graph, Leiden, UMAP
knn = NearestNeighbors(n_neighbors=15, n_jobs=-1).fit(cellpc).kneighbors_graph(
    cellpc, mode="connectivity")
src, dst = knn.nonzero()
g = ig.Graph(n=cellpc.shape[0], edges=list(zip(src.tolist(), dst.tolist())), directed=False)
g.simplify()

def leiden(res, seed=0):
    return np.array(la.find_partition(
        g, la.RBConfigurationVertexPartition, resolution_parameter=res, seed=seed).membership)

def by_size(a):
    o = np.argsort(-np.bincount(a)); rm = np.empty(a.max() + 1, int); rm[o] = np.arange(len(o))
    return rm[a]

cl29 = leiden(1.5)                      # 29 clusters - used for the gene-module annotation
cl12 = by_size(leiden(0.5))             # 12 clusters - the cell-type-blind partition
K    = cl12.max() + 1
n12  = np.bincount(cl12)

emb  = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(cellpc)
emb2 = umap.UMAP(n_neighbors=50, min_dist=0.05, random_state=42).fit_transform(cellpc)

# --------------------------------------------------- 5. blind clusters: profiles + outputs
PB12 = np.vstack([Ln[cl12 == c].mean(0) for c in range(K)]).T
Z12  = (PB12 - PB12.mean(1, keepdims=True)) / PB12.std(1, keepdims=True)

show_g, seen = [], set()
for c in range(K):
    for gg in gf[np.argsort(-Z12[:, c])[:4]]:
        if gg not in seen:
            seen.add(gg); show_g.append(gg)
gidx = [gfi[gg] for gg in show_g]

pd.DataFrame({
    "cluster": [f"C{c+1}" for c in range(K)],
    "n_cells": n12,
    "pct_cells": (100 * n12 / n12.sum()).round(2),
    "median_counts": [int(np.median(totq[cl12 == c])) for c in range(K)],
    "top10_genes": [", ".join(gf[np.argsort(-Z12[:, c])[:10]]) for c in range(K)],
}).to_csv("cell_cluster_summary.csv", index=False)

pd.DataFrame({
    "cell_id": cell_id[qc], "slice": sl[qc], "orig_slice": oslice[qc],
    "pos_x": pos[qc, 0], "pos_y": pos[qc, 1], "total_counts": totq,
    "cluster": [f"C{c+1}" for c in cl12],
    "umap1": emb[:, 0].round(3), "umap2": emb[:, 1].round(3),
}).to_csv("cell_cluster_assignments.csv", index=False)

cmap12 = [mpl.colormaps["tab20"](i) for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3]]
rs = np.random.default_rng(0).permutation(len(cl12))     # draw order, avoids overplot bias

def bare_axes(ax, x="UMAP1", y="UMAP2"):
    ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.04)
    for spn in ax.spines.values():
        spn.set_visible(False)
    for xy, tgt in [((0.075, 0.02), (0.02, 0.02)), ((0.02, 0.075), (0.02, 0.02))]:
        ax.annotate("", xy=xy, xytext=tgt, xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=META_GREY))
    ax.text(0.085, 0.017, x, fontsize=5.5, color=META_GREY, transform=ax.transAxes, va="center")
    ax.text(0.017, 0.085, y, fontsize=5.5, color=META_GREY, transform=ax.transAxes,
            rotation=90, ha="center")

fig = plt.figure(figsize=(13.4, 5.6))
axa = fig.add_axes([0.035, 0.06, 0.44, 0.86])
axb = fig.add_axes([0.565, 0.30, 0.30, 0.62])
cax = fig.add_axes([0.885, 0.42, 0.010, 0.34])
axa.scatter(emb[rs, 0], emb[rs, 1], s=0.35,
            c=np.array([cmap12[c] for c in cl12])[rs], linewidths=0, rasterized=True)
for c in range(K):
    med = np.median(emb[cl12 == c], axis=0)
    axa.text(med[0], med[1], f"C{c+1}", fontsize=9, weight="bold", color=cmap12[c],
             ha="center", va="center", zorder=6,
             path_effects=[pe.withStroke(linewidth=3, foreground="white")])
bare_axes(axa)
axa.set_title("a  147,176 cells, 12 unsupervised clusters (Leiden, res 0.5)", fontsize=9, loc="left")
im = axb.imshow(Z12[gidx].T, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto", interpolation="nearest")
axb.set_yticks(range(K)); axb.set_yticklabels([f"C{c+1}  (n={n12[c]:,})" for c in range(K)], fontsize=6.5)
axb.set_xticks(range(len(show_g)))
axb.set_xticklabels([f"$\\it{{{gg}}}$" for gg in show_g], rotation=90, fontsize=5.2)
for c in range(K):
    axb.get_yticklabels()[c].set_color(cmap12[c])
axb.set_title("b  Top-4 genes per cluster by expression z-score", fontsize=9, loc="left")
cb = fig.colorbar(im, cax=cax); cb.set_label("expression z-score across clusters", fontsize=6.5)
cb.set_ticks([-3, 0, 3]); cb.ax.tick_params(labelsize=6)
fig.savefig("cell_clusters_blind.png", dpi=300, bbox_inches="tight")

# ------------------------------------------------------------- 6. gene co-expression modules
# Gene-gene correlation across single cells is uninformative here (r -0.27..0.35,
# silhouette ~0.02 at every k). Aggregating into metacells first is what makes it work.
mc = MiniBatchKMeans(n_clusters=250, random_state=0, n_init=5, batch_size=4096).fit_predict(cellpc)
MC = np.vstack([Ln[mc == c].mean(0) for c in range(250)]).T      # genes x 250 metacells

Cm = np.corrcoef(MC)                                             # r -0.65..0.88
Dm = 1 - Cm; np.fill_diagonal(Dm, 0); Dm = (Dm + Dm.T) / 2
Zm = linkage(squareform(Dm, checks=False), method="average")
for k in range(3, 13):                                           # k=4 best (silhouette 0.123)
    lab = fcluster(Zm, k, criterion="maxclust")
    print(k, np.bincount(lab)[1:], round(silhouette_score(Dm, lab, metric="precomputed"), 3))
mod    = fcluster(Zm, 4, criterion="maxclust")
gorder = leaves_list(Zm)

# EWCE-style specificity on linear pseudobulk means over the 29 Leiden clusters
PBlin = np.vstack([np.expm1(Ln[cl29 == c]).mean(0) for c in range(cl29.max() + 1)]).T
prop  = PBlin / np.maximum(PBlin.sum(1, keepdims=True), 1e-12)
spec, topc = prop.max(1), prop.argmax(1)
x     = PBlin / np.maximum(PBlin.max(1, keepdims=True), 1e-12)
tau   = (1 - x).sum(1) / (PBlin.shape[1] - 1)
det   = np.asarray((Xq > 0).sum(0)).ravel() / Xq.shape[0]
pd.DataFrame({
    "gene": gf, "module": mod, "frac_cells_detected": det.round(4),
    "mean_counts_per_cell": np.asarray(Xq.mean(0)).ravel().round(3),
    "specificity_index": spec.round(4), "tau": tau.round(4),
    "top_cell_cluster": [f"c{t}" for t in topc],
    "specificity_reliable": det >= 0.01,        # 13/108 genes fall below this and take
}).to_csv("gene_modules_specificity.csv", index=False)   # spuriously perfect specificity

# ------------------------------------------------------------ 7. method / view comparisons
km = by_size(KMeans(n_clusters=12, n_init=10, random_state=0).fit_predict(cellpc))
gm = by_size(GaussianMixture(n_components=12, covariance_type="diag",
                             random_state=0, max_iter=200).fit_predict(cellpc))
print("ARI Leiden/k-means %.3f  Leiden/GMM %.3f  k-means/GMM %.3f" % (
    adjusted_rand_score(cl12, km), adjusted_rand_score(cl12, gm), adjusted_rand_score(km, gm)))

# 15-NN preservation of each 2-D view relative to the 25-PC space (20k subsample)
sub = np.random.default_rng(3).choice(len(cl12), 20000, replace=False)
ref = NearestNeighbors(n_neighbors=16, n_jobs=-1).fit(cellpc[sub]).kneighbors(cellpc[sub])[1][:, 1:]
def knn_overlap(XY):
    nb = NearestNeighbors(n_neighbors=16, n_jobs=-1).fit(XY[sub]).kneighbors(XY[sub])[1][:, 1:]
    return np.mean([len(set(a) & set(b)) / 15 for a, b in zip(nb, ref)])
print("15-NN preservation: 2-D PCA %.3f  2-D UMAP %.3f" % (
    knn_overlap(U[:, :2] * S[:2]), knn_overlap(emb)))
