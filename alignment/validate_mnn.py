"""
MNN Alignment Validation - Cellpose centroids + rigid transform.

Validates a castalign rigid alignment by:
  1. Loading cellpose _seg.npy masks for ex-vivo and in-vivo volumes
  2. Computing per-label centroids in voxel coordinates
  3. Applying the fitted rigid transform (invivo_ref -> ex_vivo_block) to in-vivo centroids
  4. Mutual-nearest-neighbor (MNN) match against ex-vivo centroids in voxel space
  5. Reporting the MNN distance distribution in voxel and um units

MVP scope. No size filter, no null shuffle, no pre-vs-post nonlinear comparison
(see project_alignment_validation_mnn.md for the deferred publication-grade patches).

Both volumes are assumed to be Huang lab 2P (same voxel pitch). Distances are
reported in both voxel and um; per-axis um is also reported because Z pitch is
~1.8x XY pitch.

CLI:
    python alignment/validate_mnn.py
    python alignment/validate_mnn.py --graph-path ~/castalign_graphs/by94_graph.db
    python alignment/validate_mnn.py --exvivo-seg .../exvivo_seg.npy --invivo-seg .../invivo_seg.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


HUANG_UM_PER_VOXEL = (2.0, 1.1055, 1.1055)
# Validation operates on the red (sparse) alignment channel by design — it's the
# segmentation channel and the only side with a fitted rigid edge. Green is
# joined via Identity in the graph but is not segmented here.
DEFAULT_NODE_MOVABLE = "invivo_ref_red"
DEFAULT_NODE_FIXED = "ex_vivo_block_red"


# ------------------------------------------------------------------
# Centroid extraction
# ------------------------------------------------------------------

def extract_centroids(seg_npy_path: Union[str, Path]) -> dict:
    """
    Load a cellpose ``_seg.npy`` file and compute per-label centroids.

    Returns
    -------
    dict with keys
        centroids : (N, 3) float64 — voxel coords (z, y, x)
        sizes     : (N,)   int     — voxel count per label
        labels    : (N,)   int     — cellpose label IDs (for traceback)
    """
    seg_npy_path = Path(seg_npy_path)
    if not seg_npy_path.exists():
        raise FileNotFoundError(f"Seg file not found: {seg_npy_path}")

    data = np.load(seg_npy_path, allow_pickle=True).item()
    if "masks" not in data:
        raise KeyError(
            f"{seg_npy_path.name} has no 'masks' key. Keys present: {list(data.keys())}"
        )
    masks = data["masks"]

    if masks.ndim != 3:
        raise ValueError(
            f"Expected 3D masks (z, y, x), got shape {masks.shape}. "
            "MNN validation operates on 3D segmentations."
        )

    label_ids = np.unique(masks)
    label_ids = label_ids[label_ids > 0]

    if label_ids.size == 0:
        return {
            "centroids": np.zeros((0, 3), dtype=float),
            "sizes":     np.zeros(0, dtype=int),
            "labels":    np.zeros(0, dtype=int),
        }

    # Geometric centroid per label (voxel coords, z, y, x)
    centroids = np.array(
        ndimage.center_of_mass(
            np.ones(masks.shape, dtype=bool),
            labels=masks,
            index=label_ids.tolist(),
        ),
        dtype=float,
    )

    bincount = np.bincount(masks.ravel())
    sizes = bincount[label_ids]

    return {
        "centroids": centroids,
        "sizes":     sizes.astype(int),
        "labels":    label_ids.astype(int),
    }


# ------------------------------------------------------------------
# Graph + transform
# ------------------------------------------------------------------

def load_graph_transform(graph_path: Union[str, Path], src: str, dst: str):
    """
    Load the castalign graph and return the transform mapping ``src -> dst``.

    Composition via BFS is automatic (``Graph.get_transform``), so this works
    for both direct edges and multi-hop paths.
    """
    import castalign as ca

    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    g = ca.Graph.load(str(graph_path))
    if src not in g.nodes:
        raise KeyError(f"Node '{src}' not in graph. Present: {sorted(g.nodes)}")
    if dst not in g.nodes:
        raise KeyError(f"Node '{dst}' not in graph. Present: {sorted(g.nodes)}")
    return g.get_transform(src, dst)


def apply_transform_to_centroids(transform, centroids: np.ndarray) -> np.ndarray:
    """
    Apply a castalign transform to (N, 3) voxel centroids.

    Castalign's ``Transform.__call__`` dispatches to ``.transform()`` when the
    input has shape (N, 3), returning points in the destination voxel frame.
    """
    if centroids.shape[0] == 0:
        return centroids.copy()
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) centroids, got shape {centroids.shape}")
    return np.asarray(transform(centroids), dtype=float)


# ------------------------------------------------------------------
# MNN matching
# ------------------------------------------------------------------

def mnn_match(pts_movable: np.ndarray, pts_fixed: np.ndarray) -> dict:
    """
    Mutual-nearest-neighbor match between two point sets in voxel coordinates.

    Uses Euclidean distance in voxel units (as specified in the validation
    design). Per-axis um distances are returned alongside so the caller can
    compute um-Euclidean distances without re-running the tree.

    Returns
    -------
    dict with keys
        movable_idx           : (K,)    int
        fixed_idx             : (K,)    int
        distances_voxel       : (K,)    float
        distances_um_per_axis : (K, 3)  float  (um, z / y / x absolute)
    """
    if pts_movable.shape[0] == 0 or pts_fixed.shape[0] == 0:
        return {
            "movable_idx":           np.zeros(0, dtype=int),
            "fixed_idx":             np.zeros(0, dtype=int),
            "distances_voxel":       np.zeros(0),
            "distances_um_per_axis": np.zeros((0, 3)),
        }

    tree_fixed = cKDTree(pts_fixed)
    tree_movable = cKDTree(pts_movable)

    _, nn_m_to_f = tree_fixed.query(pts_movable, k=1)
    _, nn_f_to_m = tree_movable.query(pts_fixed, k=1)

    movable_idx = np.arange(len(pts_movable))
    is_mutual = nn_f_to_m[nn_m_to_f] == movable_idx

    movable_idx = movable_idx[is_mutual]
    fixed_idx = nn_m_to_f[is_mutual]

    delta = pts_movable[movable_idx] - pts_fixed[fixed_idx]
    distances_voxel = np.linalg.norm(delta, axis=1)
    distances_um_per_axis = np.abs(delta) * np.asarray(HUANG_UM_PER_VOXEL)

    return {
        "movable_idx":           movable_idx,
        "fixed_idx":             fixed_idx,
        "distances_voxel":       distances_voxel,
        "distances_um_per_axis": distances_um_per_axis,
    }


# ------------------------------------------------------------------
# Summary + reporting
# ------------------------------------------------------------------

def summarize(mnn_result: dict, n_movable: int, n_fixed: int) -> dict:
    """Population-level summary numbers from an MNN result."""
    d_vox = mnn_result["distances_voxel"]
    d_per_axis = mnn_result["distances_um_per_axis"]
    n_mnn = int(d_vox.size)

    if n_mnn == 0:
        return {
            "n_movable": int(n_movable),
            "n_fixed":   int(n_fixed),
            "n_mnn":     0,
            "mnn_fraction":      0.0,
            "median_dist_voxel": None,
            "p95_dist_voxel":    None,
            "median_dist_um":    None,
            "p95_dist_um":       None,
            "median_xy_um":      None,
            "median_z_um":       None,
        }

    d_um = np.linalg.norm(d_per_axis, axis=1)
    smaller = max(1, min(n_movable, n_fixed))

    return {
        "n_movable":         int(n_movable),
        "n_fixed":           int(n_fixed),
        "n_mnn":             n_mnn,
        "mnn_fraction":      float(n_mnn) / smaller,
        "median_dist_voxel": float(np.median(d_vox)),
        "p95_dist_voxel":    float(np.percentile(d_vox, 95)),
        "median_dist_um":    float(np.median(d_um)),
        "p95_dist_um":       float(np.percentile(d_um, 95)),
        "median_xy_um":      float(np.median(np.linalg.norm(d_per_axis[:, 1:3], axis=1))),
        "median_z_um":       float(np.median(d_per_axis[:, 0])),
    }


def print_summary(summary: dict) -> None:
    """Pretty-print a summary dict to stdout."""
    print("=" * 60)
    print("MNN VALIDATION SUMMARY")
    print("=" * 60)
    print(f"N cells (movable / in-vivo):   {summary['n_movable']}")
    print(f"N cells (fixed   / ex-vivo):   {summary['n_fixed']}")
    print(f"N mutual NN pairs:             {summary['n_mnn']}")

    if summary["n_mnn"] == 0:
        print("  (no matches — check the data / transform)")
        print("=" * 60)
        return

    print(f"MNN fraction (of smaller set): {summary['mnn_fraction']:.1%}")
    print()
    print("Distance distribution")
    print(f"  median:  {summary['median_dist_voxel']:.2f} voxel   =  {summary['median_dist_um']:.2f} um")
    print(f"  p95:     {summary['p95_dist_voxel']:.2f} voxel   =  {summary['p95_dist_um']:.2f} um")
    print()
    print("Per-axis (median um)")
    print(f"  XY: {summary['median_xy_um']:.2f} um")
    print(f"  Z:  {summary['median_z_um']:.2f} um")
    print("=" * 60)


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

def _default_seg_path(data_path: Path, cellpose_dir: Path) -> Path:
    """Resolve the ``_seg.npy`` for a TIFF.

    Checks next-to-TIFF first (where ``batch_cellpose.py`` writes), then falls
    back to the ``cellpose/`` subfolder (legacy location).
    """
    next_to_tiff = data_path.with_name(f"{data_path.stem}_seg.npy")
    if next_to_tiff.exists():
        return next_to_tiff
    return cellpose_dir / f"{data_path.stem}_seg.npy"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="MNN alignment validation (cellpose centroids + rigid transform)."
    )
    parser.add_argument(
        "--graph-path", default=None,
        help="Override GRAPH_PATH (e.g. ~/castalign_graphs/by94_graph.db for OneDrive workaround).",
    )
    parser.add_argument(
        "--exvivo-seg", default=None,
        help="Override ex-vivo cellpose _seg.npy. Default: <INVIVO parent>/cellpose/<exvivo stem>_seg.npy",
    )
    parser.add_argument(
        "--invivo-seg", default=None,
        help="Override in-vivo cellpose _seg.npy. Default: <INVIVO parent>/cellpose/<invivo stem>_seg.npy",
    )
    parser.add_argument("--node-movable", default=DEFAULT_NODE_MOVABLE)
    parser.add_argument("--node-fixed", default=DEFAULT_NODE_FIXED)
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import local_config

    invivo_path = Path(local_config.INVIVO_PATH_RED)
    exvivo_path = Path(local_config.BLOCK_STACK_PATH_RED)
    out_dir = invivo_path.parent / "cellpose"

    if args.graph_path:
        graph_path = Path(args.graph_path).expanduser()
    elif local_config.GRAPH_PATH:
        graph_path = Path(local_config.GRAPH_PATH)
    else:
        raise ValueError(
            "GRAPH_PATH is blank in local_config.py and no --graph-path was passed.\n"
            "Either set GRAPH_PATH in local_config.py or pass --graph-path."
        )

    exvivo_seg = (
        Path(args.exvivo_seg).expanduser() if args.exvivo_seg
        else _default_seg_path(exvivo_path, out_dir)
    )
    invivo_seg = (
        Path(args.invivo_seg).expanduser() if args.invivo_seg
        else _default_seg_path(invivo_path, out_dir)
    )

    print("=" * 60)
    print("MNN ALIGNMENT VALIDATION")
    print("=" * 60)
    print(f"Graph:      {graph_path}")
    print(f"Ex-vivo:    {exvivo_seg}")
    print(f"In-vivo:    {invivo_seg}")
    print(f"Output dir: {out_dir}")
    print(f"Direction:  {args.node_movable} -> {args.node_fixed}")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)

    # [1/4] Centroids
    print("[1/4] Extracting centroids")
    d_ex = extract_centroids(exvivo_seg)
    d_in = extract_centroids(invivo_seg)
    ex_size_med = float(np.median(d_ex["sizes"])) if d_ex["sizes"].size else float("nan")
    in_size_med = float(np.median(d_in["sizes"])) if d_in["sizes"].size else float("nan")
    print(f"  ex-vivo: {len(d_ex['centroids']):>6d} cells   median size {ex_size_med:.0f} voxels")
    print(f"  in-vivo: {len(d_in['centroids']):>6d} cells   median size {in_size_med:.0f} voxels")

    np.savez(
        out_dir / "exvivo_centroids.npz",
        centroids=d_ex["centroids"], sizes=d_ex["sizes"], labels=d_ex["labels"],
    )
    np.savez(
        out_dir / "invivo_centroids.npz",
        centroids=d_in["centroids"], sizes=d_in["sizes"], labels=d_in["labels"],
    )

    # [2/4] Transform
    print()
    print(f"[2/4] Loading graph and applying transform {args.node_movable} -> {args.node_fixed}")
    transform = load_graph_transform(graph_path, args.node_movable, args.node_fixed)
    print(f"  transform: {type(transform).__name__}")
    in_in_ex_frame = apply_transform_to_centroids(transform, d_in["centroids"])
    np.savez(out_dir / "invivo_centroids_in_exvivo_frame.npz", centroids=in_in_ex_frame)

    def _bbox_str(pts, label):
        if pts.shape[0] == 0:
            return f"  {label}: (empty)"
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        return (f"  {label}:  z {mn[0]:7.1f}–{mx[0]:7.1f}   "
                f"y {mn[1]:7.1f}–{mx[1]:7.1f}   x {mn[2]:7.1f}–{mx[2]:7.1f}")

    print(_bbox_str(d_ex["centroids"], "ex-vivo            "))
    print(_bbox_str(d_in["centroids"], "in-vivo (raw)      "))
    print(_bbox_str(in_in_ex_frame,     "in-vivo (in ex-frame)"))

    # [3/4] MNN
    print()
    print("[3/4] Mutual nearest-neighbor matching")
    mnn = mnn_match(in_in_ex_frame, d_ex["centroids"])
    print(f"  matches: {mnn['movable_idx'].size}")

    # [4/4] Summary
    print()
    print("[4/4] Summarizing")
    summary = summarize(mnn, n_movable=len(d_in["centroids"]), n_fixed=len(d_ex["centroids"]))

    np.savez(
        out_dir / "mnn_results.npz",
        movable_idx=mnn["movable_idx"],
        fixed_idx=mnn["fixed_idx"],
        distances_voxel=mnn["distances_voxel"],
        distances_um_per_axis=mnn["distances_um_per_axis"],
        summary=np.array(summary, dtype=object),
    )

    print()
    print_summary(summary)
    print()
    print(f"Outputs written to: {out_dir}")
    for fn in (
        "exvivo_centroids.npz",
        "invivo_centroids.npz",
        "invivo_centroids_in_exvivo_frame.npz",
        "mnn_results.npz",
    ):
        print(f"  - {fn}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
