"""
Cellpose labels of a red 2P node -> NeuroReg block cells.

Reads ``<ANALYSIS_ROOT>/cellpose/<stem>_masks.tif`` (written by batch_cellpose.py)
and writes ``<stem>_neuroreg.mat`` beside it, in the form NeuroReg's
neuroReg.detectCells3 returns and the NeuromicsCellDetection-dtc run file
(dtc_run_BY95.m, BLOCK_CELLS = 'cellpose') loads in place of it:

  pt_list_vol    3 x N, um, [x; y; z] = [column; plane; row] — neuroReg.loadTiff's
                 grid, (1:n) * pitch, so a 0-indexed voxel (z, y, x) sits at
                 [(x+1)*XY; (z+1)*Z; (y+1)*XY]
  pt_area_vol    1 x N, cell volume in um^3 (voxel count * XY * XY * Z)
  cellpose_label 1 x N, label in the masks tif, for tracing a cell back
  shape_zyx, XY_UM, Z_UM, source_masks, source_tiff

Positions are label centroids. dtc_block_cells.m takes a cell whose plane is a
whole number for a hand click and snaps it to the soma; a centroid that lands
exactly on a plane is moved 0.001 plane deeper so it is not.

Cells outside --min-um3 / --max-um3 are dropped; the volume distribution is
printed so the bounds can be picked. NeuroReg's own SizeLimit on BY95 is
[100, 4200] um^3.

Usage (.cellpose-venv or .castalign-venv; needs numpy, scipy, tifffile):
    python alignment/cellpose_to_neuroreg.py                        # block_stack_red
    python alignment/cellpose_to_neuroreg.py --min-um3 100 --max-um3 4200
    python alignment/cellpose_to_neuroreg.py --masks <path_masks.tif>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from analysis_paths import TWO_P_RED_NODES, cellpose_dir, two_p_path, two_p_scope
from scope_profiles import spacing_zyx

PLANE_NUDGE = 1e-3   # planes


def label_centroids(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(labels, centroids (N, 3) as 0-indexed (z, y, x), voxel counts) for labels > 0."""
    from scipy import ndimage
    counts = np.bincount(masks.ravel())
    labels = np.nonzero(counts)[0]
    labels = labels[labels > 0]
    if labels.size == 0:
        return labels, np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    centroids = np.array(ndimage.center_of_mass(masks > 0, masks, labels), dtype=float)
    return labels, centroids.reshape(-1, 3), counts[labels]


def to_neuroreg(centroids_zyx: np.ndarray, xy_um: float, z_um: float) -> np.ndarray:
    """0-indexed (z, y, x) voxels -> 3 x N um, [column; plane; row], 1-indexed grid."""
    z, y, x = (centroids_zyx + 1.0).T
    on_plane = np.abs(z - np.round(z)) < 1e-6
    z = np.where(on_plane, z + PLANE_NUDGE, z)
    return np.vstack([x * xy_um, z * z_um, y * xy_um])


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node", default="block_stack_red", choices=list(TWO_P_RED_NODES),
                        help="Red 2P node whose masks are converted. Default: block_stack_red.")
    parser.add_argument("--masks", default=None,
                        help="Masks tif to convert instead of the node's batch_cellpose.py output.")
    parser.add_argument("--min-um3", type=float, default=0.0, help="Drop cells smaller than this.")
    parser.add_argument("--max-um3", type=float, default=float("inf"),
                        help="Drop cells larger than this.")
    args = parser.parse_args(argv)

    import tifffile
    from scipy.io import savemat

    scope, scope_var = two_p_scope(args.node)
    z_um, xy_um, _ = spacing_zyx(scope, scope_var)

    tiff = two_p_path(args.node)
    if args.masks:
        masks_path = Path(args.masks)
    else:
        if tiff is None:
            sys.exit(f"local_config.{TWO_P_RED_NODES[args.node]} is blank; pass --masks.")
        out = cellpose_dir()
        if out is None:
            sys.exit("ANALYSIS_ROOT is blank in local_config.py; pass --masks.")
        masks_path = out / f"{tiff.stem}_masks.tif"
    if not masks_path.exists():
        sys.exit(f"Masks not found: {masks_path}\nRun batch_cellpose.py --node {args.node} first.")

    run_json = masks_path.with_name(masks_path.name.replace("_masks.tif", "_cellpose.json"))
    if run_json.exists():
        run = json.loads(run_json.read_text())
        if run.get("scope") != scope:
            sys.exit(f"{run_json.name} was segmented under scope {run.get('scope')!r}; "
                     f"local_config's {scope_var} says {scope!r}.")
        if tiff is not None and run.get("input_tiff") != str(tiff):
            print(f"WARNING: {run_json.name} input {run.get('input_tiff')} is not "
                  f"local_config's {tiff}")

    print(f"Masks: {masks_path}")
    masks = tifffile.imread(str(masks_path))
    if masks.ndim != 3:
        sys.exit(f"Expected (Z, Y, X) masks, got shape {masks.shape}")
    print(f"  shape (Z, Y, X) {masks.shape}, {scope_var} {scope}: XY {xy_um} um, Z {z_um} um")

    labels, cents, counts = label_centroids(masks)
    vol = counts * xy_um * xy_um * z_um
    print(f"  {labels.size} labels")
    if labels.size:
        q = np.percentile(vol, [5, 25, 50, 75, 95])
        print("  volume um^3  p5 {:.0f}  p25 {:.0f}  median {:.0f}  p75 {:.0f}  p95 {:.0f}".format(*q))

    keep = (vol >= args.min_um3) & (vol <= args.max_um3)
    print(f"  kept {keep.sum()} in [{args.min_um3:g}, {args.max_um3:g}] um^3, "
          f"dropped {(~keep).sum()}")
    labels, cents, vol = labels[keep], cents[keep], vol[keep]

    pt = to_neuroreg(cents, xy_um, z_um)
    depth = pt[1]
    for lo, hi in [(0, 50), (50, 150), (150, 250), (250, np.inf)]:
        print(f"  depth {lo:>3}-{hi:<4g} um: {int(((depth >= lo) & (depth < hi)).sum())}")

    out_path = masks_path.with_name(masks_path.name.replace("_masks.tif", "_neuroreg.mat"))
    if out_path == masks_path:
        out_path = masks_path.with_suffix(".neuroreg.mat")
    savemat(str(out_path), {
        "pt_list_vol": pt,
        "pt_area_vol": vol.reshape(1, -1).astype(float),
        "cellpose_label": labels.reshape(1, -1).astype(float),
        "shape_zyx": np.array(masks.shape, dtype=float).reshape(1, -1),
        "XY_UM": float(xy_um),
        "Z_UM": float(z_um),
        "min_um3": float(args.min_um3),
        "max_um3": float(args.max_um3) if np.isfinite(args.max_um3) else -1.0,
        "source_masks": str(masks_path),
        "source_tiff": str(tiff) if tiff is not None else "",
    })
    print(f"Wrote {out_path} ({pt.shape[1]} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
