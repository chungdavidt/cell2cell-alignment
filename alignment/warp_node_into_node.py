#!/usr/bin/env python3
"""Draw one 2P volume on another volume's voxel grid through the graph's fit.

Written for NeuroReg block curation: the in-vivo red stack drawn in the ex vivo
block's frame, loaded as the second channel of GUI_StackCuration so a cell
present in both shows yellow. Any two volume nodes with a path between them work.

The source image is read from the TIFF local_config names for that node, which
is the array the graph builder stored (imageio.volread, no reorientation); the
graph copy is lossy for 2P volumes. The output has the destination TIFF's shape,
(Z, Y, X) = (pages, rows, columns), float32; voxels the source does not cover
are 0.

    g.get_transform(src, dst).transform_image(img, output_size=[(0,Z),(0,Y),(0,X)],
                                              labels=False, force_size=True)

Run in .castalign-venv:
    python alignment/warp_node_into_node.py
    python alignment/warp_node_into_node.py --src invivo_red --dst block_stack_red --out D:\\x.tif
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import tifffile
import castalign as ca

import local_config
from analysis_paths import get_analysis_root
from subslice_graph_builder import GRAPH_PATH, _derive_graph_path

NODE_TIFS = {
    "invivo_red": "INVIVO_PATH_RED",
    "invivo_green": "INVIVO_PATH_GREEN",
    "block_stack_red": "BLOCK_STACK_PATH_RED",
    "block_stack_green": "BLOCK_STACK_PATH_GREEN",
}


def node_tif(node):
    """The TIFF local_config names for a volume node."""
    if node not in NODE_TIFS:
        raise SystemExit(f"{node}: not a volume node; choose from {sorted(NODE_TIFS)}")
    path = getattr(local_config, NODE_TIFS[node], "")
    if not path:
        raise SystemExit(f"{NODE_TIFS[node]} is blank in local_config.py")
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{node}: {path} not found")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="invivo_red", help="volume to draw (default: invivo_red)")
    ap.add_argument("--dst", default="block_stack_red", help="grid to draw it on (default: block_stack_red)")
    ap.add_argument("--graph", default=None, help="graph .db (default: the builder's GRAPH_PATH rule)")
    ap.add_argument("--out", default=None,
                    help="output TIFF (default: <ANALYSIS_ROOT>/neuroreg/ZStack/processed/{src}_in_{dst}.tif)")
    args = ap.parse_args()

    graph_path = Path(args.graph) if args.graph else (
        Path(GRAPH_PATH) if GRAPH_PATH else _derive_graph_path(None, None))
    if not graph_path.exists():
        raise SystemExit(f"Graph not found: {graph_path}")
    if args.out:
        out = Path(args.out)
    else:
        root = get_analysis_root()
        if root is None:
            raise SystemExit("ANALYSIS_ROOT is blank in local_config.py; pass --out")
        out = root / "neuroreg" / "ZStack" / "processed" / f"{args.src}_in_{args.dst}.tif"

    src_path, dst_path = node_tif(args.src), node_tif(args.dst)
    g = ca.Graph.load(str(graph_path))
    for n in (args.src, args.dst):
        if n not in g.nodes:
            raise SystemExit(f"{n} not in {graph_path}")
    if not g.has_transform(args.src, args.dst):
        raise SystemExit(f"No path from {args.src} to {args.dst} in {graph_path}")
    t = g.get_transform(args.src, args.dst)

    src = tifffile.imread(str(src_path)).astype(np.float32)
    with tifffile.TiffFile(str(dst_path)) as tf:
        dst_shape = tf.series[0].shape
    if len(src.shape) != 3 or len(dst_shape) != 3:
        raise SystemExit(f"Expected 3D stacks: {args.src} {src.shape}, {args.dst} {dst_shape}")
    Z, Y, X = dst_shape
    print(f"Graph: {graph_path}")
    print(f"{args.src}: {src_path}  {src.shape}")
    print(f"{args.dst}: {dst_path}  {dst_shape}")

    warped = np.asarray(t.transform_image(src, output_size=[(0, Z), (0, Y), (0, X)],
                                          labels=False, force_size=True), dtype=np.float32)
    warped = np.nan_to_num(warped, nan=0.0)
    if warped.shape != tuple(dst_shape):
        raise SystemExit(f"transform_image returned {warped.shape}, expected {tuple(dst_shape)}")

    covered = warped != 0
    per_plane = covered.reshape(Z, -1).mean(axis=1)
    planes = np.flatnonzero(per_plane > 0.5)
    print(f"Covered voxels: {covered.mean():.1%}")
    if planes.size:
        print(f"Planes more than half covered: {planes[0] + 1}-{planes[-1] + 1} of {Z} (1-based, as in ImageJ)")
    else:
        print("No plane is more than half covered")

    out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out), warped)
    print(f"Saved: {out}  {warped.shape} float32")


if __name__ == "__main__":
    main()
