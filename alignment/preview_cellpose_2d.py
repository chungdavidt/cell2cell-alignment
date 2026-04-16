"""
Fast 2D cellpose preview on a single Z plane of a 3D TIFF.

Use to iterate on cellprob/flow/filter params without waiting hours for a
3D run. Does NOT save anything — just prints the ROI count + some stats.

Usage:
    python alignment/preview_cellpose_2d.py <tiff_path> <z_plane>
    python alignment/preview_cellpose_2d.py <tiff_path> <z_plane> --cellprob -2 --flow 0.7
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tifffile


NORMALIZE_PARAMS = {
    "lowhigh": None,
    "percentile": [0.1, 97.0],
    "normalize": True,
    "norm3D": True,
    "sharpen_radius": 60.0,
    "smooth_radius": 1.0,
    "tile_norm_blocksize": 50.0,
    "tile_norm_smooth3D": 1.0,
    "invert": False,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tiff_path", help="Path to 3D TIFF")
    parser.add_argument("z_plane", type=int, help="Z plane to preview (0-indexed)")
    parser.add_argument("--cellprob", type=float, default=-1.0, help="cellprob_threshold (default -1)")
    parser.add_argument("--flow", type=float, default=0.6, help="flow_threshold (default 0.6)")
    parser.add_argument("--min-size", type=int, default=8, help="min_size (default 8)")
    args = parser.parse_args()

    tiff_path = Path(args.tiff_path)
    if not tiff_path.exists():
        sys.exit(f"Not found: {tiff_path}")

    print(f"Loading {tiff_path.name}")
    t0 = time.time()
    stack = tifffile.imread(str(tiff_path))
    print(f"  shape: {stack.shape}, dtype: {stack.dtype}, load time: {time.time()-t0:.1f}s")

    if stack.ndim != 3:
        sys.exit(f"Expected 3D (Z, Y, X) stack, got ndim={stack.ndim}")
    if not (0 <= args.z_plane < stack.shape[0]):
        sys.exit(f"z_plane {args.z_plane} out of range [0, {stack.shape[0]})")

    plane = stack[args.z_plane]
    print(f"  plane {args.z_plane}: shape {plane.shape}, min {plane.min()}, "
          f"max {plane.max()}, mean {plane.mean():.1f}")

    print()
    print("Loading cpsam model...")
    t0 = time.time()
    from cellpose.models import CellposeModel
    model = CellposeModel(gpu=True, pretrained_model="cpsam")
    print(f"  loaded in {time.time()-t0:.1f}s (device={model.device})")

    print()
    print(f"Running 2D segmentation on plane {args.z_plane}:")
    print(f"  cellprob_threshold = {args.cellprob}")
    print(f"  flow_threshold     = {args.flow}")
    print(f"  min_size           = {args.min_size}")
    print(f"  normalize dict:    = {NORMALIZE_PARAMS}")

    t0 = time.time()
    masks, flows, styles = model.eval(
        plane,
        do_3D=False,
        normalize=NORMALIZE_PARAMS,
        cellprob_threshold=args.cellprob,
        flow_threshold=args.flow,
        min_size=args.min_size,
        niter=200,
        diameter=None,
        batch_size=8,
    )
    dt = time.time() - t0

    n_rois = int(masks.max())
    sizes = np.bincount(masks.ravel())[1:] if n_rois > 0 else np.array([])

    print()
    print(f"DONE in {dt:.1f}s")
    print(f"  ROIs found:  {n_rois}")
    if n_rois > 0:
        print(f"  size range:  {sizes.min()} – {sizes.max()} px")
        print(f"  median size: {int(np.median(sizes))} px")


if __name__ == "__main__":
    main()
