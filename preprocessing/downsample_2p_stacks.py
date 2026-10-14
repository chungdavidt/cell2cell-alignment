#!/usr/bin/env python3
"""
Downsample any 2P stack sampled finer than SCOPE to SCOPE's pitch, for the graph.

castalign fits in voxels, so every node in a fit has to sit on one pitch; the
graph's is SCOPE's (the in vivo stack). A 2P volume whose own scope is finer
(BY108's ex vivo block: 1024 px, 0.553 µm/px, declared as BLOCK_STACK_SCOPE)
enters the graph as a downsampled copy written here; the raw TIFF is untouched
and stays what cellpose and NeuroReg read. Images are only ever downsampled: a
volume coarser than SCOPE stops the run (scope_profiles.downsample_factor).

Each plane is resized in XY by utilities.resize (BARseq step 3's resize:
bilinear, anti-aliased, target round(native / factor), dtype kept); Z is not
resampled and must already match SCOPE's.

Output: <ANALYSIS_ROOT>/preprocessing/2p_downsampled/<stem>.tif (ImageJ TIFF
carrying SCOPE's pitch in µm) + <stem>.json naming the source, scopes, factor
and shapes. subslice_graph_builder.py ingests the copy in place of the raw
file and refuses one whose .json names a different source.

Usage:
    python preprocessing/downsample_2p_stacks.py --dry-run
    python preprocessing/downsample_2p_stacks.py [--force]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import local_config  # noqa: E402
from analysis_paths import downsampled_2p_path, two_p_scope  # noqa: E402
from scope_profiles import downsample_factor, get_profile  # noqa: E402

# Every 2P node and the local_config line naming its TIFF.
TWO_P_NODES = {
    "invivo_red": "INVIVO_PATH_RED",
    "invivo_green": "INVIVO_PATH_GREEN",
    "block_stack_red": "BLOCK_STACK_PATH_RED",
    "block_stack_green": "BLOCK_STACK_PATH_GREEN",
}


def plan() -> list:
    """[(node, raw path, scope, var, factor, out path)] for every configured 2P TIFF."""
    target = getattr(local_config, "SCOPE", "")
    get_profile(target)
    rows = []
    for node, cfg in TWO_P_NODES.items():
        raw = getattr(local_config, cfg, "")
        if not raw:
            continue
        scope, var = two_p_scope(node)
        factor = downsample_factor(scope, target, var, "SCOPE")
        rows.append((node, Path(raw), scope, var, factor, downsampled_2p_path(raw)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report only; read no pixels")
    ap.add_argument("--force", action="store_true", help="overwrite existing copies")
    args = ap.parse_args()

    target = getattr(local_config, "SCOPE", "")
    xy = get_profile(target)["xy_um_per_px"]
    z = get_profile(target)["z_um_per_px"]
    print(f"SCOPE (graph): {target}  ({xy} µm/px XY, {z} µm Z)")
    rows = plan()
    if not rows:
        sys.exit("No 2P TIFF is set in local_config.py.")

    todo = []
    for node, raw, scope, var, factor, out in rows:
        if factor == 1.0:
            print(f"  {node:<18} {var}={scope}: on SCOPE's pitch, ingested as is ({raw.name})")
            continue
        if out is None:
            sys.exit("ANALYSIS_ROOT is blank in local_config.py.")
        state = "exists" if out.exists() else "to write"
        print(f"  {node:<18} {var}={scope}: downsample XY by {factor:g} -> {out}  [{state}]")
        if not raw.exists():
            sys.exit(f"{node}: TIFF not found: {raw}")
        if out.exists() and not args.force:
            continue
        todo.append((node, raw, scope, var, factor, out))

    if args.dry_run or not todo:
        if args.dry_run:
            print("DRY RUN -- nothing written")
        return

    import numpy as np
    import tifffile
    from utilities.resize import downsampled_shape, imresize_bilinear

    for node, raw, scope, var, factor, out in todo:
        print(f"\n{node}: reading {raw}")
        stack = tifffile.imread(str(raw))
        if stack.ndim != 3:
            sys.exit(f"{node}: expected (Z, Y, X), got {stack.shape}")
        out_hw = downsampled_shape(stack.shape[1:], factor)
        small = np.empty((stack.shape[0], *out_hw), dtype=stack.dtype)
        for i, plane in enumerate(stack):
            small[i] = imresize_bilinear(plane, out_hw)
        out.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(out), small, imagej=True, resolution=(1 / xy, 1 / xy),
                         metadata={"spacing": z, "unit": "um", "axes": "ZYX"})
        out.with_suffix(".json").write_text(json.dumps({
            "written": datetime.now().isoformat(timespec="seconds"),
            "node": node, "source": str(raw), "source_scope": scope, "source_var": var,
            "target_scope": target, "factor": factor,
            "source_shape": list(stack.shape), "shape": list(small.shape),
            "method": "utilities.resize.imresize_bilinear per plane (BARseq step 3's)",
        }, indent=2))
        print(f"  wrote {out.name}  {stack.shape} -> {small.shape} {small.dtype}")


if __name__ == "__main__":
    main()
