#!/usr/bin/env python3
"""Explain how castalign will draw a BARseq alignment node, and why.

Answers "why is the slice inverted in napari" by walking the three places the
pixels can change between the TIFF on disk and what napari paints:

    file  ->  graph storage (utils.compress_image)  ->  layer type (alignment_gui)

Both later steps hinge on `castalign.utils.image_is_label`, and both can
surprise:

**Storage.** `compress_image` sends label-like images to lossless gzip and
everything else to a lossy branch. When the heuristic says False on a binary
marker image, the JPEG branch normalises by `np.quantile(img, .999)` -- which
on a >99.9% background image IS the background value -- so the cells are
clipped away and what comes back is float noise, not a mask. Seen on BY95's
qc20_5_ge5 nodes 2026-08-31. `compression="label"` avoids the heuristic
entirely; the graph builder now passes it for every subslice.

**Display.** On a True verdict `alignment_gui` builds a napari Labels layer
instead of an Image layer with the red colormap. A Labels layer has no
contrast limits: every distinct nonzero value is a solid arbitrary colour and
only 0 is transparent, so a nonzero background paints over everything.

This script prints, per node: the stored compression format, the stored pixel
values, the same values from the TIFF on disk, the `image_is_label` verdict for
both the raw node image and the padded volume Mode C builds from it, and -- if
napari imports -- the actual RGBA each value would be painted.

Read-only. Opens no GUI, writes nothing.

Run in .castalign-venv:
    python alignment/check_align_node_render.py
    python alignment/check_align_node_render.py --node slice22_subslice_ALIGN_qc20_5_ge5
    python alignment/check_align_node_render.py --pad-z 200 --limit 3
"""

import argparse
import ast
import sqlite3
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import castalign as ca
from castalign import utils

from analysis_paths import resolve_subslice_dir
from subslice_graph_builder import (
    GRAPH_PATH, _derive_graph_path, load_single_subslice,
)

# utils.compress_image's format codes (castalign/utils.py:146-148, 153)
_FORMATS = {0: "raw (unused)", 1: "vp9 video, LOSSY", 2: "jpeg, LOSSY",
            3: "gzip, lossless"}


_STEM_MARKER = "_subslice_ALIGN_"


def source_tif(node, subslice_dir):
    """The TIFF a subslice node was built from, found by the node's own name.

    A node is named `{stem}_{render folder}` (subslice_node_name), so the
    folder it came from is in the name even after SUBSLICE_DIR has moved on to
    a different cutoff. Both render folders live side by side under
    <OUTPUT_ROOT>/subslice_align/.
    """
    if subslice_dir is None or _STEM_MARKER not in node:
        return None
    stem, folder = node.split(_STEM_MARKER, 1)
    return Path(subslice_dir).parent / folder / f"{stem}{_STEM_MARKER[:-1]}.tif"


def value_summary(arr, limit=8):
    """'0 x 1,530,199 | 255 x 3,401' plus a flag when there are more values."""
    vals, counts = np.unique(arr, return_counts=True)
    shown = " | ".join(f"{v} x {c:,}" for v, c in
                       zip(vals[:limit].tolist(), counts[:limit].tolist()))
    if vals.size > limit:
        shown += f" | ... ({vals.size} distinct)"
    return shown, vals


def stored_format(graph_path, node):
    """The compression format castalign recorded for this node, from the .db."""
    db = Path(str(graph_path))
    if db.suffix != ".db":
        db = db.with_suffix(".db")
    if not db.exists():
        return "no .db on disk"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT info FROM node_images WHERE node_name = ?", (node,)).fetchone()
    finally:
        con.close()
    if row is None or row[0] is None:
        return "no stored image"
    try:
        kind = ast.literal_eval(row[0])
        code = int(kind[0])
        return f"{_FORMATS.get(code, code)}  (info={kind})"
    except (ValueError, SyntaxError, IndexError, TypeError):
        return f"unparsed info={row[0]!r}"


def napari_label_colors(values):
    """RGBA napari would paint each label value, or None if napari won't say."""
    try:
        import napari
        layer = napari.layers.Labels(np.zeros((1, 1), dtype=np.int64))
        out = {}
        for v in values:
            try:
                c = layer.get_color(int(v))
            except AttributeError:
                c = layer.colormap.map(np.asarray([int(v)]))[0]
            out[int(v)] = np.round(np.asarray(c, dtype=float), 3).tolist()
        return out
    except Exception as exc:                      # napari absent or API moved
        return f"unavailable ({type(exc).__name__}: {exc})"


def describe(g, node, subslice_dir, pad_z, repeat):
    print("=" * 70)
    print(node)
    print("=" * 70)

    print(f"  stored as: {stored_format(g.filename or GRAPH_PATH, node)}")

    img = np.asarray(g.get_image(node))
    summary, vals = value_summary(img)
    print(f"  in graph:  {img.shape} {img.dtype}")
    print(f"             {summary}")

    # The node carries its own render folder, so find the source TIFF by name
    # rather than through the current SUBSLICE_DIR -- they differ whenever the
    # config has moved on to another cutoff since the node was added.
    tif = source_tif(node, subslice_dir)
    if tif is not None and tif.exists():
        disk = np.asarray(load_single_subslice(tif))
        disk_summary, _ = value_summary(disk)
        print(f"  on disk:   {tif}")
        print(f"             {disk.shape} {disk.dtype}")
        print(f"             {disk_summary}")
        same = (disk.shape == img.shape and
                np.array_equal(np.asarray(disk), np.asarray(img)))
        print(f"  identical to graph: {same}"
              f"{'' if same else '   <-- STORAGE CHANGED THE PIXELS'}")
        disk_is_label = utils.image_is_label(disk)
        would = "gzip, lossless" if disk_is_label else (
            "vp9, LOSSY" if min(disk.shape) > 10 else "jpeg, LOSSY")
        print(f"  image_is_label(file): {disk_is_label}"
              f"  -> compression=\"normal\" would store it as {would}")
        print(f"                        compression=\"label\" stores it as "
              f"gzip, lossless regardless")
    else:
        print(f"  on disk:   not found"
              f"{'' if tif is None else f' ({tif})'}")

    is_label = utils.image_is_label(img)
    print(f"  image_is_label(node image): {is_label}"
          f"  -> alignment_gui builds "
          f"{'v.add_labels()' if is_label else 'v.add_image(colormap=red)'}")

    # What Mode C actually hands the GUI as the fixed base
    if repeat:
        padded = np.repeat(np.asarray(img[0])[None], 2 * pad_z + 1, axis=0)
        print(f"  padded base (repeat=True): {padded.shape}, no fill value")
    else:
        old_fill = float(img.mean()) * 0.90
        new_fill = 0 if vals.size <= 3 else old_fill
        padded = np.full((2 * pad_z + 1, *img.shape[1:]),
                         new_fill, dtype=img.dtype)
        padded[pad_z] = img[0]
        print(f"  padded base (repeat=False): {padded.shape}, "
              f"fill={np.asarray(new_fill).item()} "
              f"(pre-2026-08-31 fill would be {old_fill:.2f} -> "
              f"{np.asarray(old_fill).astype(img.dtype).item()})")
    padded_summary, padded_vals = value_summary(padded)
    print(f"             {padded_summary}")
    padded_is_label = utils.image_is_label(padded)
    print(f"  image_is_label(padded base): {padded_is_label}"
          f"  -> {'LABELS layer' if padded_is_label else 'Image layer, red colormap'}")

    if padded_is_label:
        colors = napari_label_colors(padded_vals[:8].tolist())
        print(f"  napari label colours (RGBA): {colors}")
        print("  In a Labels layer only 0 is transparent. Any other value is a "
              "solid colour,\n  so a nonzero background paints over everything "
              "and the cells are a second colour.")


def main():
    ap = argparse.ArgumentParser(
        description="Report how castalign stores and draws BARseq alignment nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--node", action="append", default=None,
                    help="node name (repeatable); default: subslice nodes in the graph")
    ap.add_argument("--pad-z", type=int, default=25,
                    help="PAD_Z used by Mode C (default: 25)")
    ap.add_argument("--repeat", action="store_true",
                    help="model REPEAT_SLICE_IN_Z = True instead of a blank pad")
    ap.add_argument("--limit", type=int, default=3,
                    help="how many nodes to report when --node is not given")
    args = ap.parse_args()

    # Blank GRAPH_PATH means "derive it", same rule the builder follows.
    graph_path = Path(GRAPH_PATH) if GRAPH_PATH else _derive_graph_path(None, None)
    if not graph_path.exists():
        raise SystemExit(f"Graph not found: {graph_path}")
    g = ca.Graph.load(str(graph_path))
    print(f"Graph: {graph_path}  ({len(g.nodes)} nodes)\n")

    subslice_dir = resolve_subslice_dir()
    print(f"SUBSLICE_DIR resolves to: {subslice_dir}\n")

    if args.node:
        nodes = args.node
        missing = [n for n in nodes if n not in g.nodes]
        if missing:
            raise SystemExit(f"Not in graph: {missing}\nHave: {sorted(g.nodes)}")
    else:
        nodes = [n for n in sorted(g.nodes) if n.startswith("slice")][:args.limit]
        if not nodes:
            raise SystemExit(f"No slice* nodes in graph. Have: {sorted(g.nodes)}")

    for n in nodes:
        describe(g, n, subslice_dir, args.pad_z, args.repeat)
        print()


if __name__ == "__main__":
    main()
