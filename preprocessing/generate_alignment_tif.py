#!/usr/bin/env python3
"""
Generate the BARseq alignment TIF: QC-passing cells at >= the marker's rolony floor,
everything else background. mScarlet by default; --marker gcamp renders GCaMP
(column 111) on the same mScarlet-defined subslice grid.

This is the image castalign fits on. It deliberately looks like a 2P mScarlet
volume — sparse bright somas on a dark field — because that is what it has to be
matched against. A dense all-cells mask field has no counterpart in the 2P, so
aligning one against the other means matching dense speckle to sparse points.

A cell is drawn only if it passes **both** gates:

    QC:      total reads >= --min-reads AND distinct genes >= --min-genes
    marker:  marker rolonies >= --min-rolonies   (mScarlet, or --marker's column)

Everything else — QC failures, non-marker cells, cells below the cutoff, the gaps
between cells — goes to background. A cell that fails either gate is not dimmed,
it is absent, exactly as a non-expressing cell is absent from a 2P mScarlet image.

**Binary by default.** Cells are flat 255. castalign rescales display intensity,
so graded values are not stable between views; 0 vs max survives any rescaling.

**Graded with --ceiling (David, 2026-09-16).** Each drawn cell gets a level from
its rolony count on a ramp that starts at the floor:

    level = floor(dim + (min(count, ceiling) - floor) / (ceiling - floor) * (255 - dim) + 0.5)

so the floor is --dim (visible, never 0), the ceiling and above are 255, and
every whole count between is its own step. The slope moves with the floor --
the same count is a different level in folders with different floors, as in
matlab/plot_marker_slices.m RAMP_FROM = 'floor'. This is NOT the fixed
[1, ceiling] ramp step 4 and check_rolony_cutoff.py use; compare levels only
within one folder. Levels compare within one castalign view, not between views.

--min-rolonies 0 draws every QC-passing cell; zero-count cells have no row in
export_subslice_cells.py's table, which keeps mScarlet > 0 only.

## --marker

--marker picks the expmat column from marker_profiles (index-only: mScarlet 113,
GCaMP 111). The subslice grid is always the mScarlet one -- this script reads
HYB_DOWNSAMPLED_DIR whatever the marker -- so a GCaMP render is pixel-for-pixel
on the same grid as the mScarlet renders, and the graph builder's Identity hub
carries a fit on either to both. GCaMP cells in FOVs outside the mScarlet
subslices fall off the image (the off-mask / out-of-bounds columns). Its 2P
counterpart is the green channel, which rides the red fit through Identity.

For GCaMP, --min-rolonies is required: ALIGN_MIN_ROLONIES is mScarlet's cutoff.
Bare --ceiling takes the marker's ceiling from marker_profiles.

## --source slice

Draws the same image on the WHOLE section instead of the subslice crop: reads
stitch_slices.py --downsample's slice{N}_CELLMASK.h5 (every FOV with a cell, no
marker filter) and writes

    <OUTPUT_ROOT>/slice_align_{marker}/{marker}_qc..._ge{n}[_sat{c}]/slice{N}_ALIGN_{folder}.tif

Same gates, same levels, same cell -> pixel formula; a subslice image is a crop
of its whole-slice one. For stacking and viewing (stack_sections.py): the names
carry no `subslice` token, so the graph builder never ingests them. The canvas
offset is read with generate_marker_cellmask_slice.canvas_offsets, which takes
stitch_slices.py's min_x/min_y (defect D13: step 3 calls the same value
min_x_offset/min_y_offset).

The whole label is filled, so somas keep the shape the segmentation gave them.

## What --min-rolonies here does and does not affect

It is a **registration** parameter, not an analysis one. It changes which cells
you can see while fitting, and therefore how well-determined the fit is. It does
NOT reach anything downstream: the transform castalign stores is a mapping
between coordinate frames, so it applies to every point in the subslice grid —
cells that were drawn, cells that were not, raw mRNA pixels, background. The
cell <-> label link (export_subslice_cells.py) takes no cutoff at all.

So pick this value purely for legibility against the 2P. Pick the analysis cutoff
separately, as a filter on the exported cell table. They need not match.

Output goes to a folder named by marker and both gates, so several coexist and
nothing is overwritten:
    <OUTPUT_ROOT>/subslice_align_mscarlet/mscarlet_qc{reads}_{genes}_ge{n}/slice{N}_subslice_ALIGN_mscarlet_qc{reads}_{genes}_ge{n}.tif
    <OUTPUT_ROOT>/subslice_align_mscarlet/mscarlet_qc{reads}_{genes}_ge{n}_sat{ceiling}/..._sat{ceiling}.tif   (graded)
    <OUTPUT_ROOT>/subslice_align_gcamp/gcamp_qc{reads}_{genes}_ge{n}[_sat{ceiling}]/slice{N}_subslice_ALIGN_gcamp_...tif
The folder prefix repeats the parent's marker because the graph builder names a
node by section number and leaf folder alone (slice22_subslice_ALIGN_gcamp_qc20_5_ge3_sat10);
each filename repeats the whole folder name so a TIF opened on its own says its
marker, QC pair, floor and ceiling. Without --out the stem is the node name.
--dim is in neither; align_render.json records it.
Unprefixed folders under <OUTPUT_ROOT>/subslice_align/, holding unsuffixed
slice{N}_subslice_ALIGN.tif, are mScarlet renders from before 2026-10-08; this
script no longer writes there.

--dim is not in the folder name. Each folder carries align_render.json, the
settings it was rendered with; a run whose settings differ refuses to write
there. The graph builder copies the same record into node metadata and raises
when a folder no longer matches the nodes built from it.

Usage:
    python preprocessing/generate_alignment_tif.py --min-rolonies 3
    python preprocessing/generate_alignment_tif.py --min-rolonies 10 --slices 22 24
    python preprocessing/generate_alignment_tif.py --min-rolonies 3 --all-cells-level 60
    python preprocessing/generate_alignment_tif.py --min-rolonies 5 --ceiling        # ramp 5 -> 15
    python preprocessing/generate_alignment_tif.py --min-rolonies 0 --ceiling 10 --dim 60
    python preprocessing/generate_alignment_tif.py --marker gcamp --min-rolonies 3 --ceiling   # ramp 3 -> 10
    python preprocessing/generate_alignment_tif.py --source slice --marker gcamp -n 3 --ceiling  # whole sections
"""

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy import sparse

from preprocessing_config import (
    FILT_NEURONS_PATH,
    HYB_DOWNSAMPLED_DIR,
    HYB_SLICE_DOWNSAMPLED_DIR,
    SLICE_ALIGN_MSCARLET_DIR,
    SLICE_ALIGN_GCAMP_DIR,
    SUBSLICE_ALIGN_MSCARLET_DIR,
    SUBSLICE_ALIGN_GCAMP_DIR,
    QC_MIN_READS,
    QC_MIN_GENES,
    ALIGN_MIN_ROLONIES,
    DOWNSAMPLE_XY,
    TARGET_XY_UM_PER_PX,
    SCOPE,
)
from utilities.mat_io import (
    load_filt_neurons,
    load_mat,
    load_cellmask_h5,
    get_expression_column,
    resolve_marker_column,
)
from utilities.image_io import imwrite_tiff
from marker_profiles import MARKERS
from analysis_paths import ALIGN_TIF_GLOB, SLICE_ALIGN_TIF_GLOB, align_tif_name

FOREGROUND = 255
SPARSE_WARN = 20        # subslices with fewer visible cells than this are flagged
DEFAULT_DIM = 40        # level of a floor-count cell in a graded render
# Read back by alignment/subslice_graph_builder.py under the same name.
RENDER_SIDECAR = "align_render.json"
# Default --marker, and the marker of a sidecar written before --marker existed.
# The builder holds the same value.
DEFAULT_MARKER = "mscarlet"
# --ceiling with no value: take the marker's ceiling. Not a str, which argparse would int().
BARE_CEILING = object()
# Output root per marker, per source.
ALIGN_ROOTS = {
    "mscarlet": SUBSLICE_ALIGN_MSCARLET_DIR,
    "gcamp": SUBSLICE_ALIGN_GCAMP_DIR,
}
SLICE_ALIGN_ROOTS = {
    "mscarlet": SLICE_ALIGN_MSCARLET_DIR,
    "gcamp": SLICE_ALIGN_GCAMP_DIR,
}
TIF_GLOBS = {"subslice": ALIGN_TIF_GLOB, "slice": SLICE_ALIGN_TIF_GLOB}
SLICE_CELLMASK_RE = re.compile(r"slice(\d+)_CELLMASK\.h5")


def render_leaf(marker, min_reads, min_genes, floor, ceiling=None):
    """Folder name for one render: mscarlet_qc20_5_ge5, gcamp_qc20_5_ge3_sat10."""
    leaf = f"{marker}_qc{min_reads}_{min_genes}_ge{floor}"
    if ceiling is not None:
        leaf += f"_sat{ceiling}"
    return leaf


def shade(counts, floor, ceiling, dim):
    """Rolony counts -> uint8 levels: floor -> dim, ceiling and above -> 255.

    The ramp starts at the floor, so its slope depends on the floor. Counts
    below the floor clip to dim; the caller draws only rows at >= floor.
    Rolony counts are whole numbers, so this is integer arithmetic: halves
    round up exactly, with no float product landing just under x.5.
    """
    c = np.clip(np.rint(np.asarray(counts, dtype=np.float64)),
                floor, ceiling).astype(np.int64)
    span = ceiling - floor
    return (dim + (2 * (c - floor) * (FOREGROUND - dim) + span)
            // (2 * span)).astype(np.uint8)


def read_render_sidecar(out_dir):
    """The `render` record in out_dir's sidecar, or None when there is none."""
    path = Path(out_dir) / RENDER_SIDECAR
    if not path.exists():
        return None
    render = json.loads(path.read_text()).get("render")
    if render is not None:
        render.setdefault("marker", DEFAULT_MARKER)
    return render


def read_render_source(out_dir):
    """The sidecar's `source`; one written before --source existed is subslice."""
    path = Path(out_dir) / RENDER_SIDECAR
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("source", "subslice")


def check_render_sidecar(out_dir, render, source="subslice"):
    """Raise if out_dir was rendered with different settings.

    --dim and --all-cells-level are not in the folder name, so without this a
    second run would mix two renders in one folder. The source sits beside
    `render`, not in it: the graph builder compares `render` with node metadata,
    and whole-slice renders never become nodes.
    """
    existing_source = read_render_source(out_dir)
    if existing_source is not None and existing_source != source:
        raise SystemExit(f"\n{out_dir} holds {existing_source} renders; this run is "
                         f"--source {source}. Pass --out elsewhere.")
    existing = read_render_sidecar(out_dir)
    if existing is None:
        if any(Path(out_dir).glob(TIF_GLOBS[source])):
            print(f"note: {out_dir.name} has ALIGN tifs but no {RENDER_SIDECAR}; "
                  f"recording this run's settings for the whole folder")
        return
    if existing != render:
        diff = "\n".join(
            f"    {k}: folder {existing.get(k)!r}, this run {render.get(k)!r}"
            for k in sorted(set(existing) | set(render))
            if existing.get(k) != render.get(k))
        raise SystemExit(
            f"\n{out_dir} was rendered with different settings:\n{diff}\n\n"
            f"Move that folder to quarantine/ (or pass --out) before rendering "
            f"these settings under its name. If its nodes are already in the "
            f"graph, the builder will refuse the new render until the graph is "
            f"rebuilt with force_rebuild=True.")


def write_render_sidecar(out_dir, render, levels, source="subslice"):
    """Record the settings, the source and the count -> level table for out_dir."""
    payload = {"source": source,
               "render": render,
               "levels": {str(c): int(v) for c, v in levels}}
    (Path(out_dir) / RENDER_SIDECAR).write_text(json.dumps(payload, indent=2) + "\n")


def qualifying_label_mask(cellmask, x_img, y_img, keep_cell, all_cells_level=0,
                          values=None):
    """uint8 image: a level on labels whose cell passed both gates, else background.

    `values` holds one uint8 level per row (graded render); None paints every
    drawn label FOREGROUND. Where two drawn rows land on one label, the higher
    level wins.

    Builds a label lookup and indexes the mask once, rather than comparing the
    whole image per cell the way steps 4 and 5 do.

    Returns (image, n_labels_drawn, n_off_mask, n_out_of_bounds).
    """
    labels = cellmask if np.issubdtype(cellmask.dtype, np.integer) \
        else cellmask.astype(np.int32)
    h, w = labels.shape

    in_bounds = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
    ids = np.zeros(x_img.size, dtype=np.int64)
    ids[in_bounds] = labels[y_img[in_bounds], x_img[in_bounds]]
    on_mask = ids > 0

    lut = np.zeros(int(labels.max()) + 1, dtype=np.uint8)
    if all_cells_level:
        # optional faint substrate; off by default -- the 2P has no all-cells layer
        lut[:] = np.uint8(all_cells_level)
    lut[0] = 0
    drawn = on_mask & keep_cell
    if drawn.any():
        if values is None:
            lut[ids[drawn]] = FOREGROUND
        else:
            # every level is >= dim > all_cells_level, so the substrate never wins
            np.maximum.at(lut, ids[drawn], np.asarray(values, dtype=np.uint8)[drawn])

    return (lut[labels],
            int(np.unique(ids[drawn]).size),
            int((~on_mask & in_bounds).sum()),
            int((~in_bounds).sum()))


def main():
    ap = argparse.ArgumentParser(
        description="Generate the BARseq alignment TIF (marker cells at >= the "
                    "floor; binary, or graded with --ceiling)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--source", choices=("subslice", "slice"), default="subslice",
                    help="subslice crop (the graph's input, default) or the whole section")
    ap.add_argument("--marker", choices=sorted(ALIGN_ROOTS), default=DEFAULT_MARKER,
                    help=f"which marker column to draw (default: {DEFAULT_MARKER})")
    ap.add_argument("--min-rolonies", "-n", type=int, default=None,
                    help=f"marker rolony floor to be drawn at all (default for "
                         f"mscarlet: config's {ALIGN_MIN_ROLONIES}; required otherwise)")
    ap.add_argument("--min-reads", type=int, default=QC_MIN_READS,
                    help=f"QC total reads floor (default: config's {QC_MIN_READS})")
    ap.add_argument("--min-genes", type=int, default=QC_MIN_GENES,
                    help=f"QC distinct genes floor (default: config's {QC_MIN_GENES})")
    ap.add_argument("--all-cells-level", type=int, default=0,
                    help="draw non-qualifying cells at this level instead of background; "
                         "0 = off, which is the point of this image (default: 0)")
    ap.add_argument("--ceiling", type=int, nargs="?", const=BARE_CEILING, default=None,
                    help="grade drawn cells by rolony count, --min-rolonies -> --dim up to "
                         "this count -> 255; bare flag = the marker's ceiling in "
                         "marker_profiles (" + ", ".join(
                             f"{m} {MARKERS[m]['ceiling']}" for m in sorted(ALIGN_ROOTS))
                         + "). Absent = binary (default)")
    ap.add_argument("--dim", type=int, default=None,
                    help=f"level of a cell at the floor in a graded render "
                         f"(default: {DEFAULT_DIM}); needs --ceiling")
    ap.add_argument("--slices", "-s", type=int, nargs="+", default=None,
                    help="specific slice IDs")
    ap.add_argument("--out", default=None, help="output directory override")
    args = ap.parse_args()

    marker = args.marker
    profile = MARKERS[marker]
    label = profile["label"]
    if args.min_rolonies is None:
        if marker != DEFAULT_MARKER:
            ap.error(f"--marker {marker} needs --min-rolonies; ALIGN_MIN_ROLONIES "
                     f"is the {MARKERS[DEFAULT_MARKER]['label']} cutoff")
        args.min_rolonies = ALIGN_MIN_ROLONIES
    if args.ceiling is BARE_CEILING:
        if profile["ceiling"] is None:
            ap.error(f"bare --ceiling: {label} has no ceiling in marker_profiles; "
                     f"pass --ceiling N")
        args.ceiling = profile["ceiling"]

    floor = args.min_rolonies
    if floor < 0:
        ap.error("--min-rolonies must be >= 0")
    if not 0 <= args.all_cells_level < FOREGROUND:
        ap.error(f"--all-cells-level must be in [0, {FOREGROUND})")
    graded = args.ceiling is not None
    ceiling = args.ceiling
    if args.dim is not None and not graded:
        ap.error("--dim sets the floor's level in a graded render; pass --ceiling too")
    dim = (DEFAULT_DIM if args.dim is None else args.dim) if graded else None
    if graded:
        if not 0 < dim < FOREGROUND:
            ap.error(f"--dim must be in (0, {FOREGROUND}); 0 is background")
        if floor >= ceiling:
            ap.error(f"--min-rolonies {floor} must be below --ceiling {ceiling}; "
                     f"the ramp would have no width")
        if ceiling - floor > FOREGROUND - dim:
            ap.error(f"{ceiling - floor + 1} counts do not fit in levels "
                     f"{dim}-{FOREGROUND} as distinct steps; lower --ceiling or --dim")
        if args.all_cells_level >= dim:
            ap.error(f"--all-cells-level {args.all_cells_level} must be below --dim {dim}, "
                     f"or other cells read as floor cells")

    whole = args.source == "slice"
    input_dir = Path(HYB_SLICE_DOWNSAMPLED_DIR if whole else HYB_DOWNSAMPLED_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Downsampled {'whole slices' if whole else 'subslices'} not found: {input_dir}\n"
            + ("Run stitch_slices.py --downsample --no-scale-bars first." if whole else
               "Run stitch_subslices.py then downsample_subslices_cellmask.py first."))
    leaf = render_leaf(marker, args.min_reads, args.min_genes, floor, ceiling)
    roots = SLICE_ALIGN_ROOTS if whole else ALIGN_ROOTS
    out_dir = Path(args.out) if args.out else Path(roots[marker]) / leaf
    out_dir.mkdir(parents=True, exist_ok=True)

    render = {
        "marker": marker,
        "mode": "graded" if graded else "binary",
        "min_reads": args.min_reads,
        "min_genes": args.min_genes,
        "min_rolonies": floor,
        "ceiling": ceiling,
        "dim": dim,
        "all_cells_level": args.all_cells_level,
    }
    check_render_sidecar(out_dir, render, args.source)
    if graded:
        ramp_counts = np.arange(floor, ceiling + 1)
        levels = list(zip(ramp_counts.tolist(),
                          shade(ramp_counts, floor, ceiling, dim).tolist()))
    else:
        levels = []

    print("=" * 60)
    print("GENERATE ALIGNMENT TIF")
    print("=" * 60)
    print(f"drawn:   QC (reads >= {args.min_reads} AND genes >= {args.min_genes})"
          f" AND {label} (column {profile['column']}) >= {floor}")
    if graded:
        print(f"shading: {floor} rolonies -> {dim}, {ceiling}+ -> {FOREGROUND}, "
              f"{len(levels)} levels (slope set by the floor)")
    else:
        print(f"shading: binary, every drawn cell {FOREGROUND}")
    print(f"         everything else -> background"
          f"{'' if not args.all_cells_level else f' (other cells at {args.all_cells_level})'}")
    print(f"source:  {'whole slices' if whole else 'subslices'}  ({input_dir})")
    print(f"scope:   {SCOPE} ({TARGET_XY_UM_PER_PX:.4f} µm/px)")
    print(f"output:  {out_dir}\n")

    print("Loading filt_neurons...")
    fn = load_filt_neurons(FILT_NEURONS_PATH)
    expmat = fn["expmat"]
    n_cells = expmat.shape[0]

    if sparse.issparse(expmat):
        total_reads = np.asarray(expmat.sum(axis=1)).ravel()
        n_genes = np.asarray((expmat > 0).sum(axis=1)).ravel()
    else:
        total_reads = np.sum(expmat, axis=1)
        n_genes = np.sum(expmat > 0, axis=1)
    pass_qc = (total_reads >= args.min_reads) & (n_genes >= args.min_genes)

    marker_col = resolve_marker_column(fn, profile["gene_name"], profile["column"])
    counts = np.asarray(get_expression_column(expmat, marker_col)).ravel()

    # The AND gate. Both must hold or the cell is not drawn.
    visible = pass_qc & (counts >= floor)
    values = shade(counts, floor, ceiling, dim) if graded else None
    print(f"  cells: {n_cells}")
    print(f"  QC-passing: {int(pass_qc.sum())} ({100 * pass_qc.sum() / n_cells:.1f}%)")
    print(f"  QC-passing AND {label} >= {floor}: {int(visible.sum())}")
    if graded:
        print(f"  of those, at >= {ceiling} ({FOREGROUND}): "
              f"{int((visible & (counts >= ceiling)).sum())}")
    print()
    if not visible.any():
        raise ValueError(f"No cell passes both gates at --min-rolonies {floor}.")

    slice_ids = np.asarray(fn["slice"]).ravel()
    pos = np.asarray(fn["pos"])

    if whole:
        from generate_marker_cellmask_slice import canvas_offsets
        found = {int(m.group(1)): f for f in input_dir.glob("slice*_CELLMASK.h5")
                 if (m := SLICE_CELLMASK_RE.fullmatch(f.name))}
        files = [found[n] for n in sorted(found)]
        use_h5 = True
        if not files:
            raise FileNotFoundError(f"No slice*_CELLMASK.h5 in: {input_dir}")
    else:
        files = sorted(input_dir.glob("slice*_subslice_CELLMASK.h5"),
                       key=lambda f: int(re.search(r"slice(\d+)_subslice", f.name).group(1)))
        use_h5 = bool(files)
        if not files:
            files = sorted(input_dir.glob("slice*_subslice_CELLMASK.mat"),
                           key=lambda f: int(re.search(r"slice(\d+)_subslice", f.name).group(1)))
        if not files:
            raise FileNotFoundError(f"No downsampled subslices found in: {input_dir}")

    want = set(args.slices) if args.slices else None
    written, sparse_slices = [], []
    write_render_sidecar(out_dir, render, levels, args.source)

    for f in files:
        slice_id = int(re.search(r"slice(\d+)_", f.name).group(1))
        if want is not None and slice_id not in want:
            continue
        in_slice = slice_ids == slice_id
        if not in_slice.any():
            continue

        if whole:
            cellmask, meta = load_cellmask_h5(f)
            min_x_offset, min_y_offset = canvas_offsets(meta, f)
        elif use_h5:
            cellmask, meta = load_cellmask_h5(f)
            min_x_offset = int(meta.get("min_x_offset", 0))
            min_y_offset = int(meta.get("min_y_offset", 0))
        else:
            data = load_mat(f)
            if "cellmask_down" not in data:
                print(f"  WARNING: cellmask_down missing in {f.name}, skipped")
                continue
            cellmask = np.asarray(data["cellmask_down"])
            min_x_offset = int(data.get("min_x_offset", 0))
            min_y_offset = int(data.get("min_y_offset", 0))
        if cellmask is None:
            print(f"  WARNING: no cellmask in {f.name}, skipped")
            continue

        # Round THEN subtract 1, matching generate_marker_cellmask_subslice.py,
        # check_rolony_cutoff.py and export_subslice_cells.py exactly.
        idx = np.where(in_slice)[0]
        x_img = np.rint((pos[idx, 0] * 2 - (min_x_offset - 1)) / DOWNSAMPLE_XY).astype(np.int64) - 1
        y_img = np.rint((pos[idx, 1] * 2 - (min_y_offset - 1)) / DOWNSAMPLE_XY).astype(np.int64) - 1

        img, n_drawn, off_mask, oob = qualifying_label_mask(
            cellmask, x_img, y_img, visible[idx], args.all_cells_level,
            None if values is None else values[idx])

        out_path = out_dir / align_tif_name(slice_id, leaf, args.source)
        imwrite_tiff(out_path, img)
        written.append((slice_id, int(visible[idx].sum()), n_drawn, off_mask, oob))
        flag = ""
        if n_drawn < SPARSE_WARN:
            sparse_slices.append(slice_id)
            flag = "   <-- too sparse to align on"
        print(f"  slice {slice_id}: {n_drawn} cells drawn, {img.shape}{flag}")

    if not written:
        raise ValueError("No subslices written.")

    print("\n" + "=" * 60)
    print(f"{'slice':>7}{'passed gates':>15}{'drawn':>8}{'off-mask':>10}{'out-of-bounds':>15}")
    for slice_id, n_pass, n_drawn, off_mask, oob in written:
        print(f"{slice_id:>7}{n_pass:>15}{n_drawn:>8}{off_mask:>10}{oob:>15}")

    tot_drawn = sum(r[2] for r in written)
    print(f"\n{len(written)} TIFs, {tot_drawn} cells drawn total "
          f"(median {int(np.median([r[2] for r in written]))} per subslice)")
    if sparse_slices:
        print(f"\n{len(sparse_slices)} subslice(s) under {SPARSE_WARN} cells: {sparse_slices}")
        print("  Too few landmarks to fit against. Lower --min-rolonies for these,")
        print("  or accept that they align on neighbouring sections' transforms.")
    if graded:
        print(f"\nrolonies -> level ({RENDER_SIDECAR}):")
        print("  " + "  ".join(f"{c}{'+' if c == ceiling else ''}:{v}" for c, v in levels))
    print(f"\n{out_dir}")
    print("--min-rolonies here is a registration parameter only. It changes what you")
    print("can see while fitting, never where the transform sends anything.")


if __name__ == "__main__":
    main()
