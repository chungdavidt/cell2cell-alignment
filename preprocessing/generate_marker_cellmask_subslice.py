#!/usr/bin/env python3
"""
Generate Marker Cell Mask Overlays for Subslices.

Creates marker overlays on CELL MASK background, on the downsampled subslices.
--marker picks which readout column is painted: mScarlet (113) or GCaMP (111),
index-only, since the panel labels these slots with stale gene names. Each marker
carries its own colour ramp, draw cutoff, saturation cap and output root --
the first three in marker_profiles.py at the project root, the last in
MARKER_OUTPUT_DIRS below; nothing else in the run differs between them.

Ported from the lab's generate_mscarlet_cellmask_subslice_anisotropic.m -- their filename, from the era of the
two-factor resample. This port is ISOTROPIC: one DOWNSAMPLE_XY covers both
in-plane axes. Pre-isotropic code is frozen in archive/anisotropic_preprocessing/.

Cell centroids from filt_neurons.mat are mapped into the downsampled image with
DOWNSAMPLE_XY, the same factor downsample_subslices_cellmask.py used
to resample it. The two MUST agree or every cell lands off its mask.

This pipeline:
    - Shows cell masks in flat grey
    - Paints cells at >= the draw cutoff on the marker's ramp -- dark red ->
      orange -> yellow for mScarlet, dark green -> green -> pale yellow-green
      for GCaMP -- linear in rolony count over the FIXED domain
      [ROLONY_RAMP_MIN, the marker's ceiling]
    - No DAPI background

The ramp domain is a constant and the cutoff does not touch it. A cell's rolony
count does not change when the cutoff moves, so its colour must not either:
--min-rolonies decides which cells are DRAWN and nothing else. Raising it
deletes cells off the bottom of the ramp and leaves every survivor the colour it
already was, so two runs compare pixel for pixel. Anchoring the ramp on the
cutoff instead -- (count - cutoff) / (CEILING - cutoff), which this code did
until 2026-09-04 -- re-shaded every surviving cell: at a cutoff of 3 a 9-rolony
cell was (255, 90, 0) and at a cutoff of 5 it was (227, 71, 0), so a colour read
off one image meant a different count in the next. Do not re-couple them. The
count / max_expr * MSCARLET_BOOST form both replace tied every cell's brightness
to the single brightest cell in the dataset -- a value that moves with the QC
gates -- and rendered BY95's median marker cell (1 rolony) darker than the grey
mask behind it. Both config constants stay defined for the two scripts still
using them.

Each ramp is a hue ramp, not a single-channel one. Counts are integers, so a
1-to-15 domain has 15 levels; on a red-only ramp those differ in ONE channel
12.75 uint8 apart and read as flat. mScarlet's three anchor colours are
check_rolony_cutoff.py's, and so is the [1, cap] domain, so a count renders
identically in both tools whenever that tool's --saturate-at equals the marker's
ceiling. Every ramp's darkest anchor must clear the grey mask field at uint8 32,
or a low-count cell is invisible against the background it sits on.

A marker's ceiling is a BY95 number, the default while the cap is being tried
out. It is per-brain like every other rolony gate: re-pick it with
check_rolony_cutoff.py before running another dataset. Moving it DOES re-shade
every cell -- it is the one bound that sets the mapping -- so it is a constant
edit, not a flag, and it names the output folder alongside the cutoff. Two runs
at the same cutoff and cap overwrite each other, and the second one wins.

run_pipeline.py does not forward its own --min-rolonies here: that flag sets the
ALIGN tifs' cutoff, a registration parameter, and this one is a display choice.
Two questions, two numbers.

--exclude-slices drops those sections from the whole run, for a section whose
data is bad: no overlay TIF, no comparison PNG, no bar in the histogram. It is
local to this script -- nothing else reads it, no config name holds it, and
run_pipeline.py does not forward it, so the pipeline still renders the section.
A number with no subslice on disk is an error, so a typo cannot write a figure
whose name claims an exclusion that did not happen.

Usage:
    python generate_marker_cellmask_subslice.py
    python generate_marker_cellmask_subslice.py --marker gcamp
    python generate_marker_cellmask_subslice.py --slice 22
    python generate_marker_cellmask_subslice.py --min-rolonies 3
    python generate_marker_cellmask_subslice.py --exclude-slices 58
    python generate_marker_cellmask_subslice.py --test

Input:
    - HYB_subslice_stitched_tif_downsampled_micronwise/
    - filt_neurons.mat (for cell positions and expression)

Output (the root is the marker's: mScarlet_cellmask_subslice/ or
GCaMP_cellmask_subslice/, so the two never overwrite each other):
    - <marker root>/rolony_ge{CUTOFF}_sat{CEILING}/
        * slice{N}_subslice_{mScarlet|GCaMP}_cellmask.tif
        * slice{N}_subslice_comparison.png
        * rolony_ramp_legend.png -- one per folder, not per slice. The domain
          is absolute and cutoff-independent, so a count maps to the same
          colour in every folder too; a higher cutoff just starts the legend
          higher. Only the ceiling changes the colours.
        * cell_count_histogram_ge{CUTOFF}_rolonies[_ex{slices}].png -- cells
          painted per section against that section's QC-passing marker+
          total. Whole runs only: --slice and --test would write one bar over
          it. The excluded numbers land in the filename, so an excluded run
          cannot overwrite a full one.

      The folder name records the draw cutoff and the ramp cap.
      QC_MIN_READS / QC_MIN_GENES also decide which cells are eligible to be
      drawn; they live in local_config.py and the run prints them.
"""

import argparse
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy import sparse
import re
import time

from preprocessing_config import (
    FILT_NEURONS_PATH,
    HYB_DOWNSAMPLED_DIR,
    GCAMP_CELLMASK_DIR,
    GCAMP_COLUMN_INDEX,
    MSCARLET_CELLMASK_DIR,
    MSCARLET_COLUMN_INDEX,
    MSCARLET_GENE_NAME,
    QC_MIN_READS,
    QC_MIN_GENES,
    EXVIVO_UM_PER_PX,
    TARGET_XY_UM_PER_PX,
    DOWNSAMPLE_XY,
    CELLMASK_BRIGHTNESS,
)
from marker_profiles import get_marker, marker_names
from utilities.mat_io import load_filt_neurons, load_mat, load_cellmask_h5, get_expression_column, resolve_marker_column
from utilities.image_io import imwrite_tiff
from utilities.visualization import create_comparison_figure, create_histogram

# Rolony count -> colour. The domain is [ROLONY_RAMP_MIN, the marker's ceiling]
# and the draw cutoff is NOT part of it: those two bounds alone decide what
# colour a count gets. The ramp floor is 1 for every marker -- a drawn cell has
# at least one rolony -- so only the ceiling is per marker.
ROLONY_RAMP_MIN = 1     # count that maps to the darkest colour; never the cutoff
CELLMASK_SCALE = 0.5    # scales CELLMASK_BRIGHTNESS -> 0.125, uint8 32

# Where each marker's overlays land. Paths need OUTPUT_ROOT, so they stay in
# preprocessing_config while the rest of the marker's facts -- column, ramp,
# floor, ceiling -- live in the stdlib-only marker_profiles at the project root.
MARKER_OUTPUT_DIRS = {
    "mscarlet": MSCARLET_CELLMASK_DIR,
    "gcamp": GCAMP_CELLMASK_DIR,
}


def make_ramp(colors):
    """The marker's anchor colours as one colormap, built once per run."""
    return LinearSegmentedColormap.from_list("rolony", colors)


def ramp_rgb(count, cmap, ceiling):
    """One rolony count -> RGB on the fixed [ROLONY_RAMP_MIN, ceiling] ramp.
    Takes no cutoff: the same count is the same colour in every run."""
    frac = np.clip(
        (count - ROLONY_RAMP_MIN) / (ceiling - ROLONY_RAMP_MIN), 0.0, 1.0)
    return cmap(frac)[:3]


def write_ramp_legend(output_dir, floor, cmap, ceiling, marker_label):
    """One legend per output folder: every colour that run can produce.

    Discrete swatches, not a gradient. Counts are integers, so a swatch per
    count is a lookup -- read a cell's colour off the image, read its rolony
    count off the legend. It runs `floor` to the ceiling because those are
    the counts that get drawn, but every swatch is `ramp_rgb`'s absolute
    colour: raising the cutoff shortens this legend from the bottom and leaves
    the remaining swatches untouched. Written as its own file, never burned
    into the overlay TIF, which shares a pixel grid with the ALIGN tif and with
    export_subslice_cells.py's y_node/x_node.
    """
    import matplotlib.pyplot as plt

    counts = list(range(floor, ceiling + 1))
    fig, ax = plt.subplots(figsize=(2.6, 0.34 * (len(counts) + 2) + 0.6))

    for row, count in enumerate(counts):
        ax.add_patch(plt.Rectangle((0, row), 1, 0.86,
                                   color=ramp_rgb(count, cmap, ceiling)))
        label = f"{count}+" if count == ceiling else str(count)
        ax.text(1.15, row + 0.43, label, va="center", fontsize=9)

    # Below the cutoff a cell is drawn in the mask field's grey,
    # indistinguishable from marker-negative.
    grey = CELLMASK_BRIGHTNESS * CELLMASK_SCALE
    ax.add_patch(plt.Rectangle((0, -1.4), 1, 0.86, color=(grey, grey, grey)))
    ax.text(1.15, -0.97, f"< {floor}", va="center", fontsize=9)

    ax.set_xlim(-0.1, 2.6)
    ax.set_ylim(-1.9, len(counts) + 0.4)
    ax.axis("off")
    ax.set_title(f"{marker_label} rolonies", fontsize=10, pad=8)
    fig.text(0.5, 0.015,
             f"fixed ramp {ROLONY_RAMP_MIN}-{ceiling}+, absolute counts",
             ha="center", fontsize=7)

    out_path = Path(output_dir) / "rolony_ramp_legend.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def drop_excluded_slices(cellmask_files, excluded):
    """`cellmask_files` without those sections, and how many files that was.

    Copied into this script rather than shared: the exclusion is one run's
    argument, not a property of the brain. A section with no subslice on disk
    is an error -- excluding a number that was never there would put a claim in
    the filename that the run did not honour.
    """
    present = set()
    for f in cellmask_files:
        m = re.search(r'slice(\d+)_subslice', f.stem)
        if m:
            present.add(int(m.group(1)))

    missing = [s for s in excluded if s not in present]
    if missing:
        raise ValueError(
            f"--exclude-slices {missing}: no subslice on disk. "
            f"present: {sorted(present)}")

    kept = []
    for f in cellmask_files:
        m = re.search(r'slice(\d+)_subslice', f.stem)
        if m and int(m.group(1)) in excluded:
            continue
        kept.append(f)
    return kept, len(cellmask_files) - len(kept)


def generate_marker_cellmask_subslice(
    target_slice: int = None,
    test_mode: bool = False,
    exclude_slices=None,
    min_rolonies: int = None,
    marker: str = "mscarlet",
):
    """
    Generate marker cell mask overlays.

    Args:
        target_slice: Process specific slice only
        test_mode: Process first subslice only
        exclude_slices: Section numbers to drop from the whole run
        min_rolonies: Draw cutoff for this run (default: the marker's floor).
            Gates which cells are drawn; never the ramp.
        marker: Which marker_profiles entry to paint -- 'mscarlet' or 'gcamp'. Decides
            the column, the ramp, the cutoff/cap defaults and the output root.
    """
    settings = get_marker(marker, min_rolonies)
    settings["out_dir"] = MARKER_OUTPUT_DIRS[marker]
    marker_label = settings["label"]
    ceiling = settings["ceiling"]
    cmap = make_ramp(settings["ramp"])

    floor = settings["floor"] if min_rolonies is None else int(min_rolonies)
    if floor < 1:
        raise ValueError(f"--min-rolonies {floor} is below 1: a drawn cell has "
                         f"at least one rolony")

    input_dir = Path(HYB_DOWNSAMPLED_DIR)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Downsampled subslices not found!\n"
            f"Run downsample_subslices_cellmask.py first.\n"
            f"Expected: {input_dir}"
        )

    # One folder per (cutoff, ramp cap), so changing either writes beside the
    # last run instead of overwriting it. The name is ge{cutoff}_sat{cap},
    # check_rolony_cutoff.py's vocabulary -- the old rolony_{floor}_{ceiling}
    # folders read as a ramp range and were one, which is the bug this pair of
    # names retires.
    output_dir = (Path(settings["out_dir"])
                  / f"rolony_ge{floor}_sat{ceiling}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 40)
    print(f"GENERATE {marker_label.upper()} CELL MASK OVERLAYS")
    print("=" * 40)
    print(f"Marker: {marker_label}, column {settings['column']}")
    print(f"Rolony ramp: {ROLONY_RAMP_MIN} -> {ceiling}+ rolonies "
          f"(fixed, absolute, cutoff-independent)")
    print(f"Draw cutoff: >= {floor} rolonies; below it, not drawn")
    if floor >= ceiling:
        print(f"  WARNING: cutoff {floor} is at or above the ramp cap "
              f"{ceiling}, so every drawn cell saturates to one colour")
    print(f"Cell mask brightness: {CELLMASK_BRIGHTNESS * CELLMASK_SCALE:.3f}")
    legend_path = write_ramp_legend(output_dir, floor, cmap, ceiling,
                                    marker_label)
    print(f"Legend: {legend_path.name}")
    print()
    print("Resolution matching:")
    print(f"  Ex vivo original: {EXVIVO_UM_PER_PX:.4f} um/px")
    print(f"  2P target:        {TARGET_XY_UM_PER_PX:.4f} um/px")
    print(f"  Downsample factor: {DOWNSAMPLE_XY:.4f}x (both dimensions)")
    print()

    if test_mode:
        print("Mode: TEST (first subslice only)")
    elif target_slice is not None:
        print(f"Mode: SLICE {target_slice} only")
    else:
        print("Mode: FULL (all subslices)")
    print()

    # Load filt_neurons
    print("Loading filt_neurons...")
    filt_neurons = load_filt_neurons(FILT_NEURONS_PATH)
    expmat = filt_neurons['expmat']
    n_cells = expmat.shape[0]
    print(f"  Total cells: {n_cells}")

    # Apply QC filters
    if sparse.issparse(expmat):
        total_reads = np.asarray(expmat.sum(axis=1)).flatten()
        total_genes = np.asarray((expmat > 0).sum(axis=1)).flatten()
    else:
        total_reads = np.sum(expmat, axis=1)
        total_genes = np.sum(expmat > 0, axis=1)

    pass_qc = (total_reads >= QC_MIN_READS) & (total_genes >= QC_MIN_GENES)
    print(f"  QC filtering: {np.sum(pass_qc)} / {n_cells} cells pass ({100*np.sum(pass_qc)/n_cells:.1f}%)")

    # Get marker expression
    marker_col = resolve_marker_column(
        filt_neurons, settings["gene_name"], settings["column"])
    marker_expression = get_expression_column(expmat, marker_col)

    # Calculate global max from QC-passed cells ONLY
    max_expr = np.max(marker_expression[pass_qc])
    print(f"Global max {marker_label} (QC-passed): {max_expr} transcripts")

    # Reported only. The ramp is absolute, so nothing divides by this.
    marker_positive = marker_expression > 0
    print(f"  {marker_label}+ cells: {np.sum(marker_positive)}")

    # Combined filter (QC-passed AND marker+)
    marker_qc_pass = pass_qc & marker_positive
    print(f"  {marker_label}+ QC-passing: {np.sum(marker_qc_pass)}")
    print(f"  at >= {floor} rolonies (drawn): "
          f"{np.sum(marker_qc_pass & (marker_expression >= floor))}")
    print(f"  at >= {ceiling} rolonies (saturated): "
          f"{np.sum(marker_qc_pass & (marker_expression >= ceiling))}\n")

    print("Position scale factor (from full-res):")
    print(f"  {1/DOWNSAMPLE_XY:.6f} on both axes (x2 for canvas, then scale)\n")

    # Find subslices to process (check for .h5 first, fall back to .mat)
    cellmask_files = list(input_dir.glob("slice*_subslice_CELLMASK.h5"))
    use_h5_format = True

    if not cellmask_files:
        cellmask_files = list(input_dir.glob("slice*_subslice_CELLMASK.mat"))
        use_h5_format = False

    if not cellmask_files:
        raise FileNotFoundError(f"No downsampled subslices found in: {input_dir}")

    print(f"Found {len(cellmask_files)} subslices ({'H5' if use_h5_format else 'MAT'} format)\n")

    # Before --slice / --test narrow the set, so a typo is caught against every
    # section on disk rather than against one run's subset.
    excluded = sorted(set(exclude_slices or []))
    excluded_note = ""
    if excluded:
        if target_slice in excluded:
            raise ValueError(
                f"--slice {target_slice} is also in --exclude-slices {excluded}")
        cellmask_files, n_dropped = drop_excluded_slices(cellmask_files, excluded)
        excluded_note = (f"excluding slice{'s' if len(excluded) > 1 else ''} "
                         f"{', '.join(str(s) for s in excluded)}: "
                         f"{n_dropped} subslices dropped")
        print(excluded_note)
        print("  (the dataset totals printed above count every section)\n")
        if not cellmask_files:
            raise ValueError("--exclude-slices dropped every subslice")

    # Filter by target slice
    if target_slice is not None:
        ext = '.h5' if use_h5_format else '.mat'
        pattern = f"slice{target_slice}_subslice_CELLMASK{ext}"
        cellmask_files = [f for f in cellmask_files if f.name == pattern]
        if not cellmask_files:
            raise ValueError(f"Slice {target_slice} not found")

    if test_mode:
        cellmask_files = cellmask_files[:1]
        print("TEST MODE: Processing first subslice only\n")

    cellmask_files.sort()

    # Get arrays for position lookup
    slice_ids = np.asarray(filt_neurons['slice']).flatten()
    pos = np.asarray(filt_neurons['pos'])

    # (slice_id, QC-passing marker+ cells, cells painted) per section
    hist_rows = []

    # Process each subslice
    for i, cellmask_file in enumerate(cellmask_files):
        base_name = cellmask_file.stem.replace('_CELLMASK', '')

        # Parse slice ID
        match = re.search(r'slice(\d+)_subslice', base_name)
        if not match:
            print(f"WARNING: Could not parse slice ID from: {base_name}")
            continue
        slice_id = int(match.group(1))

        print("=" * 40)
        print(f"[{i+1}/{len(cellmask_files)}] Processing {base_name}")
        print("=" * 40)

        # Load cellmask
        print("  Loading cellmask...")

        if use_h5_format:
            stitched_cellmask, mask_metadata = load_cellmask_h5(cellmask_file)
            min_x_offset = mask_metadata.get('min_x_offset', 0)
            min_y_offset = mask_metadata.get('min_y_offset', 0)
            print(f"    Cellmask: {stitched_cellmask.shape[0]} x {stitched_cellmask.shape[1]} "
                  f"(canvas offset: x={min_x_offset}, y={min_y_offset})")
        else:
            # Legacy .mat format
            mask_data = load_mat(cellmask_file)

            if 'cellmask_down' in mask_data:
                stitched_cellmask = np.asarray(mask_data['cellmask_down'])
            else:
                # Fallback for old files
                for key, value in mask_data.items():
                    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                        if value.size > 100:
                            stitched_cellmask = np.asarray(value)
                            print(f"  WARNING: Using {key} as cellmask (cellmask_down not found)")
                            break

            # Load canvas offset information
            if 'min_x_offset' in mask_data and 'min_y_offset' in mask_data:
                min_x_offset = int(mask_data['min_x_offset'])
                min_y_offset = int(mask_data['min_y_offset'])
                print(f"    Cellmask: {stitched_cellmask.shape[0]} x {stitched_cellmask.shape[1]} "
                      f"(canvas offset: x={min_x_offset}, y={min_y_offset})")
            else:
                print("  WARNING: Canvas offsets not found - position mapping may fail!")
                min_x_offset = 0
                min_y_offset = 0
                print(f"    Cellmask: {stitched_cellmask.shape[0]} x {stitched_cellmask.shape[1]}")

        # Filter cells by slice and QC
        in_slice = slice_ids == slice_id
        slice_marker_qc = in_slice & marker_qc_pass

        drawn = slice_marker_qc & (marker_expression >= floor)
        total_cells = np.sum(in_slice)
        marker_cells = np.sum(slice_marker_qc)
        cells_drawn = np.sum(drawn)

        print(f"  Cells in slice: {total_cells}")
        print(f"  {marker_label}+ QC-passing: {marker_cells}")
        print(f"  Drawn (>= {floor} rolonies): {cells_drawn}")
        print(f"  At ceiling (>= {ceiling}): "
              f"{np.sum(drawn & (marker_expression >= ceiling))}")

        slice_cell_indices = np.where(drawn)[0]

        if cells_drawn == 0:
            print(f"  WARNING: No cells at >= {floor} rolonies, "
                  f"saving cell mask only")

        # Create overlay
        print("  Creating overlay...")
        overlay_start = time.time()

        # Initialize RGB overlay with cell mask background
        cell_exists_mask = stitched_cellmask > 0
        cellmask_gray = cell_exists_mask.astype(float) * CELLMASK_BRIGHTNESS * CELLMASK_SCALE
        overlay_rgb = np.stack([cellmask_gray, cellmask_gray, cellmask_gray], axis=2)

        # For comparison figure
        marker_only = np.zeros_like(overlay_rgb)

        # Color each marker+ cell using position-based lookup
        cells_mapped = 0
        cells_not_found = 0

        if cells_drawn > 0:
            for cell_idx in slice_cell_indices:
                # Full-res position -> canvas space (x2) -> downsampled image
                pos_x_fullres = pos[cell_idx, 0]
                pos_y_fullres = pos[cell_idx, 1]

                # Same factor on both axes, matching how the image was resampled.
                # MATLAB: x_stitched = round((pos_x_fullres * 2 - (min_x_offset - 1)) / downsample)
                # Python: same formula but 0-indexed result
                x_stitched = round((pos_x_fullres * 2 - (min_x_offset - 1)) / DOWNSAMPLE_XY)
                y_stitched = round((pos_y_fullres * 2 - (min_y_offset - 1)) / DOWNSAMPLE_XY)

                # Convert to 0-indexed for Python
                x_stitched = int(x_stitched) - 1
                y_stitched = int(y_stitched) - 1

                # Check bounds
                if (x_stitched < 0 or x_stitched >= stitched_cellmask.shape[1] or
                    y_stitched < 0 or y_stitched >= stitched_cellmask.shape[0]):
                    cells_not_found += 1
                    continue

                # Look up cell ID from stitched cellmask
                cell_id = stitched_cellmask[y_stitched, x_stitched]
                if cell_id == 0:
                    cells_not_found += 1
                    continue

                # Find all pixels belonging to this cell
                cell_mask = stitched_cellmask == cell_id

                # Absolute rolony count -> colour. The domain is a constant and
                # `floor` is not in it, so a given count is the same colour in
                # every section and at every cutoff.
                rgb = ramp_rgb(marker_expression[cell_idx], cmap, ceiling)

                overlay_rgb[cell_mask] = rgb

                # Also update marker_only for comparison
                marker_only[cell_mask] = rgb

                cells_mapped += 1

            overlay_time = time.time() - overlay_start
            print(f"    Mapped {cells_mapped} / {len(slice_cell_indices)} cells "
                  f"({100*cells_mapped/len(slice_cell_indices):.1f}%) in {overlay_time:.2f} sec")

            if cells_not_found > 0:
                print(f"    Cells not found: {cells_not_found}")
        else:
            overlay_time = time.time() - overlay_start
            print(f"    Cell mask only (no overlay) in {overlay_time:.2f} sec")

        hist_rows.append((slice_id, marker_cells, cells_mapped))

        # Save overlay
        output_name = f"{base_name}_{marker_label}_cellmask.tif"
        output_path = output_dir / output_name
        overlay_rgb_uint8 = (np.clip(overlay_rgb, 0, 1) * 255).astype(np.uint8)
        imwrite_tiff(output_path, overlay_rgb_uint8)
        print(f"    Saved overlay: {output_name} ({output_path.stat().st_size/1e6:.1f} MB)")

        # Generate comparison figure
        print("  Generating comparison figure...")
        fig_path = create_comparison_figure(
            cellmask_gray, overlay_rgb, marker_only,
            slice_id, cells_mapped, output_dir, base_name,
            marker_label=marker_label,
        )
        print(f"    Saved comparison: {Path(fig_path).name}\n")

    # Painted vs. eligible, per section. A bar short of its dashed total is
    # the floor cut plus whatever the centroid lookup missed; the per-section
    # prints above separate the two. Sorted numerically -- cellmask_files is
    # sorted lexicographically, which puts slice10 before slice9.
    if test_mode or target_slice is not None:
        print("Histogram: skipped, partial run (the figure covers every section)\n")
    elif hist_rows:
        hist_rows.sort()
        stem = f"cell_count_histogram_ge{floor}_rolonies"
        if excluded:
            stem += "_ex" + "_".join(str(s) for s in excluded)
        hist_path = create_histogram(
            [r[0] for r in hist_rows], [r[2] for r in hist_rows],
            [r[1] for r in hist_rows], float(floor), output_dir,
            criterion_label=f">= {floor} rolonies",
            filename=f"{stem}.png",
            note=excluded_note or None,
            marker_label=marker_label,
        )
        print(f"Histogram: {Path(hist_path).name}\n")

    # Summary
    print("=" * 40)
    print("CELL MASK OVERLAY COMPLETE")
    print("=" * 40)
    print(f"Output directory: {output_dir}")
    print(f"Subslices processed: {len(cellmask_files)}")
    print(f"\nResolution: {TARGET_XY_UM_PER_PX:.4f} um/px, square pixels")
    print("\nNext steps:")
    print("  1. Review overlays in output directory")
    print("  2. Fit on the ALIGN tifs, not these -- generate_alignment_tif.py")
    print("  3. (Optional) Re-run at another --min-rolonies; its own folder")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate marker cell mask overlays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--marker', '-m',
        choices=marker_names(),
        default='mscarlet',
        help='Which readout column to paint (default: mscarlet). Selects the '
             'colour ramp, the cutoff/cap defaults and the output root, so the '
             'markers never overwrite each other.'
    )
    parser.add_argument(
        '--slice', '-s',
        type=int,
        default=None,
        help='Process specific slice only'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test mode: process first subslice only'
    )
    parser.add_argument(
        '--min-rolonies', '-n',
        type=int,
        default=None,
        help='Draw cutoff: below this a cell is not drawn (default: the '
             "marker's floor in marker_profiles). Names the output folder. Does not "
             'move the colour ramp -- a survivor keeps its colour at every '
             'cutoff.'
    )
    parser.add_argument(
        '--exclude-slices',
        type=int,
        nargs='+',
        metavar='N',
        help='Drop these sections from the whole run; the numbers land in the '
             'histogram filename'
    )

    args = parser.parse_args()

    generate_marker_cellmask_subslice(
        target_slice=args.slice,
        test_mode=args.test,
        exclude_slices=args.exclude_slices,
        min_rolonies=args.min_rolonies,
        marker=args.marker,
    )


if __name__ == '__main__':
    main()
