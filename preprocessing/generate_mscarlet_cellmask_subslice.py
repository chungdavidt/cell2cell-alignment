#!/usr/bin/env python3
"""
Generate mScarlet Cell Mask Overlays for Subslices.

Creates mScarlet overlays on CELL MASK background, on the downsampled subslices.

Ported from the lab's generate_mscarlet_cellmask_subslice_anisotropic.m -- their filename, from the era of the
two-factor resample. This port is ISOTROPIC: one DOWNSAMPLE_XY covers both
in-plane axes. Pre-isotropic code is frozen in archive/anisotropic_preprocessing/.

Cell centroids from filt_neurons.mat are mapped into the downsampled image with
DOWNSAMPLE_XY, the same factor downsample_subslices_cellmask.py used
to resample it. The two MUST agree or every cell lands off its mask.

This pipeline:
    - Shows cell masks in flat grey
    - Paints cells at >= ROLONY_FLOOR mScarlet rolonies on a dark red -> orange
      -> yellow ramp, linear in rolony count over the FIXED domain
      [ROLONY_FLOOR, ROLONY_CEILING]
    - No DAPI background

The ramp is absolute, not normalized. A 9-rolony cell is the same red in every
section and every run, so two runs compare directly. The count / max_expr *
MSCARLET_BOOST form this replaces tied every cell's brightness to the single
brightest cell in the dataset -- a value that moves with the QC gates -- and
rendered BY95's median marker cell (1 rolony) darker than the grey mask behind
it. Both config constants stay defined for the two scripts still using them.

The ramp is a hue ramp, not a red one. Counts are integers, so a 3-to-15 domain
has 13 levels; on a red-only ramp those differ in ONE channel 12.75 uint8 apart
and read as flat. The three anchor colours are check_rolony_cutoff.py's. Note
the DOMAINS still differ -- that tool ramps over [1, --saturate-at] by design --
so a given count matches across the two only when the bounds are set to match.

ROLONY_FLOOR / ROLONY_CEILING are BY95 numbers, hardcoded while the bounds are
being tried out. They are per-brain like every other rolony gate: re-pick them
with check_rolony_cutoff.py before running another dataset.

Usage:
    python generate_mscarlet_cellmask_subslice.py
    python generate_mscarlet_cellmask_subslice.py --slice 22
    python generate_mscarlet_cellmask_subslice.py --test

Input:
    - HYB_subslice_stitched_tif_downsampled_micronwise/
    - filt_neurons.mat (for cell positions and expression)

Output:
    - mScarlet_cellmask_subslice/rolony_{FLOOR}_{CEILING}/
        * slice{N}_subslice_mScarlet_cellmask.tif
        * slice{N}_subslice_comparison.png
        * rolony_ramp_legend.png -- one per folder, not per slice. The domain
          is absolute, so a count maps to the same colour in every image
          written here; a different FLOOR/CEILING is a different folder with
          its own legend.

      The folder name records the ramp only. QC_MIN_READS / QC_MIN_GENES also
      decide which cells are eligible to be drawn; they live in local_config.py
      and the run prints them.
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
from utilities.mat_io import load_filt_neurons, load_mat, load_cellmask_h5, get_expression_column, resolve_marker_column
from utilities.image_io import imwrite_tiff
from utilities.visualization import create_comparison_figure

# Rolony count -> colour, fixed domain. BY95 numbers; re-pick per brain.
ROLONY_FLOOR = 3        # below this a cell is left as grey mask, not drawn
ROLONY_CEILING = 15     # at or above this the colour saturates
CELLMASK_SCALE = 0.5    # scales CELLMASK_BRIGHTNESS -> 0.125, uint8 32

# dark red -> orange -> yellow, check_rolony_cutoff.py's anchors. The floor at
# uint8 (115, 0, 0) clears the grey field at 32.
RAMP_COLORS = [(0.45, 0.0, 0.0), (1.0, 0.35, 0.0), (1.0, 0.95, 0.25)]
_RAMP = LinearSegmentedColormap.from_list("rolony", RAMP_COLORS)


def ramp_rgb(count):
    """One rolony count -> RGB on the fixed [ROLONY_FLOOR, ROLONY_CEILING] ramp."""
    frac = np.clip(
        (count - ROLONY_FLOOR) / (ROLONY_CEILING - ROLONY_FLOOR), 0.0, 1.0)
    return _RAMP(frac)[:3]


def write_ramp_legend(output_dir):
    """One legend per output folder: every colour the ramp can produce.

    Discrete swatches, not a gradient. Counts are integers, so the ramp has
    exactly ROLONY_CEILING - ROLONY_FLOOR + 1 reachable colours and a swatch
    per count is a lookup -- read a cell's colour off the image, read its
    rolony count off the legend. Written as its own file, never burned into
    the overlay TIF, which shares a pixel grid with the ALIGN tif and with
    export_subslice_cells.py's y_node/x_node.
    """
    import matplotlib.pyplot as plt

    counts = list(range(ROLONY_FLOOR, ROLONY_CEILING + 1))
    fig, ax = plt.subplots(figsize=(2.6, 0.34 * (len(counts) + 2) + 0.6))

    for row, count in enumerate(counts):
        ax.add_patch(plt.Rectangle((0, row), 1, 0.86, color=ramp_rgb(count)))
        label = f"{count}+" if count == ROLONY_CEILING else str(count)
        ax.text(1.15, row + 0.43, label, va="center", fontsize=9)

    # The floor is a cutoff as well as the ramp's bottom: below it a cell is
    # drawn in the mask field's grey, indistinguishable from marker-negative.
    grey = CELLMASK_BRIGHTNESS * CELLMASK_SCALE
    ax.add_patch(plt.Rectangle((0, -1.4), 1, 0.86, color=(grey, grey, grey)))
    ax.text(1.15, -0.97, f"< {ROLONY_FLOOR}", va="center", fontsize=9)

    ax.set_xlim(-0.1, 2.6)
    ax.set_ylim(-1.9, len(counts) + 0.4)
    ax.axis("off")
    ax.set_title("mScarlet rolonies", fontsize=10, pad=8)
    fig.text(0.5, 0.015, "fixed ramp, absolute counts", ha="center", fontsize=7)

    out_path = Path(output_dir) / "rolony_ramp_legend.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def generate_mscarlet_cellmask_subslice(
    target_slice: int = None,
    test_mode: bool = False,
):
    """
    Generate mScarlet cell mask overlays.

    Args:
        target_slice: Process specific slice only
        test_mode: Process first subslice only
    """
    input_dir = Path(HYB_DOWNSAMPLED_DIR)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Downsampled subslices not found!\n"
            f"Run downsample_subslices_cellmask.py first.\n"
            f"Expected: {input_dir}"
        )

    # One folder per ramp, so changing the bounds writes beside the last run
    # instead of overwriting it.
    output_dir = Path(MSCARLET_CELLMASK_DIR) / f"rolony_{ROLONY_FLOOR}_{ROLONY_CEILING}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 40)
    print("GENERATE mSCARLET CELL MASK OVERLAYS")
    print("=" * 40)
    print(f"Rolony ramp: {ROLONY_FLOOR} -> {ROLONY_CEILING}+ rolonies, "
          f"dark red -> orange -> yellow (fixed, absolute)")
    print(f"  below {ROLONY_FLOOR} rolonies: not drawn")
    print(f"Cell mask brightness: {CELLMASK_BRIGHTNESS * CELLMASK_SCALE:.3f}")
    legend_path = write_ramp_legend(output_dir)
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

    # Get mScarlet expression
    mscarlet_col = resolve_marker_column(
        filt_neurons, MSCARLET_GENE_NAME, MSCARLET_COLUMN_INDEX)
    mscarlet_expression = get_expression_column(expmat, mscarlet_col)

    # Calculate global max from QC-passed cells ONLY
    max_expr = np.max(mscarlet_expression[pass_qc])
    print(f"Global max mScarlet (QC-passed): {max_expr} transcripts")

    # Reported only. The ramp is absolute, so nothing divides by this.
    mscarlet_positive = mscarlet_expression > 0
    print(f"  mScarlet+ cells: {np.sum(mscarlet_positive)}")

    # Combined filter (QC-passed AND mScarlet+)
    mscarlet_qc_pass = pass_qc & mscarlet_positive
    print(f"  mScarlet+ QC-passing: {np.sum(mscarlet_qc_pass)}")
    print(f"  at >= {ROLONY_FLOOR} rolonies (drawn): "
          f"{np.sum(mscarlet_qc_pass & (mscarlet_expression >= ROLONY_FLOOR))}")
    print(f"  at >= {ROLONY_CEILING} rolonies (saturated): "
          f"{np.sum(mscarlet_qc_pass & (mscarlet_expression >= ROLONY_CEILING))}\n")

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
        slice_mscarlet_qc = in_slice & mscarlet_qc_pass

        drawn = slice_mscarlet_qc & (mscarlet_expression >= ROLONY_FLOOR)
        total_cells = np.sum(in_slice)
        mscarlet_cells = np.sum(slice_mscarlet_qc)
        cells_drawn = np.sum(drawn)

        print(f"  Cells in slice: {total_cells}")
        print(f"  mScarlet+ QC-passing: {mscarlet_cells}")
        print(f"  Drawn (>= {ROLONY_FLOOR} rolonies): {cells_drawn}")
        print(f"  At ceiling (>= {ROLONY_CEILING}): "
              f"{np.sum(drawn & (mscarlet_expression >= ROLONY_CEILING))}")

        slice_cell_indices = np.where(drawn)[0]

        if cells_drawn == 0:
            print(f"  WARNING: No cells at >= {ROLONY_FLOOR} rolonies, "
                  f"saving cell mask only")

        # Create overlay
        print("  Creating overlay...")
        overlay_start = time.time()

        # Initialize RGB overlay with cell mask background
        cell_exists_mask = stitched_cellmask > 0
        cellmask_gray = cell_exists_mask.astype(float) * CELLMASK_BRIGHTNESS * CELLMASK_SCALE
        overlay_rgb = np.stack([cellmask_gray, cellmask_gray, cellmask_gray], axis=2)

        # For comparison figure
        mscarlet_only = np.zeros_like(overlay_rgb)

        # Color each mScarlet+ cell using position-based lookup
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

                # Absolute rolony count -> colour. The domain is fixed, so a
                # given count is the same colour in every section and every run,
                # and moving the bounds never re-shades the cells between them.
                rgb = ramp_rgb(mscarlet_expression[cell_idx])

                overlay_rgb[cell_mask] = rgb

                # Also update mscarlet_only for comparison
                mscarlet_only[cell_mask] = rgb

                cells_mapped += 1

            overlay_time = time.time() - overlay_start
            print(f"    Mapped {cells_mapped} / {len(slice_cell_indices)} cells "
                  f"({100*cells_mapped/len(slice_cell_indices):.1f}%) in {overlay_time:.2f} sec")

            if cells_not_found > 0:
                print(f"    Cells not found: {cells_not_found}")
        else:
            overlay_time = time.time() - overlay_start
            print(f"    Cell mask only (no overlay) in {overlay_time:.2f} sec")

        # Save overlay
        output_name = f"{base_name}_mScarlet_cellmask.tif"
        output_path = output_dir / output_name
        overlay_rgb_uint8 = (np.clip(overlay_rgb, 0, 1) * 255).astype(np.uint8)
        imwrite_tiff(output_path, overlay_rgb_uint8)
        print(f"    Saved overlay: {output_name} ({output_path.stat().st_size/1e6:.1f} MB)")

        # Generate comparison figure
        print("  Generating comparison figure...")
        fig_path = create_comparison_figure(
            cellmask_gray, overlay_rgb, mscarlet_only,
            slice_id, cells_mapped, output_dir, base_name
        )
        print(f"    Saved comparison: {Path(fig_path).name}\n")

    # Summary
    print("=" * 40)
    print("CELL MASK OVERLAY COMPLETE")
    print("=" * 40)
    print(f"Output directory: {output_dir}")
    print(f"Subslices processed: {len(cellmask_files)}")
    print(f"\nResolution: {TARGET_XY_UM_PER_PX:.4f} um/px, square pixels")
    print("\nNext steps:")
    print("  1. Review overlays in output directory")
    print("  2. Add overlays to LineStuffUp graph for alignment")
    print("  3. (Optional) Adjust ROLONY_FLOOR / ROLONY_CEILING and regenerate")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate mScarlet cell mask overlays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
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

    args = parser.parse_args()

    generate_mscarlet_cellmask_subslice(
        target_slice=args.slice,
        test_mode=args.test,
    )


if __name__ == '__main__':
    main()
