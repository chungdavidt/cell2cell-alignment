#!/usr/bin/env python3
"""
Generate mScarlet Cell Mask Overlays for Subslices (ANISOTROPIC).

Creates mScarlet overlays on CELL MASK background using ANISOTROPIC downsampling.

This is a Python port of generate_mscarlet_cellmask_subslice_anisotropic.m with exact fidelity.

ANISOTROPIC SCALING (matches in vivo resolution correctly):
    - X: 0.32 -> 2.34 um/px (7.31x downsample, matches in vivo X/Y)
    - Y: 0.32 -> 1.0 um/px (3.125x downsample, matches in vivo Z after xrotate=90)

This pipeline:
    - Shows cell masks in grayscale (adjustable brightness)
    - Highlights mScarlet+ cells in red
    - No DAPI background
    - Uses ANISOTROPIC resolution matching

Usage:
    python generate_mscarlet_cellmask_subslice_anisotropic.py
    python generate_mscarlet_cellmask_subslice_anisotropic.py --threshold 0.3
    python generate_mscarlet_cellmask_subslice_anisotropic.py --threshold 0.3 --cellmask 0.5
    python generate_mscarlet_cellmask_subslice_anisotropic.py --threshold 0.3 --slice 22
    python generate_mscarlet_cellmask_subslice_anisotropic.py --test

Input:
    - HYB_subslice_stitched_tif_downsampled_micronwise_anisotropic/
    - filt_neurons.mat (for cell positions and expression)

Output:
    - mScarlet_cellmask_subslice/threshold_X.XX_cellmask_X.XX_anisotropic/
        * slice{N}_subslice_mScarlet_cellmask.tif
        * slice{N}_subslice_comparison.png
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import sparse
import re
import time

from config import (
    FILT_NEURONS_PATH,
    HYB_DOWNSAMPLED_DIR,
    MSCARLET_CELLMASK_DIR,
    MSCARLET_COLUMN_INDEX,
    QC_MIN_READS,
    QC_MIN_GENES,
    EXVIVO_UM_PER_PX,
    INVIVO_XY_UM_PER_PX,
    INVIVO_Z_UM_PER_PX,
    CELLMASK_BRIGHTNESS,
    RED_OPACITY,
    MSCARLET_BOOST,
    get_threshold_folder,
)
from utils.mat_io import load_filt_neurons, load_mat, load_cellmask_h5, get_expression_column
from utils.image_io import imwrite_tiff
from utils.visualization import create_comparison_figure


def generate_mscarlet_cellmask_subslice_anisotropic(
    min_mscarlet_intensity: float = 0.0,
    cellmask_intensity: float = 0.5,
    target_slice: int = None,
    test_mode: bool = False,
):
    """
    Generate mScarlet cell mask overlays.

    Args:
        min_mscarlet_intensity: Normalized mScarlet threshold (0-1)
        cellmask_intensity: Cell mask background brightness multiplier (0-1)
        target_slice: Process specific slice only
        test_mode: Process first subslice only
    """
    input_dir = Path(HYB_DOWNSAMPLED_DIR)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Anisotropic subslices not found!\n"
            f"Run downsample_subslices_cellmask_anisotropic.py first.\n"
            f"Expected: {input_dir}"
        )

    # Create output directory with threshold info
    threshold_folder = get_threshold_folder(min_mscarlet_intensity, cellmask_intensity)
    output_dir = Path(MSCARLET_CELLMASK_DIR) / threshold_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 40)
    print("GENERATE mSCARLET CELL MASK OVERLAYS")
    print("(ANISOTROPIC RESOLUTION)")
    print("=" * 40)
    print(f"mScarlet threshold: {min_mscarlet_intensity:.2f}")
    print(f"Cell mask brightness: {cellmask_intensity:.2f} ({cellmask_intensity*100:.0f}% of base)")
    print()
    print("Resolution matching:")
    print(f"  Ex vivo original: {EXVIVO_UM_PER_PX:.2f} um/px")
    print(f"  Target X (-> in vivo X/Y): {INVIVO_XY_UM_PER_PX:.2f} um/px")
    print(f"  Target Y (-> in vivo Z): {INVIVO_Z_UM_PER_PX:.2f} um/px")
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
    mscarlet_expression = get_expression_column(expmat, MSCARLET_COLUMN_INDEX)

    # Calculate global max from QC-passed cells ONLY
    max_expr = np.max(mscarlet_expression[pass_qc])
    print(f"Global max mScarlet (QC-passed): {max_expr} transcripts")

    # Normalize ALL cells using this global max
    mscarlet_normalized = mscarlet_expression / max_expr

    # mScarlet+ cells
    mscarlet_positive = mscarlet_expression > 0
    print(f"  mScarlet+ cells: {np.sum(mscarlet_positive)}")

    # Combined filter (QC-passed AND mScarlet+)
    mscarlet_qc_pass = pass_qc & mscarlet_positive
    print(f"  mScarlet+ QC-passing: {np.sum(mscarlet_qc_pass)}\n")

    # Position scale factors for ANISOTROPIC downsampling
    downsample_x = INVIVO_XY_UM_PER_PX / EXVIVO_UM_PER_PX  # 7.31
    downsample_y = INVIVO_Z_UM_PER_PX / EXVIVO_UM_PER_PX   # 3.125

    print("Position scale factors (from full-res):")
    print(f"  X: {1/downsample_x:.6f} (x2 for canvas, then anisotropic scale)")
    print(f"  Y: {1/downsample_y:.6f} (x2 for canvas, then anisotropic scale)\n")

    # Find subslices to process (check for .h5 first, fall back to .mat)
    cellmask_files = list(input_dir.glob("slice*_subslice_CELLMASK.h5"))
    use_h5_format = True

    if not cellmask_files:
        cellmask_files = list(input_dir.glob("slice*_subslice_CELLMASK.mat"))
        use_h5_format = False

    if not cellmask_files:
        raise FileNotFoundError(f"No anisotropic subslices found in: {input_dir}")

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

        total_cells = np.sum(in_slice)
        mscarlet_cells = np.sum(slice_mscarlet_qc)
        cells_above_thresh = np.sum(slice_mscarlet_qc & (mscarlet_normalized >= min_mscarlet_intensity))

        print(f"  Cells in slice: {total_cells}")
        print(f"  mScarlet+ QC-passing: {mscarlet_cells}")
        print(f"  Above threshold ({min_mscarlet_intensity:.2f}): {cells_above_thresh}")

        # Get cells above threshold
        slice_cell_indices = np.where(slice_mscarlet_qc & (mscarlet_normalized >= min_mscarlet_intensity))[0]

        if cells_above_thresh == 0:
            print("  WARNING: No cells above threshold, saving cell mask only")

        # Create overlay
        print("  Creating overlay...")
        overlay_start = time.time()

        # Initialize RGB overlay with cell mask background
        cell_exists_mask = stitched_cellmask > 0
        cellmask_gray = cell_exists_mask.astype(float) * CELLMASK_BRIGHTNESS * cellmask_intensity
        overlay_rgb = np.stack([cellmask_gray, cellmask_gray, cellmask_gray], axis=2)

        # For comparison figure
        mscarlet_only = np.zeros_like(overlay_rgb)

        # Color each mScarlet+ cell using position-based lookup
        cells_mapped = 0
        cells_not_found = 0

        if cells_above_thresh > 0:
            for cell_idx in slice_cell_indices:
                # ANISOTROPIC position mapping
                # Full-res position * 2 (canvas space) * scale_factor - offset correction
                pos_x_fullres = pos[cell_idx, 0]
                pos_y_fullres = pos[cell_idx, 1]

                # Apply SEPARATE scale factors for X and Y
                # Canvas space: pos * 2
                # Then downsample with anisotropic factors
                # MATLAB: x_stitched = round((pos_x_fullres * 2 - (min_x_offset - 1)) / downsample_x)
                # Python: same formula but 0-indexed result
                x_stitched = round((pos_x_fullres * 2 - (min_x_offset - 1)) / downsample_x)
                y_stitched = round((pos_y_fullres * 2 - (min_y_offset - 1)) / downsample_y)

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

                # Color intensity based on normalized expression (with boost)
                intensity = min(mscarlet_normalized[cell_idx] * MSCARLET_BOOST, 1.0)
                red_value = intensity * RED_OPACITY

                # Apply red color to cell region
                overlay_rgb[cell_mask, 0] = red_value  # Red channel
                overlay_rgb[cell_mask, 1] = 0          # Green channel
                overlay_rgb[cell_mask, 2] = 0          # Blue channel

                # Also update mscarlet_only for comparison
                mscarlet_only[cell_mask, 0] = red_value

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
    print("ANISOTROPIC CELL MASK OVERLAY COMPLETE")
    print("=" * 40)
    print(f"Output directory: {output_dir}")
    print(f"Subslices processed: {len(cellmask_files)}")
    print(f"\nResolution: Anisotropic (X={INVIVO_XY_UM_PER_PX:.2f}, Y={INVIVO_Z_UM_PER_PX:.2f} um/px)")
    print("\nNext steps:")
    print("  1. Review overlays in output directory")
    print("  2. Add overlays to LineStuffUp graph for alignment")
    print("  3. (Optional) Adjust threshold and regenerate")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate mScarlet cell mask overlays (anisotropic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--threshold', '-th',
        type=float,
        default=0.0,
        help='Normalized mScarlet threshold (0-1), default: 0'
    )
    parser.add_argument(
        '--cellmask', '-cm',
        type=float,
        default=0.5,
        help='Cell mask brightness multiplier (0-1), default: 0.5'
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

    generate_mscarlet_cellmask_subslice_anisotropic(
        min_mscarlet_intensity=args.threshold,
        cellmask_intensity=args.cellmask,
        target_slice=args.slice,
        test_mode=args.test,
    )


if __name__ == '__main__':
    main()
