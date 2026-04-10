#!/usr/bin/env python3
"""
Interactive mScarlet Threshold Viewer (CELL MASK - ANISOTROPIC).

Displays multiple subslices with mScarlet+ cells on CELL MASK background.
Uses ANISOTROPIC downsampling for correct resolution matching.

This is a Python port of interactive_mscarlet_threshold_cellmask_subslice_anisotropic.m
with exact fidelity.

ANISOTROPIC SCALING:
    - X: 0.32 -> 2.34 um/px (matches in vivo X/Y)
    - Y: 0.32 -> 1.0 um/px (matches in vivo Z after xrotate=90)

PERFORMANCE: Data is cached after first run - subsequent threshold changes are FAST!

Prerequisites:
    1. Run stitch_subslices.py for desired slices
    2. Run downsample_subslices_cellmask_anisotropic.py for desired slices

Usage:
    python interactive_mscarlet_threshold_cellmask_subslice_anisotropic.py
    python interactive_mscarlet_threshold_cellmask_subslice_anisotropic.py --threshold 0.3
    python interactive_mscarlet_threshold_cellmask_subslice_anisotropic.py --threshold 0.3 --cellmask 0.5
    python interactive_mscarlet_threshold_cellmask_subslice_anisotropic.py --slices 22 44
    python interactive_mscarlet_threshold_cellmask_subslice_anisotropic.py --first 5
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import sparse
import re
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt

from config import (
    FILT_NEURONS_PATH,
    HYB_DOWNSAMPLED_DIR,
    MSCARLET_INTERACTIVE_DIR,
    MSCARLET_COLUMN_INDEX,
    QC_MIN_READS,
    QC_MIN_GENES,
    EXVIVO_UM_PER_PX,
    INVIVO_XY_UM_PER_PX,
    INVIVO_Z_UM_PER_PX,
)
from utilities.mat_io import load_filt_neurons, load_mat, load_cellmask_h5, get_expression_column
from utilities.visualization import create_histogram

# Module-level cache for data persistence across function calls
_CACHE = {
    'subslice_data': None,
    'max_expr': None,
}


def interactive_mscarlet_threshold_cellmask_subslice_anisotropic(
    min_mscarlet_intensity: float = 0.0,
    cellmask_intensity: float = 0.5,
    force_reload: bool = False,
    slice_selection=None,
):
    """
    Interactive mScarlet threshold viewer.

    Args:
        min_mscarlet_intensity: Minimum normalized mScarlet expression (0-1)
        cellmask_intensity: Cell mask background brightness (0-1)
        force_reload: Force reload from disk (default: False)
        slice_selection: Subslices to process:
            - None or 'all': All subslices
            - int: First N subslices
            - list: Specific slice IDs
    """
    CELLMASK_BRIGHTNESS = 0.3
    RED_OPACITY = 1.0
    RED_BOOST = 2.0
    SLICES_PER_FIGURE = 6  # 2x3 grid

    # Determine selection mode
    if slice_selection is None or slice_selection == 'all':
        selection_mode = 'all'
        selection_desc = 'All subslices'
    elif isinstance(slice_selection, int):
        selection_mode = 'first_n'
        selection_desc = f'Preview mode: first {slice_selection} subslices'
    else:
        selection_mode = 'specific'
        selection_desc = f'Specific subslices: {slice_selection}'

    print("=" * 40)
    print("MSCARLET CELL MASK VIEWER (ANISOTROPIC)")
    print("=" * 40)
    print(f"mScarlet threshold: {min_mscarlet_intensity:.2f} ({selection_desc})")
    print(f"Cell mask brightness: {cellmask_intensity:.2f} ({cellmask_intensity*100:.0f}% of base)\n")

    # Create output directory
    threshold_folder = f"threshold_{min_mscarlet_intensity:.2f}_cellmask_{cellmask_intensity:.2f}"
    output_dir = Path(MSCARLET_INTERACTIVE_DIR) / threshold_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    use_cache = _CACHE['subslice_data'] is not None and not force_reload

    if use_cache:
        print("Using cached data (FAST!)")
        print("   (To force reload: add --reload flag)\n")
        subslice_data = _CACHE['subslice_data']
        max_expr = _CACHE['max_expr']
    else:
        # Load data
        input_dir = Path(HYB_DOWNSAMPLED_DIR)

        if not input_dir.exists():
            raise FileNotFoundError(
                f"Anisotropic subslice directory not found!\n"
                f"Run stitch_subslices.py and downsample_subslices_cellmask_anisotropic.py first.\n"
                f"Expected: {input_dir}"
            )

        # Load filt_neurons
        print("Loading filt_neurons...")
        filt_neurons = load_filt_neurons(FILT_NEURONS_PATH)
        expmat = filt_neurons['expmat']

        # Apply QC filters
        if sparse.issparse(expmat):
            total_reads = np.asarray(expmat.sum(axis=1)).flatten()
            total_genes = np.asarray((expmat > 0).sum(axis=1)).flatten()
        else:
            total_reads = np.sum(expmat, axis=1)
            total_genes = np.sum(expmat > 0, axis=1)

        pass_qc = (total_reads >= QC_MIN_READS) & (total_genes >= QC_MIN_GENES)

        # Get mScarlet expression
        mscarlet_expression = get_expression_column(expmat, MSCARLET_COLUMN_INDEX)
        mscarlet_positive = pass_qc & (mscarlet_expression > 0)
        max_expr = np.max(mscarlet_expression[mscarlet_positive])

        print(f"mScarlet+ QC-passed cells: {np.sum(mscarlet_positive)}")
        print(f"Expression range: {np.min(mscarlet_expression[mscarlet_positive]):.0f} - {max_expr:.0f} transcripts")
        print(f"\nResolution: Anisotropic (X={INVIVO_XY_UM_PER_PX:.2f}, Y={INVIVO_Z_UM_PER_PX:.2f} um/px)\n")

        # Find available subslices (check for .h5 first, fall back to .mat)
        cellmask_files = list(input_dir.glob("slice*_subslice_CELLMASK.h5"))
        use_h5_format = True

        if not cellmask_files:
            cellmask_files = list(input_dir.glob("slice*_subslice_CELLMASK.mat"))
            use_h5_format = False

        if not cellmask_files:
            raise FileNotFoundError(f"No anisotropic subslices found in: {input_dir}")

        # Sort by slice ID
        def get_slice_id(f):
            match = re.search(r'slice(\d+)_subslice', f.name)
            return int(match.group(1)) if match else 0

        cellmask_files.sort(key=get_slice_id)
        print(f"Found {len(cellmask_files)} subslices ({'H5' if use_h5_format else 'MAT'} format)")

        # Load all subslices into memory
        print("Loading subslices and pre-computing cell masks...")
        print("(This happens once - changing thresholds will then be fast!)")

        # Anisotropic scale factors
        downsample_x = INVIVO_XY_UM_PER_PX / EXVIVO_UM_PER_PX
        downsample_y = INVIVO_Z_UM_PER_PX / EXVIVO_UM_PER_PX

        slice_ids_array = np.asarray(filt_neurons['slice']).flatten()
        pos = np.asarray(filt_neurons['pos'])

        subslice_data = []

        for cellmask_file in cellmask_files:
            base_name = cellmask_file.stem.replace('_CELLMASK', '')

            # Parse slice ID
            match = re.search(r'slice(\d+)_subslice', base_name)
            if not match:
                continue
            slice_id = int(match.group(1))

            # Find cells in this slice
            in_slice = slice_ids_array == slice_id
            slice_mscarlet_cells = in_slice & mscarlet_positive

            if np.sum(slice_mscarlet_cells) == 0:
                continue

            # Load cellmask
            if use_h5_format:
                cellmask, mask_metadata = load_cellmask_h5(cellmask_file)
                min_x_offset = mask_metadata.get('min_x_offset', 0)
                min_y_offset = mask_metadata.get('min_y_offset', 0)
            else:
                mask_data = load_mat(cellmask_file)
                if 'cellmask_down' not in mask_data:
                    print(f"  WARNING: cellmask_down not found in {cellmask_file.name}, skipping")
                    continue
                cellmask = np.asarray(mask_data['cellmask_down'])
                min_x_offset = int(mask_data.get('min_x_offset', 0))
                min_y_offset = int(mask_data.get('min_y_offset', 0))

            # Legacy compatibility check - skip if no cellmask
            if cellmask is None:
                print(f"  WARNING: No cellmask found in {cellmask_file.name}, skipping")
                continue

            # Calculate ANISOTROPIC positions
            cell_indices = np.where(slice_mscarlet_cells)[0]
            cell_positions = pos[cell_indices]

            # Apply anisotropic position transformation
            adjusted_x = (cell_positions[:, 0] * 2 - (min_x_offset - 1)) / downsample_x
            adjusted_y = (cell_positions[:, 1] * 2 - (min_y_offset - 1)) / downsample_y
            adjusted_positions = np.column_stack([adjusted_x, adjusted_y])

            # PRE-COMPUTE cell pixel lists for fast threshold changes
            cell_pixel_lists = []

            for i in range(len(cell_indices)):
                x = int(round(adjusted_positions[i, 0])) - 1  # 0-indexed
                y = int(round(adjusted_positions[i, 1])) - 1  # 0-indexed

                # Check bounds
                if x < 0 or x >= cellmask.shape[1] or y < 0 or y >= cellmask.shape[0]:
                    cell_pixel_lists.append(None)
                    continue

                # Get cell ID and find all pixels
                cell_id = cellmask[y, x]
                if cell_id == 0:
                    cell_pixel_lists.append(None)
                    continue

                # Store linear indices for this cell
                pixel_indices = np.where(cellmask == cell_id)
                cell_pixel_lists.append(pixel_indices)

            # Store subslice data
            subslice_data.append({
                'slice_id': slice_id,
                'cellmask': cellmask,
                'cell_indices': cell_indices,
                'expressions': mscarlet_expression[cell_indices],
                'positions': adjusted_positions,
                'cell_pixel_lists': cell_pixel_lists,
            })

            print(f"  Loaded subslice {slice_id} ({np.sum(slice_mscarlet_cells)} mScarlet+ cells, pre-computed masks)")

        if len(subslice_data) == 0:
            raise ValueError("No subslices loaded!")

        print(f"\nLoaded {len(subslice_data)} subslices with pre-computed cell masks")

        # Cache the data
        _CACHE['subslice_data'] = subslice_data
        _CACHE['max_expr'] = max_expr
        print("Data cached for fast threshold changes!\n")

    # Generate and display overlays
    n_subslices_total = len(subslice_data)

    # Determine which subslices to process
    if selection_mode == 'all':
        subslice_indices = list(range(n_subslices_total))
    elif selection_mode == 'first_n':
        subslice_indices = list(range(min(n_subslices_total, slice_selection)))
    else:  # specific
        all_slice_ids = [s['slice_id'] for s in subslice_data]
        subslice_indices = []
        for req_id in slice_selection:
            if req_id in all_slice_ids:
                subslice_indices.append(all_slice_ids.index(req_id))
            else:
                print(f"WARNING: Subslice ID {req_id} not found")

    n_subslices = len(subslice_indices)
    print(f"Generating overlays for {n_subslices} subslices (threshold={min_mscarlet_intensity:.2f})...")
    if n_subslices < n_subslices_total:
        print(f"  (Processing {n_subslices} of {n_subslices_total} total loaded subslices)")

    start_time = time.time()

    n_figures = (n_subslices + SLICES_PER_FIGURE - 1) // SLICES_PER_FIGURE
    print(f"Creating {n_figures} figures ({SLICES_PER_FIGURE} subslices per figure)")
    print(f"Auto-saving to: {output_dir}\n")

    # Debug counters
    total_cells_colored = 0
    total_cells_with_pixels = 0
    total_cells_empty = 0

    # Generate figures
    current_fig = None
    fig_num = 0

    for plot_idx, data_idx in enumerate(subslice_indices):
        # Create new figure every SLICES_PER_FIGURE subslices
        if plot_idx % SLICES_PER_FIGURE == 0:
            # Save previous figure if exists
            if current_fig is not None:
                fig_filename = f"figure_{fig_num}_of_{n_figures}.png"
                fig_path = output_dir / fig_filename
                current_fig.savefig(fig_path, dpi=150)
                print(f"  Saved {fig_filename}")
                plt.close(current_fig)

            # Create new figure
            fig_num = plot_idx // SLICES_PER_FIGURE + 1
            current_fig = plt.figure(figsize=(16, 10))
            current_fig.suptitle(
                f"mScarlet Cell Mask (Aniso, thresh={min_mscarlet_intensity:.2f}) - Fig {fig_num}/{n_figures}",
                fontsize=14
            )

        if (plot_idx + 1) % 5 == 0:
            print(f"  Processing subslice {plot_idx + 1}/{n_subslices}...")

        # Get subslice data
        data = subslice_data[data_idx]
        cellmask = data['cellmask']
        expressions = data['expressions']
        cell_pixel_lists = data['cell_pixel_lists']

        # Initialize overlay with cell mask background
        cell_exists_mask = cellmask > 0
        cellmask_gray = cell_exists_mask.astype(float) * CELLMASK_BRIGHTNESS * cellmask_intensity
        overlay_rgb = np.stack([cellmask_gray, cellmask_gray, cellmask_gray], axis=2)

        # Color cells above threshold
        slice_cells_colored = 0

        for i in range(len(expressions)):
            pixel_indices = cell_pixel_lists[i]
            if pixel_indices is None:
                total_cells_empty += 1
                continue

            total_cells_with_pixels += 1

            # Calculate intensity
            expr_normalized = expressions[i] / max_expr
            red_intensity = min(expr_normalized * RED_BOOST, 1.0)

            # Only color if above threshold
            if expr_normalized >= min_mscarlet_intensity:
                # Apply color using pre-computed indices
                overlay_rgb[pixel_indices[0], pixel_indices[1], 0] = red_intensity * RED_OPACITY
                overlay_rgb[pixel_indices[0], pixel_indices[1], 1] = 0
                overlay_rgb[pixel_indices[0], pixel_indices[1], 2] = 0

                slice_cells_colored += 1
                total_cells_colored += 1

        # Display in subplot
        subplot_idx = plot_idx % SLICES_PER_FIGURE + 1
        ax = current_fig.add_subplot(2, 3, subplot_idx)
        ax.imshow(np.clip(overlay_rgb, 0, 1))
        ax.set_title(f"Subslice {data['slice_id']} ({slice_cells_colored} cells)", fontsize=12)
        ax.axis('off')

    # Save the last figure
    if current_fig is not None:
        fig_filename = f"figure_{fig_num}_of_{n_figures}.png"
        fig_path = output_dir / fig_filename
        current_fig.savefig(fig_path, dpi=150)
        print(f"  Saved {fig_filename}")
        plt.close(current_fig)

    # Generate cell count histogram
    print("\nGenerating cell count histogram...")

    slice_ids_plot = []
    cells_displayed = []
    total_cells_per_slice = []

    for plot_idx, data_idx in enumerate(subslice_indices):
        data = subslice_data[data_idx]
        expressions = data['expressions']
        cells_above = np.sum((expressions / max_expr) >= min_mscarlet_intensity)
        total_cells_slice = len(expressions)

        slice_ids_plot.append(data['slice_id'])
        cells_displayed.append(cells_above)
        total_cells_per_slice.append(total_cells_slice)

    hist_path = create_histogram(
        slice_ids_plot, cells_displayed, total_cells_per_slice,
        min_mscarlet_intensity, output_dir
    )
    print(f"  Saved {Path(hist_path).name}")

    # Summary
    elapsed = time.time() - start_time
    print(f"\nDEBUG INFO:")
    print(f"  Cells with pixel data: {total_cells_with_pixels}")
    print(f"  Cells without pixels: {total_cells_empty}")
    print(f"  Cells colored (above threshold): {total_cells_colored}")

    print(f"\nCompleted in {elapsed:.1f} seconds ({elapsed/n_subslices:.2f} sec/subslice)")

    print("\n" + "=" * 40)
    print("DISPLAY COMPLETE")
    print("=" * 40)
    print(f"Created {n_figures} figures showing {n_subslices} subslices (threshold {min_mscarlet_intensity:.2f})")
    print(f"Each figure shows {SLICES_PER_FIGURE} subslices in a 2x3 grid")
    print(f"Generated cell count histogram: {Path(hist_path).name}")
    print(f"All outputs saved to: {output_dir}")

    print("\nTIPS:")
    print("Try different thresholds:")
    print("  python interactive_mscarlet_threshold_cellmask_subslice_anisotropic.py --threshold 0.3")
    print("  python interactive_mscarlet_threshold_cellmask_subslice_anisotropic.py --threshold 0.7")
    print("\nNext run with different threshold will be FAST (data cached!)")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive mScarlet threshold viewer (anisotropic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--threshold', '-th',
        type=float,
        default=0.0,
        help='Minimum normalized mScarlet expression (0-1), default: 0'
    )
    parser.add_argument(
        '--cellmask', '-cm',
        type=float,
        default=0.5,
        help='Cell mask brightness (0-1), default: 0.5'
    )
    parser.add_argument(
        '--reload', '-r',
        action='store_true',
        help='Force reload from disk'
    )
    parser.add_argument(
        '--slices', '-s',
        type=int,
        nargs='+',
        default=None,
        help='Specific slice IDs to process'
    )
    parser.add_argument(
        '--first', '-f',
        type=int,
        default=None,
        help='Process only first N subslices'
    )

    args = parser.parse_args()

    # Determine slice selection
    if args.slices is not None:
        slice_selection = args.slices
    elif args.first is not None:
        slice_selection = args.first
    else:
        slice_selection = None  # All

    interactive_mscarlet_threshold_cellmask_subslice_anisotropic(
        min_mscarlet_intensity=args.threshold,
        cellmask_intensity=args.cellmask,
        force_reload=args.reload,
        slice_selection=slice_selection,
    )


if __name__ == '__main__':
    main()
