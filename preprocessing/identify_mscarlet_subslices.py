#!/usr/bin/env python3
"""
Identify mScarlet+ Subslices for LineStuffUp Alignment.

Finds contiguous regions of FOVs with mScarlet-expressing cells per slice.
Creates subslice definitions (FOV lists) for downstream stitching and overlay.

This is a Python port of identify_mscarlet_subslices.m with exact fidelity.

Usage:
    python identify_mscarlet_subslices.py
    python identify_mscarlet_subslices.py --test  # Test mode (slice 22 only)
    python identify_mscarlet_subslices.py --slice 22  # Specific slice

Output: subslice_definitions.mat containing:
    - subslice_info: struct array with fields:
        .slice_id: Slice number
        .fov_list: Cell array of FOV names in subslice
        .fov_grid_positions: [N x 2] array of (row, col) grid positions
        .mscarlet_fovs: FOVs with mScarlet+ cells (original, before bridges)
        .bridge_fovs: FOVs added to ensure edge-connectivity
        .num_mscarlet_cells: Total mScarlet+ QC-passing cells in subslice

Algorithm:
    1. Apply QC filter: reads >= 20, genes >= 5
    2. Find FOVs with mScarlet+ cells (column 113 in Python, was 114 in MATLAB)
    3. Build 8-connectivity graph (edge or corner neighbors)
    4. Find largest connected component per slice
    5. Add bridge FOVs for diagonal-only connections
    6. Generate diagnostic visualizations
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import sparse

from config import (
    FILT_NEURONS_PATH,
    SUBSLICE_DEFINITIONS_DIR,
    SUBSLICE_DEFINITIONS_FILE,
    MSCARLET_COLUMN_INDEX,
    QC_MIN_READS,
    QC_MIN_GENES,
)
from utils.mat_io import load_filt_neurons, save_mat, sparse_to_dense, get_expression_column
from utils.graph_utils import (
    parse_fov_grid_positions,
    build_adjacency_8connected,
    find_connected_components,
    add_bridge_fovs,
    get_largest_component,
)
from utils.visualization import visualize_subslice


def identify_mscarlet_subslices(test_mode: bool = False, target_slice: int = None):
    """
    Main function to identify mScarlet+ subslices.

    Args:
        test_mode: If True, process only slice 22 (known good slice)
        target_slice: If specified, process only this slice
    """
    TEST_SLICE = 22  # Known slice with good mScarlet expression

    print("=" * 40)
    print("IDENTIFY mSCARLET SUBSLICES")
    print("=" * 40)
    if test_mode:
        print(f"Mode: TEST (slice {TEST_SLICE} only)")
    elif target_slice is not None:
        print(f"Mode: SINGLE SLICE ({target_slice})")
    else:
        print("Mode: FULL (all slices)")
    print()

    # Create output directory
    output_dir = Path(SUBSLICE_DEFINITIONS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load filt_neurons
    print("Loading filt_neurons...")
    filt_neurons = load_filt_neurons(FILT_NEURONS_PATH)

    expmat = filt_neurons['expmat']
    n_cells = expmat.shape[0]
    print(f"  Total cells in dataset: {n_cells}")

    # FOV names should already be normalized by load_filt_neurons
    fov_names = filt_neurons['fov']
    print("  FOV names normalized")

    # Apply QC filter
    print("\nApplying QC filter...")
    print(f"  QC criteria: reads >= {QC_MIN_READS} AND genes >= {QC_MIN_GENES}")

    # Calculate total reads and genes per cell
    if sparse.issparse(expmat):
        total_reads = np.asarray(expmat.sum(axis=1)).flatten()
        total_genes = np.asarray((expmat > 0).sum(axis=1)).flatten()
    else:
        total_reads = np.sum(expmat, axis=1)
        total_genes = np.sum(expmat > 0, axis=1)

    pass_qc = (total_reads >= QC_MIN_READS) & (total_genes >= QC_MIN_GENES)
    print(f"  Cells passing QC: {np.sum(pass_qc)} / {n_cells} ({100*np.sum(pass_qc)/n_cells:.1f}%)")

    # Find mScarlet+ cells
    print("\nFinding mScarlet+ cells...")
    # IMPORTANT: MATLAB column 114 -> Python index 113 (0-indexed)
    mscarlet_expression = get_expression_column(expmat, MSCARLET_COLUMN_INDEX)
    mscarlet_positive = mscarlet_expression > 0

    print(f"  mScarlet+ cells (any expression): {np.sum(mscarlet_positive)} / {n_cells} "
          f"({100*np.sum(mscarlet_positive)/n_cells:.1f}%)")

    # Combined filter
    mscarlet_qc_pass = pass_qc & mscarlet_positive
    print(f"  mScarlet+ cells passing QC: {np.sum(mscarlet_qc_pass)} "
          f"({100*np.sum(mscarlet_qc_pass)/np.sum(pass_qc):.1f}%)")
    print()

    # Get unique slices
    slice_ids = np.asarray(filt_neurons['slice']).flatten()
    unique_slices = np.unique(slice_ids[~np.isnan(slice_ids)])
    unique_slices = unique_slices.astype(int)

    if test_mode:
        if TEST_SLICE not in unique_slices:
            raise ValueError(f"Test slice {TEST_SLICE} not found in dataset")
        unique_slices = np.array([TEST_SLICE])
        print(f"Processing test slice: {TEST_SLICE}\n")
    elif target_slice is not None:
        if target_slice not in unique_slices:
            raise ValueError(f"Slice {target_slice} not found in dataset")
        unique_slices = np.array([target_slice])
        print(f"Processing slice: {target_slice}\n")
    else:
        print(f"Found {len(unique_slices)} unique slices\n")

    # Process each slice
    subslice_info_list = []

    for s_idx, slice_id in enumerate(unique_slices):
        print("=" * 40)
        print(f"[{s_idx+1}/{len(unique_slices)}] Processing slice {slice_id}")
        print("=" * 40)

        # Get cells in this slice
        in_slice = slice_ids == slice_id
        slice_mscarlet_qc = in_slice & mscarlet_qc_pass

        print(f"  Cells in slice: {np.sum(in_slice)}")
        print(f"  mScarlet+ QC-passing cells: {np.sum(slice_mscarlet_qc)}")

        if np.sum(slice_mscarlet_qc) == 0:
            print("  WARNING: No mScarlet+ cells in slice, skipping\n")
            continue

        # Get unique FOVs with mScarlet+ cells
        slice_fov_names = np.array(fov_names)[slice_mscarlet_qc]
        slice_fovs = list(np.unique(slice_fov_names))
        print(f"  FOVs with mScarlet+ cells: {len(slice_fovs)}")

        # Parse FOV names to get grid positions
        fov_positions, valid_mask = parse_fov_grid_positions(slice_fovs)

        if np.sum(valid_mask) == 0:
            print("  WARNING: No valid FOV names parsed, skipping\n")
            continue

        if np.sum(~valid_mask) > 0:
            print(f"  WARNING: {np.sum(~valid_mask)} FOVs could not be parsed")

        # Keep only valid FOVs
        valid_indices = np.where(valid_mask)[0]
        slice_fovs = [slice_fovs[i] for i in valid_indices]
        fov_positions = fov_positions[valid_mask]

        print(f"  Valid FOVs for clustering: {len(slice_fovs)}")

        # Build adjacency matrix (8-connectivity)
        adj_matrix = build_adjacency_8connected(fov_positions)

        # Find connected components
        components, num_components = find_connected_components(adj_matrix)
        print(f"  Connected components found: {num_components}")

        if num_components == 0:
            print("  WARNING: No connected components, skipping\n")
            continue

        # Find largest component
        component_sizes = [np.sum(components == c) for c in range(1, num_components + 1)]
        print(f"  Component sizes: {component_sizes}")

        largest_mask, largest_idx = get_largest_component(components, num_components)
        print(f"  Largest component: #{largest_idx} with {np.sum(largest_mask)} FOVs")

        # Get FOVs in largest component
        mscarlet_fovs = [slice_fovs[i] for i in range(len(slice_fovs)) if largest_mask[i]]
        mscarlet_positions = fov_positions[largest_mask]

        # Add bridge FOVs for diagonal connections
        print("  Adding bridge FOVs for diagonal connections...")
        bridge_fovs, bridge_positions = add_bridge_fovs(mscarlet_fovs, mscarlet_positions)
        print(f"  Bridge FOVs added: {len(bridge_fovs)}")

        # Combine mScarlet and bridge FOVs
        final_fov_list = mscarlet_fovs + bridge_fovs
        if len(bridge_positions) > 0:
            final_positions = np.vstack([mscarlet_positions, bridge_positions])
        else:
            final_positions = mscarlet_positions

        # Count mScarlet+ cells in final subslice
        fov_names_array = np.array(fov_names)
        in_subslice_fovs = np.isin(fov_names_array, final_fov_list)
        subslice_mscarlet_cells = in_slice & in_subslice_fovs & mscarlet_qc_pass
        num_mscarlet_cells = int(np.sum(subslice_mscarlet_cells))

        print(f"  Total FOVs in subslice: {len(final_fov_list)} "
              f"({len(mscarlet_fovs)} mScarlet + {len(bridge_fovs)} bridge)")
        print(f"  Total mScarlet+ cells in subslice: {num_mscarlet_cells}")

        # Save subslice info
        subslice_info = {
            'slice_id': int(slice_id),
            'fov_list': final_fov_list,
            'fov_grid_positions': final_positions,
            'mscarlet_fovs': mscarlet_fovs,
            'bridge_fovs': bridge_fovs,
            'num_mscarlet_cells': num_mscarlet_cells,
        }
        subslice_info_list.append(subslice_info)

        # Generate diagnostic visualization
        viz_path = visualize_subslice(
            slice_id, mscarlet_fovs, mscarlet_positions,
            bridge_fovs, bridge_positions, output_dir
        )
        print(f"  Saved diagnostic plot: {viz_path}")
        print()

    # Save results
    print("=" * 40)
    print("SAVING RESULTS")
    print("=" * 40)
    print(f"Total slices with subslices: {len(subslice_info_list)}")
    print(f"Output file: {SUBSLICE_DEFINITIONS_FILE}")

    # Convert to format compatible with MATLAB
    save_mat(SUBSLICE_DEFINITIONS_FILE, {'subslice_info': subslice_info_list}, format='5')
    print("Saved subslice definitions\n")

    # Summary
    print("=" * 40)
    print("SUMMARY")
    print("=" * 40)
    for info in subslice_info_list:
        print(f"Slice {info['slice_id']}: {len(info['fov_list'])} FOVs "
              f"({len(info['mscarlet_fovs'])} mScarlet + {len(info['bridge_fovs'])} bridge), "
              f"{info['num_mscarlet_cells']} cells")

    print("\nNext steps:")
    print(f"  1. Review diagnostic plots in: {output_dir}")
    print("  2. (Optional) Edit subslice definitions using edit_subslice_definitions.py")
    print("  3. Run stitch_subslices.py to create stitched subslice images")
    print()

    return subslice_info_list


def main():
    parser = argparse.ArgumentParser(
        description="Identify mScarlet+ subslices for alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test mode: process only slice 22'
    )
    parser.add_argument(
        '--slice', '-s',
        type=int,
        default=None,
        help='Process specific slice only'
    )

    args = parser.parse_args()

    identify_mscarlet_subslices(
        test_mode=args.test,
        target_slice=args.slice
    )


if __name__ == '__main__':
    main()
