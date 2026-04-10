#!/usr/bin/env python3
"""
Refine Subslice Definitions by mScarlet Threshold.

Filters FOVs in subslice definitions based on mScarlet intensity threshold.
Only keeps FOVs with sufficient bright (above-threshold) mScarlet+ cells.

This is a Python port of refine_subslices_by_threshold.m with exact fidelity.

Usage:
    python refine_subslices_by_threshold.py --threshold 0.3
    python refine_subslices_by_threshold.py --threshold 0.3 --min-cells 5
    python refine_subslices_by_threshold.py --threshold 0.3 --bridge-strategy adaptive

Input:
    - subslice_definitions.mat (from identify_mscarlet_subslices.py)
    - filt_neurons.mat (raw transcriptomics data)

Output:
    - subslice_definitions_thresh_X.XX.mat (refined definitions)
    - slice{N}_refinement_comparison_thresh_X.XX.png (comparison plots)

Algorithm:
    1. For each slice, count mScarlet+ cells per FOV that pass threshold
    2. Keep only FOVs with >= min_cells bright cells
    3. Handle bridge FOVs (adaptive/keep/remove)
    4. Ensure connectivity via 8-connected component analysis
    5. Save refined definitions
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt

from config import (
    FILT_NEURONS_PATH,
    SUBSLICE_DEFINITIONS_FILE,
    SUBSLICE_DEFINITIONS_DIR,
    QC_MIN_READS,
    QC_MIN_GENES,
    MSCARLET_COLUMN_INDEX,
)
from utilities.mat_io import load_filt_neurons, load_mat, save_mat


def build_adjacency_8connected(positions: np.ndarray) -> np.ndarray:
    """
    Build 8-connected adjacency matrix from grid positions.

    Args:
        positions: [N x 2] array of (row, col) grid positions

    Returns:
        [N x N] boolean adjacency matrix
    """
    n = len(positions)
    adj = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            row_diff = abs(positions[i, 0] - positions[j, 0])
            col_diff = abs(positions[i, 1] - positions[j, 1])

            # 8-connected: adjacent if row and col differ by at most 1
            # but not both zero (that would be same position)
            if row_diff <= 1 and col_diff <= 1 and (row_diff + col_diff) > 0:
                adj[i, j] = True
                adj[j, i] = True

    return adj


def find_connected_components(adj: np.ndarray) -> tuple:
    """
    Find connected components using BFS.

    Args:
        adj: [N x N] boolean adjacency matrix

    Returns:
        Tuple of (components, n_components)
        - components: [N] array of component IDs (1-indexed)
        - n_components: number of connected components
    """
    n = adj.shape[0]
    components = np.zeros(n, dtype=int)
    current_component = 0

    for start in range(n):
        if components[start] != 0:
            continue

        # Start new component
        current_component += 1
        queue = deque([start])
        components[start] = current_component

        while queue:
            node = queue.popleft()
            for neighbor in range(n):
                if adj[node, neighbor] and components[neighbor] == 0:
                    components[neighbor] = current_component
                    queue.append(neighbor)

    return components, current_component


def add_bridge_fovs(fov_list: list, positions: np.ndarray, all_fov_names: list) -> tuple:
    """
    Add minimal bridge FOVs to connect diagonal-only neighbors.

    Args:
        fov_list: List of FOV names
        positions: [N x 2] array of (row, col) positions
        all_fov_names: List of all possible FOV names (for name generation)

    Returns:
        Tuple of (bridge_fovs, bridge_positions)
    """
    n = len(fov_list)
    bridge_fovs = []
    bridge_positions = []

    # Extract prefix from existing FOV names
    if fov_list:
        # Parse FOV name format: MAX_Pos5_ROW_COL
        parts = fov_list[0].split('_')
        if len(parts) >= 2:
            prefix = '_'.join(parts[:-2])  # Everything except last two parts
        else:
            prefix = 'MAX_Pos5'
    else:
        prefix = 'MAX_Pos5'

    # Find diagonal-only pairs
    for i in range(n):
        for j in range(i + 1, n):
            row_diff = abs(positions[i, 0] - positions[j, 0])
            col_diff = abs(positions[i, 1] - positions[j, 1])

            # Diagonal-only: both row and col differ by 1
            if row_diff == 1 and col_diff == 1:
                # Choose bridge position: prefer min row, then min col
                if positions[i, 0] < positions[j, 0]:
                    bridge_row = int(positions[i, 0])
                    bridge_col = int(positions[j, 1])
                elif positions[i, 0] > positions[j, 0]:
                    bridge_row = int(positions[j, 0])
                    bridge_col = int(positions[i, 1])
                else:
                    # Same row (shouldn't happen for diagonal)
                    bridge_row = int(positions[i, 0])
                    bridge_col = min(int(positions[i, 1]), int(positions[j, 1]))

                # Check if bridge position already exists
                bridge_pos = np.array([bridge_row, bridge_col])
                already_exists = False

                for k in range(n):
                    if np.array_equal(positions[k], bridge_pos):
                        already_exists = True
                        break

                for bp in bridge_positions:
                    if np.array_equal(bp, bridge_pos):
                        already_exists = True
                        break

                if not already_exists:
                    # Generate bridge FOV name
                    bridge_name = f"{prefix}_{bridge_row:03d}_{bridge_col:03d}"
                    bridge_fovs.append(bridge_name)
                    bridge_positions.append(bridge_pos)

    return bridge_fovs, np.array(bridge_positions) if bridge_positions else np.empty((0, 2))


def refine_subslices_by_threshold(
    min_intensity: float,
    min_cells: int = 3,
    bridge_strategy: str = 'adaptive',
):
    """
    Refine subslice definitions by filtering FOVs based on mScarlet threshold.

    Args:
        min_intensity: Normalized mScarlet intensity threshold (0-1)
        min_cells: Minimum bright cells required per FOV (default: 3)
        bridge_strategy: How to handle bridge FOVs
            - 'adaptive': Recompute minimal bridges for remaining FOVs
            - 'keep': Keep all original bridges
            - 'remove': Remove all bridges
    """
    # Validate inputs
    if not 0 <= min_intensity <= 1:
        raise ValueError(f"min_intensity must be between 0 and 1, got {min_intensity}")
    if min_cells < 1:
        raise ValueError(f"min_cells must be >= 1, got {min_cells}")
    if bridge_strategy not in ['adaptive', 'keep', 'remove']:
        raise ValueError(f"bridge_strategy must be 'adaptive', 'keep', or 'remove'")

    print("=" * 50)
    print("REFINE SUBSLICES BY THRESHOLD")
    print("=" * 50)
    print(f"  Threshold: {min_intensity:.2f}")
    print(f"  Min cells per FOV: {min_cells}")
    print(f"  Bridge strategy: {bridge_strategy}")
    print()

    # Define output path
    output_file = Path(SUBSLICE_DEFINITIONS_DIR) / f"subslice_definitions_thresh_{min_intensity:.2f}.mat"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load original definitions
    print("Loading subslice definitions...")
    if not Path(SUBSLICE_DEFINITIONS_FILE).exists():
        raise FileNotFoundError(
            f"Subslice definitions not found!\n"
            f"Run identify_mscarlet_subslices.py first.\n"
            f"Expected: {SUBSLICE_DEFINITIONS_FILE}"
        )

    definitions_data = load_mat(SUBSLICE_DEFINITIONS_FILE)
    subslice_info_list = definitions_data['subslice_info']

    # Handle different formats
    if isinstance(subslice_info_list, np.ndarray):
        subslice_info_list = list(subslice_info_list.flatten())
    elif not isinstance(subslice_info_list, list):
        subslice_info_list = [subslice_info_list]

    print(f"  Loaded {len(subslice_info_list)} slices")

    # Load filt_neurons
    print("Loading filt_neurons...")
    filt_neurons = load_filt_neurons(FILT_NEURONS_PATH)

    # Extract data arrays
    fov_names = filt_neurons['fov']
    slice_ids = np.asarray(filt_neurons['slice']).flatten()
    expmat = np.asarray(filt_neurons['expmat'])

    # QC filter
    total_reads = np.sum(expmat, axis=1)
    genes_expressed = np.sum(expmat > 0, axis=1)
    qc_pass = (total_reads >= QC_MIN_READS) & (genes_expressed >= QC_MIN_GENES)

    # Get mScarlet expression and normalize
    mscarlet_expression = expmat[:, MSCARLET_COLUMN_INDEX]
    qc_mscarlet = mscarlet_expression[qc_pass]
    max_mscarlet = np.max(qc_mscarlet) if len(qc_mscarlet) > 0 else 1.0
    mscarlet_normalized = mscarlet_expression / max_mscarlet

    print(f"  QC-passing cells: {np.sum(qc_pass)}")
    print(f"  Max mScarlet (QC-passed): {max_mscarlet:.2f}")
    print()

    # Process each slice
    refined_list = []
    summary_stats = []

    for s_idx, subslice_info in enumerate(subslice_info_list):
        # Handle both dict and object access
        if isinstance(subslice_info, dict):
            slice_id = int(subslice_info['slice_id'])
            mscarlet_fovs_orig = subslice_info.get('mscarlet_fovs', [])
            bridge_fovs_orig = subslice_info.get('bridge_fovs', [])
            fov_grid_positions_orig = subslice_info.get('fov_grid_positions', np.array([]))
        else:
            slice_id = int(subslice_info.slice_id)
            mscarlet_fovs_orig = getattr(subslice_info, 'mscarlet_fovs', [])
            bridge_fovs_orig = getattr(subslice_info, 'bridge_fovs', [])
            fov_grid_positions_orig = getattr(subslice_info, 'fov_grid_positions', np.array([]))

        # Normalize FOV lists
        if isinstance(mscarlet_fovs_orig, np.ndarray):
            mscarlet_fovs_orig = [str(f) for f in mscarlet_fovs_orig.flatten()]
        elif isinstance(mscarlet_fovs_orig, str):
            mscarlet_fovs_orig = [mscarlet_fovs_orig]
        else:
            mscarlet_fovs_orig = list(mscarlet_fovs_orig) if mscarlet_fovs_orig else []

        if isinstance(bridge_fovs_orig, np.ndarray):
            bridge_fovs_orig = [str(f) for f in bridge_fovs_orig.flatten()]
        elif isinstance(bridge_fovs_orig, str):
            bridge_fovs_orig = [bridge_fovs_orig]
        else:
            bridge_fovs_orig = list(bridge_fovs_orig) if bridge_fovs_orig else []

        if isinstance(fov_grid_positions_orig, np.ndarray):
            fov_grid_positions_orig = np.asarray(fov_grid_positions_orig)
        else:
            fov_grid_positions_orig = np.array([])

        print(f"[Slice {slice_id}] Processing...")
        print(f"  Original mScarlet FOVs: {len(mscarlet_fovs_orig)}")
        print(f"  Original bridge FOVs: {len(bridge_fovs_orig)}")

        # Count bright cells per FOV
        bright_counts = {}
        for fov_name in mscarlet_fovs_orig:
            in_fov = np.array([f == fov_name for f in fov_names])
            in_slice = slice_ids == slice_id
            cell_mask = in_fov & in_slice & qc_pass

            cell_indices = np.where(cell_mask)[0]
            bright_count = np.sum(mscarlet_normalized[cell_indices] >= min_intensity)
            bright_counts[fov_name] = bright_count

        # Filter FOVs by bright cell count
        filtered_mscarlet_fovs = [
            fov for fov in mscarlet_fovs_orig
            if bright_counts.get(fov, 0) >= min_cells
        ]

        print(f"  Filtered mScarlet FOVs: {len(filtered_mscarlet_fovs)} (pass rate: {100*len(filtered_mscarlet_fovs)/max(1,len(mscarlet_fovs_orig)):.1f}%)")

        if len(filtered_mscarlet_fovs) == 0:
            print(f"  WARNING: No FOVs pass threshold - excluding slice")
            refined_info = {
                'slice_id': slice_id,
                'fov_list': [],
                'mscarlet_fovs': [],
                'bridge_fovs': [],
                'fov_grid_positions': np.array([]).reshape(0, 2),
                'num_mscarlet_cells': 0,
                'excluded': True,
                'exclusion_reason': 'No FOVs pass threshold',
            }
            refined_list.append(refined_info)
            summary_stats.append((slice_id, 0, len(mscarlet_fovs_orig), 0))
            print()
            continue

        # Get positions for filtered FOVs
        filtered_positions = []
        for fov in filtered_mscarlet_fovs:
            # Parse position from FOV name
            parts = fov.split('_')
            if len(parts) >= 2:
                try:
                    row = int(parts[-2])
                    col = int(parts[-1])
                    filtered_positions.append([row, col])
                except ValueError:
                    pass

        filtered_positions = np.array(filtered_positions) if filtered_positions else np.empty((0, 2))

        # Handle bridge FOVs based on strategy
        if bridge_strategy == 'keep':
            final_bridge_fovs = bridge_fovs_orig
            # Get bridge positions
            bridge_positions = []
            for fov in final_bridge_fovs:
                parts = fov.split('_')
                if len(parts) >= 2:
                    try:
                        row = int(parts[-2])
                        col = int(parts[-1])
                        bridge_positions.append([row, col])
                    except ValueError:
                        pass
            bridge_positions = np.array(bridge_positions) if bridge_positions else np.empty((0, 2))

        elif bridge_strategy == 'remove':
            final_bridge_fovs = []
            bridge_positions = np.empty((0, 2))

        else:  # adaptive
            final_bridge_fovs, bridge_positions = add_bridge_fovs(
                filtered_mscarlet_fovs,
                filtered_positions,
                list(fov_names)
            )

        print(f"  Bridge FOVs ({bridge_strategy}): {len(final_bridge_fovs)}")

        # Combine FOVs and positions
        all_fovs = filtered_mscarlet_fovs + list(final_bridge_fovs)
        if len(bridge_positions) > 0 and len(filtered_positions) > 0:
            all_positions = np.vstack([filtered_positions, bridge_positions])
        elif len(filtered_positions) > 0:
            all_positions = filtered_positions
        else:
            all_positions = np.empty((0, 2))

        # Check connectivity
        if len(all_positions) > 1:
            adj = build_adjacency_8connected(all_positions)
            components, n_components = find_connected_components(adj)

            if n_components > 1:
                print(f"  WARNING: {n_components} disconnected components found")

                # Keep only largest component
                component_sizes = [np.sum(components == c) for c in range(1, n_components + 1)]
                largest_component = np.argmax(component_sizes) + 1

                keep_mask = components == largest_component
                all_fovs = [f for i, f in enumerate(all_fovs) if keep_mask[i]]
                all_positions = all_positions[keep_mask]

                # Recategorize FOVs
                filtered_mscarlet_fovs = [f for f in all_fovs if f in filtered_mscarlet_fovs]
                final_bridge_fovs = [f for f in all_fovs if f not in filtered_mscarlet_fovs]

                print(f"  Kept largest component: {len(all_fovs)} FOVs")

        # Count total bright cells
        total_bright_cells = sum(
            bright_counts.get(fov, 0) for fov in filtered_mscarlet_fovs
        )

        # Create refined info
        refined_info = {
            'slice_id': slice_id,
            'fov_list': all_fovs,
            'mscarlet_fovs': filtered_mscarlet_fovs,
            'bridge_fovs': list(final_bridge_fovs),
            'fov_grid_positions': all_positions,
            'num_mscarlet_cells': total_bright_cells,
            'excluded': False,
            'exclusion_reason': '',
        }
        refined_list.append(refined_info)

        summary_stats.append((
            slice_id,
            len(filtered_mscarlet_fovs),
            len(mscarlet_fovs_orig),
            total_bright_cells
        ))

        print(f"  Final: {len(all_fovs)} FOVs, {total_bright_cells} bright cells")
        print()

    # Add metadata to refined definitions
    refined_data = {
        'subslice_info': refined_list,
        'threshold': min_intensity,
        'min_cells_per_fov': min_cells,
        'bridge_strategy': bridge_strategy,
        'refinement_date': str(datetime.now()),
    }

    # Save refined definitions
    print("Saving refined definitions...")
    save_mat(output_file, refined_data, format='5')
    print(f"  Saved: {output_file}")
    print()

    # Generate comparison plots
    print("Generating comparison plots...")
    plot_dir = Path(SUBSLICE_DEFINITIONS_DIR)

    for s_idx, (refined_info, orig_info) in enumerate(zip(refined_list, subslice_info_list)):
        if isinstance(refined_info, dict):
            slice_id = refined_info['slice_id']
            if refined_info.get('excluded', False):
                continue
        else:
            slice_id = refined_info.slice_id
            if getattr(refined_info, 'excluded', False):
                continue

        # Get original positions
        if isinstance(orig_info, dict):
            orig_positions = orig_info.get('fov_grid_positions', np.array([]))
            orig_mscarlet = orig_info.get('mscarlet_fovs', [])
            orig_bridge = orig_info.get('bridge_fovs', [])
        else:
            orig_positions = getattr(orig_info, 'fov_grid_positions', np.array([]))
            orig_mscarlet = getattr(orig_info, 'mscarlet_fovs', [])
            orig_bridge = getattr(orig_info, 'bridge_fovs', [])

        if isinstance(orig_positions, np.ndarray) and orig_positions.size > 0:
            orig_positions = np.asarray(orig_positions)
        else:
            continue

        # Get refined positions
        if isinstance(refined_info, dict):
            ref_positions = refined_info.get('fov_grid_positions', np.array([]))
            ref_mscarlet = refined_info.get('mscarlet_fovs', [])
            ref_bridge = refined_info.get('bridge_fovs', [])
        else:
            ref_positions = getattr(refined_info, 'fov_grid_positions', np.array([]))
            ref_mscarlet = getattr(refined_info, 'mscarlet_fovs', [])
            ref_bridge = getattr(refined_info, 'bridge_fovs', [])

        if isinstance(ref_positions, np.ndarray) and ref_positions.size > 0:
            ref_positions = np.asarray(ref_positions)
        else:
            ref_positions = np.empty((0, 2))

        # Create comparison figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original
        ax = axes[0]
        ax.set_title(f'Original (Slice {slice_id})')
        if orig_positions.size > 0:
            # Plot all positions
            ax.scatter(orig_positions[:, 1], orig_positions[:, 0],
                      c='lightgray', s=100, label='All FOVs')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.legend()

        # Refined
        ax = axes[1]
        ax.set_title(f'Refined (threshold={min_intensity:.2f})')
        if ref_positions.size > 0:
            # Color by type
            colors = []
            for i, fov in enumerate(refined_info['fov_list'] if isinstance(refined_info, dict) else refined_info.fov_list):
                if fov in (ref_mscarlet if isinstance(ref_mscarlet, list) else list(ref_mscarlet)):
                    colors.append('red')
                else:
                    colors.append('green')

            if len(colors) == len(ref_positions):
                ax.scatter(ref_positions[:, 1], ref_positions[:, 0],
                          c=colors, s=100)
            else:
                ax.scatter(ref_positions[:, 1], ref_positions[:, 0],
                          c='blue', s=100)

        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.invert_yaxis()
        ax.set_aspect('equal')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='mScarlet+ FOVs'),
            Patch(facecolor='green', label='Bridge FOVs'),
        ]
        ax.legend(handles=legend_elements)

        plt.tight_layout()

        # Save plot
        plot_file = plot_dir / f"slice{slice_id}_refinement_comparison_thresh_{min_intensity:.2f}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close(fig)

    print(f"  Saved comparison plots to: {plot_dir}")
    print()

    # Print summary
    print("=" * 50)
    print("REFINEMENT SUMMARY")
    print("=" * 50)
    print(f"{'Slice':<8} {'Kept':<8} {'Original':<10} {'Retention':<12} {'Cells':<8}")
    print("-" * 50)

    total_kept = 0
    total_orig = 0
    for slice_id, kept, orig, cells in summary_stats:
        retention = 100 * kept / orig if orig > 0 else 0
        print(f"{slice_id:<8} {kept:<8} {orig:<10} {retention:>6.1f}%      {cells:<8}")
        total_kept += kept
        total_orig += orig

    print("-" * 50)
    overall_retention = 100 * total_kept / total_orig if total_orig > 0 else 0
    print(f"{'TOTAL':<8} {total_kept:<8} {total_orig:<10} {overall_retention:>6.1f}%")
    print()

    print("Output files:")
    print(f"  Definitions: {output_file}")
    print(f"  Plots: {plot_dir}/slice*_refinement_comparison_thresh_{min_intensity:.2f}.png")
    print()
    print("Next steps:")
    print(f"  1. Run stitch_subslices.py with refined definitions")
    print(f"  2. Or manually review plots to verify refinement quality")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Refine subslice definitions by mScarlet threshold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        required=True,
        help='mScarlet intensity threshold (0-1, normalized)'
    )
    parser.add_argument(
        '--min-cells', '-m',
        type=int,
        default=3,
        help='Minimum bright cells per FOV to keep (default: 3)'
    )
    parser.add_argument(
        '--bridge-strategy', '-b',
        choices=['adaptive', 'keep', 'remove'],
        default='adaptive',
        help='Bridge FOV strategy: adaptive (recompute), keep (original), remove (none)'
    )

    args = parser.parse_args()

    refine_subslices_by_threshold(
        min_intensity=args.threshold,
        min_cells=args.min_cells,
        bridge_strategy=args.bridge_strategy,
    )


if __name__ == '__main__':
    main()
