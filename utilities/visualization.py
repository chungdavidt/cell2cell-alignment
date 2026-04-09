"""
Visualization utilities for preprocessing pipeline.

Provides:
- Subslice grid visualization
- Comparison figures (cellmask, overlay, mScarlet-only)
- Cell count histograms
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import List, Optional, Union


def visualize_subslice(
    slice_id: int,
    mscarlet_fovs: List[str],
    mscarlet_positions: np.ndarray,
    bridge_fovs: List[str],
    bridge_positions: np.ndarray,
    output_dir: Union[str, Path],
    edited: bool = False,
) -> str:
    """
    Create diagnostic visualization of subslice FOV grid.

    Args:
        slice_id: Slice number
        mscarlet_fovs: List of mScarlet+ FOV names
        mscarlet_positions: (N, 2) array of (row, col) positions
        bridge_fovs: List of bridge FOV names
        bridge_positions: (M, 2) array of bridge positions
        output_dir: Output directory for saved figure
        edited: If True, add "(EDITED)" to title

    Returns:
        Path to saved figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine all positions
    all_positions = mscarlet_positions
    if len(bridge_positions) > 0:
        all_positions = np.vstack([mscarlet_positions, bridge_positions])

    n_mscarlet = len(mscarlet_fovs)

    # Determine grid bounds
    min_row = int(np.min(all_positions[:, 0]))
    max_row = int(np.max(all_positions[:, 0]))
    min_col = int(np.min(all_positions[:, 1]))
    max_col = int(np.max(all_positions[:, 1]))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot mScarlet+ FOVs (green)
    for i in range(n_mscarlet):
        r, c = mscarlet_positions[i]
        rect = Rectangle(
            (c - 0.4, r - 0.4), 0.8, 0.8,
            facecolor='#33CC33',
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(c, r, f'M{i+1}', ha='center', va='center',
                fontsize=8, fontweight='bold')

    # Plot bridge FOVs (yellow)
    for i, fov in enumerate(bridge_fovs):
        r, c = bridge_positions[i]
        rect = Rectangle(
            (c - 0.4, r - 0.4), 0.8, 0.8,
            facecolor='#FFFF4D',
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(c, r, f'B{i+1}', ha='center', va='center',
                fontsize=8, fontweight='bold')

    # Draw grid lines
    for r in range(min_row, max_row + 2):
        ax.plot([min_col - 0.5, max_col + 0.5], [r - 0.5, r - 0.5],
                'k:', linewidth=0.5)
    for c in range(min_col, max_col + 2):
        ax.plot([c - 0.5, c - 0.5], [min_row - 0.5, max_row + 0.5],
                'k:', linewidth=0.5)

    # Labels and formatting
    ax.set_xlabel('Grid Column', fontsize=12)
    ax.set_ylabel('Grid Row', fontsize=12)

    title = f'Slice {slice_id} Subslice'
    if edited:
        title += ' (EDITED)'
    title += f': {n_mscarlet} mScarlet FOVs + {len(bridge_fovs)} Bridge FOVs'
    ax.set_title(title, fontsize=14)

    ax.set_xlim(min_col - 1, max_col + 1)
    ax.set_ylim(max_row + 1, min_row - 1)  # Reversed for row 0 at top
    ax.set_aspect('equal')
    ax.grid(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#33CC33', edgecolor='black', label='mScarlet+ FOV'),
        Rectangle((0, 0), 1, 1, facecolor='#FFFF4D', edgecolor='black', label='Bridge FOV'),
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()

    # Save figure
    suffix = '_EDITED' if edited else ''
    output_file = output_dir / f'slice{slice_id}_subslice_grid{suffix}.png'
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return str(output_file)


def create_comparison_figure(
    cellmask_gray: np.ndarray,
    overlay_rgb: np.ndarray,
    mscarlet_only: np.ndarray,
    slice_id: int,
    cells_mapped: int,
    output_dir: Union[str, Path],
    base_name: str,
) -> str:
    """
    Generate 3-panel comparison figure.

    Args:
        cellmask_gray: Grayscale cell mask image
        overlay_rgb: RGB overlay image
        mscarlet_only: RGB image with only mScarlet cells (no background)
        slice_id: Slice number
        cells_mapped: Number of cells successfully mapped
        output_dir: Output directory
        base_name: Base filename for output

    Returns:
        Path to saved figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Cell mask
    axes[0].imshow(cellmask_gray, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Cell Mask (Gray)')
    axes[0].axis('off')

    # Panel 2: Overlay
    axes[1].imshow(np.clip(overlay_rgb, 0, 1))
    if cells_mapped > 0:
        axes[1].set_title(f'mScarlet+ Overlay ({cells_mapped} cells)')
    else:
        axes[1].set_title('Cell Mask Only (No cells above threshold)')
    axes[1].axis('off')

    # Panel 3: mScarlet only
    axes[2].imshow(np.clip(mscarlet_only, 0, 1))
    axes[2].set_title('mScarlet Only')
    axes[2].axis('off')

    plt.tight_layout()

    output_file = output_dir / f'{base_name}_comparison.png'
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return str(output_file)


def create_histogram(
    slice_ids: List[int],
    cells_displayed: List[int],
    total_cells: List[int],
    threshold: float,
    output_dir: Union[str, Path],
) -> str:
    """
    Create cell count histogram per subslice.

    Args:
        slice_ids: List of slice IDs
        cells_displayed: Cells above threshold per slice
        total_cells: Total mScarlet+ cells per slice
        threshold: mScarlet threshold used
        output_dir: Output directory

    Returns:
        Path to saved figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(slice_ids))

    # Bar plot
    bars = ax.bar(x, cells_displayed, color='#CC3333', edgecolor='black')

    # Add total cell reference lines
    for i, (displayed, total) in enumerate(zip(cells_displayed, total_cells)):
        ax.plot([i - 0.3, i + 0.3], [total, total], 'k--', linewidth=1.5)

        # Value label on bar
        ax.text(i, displayed, str(displayed),
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Slice ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Cells Displayed', fontsize=14, fontweight='bold')
    ax.set_title(f'mScarlet+ Cell Counts per Subslice (Threshold: {threshold:.2f})',
                 fontsize=16, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in slice_ids])

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#CC3333', edgecolor='black',
                  label='Cells above threshold'),
        Line2D([0], [0], linestyle='--', color='black',
               label='Total mScarlet+ cells'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=12)

    # Percentage labels
    ymin, ymax = ax.get_ylim()
    y_label_pos = ymin - (ymax - ymin) * 0.08

    for i, (displayed, total) in enumerate(zip(cells_displayed, total_cells)):
        if total > 0:
            percentage = (displayed / total) * 100
            ax.text(i, y_label_pos, f'{percentage:.0f}%',
                    ha='center', va='top', fontsize=8, rotation=45)

    ax.set_ylim(0, max(max(cells_displayed), max(total_cells)) * 1.15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f'cell_count_histogram_threshold_{threshold:.2f}.png'
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return str(output_file)


def create_subslice_grid_figure(
    cellmask: np.ndarray,
    slice_id: int,
    n_cells: int,
    output_dir: Union[str, Path],
) -> str:
    """
    Create a simple figure showing the subslice cellmask.

    Args:
        cellmask: 2D array with cell IDs
        slice_id: Slice number
        n_cells: Number of unique cells
        output_dir: Output directory

    Returns:
        Path to saved figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Show cellmask with cell boundaries
    ax.imshow(cellmask > 0, cmap='gray')
    ax.set_title(f'Slice {slice_id} - {n_cells} cells')
    ax.axis('off')

    plt.tight_layout()

    output_file = output_dir / f'slice{slice_id}_cellmask_preview.png'
    fig.savefig(output_file, dpi=100)
    plt.close(fig)

    return str(output_file)
