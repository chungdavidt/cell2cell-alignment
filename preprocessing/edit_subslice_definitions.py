#!/usr/bin/env python3
"""
Edit Subslice Definitions - Interactive tool to manually edit FOV assignments.

This is a Python port of edit_subslice_definitions.m with exact fidelity.

Usage:
    python edit_subslice_definitions.py                    # Interactive mode
    python edit_subslice_definitions.py --slice 22         # Edit slice 22
    python edit_subslice_definitions.py --threshold 0.3 --slice 22  # With overlay viewing

Features:
    - Add/remove FOVs from slice definitions
    - View FOV details and cell counts
    - View mScarlet overlay images at specified threshold
    - Automatic metadata recalculation
    - Return to menu by pressing Enter at any prompt

The threshold parameter ONLY affects which overlay images you can view while
editing. It does NOT change which definitions you're editing.

Input:  subslice_definitions.mat (from identify_mscarlet_subslices.py)
Output: Updated subslice_definitions.mat + regenerated diagnostic plots
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import sparse

from config import (
    FILT_NEURONS_PATH,
    SUBSLICE_DEFINITIONS_DIR,
    SUBSLICE_DEFINITIONS_FILE,
    MSCARLET_LABELLED_DIR,
    MSCARLET_COLUMN_INDEX,
    QC_MIN_READS,
    QC_MIN_GENES,
    get_threshold_folder,
)
from utils.mat_io import load_filt_neurons, load_mat, save_mat, get_expression_column
from utils.graph_utils import parse_fov_grid_positions
from utils.visualization import visualize_subslice


def display_subslice(subslice: dict):
    """Display current subslice information."""
    print("Current subslice:")
    print(f"  Slice ID: {subslice['slice_id']}")
    print(f"  Total FOVs: {len(subslice['fov_list'])}")
    print(f"  mScarlet+ FOVs: {len(subslice['mscarlet_fovs'])}")
    print(f"  Bridge FOVs: {len(subslice['bridge_fovs'])}")
    print(f"  mScarlet+ cells: {subslice['num_mscarlet_cells']}\n")

    print("FOV List:")
    for i, fov_name in enumerate(subslice['fov_list']):
        is_mscarlet = fov_name in subslice['mscarlet_fovs']
        is_bridge = fov_name in subslice['bridge_fovs']

        if is_mscarlet:
            type_str = '[mScarlet]'
        elif is_bridge:
            type_str = '[Bridge]'
        else:
            type_str = '[Unknown]'

        print(f"  {i+1:2d}. {fov_name:30s} {type_str}")


def count_mscarlet_cells(fov_list: list, slice_id: int, filt_neurons: dict,
                         mscarlet_qc_pass: np.ndarray) -> int:
    """Count total mScarlet+ QC-passing cells in FOV list."""
    fov_names = filt_neurons['fov']
    slice_ids = np.asarray(filt_neurons['slice']).flatten()

    in_slice = slice_ids == slice_id
    in_subslice = np.isin(fov_names, fov_list)
    return int(np.sum(in_slice & in_subslice & mscarlet_qc_pass))


def remove_fov(subslice: dict, fov_name: str, filt_neurons: dict,
               mscarlet_qc_pass: np.ndarray) -> tuple:
    """Remove FOV from subslice."""
    if fov_name not in subslice['fov_list']:
        print(f"  WARNING: FOV '{fov_name}' not found in subslice")
        return subslice, False

    # Find index
    idx = subslice['fov_list'].index(fov_name)

    # Remove from main list
    subslice['fov_list'] = [f for f in subslice['fov_list'] if f != fov_name]

    # Remove from grid positions
    if 'fov_grid_positions' in subslice and len(subslice['fov_grid_positions']) > idx:
        subslice['fov_grid_positions'] = np.delete(
            subslice['fov_grid_positions'], idx, axis=0
        )

    # Remove from mScarlet or bridge list
    if fov_name in subslice['mscarlet_fovs']:
        subslice['mscarlet_fovs'] = [f for f in subslice['mscarlet_fovs'] if f != fov_name]
        print(f"  Removed mScarlet+ FOV: {fov_name}")
    elif fov_name in subslice['bridge_fovs']:
        subslice['bridge_fovs'] = [f for f in subslice['bridge_fovs'] if f != fov_name]
        print(f"  Removed bridge FOV: {fov_name}")

    # Recalculate mScarlet+ cell count
    subslice['num_mscarlet_cells'] = count_mscarlet_cells(
        subslice['fov_list'], subslice['slice_id'], filt_neurons, mscarlet_qc_pass
    )
    print(f"  Updated mScarlet+ cell count: {subslice['num_mscarlet_cells']}")

    return subslice, True


def add_fov(subslice: dict, fov_name: str, filt_neurons: dict,
            mscarlet_qc_pass: np.ndarray) -> tuple:
    """Add FOV to subslice."""
    if fov_name in subslice['fov_list']:
        print(f"  WARNING: FOV '{fov_name}' already in subslice")
        return subslice, False

    # Parse FOV name to get grid position
    positions, valid_mask = parse_fov_grid_positions([fov_name])
    if not valid_mask[0]:
        print("  WARNING: Invalid FOV name format. Expected: MAX_Pos#_XXX_YYY")
        return subslice, False

    row, col = positions[0]

    # Check if FOV has mScarlet+ cells
    fov_names = filt_neurons['fov']
    slice_ids = np.asarray(filt_neurons['slice']).flatten()

    in_slice = slice_ids == subslice['slice_id']
    in_fov = np.array([f == fov_name for f in fov_names])
    fov_mscarlet_cells = np.sum(in_slice & in_fov & mscarlet_qc_pass)

    # Add to appropriate list
    if fov_mscarlet_cells > 0:
        subslice['mscarlet_fovs'].append(fov_name)
        print(f"  Added as mScarlet+ FOV ({fov_mscarlet_cells} cells)")
    else:
        subslice['bridge_fovs'].append(fov_name)
        print(f"  Added as bridge FOV (0 mScarlet+ cells)")

    # Add to main list and positions
    subslice['fov_list'].append(fov_name)
    if 'fov_grid_positions' in subslice:
        subslice['fov_grid_positions'] = np.vstack([
            subslice['fov_grid_positions'],
            [row, col]
        ])
    else:
        subslice['fov_grid_positions'] = np.array([[row, col]])

    # Recalculate mScarlet+ cell count
    subslice['num_mscarlet_cells'] = count_mscarlet_cells(
        subslice['fov_list'], subslice['slice_id'], filt_neurons, mscarlet_qc_pass
    )
    print(f"  Updated mScarlet+ cell count: {subslice['num_mscarlet_cells']}")

    return subslice, True


def view_fov_details(fov_name: str, filt_neurons: dict, mscarlet_qc_pass: np.ndarray):
    """Display details about a specific FOV."""
    print(f"\n--- FOV Details: {fov_name} ---")

    # Parse grid position
    positions, valid_mask = parse_fov_grid_positions([fov_name])
    if valid_mask[0]:
        row, col = positions[0]
        print(f"  Grid position: (row={int(row)}, col={int(col)})")

    # Count cells
    fov_names = filt_neurons['fov']
    in_fov = np.array([f == fov_name for f in fov_names])
    total_cells = int(np.sum(in_fov))
    mscarlet_cells = int(np.sum(in_fov & mscarlet_qc_pass))

    print(f"  Total cells in FOV: {total_cells}")
    print(f"  mScarlet+ QC-passing cells: {mscarlet_cells}")

    if total_cells > 0:
        print(f"  mScarlet+ percentage: {100*mscarlet_cells/total_cells:.1f}%")

    # Slice distribution
    if total_cells > 0:
        slice_ids = np.asarray(filt_neurons['slice']).flatten()
        fov_slices = np.unique(slice_ids[in_fov])
        fov_slices = fov_slices[~np.isnan(fov_slices)].astype(int)
        print(f"  Present in slices: {list(fov_slices)}")

    print()


def view_overlay_image(slice_id: int, overlay_dir: Path):
    """Display mScarlet overlay image for the specified slice."""
    import matplotlib
    matplotlib.use('TkAgg')  # Try interactive backend
    import matplotlib.pyplot as plt

    overlay_file = overlay_dir / f"slice{slice_id}_subslice_mScarlet_overlay_DAPI.tif"

    if not overlay_file.exists():
        print(f"\n  WARNING: Overlay image not found: {overlay_file}")
        print("  You may need to run generate_mscarlet_overlay_subslice.py first.\n")
        return

    print(f"\n  Opening overlay image: {overlay_file}")

    try:
        from utils.image_io import imread_tiff
        overlay_img = imread_tiff(overlay_file)

        plt.figure(figsize=(12, 9))
        plt.imshow(overlay_img)
        plt.title(f"Slice {slice_id} - mScarlet Overlay")
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        print("  Image displayed. Close the figure window when done.\n")
    except Exception as e:
        print(f"  ERROR: Could not display image: {e}\n")


def view_labelled_fovs(slice_id: int, threshold: float):
    """Display labelled mScarlet overlay for the specified slice."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    threshold_folder = get_threshold_folder(threshold)
    labelled_dir = Path(MSCARLET_LABELLED_DIR) / threshold_folder
    labelled_file = labelled_dir / f"slice_{slice_id:02d}_labelled.png"

    if not labelled_file.exists():
        print(f"\n  WARNING: Labelled overlay not found: {labelled_file}")
        print(f"  Run: generate_mscarlet_overlay_labelled.py --threshold {threshold} --slice {slice_id}\n")
        return

    print(f"\n  Opening labelled overlay: {labelled_file}")

    try:
        from PIL import Image
        labelled_img = np.array(Image.open(labelled_file))

        plt.figure(figsize=(12, 9))
        plt.imshow(labelled_img)
        plt.title(f"Slice {slice_id} - Labelled mScarlet Overlay (threshold {threshold:.2f})")
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        print("  Image displayed. Close the figure window when done.\n")
    except Exception as e:
        print(f"  ERROR: Could not display image: {e}\n")


def edit_subslice_definitions(threshold: float = 0.0, target_slice: int = None):
    """
    Main interactive editing function.

    Args:
        threshold: Threshold for viewing overlays (affects display only)
        target_slice: If specified, edit only this slice
    """
    definitions_file = Path(SUBSLICE_DEFINITIONS_FILE)
    definitions_dir = Path(SUBSLICE_DEFINITIONS_DIR)

    threshold_folder = get_threshold_folder(threshold)
    overlay_dir = Path(MSCARLET_LABELLED_DIR).parent / "mScarlet_overlay_dapi_subslice" / threshold_folder

    if not definitions_file.exists():
        raise FileNotFoundError(
            f"Subslice definitions not found!\n"
            f"Run identify_mscarlet_subslices.py first.\n"
            f"Expected: {definitions_file}"
        )

    print("=" * 40)
    print("EDIT SUBSLICE DEFINITIONS")
    print("=" * 40)
    print(f"Threshold: {threshold:.2f}")
    print(f"Overlay directory: {overlay_dir}\n")

    # Load data
    print("Loading subslice definitions...")
    definitions_data = load_mat(definitions_file)
    subslice_info_list = definitions_data['subslice_info']

    # Convert to list of dicts if needed
    if isinstance(subslice_info_list, np.ndarray):
        subslice_info_list = [
            {k: v for k, v in zip(s.dtype.names, s)} if hasattr(s, 'dtype') else s
            for s in subslice_info_list.flatten()
        ]

    print(f"  Loaded {len(subslice_info_list)} subslices")

    print("Loading filt_neurons...")
    filt_neurons = load_filt_neurons(FILT_NEURONS_PATH)
    expmat = filt_neurons['expmat']

    # QC filter and mScarlet expression
    if sparse.issparse(expmat):
        total_reads = np.asarray(expmat.sum(axis=1)).flatten()
        total_genes = np.asarray((expmat > 0).sum(axis=1)).flatten()
    else:
        total_reads = np.sum(expmat, axis=1)
        total_genes = np.sum(expmat > 0, axis=1)

    pass_qc = (total_reads >= QC_MIN_READS) & (total_genes >= QC_MIN_GENES)
    mscarlet_expression = get_expression_column(expmat, MSCARLET_COLUMN_INDEX)
    mscarlet_positive = mscarlet_expression > 0
    mscarlet_qc_pass = pass_qc & mscarlet_positive

    print("  Data loaded\n")

    # Outer loop for slice selection
    while True:
        # Select slice to edit
        if target_slice is None:
            # Show menu
            print("Available slices:")
            for i, info in enumerate(subslice_info_list):
                slice_id = info['slice_id'] if isinstance(info, dict) else info.slice_id
                fov_list = info['fov_list'] if isinstance(info, dict) else info.fov_list
                num_cells = info['num_mscarlet_cells'] if isinstance(info, dict) else info.num_mscarlet_cells
                print(f"  {i+1}. Slice {slice_id} ({len(fov_list)} FOVs, {num_cells} mScarlet+ cells)")
            print()

            try:
                slice_choice = input("Enter slice number to edit (or 0 to cancel): ")
                if slice_choice.strip() == '' or int(slice_choice) == 0:
                    print("Cancelled.")
                    return
                slice_choice = int(slice_choice)
            except ValueError:
                print("Invalid input.")
                continue

            # Find subslice with this slice_id
            subslice_idx = None
            for i, info in enumerate(subslice_info_list):
                s_id = info['slice_id'] if isinstance(info, dict) else info.slice_id
                if s_id == slice_choice:
                    subslice_idx = i
                    break

            if subslice_idx is None:
                print(f"Slice {slice_choice} not found in subslice definitions")
                continue
        else:
            # Direct slice specification
            subslice_idx = None
            for i, info in enumerate(subslice_info_list):
                s_id = info['slice_id'] if isinstance(info, dict) else info.slice_id
                if s_id == target_slice:
                    subslice_idx = i
                    break

            if subslice_idx is None:
                raise ValueError(f"Slice {target_slice} not found in subslice definitions")

        subslice = subslice_info_list[subslice_idx]
        if not isinstance(subslice, dict):
            subslice = {k: v for k, v in zip(subslice.dtype.names, subslice)}

        slice_id = subslice['slice_id']
        print("=" * 40)
        print(f"EDITING SLICE {slice_id}")
        print("=" * 40 + "\n")

        # Interactive editing loop
        modified = False

        while True:
            display_subslice(subslice)

            print("\nOptions:")
            print("  1. Remove FOV")
            print("  2. Add FOV")
            print("  3. View FOV details")
            print("  4. View mScarlet overlay image")
            print("  5. View labelled mScarlet overlay (FOV labels + cells)")
            if target_slice is None:
                print("  6. Save and return to menu")
                print("  7. Back to slice selection (discard changes)\n")
            else:
                print("  6. Save and exit")
                print("  7. Exit without saving\n")

            choice = input("Enter choice: ").strip()

            if choice == '1':
                # Remove FOV
                fov_input = input("Enter FOV number or name to remove (Enter to cancel): ").strip()
                if not fov_input:
                    continue

                try:
                    fov_idx = int(fov_input)
                    if 1 <= fov_idx <= len(subslice['fov_list']):
                        fov_name = subslice['fov_list'][fov_idx - 1]
                    else:
                        fov_name = fov_input
                except ValueError:
                    fov_name = fov_input

                subslice, success = remove_fov(subslice, fov_name, filt_neurons, mscarlet_qc_pass)
                if success:
                    modified = True

            elif choice == '2':
                # Add FOV
                fov_name = input("Enter FOV name to add (Enter to cancel): ").strip()
                if not fov_name:
                    continue

                subslice, success = add_fov(subslice, fov_name, filt_neurons, mscarlet_qc_pass)
                if success:
                    modified = True

            elif choice == '3':
                # View FOV details
                fov_input = input("Enter FOV number or name to view (Enter to cancel): ").strip()
                if not fov_input:
                    continue

                try:
                    fov_idx = int(fov_input)
                    if 1 <= fov_idx <= len(subslice['fov_list']):
                        fov_name = subslice['fov_list'][fov_idx - 1]
                    else:
                        fov_name = fov_input
                except ValueError:
                    fov_name = fov_input

                view_fov_details(fov_name, filt_neurons, mscarlet_qc_pass)

            elif choice == '4':
                # View mScarlet overlay image
                view_overlay_image(slice_id, overlay_dir)

            elif choice == '5':
                # View labelled mScarlet overlay
                view_labelled_fovs(slice_id, threshold)

            elif choice == '6':
                # Save and exit/return
                if modified:
                    print("\nSaving changes...")
                    subslice_info_list[subslice_idx] = subslice
                    save_mat(definitions_file, {'subslice_info': subslice_info_list}, format='7.3')
                    print(f"Saved updated definitions to: {definitions_file}")

                    # Regenerate diagnostic plot
                    print("Regenerating diagnostic plot...")
                    mscarlet_positions = subslice['fov_grid_positions'][:len(subslice['mscarlet_fovs'])]
                    if len(subslice['bridge_fovs']) > 0:
                        bridge_positions = subslice['fov_grid_positions'][len(subslice['mscarlet_fovs']):]
                    else:
                        bridge_positions = np.zeros((0, 2))

                    visualize_subslice(
                        slice_id,
                        subslice['mscarlet_fovs'],
                        mscarlet_positions,
                        subslice['bridge_fovs'],
                        bridge_positions,
                        definitions_dir,
                        edited=True
                    )
                    print("Updated diagnostic plot\n")
                else:
                    print("\nNo changes made.")

                if target_slice is not None:
                    print("Done.")
                    return
                else:
                    break  # Return to slice selection

            elif choice == '7':
                # Exit without saving
                if modified:
                    confirm = input("You have unsaved changes. Go back anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue

                if target_slice is not None:
                    print("Exiting without saving.")
                    return

                print("\nReturning to slice selection...\n")
                break

            else:
                print("Invalid choice. Please enter 1-7.\n")

        # If target_slice was specified, exit after editing once
        if target_slice is not None:
            return


def main():
    parser = argparse.ArgumentParser(
        description="Interactive tool to edit subslice FOV definitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--threshold', '-th',
        type=float,
        default=0.0,
        help='Threshold for viewing overlays (affects display only)'
    )
    parser.add_argument(
        '--slice', '-s',
        type=int,
        default=None,
        help='Directly edit specified slice'
    )

    args = parser.parse_args()

    edit_subslice_definitions(
        threshold=args.threshold,
        target_slice=args.slice,
    )


if __name__ == '__main__':
    main()
