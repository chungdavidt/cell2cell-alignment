#!/usr/bin/env python3
"""
Stitch Subslices - Create Stitched FOV Images for Subslices.

Stitches FOVs in subslice definitions to create composite images.
Uses regression-based positioning (same as ben_generate_in_situ_stack_dtc.m).

This is a Python port of stitch_subslices.m with exact fidelity.

Usage:
    python stitch_subslices.py
    python stitch_subslices.py --slice 22     # Stitch specific slice only
    python stitch_subslices.py --test         # Test mode (first subslice only)

Input:
    - subslice_definitions.mat (from identify_mscarlet_subslices.py)
    - hyb/ directory with FOV TIFFs (multi-page: GCAMP, mScarlet, DAPI)
    - hyb_channels/ directory with extracted channels and cellmasks

Output:
    - HYB_subslice_stitched_tif/ directory with:
        * slice{N}_subslice_GCAMP.tif
        * slice{N}_subslice_MSCARLET.tif
        * slice{N}_subslice_DAPI.tif
        * slice{N}_subslice_CELLMASK.h5 (HDF5 format, Python-native)

Algorithm (same as ben_generate_in_situ_stack_dtc.m):
    1. For each FOV, use linear regression on filt_neurons positions
    2. Calculate global offset: pos*2 = offset + pos40x
    3. Place FOV on canvas at calculated offset
    4. Use max-projection for overlapping regions
    5. Stitch cellmasks with global ID offsets to preserve cell identity
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import sparse
import time

from config import (
    FILT_NEURONS_PATH,
    SUBSLICE_DEFINITIONS_FILE,
    HYB_ROOT,
    HYB_CHANNELS_DIR,
    HYB_STITCHED_DIR,
    FOV_SIZE,
)
from utils.mat_io import load_filt_neurons, load_mat, save_mat, save_cellmask_h5
from utils.image_io import load_fov_images, imwrite_tiff, get_file_size_mb
from utils.regression import calculate_fov_offset


def stitch_fov_channels(
    fov_list: list,
    slice_id: int,
    filt_neurons: dict,
    hyb_root: Path,
    channels_root: Path,
) -> tuple:
    """
    Stitch FOVs using regression-based positioning.

    Args:
        fov_list: List of FOV names to stitch
        slice_id: Slice number
        filt_neurons: filt_neurons data dict
        hyb_root: Path to hyb/ directory
        channels_root: Path to extracted channels directory

    Returns:
        Tuple of (gcamp_stitched, dapi_stitched, mscarlet_stitched,
                  cellmask_stitched, min_x, min_y, fov_offsets_dict)

        fov_offsets_dict: Dict mapping FOV name -> [x_offset, y_offset]
                          for downstream position mapping
    """
    n_fovs = len(fov_list)
    print("  Calculating FOV offsets using regression...")

    # Get data arrays
    fov_names = filt_neurons['fov']
    slice_ids = np.asarray(filt_neurons['slice']).flatten()
    pos = np.asarray(filt_neurons['pos'])
    pos40x = np.asarray(filt_neurons['pos40x'])

    # Initialize offsets array and dict for output
    fov_offsets = np.zeros((n_fovs, 2))  # [x_offset, y_offset]
    fov_loaded_flags = np.zeros(n_fovs, dtype=bool)
    fov_offsets_dict = {}  # FOV name -> [x_offset, y_offset] for MATLAB compatibility

    # Calculate offset for each FOV using regression on cell positions
    for i, fov_name in enumerate(fov_list):
        # Get cells in this FOV and slice
        in_fov = np.array([f == fov_name for f in fov_names])
        in_slice = slice_ids == slice_id
        cell_mask = in_fov & in_slice
        i_cell = np.where(cell_mask)[0]

        if len(i_cell) < 3:
            print(f"    WARNING: FOV {fov_name} has < 3 cells, skipping")
            continue

        try:
            # Regression: pos*2 = offset + pos40x
            x_offset, y_offset = calculate_fov_offset(
                pos[i_cell],
                pos40x[i_cell],
                scale_factor=2.0
            )

            fov_offsets[i] = [x_offset, y_offset]
            fov_loaded_flags[i] = True
            # Store in dict for MATLAB-compatible output
            fov_offsets_dict[fov_name] = [x_offset, y_offset]
        except Exception as e:
            print(f"    WARNING: Failed to calculate offset for {fov_name}: {e}")
            continue

    n_loaded = np.sum(fov_loaded_flags)
    print(f"    Calculated offsets for {n_loaded}/{n_fovs} FOVs")

    if n_loaded == 0:
        raise ValueError("No FOVs could be positioned (insufficient cells)")

    # Determine canvas size
    valid_offsets = fov_offsets[fov_loaded_flags]

    min_x = int(np.min(valid_offsets[:, 0]))
    max_x = int(np.max(valid_offsets[:, 0])) + FOV_SIZE
    min_y = int(np.min(valid_offsets[:, 1]))
    max_y = int(np.max(valid_offsets[:, 1])) + FOV_SIZE

    canvas_width = max_x - min_x
    canvas_height = max_y - min_y

    print(f"    Canvas size: {canvas_height} x {canvas_width} pixels")
    print(f"    Memory required: ~{canvas_height * canvas_width * 2 / 1e6:.1f} MB per channel")

    # Initialize canvases
    gcamp_stitched = np.zeros((canvas_height, canvas_width), dtype=np.uint16)
    dapi_stitched = np.zeros((canvas_height, canvas_width), dtype=np.uint16)
    mscarlet_stitched = np.zeros((canvas_height, canvas_width), dtype=np.uint16)
    cellmask_stitched = np.zeros((canvas_height, canvas_width), dtype=np.uint32)

    current_max_cell_id = 0  # Track max cell ID for global offsets

    # Stitch each FOV
    print("  Stitching FOVs...")
    for i in range(n_fovs):
        if not fov_loaded_flags[i]:
            continue

        fov_name = fov_list[i]
        print(f"    [{i+1:2d}/{n_fovs:2d}] {fov_name}")

        # Load FOV images
        gcamp, dapi, mscarlet, cellmask = load_fov_images(
            fov_name, hyb_root, channels_root
        )

        if gcamp is None:
            print("      WARNING: Failed to load FOV, skipping")
            continue

        # Get offset (adjusted for canvas origin)
        # MATLAB: x_off = fov_offsets(i, 1) - min_x + 1
        # Python: x_off = fov_offsets[i, 0] - min_x (0-indexed)
        x_off = int(fov_offsets[i, 0] - min_x)
        y_off = int(fov_offsets[i, 1] - min_y)

        # Determine FOV size (actual loaded size)
        fov_h, fov_w = gcamp.shape[:2]

        # Canvas region bounds
        canvas_y1 = y_off
        canvas_y2 = y_off + fov_h
        canvas_x1 = x_off
        canvas_x2 = x_off + fov_w

        # Clip to canvas bounds
        canvas_y1_clipped = max(0, canvas_y1)
        canvas_y2_clipped = min(canvas_height, canvas_y2)
        canvas_x1_clipped = max(0, canvas_x1)
        canvas_x2_clipped = min(canvas_width, canvas_x2)

        # FOV region bounds (corresponding to canvas region)
        fov_y1 = canvas_y1_clipped - y_off
        fov_y2 = canvas_y2_clipped - y_off
        fov_x1 = canvas_x1_clipped - x_off
        fov_x2 = canvas_x2_clipped - x_off

        # Max-projection stitching for images
        gcamp_stitched[canvas_y1_clipped:canvas_y2_clipped, canvas_x1_clipped:canvas_x2_clipped] = np.maximum(
            gcamp_stitched[canvas_y1_clipped:canvas_y2_clipped, canvas_x1_clipped:canvas_x2_clipped],
            gcamp[fov_y1:fov_y2, fov_x1:fov_x2]
        )

        if dapi is not None:
            dapi_stitched[canvas_y1_clipped:canvas_y2_clipped, canvas_x1_clipped:canvas_x2_clipped] = np.maximum(
                dapi_stitched[canvas_y1_clipped:canvas_y2_clipped, canvas_x1_clipped:canvas_x2_clipped],
                dapi[fov_y1:fov_y2, fov_x1:fov_x2]
            )

        if mscarlet is not None:
            mscarlet_stitched[canvas_y1_clipped:canvas_y2_clipped, canvas_x1_clipped:canvas_x2_clipped] = np.maximum(
                mscarlet_stitched[canvas_y1_clipped:canvas_y2_clipped, canvas_x1_clipped:canvas_x2_clipped],
                mscarlet[fov_y1:fov_y2, fov_x1:fov_x2]
            )

        # Cellmask stitching with global ID offset
        # MATLAB-compatible: overwrite canvas with offset cell IDs (not "first wins")
        if cellmask is not None:
            cellmask_region = cellmask[fov_y1:fov_y2, fov_x1:fov_x2].astype(np.uint32)
            mask = cellmask_region > 0

            if np.any(mask):
                # Get canvas region and add offset to non-zero cells
                canvas_region = cellmask_stitched[canvas_y1_clipped:canvas_y2_clipped, canvas_x1_clipped:canvas_x2_clipped]
                # Overwrite canvas where current FOV has cells (MATLAB behavior)
                canvas_region[mask] = cellmask_region[mask] + current_max_cell_id
                cellmask_stitched[canvas_y1_clipped:canvas_y2_clipped, canvas_x1_clipped:canvas_x2_clipped] = canvas_region

                # Update max cell ID from ENTIRE stitched array (MATLAB: current_max_cell_id = max(cellmask_stitched(:)))
                current_max_cell_id = int(np.max(cellmask_stitched))

    print(f"    Stitched {np.sum(fov_loaded_flags)} FOVs")
    n_unique_cells = len(np.unique(cellmask_stitched[cellmask_stitched > 0]))
    print(f"    Total unique cells in stitched cellmask: {n_unique_cells}")

    return gcamp_stitched, dapi_stitched, mscarlet_stitched, cellmask_stitched, min_x, min_y, fov_offsets_dict


def stitch_subslices(target_slice: int = None, test_mode: bool = False):
    """
    Main function to stitch subslices.

    Args:
        target_slice: If specified, process only this slice
        test_mode: If True, process only first subslice
    """
    print("=" * 40)
    print("STITCH SUBSLICES")
    print("=" * 40)
    if test_mode:
        print("Mode: TEST (first subslice only)")
    elif target_slice is not None:
        print(f"Mode: SLICE {target_slice} only")
    else:
        print("Mode: FULL (all subslices)")
    print()

    # Create output directory
    output_dir = Path(HYB_STITCHED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load subslice definitions
    print("Loading subslice definitions...")
    if not Path(SUBSLICE_DEFINITIONS_FILE).exists():
        raise FileNotFoundError(
            f"Subslice definitions not found!\n"
            f"Run identify_mscarlet_subslices.py first.\n"
            f"Expected: {SUBSLICE_DEFINITIONS_FILE}"
        )

    definitions_data = load_mat(SUBSLICE_DEFINITIONS_FILE)
    subslice_info_list = definitions_data['subslice_info']

    # Handle different formats from load_mat
    if isinstance(subslice_info_list, np.ndarray):
        subslice_info_list = list(subslice_info_list.flatten())
    elif hasattr(subslice_info_list, '_fieldnames'):
        # Single mat_struct - wrap in list
        subslice_info_list = [subslice_info_list]
    elif not isinstance(subslice_info_list, list):
        subslice_info_list = [subslice_info_list]

    print(f"  Loaded {len(subslice_info_list)} subslices")

    # Load filt_neurons
    print("Loading filt_neurons...")
    filt_neurons = load_filt_neurons(FILT_NEURONS_PATH)
    print("  Data loaded\n")

    # Filter subslices to process
    if target_slice is not None:
        subslice_info_list = [
            s for s in subslice_info_list
            if (s['slice_id'] if isinstance(s, dict) else s.slice_id) == target_slice
        ]
        if len(subslice_info_list) == 0:
            raise ValueError(f"Slice {target_slice} not found in subslice definitions")

    if test_mode:
        print("TEST MODE: Processing first subslice only\n")
        subslice_info_list = subslice_info_list[:1]

    # Process each subslice
    hyb_root = Path(HYB_ROOT)
    channels_root = Path(HYB_CHANNELS_DIR)

    for s_idx, subslice_info in enumerate(subslice_info_list):
        # Handle both dict and object access
        if isinstance(subslice_info, dict):
            slice_id = subslice_info['slice_id']
            fov_list = subslice_info['fov_list']
            mscarlet_fovs = subslice_info.get('mscarlet_fovs', [])
            bridge_fovs = subslice_info.get('bridge_fovs', [])
        else:
            slice_id = subslice_info.slice_id
            fov_list = subslice_info.fov_list
            mscarlet_fovs = subslice_info.mscarlet_fovs if hasattr(subslice_info, 'mscarlet_fovs') else []
            bridge_fovs = subslice_info.bridge_fovs if hasattr(subslice_info, 'bridge_fovs') else []

        # Ensure fov_list is a list of strings (not characters from a single string)
        if isinstance(fov_list, str):
            # Single FOV name - wrap in list
            fov_list = [fov_list]
        elif isinstance(fov_list, np.ndarray):
            fov_list = [str(f) for f in fov_list.flatten()]
        elif not isinstance(fov_list, list):
            # Try to convert, but check if it's iterable of strings vs single string
            fov_list = list(fov_list)
            # If we got single characters, it was a string that got split
            if len(fov_list) > 0 and all(len(f) == 1 for f in fov_list):
                # Rejoin - this was a single FOV name
                fov_list = [''.join(fov_list)]

        # Same handling for mscarlet_fovs and bridge_fovs
        if isinstance(mscarlet_fovs, str):
            mscarlet_fovs = [mscarlet_fovs]
        elif isinstance(mscarlet_fovs, np.ndarray):
            mscarlet_fovs = [str(f) for f in mscarlet_fovs.flatten()]
        elif not isinstance(mscarlet_fovs, list):
            mscarlet_fovs = list(mscarlet_fovs) if mscarlet_fovs else []

        if isinstance(bridge_fovs, str):
            bridge_fovs = [bridge_fovs]
        elif isinstance(bridge_fovs, np.ndarray):
            bridge_fovs = [str(f) for f in bridge_fovs.flatten()]
        elif not isinstance(bridge_fovs, list):
            bridge_fovs = list(bridge_fovs) if bridge_fovs else []

        print("=" * 40)
        print(f"[{s_idx+1}/{len(subslice_info_list)}] Stitching Slice {slice_id} Subslice")
        print("=" * 40)
        print(f"  FOVs to stitch: {len(fov_list)}")
        print(f"  mScarlet+ FOVs: {len(mscarlet_fovs)}")
        print(f"  Bridge FOVs: {len(bridge_fovs)}\n")

        # Stitch each channel
        stitch_start = time.time()

        gcamp_stitched, dapi_stitched, mscarlet_stitched, cellmask_stitched, min_x, min_y, fov_offsets_dict = \
            stitch_fov_channels(fov_list, slice_id, filt_neurons, hyb_root, channels_root)

        stitch_time = time.time() - stitch_start
        print(f"  Stitching completed in {stitch_time:.1f} seconds\n")

        # Save outputs
        print("  Saving stitched channels...")
        save_start = time.time()

        output_prefix = f"slice{slice_id}_subslice"

        # Save GCAMP
        gcamp_file = output_dir / f"{output_prefix}_GCAMP.tif"
        imwrite_tiff(gcamp_file, gcamp_stitched)
        print(f"    GCAMP: {get_file_size_mb(gcamp_file):.1f} MB")

        # Save DAPI
        dapi_file = output_dir / f"{output_prefix}_DAPI.tif"
        imwrite_tiff(dapi_file, dapi_stitched)
        print(f"    DAPI: {get_file_size_mb(dapi_file):.1f} MB")

        # Save mScarlet
        mscarlet_file = output_dir / f"{output_prefix}_MSCARLET.tif"
        imwrite_tiff(mscarlet_file, mscarlet_stitched)
        print(f"    mSCARLET: {get_file_size_mb(mscarlet_file):.1f} MB")

        # Save cellmask with offset information for position mapping
        # Using HDF5 format (.h5) - Python-native, also readable by MATLAB via h5read()
        cellmask_file = output_dir / f"{output_prefix}_CELLMASK.h5"
        save_cellmask_h5(cellmask_file, cellmask_stitched, metadata={
            'fov_offsets': fov_offsets_dict,  # FOV name -> [x_offset, y_offset]
            'min_x': min_x,
            'min_y': min_y,
        })
        print(f"    CELLMASK: {get_file_size_mb(cellmask_file):.1f} MB")

        save_time = time.time() - save_start
        print(f"  Files saved in {save_time:.1f} seconds\n")

    # Summary
    print("=" * 40)
    print("STITCHING COMPLETE")
    print("=" * 40)
    print(f"Output directory: {output_dir}")
    print(f"Subslices stitched: {len(subslice_info_list)}")
    print("\nNext steps:")
    print("  1. Run downsample_subslices_cellmask_anisotropic.py to match in-vivo resolution")
    print("  2. Run generate_mscarlet_cellmask_subslice_anisotropic.py to create overlays")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Stitch FOVs into composite subslice images",
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

    stitch_subslices(
        target_slice=args.slice,
        test_mode=args.test
    )


if __name__ == '__main__':
    main()
