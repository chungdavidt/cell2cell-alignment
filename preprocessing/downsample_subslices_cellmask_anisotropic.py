#!/usr/bin/env python3
"""
Downsample Stitched Subslices with Anisotropic Scaling.

Downsamples ALL channels (DAPI, GCAMP, MSCARLET, CELLMASK) from stitched subslices
using ANISOTROPIC scaling to match in-vivo resolution correctly in both dimensions.

This is a Python port of downsample_subslices_refined.m with exact fidelity,
updated to use anisotropic scaling.

ANISOTROPIC SCALING (matches in vivo resolution):
    - X: 0.32 -> 2.34 um/px (7.31x downsample, matches in vivo X/Y)
    - Y: 0.32 -> 1.0 um/px (3.125x downsample, matches in vivo Z after xrotate=90)

Usage:
    python downsample_subslices_cellmask_anisotropic.py
    python downsample_subslices_cellmask_anisotropic.py --slice 22  # Process slice 22 only
    python downsample_subslices_cellmask_anisotropic.py --cellmask-only  # Only cellmasks (legacy)

Input:
    - HYB_subslice_stitched_tif/ (from stitch_subslices.py)
        * slice{N}_subslice_DAPI.tif
        * slice{N}_subslice_GCAMP.tif
        * slice{N}_subslice_MSCARLET.tif
        * slice{N}_subslice_CELLMASK.h5 (HDF5 format)

Output:
    - HYB_subslice_stitched_tif_downsampled_micronwise_anisotropic/
        * slice{N}_subslice_DAPI.tif (bilinear downsampled)
        * slice{N}_subslice_GCAMP.tif (bilinear downsampled)
        * slice{N}_subslice_MSCARLET.tif (bilinear downsampled)
        * slice{N}_subslice_CELLMASK.h5 (nearest-neighbor, preserves cell IDs)
        * downsample_metadata.mat (scale factors, timestamps)

Downsampling:
    - X dimension: 7.31x (0.32 -> 2.34 um/px)
    - Y dimension: 3.125x (0.32 -> 1.0 um/px)
    - Method: Bilinear for images, Nearest-neighbor for cellmask
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import re

try:
    from skimage.transform import resize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    from scipy.ndimage import zoom

from config import (
    HYB_STITCHED_DIR,
    HYB_DOWNSAMPLED_DIR,
    EXVIVO_UM_PER_PX,
    INVIVO_XY_UM_PER_PX,
    INVIVO_Z_UM_PER_PX,
    DOWNSAMPLE_X,
    DOWNSAMPLE_Y,
)
from utilities.mat_io import load_mat, save_mat, load_cellmask_h5, save_cellmask_h5
from utilities.image_io import imread_tiff, imwrite_tiff


def imresize_nearest(image: np.ndarray, output_shape: tuple) -> np.ndarray:
    """
    Resize image using nearest-neighbor interpolation.

    This preserves cell IDs in segmentation masks.

    Args:
        image: Input image array
        output_shape: Target (height, width) shape

    Returns:
        Resized image
    """
    if HAS_SKIMAGE:
        # order=0 is nearest-neighbor
        # preserve_range=True keeps original dtype values
        return resize(
            image,
            output_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(image.dtype)
    else:
        # Use scipy zoom as fallback
        zoom_factors = (output_shape[0] / image.shape[0],
                        output_shape[1] / image.shape[1])
        return zoom(image, zoom_factors, order=0)


def imresize_bilinear(image: np.ndarray, output_shape: tuple) -> np.ndarray:
    """
    Resize image using bilinear interpolation.

    This is appropriate for intensity images (DAPI, GCAMP, MSCARLET).
    Matches MATLAB's imresize with 'bilinear' method.

    Args:
        image: Input image array (uint16 typically)
        output_shape: Target (height, width) shape

    Returns:
        Resized image with same dtype as input
    """
    original_dtype = image.dtype

    if HAS_SKIMAGE:
        # order=1 is bilinear interpolation
        # preserve_range=True keeps original intensity range
        # anti_aliasing=True for quality downsampling (like MATLAB default)
        result = resize(
            image,
            output_shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True
        )
    else:
        # Use scipy zoom as fallback
        zoom_factors = (output_shape[0] / image.shape[0],
                        output_shape[1] / image.shape[1])
        result = zoom(image.astype(np.float64), zoom_factors, order=1)

    # Preserve original dtype (important for uint16 images)
    if np.issubdtype(original_dtype, np.integer):
        result = np.clip(result, np.iinfo(original_dtype).min, np.iinfo(original_dtype).max)
    return result.astype(original_dtype)


def downsample_subslices_cellmask_anisotropic(target_slice: int = None, cellmask_only: bool = False):
    """
    Main function to downsample all channels with anisotropic scaling.

    Args:
        target_slice: If specified, process only this slice
        cellmask_only: If True, only process cellmasks (legacy behavior)
    """
    input_dir = Path(HYB_STITCHED_DIR)
    output_dir = Path(HYB_DOWNSAMPLED_DIR)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Stitched subslices not found!\n"
            f"Run stitch_subslices.py first.\n"
            f"Expected: {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 40)
    if cellmask_only:
        print("DOWNSAMPLE CELLMASKS ONLY (ANISOTROPIC)")
    else:
        print("DOWNSAMPLE ALL CHANNELS (ANISOTROPIC)")
    print("=" * 40)
    print("Resolution mapping:")
    print(f"  Ex vivo original: {EXVIVO_UM_PER_PX:.2f} um/px (both X and Y)")
    print()
    print("  In vivo targets:")
    print(f"    X (Lateral-Medial): {INVIVO_XY_UM_PER_PX:.2f} um/px")
    print(f"    Y (Dorsal-Ventral): {INVIVO_Z_UM_PER_PX:.2f} um/px")
    print()
    print("  Downsample factors:")
    print(f"    X dimension: {DOWNSAMPLE_X:.4f}x (ex vivo X -> in vivo X)")
    print(f"    Y dimension: {DOWNSAMPLE_Y:.4f}x (ex vivo Y -> in vivo Z)")
    print()
    if not cellmask_only:
        print("  Channels to process: DAPI, GCAMP, MSCARLET, CELLMASK")
        print("  Image interpolation: Bilinear (for quality)")
        print("  Cellmask interpolation: Nearest-neighbor (preserves cell IDs)")
        print()

    # Find stitched subslices (check for .h5 first, fall back to .mat)
    cellmask_files = list(input_dir.glob("slice*_subslice_CELLMASK.h5"))
    use_h5_format = True

    if not cellmask_files:
        # Fall back to .mat format for backwards compatibility
        cellmask_files = list(input_dir.glob("slice*_subslice_CELLMASK.mat"))
        use_h5_format = False

    if not cellmask_files:
        raise FileNotFoundError(f"No stitched cellmasks found in: {input_dir}")

    print(f"Found {len(cellmask_files)} stitched cellmasks ({'H5' if use_h5_format else 'MAT'} format)\n")

    # Filter by target slice if specified
    if target_slice is not None:
        ext = '.h5' if use_h5_format else '.mat'
        pattern = f"slice{target_slice}_subslice_CELLMASK{ext}"
        cellmask_files = [f for f in cellmask_files if f.name == pattern]
        if not cellmask_files:
            raise ValueError(f"Slice {target_slice} not found")
        print(f"Processing slice {target_slice} only\n")

    # Sort files for consistent ordering
    cellmask_files.sort()

    # Metadata for tracking
    metadata = {
        'downsample_x': DOWNSAMPLE_X,
        'downsample_y': DOWNSAMPLE_Y,
        'exvivo_resolution_um_per_pixel': EXVIVO_UM_PER_PX,
        'invivo_x_resolution_um_per_pixel': INVIVO_XY_UM_PER_PX,
        'invivo_y_resolution_um_per_pixel': INVIVO_Z_UM_PER_PX,
        'processing_date': str(datetime.now()),
        'slices_processed': [],
    }

    # Process each subslice
    for i, cellmask_file in enumerate(cellmask_files):
        base_name = cellmask_file.stem.replace('_CELLMASK', '')

        # Parse slice number
        match = re.search(r'slice(\d+)_subslice', base_name)
        if not match:
            print(f"WARNING: Could not parse slice ID from: {base_name}")
            continue
        slice_id = int(match.group(1))

        print("=" * 40)
        print(f"[{i+1}/{len(cellmask_files)}] Processing {base_name}")
        print("=" * 40)

        # Load cellmask
        cellmask_output_path = output_dir / f"{base_name}_CELLMASK.h5"

        print("  Loading cellmask...")

        # Load based on format
        if use_h5_format:
            cellmask_fullres, input_metadata = load_cellmask_h5(cellmask_file)
            min_x_offset = input_metadata.get('min_x', 0)
            min_y_offset = input_metadata.get('min_y', 0)
        else:
            # Legacy .mat format
            mask_data = load_mat(cellmask_file)

            # Get cellmask variable
            if 'cellmask_stitched' in mask_data:
                cellmask_fullres = np.asarray(mask_data['cellmask_stitched'])
            else:
                # Find the largest numeric array
                cellmask_fullres = None
                for key, value in mask_data.items():
                    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                        if value.size > 100:
                            cellmask_fullres = np.asarray(value)
                            break

                if cellmask_fullres is None:
                    print(f"  WARNING: No cellmask found in {cellmask_file.name}, skipping")
                    continue

            # Load offset information
            min_x_offset = int(mask_data.get('min_x', 0))
            min_y_offset = int(mask_data.get('min_y', 0))

        orig_h, orig_w = cellmask_fullres.shape
        print(f"    Original size: {orig_h} x {orig_w} (H x W)")
        print(f"    Canvas offset: x={min_x_offset}, y={min_y_offset}")

        # Calculate new dimensions with ANISOTROPIC scaling
        new_h = round(orig_h / DOWNSAMPLE_Y)  # Y dimension
        new_w = round(orig_w / DOWNSAMPLE_X)  # X dimension

        print("  Downsampling anisotropically...")
        print(f"    Target size: {new_h} x {new_w} (H x W)")
        print(f"    X: {DOWNSAMPLE_X:.4f}x downsample (W: {orig_w} -> {new_w})")
        print(f"    Y: {DOWNSAMPLE_Y:.4f}x downsample (H: {orig_h} -> {new_h})")

        # Downsample using nearest-neighbor (preserve cell IDs)
        cellmask_downsampled = imresize_nearest(cellmask_fullres, (new_h, new_w))

        # Count unique cells
        n_cells_orig = len(np.unique(cellmask_fullres[cellmask_fullres > 0]))
        n_cells_down = len(np.unique(cellmask_downsampled[cellmask_downsampled > 0]))
        preservation_pct = 100 * n_cells_down / n_cells_orig if n_cells_orig > 0 else 0
        print(f"    Unique cells: {n_cells_orig} -> {n_cells_down} ({preservation_pct:.1f}% preserved)")

        # Save downsampled cellmask with metadata (HDF5 format)
        save_cellmask_h5(cellmask_output_path, cellmask_downsampled, metadata={
            'DOWNSAMPLE_X': DOWNSAMPLE_X,
            'DOWNSAMPLE_Y': DOWNSAMPLE_Y,
            'min_x_offset': min_x_offset,
            'min_y_offset': min_y_offset,
            'EXVIVO_UM_PER_PX': EXVIVO_UM_PER_PX,
            'INVIVO_XY_UM_PER_PX': INVIVO_XY_UM_PER_PX,
            'INVIVO_Z_UM_PER_PX': INVIVO_Z_UM_PER_PX,
        })

        file_size = cellmask_output_path.stat().st_size / 1e6
        print(f"    Saved CELLMASK: {file_size:.1f} MB")

        # Process TIF channels if not cellmask-only mode
        if not cellmask_only:
            channels = ['DAPI', 'GCAMP', 'MSCARLET']
            for channel in channels:
                tif_file = input_dir / f"{base_name}_{channel}.tif"
                tif_output_path = output_dir / f"{base_name}_{channel}.tif"

                if not tif_file.exists():
                    print(f"    WARNING: {channel} TIF not found, skipping")
                    continue

                print(f"  Downsampling {channel}...")
                try:
                    # Load TIF image
                    img_fullres = imread_tiff(tif_file)

                    # Verify dimensions match cellmask
                    if img_fullres.shape[:2] != (orig_h, orig_w):
                        print(f"    WARNING: {channel} size {img_fullres.shape[:2]} != cellmask size {(orig_h, orig_w)}")

                    # Downsample using bilinear interpolation (for quality)
                    img_downsampled = imresize_bilinear(img_fullres, (new_h, new_w))

                    # Save downsampled TIF
                    imwrite_tiff(tif_output_path, img_downsampled)

                    tif_size = tif_output_path.stat().st_size / 1e6
                    print(f"    Saved {channel}: {tif_size:.1f} MB")

                except Exception as e:
                    print(f"    ERROR processing {channel}: {e}")
                    continue

        print()

        # Update metadata
        metadata['slices_processed'].append(slice_id)

    # Save metadata
    metadata_file = output_dir / 'downsample_metadata.mat'
    save_mat(metadata_file, {'metadata': metadata}, format='7.3')

    print("=" * 40)
    print("DOWNSAMPLING COMPLETE")
    print("=" * 40)
    print(f"Processed slices: {len(metadata['slices_processed'])}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved: {metadata_file}")
    print("\nDownsampling summary:")
    print(f"  X factor: {DOWNSAMPLE_X:.4f}x (0.32 -> 2.34 um/px)")
    print(f"  Y factor: {DOWNSAMPLE_Y:.4f}x (0.32 -> 1.0 um/px)")
    print("  Method: Nearest-neighbor (preserves cell IDs)")
    print("  Result: NON-SQUARE pixels but PHYSICALLY CORRECT!")
    print("\nNext step:")
    print("  Run generate_mscarlet_cellmask_subslice_anisotropic.py to create overlays")
    print("  Or run interactive_mscarlet_threshold_cellmask_subslice_anisotropic.py for interactive viewer")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Downsample all channels with anisotropic scaling",
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
        '--cellmask-only',
        action='store_true',
        help='Only process cellmasks (skip TIF channels)'
    )

    args = parser.parse_args()

    downsample_subslices_cellmask_anisotropic(
        target_slice=args.slice,
        cellmask_only=args.cellmask_only
    )


if __name__ == '__main__':
    main()
