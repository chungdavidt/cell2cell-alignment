#!/usr/bin/env python3
"""
Create Aligned Ex Vivo Volume from 2D Slices.

Creates a 3D aligned brain stack from individual 2D ex-vivo tissue slices
using rotation and translation parameters from an alignment CSV file.

This is a Python port of create_ex_vivo_volume_dtc.m and rotate_shift_matrix.m.

Usage:
    python create_ex_vivo_volume.py --input stitched_hyb.mat --alignment alignment.csv
    python create_ex_vivo_volume.py --input stitched_hyb.mat --alignment alignment.csv --slide 2

Input:
    - stitched_hyb.mat: Contains stitched slice images (variables named JH302_SLIDE_SLICE)
    - alignment.csv: Contains alignment parameters with columns:
        * Slide, Slice, X_px_, Y_px_, Degrees

Output:
    - aligned_brain_stack/
        * aligned_stack_DAPI.tif (3D stack)
        * alignment_metadata.mat (transformation info)

Algorithm:
    1. Load stitched slice images from MAT file
    2. For each slice, read alignment parameters (rotation, x/y shift)
    3. Apply inverse rotation + translation using bilinear interpolation
    4. Stack aligned slices into 3D volume
    5. Optionally downsample the volume
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

from scipy.ndimage import map_coordinates

from utils.mat_io import load_mat, save_mat
from utils.image_io import imwrite_tiff

try:
    from skimage.transform import resize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    from scipy.ndimage import zoom


def rotate_shift_matrix(
    input_matrix: np.ndarray,
    theta_deg: float,
    shift_x: float,
    shift_y: float,
) -> np.ndarray:
    """
    Apply rotation and translation to a 2D image using inverse mapping.

    This is a Python port of rotate_shift_matrix.m with exact fidelity.
    Uses backward warping: for each output pixel, find the corresponding
    source pixel using inverse transformation.

    Args:
        input_matrix: 2D input image array
        theta_deg: Rotation angle in degrees (positive = counterclockwise)
        shift_x: Translation in X (column) direction in pixels
        shift_y: Translation in Y (row) direction in pixels

    Returns:
        Rotated and shifted image (same size as input)
    """
    rows, cols = input_matrix.shape
    output = np.zeros_like(input_matrix)

    # Image center (MATLAB convention: center of image)
    cx = (cols + 1) / 2
    cy = (rows + 1) / 2

    # Convert angle to radians
    theta = np.deg2rad(theta_deg)

    # Inverse rotation matrix (to map output -> input)
    # R_inv = [cos(θ)   sin(θ)]
    #         [-sin(θ)  cos(θ)]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Create coordinate grids for output image
    # Using 1-based indexing like MATLAB
    jj, ii = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))

    # Shift output coordinates relative to center and apply translation
    x = jj - cx - shift_x
    y = ii - cy - shift_y

    # Apply inverse rotation to get source coordinates
    x_src = x * cos_t + y * sin_t + cx
    y_src = -x * sin_t + y * cos_t + cy

    # Convert to 0-based indexing for scipy
    x_src_0 = x_src - 1
    y_src_0 = y_src - 1

    # Use bilinear interpolation via map_coordinates
    # map_coordinates expects coordinates as (row, col) = (y, x)
    coords = np.array([y_src_0.ravel(), x_src_0.ravel()])

    # Interpolate with bilinear (order=1), fill out-of-bounds with 0
    output_flat = map_coordinates(
        input_matrix.astype(np.float64),
        coords,
        order=1,  # Bilinear interpolation
        mode='constant',
        cval=0.0,
    )

    output = output_flat.reshape(rows, cols).astype(input_matrix.dtype)

    return output


def downsample_3d(volume: np.ndarray, factors: tuple) -> np.ndarray:
    """
    Downsample a 3D volume using cubic interpolation.

    Args:
        volume: 3D array (Z, Y, X) or (slices, height, width)
        factors: Downsample factors for each dimension (z_factor, y_factor, x_factor)

    Returns:
        Downsampled volume
    """
    new_shape = tuple(int(round(s / f)) for s, f in zip(volume.shape, factors))

    if HAS_SKIMAGE:
        return resize(
            volume,
            new_shape,
            order=3,  # Cubic interpolation
            preserve_range=True,
            anti_aliasing=True,
        ).astype(volume.dtype)
    else:
        zoom_factors = tuple(1.0 / f for f in factors)
        return zoom(volume.astype(np.float64), zoom_factors, order=3).astype(volume.dtype)


def create_ex_vivo_volume(
    input_mat: str,
    alignment_csv: str,
    output_dir: str = None,
    slide_id: int = None,
    downsample_xy: float = 1.0,
    downsample_z: float = 1.0,
):
    """
    Create aligned 3D volume from 2D slices.

    Args:
        input_mat: Path to stitched_hyb.mat file
        alignment_csv: Path to alignment.csv file
        output_dir: Output directory (default: aligned_brain_stack/)
        slide_id: Filter to specific slide ID (default: process all)
        downsample_xy: XY downsample factor (default: 1.0 = no downsampling)
        downsample_z: Z downsample factor (default: 1.0 = no downsampling)
    """
    input_mat = Path(input_mat)
    alignment_csv = Path(alignment_csv)

    if output_dir is None:
        output_dir = input_mat.parent / 'aligned_brain_stack'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("CREATE ALIGNED EX VIVO VOLUME")
    print("=" * 50)
    print(f"  Input MAT: {input_mat}")
    print(f"  Alignment CSV: {alignment_csv}")
    print(f"  Output dir: {output_dir}")
    if slide_id is not None:
        print(f"  Slide filter: {slide_id}")
    print()

    # Load alignment parameters
    print("Loading alignment parameters...")
    if not alignment_csv.exists():
        raise FileNotFoundError(f"Alignment CSV not found: {alignment_csv}")

    alignment_df = pd.read_csv(alignment_csv)
    print(f"  Loaded {len(alignment_df)} alignment entries")

    # Check required columns
    required_cols = ['Slide', 'Slice', 'X_px_', 'Y_px_', 'Degrees']
    # Handle alternative column names
    col_mapping = {}
    for col in required_cols:
        if col in alignment_df.columns:
            col_mapping[col] = col
        elif col.replace('_', '') in alignment_df.columns:
            col_mapping[col] = col.replace('_', '')
        elif col.lower() in [c.lower() for c in alignment_df.columns]:
            for c in alignment_df.columns:
                if c.lower() == col.lower():
                    col_mapping[col] = c
                    break

    # Rename columns to standard names
    alignment_df = alignment_df.rename(columns={v: k for k, v in col_mapping.items()})

    # Filter by slide if specified
    if slide_id is not None:
        alignment_df = alignment_df[alignment_df['Slide'] == slide_id]
        print(f"  Filtered to slide {slide_id}: {len(alignment_df)} slices")

    if len(alignment_df) == 0:
        raise ValueError("No alignment entries found after filtering")

    # Sort by slice number
    alignment_df = alignment_df.sort_values('Slice')
    print()

    # Load stitched images
    print("Loading stitched images...")
    if not input_mat.exists():
        raise FileNotFoundError(f"Input MAT file not found: {input_mat}")

    mat_data = load_mat(input_mat)
    print(f"  Variables in MAT file: {list(mat_data.keys())}")

    # Find slice images (format: JH302_SLIDE_SLICE or similar)
    slice_images = {}
    for key, value in mat_data.items():
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.size > 1000:
            # Try to parse slice info from variable name
            parts = key.split('_')
            if len(parts) >= 3:
                try:
                    parsed_slide = int(parts[-2])
                    parsed_slice = int(parts[-1])
                    if slide_id is None or parsed_slide == slide_id:
                        slice_images[(parsed_slide, parsed_slice)] = value
                except ValueError:
                    pass

    if len(slice_images) == 0:
        # Fall back to using any 2D arrays as slices in order
        print("  WARNING: Could not parse slice names, using all 2D arrays")
        for idx, (key, value) in enumerate(mat_data.items()):
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.size > 1000:
                slice_images[(slide_id or 1, idx + 1)] = value

    print(f"  Found {len(slice_images)} slice images")
    print()

    # Determine output volume size
    # Use largest slice dimensions for consistency
    max_rows = max(img.shape[0] for img in slice_images.values())
    max_cols = max(img.shape[1] for img in slice_images.values())
    n_slices = len(alignment_df)

    print(f"Volume dimensions: {n_slices} x {max_rows} x {max_cols} (Z x H x W)")
    print()

    # Create aligned volume
    print("Aligning slices...")
    aligned_volume = np.zeros((n_slices, max_rows, max_cols), dtype=np.uint16)

    alignment_info = []

    for z_idx, (_, row) in enumerate(alignment_df.iterrows()):
        slice_num = int(row['Slice'])
        slide_num = int(row['Slide'])
        x_shift = float(row['X_px_'])
        y_shift = float(row['Y_px_'])
        rotation = float(row['Degrees'])

        print(f"  [{z_idx + 1}/{n_slices}] Slice {slice_num}: rot={rotation:.1f}°, shift=({x_shift:.1f}, {y_shift:.1f})")

        # Get slice image
        key = (slide_num, slice_num)
        if key not in slice_images:
            print(f"    WARNING: Slice image not found, skipping")
            continue

        slice_img = slice_images[key]

        # Pad or crop to match volume dimensions
        padded = np.zeros((max_rows, max_cols), dtype=slice_img.dtype)
        h, w = slice_img.shape
        padded[:min(h, max_rows), :min(w, max_cols)] = slice_img[:min(h, max_rows), :min(w, max_cols)]

        # Apply rotation and translation
        aligned = rotate_shift_matrix(padded, rotation, x_shift, y_shift)

        # Store in volume
        aligned_volume[z_idx] = aligned.astype(np.uint16)

        # Record alignment info
        alignment_info.append({
            'z_index': z_idx,
            'slice_id': slice_num,
            'slide_id': slide_num,
            'rotation_deg': rotation,
            'shift_x': x_shift,
            'shift_y': y_shift,
        })

    print()

    # Apply downsampling if requested
    if downsample_xy > 1.0 or downsample_z > 1.0:
        print(f"Downsampling volume: XY={downsample_xy}x, Z={downsample_z}x")
        aligned_volume = downsample_3d(
            aligned_volume,
            (downsample_z, downsample_xy, downsample_xy)
        )
        print(f"  New dimensions: {aligned_volume.shape}")
        print()

    # Save outputs
    print("Saving aligned volume...")

    # Save as multi-page TIFF
    stack_file = output_dir / 'aligned_stack_DAPI.tif'
    imwrite_tiff(stack_file, aligned_volume)
    stack_size = stack_file.stat().st_size / 1e6
    print(f"  Stack: {stack_file} ({stack_size:.1f} MB)")

    # Save metadata
    metadata = {
        'alignment_info': alignment_info,
        'original_dimensions': [n_slices, max_rows, max_cols],
        'final_dimensions': list(aligned_volume.shape),
        'downsample_xy': downsample_xy,
        'downsample_z': downsample_z,
        'input_mat': str(input_mat),
        'alignment_csv': str(alignment_csv),
        'processing_date': str(datetime.now()),
    }
    metadata_file = output_dir / 'alignment_metadata.mat'
    save_mat(metadata_file, metadata, format='7.3')
    print(f"  Metadata: {metadata_file}")
    print()

    # Summary
    print("=" * 50)
    print("ALIGNMENT COMPLETE")
    print("=" * 50)
    print(f"  Aligned slices: {n_slices}")
    print(f"  Volume shape: {aligned_volume.shape}")
    print(f"  Output: {output_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Create aligned 3D ex-vivo volume from 2D slices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to stitched_hyb.mat file'
    )
    parser.add_argument(
        '--alignment', '-a',
        type=str,
        required=True,
        help='Path to alignment.csv file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: aligned_brain_stack/)'
    )
    parser.add_argument(
        '--slide', '-s',
        type=int,
        default=None,
        help='Filter to specific slide ID'
    )
    parser.add_argument(
        '--downsample-xy',
        type=float,
        default=1.0,
        help='XY downsample factor (default: 1.0 = no downsampling)'
    )
    parser.add_argument(
        '--downsample-z',
        type=float,
        default=1.0,
        help='Z downsample factor (default: 1.0 = no downsampling)'
    )

    args = parser.parse_args()

    create_ex_vivo_volume(
        input_mat=args.input,
        alignment_csv=args.alignment,
        output_dir=args.output,
        slide_id=args.slide,
        downsample_xy=args.downsample_xy,
        downsample_z=args.downsample_z,
    )


if __name__ == '__main__':
    main()
