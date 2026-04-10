#!/usr/bin/env python3
r"""
Prepare In-Vivo Volume for LineStuffUp Pipeline.

Flips the in vivo 2P imaging stack vertically (top-bottom flip)
to correct for mirror artifact in acquisition.

This is a Python port of flip_invivo_stack.py, using tifffile instead of imageio.

Flip: Y-axis (axis=1 in ZYX ordering)

Usage:
    python prepare_invivo_volume.py
    python prepare_invivo_volume.py --input /path/to/stack.tiff
    python prepare_invivo_volume.py --input /path/to/stack.tiff --output /path/to/output_dir

Input:
    - Raw in-vivo TIFF stack (ZYX ordering from tifffile)

Output:
    - in_vivo_flip_corrected/{input_stem}_flipped.tiff
"""

import argparse
import numpy as np
from pathlib import Path
import tifffile
import time

# Default paths (from local_config via config.py)
from config import DATA_ROOT, OUTPUT_ROOT

DEFAULT_RAW_DATA_ROOT = Path(DATA_ROOT)
DEFAULT_INVIVO_DIR = DEFAULT_RAW_DATA_ROOT / "in_vivo_stack"
DEFAULT_OUTPUT_ROOT = Path(OUTPUT_ROOT)


def flip_invivo_stack(
    input_path: Path,
    output_dir: Path,
    axis: int = 1  # Y-axis for ZYX stacks
) -> Path:
    """
    Flip a TIFF stack along specified axis.

    Parameters
    ----------
    input_path : Path
        Path to input TIFF file
    output_dir : Path
        Directory to save flipped file
    axis : int
        Axis to flip along (1 = Y-axis for ZYX, default: 1)

    Returns
    -------
    Path
        Path to output file
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IN VIVO STACK FLIP CORRECTION")
    print("=" * 70)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"Flip axis: {axis} (Y-axis vertical flip)")
    print()

    start_time = time.time()

    # Read stack
    print(f"Reading... ", end="", flush=True)
    stack = tifffile.imread(str(input_path))
    original_shape = stack.shape
    print(f"shape {original_shape}")

    # Flip along Y-axis
    print(f"Flipping along axis {axis}... ", end="", flush=True)
    flipped = np.flip(stack, axis=axis)
    print("done")

    # Verify flip
    print(f"Original shape: {original_shape}")
    print(f"Flipped shape:  {flipped.shape}")
    assert flipped.shape == original_shape, "Shape mismatch after flip!"

    # Save flipped stack with "_flipped" suffix
    output_name = input_path.stem + "_flipped" + input_path.suffix
    output_path = output_dir / output_name
    print(f"Saving to {output_path.name}... ", end="", flush=True)

    # Ensure contiguous array for efficient saving
    flipped = np.ascontiguousarray(flipped)

    tifffile.imwrite(
        str(output_path),
        flipped,
        photometric='minisblack',
        compression=None,
    )

    elapsed = time.time() - start_time
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"done ({elapsed:.1f}s, {file_size_mb:.1f} MB)")

    print()
    print("=" * 70)
    print("FLIP CORRECTION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    print("\nNext steps:")
    print("1. Verify flip is correct by viewing images")
    print("2. Update notebook to use flipped files")
    print("3. Rebuild graph with corrected orientation")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Flip in-vivo volume along Y-axis to correct mirror artifact",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Path to raw TIFF (default: JH302_1x_ch2.tiff in raw data)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: output/in_vivo_flip_corrected)'
    )
    parser.add_argument(
        '--axis', '-a',
        type=int,
        default=1,
        help='Axis to flip along (default: 1 for Y-axis in ZYX)'
    )
    parser.add_argument(
        '--channel',
        type=str,
        choices=['ch1', 'ch2'],
        default='ch2',
        help='Channel to process (default: ch2)'
    )
    parser.add_argument(
        '--magnification',
        type=str,
        choices=['1x', '2x'],
        default='1x',
        help='Magnification level (default: 1x)'
    )

    args = parser.parse_args()

    # Build input path from channel/magnification if not specified
    input_path = args.input
    if input_path is None:
        input_path = DEFAULT_INVIVO_DIR / f"JH302_{args.magnification}_{args.channel}.tiff"
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        exit(1)

    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / "in_vivo_flip_corrected"

    flip_invivo_stack(input_path, output_dir, axis=args.axis)


if __name__ == '__main__':
    main()
