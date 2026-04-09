#!/usr/bin/env python3
"""
Preprocessing Pipeline Runner

Runs the complete mScarlet preprocessing pipeline:
1. identify_mscarlet_subslices.py - Find FOV clusters with mScarlet+ cells
2. stitch_subslices.py - Stitch FOVs into composite images
3. downsample_subslices_cellmask_anisotropic.py - Downsample to match in-vivo resolution
4. generate_mscarlet_cellmask_subslice_anisotropic.py - Create mScarlet overlays
5. interactive_mscarlet_threshold_cellmask_subslice_anisotropic.py - Generate figures

Usage:
    python run_pipeline.py                    # Run full pipeline (all slices)
    python run_pipeline.py --slice 22         # Run on specific slice only
    python run_pipeline.py --test             # Test mode (slice 22 only)
    python run_pipeline.py --start-from 3     # Resume from step 3

Author: DTC
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


# Pipeline steps
STEPS = [
    {
        'name': 'Identify Subslices',
        'script': 'identify_mscarlet_subslices.py',
        'description': 'Find FOV clusters with mScarlet+ cells',
    },
    {
        'name': 'Stitch FOVs',
        'script': 'stitch_subslices.py',
        'description': 'Stitch FOVs into composite images (DAPI, GCAMP, MSCARLET, CELLMASK)',
    },
    {
        'name': 'Downsample All Channels',
        'script': 'downsample_subslices_cellmask_anisotropic.py',
        'description': 'Downsample all channels to match in-vivo resolution (anisotropic)',
    },
    {
        'name': 'Generate Overlays',
        'script': 'generate_mscarlet_cellmask_subslice_anisotropic.py',
        'description': 'Create mScarlet cell overlays on cellmask',
    },
    {
        'name': 'Generate Figures',
        'script': 'interactive_mscarlet_threshold_cellmask_subslice_anisotropic.py',
        'description': 'Generate visualization figures',
    },
]

# Optional steps (not in main pipeline, run separately)
OPTIONAL_STEPS = {
    'refine': {
        'name': 'Refine by Threshold',
        'script': 'refine_subslices_by_threshold.py',
        'description': 'Filter FOVs by mScarlet intensity threshold',
        'requires_args': ['--threshold'],
    },
    'align': {
        'name': 'Create Aligned Volume',
        'script': 'create_ex_vivo_volume.py',
        'description': 'Create 3D aligned brain stack from 2D slices',
        'requires_args': ['--input', '--alignment'],
    },
}


def run_step(step_num, step_info, python_exe, script_dir, extra_args=None, dry_run=False):
    """Run a single pipeline step."""
    script_path = script_dir / step_info['script']

    if not script_path.exists():
        print(f"  ERROR: Script not found: {script_path}")
        return False

    cmd = [python_exe, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {step_info['name']}")
    print(f"{'='*60}")
    print(f"Script: {step_info['script']}")
    print(f"Description: {step_info['description']}")
    if extra_args:
        print(f"Arguments: {' '.join(extra_args)}")
    print()

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            check=False,
            text=True,
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ Step {step_num} completed in {elapsed:.1f} seconds")
            return True
        else:
            print(f"\n✗ Step {step_num} FAILED (exit code {result.returncode})")
            return False

    except Exception as e:
        print(f"\n✗ Step {step_num} ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run the complete mScarlet preprocessing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Full pipeline, all slices
  python run_pipeline.py --slice 22         # Specific slice only
  python run_pipeline.py --test             # Test mode (slice 22)
  python run_pipeline.py --start-from 3     # Resume from step 3
  python run_pipeline.py --dry-run          # Show commands without running
"""
    )
    parser.add_argument('--slice', '-s', type=int, help='Process specific slice only')
    parser.add_argument('--test', '-t', action='store_true', help='Test mode: process slice 22 only')
    parser.add_argument('--start-from', type=int, default=1, help='Start from step N (1-5)')
    parser.add_argument('--stop-after', type=int, default=5, help='Stop after step N (1-5)')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without executing')
    parser.add_argument('--threshold', type=float, default=0.0, help='mScarlet threshold (for steps 4-5)')
    parser.add_argument('--python', type=str, help='Python executable path')

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent.resolve()

    # Find Python executable
    if args.python:
        python_exe = args.python
    else:
        # Try to use the venv python
        venv_python = Path('/home/dtc/lab/venvs/preprocessing/bin/python')
        if venv_python.exists():
            python_exe = str(venv_python)
        else:
            python_exe = sys.executable

    # Build extra arguments for scripts
    extra_args = []
    if args.test:
        extra_args.append('--slice')
        extra_args.append('22')
    elif args.slice:
        extra_args.append('--slice')
        extra_args.append(str(args.slice))

    # Threshold args for steps 4-5
    threshold_args = ['--threshold', str(args.threshold)]

    # Print header
    print("="*60)
    print("mSCARLET PREPROCESSING PIPELINE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {python_exe}")
    print(f"Script directory: {script_dir}")
    if args.test:
        print("Mode: TEST (slice 22 only)")
    elif args.slice:
        print(f"Mode: SINGLE SLICE ({args.slice})")
    else:
        print("Mode: FULL (all slices)")
    print(f"Steps: {args.start_from} to {args.stop_after}")
    if args.dry_run:
        print("DRY RUN: Commands will be shown but not executed")

    # Run pipeline
    total_start = time.time()
    failed_step = None

    for i, step in enumerate(STEPS, 1):
        if i < args.start_from:
            print(f"\nSkipping step {i}: {step['name']}")
            continue
        if i > args.stop_after:
            print(f"\nStopping before step {i}: {step['name']}")
            break

        # Add threshold args for steps 4-5
        step_args = extra_args.copy()
        if i >= 4 and args.threshold > 0:
            step_args.extend(threshold_args)

        # For step 5, limit to first figure in test mode
        if i == 5 and args.test:
            step_args.extend(['--first', '1'])

        success = run_step(i, step, python_exe, script_dir, step_args, args.dry_run)

        if not success:
            failed_step = i
            print(f"\nPipeline stopped at step {i}")
            print(f"To resume: python run_pipeline.py --start-from {i} {' '.join(extra_args)}")
            break

    # Summary
    total_elapsed = time.time() - total_start

    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")

    if failed_step:
        print(f"Status: FAILED at step {failed_step}")
        sys.exit(1)
    else:
        print("Status: COMPLETED successfully")
        print(f"\nOutput locations:")
        print(f"  Subslice definitions: /home/dtc/lab/output/subslice_definitions/")
        print(f"  Stitched images: /home/dtc/lab/output/HYB_subslice_stitched_tif/")
        print(f"  Downsampled: /home/dtc/lab/output/HYB_subslice_stitched_tif_downsampled_micronwise_anisotropic/")
        print(f"  Overlays: /home/dtc/lab/output/mScarlet_cellmask_subslice/")
        print(f"  Figures: /home/dtc/lab/output/mScarlet_cellmask_interactive_subslice_anisotropic/")


if __name__ == '__main__':
    main()
