#!/usr/bin/env python3
"""
Preprocessing Pipeline Runner

Runs the complete mScarlet preprocessing pipeline:
1. identify_mscarlet_subslices.py - Find FOV clusters with mScarlet+ cells
2. stitch_subslices.py - Stitch FOVs into composite images
3. downsample_subslices_cellmask.py - Downsample to match in-vivo resolution
4. generate_mscarlet_cellmask_subslice.py - Create mScarlet overlays
5. interactive_mscarlet_threshold_cellmask_subslice.py - Generate figures

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
        'script': 'downsample_subslices_cellmask.py',
        'description': 'Downsample all channels to the 2P in-plane pixel size',
    },
    {
        'name': 'Generate Overlays',
        'script': 'generate_mscarlet_cellmask_subslice.py',
        'description': 'Create mScarlet cell overlays on cellmask',
        'takes_threshold': True,
    },
    {
        'name': 'Generate Figures',
        'script': 'interactive_mscarlet_threshold_cellmask_subslice.py',
        'description': 'Generate visualization figures',
        'takes_threshold': True,
    },
    {
        # Its slice flag is plural and variadic, and it has no --threshold:
        # the gates are QC + rolony count, not the overlay's display threshold.
        'name': 'Generate Alignment TIFs',
        'script': 'generate_alignment_tif.py',
        'description': 'Binary marker-only images the graph builder aligns on',
        'slice_flag': '--slices',
    },
]

# Orientation assignment is deliberately NOT a step here. It is interactive --
# it needs a display and four answers by hand -- so it cannot run in this
# subprocess chain. It is still part of preprocessing, and it runs LAST: after
# this pipeline, immediately before subslice_graph_builder.py ingests the
# alignment TIFs. See the reminder printed at the end of a full run.

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
    parser.add_argument('--start-from', type=int, default=1,
                        help=f'Start from step N (1-{len(STEPS)})')
    parser.add_argument('--stop-after', type=int, default=len(STEPS),
                        help=f'Stop after step N (1-{len(STEPS)})')
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
        python_exe = sys.executable

    # The slice selector, if any. Its flag differs per script, so it is applied
    # per step rather than shared.
    target_slice = '22' if args.test else (str(args.slice) if args.slice else None)
    extra_args = ['--slice', target_slice] if target_slice else []

    # Threshold args, forwarded only to the steps that declare they take it
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

        step_args = []
        if target_slice:
            step_args += [step.get('slice_flag', '--slice'), target_slice]
        if step.get('takes_threshold') and args.threshold > 0:
            step_args += threshold_args

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
        from preprocessing_config import (
            OUTPUT_ROOT, SUBSLICE_DEFINITIONS_DIR, HYB_STITCHED_DIR,
            HYB_DOWNSAMPLED_DIR, MSCARLET_CELLMASK_DIR, MSCARLET_INTERACTIVE_DIR,
            SUBSLICE_ALIGN_DIR,
        )
        print("Status: COMPLETED successfully")
        print(f"\nOutput locations:")
        print(f"  Subslice definitions: {SUBSLICE_DEFINITIONS_DIR}")
        print(f"  Stitched images: {HYB_STITCHED_DIR}")
        print(f"  Downsampled: {HYB_DOWNSAMPLED_DIR}")
        print(f"  Overlays: {MSCARLET_CELLMASK_DIR}")
        print(f"  Figures: {MSCARLET_INTERACTIVE_DIR}")
        print(f"  Alignment TIFs: {SUBSLICE_ALIGN_DIR}")
        print("\nNext, in this order:")
        print("  1. python preprocessing/assign_orientation.py --modality barseq_subslice")
        print("     Last preprocessing act for this modality. Assign on the image the")
        print("     graph will ingest -- any flip introduced after this invalidates the code.")
        print("  2. python alignment/subslice_graph_builder.py")
        print("     Stamps the orientation onto every node it adds. A node added before")
        print("     the code exists carries none, and the handedness guard cannot fire.")


if __name__ == '__main__':
    main()
