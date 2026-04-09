"""
MATLAB to Python Pipeline Migration
====================================

This package ports the MATLAB preprocessing scripts for the LineStuffUp alignment pipeline
to Python with exact fidelity. No shortcuts, no redesigns.

Critical: Preserve cell position tracking throughout (for barseq gene expression linkage).

Pipeline execution order:
    1. identify_mscarlet_subslices.py -> subslice_definitions.mat
    2. stitch_subslices.py -> HYB_subslice_stitched_tif/
    3. downsample_subslices_cellmask_anisotropic.py -> downsampled cellmasks
    4. generate_mscarlet_cellmask_subslice_anisotropic.py -> mScarlet overlays
    5. interactive_mscarlet_threshold_cellmask_subslice_anisotropic.py -> interactive viewer
    6. generate_mscarlet_overlay_labelled.py -> labelled FOV overlays

Utility script (anytime use):
    - edit_subslice_definitions.py -> modify subslice_definitions.mat

Constants:
    - Coordinates are always (z, y, x) for CASTalign compatibility
    - MATLAB 1-indexed -> Python 0-indexed
    - mScarlet column 114 in MATLAB -> index 113 in Python
"""

__version__ = "1.0.0"
