"""
MATLAB to Python Pipeline Migration
====================================

This package ports the MATLAB preprocessing scripts for the LineStuffUp alignment pipeline
to Python with exact fidelity. No shortcuts, no redesigns.

Critical: Preserve cell position tracking throughout (for barseq gene expression linkage).

Pipeline execution order:
    1. identify_mscarlet_subslices.py -> subslice_definitions.mat
    2. stitch_subslices.py -> HYB_subslice_stitched_tif/
    3. downsample_subslices_cellmask.py -> downsampled cellmasks
    4. generate_mscarlet_cellmask_subslice.py -> mScarlet overlays
    5. interactive_mscarlet_threshold_cellmask_subslice.py -> batch figures (Agg,
       not interactive despite the name)
    6. generate_alignment_tif.py -> binary marker-only tifs the graph builder fits on

generate_mscarlet_overlay_labelled.py is NOT step 6 -- it is an optional inspection
tool driven by edit_subslice_definitions.py.s

Utility script (anytime use):
    - edit_subslice_definitions.py -> modify subslice_definitions.mat

Constants:
    - Coordinates are always (z, y, x) for CASTalign compatibility
    - MATLAB 1-indexed -> Python 0-indexed
    - mScarlet column 114 in MATLAB -> index 113 in Python
"""

__version__ = "1.0.0"
