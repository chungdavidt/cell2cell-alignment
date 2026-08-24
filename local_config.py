"""
Local configuration — machine-specific paths. Subject: BY95.
"""

# One root for every derived output: preprocessing/, alignment/, cellpose/,
# plus orientation.json. Its folder name is the subject label ("BY95").
ANALYSIS_ROOT = r"C:\Users\David\lab_local\projects\cell_type\analysis\BY95"

# Path to raw BARseq data
DATA_ROOT = r"C:\Users\David\lab_local\projects\cell_type\data\050526 BY95\allen_transcriptomics\BY95"

# Blank -> <ANALYSIS_ROOT>/preprocessing
OUTPUT_ROOT = r""

# Per-FOV raw directory (MAX_Pos*_*_* folders). Blank -> auto-detect;
# BY95 has <DATA_ROOT>/hyb_raw_files.
HYB_ROOT = r""

# Threshold folder from step 4 (generate_mscarlet_cellmask_subslice.py), named
# relative so the directories above it are inherited from preprocessing_config.
# 0.00/0.50 are that script's defaults; change if you export another pair.
SUBSLICE_DIR = r"threshold_0.00_cellmask_0.50"

# TODO(David, on Windows): fill in the four BY95 2-photon TIFFs from
# "050526 BY95\". Blank = that node is skipped; the paths below were JH302's
# (retired 2026-08-22) and were removed rather than carried over.
# Path to ex-vivo block RED channel (3D 2-photon volume, .tif/.tiff)
BLOCK_STACK_PATH_RED = r""
# Path to ex-vivo block GREEN channel (signal of interest, optional)
BLOCK_STACK_PATH_GREEN = r""

# Path to in-vivo RED channel 2-photon stack (.tif/.tiff)
INVIVO_PATH_RED = r""
# Path to in-vivo GREEN channel 2-photon stack (signal of interest, optional)
INVIVO_PATH_GREEN = r""

# Blank -> <ANALYSIS_ROOT>/alignment/BY95_graph.db. The old reason for pinning
# this (a date-prefixed data folder doubling into the filename) is gone now
# that the path comes from ANALYSIS_ROOT.
GRAPH_PATH = r""

# Microscope that acquired this subject's data; profiles in scope_profiles.py.
# Source of truth for every pixel size — TIFF XResolution is read only to check
# it, and a disagreement is a hard error.
#   "huang_lab"        0.3910 µm/px XY, 1.0 µm Z   BY95
#   "huang_lab_566um"  1.1055 µm/px XY, 2.0 µm Z   by84, by94, by89
SCOPE = "huang_lab"
