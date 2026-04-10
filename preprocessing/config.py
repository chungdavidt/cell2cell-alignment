"""
Configuration for preprocessing pipeline.

User must configure DATA_ROOT and OUTPUT_ROOT for their environment.
All other paths are derived from these.
"""

import os
import sys
from pathlib import Path

# =============================================================================
# USER CONFIGURATION - loaded from local_config.py at project root
# =============================================================================

# Add project root to path for local_config import
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from local_config import DATA_ROOT, OUTPUT_ROOT
except ImportError:
    raise ImportError(
        "local_config.py not found.\n"
        "Copy local_config.example.py to local_config.py and fill in your paths:\n"
        "    cp local_config.example.py local_config.py"
    )

# =============================================================================
# DERIVED PATHS - Automatically computed from DATA_ROOT and OUTPUT_ROOT
# =============================================================================

# Input data paths
HYB_ROOT = os.path.join(DATA_ROOT, "hyb")
FILT_NEURONS_PATH = os.path.join(DATA_ROOT, "filt_neurons.mat")

# Output subdirectories
SUBSLICE_DEFINITIONS_DIR = os.path.join(OUTPUT_ROOT, "subslice_definitions")
HYB_CHANNELS_DIR = os.path.join(OUTPUT_ROOT, "hyb_channels")
HYB_STITCHED_DIR = os.path.join(OUTPUT_ROOT, "HYB_subslice_stitched_tif")
HYB_DOWNSAMPLED_DIR = os.path.join(OUTPUT_ROOT, "HYB_subslice_stitched_tif_downsampled_micronwise_anisotropic")
MSCARLET_CELLMASK_DIR = os.path.join(OUTPUT_ROOT, "mScarlet_cellmask_subslice")
MSCARLET_INTERACTIVE_DIR = os.path.join(OUTPUT_ROOT, "mScarlet_cellmask_interactive_subslice_anisotropic")
MSCARLET_LABELLED_DIR = os.path.join(OUTPUT_ROOT, "mScarlet_overlay_dapi_labelled")

# Specific output files
SUBSLICE_DEFINITIONS_FILE = os.path.join(SUBSLICE_DEFINITIONS_DIR, "subslice_definitions.mat")

# =============================================================================
# RESOLUTION CONSTANTS
# =============================================================================

# Ex vivo resolution
EXVIVO_UM_PER_PX = 0.32  # micrometers per pixel

# In vivo resolution (after xrotate=90 transformation)
INVIVO_XY_UM_PER_PX = 2.34  # lateral-medial (X in vivo = X/Y in ex vivo)
INVIVO_Z_UM_PER_PX = 1.0    # dorsal-ventral (Z in vivo = Z in ex vivo, after rotation)

# Anisotropic downsample factors
# X: ex vivo -> in vivo X/Y resolution
DOWNSAMPLE_X = INVIVO_XY_UM_PER_PX / EXVIVO_UM_PER_PX  # 7.3125
# Y: ex vivo -> in vivo Z resolution
DOWNSAMPLE_Y = INVIVO_Z_UM_PER_PX / EXVIVO_UM_PER_PX   # 3.125

# =============================================================================
# DATA CONSTANTS
# =============================================================================

# Standard FOV size in pixels
FOV_SIZE = 3200

# mScarlet column index (0-indexed in Python, was 114 in MATLAB which is 1-indexed)
MSCARLET_COLUMN_INDEX = 113  # Python 0-indexed

# QC thresholds
QC_MIN_READS = 20
QC_MIN_GENES = 5

# =============================================================================
# VISUALIZATION CONSTANTS
# =============================================================================

# Cell mask overlay parameters
CELLMASK_BRIGHTNESS = 0.25
RED_OPACITY = 0.95
MSCARLET_BOOST = 1.2

# Label parameters
LABEL_FONT_SIZE = 48
LABEL_COLOR = "black"
LABEL_TEXT_COLOR = "yellow"
DAPI_BRIGHTNESS = 0.35

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_output_dirs():
    """Create all output directories if they don't exist."""
    dirs = [
        SUBSLICE_DEFINITIONS_DIR,
        HYB_CHANNELS_DIR,
        HYB_STITCHED_DIR,
        HYB_DOWNSAMPLED_DIR,
        MSCARLET_CELLMASK_DIR,
        MSCARLET_INTERACTIVE_DIR,
        MSCARLET_LABELLED_DIR,
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def validate_paths():
    """Validate that required input paths exist."""
    errors = []

    if not os.path.exists(DATA_ROOT):
        errors.append(f"DATA_ROOT not found: {DATA_ROOT}")

    if not os.path.exists(FILT_NEURONS_PATH):
        errors.append(f"filt_neurons.mat not found: {FILT_NEURONS_PATH}")

    if not os.path.exists(HYB_ROOT):
        errors.append(f"hyb directory not found: {HYB_ROOT}")

    if errors:
        raise FileNotFoundError(
            "Required paths not found. Please update config.py:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    return True


def get_threshold_folder(threshold, cellmask_intensity=None):
    """
    Generate threshold folder name matching MATLAB convention.

    Args:
        threshold: mScarlet threshold (0-1)
        cellmask_intensity: Optional cell mask brightness multiplier

    Returns:
        Folder name string
    """
    if cellmask_intensity is not None:
        return f"threshold_{threshold:.2f}_cellmask_{cellmask_intensity:.2f}_anisotropic"
    else:
        return f"threshold_{threshold:.2f}_downsampled"
