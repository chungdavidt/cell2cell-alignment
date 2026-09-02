"""
Configuration for preprocessing pipeline.

User must configure DATA_ROOT, SCOPE, QC_MIN_READS, QC_MIN_GENES and
ALIGN_MIN_ROLONIES in local_config.py; OUTPUT_ROOT is optional there.
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
    import local_config
except ImportError:
    raise ImportError(
        "local_config.py not found.\n"
        "Copy local_config.example.py to local_config.py and fill in your paths:\n"
        "    cp local_config.example.py local_config.py"
    )

DATA_ROOT = getattr(local_config, "DATA_ROOT", "")
if not DATA_ROOT:
    raise ValueError(
        "DATA_ROOT is not set in local_config.py.\n"
        "Set it to the dataset folder containing filt_neurons.mat and the hyb/ "
        "(or hyb_raw_files/) per-FOV directory."
    )

# OUTPUT_ROOT precedence:
#   1. explicit OUTPUT_ROOT in local_config.py
#   2. <ANALYSIS_ROOT>/preprocessing/  — the per-subject analysis tree, which
#      keeps derived outputs out of the data tree (preferred)
#   3. <DATA_ROOT>/preprocessing/ — older co-located layout, for subjects with
#      no ANALYSIS_ROOT set
# OUTPUT_ROOT is consumed only by preprocessing.
from analysis_paths import analysis_subdir, PREPROCESSING_SUBDIR

_ANALYSIS_PREPROCESSING = analysis_subdir(PREPROCESSING_SUBDIR)
OUTPUT_ROOT = (
    getattr(local_config, "OUTPUT_ROOT", "")
    or (str(_ANALYSIS_PREPROCESSING) if _ANALYSIS_PREPROCESSING is not None else "")
    or os.path.join(DATA_ROOT, "preprocessing")
)

# Guard: a relative OUTPUT_ROOT would silently resolve against the run cwd
# (run_pipeline.py runs each step with cwd=preprocessing/), dumping outputs into
# the code folder. Require absolute so that footgun can never recur.
if not os.path.isabs(OUTPUT_ROOT):
    raise ValueError(
        f"OUTPUT_ROOT resolved to a non-absolute path: {OUTPUT_ROOT!r}\n"
        "Set DATA_ROOT (and OUTPUT_ROOT, if you set it explicitly) to absolute "
        "paths in local_config.py."
    )

# =============================================================================
# DERIVED PATHS - Automatically computed from DATA_ROOT and OUTPUT_ROOT
# =============================================================================

# Input data paths
#
# HYB_ROOT: the per-FOV raw directory. Its name is not stable across datasets
# (JH302 = "hyb", BY95 allen_transcriptomics = "hyb_raw_files"), so an explicit
# HYB_ROOT in local_config.py wins; blank auto-detects the first name in
# HYB_DIRNAME_CANDIDATES that exists under DATA_ROOT, falling back to the first
# candidate so validate_paths() reports a concrete missing path.
HYB_DIRNAME_CANDIDATES = ("hyb", "hyb_raw_files")


def _resolve_hyb_root():
    explicit = getattr(local_config, "HYB_ROOT", "")
    if explicit:
        if not os.path.isabs(explicit):
            raise ValueError(
                f"HYB_ROOT in local_config.py must be absolute: {explicit!r}"
            )
        return explicit
    for name in HYB_DIRNAME_CANDIDATES:
        candidate = os.path.join(DATA_ROOT, name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(DATA_ROOT, HYB_DIRNAME_CANDIDATES[0])


HYB_ROOT = _resolve_hyb_root()
FILT_NEURONS_PATH = os.path.join(DATA_ROOT, "filt_neurons.mat")

# Output subdirectories
SUBSLICE_DEFINITIONS_DIR = os.path.join(OUTPUT_ROOT, "subslice_definitions")
HYB_CHANNELS_DIR = os.path.join(OUTPUT_ROOT, "hyb_channels")
HYB_STITCHED_DIR = os.path.join(OUTPUT_ROOT, "HYB_subslice_stitched_tif")
HYB_DOWNSAMPLED_DIR = os.path.join(OUTPUT_ROOT, "HYB_subslice_stitched_tif_downsampled_micronwise")
# Whole sections, every FOV that has a cell in the slice. Written by
# stitch_slices.py, outside the 7-step run. The name carries no "subslice"
# token so the step 3 and graph-builder globs cannot pick these up.
HYB_SLICE_STITCHED_DIR = os.path.join(OUTPUT_ROOT, "HYB_slice_stitched_tif")
MSCARLET_CELLMASK_DIR = os.path.join(OUTPUT_ROOT, "mScarlet_cellmask_subslice")
# Binary marker-only images the graph aligns on. One subdirectory per gate
# combination, named by generate_alignment_tif.py.
SUBSLICE_ALIGN_DIR = os.path.join(OUTPUT_ROOT, "subslice_align")
MSCARLET_INTERACTIVE_DIR = os.path.join(OUTPUT_ROOT, "mScarlet_cellmask_interactive_subslice")
MSCARLET_LABELLED_DIR = os.path.join(OUTPUT_ROOT, "mScarlet_overlay_dapi_labelled")

# Specific output files
SUBSLICE_DEFINITIONS_FILE = os.path.join(SUBSLICE_DEFINITIONS_DIR, "subslice_definitions.mat")

# =============================================================================
# RESOLUTION CONSTANTS
# =============================================================================

# Pixel sizes come from the scope declared as SCOPE in local_config.py. The
# profile table lives in scope_profiles.py at the project root — stdlib-only, so
# preprocessing, alignment and the cellpose venv all read the same numbers.
#
# Sections are cut in the 2P imaging plane, so both BARseq in-plane axes map to
# 2P XY and ONE factor covers both. (Sections used to be cut coronal against an
# axial 2P, which is why this was two factors -- 7.3125 for X, 3.125 for Y --
# off Li lab optics that belong to the retired JH302.)
from scope_profiles import (
    EXVIVO_UM_PER_PX,
    get_profile,
    resolve_scope_name,
)

SCOPE = resolve_scope_name(getattr(local_config, "SCOPE", ""))
SCOPE_PROFILE = get_profile(SCOPE)          # raises if blank or unknown

# In-plane pixel size the BARseq images are resampled to, µm/px.
TARGET_XY_UM_PER_PX = SCOPE_PROFILE['xy_um_per_px']

# Isotropic downsample factor: new_size = original_size / DOWNSAMPLE_XY.
# huang_lab (1.1000) -> 3.4375x, li_lab (2.34) -> 7.3125x.
DOWNSAMPLE_XY = TARGET_XY_UM_PER_PX / EXVIVO_UM_PER_PX

# =============================================================================
# DATA CONSTANTS
# =============================================================================

# Standard FOV size in pixels
FOV_SIZE = 3200

# mScarlet column index (0-indexed in Python, was 114 in MATLAB which is 1-indexed)
# Marker columns in expmat.
#
# The gene list MISLABELS these slots on this panel: the last entries are hyb
# readout channels, not barcoded genes (their second column holds channel
# numbers 1/2/4 instead of a 7-mer), and they still carry template gene names.
# Index 113 is labelled "Slc30a3" but is mScarlet, index 111 is labelled
# "Slc17a7" but is GCaMP — confirmed by the lab's own Gen_mScarlet_plots.m
# (expmat(:,114)) and Gen_GCaMP_plots.m (expmat(:,112)), MATLAB 1-indexed.
#
# So look-up by name is OFF by default: blank MSCARLET_GENE_NAME means "trust
# the index". Set it to a real name only for a panel that labels its marker
# honestly, and resolve_marker_column will then raise rather than read the
# wrong column if that name is missing.
MSCARLET_GENE_NAME = ""
MSCARLET_COLUMN_INDEX = 113  # Python 0-indexed (MATLAB 114)
GCAMP_COLUMN_INDEX = 111     # Python 0-indexed (MATLAB 112)

# QC thresholds and the alignment rolony floor. ALL THREE ARE PER-BRAIN and are
# declared in local_config.py beside SCOPE, not here -- switching subjects then
# touches one file. Every consumer still imports them from this module, so
# nothing downstream changed when they moved (2026-08-27).
#
# 20/5 is the lab's cell-typing QC: their reported BY95 pass rate of 26.4%
# reproduces exactly at those values, which check_qc_metrics.py verifies. The
# same lab's marker plotting deliberately runs ungated at 0/0, so that marker
# detection does not depend on transcriptome quality -- two thresholds answering
# two different questions. This pipeline runs at the cell-typing pair, so a cell
# only counts as marker+ if its transcriptome is trustworthy; on BY95 that
# drops ~74% of rows before the marker column is read.
#
# Required, and unset is a hard error rather than a default, for the same reason
# SCOPE is: silently inheriting the previous brain's numbers is the exact
# failure these are meant to prevent. 0 is a legitimate value, so the unset
# sentinel is None, not 0.


def _require_threshold(name: str, what: str, how: str) -> int:
    value = getattr(local_config, name, None)
    if value is None:
        raise ValueError(
            f"\n{'='*60}\n"
            f"{name} is not set in local_config.py.\n\n"
            f"{what}\n\n"
            f"It is per-brain, so it has no default -- inheriting the previous\n"
            f"brain's number silently is the failure this guard exists to stop.\n"
            f"{how}\n"
            f"{'='*60}"
        )
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"{name} in local_config.py must be a non-negative int, got {value!r}"
        )
    return value


_QC_SOURCE = (
    "Measure it: run check_qc_metrics.py <DATA_ROOT> --no-reported and read the\n"
    "reads/genes sweep. If the lab reports a pass rate for this brain, pass it as\n"
    "--lab-reads / --lab-genes and the tool says whether a candidate pair\n"
    "reproduces it. 0 is a valid answer -- it leaves marker detection ungated on\n"
    "transcriptome quality, which is a different question, not a wrong one."
)

QC_MIN_READS = _require_threshold(
    "QC_MIN_READS", "Total-reads floor for a cell to pass QC.", _QC_SOURCE)
QC_MIN_GENES = _require_threshold(
    "QC_MIN_GENES", "Distinct-genes floor for a cell to pass QC.", _QC_SOURCE)

# mScarlet rolony floor for a cell to be DRAWN into the alignment TIF.
#
# A REGISTRATION knob, not an analysis threshold. The fit is stored as frame
# parameters on the grid, so it applies to cells that were never drawn --
# raising it changes what you can see while aligning and nothing else. Pick it
# by eye with check_rolony_cutoff.py, which draws the raw mScarlet TIF beside
# the painted masks, then record it in local_config.py so a pipeline run
# reproduces it without a command-line flag.
#
# generate_alignment_tif.py names its output folder after the gates
# (qc{reads}_{genes}_ge{n}), so changing this writes somewhere new instead of
# overwriting the previous choice.
ALIGN_MIN_ROLONIES = _require_threshold(
    "ALIGN_MIN_ROLONIES",
    "mScarlet rolony floor for a cell to be drawn into the alignment TIF.",
    "Pick it by eye: run check_rolony_cutoff.py, which draws the raw mScarlet\n"
    "TIF beside the painted masks, and read off where the kept cells stop\n"
    "tracking real signal. 1 draws every marker+ cell that passes QC.")

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
        errors.append(
            f"hyb directory not found: {HYB_ROOT}\n"
            f"    (searched {list(HYB_DIRNAME_CANDIDATES)} under DATA_ROOT; "
            f"set HYB_ROOT in local_config.py for a different name)"
        )

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
        return f"threshold_{threshold:.2f}_cellmask_{cellmask_intensity:.2f}"
    else:
        return f"threshold_{threshold:.2f}_downsampled"


def mscarlet_subslice_dir(threshold, cellmask_intensity=None):
    """
    Full path to the overlay output for one threshold pair.

    The one place this composition lives. Step 4 writes here, and a relative
    SUBSLICE_DIR in local_config.py resolves against MSCARLET_CELLMASK_DIR
    (analysis_paths.resolve_subslice_dir), so the graph builder inherits the
    layout instead of repeating it as a hand-typed absolute path.
    """
    return os.path.join(
        MSCARLET_CELLMASK_DIR, get_threshold_folder(threshold, cellmask_intensity)
    )
