"""
Local configuration - COPY THIS FILE to local_config.py and fill in your paths.

    cp local_config.example.py local_config.py

local_config.py is TRACKED in this repo as of 81dc5d8 — it is no longer
gitignored, so paths in it travel between machines and every edit shows as a
diff. Keep credentials out of it.
"""

# ---------------------------------------------------------------------
# ANALYSIS_ROOT — one folder holding every derived output for a subject:
#
#   <ANALYSIS_ROOT>/preprocessing/   BARseq preprocessing outputs
#   <ANALYSIS_ROOT>/alignment/       <subject>_graph.db
#   <ANALYSIS_ROOT>/cellpose/        *_seg.npy, sweep_in_vivo_*, combo_in_vivo_*
#
# Set it per subject and the rest follows; nothing derived lands in the data
# tree. Absolute path required. Blank keeps every consumer on its older
# data-tree-adjacent default, so existing subjects need no change.
# Explicit OUTPUT_ROOT / GRAPH_PATH below still override it.
# Example:
#   "C:/Users/David/lab_local/projects/cell_type/analysis/BY95"
# ---------------------------------------------------------------------
ANALYSIS_ROOT = ""

# Path to raw BARseq data (Batch3_JH302)
# Examples:
#   Windows:  "C:/Users/Li Lab/Documents/Data_ALM_cell_type_transcriptom/Batch3_JH302"
#   Mac:      "/Volumes/home/lab/raw_data/Data_ALM_cell_type_transcriptom/Batch3_JH302"
#   Linux:    "/home/yourname/lab/raw_data/Data_ALM_cell_type_transcriptom/Batch3_JH302"
DATA_ROOT = ""

# Path to the preprocessing output directory. Override only — leave BLANK and
# it resolves to <ANALYSIS_ROOT>/preprocessing/, or, when ANALYSIS_ROOT is
# blank too, to <DATA_ROOT>/preprocessing/ (the older co-located layout).
# Consumed only by preprocessing; the alignment graph builder ignores it.
OUTPUT_ROOT = ""

# ---------------------------------------------------------------------
# Per-brain thresholds — REQUIRED. Unset is a hard error, like SCOPE.
#
# Read by preprocessing_config, which every pipeline script imports from, so
# these are the single place they are set and no command-line flag is needed for
# a run to reproduce them. 0 is a valid value, which is why unset is None rather
# than 0.
#
# QC_MIN_READS / QC_MIN_GENES — measure them for a new brain with
# check_qc_metrics.py <DATA_ROOT> --no-reported, which sweeps reads/genes and
# prints the pass rate at each pair; --lab-reads / --lab-genes checks a candidate
# against a pass rate the lab reports. Never carry another brain's numbers over.
# BY95's cell-typing QC is 20 / 5 (26.4% pass). 0 / 0 leaves marker detection
# ungated on transcriptome quality — a different question, not a wrong answer.
#
# ALIGN_MIN_ROLONIES — mScarlet rolony floor for a cell to be drawn into the
# alignment TIF. A registration knob, not an analysis threshold; pick it by eye
# with check_rolony_cutoff.py. It is the "ge" in SUBSLICE_DIR.
# ---------------------------------------------------------------------
QC_MIN_READS = None
QC_MIN_GENES = None
ALIGN_MIN_ROLONIES = None

# Path to the per-FOV raw directory (the one holding MAX_Pos*_*_* folders, each
# with alignedn2vhyb01.tif + cellmask.mat).
# Leave BLANK to auto-detect: <DATA_ROOT>/hyb, then <DATA_ROOT>/hyb_raw_files.
# Set an absolute path only when the folder is named something else or lives
# outside DATA_ROOT. Consumed only by preprocessing.
#   JH302:                      <DATA_ROOT>/hyb             (auto-detected)
#   BY95 allen_transcriptomics: <DATA_ROOT>/hyb_raw_files   (auto-detected)
HYB_ROOT = ""

# ---------------------------------------------------------------------
# Graph inputs — the graph builder adds nodes for the paths that are set.
#
# Each 2P volume has TWO channels: red (sparsely-labelled, used for alignment
# fitting + Cellpose segmentation) and green (signal of interest). The two
# channels are co-registered in hardware and share a single fitted transform —
# they enter the graph as sibling nodes joined by a `castalign.base.Identity`
# edge. Rigid + nonlinear edges are fitted only on the `_red ↔ _red` pair.
#
# Rules:
#   blank ("")                            → skip that node
#   set but file/dir doesn't exist        → hard error (catches typos)
#   set and exists                        → add to graph
#   GREEN set without RED for same volume → hard error (would dangle Identity)
#
# At least one of the four 2P paths (or SUBSLICE_DIR) must be set.
# ---------------------------------------------------------------------

# Path to ex-vivo block RED channel (3D 2-photon volume, .tif/.tiff)
# Sparsely-labelled alignment channel. This is the 2P image of the tissue
# block before slicing.
# Examples:
#   Windows:  "C:/Users/Li Lab/Documents/data/gad2 by94 red exvivo max proj.tif"
#   Mac:      "/Users/yourname/data/gad2 by94 red exvivo max proj.tif"
BLOCK_STACK_PATH_RED = ""

# Path to ex-vivo block GREEN channel (signal of interest, optional).
# Same voxel grid as the red channel (co-registered in hardware).
# Leave blank if you don't have a green volume yet.
BLOCK_STACK_PATH_GREEN = ""

# Path to in-vivo RED channel 2-photon stack (.tif/.tiff)
# Sparsely-labelled alignment channel.
# - Li lab data: typically the preprocessed/flipped output, e.g.
#       OUTPUT_ROOT/in_vivo_flip_corrected/JH302_1x_ch2_flipped.tiff
# - Huang lab data: typically the raw max-projection TIFF (no preprocessing needed)
# Examples:
#   Windows:  "C:/Users/Li Lab/Documents/output/in_vivo_flip_corrected/JH302_1x_ch2_flipped.tiff"
#   Mac:      "/Users/yourname/data/gad2 by94 red invivo max proj.tif"
#   Linux:    "/home/yourname/lab/output/in_vivo_flip_corrected/JH302_1x_ch2_flipped.tiff"
INVIVO_PATH_RED = ""

# Path to in-vivo GREEN channel 2-photon stack (signal of interest, optional).
# Same voxel grid as the red channel (co-registered in hardware).
# Leave blank if you don't have a green volume yet.
INVIVO_PATH_GREEN = ""

# Directory of BARseq subslice images the graph builder ingests.
# Leave blank ("") for 2P-only alignment (no BARseq data).
#
# PREFERRED: a RELATIVE path, which names only the trailing folder and inherits
# everything above it from preprocessing_config, so a rename of the output dirs
# does not reach this file. It is resolved against BOTH output roots:
#   SUBSLICE_DIR = "qc20_5_ge1"                   # step 6 ALIGN tifs
#   -> <OUTPUT_ROOT>/subslice_align/qc20_5_ge1
#   SUBSLICE_DIR = "threshold_0.00_cellmask_0.50" # step 4 overlays
#   -> <OUTPUT_ROOT>/mScarlet_cellmask_subslice/threshold_0.00_cellmask_0.50
# The ALIGN tifs are what the builder prefers and what it fits on — binary and
# marker-only. The step 4 overlay is an RGB display figure and is the fallback.
# The qc/ge numbers are QC_MIN_READS, QC_MIN_GENES and ALIGN_MIN_ROLONIES; each
# combination writes its own folder, so this line selects which one is ingested.
# A wrong folder name lists the ones that do exist. An absolute path is still
# used verbatim, for subslices preprocessing did not write.
SUBSLICE_DIR = ""

# Path to the alignment graph file (.db) — used by both the graph builder and
# the notebook. Leave BLANK to auto-derive: <ANALYSIS_ROOT>/alignment/
# <subject>_graph.db when ANALYSIS_ROOT is set, else
# <2P TIFF parent>/alignment/<folder name>_graph.db.
# Examples:
#   Windows:  "C:/Users/Li Lab/Documents/output/linestuffup_output/castalign_test.db"
#   Mac:      "/Volumes/home/lab/output/linestuffup_output/castalign_test.db"
#   Linux:    "/home/yourname/lab/output/linestuffup_output/castalign_test.db"
GRAPH_PATH = ""

# ---------------------------------------------------------------------
# SCOPE — the microscope that acquired this subject's data.
#
# One declaration, and it is the source of truth for pixel size everywhere:
#   preprocessing   BARseq resample factor (xy_um_per_px / 0.32)
#   graph builder   spacing on every 2P node and every BARseq subslice node
#   validate_mnn    µm scale for centroid distances
#
# Values (defined in scope_profiles.py at the project root):
#   "li_lab"           2.34   µm/px XY, 1.0 µm Z, 1200 µm FOV    JH302 (retired)
#   "huang_lab"        1.1000 µm/px XY, 1.0 µm Z, 563.2 µm FOV   BY95
#
# huang_lab was 0.3910 µm/px over a 200.19 µm field until 2026-08-30, when the
# stacks' own XResolution tag was read and said 1.1000. "huang_lab_566um"
# (1.1055, by84/by94/by89) went with it: 563.2 and 566.08 µm are 0.51% apart,
# so the "two zooms ~2.83x apart" story was an artifact of the wrong 200.19 and
# the two keys collided inside the 5% tolerance that identify_scope uses.
# Those three subjects now hard-error until their own tags are read.
#
# Required. Blank is a hard error. TIFF resolution metadata, where present, is
# read only to CHECK this line: a file whose pixel size disagrees with SCOPE
# stops the run rather than silently overriding it.
# ---------------------------------------------------------------------
SCOPE = ""
