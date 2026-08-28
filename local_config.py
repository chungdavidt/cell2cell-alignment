"""
Local configuration — machine-specific paths. Subject: BY95.

TRACKED IN GIT since 81dc5d8, not gitignored. Paths here travel between the
WSL edit host and the Windows execution host, and every edit shows as a repo
diff. Do not put credentials in this file.

Blank is a convention, not an omission: for OUTPUT_ROOT, HYB_ROOT and
GRAPH_PATH it means "derive it", and each says below what it derives to. SCOPE
is the exception — it is authoritative, and blank is a hard error.

Switching subjects: ANALYSIS_ROOT, DATA_ROOT, SCOPE, the four 2P paths, and
SUBSLICE_DIR's cutoff. Nothing in the notebook or the graph builder changes.
"""

# ---------------------------------------------------------------------
# Subject
# ---------------------------------------------------------------------

# One root for every derived output — preprocessing/, alignment/, cellpose/,
# plus orientation.json at its top level. Its folder name is the subject label
# ("BY95"), which is what names the graph .db.
ANALYSIS_ROOT = r"C:\Users\David\lab_local\projects\cell_type\analysis\BY95"

# Microscope that acquired this subject's data; profiles in scope_profiles.py.
# Source of truth for every pixel size — the BARseq resample factor, node
# spacings, and validate_mnn's µm scale. TIFF XResolution is read only to check
# it, and a disagreement is a hard error.
#   "huang_lab"        0.3910 µm/px XY, 1.0 µm Z   BY95
#   "huang_lab_566um"  1.1055 µm/px XY, 2.0 µm Z   by84, by94, by89
SCOPE = "huang_lab"

# ---------------------------------------------------------------------
# Per-brain thresholds
#
# All three are read by preprocessing_config, which every pipeline script and
# probe imports from — so these lines are the single place they are set, and no
# command-line flag is needed to make a run reproduce them. Unset is a hard
# error, not a default: silently inheriting the previous brain's numbers is the
# failure they exist to prevent. 0 is a valid value.
# ---------------------------------------------------------------------

# QC floors for a cell to count at all. 20/5 is the lab's cell-typing QC — their
# reported BY95 pass rate of 26.4% reproduces exactly here, and it drops ~74% of
# rows before the marker column is read. Gates step 1 subslice selection, not
# just the overlays.
#
# Measure them for a new brain: check_qc_metrics.py <DATA_ROOT> --no-reported
# sweeps reads/genes and prints the pass rate at each pair; --lab-reads /
# --lab-genes checks a candidate against a pass rate the lab reports. 0/0 leaves
# marker detection ungated on transcriptome quality — a different question from
# this one, not a different answer to it.
QC_MIN_READS = 20
QC_MIN_GENES = 5

# mScarlet rolony floor for a cell to be DRAWN into the alignment TIF.
# A registration knob: the fit is stored as frame parameters on the grid, so it
# applies to cells that were never drawn. Pick it by eye with
# check_rolony_cutoff.py, then set it here — it is the "ge" in SUBSLICE_DIR
# below, and each value writes its own folder rather than overwriting.
ALIGN_MIN_ROLONIES = 3

# ---------------------------------------------------------------------
# Raw input (read-only)
# ---------------------------------------------------------------------

DATA_ROOT = r"C:\Users\David\lab_local\projects\cell_type\data\050526 BY95\allen_transcriptomics\BY95"

# Per-FOV raw directory (the MAX_Pos*_*_* folders).
# Blank -> auto-detect; BY95 resolves to <DATA_ROOT>/hyb_raw_files.
HYB_ROOT = r""

# ---------------------------------------------------------------------
# Derived output
# ---------------------------------------------------------------------

# Blank -> <ANALYSIS_ROOT>/preprocessing. Consumed only by preprocessing; the
# graph builder reaches the same tree through ANALYSIS_ROOT.
# Anything an older run wrote directly under <ANALYSIS_ROOT> is an orphan from
# before this moved a level deeper — not read, not overwritten.
OUTPUT_ROOT = r""

# Blank -> <ANALYSIS_ROOT>/alignment/BY95_graph.db.
GRAPH_PATH = r""

# ---------------------------------------------------------------------
# What the graph ingests
#
# Each 2P volume enters as a _red + _green pair joined by Identity; rigid and
# nonlinear edges are fitted only on _red <-> _red. Blank skips that node,
# set-but-missing is a hard error, GREEN without its RED is a hard error.
# At least one of these five must be set.
# ---------------------------------------------------------------------

# BARseq subslices. Relative, so everything above it comes from
# preprocessing_config and a rename never reaches this file. Resolved against
# BOTH output roots:
#   "qc20_5_ge3"  -> subslice_align/qc20_5_ge3/   (step 6 ALIGN tifs)
# ALIGN tifs are the only image the builder ingests — binary, marker-only. A
# folder holding none (a step 4 overlay folder, say) is a hard error; the step 4
# overlay is an RGB display figure whose BT.601 collapse renders BY95's median
# marker cell darker than the mask field behind it.
# The number is ALIGN_MIN_ROLONIES, picked by eye with check_rolony_cutoff.py;
# each cutoff writes its own folder, so change it here to switch which one is
# ingested. A wrong name raises and lists the folders that do exist.
# NOTE: the folder name is part of the node name (slice22_subslice_ALIGN_qc20_5_ge3),
# so switching cutoffs ADDS a parallel node set to an existing graph rather than
# replacing one. The old set keeps its own alignment; the new one is aligned by
# hand. force_rebuild=True is the reset.
SUBSLICE_DIR = r"qc20_5_ge3"

# TODO(David, on Windows): fill in the four BY95 2-photon TIFFs from
# "050526 BY95\". Until then the graph is subslice-only and the cross-modality
# handedness guard has nothing to compare. The paths that were here were
# JH302's (retired 2026-08-22) and were removed rather than carried over.

# Ex-vivo block, 2P volume of the tissue before slicing.
BLOCK_STACK_PATH_RED = r""
BLOCK_STACK_PATH_GREEN = r""

# In-vivo 2P stack.
INVIVO_PATH_RED = r""
INVIVO_PATH_GREEN = r""
