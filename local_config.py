"""
Local configuration — machine-specific paths. Subject: BY108.

BY95's values are in git at e5e9aa2.

TRACKED IN GIT since 81dc5d8, not gitignored. Paths here travel between the
WSL edit host and the Windows execution host, and every edit shows as a repo
diff. Do not put credentials in this file.

Blank is a convention, not an omission: for OUTPUT_ROOT, HYB_ROOT and
GRAPH_PATH it means "derive it", and each says below what it derives to. SCOPE
is the exception — it is authoritative, and blank is a hard error.

Switching subjects: ANALYSIS_ROOT, DATA_ROOT, SCOPE (+ BLOCK_STACK_SCOPE), the four 2P paths, and
SUBSLICE_DIR's cutoff. Nothing in the notebook or the graph builder changes.
"""

# ---------------------------------------------------------------------
# Subject
# ---------------------------------------------------------------------

# One root for every derived output — preprocessing/, alignment/, cellpose/,
# plus orientation.json at its top level. Its folder name is the subject label
# ("BY108"), which is what names the graph .db.
ANALYSIS_ROOT = r"D:\lab_local\projects\cell_type\analysis\BY108"

# Microscope that acquired this subject's data; profiles in scope_profiles.py.
# Source of truth for every pixel size — the BARseq resample factor, node
# spacings, and validate_mnn's µm scale. TIFF XResolution is read only to check
# it, and a disagreement is a hard error.
#   "huang_lab"   1.1000 µm/px XY, 1.0 µm Z   BY95 (563.2 µm FOV)
#   "huang_lab_bruker_512px"   1.1056 µm/px XY, 1.0 µm Z, 566.08 µm FOV  (Bruker XML)
#   "huang_lab_bruker_1024px"  0.5528 µm/px XY, same field at 1024 px
# BY108's stacks carry no calibration tag; the pitch is from the rig's Prairie
# View XML for 16x zoom 2 at 512 px (in vivo), halved for the 1024 px block.
# by84/by94/by89 named "huang_lab_566um", deleted 2026-08-30 — re-derive
# their pitch from their own XResolution tags before running them again.
SCOPE = "huang_lab_bruker_512px"

# The ex vivo block's scope when it was sampled at another pitch; applies to
# the block_stack_* nodes only. Blank = SCOPE. BARseq is resampled to SCOPE's.
BLOCK_STACK_SCOPE = "huang_lab_bruker_1024px"

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
# BY108 has no BARseq: unset. BY95 was 20 / 5 / 3.
QC_MIN_READS = None
QC_MIN_GENES = None

# mScarlet rolony floor for a cell to be DRAWN into the alignment TIF.
# A registration knob: the fit is stored as frame parameters on the grid, so it
# applies to cells that were never drawn. Pick it by eye with
# check_rolony_cutoff.py, then set it here — it is the "ge" in SUBSLICE_DIR
# below, and each value writes its own folder rather than overwriting.
ALIGN_MIN_ROLONIES = None

# ---------------------------------------------------------------------
# Raw input (read-only)
# ---------------------------------------------------------------------

# Blank = no BARseq for this subject yet: preprocessing will not run, and the
# graph builder skips SUBSLICE_DIR and SUBSLICE_RAW_CHANNELS and builds from the
# 2P paths alone.
DATA_ROOT = r""

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

# Blank -> <ANALYSIS_ROOT>/alignment/BY108_graph.db.
GRAPH_PATH = r""

# ---------------------------------------------------------------------
# What the graph ingests
#
# Each 2P volume enters as a _red + _green pair joined by Identity; rigid and
# nonlinear edges are fitted only on _red <-> _red. Blank skips that node;
# set but not on disk yet is skipped with a warning, so a 2P-only graph builds
# before any BARseq exists. GREEN without its RED is a hard error, and so is
# nothing on disk at all.
# ---------------------------------------------------------------------

# BARseq subslices. Relative, so everything above it comes from
# preprocessing_config and a rename never reaches this file. Resolved against
# every output root:
#   "mscarlet_qc20_5_ge3" -> subslice_align_mscarlet/mscarlet_qc20_5_ge3/   (step 6 ALIGN tifs)
#   "gcamp_qc20_5_ge3_sat10" -> subslice_align_gcamp/, generate_alignment_tif.py --marker gcamp
#   "qc20_5_ge5"  -> subslice_align/qc20_5_ge5/   (unprefixed, written before 2026-10-08)
# ALIGN tifs are the only image the builder ingests — marker-only, binary or
# graded. A folder holding none (a step 4 overlay folder, say) is skipped with a
# warning, never ingested; the step 4 overlay is an RGB display figure whose BT.601 collapse renders
# BY95's median marker cell darker than the mask field behind it.
# The number is ALIGN_MIN_ROLONIES, picked by eye with check_rolony_cutoff.py;
# each cutoff writes its own folder, so change it here to switch which one is
# ingested. A wrong name raises and lists the folders that do exist.
# NOTE: the folder name is part of the node name (slice22_subslice_ALIGN_mscarlet_qc20_5_ge3),
# so switching cutoffs ADDS a parallel node set to an existing graph rather than
# replacing one. The old set keeps its own alignment; the new one is aligned by
# hand. force_rebuild=True is the reset.
SUBSLICE_DIR = r""


# BARseq raw fluorescence channels to carry into the graph beside the
# ALIGN renders, as Identity siblings of them. Empty list = skip.
#
#   []              nothing (previous behaviour)
#   ["MSCARLET"]    the channel with a 2P counterpart -- overlay it on invivo_red
#   ["MSCARLET", "GCAMP"]   add the BARseq GCaMP readout too
#
# Read from HYB_subslice_stitched_tif_downsampled_micronwise/, where
# downsample_subslices_cellmask.py resamples EVERY channel to one target shape
# computed from the cellmask -- so these sit pixel-for-pixel on the grid the
# ALIGN tif is painted on, and Identity is their literal relationship.
#
# Node names carry no cutoff (slice22_mscarlet): there is one raw image per
# section per channel however many rolony cutoffs were rendered from it. The
# FIRST channel is the section's Identity hub; every ALIGN render of that
# section links to it, so a fit made on any one of them reaches all of them.
# The flip side: two independent fits on two renders of the same section are
# then contradictory, and BFS resolves by path length. Fit one node per section.
# Stored losslessly, unlike the 2P volumes -- these are uint16 fluorescence.
SUBSLICE_RAW_CHANNELS = []


# Ex-vivo block, 2P volume of the tissue before slicing.
BLOCK_STACK_PATH_RED = r"D:\lab_local\projects\cell_type\data\20261006 BY108\BY108 exvivo red.tif"
BLOCK_STACK_PATH_GREEN = r""

# In-vivo 2P stack.
INVIVO_PATH_RED = r"D:\lab_local\projects\cell_type\data\20261006 BY108\BY108 invivo red.tif"
INVIVO_PATH_GREEN = r""
