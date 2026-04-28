"""
Local configuration - COPY THIS FILE to local_config.py and fill in your paths.

    cp local_config.example.py local_config.py

local_config.py is gitignored and will not be committed.
Each person sets their own paths once.
"""

# Path to raw BARseq data (Batch3_JH302)
# Examples:
#   Windows:  "C:/Users/Li Lab/Documents/Data_ALM_cell_type_transcriptom/Batch3_JH302"
#   Mac:      "/Volumes/home/lab/raw_data/Data_ALM_cell_type_transcriptom/Batch3_JH302"
#   Linux:    "/home/yourname/lab/raw_data/Data_ALM_cell_type_transcriptom/Batch3_JH302"
DATA_ROOT = ""

# Path to output directory
# Examples:
#   Windows:  "C:/Users/Li Lab/Documents/output"
#   Mac:      "/Volumes/home/lab/output"
#   Linux:    "/home/yourname/lab/output"
OUTPUT_ROOT = ""

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

# Path to directory of BARseq anisotropic subslice overlays.
# Leave blank ("") for 2P-only alignment (no BARseq data).
# Typical location when you do have BARseq:
#   OUTPUT_ROOT + "/mScarlet_cellmask_subslice/threshold_0.30_cellmask_0.50_anisotropic"
SUBSLICE_DIR = ""

# Path to the alignment graph file (.db) — used by both the graph builder and
# the notebook. Typically under OUTPUT_ROOT/linestuffup_output/.
# Examples:
#   Windows:  "C:/Users/Li Lab/Documents/output/linestuffup_output/castalign_test.db"
#   Mac:      "/Volumes/home/lab/output/linestuffup_output/castalign_test.db"
#   Linux:    "/home/yourname/lab/output/linestuffup_output/castalign_test.db"
GRAPH_PATH = ""
