"""
Local configuration - COPY THIS FILE to local_config.py and fill in your paths.

    cp local_config.example.py local_config.py

local_config.py is gitignored and will not be committed.
Each person sets their own paths once.
"""

# Path to the castalign repository root
# This is the folder that CONTAINS the castalign/ package folder
# Examples:
#   Windows:  "C:/Users/YourName/Desktop/castalign"
#   Mac:      "/Users/yourname/programs/castalign"
#   Linux:    "/home/yourname/lab/programs/castalign"
CASTALIGN_ROOT = ""

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
# Graph inputs — the graph builder adds a node for each path that is set.
# Leave any of these blank ("") to skip that node type.
# A path that is set but doesn't exist on disk is a hard error (typo guard).
# At least one of the three must be set.
# ---------------------------------------------------------------------

# Path to ex-vivo block image (3D 2-photon volume, .tif/.tiff)
# This is the 2P image of the tissue block before slicing.
# Examples:
#   Windows:  "C:/Users/Li Lab/Documents/data/gad2 by94 red exvivo max proj.tif"
#   Mac:      "/Users/yourname/data/gad2 by94 red exvivo max proj.tif"
BLOCK_STACK_PATH = ""

# Path to in-vivo 2-photon stack (.tif/.tiff)
# - Li lab data: typically the preprocessed/flipped output, e.g.
#       OUTPUT_ROOT/in_vivo_flip_corrected/JH302_1x_ch2_flipped.tiff
# - Huang lab data: typically the raw max-projection TIFF (no preprocessing needed)
# Examples:
#   Windows:  "C:/Users/Li Lab/Documents/output/in_vivo_flip_corrected/JH302_1x_ch2_flipped.tiff"
#   Mac:      "/Users/yourname/data/gad2 by94 red invivo max proj.tif"
#   Linux:    "/home/yourname/lab/output/in_vivo_flip_corrected/JH302_1x_ch2_flipped.tiff"
INVIVO_PATH = ""

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
