"""
Local configuration — machine-specific paths.
"""

# Path to raw BARseq data
DATA_ROOT = r"C:\Users\David\lab_local\projects\cell_type\data\050526 BY95\allen_transcriptomics\BY95"

# Path to output directory
OUTPUT_ROOT = r"C:\Users\David\lab_local\projects\cell_type\analysis\BY95"

# Path to directory of BARseq anisotropic subslice overlays (blank = skip)
SUBSLICE_DIR = r"C:\Users\David\lab_local\projects\cell_type\data\052026 JH302\preprocessing\mScarlet_cellmask_subslice\threshold_0.30_cellmask_0.50_anisotropic"

# Path to ex-vivo block RED channel (3D 2-photon volume, .tif/.tiff)
BLOCK_STACK_PATH_RED = r""
# Path to ex-vivo block GREEN channel (signal of interest, optional)
BLOCK_STACK_PATH_GREEN = r""

# Path to in-vivo RED channel 2-photon stack (.tif/.tiff)
INVIVO_PATH_RED = r"C:\Users\David\lab_local\projects\cell_type\data\052026 JH302\JH302_1x_ch1.tiff"
# Path to in-vivo GREEN channel 2-photon stack (signal of interest, optional)
INVIVO_PATH_GREEN = r"C:\Users\David\lab_local\projects\cell_type\data\052026 JH302\JH302_1x_ch2.tiff"

# Path to the alignment graph file (.db)
# Blank = auto-derived to <data parent>/alignment/<subject>_graph.db,
# which would duplicate the dated folder name in the filename
# ("050526 BY95/alignment/050526 BY95_graph.db"). Pinning explicitly
# to keep the filename clean.
GRAPH_PATH = r""

# Scope fallback for files lacking pixel calibration metadata.
# Values: "li_lab" | "huang_lab" | "" (blank).
# Behavior: if the TIFF has XResolution metadata, the autodetector uses it
# (this fallback is ignored). If metadata is absent, the builder uses the
# scope named here for that modality. Blank with absent metadata is a hard
# error.
SCOPE_FALLBACK_INVIVO = "li_lab"
SCOPE_FALLBACK_BLOCK = ""
