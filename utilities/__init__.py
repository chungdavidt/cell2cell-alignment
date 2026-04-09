"""
Utility modules for preprocessing pipeline.

Modules:
    - mat_io: .mat file loading/saving
    - image_io: TIFF I/O (multi-page support)
    - regression: Linear regression (replaces MATLAB regress())
    - graph_utils: BFS, connected components, adjacency
    - visualization: Plotting helpers
"""

from .mat_io import load_mat, save_mat, load_filt_neurons
from .image_io import imread_tiff, imwrite_tiff, imread_multipage
from .regression import linear_regression, regress
from .graph_utils import (
    build_adjacency_8connected,
    find_connected_components,
    add_bridge_fovs,
    parse_fov_grid_positions,
)
from .visualization import (
    visualize_subslice,
    create_comparison_figure,
    create_histogram,
)
