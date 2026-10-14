"""
Resize used by every downsample that feeds the graph.

imresize_bilinear is preprocessing/downsample_subslices_cellmask.py's (BARseq
step 3), copied rather than imported: that module imports preprocessing_config,
which raises when DATA_ROOT is blank (a subject with no BARseq). Step 3 keeps
its own copy unchanged.
"""

import numpy as np


def imresize_bilinear(image: np.ndarray, output_shape: tuple) -> np.ndarray:
    """Bilinear resize with anti-aliasing, original dtype kept (step 3's)."""
    original_dtype = image.dtype
    try:
        from skimage.transform import resize
        result = resize(image, output_shape, order=1, preserve_range=True,
                        anti_aliasing=True)
    except ImportError:
        from scipy.ndimage import zoom
        zoom_factors = (output_shape[0] / image.shape[0],
                        output_shape[1] / image.shape[1])
        result = zoom(image.astype(np.float64), zoom_factors, order=1)
    if np.issubdtype(original_dtype, np.integer):
        result = np.clip(result, np.iinfo(original_dtype).min,
                         np.iinfo(original_dtype).max)
    return result.astype(original_dtype)


def downsampled_shape(shape_hw: tuple, factor: float) -> tuple:
    """Step 3's target shape: round(native / factor) per axis."""
    return tuple(round(n / factor) for n in shape_hw)
