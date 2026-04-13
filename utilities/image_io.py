"""
Image I/O utilities for TIFF files.

Handles:
- Single-page TIFFs
- Multi-page TIFFs (common for microscopy data)
- 16-bit images (uint16)
- RGB images

Uses tifffile for reliable multi-page TIFF support.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, List

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    import warnings
    warnings.warn("tifffile not installed. Using PIL for TIFF I/O (limited multi-page support).")


def imread_tiff(filepath: Union[str, Path]) -> np.ndarray:
    """
    Read a TIFF file.

    Args:
        filepath: Path to TIFF file

    Returns:
        Image as numpy array
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TIFF file not found: {filepath}")

    if HAS_TIFFFILE:
        return tifffile.imread(str(filepath))
    else:
        from PIL import Image
        img = Image.open(filepath)
        return np.array(img)


def imread_multipage(filepath: Union[str, Path], page: Optional[int] = None) -> np.ndarray:
    """
    Read a multi-page TIFF file.

    Args:
        filepath: Path to TIFF file
        page: Specific page to read (0-indexed). If None, returns all pages.

    Returns:
        Image as numpy array. If multi-page and page=None, shape is (pages, H, W).

    Note:
        Page indices are 0-based in Python (unlike MATLAB which is 1-based).
        Example: MATLAB imread(file, 1) -> Python imread_multipage(file, 0)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TIFF file not found: {filepath}")

    if HAS_TIFFFILE:
        img = tifffile.imread(str(filepath))
        if page is not None:
            if img.ndim >= 3 and page < img.shape[0]:
                return img[page]
            elif img.ndim == 2 and page == 0:
                return img
            else:
                raise IndexError(f"Page {page} not found in TIFF with shape {img.shape}")
        return img
    else:
        from PIL import Image
        img = Image.open(filepath)

        if page is not None:
            try:
                img.seek(page)
                return np.array(img)
            except EOFError:
                raise IndexError(f"Page {page} not found in TIFF")

        # Read all pages
        pages = []
        try:
            while True:
                pages.append(np.array(img))
                img.seek(len(pages))
        except EOFError:
            pass

        if len(pages) == 1:
            return pages[0]
        return np.stack(pages, axis=0)


def imwrite_tiff(
    filepath: Union[str, Path],
    image: np.ndarray,
    compression: Optional[str] = None,
    photometric: Optional[str] = None,
) -> None:
    """
    Write a TIFF file.

    Args:
        filepath: Output path
        image: Image array to write
        compression: Compression method ('deflate', 'lzw', None)
        photometric: Photometric interpretation ('minisblack', 'rgb', None)

    Note:
        - uint16 images are written as-is (common for microscopy)
        - uint8 images are written as-is
        - float images are scaled to uint16 or uint8
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Handle float images
    if image.dtype in (np.float32, np.float64):
        if image.max() <= 1.0:
            # Assume normalized to [0, 1]
            image = (image * 255).astype(np.uint8)
        else:
            # Assume 16-bit range
            image = np.clip(image, 0, 65535).astype(np.uint16)

    if HAS_TIFFFILE:
        kwargs = {}
        if compression:
            kwargs['compression'] = compression
        if photometric:
            kwargs['photometric'] = photometric

        tifffile.imwrite(str(filepath), image, **kwargs)
    else:
        from PIL import Image
        if image.ndim == 3 and image.shape[2] == 3:
            mode = 'RGB'
        elif image.dtype == np.uint16:
            mode = 'I;16'
        else:
            mode = 'L'

        pil_img = Image.fromarray(image, mode=mode)
        pil_img.save(filepath)


def get_tiff_info(filepath: Union[str, Path]) -> dict:
    """
    Get information about a TIFF file.

    Returns:
        Dictionary with keys:
            - shape: Image dimensions
            - dtype: Data type
            - pages: Number of pages
            - photometric: Photometric interpretation
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TIFF file not found: {filepath}")

    if HAS_TIFFFILE:
        with tifffile.TiffFile(str(filepath)) as tif:
            pages = len(tif.pages)
            page0 = tif.pages[0]
            return {
                'shape': page0.shape if pages == 1 else (pages,) + page0.shape,
                'dtype': page0.dtype,
                'pages': pages,
                'photometric': str(page0.photometric) if hasattr(page0, 'photometric') else None,
            }
    else:
        from PIL import Image
        img = Image.open(filepath)

        # Count pages
        pages = 0
        try:
            while True:
                pages += 1
                img.seek(pages)
        except EOFError:
            pass

        img.seek(0)
        arr = np.array(img)

        return {
            'shape': arr.shape if pages == 1 else (pages,) + arr.shape,
            'dtype': arr.dtype,
            'pages': pages,
            'photometric': img.mode,
        }


def get_tiff_resolution(filepath: Union[str, Path]) -> dict:
    """
    Extract physical resolution from TIFF metadata.

    Reads XResolution/YResolution TIFF tags and ImageJ metadata
    to determine pixel size in microns.

    Returns:
        Dictionary with keys:
            - xy_um_per_px: XY pixel size in microns (None if undetectable)
            - z_um_per_px: Z spacing in microns (None if not in metadata)
            - shape: Volume dimensions
            - dtype: Data type
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TIFF file not found: {filepath}")

    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required for resolution autodetection")

    with tifffile.TiffFile(str(filepath)) as tif:
        page = tif.pages[0]
        shape = tif.series[0].shape
        dtype = tif.series[0].dtype
        ij_meta = tif.imagej_metadata or {}

        # XY resolution from TIFF tags
        xy_um_per_px = None
        x_tag = page.tags.get('XResolution')

        if x_tag is not None:
            num, denom = x_tag.value
            if num > 0 and denom > 0:
                pixels_per_unit = num / denom
                unit = ij_meta.get('unit', '')
                res_unit_tag = page.tags.get('ResolutionUnit')
                res_unit = res_unit_tag.value if res_unit_tag else None

                if unit == 'micron':
                    # ImageJ stores pixels_per_micron in XResolution when unit='micron'
                    xy_um_per_px = 1.0 / pixels_per_unit
                elif res_unit == 3:  # centimeter
                    xy_um_per_px = 1e4 / pixels_per_unit
                elif res_unit == 2:  # inch
                    xy_um_per_px = 25400.0 / pixels_per_unit

        # Z spacing from ImageJ metadata
        z_um_per_px = None
        if 'spacing' in ij_meta:
            z_um_per_px = float(ij_meta['spacing'])

        return {
            'xy_um_per_px': xy_um_per_px,
            'z_um_per_px': z_um_per_px,
            'shape': shape,
            'dtype': dtype,
        }


def load_fov_images(
    fov_name: str,
    hyb_root: Union[str, Path],
    channels_root: Optional[Union[str, Path]] = None,
) -> tuple:
    """
    Load FOV images (GCAMP, DAPI, mScarlet, cellmask).

    This mimics the MATLAB load_fov_images function from stitch_subslices.m.

    Args:
        fov_name: FOV name (e.g., 'MAX_Pos5_012_034')
        hyb_root: Path to hyb/ directory
        channels_root: Optional path to extracted channels directory

    Returns:
        Tuple of (gcamp, dapi, mscarlet, cellmask)
        Returns None for missing images.

    Note:
        Multi-page TIFF structure (alignedn2vhyb01.tif):
            - Page 0: GCAMP (was page 1 in MATLAB)
            - Page 3: mScarlet (was page 4 in MATLAB)
            - Page 4: DAPI (was page 5 in MATLAB)
    """
    from .mat_io import load_mat

    hyb_root = Path(hyb_root)
    gcamp = None
    dapi = None
    mscarlet = None
    cellmask = None

    # Try loading from hyb/ (multi-page TIFF)
    hyb_file = hyb_root / fov_name / 'alignedn2vhyb01.tif'
    if hyb_file.exists():
        try:
            info = get_tiff_info(hyb_file)
            if info['pages'] >= 5:
                # Page indices: 0=GCAMP, 3=mScarlet, 4=DAPI (0-indexed)
                gcamp = imread_multipage(hyb_file, 0)
                mscarlet = imread_multipage(hyb_file, 3)
                dapi = imread_multipage(hyb_file, 4)
        except Exception:
            pass

    # Fall back to extracted channels if hyb/ failed
    if gcamp is None and channels_root is not None:
        channels_root = Path(channels_root)
        channels_dir = channels_root / fov_name

        gcamp_file = channels_dir / 'GCAMP.tif'
        if gcamp_file.exists():
            gcamp = imread_tiff(gcamp_file)

        mscarlet_file = channels_dir / 'MSCARLET.tif'
        if mscarlet_file.exists():
            mscarlet = imread_tiff(mscarlet_file)

        dapi_file = channels_dir / 'DAPI.tif'
        if dapi_file.exists():
            dapi = imread_tiff(dapi_file)

    # Load cellmask
    cellmask_file = None
    if channels_root is not None:
        cellmask_file = Path(channels_root) / fov_name / 'cellmask.mat'

    if cellmask_file is None or not cellmask_file.exists():
        cellmask_file = hyb_root / fov_name / 'cellmask.mat'

    if cellmask_file.exists():
        try:
            mask_data = load_mat(cellmask_file)

            # Find cellmask variable
            possible_names = ['maski', 'cellmask', 'mask', 'segmentation', 'seg']
            for name in possible_names:
                if name in mask_data:
                    cellmask = np.asarray(mask_data[name])
                    break

            # Fall back to first numeric array
            if cellmask is None:
                for key, value in mask_data.items():
                    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                        if 'metadata' not in key.lower():
                            cellmask = np.asarray(value)
                            break
        except Exception:
            pass

    return gcamp, dapi, mscarlet, cellmask


def get_file_size_mb(filepath: Union[str, Path]) -> float:
    """Get file size in megabytes."""
    filepath = Path(filepath)
    if filepath.exists():
        return filepath.stat().st_size / 1e6
    return 0.0
