"""
MAT file I/O utilities.

Handles loading and saving .mat files with proper handling of:
- Sparse matrices
- Struct arrays
- Cell arrays
- Numeric arrays

IMPORTANT: MATLAB arrays are 1-indexed, Python arrays are 0-indexed.
"""

import numpy as np
from scipy import io as sio
from scipy import sparse
from pathlib import Path


def load_mat(filepath, squeeze_me=True, struct_as_record=False):
    """
    Load a .mat file.

    Args:
        filepath: Path to .mat file
        squeeze_me: Remove single-dimensional entries (default True)
        struct_as_record: Return structs as numpy records (default False)

    Returns:
        Dictionary of variables from .mat file
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MAT file not found: {filepath}")

    # Check if file is HDF5 format (MATLAB v7.3)
    # MATLAB v7.3 files have HDF5 signature at offset 512
    try:
        import h5py
        is_hdf5 = False

        # Check header for "MATLAB 7.3" or HDF5 signature
        with open(filepath, 'rb') as f:
            header = f.read(16)
            if b'MATLAB 7.3' in header:
                is_hdf5 = True
            else:
                # Check for HDF5 signature at start or offset 512
                if header[:4] == b'\x89HDF':
                    is_hdf5 = True
                else:
                    f.seek(512)
                    magic = f.read(4)
                    if magic == b'\x89HDF':
                        is_hdf5 = True

        if is_hdf5:
            with h5py.File(filepath, 'r') as f:
                return _load_hdf5_mat(f)
    except ImportError:
        pass  # h5py not installed, fall back to scipy

    # Try scipy for older .mat formats (v5, v7)
    return sio.loadmat(
        str(filepath),
        squeeze_me=squeeze_me,
        struct_as_record=struct_as_record
    )


def _load_hdf5_mat(f):
    """Load variables from HDF5 .mat file."""
    result = {}
    for key in f.keys():
        if key.startswith('#'):  # Skip HDF5 metadata
            continue
        result[key] = _convert_hdf5_to_numpy(f[key])
    return result


def _convert_hdf5_to_numpy(item, file=None):
    """Convert HDF5 dataset/group to numpy array."""
    import h5py
    if file is None and hasattr(item, 'file'):
        file = item.file

    if isinstance(item, h5py.Dataset):
        data = item[()]

        # Handle MATLAB cell arrays stored as object references
        if data.dtype == np.object_ or str(data.dtype).startswith('object'):
            # Check if MATLAB_class indicates cell array
            matlab_class = item.attrs.get('MATLAB_class', b'')
            if isinstance(matlab_class, bytes):
                matlab_class = matlab_class.decode('utf-8')

            if matlab_class == 'cell':
                # Dereference all objects
                result = []
                for ref in data.flatten():
                    if isinstance(ref, h5py.Reference):
                        result.append(_convert_hdf5_to_numpy(file[ref], file))
                    else:
                        result.append(ref)
                return result

        # Handle MATLAB strings stored as uint16 arrays
        # Only decode as string if the values are valid unicode code points
        # and the array is small (likely a single string, not an index array)
        if item.dtype == np.uint16 and len(data.shape) == 2:
            # Check if it looks like a string (small, values in printable range)
            # Large arrays with small values are likely numeric indices
            flat = data.flatten()
            if len(flat) < 1000 and len(flat) > 0:
                # Check if values are in printable ASCII/Unicode range
                if flat.min() >= 32 and flat.max() < 128:
                    try:
                        return ''.join(chr(c) for c in flat if c > 0)
                    except (ValueError, TypeError):
                        pass
        return data

    elif isinstance(item, h5py.Group):
        # Check MATLAB class attribute
        matlab_class = None
        if 'MATLAB_class' in item.attrs:
            matlab_class = item.attrs['MATLAB_class']
            if isinstance(matlab_class, bytes):
                matlab_class = matlab_class.decode('utf-8')

        # Handle sparse matrices
        if matlab_class == 'sparse' or ('data' in item and 'ir' in item and 'jc' in item):
            return _load_sparse_matrix(item)

        # Handle structs
        if matlab_class == 'struct':
            return _load_struct(item, file)

        # Handle cell arrays
        if matlab_class == 'cell':
            return _load_cell_array(item, file)

        # Handle categorical arrays (stored as indices + categories)
        if matlab_class == 'categorical':
            return _load_categorical(item, file)

        # Default: return as dict (recurse into group)
        return {k: _convert_hdf5_to_numpy(v, file) for k, v in item.items() if not k.startswith('#')}

    return item


def _load_sparse_matrix(group):
    """Load MATLAB sparse matrix from HDF5 group."""
    data = group['data'][()]
    ir = group['ir'][()]  # Row indices
    jc = group['jc'][()]  # Column pointers

    # MATLAB uses CSC format
    # Number of columns is len(jc) - 1
    # Number of rows is max(ir) + 1 (or from MATLAB_sparse_nrows attribute)
    ncols = len(jc) - 1
    if 'MATLAB_sparse' in group.attrs:
        nrows = int(group.attrs['MATLAB_sparse'])
    else:
        nrows = int(ir.max()) + 1 if len(ir) > 0 else 0

    return sparse.csc_matrix((data, ir, jc), shape=(nrows, ncols))


def _load_struct(group, file=None):
    """Load MATLAB struct from HDF5 group."""
    if file is None:
        file = group.file
    result = {}
    for key in group.keys():
        if not key.startswith('#'):
            result[key] = _convert_hdf5_to_numpy(group[key], file)
    return result


def _load_cell_array(group, file=None):
    """Load MATLAB cell array from HDF5 group."""
    import h5py
    if file is None:
        file = group.file
    refs = group[()]
    if refs.ndim == 0:
        return []

    result = []
    for ref in refs.flatten():
        if isinstance(ref, h5py.Reference):
            result.append(_convert_hdf5_to_numpy(file[ref], file))
        else:
            result.append(ref)
    return result


def _load_categorical(group, file=None):
    """Load MATLAB categorical array from HDF5 group."""
    if file is None:
        file = group.file

    # Categorical arrays have 'codes' (indices) and 'categories'
    if 'codes' in group:
        codes = group['codes'][()]
    elif 'data' in group:
        codes = group['data'][()]
    else:
        # Fall back to raw data
        return _convert_hdf5_to_numpy(group[()], file)

    # Get categories (the actual string values)
    if 'categories' in group:
        cats = group['categories']
        if isinstance(cats, h5py.Group):
            categories = []
            for ref in cats[()].flatten():
                if isinstance(ref, h5py.Reference):
                    val = file[ref][()]
                    if val.dtype == np.uint16:
                        categories.append(''.join(chr(c) for c in val.flatten() if c > 0))
                    else:
                        categories.append(val)
                else:
                    categories.append(ref)
        else:
            categories = cats[()]
    else:
        # Return indices if no categories
        return codes.flatten()

    # Convert indices to actual values (MATLAB is 1-indexed)
    result = []
    for idx in codes.flatten():
        if idx > 0 and idx <= len(categories):
            result.append(categories[int(idx) - 1])
        else:
            result.append(None)
    return result


def save_mat(filepath, data_dict, do_compression=True, format='5'):
    """
    Save variables to a .mat file.

    Args:
        filepath: Output path
        data_dict: Dictionary of variable names to values
        do_compression: Compress data (default True)
        format: '5' for MATLAB 5 format, '7.3' for HDF5 format

    Note:
        Use format='7.3' for large arrays (>2GB) or when compatibility with
        MATLAB -v7.3 is required.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == '7.3':
        _save_hdf5_mat(filepath, data_dict, do_compression)
    else:
        sio.savemat(
            str(filepath),
            data_dict,
            do_compression=do_compression
        )


def _save_hdf5_mat(filepath, data_dict, do_compression):
    """Save to HDF5 format (MATLAB v7.3)."""
    import h5py
    with h5py.File(filepath, 'w') as f:
        for key, value in data_dict.items():
            _write_hdf5_variable(f, key, value, do_compression)


def _write_hdf5_variable(f, name, value, compress):
    """Write a single variable to HDF5 file."""
    if isinstance(value, np.ndarray):
        if compress and value.size > 1000:
            f.create_dataset(name, data=value, compression='gzip')
        else:
            f.create_dataset(name, data=value)
    elif isinstance(value, (int, float)):
        f.create_dataset(name, data=np.array(value))
    elif isinstance(value, str):
        f.create_dataset(name, data=np.array(value, dtype='S'))
    elif isinstance(value, dict):
        grp = f.create_group(name)
        for k, v in value.items():
            _write_hdf5_variable(grp, k, v, compress)
    elif isinstance(value, (list, tuple)):
        f.create_dataset(name, data=np.array(value))
    elif sparse.issparse(value):
        # Convert sparse to dense for HDF5
        f.create_dataset(name, data=value.toarray(), compression='gzip' if compress else None)
    else:
        f.create_dataset(name, data=np.array(value))


def load_filt_neurons(filepath):
    """
    Load filt_neurons.mat with proper handling of its structure.

    Returns a dict-like object with fields:
        - expmat: Expression matrix (sparse or dense)
        - pos: Cell positions (N x 2)
        - pos40x: 40x positions (N x 2)
        - fov: FOV names (list of strings)
        - fov_names: Unique FOV names (if fov is numeric indices)
        - slice: Slice IDs

    IMPORTANT: mScarlet is in column index 113 (0-indexed in Python).
    In MATLAB it was column 114 (1-indexed).
    """
    data = load_mat(filepath, squeeze_me=True, struct_as_record=False)

    # Handle nested structure
    if 'filt_neurons' in data:
        fn = data['filt_neurons']
        # Could be a numpy void (struct) or dict
        if hasattr(fn, 'dtype') and fn.dtype.names is not None:
            # numpy structured array
            result = {name: fn[name] for name in fn.dtype.names}
        elif isinstance(fn, dict):
            result = fn
        else:
            result = data
    else:
        result = data

    # Normalize FOV names to list of strings
    result = _normalize_fov_names(result)

    # Fix array orientations from HDF5 (MATLAB stores column-major)
    # pos and pos40x should be (N, 2) not (2, N)
    for key in ['pos', 'pos40x', 'depth']:
        if key in result and isinstance(result[key], np.ndarray):
            if result[key].shape[0] == 2 and result[key].shape[1] > 2:
                result[key] = result[key].T

    # Squeeze 1D arrays: slice, angle, id should be (N,) not (1, N)
    for key in ['slice', 'angle', 'id', 'orig_slice']:
        if key in result and isinstance(result[key], np.ndarray):
            result[key] = result[key].flatten()

    # Convert sparse matrices to proper format
    if 'expmat' in result and sparse.issparse(result['expmat']):
        # Keep as sparse for memory efficiency
        pass
    elif 'expmat' in result:
        # Convert to sparse if dense and large
        expmat = np.asarray(result['expmat'])
        if expmat.size > 1e6:
            result['expmat'] = sparse.csr_matrix(expmat)

    return result


def _normalize_fov_names(data):
    """
    Normalize FOV names to a list of strings.

    Handles:
        - Numeric indices referencing fov_names
        - String arrays
        - Categorical arrays
        - Cell arrays of strings
    """
    if 'fov' not in data:
        return data

    fov = data['fov']

    # If numeric, convert using fov_names lookup
    if np.issubdtype(np.asarray(fov).dtype, np.number):
        if 'fov_names' in data:
            fov_names = data['fov_names']
            # Ensure fov_names is a list of strings
            if isinstance(fov_names, np.ndarray):
                if fov_names.dtype.kind == 'U':  # Unicode string
                    fov_names = fov_names.tolist()
                elif fov_names.dtype.kind == 'S':  # Byte string
                    fov_names = [s.decode('utf-8') for s in fov_names.flatten()]
                elif fov_names.dtype.kind == 'O':  # Object array
                    fov_names = [str(s) for s in fov_names.flatten()]

            # Convert numeric indices to names (MATLAB is 1-indexed)
            fov_indices = np.asarray(fov).flatten()
            data['fov'] = [fov_names[int(idx) - 1] for idx in fov_indices]
        else:
            raise ValueError("fov is numeric but fov_names not found")

    # If numpy array of strings, convert to list
    elif isinstance(fov, np.ndarray):
        if fov.dtype.kind in ('U', 'S', 'O'):
            data['fov'] = [str(s) for s in fov.flatten()]

    # If already list, ensure strings
    elif isinstance(fov, (list, tuple)):
        data['fov'] = [str(s) for s in fov]

    return data


def sparse_to_dense(mat):
    """Convert sparse matrix to dense numpy array."""
    if sparse.issparse(mat):
        return mat.toarray()
    return np.asarray(mat)


def get_expression_column(expmat, column_idx):
    """
    Get a specific column from expression matrix.

    Args:
        expmat: Expression matrix (sparse or dense)
        column_idx: Column index (0-indexed in Python!)

    Returns:
        1D numpy array of expression values
    """
    if sparse.issparse(expmat):
        return np.asarray(expmat[:, column_idx].todense()).flatten()
    return np.asarray(expmat[:, column_idx]).flatten()


# =============================================================================
# HDF5 Cellmask I/O (Python-native format)
# =============================================================================

def save_cellmask_h5(filepath, cellmask, metadata=None, compression='gzip'):
    """
    Save cellmask data to HDF5 format (.h5).

    This is the preferred Python-native format for cellmask data.
    Can still be read by MATLAB with h5read().

    Args:
        filepath: Output path (should end with .h5)
        cellmask: 2D uint32 array of cell IDs
        metadata: Dict of metadata (fov_offsets, min_x, min_y, etc.)
        compression: Compression method ('gzip', 'lzf', or None)

    Example:
        save_cellmask_h5('slice22_CELLMASK.h5', cellmask, {
            'fov_offsets': {'FOV1': [100, 200], 'FOV2': [300, 400]},
            'min_x': 50,
            'min_y': 75,
            'DOWNSAMPLE_X': 7.31,
            'DOWNSAMPLE_Y': 3.125,
        })
    """
    import h5py

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        # Save cellmask array
        f.create_dataset(
            'cellmask',
            data=cellmask.astype(np.uint32),
            compression=compression,
            chunks=True,  # Enable chunking for better compression
        )

        # Save metadata
        if metadata:
            meta_grp = f.create_group('metadata')
            for key, value in metadata.items():
                if key == 'fov_offsets' and isinstance(value, dict):
                    # Special handling for fov_offsets dict
                    fov_grp = meta_grp.create_group('fov_offsets')
                    for fov_name, offset in value.items():
                        fov_grp.create_dataset(fov_name, data=np.array(offset))
                elif isinstance(value, dict):
                    # Nested dict
                    sub_grp = meta_grp.create_group(key)
                    for k, v in value.items():
                        if isinstance(v, str):
                            sub_grp.create_dataset(k, data=np.bytes_(v))
                        else:
                            sub_grp.create_dataset(k, data=np.array(v))
                elif isinstance(value, str):
                    meta_grp.create_dataset(key, data=np.bytes_(value))
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    # Handle Python ints/floats AND numpy scalar types (int64, float64, etc.)
                    meta_grp.create_dataset(key, data=value)
                elif isinstance(value, np.ndarray):
                    meta_grp.create_dataset(key, data=value, compression=compression if value.size > 100 else None)
                elif isinstance(value, (list, tuple)):
                    meta_grp.create_dataset(key, data=np.array(value))

        # Add file format version for future compatibility
        f.attrs['format_version'] = '1.0'
        f.attrs['created_by'] = 'preprocessing_python'


def load_cellmask_h5(filepath):
    """
    Load cellmask data from HDF5 format (.h5).

    Args:
        filepath: Path to .h5 file

    Returns:
        Tuple of (cellmask, metadata)
        - cellmask: 2D uint32 array
        - metadata: Dict with all metadata including fov_offsets
    """
    import h5py

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Cellmask file not found: {filepath}")

    with h5py.File(filepath, 'r') as f:
        # Load cellmask
        cellmask = f['cellmask'][()]

        # Load metadata
        metadata = {}
        if 'metadata' in f:
            meta_grp = f['metadata']
            for key in meta_grp.keys():
                item = meta_grp[key]
                if isinstance(item, h5py.Group):
                    # Handle nested groups (like fov_offsets)
                    if key == 'fov_offsets':
                        metadata[key] = {
                            fov_name: item[fov_name][()].tolist()
                            for fov_name in item.keys()
                        }
                    else:
                        metadata[key] = {
                            k: _decode_h5_value(item[k])
                            for k in item.keys()
                        }
                else:
                    metadata[key] = _decode_h5_value(item)

    return cellmask, metadata


def _decode_h5_value(item):
    """Decode HDF5 dataset value to Python type."""
    value = item[()]
    if isinstance(value, bytes):
        return value.decode('utf-8')
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value
    return value
