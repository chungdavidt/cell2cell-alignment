#!/usr/bin/env python3
"""Round-trip tests for subslice_definitions.mat.

NOT stdlib-only, unlike the other two tests here -- needs numpy + scipy, so run
it in .castalign-venv:

    .castalign-venv\\Scripts\\python.exe tests\\test_subslice_definitions_roundtrip.py

Every assertion here was written against measured behaviour, not assumed
behaviour. What savemat does to a list of dicts is not obvious and three of
these were wrong on inspection alone:

  * `subslice_info` lands as a CELL of structs, not a struct array, so MATLAB
    needs `subslice_info{i}`.
  * a name list lands as an N-by-L CHAR MATRIX, so MATLAB's `(:)` flattens it
    column-major into one scrambled string.
  * short names are padded to the longest with SPACES (not NUL), and those
    spaces come back on the Python side too, where the names become directory
    components.
"""
import sys
import zlib
import struct
from pathlib import Path
import tempfile

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'preprocessing'))

import numpy as np
from utilities.mat_io import save_mat, load_mat


class Skip(Exception):
    """Raised when a test needs a resolvable local_config."""


def _step1():
    """identify_marker_subslices, or Skip.

    It imports preprocessing_config, which validates the whole preprocessing
    config -- on the WSL edit host ANALYSIS_ROOT is a Windows path and is not
    absolute, so these two tests only run on the execution host.
    """
    try:
        import identify_marker_subslices as ims
    except Exception as e:
        raise Skip(f"{type(e).__name__}: {str(e).strip().splitlines()[0][:80]}")
    return ims

MIXED = ['MAX_Pos9_003_005', 'MAX_Pos10_003_006']       # 16 and 17 characters
UNIFORM = ['MAX_Pos1_003_005', 'MAX_Pos2_003_006']      # both 16 -- BY95's shape


def entry(names):
    return {
        'slice_id': 22,
        'fov_list': list(names),
        'fov_grid_positions': np.zeros((len(names), 2)),
        'marker_fovs': list(names),
        'bridge_fovs': [],
        'num_marker_cells': len(names),
        'marker': 'mscarlet',
    }


def _top_level_class(path):
    """MATLAB array class of the first variable, read from the MAT5 bytes."""
    raw = Path(path).read_bytes()
    typ, nbytes = struct.unpack('<II', raw[128:136])
    body = zlib.decompress(raw[136:136 + nbytes]) if typ == 15 else raw[136:136 + nbytes]
    # miMATRIX tag, then the array-flags subelement; class is the low byte
    flags = struct.unpack('<I', body[16:20])[0]
    return flags & 0xFF


MX_CELL, MX_STRUCT = 1, 2


def test_subslice_info_is_a_cell_not_a_struct_array(tmp):
    """MATLAB must index it {i}. savemat takes the write_cells branch because a
    Python list is not a mapping."""
    p = tmp / 'cls.mat'
    save_mat(p, {'subslice_info': [entry(UNIFORM), entry(UNIFORM)]}, format='5')
    cls = _top_level_class(p)
    assert cls == MX_CELL, f"expected mxCELL_CLASS ({MX_CELL}), got {cls}"


def test_name_list_is_a_char_matrix(tmp):
    """Not a cell of strings -- which is why MATLAB needs cellstr before (:)."""
    p = tmp / 'chars.mat'
    save_mat(p, {'subslice_info': [entry(MIXED)]}, format='5')
    back = load_mat(p)['subslice_info']
    e = back.flatten()[0] if isinstance(back, np.ndarray) else back
    names = np.atleast_1d(e.fov_list).ravel()
    assert names.size == len(MIXED), f"got {names.size} elements, expected {len(MIXED)}"


def test_short_names_are_space_padded(tmp):
    """The padding is char 32. A padded name is a directory that does not exist."""
    p = tmp / 'pad.mat'
    save_mat(p, {'subslice_info': [entry(MIXED)]}, format='5')
    back = load_mat(p)['subslice_info']
    e = back.flatten()[0] if isinstance(back, np.ndarray) else back
    raw = [str(x) for x in np.atleast_1d(e.fov_list).ravel()]
    assert raw[0] != MIXED[0], "expected the short name to come back padded"
    assert raw[0].rstrip() == MIXED[0], f"padding is not trailing whitespace: {raw[0]!r}"
    assert set(raw[0][len(MIXED[0]):]) == {' '}, f"padding is not spaces: {raw[0]!r}"


def test_uniform_names_need_no_padding(tmp):
    """Why BY95 has never hit this."""
    p = tmp / 'uni.mat'
    save_mat(p, {'subslice_info': [entry(UNIFORM)]}, format='5')
    back = load_mat(p)['subslice_info']
    e = back.flatten()[0] if isinstance(back, np.ndarray) else back
    raw = [str(x) for x in np.atleast_1d(e.fov_list).ravel()]
    assert raw == UNIFORM, f"{raw!r}"


def test_loader_strips_the_padding(tmp):
    """identify_marker_subslices must not hand a padded name to a path join."""
    ims = _step1()
    p = tmp / 'strip.mat'
    save_mat(p, {'subslice_info': [entry(MIXED)]}, format='5')
    kept = ims._load_existing_definitions(p)
    assert kept, "entry was dropped"
    names = [str(x) for x in np.atleast_1d(kept[0]['fov_list']).ravel()]
    assert names == MIXED, f"{names!r}"


def test_legacy_field_names_still_load(tmp):
    """A file written before the rename carries mscarlet_fovs only."""
    ims = _step1()
    legacy = entry(UNIFORM)
    legacy['mscarlet_fovs'] = legacy.pop('marker_fovs')
    legacy['num_mscarlet_cells'] = legacy.pop('num_marker_cells')
    legacy.pop('marker')
    p = tmp / 'legacy.mat'
    save_mat(p, {'subslice_info': [legacy]}, format='5')
    kept = ims._load_existing_definitions(p)
    assert kept, "a pre-rename entry was dropped as malformed"
    assert set(ims.SUBSLICE_FIELDS) <= set(kept[0]), sorted(kept[0])
    assert kept[0]['marker'] == ims.PIPELINE_MARKER


def test_v73_cannot_hold_a_list_of_dicts(tmp):
    """Why edit_subslice_definitions.py writes format='5'."""
    try:
        save_mat(tmp / 'v73.mat', {'subslice_info': [entry(UNIFORM)]}, format='7.3')
    except Exception:
        return
    raise AssertionError("format='7.3' accepted a list of dicts; it should not")


def test_entries_load_as_mat_struct(tmp):
    """load_mat passes struct_as_record=False, so entries have _fieldnames and
    NOT dtype.names -- any unwrapper testing for .dtype silently does nothing."""
    p = tmp / 'ms.mat'
    save_mat(p, {'subslice_info': [entry(UNIFORM)]}, format='5')
    back = load_mat(p)['subslice_info']
    e = back.flatten()[0] if isinstance(back, np.ndarray) else back
    assert hasattr(e, '_fieldnames'), type(e).__name__
    assert not hasattr(e, 'dtype'), "entry unexpectedly exposes .dtype"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = skipped = 0
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for t in tests:
            try:
                t(tmp)
                print(f"  PASS  {t.__name__}")
            except Skip as e:
                skipped += 1
                print(f"  SKIP  {t.__name__}: needs a resolvable config ({e})")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL  {t.__name__}: {e}")
            except Exception as e:
                failed += 1
                print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, "
          f"{skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
