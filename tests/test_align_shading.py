#!/usr/bin/env python3
"""Tests for generate_alignment_tif.py's graded mode (--ceiling) and its sidecar.

NOT stdlib-only -- needs numpy + tifffile, and the builder tests need castalign,
so run it in .castalign-venv on the execution host:

    .castalign-venv\\Scripts\\python.exe tests\\test_align_shading.py

Everything runs on a synthetic label mask; no dataset is read. Importing the
script imports preprocessing_config, which validates local_config, so on a host
without a resolvable config every test SKIPs.
"""
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'preprocessing'))

import numpy as np


class Skip(Exception):
    """Raised when a test needs a resolvable local_config or castalign."""


def _gat():
    try:
        import generate_alignment_tif as gat
    except Exception as e:
        raise Skip(f"{type(e).__name__}: {str(e).strip().splitlines()[0][:80]}")
    return gat


def _sgb():
    try:
        import alignment.subslice_graph_builder as sgb
    except Exception as e:
        raise Skip(f"{type(e).__name__}: {str(e).strip().splitlines()[0][:80]}")
    return sgb


def previous_qualifying_label_mask(cellmask, x_img, y_img, keep_cell, all_cells_level=0):
    """generate_alignment_tif.qualifying_label_mask as of d8c44a4, before --ceiling."""
    labels = cellmask if np.issubdtype(cellmask.dtype, np.integer) \
        else cellmask.astype(np.int32)
    h, w = labels.shape
    in_bounds = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
    ids = np.zeros(x_img.size, dtype=np.int64)
    ids[in_bounds] = labels[y_img[in_bounds], x_img[in_bounds]]
    on_mask = ids > 0
    lut = np.zeros(int(labels.max()) + 1, dtype=np.uint8)
    if all_cells_level:
        lut[:] = np.uint8(all_cells_level)
    lut[0] = 0
    drawn = on_mask & keep_cell
    if drawn.any():
        lut[ids[drawn]] = 255
    return (lut[labels],
            int(np.count_nonzero(lut == 255)),
            int((~on_mask & in_bounds).sum()),
            int((~in_bounds).sum()))


# 8x8 mask, labels 1-4 in 3x3 blocks, 0 between them
MASK = np.zeros((8, 8), dtype=np.int32)
MASK[0:3, 0:3] = 1
MASK[0:3, 4:7] = 2
MASK[4:7, 0:3] = 3
MASK[4:7, 4:7] = 4
# rows: one centroid inside each label, one on background, one out of bounds
X = np.array([1, 5, 1, 5, 3, 20])
Y = np.array([1, 1, 5, 5, 3, 1])


def level_at(img, label):
    vals = np.unique(img[MASK == label])
    assert vals.size == 1, f"label {label} holds {vals.tolist()}"
    return int(vals[0])


def test_binary_output_unchanged(tmp):
    gat = _gat()
    for keep in (np.array([1, 1, 1, 1, 1, 1], bool), np.array([1, 0, 1, 0, 1, 1], bool)):
        for substrate in (0, 60):
            new = gat.qualifying_label_mask(MASK, X, Y, keep, substrate)
            old = previous_qualifying_label_mask(MASK, X, Y, keep, substrate)
            assert np.array_equal(new[0], old[0]), f"pixels differ, substrate {substrate}"
            assert new[0].dtype == old[0].dtype == np.uint8
            assert new[1:] == old[1:], f"counts {new[1:]} != {old[1:]}"


def test_ramp_table_at_defaults(tmp):
    gat = _gat()
    got = gat.shade(np.arange(5, 16), 5, 15, 40).tolist()
    assert got == [40, 62, 83, 105, 126, 148, 169, 191, 212, 234, 255], got


def test_graded_levels_on_labels(tmp):
    gat = _gat()
    counts = np.array([5, 10, 15, 45, 7, 7])
    values = gat.shade(counts, 5, 15, 40)
    keep = np.ones(6, bool)
    img, n_drawn, off_mask, oob = gat.qualifying_label_mask(MASK, X, Y, keep, 0, values)
    assert img.dtype == np.uint8
    assert [level_at(img, k) for k in (1, 2, 3, 4)] == [40, 148, 255, 255]
    assert level_at(img, 0) == 0
    assert (n_drawn, off_mask, oob) == (4, 1, 1), (n_drawn, off_mask, oob)


def test_undrawn_cells_are_background(tmp):
    gat = _gat()
    values = gat.shade(np.array([5, 10, 15, 15, 5, 5]), 5, 15, 40)
    keep = np.array([1, 0, 1, 0, 1, 1], bool)      # rows 2 and 4 fail a gate
    img, n_drawn, _, _ = gat.qualifying_label_mask(MASK, X, Y, keep, 0, values)
    assert [level_at(img, k) for k in (1, 2, 3, 4)] == [40, 0, 255, 0]
    assert n_drawn == 2


def test_substrate_sits_under_graded_cells(tmp):
    gat = _gat()
    values = gat.shade(np.array([5, 10, 15, 15, 5, 5]), 5, 15, 40)
    keep = np.array([1, 0, 1, 1, 1, 1], bool)
    img, _, _, _ = gat.qualifying_label_mask(MASK, X, Y, keep, 20, values)
    assert [level_at(img, k) for k in (0, 1, 2, 3, 4)] == [0, 40, 20, 255, 255]


def test_floor_zero_is_visible(tmp):
    gat = _gat()
    levels = gat.shade(np.arange(0, 16), 0, 15, 40)
    assert levels[0] == 40 and levels[-1] == 255
    assert np.all(np.diff(levels.astype(int)) > 0), levels.tolist()


def test_widest_ramp_stays_distinct(tmp):
    gat = _gat()
    levels = gat.shade(np.arange(0, 216), 0, 215, 40).astype(int)
    assert levels.size == 216 and np.all(np.diff(levels) > 0)


def test_shared_label_takes_max(tmp):
    gat = _gat()
    x, y = np.array([1, 2]), np.array([1, 2])       # both rows on label 1
    keep = np.ones(2, bool)
    for vals in (np.array([62, 148], np.uint8), np.array([148, 62], np.uint8)):
        img, n_drawn, _, _ = gat.qualifying_label_mask(MASK, x, y, keep, 0, vals)
        assert level_at(img, 1) == 148
        assert n_drawn == 1


def test_cli_guards(tmp):
    gat = _gat()
    bad = [
        ["-n", "-1"],
        ["-n", "5", "--dim", "50"],                         # --dim without --ceiling
        ["-n", "15", "--ceiling", "15"],                    # no ramp width
        ["-n", "0", "--ceiling", "255"],                    # 256 counts in 216 levels
        ["-n", "5", "--ceiling", "--dim", "0"],
        ["-n", "5", "--ceiling", "--dim", "255"],
        ["-n", "5", "--ceiling", "--all-cells-level", "40"],
        ["--marker", "gcamp"],                              # no --min-rolonies
        ["--marker", "gcamp", "--ceiling"],
        ["--marker", "nope", "-n", "3"],
    ]
    for argv in bad:
        with mock.patch.object(sys, "argv", ["generate_alignment_tif.py", *argv]), \
                mock.patch("sys.stderr"):
            try:
                gat.main()
            except SystemExit as e:
                assert e.code == 2, f"{argv}: exit {e.code}, expected an argparse error"
            else:
                raise AssertionError(f"{argv}: accepted")


def _render(**over):
    r = {"marker": "mscarlet", "mode": "graded", "min_reads": 20, "min_genes": 5, "min_rolonies": 5,
         "ceiling": 15, "dim": 40, "all_cells_level": 0}
    r.update(over)
    return r


def test_sidecar_refuses_other_settings(tmp):
    gat = _gat()
    folder = tmp / "qc20_5_ge5_sat15"
    folder.mkdir()
    gat.check_render_sidecar(folder, _render())            # empty folder: fine
    gat.write_render_sidecar(folder, _render(), [(5, 40), (15, 255)])
    gat.check_render_sidecar(folder, _render())            # same settings: fine
    assert gat.read_render_sidecar(folder) == _render()
    try:
        gat.check_render_sidecar(folder, _render(dim=60))
    except SystemExit as e:
        assert "dim" in str(e.code), e.code
    else:
        raise AssertionError("different --dim accepted")


def test_sidecar_adopts_unrecorded_folder(tmp):
    gat = _gat()
    folder = tmp / "qc20_5_ge5"
    folder.mkdir()
    gat.imwrite_tiff(folder / "slice22_subslice_ALIGN.tif", np.zeros((8, 8), np.uint8))
    with mock.patch("sys.stdout"):
        gat.check_render_sidecar(folder, _render(mode="binary", ceiling=None, dim=None))


def test_builder_reads_the_same_sidecar(tmp):
    gat, sgb = _gat(), _sgb()
    assert sgb.ALIGN_RENDER_SIDECAR == gat.RENDER_SIDECAR
    folder = tmp / "reader"
    folder.mkdir()
    assert sgb.read_align_render(folder) is None
    gat.write_render_sidecar(folder, _render(), [])
    assert sgb.read_align_render(folder) == _render()


def test_builder_refuses_rerendered_folder(tmp):
    gat, sgb = _gat(), _sgb()
    folder = tmp / "builder"
    folder.mkdir()
    tif = folder / "slice22_subslice_ALIGN.tif"
    gat.imwrite_tiff(tif, np.zeros((8, 8), np.uint8))
    name = "slice22_subslice_ALIGN_builder"
    present = [(tif, name)]

    def graph(meta):
        return types.SimpleNamespace(node_metadata={name: meta})

    stored = {"shape": (1, 8, 8), "render": _render()}
    sgb.assert_stored_shapes_match(graph(stored), present, verbose=False,
                                   render=_render())
    sgb.assert_stored_shapes_match(graph({"shape": (1, 8, 8)}), present,
                                   verbose=False, render=_render(dim=60))
    sgb.assert_stored_shapes_match(graph(stored), present, verbose=False,
                                   render=None)
    try:
        sgb.assert_stored_shapes_match(graph(stored), present, verbose=False,
                                       render=_render(dim=60))
    except ValueError as e:
        assert "'dim': 40" in str(e) and "'dim': 60" in str(e), str(e)
    else:
        raise AssertionError("re-rendered folder accepted")
    # castalign stores node metadata with repr and reads it back with eval
    assert eval(repr(stored)) == stored


def test_render_leaf_per_marker(tmp):
    gat = _gat()
    assert gat.render_leaf("mscarlet", 20, 5, 5) == "mscarlet_qc20_5_ge5"
    assert gat.render_leaf("mscarlet", 20, 5, 5, 15) == "mscarlet_qc20_5_ge5_sat15"
    assert gat.render_leaf("gcamp", 20, 5, 3, 10) == "gcamp_qc20_5_ge3_sat10"
    assert gat.ALIGN_ROOTS["mscarlet"] == gat.SUBSLICE_ALIGN_MSCARLET_DIR
    assert gat.ALIGN_ROOTS["gcamp"] == gat.SUBSLICE_ALIGN_GCAMP_DIR
    assert gat.ALIGN_ROOTS["gcamp"] != gat.ALIGN_ROOTS["mscarlet"]


def test_sidecar_without_marker_is_mscarlet(tmp):
    gat, sgb = _gat(), _sgb()
    assert sgb.ALIGN_RENDER_DEFAULT_MARKER == gat.DEFAULT_MARKER
    folder = tmp / "pre_marker"
    folder.mkdir()
    old = {k: v for k, v in _render().items() if k != "marker"}
    gat.write_render_sidecar(folder, old, [])
    gat.check_render_sidecar(folder, _render())             # mScarlet re-render: fine
    assert sgb.read_align_render(folder) == _render()
    try:
        gat.check_render_sidecar(folder, _render(marker="gcamp"))
    except SystemExit as e:
        assert "marker" in str(e.code), e.code
    else:
        raise AssertionError("gcamp render accepted into an mScarlet folder")


def test_builder_accepts_node_without_marker(tmp):
    gat, sgb = _gat(), _sgb()
    folder = tmp / "pre_marker_node"
    folder.mkdir()
    tif = folder / "slice22_subslice_ALIGN.tif"
    gat.imwrite_tiff(tif, np.zeros((8, 8), np.uint8))
    name = "slice22_subslice_ALIGN_pre_marker_node"
    old = {k: v for k, v in _render().items() if k != "marker"}
    g = types.SimpleNamespace(node_metadata={name: {"shape": (1, 8, 8), "render": old}})
    sgb.assert_stored_shapes_match(g, [(tif, name)], verbose=False, render=_render())
    try:
        sgb.assert_stored_shapes_match(g, [(tif, name)], verbose=False,
                                       render=_render(marker="gcamp"))
    except ValueError as e:
        assert "'marker'" in str(e), str(e)
    else:
        raise AssertionError("marker change accepted")


def test_align_tif_names(tmp):
    import analysis_paths as ap
    sgb = _sgb()
    assert ap.align_tif_name(22, "gcamp_qc20_5_ge3_sat10") == \
        "slice22_subslice_ALIGN_gcamp_qc20_5_ge3_sat10.tif"
    assert ap.align_tif_name(22) == "slice22_subslice_ALIGN.tif"
    for name, n in [("slice22_subslice_ALIGN_gcamp_qc20_5_ge3_sat10.tif", 22),
                    ("slice7_subslice_ALIGN_mscarlet_qc0_0_ge0.tif", 7),
                    ("slice22_subslice_ALIGN.tif", 22),
                    ("slice22_subslice_MSCARLET.tif", None),
                    ("slice22_subslice_ALIGN_gcamp.tif.bak", None)]:
        assert ap.align_tif_slice(name) == n, (name, ap.align_tif_slice(name))
    # node name: section + folder, never the file's suffix
    folder = "gcamp_qc20_5_ge3_sat10"
    assert sgb.subslice_node_name(ap.align_tif_name(22, folder), folder) == \
        "slice22_subslice_ALIGN_gcamp_qc20_5_ge3_sat10"
    assert sgb.subslice_node_name("slice22_subslice_ALIGN.tif", "qc20_5_ge5") == \
        "slice22_subslice_ALIGN_qc20_5_ge5"


def test_align_tifs_one_per_section(tmp):
    import analysis_paths as ap
    folder = tmp / "two_per_section"
    folder.mkdir()
    for name in ("slice10_subslice_ALIGN_mscarlet_qc20_5_ge5.tif",
                 "slice2_subslice_ALIGN_mscarlet_qc20_5_ge5.tif",
                 "slice2_subslice_DAPI.tif"):
        (folder / name).write_bytes(b"")
    assert [p.name for p in ap.align_tifs(folder)] == [
        "slice2_subslice_ALIGN_mscarlet_qc20_5_ge5.tif",
        "slice10_subslice_ALIGN_mscarlet_qc20_5_ge5.tif"]
    (folder / "slice2_subslice_ALIGN.tif").write_bytes(b"")
    try:
        ap.align_tifs(folder)
    except ValueError as e:
        assert "Section 2" in str(e), e
    else:
        raise AssertionError("two ALIGN tifs for one section accepted")


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
                print(f"  SKIP  {t.__name__}: {e}")
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
