"""cellpose_to_neuroreg.py: centroids land on neuroReg.loadTiff's grid.

Needs numpy, scipy, tifffile (.cellpose-venv or .castalign-venv):
    python tests/test_cellpose_to_neuroreg.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "alignment"))
sys.path.insert(0, str(ROOT))
import cellpose_to_neuroreg as c2n  # noqa: E402

XY, Z = 1.1, 1.0


def test_grid_and_order():
    # One voxel at 0-indexed (z=4, y=10, x=20): loadTiff puts row 11, column 21,
    # page 5 at x = 21*XY, y = 5*Z, z = 11*XY.
    pt = c2n.to_neuroreg(np.array([[4.0, 10.0, 20.0]]), XY, Z)
    assert np.allclose(pt[:, 0], [21 * XY, 5 * Z + c2n.PLANE_NUDGE * Z, 11 * XY])


def test_off_plane_untouched():
    pt = c2n.to_neuroreg(np.array([[4.5, 10.0, 20.0]]), XY, Z)
    assert np.isclose(pt[1, 0], 5.5 * Z)


def test_centroids_and_counts():
    m = np.zeros((10, 30, 40), dtype=np.uint16)
    m[2:5, 5:8, 10:13] = 3          # 27 voxels, centre (3, 6, 11)
    m[6:8, 20:22, 30:34] = 7        # 16 voxels, centre (6.5, 20.5, 31.5)
    labels, cents, counts = c2n.label_centroids(m)
    assert labels.tolist() == [3, 7]
    assert counts.tolist() == [27, 16]
    assert np.allclose(cents, [[3, 6, 11], [6.5, 20.5, 31.5]])


def test_cli_writes_mat():
    import tifffile
    from scipy.io import loadmat
    m = np.zeros((10, 30, 40), dtype=np.uint16)
    m[6:8, 20:22, 30:34] = 7
    m[1, 1, 1] = 9                  # 1 voxel, dropped by --min-um3
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "vol_masks.tif"
        tifffile.imwrite(str(p), m)
        assert c2n.main(["--masks", str(p), "--min-um3", "2"]) == 0
        out = loadmat(str(Path(d) / "vol_neuroreg.mat"))
    assert out["pt_list_vol"].shape == (3, 1)
    assert np.allclose(out["pt_list_vol"][:, 0], [32.5 * XY, 7.5 * Z, 21.5 * XY])
    assert np.isclose(out["pt_area_vol"][0, 0], 16 * XY * XY * Z)
    assert out["cellpose_label"][0, 0] == 7


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
