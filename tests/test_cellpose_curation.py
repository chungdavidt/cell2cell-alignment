"""cellpose_curation.py: found / missed / extra counts and the grid's override output.

Needs numpy (.cellpose-venv or .castalign-venv):
    python tests/test_cellpose_curation.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "alignment"))
sys.path.insert(0, str(ROOT))
import cellpose_curation as cc  # noqa: E402


def _truth():
    t = np.zeros((20, 20), dtype=np.uint16)
    t[2:6, 2:6] = 1        # 16 px
    t[10:14, 10:14] = 2    # 16 px
    return t


def test_exact_and_relabelled():
    t = _truth()
    p = np.where(t == 1, 7, np.where(t == 2, 3, 0))
    m = cc.match(t, p, 0.5)
    assert (m["found"], m["missed"], m["extra"]) == (2, 0, 0)
    assert cc.same_outlines(t, p)


def test_missed_and_extra():
    t = _truth()
    p = np.zeros_like(t)
    p[2:6, 2:6] = 5        # cell 1 found
    p[16:19, 16:19] = 9    # nothing there
    m = cc.match(t, p, 0.5)
    assert (m["found"], m["missed"], m["extra"]) == (1, 1, 1)
    assert not cc.same_outlines(t, p)


def test_iou_threshold():
    t = _truth()
    p = np.zeros_like(t)
    p[2:6, 2:4] = 1        # half of cell 1: IoU 0.5
    p[10:14, 10:11] = 2    # quarter of cell 2: IoU 0.25
    m = cc.match(t, p, 0.5)
    assert (m["found"], m["missed"], m["extra"]) == (1, 1, 1)


def test_one_outline_one_cell():
    # One outline covering both cells matches at most one of them.
    t = np.zeros((10, 10), dtype=np.uint16)
    t[0:5, 0:5] = 1
    t[0:5, 5:6] = 2
    p = np.zeros_like(t)
    p[0:5, 0:6] = 1
    m = cc.match(t, p, 0.1)
    assert (m["found"], m["missed"], m["extra"]) == (1, 1, 0)


def test_empty_plane():
    z = np.zeros((5, 5), dtype=np.uint16)
    t = cc.totals([cc.match(z, z, 0.5)])
    assert t["f1"] == 1.0 and t["found"] == 0


def test_grid_holds_vanilla_and_override():
    combos = cc.grid_combos()
    assert sum(cc.is_vanilla(c) for c in combos) == 1
    v = next(c for c in combos if cc.is_vanilla(c))
    assert cc.override_for(v) == {}
    c = dict(v, diameter=15, sharpen_radius=60)
    o = cc.override_for(c)
    assert o["diameter"] == 15
    assert o["normalize"]["sharpen_radius"] == 60
    assert o["normalize"]["percentile"] == [1.0, 99.0]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
