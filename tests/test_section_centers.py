#!/usr/bin/env python3
"""
Checks on section_centers: the whole-pixel layout and the record's round trip.

stdlib only, so it runs anywhere:

    python3 tests/test_section_centers.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import section_centers as sc

fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


print("pixel rounding")
check("halves go up", [sc.pixel(v) for v in (2.5, 3.5, -0.5, 2.49)] == [3, 4, 0, 2])

print("layout")
centres = {20: (10.0, 30.0), 21: (25.4, 5.6), 22: (40.0, 40.0)}
shapes = {20: (50, 60), 21: (30, 20), 22: (80, 45)}
(H, W), shifts = sc.layout(centres, shapes)
landed = {s: (sc.pixel(y) + shifts[s][0], sc.pixel(x) + shifts[s][1])
          for s, (y, x) in centres.items()}
check("every centre on one canvas pixel", len(set(landed.values())) == 1, str(landed))
check("no section cropped",
      all(dy >= 0 and dx >= 0 and dy + shapes[s][0] <= H and dx + shapes[s][1] <= W
          for s, (dy, dx) in shifts.items()), f"canvas {H}x{W} shifts {shifts}")
# smallest canvas: some section touches each of the four edges
check("canvas is the smallest that fits",
      min(dy for dy, _ in shifts.values()) == 0
      and min(dx for _, dx in shifts.values()) == 0
      and max(dy + shapes[s][0] for s, (dy, _) in shifts.items()) == H
      and max(dx + shapes[s][1] for s, (_, dx) in shifts.items()) == W)
check("worked numbers", ((H, W), shifts) ==
      ((80, 70), {20: (30, 10), 21: (15, 34), 22: (0, 0)}), f"{(H, W)} {shifts}")

(H1, W1), s1 = sc.layout({5: (3.0, 4.0)}, {5: (7, 9)})
check("one section is its own canvas", (H1, W1) == (7, 9) and s1 == {5: (0, 0)})

try:
    sc.layout({}, {})
    check("empty layout raises", False)
except ValueError:
    check("empty layout raises", True)

print("sources")
check("raw names", [sc.raw_tif_name("slice", 22, "dapi"), sc.raw_tif_name("subslice", 22, "GCAMP")]
      == ["slice22_DAPI.tif", "slice22_subslice_GCAMP.tif"])
check("one record per source", sc.CENTERS_FILENAMES ==
      {"slice": "section_centers_slice.json", "subslice": "section_centers.json"})
check("record without source is subslice", sc.record_source({"sections": {}}) == "subslice")
check("record source read", sc.record_source({"source": "slice", "sections": {}}) == "slice")

print("record")
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "sub" / sc.CENTERS_FILENAME
    rec = sc.load(path)
    check("missing file loads empty", rec == {"image": None, "sections": {}})
    sc.set_center(rec, 22, 812.444, 1040.0, (1630, 2210), "slice22_subslice_DAPI.tif")
    rec["image"] = "DAPI"
    sc.save(path, rec)
    back = sc.load(path)
    check("round trip", back == rec, str(back))
    check("centers()", sc.centers(back) == {22: (812.44, 1040.0)})
    check("shapes()", sc.shapes(back) == {22: (1630, 2210)})
    check("no temp file left", not path.with_suffix(".json.tmp").exists())
    check("explicit path wins", sc.centers_path(str(path)) == path)

print(f"\nFAILURES: {', '.join(fails) if fails else 'none'}")
sys.exit(1 if fails else 0)
