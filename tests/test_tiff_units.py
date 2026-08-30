#!/usr/bin/env python3
"""
Checks on scope_profiles' TIFF resolution-unit parser.

stdlib only and no I/O, so it runs anywhere — the parsing is pure, which is the
reason it lives in scope_profiles.py rather than in utilities/image_io.py next
to tifffile.

The case that matters is BY95: its 2P stacks carry XResolution 909090/1000000
with ImageJ unit µm written as a literal escape. The old parser tested for the
string 'micron' only, returned None, and assert_matches_metadata read that as
"nothing to check" — so a 2.81x pixel-size error passed the guard built to catch
exactly it.

    python3 tests/test_tiff_units.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scope_profiles as sp

fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


def close(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol


# ---------------------------------------------------------------- 1: BY95
print("1. the real BY95 case")
BY95_XRES = 909090 / 1000000
ESCAPED_MICRON = "\\u00B5m"        # seven characters, as tifffile hands it over

# ImageJ stored 1/1.1 rounded to six decimals, so inverting it lands 1.1e-6 off
# 1.1 exactly. That is the tag's own precision, not a parser error.
check("escaped micron unit parses",
      close(sp.um_per_px_from_resolution(BY95_XRES, ESCAPED_MICRON, 2), 1.1, 1e-5),
      str(sp.um_per_px_from_resolution(BY95_XRES, ESCAPED_MICRON, 2)))
check("...and it is 1.1000 to four places",
      round(sp.um_per_px_from_resolution(BY95_XRES, ESCAPED_MICRON, None), 4) == 1.1)
check("512 px of it is a 563.2 µm field",
      close(round(512 * sp.um_per_px_from_resolution(BY95_XRES, ESCAPED_MICRON, None), 1), 563.2))
check("the old 'micron'-only test would have missed it",
      ESCAPED_MICRON != 'micron' and sp.normalize_unit(ESCAPED_MICRON) != 'micron')
check("it lands within tolerance of huang_lab",
      sp.identify_scope(sp.um_per_px_from_resolution(BY95_XRES, ESCAPED_MICRON, None)) == 'huang_lab')

# ---------------------------------------------------------------- 2: spellings
print("\n2. accepted µm spellings, all -> 2.0 µm/px at XResolution 0.5")
for spelling in ["um", "µm", "μm", "\\u00B5m", "\\u03BCm", "micron", "microns",
                 "micrometer", "micrometers", "micrometre", "micrometres",
                 "  UM  ", "Micron", "\\u00b5m"]:
    check(f"{spelling!r}", close(sp.um_per_px_from_resolution(0.5, spelling, None), 2.0),
          str(sp.um_per_px_from_resolution(0.5, spelling, None)))

check("both mus normalize to a one-character unit",
      len(sp.normalize_unit("\\u00B5m")) == 2 and len(sp.normalize_unit("\\u03BCm")) == 2)

# ---------------------------------------------------------------- 3: ordering
print("\n3. a recognised ImageJ unit beats ResolutionUnit")
check("micron + ResolutionUnit 2 (inch) -> micron, not inch",
      close(sp.um_per_px_from_resolution(0.5, "\\u00B5m", 2), 2.0),
      str(sp.um_per_px_from_resolution(0.5, "\\u00B5m", 2)))
check("micron + ResolutionUnit 3 (cm) -> micron, not cm",
      close(sp.um_per_px_from_resolution(0.5, "micron", 3), 2.0))
check("the inch reading would have been wrong by 25400x",
      close(sp.um_per_px_from_resolution(0.5, None, 2), 50800.0))

# ---------------------------------------------------------------- 4: fallbacks
print("\n4. ResolutionUnit fallbacks when the ImageJ unit is absent or unknown")
check("no unit + ResolutionUnit 3 (cm)",
      close(sp.um_per_px_from_resolution(1e4, None, 3), 1.0))
check("no unit + ResolutionUnit 2 (inch)",
      close(sp.um_per_px_from_resolution(25400.0, None, 2), 1.0))
check("empty unit falls back too", close(sp.um_per_px_from_resolution(1e4, "", 3), 1.0))
check("unknown unit falls back to ResolutionUnit",
      close(sp.um_per_px_from_resolution(1e4, "furlongs", 3), 1.0))
check("ResolutionUnit as an int-like enum still resolves",
      close(sp.um_per_px_from_resolution(1e4, None, True + 2), 1.0))
check("non-micron lengths convert: mm", close(sp.um_per_px_from_resolution(1.0, "mm", None), 1e3))
check("non-micron lengths convert: nm", close(sp.um_per_px_from_resolution(1.0, "nm", None), 1e-3))
check("non-micron lengths convert: cm", close(sp.um_per_px_from_resolution(1.0, "cm", None), 1e4))
check("non-micron lengths convert: inch", close(sp.um_per_px_from_resolution(1.0, "inch", None), 25400.0))

# ---------------------------------------------------------------- 5: undecidable
print("\n5. undecidable -> None, never a guess")
check("ResolutionUnit 1 (none) with no ImageJ unit",
      sp.um_per_px_from_resolution(300.0, None, 1) is None)
check("no unit hints at all", sp.um_per_px_from_resolution(300.0, None, None) is None)
check("unknown unit with no ResolutionUnit",
      sp.um_per_px_from_resolution(300.0, "parsecs", None) is None)
check("zero XResolution", sp.um_per_px_from_resolution(0.0, "micron", None) is None)
check("negative XResolution", sp.um_per_px_from_resolution(-1.0, "micron", None) is None)
check("non-numeric XResolution", sp.um_per_px_from_resolution("abc", "micron", None) is None)
check("None XResolution", sp.um_per_px_from_resolution(None, "micron", None) is None)

# ---------------------------------------------------------------- 6: guard
print("\n6. what the parsed number does downstream")
check("a 72-DPI inch default is still rejected as implausible",
      not sp.is_plausible_xy(sp.um_per_px_from_resolution(72.0, None, 2)))
check("1.1000 passes assert_matches_metadata against huang_lab",
      sp.assert_matches_metadata('huang_lab', 1.1000) is None)

raised = False
try:
    sp.assert_matches_metadata('huang_lab', 0.3910)
except ValueError:
    raised = True
check("the old 0.3910 would now RAISE against huang_lab", raised)

check("huang_lab_566um is gone", 'huang_lab_566um' not in sp.MICROSCOPE_PROFILES)
check("no two profiles collide inside identify_scope's tolerance",
      all(sp.identify_scope(p['xy_um_per_px']) == name
          for name, p in sp.MICROSCOPE_PROFILES.items()))
check("huang_lab resamples BARseq by 3.4375x",
      close(sp.downsample_xy('huang_lab'), 3.4375))

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
