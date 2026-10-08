"""Per-section centre clicks, and the whole-pixel layout that stacks sections on them.

Lives at the project root next to ``orientation.py`` and imports nothing beyond
the stdlib, so the picker (matplotlib), the stacker (numpy + tifffile) and a later
point-cloud step all read one record through one module.

One record per SOURCE, because the two sources are different grids:

    slice     the whole section, stitch_slices.py --downsample
              (HYB_slice_stitched_tif_downsampled_micronwise/slice{N}_{CH}.tif)
              -> <ANALYSIS_ROOT>/section_centers_slice.json
    subslice  the marker-defined region, pipeline step 3
              (HYB_subslice_stitched_tif_downsampled_micronwise/slice{N}_subslice_{CH}.tif)
              -> <ANALYSIS_ROOT>/section_centers.json

A record::

    {
      "source": "slice",
      "image": "DAPI",
      "sections": {
        "22": {"y": 812.4, "x": 1040.0, "shape": [1630, 2210],
               "file": "slice22_subslice_DAPI.tif"},
        ...
      }
    }

``y``/``x`` are 0-indexed pixel coordinates in that section's downsampled grid
for the record's source. Every channel of one source shares that grid (and for
subslice, so does every ALIGN render), so one click serves all of them. A
record without ``source`` predates the field and is subslice. ``shape`` is the grid the click was made on; a
consumer whose image has another shape is on a different grid (step 3 re-run at
another pitch) and must refuse the click rather than misplace the section.

Stacking is translation only, by whole pixels: no resampling, so pixel values
are unchanged. A section pixel (y, x) lands at (y + dy, x + dx) on the canvas,
and every section's rounded centre lands on the same canvas pixel.
"""

import json
import math
import os
from pathlib import Path

from analysis_paths import get_analysis_root

SOURCES = ("slice", "subslice")
CENTERS_FILENAMES = {"slice": "section_centers_slice.json",
                     "subslice": "section_centers.json"}
CENTERS_FILENAME = CENTERS_FILENAMES["subslice"]
RAW_TIF_NAMES = {"slice": "slice{n}_{channel}.tif",
                 "subslice": "slice{n}_subslice_{channel}.tif"}


def raw_tif_name(source, slice_no, channel) -> str:
    """Downsampled raw channel filename: `slice22_DAPI.tif` / `slice22_subslice_DAPI.tif`."""
    return RAW_TIF_NAMES[source].format(n=int(slice_no), channel=channel.upper())


def record_source(record) -> str:
    """The record's source; a record from before the field is subslice."""
    return record.get("source") or "subslice"


def centers_path(explicit=None, source="subslice") -> Path:
    """``<ANALYSIS_ROOT>/<CENTERS_FILENAMES[source]>``; `explicit` (an ``--centers`` value) wins."""
    if explicit:
        return Path(explicit).expanduser()
    root = get_analysis_root()
    if root is None:
        raise ValueError(
            f"ANALYSIS_ROOT is not set in local_config.py, so there is no place "
            f"to keep {CENTERS_FILENAMES[source]}. Set it, or pass --centers <path>.")
    return root / CENTERS_FILENAMES[source]


def load(path) -> dict:
    """The record at `path`, or an empty one when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return {"image": None, "sections": {}}
    record = json.loads(path.read_text())
    record.setdefault("sections", {})
    return record


def save(path, record) -> None:
    """Write `record` to `path` via a temporary file, so an interrupted write leaves the old one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def set_center(record, slice_no, y, x, shape, file) -> None:
    """Record one section's centre, in place."""
    record["sections"][str(int(slice_no))] = {
        "y": round(float(y), 2), "x": round(float(x), 2),
        "shape": [int(shape[0]), int(shape[1])], "file": str(file),
    }


def centers(record) -> dict:
    """``{slice_no: (y, x)}`` from a record."""
    return {int(k): (v["y"], v["x"]) for k, v in record["sections"].items()}


def shapes(record) -> dict:
    """``{slice_no: (H, W)}`` the clicks were made on."""
    return {int(k): tuple(v["shape"]) for k, v in record["sections"].items()}


def pixel(v) -> int:
    """Nearest whole pixel, halves up (Python's round() sends 2.5 to 2)."""
    return int(math.floor(v + 0.5))


def layout(section_centers, section_shapes):
    """Canvas shape and per-section shifts that put every centre on one pixel.

    `section_centers` is ``{slice_no: (y, x)}``, `section_shapes` ``{slice_no: (H, W)}``.
    Returns ``((H, W), {slice_no: (dy, dx)})``: the smallest canvas holding every
    section uncropped, and the whole-pixel shift for each. Every rounded centre
    lands at the same canvas pixel, ``(max rounded y, max rounded x)``.
    """
    if not section_centers:
        raise ValueError("no section centres to lay out")
    cy = {s: pixel(y) for s, (y, _) in section_centers.items()}
    cx = {s: pixel(x) for s, (_, x) in section_centers.items()}
    above = max(cy.values())
    left = max(cx.values())
    below = max(section_shapes[s][0] - cy[s] for s in section_centers)
    right = max(section_shapes[s][1] - cx[s] for s in section_centers)
    shifts = {s: (above - cy[s], left - cx[s]) for s in section_centers}
    return (above + below, left + right), shifts
