#!/usr/bin/env python3
"""
Short pitch report for the 2P stacks: what each TIFF says about its µm/px, and
what any Prairie View XML under the data folder says. A few lines per file.

Per TIFF (default: the four 2P paths in local_config):
  shape / axes / dtype, the field each candidate pitch implies, XResolution +
  ResolutionUnit + ImageJ unit/spacing -> µm/px, OME PhysicalSizeX, Software,
  and the first line of the ImageDescription.

Per Prairie View XML (<PVScan>) found recursively under --search (default: the
folder holding BLOCK_STACK_PATH_RED; ZEN *_metadata.xml files skipped), one
line: date, opticalZoom, micronsPerPixel X, pixelsPerLine x linesPerFrame,
objective, frame count.

Reads only.

Usage:
    python preprocessing/check_2p_pitch.py
    python preprocessing/check_2p_pitch.py "<a.tif>" "<b.tif>" --search "<folder>"
    python preprocessing/check_2p_pitch.py --no-search
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import local_config  # noqa: E402
from scope_profiles import MICROSCOPE_PROFILES, um_per_px_from_resolution  # noqa: E402

PATH_VARS = ("INVIVO_PATH_RED", "INVIVO_PATH_GREEN", "BLOCK_STACK_PATH_RED", "BLOCK_STACK_PATH_GREEN")
CANDIDATES = ("huang_lab", "huang_lab_bruker_512px", "huang_lab_bruker_1024px")


def tiff_report(path: Path):
    import tifffile
    print(f"\n{path.name}")
    if not path.exists():
        print("  not found")
        return
    with tifffile.TiffFile(str(path)) as t:
        s = t.series[0]
        page = t.pages[0]
        tags = page.tags
        ij = t.imagej_metadata or {}
        print(f"  shape {tuple(s.shape)} axes {s.axes} {s.dtype}")
        w = s.shape[-1]
        fields = ", ".join(f"{c} {MICROSCOPE_PROFILES[c]['xy_um_per_px'] * w:.0f} um"
                           for c in CANDIDATES if c in MICROSCOPE_PROFILES)
        print(f"  field at {w} px: {fields}")
        if "XResolution" in tags:
            num, den = tags["XResolution"].value
            unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else None
            um = um_per_px_from_resolution(num / den if den else None, ij.get("unit"), unit)
            print(f"  XResolution {num}/{den}, ResolutionUnit {unit}, ImageJ unit {ij.get('unit')!r}, "
                  f"spacing {ij.get('spacing')} -> {'none' if um is None else f'{um:.5f} um/px'}")
        else:
            print("  XResolution: absent")
        if t.is_ome:
            m = re.findall(r'PhysicalSize([XYZ])="([^"]+)"', t.ome_metadata or "")
            print(f"  OME PhysicalSize: {dict(m) or 'absent'}")
        sw = tags["Software"].value if "Software" in tags else None
        desc = (page.description or "").strip().splitlines()
        print(f"  Software {sw!r}; ImageDescription: {desc[0][:120]!r}" if desc
              else f"  Software {sw!r}; ImageDescription: none")


def pv_values(root) -> dict:
    out = {}
    shard = root.find("PVStateShard")
    if shard is None:
        return out
    for sv in shard.findall("PVStateValue"):
        key = sv.get("key")
        if sv.get("value") is not None:
            out[key] = sv.get("value")
        else:
            out[key] = {iv.get("index"): iv.get("value") for iv in sv.findall("IndexedValue")}
    return out


def search_xml(folder: Path):
    print(f"\nPrairie View XMLs under {folder}")
    hits = 0
    for p in sorted(folder.rglob("*.xml")):
        if p.name.endswith("_metadata.xml"):
            continue
        try:
            with open(p, "rb") as f:
                if b"<PVScan" not in f.read(4096):
                    continue
            root = ET.parse(p).getroot()
        except (OSError, ET.ParseError):
            continue
        hits += 1
        v = pv_values(root)
        mpp = v.get("micronsPerPixel", {})
        mpp = mpp.get("XAxis") if isinstance(mpp, dict) else mpp
        frames = sum(len(s.findall("Frame")) for s in root.findall("Sequence"))
        print(f"  {p.relative_to(folder)}: {root.get('date')}  zoom {v.get('opticalZoom')}  "
              f"um/px {mpp}  {v.get('pixelsPerLine')}x{v.get('linesPerFrame')}  "
              f"{v.get('objectiveLens')}  {frames} frames")
    if not hits:
        print("  none")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tifs", nargs="*", type=Path, help="default: the four 2P paths in local_config")
    ap.add_argument("--search", type=Path, default=None,
                    help="folder searched recursively for Prairie View XMLs; "
                         "default: BLOCK_STACK_PATH_RED's folder")
    ap.add_argument("--no-search", action="store_true")
    args = ap.parse_args()

    tifs = args.tifs or [Path(getattr(local_config, v)) for v in PATH_VARS if getattr(local_config, v, "")]
    for p in tifs:
        tiff_report(p)
    if args.no_search:
        return
    folder = args.search or (tifs[0].parent if tifs else None)
    if folder is None or not folder.is_dir():
        sys.exit(f"\nsearch folder not found: {folder}")
    search_xml(folder)


if __name__ == "__main__":
    main()
