#!/usr/bin/env python3
"""
Print the acquisition calibration Prairie View recorded for a Bruker run.

Bruker writes one `<run>.xml` beside each run's per-plane `.ome.tif` files. It
holds the scan settings (micronsPerPixel, opticalZoom, objective, frame size)
and a positionCurrent per frame, from which the z step follows. Stacks
converted to a single TIFF elsewhere (bruker_hyperstack.tif, the split red
stacks) carry none of it, so this XML is the source for a scope profile.

Read-only. Stdlib only; tifffile is used, if installed, to also print the OME
size fields of a .tif argument. Does not import preprocessing_config, so it
runs with DATA_ROOT blank.

Usage:
    python preprocessing/check_bruker_metadata.py <run>.ome.tif   # finds <run>.xml beside it
    python preprocessing/check_bruker_metadata.py <run folder>     # every Prairie View .xml in it
    python preprocessing/check_bruker_metadata.py <run>.xml
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

SCAN_KEYS = (
    "micronsPerPixel", "opticalZoom", "objectiveLens", "objectiveLensMag",
    "objectiveLensNA", "pixelsPerLine", "linesPerFrame", "zDevice",
    "framePeriod", "dwellTime", "rastersPerFrame", "activeMode",
)


def state_values(shard) -> dict:
    """{key: value} from a PVStateShard. Indexed keys give {index: value};
    subindexed keys (positionCurrent) give {index: {subindex: float}}."""
    out = {}
    if shard is None:
        return out
    for sv in shard.findall("PVStateValue"):
        key = sv.get("key")
        if sv.get("value") is not None:
            out[key] = sv.get("value")
            continue
        indexed = {iv.get("index"): iv.get("value") for iv in sv.findall("IndexedValue")}
        sub = {}
        for svs in sv.findall("SubindexedValues"):
            sub[svs.get("index")] = {
                int(s.get("subindex")): float(s.get("value"))
                for s in svs.findall("SubindexedValue")
            }
        out[key] = indexed or sub
    return out


def xml_for(path: Path) -> list:
    """Prairie View XMLs for a .tif, a folder, or an .xml path."""
    if path.suffix.lower() == ".xml":
        return [path]
    folder = path if path.is_dir() else path.parent
    xmls = sorted(folder.glob("*.xml"))
    if path.is_file():
        # <run>_Cycle00001_Ch1_000001.ome.tif -> <run>.xml
        xmls = [x for x in xmls if path.name.startswith(x.stem + "_")] or xmls
    return [x for x in xmls if _is_pvscan(x)]


def _is_pvscan(path: Path) -> bool:
    with open(path, "rb") as f:
        return b"<PVScan" in f.read(4096)


def describe_steps(values: list) -> str:
    if len(values) < 2:
        return f"{len(values)} value(s): {values}"
    steps = Counter(round(b - a, 4) for a, b in zip(values, values[1:]))
    return (f"{len(values)} frames, {values[0]:g} -> {values[-1]:g} "
            f"(span {values[-1] - values[0]:g}), steps {steps.most_common(4)}")


def report_xml(path: Path) -> None:
    root = ET.parse(path).getroot()
    print(f"\n== {path}")
    print(f"  PVScan version {root.get('version')}  date {root.get('date')}")

    scan = state_values(root.find("PVStateShard"))
    print("  scan settings:")
    for key in SCAN_KEYS:
        if key in scan:
            print(f"    {key}: {scan[key]}")
    if "positionCurrent" in scan:
        print(f"    positionCurrent (start): {scan['positionCurrent']}")

    sequences = root.findall("Sequence")
    print(f"  sequences: {len(sequences)}  "
          f"{dict(Counter(s.get('type') for s in sequences))}")
    frames = [f for s in sequences for f in s.findall("Frame")]
    per_seq = Counter(len(s.findall("Frame")) for s in sequences)
    print(f"  frames: {len(frames)}  (frames per sequence: {dict(per_seq)})")
    channels = Counter(fl.get("channelName") or fl.get("channel")
                       for f in frames for fl in f.findall("File"))
    print(f"  files per channel: {dict(channels)}")

    # Settings that change inside the run: a frame-level shard overrides the scan.
    changed = Counter()
    for seq in sequences:
        for shard in [seq.find("PVStateShard")] + [f.find("PVStateShard") for f in seq.findall("Frame")]:
            for key, val in state_values(shard).items():
                if key in SCAN_KEYS and val != scan.get(key):
                    changed[(key, str(val))] += 1
    if changed:
        print("  settings that differ from the scan header inside the run:")
        for (key, val), n in changed.most_common(10):
            print(f"    {key} = {val}  ({n} shard(s))")

    # Z per frame, per device (subindex), within the first sequence and across sequences.
    def z_of(frame):
        pos = state_values(frame.find("PVStateShard")).get("positionCurrent", {})
        return pos.get("ZAxis") if isinstance(pos, dict) else None

    first = [z_of(f) for f in (sequences[0].findall("Frame") if sequences else [])]
    first = [z for z in first if z]
    if first:
        print("  Z within sequence 1, by device subindex:")
        for sub in sorted({k for z in first for k in z}):
            print(f"    subindex {sub}: {describe_steps([z[sub] for z in first if sub in z])}")
        print(f"    sum of subindices: {describe_steps([sum(z.values()) for z in first])}")
    else:
        print("  Z: no per-frame positionCurrent in sequence 1")
    if len(sequences) > 1:
        starts = [z_of(s.findall('Frame')[0]) for s in sequences if s.findall('Frame')]
        starts = [sum(z.values()) for z in starts if z]
        if starts:
            print(f"  Z at the first frame of each sequence: {describe_steps(starts)}")


def report_tif(path: Path) -> None:
    try:
        import tifffile
    except ImportError:
        print("  (tifffile not installed: OME fields skipped)")
        return
    with tifffile.TiffFile(path) as t:
        s = t.series[0]
        print(f"\n== {path.name}\n  shape {s.shape} axes {s.axes} {s.dtype}")
        print("  OME:", re.findall(
            r'(PhysicalSize[XYZ](?:Unit)?|Size[XYZTC])="([^"]+)"', t.ome_metadata or ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="a run's .ome.tif, its folder, or its .xml")
    path = ap.parse_args().path
    if not path.exists():
        sys.exit(f"Not found: {path}")
    if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
        report_tif(path)
    xmls = xml_for(path)
    if not xmls:
        folder = path if path.is_dir() else path.parent
        sys.exit(f"No Prairie View .xml in {folder}. Contents: "
                 f"{sorted(p.name for p in folder.iterdir() if not p.name.endswith('.tif'))[:30]}")
    for x in xmls:
        report_xml(x)


if __name__ == "__main__":
    main()
