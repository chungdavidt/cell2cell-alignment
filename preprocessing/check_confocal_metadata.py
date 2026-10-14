#!/usr/bin/env python3
"""
List a confocal export and print each image's geometry and calibration.

For each TIFF-family file (.tif/.tiff/.ome.tif/.lsm): series shape, axes and
dtype, and the pixel size from every place it can be recorded -- OME
PhysicalSize, ImageJ unit + XResolution + spacing, the TIFF ResolutionUnit, and
Zeiss LSM voxel sizes -- so a disagreement between them is visible. Channel
names come from OME or ImageJ labels where present. Other image formats
(.czi, .nd2, .lif, .oir, .ims, ...) are listed with their size and the reader
module that would open them, if installed. Sidecar text/XML files are listed.

Read-only: opens nothing for writing. Does not import preprocessing_config.

Usage:
    python preprocessing/check_confocal_metadata.py "<confocal folder or file>"
    python preprocessing/check_confocal_metadata.py "<folder>" --max-files 5
"""

import argparse
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scope_profiles import um_per_px_from_resolution  # noqa: E402

TIFF_SUFFIXES = (".tif", ".tiff", ".lsm", ".btf")
OTHER_IMAGE_READERS = {
    ".czi": "czifile / aicsimageio", ".nd2": "nd2", ".lif": "readlif",
    ".oir": "oiffile / aicsimageio", ".oib": "oiffile", ".ims": "h5py",
    ".vsi": "bioformats", ".sld": "bioformats", ".lof": "readlif",
}
SIDECAR_SUFFIXES = (".xml", ".txt", ".json", ".csv", ".ini", ".log", ".oif", ".xlif")


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def describe_tiff(path: Path) -> None:
    import tifffile
    with tifffile.TiffFile(path) as t:
        for i, s in enumerate(t.series[:4]):
            print(f"    series {i}: shape {s.shape} axes {s.axes} {s.dtype}")
        if len(t.series) > 4:
            print(f"    ... {len(t.series)} series")
        page = t.pages[0]
        tags = page.tags

        if t.is_ome:
            ome = t.ome_metadata or ""
            sizes = dict(re.findall(
                r'\b(PhysicalSize[XYZ](?:Unit)?|Size[XYZCT]|DimensionOrder)="([^"]+)"', ome))
            print(f"    OME: {sizes}")
            names = re.findall(r'<Channel\b[^>]*\bName="([^"]*)"', ome)
            if names:
                print(f"    OME channels: {names}")
            for key in ("NominalMagnification", "LensNA", "Model"):
                m = re.search(rf'\b{key}="([^"]+)"', ome)
                if m:
                    print(f"    OME {key}: {m.group(1)}")

        ij = t.imagej_metadata
        if ij:
            short = {k: v for k, v in ij.items() if k not in ("Info", "Labels", "Ranges", "LUTs", "Properties")}
            print(f"    ImageJ: {short}")
            if ij.get("Labels"):
                print(f"    ImageJ labels: {list(dict.fromkeys(ij['Labels']))[:8]}")
            info = str(ij.get("Info", ""))
            hits = re.findall(r"[^\n]*(?:[Pp]ixel|[Vv]oxel|[Ss]caling|[Zz]oom|[Oo]bjective|[Ss]tep|[Cc]hannel ?[Nn]ame|[Dd]ye|[Ww]avelength)[^\n]*", info)
            for h in hits[:20]:
                print(f"      Info: {h.strip()[:200]}")

        if t.is_lsm:
            m = t.lsm_metadata or {}
            vox = [m.get(k) for k in ("VoxelSizeX", "VoxelSizeY", "VoxelSizeZ")]
            print(f"    LSM voxel size (um): {[None if v is None else v * 1e6 for v in vox]}")
            if m.get("ChannelColors"):
                print(f"    LSM channels: {m['ChannelColors']}")

        if "XResolution" in tags:
            num, den = tags["XResolution"].value
            unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else None
            ij_unit = (ij or {}).get("unit")
            um = um_per_px_from_resolution(num / den if den else None, ij_unit, unit)
            print(f"    XResolution {num}/{den}, ResolutionUnit {unit}, ImageJ unit {ij_unit!r}"
                  f" -> {um if um is None else f'{um:.5f}'} um/px")
        if "Software" in tags:
            print(f"    Software: {tags['Software'].value}")
        desc = page.description or ""
        if desc and not t.is_ome and not ij:
            print(f"    ImageDescription: {desc[:400]!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="confocal folder (searched recursively) or one file")
    ap.add_argument("--max-files", type=int, default=12,
                    help="TIFF files to open in detail; the rest are only listed (default 12)")
    args = ap.parse_args()

    if not args.path.exists():
        sys.exit(f"Not found: {args.path}")
    files = [args.path] if args.path.is_file() else sorted(p for p in args.path.rglob("*") if p.is_file())
    root = args.path if args.path.is_dir() else args.path.parent

    print(f"{len(files)} file(s) under {root}")
    print(f"  by extension: {dict(Counter(''.join(p.suffixes[-2:]).lower() or '(none)' for p in files))}")
    folders = Counter(str(p.parent.relative_to(root)) for p in files)
    print(f"  folders: {dict(folders)}")

    sidecars = [p for p in files if p.suffix.lower() in SIDECAR_SUFFIXES]
    if sidecars:
        print("\nSidecar files:")
        for p in sidecars[:30]:
            print(f"  {p.relative_to(root)}  ({p.stat().st_size / 1e3:.1f} kB)")

    others = [p for p in files if p.suffix.lower() in OTHER_IMAGE_READERS]
    if others:
        print("\nNon-TIFF image files:")
        for p in others[:30]:
            reader = OTHER_IMAGE_READERS[p.suffix.lower()]
            have = [m for m in re.split(r"[ /]+", reader) if m and installed(m)]
            print(f"  {p.relative_to(root)}  ({p.stat().st_size / 1e6:.1f} MB)  reader: {reader}"
                  f"  installed: {have or 'none'}")

    tiffs = [p for p in files if p.suffix.lower() in TIFF_SUFFIXES]
    if tiffs:
        if not installed("tifffile"):
            sys.exit("\ntifffile is not installed in this environment; run in .castalign-venv.")
        print(f"\nTIFF files: {len(tiffs)} (opening {min(len(tiffs), args.max_files)})")
        for p in tiffs[:args.max_files]:
            print(f"\n  {p.relative_to(root)}  ({p.stat().st_size / 1e6:.1f} MB)")
            try:
                describe_tiff(p)
            except Exception as e:
                print(f"    could not read: {type(e).__name__}: {e}")
        for p in tiffs[args.max_files:]:
            print(f"  (listed only) {p.relative_to(root)}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
