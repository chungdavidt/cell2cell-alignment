#!/usr/bin/env python3
"""
µm/px of every confocal sub-stack, as the values to type into Fiji.

Two images per sub-stack S{k}.{n}:
  ZEN export   CONFOCAL_DIR/S{k}.{n}.tiff (stitched RGB) and its tiles, at the
               pitch in S{k}.{n}.tiff_metadata.xml (Scaling X/Y/Z).
  graph node   <ANALYSIS_ROOT>/preprocessing/confocal/{label}/
               slice{k}_confocal_{label}_part{n}.tif, written by
               prepare_confocal_sections.py with no calibration tag. Its pitch is
               the native extent over the output shape, per axis: shapes are
               rounded, so it differs slightly from the nominal
               SCOPE pitch x CONFOCAL_EXTRA_DOWNSAMPLE.

For each file: what its TIFF header already says (what Fiji shows on open),
and the Fiji values: Image > Properties (Unit "micron", Pixel width/height),
or Analyze > Set Scale (Distance in pixels 1, Known distance = µm/px).

Reads headers and XML only; writes nothing.

Usage:
    python preprocessing/check_confocal_pitch.py
    python preprocessing/check_confocal_pitch.py --section S2.1 S2.2
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT, _ROOT / "preprocessing"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import local_config  # noqa: E402
from analysis_paths import confocal_label, confocal_output_dir  # noqa: E402
from prepare_confocal_sections import discover_sections, read_zen_xml  # noqa: E402
from scope_profiles import get_profile, um_per_px_from_resolution  # noqa: E402


def header_pitch(path: Path) -> str:
    """What the TIFF header records as µm/px, or why there is none."""
    if not path.exists():
        return "file not found"
    import tifffile
    with tifffile.TiffFile(str(path)) as t:
        s = t.series[0]
        tags = t.pages[0].tags
        ij = t.imagej_metadata or {}
        shape = f"shape {tuple(s.shape)} {s.axes}"
        if "XResolution" not in tags:
            return f"{shape}; no XResolution tag (Fiji opens it in pixels)"
        num, den = tags["XResolution"].value
        unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else None
        um = um_per_px_from_resolution(num / den if den else None, ij.get("unit"), unit)
        um_text = "no usable unit" if um is None else f"{um:.5f} um/px"
        return (f"{shape}; XResolution {num}/{den}, ResolutionUnit {unit}, "
                f"ImageJ unit {ij.get('unit')!r} -> {um_text}")


def fiji_line(y_um: float, x_um: float) -> str:
    return (f"Fiji: Pixel width {x_um:.5f}, Pixel height {y_um:.5f} micron  "
            f"(= {1 / x_um:.5f} px/um)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", nargs="+", metavar="S1.1", help="only these sub-stacks")
    args = ap.parse_args()

    conf_dir = Path(getattr(local_config, "CONFOCAL_DIR", "") or ".")
    channel = getattr(local_config, "CONFOCAL_CHANNEL", "")
    conf_scope = getattr(local_config, "CONFOCAL_SCOPE", "")
    graph_scope = getattr(local_config, "SCOPE", "")
    extra = float(getattr(local_config, "CONFOCAL_EXTRA_DOWNSAMPLE", 1.0))
    conf_xy = get_profile(conf_scope, "CONFOCAL_SCOPE")["xy_um_per_px"]
    target_xy = get_profile(graph_scope, "SCOPE")["xy_um_per_px"]
    print(f"CONFOCAL_DIR   {conf_dir}")
    print(f"CONFOCAL_SCOPE {conf_scope}: {conf_xy} um/px")
    print(f"SCOPE          {graph_scope}: {target_xy} um/px")
    print(f"graph node nominal pitch: {target_xy:.5f} x CONFOCAL_EXTRA_DOWNSAMPLE {extra:g} "
          f"= {target_xy * extra:.5f} um/px")

    out_dir = confocal_output_dir(channel) if channel else None
    sidecar = {}
    if out_dir is not None and (out_dir / "confocal_sections.json").exists():
        sidecar = json.loads((out_dir / "confocal_sections.json").read_text()).get("sections", {})
        print(f"node folder    {out_dir}  ({len(sidecar)} records in confocal_sections.json)")
    else:
        print(f"node folder    {out_dir}: no confocal_sections.json")

    if not conf_dir.is_dir():
        sys.exit(f"CONFOCAL_DIR not found: {conf_dir}")
    sections = discover_sections(conf_dir)
    if args.section:
        sections = [s for s in sections if s[2] in set(args.section)]
    if not sections:
        sys.exit("No S{k}.{n}.tiff_metadata.xml matched.")

    rows = []
    for slide, n, name, xml, _tiles in sections:
        meta = read_zen_xml(xml)
        sc, size = meta["scaling_um"], meta["size"]
        x_um, y_um, z_um = sc.get("X"), sc.get("Y"), sc.get("Z")
        print(f"\n{name}")
        print(f"  ZEN XML  Scaling X {x_um} Y {y_um} Z {z_um} um; "
              f"SizeX {size.get('SizeX')} SizeY {size.get('SizeY')} SizeZ {size.get('SizeZ')}")
        stitched = conf_dir / f"{name}.tiff"
        print(f"  {stitched.name}: {header_pitch(stitched)}")
        if x_um and y_um:
            print(f"    {fiji_line(y_um, x_um)}")
            if z_um:
                print(f"    tiles: Voxel depth {z_um:g} micron (z step between planes)")

        rec = sidecar.get(name)
        label = confocal_label(channel) if channel else "?"
        node_name = f"slice{slide}_confocal_{label}_part{n}"
        node_path = out_dir / f"{node_name}.tif" if out_dir is not None else None
        if rec is None:
            print(f"  {node_name}.tif: no record in confocal_sections.json")
            rows.append((name, node_name, x_um, None, None))
            continue
        nh, nw = rec["native_shape"]
        oh, ow = rec["output_shape"]
        nat = rec.get("native_um_per_px") or x_um
        node_y, node_x = nh * nat / oh, nw * nat / ow
        print(f"  {node_name}.tif: {header_pitch(node_path)}")
        print(f"    native {nh} x {nw} at {nat} um/px -> {oh} x {ow}, factor {rec['factor']:.6f}")
        print(f"    {fiji_line(node_y, node_x)}")
        rows.append((name, node_name, x_um, node_y, node_x))

    print("\nsummary (um/px)")
    print(f"  {'ZEN':<7} {'graph node':<34} {'ZEN X':>9} {'node Y':>9} {'node X':>9}")
    for name, node, zx, ny, nx in rows:
        f = lambda v: f"{v:9.5f}" if v else f"{'-':>9}"
        print(f"  {name:<7} {node:<34} {f(zx)} {f(ny)} {f(nx)}")


if __name__ == "__main__":
    main()
