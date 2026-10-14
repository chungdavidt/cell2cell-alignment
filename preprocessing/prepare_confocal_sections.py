#!/usr/bin/env python3
"""
Confocal sections -> one projected, downsampled image per section for the graph.

Input is a ZEN TIFF export (local_config.CONFOCAL_DIR), one per section:

    S1.1.tiff                    stitched RGB, every channel merged (not used)
    S1.1.tiff_metadata.xml       ZEN metadata: Scaling, channels, image size
    S1.1.tiff_files/             tiles, one per z plane, channel and mosaic tile:
        S1.1_h0s0t0z{z}c{c}x{x0}-{w}y{y0}-{h}m{m}.tiff

The stitched S{n}.tiff merges the channels into one RGB (tdTomato's orange and
DAPI's magenta both feed R), so it cannot give one channel back. The tiles hold
one channel each, rendered in that channel's colour. Per section:

  1. channel   CONFOCAL_CHANNEL is looked up by name among the image's channels
               in the XML; its index is the tiles' c{c} and its colour picks the
               RGB component that carries the intensity (tdTomato #FF4100 -> R).
  2. stitch    each tile is pasted at the offset in its filename (x{x0}, y{y0});
               overlaps take the max. The canvas must equal the XML's SizeX/SizeY.
  3. project   max over the z planes.
  4. pitch     CONFOCAL_SCOPE is the declared pitch; each section's XML Scaling
               must agree within 5% or the run stops.
  5. downsample to SCOPE's pitch (the graph grid, the in vivo stack's), by the
               same resize as downsample_subslices_cellmask.py: bilinear with
               anti-aliasing, target shape round(native / factor), dtype kept.
               A confocal coarser than SCOPE stops the run: images are only
               ever downsampled.

Section order: S{slide}.{n} sorted by slide then n is the cutting order
(David, 2026-10-14), numbered slice1, slice2, ... in that order.

Values are ZEN's 8-bit display rendering, not the 16-bit acquisition; no .czi
was kept.

Output, under <ANALYSIS_ROOT>/preprocessing/confocal/{label}/:
    slice{N}_confocal_{label}.tif     2D, dtype of the tiles
    confocal_sections.json            per section: source, slice number, tiles,
                                      z planes, pitches, factor, shapes

Reads CONFOCAL_DIR only; writes nothing next to it. Does not import
preprocessing_config, so it runs with DATA_ROOT blank (no BARseq).

Usage:
    python preprocessing/prepare_confocal_sections.py --dry-run
    python preprocessing/prepare_confocal_sections.py
    python preprocessing/prepare_confocal_sections.py --section S1.1 --force
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import local_config  # noqa: E402
from analysis_paths import analysis_subdir, PREPROCESSING_SUBDIR  # noqa: E402
from scope_profiles import assert_matches_metadata, get_profile  # noqa: E402

CONFOCAL_SUBDIR = "confocal"
SECTION_RE = re.compile(r"^S(\d+)\.(\d+)$")
TILE_RE = re.compile(
    r"^(?P<sec>.+?)_h\d+s\d+t\d+z(?P<z>\d+)c(?P<c>\d+)"
    r"x(?P<x>\d+)-(?P<w>\d+)y(?P<y>\d+)-(?P<h>\d+)m(?P<m>\d+)\.tiff?$")
# Mean |component - value * weight| above this (8-bit levels) means the tile is
# not a single channel rendered in the XML colour.
COLOUR_FIT_MAX_ERROR = 2.0


# ---------------------------------------------------------------------------
# ZEN metadata
# ---------------------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_zen_xml(path: Path) -> dict:
    """Pitch, image size, z planes and channels from a ZEN *_metadata.xml.

    Scaling/Items/Distance values are metres per pixel. Image-level channels
    are the Channel elements whose Id is "Channel:{k}"; k is the tiles' c{k}.
    """
    root = ET.parse(path).getroot()
    scaling = {}
    for scl in root.iter():
        if _local(scl.tag) != "Scaling":
            continue
        for d in scl.iter():
            if _local(d.tag) == "Distance" and d.get("Id"):
                val = next((c.text for c in d if _local(c.tag) == "Value"), None)
                if val:
                    scaling[d.get("Id")] = float(val) * 1e6
    size = {}
    for img in root.iter():
        if _local(img.tag) != "Image":
            continue
        for child in img:
            name = _local(child.tag)
            if re.fullmatch(r"Size[XYZCM]", name) and child.text and name not in size:
                size[name] = int(child.text)
    channels = {}
    for ch in root.iter():
        cid = ch.get("Id") or ""
        if _local(ch.tag) == "Channel" and cid.startswith("Channel:"):
            k = int(cid.split(":")[1])
            if k in channels:
                continue
            colour = next((c.text for c in ch if _local(c.tag) == "Color"), None)
            channels[k] = {"name": ch.get("Name"), "colour": colour}
    return {"scaling_um": scaling, "size": size, "channels": channels}


def colour_weights(argb: str) -> tuple:
    """(R, G, B) weights in [0, 1] from ZEN's #AARRGGBB."""
    h = argb.lstrip("#")[-6:]
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def channel_label(name: str) -> str:
    """'tdTom-T2' -> 'tdtom': the dye part of a ZEN channel name, lowercase."""
    return re.sub(r"[^a-z0-9]", "", name.split("-")[0].lower())


# ---------------------------------------------------------------------------
# Sections and tiles
# ---------------------------------------------------------------------------

def discover_sections(confocal_dir: Path) -> list:
    """[(slide, n, name, xml, tile folder)] in cutting order."""
    out = []
    for xml in confocal_dir.glob("*.tiff_metadata.xml"):
        name = xml.name[:-len(".tiff_metadata.xml")]
        m = SECTION_RE.match(name)
        if not m:
            continue
        out.append((int(m.group(1)), int(m.group(2)), name, xml,
                    confocal_dir / f"{name}.tiff_files"))
    return sorted(out)


def section_tiles(tile_dir: Path, section: str, channel: int) -> dict:
    """{z: [(path, x0, y0, w, h, m)]} for one channel of one section."""
    by_z = defaultdict(list)
    for p in tile_dir.iterdir():
        m = TILE_RE.match(p.name)
        if not m or m.group("sec") != section or int(m.group("c")) != channel:
            continue
        by_z[int(m.group("z"))].append(
            (p, int(m.group("x")), int(m.group("y")),
             int(m.group("w")), int(m.group("h")), int(m.group("m"))))
    return dict(sorted(by_z.items()))


# ---------------------------------------------------------------------------
# Resize -- copied from downsample_subslices_cellmask.imresize_bilinear (step 3),
# which cannot be imported here: its module imports preprocessing_config, and
# that raises with DATA_ROOT blank.
# ---------------------------------------------------------------------------

def imresize_bilinear(image, output_shape: tuple):
    """Bilinear resize with anti-aliasing, original dtype kept (step 3's)."""
    import numpy as np
    original_dtype = image.dtype
    try:
        from skimage.transform import resize
        result = resize(image, output_shape, order=1, preserve_range=True,
                        anti_aliasing=True)
    except ImportError:
        from scipy.ndimage import zoom
        zoom_factors = (output_shape[0] / image.shape[0],
                        output_shape[1] / image.shape[1])
        result = zoom(image.astype(np.float64), zoom_factors, order=1)
    if np.issubdtype(original_dtype, np.integer):
        result = np.clip(result, np.iinfo(original_dtype).min,
                         np.iinfo(original_dtype).max)
    return result.astype(original_dtype)


def build_projection(tiles_by_z: dict, canvas_hw: tuple, weights: tuple):
    """Stitch each z plane from its tiles (max in overlaps), then max over z.

    Returns (image, colour-fit error): the intensity is the RGB component with
    the largest colour weight; the error is the mean |other component -
    value * its weight| over nonzero pixels, which is ~0 when the tile really is
    one channel rendered in that colour.
    """
    import numpy as np
    import tifffile
    comp = max(range(3), key=lambda i: weights[i])
    canvas = None
    errors = []
    for z, tiles in tiles_by_z.items():
        for path, x0, y0, w, h, _ in tiles:
            rgb = tifffile.imread(str(path))
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f"{path.name}: expected (H, W, 3) RGB, got {rgb.shape}")
            if rgb.shape[:2] != (h, w):
                raise ValueError(f"{path.name}: shape {rgb.shape[:2]} but name says {(h, w)}")
            val = rgb[..., comp]
            if canvas is None:
                canvas = np.zeros(canvas_hw, dtype=val.dtype)
            nz = val > 0
            if nz.any():
                v = val[nz].astype(np.float64)
                for i in range(3):
                    if i != comp:
                        errors.append(np.abs(rgb[..., i][nz] - v * weights[i]).mean())
            region = canvas[y0:y0 + h, x0:x0 + w]
            np.maximum(region, val, out=region)
    return canvas, (max(errors) if errors else 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", nargs="+", metavar="S1.1",
                    help="only these sections (slice numbers still count every section)")
    ap.add_argument("--channel", default=None,
                    help="ZEN channel name; default local_config.CONFOCAL_CHANNEL")
    ap.add_argument("--dry-run", action="store_true",
                    help="report sections, tiles, pitches and shapes; read no pixels")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    confocal_dir = getattr(local_config, "CONFOCAL_DIR", "")
    if not confocal_dir:
        sys.exit("CONFOCAL_DIR is blank in local_config.py.")
    confocal_dir = Path(confocal_dir)
    if not confocal_dir.is_dir():
        sys.exit(f"CONFOCAL_DIR not found: {confocal_dir}")
    channel_name = args.channel or getattr(local_config, "CONFOCAL_CHANNEL", "")
    if not channel_name:
        sys.exit("CONFOCAL_CHANNEL is blank in local_config.py; pass --channel.")
    conf_scope = getattr(local_config, "CONFOCAL_SCOPE", "")
    conf_xy = get_profile(conf_scope, "CONFOCAL_SCOPE")["xy_um_per_px"]
    graph_scope = getattr(local_config, "SCOPE", "")
    target_xy = get_profile(graph_scope, "SCOPE")["xy_um_per_px"]
    if conf_xy > target_xy:
        sys.exit(f"CONFOCAL_SCOPE {conf_xy} um/px is coarser than SCOPE {target_xy}: "
                 f"that would be an upsample, and images are only downsampled.")
    factor = target_xy / conf_xy

    out_root = analysis_subdir(PREPROCESSING_SUBDIR)
    if out_root is None:
        sys.exit("ANALYSIS_ROOT is blank in local_config.py.")

    sections = discover_sections(confocal_dir)
    if not sections:
        sys.exit(f"No S{{slide}}.{{n}}.tiff_metadata.xml in {confocal_dir}")

    print(f"CONFOCAL_DIR:   {confocal_dir}")
    print(f"channel:        {channel_name}")
    print(f"CONFOCAL_SCOPE: {conf_scope}  ({conf_xy} um/px)")
    print(f"SCOPE (graph):  {graph_scope}  ({target_xy} um/px)")
    print(f"downsample:     native / {factor:.6f}")
    print(f"sections:       {len(sections)}, numbered in cutting order")

    label = channel_label(channel_name)
    out_dir = out_root / CONFOCAL_SUBDIR / label
    print(f"output:         {out_dir}")
    if args.dry_run:
        print("DRY RUN -- no pixels read, nothing written")

    wanted = set(args.section) if args.section else None
    records = []
    for number, (slide, n, name, xml, tile_dir) in enumerate(sections, start=1):
        if wanted is not None and name not in wanted:
            continue
        meta = read_zen_xml(xml)
        idx = next((k for k, ch in meta["channels"].items()
                    if ch["name"] == channel_name), None)
        if idx is None:
            sys.exit(f"{name}: no channel {channel_name!r}; channels in {xml.name}: "
                     f"{ {k: ch['name'] for k, ch in meta['channels'].items()} }")
        weights = colour_weights(meta["channels"][idx]["colour"])
        xy_meta = meta["scaling_um"].get("X")
        assert_matches_metadata(conf_scope, xy_meta, source_name=xml.name,
                                var_name="CONFOCAL_SCOPE")
        if not tile_dir.is_dir():
            sys.exit(f"{name}: tile folder not found: {tile_dir}")
        tiles_by_z = section_tiles(tile_dir, name, idx)
        if not tiles_by_z:
            sys.exit(f"{name}: no c{idx} tiles in {tile_dir}")
        tiles_per_z = {z: len(t) for z, t in tiles_by_z.items()}
        canvas_w = max(x0 + w for t in tiles_by_z.values() for _, x0, _, w, _, _ in t)
        canvas_h = max(y0 + h for t in tiles_by_z.values() for _, _, y0, _, h, _ in t)
        size = meta["size"]
        if (size.get("SizeX"), size.get("SizeY")) != (canvas_w, canvas_h):
            sys.exit(f"{name}: tiles span {canvas_w} x {canvas_h} but the XML image is "
                     f"{size.get('SizeX')} x {size.get('SizeY')} (SizeX x SizeY)")
        out_hw = (round(canvas_h / factor), round(canvas_w / factor))
        z_um = meta["scaling_um"].get("Z")
        out_path = out_dir / f"slice{number}_confocal_{label}.tif"
        print(f"\n  {name} -> slice{number}: c{idx} {channel_name} "
              f"(colour {meta['channels'][idx]['colour']}, component {'RGB'[max(range(3), key=lambda i: weights[i])]})")
        print(f"    XML pitch {xy_meta:.5f} um/px, z planes {sorted(tiles_by_z)} "
              f"({z_um:g} um apart), tiles per z {tiles_per_z}")
        print(f"    canvas {canvas_h} x {canvas_w} -> {out_hw[0]} x {out_hw[1]}  ({out_path.name})")

        record = {
            "section": name, "slice": number, "source_xml": str(xml),
            "channel": channel_name, "channel_index": idx,
            "z_planes": sorted(tiles_by_z), "z_step_um": z_um,
            "tiles_per_z": tiles_per_z, "projection": "max over z; max in tile overlaps",
            "native_um_per_px": xy_meta, "confocal_scope": conf_scope,
            "target_um_per_px": target_xy, "graph_scope": graph_scope,
            "factor": factor, "native_shape": [canvas_h, canvas_w],
            "output_shape": list(out_hw), "output": str(out_path),
        }
        if args.dry_run:
            records.append(record)
            continue
        if out_path.exists() and not args.force:
            print(f"    exists, skipped (--force overwrites)")
            continue

        import tifffile
        native, colour_error = build_projection(tiles_by_z, (canvas_h, canvas_w), weights)
        if colour_error > COLOUR_FIT_MAX_ERROR:
            sys.exit(f"{name}: tiles do not look like {channel_name} alone in colour "
                     f"{meta['channels'][idx]['colour']} (mean error {colour_error:.2f} levels)")
        small = imresize_bilinear(native, out_hw)
        out_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(out_path), small)
        record["colour_fit_error"] = round(float(colour_error), 3)
        record["native_max"] = int(native.max())
        records.append(record)
        print(f"    wrote {out_path.name}  {small.shape} {small.dtype}, "
              f"native max {native.max()}, colour fit error {colour_error:.2f}")

    if args.dry_run or not records:
        return
    sidecar = out_dir / "confocal_sections.json"
    previous = json.loads(sidecar.read_text()).get("sections", {}) if sidecar.exists() else {}
    previous.update({r["section"]: r for r in records})
    sidecar.write_text(json.dumps({
        "written": datetime.now().isoformat(timespec="seconds"),
        "confocal_dir": str(confocal_dir), "sections": previous,
    }, indent=2))
    print(f"\nWrote {len(records)} section(s); record: {sidecar}")


if __name__ == "__main__":
    main()
