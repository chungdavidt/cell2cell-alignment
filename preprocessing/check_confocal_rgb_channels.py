#!/usr/bin/env python3
"""
Which stains a ZEN TIFF export holds: the XML's channel list against the pixels
of the stitched S{k}.{n}.tiff and of its tiles.

Per sub-stack:
  1. XML     image channels (Id "Channel:k": name, display colour), SizeC, and
             every other <Channel> element naming a dye, with its element path
             so an acquisition-setup entry is distinguishable from an exported
             channel.
  2. stitched S{k}.{n}.tiff, RGB: per component max / mean / nonzero pixels.
             Then whether G and B are fixed functions of R:
               - for each R value, how many distinct G (B) values occur;
                 one channel in one colour gives 1-2 (rounding), a second
                 stain in that component gives many
               - least-squares slope G/R and B/R vs the ratio the XML colours
                 predict for the exported channel(s)
               - fraction of pixels more than 2 levels off slope * R
  3. tiles   prefixes, c / z indices, count; one tile per c read: shape, dtype,
             max, and mean |G-R|, |B-R| when RGB (0 = grayscale = one channel).

Reads only. Run in .castalign-venv.

Usage:
    python preprocessing/check_confocal_rgb_channels.py                 # every sub-stack of CONFOCAL_DIR
    python preprocessing/check_confocal_rgb_channels.py --section S1.1
    python preprocessing/check_confocal_rgb_channels.py --dir "<export folder>" --section S1.1
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT, _ROOT / "preprocessing"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import local_config  # noqa: E402
from prepare_confocal_sections import TILE_RE, discover_sections, read_tiff_as_is  # noqa: E402

DYES = ("dapi", "tdtom", "egfp", "gfp", "af647", "af488", "af555", "af594", "cy3", "cy5", "mscarlet")


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def hex_rgb(colour):
    """'#FFFF4100' or '#FF4100' -> (255, 65, 0); None if unreadable."""
    if not colour:
        return None
    h = colour.lstrip("#")
    if len(h) == 8:      # ZEN writes #AARRGGBB
        h = h[2:]
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def xml_report(xml: Path):
    root = ET.parse(xml).getroot()
    image_channels, other = {}, Counter()

    def walk(el, trail):
        name = _local(el.tag)
        if name == "Channel":
            cid, cname = el.get("Id") or "", el.get("Name") or ""
            kids = {_local(c.tag): (c.text or "").strip() for c in el}
            if cid.startswith("Channel:") and cid not in image_channels:
                image_channels[cid] = (cname, kids.get("Color"), kids.get("Fluor") or kids.get("DyeName"))
            elif not cid.startswith("Channel:"):
                other[(cname, "/".join(trail[-3:]))] += 1
        for c in el:
            walk(c, trail + [name])

    walk(root, [])
    size = {}
    for el in root.iter():
        if _local(el.tag) == "Image":
            for c in el:
                n = _local(c.tag)
                if n in ("SizeC", "SizeZ", "SizeX", "SizeY", "SizeM") and n not in size and c.text:
                    size[n] = int(c.text)
    print(f"  XML {xml.name}")
    print(f"    image Size*: {size}")
    print(f"    exported image channels (Id 'Channel:k'):")
    for cid, (n, col, fluor) in sorted(image_channels.items()):
        print(f"      {cid}: name {n!r}, colour {col} -> RGB {hex_rgb(col)}, fluor {fluor!r}")
    dye_named = {k: v for k, v in other.items() if any(d in k[0].lower() for d in DYES)}
    print(f"    other <Channel> elements naming a dye (name, parent path) x count:")
    for (n, path), cnt in sorted(dye_named.items()):
        print(f"      {n!r} under {path} x{cnt}")
    hits = [el for el in root.iter() if (el.text or "").strip().lower() in ("dapi", "dapi-t4")
            or (el.get("Name") or "").lower().startswith("dapi")]
    print(f"    elements whose Name or text is DAPI: {len(hits)}")
    return image_channels


def stitched_report(path: Path, image_channels: dict):
    import numpy as np
    if not path.exists():
        print(f"  stitched {path.name}: not found")
        return
    a = read_tiff_as_is(path)
    print(f"  stitched {path.name}: shape {a.shape} {a.dtype}")
    if a.ndim != 3 or a.shape[-1] < 3:
        print("    not RGB; nothing to split")
        return
    R, G, B = (a[..., i].astype(np.int64) for i in range(3))
    for c, x in zip("RGB", (R, G, B)):
        print(f"    {c}: max {x.max():4d}  mean {x.mean():7.3f}  nonzero px {int((x > 0).sum()):>10d}")
    cols = [hex_rgb(col) for _, col, _ in image_channels.values() if hex_rgb(col)]
    if len(cols) == 1 and cols[0][0]:
        r0, g0, b0 = cols[0]
        print(f"    predicted from the one XML colour {cols[0]}: G/R {g0 / r0:.4f}, B/R {b0 / r0:.4f}")
    elif cols:
        print(f"    XML colours of {len(cols)} channels: {cols} (a sum of these, no single ratio)")
    on = R > 0
    for c, x in (("G", G), ("B", B)):
        if not on.any():
            break
        r, y = R[on], x[on]
        slope = float((r * y).sum() / (r * r).sum())
        off = np.abs(y - slope * r) > 2
        pairs = np.unique(np.stack([r, y], 1), axis=0)
        per_r = Counter(pairs[:, 0].tolist())
        worst = max(per_r.values())
        print(f"    {c} vs R (pixels with R > 0): slope {slope:.4f}; "
              f"distinct {c} values per R value: median {int(np.median(list(per_r.values())))}, "
              f"max {worst}; >2 levels off slope*R: {100 * off.mean():.4f}%")
        only = int(((x > 0) & ~on).sum())
        print(f"      pixels with {c} > 0 and R = 0: {only}")


def tiles_report(tile_dir: Path):
    import numpy as np
    if not tile_dir.is_dir():
        print(f"  tiles: folder not found {tile_dir.name}")
        return
    files = [p for p in tile_dir.iterdir() if p.is_file()]
    hits = [(p, TILE_RE.match(p.name)) for p in files]
    hits = [(p, m) for p, m in hits if m]
    print(f"  tiles {tile_dir.name}: {len(files)} files, {len(hits)} tile images; "
          f"prefixes {sorted({m.group('sec') for _, m in hits})}, "
          f"c {dict(Counter(int(m.group('c')) for _, m in hits))}, "
          f"z {sorted({int(m.group('z')) for _, m in hits})}")
    for c in sorted({int(m.group("c")) for _, m in hits}):
        p = sorted(p for p, m in hits if int(m.group("c")) == c)[0]
        t = read_tiff_as_is(p)
        line = f"    c{c} {p.name}: shape {t.shape} {t.dtype} max {t.max()}"
        if t.ndim == 3 and t.shape[-1] >= 3:
            r = t[..., 0].astype(np.int64)
            line += (f", mean |G-R| {np.abs(t[..., 1] - r).mean():.3f}, "
                     f"mean |B-R| {np.abs(t[..., 2] - r).mean():.3f}, "
                     f"max R/G/B {t[..., 0].max()}/{t[..., 1].max()}/{t[..., 2].max()}")
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="export folder; default local_config.CONFOCAL_DIR")
    ap.add_argument("--section", nargs="+", metavar="S1.1", help="only these sub-stacks")
    args = ap.parse_args()
    d = Path(args.dir or getattr(local_config, "CONFOCAL_DIR", ""))
    if not d.is_dir():
        sys.exit(f"export folder not found: {d}")
    sections = [s for s in discover_sections(d) if not args.section or s[2] in args.section]
    if not sections:
        sys.exit(f"no matching S{{k}}.{{n}}.tiff_metadata.xml in {d}")
    print(f"{d}\n")
    for _, _, name, xml, tile_dir in sections:
        print(f"=== {name} ===")
        chans = xml_report(xml)
        stitched_report(d / f"{name}.tiff", chans)
        tiles_report(tile_dir)
        print()


if __name__ == "__main__":
    main()
