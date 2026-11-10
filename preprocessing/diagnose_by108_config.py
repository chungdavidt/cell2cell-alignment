#!/usr/bin/env python3
"""
Read-only diagnosis of the current local_config.py against the files it names,
the outputs already written, and the graph. Written for BY108 after the config
moved to the confocal re-export ("redo full slice").

Sections:
  1. config      every changed name vs a reference commit (default fff35c4,
                 the last BY108 config), plus BY95 values that no longer apply.
  2. 2P pitch    SCOPE / BLOCK_STACK_SCOPE, each stack's shape from its header,
                 the field of view each label implies, and which file the
                 builder would load (raw or the 2p_downsampled copy).
  3. red/green   per 2P volume: shapes, and the correlation of a block-averaged
                 middle plane of red against green as-is, X-flipped and
                 Y-flipped (whether the two channels share one frame).
  4. confocal    per sub-stack of CONFOCAL_DIR: every channel in the XML, Scaling,
                 Size*, tile names vs prepare_confocal_sections.TILE_RE, tile
                 c/z/m indices, tile shape/dtype/grayscale, canvas vs SizeX/Y;
                 then the same summary for the previous export, side by side.
  5. outputs     what is already in <ANALYSIS_ROOT>/preprocessing/confocal/<label>/
                 and which export it came from.
  6. graph       nodes in GRAPH_PATH, their recorded source/shape, and which ones
                 the builder would skip by name.

Reads only; writes nothing. Run in .castalign-venv (tifffile, numpy; castalign
for section 6). Does not import preprocessing_config.

Usage:
    python preprocessing/diagnose_by108_config.py
    python preprocessing/diagnose_by108_config.py --no-pixels --no-graph
    python preprocessing/diagnose_by108_config.py --ref-commit fff35c4 --old-confocal-dir "<dir>"
"""

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT, _ROOT / "preprocessing"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import local_config  # noqa: E402
from analysis_paths import (confocal_label, confocal_output_dir,  # noqa: E402
                            confocal_tifs, downsampled_2p_path)
from scope_profiles import get_profile  # noqa: E402
from prepare_confocal_sections import (TILE_RE, discover_sections,  # noqa: E402
                                       read_zen_xml, read_tiff_as_is)

FLAG = "  !! "
OK = "     "


def header(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def cfg(name, default=None):
    return getattr(local_config, name, default)


# ---------------------------------------------------------------------------
# 1. config
# ---------------------------------------------------------------------------

def config_at_commit(commit: str) -> dict:
    """Top-level literal assignments in local_config.py at `commit`."""
    out = subprocess.run(["git", "-C", str(_ROOT), "show", f"{commit}:local_config.py"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    vals = {}
    for node in ast.parse(out.stdout).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                vals[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return vals


def section_config(ref_commit: str) -> dict:
    header(f"1. local_config.py vs {ref_commit}")
    try:
        ref = config_at_commit(ref_commit)
    except Exception as e:
        print(f"{FLAG}could not read {ref_commit}: {e}")
        return {}
    names = sorted(n for n in set(ref) | {k for k in dir(local_config) if k.isupper()})
    changed = [n for n in names if ref.get(n, "<absent>") != cfg(n, "<absent>")]
    if not changed:
        print(f"{OK}no differences")
    for n in changed:
        print(f"  {n}\n      {ref_commit}: {ref.get(n, '<absent>')!r}\n      now:     {cfg(n, '<absent>')!r}")

    print()
    if not cfg("DATA_ROOT"):
        carried = {n: cfg(n) for n in ("QC_MIN_READS", "QC_MIN_GENES", "ALIGN_MIN_ROLONIES",
                                       "SUBSLICE_DIR", "SUBSLICE_RAW_CHANNELS")
                   if cfg(n) not in (None, "", [])}
        if carried:
            print(f"{FLAG}DATA_ROOT is blank (no BARseq) but BARseq names are set: {carried}")
            print(f"     The builder skips them while DATA_ROOT is blank; they are BY95's values.")
    return ref


# ---------------------------------------------------------------------------
# 2. 2P pitch and 3. red/green
# ---------------------------------------------------------------------------

def tiff_shape(path: Path):
    import tifffile
    with tifffile.TiffFile(str(path)) as t:
        s = t.series[0]
        return tuple(s.shape), str(s.dtype), len(t.pages)


def section_2p_pitch():
    header("2. 2P pitch: SCOPE / BLOCK_STACK_SCOPE vs stack shapes")
    scope = cfg("SCOPE", "")
    block_scope = cfg("BLOCK_STACK_SCOPE", "") or scope
    print(f"  SCOPE             = {scope!r}  ({get_profile(scope)['xy_um_per_px']} um/px)")
    print(f"  BLOCK_STACK_SCOPE = {cfg('BLOCK_STACK_SCOPE', '')!r}"
          f"  -> block read at {block_scope!r} ({get_profile(block_scope)['xy_um_per_px']} um/px)")
    fields = {}
    for var, sc in [("INVIVO_PATH_RED", scope), ("INVIVO_PATH_GREEN", scope),
                    ("BLOCK_STACK_PATH_RED", block_scope), ("BLOCK_STACK_PATH_GREEN", block_scope)]:
        raw = cfg(var, "")
        if not raw:
            print(f"  {var}: blank")
            continue
        path = Path(raw)
        if not path.exists():
            print(f"{FLAG}{var}: not found: {path}")
            continue
        shape, dtype, pages = tiff_shape(path)
        xy = get_profile(sc)["xy_um_per_px"]
        field = shape[-1] * xy
        fields[var] = field
        print(f"  {var}: {path.name}\n      shape {shape} {dtype}, {pages} pages -> "
              f"{shape[-1]} px x {xy:.5f} = {field:.1f} um field at {sc!r}")
        target = get_profile(scope)["xy_um_per_px"]
        if abs(xy / target - 1) >= 0.005:
            copy = downsampled_2p_path(path)
            src = None
            if copy is not None and copy.with_suffix(".json").exists():
                src = json.loads(copy.with_suffix(".json").read_text()).get("source")
            state = ("missing" if copy is None or not copy.exists()
                     else "made from this file" if src == str(path)
                     else f"made from {src!r} -> builder raises")
            print(f"      finer than SCOPE: builder loads {copy} ({state})")
        else:
            print(f"      same pitch as SCOPE: builder loads the raw file")
    inv, blk = fields.get("INVIVO_PATH_RED"), fields.get("BLOCK_STACK_PATH_RED")
    if inv and blk:
        ratio = blk / inv
        mark = FLAG if abs(ratio - 1) > 0.05 else OK
        print(f"\n{mark}block field / in vivo field = {blk:.1f} / {inv:.1f} um = {ratio:.3f}")
        if abs(ratio - 1) > 0.05:
            print(f"     2026-10-14 record: block taken at the in vivo zoom, so its field equals"
                  f" in vivo's;\n     a 2x here means the block's pitch label is off by 2x "
                  f"(huang_lab_bruker_1024px was BLOCK_STACK_SCOPE then).")
    stale = downsampled_2p_path(Path(cfg("BLOCK_STACK_PATH_RED") or "x"))
    if stale is not None and stale.parent.is_dir():
        print(f"\n  {stale.parent}:")
        for j in sorted(stale.parent.glob("*.json")):
            print(f"      {j.with_suffix('.tif').name}  <- {json.loads(j.read_text()).get('source')}")


def block_mean(a, k=8):
    import numpy as np
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def corr(a, b):
    import numpy as np
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else float("nan")


def section_red_green():
    import numpy as np
    import tifffile
    header("3. red vs green: same frame? (middle plane, 8x8 block mean, Pearson r)")
    print("     r is highest for the flip that puts green in red's frame; it measures "
          "shared structure\n     (vessels, background, bleed-through), not cell identity.")
    for vol, rv, gv in [("invivo", "INVIVO_PATH_RED", "INVIVO_PATH_GREEN"),
                        ("block_stack", "BLOCK_STACK_PATH_RED", "BLOCK_STACK_PATH_GREEN")]:
        r, g = cfg(rv, ""), cfg(gv, "")
        if not (r and g and Path(r).exists() and Path(g).exists()):
            print(f"  {vol}: red or green blank/missing, skipped")
            continue
        rs, _, rp = tiff_shape(Path(r))
        gs, _, gp = tiff_shape(Path(g))
        print(f"\n  {vol}: red {Path(r).name} {rs}\n  {' ' * len(vol)}  green {Path(g).name} {gs}")
        if rs != gs:
            print(f"{FLAG}shapes differ: an Identity edge between them is not pixel-to-pixel")
        with tifffile.TiffFile(r) as tr, tifffile.TiffFile(g) as tg:
            if rp != gp:
                print(f"{FLAG}page counts differ ({rp} vs {gp}); comparing page {min(rp, gp) // 2}")
            k = min(rp, gp) // 2
            a = block_mean(tr.pages[k].asarray().astype(np.float64))
            b = block_mean(tg.pages[k].asarray().astype(np.float64))
        if a.shape != b.shape:
            print(f"{FLAG}page shapes differ: {a.shape} vs {b.shape}")
            continue
        scores = {"as-is": corr(a, b), "green X-flipped": corr(a, b[:, ::-1]),
                  "green Y-flipped": corr(a, b[::-1, :]), "green XY-flipped": corr(a, b[::-1, ::-1])}
        if a.shape[0] == a.shape[1]:
            scores["green transposed"] = corr(a, b.T)
        best = max(scores, key=lambda s: scores[s])
        for s, v in scores.items():
            print(f"      {s:18s} r = {v:+.3f}{'   <- best' if s == best else ''}")
        if best != "as-is":
            print(f"{FLAG}green matches red best when {best}: the Identity edge "
                  f"{vol}_red <-> {vol}_green would mirror green")


# ---------------------------------------------------------------------------
# 4. confocal export
# ---------------------------------------------------------------------------

def xml_channel_elements(xml: Path) -> list:
    """Every <Channel> element: Id, Name and short child texts naming the dye."""
    import xml.etree.ElementTree as ET
    out = []
    for ch in ET.parse(xml).getroot().iter():
        if ch.tag.rsplit("}", 1)[-1] != "Channel":
            continue
        kids = {c.tag.rsplit("}", 1)[-1]: (c.text or "").strip() for c in ch}
        info = {k: v for k, v in kids.items()
                if k in ("Fluor", "DyeName", "Color", "ExcitationWavelength",
                         "EmissionWavelength", "AcquisitionMode", "IlluminationType")
                and v}
        out.append((ch.get("Id"), ch.get("Name"), info))
    return out


def describe_export(confocal_dir: Path, channel_name: str, read_pixels: bool,
                    verbose: bool) -> dict:
    """Per-sub-stack summary; prints detail when verbose."""
    import numpy as np
    summary = {}
    sections = discover_sections(confocal_dir)
    if verbose:
        others = Counter(p.suffix.lower() or "(dir)" for p in confocal_dir.iterdir())
        print(f"  {confocal_dir}\n  entries by suffix: {dict(others)}")
        print(f"  sub-stacks matched by S{{k}}.{{n}}.tiff_metadata.xml: {[s[2] for s in sections]}")
        unmatched = [p.name for p in confocal_dir.glob("*_metadata.xml")
                     if p.name not in {s[3].name for s in sections}]
        if unmatched:
            print(f"{FLAG}metadata XMLs not matched by the section pattern: {unmatched}")
    for slide, n, name, xml, tile_dir in sections:
        meta = read_zen_xml(xml)
        chans = {k: c["name"] for k, c in meta["channels"].items()}
        rec = {"channels": chans, "scaling_um": meta["scaling_um"], "size": meta["size"]}
        if verbose:
            print(f"\n  {name}  (slice{slide} part{n})")
            print(f"      image channels (Id 'Channel:k', what prepare_confocal_sections reads): {chans}")
            hit = [k for k, v in chans.items() if v == channel_name]
            print(f"{OK if hit else FLAG}CONFOCAL_CHANNEL {channel_name!r} -> "
                  f"{'c' + str(hit[0]) if hit else 'no match'}")
            names = Counter((cid, nm) for cid, nm, _ in xml_channel_elements(xml))
            print(f"      all <Channel> elements (Id, Name) x count: {dict(names)}")
            print(f"      Scaling um: {meta['scaling_um']}   Size: {meta['size']}")
        if not tile_dir.is_dir():
            rec["tiles"] = None
            if verbose:
                print(f"{FLAG}tile folder missing: {tile_dir.name}")
                stitched = confocal_dir / f"{name}.tiff"
                if stitched.exists():
                    print(f"      stitched {stitched.name}: {tiff_shape(stitched)}")
            summary[name] = rec
            continue
        files = sorted(p for p in tile_dir.iterdir() if p.is_file())
        parsed = [(p, TILE_RE.match(p.name)) for p in files]
        good = [(p, m) for p, m in parsed if m and m.group("sec") == name]
        bad = [p.name for p, m in parsed if not (m and m.group("sec") == name)]
        cs = Counter(int(m.group("c")) for _, m in good)
        zs = sorted({int(m.group("z")) for _, m in good})
        ms = sorted({int(m.group("m")) for _, m in good})
        rec.update(tiles=len(good), unmatched=len(bad), c=dict(cs), z=zs, m=len(ms))
        if good:
            w = max(int(m.group("x")) + int(m.group("w")) for _, m in good)
            h = max(int(m.group("y")) + int(m.group("h")) for _, m in good)
            rec["canvas_xy"] = (w, h)
        if verbose:
            print(f"      tiles: {len(files)} files, {len(good)} match TILE_RE; "
                  f"c {dict(cs)}, z {zs}, {len(ms)} mosaic positions")
            if bad:
                print(f"{FLAG}{len(bad)} tile names do not match TILE_RE, e.g. {bad[:3]}")
            if good:
                size = (meta["size"].get("SizeX"), meta["size"].get("SizeY"))
                mark = OK if size == rec["canvas_xy"] else FLAG
                print(f"{mark}tile canvas {rec['canvas_xy']} vs XML SizeX/SizeY {size}")
            if missing := sorted(set(chans) - set(cs)):
                print(f"{FLAG}XML channels with no tiles: {missing}")
            if extra := sorted(set(cs) - set(chans)):
                print(f"{FLAG}tile c-indices with no XML channel: {extra}")
        if read_pixels and good:
            p = good[0][0]
            a = read_tiff_as_is(p)
            rec["tile0"] = (tuple(a.shape), str(a.dtype))
            line = f"      first tile {p.name}: shape {a.shape} {a.dtype}, max {a.max()}"
            if a.ndim == 3 and a.shape[-1] in (3, 4):
                r = a[..., 0].astype(np.int32)
                err = max(np.abs(a[..., i].astype(np.int32) - r).mean() for i in (1, 2))
                line += f", mean |G-R| or |B-R| {err:.2f}"
                rec["gray_err"] = round(float(err), 2)
            elif a.ndim == 2:
                line += "  (single plane, not RGB: build_projection expects (H, W, 3))"
            if verbose:
                print(line)
        summary[name] = rec
    return summary


def section_confocal(read_pixels: bool, old_dir) -> None:
    header("4. confocal export: CONFOCAL_DIR")
    cdir, ch = cfg("CONFOCAL_DIR", ""), cfg("CONFOCAL_CHANNEL", "")
    if not cdir or not Path(cdir).is_dir():
        print(f"{FLAG}CONFOCAL_DIR blank or not found: {cdir!r}")
        return
    new = describe_export(Path(cdir), ch, read_pixels, verbose=True)
    names = sorted({v for r in new.values() for v in r["channels"].values()})
    print(f"\n  channel names across this export: {names}")
    if ch not in names and names:
        print(f"{FLAG}CONFOCAL_CHANNEL = {ch!r} is in none of them")

    if old_dir is None or not Path(old_dir).is_dir():
        print(f"\n  previous export: not given or not found ({old_dir!r}); comparison skipped")
        return
    if Path(old_dir).resolve() == Path(cdir).resolve():
        return
    print(f"\n  previous export: {old_dir}")
    old = describe_export(Path(old_dir), ch, read_pixels, verbose=False)
    keys = sorted(set(old) | set(new), key=lambda s: tuple(int(x) for x in s[1:].split(".")))
    print(f"\n  {'sub-stack':9s} | {'previous':58s} | now")
    for k in keys:
        def short(r):
            if r is None:
                return "-"
            s = (f"ch {list(r['channels'].values())} xy "
                 f"{r['scaling_um'].get('X', float('nan')):.4f} z {r['scaling_um'].get('Z', float('nan')):.2f} "
                 f"{r['size'].get('SizeX')}x{r['size'].get('SizeY')} Z{r['size'].get('SizeZ')}")
            if r.get("tiles") is not None:
                s += f" tiles {r['tiles']} z{r.get('z')}"
            if r.get("tile0"):
                s += f" {r['tile0'][0]} {r['tile0'][1]}"
            return s
        print(f"  {k:9s} | {short(old.get(k)):58s} | {short(new.get(k))}")


# ---------------------------------------------------------------------------
# 5. outputs and 6. graph
# ---------------------------------------------------------------------------

def section_outputs():
    header("5. confocal outputs already written")
    ch = cfg("CONFOCAL_CHANNEL", "")
    out = confocal_output_dir(ch) if ch else None
    if out is None:
        print(f"{FLAG}ANALYSIS_ROOT or CONFOCAL_CHANNEL blank")
        return
    print(f"  {out}  (label {confocal_label(ch)!r} from CONFOCAL_CHANNEL {ch!r})")
    if not out.is_dir():
        print(f"{OK}does not exist yet")
        return
    tifs = confocal_tifs(out)
    print(f"  {len(tifs)} sub-stack tif(s): {[t.stem for t in tifs]}")
    side = out / "confocal_sections.json"
    if side.exists():
        rec = json.loads(side.read_text())
        print(f"  confocal_sections.json written {rec.get('written')} from {rec.get('confocal_dir')}")
        dirs = Counter(Path(r.get("source_xml", "")).parent.as_posix() for r in rec.get("sections", {}).values())
        print(f"  per-record source folders: {dict(dirs)}")
        if rec.get("confocal_dir") != cfg("CONFOCAL_DIR"):
            print(f"{FLAG}outputs came from a different CONFOCAL_DIR. prepare_confocal_sections.py "
                  f"skips an existing\n     output without --force (keeping the old export's "
                  f"pixels) and merges records into this json;\n     sub-stacks absent from the new "
                  f"export stay on disk and the builder ingests them.")
        for r in rec.get("sections", {}).values():
            print(f"      {r.get('node')}: native {r.get('native_shape')} -> {r.get('output_shape')}, "
                  f"factor {r.get('factor', 0):.4f}, graph_scope {r.get('graph_scope')}")


def section_graph():
    header("6. graph: GRAPH_PATH")
    path = cfg("GRAPH_PATH", "")
    if not path:
        print("  GRAPH_PATH blank: builder derives it; skipped")
        return
    if not Path(path).exists():
        print(f"{OK}{path} does not exist; the builder creates it")
        return
    try:
        import castalign as ca
    except ImportError as e:
        print(f"{FLAG}castalign not importable ({e}); run in .castalign-venv")
        return
    g = ca.Graph.load(str(path))
    print(f"  {path}: {len(g.nodes)} nodes")
    md = g.node_metadata
    for n in g.nodes:
        m = md.get(n, {}) or {}
        keys = {k: m[k] for k in ("spacing", "shape", "source", "extra_downsample", "modality")
                if k in m}
        print(f"      {n}: {keys}")
    edges = sorted((a, b) for a in g.edges for b in g.edges[a])
    print(f"  edges: {edges}")
    print("\n     The builder skips invivo_*/block_stack_* by name, with no source or shape check:")
    print("     a node built from an earlier file stays in the graph after the config path changes.")
    for node, var in [("block_stack_red", "BLOCK_STACK_PATH_RED"), ("invivo_red", "INVIVO_PATH_RED"),
                      ("block_stack_green", "BLOCK_STACK_PATH_GREEN"),
                      ("invivo_green", "INVIVO_PATH_GREEN")]:
        if node in g.nodes and cfg(var):
            print(f"{FLAG}{node} is in the graph; {var} = {Path(cfg(var)).name} would not replace it")
    cdir = cfg("CONFOCAL_DIR", "")
    if cdir and Path(cdir).is_dir():
        live = {f"slice{s}_confocal_{confocal_label(cfg('CONFOCAL_CHANNEL', ''))}_part{n}"
                for s, n, *_ in discover_sections(Path(cdir))}
        conf = [n for n in g.nodes if "_confocal_" in n]
        gone = [n for n in conf if n not in live]
        if gone:
            print(f"{FLAG}confocal nodes with no sub-stack in the current CONFOCAL_DIR: {gone}")
        print(f"     confocal nodes whose name the new export reuses (pixels from the old export "
              f"unless replaced): {sorted(set(conf) & live)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-commit", default="fff35c4",
                    help="commit whose local_config.py to compare against (default fff35c4)")
    ap.add_argument("--old-confocal-dir", default=None,
                    help="previous confocal export; default CONFOCAL_DIR at --ref-commit")
    ap.add_argument("--no-pixels", action="store_true",
                    help="headers and XML only: skip the red/green planes and the tile reads")
    ap.add_argument("--no-graph", action="store_true", help="skip loading GRAPH_PATH")
    args = ap.parse_args()

    ref = section_config(args.ref_commit)
    section_2p_pitch()
    if not args.no_pixels:
        section_red_green()
    section_confocal(not args.no_pixels, args.old_confocal_dir or ref.get("CONFOCAL_DIR"))
    section_outputs()
    if not args.no_graph:
        section_graph()


if __name__ == "__main__":
    main()
