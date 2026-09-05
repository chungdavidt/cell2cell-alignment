"""
Read-only probe: dump every scrap of metadata a TIFF carries.

Written for one question -- does <HYB_ROOT>/<FOV>/alignedn2vhyb01.tif say
anywhere which of its 6 pages is mScarlet? -- but it takes any TIFF path, so the
2P stacks in local_config answer the same way.

utilities/image_io.py `load_fov_images` hardcodes page 0 = GCAMP, 3 = mScarlet,
4 = DAPI, indices carried over from JH302. filt_neurons.mat corroborates them for
BY95 -- the 114x2 `genes` cell holds channel numbers 1/2/4 in column 2 for expmat
columns 111/112/113 (GCaMP / unknown / mScarlet) where every other row holds a
7-mer barcode -- but the TIFF itself has never been asked.

Per file it prints:
  format flags   is_imagej / is_ome / is_scanimage / ... , byte order, BigTIFF
  series         axes, shape, dtype, pages per series
  metadata       ImageJ (Info, Labels, LUTs, channels=), OME-XML <Channel> names,
                 ScanImage, MicroManager (ChNames), tifffile "shaped"
  page tags      every tag of every page: code, name, dtype, count, value
  keyword sweep  every metadata string searched for channel vocabulary
                 (mscarlet, gcamp, dapi, tdtomato, 561, ch2, ...) with its source
  siblings       other files in the same directory, since a channel name absent
                 from the TIFF is often in a companion file

Writes nothing unless --json is given.

Usage:
    python preprocessing/check_tiff_metadata.py                  # first FOV under HYB_ROOT
    python preprocessing/check_tiff_metadata.py -n 3             # compare 3 FOVs
    python preprocessing/check_tiff_metadata.py "C:\\path\\to\\any.tif"
    python preprocessing/check_tiff_metadata.py --full           # no value truncation
    python preprocessing/check_tiff_metadata.py --dump-text      # print small sibling text files
    python preprocessing/check_tiff_metadata.py --json meta.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tifffile

HYB_TIF_NAME = "alignedn2vhyb01.tif"

# tifffile format flags worth reporting; absent attributes are skipped, since
# the set differs between tifffile versions.
FORMAT_FLAGS = (
    "is_imagej", "is_ome", "is_shaped", "is_scanimage", "is_micromanager",
    "is_lsm", "is_stk", "is_fluoview", "is_nih", "is_andor", "is_philips",
    "is_ndpi", "is_svs", "is_qpi", "is_bigtiff", "is_mmstack",
)

# Named metadata blocks, in the order they are printed.
METADATA_BLOCKS = (
    ("imagej_metadata", "ImageJ"),
    ("ome_metadata", "OME-XML"),
    ("scanimage_metadata", "ScanImage"),
    ("micromanager_metadata", "MicroManager"),
    ("shaped_metadata", "tifffile shaped"),
    ("stk_metadata", "STK"),
    ("lsm_metadata", "LSM"),
    ("fluoview_metadata", "FluoView"),
    ("nih_metadata", "NIH Image"),
)

# Tags printed in full however long they are -- these are where a name hides.
VERBOSE_TAGS = {
    "ImageDescription", "PageName", "Software", "Artist", "DateTime",
    "HostComputer", "Make", "Model", "Copyright", "DocumentName",
    "TargetPrinter", "InkNames", "UniqueCameraModel", "CameraSerialNumber",
}

# Tags that are per-strip offset/length arrays -- long, and never a channel name.
BULK_TAGS = {"StripOffsets", "StripByteCounts", "TileOffsets", "TileByteCounts",
             "ColorMap", "TransferFunction"}

# Channel vocabulary for the keyword sweep. A hit is reported with its source
# and surrounding text; a miss is the answer that the file names nothing.
TOKENS = (
    "mscarlet", "scarlet", "mcherry", "tdtomato", "tdtom", "rfp", "gcamp",
    "gfp", "yfp", "egfp", "dapi", "hoechst", "cy3", "cy5", "alexa", "atto",
    "channel", "chan", "wavelength", "emission", "excitation", "laser",
    "filter", "lut", "405", "488", "561", "594", "640", "647",
)

TEXT_SUFFIXES = {".txt", ".json", ".xml", ".csv", ".log", ".ini", ".cfg",
                 ".yaml", ".yml", ".md"}

MAX_VALUE_CHARS = 400
MAX_ARRAY_ITEMS = 8
MAX_TEXT_BYTES = 64_000


def as_text(value):
    """A tag or metadata value as a printable string."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def shorten(text, full):
    text = text.replace("\r\n", "\n")
    if full or len(text) <= MAX_VALUE_CHARS:
        return text
    return f"{text[:MAX_VALUE_CHARS]} ... [{len(text)} chars total, --full to see it all]"


def indent(text, pad="      "):
    """Multi-line values line up under their tag name."""
    lines = text.split("\n")
    if len(lines) == 1:
        return text
    return ("\n" + pad).join(lines)


def format_tag_value(tag, full):
    """Tag value as a string, with bulk arrays cut down to their first items."""
    value = tag.value
    name = tag.name

    if name in BULK_TAGS and hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        items = list(value)[:MAX_ARRAY_ITEMS]
        more = "" if len(value) <= MAX_ARRAY_ITEMS else f" ... (+{len(value) - MAX_ARRAY_ITEMS} more)"
        return f"[{', '.join(str(i) for i in items)}{more}]"

    text = as_text(value)
    if name in VERBOSE_TAGS:
        return indent(text if full else shorten(text, full))
    return indent(shorten(text, full))


def walk_strings(obj, source, out, key=None):
    """Collect (source, key, string) from any nested metadata structure."""
    label = f"{source}" if key is None else f"{source}:{key}"
    if isinstance(obj, bytes):
        out.append((label, obj.decode("utf-8", errors="replace")))
    elif isinstance(obj, str):
        out.append((label, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.append((label, str(k)))
            walk_strings(v, source, out, key=k if key is None else f"{key}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            walk_strings(v, source, out, key=f"{key or ''}[{i}]")


def ome_channels(xml):
    """<Channel> Name/ID/Fluor attributes, which is the one place a TIFF names
    its channels in a standard way."""
    found = []
    for match in re.finditer(r"<Channel\b([^>]*)/?>", xml):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
        found.append(attrs)
    return found


def channel_names_by_page(tf):
    """[(page index, name, where it came from)] for every page a name can be
    attached to. The four places a TIFF can name a plane; empty means it does
    not name them at all."""
    found = []

    imagej = getattr(tf, "imagej_metadata", None) or {}
    for i, label in enumerate(imagej.get("Labels") or []):
        found.append((i, as_text(label), "ImageJ Labels"))

    ome = getattr(tf, "ome_metadata", None)
    if ome:
        # <Channel> order is page order only while the series is channel-major;
        # the series line above says which axes it actually has.
        for i, attrs in enumerate(ome_channels(ome)):
            name = attrs.get("Name") or attrs.get("Fluor") or attrs.get("ID")
            if name:
                found.append((i, name, "OME <Channel>"))

    mm = getattr(tf, "micromanager_metadata", None) or {}
    summary = mm.get("Summary") if isinstance(mm, dict) else None
    for i, name in enumerate((summary or {}).get("ChNames") or []):
        found.append((i, as_text(name), "MicroManager ChNames"))

    for i, page in enumerate(tf.pages):
        page = page.aspage() if hasattr(page, "aspage") else page
        tag = page.tags.get("PageName")
        if tag is not None:
            found.append((i, as_text(tag.value), "PageName tag"))

    return found


def probe_file(path, full=False):
    """Everything one TIFF says about itself. Returns a dict for --json."""
    print("=" * 70)
    print(path)
    print("=" * 70)
    size = path.stat().st_size
    print(f"  size: {size / 1e6:.1f} MB")

    record = {"path": str(path), "size_bytes": size, "pages": [], "metadata": {}}
    strings = []

    with tifffile.TiffFile(path) as tf:
        flags = [name for name in FORMAT_FLAGS if getattr(tf, name, False)]
        print(f"  byteorder: {tf.byteorder}   flags: {', '.join(flags) if flags else 'none'}")
        record["flags"] = flags
        record["byteorder"] = tf.byteorder

        print(f"\n  --- series ({len(tf.series)}) ---")
        for i, series in enumerate(tf.series):
            axes = getattr(series, "axes", "?")
            kind = getattr(series, "kind", "?")
            print(f"    series {i}: shape={series.shape} axes={axes} "
                  f"dtype={series.dtype} kind={kind} pages={len(series.pages)}")
            record.setdefault("series", []).append(
                {"shape": list(series.shape), "axes": axes,
                 "dtype": str(series.dtype), "kind": str(kind)})

        for attr, label in METADATA_BLOCKS:
            try:
                meta = getattr(tf, attr, None)
            except Exception as exc:                      # some blocks parse lazily
                print(f"\n  --- {label} --- read FAILED: {type(exc).__name__}: {exc}")
                continue
            if not meta:
                continue
            print(f"\n  --- {label} metadata ---")
            record["metadata"][label] = as_text(meta) if isinstance(meta, (str, bytes)) else repr(meta)
            walk_strings(meta, label, strings)
            if isinstance(meta, str):
                print(f"    {indent(shorten(meta, full), '    ')}")
                if label == "OME-XML":
                    for ch in ome_channels(meta):
                        print(f"    <Channel> {ch}")
                    record["ome_channels"] = ome_channels(meta)
            elif isinstance(meta, dict):
                for key, value in meta.items():
                    print(f"    {key}: {indent(shorten(as_text(value), full), '      ')}")
            else:
                print(f"    {indent(shorten(as_text(meta), full), '    ')}")

        print(f"\n  --- pages ({len(tf.pages)}) ---")
        for i, page in enumerate(tf.pages):
            page = page.aspage() if hasattr(page, "aspage") else page
            desc = (f"    page {i}: shape={page.shape} dtype={page.dtype} "
                    f"photometric={getattr(page, 'photometric', None)} "
                    f"compression={getattr(page, 'compression', None)} "
                    f"samples={getattr(page, 'samplesperpixel', None)} "
                    f"bits={getattr(page, 'bitspersample', None)}")
            print(desc)
            page_record = {"index": i, "shape": list(page.shape),
                           "dtype": str(page.dtype), "tags": {}}
            for tag in page.tags:
                text = format_tag_value(tag, full)
                print(f"      {tag.code:>5}  {tag.name:<22} {text}")
                page_record["tags"][tag.name] = shorten(as_text(tag.value), full=False)
                if tag.name not in BULK_TAGS:
                    walk_strings(tag.value, f"page{i}", strings, key=tag.name)
            record["pages"].append(page_record)

        names = channel_names_by_page(tf)

    print("\n  --- channel names by page ---")
    if names:
        for index, name, source in names:
            print(f"    page {index}: {name}   [{source}]")
    else:
        print("    none: nothing in this file names a page.")
    if path.name == HYB_TIF_NAME:
        print("    load_fov_images reads page 0 as GCAMP, 3 as mScarlet, 4 as DAPI.")
    record["channel_names"] = [{"page": i, "name": n, "source": s} for i, n, s in names]

    hits = keyword_sweep(strings, full)
    record["keyword_hits"] = hits
    return record


def keyword_sweep(strings, full):
    """Report every channel-vocabulary token found in any metadata string."""
    print("\n  --- keyword sweep ---")
    hits = []
    seen = set()
    for source, text in strings:
        low = text.lower()
        for token in TOKENS:
            if token not in low:
                continue
            key = (source, token, text[:80])
            if key in seen:
                continue
            seen.add(key)
            hits.append({"source": source, "token": token,
                         "text": shorten(text, full)})
    if not hits:
        print("    no channel vocabulary anywhere in this file's metadata.")
        print("    The TIFF does not name its channels -- page identity has to come")
        print("    from the acquisition/processing code or from the pixels.")
        return hits
    for hit in hits:
        print(f"    [{hit['token']}] in {hit['source']}: {indent(hit['text'], '      ')}")
    return hits


def list_siblings(directory, dump_text=False, full=False):
    """What else sits beside the TIFF -- a channel name it lacks is often here."""
    print(f"\n  --- other files in {directory.name}/ ---")
    entries = sorted(p for p in directory.iterdir() if p.is_file())
    if not entries:
        print("    (none)")
        return []
    listing = []
    for path in entries:
        size = path.stat().st_size
        print(f"    {path.name}  ({size / 1e6:.2f} MB)")
        listing.append({"name": path.name, "size_bytes": size})
        if dump_text and path.suffix.lower() in TEXT_SUFFIXES and size <= MAX_TEXT_BYTES:
            text = path.read_text(encoding="utf-8", errors="replace")
            print(f"      {indent(shorten(text, full), '      ')}")
    return listing


def resolve_targets(args):
    """(paths, note). An explicit path wins; otherwise walk HYB_ROOT's FOVs."""
    if args.path:
        return [Path(p) for p in args.path], "explicit path"

    from preprocessing_config import HYB_ROOT
    hyb_root = Path(HYB_ROOT)
    if not hyb_root.exists():
        raise SystemExit(f"HYB_ROOT does not exist: {hyb_root}")

    paths = []
    for fov_dir in sorted(p for p in hyb_root.iterdir() if p.is_dir()):
        tif = fov_dir / HYB_TIF_NAME
        if tif.exists():
            paths.append(tif)
        if len(paths) >= args.n:
            break
    if not paths:
        raise SystemExit(f"No {HYB_TIF_NAME} under {hyb_root}")
    return paths, f"{HYB_ROOT}"


def main():
    ap = argparse.ArgumentParser(
        description="Dump every scrap of metadata a TIFF carries",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("path", nargs="*", help=f"TIFF(s); default: {HYB_TIF_NAME} under HYB_ROOT")
    ap.add_argument("-n", type=int, default=1, help="FOVs to read when no path is given (default 1)")
    ap.add_argument("--full", action="store_true", help="do not truncate long values")
    ap.add_argument("--dump-text", action="store_true", help="print small text files sitting beside the TIFF")
    ap.add_argument("--no-siblings", action="store_true", help="skip the directory listing")
    ap.add_argument("--json", default=None, help="write the dump to this path")
    args = ap.parse_args()

    paths, note = resolve_targets(args)
    print(f"tifffile {tifffile.__version__}   source: {note}\n")

    records = []
    for path in paths:
        if not path.exists():
            print(f"MISSING: {path}\n")
            continue
        try:
            record = probe_file(path, full=args.full)
        except Exception as exc:
            print(f"  !! could not read as TIFF: {type(exc).__name__}: {exc}")
            record = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
        if not args.no_siblings:
            record["siblings"] = list_siblings(path.parent, args.dump_text, args.full)
        records.append(record)
        print()

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        print(f"JSON: {out}")


if __name__ == "__main__":
    main()
