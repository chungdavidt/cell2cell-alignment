"""
Read-only structural probe for a new BARseq dataset.

Prints what the preprocessing pipeline's hardcoded JH302 assumptions expect vs.
what the dataset actually contains. Writes nothing.

Usage:
    python preprocessing/inspect_dataset_structure.py <DATA_ROOT>
    python preprocessing/inspect_dataset_structure.py          # uses local_config.DATA_ROOT
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Per-FOV raw dir name varies by dataset; mirrors preprocessing_config.
HYB_DIRNAME_CANDIDATES = ("hyb", "hyb_raw_files")


def probe_filt_neurons(path):
    print(f"\n=== filt_neurons.mat ===\n{path}")
    if not path.exists():
        print("  MISSING")
        return
    from utilities.mat_io import load_mat, load_filt_neurons

    raw = load_mat(path)
    print(f"  top-level vars: {[k for k in raw if not k.startswith('__')]}")
    try:
        fn = load_filt_neurons(path)
    except Exception as e:
        print(f"  load_filt_neurons FAILED: {type(e).__name__}: {e}")
        return
    print(f"  parsed fields: {sorted(fn.keys())}")
    for key in sorted(fn.keys()):
        v = fn[key]
        # getattr's default must not be evaluated eagerly here: len() on a sparse
        # matrix raises rather than returning a size.
        if hasattr(v, "shape"):
            shape = v.shape
        elif hasattr(v, "__len__"):
            shape = f"len={len(v)}"
        else:
            shape = "scalar"
        print(f"  {key:16s} shape={shape} type={type(v).__name__}")

    for key in ("expmat", "pos", "fov", "slice"):
        if key not in fn:
            print(f"  !! required field {key!r} is absent")

    if "expmat" in fn:
        m = fn["expmat"]
        ncols = m.shape[1] if m.ndim == 2 else None
        print(f"  expmat shape after orientation: {m.shape} (cells, genes)")
        print(f"  expmat gene columns: {ncols} (JH302 panel: 114, mScarlet last)")

    genes = fn.get("genes")
    if genes:
        alt = fn.get("genes_alt") or [""] * len(genes)
        print(f"  gene list ({len(genes)} entries, index: name | second column):")
        for i, name in enumerate(genes):
            print(f"    {i:3d}  {name:<24} | {alt[i] if i < len(alt) else ''}")
        from utilities.mat_io import resolve_marker_column
        for marker in ("mScarlet", "GCaMP"):
            try:
                print(f"  {marker} -> column {resolve_marker_column(fn, marker)}")
            except ValueError as e:
                print(f"  {marker}: {e}")
    else:
        print("  no gene list — marker column falls back to MSCARLET_COLUMN_INDEX (113)")
    if "slice" in fn:
        sl = np.asarray(fn["slice"]).ravel().astype(float)
        n_nan = int(np.isnan(sl).sum())
        real = sl[~np.isnan(sl)]
        uniq = np.unique(real)
        print(f"  slice ids: n={sl.size}, NaN={n_nan} "
              f"({100*n_nan/sl.size:.1f}% of cells unassigned)")
        print(f"  slice values: {uniq.size} unique, "
              f"range {real.min():.0f}..{real.max():.0f}")
        print(f"    {uniq.astype(int).tolist()}")
    for key in ("uniq_slice", "slice_boundaries", "orig_slice"):
        if key in fn:
            v = np.asarray(fn[key]).ravel()
            preview = v[:20].tolist()
            print(f"  {key}: {v.size} values{'' if v.size <= 20 else ' (first 20)'} {preview}")
    if "fov" in fn:
        fovs = list(fn["fov"])
        print(f"  fov entries: {len(fovs)} unique={len(set(map(str, fovs)))}")
        print(f"  fov sample: {[str(x) for x in fovs[:5]]}")


def _probe_tiff(tif: Path) -> None:
    """Report page layout, or diagnose the file when it won't open as a TIFF."""
    import tifffile

    try:
        with tifffile.TiffFile(tif) as tf:
            print(f"  {tif.name}: pages={len(tf.pages)} "
                  f"shape={tf.pages[0].shape} dtype={tf.pages[0].dtype}")
            print("  (JH302 layout: page 0=GCAMP, 3=mScarlet, 4=DAPI; needs >=5 pages)")
            for i, pg in enumerate(tf.pages[:8]):
                print(f"    page {i}: {pg.shape} {pg.dtype}")
            if tf.imagej_metadata:
                print(f"  imagej_metadata: {tf.imagej_metadata}")
            xres = tf.pages[0].tags.get("XResolution")
            print(f"  XResolution: {xres.value if xres else None}")
        return
    except Exception as exc:
        print(f"  !! could not open as TIFF: {type(exc).__name__}: {exc}")

    size = tif.stat().st_size
    print(f"  file size: {size/1e6:.1f} MB")
    with open(tif, "rb") as f:
        head = f.read(4096)
        f.seek(max(0, size - 64))
        tail = f.read(64)
    print(f"  first 64 bytes: {head[:64].hex(' ')}")
    print(f"  last 64 bytes:  {tail.hex(' ')}")
    print(f"  leading zero bytes: {len(head) - len(head.lstrip(bytes([0])))}")
    for magic, label in ((b"II*\x00", "little-endian TIFF"),
                         (b"MM\x00*", "big-endian TIFF"),
                         (b"II+\x00", "BigTIFF LE"),
                         (b"MM\x00+", "BigTIFF BE"),
                         (b"\x89HDF", "HDF5")):
        off = head.find(magic)
        print(f"  {label} magic in first 4KB: "
              f"{'no' if off < 0 else f'yes at offset {off}'}")


def probe_hyb(hyb_root):
    print(f"\n=== {hyb_root.name}/ ===\n{hyb_root}")
    if not hyb_root.exists():
        print("  MISSING")
        return
    all_dirs = sorted(p for p in hyb_root.iterdir() if p.is_dir())
    print(f"  subdirectories: {len(all_dirs)}")

    from utilities.graph_utils import parse_fov_grid_positions

    names = [p.name for p in all_dirs]
    fov_dirs = []
    if names:
        pos, valid = parse_fov_grid_positions(names)
        fov_dirs = [d for d, v in zip(all_dirs, valid) if v]
        print(f"  MAX_Pos{{N}}_{{row}}_{{col}} parse: {valid.sum()}/{len(names)} valid")
        print(f"  FOV names: {[p.name for p in fov_dirs[:5]]}")
        if valid.any():
            ok = pos[valid]
            print(f"  grid rows {ok[:,0].min():.0f}..{ok[:,0].max():.0f} "
                  f"cols {ok[:,1].min():.0f}..{ok[:,1].max():.0f}")
        non_fov = [n for n, v in zip(names, valid) if not v]
        if non_fov:
            print(f"  non-FOV subdirs (ignored): {non_fov}")

    if not fov_dirs:
        print("  no directory parsed as a FOV name")
        return
    d = fov_dirs[0]
    print(f"\n  --- contents of {d.name} ---")
    for f in sorted(d.iterdir())[:20]:
        print(f"    {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

    tif = d / "alignedn2vhyb01.tif"
    print(f"\n  alignedn2vhyb01.tif present: {tif.exists()}")
    if not tif.exists():
        cands = sorted(d.glob("*.tif")) + sorted(d.glob("*.tiff"))
        print(f"  other TIFFs: {[c.name for c in cands]}")
        tif = cands[0] if cands else None
    if tif is not None:
        _probe_tiff(tif)

    cm = d / "cellmask.mat"
    print(f"\n  cellmask.mat present: {cm.exists()}")
    if cm.exists():
        from utilities.mat_io import load_mat
        try:
            keys = [k for k in load_mat(cm) if not k.startswith("__")]
            print(f"  cellmask vars: {keys}  (tried: maski/cellmask/mask/segmentation/seg)")
        except Exception as e:
            print(f"  cellmask load FAILED: {type(e).__name__}: {e}")
    else:
        print(f"  other .mat/.h5 in dir: "
              f"{[f.name for f in d.iterdir() if f.suffix in ('.mat', '.h5')]}")


def main():
    if len(sys.argv) > 1:
        data_root = Path(sys.argv[1])
    else:
        import local_config
        data_root = Path(local_config.DATA_ROOT)

    print(f"DATA_ROOT = {data_root}")
    print(f"exists: {data_root.exists()}")
    if data_root.exists():
        print("\n=== top level ===")
        for p in sorted(data_root.iterdir())[:40]:
            kind = "dir " if p.is_dir() else "file"
            print(f"  {kind} {p.name}")

    probe_filt_neurons(data_root / "filt_neurons.mat")

    hyb_root = next(
        (data_root / n for n in HYB_DIRNAME_CANDIDATES if (data_root / n).is_dir()),
        data_root / HYB_DIRNAME_CANDIDATES[0],
    )
    probe_hyb(hyb_root)


if __name__ == "__main__":
    main()
