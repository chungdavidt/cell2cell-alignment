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
    for key in ("expmat", "pos", "pos40x", "fov", "slice", "depth", "angle", "id"):
        if key not in fn:
            print(f"  {key:8s} ABSENT")
            continue
        v = fn[key]
        shape = getattr(v, "shape", f"len={len(v)}" if hasattr(v, "__len__") else "?")
        print(f"  {key:8s} shape={shape} type={type(v).__name__}")

    if "expmat" in fn:
        m = fn["expmat"]
        ncols = m.shape[1] if m.ndim == 2 else None
        print(f"  expmat shape after orientation: {m.shape} (cells, genes)")
        print(f"  expmat gene columns: {ncols} (JH302 panel: 114, mScarlet last)")

    genes = fn.get("genes")
    if genes:
        print(f"  gene list: {len(genes)} names, last 8: {genes[-8:]}")
        if fn.get("genes_alt"):
            print(f"  second name column present, last 8: {fn['genes_alt'][-8:]}")
        from utilities.mat_io import resolve_marker_column
        for marker in ("mScarlet", "GCaMP"):
            try:
                print(f"  {marker} -> column {resolve_marker_column(fn, marker)}")
            except ValueError as e:
                print(f"  {marker}: {e}")
    else:
        print("  no gene list — marker column falls back to MSCARLET_COLUMN_INDEX (113)")
    if "slice" in fn:
        s = np.asarray(fn["slice"]).ravel()
        print(f"  slice ids: n={s.size} unique={np.unique(s).size} range={s.min()}..{s.max()}")
    if "fov" in fn:
        fovs = list(fn["fov"])
        print(f"  fov entries: {len(fovs)} unique={len(set(map(str, fovs)))}")
        print(f"  fov sample: {[str(x) for x in fovs[:5]]}")


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
        import tifffile
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
