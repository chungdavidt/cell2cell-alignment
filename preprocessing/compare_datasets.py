"""
Read-only side-by-side comparison of two BARseq datasets.

Extracts the same summary from each dataset and prints the fields that differ,
so a new dataset can be diffed against one the pipeline already handles.
Writes nothing unless --json is given.

Usage:
    python preprocessing/compare_datasets.py <DATA_ROOT_A> <DATA_ROOT_B>
    python preprocessing/compare_datasets.py <DATA_ROOT_A> <DATA_ROOT_B> --all
    python preprocessing/compare_datasets.py <DATA_ROOT> --json snapshot.json
    python preprocessing/compare_datasets.py snapshot.json <DATA_ROOT_B>

Either positional argument may be a dataset folder or a .json snapshot written
by an earlier --json run, so a dataset can be compared against a copy that no
longer exists.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Mirrors preprocessing_config; kept local so this runs without local_config.
HYB_DIRNAME_CANDIDATES = ("hyb", "hyb_raw_files")
MSCARLET_COLUMN_INDEX = 113
GCAMP_COLUMN_INDEX = 111
QC_MIN_READS = 20
QC_MIN_GENES = 5


def _col(expmat, idx):
    import scipy.sparse as sp

    if idx >= expmat.shape[1]:
        return None
    col = expmat[:, idx]
    return np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()


def _summarize_filt_neurons(path, s):
    if not path.exists():
        s["filt_neurons"] = "MISSING"
        return
    from utilities.mat_io import load_mat, load_filt_neurons

    s["fn.top_level_vars"] = sorted(k for k in load_mat(path) if not k.startswith("__"))
    try:
        fn = load_filt_neurons(path)
    except Exception as e:
        s["fn.load"] = f"FAILED {type(e).__name__}: {e}"
        return
    s["fn.fields"] = sorted(fn.keys())

    expmat = fn.get("expmat")
    if expmat is not None and getattr(expmat, "ndim", 0) == 2:
        import scipy.sparse as sp

        s["fn.n_cells"], s["fn.n_genes"] = int(expmat.shape[0]), int(expmat.shape[1])
        s["fn.expmat_dtype"] = str(expmat.dtype)
        s["fn.expmat_sparse"] = bool(sp.issparse(expmat))
        reads = np.asarray(expmat.sum(axis=1)).ravel()
        ngene = np.asarray((expmat > 0).sum(axis=1)).ravel()
        s["fn.reads_per_cell_median"] = float(np.median(reads))
        s["fn.reads_per_cell_mean"] = round(float(reads.mean()), 2)
        s["fn.genes_per_cell_median"] = float(np.median(ngene))
        passing = int(((reads >= QC_MIN_READS) & (ngene >= QC_MIN_GENES)).sum())
        s[f"fn.qc_pass_pct (reads>={QC_MIN_READS},genes>={QC_MIN_GENES})"] = round(
            100 * passing / reads.size, 1
        )

    genes = fn.get("genes")
    if genes:
        s["fn.gene_panel_size"] = len(genes)
        s["fn.gene_names"] = list(genes)
        s["fn.has_genes_alt"] = bool(fn.get("genes_alt"))
    for marker, idx in (("mScarlet", MSCARLET_COLUMN_INDEX), ("GCaMP", GCAMP_COLUMN_INDEX)):
        if genes and idx < len(genes):
            s[f"fn.{marker}_col{idx}_label"] = genes[idx]
        col = _col(expmat, idx) if expmat is not None else None
        if col is not None:
            nz = int((col > 0).sum())
            s[f"fn.{marker}_col{idx}_pos_pct"] = round(100 * nz / col.size, 2)
            s[f"fn.{marker}_col{idx}_max"] = float(col.max())
            s[f"fn.{marker}_col{idx}_median_pos"] = (
                float(np.median(col[col > 0])) if nz else 0.0
            )

    if "slice" in fn:
        sl = np.asarray(fn["slice"]).ravel().astype(float)
        real = sl[~np.isnan(sl)]
        s["fn.slice_nan_pct"] = round(100 * float(np.isnan(sl).sum()) / sl.size, 1)
        s["fn.slice_n_unique"] = int(np.unique(real).size) if real.size else 0
        s["fn.slice_range"] = (
            [float(real.min()), float(real.max())] if real.size else None
        )
        s["fn.slice_ids"] = np.unique(real).astype(int).tolist() if real.size else []

    for key in ("uniq_slice", "slice_boundaries", "orig_slice"):
        if key in fn:
            s[f"fn.{key}_size"] = int(np.asarray(fn[key], dtype=object).ravel().size)

    for key in ("pos", "pos40x", "depth", "angle"):
        if key in fn:
            v = np.asarray(fn[key], dtype=float)
            s[f"fn.{key}_shape"] = list(v.shape)
            finite = v[np.isfinite(v)]
            if finite.size:
                s[f"fn.{key}_range"] = [round(float(finite.min()), 1),
                                        round(float(finite.max()), 1)]

    if "fov" in fn:
        fovs = [str(x) for x in fn["fov"]]
        s["fn.fov_entries"] = len(fovs)
        s["fn.fov_unique"] = len(set(fovs))
    if "fov_names" in fn:
        s["fn.fov_names_count"] = len(list(fn["fov_names"]))
    for key in ("brain_name", "batch_num", "uid"):
        if key in fn:
            v = fn[key]
            s[f"fn.{key}"] = (str(v)[:60] if np.size(v) < 5
                              else f"size={np.size(v)}")


def _summarize_hyb(data_root, s):
    hyb_root = next(
        (data_root / n for n in HYB_DIRNAME_CANDIDATES if (data_root / n).is_dir()),
        None,
    )
    s["hyb.dirname"] = hyb_root.name if hyb_root else "MISSING"
    if hyb_root is None:
        return
    from utilities.graph_utils import parse_fov_grid_positions

    all_dirs = sorted(p for p in hyb_root.iterdir() if p.is_dir())
    names = [p.name for p in all_dirs]
    s["hyb.subdirs"] = len(all_dirs)
    if not names:
        return
    pos, valid = parse_fov_grid_positions(names)
    fov_dirs = [d for d, v in zip(all_dirs, valid) if v]
    s["hyb.fov_dirs"] = len(fov_dirs)
    s["hyb.non_fov_subdirs"] = [n for n, v in zip(names, valid) if not v]
    if valid.any():
        ok = pos[valid]
        s["hyb.grid_rows"] = [int(ok[:, 0].min()), int(ok[:, 0].max())]
        s["hyb.grid_cols"] = [int(ok[:, 1].min()), int(ok[:, 1].max())]
    if not fov_dirs:
        return

    d = fov_dirs[0]
    s["hyb.first_fov"] = d.name
    s["hyb.first_fov_files"] = sorted(f.name for f in d.iterdir())

    tif = d / "alignedn2vhyb01.tif"
    s["hyb.alignedn2vhyb01_present"] = tif.exists()
    if not tif.exists():
        cands = sorted(d.glob("*.tif")) + sorted(d.glob("*.tiff"))
        tif = cands[0] if cands else None
    if tif is not None:
        s["hyb.tif_MB"] = round(tif.stat().st_size / 1e6, 1)
        import tifffile

        try:
            with tifffile.TiffFile(tif) as tf:
                s["hyb.tif_pages"] = len(tf.pages)
                s["hyb.tif_shape"] = list(tf.pages[0].shape)
                s["hyb.tif_dtype"] = str(tf.pages[0].dtype)
                xres = tf.pages[0].tags.get("XResolution")
                s["hyb.tif_xresolution"] = list(xres.value) if xres else None
                s["hyb.tif_imagej_metadata"] = (
                    sorted(tf.imagej_metadata) if tf.imagej_metadata else None
                )
        except Exception as e:
            s["hyb.tif_open"] = f"FAILED {type(e).__name__}: {e}"

    cm = d / "cellmask.mat"
    s["hyb.cellmask_present"] = cm.exists()
    if cm.exists():
        s["hyb.cellmask_MB"] = round(cm.stat().st_size / 1e6, 1)
        from utilities.mat_io import load_mat

        try:
            s["hyb.cellmask_vars"] = sorted(
                k for k in load_mat(cm) if not k.startswith("__")
            )
        except Exception as e:
            s["hyb.cellmask_load"] = f"FAILED {type(e).__name__}: {e}"


def summarize(data_root: Path) -> dict:
    s = {"data_root": str(data_root), "exists": data_root.exists()}
    if not data_root.exists():
        return s
    s["top_level.dirs"] = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    s["top_level.files"] = sorted(p.name for p in data_root.iterdir() if p.is_file())
    _summarize_filt_neurons(data_root / "filt_neurons.mat", s)
    _summarize_hyb(data_root, s)
    return s


def _load(arg):
    p = Path(arg)
    return json.loads(p.read_text()) if p.suffix == ".json" else summarize(p)


def _fmt(v, width=44):
    if v is None:
        return "-"
    if isinstance(v, list):
        text = f"[{len(v)}] {v[:6]}" if len(v) > 6 else str(v)
    else:
        text = str(v)
    return text if len(text) <= width else text[: width - 1] + "…"


def _list_delta(a, b):
    """Describe how two lists differ, instead of printing both in full."""
    sa, sb = set(map(str, a)), set(map(str, b))
    only_a, only_b = sorted(sa - sb), sorted(sb - sa)
    lines = []
    if only_a:
        lines.append(f"      only in A ({len(only_a)}): {only_a[:8]}")
    if only_b:
        lines.append(f"      only in B ({len(only_b)}): {only_b[:8]}")
    if not lines and list(map(str, a)) != list(map(str, b)):
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if str(x) != str(y)), None)
        lines.append(f"      same members, order differs from index {first}")
    return lines


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("a", help="DATA_ROOT or .json snapshot")
    ap.add_argument("b", nargs="?", help="DATA_ROOT or .json snapshot")
    ap.add_argument("--json", metavar="PATH",
                    help="write A's summary as a snapshot and exit")
    ap.add_argument("--all", action="store_true",
                    help="print matching fields too, not only differences")
    args = ap.parse_args()

    sa = _load(args.a)
    if args.json:
        Path(args.json).write_text(json.dumps(sa, indent=1, default=str))
        print(f"wrote {args.json}  ({len(sa)} fields)")
        return
    if not args.b:
        ap.error("need a second dataset, or --json to write a snapshot")
    sb = _load(args.b)

    keys = [k for k in list(sa) + [k for k in sb if k not in sa]
            if k != "data_root"]
    name_a = Path(sa.get("data_root", args.a)).name
    name_b = Path(sb.get("data_root", args.b)).name
    print(f"A = {sa.get('data_root', args.a)}")
    print(f"B = {sb.get('data_root', args.b)}")
    print(f"\n{'':2}{'field':<40} {name_a[:44]:<44} {name_b[:44]}")
    print("-" * 134)

    n_diff = 0
    for k in keys:
        va, vb = sa.get(k), sb.get(k)
        same = va == vb
        if same and not args.all:
            continue
        n_diff += not same
        print(f"{'  ' if same else '! '}{k:<40} {_fmt(va):<44} {_fmt(vb)}")
        if not same and isinstance(va, list) and isinstance(vb, list):
            for line in _list_delta(va, vb):
                print(line)

    print("-" * 134)
    print(f"{n_diff} differing fields of {len(keys)}")


if __name__ == "__main__":
    main()
