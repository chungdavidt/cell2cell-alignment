r"""
Hand-curated cellpose outlines, and every parameter combination scored against them.

export  Saves chosen Z planes of a red 2P volume as single-plane TIFFs, each with
        cellpose's default 2D outlines as ``<plane>_seg.npy`` beside it, into
        ``<ANALYSIS_ROOT>/cellpose/curation_<tag>/``. Opening a plane TIFF in the
        cellpose GUI loads those outlines; fix them there (delete wrong ones, draw
        missed cells) and save with Ctrl+S, which overwrites the ``_seg.npy``.
        Planes already in the folder are never overwritten, so planes can be added
        later. ``--blank`` exports planes with no starting outlines.

score   Runs every combination in GRID on the curated planes and compares each
        result with the curated outlines. A curated cell is FOUND when one
        predicted outline overlaps it with intersection-over-union >= --iou
        (default 0.5; each outline is used for one cell at most), MISSED otherwise;
        a predicted outline matching no curated cell is EXTRA.
        F1 = 2*found / (2*found + missed + extra). Combinations are ranked by F1,
        then found, then mean IoU of the found cells. Writes scores.csv,
        per_plane.csv, best.json and outline PNGs (curated yellow, predicted cyan)
        into ``curation_<tag>/score_grid_<YYYYMMDD_HHMMSS>/``, and prints the best
        combination as the line to paste into batch_cellpose.EVAL_OVERRIDES.

        ``score --batch`` (or ``--masks <path>``) scores a 3D run of
        batch_cellpose.py on the same planes instead of running the grid.

A plane whose saved outlines are identical to the exported ones counts as not
curated and is left out, unless named in --accept-unchanged (the default outlines
were already right, or a --blank plane has no cells).

The grid runs cellpose in 2D, one plane at a time; batch_cellpose.py runs 3D on the
whole volume, so score its output with --batch after the full run.

Usage (.cellpose-venv):
    .cellpose-venv\\Scripts\\python.exe alignment\\cellpose_curation.py export --z-planes 120,151,180,220
    .cellpose-venv\\Scripts\\python.exe alignment\\cellpose_curation.py export --n 15        # evenly spaced
    .cellpose-venv\\Scripts\\python.exe -m cellpose                                          # open a z*.tif, fix, Ctrl+S
    .cellpose-venv\\Scripts\\python.exe alignment\\cellpose_curation.py score --dry-run
    .cellpose-venv\\Scripts\\python.exe alignment\\cellpose_curation.py score
    .cellpose-venv\\Scripts\\python.exe alignment\\cellpose_curation.py score --accept-unchanged 151,220
    .cellpose-venv\\Scripts\\python.exe alignment\\cellpose_curation.py score --batch          # 3D run
    add --node block_stack_red to any of these for the ex vivo block
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Same as sweep_in_vivo_2d.py: cellpose's defaults, normalize in dict form.
VANILLA_NORMALIZE_DICT = {
    "normalize":           True,
    "norm3D":              True,
    "invert":              False,
    "lowhigh":             None,
    "percentile":          [1.0, 99.0],
    "sharpen_radius":      0,
    "smooth_radius":       0,
    "tile_norm_blocksize": 0,
    "tile_norm_smooth3D":  1,
}
VANILLA_EVAL = {
    "cellprob_threshold": 0.0,
    "flow_threshold":     0.4,
    "diameter":           None,
    "min_size":           8,
    "niter":              200,
}
BATCH_SIZE = 8

# (param, values, kind). Every list holds the vanilla value, so vanilla is one of
# the combinations. 4*3*3*2*2*2 = 288 combinations.
GRID: list[tuple[str, list, str]] = [
    ("diameter",            [None, 10, 15, 20],         "eval"),
    ("cellprob_threshold",  [-1.0, 0.0, 1.0],           "eval"),
    ("flow_threshold",      [0.4, 0.6, 0.8],            "eval"),
    ("sharpen_radius",      [0, 60],                    "normalize"),
    ("tile_norm_blocksize", [0, 100],                   "normalize"),
    ("percentile",          [[1.0, 99.0], [0.1, 99.9]], "normalize"),
]

# Same as sweep_in_vivo_2d.SWEEP_TAGS.
TAGS = {"invivo_red": "in_vivo", "block_stack_red": "block_stack"}

TRUTH_COLOR = "#FFE066"
PRED_COLOR = "#00E5E5"
PLANES_PER_PAGE = 5
CELL_SIZE_INCHES = 2.5
DPI = 150
MANIFEST = "curation.json"


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match(truth: np.ndarray, pred: np.ndarray, iou_min: float) -> dict:
    """One-to-one match of two label images by IoU, highest first; labels > 0 are cells."""
    t_ids, t_inv = np.unique(truth.ravel(), return_inverse=True)
    p_ids, p_inv = np.unique(pred.ravel(), return_inverse=True)
    t_area = np.bincount(t_inv)
    p_area = np.bincount(p_inv)
    n_p = len(p_ids)
    pair, inter = np.unique(t_inv.astype(np.int64) * n_p + p_inv, return_counts=True)
    ti, pi = pair // n_p, pair % n_p
    fg = (t_ids[ti] > 0) & (p_ids[pi] > 0)
    ti, pi, inter = ti[fg], pi[fg], inter[fg]
    iou = inter / (t_area[ti] + p_area[pi] - inter)

    used_t, used_p, ious = set(), set(), []
    for k in np.argsort(-iou, kind="stable"):
        if iou[k] < iou_min:
            break
        if ti[k] in used_t or pi[k] in used_p:
            continue
        used_t.add(ti[k])
        used_p.add(pi[k])
        ious.append(float(iou[k]))

    n_truth = int((t_ids > 0).sum())
    n_pred = int((p_ids > 0).sum())
    found = len(ious)
    return {"found": found, "missed": n_truth - found, "extra": n_pred - found,
            "n_truth": n_truth, "n_pred": n_pred, "iou_sum": float(sum(ious))}


def totals(rows: list[dict]) -> dict:
    t = {k: sum(r[k] for r in rows) for k in ("found", "missed", "extra", "n_truth", "n_pred", "iou_sum")}
    denom = 2 * t["found"] + t["missed"] + t["extra"]
    t["f1"] = 1.0 if denom == 0 else 2 * t["found"] / denom
    t["precision"] = t["found"] / t["n_pred"] if t["n_pred"] else 1.0
    t["recall"] = t["found"] / t["n_truth"] if t["n_truth"] else 1.0
    t["mean_iou"] = t["iou_sum"] / t["found"] if t["found"] else 0.0
    return t


def n_cells(masks: np.ndarray) -> int:
    return int((np.unique(masks) > 0).sum())


def same_outlines(a: np.ndarray, b: np.ndarray) -> bool:
    """Same cells with the same pixels, whatever the label numbers."""
    m = match(a, b, 1.0)
    return m["missed"] == 0 and m["extra"] == 0


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def grid_combos() -> list[dict]:
    """[{param: value}] over the full product of GRID."""
    names = [p for p, _, _ in GRID]
    return [dict(zip(names, vals)) for vals in itertools.product(*[v for _, v, _ in GRID])]


def build_kwargs(combo: dict) -> dict:
    kinds = {p: k for p, _, k in GRID}
    norm = deepcopy(VANILLA_NORMALIZE_DICT)
    ev = deepcopy(VANILLA_EVAL)
    for name, value in combo.items():
        (norm if kinds[name] == "normalize" else ev)[name] = value
    return {"do_3D": False, "normalize": norm, **ev, "batch_size": BATCH_SIZE}


def vanilla_kwargs() -> dict:
    return {"do_3D": False, "normalize": True, **VANILLA_EVAL, "batch_size": BATCH_SIZE}


def is_vanilla(combo: dict) -> bool:
    kinds = {p: k for p, _, k in GRID}
    return all(
        (VANILLA_NORMALIZE_DICT if kinds[n] == "normalize" else VANILLA_EVAL)[n] == v
        for n, v in combo.items())


def override_for(combo: dict) -> dict:
    """The combination as an EVAL_OVERRIDES entry: only what differs from batch_cellpose's defaults."""
    kinds = {p: k for p, _, k in GRID}
    out, norm = {}, {}
    for name, value in combo.items():
        if kinds[name] == "normalize":
            if VANILLA_NORMALIZE_DICT[name] != value:
                norm[name] = value
        elif VANILLA_EVAL[name] != value:
            out[name] = value
    if norm:
        out["normalize"] = {**VANILLA_NORMALIZE_DICT, **norm}
    return out


def combo_label(combo: dict) -> str:
    short = {"diameter": "diam", "cellprob_threshold": "cellprob", "flow_threshold": "flow",
             "sharpen_radius": "sharpen", "tile_norm_blocksize": "tile", "percentile": "pct"}
    return " ".join(f"{short.get(k, k)}={v}" for k, v in combo.items())


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def curation_dir(node: str) -> Path:
    from analysis_paths import cellpose_dir
    root = cellpose_dir()
    if root is None:
        sys.exit("ANALYSIS_ROOT is blank in local_config.py; curation is written under it.")
    return root / f"curation_{TAGS[node]}"


def plane_name(z: int, n_planes: int) -> str:
    return f"z{z:0{len(str(n_planes - 1))}d}"


def load_seg_masks(path: Path) -> np.ndarray:
    seg = np.load(str(path), allow_pickle=True).item()
    return np.asarray(seg["masks"]).squeeze()


def load_manifest(cdir: Path) -> dict:
    path = cdir / MANIFEST
    if not path.exists():
        sys.exit(f"No {MANIFEST} in {cdir}. Run the export step first.")
    return json.loads(path.read_text())


def load_planes(cdir: Path, manifest: dict, accept: set | str) -> list[dict]:
    """Curated planes as [{z, name, image, truth}]; prints the ones left out and why."""
    import tifffile
    out = []
    print(f"Planes in {cdir}:")
    for z_str, info in sorted(manifest["planes"].items(), key=lambda kv: int(kv[0])):
        z, name = int(z_str), info["name"]
        image = tifffile.imread(str(cdir / f"{name}.tif"))
        start = tifffile.imread(str(cdir / "_start" / f"{name}_masks.tif"))
        seg_path = cdir / f"{name}_seg.npy"
        truth = load_seg_masks(seg_path) if seg_path.exists() else np.zeros_like(start)
        if truth.shape != image.shape:
            sys.exit(f"{seg_path.name}: masks shape {truth.shape}, plane shape {image.shape}")
        accepted = accept == "all" or z in accept
        if same_outlines(truth, start) and not accepted:
            print(f"  z={z:<4} {name}: unchanged since export, left out "
                  f"(--accept-unchanged {z} to use it as is)")
            continue
        print(f"  z={z:<4} {name}: {n_cells(truth)} curated cells")
        out.append({"z": z, "name": name, "image": image, "truth": truth})
    if not out:
        sys.exit("No curated planes.")
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_outlines(ax, masks: np.ndarray, color: str, width: float) -> None:
    ids = np.unique(masks)
    ids = ids[ids > 0]
    if ids.size == 0:
        return
    # Consecutive labels, so half-integer levels trace every boundary in one call.
    relabel = np.searchsorted(np.concatenate([[0], ids]), masks)
    ax.contour(relabel, levels=np.arange(0.5, ids.size + 0.5, 1.0), colors=color, linewidths=width)


def render_pages(out_dir: Path, stem: str, planes: list[dict],
                 columns: list[tuple[str, list | None]], iou_min: float) -> None:
    """columns: (header, per-plane predicted masks, or None for the curated-only column)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pages = math.ceil(len(planes) / PLANES_PER_PAGE)
    for page in range(n_pages):
        idx = list(range(page * PLANES_PER_PAGE, min(len(planes), (page + 1) * PLANES_PER_PAGE)))
        fig, axes = plt.subplots(len(idx), len(columns), squeeze=False, constrained_layout=True,
                                 figsize=(CELL_SIZE_INCHES * len(columns), CELL_SIZE_INCHES * len(idx)))
        for r, i in enumerate(idx):
            p = planes[i]
            lo, hi = np.percentile(p["image"], [1, 99])
            for c, (header, preds) in enumerate(columns):
                ax = axes[r, c]
                ax.imshow(p["image"], cmap="gray", vmin=lo, vmax=hi)
                if preds is None:
                    draw_outlines(ax, p["truth"], TRUTH_COLOR, 0.8)
                    text = f"{n_cells(p['truth'])} curated"
                else:
                    draw_outlines(ax, p["truth"], TRUTH_COLOR, 0.5)
                    draw_outlines(ax, preds[i], PRED_COLOR, 0.8)
                    m = match(p["truth"], preds[i], iou_min)
                    text = f"found {m['found']}/{m['n_truth']}  extra {m['extra']}"
                ax.text(0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top",
                        color="white", fontsize=6,
                        bbox=dict(facecolor="black", alpha=0.6, pad=1.5, edgecolor="none"))
                if r == 0:
                    ax.set_title(header, fontsize=7)
                if c == 0:
                    ax.set_ylabel(f"z={p['z']}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
        suffix = f"_p{page + 1}" if n_pages > 1 else ""
        fig.suptitle("curated = yellow, cellpose = cyan", fontsize=10)
        fig.savefig(out_dir / f"{stem}{suffix}.png", dpi=DPI)
        plt.close(fig)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def cmd_export(args) -> int:
    import tifffile
    import local_config
    from analysis_paths import TWO_P_RED_NODES, two_p_path

    tiff = Path(args.tiff) if args.tiff else two_p_path(args.node)
    if tiff is None:
        sys.exit(f"local_config.{TWO_P_RED_NODES[args.node]} is blank; pass --tiff.")
    if not tiff.exists():
        sys.exit(f"TIFF not found: {tiff}")
    cdir = curation_dir(args.node)

    manifest = None
    if (cdir / MANIFEST).exists():
        manifest = json.loads((cdir / MANIFEST).read_text())
        if manifest["tiff"] != str(tiff):
            sys.exit(f"{cdir} holds planes of {manifest['tiff']}, not {tiff}. "
                     f"Move that folder aside to curate another volume.")

    with tifffile.TiffFile(str(tiff)) as tf:
        shape = tuple(tf.series[0].shape)
    if len(shape) != 3:
        sys.exit(f"Expected a (Z, Y, X) volume, got shape {shape}")
    n_planes = shape[0]

    if args.z_planes:
        zs = sorted({int(s) for s in args.z_planes.split(",")})
    else:
        zs = sorted({int(round(v)) for v in np.linspace(0, n_planes - 1, args.n + 2)[1:-1]})
    bad = [z for z in zs if not 0 <= z < n_planes]
    if bad:
        sys.exit(f"Z planes {bad} outside [0, {n_planes}).")

    have = set(int(z) for z in manifest["planes"]) if manifest else set()
    new = [z for z in zs if z not in have]
    print(f"Volume: {tiff}  shape (Z, Y, X) {shape}")
    print(f"Curation folder: {cdir}")
    print(f"Planes asked for: {zs}")
    if have & set(zs):
        print(f"  already exported, left as they are: {sorted(have & set(zs))}")
    print(f"  to export: {new} ({'no starting outlines' if args.blank else 'default cellpose outlines'})")
    if args.dry_run or not new:
        return 0

    stack = tifffile.imread(str(tiff))
    (cdir / "_start").mkdir(parents=True, exist_ok=True)
    if manifest is None:
        manifest = {"node": args.node, "tiff": str(tiff), "shape_zyx": list(shape),
                    "scope": getattr(local_config, "SCOPE", ""),
                    "start_eval": vanilla_kwargs(), "planes": {}}

    model = None
    if not args.blank:
        from cellpose import io as cp_io
        from cellpose.models import CellposeModel
        print("Loading cpsam model...")
        model = CellposeModel(gpu=True, pretrained_model="cpsam")

    for z in new:
        name = plane_name(z, n_planes)
        plane = stack[z]
        tifffile.imwrite(str(cdir / f"{name}.tif"), plane)
        if model is not None:
            masks, flows, _ = model.eval(plane, **vanilla_kwargs())
            # cellpose writes <file stem>_seg.npy beside the file name it is given.
            cp_io.masks_flows_to_seg(plane, masks, flows, str(cdir / f"{name}.tif"))
            if not (cdir / f"{name}_seg.npy").exists():
                sys.exit(f"cellpose did not write {cdir / (name + '_seg.npy')}")
        else:
            masks = np.zeros(plane.shape, dtype=np.uint16)
        tifffile.imwrite(str(cdir / "_start" / f"{name}_masks.tif"), masks.astype(np.uint16))
        manifest["planes"][str(z)] = {"name": name, "start": "blank" if args.blank else "default",
                                      "exported": datetime.now().isoformat(timespec="seconds")}
        print(f"  z={z:<4} {name}.tif  {int(masks.max())} starting outlines")
        (cdir / MANIFEST).write_text(json.dumps(manifest, indent=2))

    print(f"\nNext: .cellpose-venv\\Scripts\\python.exe -m cellpose, open each {cdir}\\z*.tif,")
    print("fix the outlines, Ctrl+S. Then: alignment\\cellpose_curation.py score --dry-run")
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def parse_accept(text: str | None) -> set | str:
    if not text:
        return set()
    if text.strip().lower() == "all":
        return "all"
    return {int(s) for s in text.split(",")}


def cmd_score(args) -> int:
    cdir = curation_dir(args.node)
    manifest = load_manifest(cdir)
    planes = load_planes(cdir, manifest, parse_accept(args.accept_unchanged))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.batch or args.masks:
        return score_3d(args, cdir, manifest, planes, stamp)

    combos = grid_combos()
    print(f"\n{len(combos)} combinations x {len(planes)} planes = {len(combos) * len(planes)} cellpose runs")
    for p, values, kind in GRID:
        print(f"  {p:<20} {values}")
    if args.dry_run:
        return 0

    from cellpose.models import CellposeModel
    out_dir = cdir / f"score_grid_{stamp}"
    out_dir.mkdir(parents=True)
    print(f"Output: {out_dir}\nLoading cpsam model...")
    model = CellposeModel(gpu=True, pretrained_model="cpsam")

    def run(combo):
        kw = build_kwargs(combo)
        return [model.eval(p["image"], **kw)[0] for p in planes]

    results, per_plane = [], []
    t0 = time.time()
    for i, combo in enumerate(combos):
        preds = run(combo)
        rows = [match(p["truth"], m, args.iou) for p, m in zip(planes, preds)]
        for p, r in zip(planes, rows):
            per_plane.append({"combo": i, "z": p["z"], **{k: r[k] for k in ("found", "missed", "extra", "n_truth", "n_pred")}})
        t = totals(rows)
        results.append({"combo": i, "params": combo, **t})
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(combos) - i - 1)
        print(f"  [{i + 1:>4}/{len(combos)}] F1 {t['f1']:.3f}  found {t['found']}/{t['n_truth']}  "
              f"extra {t['extra']}  {combo_label(combo)}   ({eta / 60:.0f} min left)")

    results.sort(key=lambda r: (-r["f1"], -r["found"], -r["mean_iou"]))
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    fields = ["rank", "combo", "f1", "precision", "recall", "found", "missed", "extra",
              "n_truth", "n_pred", "mean_iou"] + [p for p, _, _ in GRID]
    with open(out_dir / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({**{k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in fields[:11]},
                        **{p: r["params"][p] for p, _, _ in GRID}})
    with open(out_dir / "per_plane.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["combo", "z", "found", "missed", "extra", "n_truth", "n_pred"])
        w.writeheader()
        w.writerows(per_plane)

    vanilla = next(r for r in results if is_vanilla(r["params"]))
    best = results[0]
    print(f"\nTop {min(10, len(results))} of {len(results)} (IoU >= {args.iou}):")
    for r in results[:10]:
        print(f"  {r['rank']:>3}. F1 {r['f1']:.3f}  found {r['found']}/{r['n_truth']}  "
              f"missed {r['missed']}  extra {r['extra']}  {combo_label(r['params'])}")
    print(f"Defaults: rank {vanilla['rank']}, F1 {vanilla['f1']:.3f}, found {vanilla['found']}/"
          f"{vanilla['n_truth']}, extra {vanilla['extra']}")

    override = override_for(best["params"])
    line = f'"{args.node}": {override!r},'
    print(f"\nBest, for EVAL_OVERRIDES in alignment/batch_cellpose.py:\n    {line}")
    (out_dir / "best.json").write_text(json.dumps(
        {"node": args.node, "iou": args.iou, "planes": [p["z"] for p in planes],
         "best": {k: best[k] for k in ("f1", "found", "missed", "extra", "n_truth", "mean_iou")},
         "best_params": best["params"], "eval_override": override,
         "defaults": {k: vanilla[k] for k in ("rank", "f1", "found", "missed", "extra")}},
        indent=2, default=str))

    top = [r for r in results[:args.top]]
    columns = [("curated", None), (f"defaults (rank {vanilla['rank']})", run(vanilla["params"]))]
    for r in top:
        if r is not vanilla:
            columns.append((f"#{r['rank']} F1 {r['f1']:.2f}\n{combo_label(r['params'])}", run(r["params"])))
    render_pages(out_dir, "top", planes, columns, args.iou)
    print(f"\nWrote scores.csv, per_plane.csv, best.json, top*.png to {out_dir}")
    return 0


def score_3d(args, cdir, manifest, planes, stamp) -> int:
    import tifffile
    if args.masks:
        masks_path = Path(args.masks)
    else:
        masks_path = cdir.parent / f"{Path(manifest['tiff']).stem}_masks.tif"
    if not masks_path.exists():
        sys.exit(f"Masks not found: {masks_path}\nRun batch_cellpose.py --node {args.node} first.")
    print(f"\n3D masks: {masks_path}")
    if args.dry_run:
        return 0
    vol = tifffile.imread(str(masks_path))
    if list(vol.shape) != list(manifest["shape_zyx"]):
        sys.exit(f"Masks shape {vol.shape}, curated volume {manifest['shape_zyx']}")

    preds = [vol[p["z"]] for p in planes]
    rows = [match(p["truth"], m, args.iou) for p, m in zip(planes, preds)]
    for p, r in zip(planes, rows):
        print(f"  z={p['z']:<4} found {r['found']}/{r['n_truth']}  missed {r['missed']}  extra {r['extra']}")
    t = totals(rows)
    print(f"Total (IoU >= {args.iou}): F1 {t['f1']:.3f}  found {t['found']}/{t['n_truth']}  "
          f"missed {t['missed']}  extra {t['extra']}  mean IoU {t['mean_iou']:.2f}")

    out_dir = cdir / f"score_3d_{stamp}"
    out_dir.mkdir(parents=True)
    run_json = masks_path.with_name(masks_path.name.replace("_masks.tif", "_cellpose.json"))
    params = json.loads(run_json.read_text()).get("eval_params") if run_json.exists() else None
    (out_dir / "score.json").write_text(json.dumps(
        {"masks": str(masks_path), "eval_params": params, "iou": args.iou,
         "per_plane": [{"z": p["z"], **r} for p, r in zip(planes, rows)],
         "total": t}, indent=2, default=str))
    render_pages(out_dir, "3d", planes, [("curated", None), ("3D run", preds)], args.iou)
    print(f"Wrote score.json, 3d*.png to {out_dir}")
    return 0


# ---------------------------------------------------------------------------

def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="Save planes and starting outlines for curation.")
    sc = sub.add_parser("score", help="Score every GRID combination, or a 3D run, against the curation.")
    for p in (ex, sc):
        p.add_argument("--node", default="invivo_red", choices=list(TAGS),
                       help="Red 2P node. Default: invivo_red.")
        p.add_argument("--dry-run", action="store_true", help="Print the plan, run nothing.")

    ex.add_argument("--tiff", default=None, help="Volume to curate instead of the node's local_config TIFF.")
    ex.add_argument("--z-planes", default=None, help="Comma-separated 0-indexed Z planes.")
    ex.add_argument("--n", type=int, default=15, help="Without --z-planes: this many evenly spaced planes.")
    ex.add_argument("--blank", action="store_true", help="No starting outlines; draw every cell by hand.")

    sc.add_argument("--iou", type=float, default=0.5, help="Overlap needed for a found cell. Default 0.5.")
    sc.add_argument("--accept-unchanged", default=None,
                    help="Z planes (comma-separated, or 'all') whose exported outlines are used as curated.")
    sc.add_argument("--top", type=int, default=4, help="Combinations drawn in top*.png. Default 4.")
    sc.add_argument("--batch", action="store_true",
                    help="Score batch_cellpose.py's 3D masks for the node instead of the grid.")
    sc.add_argument("--masks", default=None, help="Score this 3D masks tif instead of the grid.")

    args = parser.parse_args(argv)
    return cmd_export(args) if args.cmd == "export" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
