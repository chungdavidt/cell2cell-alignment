r"""
Hand-curated cellpose outlines, and every parameter combination scored against them.

depth   First step for a new TIFF. Without --keep, draws every --every'th plane
        (default 10) into ``<ANALYSIS_ROOT>/cellpose/depth_<tag>/planes_every<N>.png``
        to judge by eye which planes hold cells (surface junk, dim deep planes).
        ``--keep FIRST-LAST`` (or ``all``) records the planes to keep in
        ``<ANALYSIS_ROOT>/cellpose/depth_ranges.json``. batch_cellpose.py refuses
        a TIFF with no range and zeroes cells outside it; export picks planes
        inside it; score, train and stitch leave curated planes outside it out.

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

train   Retrains cellpose's cpsam model on the curated planes, holding some out
        (every 4th by depth, or --test-planes). Scores cpsam and the retrained
        model on the held-out planes only, each with the default settings and
        with the latest score_grid_* best settings, and writes the model,
        losses.csv, scores.json and heldout*.png into
        ``curation_<tag>/train_<YYYYMMDD_HHMMSS>/``. Prints the line for
        batch_cellpose.MODEL_PATHS. --no-holdout trains on every plane and
        scores nothing (the final model once the held-out score is known).

stitch  Takes batch_cellpose.py's output from a run with do_3D False and
        stitch_threshold > 0, splits it back into each plane's outlines, re-joins
        them at each --thresholds value (cellpose's stitch3D: an outline joins
        the cell on the plane above when their IoU is at least the value), drops
        3D cells spanning fewer than each --min-planes value, and scores every
        pair on the curated planes. Writes stitch_<YYYYMMDD_HHMMSS>/stitch.json.
        Run it on a batch made with MIN_PLANES 0: cells the batch already
        dropped are not in its output to re-join. Planes are aligned before
        joining when batch_cellpose.ALIGN_PLANES is on for the node (--no-align).

--skip-planes leaves curated planes out of score, train and stitch.

A plane whose saved outlines are identical to the exported ones counts as not
curated and is left out, unless named in --accept-unchanged (the default outlines
were already right, or a --blank plane has no cells).

The grid runs cellpose in 2D, one plane at a time; batch_cellpose.py runs 3D on the
whole volume, so score its output with --batch after the full run.

Usage (.cellpose-venv):
    .cellpose-venv\Scripts\python.exe alignment\cellpose_curation.py depth                 # look
    .cellpose-venv\Scripts\python.exe alignment\cellpose_curation.py depth --keep 50-400   # record
    .cellpose-venv\Scripts\python.exe alignment\cellpose_curation.py export --z-planes 120,151,180,220
    .cellpose-venv\Scripts\python.exe alignment\cellpose_curation.py export --n 15        # evenly spaced
    .cellpose-venv\Scripts\python.exe -m cellpose                                          # open a z*.tif, fix, Ctrl+S
    .cellpose-venv\Scripts\python.exe alignment\cellpose_curation.py score --dry-run
    .cellpose-venv\Scripts\python.exe alignment\cellpose_curation.py score
    .cellpose-venv\Scripts\python.exe alignment\cellpose_curation.py score --accept-unchanged 151,220
    .cellpose-venv\Scripts\python.exe alignment\cellpose_curation.py score --batch          # 3D run
    .cellpose-venv\Scripts\python.exe alignment\cellpose_curation.py stitch --skip-planes 20,45
    .cellpose-venv\Scripts\python.exe alignment\cellpose_curation.py train --dry-run         # split
    .cellpose-venv\Scripts\python.exe alignment\cellpose_curation.py train
    .cellpose-venv\Scripts\python.exe alignment\cellpose_curation.py train --no-holdout
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

    used_t, used_p, ious, matched = set(), set(), [], []
    for k in np.argsort(-iou, kind="stable"):
        if iou[k] < iou_min:
            break
        if ti[k] in used_t or pi[k] in used_p:
            continue
        used_t.add(ti[k])
        used_p.add(pi[k])
        ious.append(float(iou[k]))
        matched.append(int(p_ids[pi[k]]))

    n_truth = int((t_ids > 0).sum())
    n_pred = int((p_ids > 0).sum())
    found = len(ious)
    return {"found": found, "missed": n_truth - found, "extra": n_pred - found,
            "n_truth": n_truth, "n_pred": n_pred, "iou_sum": float(sum(ious)),
            "matched": matched}


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


def load_planes(cdir: Path, manifest: dict, accept: set | str, skip: set = frozenset()) -> list[dict]:
    """Curated planes as [{z, name, image, truth}]; prints the ones left out and why."""
    import tifffile
    from batch_cellpose import load_depth_range
    keep_z = load_depth_range(cdir.parent, Path(manifest["tiff"]))
    out = []
    print(f"Planes in {cdir}:")
    for z_str, info in sorted(manifest["planes"].items(), key=lambda kv: int(kv[0])):
        z, name = int(z_str), info["name"]
        if z in skip:
            print(f"  z={z:<4} {name}: left out (--skip-planes)")
            continue
        if keep_z and not keep_z[0] <= z <= keep_z[1]:
            print(f"  z={z:<4} {name}: left out (outside depth range {keep_z[0]}-{keep_z[1]})")
            continue
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
    from analysis_paths import TWO_P_RED_NODES, two_p_path, two_p_scope

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
    from batch_cellpose import load_depth_range
    first, last = load_depth_range(cdir.parent, tiff) or (0, n_planes - 1)

    if args.z_planes:
        zs = sorted({int(s) for s in args.z_planes.split(",")})
    else:
        zs = sorted({int(round(v)) for v in np.linspace(first, last, args.n + 2)[1:-1]})
    bad = [z for z in zs if not first <= z <= last]
    if bad:
        sys.exit(f"Z planes {bad} outside {first}-{last} (depth range, or the volume).")

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
                    "scope": two_p_scope(args.node)[0],
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
    planes = load_planes(cdir, manifest, parse_accept(args.accept_unchanged), parse_skip(args.skip_planes))
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


def parse_skip(text: str | None) -> set:
    return {int(t) for t in text.split(",")} if text else set()


def batch_masks_path(args, cdir: Path, manifest: dict) -> Path:
    if args.masks:
        return Path(args.masks)
    return cdir.parent / f"{Path(manifest['tiff']).stem}_masks.tif"


def plane_spans(vol: np.ndarray) -> dict:
    """{label: number of Z planes from its first to its last}."""
    from scipy import ndimage
    return {lab: sl[0].stop - sl[0].start
            for lab, sl in enumerate(ndimage.find_objects(vol), start=1) if sl is not None}


def print_spans(span: dict, rows: list[dict], preds: list, n_z: int) -> None:
    """Planes spanned per 3D cell: all cells, those matching a curated cell, and
    those on a curated plane matching none."""
    found_ids = {lab for r in rows for lab in r["matched"]}
    extra_ids = {int(lab) for m in preds for lab in np.unique(m) if lab > 0} - found_ids
    groups = [("all 3D cells", list(span.values())),
              ("matching a curated cell", [span[l] for l in found_ids]),
              ("on a curated plane, matching none", [span[l] for l in extra_ids])]
    bins = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 20), (21, 50), (51, n_z)]
    print("\nPlanes spanned per 3D cell:")
    print(f"  {'':<40}" + "".join(f"{f'{a}-{b}':>9}" for a, b in bins))
    for name, s in groups:
        s = np.asarray(s)
        print(f"  {f'{name} ({len(s)})':<40}" + "".join(f"{int(((s >= a) & (s <= b)).sum()):>9}" for a, b in bins))


def score_3d(args, cdir, manifest, planes, stamp) -> int:
    import tifffile
    masks_path = batch_masks_path(args, cdir, manifest)
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
    print_spans(plane_spans(vol), rows, preds, vol.shape[0])
    print(f"Wrote score.json, 3d*.png to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# depth
# ---------------------------------------------------------------------------

def cmd_depth(args) -> int:
    import tifffile
    from analysis_paths import TWO_P_RED_NODES, cellpose_dir, two_p_path
    from batch_cellpose import load_depth_range, save_depth_range

    tiff = Path(args.tiff) if args.tiff else two_p_path(args.node)
    if tiff is None:
        sys.exit(f"local_config.{TWO_P_RED_NODES[args.node]} is blank; pass --tiff.")
    if not tiff.exists():
        sys.exit(f"TIFF not found: {tiff}")
    root = cellpose_dir()
    if root is None:
        sys.exit("ANALYSIS_ROOT is blank in local_config.py.")
    with tifffile.TiffFile(str(tiff)) as tf:
        shape = tuple(tf.series[0].shape)
    n_planes = shape[0]
    current = load_depth_range(root, tiff)
    print(f"Volume: {tiff}  shape (Z, Y, X) {shape}")
    print(f"Depth range now: {f'planes {current[0]}-{current[1]}' if current else 'not set'}")

    if args.keep:
        if args.keep == "all":
            first, last = 0, n_planes - 1
        else:
            try:
                first, last = (int(v) for v in args.keep.split("-"))
            except ValueError:
                sys.exit(f"--keep {args.keep!r}: expected FIRST-LAST, e.g. 50-400, or 'all'.")
        if not 0 <= first <= last < n_planes:
            sys.exit(f"--keep {first}-{last} must lie in 0-{n_planes - 1}, first <= last.")
        if args.dry_run:
            print(f"Would set planes {first}-{last}.")
            return 0
        save_depth_range(root, tiff, first, last)
        print(f"Set planes {first}-{last} in {root / 'depth_ranges.json'}")
        return 0

    zs = list(range(0, n_planes, args.every))
    out_dir = root / f"depth_{TAGS[args.node]}"
    png = out_dir / f"planes_every{args.every}.png"
    print(f"Drawing {len(zs)} planes (every {args.every}) into {png}")
    if args.dry_run:
        return 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stack = tifffile.imread(str(tiff))
    cols = 8
    rows = math.ceil(len(zs) / cols)
    tile = 2.0
    fig, axes = plt.subplots(rows, cols, figsize=(cols * tile, rows * tile), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, z in zip(axes.flat, zs):
        img = stack[z].astype(np.float32)
        lo, hi = np.percentile(img, [1, 99.5])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
        out = current is not None and not current[0] <= z <= current[1]
        ax.set_title(f"z={z}{'  (outside range)' if out else ''}", fontsize=8,
                     color="red" if out else "black")
    fig.suptitle(f"{tiff.name}: every {args.every}th plane, each scaled to its own 1-99.5 percentile",
                 fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {png}\nThen record the planes to keep:\n"
          f"  .cellpose-venv\\Scripts\\python.exe alignment\\cellpose_curation.py depth --node {args.node} --keep FIRST-LAST")
    return 0


# ---------------------------------------------------------------------------
# stitch
# ---------------------------------------------------------------------------

def cmd_stitch(args) -> int:
    """Re-join a stitched batch run's per-plane outlines at other overlap cutoffs,
    drop cells spanning fewer than N planes, and score each result."""
    import tifffile
    cdir = curation_dir(args.node)
    manifest = load_manifest(cdir)
    planes = load_planes(cdir, manifest, parse_accept(args.accept_unchanged), parse_skip(args.skip_planes))
    masks_path = batch_masks_path(args, cdir, manifest)
    if not masks_path.exists():
        sys.exit(f"Masks not found: {masks_path}\nRun batch_cellpose.py --node {args.node} first.")
    thresholds = [float(t) for t in args.thresholds.split(",")]
    mins = [int(k) for k in args.min_planes.split(",")]
    print(f"\n3D masks: {masks_path}\nStitch thresholds {thresholds}, min planes {mins}")
    if args.dry_run:
        return 0

    from cellpose.utils import fill_holes_and_remove_small_masks, stitch3D
    from batch_cellpose import ALIGN_PLANES, plane_offsets, stitch_aligned
    align = ALIGN_PLANES.get(args.node, False) and not args.no_align
    offsets = plane_offsets(tifffile.imread(manifest["tiff"])) if align else None
    print(f"Align planes before joining: {align}")
    vol = tifffile.imread(str(masks_path))
    if list(vol.shape) != list(manifest["shape_zyx"]):
        sys.exit(f"Masks shape {vol.shape}, curated volume {manifest['shape_zyx']}")
    # Each plane's outlines numbered 1..n on their own, as cellpose has them before stitching.
    flat = np.zeros(vol.shape, dtype=np.int32)
    for z in range(vol.shape[0]):
        ids, inv = np.unique(vol[z], return_inverse=True)
        flat[z] = inv.reshape(vol.shape[1:]) + (0 if ids[0] == 0 else 1)
    del vol

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    print(f"\n  {'stitch':>6} {'min':>4} {'cells':>6} {'F1':>6} {'found':>9} {'extra':>6} "
          f"{'3D cells found':>15} {'of them >50 planes':>19}")
    for t in thresholds:
        joined = stitch_aligned(flat, offsets, t) if align else stitch3D(flat.copy(), stitch_threshold=t)
        st = fill_holes_and_remove_small_masks(joined, min_size=8)
        span = plane_spans(st)
        labs = np.fromiter(span.keys(), dtype=np.int64, count=len(span))
        lens = np.fromiter(span.values(), dtype=np.int64, count=len(span))
        for k in mins:
            keep = np.zeros(int(st.max()) + 1, dtype=bool)
            keep[labs[lens >= k]] = True
            preds = [np.where(keep[st[p["z"]]], st[p["z"]], 0) for p in planes]
            rows = [match(p["truth"], m, args.iou) for p, m in zip(planes, preds)]
            tot = totals(rows)
            found_ids = {lab for r in rows for lab in r["matched"]}
            long_ = sum(span[l] > 50 for l in found_ids)
            n = int((lens >= k).sum())
            print(f"  {t:>6.2f} {k:>4} {n:>6} {tot['f1']:>6.3f} {tot['found']:>4}/{tot['n_truth']:<4} "
                  f"{tot['extra']:>6} {len(found_ids):>15} {long_:>19}")
            results.append({"stitch_threshold": t, "min_planes": k, "n_cells": n,
                            "n_found_3d": len(found_ids), "n_found_3d_over_50_planes": long_,
                            **{key: v for key, v in tot.items()}})
        del st

    out_dir = cdir / f"stitch_{stamp}"
    out_dir.mkdir(parents=True)
    (out_dir / "stitch.json").write_text(json.dumps(
        {"masks": str(masks_path), "iou": args.iou, "planes": [p["z"] for p in planes],
         "results": results}, indent=2))
    print(f"\n'3D cells found': distinct 3D cells matching a curated cell. Fewer than 'found'\n"
          f"means curated cells on different planes fell into one 3D cell.\nWrote {out_dir / 'stitch.json'}")
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def latest_grid_best(cdir: Path) -> dict | None:
    runs = sorted(cdir.glob("score_grid_*/best.json"))
    if not runs:
        return None
    best = json.loads(runs[-1].read_text())
    return {"source": str(runs[-1]), "params": best["best_params"]}


def cmd_train(args) -> int:
    cdir = curation_dir(args.node)
    manifest = load_manifest(cdir)
    planes = load_planes(cdir, manifest, parse_accept(args.accept_unchanged), parse_skip(args.skip_planes))
    zs = [p["z"] for p in planes]

    if args.no_holdout:
        test_z = []
    elif args.test_planes:
        test_z = sorted({int(s) for s in args.test_planes.split(",")})
        if set(test_z) - set(zs):
            sys.exit(f"--test-planes {sorted(set(test_z) - set(zs))} are not curated planes.")
    else:
        test_z = zs[1::4]
    train = [p for p in planes if p["z"] not in test_z]
    test = [p for p in planes if p["z"] in test_z]
    if not train:
        sys.exit("No planes left to train on.")
    print(f"\nTrain: {len(train)} planes, {sum(n_cells(p['truth']) for p in train)} cells  "
          f"z {[p['z'] for p in train]}")
    print(f"Held out: {len(test)} planes, {sum(n_cells(p['truth']) for p in test)} cells  "
          f"z {[p['z'] for p in test]}")
    grid = latest_grid_best(cdir)
    if test:
        print(f"Grid best settings: {combo_label(grid['params']) if grid else 'none (no score_grid_* run)'}")
    print(f"Epochs {args.epochs}, learning rate {args.learning_rate}")
    if args.dry_run:
        return 0

    import torch
    from cellpose import train as cp_train
    from cellpose.models import CellposeModel

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cdir / f"train_{stamp}"
    out_dir.mkdir(parents=True)
    print(f"Output: {out_dir}\nLoading cpsam model...")
    model = CellposeModel(gpu=True, pretrained_model="cpsam")
    t0 = time.time()
    model_path, train_losses, test_losses = cp_train.train_seg(
        model.net,
        train_data=[p["image"] for p in train], train_labels=[p["truth"] for p in train],
        test_data=[p["image"] for p in test] or None,
        test_labels=[p["truth"] for p in test] or None,
        learning_rate=args.learning_rate, weight_decay=0.1, n_epochs=args.epochs,
        min_train_masks=1, save_path=str(out_dir),
        model_name=f"cpsam_{TAGS[args.node]}_{stamp}")
    model_path = Path(model_path)
    print(f"Trained in {(time.time() - t0) / 60:.1f} min: {model_path}")
    del model
    torch.cuda.empty_cache()

    with open(out_dir / "losses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "test_loss"])
        for e, tr in enumerate(np.asarray(train_losses)):
            # train_seg fills test_loss only on epoch 0, 5 and every 10th; the rest stay 0.
            te = np.asarray(test_losses)[e] if test_losses is not None and len(test_losses) > e else 0
            w.writerow([e, float(tr), float(te) if te else ""])

    record = {"node": args.node, "model": str(model_path), "epochs": args.epochs,
              "learning_rate": args.learning_rate, "weight_decay": 0.1,
              "train_planes": [p["z"] for p in train], "test_planes": [p["z"] for p in test],
              "iou": args.iou, "grid_best": grid, "scores": {}}

    if test:
        settings = [("default settings", vanilla_kwargs())]
        if grid:
            settings.append(("grid best settings", build_kwargs(grid["params"])))
        columns = [("curated", None)]
        for model_name, label in (("cpsam", "cpsam"), (str(model_path), "retrained")):
            m = CellposeModel(gpu=True, pretrained_model=model_name)
            for s_label, kw in settings:
                preds = [m.eval(p["image"], **kw)[0] for p in test]
                t = totals([match(p["truth"], pr, args.iou) for p, pr in zip(test, preds)])
                key = f"{label}, {s_label}"
                record["scores"][key] = t
                columns.append((key, preds))
            del m
            torch.cuda.empty_cache()

        print(f"\nHeld-out planes {[p['z'] for p in test]} (IoU >= {args.iou}):")
        for key, t in record["scores"].items():
            print(f"  {key:<34} F1 {t['f1']:.3f}  found {t['found']}/{t['n_truth']}  "
                  f"missed {t['missed']}  extra {t['extra']}")
        if grid:
            print("  (the grid best settings were chosen on all curated planes, held-out ones included)")
        render_pages(out_dir, "heldout", test, columns, args.iou)

    (out_dir / "scores.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"\nFor MODEL_PATHS in alignment/batch_cellpose.py:\n    \"{args.node}\": r\"{model_path}\",")
    print(f"Wrote losses.csv, scores.json{', heldout*.png' if test else ''} to {out_dir}")
    return 0


# ---------------------------------------------------------------------------

def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="Save planes and starting outlines for curation.")
    sc = sub.add_parser("score", help="Score every GRID combination, or a 3D run, against the curation.")
    tr = sub.add_parser("train", help="Retrain cpsam on the curated planes; score on held-out planes.")
    st = sub.add_parser("stitch", help="Re-join a stitched batch run at other cutoffs; score each.")
    dp = sub.add_parser("depth", help="Draw planes to judge by eye; --keep records the Z planes to segment.")
    for p in (ex, sc, tr, st, dp):
        p.add_argument("--node", default="invivo_red", choices=list(TAGS),
                       help="Red 2P node. Default: invivo_red.")
        p.add_argument("--dry-run", action="store_true", help="Print the plan, run nothing.")

    ex.add_argument("--tiff", default=None, help="Volume to curate instead of the node's local_config TIFF.")
    ex.add_argument("--z-planes", default=None, help="Comma-separated 0-indexed Z planes.")
    ex.add_argument("--n", type=int, default=15, help="Without --z-planes: this many evenly spaced planes.")
    ex.add_argument("--blank", action="store_true", help="No starting outlines; draw every cell by hand.")

    for p in (sc, tr, st):
        p.add_argument("--iou", type=float, default=0.5, help="Overlap needed for a found cell. Default 0.5.")
        p.add_argument("--accept-unchanged", default=None,
                       help="Z planes (comma-separated, or 'all') whose exported outlines are used as curated.")
        p.add_argument("--skip-planes", default=None,
                       help="Curated Z planes to leave out, comma-separated (e.g. surface planes).")
    sc.add_argument("--top", type=int, default=4, help="Combinations drawn in top*.png. Default 4.")
    sc.add_argument("--batch", action="store_true",
                    help="Score batch_cellpose.py's 3D masks for the node instead of the grid.")
    sc.add_argument("--masks", default=None, help="Score this 3D masks tif instead of the grid.")

    tr.add_argument("--test-planes", default=None,
                    help="Held-out Z planes, comma-separated. Default: every 4th curated plane by depth.")
    tr.add_argument("--no-holdout", action="store_true", help="Train on every curated plane, score nothing.")
    tr.add_argument("--epochs", type=int, default=100, help="Default 100.")
    tr.add_argument("--learning-rate", type=float, default=1e-5, help="Default 1e-5.")

    dp.add_argument("--tiff", default=None, help="Volume instead of the node's local_config TIFF.")
    dp.add_argument("--every", type=int, default=10, help="Draw every Nth plane. Default 10.")
    dp.add_argument("--keep", default=None, help="Record the planes to keep: FIRST-LAST (inclusive) or 'all'.")

    st.add_argument("--masks", default=None, help="Stitched masks tif. Default: the node's batch_cellpose.py output.")
    st.add_argument("--thresholds", default="0.25,0.4,0.5,0.6,0.75",
                    help="Overlap (IoU) needed to join outlines on neighbouring planes. Default 0.25,0.4,0.5,0.6,0.75.")
    st.add_argument("--no-align", action="store_true",
                    help="Join without aligning planes, even if batch_cellpose.ALIGN_PLANES is on for the node.")
    st.add_argument("--min-planes", default="1,3,5,7",
                    help="Drop 3D cells spanning fewer planes. Default 1,3,5,7.")

    args = parser.parse_args(argv)
    return {"export": cmd_export, "score": cmd_score, "train": cmd_train, "stitch": cmd_stitch,
            "depth": cmd_depth}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
