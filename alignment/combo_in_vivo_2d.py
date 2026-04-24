"""
Combo 2D cellpose parameter sweep — data-driven follow-up to sweep_in_vivo_2d.py.

Reads a prior sweep's summary.csv + config.json, picks the top-2 values
strictly superior to vanilla for each knob (pins to vanilla when no superior
value exists, pairs the single winner with vanilla when only one exists),
then runs the full Cartesian product of those picks on the same Z planes as
the source sweep. Each combo is scored by an additive model
``vanilla_sum + Σ (pick_sum − vanilla_sum)`` and rendered in score-descending
order, paginated so every PNG fits ``[vanilla, C?, C?, C?, C?]``. A
predicted-vs-actual pivot at the end surfaces knob-interaction residuals.

Design is agnostic: knob identity, kinds, and authoritative values come from
the source sweep's config.json, so any future sweep with a different knob set
works without code changes.

Outputs go to ``<tiff parent>/combo_in_vivo_<YYYYMMDD_HHMMSS>/`` — never into
the repo.

Usage:
    source .cellpose-venv/bin/activate          # Linux/Mac
    # .cellpose-venv\Scripts\activate            # Windows
    python alignment/combo_in_vivo_2d.py                               # auto-pick latest sweep next to local_config.INVIVO_PATH
    python alignment/combo_in_vivo_2d.py --dry-run                     # auto-pick + preview
    python alignment/combo_in_vivo_2d.py --from-sweep <sweep_dir>      # explicit override
    python alignment/combo_in_vivo_2d.py --tiff <path>                 # auto-pick latest sweep next to a non-default TIFF
    python alignment/combo_in_vivo_2d.py --from-sweep <sweep_dir> --z-planes 25,98,108
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VANILLA_SENTINEL  = "__VANILLA_NOOP__"   # in-memory marker, never written to CSV
CONTOUR_COLOR     = "#00E5E5"
MAX_COLS_PER_PAGE = 5            # vanilla + <=4 combos per page
CELL_SIZE_INCHES  = 2.5
DPI               = 150
BATCH_SIZE        = 8


# ---------------------------------------------------------------------------
# _Tee (copied from alignment/sweep_in_vivo_2d.py — alignment/ has no __init__.py)
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, s):
        self.stream.write(s)
        self.log_file.write(s)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def fileno(self):
        return self.stream.fileno()


# ---------------------------------------------------------------------------
# Helpers (format/eval/render — copied from sweep_in_vivo_2d.py)
# ---------------------------------------------------------------------------

def format_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return str(list(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def eval_one(model, plane: np.ndarray, kwargs: dict) -> tuple[np.ndarray, float]:
    t0 = time.time()
    masks, _flows, _styles = model.eval(plane, **kwargs)
    return masks, time.time() - t0


def draw_contours(ax, masks: np.ndarray) -> None:
    """Draw per-label outlines in one ax.contour call (matplotlib-only, no skimage)."""
    m = int(masks.max()) if masks.size else 0
    if m == 0:
        return
    levels = np.arange(0.5, m + 0.5, 1.0)
    try:
        ax.contour(masks, levels=levels, colors=CONTOUR_COLOR, linewidths=0.8)
    except Exception as exc:
        print(f"    (contour draw skipped: {type(exc).__name__}: {exc})")


def render_page(out_path: Path, stack: np.ndarray, z_indices: list[int],
                col_masks: list[list[np.ndarray]], col_headers: list[str],
                suptitle: str) -> None:
    n_rows = len(z_indices)
    n_cols = len(col_headers)
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(CELL_SIZE_INCHES * n_cols, CELL_SIZE_INCHES * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    for r, z in enumerate(z_indices):
        plane = stack[z]
        v_lo, v_hi = np.percentile(plane, [1, 99])
        for c, header in enumerate(col_headers):
            ax = axes[r, c]
            ax.imshow(plane, cmap="gray", vmin=v_lo, vmax=v_hi)
            masks = col_masks[r][c]
            draw_contours(ax, masks)
            n = int(masks.max()) if masks.size else 0
            ax.text(
                0.98, 0.98, f"n={n}",
                transform=ax.transAxes, ha="right", va="top",
                color="#FFE066", fontsize=7,
                bbox=dict(facecolor="black", alpha=0.5, pad=1.5, edgecolor="none"),
            )
            if r == 0:
                ax.set_title(header, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"Z={z}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep-input loading + per-knob pick logic
# ---------------------------------------------------------------------------

def find_latest_sweep(search_dir: Path) -> Path | None:
    """Return the most-recent valid ``sweep_in_vivo_*`` dir in search_dir, or None.

    Timestamp naming (``sweep_in_vivo_YYYYMMDD_HHMMSS``) sorts lexically.
    A dir must contain both ``summary.csv`` and ``config.json`` to be returned.
    """
    if not search_dir.is_dir():
        return None
    candidates = sorted(search_dir.glob("sweep_in_vivo_*"),
                        key=lambda p: p.name, reverse=True)
    for c in candidates:
        if c.is_dir() and (c / "summary.csv").exists() and (c / "config.json").exists():
            return c
    return None


def load_sweep_inputs(sweep_dir: Path) -> tuple[list[dict], dict]:
    csv_path = sweep_dir / "summary.csv"
    cfg_path = sweep_dir / "config.json"
    if not csv_path.exists():
        sys.exit(f"Missing summary.csv in source sweep: {csv_path}")
    if not cfg_path.exists():
        sys.exit(f"Missing config.json in source sweep: {cfg_path}")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(cfg_path) as f:
        cfg = json.load(f)
    return rows, cfg


def coerce_value(val_str: str, authoritative_values: list):
    """Invert format_value: match CSV string back to its authoritative-typed value."""
    for v in authoritative_values:
        if format_value(v) == val_str:
            return v
    raise ValueError(
        f"Cannot coerce CSV value {val_str!r} to any authoritative value in "
        f"{authoritative_values!r}"
    )


def aggregate_sums(summary_rows: list[dict], sweeps_config: list[dict]
                   ) -> tuple[dict, dict]:
    """Dedupe by (param, value_str, z_plane), then sum n_rois across Z per value.

    Returns (sums[param][value_str], vanilla_sums[param]).
    """
    known_params = {s["param"] for s in sweeps_config}
    seen: dict[tuple, int] = {}
    for row in summary_rows:
        p = row["param"]
        if p not in known_params:
            raise ValueError(
                f"summary.csv has param {p!r} not found in config.json sweeps "
                f"(known: {sorted(known_params)})"
            )
        z = int(row["z_plane"])
        val_str = row["value"]
        n = int(row["n_rois"])
        key = (p, val_str, z)
        if key in seen:
            if seen[key] != n:
                raise ValueError(
                    f"summary.csv has inconsistent n_rois for {key!r}: "
                    f"{seen[key]} vs {n} — source sweep is malformed."
                )
            continue
        seen[key] = n

    sums: dict[str, dict[str, int]] = {}
    vanilla_sums: dict[str, int] = {}
    for (p, val_str, _z), n in seen.items():
        if val_str == "vanilla":
            vanilla_sums[p] = vanilla_sums.get(p, 0) + n
        else:
            sums.setdefault(p, {}).setdefault(val_str, 0)
            sums[p][val_str] += n

    for s in sweeps_config:
        if s["param"] not in vanilla_sums:
            raise ValueError(
                f"No vanilla rows in summary.csv for param {s['param']!r}"
            )
    return sums, vanilla_sums


def pick_per_knob(sums: dict, vanilla_sums: dict, sweeps_config: list[dict]
                  ) -> dict[str, list]:
    """Top-2 strictly superior to vanilla, sentinel-filled where fewer exist."""
    picks: dict[str, list] = {}
    for s in sweeps_config:
        p = s["param"]
        authoritative = s["values"]
        superior = [
            (vs, n) for vs, n in sums.get(p, {}).items()
            if n > vanilla_sums[p]
        ]
        superior.sort(key=lambda vn: (-vn[1], vn[0]))
        if not superior:
            picks[p] = [VANILLA_SENTINEL]
        elif len(superior) == 1:
            picks[p] = [coerce_value(superior[0][0], authoritative),
                        VANILLA_SENTINEL]
        else:
            picks[p] = [coerce_value(superior[0][0], authoritative),
                        coerce_value(superior[1][0], authoritative)]
    return picks


# ---------------------------------------------------------------------------
# Combo kwargs + score
# ---------------------------------------------------------------------------

def build_combo_kwargs(combo: dict, vanilla_normalize_dict: dict,
                       vanilla_eval: dict, kinds_by_param: dict) -> dict:
    """N-way-override version of sweep's build_sweep_kwargs.

    If any normalize-family knob has a non-sentinel pick, build a dict from
    cellpose defaults and apply all normalize overrides at once; otherwise use
    ``normalize=True`` (bool form, matches the vanilla column).
    """
    norm_overrides = {p: v for p, v in combo.items()
                      if v is not VANILLA_SENTINEL and kinds_by_param[p] == "normalize"}
    eval_overrides = {p: v for p, v in combo.items()
                      if v is not VANILLA_SENTINEL and kinds_by_param[p] == "eval"}
    if norm_overrides:
        norm = deepcopy(vanilla_normalize_dict)
        norm.update(norm_overrides)
    else:
        norm = True
    ev = deepcopy(vanilla_eval)
    ev.update(eval_overrides)
    return {"do_3D": False, "normalize": norm, **ev, "batch_size": BATCH_SIZE}


def predicted_score(combo: dict, sums: dict, vanilla_sums: dict,
                    vanilla_sum_one_run: int) -> float:
    return vanilla_sum_one_run + sum(
        sums[p][format_value(v)] - vanilla_sums[p]
        for p, v in combo.items() if v is not VANILLA_SENTINEL
    )


def legend_cell(v) -> str:
    if v is VANILLA_SENTINEL:
        return "<vanilla>"
    if isinstance(v, (list, tuple)):
        return json.dumps(list(v))
    return str(v)


def config_pick(v):
    """JSON-native rendering for config.json (preserves int/float/list types)."""
    if v is VANILLA_SENTINEL:
        return "<vanilla>"
    return v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-sweep", default=None,
        help="Path to source sweep directory (must contain summary.csv + config.json). "
             "Default: most recent sweep_in_vivo_* next to --tiff, or next to "
             "local_config.INVIVO_PATH when --tiff isn't passed either.",
    )
    parser.add_argument(
        "--tiff", default=None,
        help="Override input TIFF. Default: source sweep's tiff_path.",
    )
    parser.add_argument(
        "--z-planes", default=None,
        help="Comma-separated Z indices. Default: source sweep's z_planes. "
             "Planes not present in the source trigger a WARNING because predictions "
             "come from source sums (no apples-to-apples delta).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the full plan without loading the model or writing anything.",
    )
    args = parser.parse_args(argv)

    if args.from_sweep:
        sweep_dir = Path(args.from_sweep).resolve()
        if not sweep_dir.exists():
            sys.exit(f"Source sweep directory not found: {sweep_dir}")
        if not sweep_dir.is_dir():
            sys.exit(f"--from-sweep is not a directory: {sweep_dir}")
    else:
        if args.tiff:
            search_near = Path(args.tiff).resolve().parent
            search_label = f"--tiff parent ({search_near})"
        else:
            project_root = Path(__file__).resolve().parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            try:
                import local_config
            except ImportError:
                sys.exit("No --from-sweep passed and local_config.py not importable. "
                         "Pass --from-sweep explicitly.")
            invivo_path = getattr(local_config, "INVIVO_PATH", None)
            if not invivo_path:
                sys.exit("No --from-sweep passed and local_config.INVIVO_PATH is empty. "
                         "Pass --from-sweep explicitly or set INVIVO_PATH.")
            search_near = Path(invivo_path).parent
            search_label = f"local_config.INVIVO_PATH parent ({search_near})"
        sweep_dir = find_latest_sweep(search_near)
        if sweep_dir is None:
            sys.exit(f"No sweep_in_vivo_* dir with summary.csv + config.json found in "
                     f"{search_label}. Pass --from-sweep explicitly.")
        print(f"Auto-picked most recent sweep: {sweep_dir.name}")
        print(f"  (from {search_label})")

    summary_rows, source_cfg = load_sweep_inputs(sweep_dir)
    sweeps_config = source_cfg["sweeps"]
    vanilla_normalize      = source_cfg["vanilla_normalize"]       # bool (echoed to our config.json)
    vanilla_normalize_dict = source_cfg["vanilla_normalize_dict"]
    vanilla_eval           = source_cfg["vanilla_eval"]

    sums, vanilla_sums = aggregate_sums(summary_rows, sweeps_config)
    if len(set(vanilla_sums.values())) != 1:
        raise AssertionError(
            f"Vanilla sums differ across knobs: {vanilla_sums!r}. "
            f"Source sweep summary.csv appears malformed "
            f"(vanilla should be identical across knobs since it's cached per Z)."
        )
    vanilla_sum_one_run = next(iter(vanilla_sums.values()))
    picks = pick_per_knob(sums, vanilla_sums, sweeps_config)

    param_names    = [s["param"] for s in sweeps_config]
    kinds_by_param = {s["param"]: s["kind"] for s in sweeps_config}
    combo_value_lists = [picks[p] for p in param_names]
    combos_raw = list(itertools.product(*combo_value_lists))
    combos = [dict(zip(param_names, vals)) for vals in combos_raw]

    # Score + sort (desc score, asc n_nv, asc orig_idx).
    scored: list[tuple[float, int, int, dict]] = []
    for idx, combo in enumerate(combos):
        score = predicted_score(combo, sums, vanilla_sums, vanilla_sum_one_run)
        n_nv = sum(1 for v in combo.values() if v is not VANILLA_SENTINEL)
        scored.append((score, n_nv, idx, combo))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))

    tiff_path = Path(args.tiff) if args.tiff else Path(source_cfg["tiff_path"])
    if not tiff_path.exists():
        sys.exit(f"TIFF not found: {tiff_path}")

    source_z = [int(z) for z in source_cfg["z_planes"]]
    if args.z_planes:
        z_indices = [int(x) for x in args.z_planes.split(",")]
        not_in_source = [z for z in z_indices if z not in source_z]
        if not_in_source:
            print(f"WARNING: --z-planes contains {not_in_source} not in source sweep "
                  f"{source_z}. Predicted-vs-actual comparison will not be apples-to-apples "
                  f"because predicted scores come from source sums. Proceeding anyway.")
    else:
        z_indices = source_z

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = tiff_path.parent / f"combo_in_vivo_{stamp}"

    n_combos      = len(combos)
    page_size     = MAX_COLS_PER_PAGE - 1
    n_pages       = max(1, math.ceil(n_combos / page_size))
    n_total_evals = len(z_indices) * (1 + n_combos)

    all_pinned = all(picks[p] == [VANILLA_SENTINEL] for p in param_names)

    if args.dry_run:
        print("DRY RUN — combo plan")
        print(f"  Source sweep: {sweep_dir}")
        print(f"  TIFF:         {tiff_path}")
        print(f"  Output dir (would create): {out_dir}")
        print(f"  Z planes: {z_indices}")
        print(f"  vanilla_sum (one run, from source): {vanilla_sum_one_run}")
        print(f"  Picks per knob:")
        for s in sweeps_config:
            p = s["param"]
            disp = [legend_cell(v) for v in picks[p]]
            print(f"    {p} (vanilla={vanilla_sums[p]}): {disp}")
        shape_str = " x ".join(str(len(pl)) for pl in combo_value_lists)
        print(f"  n_combos:      {n_combos}  ({shape_str})")
        print(f"  n_pages:       {n_pages}  (up to {page_size} combos per page)")
        print(f"  n_total_evals: {n_total_evals}  "
              f"({len(z_indices)} vanilla + {n_combos} x {len(z_indices)})")
        if all_pinned:
            print(f"  WARNING: All knobs pinned to vanilla — no combos will be evaluated.")
        print()
        for combo_id, (score, nnv, _orig_idx, combo) in enumerate(scored, start=1):
            desc = "  ".join(f"{p}={legend_cell(combo[p])}" for p in param_names)
            print(f"  C{combo_id:<3}  pred {score:>5.1f}  nnv={nnv}  {desc}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=False)

    log_path = out_dir / "combo.log"
    real_stdout, real_stderr = sys.stdout, sys.stderr
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(real_stdout, log_f)
    sys.stderr = _Tee(real_stderr, log_f)

    exit_code = 0
    try:
        config = {
            "tiff_path":              str(tiff_path),
            "sweep_dir":              str(sweep_dir),
            "z_planes":               list(z_indices),
            "vanilla_normalize":      vanilla_normalize,
            "vanilla_normalize_dict": vanilla_normalize_dict,
            "vanilla_eval":           vanilla_eval,
            "sweeps":                 sweeps_config,
            "picks_per_knob":         {p: [config_pick(v) for v in picks[p]]
                                       for p in param_names},
            "vanilla_sum_one_run":    vanilla_sum_one_run,
            "n_combos":               n_combos,
            "n_total_evals":          n_total_evals,
            "contour_color":          CONTOUR_COLOR,
            "max_cols_per_page":      MAX_COLS_PER_PAGE,
            "batch_size":             BATCH_SIZE,
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"Wrote config.json")

        legend_path = out_dir / "combo_legend.csv"
        with open(legend_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["combo_id", "predicted_score", *param_names])
            writer.writerow(
                ["vanilla", vanilla_sum_one_run, *[legend_cell(VANILLA_SENTINEL)] * len(param_names)]
            )
            for combo_id, (score, _nnv, _orig_idx, combo) in enumerate(scored, start=1):
                writer.writerow(
                    [f"C{combo_id}", score, *[legend_cell(combo[p]) for p in param_names]]
                )
        print(f"Wrote combo_legend.csv")

        if all_pinned:
            print("\nWARNING: All knobs pinned to vanilla — no combos to evaluate.")
            print("Writing empty summary.csv and exiting (skipping model load).")
            sum_path = out_dir / "summary.csv"
            with open(sum_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "combo_id", "predicted_score", "z_plane", "page", "n_rois",
                    "median_size_px", "n_tiny_below_2x_min_size", "runtime_s",
                ])
                writer.writeheader()
            return 0

        print(f"\nLoading TIFF: {tiff_path}")
        t0 = time.time()
        stack = tifffile.imread(str(tiff_path))
        print(f"  shape: {stack.shape}, dtype: {stack.dtype}, load: {time.time()-t0:.1f}s")
        if stack.ndim != 3:
            sys.exit(f"Expected 3D (Z, Y, X) stack, got ndim={stack.ndim}")
        for z in z_indices:
            if not (0 <= z < stack.shape[0]):
                sys.exit(f"Z plane {z} out of range [0, {stack.shape[0]}).")

        print("\nLoading cpsam model...")
        t0 = time.time()
        from cellpose.models import CellposeModel
        model = CellposeModel(gpu=True, pretrained_model="cpsam")
        print(f"  loaded in {time.time() - t0:.1f}s (device={model.device})")

        tiny_thresh = 2 * vanilla_eval["min_size"]

        def plane_stats(masks: np.ndarray) -> tuple[int, int, int]:
            n = int(masks.max()) if masks.size else 0
            if n > 0:
                sizes = np.bincount(masks.ravel())[1:]
                median_size = int(np.median(sizes))
                n_tiny = int((sizes < tiny_thresh).sum())
            else:
                median_size = 0
                n_tiny = 0
            return n, median_size, n_tiny

        combo_page: dict[int, int] = {}
        for page_idx in range(1, n_pages + 1):
            start = (page_idx - 1) * page_size
            end   = min(start + page_size, n_combos)
            for i in range(start, end):
                combo_page[i + 1] = page_idx

        vanilla_masks: dict[int, np.ndarray] = {}
        vanilla_stats: dict[int, tuple[int, int, int, float]] = {}
        print("\nEvaluating vanilla on each Z plane (cached across combos)...")
        vanilla_kwargs = {
            "do_3D": False, "normalize": True, **vanilla_eval, "batch_size": BATCH_SIZE,
        }
        for z in z_indices:
            plane = stack[z]
            print(f"  Z={z:>4}  vanilla ... ", end="", flush=True)
            v_masks, v_dt = eval_one(model, plane, vanilla_kwargs)
            n, msz, tiny = plane_stats(v_masks)
            vanilla_masks[z] = v_masks
            vanilla_stats[z] = (n, msz, tiny, v_dt)
            print(f"n={n} ({v_dt:.1f}s)")

        combo_masks: dict[int, dict[int, np.ndarray]] = {}
        combo_stats: dict[int, dict[int, tuple[int, int, int, float]]] = {}
        print("\nEvaluating combos...")
        for combo_id, (score, _nnv, _orig_idx, combo) in enumerate(scored, start=1):
            kwargs = build_combo_kwargs(combo, vanilla_normalize_dict, vanilla_eval, kinds_by_param)
            desc = "  ".join(f"{p}={legend_cell(combo[p])}" for p in param_names)
            print(f"\n--- C{combo_id} (pred {score:.1f}, page {combo_page[combo_id]}/{n_pages})  {desc} ---")
            combo_masks[combo_id] = {}
            combo_stats[combo_id] = {}
            for z in z_indices:
                plane = stack[z]
                print(f"  Z={z:>4}  C{combo_id} ... ", end="", flush=True)
                masks, dt = eval_one(model, plane, kwargs)
                n, msz, tiny = plane_stats(masks)
                combo_masks[combo_id][z] = masks
                combo_stats[combo_id][z] = (n, msz, tiny, dt)
                print(f"n={n} ({dt:.1f}s)")

        print("\nRendering pages...")
        for page_idx in range(1, n_pages + 1):
            start = (page_idx - 1) * page_size
            end   = min(start + page_size, n_combos)
            page_combo_ids = list(range(start + 1, end + 1))
            if not page_combo_ids:
                continue
            col_headers = ["vanilla"] + [
                f"C{cid}\npred {scored[cid - 1][0]:.0f}" for cid in page_combo_ids
            ]
            col_masks: list[list[np.ndarray]] = []
            for z in z_indices:
                row = [vanilla_masks[z]]
                for cid in page_combo_ids:
                    row.append(combo_masks[cid][z])
                col_masks.append(row)
            hi = scored[page_combo_ids[0] - 1][0]
            lo = scored[page_combo_ids[-1] - 1][0]
            first, last = page_combo_ids[0], page_combo_ids[-1]
            if n_pages == 1:
                out_path = out_dir / "combos.png"
                suptitle = f"combos: C{first}..C{last}, predicted {hi:.0f}–{lo:.0f}"
            else:
                out_path = out_dir / f"combos_p{page_idx}.png"
                suptitle = (f"combos (page {page_idx}/{n_pages}): "
                            f"C{first}..C{last}, predicted {hi:.0f}–{lo:.0f}")
            render_page(out_path, stack, z_indices, col_masks, col_headers, suptitle)
            print(f"  wrote {out_path.name}")

        sum_path = out_dir / "summary.csv"
        with open(sum_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "combo_id", "predicted_score", "z_plane", "page", "n_rois",
                "median_size_px", "n_tiny_below_2x_min_size", "runtime_s",
            ])
            writer.writeheader()
            for z in z_indices:
                n, msz, tiny, rt = vanilla_stats[z]
                writer.writerow({
                    "combo_id":                  "vanilla",
                    "predicted_score":           "",
                    "z_plane":                   z,
                    "page":                      0,
                    "n_rois":                    n,
                    "median_size_px":            msz,
                    "n_tiny_below_2x_min_size":  tiny,
                    "runtime_s":                 round(rt, 2),
                })
            for combo_id, (score, _nnv, _orig_idx, _combo) in enumerate(scored, start=1):
                for z in z_indices:
                    n, msz, tiny, rt = combo_stats[combo_id][z]
                    writer.writerow({
                        "combo_id":                  f"C{combo_id}",
                        "predicted_score":           score,
                        "z_plane":                   z,
                        "page":                      combo_page[combo_id],
                        "n_rois":                    n,
                        "median_size_px":            msz,
                        "n_tiny_below_2x_min_size":  tiny,
                        "runtime_s":                 round(rt, 2),
                    })
        print(f"\nWrote summary.csv")

        print()
        print("=" * 64)
        print("PREDICTED vs ACTUAL ROI TOTALS (sum across Z)")
        print("=" * 64)
        print(f"{'combo_id':<8} | {'predicted':>9} | {'actual_total':>12} | {'delta':>6}")
        print(f"{'-'*8}-+-{'-'*9}-+-{'-'*12}-+-{'-'*6}")
        vanilla_actual = sum(vanilla_stats[z][0] for z in z_indices)
        print(f"{'vanilla':<8} | {vanilla_sum_one_run:>9.1f} | "
              f"{vanilla_actual:>12} | {vanilla_actual - vanilla_sum_one_run:>+6.1f}")
        deltas: list[float] = []
        for combo_id, (score, _nnv, _orig_idx, _combo) in enumerate(scored, start=1):
            actual = sum(combo_stats[combo_id][z][0] for z in z_indices)
            delta = actual - score
            deltas.append(delta)
            print(f"{'C' + str(combo_id):<8} | {score:>9.1f} | "
                  f"{actual:>12} | {delta:>+6.1f}")
        if deltas:
            mean_abs = float(np.mean(np.abs(deltas)))
            print(f"\nAdditive-model residuals: "
                  f"mean|delta|={mean_abs:.2f}, min={min(deltas):+.1f}, max={max(deltas):+.1f}")

        print(f"\nAll outputs: {out_dir}")

    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout = real_stdout
        sys.stderr = real_stderr
        log_f.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
