"""
2D cellpose parameter sweep on a single TIFF, with visual QC grids.

Runs OFAT sweeps of cellpose normalize + eval parameters on 5 chosen Z planes
of a 3D TIFF and emits one PNG per sweep (paginated to <=5 cols) with ROI
contours drawn on the raw plane. Decision metric is visual overlay quality
(per ``feedback_visual_qc_beats_roi_count.md``); ROI counts in the pivot print
and summary.csv are supporting evidence, not the target.

Each non-vanilla column = vanilla + exactly one swept parameter. Vanilla
column is cellpose defaults (``normalize=True``, ``cellprob_threshold=0.0``,
``flow_threshold=0.4``).

Outputs go to ``<tiff parent>/sweep_in_vivo_<YYYYMMDD_HHMMSS>/`` — never into
the repo.

Usage:
    source .cellpose-venv/bin/activate          # Linux/Mac
    # .cellpose-venv\Scripts\activate            # Windows
    python alignment/sweep_in_vivo_2d.py                               # use local_config.INVIVO_PATH
    python alignment/sweep_in_vivo_2d.py --tiff <path>                 # override input
    python alignment/sweep_in_vivo_2d.py --z-planes 20,60,100,140,180  # override Z picker
    python alignment/sweep_in_vivo_2d.py --dry-run                     # print plan only
"""
from __future__ import annotations

import argparse
import csv
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
# Config (all fixed per plan ~/.claude/plans/optimized-toasting-kahan.md)
# ---------------------------------------------------------------------------

# Vanilla reference column (leftmost on every page). Each non-vanilla column
# starts from vanilla and applies exactly one swept parameter.
VANILLA_NORMALIZE = True  # bool — used for the vanilla column itself

VANILLA_NORMALIZE_DICT = {  # dict form equivalent to the bool, matches cellpose defaults
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

# (param_name, values, kind)
SWEEPS: list[tuple[str, list, str]] = [
    ("sharpen_radius",      [0, 15, 30, 60, 120],                     "normalize"),
    ("tile_norm_blocksize", [0, 25, 50, 100],                         "normalize"),
    ("percentile",          [[1, 99], [0.1, 99.9], [5, 99]],          "normalize"),
    ("cellprob_threshold",  [-3.0, -1.0, 0.0, 1.0],                   "eval"),
    ("flow_threshold",      [0.4, 0.6, 0.8],                          "eval"),
]

CONTOUR_COLOR      = "#00E5E5"
MAX_COLS_PER_PAGE  = 5          # vanilla + <=4 swept values per page
N_Z_PLANES         = 5
CELL_SIZE_INCHES   = 2.5
DPI                = 150
BATCH_SIZE         = 8


# ---------------------------------------------------------------------------
# _Tee (copied from alignment/batch_cellpose.py to avoid cross-file imports;
# the alignment/ dir has no __init__.py)
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
# Z-plane picker
# ---------------------------------------------------------------------------

def pick_z_planes(stack: np.ndarray, n: int = N_Z_PLANES) -> list[int]:
    """Hybrid signal-based picker: 1 bottom-Q, 3 middle, 1 top-Q.

    Ranks planes by above-background signal ``(plane - plane.min()).sum()``,
    splits into quartiles, and samples deterministically from each bucket.
    """
    n_planes = stack.shape[0]
    if n_planes <= n:
        return list(range(n_planes))

    plane_min = stack.reshape(n_planes, -1).min(axis=1, keepdims=False)
    sig = stack.reshape(n_planes, -1).sum(axis=1) - plane_min * stack.shape[1] * stack.shape[2]
    rank = np.argsort(sig)  # ascending by signal

    q1_end   = max(1, n_planes // 4)
    q4_start = min(n_planes - 1, 3 * n_planes // 4)
    bottom_q = rank[:q1_end]
    middle_q = rank[q1_end:q4_start]
    top_q    = rank[q4_start:]

    picks: list[int] = []
    if len(bottom_q):
        picks.append(int(bottom_q[len(bottom_q) // 2]))
    if len(middle_q):
        k = min(3, len(middle_q))
        idxs = np.linspace(0, len(middle_q) - 1, k).astype(int)
        picks.extend(int(middle_q[i]) for i in idxs)
    if len(top_q):
        picks.append(int(top_q[len(top_q) // 2]))

    # Deduplicate, backfill from overall rank if we're short.
    seen = set(picks)
    for z in rank:
        if len(picks) >= n:
            break
        z_int = int(z)
        if z_int not in seen:
            picks.append(z_int)
            seen.add(z_int)

    return sorted(picks[:n])


# ---------------------------------------------------------------------------
# Eval kwargs builders
# ---------------------------------------------------------------------------

def build_vanilla_kwargs() -> dict:
    return {
        "do_3D":      False,
        "normalize":  VANILLA_NORMALIZE,  # bool
        **VANILLA_EVAL,
        "batch_size": BATCH_SIZE,
    }


def build_sweep_kwargs(param_name: str, value, kind: str) -> dict:
    if kind == "normalize":
        norm = deepcopy(VANILLA_NORMALIZE_DICT)  # dict starting from cellpose defaults
        norm[param_name] = value
        ev = VANILLA_EVAL
    elif kind == "eval":
        norm = VANILLA_NORMALIZE  # bool, same as vanilla column
        ev = deepcopy(VANILLA_EVAL)
        ev[param_name] = value
    else:
        raise ValueError(f"unknown sweep kind: {kind!r}")
    return {
        "do_3D":      False,
        "normalize":  norm,
        **ev,
        "batch_size": BATCH_SIZE,
    }


def format_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return str(list(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def paginate(values: list, max_swept: int = MAX_COLS_PER_PAGE - 1) -> list[list]:
    """Split sweep values across pages so vanilla + values fit in MAX_COLS_PER_PAGE."""
    n_pages = max(1, math.ceil(len(values) / max_swept))
    return [values[i * max_swept : (i + 1) * max_swept] for i in range(n_pages)]


# ---------------------------------------------------------------------------
# Eval one plane
# ---------------------------------------------------------------------------

def eval_one(model, plane: np.ndarray, kwargs: dict) -> tuple[np.ndarray, float]:
    t0 = time.time()
    masks, _flows, _styles = model.eval(plane, **kwargs)
    return masks, time.time() - t0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def draw_contours(ax, masks: np.ndarray) -> None:
    """Draw per-label outlines in one ax.contour call (matplotlib-only, no skimage)."""
    m = int(masks.max()) if masks.size else 0
    if m == 0:
        return
    # Half-integer levels capture every label boundary (background-to-cell and
    # cell-to-cell) in a single pass.
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
# Pivot print
# ---------------------------------------------------------------------------

def print_pivots(pivot_data: dict, z_indices: list[int]) -> None:
    print()
    print("=" * 72)
    print("ROI COUNT PIVOTS (rows = param value, cols = Z plane)")
    print("=" * 72)
    val_w = 18
    z_w   = 8
    for param_name, by_value in pivot_data.items():
        print(f"\n{param_name}:")
        header_cells = [f"{'value':<{val_w}}"] + [f"{'z=' + str(z):>{z_w}}" for z in z_indices]
        print("  " + "  ".join(header_cells))
        for val_str, by_z in by_value.items():
            data_cells = [f"{val_str:<{val_w}}"]
            for z in z_indices:
                cell = by_z.get(z)
                cell_str = str(cell) if cell is not None else "-"
                data_cells.append(f"{cell_str:>{z_w}}")
            print("  " + "  ".join(data_cells))


# ---------------------------------------------------------------------------
# Plan building (shared by dry-run and real run)
# ---------------------------------------------------------------------------

def build_plan() -> list[dict]:
    plan_rows: list[dict] = []
    for param_name, values, kind in SWEEPS:
        pages = paginate(values)
        multi = len(pages) > 1
        for page_idx, page_values in enumerate(pages, start=1):
            suffix = f"_p{page_idx}" if multi else ""
            plan_rows.append({
                "param":       param_name,
                "kind":        kind,
                "page":        page_idx,
                "total_pages": len(pages),
                "values":      page_values,
                "filename":    f"{param_name}{suffix}.png",
            })
    return plan_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiff", default=None,
        help="Override input TIFF. Default: local_config.INVIVO_PATH.",
    )
    parser.add_argument(
        "--z-planes", default=None,
        help="Comma-separated Z indices (e.g. '20,60,100,140,180'). "
             "Overrides the hybrid picker.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the full sweep plan (Z planes + per-page cols + filenames) "
             "without loading the model or writing anything.",
    )
    args = parser.parse_args(argv)

    # Resolve TIFF path.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if args.tiff:
        tiff_path = Path(args.tiff)
    else:
        import local_config
        if not getattr(local_config, "INVIVO_PATH", None):
            sys.exit("local_config.INVIVO_PATH is empty — pass --tiff instead.")
        tiff_path = Path(local_config.INVIVO_PATH)

    if not tiff_path.exists():
        sys.exit(f"TIFF not found: {tiff_path}")

    # Load stack.
    print(f"Loading TIFF: {tiff_path}")
    t0 = time.time()
    stack = tifffile.imread(str(tiff_path))
    print(f"  shape: {stack.shape}, dtype: {stack.dtype}, load: {time.time()-t0:.1f}s")
    if stack.ndim != 3:
        sys.exit(f"Expected 3D (Z, Y, X) stack, got ndim={stack.ndim}")

    # Pick Z planes.
    if args.z_planes:
        z_indices = [int(x) for x in args.z_planes.split(",")]
        for z in z_indices:
            if not (0 <= z < stack.shape[0]):
                sys.exit(f"Z plane {z} out of range [0, {stack.shape[0]}).")
    else:
        z_indices = pick_z_planes(stack)
    print(f"Z planes: {z_indices}")

    # Resolve output dir.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = tiff_path.parent / f"sweep_in_vivo_{stamp}"

    plan_rows = build_plan()
    n_non_vanilla = sum(len(r["values"]) for r in plan_rows)
    n_vanilla     = len(z_indices)
    n_total_evals = n_vanilla + n_non_vanilla * len(z_indices)

    if args.dry_run:
        print("\nDRY RUN — sweep plan")
        print(f"  Output dir (would create): {out_dir}")
        print(f"  Z planes: {z_indices}")
        print(f"  Vanilla evaluations (1 per Z, cached): {n_vanilla}")
        print(f"  Non-vanilla evaluations: "
              f"{n_non_vanilla} param-values x {len(z_indices)} Z = "
              f"{n_non_vanilla * len(z_indices)}")
        print(f"  Total model.eval calls: {n_total_evals}")
        print()
        for r in plan_rows:
            cols = ["vanilla"] + [format_value(v) for v in r["values"]]
            print(f"  [{r['filename']}] page {r['page']}/{r['total_pages']} "
                  f"({r['kind']}): cols = {cols}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output dir: {out_dir}")

    # Tee stdout/stderr into sweep.log.
    log_path = out_dir / "sweep.log"
    real_stdout, real_stderr = sys.stdout, sys.stderr
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(real_stdout, log_f)
    sys.stderr = _Tee(real_stderr, log_f)

    exit_code = 0
    try:
        # Write config.json upfront so it exists even if the run dies mid-sweep.
        config = {
            "tiff_path":              str(tiff_path),
            "z_planes":               list(z_indices),
            "vanilla_normalize":      VANILLA_NORMALIZE,       # bool
            "vanilla_normalize_dict": VANILLA_NORMALIZE_DICT,  # dict form used for sweeping normalize-family params
            "vanilla_eval":           VANILLA_EVAL,
            "sweeps": [
                {"param": p, "values": v, "kind": k} for p, v, k in SWEEPS
            ],
            "contour_color":      CONTOUR_COLOR,
            "max_cols_per_page":  MAX_COLS_PER_PAGE,
            "batch_size":         BATCH_SIZE,
            "n_total_evals":      n_total_evals,
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"Wrote config.json")

        # Load model once.
        print("\nLoading cpsam model...")
        t0 = time.time()
        from cellpose.models import CellposeModel
        model = CellposeModel(gpu=True, pretrained_model="cpsam")
        print(f"  loaded in {time.time() - t0:.1f}s (device={model.device})")

        # Vanilla cache: z -> masks. The same vanilla config appears as col 0 on
        # every page; compute once per Z and reuse.
        vanilla_masks: dict[int, np.ndarray] = {}
        summary_rows: list[dict] = []
        # pivot_data[param][value_str][z] = n_rois
        pivot_data: dict[str, dict] = {}

        tiny_thresh = 2 * VANILLA_EVAL["min_size"]

        def record_row(param_name, value_str, z, page, masks, runtime_s):
            n = int(masks.max()) if masks.size else 0
            if n > 0:
                sizes = np.bincount(masks.ravel())[1:]
                median_size = int(np.median(sizes))
                n_tiny = int((sizes < tiny_thresh).sum())
            else:
                median_size = 0
                n_tiny = 0
            summary_rows.append({
                "param":                     param_name,
                "value":                     value_str,
                "z_plane":                   z,
                "page":                      page,
                "n_rois":                    n,
                "median_size_px":            median_size,
                "n_tiny_below_2x_min_size":  n_tiny,
                "runtime_s":                 round(runtime_s, 2),
            })
            pivot_data.setdefault(param_name, {}).setdefault(value_str, {})[z] = n

        for param_name, values, kind in SWEEPS:
            pages = paginate(values)
            multi = len(pages) > 1

            for page_idx, page_values in enumerate(pages, start=1):
                suffix = f"_p{page_idx}" if multi else ""
                out_path = out_dir / f"{param_name}{suffix}.png"
                suptitle = (
                    f"{param_name} (page {page_idx}/{len(pages)})"
                    if multi else param_name
                )
                col_headers = ["vanilla"] + [format_value(v) for v in page_values]
                print(f"\n--- {out_path.name}  [{param_name}, page {page_idx}/{len(pages)}] ---")

                # col_masks[row][col] = masks array for (z_indices[row], col)
                col_masks: list[list[np.ndarray]] = [
                    [np.zeros((1, 1), dtype=np.int32)] * len(col_headers)
                    for _ in z_indices
                ]

                for r, z in enumerate(z_indices):
                    plane = stack[z]

                    # Col 0: vanilla (cached across pages).
                    if z not in vanilla_masks:
                        vkwargs = build_vanilla_kwargs()
                        print(f"  Z={z:>4}  vanilla ... ", end="", flush=True)
                        v_masks, v_dt = eval_one(model, plane, vkwargs)
                        vanilla_masks[z] = v_masks
                        print(f"n={int(v_masks.max())} ({v_dt:.1f}s)")
                    else:
                        v_masks = vanilla_masks[z]
                        v_dt = 0.0
                        print(f"  Z={z:>4}  vanilla (cached) n={int(v_masks.max())}")
                    col_masks[r][0] = v_masks
                    record_row(param_name, "vanilla", z, page_idx, v_masks, v_dt)

                    # Cols 1..: swept values.
                    for c, val in enumerate(page_values, start=1):
                        kwargs = build_sweep_kwargs(param_name, val, kind)
                        val_str = format_value(val)
                        print(f"  Z={z:>4}  {param_name}={val_str} ... ", end="", flush=True)
                        masks, dt = eval_one(model, plane, kwargs)
                        n = int(masks.max())
                        print(f"n={n} ({dt:.1f}s)")
                        col_masks[r][c] = masks
                        record_row(param_name, val_str, z, page_idx, masks, dt)

                render_page(out_path, stack, z_indices, col_masks, col_headers, suptitle)
                print(f"  wrote {out_path.name}")

        # summary.csv
        csv_path = out_dir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "param", "value", "z_plane", "page", "n_rois",
                "median_size_px", "n_tiny_below_2x_min_size", "runtime_s",
            ])
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"\nWrote summary.csv ({len(summary_rows)} rows): {csv_path.name}")

        print_pivots(pivot_data, z_indices)
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
