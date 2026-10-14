r"""
Sideways shift between neighbouring Z planes of a red 2P volume.

For every plane z, measures how far its image content sits from plane z-1 in Y and
X (pixels), by phase correlation: both planes Hann-windowed, normalized
cross-power spectrum, peak located to a fraction of a pixel by a parabola through
the peak and its two neighbours on each axis. A shift of (dy, dx) means plane z
looks like plane z-1 moved dy pixels down and dx right. ``peak`` is the height
of the correlation peak (1 = identical up to the shift; near 0 = no common
structure, so the shift on that row means nothing).

Cellpose's stitching joins an outline to the one on the plane above only if they
overlap by IoU >= stitch_threshold; a cell 10-15 px across loses half its overlap
to a shift of a few pixels, so a jump splits every cell crossing it.

Before measuring, shifts one middle plane by a known amount with np.roll and
prints what it recovers, to confirm the sign convention.

Reads the TIFF only. Writes ``<ANALYSIS_ROOT>/cellpose/plane_shift/<stem>_shift.csv``
(z, dy, dx, shift, peak, cumulative dy/dx from the first plane) and
``<stem>_shift.png`` (shift per plane; curated planes and the depth range marked).

Usage (.cellpose-venv):
    .cellpose-venv\Scripts\python.exe alignment\check_plane_shift.py
    .cellpose-venv\Scripts\python.exe alignment\check_plane_shift.py --node block_stack_red --threshold 2
    .cellpose-venv\Scripts\python.exe alignment\check_plane_shift.py --tiff <path> --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SELF_TEST_SHIFT = (3, -2)


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1])).astype(np.float32)


def spectrum(plane: np.ndarray, window: np.ndarray) -> np.ndarray:
    a = plane.astype(np.float32)
    return np.fft.fft2((a - a.mean()) * window)


def _subpixel(c_minus: float, c0: float, c_plus: float) -> float:
    denom = c_minus - 2 * c0 + c_plus
    return 0.0 if denom == 0 else 0.5 * (c_minus - c_plus) / denom


def phase_shift(f_prev: np.ndarray, f_cur: np.ndarray) -> tuple[float, float, float]:
    """(dy, dx, peak): current plane = previous plane moved by (dy, dx)."""
    cross = f_cur * np.conj(f_prev)
    cross /= np.maximum(np.abs(cross), 1e-12)
    r = np.real(np.fft.ifft2(cross))
    ny, nx = r.shape
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
    peak = float(r[iy, ix])
    dy = iy + _subpixel(r[(iy - 1) % ny, ix], r[iy, ix], r[(iy + 1) % ny, ix])
    dx = ix + _subpixel(r[iy, (ix - 1) % nx], r[iy, ix], r[iy, (ix + 1) % nx])
    if dy > ny / 2:
        dy -= ny
    if dx > nx / 2:
        dx -= nx
    return float(dy), float(dx), peak


def curated_planes(node: str) -> list[int]:
    from cellpose_curation import MANIFEST, curation_dir
    path = curation_dir(node) / MANIFEST
    if not path.exists():
        return []
    return sorted(int(z) for z in json.loads(path.read_text())["planes"])


def main(argv: list | None = None) -> int:
    from analysis_paths import TWO_P_RED_NODES, cellpose_dir, two_p_path
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node", default="invivo_red", choices=list(TWO_P_RED_NODES),
                        help="Red 2P node. Default: invivo_red.")
    parser.add_argument("--tiff", default=None, help="Volume instead of the node's local_config TIFF.")
    parser.add_argument("--threshold", type=float, default=1.5,
                        help="List planes shifted more than this many pixels from the one above. Default 1.5.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan, read nothing.")
    args = parser.parse_args(argv)

    import tifffile
    tiff = Path(args.tiff) if args.tiff else two_p_path(args.node)
    if tiff is None:
        sys.exit(f"local_config.{TWO_P_RED_NODES[args.node]} is blank; pass --tiff.")
    if not tiff.exists():
        sys.exit(f"TIFF not found: {tiff}")
    root = cellpose_dir()
    if root is None:
        sys.exit("ANALYSIS_ROOT is blank in local_config.py.")
    out_dir = root / "plane_shift"
    csv_path = out_dir / f"{tiff.stem}_shift.csv"
    png_path = out_dir / f"{tiff.stem}_shift.png"
    print(f"Volume: {tiff}\nWrites: {csv_path}\n        {png_path}")
    if args.dry_run:
        return 0

    stack = tifffile.imread(str(tiff))
    if stack.ndim != 3:
        sys.exit(f"Expected a (Z, Y, X) volume, got shape {stack.shape}")
    n_z = stack.shape[0]
    window = hann2d(stack.shape[1:])

    mid = stack[n_z // 2]
    dy, dx, peak = phase_shift(spectrum(mid, window),
                               spectrum(np.roll(mid, SELF_TEST_SHIFT, axis=(0, 1)), window))
    print(f"Self-test on plane {n_z // 2}: moved {SELF_TEST_SHIFT}, measured ({dy:.2f}, {dx:.2f}), peak {peak:.2f}")
    if abs(dy - SELF_TEST_SHIFT[0]) > 0.5 or abs(dx - SELF_TEST_SHIFT[1]) > 0.5:
        sys.exit("Self-test failed: measured shift does not match the applied one.")

    rows = []
    f_prev = spectrum(stack[0], window)
    cum_y = cum_x = 0.0
    for z in range(1, n_z):
        f_cur = spectrum(stack[z], window)
        dy, dx, peak = phase_shift(f_prev, f_cur)
        cum_y += dy
        cum_x += dx
        rows.append({"z": z, "dy": dy, "dx": dx, "shift": float(np.hypot(dy, dx)), "peak": peak,
                     "cum_dy": cum_y, "cum_dx": cum_x})
        f_prev = f_cur

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    from batch_cellpose import load_depth_range
    keep_z = load_depth_range(root, tiff)
    curated = curated_planes(args.node) if not args.tiff else []
    shifts = np.array([r["shift"] for r in rows])
    in_range = [r for r in rows if keep_z is None or keep_z[0] <= r["z"] <= keep_z[1]]
    s_in = np.array([r["shift"] for r in in_range])
    big = [r for r in in_range if r["shift"] > args.threshold]

    rng = f"planes {keep_z[0]}-{keep_z[1]}" if keep_z else "all planes (no depth range set)"
    print(f"\nShift from the plane above, {rng}: median {np.median(s_in):.2f} px, "
          f"95th percentile {np.percentile(s_in, 95):.2f} px, max {s_in.max():.2f} px")
    print(f"Drift over the whole volume: {rows[-1]['cum_dy']:.1f} px in Y, {rows[-1]['cum_dx']:.1f} px in X")
    print(f"\nPlanes shifted more than {args.threshold} px from the plane above ({len(big)}):")
    for r in big:
        mark = "  curated" if r["z"] in curated else ""
        print(f"  z={r['z']:<4} dy {r['dy']:+6.2f}  dx {r['dx']:+6.2f}  shift {r['shift']:5.2f}  "
              f"peak {r['peak']:.2f}{mark}")
    if curated:
        print("\nCurated planes, shift from the plane above / to the plane below:")
        by_z = {r["z"]: r for r in rows}
        for z in curated:
            a, b = by_z.get(z), by_z.get(z + 1)
            print(f"  z={z:<4} " + (f"{a['shift']:5.2f} px" if a else "   -    ") + "  /  "
                  + (f"{b['shift']:5.2f} px" if b else "-"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zs = np.array([r["z"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(zs, [r["dy"] for r in rows], lw=0.8, label="dy")
    axes[0].plot(zs, [r["dx"] for r in rows], lw=0.8, label="dx")
    axes[0].set_ylabel("shift from plane above (px)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(zs, shifts, lw=0.8, color="k")
    axes[1].axhline(args.threshold, color="r", lw=0.8, ls="--")
    axes[1].set_ylabel("|shift| (px)")
    axes[2].plot(zs, [r["peak"] for r in rows], lw=0.8, color="g")
    axes[2].set_ylabel("correlation peak")
    axes[2].set_xlabel("z (plane)")
    for ax in axes:
        for z in curated:
            ax.axvline(z, color="orange", lw=0.6, ls=":")
        if keep_z:
            ax.axvspan(0, keep_z[0], color="0.85", lw=0)
            ax.axvspan(keep_z[1], n_z - 1, color="0.85", lw=0)
    fig.suptitle(f"{tiff.name}: shift of each plane from the one above "
                 f"(orange dotted = curated planes, grey = outside depth range)", fontsize=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {csv_path}\n      {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
