#!/usr/bin/env python3
"""Audit every stored slice fit: what the edge holds, where it puts the slice, what the graph viewer draws.

For every 1-plane node (a section) with a path to a 2P volume:

  route    the section's direct edges, and the BFS route to the viewer frame (the route the
           graph viewer cell draws through)
  stages   every stage of the stored target -> section edge (the Mode C save), in order
  clicks   every point stage still stored with its clicks (N, V, R, P; Mode C stores T as
           MatrixFixed, without clicks): per pair, the block cell in the frame the stage was
           fitted in (its z is the plane castalign recorded for the click), the same cell in raw
           block voxels, the section cell, and the depth the pair asks for (block z - section z)
  sheet    the section plane carried into the target frame and into the viewer frame: z range,
           tilt, fold (depth off the best flat plane) and nonlinear part (off the best affine),
           with a coarse map of the fold
  inverse  target -> section -> target and section -> target -> section round trips
  domain   each nonlinear stage on the section -> viewer route that runs castalign's fast direction
           (invert False) asserts every input point lies inside its triangulation (clicks + their
           hull scaled x1000 for V, x100 for N): base.py "Points outside domain". Fed the 8 box
           corners the graph viewer transforms first (origin_and_maxpos, 2-plane section) and the
           section sheet; reports the click spread (one tiny singular value = clicks on a plane for
           V, on a line for N) and the points left outside
  compose  hop-by-hop vs g.get_transform(section, viewer frame)
  render   a dot drawn on the section at each section click and at its centre, resampled with
           castalign's transform_image as the graph viewer cell does (2-plane path); its
           centroid vs the analytic position, and vs where the matching block click lands

Read-only: loads the graph, writes nothing.

--click-test NODE opens castalign's text menu + GUI the way the Mode C cell does (section padded
to 2*200+1 planes and repeated in every plane = fixed, target volume = movable, starting from the
stored fit) and, after q, prints the planes castalign recorded for every click made in that
session. Nothing is saved to the graph.

Run in .castalign-venv:
    python alignment/diagnose_slice_fits.py
    python alignment/diagnose_slice_fits.py --slices slice2_confocal_tdtom_part1 slice2_confocal_tdtom_part2
    python alignment/diagnose_slice_fits.py --viewer-space block_stack_red --no-render
    python alignment/diagnose_slice_fits.py --click-test slice2_confocal_tdtom_part1
"""

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import scipy.spatial
import castalign as ca
import castalign.utils as ca_utils

from subslice_graph_builder import GRAPH_PATH, _derive_graph_path

PAD_Z = 200                 # the Mode C cell's PAD_Z
TARGET_PREFERENCE = ("block_stack_red", "invivo_red", "block_stack_green", "invivo_green")


def fmt(a, p=1):
    return np.array2string(np.asarray(a, dtype=float), precision=p, suppress_small=True,
                           floatmode="fixed", max_line_width=200)


def header(text):
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100)


# ---------------------------------------------------------------- transform structure

def stage_list(t):
    """[(stage, prior)] in application order. prior = the composed transform applied before stage, None for the first.

    castalign composes `a + b` into an object whose .b is b and whose pretransform() returns a.
    """
    out = []
    x = t
    while hasattr(x, "b"):
        prior = x.pretransform()
        out.append((x.b, prior))
        x = prior
    out.append((x, None))
    return out[::-1]


def is_point(st):
    return isinstance(st, ca.base.PointTransform)


def is_nonlinear(st):
    return is_point(st) and not hasattr(st, "matrix")


def describe(st):
    name = type(st).__name__
    if is_point(st):
        n = len(np.atleast_2d(np.asarray(st.points_start)))
        kind = "nonlinear" if is_nonlinear(st) else "affine"
        return f"{name} ({kind})  {n} pairs  params {dict(st.params)}"
    if hasattr(st, "matrix"):
        M = np.asarray(st.matrix, dtype=float)
        tr = -np.asarray(st.shift, dtype=float)          # castalign maps p -> p @ M - shift
        if np.allclose(M, np.eye(3)):
            return f"{name}  translate z,y,x {fmt(tr)}"
        z_img = M[0]                                      # where a unit step along source z goes
        tilt = np.degrees(np.arccos(np.clip(abs(z_img[0]) / np.linalg.norm(z_img), 0, 1)))
        return (f"{name}  det {np.linalg.det(M):+.4f}  row norms {fmt(np.linalg.norm(M, axis=1), 3)}  "
                f"z-axis tilt {tilt:.2f} deg  offset z,y,x {fmt(tr)}")
    return f"{name}  {getattr(st, 'params', '')}"


def exact_rigid_stages(t):
    """Copy of the Mode C cell's _exact_rigid_stages: every TranslateRotate (T) stage as MatrixFixed."""
    if hasattr(t, "b"):
        return exact_rigid_stages(t.pretransform()) + exact_rigid_stages(t.b)
    if type(t) is ca.TranslateRotate:
        M, s = np.asarray(t.matrix, dtype=float), np.asarray(t.shift, dtype=float)
        return ca.MatrixFixed(**{f"a{i + 1}{j + 1}": float(M[i, j]) for i in range(3) for j in range(3)},
                              z=-s[0], y=-s[1], x=-s[2])
    return t


# ---------------------------------------------------------------- geometry

def sheet_points(shape, n=40):
    yy, xx = np.meshgrid(np.linspace(0, shape[1] - 1, n), np.linspace(0, shape[2] - 1, n), indexing="ij")
    return np.c_[np.zeros(yy.size), yy.ravel(), xx.ravel()]


def affine_residual(P, Q):
    A = np.c_[P, np.ones(len(P))]
    return Q - A @ np.linalg.lstsq(A, Q, rcond=None)[0]


def plane_fit(Q):
    """Fit z = a*y + b*x + c to mapped sheet points. Returns (tilt deg, z residuals)."""
    A = np.c_[Q[:, 1], Q[:, 2], np.ones(len(Q))]
    coef = np.linalg.lstsq(A, Q[:, 0], rcond=None)[0]
    return np.degrees(np.arctan(np.hypot(coef[0], coef[1]))), Q[:, 0] - A @ coef


def coarse_map(P, vals, mask, shape, k=6):
    iy = np.minimum((P[:, 1] / shape[1] * k).astype(int), k - 1)
    ix = np.minimum((P[:, 2] / shape[2] * k).astype(int), k - 1)
    rows = []
    for r in range(k):
        cells = []
        for c in range(k):
            m = mask & (iy == r) & (ix == c)
            cells.append(f"{np.nanmean(vals[m]):+7.1f}" if m.any() else "      .")
        rows.append(" ".join(cells))
    return rows


def sheet_report(label, t, shape, ref_shape=None):
    P = sheet_points(shape)
    try:
        Q = np.asarray(t.transform(P), dtype=float)
    except Exception as e:
        print(f"    {label}: transform failed: {type(e).__name__}: {e}")
        return
    inside = (np.ones(len(Q), bool) if ref_shape is None
              else np.all((Q >= 0) & (Q < np.asarray(ref_shape)), axis=1))
    where = "" if ref_shape is None else f" of {tuple(int(v) for v in ref_shape)}"
    print(f"    {label}: {inside.sum()}/{len(Q)} sheet points inside the volume{where}")
    if inside.sum() < 4:
        return
    tilt, zres = plane_fit(Q[inside])
    R = affine_residual(P[inside], Q[inside])
    print(f"      z {Q[inside, 0].min():.1f}..{Q[inside, 0].max():.1f}   y {Q[inside, 1].min():.0f}..{Q[inside, 1].max():.0f}"
          f"   x {Q[inside, 2].min():.0f}..{Q[inside, 2].max():.0f}   tilt vs z planes {tilt:.1f} deg")
    print(f"      fold (depth off best flat plane): max {np.abs(zres).max():.2f}  rms {np.sqrt(np.mean(zres ** 2)):.2f} vox")
    print(f"      nonlinear part (off best affine): depth max {np.abs(R[:, 0]).max():.2f}"
          f"   in-plane max {np.linalg.norm(R[:, 1:], axis=1).max():.2f} vox")
    full = np.full(len(Q), np.nan)
    full[inside] = zres
    print("      fold map, mean depth off the flat plane per cell (rows = section y, cols = section x, . = outside):")
    for row in coarse_map(P, full, inside, shape):
        print("        " + row)


# ---------------------------------------------------------------- per-section checks

def click_report(fwd, target):
    """Print every point stage's clicks. Returns [(block_raw (n,3), stage name)]."""
    found = []
    for i, (st, prior) in enumerate(stage_list(fwd)):
        if not is_point(st):
            continue
        start = np.atleast_2d(np.asarray(st.points_start, dtype=float))
        end = np.atleast_2d(np.asarray(st.points_end, dtype=float))
        raw = np.asarray(prior.inverse_transform(start), dtype=float) if prior is not None else start
        name = type(st).__name__
        print(f"    stage {i}: {name}, {len(start)} pairs")
        print(f"      {'pair':>4}  {'block click, fit frame z,y,x':>30}  {'raw ' + target + ' voxel z,y,x':>34}  "
              f"{'section click z,y,x':>28}  {'depth asked':>11}")
        for k in range(len(start)):
            print(f"      {k:>4}  {fmt(start[k]):>30}  {fmt(raw[k]):>34}  {fmt(end[k]):>28}  {start[k, 0] - end[k, 0]:>+11.2f}")
        print(f"      block-click z spread {np.ptp(start[:, 0]):.2f}   section-click z spread {np.ptp(end[:, 0]):.2f}"
              f"   raw {target} z {raw[:, 0].min():.1f}..{raw[:, 0].max():.1f}")
        if is_nonlinear(st) and np.ptp(start[:, 0] - end[:, 0]) < 0.5:
            print("      -> every pair asks for the same depth: this stage warps in-plane only, no fold in depth")
        try:
            err = np.abs(np.asarray(st.transform(start)) - end).max()
            print(f"      stage maps its block clicks onto its section clicks: max error {err:.2e}")
        except Exception as e:
            print(f"      stage transform failed: {type(e).__name__}: {e}")
        found.append((raw, name))
    if not found:
        print("    no point stage with clicks in this edge")
    return found


def roundtrip_report(fwd, inv, target_shape, sec_shape):
    rng = np.random.default_rng(0)
    try:
        Pb = rng.uniform(0, 1, (300, 3)) * np.asarray(target_shape, dtype=float)
        e1 = np.abs(np.asarray(inv.transform(fwd.transform(Pb))) - Pb).max()
        print(f"    target -> section -> target  max error {e1:.2e} vox")
    except Exception as e:
        print(f"    target -> section -> target  failed: {type(e).__name__}: {e}")
    try:
        Ps = sheet_points(sec_shape, 15)
        Ps[:, 0] = rng.uniform(-10, 10, len(Ps))
        e2 = np.abs(np.asarray(fwd.transform(inv.transform(Ps))) - Ps).max()
        print(f"    section -> target -> section max error {e2:.2e} vox")
    except Exception as e:
        print(f"    section -> target -> section failed: {type(e).__name__}: {e}")


def flat_stages(t):
    """Every leaf stage of t in application order (nested `a + b` compositions expanded)."""
    if hasattr(t, "b"):
        return flat_stages(t.pretransform()) + flat_stages(t.b)
    return [t]


def domain_report(label, t, sec_shape):
    """Replay castalign's fast-direction containment test for every nonlinear stage of t."""
    H, W = sec_shape[1], sec_shape[2]
    corners = np.array([[z, y, x] for z in (0, 2) for y in (0, H) for x in (0, W)], dtype=float)
    P = np.concatenate([corners, sheet_points(sec_shape)])
    n_c = len(corners)
    Q = P
    found = False
    for i, st in enumerate(flat_stages(t)):
        if is_nonlinear(st) and hasattr(st, "all_points_start"):
            found = True
            flat = hasattr(st, "B")
            inv = bool(st.params.get("invert"))
            clicks = np.atleast_2d(np.asarray(st.points_start, dtype=float))
            C = clicks @ st.B if flat else clicks
            sv = np.linalg.svd(C - C.mean(axis=0), compute_uv=False)
            print(f"    {label} stage {i}: {type(st).__name__}, {len(clicks)} pairs, invert {inv}")
            print(f"      input-side clicks z {clicks[:, 0].min():.1f}..{clicks[:, 0].max():.1f}"
                  f"  y {clicks[:, 1].min():.0f}..{clicks[:, 1].max():.0f}  x {clicks[:, 2].min():.0f}..{clicks[:, 2].max():.0f}")
            print(f"      click spread, singular values{' in the projected plane' if flat else ' (z,y,x)'}: {fmt(sv, 2)}"
                  + (f"   normal z,y,x {fmt(st.normal, 3)}" if flat else ""))
            if inv:
                print("      invert True: slow path, this direction has no 'Points outside domain' assert")
            else:
                tri = np.asarray(st.all_points_start, dtype=float)
                try:
                    if flat:
                        out = scipy.spatial.Delaunay(tri @ st.B).find_simplex(Q @ st.B) < 0
                    else:
                        rns = np.random.RandomState(0).randn(*tri.shape) * .0001
                        out = scipy.spatial.Delaunay(tri + rns).find_simplex(Q) < 0
                except Exception as e:
                    print(f"      containment test failed: {type(e).__name__}: {e}")
                    return
                print(f"      outside its domain: {int(out[:n_c].sum())}/{n_c} box corners, "
                      f"{int(out[n_c:].sum())}/{len(P) - n_c} sheet points")
                for k in np.flatnonzero(out[:n_c]):
                    print(f"        corner {fmt(P[k], 0)} enters this stage at {fmt(Q[k])}")
        try:
            Q = np.asarray(st.transform(Q), dtype=float)
        except Exception as e:
            print(f"    {label} stage {i} {type(st).__name__}: transform failed: {type(e).__name__}: {e}")
            return
    if not found:
        print(f"    {label}: no nonlinear stage")


def compose_report(g, s, viewer, sec_shape):
    chain = [s] + list(g.get_chain(s, viewer))
    P = sheet_points(sec_shape, 15)
    q = P
    for a, b in zip(chain, chain[1:]):
        q = np.asarray(g.edges[a][b].transform(q))
    t = g.get_transform(s, viewer)
    err = np.abs(q - np.asarray(t.transform(P))).max()
    print(f"    route {' -> '.join(chain)}")
    print(f"    hop-by-hop vs get_transform: max difference {err:.2e} vox")
    return t


def render_dot(t, sec_shape, y, x, radius=3):
    """Dot at (y, x) on the section, 2 planes as the viewer's no-slab path; returns (rendered centroid, analytic)."""
    H, W = sec_shape[1], sec_shape[2]
    img = np.zeros((2, H, W), dtype=np.float32)
    y0, y1 = max(int(y) - radius, 0), min(int(y) + radius + 1, H)
    x0, x1 = max(int(x) - radius, 0), min(int(x) + radius + 1, W)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
    img[:, y0:y1, x0:x1] = disc.astype(np.float32)
    cy, cx = (yy[disc].mean(), xx[disc].mean()) if disc.any() else (y, x)
    expected = np.asarray(t.transform(np.array([[0.5, cy, cx]])), dtype=float)[0]
    lo = np.floor(expected - np.array([8, 14, 14]))
    hi = lo + np.array([16, 28, 28])
    out = t.transform_image(img, output_size=[(float(lo[i]), float(hi[i])) for i in range(3)],
                            force_size=True, labels=False)
    o = np.nan_to_num(np.asarray(out, dtype=float))
    origin = np.asarray(getattr(out, "origin", lo), dtype=float)
    if o.sum() <= 0:
        return None, expected
    idx = np.indices(o.shape).reshape(3, -1)
    return (idx @ o.ravel()) / o.sum() + origin, expected


def render_report(g, t_view, s, target, viewer, sec_shape, clicks, fwd):
    t_target_view = g.get_transform(target, viewer) if g.has_transform(target, viewer) else None
    jobs = [("centre", (sec_shape[1] - 1) / 2, (sec_shape[2] - 1) / 2, None)]
    for raw, name in clicks:
        try:
            sec = np.asarray(fwd.transform(raw), dtype=float)       # block click -> section node coords
        except Exception as e:
            print(f"    {name}: could not map block clicks to the section: {e}")
            continue
        for k in range(len(raw)):
            jobs.append((f"{name} pair {k}", sec[k, 1], sec[k, 2], (raw[k], sec[k, 0])))
    print(f"    {'dot':<26} {'rendered centroid z,y,x':>26} {'analytic z,y,x':>26} {'render err':>10} {'block click there':>26} {'dist':>6}")
    for label, y, x, extra in jobs:
        t0 = time.perf_counter()
        try:
            got, exp = render_dot(t_view, sec_shape, y, x)
        except Exception as e:
            print(f"    {label:<26} render failed: {type(e).__name__}: {e}")
            continue
        err = "  not drawn" if got is None else f"{np.linalg.norm(got - exp):10.2f}"
        blk, dist = "", ""
        if extra is not None and t_target_view is not None:
            raw_pt, sec_z = extra
            b = np.asarray(t_target_view.transform(raw_pt[None]), dtype=float)[0]
            blk, dist = fmt(b), f"{np.linalg.norm(b - exp):6.2f}"
            if abs(sec_z) > 0.5:
                dist += f"  (click sits {sec_z:+.1f} off the section plane)"
        print(f"    {label:<26} {fmt(got) if got is not None else '-':>26} {fmt(exp):>26} {err:>10} {blk:>26} {dist}"
              f"   {time.perf_counter() - t0:.1f}s")
    print("    render err = rendered dot vs its analytic position (z centroid of a 2-plane dot = plane 0.5);"
          " dist = section dot vs the matching block click, both in the viewer frame")


def pick_target(g, s, volumes):
    nbrs = list(g.edges[s].keys())
    vol_nbrs = [n for n in nbrs if n in volumes]
    for name in TARGET_PREFERENCE:
        if name in vol_nbrs:
            return name, nbrs, vol_nbrs
    return (vol_nbrs[0] if vol_nbrs else None), nbrs, vol_nbrs


# ---------------------------------------------------------------- click test

def click_test(g, s, volumes):
    import castalign.gui as ca_gui
    ca_utils.image_is_label = lambda img: False   # as the notebook's startup cell
    target, _, _ = pick_target(g, s, volumes)
    if target is None:
        target = next((n for n in TARGET_PREFERENCE if n in volumes), None)
    if target is None:
        raise SystemExit("no 2P volume in the graph")
    sec = np.asarray(g.get_image(s))
    fixed = np.repeat(sec[0][None], 2 * PAD_Z + 1, axis=0)
    if g.has_transform(target, s):
        start = exact_rigid_stages(g.get_transform(target, s)) + ca.TranslateFixed(z=PAD_Z)
        n_before = len(stage_list(start))
        print(f"Starting from the stored fit {target} -> {s}")
    else:
        start = ca.Identity()
        n_before = 0
        print(f"No fit for {s}: starting from Identity")
    print(f"Movable {target} {g.get_image(target).shape}   fixed: {s} repeated in {fixed.shape[0]} planes, its plane = {PAD_Z}")
    print("""
In the text menu press N. In the GUI:
  1. 'Add new point', click a section cell; scroll the z slider ~15 planes, note the number, click a block cell.
  2. 'Add new point', click a section cell; switch napari to 3D, click a block cell you see at another depth.
  3. 'Add new point', one more pair anywhere (N needs 3).
  4. 'Perform transform', close the window, then q in the text menu.
Nothing is saved.
""")
    t = ca_gui.align_interactive(nodes_movable=g.get_image(target), nodes_fixed=fixed, transform=start)
    del fixed
    header(f"CLICK TEST RESULT: {s}")
    new = stage_list(t)[n_before:]
    pts = [(st, prior) for st, prior in new if is_point(st)]
    if not pts:
        print("No point stage was added in this session (was 'Perform transform' pressed before closing?)")
        return
    for st, prior in pts:
        start_pts = np.atleast_2d(np.asarray(st.points_start, dtype=float))
        end_pts = np.atleast_2d(np.asarray(st.points_end, dtype=float))
        print(f"{type(st).__name__}: {len(start_pts)} pairs")
        for k in range(len(start_pts)):
            print(f"  pair {k}: block click z {start_pts[k, 0]:8.2f}   section click z {end_pts[k, 0]:8.2f}"
                  f"   depth asked {start_pts[k, 0] - end_pts[k, 0]:+7.2f}")
    print("\nA block click made after scrolling to plane P should read z = P (to within the slider grid's offset).")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graph", default=None, help="graph .db (default: the builder's GRAPH_PATH rule)")
    ap.add_argument("--slices", nargs="+", default=None, help="section nodes to audit (default: every 1-plane node)")
    ap.add_argument("--viewer-space", default="invivo_red", help="frame the graph viewer draws in (default: invivo_red)")
    ap.add_argument("--no-render", action="store_true", help="skip the dot render check")
    ap.add_argument("--click-test", metavar="NODE", default=None,
                    help="open castalign's GUI on NODE as Mode C does and report the recorded click planes; saves nothing")
    args = ap.parse_args()

    graph_path = Path(args.graph) if args.graph else (Path(GRAPH_PATH) if GRAPH_PATH else _derive_graph_path(None, None))
    if not graph_path.exists():
        raise SystemExit(f"Graph not found: {graph_path}")
    g = ca.Graph.load(str(graph_path))
    print(f"Graph: {graph_path}   {len(g.nodes)} nodes   castalign {getattr(ca, '__version__', '?')}")

    shapes = {n: tuple(int(v) for v in np.asarray(g.get_image(n)).shape) for n in g.nodes}
    volumes = [n for n, sh in shapes.items() if len(sh) == 3 and sh[0] > 1]
    sections = [n for n, sh in shapes.items() if len(sh) == 3 and sh[0] == 1]

    if args.click_test:
        if args.click_test not in g.nodes:
            raise SystemExit(f"{args.click_test} not in the graph")
        click_test(g, args.click_test, volumes)
        return

    header("GRAPH")
    for n in volumes:
        print(f"  volume   {n:44s} {shapes[n]}")
    for n in sections:
        print(f"  section  {n:44s} {shapes[n]}")
    print("\n  edges (one line per pair; stage count of the direction listed):")
    seen = set()
    for a in g.nodes:
        for b, e in g.edges[a].items():
            if (b, a) in seen:
                continue
            seen.add((a, b))
            st = stage_list(e)
            nl = [type(x).__name__ for x, _ in st if is_nonlinear(x)]
            print(f"    {a} -> {b}: {len(st)} stages" + (f"   NONLINEAR {nl}" if nl else ""))

    viewer = args.viewer_space
    if viewer not in g.nodes:
        raise SystemExit(f"--viewer-space {viewer} not in the graph")
    view_shape = shapes[viewer]

    todo = args.slices or sections
    for s in todo:
        header(f"SECTION {s}")
        if s not in g.nodes:
            print("  not in the graph")
            continue
        sec_shape = shapes[s]
        target, nbrs, vol_nbrs = pick_target(g, s, volumes)
        print(f"  shape {sec_shape}   direct edges to: {nbrs or 'none'}")
        if len(vol_nbrs) > 1:
            print(f"  NOTE: direct edges to {len(vol_nbrs)} volumes {vol_nbrs}; BFS takes the shortest route, not the newest fit")
        if target is None:
            print("  no direct edge to a 2P volume; nothing to audit")
            continue
        fwd, inv = g.edges[target][s], g.edges[s][target]

        print(f"\n  [stages] stored {target} -> {s}")
        for i, (st, _) in enumerate(stage_list(fwd)):
            print(f"    {i:>2}  {describe(st)}")

        print(f"\n  [clicks]")
        clicks = click_report(fwd, target)

        print(f"\n  [sheet] the section plane (z = 0) carried out of the section")
        sheet_report(f"in {target}", inv, sec_shape, shapes[target])
        if g.has_transform(s, viewer) and viewer != target:
            sheet_report(f"in {viewer} (viewer frame)", g.get_transform(s, viewer), sec_shape, view_shape)

        print(f"\n  [inverse]")
        roundtrip_report(fwd, inv, shapes[target], sec_shape)

        print(f"\n  [domain]")
        routed = g.has_transform(s, viewer)
        domain_report(f"{s} -> {viewer if routed else target}",
                      g.get_transform(s, viewer) if routed else inv, sec_shape)

        if not routed:
            print(f"\n  no route from {s} to {viewer}; compose and render skipped")
            continue
        print(f"\n  [compose] {s} -> {viewer}")
        try:
            t_view = compose_report(g, s, viewer, sec_shape)
        except Exception as e:
            print(f"    compose failed: {type(e).__name__}: {e}; render skipped")
            continue

        if not args.no_render:
            print(f"\n  [render] into {viewer}")
            render_report(g, t_view, s, target, viewer, sec_shape, clicks, fwd)


if __name__ == "__main__":
    main()
