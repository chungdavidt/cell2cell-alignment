#!/usr/bin/env python3
"""Step-by-step check of how one slice is oriented in Mode C and in Mode A.

castalign_testground.ipynb shows a BARseq slice two ways:

    Mode C            the slice is the FIXED image, drawn as stored; the target
                      (ex-vivo block or in-vivo) is warped onto it.
    Mode A            the target is the FIXED image, drawn as stored; the slice
                      is warped onto it by g.get_transform(slice, target).

This script replays each step with castalign's own calls on the real graph and
prints PASS / FAIL / INFO with numbers, so a mirrored or transposed slice can be
traced to the step that produces it:

    1  slice TIFF on disk      vs  slice node image in the graph
    2  target TIFF on disk     vs  target node image in the graph
    3  NOTEBOOK REPLAY: the startup cell's helper functions, the Mode C cell and
       the Mode A cell are executed on a COPY of the graph, with ca_gui.align_interactive
       replaced by a recorder. It captures exactly which arrays and which
       start transform each mode hands castalign, and what Mode C saves. The
       recorder answers Mode C with the fit already stored in the graph, so
       the replayed save reproduces the real edge; it answers Mode A with
       Identity, so Mode A saves nothing.
    4  the stored routes target <-> slice, hop by hop, split into components
    5  Mode A's start transform is the exact inverse of the Mode C route
    6  pressing a key in Mode A starts from that same transform
    7  what each component does to the slice's on-screen orientation
    8  image evidence: an asymmetric test pattern, and the real slice, pushed
       through the same castalign call Mode A uses, matched against the 8
       square symmetries of the original
    9  a figure built from the arrays and transforms the replay recorded:
       the Mode C view, the Mode A view (one plane, as napari's 2D view draws
       it) and the test pattern
   10  what Mode A's 2D view can show: which target planes the warped slice
       lands on, how much of it one plane holds, and the fit's tilt

Screen convention (napari's default 2D view of (z, y, x) data): y points down,
x points right. "rotate 90 counter-clockwise" is np.rot90(img, 1).

Read-only: the graph is opened with castalign's Graph.load, which connects with
sqlite mode=ro. Nothing is saved to the graph. Output: a text report and a PNG
in --out (default <graph folder>/diagnostics/<slice>/). The replay's graph
copy is written there too and deleted afterwards unless --keep-copy.

Run in .castalign-venv:
    python alignment/diagnose_slice_orientation.py --slice slice22_subslice_ALIGN_qc0_3_ge0_sat5
    python alignment/diagnose_slice_orientation.py --slice slice22_subslice_ALIGN_qc0_3_ge0_sat5 --target invivo_red
    python alignment/diagnose_slice_orientation.py --slice ... --skip-target-file   # step 2 loads the whole target TIFF
    python alignment/diagnose_slice_orientation.py --slice ... --keep-copy          # keep the replayed graph copy
    python alignment/diagnose_slice_orientation.py --self-test   # known answers first: proves the checks on this install
"""

import argparse
import ast
import gc
import json
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

SYMMETRIES = {
    # name: (numpy operation on a (y, x) image, 2x2 map of row vectors (y, x))
    "identity": (lambda a: a, [[1, 0], [0, 1]]),
    "rotate 90 counter-clockwise": (lambda a: np.rot90(a, 1), [[0, 1], [-1, 0]]),
    "rotate 180": (lambda a: np.rot90(a, 2), [[-1, 0], [0, -1]]),
    "rotate 90 clockwise": (lambda a: np.rot90(a, 3), [[0, -1], [1, 0]]),
    "flip left-right": (lambda a: a[:, ::-1], [[1, 0], [0, -1]]),
    "flip up-down": (lambda a: a[::-1, :], [[-1, 0], [0, 1]]),
    "transpose (y<->x)": (lambda a: a.T, [[0, 1], [1, 0]]),
    "anti-transpose": (lambda a: np.rot90(a, 2).T, [[0, -1], [-1, 0]]),
}
MIRRORED = {"flip left-right", "flip up-down", "transpose (y<->x)", "anti-transpose"}


class Report:
    def __init__(self):
        self.lines = []
        self.fails = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def status(self, step, ok, text):
        tag = "PASS" if ok is True else "FAIL" if ok is False else "INFO"
        if ok is False:
            self.fails.append(step)
        self(f"  [{tag}] {text}")


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------

def components(t, ca):
    """A transform split into the pieces castalign composed, in application order."""
    out = []
    while hasattr(t, "b") and hasattr(type(t), "pretransform"):
        out.append(t.b)
        t = type(t).pretransform()
    if not isinstance(t, ca.Identity):
        out.append(t)
    return list(reversed(out))


def local_map(t, point):
    """3x3 rows = where unit steps along z, y, x from `point` go under t (castalign row convention)."""
    p = np.asarray(point, dtype=float)
    base = np.asarray(t.transform(p[None]), dtype=float)[0]
    rows = [np.asarray(t.transform((p + e)[None]), dtype=float)[0] - base
            for e in np.eye(3)]
    return np.asarray(rows), base


def nearest_symmetry(block):
    """Closest of the 8 square symmetries to a 2x2 (y, x) block, and how close."""
    b = np.asarray(block, dtype=float)
    scale = np.sqrt(abs(np.linalg.det(b))) or 1.0
    bn = b / scale
    best = max(SYMMETRIES, key=lambda k: float(np.sum(np.asarray(SYMMETRIES[k][1]) * bn)))
    score = float(np.sum(np.asarray(SYMMETRIES[best][1]) * bn)) / 2.0   # 1.0 = exact
    return best, score


def describe_map(J):
    """Human description of what a 3x3 local map does to the slice plane on screen."""
    yx = J[1:, 1:]                       # images of +y and +x, screen components
    d2 = float(np.linalg.det(yx))
    d3 = float(np.linalg.det(J))
    sym, score = nearest_symmetry(yx)
    # castalign row convention: [[cos, sin], [-sin, cos]] is a counter-clockwise turn on screen
    # (zrotate=90 gives np.rot90's map). A mirrored map is written as flip left-right after a turn.
    unmirrored = yx if d2 >= 0 else yx @ np.diag([1.0, -1.0])
    angle = float(np.degrees(np.arctan2(unmirrored[0, 1], unmirrored[0, 0])))
    return {
        "angle": angle,
        "det3": d3,
        "turned_over": bool(J[0, 0] < 0),
        "screen_det": d2,
        "mirrored_on_screen": d2 < 0,
        "nearest": sym,
        "closeness": score,
        "y_to": J[1, 1:],
        "x_to": J[2, 1:],
    }


def fmt_map(d):
    return (f"det={d['det3']:+.3f}  turned over={'YES' if d['turned_over'] else 'no'}  "
            f"on-screen det={d['screen_det']:+.3f}  "
            f"mirrored on screen={'YES' if d['mirrored_on_screen'] else 'no'}  "
            f"nearest={d['nearest']} (closeness {d['closeness']:.2f})  "
            f"= {'flip left-right after ' if d['mirrored_on_screen'] else ''}"
            f"{d['angle']:+.1f} deg counter-clockwise")


def params_of(t):
    return getattr(t, "params", {})


def footprint(rendered2d):
    ys, xs = np.where(rendered2d > 0)
    if ys.size == 0:
        return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def best_symmetry_match(rendered2d, original2d, box):
    """Correlate a rendered image, cropped to `box`, against the 8 symmetries of the original."""
    from scipy import ndimage
    if box is None:
        return None, {}
    y0, y1, x0, x1 = box
    crop = rendered2d[y0:y1, x0:x1].astype(float)
    scores = {}
    for name, (op, _) in SYMMETRIES.items():
        ref = np.asarray(op(original2d), dtype=float)
        zoomed = ndimage.zoom(crop, (ref.shape[0] / crop.shape[0], ref.shape[1] / crop.shape[1]), order=1)
        zoomed = zoomed[:ref.shape[0], :ref.shape[1]]
        if zoomed.shape != ref.shape:
            continue
        a, b = zoomed.ravel(), ref.ravel()
        if a.std() == 0 or b.std() == 0:
            continue
        scores[name] = float(np.corrcoef(a, b)[0, 1])
    if not scores:
        return None, {}
    return max(scores, key=scores.get), scores


def probe_pattern(h, w):
    """Asymmetric test image: a frame, an 'F', and a block in the top-left corner."""
    img = np.zeros((h, w), dtype=np.uint8)
    t = max(2, min(h, w) // 25)
    img[:t, :] = img[-t:, :] = img[:, :t] = img[:, -t:] = 60          # frame
    y0, x0 = h // 5, w // 3
    fh, fw = (3 * h) // 5, w // 3
    img[y0:y0 + fh, x0:x0 + 2 * t] = 255                              # F stem
    img[y0:y0 + 2 * t, x0:x0 + fw] = 255                              # F top bar
    img[y0 + fh // 2:y0 + fh // 2 + 2 * t, x0:x0 + (2 * fw) // 3] = 255  # F middle bar
    img[2 * t:2 * t + h // 8, 2 * t:2 * t + w // 8] = 160             # top-left block
    return img


def render_like_mode_a(t, img3d):
    """What alignment_gui draws for a movable image: the warped array and its napari translate."""
    out = t.transform_image(img3d, output_size=None, labels=False, force_size=False)
    origin = np.asarray(t.origin_and_maxpos(img3d, output_size=None, force_size=False)[0])
    return np.asarray(out, dtype=np.float32), origin


# ----------------------------------------------------------------------------
# the steps
# ----------------------------------------------------------------------------

def step1_slice_file(rep, slice_img, slice_tif, loader):
    rep("\nSTEP 1  slice TIFF on disk vs slice node image in the graph")
    if slice_tif is None or not Path(slice_tif).exists():
        rep.status(1, None, f"TIFF not found ({slice_tif}); pass --slice-tif to check this step")
        return
    disk = np.asarray(loader(slice_tif))
    node = np.asarray(slice_img)
    rep(f"  file:  {slice_tif}")
    rep(f"  file shape {disk.shape} {disk.dtype}   node shape {node.shape} {node.dtype}")
    if disk.shape[1:] == node.shape[1:] and np.array_equal(disk, node):
        rep.status(1, True, "node image is identical to the file, pixel for pixel")
        return
    hits = [n for n, (op, _) in SYMMETRIES.items()
            if op(disk[0]).shape == node[0].shape and np.array_equal(op(disk[0]), node[0])]
    rep.status(1, False, f"node image differs from the file; exact match under: {hits or 'none of the 8 symmetries'}")


def step2_target_file(rep, target_img, target_tif, loader):
    rep("\nSTEP 2  target TIFF on disk vs target node image in the graph (stored lossy, so correlation)")
    if target_tif is None:
        rep.status(2, None, "skipped (no TIFF path; pass --target-tif, or drop --skip-target-file)")
        return
    disk = np.asarray(loader(target_tif), dtype=np.float32)
    node = np.asarray(target_img, dtype=np.float32)
    rep(f"  file:  {target_tif}")
    rep(f"  file shape {disk.shape}   node shape {node.shape}")
    if disk.shape[0] != node.shape[0]:
        rep.status(2, False, "different number of z planes")
        return
    Z = disk.shape[0]
    spread = node.reshape(Z, -1).std(axis=1)
    planes = [int(z) for z in np.argsort(spread)[::-1][:3] if spread[z] > 0]
    rep(f"  comparing node planes {planes} (the three with the most signal)")
    results = {}
    for zorder in ("same z order", "z reversed"):
        for name, (op, _) in SYMMETRIES.items():
            vals = []
            for z in planes:
                zf = z if zorder == "same z order" else Z - 1 - z
                a = op(disk[zf])[::2, ::2]
                b = node[z][::2, ::2]
                if a.shape != b.shape:
                    vals = None
                    break
                if a.std() == 0 or b.std() == 0:
                    vals.append(0.0)
                    continue
                vals.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
            if vals:
                results[(zorder, name)] = float(np.mean(vals))
    ranked = sorted(results.items(), key=lambda kv: -kv[1])
    for (zorder, name), r in ranked[:4]:
        rep(f"    r={r:+.3f}  file {name}, {zorder}")
    best = ranked[0][0] if ranked else None
    rep.status(2, best == ("same z order", "identity"),
               f"best match: {best} (r={ranked[0][1]:+.3f})" if ranked else "no comparable symmetry")


def _cell_source(nb, marker):
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            if marker in src:
                return "\n".join(l for l in src.splitlines()
                                 if not l.lstrip().startswith(("%", "!")))
    raise SystemExit(f"no notebook code cell contains {marker!r}")


def _function_source(src, names):
    tree = ast.parse(src)
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in found}
    if missing:
        raise SystemExit(f"notebook cell is missing {sorted(missing)}")
    return "\n\n".join(ast.get_source_segment(src, n) for n in found)


class GuiRecorder:
    """Stands in for castalign.gui: records each align_interactive call and returns a scripted answer."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def align_interactive(self, nodes_movable=None, nodes_fixed=None, graph=None,
                          transform=None, references=(), start=None):
        self.calls.append({"movable": nodes_movable, "fixed": nodes_fixed,
                           "graph": graph, "transform": transform})
        return self.answers.pop(0)


def _same_points(t1, t2, shape, seed=0):
    pts = np.random.default_rng(seed).uniform(0, 1, (200, 3)) * np.asarray(shape, dtype=float)
    return float(np.max(np.abs(np.asarray(t1.transform(pts)) - np.asarray(t2.transform(pts)))))


def step3_notebook_replay(rep, g, ca, graph_path, notebook, slice_name, target, work_dir):
    rep(f"\nSTEP 3  notebook replay: {notebook.name} cells executed on a copy of the graph")
    nb = json.loads(Path(notebook).read_text(encoding="utf-8"))
    # The helpers live in the startup cell, which also imports castalign.gui,
    # loads the REAL graph from local_config and sets slice_node = None. Run
    # only the function definitions the mode cells call.
    cell_utils = _function_source(_cell_source(nb, "# ALIGNMENT UTILITIES"),
                                  {"get_initial_transform_slice_to_target", "load_and_pad_slice",
                                   "save_alignment", "get_previous_transform"})
    cell_picker = _cell_source(nb, "# CHAIN PICKER")
    cell_mode_c = _cell_source(nb, "# MODE C:")
    cell_mode_a = _cell_source(nb, "# MODE A:")
    pad_m = re.search(r"^PAD_Z\s*=\s*(\d+)", cell_mode_c, re.M)
    rep_m = re.search(r"^REPEAT_SLICE_IN_Z\s*=\s*(True|False)", cell_mode_c, re.M)
    if not pad_m or not rep_m:
        raise SystemExit("Mode C cell no longer sets PAD_Z / REPEAT_SLICE_IN_Z at top level")
    pad_z, repeat = int(pad_m.group(1)), rep_m.group(1) == "True"
    rep(f"  Mode C cell as committed: PAD_Z = {pad_z}, REPEAT_SLICE_IN_Z = {repeat}")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    copy_path = work_dir / "replay_copy.db"
    shutil.copy(graph_path, copy_path)
    gcopy = ca.Graph.load(str(copy_path))
    rep(f"  graph copy: {copy_path}")

    # The Mode C GUI answer: the fit the real graph already stores (route with -PAD_Z undone).
    stored_route = g.get_transform(target, slice_name)
    mode_c_answer = stored_route + ca.TranslateFixed(z=pad_z)
    recorder = GuiRecorder([mode_c_answer, ca.Identity()])

    has_block = target.startswith("block")
    slice_nodes = sorted(n for n in gcopy.nodes if "_subslice_ALIGN" in n)
    ns = {
        "__name__": "notebook_replay", "np": np, "ca": ca, "re": re, "gc": gc, "Path": Path,
        "g": gcopy, "GRAPH_PATH": copy_path,
        "ca_gui": types.SimpleNamespace(align_interactive=recorder.align_interactive),
        "HAS_BLOCK": has_block,
        "block_node": target if has_block else None,
        "invivo_node": ("invivo_red" if "invivo_red" in gcopy.nodes else None) if has_block else target,
        "slice_node": slice_name,
        "slice_nodes": slice_nodes,
        "aligned_slices": [n for n in slice_nodes if gcopy.has_transform(n, target)],
        "unaligned_slices": [n for n in slice_nodes if not gcopy.has_transform(n, target)],
        "pearsonr": None, "tifffile": None,
    }
    exec(_function_source(cell_picker, {"extract_slice_number", "slice_variant", "get_selected_slice"}), ns)
    exec(cell_utils, ns)
    ns["refresh_slice_dropdown"] = lambda: None          # bound to an ipywidget in the notebook

    rep("  --- Mode C cell output ---")
    exec(cell_mode_c, ns)
    rep("  --- Mode A cell output ---")
    exec(cell_mode_a, ns)
    rep("  --- end of notebook output ---")

    if len(recorder.calls) != 2:
        rep.status(3, False, f"expected 2 align_interactive calls, recorded {len(recorder.calls)}")
        return None
    C, A = recorder.calls
    s_img = np.asarray(g.get_image(slice_name))
    t_img = np.asarray(g.get_image(target))
    H, W = s_img.shape[1:]

    rep("  Mode C call:")
    fixed_c = np.asarray(C["fixed"])
    rep.status(3, isinstance(C["movable"], np.ndarray) and np.array_equal(np.asarray(C["movable"]), t_img),
               f"movable = target node image {t_img.shape}, same array values")
    rep.status(3, fixed_c.shape == (2 * pad_z + 1, H, W),
               f"fixed shape {fixed_c.shape} = (2*PAD_Z+1, H, W) = {(2 * pad_z + 1, H, W)}")
    planes = range(fixed_c.shape[0]) if repeat else [pad_z]
    ok_planes = all(np.array_equal(fixed_c[k], s_img[0]) for k in planes)
    rep.status(3, ok_planes, f"fixed plane(s) {'0..' + str(2 * pad_z) if repeat else pad_z} equal the slice node image, same (y, x) order")
    rep.status(3, isinstance(C["transform"], ca.Identity), f"start transform: {C['transform']!r}")
    rep.status(3, C["graph"] is None, f"graph= passed: {C['graph'] is not None}")
    saved = gcopy.get_transform(target, slice_name)
    err = _same_points(saved, mode_c_answer + ca.TranslateFixed(z=-pad_z), t_img.shape)
    rep.status(3, err < 1e-6, f"Mode C saved (GUI fit + TranslateFixed(z=-{pad_z})); max point error {err:.2e}")
    err = _same_points(saved, stored_route, t_img.shape)
    rep.status(3, err < 1e-6, f"replayed save reproduces the route in the real graph; max point error {err:.2e}")

    rep("  Mode A call:")
    rep.status(3, isinstance(A["movable"], np.ndarray) and np.array_equal(np.asarray(A["movable"]), s_img),
               f"movable = slice node image {s_img.shape}, same array values")
    rep.status(3, isinstance(A["fixed"], np.ndarray) and np.array_equal(np.asarray(A["fixed"]), t_img),
               f"fixed = target node image {t_img.shape}, same array values")
    seed = A["transform"]
    rep(f"  start transform: {seed!r}")
    pts = np.random.default_rng(2).uniform(0, 1, (200, 3)) * np.asarray(t_img.shape, dtype=float)
    err = float(np.max(np.abs(np.asarray(seed.transform(saved.transform(pts))) - pts)))
    rep.status(3, err < 1e-3, f"Mode A start transform undoes what Mode C saved; max |seed(saved(p)) - p| = {err:.2e}")
    err = _same_points(seed, g.get_transform(slice_name, target), s_img.shape)
    rep.status(3, err < 1e-6, f"Mode A start transform = g.get_transform(slice, target) on the real graph; max error {err:.2e}")
    return {"C": C, "A": A, "pad_z": pad_z, "repeat": repeat, "mode_c_fit": mode_c_answer,
            "seed": seed, "copy_path": copy_path}


def _route_parts(rep, g, ca, frm, to):
    chain = [frm] + list(g.get_chain(frm, to))
    rep(f"  route: {' -> '.join(chain)}")
    parts = []
    for a, b in zip(chain, chain[1:]):
        e = g.edges[a][b]
        rep(f"  hop {a} -> {b}:")
        rep(f"    {e!r}")
        for c in components(e, ca):
            parts.append((f"{a}->{b}", c))
    rep(f"  {len(parts)} component(s), in the order they are applied:")
    for i, (hop, c) in enumerate(parts):
        rep(f"    #{i}  [{hop}]  {c!r}")
    return chain, parts


def step4_route(rep, g, ca, target, slice_name):
    rep(f"\nSTEP 4  stored routes between {target} and {slice_name}")
    direct = slice_name in g.edges.get(target, {})
    rep.status(4, None, f"direct edge {target} <-> {slice_name}: {'yes' if direct else 'NO (composed route)'}")
    rep(f"  (a) {target} -> {slice_name}, the direction Mode C saves (its GUI fit + TranslateFixed(z=-PAD_Z)):")
    _route_parts(rep, g, ca, target, slice_name)
    rep(f"  (b) {slice_name} -> {target}, the stored inverse edges Mode A composes:")
    chain_rev, parts_rev = _route_parts(rep, g, ca, slice_name, target)
    return chain_rev, parts_rev


def step5_inverse(rep, g, target, slice_name, target_shape):
    rep(f"\nSTEP 5  Mode A's start transform g.get_transform({slice_name}, {target}) is the inverse of step 4")
    fwd = g.get_transform(target, slice_name)
    seed = g.get_transform(slice_name, target)
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, (200, 3)) * np.asarray(target_shape, dtype=float)
    try:
        err = float(np.max(np.abs(np.asarray(seed.transform(fwd.transform(pts))) - pts)))
    except Exception as e:                     # nonlinear numerical inverse limits
        rep.status(5, None, f"could not evaluate ({type(e).__name__}: {e})")
        return fwd, seed
    rep.status(5, err < 1e-3, f"max |seed(route(p)) - p| over 200 random target points = {err:.2e} px")
    if hasattr(seed, "matrix") and hasattr(fwd, "matrix"):
        merr = float(np.max(np.abs(np.asarray(seed.matrix) @ np.asarray(fwd.matrix) - np.eye(3))))
        rep.status(5, merr < 1e-6, f"seed.matrix x route.matrix = I to within {merr:.2e}")
    return fwd, seed


def step6_keypress(rep, ca, seed, slice_shape):
    rep("\nSTEP 6  pressing t in Mode A (align_interactive: t + TranslateRotateFixed) starts from the same transform")
    try:
        started = (seed + ca.TranslateRotateFixed)()
    except Exception as e:
        rep.status(6, None, f"could not compose ({type(e).__name__}: {e})")
        return
    pts = np.random.default_rng(1).uniform(0, 1, (200, 3)) * np.asarray(slice_shape, dtype=float)
    err = float(np.max(np.abs(np.asarray(started.transform(pts)) - np.asarray(seed.transform(pts)))))
    rep.status(6, err < 1e-6, f"max difference from the seed before any slider moves = {err:.2e} px")


def step7_orientation(rep, parts, slice_shape):
    rep("\nSTEP 7  on-screen orientation, component by component (slice centre, castalign's own transform())")
    rep("  'turned over' = the slice's +z now points to -z (a 180-degree turn about an in-plane axis).")
    rep("  'mirrored on screen' = the slice would look like its mirror image in napari's (y down, x right) view.")
    centre = np.asarray([0.0, slice_shape[1] / 2.0, slice_shape[2] / 2.0])
    inv_parts = parts            # step 4 (b): the slice -> target components Mode A applies, in order
    running = centre
    J_total = np.eye(3)
    first_mirror = None
    for i, (hop, c) in enumerate(inv_parts):
        J, running_next = local_map(c, running)
        J_total = J_total @ J
        d_own = describe_map(J)
        d_cum = describe_map(J_total)
        rep(f"  [{hop}] {c!r}")
        rep(f"      this piece:  {fmt_map(d_own)}")
        rep(f"      so far:      {fmt_map(d_cum)}")
        p = params_of(c)
        rot = {k: p[k] for k in ("zrotate", "yrotate", "xrotate") if k in p}
        if rot:
            rep(f"      rotation parameters: {rot}   invert={p.get('invert')}")
        if first_mirror is None and d_cum["mirrored_on_screen"]:
            first_mirror = (hop, c, d_own)
        running = running_next
    final = describe_map(J_total)
    rep("")
    rep(f"  Mode A draws the slice (relative to how Mode C draws it): {final['nearest']}"
        f" (closeness {final['closeness']:.2f}), mirrored on screen: {'YES' if final['mirrored_on_screen'] else 'no'}")
    rep(f"    exactly: {'flip left-right after ' if final['mirrored_on_screen'] else ''}"
        f"a {final['angle']:+.1f} degree counter-clockwise turn")
    rep(f"    slice +y (down)  now points to screen (dy, dx) = ({final['y_to'][0]:+.3f}, {final['y_to'][1]:+.3f})")
    rep(f"    slice +x (right) now points to screen (dy, dx) = ({final['x_to'][0]:+.3f}, {final['x_to'][1]:+.3f})")
    if final["mirrored_on_screen"] and first_mirror is not None:
        hop, c, d_own = first_mirror
        why = ("a real reflection (det < 0)" if d_own["det3"] < 0 else
               "a turn-over (proper rotation, det > 0, that flips the slice face down)" if d_own["turned_over"] else
               "an in-plane mirror")
        rep(f"  The mirror first appears at [{hop}] {c!r}: {why}.")
    rep.status(7, None, "orientation computed from castalign's transform() on the stored graph")
    return final


def step8_images(rep, ca, seed, slice_img, final):
    rep("\nSTEP 8  image evidence: the same castalign call Mode A uses (transform_image, labels=False)")
    s = np.asarray(slice_img)
    probe = probe_pattern(s.shape[1], s.shape[2])
    out = {}
    box = None
    for label, img2d in (("test pattern", probe), ("real slice", s[0])):
        rendered, origin = render_like_mode_a(seed, img2d[None])
        if rendered.shape[0] == 0 or not rendered.any():
            # castalign keeps the render origin as float32; for a one-plane image under a rotation
            # the sampled z can land just outside [0, 1] (or the output gets 0 planes), so napari
            # is handed an empty or all-zero layer. Measured ~50% of sub-pixel positions.
            rep.status(8, None, f"{label}: castalign rendered {'no planes' if rendered.shape[0] == 0 else 'all zeros'} "
                                f"{rendered.shape} -- the slice would be INVISIBLE in Mode A (float32 origin rounding)")
            flat = np.zeros(rendered.shape[1:], dtype=np.float32)
            out[label] = (flat, origin, rendered)
            rep.image_matches[label] = None
            continue
        flat = rendered.max(axis=0)
        if box is None:
            box = footprint(flat)          # the pattern's frame fills the slice rectangle
        best, scores = best_symmetry_match(flat, img2d, box)
        out[label] = (flat, origin, rendered)
        rep.image_matches[label] = best
        if best is None:
            rep.status(8, None, f"{label}: nothing rendered to compare")
            continue
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        rep(f"  {label}: rendered {rendered.shape} at napari translate {np.round(origin, 1).tolist()}")
        for name, r in ranked[:3]:
            rep(f"    r={r:+.3f}  {name}")
        if final is not None and final["closeness"] > 0.95:
            rep.status(8, best == final["nearest"],
                       f"{label}: best image match '{best}' vs step 7 '{final['nearest']}'")
        else:
            rep.status(8, None, f"{label}: best image match '{best}' (step 7 map is not close to a 90-degree symmetry, so no agreement test)")
    return out, probe


def step10_mode_a_planes(rep, seed, slice_img, rendered_real):
    rep("\nSTEP 10  what Mode A's 2D view can show (napari draws ONE z plane of each layer at a time)")
    s = np.asarray(slice_img)
    H, W = s.shape[1:]
    o = np.asarray(seed.transform(np.asarray([[0.0, 0.0, 0.0]])))[0]
    n = np.asarray(seed.transform(np.asarray([[1.0, 0.0, 0.0]])))[0] - o      # where the slice's +z points
    tilt = float(np.degrees(np.arccos(min(1.0, abs(n[0]) / np.linalg.norm(n)))))
    corners = np.asarray(seed.transform(np.asarray([[0.0, y, x] for y in (0, H) for x in (0, W)], float)))
    centre_z = float(np.asarray(seed.transform(np.asarray([[0.0, H / 2.0, W / 2.0]])))[0][0])
    rep(f"  slice normal vs target z axis: {tilt:.2f} degrees"
        f"{' (turned over: slice +z points to target -z)' if n[0] < 0 else ''}")
    rep(f"  slice corners land on target z = {corners[:, 0].min():.2f} .. {corners[:, 0].max():.2f}; "
        f"centre z = {centre_z:.2f}")
    flat, origin, rendered = rendered_real
    if rendered.shape[0] == 0 or not rendered.any():
        rep.status(10, False, "castalign rendered the slice blank -- Mode A shows NOTHING at any z")
        return None
    thr = 0.25 * float(rendered.max())
    foot = int((rendered.max(axis=0) > thr).sum())
    counts = [int((rendered[k] > thr).sum()) for k in range(rendered.shape[0])]
    planes = [float(origin[0]) + k for k, c in enumerate(counts) if c > 0]
    k_best = int(np.argmax(counts))
    z_best = float(origin[0]) + k_best
    frac = counts[k_best] / foot if foot else 0.0
    rep(f"  warped slice has signal on {len(planes)} target plane(s), z = {planes[0]:.1f} .. {planes[-1]:.1f}")
    rep(f"  best single plane z = {z_best:.1f} holds {frac:.0%} of the slice's drawn footprint")
    if len(planes) > 1:
        rep("  Mode C resamples the TARGET onto the slice's plane, so it shows the whole slice against one")
        rep("  oblique section. Mode A cannot: it shows the target's own planes, and on each one only the")
        rep("  strip of the slice that crosses it. Scroll z in Mode A, or view in 3D, to see the rest.")
    rep.status(10, None, f"Mode A 2D: scroll to z = {round(z_best)}; one plane shows {frac:.0%} of the slice "
                         f"(tilt {tilt:.2f} degrees)")
    rep.mode_a_plane = {"z": z_best, "k": k_best, "frac": frac, "tilt": tilt, "n_planes": len(planes)}
    return rep.mode_a_plane


def step9_figure(rep, out_png, replay, rendered, probe, plane=None):
    rep("\nSTEP 9  figure, from the arrays and transforms the replay recorded")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C, A, pad_z = replay["C"], replay["A"], replay["pad_z"]
    fixed_c = np.asarray(C["fixed"])
    H, W = fixed_c.shape[1:]
    # Mode C: castalign draws the movable (target) through the GUI fit on a grid that starts at
    # origin_and_maxpos(...)[0]. Sampling the same mapping on the slice's own pixel grid gives the
    # same values up to that sub-pixel offset (on the GUI's exact grid the two are identical).
    try:
        tgt_c = np.asarray(replay["mode_c_fit"].transform_image(
            np.asarray(C["movable"]), output_size=[(pad_z, pad_z + 1), (0, H), (0, W)],
            labels=False, force_size=True))[0]
    except Exception as e:
        rep(f"  (could not render the Mode C target view: {type(e).__name__}: {e})")
        tgt_c = np.zeros((H, W))
    # Mode A: castalign draws the movable (slice) through the start transform, at a napari translate.
    seed = A["transform"]
    flat_a, origin_a, rendered_a = rendered["real slice"]
    flat_p = rendered["test pattern"][0]
    T = np.asarray(A["fixed"])
    if plane is not None:
        # napari's 2D view: one plane of the warped slice, over the target plane at the same world z
        flat_a = rendered_a[plane["k"]]
        zc = int(np.clip(round(plane["z"]), 0, T.shape[0] - 1))
        a_title = f"MODE A movable: slice on plane z={zc} ({plane['frac']:.0%} of it)"
    else:
        centre = np.asarray(seed.transform(np.asarray([[0.0, H / 2.0, W / 2.0]])))[0]
        zc = int(np.clip(round(centre[0]), 0, T.shape[0] - 1))
        a_title = "MODE A movable: slice via start transform"
    y0, x0 = int(round(origin_a[1])), int(round(origin_a[2]))
    tgt_a = np.zeros_like(flat_a)
    ys0, xs0 = max(0, y0), max(0, x0)
    ys1, xs1 = min(T.shape[1], y0 + flat_a.shape[0]), min(T.shape[2], x0 + flat_a.shape[1])
    if ys1 > ys0 and xs1 > xs0:
        tgt_a[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = T[zc, ys0:ys1, xs0:xs1]

    def norm(a):
        a = np.asarray(a, dtype=float)
        hi = np.percentile(a, 99.5) if a.size else 1
        return np.clip(a / (hi or 1), 0, 1)

    fig, ax = plt.subplots(3, 3, figsize=(15, 14))
    panels = [
        (ax[0, 0], norm(fixed_c[pad_z]), f"MODE C fixed: slice, plane {pad_z}"),
        (ax[0, 1], norm(tgt_c), f"MODE C movable: target via GUI fit, plane {pad_z}"),
        (ax[0, 2], None, "MODE C overlay: slice red, target green"),
        (ax[1, 0], norm(flat_a), a_title),
        (ax[1, 1], norm(tgt_a), f"MODE A fixed: target, plane z={zc}"),
        (ax[1, 2], None, "MODE A overlay: target red, slice green"),
        (ax[2, 0], norm(probe), "test pattern: Mode C view (as stored)"),
        (ax[2, 1], norm(flat_p), "test pattern: Mode A view"),
    ]
    for a, img, title in panels:
        if img is not None:
            a.imshow(img, cmap="gray", interpolation="nearest")
        a.set_title(title, fontsize=10)
        a.set_xlabel("x  ->")
        a.set_ylabel("<-  y (down)")
    rgb = np.zeros((H, W, 3))
    rgb[..., 0] = norm(fixed_c[pad_z])
    rgb[..., 1] = norm(tgt_c)
    ax[0, 2].imshow(rgb)
    rgb2 = np.zeros((*flat_a.shape, 3))
    rgb2[..., 0] = norm(tgt_a)
    rgb2[..., 1] = norm(flat_a)
    ax[1, 2].imshow(rgb2)
    ax[2, 2].axis("off")
    ax[2, 2].text(0, 0.5, "\n".join(l[:110] for l in rep.lines if l.strip().startswith(("[PASS]", "[FAIL]", "Mode A draws")))[-2500:],
                  fontsize=6, family="monospace", va="center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    rep.status(9, None, f"wrote {out_png}")


def run(g, ca, graph_path, notebook, slice_name, target, slice_tif, slice_loader,
        target_tif, target_loader, out_dir, keep_copy=False):
    rep = Report()
    rep.final = None
    rep.mode_a_plane = None
    rep.image_matches = {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep(f"graph:       {graph_path}")
    rep(f"notebook:    {notebook}")
    rep(f"slice node:  {slice_name}")
    rep(f"target node: {target}")
    slice_img = g.get_image(slice_name)
    target_img = g.get_image(target)
    rep(f"slice node image {np.asarray(slice_img).shape}, target node image {np.asarray(target_img).shape}")

    final = None
    rep("\nSTEP 0  what Mode A starts from")
    if g.has_transform(slice_name, target):
        rep.status(0, None, "a route exists, so Mode A starts from g.get_transform(slice, target)")
    else:
        rep.status(0, False, "NO route: Mode A would start from a neighbouring slice's transform "
                             "(get_previous_transform) or Identity -- this report cannot continue")
        return rep

    step1_slice_file(rep, slice_img, slice_tif, slice_loader)
    step2_target_file(rep, target_img, target_tif, target_loader)
    work = Path(tempfile.mkdtemp(prefix="replay_", dir=out_dir))
    try:
        replay = step3_notebook_replay(rep, g, ca, graph_path, Path(notebook), slice_name, target, work)
        if replay is None:
            return rep
        _, parts_rev = step4_route(rep, g, ca, target, slice_name)
        step5_inverse(rep, g, target, slice_name, np.asarray(target_img).shape)
        seed = replay["seed"]
        step6_keypress(rep, ca, seed, np.asarray(slice_img).shape)
        final = step7_orientation(rep, parts_rev, np.asarray(slice_img).shape)
        rep.final = final
        rendered, probe = step8_images(rep, ca, seed, replay["A"]["movable"], final)
        plane = step10_mode_a_planes(rep, seed, replay["A"]["movable"], rendered["real slice"])
        step9_figure(rep, out_dir / "orientation_steps.png", replay, rendered, probe, plane)
    finally:
        if not keep_copy:
            gc.collect()
            shutil.rmtree(work, ignore_errors=True)

    rep("\nSUMMARY")
    rep(f"  failed steps: {sorted(set(rep.fails)) or 'none'}")
    if final is not None:
        rep(f"  Mode A draws this slice as: {final['nearest']} of how Mode C draws it"
            f" (mirrored on screen: {'YES' if final['mirrored_on_screen'] else 'no'})")
    if rep.mode_a_plane is not None:
        p = rep.mode_a_plane
        rep(f"  Mode A 2D view: the slice spans {p['n_planes']} target plane(s); at z = {round(p['z'])} one plane "
            f"shows {p['frac']:.0%} of it (tilt {p['tilt']:.2f} degrees)")
    (out_dir / "orientation_steps.txt").write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    print(f"\nreport: {out_dir / 'orientation_steps.txt'}")
    return rep


SELF_TEST_CASES = [
    # name, rotation of the true slice->target map, extra flip, expected on-screen result, mirrored
    ("translation only", {}, False, "identity", False),
    ("zrotate 90", {"zrotate": 90.0}, False, "rotate 90 counter-clockwise", False),
    ("yrotate 180 (turn-over)", {"yrotate": 180.0}, False, "flip left-right", True),
    ("zrotate -90 + yrotate 180 (turn-over)", {"zrotate": -90.0, "yrotate": 180.0}, False, "transpose (y<->x)", True),
    ("flip x (reflection)", {}, True, "flip left-right", True),
    ("xrotate 5 (tilted cut)", {"xrotate": 5.0}, False, "identity", False),
]


def self_test(ca, notebook, keep=False):
    """Run every step on synthetic graphs whose answer is known, using this machine's castalign."""
    import tifffile
    from scipy import ndimage
    pad, (H, W), (Z, Y, X) = 200, (60, 90), (60, 140, 140)
    pattern = np.zeros((H, W), np.uint8)
    pattern[5:15, 5:40] = 200
    pattern[20:55, 60:65] = 255
    pattern[40:45, 20:60] = 120
    pattern[50:58, 5:10] = 80
    root = Path(tempfile.mkdtemp(prefix="orientation_selftest_"))
    rows = []
    try:
        for name, rot, flip, expect, expect_mirror in SELF_TEST_CASES:
            d = root / re.sub(r"[^a-z0-9]+", "_", name.lower())
            d.mkdir()
            R = ca.TranslateRotateFixed(**rot)
            if flip:
                R = ca.FlipFixed(x=True, xthickness=W) + R
            corners = np.array([[pad + dz, y, x] for dz in (0, 1) for y in (0, H) for x in (0, W)], float)
            shift = np.array([25.0, 20.0, 20.0]) - np.asarray(R.transform(corners)).min(axis=0)
            truth = R + ca.TranslateFixed(z=shift[0], y=shift[1], x=shift[2])     # padded slice -> target
            padded = np.zeros((2 * pad + 1, H, W), np.float32)
            padded[pad] = pattern
            target = np.asarray(truth.transform_image(padded, output_size=[(0, Z), (0, Y), (0, X)],
                                                      labels=False, force_size=True), np.float32)
            texture = ndimage.gaussian_filter(np.random.default_rng(7).normal(size=target.shape), 3)
            target = (target + 30 + 200 * texture).clip(0, None).astype(np.float32)
            g = ca.Graph("selftest")
            sname, tname = "slice22_subslice_ALIGN_selftest", "block_stack_red"
            g.add_node(sname, image=pattern[None], compression="label")         # as the builder stores them
            g.add_node(tname, image=target, compression="high")
            g.add_edge(tname, sname, truth.invert() + ca.TranslateFixed(z=-pad))  # a perfect Mode C fit, saved as cell 22 does
            db = d / "selftest.db"
            g.save(str(db))
            tifffile.imwrite(d / "slice.tif", pattern)
            tifffile.imwrite(d / "target.tif", target)
            g2 = ca.Graph.load(str(db))
            print(f"\n{'#' * 20} SELF-TEST: {name} {'#' * 20}")
            rep = run(g2, ca, db, notebook, sname, tname, d / "slice.tif",
                      lambda p: tifffile.imread(p)[None], d / "target.tif",
                      lambda p: tifffile.imread(p).astype(np.float32), d / "out")
            got = rep.final["nearest"] if rep.final else None
            mirror = rep.final["mirrored_on_screen"] if rep.final else None
            imgs = sorted(set(rep.image_matches.values()))
            ok = (not rep.fails and got == expect and mirror == expect_mirror and imgs == [expect])
            p = rep.mode_a_plane
            # step 10: a flat fit puts the whole slice on one plane; a tilted one cannot
            ok = ok and p is not None and ((p["frac"] > 0.9) == (abs(rot.get("xrotate", 0.0)) < 1))
            plane_txt = f"step 10 {p['frac']:.0%} on z={round(p['z'])}, {p['n_planes']} planes" if p else "step 10 blank"
            rows.append((name, expect, got, imgs, sorted(set(rep.fails)), plane_txt, ok))
    finally:
        if not keep:
            gc.collect()
            shutil.rmtree(root, ignore_errors=True)
    print("\n" + "=" * 100)
    print("SELF-TEST SUMMARY (known answer vs what the diagnostic reports)")
    for name, expect, got, imgs, fails, plane_txt, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:40s} expected {expect!r:30s} step 7 {got!r:30s} "
              f"step 8 {imgs} {plane_txt} failed steps {fails or 'none'}")
    if keep:
        print(f"  files kept in {root}")
    return all(r[-1] for r in rows)


def main():
    ap = argparse.ArgumentParser(description="Trace how one slice is oriented in Mode C and Mode A",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--slice", default=None, help="slice node name, as in the Chain Picker")
    ap.add_argument("--self-test", action="store_true",
                    help="run every step on synthetic graphs with known answers, then exit")
    ap.add_argument("--target", default=None,
                    help="target node (default: block_stack_red if in the graph, else invivo_red; "
                         "use whatever the Chain Picker's slice target was)")
    ap.add_argument("--slice-tif", default=None, help="override the slice TIFF path for step 1")
    ap.add_argument("--target-tif", default=None, help="override the target TIFF path for step 2")
    ap.add_argument("--skip-target-file", action="store_true", help="skip step 2 (loads the whole target TIFF)")
    ap.add_argument("--graph", default=None, help="graph .db (default: the builder's GRAPH_PATH rule)")
    ap.add_argument("--notebook", default=str(_HERE / "castalign_testground.ipynb"),
                    help="notebook whose cells are replayed (default: alignment/castalign_testground.ipynb)")
    ap.add_argument("--keep-copy", action="store_true", help="keep the replayed graph copy in the output folder")
    ap.add_argument("--out", default=None, help="output folder (default: <graph folder>/diagnostics/<slice>)")
    args = ap.parse_args()

    import castalign as ca
    if args.self_test:
        raise SystemExit(0 if self_test(ca, args.notebook, keep=args.keep_copy) else 1)
    if not args.slice:
        ap.error("--slice is required (or pass --self-test)")
    import subslice_graph_builder as sgb
    from check_align_node_render import source_tif

    if args.graph:
        graph_path = Path(args.graph)
    elif sgb.GRAPH_PATH:
        graph_path = Path(sgb.GRAPH_PATH)
    else:
        graph_path = sgb._derive_graph_path(
            Path(sgb.BLOCK_STACK_PATH_RED) if sgb.BLOCK_STACK_PATH_RED else None,
            Path(sgb.INVIVO_PATH_RED) if sgb.INVIVO_PATH_RED else None)
    if not graph_path.exists():
        raise SystemExit(f"Graph not found: {graph_path}")
    g = ca.Graph.load(str(graph_path))
    print(f"graph: {graph_path} ({len(g.nodes)} nodes, opened read-only)")

    if args.slice not in g.nodes:
        raise SystemExit(f"{args.slice} is not in the graph. Slice nodes: "
                         f"{sorted(n for n in g.nodes if n.startswith('slice'))}")
    target = args.target or ("block_stack_red" if "block_stack_red" in g.nodes else "invivo_red")
    if target not in g.nodes:
        raise SystemExit(f"target {target} is not in the graph. Nodes: {sorted(g.nodes)}")

    slice_tif = args.slice_tif
    try:
        if slice_tif is None and "_subslice_ALIGN_" in args.slice:
            from analysis_paths import resolve_subslice_dir
            slice_tif = source_tif(args.slice, resolve_subslice_dir())
        elif slice_tif is None:
            m = re.match(r"slice(\d+)_([a-z]+)$", args.slice)
            if m:
                from analysis_paths import hyb_downsampled_dir
                slice_tif = sgb.raw_channel_tif(hyb_downsampled_dir(), int(m.group(1)), m.group(2))
    except Exception as e:           # step 1 is skipped, not the whole report
        print(f"could not locate the slice TIFF ({type(e).__name__}: {e}); pass --slice-tif")
        slice_tif = None

    target_tif, target_loader = None, None
    if not args.skip_target_file:
        paths = {"invivo_red": sgb.INVIVO_PATH_RED, "invivo_green": sgb.INVIVO_PATH_GREEN,
                 "block_stack_red": sgb.BLOCK_STACK_PATH_RED, "block_stack_green": sgb.BLOCK_STACK_PATH_GREEN}
        target_tif = args.target_tif or paths.get(target) or None
        target_loader = sgb.load_block_stack if target.startswith("block") else sgb.load_invivo_stack

    out_dir = Path(args.out) if args.out else graph_path.parent / "diagnostics" / args.slice
    run(g, ca, graph_path, args.notebook, args.slice, target, slice_tif, sgb.load_single_subslice,
        target_tif, target_loader, out_dir, keep_copy=args.keep_copy)


if __name__ == "__main__":
    main()
