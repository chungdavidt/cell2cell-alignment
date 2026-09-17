"""Report how each castalign napari window is showing the (y, x) face of the data.

The same array can look transposed or mirrored in one castalign window and not
in another purely because of the view (measured on napari 0.7.0 with castalign's
own alignment_gui):

    Ctrl+T / transpose button   2D and 3D: the image is transposed
    Ctrl+E / roll button        z goes on screen (the "vertical beam" look)
    3D orbit to the other face  mirrored (flip left-right / up-down, or a
                                transpose if the camera is also turned 90 deg)

None of these carry over to the next window; each align_interactive call opens
a fresh napari.Viewer. napari's console is disabled under Jupyter and the
notebook is blocked while the window is open, so this has to be installed
before the mode cell runs:

    import napari_view_state; napari_view_state.install()     # notebook cell, once
    ... run Mode C / Mode A ...
    napari_view_state.uninstall()

Each window then labels its axes z/y/x, shows the axes overlay, prints its view
state when it opens, whenever the verdict changes (axis order, 2D/3D, camera),
and on F9. The verdict is relative to the array as stored (y down, x right).
"""

import numpy as np

_SYMMETRIES = {
    "identity": [[1, 0], [0, 1]],
    "rotate 90 counter-clockwise": [[0, 1], [-1, 0]],
    "rotate 180": [[-1, 0], [0, -1]],
    "rotate 90 clockwise": [[0, -1], [1, 0]],
    "flip left-right": [[1, 0], [0, -1]],
    "flip up-down": [[-1, 0], [0, 1]],
    "transpose (y<->x)": [[0, 1], [1, 0]],
    "anti-transpose": [[0, -1], [-1, 0]],
}


def classify_view(displayed, ndisplay, orientation2d=("down", "right"),
                  view_direction=(-1, 0, 0), up_direction=(0, -1, 0)):
    """Verdict for how the data's (y, x) face appears on screen.

    `displayed` are data axis indices in napari's displayed order (dims.displayed);
    `view_direction` / `up_direction` are in that same displayed order (3D only).
    Returns (verdict, 2x2 map) where row 0 is where data +y goes on screen and
    row 1 where data +x goes, as (down, right) components.
    """
    displayed = [int(a) for a in displayed]
    if ndisplay == 2:
        if set(displayed) != {1, 2}:
            return "z on screen (side view): the (y, x) face is not shown", None
        sign = {0: 1 if orientation2d[0] == "down" else -1,
                1: 1 if orientation2d[1] == "right" else -1}
        P = np.zeros((2, 2))
        for row, axis in enumerate((1, 2)):
            slot = displayed.index(axis)
            P[row, slot] = sign[slot]
    else:
        v = np.asarray(view_direction, dtype=float)
        u = np.asarray(up_direction, dtype=float)
        down, right = -u, np.cross(v, u)
        jz, jy, jx = displayed.index(0), displayed.index(1), displayed.index(2)
        if abs(v[jz]) < 0.7:
            return "z on screen (camera looks from the side): the (y, x) face is edge-on", None
        P = np.asarray([[down[jy], right[jy]], [down[jx], right[jx]]])
    scores = {k: float(np.sum(np.asarray(m) * P)) for k, m in _SYMMETRIES.items()}
    best = max(scores, key=scores.get)
    return best, P


def view_state(viewer):
    d, c = viewer.dims, viewer.camera
    verdict, _ = classify_view(d.displayed, d.ndisplay, tuple(str(o) for o in c.orientation2d),
                               c.view_direction, c.up_direction)
    labels = [d.axis_labels[i] for i in d.displayed]
    return (f"(y, x) face shown as: {verdict} | displayed={labels} ndisplay={d.ndisplay} "
            f"order={tuple(int(i) for i in d.order)} angles={tuple(round(float(a), 1) for a in c.angles)}")


def install():
    """Wrap napari.Viewer (as castalign.gui uses it) for the rest of the session."""
    import castalign.gui as ca_gui
    napari = ca_gui.napari
    base = getattr(napari, "_unpatched_Viewer", napari.Viewer)
    napari._unpatched_Viewer = base

    class _StateViewer(base):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("axis_labels", ("z", "y", "x"))
            super().__init__(*args, **kwargs)
            self.axes.visible = True
            self.axes.labels = True
            self._last_verdict = None
            self.bind_key("F9", lambda v: print(view_state(v)), overwrite=True)
            for ev in (self.dims.events.order, self.dims.events.ndisplay, self.camera.events.angles):
                ev.connect(self._report_if_changed)
            self._report_if_changed()

        def _report_if_changed(self, event=None):
            state = view_state(self)
            verdict = state.split(" | ")[0]
            if verdict != self._last_verdict:
                self._last_verdict = verdict
                print(f"[napari view] {state}")

    napari.Viewer = _StateViewer
    print("napari_view_state installed: castalign windows will report their view state (F9 to print).")


def uninstall():
    import castalign.gui as ca_gui
    napari = ca_gui.napari
    if hasattr(napari, "_unpatched_Viewer"):
        napari.Viewer = napari._unpatched_Viewer
        print("napari_view_state removed.")
