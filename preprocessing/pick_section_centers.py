#!/usr/bin/env python3
"""
Click a centre on every BARseq section, for stack_sections.py to stack them on.

Walks each section's downsampled DAPI in slice-number order. Click the centre
and it moves on. DAPI because it shows the anatomy; every other image of the
same source sits on the same grid, so one click serves them all.

--source slice (default)  the whole section, stitch_slices.py --downsample:
                          HYB_slice_stitched_tif_downsampled_micronwise/slice{N}_DAPI.tif
                          -> <ANALYSIS_ROOT>/section_centers_slice.json
--source subslice         the marker-defined region, pipeline step 3:
                          HYB_subslice_stitched_tif_downsampled_micronwise/slice{N}_subslice_DAPI.tif
                          -> <ANALYSIS_ROOT>/section_centers.json

The two are different grids, so each has its own record. It NEVER writes an
image; the record is saved after every click, so an interrupted pass resumes at
the first section without a centre.

Ghost: the previous centred section is drawn in magenta, following the cursor
with ITS centre under the pointer. Move until the anatomy lines up, then click;
this keeps "centre" the same place from section to section. g toggles it.

Keys:
    click   record the centre and advance
    Enter   keep this section's recorded centre and advance
    s       skip (no centre recorded) and advance
    b       back one section
    g       ghost on / off
    q       stop; everything clicked is already saved

Usage:
    python preprocessing/pick_section_centers.py                 # whole slices; resume where it stopped
    python preprocessing/pick_section_centers.py --source subslice
    python preprocessing/pick_section_centers.py --slices 22 24  # only these
    python preprocessing/pick_section_centers.py --redo          # walk every section again
    python preprocessing/pick_section_centers.py --show          # print what is recorded
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Sets the interactive backend before pyplot loads.
from assign_orientation import by_slice_number, load_display
import matplotlib.pyplot as plt

import section_centers

GHOST_MAX_PX = 1500     # ghost drawn at a stride so its long side is at most this


MADE_BY = {"slice": "stitch_slices.py --downsample --no-scale-bars",
           "subslice": "the preprocessing pipeline through step 3"}


def find_sections(dapi_dir, slice_ids=None, source="subslice"):
    """``[(slice_no, path)]`` for every downsampled DAPI of `source`, in slice order."""
    template = section_centers.RAW_TIF_NAMES[source]
    glob = template.format(n="*", channel="DAPI")
    # Anchored: as a glob, slice*_DAPI.tif also matches slice22_subslice_DAPI.tif.
    pattern = re.compile("^" + re.escape(template.format(n="@", channel="DAPI"))
                         .replace("@", r"(\d+)") + "$")
    sections = []
    for p in sorted(Path(dapi_dir).glob(glob)):
        m = pattern.match(p.name)
        if m and (slice_ids is None or int(m.group(1)) in slice_ids):
            sections.append((int(m.group(1)), p))
    sections.sort()
    if not sections:
        raise SystemExit(f"No {glob} under {dapi_dir}"
                         + (f" for slices {sorted(slice_ids)}" if slice_ids else "")
                         + f". Run {MADE_BY[source]} first.")
    if slice_ids:
        missing = sorted(set(slice_ids) - {n for n, _ in sections})
        if missing:
            raise SystemExit(f"No DAPI for slice(s) {missing} under {dapi_dir}")
    return sections


class CenterPicker:
    """One click per section; the record is saved after each."""

    def __init__(self, sections, record, out_path, start=0, ghost=True, source="subslice"):
        self.sections = sections
        self.source = source
        self.record = record
        self.out_path = out_path
        self.index = start
        self.ghost_on = ghost
        self.ghost = None
        self.ghost_center = None

        # matplotlib's single-key shortcuts (s = save figure, q = quit, g = grid)
        # would fire underneath these keys.
        for key in list(plt.rcParams):
            if key.startswith("keymap."):
                plt.rcParams[key] = []

        self.fig, self.ax = plt.subplots(figsize=(10, 9))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.load_current()

    @property
    def slice_no(self):
        return self.sections[self.index][0]

    @property
    def path(self):
        return self.sections[self.index][1]

    def recorded(self, slice_no):
        return self.record["sections"].get(str(slice_no))

    def previous_centred(self):
        """(slice_no, path, entry) of the nearest earlier section with a centre."""
        for i in range(self.index - 1, -1, -1):
            n, p = self.sections[i]
            entry = self.recorded(n)
            if entry:
                return n, p, entry
        return None

    def load_current(self):
        display, _ = load_display(self.path)
        self.shape = display.shape
        self.ax.clear()
        self.ghost = None
        self.ax.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        self.ax.set_autoscale_on(False)
        self.ax.set_xlabel(f"+x ->        {self.path.name}")
        self.ax.set_ylabel("+y  (downward)")

        entry = self.recorded(self.slice_no)
        if entry:
            self.ax.plot(entry["x"], entry["y"], "+", color="cyan", ms=28, mew=2.5)

        prev = self.previous_centred() if self.ghost_on else None
        if prev:
            self.make_ghost(prev)
        self.title()

    def make_ghost(self, prev):
        """Magenta copy of the previous centred section, strided to GHOST_MAX_PX."""
        _, path, entry = prev
        img, _ = load_display(path)
        k = max(1, int(np.ceil(max(img.shape) / GHOST_MAX_PX)))
        small = img[::k, ::k]
        rgba = np.zeros(small.shape + (4,), np.float32)
        rgba[..., 0] = 1.0
        rgba[..., 2] = 1.0
        rgba[..., 3] = 0.55 * small
        self.ghost_hw = img.shape
        self.ghost_center = (entry["y"], entry["x"])
        self.ghost = self.ax.imshow(rgba, interpolation="nearest",
                                    extent=self.ghost_extent(*self.ghost_center),
                                    visible=False, zorder=3)

    def ghost_extent(self, cy, cx):
        """Extent placing the ghost's centre at canvas (cy, cx)."""
        gy, gx = self.ghost_center
        h, w = self.ghost_hw
        left = cx - gx - 0.5
        top = cy - gy - 0.5
        return (left, left + w, top + h, top)

    def title(self):
        entry = self.recorded(self.slice_no)
        done = sum(1 for n, _ in self.sections if self.recorded(n))
        state = (f"recorded ({entry['y']:.0f}, {entry['x']:.0f}) — Enter keeps it"
                 if entry else "no centre yet")
        ghost = ("ghost: previous section follows the cursor" if self.ghost is not None
                 else "ghost off" if not self.ghost_on else "no earlier centred section")
        self.ax.set_title(
            f"slice {self.slice_no}   [{self.index + 1}/{len(self.sections)}]   "
            f"{done} centred   —   {state}\n"
            f"click = centre   s = skip   b = back   g = ghost   q = stop      ({ghost})",
            fontsize=11)
        self.fig.canvas.draw_idle()

    # -- events -------------------------------------------------------------
    def on_move(self, event):
        if self.ghost is None:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            if self.ghost.get_visible():
                self.ghost.set_visible(False)
                self.fig.canvas.draw_idle()
            return
        self.ghost.set_extent(self.ghost_extent(event.ydata, event.xdata))
        self.ghost.set_visible(True)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.button != 1:
            return
        y, x = float(event.ydata), float(event.xdata)
        section_centers.set_center(self.record, self.slice_no, y, x,
                                   self.shape, self.path.name)
        self.record["image"] = "DAPI"
        self.record["source"] = self.source
        section_centers.save(self.out_path, self.record)
        print(f"  slice {self.slice_no}: centre (y {y:.1f}, x {x:.1f})")
        self.advance()

    def on_key(self, event):
        key = (event.key or "").lower()
        if key == "q":
            print("  stopping — every click is already saved")
            plt.close(self.fig)
        elif key in ("enter", "return"):
            if self.recorded(self.slice_no):
                self.advance()
            else:
                print(f"  slice {self.slice_no} has no centre — click one, or s to skip")
        elif key == "s":
            print(f"  slice {self.slice_no}: skipped")
            self.advance()
        elif key == "b":
            if self.index == 0:
                print("  already at the first section")
            else:
                self.index -= 1
                self.load_current()
        elif key == "g":
            self.ghost_on = not self.ghost_on
            self.load_current()

    def advance(self):
        if self.index + 1 >= len(self.sections):
            print("  last section done")
            plt.close(self.fig)
            return
        self.index += 1
        self.load_current()

    def run(self):
        plt.show()


def report(record, sections=None):
    entries = record["sections"]
    print(f"{len(entries)} section(s) centred"
          + (f" (source: {section_centers.record_source(record)}, "
             f"image: {record.get('image')})" if entries else ""))
    for k in sorted(entries, key=int):
        e = entries[k]
        print(f"  slice {int(k):>3}   y {e['y']:>8.1f}   x {e['x']:>8.1f}   "
              f"grid {e['shape'][0]}x{e['shape'][1]}")
    if sections is not None:
        missing = [n for n, _ in sections if str(n) not in entries]
        if missing:
            print(f"no centre: {missing}")


def main():
    ap = argparse.ArgumentParser(
        description="Click a centre on every BARseq section (writes section_centers.json)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--source", choices=section_centers.SOURCES, default="slice",
                    help="whole slices or step 3 subslices (default: slice)")
    ap.add_argument("--slices", type=int, nargs="+", default=None, metavar="N",
                    help="only these sections")
    ap.add_argument("--redo", action="store_true",
                    help="start at the first section even if it has a centre")
    ap.add_argument("--no-ghost", action="store_true", help="start with the ghost off")
    ap.add_argument("--centers", default=None,
                    help="record path (default: <ANALYSIS_ROOT>/section_centers_slice.json, "
                         "or section_centers.json for --source subslice)")
    ap.add_argument("--show", action="store_true", help="print the record and exit")
    args = ap.parse_args()

    out_path = section_centers.centers_path(args.centers, args.source)
    record = section_centers.load(out_path)
    if args.show:
        print(f"record: {out_path}")
        report(record)
        return
    if record["sections"] and section_centers.record_source(record) != args.source:
        raise SystemExit(f"{out_path} holds {section_centers.record_source(record)} centres; "
                         f"this run is --source {args.source}. Pass a different --centers.")

    from preprocessing_config import HYB_DOWNSAMPLED_DIR, HYB_SLICE_DOWNSAMPLED_DIR
    dapi_dir = HYB_SLICE_DOWNSAMPLED_DIR if args.source == "slice" else HYB_DOWNSAMPLED_DIR
    sections = find_sections(dapi_dir, set(args.slices) if args.slices else None,
                             args.source)

    if args.redo:
        start = 0
    else:
        todo = [i for i, (n, _) in enumerate(sections) if str(n) not in record["sections"]]
        if not todo:
            print(f"Every section already has a centre ({out_path}).")
            print("Pass --redo to walk them again, or --slices N --redo to redo some.")
            report(record, sections)
            return
        start = todo[0]

    print(f"record: {out_path}")
    print(f"{len(sections)} section(s); starting at slice {sections[start][0]}")
    CenterPicker(sections, record, out_path, start=start,
                 ghost=not args.no_ghost, source=args.source).run()
    print()
    report(record, sections)


if __name__ == "__main__":
    main()
