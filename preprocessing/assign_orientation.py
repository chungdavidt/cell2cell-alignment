#!/usr/bin/env python3
"""
Assign anatomical directions to a modality by clicking them on its image.

Orientation is not recoverable from image content — nothing in a BARseq section
or a 2P stack says which edge is anterior — so a human assignment step is the
only source of the information. This tool asks for it and records the answer.

It NEVER writes an image. No transposing, no flipping, no re-saving of any TIFF,
not behind a flag: the only file it writes is <ANALYSIS_ROOT>/orientation.json.
Applying a reindex is a separate future script that emits new derived copies.

What the code describes is the FRAME, not the pixels. Every preprocessing output
for a section shares one canvas, one origin and one axis order, so DAPI, the step
4 overlay and every generate_alignment_tif.py cutoff folder all yield the same
answer. Choosing a different --min-rolonies therefore never invalidates an
assignment; only a geometry change -- a flip, a transpose, an axis reorder --
would, and nothing in preprocessing does that.

So assign on whatever is easiest to read. The default is the downsampled DAPI for
that reason: a marker-only image at a high cutoff can be too sparse to find
anterior on.

One code for a whole section series assumes every section was mounted the same
way up, and that is an assumption, not a fact: a section mounted face-down is a
mirror, and no rotation or translation undoes a mirror. So barseq_subslice walks
the series BY DEFAULT, answering every section and recording a code per section
under `sections` in the same JSON; the modality-level code becomes the majority.
Walking it is also how you find out whether the assumption held for a brain.

invivo and block_stack are one acquisition with one frame, so they are
assigned once, from a single image. That single-image form is available for
barseq_subslice too, behind an explicit --image, and it records a modality-level
code that REPLACES any per-section records — which is why it asks for --replace
before overwriting them.

Usage:
    python preprocessing/assign_orientation.py --modality barseq_subslice           # every section
    python preprocessing/assign_orientation.py --modality barseq_subslice --slice 22
    python preprocessing/assign_orientation.py --modality barseq_subslice --slices 4 9 22
    python preprocessing/assign_orientation.py --modality invivo     # image from local_config
    python preprocessing/assign_orientation.py --modality barseq_subslice --image <path> --replace
    python preprocessing/assign_orientation.py --single-plane ...        # no section series
    python preprocessing/assign_orientation.py --show                    # print what is recorded

The four questions, over one displayed image (3D input is max-projected along z
for display only):

    1. click the ANTERIOR side      -> posterior is the opposite edge
    2. click the MEDIAL side        -> only the two perpendicular edges accept
    3. press L / R                  -> which hemisphere
    4. press D / V                  -> is the first section dorsal-most or ventral-most

Each label is drawn as its answer is given, so a misclick is visible when it is
made rather than three answers later; x restarts the section at any point.
Medial carries no R/L letter until the hemisphere is answered, because that is
the answer which resolves it.

then Enter to write, any other key to start the pass over. Opposites are implied
by construction, so a self-contradictory answer cannot be entered.

Walking the series runs the same four questions per section, except the
dorsal/ventral one: that describes the CUTTING ORDER, not an individual
section, so it is asked once and applies to the series. Handedness still varies
section to section, because the in-plane answers do -- which is what makes a
face-down section visible. Navigation is Enter to record and advance, x to redo
the current section (from any stage, not only from confirm), b to step back, q
to stop and keep what is recorded. Each
section is written as it is confirmed, so an interrupted pass resumes where it
stopped rather than starting over. The dorsal/ventral answer is asked on the
first section, so redoing that section is what re-asks it.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
# A real interactive backend, unlike the threshold viewer's forced 'Agg' — this
# tool is clicks and keypresses. MPLBACKEND, if set, wins.
if not os.environ.get('MPLBACKEND'):
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import orientation
from utilities.image_io import imread_tiff

# Modality keys match graph node base names, so the graph builder looks them up
# directly. All BARseq subslices in a dataset share one orientation, so a single
# subslice image covers them all.
KNOWN_MODALITIES = ('barseq_subslice', 'invivo', 'block_stack')

# The one modality that is a SERIES of independently mounted sections rather
# than a single acquisition, so it is the one walked section by section.
SERIES_MODALITY = 'barseq_subslice'

CONFIRM_KEYS = ('enter', 'return')
DISPLAY_PERCENTILES = (1.0, 99.7)


# ---------------------------------------------------------------------------
# Image loading (read-only, display only)
# ---------------------------------------------------------------------------

def by_slice_number(paths):
    """Section files in slice-NUMBER order.

    ``sorted()`` on the filenames is lexicographic, which puts slice10 ahead of
    slice2 and made the single-image default the wrong section.
    """
    def key(path):
        match = re.match(r"slice(\d+)_subslice", path.name)
        return (0, int(match.group(1))) if match else (1, 0)
    return sorted(paths, key=key)


def default_image(modality: str, slice_id=None):
    """The image a modality is normally assigned on, from local_config.py.

    local_config, not preprocessing_config: the latter resolves SCOPE at import
    and raises when it is blank, and orientation has nothing to do with pixel
    size.
    """
    try:
        import local_config
    except ImportError:
        raise SystemExit(
            "local_config.py not found, so there is no default image for "
            f"{modality!r}. Pass --image explicitly."
        )

    if modality == 'invivo':
        path = getattr(local_config, 'INVIVO_PATH_RED', '')
    elif modality == 'block_stack':
        path = getattr(local_config, 'BLOCK_STACK_PATH_RED', '')
    elif modality == 'barseq_subslice':
        from analysis_paths import resolve_subslice_dir
        subslice_dir = resolve_subslice_dir()
        if subslice_dir is None:
            return None
        stem = f"slice{slice_id}" if slice_id is not None else "slice*"
        # What the letters describe is the FRAME, not the pixels: every step 3
        # output for a section shares one canvas, one origin and one axis
        # order, and generate_alignment_tif.py draws into that same frame. So
        # any of them yields the same code, and changing --min-rolonies -- which
        # changes only which cells are drawn -- cannot invalidate an assignment.
        #
        # That frees the default to be the most LEGIBLE image rather than the one
        # the graph ingests. DAPI shows the anatomy you are clicking on; a
        # marker-only ALIGN tif at a high cutoff can be nearly empty and is a
        # poor thing to identify anterior on. Order: DAPI, then ALIGN.
        from preprocessing_config import HYB_DOWNSAMPLED_DIR
        candidates = [
            by_slice_number(Path(HYB_DOWNSAMPLED_DIR).glob(f"{stem}_subslice_DAPI.tif")),
            by_slice_number(subslice_dir.glob(f"{stem}_subslice_ALIGN.tif")),
        ]
        for files in candidates:
            if files:
                return files[0]
        raise FileNotFoundError(
            f"No {stem}_subslice_DAPI.tif under\n"
            f"  {HYB_DOWNSAMPLED_DIR}\n"
            f"and no {stem}_subslice_ALIGN.tif under\n"
            f"  {subslice_dir}\n"
            f"Run the preprocessing pipeline first, or pass --image explicitly."
        )
    else:
        return None

    return Path(path) if path else None


def discover_sections(slice_ids=None):
    """``[(slice_id, path)]`` for every BARseq section, in slice order.

    Same candidate order as `default_image` — downsampled DAPI first, because
    it is the legible one — but resolved across the whole series instead of one
    file. The first candidate that yields any section wins outright, so the run
    is never a mixture of DAPI for some sections and ALIGN tifs for others.

    `slice_ids` restricts the pass to named sections, for re-doing a handful
    without walking the series again.
    """
    from analysis_paths import resolve_subslice_dir
    from preprocessing_config import HYB_DOWNSAMPLED_DIR

    subslice_dir = resolve_subslice_dir()
    candidates = [(Path(HYB_DOWNSAMPLED_DIR), "_subslice_DAPI.tif")]
    if subslice_dir is not None:
        candidates += [(subslice_dir, "_subslice_ALIGN.tif")]

    for root, suffix in candidates:
        found = {}
        for path in sorted(Path(root).glob(f"slice*{suffix}")):
            match = re.match(r"slice(\d+)_subslice", path.name)
            if match:
                found[int(match.group(1))] = path
        if not found:
            continue
        if slice_ids:
            missing = [s for s in slice_ids if s not in found]
            if missing:
                raise SystemExit(
                    f"No {suffix} for slice(s) {missing} under {root}. "
                    f"Present: {sorted(found)}"
                )
            found = {s: found[s] for s in slice_ids}
        return [(s, found[s]) for s in sorted(found)]

    searched = "\n".join(f"  {root}  ({suffix})" for root, suffix in candidates)
    raise FileNotFoundError(
        f"No BARseq section images found. Searched:\n{searched}\n"
        f"Run the preprocessing pipeline first."
    )


def load_display(path: Path):
    """Read `path` and reduce it to one contrast-stretched 2D image.

    Returns ``(display, n_planes)``. 3D stacks are max-projected along z, and
    `n_planes` is their z count — it decides only how the dorsal/ventral
    question is worded, since "the first section" of a 2P volume is its first
    imaged plane. The array returned is a display copy; the file on disk is
    never touched.
    """
    img = imread_tiff(path)
    n_planes = 1

    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img[..., :3].astype(np.float32).mean(axis=-1)
    elif img.ndim == 3:
        n_planes = img.shape[0]
        print(f"  3D input {img.shape} — max projection along {n_planes} z planes "
              f"for display")
        img = img.max(axis=0)
    elif img.ndim != 2:
        raise ValueError(f"Cannot display a {img.ndim}D array: shape {img.shape}")

    img = img.astype(np.float32)
    lo, hi = np.percentile(img, DISPLAY_PERCENTILES)
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros_like(img), n_planes
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0), n_planes


# ---------------------------------------------------------------------------
# The interaction
# ---------------------------------------------------------------------------

def nearest_edge(x, y, width, height, allowed):
    """Which of `allowed` edges the click at (x, y) is closest to."""
    distance = {
        'top': y / height,
        'bottom': (height - y) / height,
        'left': x / width,
        'right': (width - x) / width,
    }
    return min(allowed, key=lambda edge: distance[edge])


class OrientationPicker:
    """Sequential prompts over one section image or a whole series.

    One image and a series are the same interaction. A series adds navigation,
    a record written per section as it is confirmed, and a proposal carried
    forward from the section before — most sections of a well-mounted brain
    give the same answers, so the ones that do not are what you are looking for.
    The proposal is drawn as labels and still needs an explicit Enter, so it
    speeds up agreeing without letting you agree to something unseen.
    """

    def __init__(self, sections, modality, single_plane=False,
                 propagate=True, existing=None, on_record=None):
        self.sections = list(sections)          # [(slice_id or None, path)]
        self.modality = modality
        self.single_plane = single_plane
        self.propagate = propagate
        self.on_record = on_record
        self.stack = len(self.sections) > 1

        # slice_id -> answers, seeded from what is already recorded so an
        # interrupted pass resumes instead of restarting.
        self.answers = dict(existing or {})
        # The cutting order is a property of the SERIES, not of a section: it is
        # the same answer every time, so it is asked once and reused. Handedness
        # still varies section to section because the in-plane answers do.
        self.first_section = next(
            (a['first_section'] for a in self.answers.values()
             if a.get('first_section')), None)

        self.index = 0
        self.pending = {}
        self.labels = []
        self.result = None
        self.quit_early = False

        # Matplotlib's default single-key shortcuts (s = save figure, q = quit,
        # l = log scale) would fire underneath the answers.
        for key in list(plt.rcParams):
            if key.startswith('keymap.'):
                plt.rcParams[key] = []

        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.load_current()

    # -- sections ---------------------------------------------------------
    @property
    def slice_id(self):
        return self.sections[self.index][0]

    @property
    def path(self):
        return self.sections[self.index][1]

    def previous_answers(self):
        """In-plane answers from the nearest earlier section that has any."""
        for i in range(self.index - 1, -1, -1):
            prior = self.answers.get(self.sections[i][0])
            if prior:
                return prior
        return None

    def load_current(self):
        """Read the current section, reset the canvas, seed its proposal."""
        display, self.n_planes = load_display(self.path)
        self.height, self.width = display.shape
        self.display = display

        self.ax.clear()
        self.labels = []
        self.ax.imshow(display, cmap='gray', vmin=0, vmax=1,
                       interpolation='nearest')
        self.ax.set_xlabel(f"+x ->        {self.path.name}")
        self.ax.set_ylabel("+y  (downward)")

        seed = self.answers.get(self.slice_id)
        self.from_previous = False
        if seed is None and self.propagate:
            seed = self.previous_answers()
            self.from_previous = seed is not None
        self.pending = {k: v for k, v in (seed or {}).items()
                        if k in ('anterior_edge', 'medial_edge', 'hemisphere')}
        # A hand-edited record could name both in-plane answers on one axis,
        # which derive_code rejects. Drop the pair rather than propose it.
        anterior, medial = self.pending.get('anterior_edge'), self.pending.get('medial_edge')
        if anterior and medial and medial not in orientation.perpendicular_edges(anterior):
            self.pending.pop('anterior_edge')
            self.pending.pop('medial_edge')
            self.from_previous = False
        self.prompt()

    # -- state ------------------------------------------------------------
    @property
    def stage(self):
        for name in ('anterior_edge', 'medial_edge', 'hemisphere'):
            if name not in self.pending:
                return name
        if not self.single_plane and self.first_section is None:
            return 'first_section'
        return 'confirm'

    def restart(self):
        """Clear this section's answers and ask again from the top.

        The cutting order is a series-level answer asked on the first section,
        so redoing THERE re-asks it and redoing anywhere else leaves it alone.
        Without that, a wrong dorsal/ventral answer could only be corrected by
        re-running the whole pass.
        """
        self.pending = {}
        self.from_previous = False
        if not self.stack or self.index == 0:
            self.first_section = None
        self.prompt()

    def progress(self):
        if not self.stack:
            return self.modality
        return (f"{self.modality}   [{self.index + 1}/{len(self.sections)}]"
                f"   slice {self.slice_id}")

    def draw_answered(self):
        """Redraw every label the answers so far support.

        Called on each prompt, so a misclick shows up the moment it is made
        rather than three answers later. Medial carries no letter until the
        hemisphere is answered — the hemisphere is what turns medial/lateral
        into R/L — so it is drawn unlettered until then.
        """
        self.clear_labels()
        anterior = self.pending.get('anterior_edge')
        if anterior:
            self.edge_label(anterior, "ANTERIOR", 'yellow')
            self.edge_label(orientation.OPPOSITE_EDGE[anterior], "POSTERIOR", 'yellow')

        medial = self.pending.get('medial_edge')
        if medial:
            hemisphere = self.pending.get('hemisphere')
            if hemisphere:
                letter = 'R' if hemisphere == 'left' else 'L'
                medial_text = f"MEDIAL ({letter})"
                lateral_text = f"LATERAL ({orientation.opposite(letter)})"
            else:
                medial_text, lateral_text = "MEDIAL", "LATERAL"
            self.edge_label(medial, medial_text, 'cyan')
            self.edge_label(orientation.OPPOSITE_EDGE[medial], lateral_text, 'cyan')

    def prompt(self):
        stage = self.stage
        text = {
            'anterior_edge': "Click the ANTERIOR side  (nearest edge wins)",
            'medial_edge': "Click the MEDIAL side  (only the perpendicular edges)",
            'hemisphere': "Which hemisphere?   press  L  or  R",
            'first_section': (
                f"Is z = 0 (the first of {self.n_planes} planes) dorsal-most or "
                f"ventral-most?   press  D  or  V"
                if self.n_planes > 1 else
                "Is the FIRST section dorsal-most or ventral-most?   press  D  or  V"
            ),
        }.get(stage)
        if stage == 'confirm':
            self.show_result()
            return
        self.draw_answered()
        self.ax.set_title(f"{self.progress()}\n{text}", fontsize=12)
        self.fig.canvas.draw_idle()

    # -- events -----------------------------------------------------------
    def on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        stage = self.stage
        if stage == 'anterior_edge':
            allowed = orientation.EDGES
        elif stage == 'medial_edge':
            allowed = orientation.perpendicular_edges(self.pending['anterior_edge'])
        else:
            return
        edge = nearest_edge(event.xdata, event.ydata, self.width, self.height, allowed)
        self.pending[stage] = edge
        print(f"  {stage}: {edge}")
        self.prompt()

    def on_key(self, event):
        key = (event.key or '').lower()
        stage = self.stage

        # x restarts the current section from ANY stage, not only from confirm.
        # With labels drawn as they are answered, a misclick is visible
        # immediately, so there has to be a way to fix it immediately.
        if key == 'x':
            print(f"  redoing {'this section' if self.stack else 'from the top'}")
            self.restart()
            return

        if self.stack:
            if key == 'q':
                print("  stopping — everything recorded so far is written")
                self.quit_early = True
                self.finish()
                return
            if key == 'b':
                self.step_back()
                return

        if stage == 'confirm':
            if key in CONFIRM_KEYS:
                self.record_current()
                self.advance()
            elif not self.stack:
                print("  restarting")
                self.restart()
            return

        if stage == 'hemisphere' and key in ('l', 'r'):
            self.pending['hemisphere'] = 'left' if key == 'l' else 'right'
            print(f"  hemisphere: {self.pending['hemisphere']}")
            self.prompt()
        elif stage == 'first_section' and key in ('d', 'v'):
            self.first_section = 'dorsal' if key == 'd' else 'ventral'
            print(f"  first_section: {self.first_section}-most"
                  f"{' (applies to the whole series)' if self.stack else ''}")
            self.restamp_recorded()
            self.prompt()

    # -- navigation -------------------------------------------------------
    def record_current(self):
        answers = dict(self.pending)
        answers['first_section'] = None if self.single_plane else self.first_section
        answers['image'] = str(self.path)
        self.answers[self.slice_id] = answers
        if self.on_record:
            self.on_record(self.slice_id, answers)

    def restamp_recorded(self):
        """Re-record sections answered under a different cutting order.

        `first_section` sets the z letter of every section in the series, so
        redoing the first section and answering D/V differently would otherwise
        leave the already-recorded sections carrying the old z — a record that
        disagrees with itself, and a `majority_code` split across two codes that
        differ only in a letter nobody re-answered.
        """
        stale = [sid for sid, answers in self.answers.items()
                 if answers.get('first_section') not in (None, self.first_section)]
        if not stale:
            return
        print(f"  cutting order changed — restamping {len(stale)} recorded section(s)")
        for sid in sorted(stale, key=lambda v: (v is None, v)):
            self.answers[sid]['first_section'] = self.first_section
            if self.on_record:
                self.on_record(sid, self.answers[sid])

    def advance(self):
        if self.index + 1 >= len(self.sections):
            self.finish()
            return
        self.index += 1
        self.load_current()

    def step_back(self):
        if self.index == 0:
            print("  already at the first section")
            return
        self.index -= 1
        self.load_current()

    def finish(self):
        self.result = dict(self.answers)
        plt.close(self.fig)

    # -- drawing ----------------------------------------------------------
    def clear_labels(self):
        for artist in self.labels:
            artist.remove()
        self.labels = []

    def edge_label(self, edge, text, color):
        position = {
            'top': (0.5, 0.98, 'center', 'top'),
            'bottom': (0.5, 0.02, 'center', 'bottom'),
            'left': (0.02, 0.5, 'left', 'center'),
            'right': (0.98, 0.5, 'right', 'center'),
        }[edge]
        x, y, ha, va = position
        self.labels.append(self.ax.text(
            x, y, text, transform=self.ax.transAxes, ha=ha, va=va,
            color=color, fontsize=13, fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=3),
        ))

    def show_result(self):
        """Draw all six labels, print the code, wait for an explicit confirm."""
        a = self.pending
        code = orientation.derive_code(
            a['anterior_edge'], a['medial_edge'], a['hemisphere'],
            None if self.single_plane else self.first_section,
        )
        self.draw_answered()

        if self.single_plane:
            through = ("DORSAL / VENTRAL: single plane, no series\n"
                       f"+z (into the stack) = {orientation.LETTER_NAME[code[0]]}"
                       " — derived, not answered")
        else:
            first = ("z = 0" if self.n_planes > 1 else "first section")
            through = (f"DORSAL / VENTRAL: {first} is {self.first_section}-most\n"
                       f"+z (into the stack) = {orientation.LETTER_NAME[code[0]]}")
        self.labels.append(self.ax.text(
            0.5, 0.5, through, transform=self.ax.transAxes,
            ha='center', va='center', color='white', fontsize=11,
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=5),
        ))

        note, colour = self.deviation_note(code)
        if note:
            self.labels.append(self.ax.text(
                0.5, 0.11, note, transform=self.ax.transAxes,
                ha='center', va='center', color=colour, fontsize=11,
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=4),
            ))

        keys = ("Enter = record and advance,  x = redo,  b = back,  q = stop"
                if self.stack else
                "Enter = record,  x or any other key = start over")
        self.ax.set_title(
            f"{self.progress()}   ->   {code}   ({orientation.describe(code)})\n{keys}",
            fontsize=12,
        )
        self.fig.canvas.draw_idle()

        print()
        print(f"  code: {code}   {orientation.describe(code)}")
        if note:
            print(f"  {note}")
        print(f"  {keys}")

    def deviation_note(self, code):
        """Flag a section whose code differs from the one before it.

        A differing handedness is the one that matters: it is a mirror, and no
        alignment undoes a mirror. A differing code at the SAME handedness is a
        90-degree mounting difference, which is a rotation.
        """
        if not self.stack:
            return None, None
        prior = self.previous_answers()
        if not prior or not prior.get('anterior_edge'):
            return None, None
        prior_code = orientation.derive_code(
            prior['anterior_edge'], prior['medial_edge'], prior['hemisphere'],
            None if self.single_plane else self.first_section,
        )
        if code == prior_code:
            return (("proposed from the previous section — confirm or press x"
                     if self.from_previous else
                     f"same as the previous section ({prior_code})"), 'lightgreen')
        if orientation.agree(code, prior_code):
            return (f"DIFFERS from the previous section ({prior_code}) — "
                    f"rotated, same handedness", 'orange')
        return (f"MIRRORED relative to the previous section ({prior_code}) — "
                f"opposite handedness", 'red')

    def run(self):
        plt.show()
        return self.result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def show(path):
    record = orientation.load(path)
    print(f"orientation record: {path}"
          f"{'' if Path(path).exists() else '  (does not exist yet)'}")
    subject = record.get('subject')
    if subject:
        print(f"subject: {subject}")
    modalities = record.get('modalities', {})
    if not modalities:
        print("no modalities assigned")
        return record
    for name, entry in sorted(modalities.items()):
        code = entry.get('code')
        print()
        print(f"  {name}: {code}   {orientation.describe(code)}")
        print(f"    anterior edge {entry.get('anterior_edge')}, "
              f"medial edge {entry.get('medial_edge')}, "
              f"{entry.get('hemisphere')} hemisphere, "
              f"first section {entry.get('first_section')}")
        print(f"    image: {entry.get('image')}")
        print(f"    assigned: {entry.get('assigned')}")
        if entry.get('sections'):
            print(f"    per section: {entry.get('n_agree_with_code')} of "
                  f"{entry.get('n_sections')} sections carry {code}")
            report_sections(orientation.section_codes(path, name), indent=4)
    report_handedness(orientation.codes(path))
    return record


def report_sections(section_map, indent=2):
    """Group a modality's sections by code and name the odd ones out.

    The modality-level code is a majority summary, so this is where a
    differently-mounted section actually becomes visible.
    """
    pad = " " * indent
    if not section_map:
        return
    by_code = {}
    for slice_id, code in section_map.items():
        by_code.setdefault(code, []).append(slice_id)

    if len(by_code) == 1:
        code = next(iter(by_code))
        print(f"{pad}all {len(section_map)} sections agree: {code}")
        return

    print(f"{pad}{len(by_code)} distinct codes across {len(section_map)} sections:")
    for code, ids in sorted(by_code.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        listed = ", ".join(str(i) for i in sorted(ids))
        print(f"{pad}  {code}  ({len(ids):>3}): {listed}")

    groups = orientation.handedness_groups(section_map)
    if groups[1] and groups[-1]:
        smaller = groups[1] if len(groups[1]) <= len(groups[-1]) else groups[-1]
        print(f"{pad}MIRRORED SECTIONS: {', '.join(str(i) for i in smaller)}")
        print(f"{pad}  These were mounted the other way up. A mirror is not a")
        print(f"{pad}  rotation — no alignment undoes it.")
    else:
        print(f"{pad}All sections share a handedness — the differences are")
        print(f"{pad}rotations, which alignment resolves.")


def report_handedness(code_map):
    """Print whether the recorded modalities share a handedness."""
    if len(code_map) < 2:
        return
    print()
    disagreement = orientation.disagreeing_pair(code_map)
    if disagreement is None:
        print("handedness: all recorded modalities agree")
        return
    name_a, name_b = disagreement
    print(f"HANDEDNESS DISAGREEMENT: {name_a} ({code_map[name_a]}) vs "
          f"{name_b} ({code_map[name_b]})")
    print("  One acquisition mirrored the tissue, or one label is wrong. No")
    print("  rotation or translation reconciles a mirror — a flip would map the")
    print("  left hemisphere onto the right. The graph builder refuses to build")
    print("  until these agree.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def resolve_subject(explicit):
    if explicit is not None:
        return explicit
    from analysis_paths import subject_name
    return subject_name()


def assign_one(args, out_path, parser):
    """Assign one image, recorded at the modality level.

    Right for invivo and block_stack, which are one acquisition with one
    frame. For a section series it records one code and asserts every section
    shares it, so it REPLACES per-section records rather than updating them —
    `orientation.save` writes the modality entry whole. That is silent data
    loss, so it is refused unless --replace says it is intended.
    """
    prior = orientation.load(out_path).get('modalities', {}).get(args.modality, {})
    if prior.get('sections') and not args.replace:
        parser.error(
            f"{args.modality} already has per-section records "
            f"({len(prior['sections'])} sections, code {prior.get('code')}), and a "
            f"single-image assignment would replace them with one code for the "
            f"whole series.\n"
            f"  Update a section:   --slice N   (or --slices N M ...)\n"
            f"  See what is on file: --show\n"
            f"  Replace anyway:     --replace"
        )

    image_path = (Path(args.image) if args.image
                  else default_image(args.modality, args.slice))
    if image_path is None:
        parser.error(f"No --image given and local_config.py has no path for "
                     f"{args.modality!r}.")
    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")

    print(f"modality: {args.modality}")
    print(f"image:    {image_path}")
    print()

    result = OrientationPicker([(None, image_path)], args.modality,
                               single_plane=args.single_plane).run()
    if not result:
        print("Closed without confirming — nothing written.")
        return
    answers = result[None]

    entry = orientation.make_entry(
        answers['anterior_edge'], answers['medial_edge'], answers['hemisphere'],
        answers.get('first_section'), image=image_path,
    )
    orientation.save(out_path, args.modality, entry,
                     subject=resolve_subject(args.subject))
    print()
    print(f"Recorded {args.modality} = {entry['code']} in {out_path}")
    print("No image was written.")
    report_handedness(orientation.codes(out_path))


def assign_series(args, out_path, parser):
    """Walk a BARseq section series, recording a code per section.

    One code per modality assumes uniform mounting. This drops that assumption:
    every section is answered on its own, and a section mounted face-down shows
    up as a handedness that disagrees with its neighbours.
    """
    if args.modality != SERIES_MODALITY:
        parser.error(
            f"a section pass walks a BARseq series, but {args.modality!r} is one "
            f"acquisition with one frame — there are no sections to step through. "
            f"Assign it with --image or its local_config default."
        )
    if args.image:
        parser.error(
            "--image names one file, so it records one code for the whole "
            "series. Drop it to walk the sections, or keep it and add --replace."
        )
    if args.single_plane:
        parser.error(
            "--single-plane derives the z letter from the in-plane answers, "
            "choosing whichever one makes the handedness consistent. Every "
            "section would then agree with every other by construction and no "
            "mirrored section could ever be flagged — which is the only reason "
            "to walk the series. Answer the dorsal/ventral question instead."
        )

    # --slice and --slices differ only in how many sections they name; both
    # restrict the same pass and both record per section.
    wanted = args.slices or ([args.slice] if args.slice is not None else None)
    sections = discover_sections(wanted)
    print(f"modality: {args.modality}")
    print(f"sections: {len(sections)}  "
          f"(slice {sections[0][0]} .. {sections[-1][0]})")
    print(f"source:   {sections[0][1].parent}")

    # Anything already on file seeds the pass, so an interrupted run resumes.
    prior = orientation.load(out_path).get('modalities', {}).get(args.modality, {})
    existing = {int(k): v for k, v in (prior.get('sections') or {}).items()
                if str(k).lstrip('-').isdigit()}
    if existing:
        print(f"resuming: {len(existing)} section(s) already recorded")
    if args.fresh:
        print("fresh:    no proposal carried forward; every section answered from scratch")
    print()

    subject = resolve_subject(args.subject)
    recorded = dict(existing)

    def on_record(slice_id, answers):
        recorded[slice_id] = orientation.make_entry(
            answers['anterior_edge'], answers['medial_edge'],
            answers['hemisphere'], answers.get('first_section'),
            image=answers.get('image'),
        )
        # Written as each section is confirmed rather than at the end: a
        # 62-section pass is not one sitting, and q or a closed window should
        # leave the finished sections on file.
        orientation.save_sections(out_path, args.modality, recorded,
                                  subject=subject)
        print(f"  recorded slice {slice_id} = {recorded[slice_id]['code']} "
              f"({len(recorded)} on file)")

    OrientationPicker(sections, args.modality,
                      single_plane=args.single_plane,
                      propagate=not args.fresh,
                      existing=existing,
                      on_record=on_record).run()

    if not recorded:
        print("Nothing confirmed — nothing written.")
        return

    entry = orientation.load(out_path)['modalities'][args.modality]
    print()
    print(f"Recorded {len(recorded)} section(s) in {out_path}")
    print("No image was written.")
    print(f"modality code (majority of sections): {entry['code']}   "
          f"{orientation.describe(entry['code'])}")
    report_sections(orientation.section_codes(out_path, args.modality))
    report_handedness(orientation.codes(out_path))
    print()
    print("NOTE: subslice_graph_builder.py reads the modality-level code only, so")
    print("      every subslice node is still stamped with the majority code. The")
    print("      per-section records above are not yet consumed by anything.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--modality', default=None,
                   help=f"Graph node base name. Known: {', '.join(KNOWN_MODALITIES)}")
    p.add_argument('--image', default=None,
                   help='Assign on this one image and record ONE code for the '
                        'modality (default for invivo and block_stack, which '
                        'are single acquisitions). For barseq_subslice this skips '
                        'the section pass, so it needs --replace to overwrite '
                        'per-section records.')
    p.add_argument('--slice', type=int, default=None,
                   help='Restrict the section pass to this one section, for '
                        're-doing it without walking the series again.')
    p.add_argument('--all-slices', action='store_true',
                   help='Accepted and redundant: barseq_subslice walks every '
                        'section by default.')
    p.add_argument('--slices', type=int, nargs='+', default=None,
                   help='Restrict the section pass to these sections.')
    p.add_argument('--replace', action='store_true',
                   help='Allow a single-image assignment to replace existing '
                        'per-section records with one code for the whole series.')
    p.add_argument('--fresh', action='store_true',
                   help='Do not carry the previous section\'s answers forward as a '
                        'proposal. Slower, but nothing is pre-agreed.')
    p.add_argument('--out', default=None,
                   help='orientation.json path (default: <ANALYSIS_ROOT>/orientation.json)')
    p.add_argument('--single-plane', action='store_true',
                   help='No section series: skip the dorsal/ventral question and derive '
                        'the z letter from the in-plane answers')
    p.add_argument('--subject', default=None,
                   help='Subject label to stamp (default: the ANALYSIS_ROOT folder name)')
    p.add_argument('--show', action='store_true',
                   help='Print what is recorded and exit')
    args = p.parse_args()

    try:
        out_path = orientation.orientation_path(args.out)
    except ValueError as e:
        raise SystemExit(str(e))

    if args.show:
        show(out_path)
        return

    if not args.modality:
        p.error("--modality is required (or use --show)")
    if args.modality not in KNOWN_MODALITIES:
        print(f"NOTE: {args.modality!r} is not one of the modality keys the graph "
              f"builder reads ({', '.join(KNOWN_MODALITIES)}).")
        print("      It will be recorded, but nothing will look it up.")

    # A BARseq series is walked section by section unless --image explicitly
    # names one file: sections are mounted one at a time, so uniform mounting is
    # the thing being measured, never the thing assumed.
    walk = (args.modality == SERIES_MODALITY and not args.image
            and not args.single_plane)
    if walk or args.all_slices or args.slices or (
            args.slice is not None and args.modality == SERIES_MODALITY):
        assign_series(args, out_path, p)
    else:
        assign_one(args, out_path, p)


if __name__ == '__main__':
    main()
