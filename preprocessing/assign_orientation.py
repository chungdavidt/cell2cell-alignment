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

Usage:
    python preprocessing/assign_orientation.py --modality barseq_subslice --image <path>
    python preprocessing/assign_orientation.py --modality barseq_subslice --slice 22
    python preprocessing/assign_orientation.py --modality invivo_ref     # image from local_config
    python preprocessing/assign_orientation.py --single-plane ...        # no section series
    python preprocessing/assign_orientation.py --show                    # print what is recorded

The four questions, over one displayed image (3D input is max-projected along z
for display only):

    1. click the ANTERIOR side      -> posterior is the opposite edge
    2. click the MEDIAL side        -> only the two perpendicular edges accept
    3. press L / R                  -> which hemisphere
    4. press D / V                  -> is the first section dorsal-most or ventral-most

then Enter to write, any other key to start the pass over. Opposites are implied
by construction, so a self-contradictory answer cannot be entered.
"""

import argparse
import os
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
KNOWN_MODALITIES = ('barseq_subslice', 'invivo_ref', 'ex_vivo_block')

CONFIRM_KEYS = ('enter', 'return')
DISPLAY_PERCENTILES = (1.0, 99.7)


# ---------------------------------------------------------------------------
# Image loading (read-only, display only)
# ---------------------------------------------------------------------------

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

    if modality == 'invivo_ref':
        path = getattr(local_config, 'INVIVO_PATH_RED', '')
    elif modality == 'ex_vivo_block':
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
        # poor thing to identify anterior on. Order: DAPI, then ALIGN, then the
        # step 4 overlay.
        from preprocessing_config import HYB_DOWNSAMPLED_DIR
        candidates = [
            sorted(Path(HYB_DOWNSAMPLED_DIR).glob(f"{stem}_subslice_DAPI.tif")),
            sorted(subslice_dir.glob(f"{stem}_subslice_ALIGN.tif")),
            sorted(subslice_dir.glob(f"{stem}_subslice_mScarlet_cellmask.tif")),
        ]
        for files in candidates:
            if files:
                return files[0]
        raise FileNotFoundError(
            f"No {stem}_subslice_DAPI.tif under\n"
            f"  {HYB_DOWNSAMPLED_DIR}\n"
            f"and no {stem}_subslice_ALIGN.tif or {stem}_subslice_mScarlet_cellmask.tif "
            f"under\n"
            f"  {subslice_dir}\n"
            f"Run the preprocessing pipeline first, or pass --image explicitly."
        )
    else:
        return None

    return Path(path) if path else None


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
    """Sequential prompts over one displayed image. Returns the raw answers."""

    def __init__(self, display, modality, image_path, single_plane=False,
                 n_planes=1):
        self.display = display
        self.modality = modality
        self.image_path = image_path
        self.single_plane = single_plane
        # A volume's "first section" is its first imaged plane; say so.
        self.n_planes = n_planes
        self.height, self.width = display.shape
        self.answers = {}
        self.labels = []
        self.result = None

        # Matplotlib's default single-key shortcuts (s = save figure, q = quit,
        # l = log scale) would fire underneath the answers.
        for key in list(plt.rcParams):
            if key.startswith('keymap.'):
                plt.rcParams[key] = []

        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.ax.imshow(self.display, cmap='gray', vmin=0, vmax=1,
                       interpolation='nearest')
        self.ax.set_xlabel(f"+x ->        {Path(image_path).name}")
        self.ax.set_ylabel("+y  (downward)")
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.restart()

    # -- state ------------------------------------------------------------
    @property
    def stage(self):
        for name in ('anterior_edge', 'medial_edge', 'hemisphere', 'first_section'):
            if name == 'first_section' and self.single_plane:
                continue
            if name not in self.answers:
                return name
        return 'confirm'

    def restart(self):
        self.answers = {}
        self.clear_labels()
        self.prompt()

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
        self.ax.set_title(f"{self.modality}\n{text}", fontsize=12)
        self.fig.canvas.draw_idle()

    # -- events -----------------------------------------------------------
    def on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        stage = self.stage
        if stage == 'anterior_edge':
            allowed = orientation.EDGES
        elif stage == 'medial_edge':
            allowed = orientation.perpendicular_edges(self.answers['anterior_edge'])
        else:
            return
        edge = nearest_edge(event.xdata, event.ydata, self.width, self.height, allowed)
        self.answers[stage] = edge
        print(f"  {stage}: {edge}")
        self.prompt()

    def on_key(self, event):
        key = (event.key or '').lower()
        stage = self.stage

        if stage == 'confirm':
            if key in CONFIRM_KEYS:
                self.result = dict(self.answers)
                plt.close(self.fig)
            else:
                print("  restarting")
                self.restart()
            return

        if stage == 'hemisphere' and key in ('l', 'r'):
            self.answers['hemisphere'] = 'left' if key == 'l' else 'right'
            print(f"  hemisphere: {self.answers['hemisphere']}")
            self.prompt()
        elif stage == 'first_section' and key in ('d', 'v'):
            self.answers['first_section'] = 'dorsal' if key == 'd' else 'ventral'
            print(f"  first_section: {self.answers['first_section']}-most")
            self.prompt()

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
        a = self.answers
        code = orientation.derive_code(
            a['anterior_edge'], a['medial_edge'], a['hemisphere'],
            a.get('first_section'),
        )
        posterior_edge = orientation.OPPOSITE_EDGE[a['anterior_edge']]
        lateral_edge = orientation.OPPOSITE_EDGE[a['medial_edge']]
        medial_letter = 'R' if a['hemisphere'] == 'left' else 'L'

        self.clear_labels()
        self.edge_label(a['anterior_edge'], "ANTERIOR", 'yellow')
        self.edge_label(posterior_edge, "POSTERIOR", 'yellow')
        self.edge_label(a['medial_edge'], f"MEDIAL ({medial_letter})", 'cyan')
        self.edge_label(lateral_edge,
                        f"LATERAL ({orientation.opposite(medial_letter)})", 'cyan')

        if self.single_plane:
            through = ("DORSAL / VENTRAL: single plane, no series\n"
                       f"+z (into the stack) = {orientation.LETTER_NAME[code[0]]}"
                       " — derived, not answered")
        else:
            first = ("z = 0" if self.n_planes > 1 else "first section")
            through = (f"DORSAL / VENTRAL: {first} is {a['first_section']}-most\n"
                       f"+z (into the stack) = {orientation.LETTER_NAME[code[0]]}")
        self.labels.append(self.ax.text(
            0.5, 0.5, through, transform=self.ax.transAxes,
            ha='center', va='center', color='white', fontsize=11,
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=5),
        ))

        self.ax.set_title(
            f"{self.modality}   ->   {code}   ({orientation.describe(code)})\n"
            f"Enter = record,  any other key = start over",
            fontsize=12,
        )
        self.fig.canvas.draw_idle()

        print()
        print(f"  code: {code}   {orientation.describe(code)}")
        print(f"  Enter to record, any other key to start over.")

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
    report_handedness(orientation.codes(path))
    return record


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

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--modality', default=None,
                   help=f"Graph node base name. Known: {', '.join(KNOWN_MODALITIES)}")
    p.add_argument('--image', default=None,
                   help='Image to assign on (default: the modality path in local_config.py)')
    p.add_argument('--slice', type=int, default=None,
                   help='For barseq_subslice: which section to display '
                        '(default: the lowest-numbered one). Every section shares '
                        'the frame, so this only changes what you look at.')
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

    image_path = (Path(args.image) if args.image
                  else default_image(args.modality, args.slice))
    if image_path is None:
        p.error(f"No --image given and local_config.py has no path for "
                f"{args.modality!r}.")
    if not image_path.exists():
        p.error(f"Image not found: {image_path}")

    print(f"modality: {args.modality}")
    print(f"image:    {image_path}")
    display, n_planes = load_display(image_path)
    print(f"  display: {display.shape[1]} x {display.shape[0]} (x, y), "
          f"contrast stretched to the {DISPLAY_PERCENTILES[0]}-"
          f"{DISPLAY_PERCENTILES[1]} percentile range")
    print()

    answers = OrientationPicker(display, args.modality, image_path,
                                single_plane=args.single_plane,
                                n_planes=n_planes).run()
    if answers is None:
        print("Closed without confirming — nothing written.")
        return

    entry = orientation.make_entry(
        answers['anterior_edge'], answers['medial_edge'], answers['hemisphere'],
        answers.get('first_section'), image=image_path,
    )
    subject = args.subject
    if subject is None:
        from analysis_paths import subject_name
        subject = subject_name()

    orientation.save(out_path, args.modality, entry, subject=subject)
    print()
    print(f"Recorded {args.modality} = {entry['code']} in {out_path}")
    print("No image was written.")
    report_handedness(orientation.codes(out_path))


if __name__ == '__main__':
    main()
