"""Per-marker readout facts, shared by every script that paints or counts one.

Lives at the project root next to ``scope_profiles.py`` and ``analysis_paths.py``,
and imports nothing beyond the stdlib — the preprocessing steps, the probes that
run standalone against an arbitrary ``filt_neurons.mat``, and any venv can all
import it without pulling in numpy or matplotlib.

One entry per marker, keyed by the name its callers take on ``--marker``::

    from marker_profiles import MARKERS, get_marker

    profile = get_marker("gcamp")        # raises if its ceiling is unset
    column  = profile["column"]          # 111

Fields:

``label``        Display name, and the token that names output folders and files.
``column``       0-indexed expmat column. **Index-only**: this panel labels its
                 readout slots with stale gene names, so ``gene_name`` is blank
                 and ``utilities.mat_io.resolve_marker_column`` falls through to
                 the index. Never resolve a marker by name here.
``gene_name``    Blank for every marker, for the reason above. The field exists
                 so a dataset whose labels ARE trustworthy can fill it in.
``raw_channel``  Suffix of the stitched channel TIF ``stitch_subslices.py``
                 writes (``slice{N}_subslice_MSCARLET.tif``), and the vocabulary
                 ``local_config.SUBSLICE_RAW_CHANNELS`` already speaks.
``ramp``         Three anchor colours, dark → bright, as 0-1 RGB tuples. Plain
                 tuples so this module stays stdlib-only; the caller builds the
                 colormap. Each ramp is a HUE ramp, not a single-channel one:
                 counts are integers, so a 1-to-15 domain has 15 levels, and on a
                 one-channel ramp those sit 12.75 uint8 apart and read as flat.
                 **The darkest anchor must clear the grey cellmask field at uint8
                 32** or a low-count cell is invisible against its background.
``floor``        Default draw cutoff: below it a cell is not painted.
``ceiling``      The count at which colour saturates. The ramp domain is
                 ``[1, ceiling]`` and the cutoff is NOT in it, so raising the
                 cutoff deletes cells without re-shading the survivors.

``floor`` and ``ceiling`` are per marker AND per brain, the same rule as
``QC_MIN_READS`` / ``QC_MIN_GENES``. A blank one raises rather than borrowing the
other marker's — the ceiling is the one bound that sets count → colour, so a
borrowed one silently renders a domain nobody measured. Measure a new brain with::

    python preprocessing/count_rolonies_per_slice.py --marker gcamp --distribution

Output directories are deliberately NOT here: they need ``OUTPUT_ROOT`` and live
in ``preprocessing/preprocessing_config.py`` as ``MSCARLET_CELLMASK_DIR`` /
``GCAMP_CELLMASK_DIR``.
"""

MARKERS = {
    "mscarlet": {
        "label": "mScarlet",
        "column": 113,          # Python 0-indexed (MATLAB 114)
        "gene_name": "",
        "raw_channel": "MSCARLET",
        # dark red -> orange -> yellow; the floor at uint8 (115, 0, 0).
        # check_rolony_cutoff.py's anchors, so a count renders the same in both
        # tools whenever that tool's --saturate-at equals this ceiling.
        "ramp": [(0.45, 0.0, 0.0), (1.0, 0.35, 0.0), (1.0, 0.95, 0.25)],
        "floor": 5,             # BY95
        "ceiling": 15,          # BY95
    },
    "gcamp": {
        "label": "GCaMP",
        "column": 111,          # Python 0-indexed (MATLAB 112)
        "gene_name": "",
        "raw_channel": "GCAMP",
        # dark green -> green -> pale yellow-green; the floor at uint8 (0, 107, 26)
        "ramp": [(0.0, 0.42, 0.10), (0.15, 0.85, 0.20), (0.80, 1.0, 0.40)],
        # BY95, measured 2026-09-05 with the census above. Of 88,782 QC-passing
        # GCaMP+ cells: 39,594 at >=2, 17,298 at >=3, 4,521 at >=5, 375 above 10,
        # max 57. 60% of QC-passing cells carry >=1 rolony, so 1 and 2 are not
        # worth painting; 3 keeps the 12,777 cells a cutoff of 5 drops. The cap
        # is 10 because the tail past it is 0.42% of the population -- at
        # mScarlet's 15 the drawn cells bunch in the bottom third of the ramp.
        "floor": 3,
        "ceiling": 10,
    },
}


def get_marker(name, min_rolonies=None):
    """The MARKERS entry for `name`, with its per-brain numbers checked.

    An unset ceiling is a hard error, never a fallback to the other marker's:
    the ceiling is the one bound that sets count -> colour, so a borrowed one
    silently renders a domain nobody measured. An unset floor is an error too
    unless --min-rolonies supplied it on this run.

    Returns a copy, so a caller cannot edit the table for everyone else.
    """
    try:
        marker = dict(MARKERS[name])
    except KeyError:
        raise ValueError(
            f"--marker {name}: not one of {sorted(MARKERS)}") from None

    if marker["ceiling"] is None:
        raise ValueError(
            f"{marker['label']} has no ramp ceiling set in MARKERS "
            f"({__file__}).\n"
            f"Measure it for this brain, then write it in:\n"
            f"    python preprocessing/count_rolonies_per_slice.py "
            f"--marker {name} --distribution")

    if min_rolonies is None and marker["floor"] is None:
        raise ValueError(
            f"{marker['label']} has no draw cutoff set in MARKERS "
            f"({__file__}).\n"
            f"Pass --min-rolonies N for this run, or measure it once and write "
            f"it in:\n"
            f"    python preprocessing/count_rolonies_per_slice.py "
            f"--marker {name} --distribution")

    return marker


def marker_names():
    """Marker keys, sorted — for argparse `choices`."""
    return sorted(MARKERS)
