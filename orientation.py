"""Anatomical orientation codes for image volumes, shared by every pipeline.

Lives at the project root, next to ``scope_profiles.py`` and
``analysis_paths.py``, and imports nothing beyond the stdlib: preprocessing runs
before the graph exists and the graph builder runs after, so both sides import
this without pulling in numpy or castalign.

A code is three letters, one per array axis ``(z, y, x)``, each naming the
anatomical direction of INCREASING index — the nibabel/ITK convention::

    "VPR"  ->  +z ventral, +y posterior, +x the animal's right

Letters are absolute: ``R``/``L`` are the animal's right and left, never
"medial"/"lateral". Medial and lateral are hemisphere-relative and are what the
assignment GUI asks about; they are converted to ``R``/``L`` here and never
stored as letters.

Nothing in this module reads or writes an image. It derives codes from clicked
answers, compares two codes, and reads/writes the JSON record —
``<ANALYSIS_ROOT>/orientation.json``.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from analysis_paths import get_analysis_root

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Display convention: x increases rightward, y increases downward.
EDGES = ('top', 'bottom', 'left', 'right')
OPPOSITE_EDGE = {'top': 'bottom', 'bottom': 'top', 'left': 'right', 'right': 'left'}
EDGE_AXIS = {'top': 'y', 'bottom': 'y', 'left': 'x', 'right': 'x'}

HEMISPHERES = ('left', 'right')
FIRST_SECTIONS = ('dorsal', 'ventral')

OPPOSITE_LETTER = {'A': 'P', 'P': 'A', 'R': 'L', 'L': 'R', 'D': 'V', 'V': 'D'}
LETTER_NAME = {
    'A': 'anterior', 'P': 'posterior',
    'R': 'right', 'L': 'left',
    'D': 'dorsal', 'V': 'ventral',
}

# Unit vectors in a right-handed anatomical basis (right, anterior, dorsal).
LETTER_VECTOR = {
    'R': (1, 0, 0), 'L': (-1, 0, 0),
    'A': (0, 1, 0), 'P': (0, -1, 0),
    'D': (0, 0, 1), 'V': (0, 0, -1),
}

# Sign of `handedness()` for a volume labelled without a mirror, given array
# axis order (z, y, x) and a display whose y runs DOWNWARD. It is -1 only
# because of those two conventions, not because -1 is "correct" — never report
# a single volume's handedness as right or wrong. Its only use is comparison:
# two modalities of the same animal must share a sign (see `agree`).
CONSISTENT_HANDEDNESS = -1

ORIENTATION_FILENAME = "orientation.json"


def perpendicular_edges(edge: str) -> tuple:
    """The two edges on the other in-plane axis from `edge`."""
    _check(edge, EDGES, "edge")
    return ('left', 'right') if EDGE_AXIS[edge] == 'y' else ('top', 'bottom')


def opposite(letter: str) -> str:
    """The letter naming the opposite anatomical direction."""
    letter = letter.upper()
    _check(letter, tuple(OPPOSITE_LETTER), "letter")
    return OPPOSITE_LETTER[letter]


def _check(value, allowed, what):
    if value not in allowed:
        raise ValueError(f"{what} must be one of {list(allowed)}, got {value!r}")


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------

def derive_code(anterior_edge: str,
                medial_edge: str,
                hemisphere: str,
                first_section: Optional[str] = None) -> str:
    """Three-letter ``(z, y, x)`` code from the four clicked answers.

    Parameters
    ----------
    anterior_edge : which of the four displayed edges is anterior.
    medial_edge : which edge is medial. Must be perpendicular to
        `anterior_edge` — the two answers name different in-plane axes.
    hemisphere : 'left' or 'right'. Required: it is what turns medial/lateral
        into the absolute letters R/L.
    first_section : 'dorsal' or 'ventral' — is index 0 of the stack (or of the
        section series) the dorsal-most or the ventral-most plane. ``None`` for
        a single plane with no series, in which case the z letter is DERIVED
        from the in-plane answers under `CONSISTENT_HANDEDNESS`. A derived z
        makes `handedness` non-informative for that volume, so `agree` cannot
        catch a mirror against it; answer it whenever a series exists.

    Returns
    -------
    str
        e.g. ``anterior=top, medial=right, hemisphere=left,
        first_section=dorsal`` -> ``"VPR"``.
    """
    _check(anterior_edge, EDGES, "anterior_edge")
    _check(medial_edge, EDGES, "medial_edge")
    _check(hemisphere, HEMISPHERES, "hemisphere")
    if medial_edge not in perpendicular_edges(anterior_edge):
        raise ValueError(
            f"medial_edge {medial_edge!r} is on the same axis as anterior_edge "
            f"{anterior_edge!r}; it must be one of "
            f"{list(perpendicular_edges(anterior_edge))}"
        )

    # Increasing index runs AWAY from the named edge, so the edge's letter is
    # the one that lands on the opposite side.
    letters = {}
    letters[EDGE_AXIS[anterior_edge]] = 'A' if anterior_edge in ('bottom', 'right') else 'P'

    # 'medial'/'lateral' live in their own alphabet until hemisphere resolves
    # them; L means lateral there and left in the output, and conflating the
    # two is the obvious bug.
    toward_medial = medial_edge in ('bottom', 'right')
    medial_letter = 'R' if hemisphere == 'left' else 'L'
    letters[EDGE_AXIS[medial_edge]] = medial_letter if toward_medial else opposite(medial_letter)

    if first_section is None:
        letters['z'] = _z_from_handedness(letters['y'], letters['x'])
    else:
        _check(first_section, FIRST_SECTIONS, "first_section")
        letters['z'] = 'V' if first_section == 'dorsal' else 'D'

    return letters['z'] + letters['y'] + letters['x']


def _z_from_handedness(y_letter: str, x_letter: str) -> str:
    """The D/V letter that makes the code's handedness `CONSISTENT_HANDEDNESS`."""
    for candidate in ('D', 'V'):
        if handedness(candidate + y_letter + x_letter) == CONSISTENT_HANDEDNESS:
            return candidate
    raise AssertionError(f"no z letter fits ({y_letter}, {x_letter})")


def validate_code(code: str) -> str:
    """Return `code` upper-cased, raising if it is not a well-formed code."""
    if not isinstance(code, str) or len(code) != 3:
        raise ValueError(f"orientation code must be 3 letters, got {code!r}")
    code = code.upper()
    for letter in code:
        if letter not in OPPOSITE_LETTER:
            raise ValueError(
                f"{letter!r} in {code!r} is not one of {sorted(OPPOSITE_LETTER)}"
            )
    axes = {frozenset((letter, opposite(letter))) for letter in code}
    if len(axes) != 3:
        raise ValueError(
            f"{code!r} does not name three distinct anatomical axes — each of "
            f"A/P, R/L and D/V must appear exactly once"
        )
    return code


def handedness(code: str) -> int:
    """Sign of the determinant of the code's three direction vectors.

    ``+1`` or ``-1``. Meaningless on its own — see `CONSISTENT_HANDEDNESS`.
    Use it only through `agree`.
    """
    code = validate_code(code)
    c1, c2, c3 = (LETTER_VECTOR[letter] for letter in code)
    det = (c1[0] * (c2[1] * c3[2] - c2[2] * c3[1])
           - c2[0] * (c1[1] * c3[2] - c1[2] * c3[1])
           + c3[0] * (c1[1] * c2[2] - c1[2] * c2[1]))
    return 1 if det > 0 else -1


def agree(code_a: str, code_b: str) -> bool:
    """True when two volumes of the same animal share a handedness.

    A False means one acquisition mirrored the tissue or one label is wrong.
    No rotation or translation reconciles a mirror — it would map the left
    hemisphere onto the right — so this is a hard stop for alignment, not a
    warning to look at later.
    """
    return handedness(code_a) == handedness(code_b)


def describe(code: str) -> str:
    """``'VPR'`` -> ``'+z ventral, +y posterior, +x right'``."""
    code = validate_code(code)
    return ", ".join(
        f"+{axis} {LETTER_NAME[letter]}"
        for axis, letter in zip('zyx', code)
    )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def orientation_path(explicit=None) -> Path:
    """Where the record lives: ``<ANALYSIS_ROOT>/orientation.json``.

    The analysis root itself, not a subdirectory — the information spans
    preprocessing and alignment. `explicit` (an ``--out`` value) wins outright.
    """
    if explicit:
        return Path(explicit).expanduser()
    root = get_analysis_root()
    if root is None:
        raise ValueError(
            f"\n{'='*60}\n"
            f"ANALYSIS_ROOT is not set in local_config.py, so there is no\n"
            f"place to keep {ORIENTATION_FILENAME}.\n\n"
            f"Either set it (derived outputs for a subject live under one\n"
            f"root, e.g. .../cell_type/analysis/BY95), or pass an explicit\n"
            f"path:\n\n"
            f"    --out <path>/{ORIENTATION_FILENAME}\n"
            f"{'='*60}"
        )
    return root / ORIENTATION_FILENAME


def make_entry(anterior_edge: str,
               medial_edge: str,
               hemisphere: str,
               first_section: Optional[str] = None,
               image=None) -> dict:
    """One modality's record: the derived code plus the raw answers.

    The raw answers are stored so a later convention change (R/L vs M/L, a
    different letter order) re-derives without anyone re-clicking anything.
    """
    return {
        'code': derive_code(anterior_edge, medial_edge, hemisphere, first_section),
        'anterior_edge': anterior_edge,
        'medial_edge': medial_edge,
        'hemisphere': hemisphere,
        'first_section': first_section,
        'image': None if image is None else str(image),
        'assigned': datetime.now().isoformat(timespec='seconds'),
    }


def section_key(slice_id) -> str:
    """Canonical JSON key for a section. JSON object keys are strings anyway."""
    return str(int(slice_id))


def sort_sections(keys):
    """Section keys in numeric order, tolerating a non-numeric one."""
    return sorted(keys, key=lambda k: (0, int(k)) if str(k).lstrip('-').isdigit()
                  else (1, str(k)))


def majority_code(section_entries: dict):
    """The code most sections share, ties broken by the lowest section key.

    This is the modality's reference frame. It is a *summary* of the per-section
    records, never a substitute for them: when sections disagree it is the code
    the majority were mounted at, and the minority are the deviants.
    """
    counts = {}
    for key in sort_sections(section_entries):
        code = section_entries[key].get('code')
        if code:
            counts[code] = counts.get(code, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    for key in sort_sections(section_entries):
        code = section_entries[key].get('code')
        if code and counts[code] == best:
            return code
    return None


def handedness_groups(code_map: dict) -> dict:
    """``{+1: [names], -1: [names]}`` — which volumes share a handedness.

    Two non-empty groups mean a mirror sits between them. Across modalities that
    is unfixable downstream; within one modality it is the set of sections
    mounted the other way up.
    """
    groups = {1: [], -1: []}
    for name in sorted(code_map, key=lambda n: (0, int(n)) if str(n).lstrip('-').isdigit()
                       else (1, str(n))):
        groups[handedness(code_map[name])].append(name)
    return groups


def load(path=None) -> dict:
    """Read the record. Missing file -> ``{'modalities': {}}``."""
    path = Path(path) if path else orientation_path()
    if not path.exists():
        return {'modalities': {}}
    with open(path) as f:
        record = json.load(f)
    record.setdefault('modalities', {})
    return record


def save(path, modality: str, entry: dict, subject: Optional[str] = None) -> Path:
    """Write `entry` under `modality`, leaving every other modality alone."""
    path = Path(path)
    record = load(path)
    if subject:
        record['subject'] = subject
    record['modalities'][modality] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(record, f, indent=2, sort_keys=False)
        f.write("\n")
    return path


def save_sections(path,
                  modality: str,
                  section_entries: dict,
                  subject: Optional[str] = None,
                  image=None) -> Path:
    """Write one entry PER SECTION under `modality`.

    Sections of one BARseq series are not guaranteed to have been mounted the
    same way up, and a section mounted face-down is a mirror that no alignment
    undoes. So each section carries its own code and its own raw answers, keyed
    by slice number — the number, not the node name, because node names carry
    the cutoff-folder suffix and would break the property that changing
    ``--min-rolonies`` never invalidates an assignment.

    The modality-level fields stay populated with the MAJORITY section's code
    and answers, so `codes` and every existing reader keep working unchanged.
    Read `sections` when you need the per-section truth; the top-level code is
    a summary, and where sections disagree it describes only the majority.
    """
    entries = {section_key(k): v for k, v in section_entries.items()}
    consensus = majority_code(entries)
    n_agree = sum(1 for e in entries.values() if e.get('code') == consensus)

    reference = next(
        (entries[k] for k in sort_sections(entries)
         if entries[k].get('code') == consensus),
        {},
    )
    entry = {
        'code': consensus,
        'anterior_edge': reference.get('anterior_edge'),
        'medial_edge': reference.get('medial_edge'),
        'hemisphere': reference.get('hemisphere'),
        'first_section': reference.get('first_section'),
        # Falls back to the reference section's own image, so the modality
        # entry names something real rather than null.
        'image': str(image) if image is not None else reference.get('image'),
        'assigned': datetime.now().isoformat(timespec='seconds'),
        'per_section': True,
        'n_sections': len(entries),
        'n_agree_with_code': n_agree,
        'sections': {k: entries[k] for k in sort_sections(entries)},
    }
    return save(path, modality, entry, subject=subject)


def section_codes(path=None, modality: Optional[str] = None) -> dict:
    """``{slice_id: code}`` for one modality's per-section records.

    Empty when that modality was assigned as a whole rather than per section.
    Keys are ints where the section key is numeric, so they sort naturally.
    """
    record = load(path)
    entry = record.get('modalities', {}).get(modality, {})
    out = {}
    for key, section in (entry.get('sections') or {}).items():
        if section.get('code'):
            out[int(key) if str(key).lstrip('-').isdigit() else key] = section['code']
    return out


def codes(path=None) -> dict:
    """``{modality: code}`` for every modality in the record."""
    return {
        name: entry['code']
        for name, entry in load(path).get('modalities', {}).items()
        if entry.get('code')
    }


def disagreeing_pair(code_map: dict):
    """First ``(name_a, name_b)`` in `code_map` whose handedness differs, or None."""
    items = sorted(code_map.items())
    for i, (name_a, code_a) in enumerate(items):
        for name_b, code_b in items[i + 1:]:
            if not agree(code_a, code_b):
                return name_a, name_b
    return None
