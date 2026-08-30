"""Microscope pixel-size profiles, shared by every pipeline.

Lives at the project root, next to ``local_config.py`` and ``analysis_paths.py``,
and imports nothing beyond the stdlib — preprocessing, alignment, and the
cellpose venv can all import it without pulling in numpy, tifffile or castalign.

``local_config.py`` names the scope that acquired a subject's data::

    SCOPE = "huang_lab"

That single declaration is the source of truth for pixel size everywhere:

- preprocessing resamples the BARseq images to the scope's XY pitch
  (``DOWNSAMPLE_XY = xy_um_per_px / 0.32``)
- the graph builder stamps ``(z, xy, xy)`` on every 2P node and
  ``(20.0, xy, xy)`` on every BARseq subslice node
- ``validate_mnn.py`` scores centroid distances in µm with the same pitch

A TIFF's own ``XResolution`` metadata is still read where available, but only to
CHECK the declaration — ``assert_matches_metadata()`` raises when the two
disagree. It is never used as an independent source, and it is never written:
raw acquisition files are read-only.
"""

from typing import Optional, Tuple

# Known microscope profiles. XY pixel size identifies the scope; Z is its
# nominal step. To add a scope, add an entry here — nothing else needs editing.
MICROSCOPE_PROFILES = {
    'li_lab': {
        'xy_um_per_px': 2.34,       # 512 px / 1200 µm FOV
        'z_um_per_px': 1.0,
        'description': 'Li lab 2P (1200 µm FOV)',
    },
    # Read from the stacks' own XResolution tag 2026-08-30: BY95's 2P files
    # carry 909090/1000000 with ImageJ unit µm, and ImageJ stores 1/pixelWidth,
    # so the pitch is 1.1000 µm/px over a 563.2 µm field.
    #
    # This replaced 0.3910 µm/px / a 200.19 µm field, which was a verbal report
    # contradicted by every one of BY95's own stacks. It survived because
    # get_tiff_resolution tested for the literal string 'micron' and these files
    # say 'µm', so assert_matches_metadata never had a number to check.
    #
    # 'huang_lab_566um' (1.1055 µm/px, 566.08 µm, 2 µm Z — by84, by94, by89) was
    # deleted the same day. 563.2 and 566.08 are 0.51% apart, so the "one
    # scanner at two zooms ~2.83x apart" story was itself an artifact of the
    # wrong 200.19: those are most likely one setting measured two ways. They
    # also collided inside identify_scope's 5% tolerance, which returns the
    # first match in insertion order — with both present the XY check could no
    # longer tell them apart. by84/by94/by89 now hard-error until their profile
    # is re-derived from their own tags, which is also the measurement that
    # settles whether the two were ever different.
    'huang_lab': {
        'xy_um_per_px': 1.1000,     # 512 px / 563.2 µm FOV, from TIFF XResolution
        'z_um_per_px': 1.0,         # 401 planes, 400 µm deep
        'description': 'Huang lab 2P (563.2 µm FOV, 1 µm Z, 401 planes)',
    },
}

# Accepted SCOPE spellings that are not profile keys. Empty since 2026-08-30:
# 'huang_lab_200um' aliased to huang_lab and named a field size that never
# described these stacks. resolve_scope_name still reads this.
SCOPE_ALIASES = {}

# Largest XY pixel size (µm/px) we'll accept as a *real* microscope calibration.
# Cellular 2P imaging is sub-~5 µm/px (li_lab 2.34, huang_lab 1.10).
# Anything coarser read from TIFF metadata is almost always an uncalibrated
# screen/print DPI default rather than a true pixel size — e.g. the 72-DPI inch
# default reads as 25400/72 = 352.78 µm/px (300 DPI → 84.7, 600 → 42.3, all
# > 20). Tune here if a genuinely coarse scope is ever added.
MAX_PLAUSIBLE_XY_UM_PER_PX = 20.0

# Physical thickness of a BARseq section, µm. Invariant across datasets.
SECTION_THICKNESS_UM = 20.0

# Ex vivo BARseq in-plane pixel size, µm/px. Invariant across datasets.
EXVIVO_UM_PER_PX = 0.32


def resolve_scope_name(name: str) -> str:
    """Map a SCOPE value through SCOPE_ALIASES to a profile key."""
    return SCOPE_ALIASES.get(name, name)


def valid_scopes() -> list:
    """Profile keys, sorted — for error messages."""
    return sorted(MICROSCOPE_PROFILES.keys())


def get_profile(scope: str, var_name: str = "SCOPE") -> dict:
    """
    Return the profile dict for a scope name.

    Raises ValueError if `scope` is blank or not a known profile. `var_name`
    names the config variable to quote in the message.
    """
    if not scope:
        raise ValueError(
            f"\n{'='*60}\n"
            f"{var_name} is not set in local_config.py.\n\n"
            f"Declare which microscope acquired this subject's data. It is the\n"
            f"source of truth for pixel size in every pipeline: the BARseq\n"
            f"resample factor, the graph node spacings, and MNN scoring.\n\n"
            f"Valid values:\n"
            f"{describe_profiles()}\n\n"
            f"Example:\n"
            f"    {var_name} = \"huang_lab\"\n"
            f"{'='*60}"
        )
    key = resolve_scope_name(scope)
    if key not in MICROSCOPE_PROFILES:
        raise ValueError(
            f"\n{'='*60}\n"
            f"{var_name} = {scope!r} is not a known microscope.\n\n"
            f"Valid values:\n"
            f"{describe_profiles()}\n\n"
            f"Edit local_config.py, or add the scope to MICROSCOPE_PROFILES in\n"
            f"scope_profiles.py.\n"
            f"{'='*60}"
        )
    return MICROSCOPE_PROFILES[key]


def describe_profiles() -> str:
    """Indented one-line-per-scope listing, for error messages."""
    return "\n".join(
        f"    {name:<16} XY {p['xy_um_per_px']:>7.4f} µm/px, Z {p['z_um_per_px']:>4} µm"
        f"   ({p['description']})"
        for name, p in MICROSCOPE_PROFILES.items()
    )


def spacing_zyx(scope: str, var_name: str = "SCOPE") -> Tuple[float, float, float]:
    """(Z, Y, X) µm/px for a 2P volume acquired on `scope`."""
    p = get_profile(scope, var_name)
    return (p['z_um_per_px'], p['xy_um_per_px'], p['xy_um_per_px'])


def subslice_spacing_zyx(scope: str, var_name: str = "SCOPE") -> Tuple[float, float, float]:
    """
    (Z, Y, X) µm/px for a BARseq subslice resampled to `scope`'s XY pitch.

    Z is the physical section thickness, not a pixel size — a section is one
    plane thick. Sections are cut in the 2P imaging plane, so the scope's XY
    pitch applies to both in-plane axes.
    """
    p = get_profile(scope, var_name)
    return (SECTION_THICKNESS_UM, p['xy_um_per_px'], p['xy_um_per_px'])


def downsample_xy(scope: str, var_name: str = "SCOPE") -> float:
    """
    BARseq resample factor for `scope`, as a divisor: new = original / factor.

    huang_lab (1.1000) → 3.4375x; li_lab (2.34) → 7.3125x.
    """
    return get_profile(scope, var_name)['xy_um_per_px'] / EXVIVO_UM_PER_PX


def identify_scope(xy_um_per_px: float, tolerance: float = 0.05) -> Optional[str]:
    """
    Profile key whose XY pitch matches `xy_um_per_px` within `tolerance`
    (relative), or None if none does.
    """
    for name, profile in MICROSCOPE_PROFILES.items():
        expected = profile['xy_um_per_px']
        if abs(xy_um_per_px - expected) / expected < tolerance:
            return name
    return None


def is_plausible_xy(xy_um_per_px: float) -> bool:
    """False for an uncalibrated DPI default masquerading as a pixel size."""
    return 0 < xy_um_per_px <= MAX_PLAUSIBLE_XY_UM_PER_PX


# ------------------------------------------------------------------
# TIFF resolution units
#
# Lives here, not in utilities/image_io.py, because this module is stdlib-only:
# tests/ runs on WSL where neither numpy nor tifffile is installed, and this is
# the parser whose blind spot let a 2.81x pixel-size error through unreported
# for a week. Pure function, no I/O, so it is testable there.
# ------------------------------------------------------------------

# Unit spelling → µm per unit. ImageJ escapes the micron sign when it writes
# its info string and tifffile passes the escape through undecoded, so the
# `unit` value read off a real file is usually the seven characters
# backslash-u-0-0-B-5-m, not the one-character glyph. normalize_unit decodes
# before matching; a set-membership test on the glyph alone misses every
# ImageJ-written file, which is how BY95's 2P stacks read as unitless for a
# week. Keys here are the decoded forms.
UNIT_TO_UM = {
    'um': 1.0,
    'µm': 1.0,          # MICRO SIGN
    'μm': 1.0,          # GREEK SMALL LETTER MU
    'micron': 1.0,
    'microns': 1.0,
    'micrometer': 1.0,
    'micrometers': 1.0,
    'micrometre': 1.0,
    'micrometres': 1.0,
    'nm': 1e-3,
    'mm': 1e3,
    'cm': 1e4,
    'm': 1e6,
    'inch': 25400.0,
    'in': 25400.0,
}

# TIFF ResolutionUnit tag values that name a length. 1 means "no unit" and
# decides nothing.
RESOLUTION_UNIT_TO_UM = {
    2: 25400.0,   # inch
    3: 1e4,       # centimeter
}


def normalize_unit(unit) -> str:
    """
    Canonical lowercase unit spelling, with any ``\\uXXXX`` escape decoded.

    Returns '' for None or a value that decodes to nothing.
    """
    if unit is None:
        return ''
    text = str(unit)
    if '\\u' in text or '\\U' in text or '\\x' in text:
        try:
            text = text.encode('ascii', 'backslashreplace').decode('unicode_escape')
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
    return text.strip().lower()


def um_per_px_from_resolution(
    pixels_per_unit,
    imagej_unit=None,
    resolution_unit=None,
) -> Optional[float]:
    """
    µm/px from a TIFF XResolution value plus unit hints, or None if undecidable.

    `pixels_per_unit` is XResolution already divided out (numerator/denominator).
    `imagej_unit` is ImageJ's ``unit`` metadata; `resolution_unit` is the TIFF
    ResolutionUnit tag (2 inch, 3 cm), accepted as an int or an enum.

    A recognised ImageJ unit WINS over ResolutionUnit. ImageJ leaves
    ResolutionUnit at its inch default on µm images, so consulting the tag first
    converts a micron file with 25400 and returns a confidently wrong number
    where None is the honest answer.
    """
    try:
        pixels_per_unit = float(pixels_per_unit)
    except (TypeError, ValueError):
        return None
    if not pixels_per_unit > 0:
        return None

    um_per_unit = UNIT_TO_UM.get(normalize_unit(imagej_unit))

    if um_per_unit is None and resolution_unit is not None:
        try:
            um_per_unit = RESOLUTION_UNIT_TO_UM.get(int(resolution_unit))
        except (TypeError, ValueError):
            um_per_unit = None

    if um_per_unit is None:
        return None
    return um_per_unit / pixels_per_unit


def assert_matches_metadata(
    scope: str,
    xy_um_per_px: Optional[float],
    z_um_per_px: Optional[float] = None,
    source_name: str = "the file",
    tolerance: float = 0.05,
    var_name: str = "SCOPE",
) -> None:
    """
    Check a file's own resolution metadata against the declared scope.

    `scope` is authoritative — this never changes what pixel size gets used, it
    only refuses to continue when the file disagrees, which means either the
    SCOPE line is wrong or the file is not from this subject.

    Silently returns when there is nothing to check: no metadata (`None`), or
    metadata that is an uncalibrated DPI default (`is_plausible_xy` false —
    common, since raw BARseq TIFFs carry a 72-DPI placeholder).
    """
    if xy_um_per_px is None or not is_plausible_xy(xy_um_per_px):
        return

    profile = get_profile(scope, var_name)
    expected = profile['xy_um_per_px']
    if abs(xy_um_per_px - expected) / expected < tolerance:
        return

    detected = identify_scope(xy_um_per_px, tolerance)
    reads_as = (
        f"{xy_um_per_px:.4f} µm/px, which is {detected}"
        if detected else
        f"{xy_um_per_px:.4f} µm/px, which matches no known scope"
    )
    ratio = max(expected / xy_um_per_px, xy_um_per_px / expected)
    raise ValueError(
        f"\n{'='*60}\n"
        f"Scope mismatch: {source_name} was not acquired on {var_name}.\n\n"
        f"  {var_name} = {scope!r}  →  {expected:.4f} µm/px "
        f"({profile['description']})\n"
        f"  {source_name} reads {reads_as}\n\n"
        f"Pixel size comes from {var_name}, so continuing would misstate this\n"
        f"file's physical size by {ratio:.2f}x. Either {var_name} is wrong for\n"
        f"this subject, or this file belongs to a different one.\n\n"
        f"Valid values:\n"
        f"{describe_profiles()}\n"
        f"{'='*60}"
    )


def unknown_scope_message(xy_um_per_px: float, source_name: str = "your image") -> str:
    """Copy-pasteable 'add a profile' message for an unrecognized pixel size."""
    return (
        f"\n{'='*60}\n"
        f"Could not identify microscope from image resolution.\n\n"
        f"  Image:       {source_name}\n"
        f"  XY detected: {xy_um_per_px:.4f} µm/px\n\n"
        f"This doesn't match any known microscope:\n"
        f"{describe_profiles()}\n\n"
        f"To fix, add your microscope to MICROSCOPE_PROFILES in\n"
        f"scope_profiles.py:\n\n"
        f"    'your_scope_name': {{\n"
        f"        'xy_um_per_px': {xy_um_per_px:.4f},\n"
        f"        'z_um_per_px': <your Z spacing in µm>,\n"
        f"        'description': '<scope name> (<FOV size> FOV)',\n"
        f"    }},\n"
        f"{'='*60}"
    )
