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
    # The Huang lab 2P runs one 512-px scanner at two zoom settings, ~2.83x
    # apart. 'huang_lab' is the CURRENT one (200.19 µm FOV, 1 µm Z, 401 z
    # levels) — that is what a bare SCOPE = "huang_lab" means. The older
    # 566.08 µm setting keeps its own key: by84, by94 and by89 were acquired
    # on it, and picking the wrong one scales XY by 2.83x.
    'huang_lab': {
        'xy_um_per_px': 0.3910,     # 512 px / 200.19 µm FOV
        'z_um_per_px': 1.0,         # 401 z levels
        'description': 'Huang lab 2P (200.19 µm FOV, 1 µm Z)',
    },
    'huang_lab_566um': {
        'xy_um_per_px': 1.1055,     # 512 px / 566.08 µm FOV
        'z_um_per_px': 2.0,
        'description': 'Huang lab 2P, older zoom (566.08 µm FOV, 2 µm Z)',
    },
}

# Accepted SCOPE spellings that are not profile keys. Kept so a config written
# against the 2026-08-21 naming keeps working.
SCOPE_ALIASES = {
    'huang_lab_200um': 'huang_lab',
}

# Largest XY pixel size (µm/px) we'll accept as a *real* microscope calibration.
# Cellular 2P imaging is sub-~5 µm/px (li_lab 2.34, huang_lab 0.39 / 1.11).
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

    huang_lab (0.3910) → 1.2219x; huang_lab_566um (1.1055) → 3.4547x.
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
