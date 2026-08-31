"""
Subslice Graph Builder - LineStuffUp Alignment Graph for BARseq Subslices

Builds the alignment graph from the 2P volumes named in local_config.py plus the
downsampled BARseq subslices produced by preprocessing/.

The only BARseq image ingested is the binary marker-only ALIGN tif written by
preprocessing/generate_alignment_tif.py; a folder holding none is an error. A
subslice node is named for its file stem AND its source folder, so each rolony
cutoff (`qc20_5_ge1`, `qc20_5_ge5`, ...) is a distinct node set. Re-pointing
SUBSLICE_DIR and re-running adds a parallel set rather than silently keeping the
pixels already stored under a colliding name; alignment is redone by hand.

Pixel sizes come from the scope declared as SCOPE in local_config.py, via the
shared profile table in scope_profiles.py. 2P nodes get (z, xy, xy); BARseq
subslice nodes get (20.0, xy, xy) — Z is the physical section thickness, and
sections are cut in the 2P imaging plane so one pitch covers both in-plane axes.

A TIFF's own XResolution metadata, where present, is read only to CHECK the
declaration: a file whose pixel size disagrees with SCOPE stops the run.

Each node also carries the anatomical orientation code recorded for its modality
by preprocessing/assign_orientation.py, so the node is self-describing. Missing
or unlabelled is fine (the key is None); two labelled modalities that disagree on
handedness are not, and stop the run — see `check_orientation_handedness`.

Author: DTC
Date: 2024-12-15
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Union, List

# Add project root to path for local_config import
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import local_config
except ImportError:
    raise ImportError(
        "local_config.py not found.\n"
        "Copy local_config.example.py to local_config.py and fill in your paths:\n"
        "    cp local_config.example.py local_config.py"
    )

# Required: output graph location
try:
    GRAPH_PATH = local_config.GRAPH_PATH
except AttributeError as e:
    raise ImportError(
        f"local_config.py is missing a required variable: {e}\n"
        "GRAPH_PATH is required. See local_config.example.py."
    )

# Optional data inputs — blank string or missing attribute = skip that node.
# Each 2P volume has two co-registered channels (red = sparse alignment channel,
# green = signal of interest). Both channels enter the graph as sibling nodes
# joined by `castalign.base.Identity`; rigid + nonlinear fits run only on red.
INVIVO_PATH_RED = getattr(local_config, "INVIVO_PATH_RED", "")
INVIVO_PATH_GREEN = getattr(local_config, "INVIVO_PATH_GREEN", "")
BLOCK_STACK_PATH_RED = getattr(local_config, "BLOCK_STACK_PATH_RED", "")
BLOCK_STACK_PATH_GREEN = getattr(local_config, "BLOCK_STACK_PATH_GREEN", "")
SUBSLICE_DIR = getattr(local_config, "SUBSLICE_DIR", "")

# BARseq raw fluorescence channels to carry into the graph beside the ALIGN
# renders, e.g. ["MSCARLET"]. Empty (or missing) = skip them. Every channel
# preprocessing downsampled is available: DAPI, GCAMP, MSCARLET. They are read
# from HYB_DOWNSAMPLED_DIR, which is where downsample_subslices_cellmask.py
# resamples ALL channels to one target shape computed from the cellmask — so a
# raw channel image sits on exactly the grid the ALIGN tif is painted on.
SUBSLICE_RAW_CHANNELS = [
    str(c).upper() for c in getattr(local_config, "SUBSLICE_RAW_CHANNELS", []) or []
]

# The microscope that acquired this subject's data — the source of truth for
# every pixel size below. Validated at first use, not at import, so a module
# import (or --help) does not require a complete config.
SCOPE = getattr(local_config, "SCOPE", "")

# Node names this builder no longer writes. Presence of any in a loaded graph
# triggers the migration guard in build_subslice_graph(). Two generations:
# the pre-multi-channel schema had no _red/_green suffix at all, and the 2P
# bases were renamed 2026-08-30 to match local_config's vocabulary
# (INVIVO_PATH_* -> invivo_*, BLOCK_STACK_PATH_* -> block_stack_*).
LEGACY_NODE_NAMES = {
    "invivo_ref", "ex_vivo_block",
    "invivo_ref_red", "invivo_ref_green",
    "ex_vivo_block_red", "ex_vivo_block_green",
    "invivo_ref_z_tilt_corrected_red", "invivo_ref_z_tilt_corrected_green",
}

import castalign as ca
import numpy as np
import imageio.v2 as imageio
from utilities.image_io import get_tiff_resolution
import orientation
from analysis_paths import (
    hyb_downsampled_dir,
    analysis_subdir,
    resolve_subslice_dir,
    subject_name,
    ALIGNMENT_SUBDIR,
)
from scope_profiles import (
    assert_matches_metadata,
    get_profile,
    is_plausible_xy,
    spacing_zyx,
    subslice_spacing_zyx,
)


# ============================================
# Data Loading Functions
# ============================================

def load_invivo_stack(path: Union[str, Path]) -> np.ndarray:
    """
    Load in vivo TIFF stack.

    Returns
    -------
    np.ndarray
        3D stack (Z, Y, X) - shape (399, 512, 512)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"In vivo stack not found: {path}")

    print(f"Loading in vivo stack: {path.name}")
    stack = imageio.volread(str(path))

    print(f"  Shape: {stack.shape} (Z, Y, X)")
    print(f"  Type: {stack.dtype}")

    return stack


def _slice_number(path: Union[str, Path]) -> Optional[int]:
    """Section number from a `slice{N}_subslice_ALIGN.tif` filename."""
    m = re.match(r"slice(\d+)_subslice", Path(path).name)
    return int(m.group(1)) if m else None


def subslice_node_name(path: Union[str, Path],
                       subslice_dir: Union[str, Path]) -> str:
    """Node name for a subslice file: its stem plus its source folder.

    The folder is the rolony cutoff (`qc20_5_ge1`, `qc20_5_ge5`, ...), so two
    cutoffs of the same section become two distinct nodes that coexist in one
    graph instead of colliding on `slice10_subslice_ALIGN`.
    """
    return f"{Path(path).stem}_{Path(subslice_dir).name}"


def raw_channel_node_name(slice_no: int, channel: str) -> str:
    """Node name for a raw channel image: `slice22_mscarlet`.

    No cutoff suffix, unlike an ALIGN node: the raw image is the same pixels
    whatever rolony cutoff was rendered from it, so there is exactly one per
    section per channel and every ALIGN render of that section is an Identity
    sibling of it.
    """
    return f"slice{int(slice_no)}_{channel.lower()}"


def raw_channel_tif(directory: Union[str, Path], slice_no: int,
                    channel: str) -> Path:
    """Path to one downsampled raw channel image."""
    return Path(directory) / f"slice{int(slice_no)}_subslice_{channel.upper()}.tif"


def align_nodes_by_slice(graph: ca.Graph) -> dict:
    """{section number: [ALIGN node names]} for every subslice node in the graph.

    One section can hold several ALIGN nodes, one per rolony cutoff rendered.
    """
    out = {}
    for name in graph.nodes:
        if "_subslice_ALIGN" not in name:
            continue
        m = re.match(r"slice(\d+)_", name)
        if m:
            out.setdefault(int(m.group(1)), []).append(name)
    return {k: sorted(v) for k, v in sorted(out.items())}


def discover_subslices(
    directory: Union[str, Path]
) -> List[Path]:
    """
    Discover available BARseq subslice files.

    Returns
    -------
    List[Path]
        Sorted list of subslice file paths
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Subslice directory not found: {directory}")

    # The ALIGN tifs are what this fits on: binary, marker-only, written by
    # preprocessing/generate_alignment_tif.py. Nothing else is accepted -- the
    # step 4 overlay is an RGB display figure, so load_single_subslice()
    # collapses it by BT.601 luminance, which on BY95 renders the median marker
    # cell DARKER than the mask field behind it.
    files = sorted(directory.glob("slice*_subslice_ALIGN.tif"))
    if not files:
        raise FileNotFoundError(
            f"No slice*_subslice_ALIGN.tif in:\n"
            f"  {directory}\n"
            f"Run preprocessing/generate_alignment_tif.py and point SUBSLICE_DIR "
            f"at its output folder."
        )

    print(f"Found {len(files)} BARseq alignment TIFs")
    return files


def load_single_subslice(path: Union[str, Path]) -> np.ndarray:
    """
    Load a single BARseq subslice.

    Returns
    -------
    np.ndarray
        Image with shape (1, H, W) for LineStuffUp compatibility
    """
    img = imageio.imread(str(path))

    # Add Z dimension if 2D grayscale
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
    # Handle RGB/RGBA - convert to grayscale using ITU-R BT.601 luminance formula
    elif img.ndim == 3 and img.shape[-1] in [3, 4]:
        img_gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
        img = img_gray[np.newaxis, :, :]

    return img


def load_block_stack(path: Union[str, Path]) -> np.ndarray:
    """
    Load ex-vivo block TIFF stack.

    Returns
    -------
    np.ndarray
        3D stack as float32
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Block image not found: {path}")

    print(f"Loading block stack: {path.name}")
    stack = imageio.volread(str(path)).astype(np.float32)

    print(f"  Shape: {stack.shape}")
    print(f"  Type: {stack.dtype}")

    return stack


def spacing_for_tiff(tiff_path: Union[str, Path], scope: str = None) -> tuple:
    """
    (Z, Y, X) spacing for a 2P TIFF, from the declared SCOPE.

    SCOPE is authoritative. The file's own XResolution metadata is read, but
    only to check the declaration: a plausible pixel size that disagrees with
    SCOPE raises (`assert_matches_metadata`), because one of the two is wrong
    and guessing which would scale the volume silently. Absent metadata, or the
    72-DPI placeholder BARseq TIFFs carry, is fine — SCOPE covers it.

    Returns
    -------
    tuple
        ((z, y, x) spacing in µm/px, scope_name)
    """
    scope = SCOPE if scope is None else scope
    profile = get_profile(scope)

    res = get_tiff_resolution(tiff_path)
    xy_meta = res.get('xy_um_per_px')
    assert_matches_metadata(
        scope, xy_meta, source_name=Path(tiff_path).name,
    )

    z, y, x = spacing_zyx(scope)
    print(f"  Scope: {scope} ({profile['description']})")
    print(f"  Spacing (Z, Y, X): ({z}, {y:.4f}, {x:.4f}) µm/px")
    if xy_meta is None:
        print("    (no resolution metadata in file — nothing to cross-check)")
    elif not is_plausible_xy(xy_meta):
        print(f"    (file metadata reads {xy_meta:.2f} µm/px, an uncalibrated "
              f"DPI default — ignored)")
    else:
        print(f"    (file metadata agrees: {xy_meta:.4f} µm/px)")
    return (z, y, x), scope


# ============================================
# Orientation
# ============================================

# Modality key in orientation.json for each set of graph nodes. The 2P keys are
# the node base names; every BARseq subslice in a dataset shares one code.
SUBSLICE_MODALITY = "barseq_subslice"


def load_orientation_codes() -> dict:
    """
    ``{modality: code}`` from ``<ANALYSIS_ROOT>/orientation.json``.

    Labelling is optional: no ANALYSIS_ROOT, no file, or a modality that was
    never assigned all yield no code and a one-line notice. Nodes then carry
    ``orientation=None``, which nothing existing reads.
    """
    try:
        path = orientation.orientation_path()
    except ValueError:
        print("  Orientation: ANALYSIS_ROOT not set — nodes get orientation=None")
        return {}

    if not path.exists():
        print(f"  Orientation: no {path.name} yet — nodes get orientation=None")
        print(f"               (assign with preprocessing/assign_orientation.py)")
        return {}

    codes = orientation.codes(path)
    print(f"  Orientation: {path}")
    for name, code in sorted(codes.items()):
        print(f"    {name}: {code}  ({orientation.describe(code)})")
    return codes


def check_orientation_handedness(codes: dict, modalities) -> None:
    """
    Refuse to build when two labelled modalities disagree on handedness.

    A mirror is not an alignment problem — no rotation or translation undoes
    it, and a flip that fixed the image would map the left hemisphere onto the
    right. This is the failure that otherwise passes alignment QC silently.
    """
    relevant = {m: codes[m] for m in modalities if m in codes}
    disagreement = orientation.disagreeing_pair(relevant)
    if disagreement is None:
        return

    name_a, name_b = disagreement
    raise ValueError(
        f"\n{'='*60}\n"
        f"Orientation handedness disagreement between two modalities of this\n"
        f"subject:\n\n"
        f"  {name_a:<18} {relevant[name_a]}  ({orientation.describe(relevant[name_a])})\n"
        f"  {name_b:<18} {relevant[name_b]}  ({orientation.describe(relevant[name_b])})\n\n"
        f"One acquisition mirrored the tissue, or one label is wrong. Alignment\n"
        f"cannot succeed through a mirror: no rotation or translation maps one\n"
        f"onto the other, and a flip would map the left hemisphere onto the\n"
        f"right.\n\n"
        f"Re-check both with:\n"
        f"    python preprocessing/assign_orientation.py --show\n"
        f"and re-assign the wrong one.\n"
        f"{'='*60}"
    )


# ============================================
# Graph Operations
# ============================================

def create_subslice_graph(name: str = "castalign_test") -> ca.Graph:
    """Create new empty LineStuffUp graph."""
    print(f"Creating graph '{name}'...")
    return ca.Graph(name)


def _add_volume_channels(
    graph: ca.Graph,
    base_name: str,
    *,
    red_stack: Optional[np.ndarray] = None,
    green_stack: Optional[np.ndarray] = None,
    red_spacing: Optional[tuple] = None,
    green_spacing: Optional[tuple] = None,
    orientation_code: Optional[str] = None,
) -> ca.Graph:
    """
    Add red and/or green channel nodes for a 2P volume, joined by a
    `castalign.base.Identity` edge once both channels exist.

    Per-channel idempotent: if a node already exists in the graph it is
    skipped silently. The Identity edge is added only once both `_red` and
    `_green` nodes are present and is itself idempotent.
    """
    red_node = f"{base_name}_red"
    green_node = f"{base_name}_green"
    default_spacing = spacing_zyx(SCOPE)

    if red_stack is not None and red_node not in graph.nodes:
        spacing = red_spacing if red_spacing is not None else default_spacing
        graph.add_node(
            red_node, image=red_stack, compression="high",
            metadata={'spacing': spacing, 'orientation': orientation_code},
        )
        print(f"  Added node: {red_node}  shape {red_stack.shape}  spacing {spacing} µm/px (Z, Y, X)")

    if green_stack is not None and green_node not in graph.nodes:
        spacing = green_spacing if green_spacing is not None else default_spacing
        graph.add_node(
            green_node, image=green_stack, compression="high",
            metadata={'spacing': spacing, 'orientation': orientation_code},
        )
        print(f"  Added node: {green_node}  shape {green_stack.shape}  spacing {spacing} µm/px (Z, Y, X)")

    if red_node in graph.nodes and green_node in graph.nodes:
        if green_node not in graph.edges.get(red_node, {}):
            graph.add_edge(red_node, green_node, ca.Identity())
            print(f"  Added Identity edge: {red_node} <-> {green_node}")

    return graph


def add_invivo_to_graph(
    graph: ca.Graph,
    *,
    red_stack: Optional[np.ndarray] = None,
    green_stack: Optional[np.ndarray] = None,
    red_spacing: Optional[tuple] = None,
    green_spacing: Optional[tuple] = None,
    orientation_code: Optional[str] = None,
    base_name: str = "invivo",
) -> ca.Graph:
    """Add in-vivo red/green nodes joined by Identity. See `_add_volume_channels`."""
    print(f"Adding in-vivo channels under base '{base_name}'...")
    return _add_volume_channels(
        graph, base_name,
        red_stack=red_stack, green_stack=green_stack,
        red_spacing=red_spacing, green_spacing=green_spacing,
        orientation_code=orientation_code,
    )


def add_block_to_graph(
    graph: ca.Graph,
    *,
    red_stack: Optional[np.ndarray] = None,
    green_stack: Optional[np.ndarray] = None,
    red_spacing: Optional[tuple] = None,
    green_spacing: Optional[tuple] = None,
    orientation_code: Optional[str] = None,
    base_name: str = "block_stack",
) -> ca.Graph:
    """Add ex-vivo block red/green nodes joined by Identity. See `_add_volume_channels`."""
    print(f"Adding ex-vivo block channels under base '{base_name}'...")
    return _add_volume_channels(
        graph, base_name,
        red_stack=red_stack, green_stack=green_stack,
        red_spacing=red_spacing, green_spacing=green_spacing,
        orientation_code=orientation_code,
    )


def assert_stored_shapes_match(
    graph: ca.Graph,
    present: List[tuple],
    verbose: bool = True,
) -> None:
    """
    Raise when a node already in the graph was built from differently-sized pixels.

    A subslice node is named for its file stem and its cutoff folder
    (`subslice_node_name`), and neither encodes the resample factor. So after
    regenerating the ALIGN tifs at a new DOWNSAMPLE_XY every name still matches,
    every file counts as "already in graph", and the run prints "All subslices
    already in graph!" while the old pixels stay in the .db — the graph then
    reports a physical size it no longer has. That is what happened when
    huang_lab went from 0.3910 to 1.1000 µm/px on 2026-08-30.

    Shape is read from each node's own metadata, written at add time, against
    the TIFF header on disk — neither side decompresses an image. Nodes added
    before the shape was recorded cannot be checked and say so rather than
    failing: `force_rebuild=True` is the way to reset those.
    """
    if not present:
        return

    mismatched = []
    unrecorded = []
    for path, node_name in present:
        stored = (graph.node_metadata.get(node_name) or {}).get('shape')
        if stored is None:
            unrecorded.append(node_name)
            continue
        on_disk = get_tiff_resolution(path)['shape']
        # Stored is (1, H, W); a 2D TIFF's series shape is (H, W).
        if tuple(stored)[-2:] != tuple(on_disk)[-2:]:
            mismatched.append((node_name, tuple(stored), tuple(on_disk), path))

    if unrecorded and verbose:
        print(f"  {len(unrecorded)} existing node(s) carry no shape — not checked "
              f"(added before 2026-08-30; force_rebuild=True to reset)")

    if not mismatched:
        return

    lines = "\n".join(
        f"    {name}\n"
        f"      in graph: {stored}\n"
        f"      on disk:  {disk}   ({Path(p).name})"
        for name, stored, disk, p in mismatched
    )
    raise ValueError(
        f"\n{'='*60}\n"
        f"{len(mismatched)} subslice node(s) do not match the images on disk.\n\n"
        f"{lines}\n\n"
        f"The node name carries the cutoff folder, not the resample factor, so\n"
        f"these would have been silently kept at their old pixel size. Either\n"
        f"the images were regenerated at a different DOWNSAMPLE_XY, or\n"
        f"SUBSLICE_DIR points at a different render.\n\n"
        f"Rebuild with force_rebuild=True — and note that drops every fit on\n"
        f"these nodes, which were made against images that no longer exist.\n"
        f"{'='*60}"
    )


def add_subslices_to_graph(
    graph: ca.Graph,
    subslice_dir: Union[str, Path],
    save_every: int = 10,
    output_path: Optional[Union[str, Path]] = None,
    orientation_code: Optional[str] = None,
    slices: Optional[Iterable[int]] = None,
    dry_run: bool = False,
    verbose: bool = True
) -> ca.Graph:
    """
    Add downsampled BARseq subslices from one render folder to the graph.

    Subslice spacing: (Z, Y, X) = (20.0, xy, xy) µm/px, where xy is SCOPE's
    in-plane pitch — the same pitch preprocessing resampled the images to.
    Z is the physical section thickness, not a pixel size.

    `slices` restricts the add to those section numbers; None means every file
    in the folder. `dry_run` reports what would be added and writes nothing.
    """
    spacing = subslice_spacing_zyx(SCOPE)

    files = discover_subslices(subslice_dir)

    if slices is not None:
        wanted = {int(n) for n in slices}
        kept = [f for f in files if _slice_number(f) in wanted]
        missing = wanted - {_slice_number(f) for f in files}
        if missing:
            raise FileNotFoundError(
                f"--slices asked for sections not in {subslice_dir}:\n"
                f"  {sorted(missing)}\n"
                f"Present: {sorted(n for n in (_slice_number(f) for f in files) if n is not None)}"
            )
        files = kept

    # Check which are already in graph
    existing_nodes = set(graph.nodes)
    files_to_add = []
    already_present = []

    for f in files:
        # e.g., "slice10_subslice_ALIGN_qc20_5_ge1"
        node_name = subslice_node_name(f, subslice_dir)
        if node_name not in existing_nodes:
            files_to_add.append((f, node_name))
        else:
            already_present.append((f, node_name))

    assert_stored_shapes_match(graph, already_present, verbose=verbose)

    if verbose:
        print(f"\nSubslice Loading:")
        print(f"  Total files: {len(files)}")
        print(f"  Already in graph: {len(files) - len(files_to_add)}")
        print(f"  To add: {len(files_to_add)}")

    if len(files_to_add) == 0:
        print("All subslices already in graph!")
        return graph

    if dry_run:
        for _, node_name in files_to_add:
            print(f"  would ADD: {node_name}")
        return graph

    added = 0
    for i, (fpath, node_name) in enumerate(files_to_add, 1):
        try:
            img = load_single_subslice(fpath)
            # (Z, Y, X) µm/px, the one orientation every subslice shares, and
            # the shape this node's pixels were built at — see
            # assert_stored_shapes_match for what reads it back.
            metadata = {
                'spacing': spacing,
                'orientation': orientation_code,
                'shape': tuple(int(n) for n in img.shape),
            }
            # compression="label" forces castalign's lossless gzip branch
            # (utils.compress_image:153). "normal" leaves the choice to
            # utils.image_is_label, and when that says False a binary marker
            # image goes down the JPEG branch, where the normaliser divides by
            # np.quantile(img, .999) -- which on a >99.9% background image IS
            # the background value, so the cells are clipped away and what
            # comes back is float noise, not a mask. Measured on BY95's
            # qc20_5_ge5 nodes 2026-08-31: stored info [2, 1, 90, ...], read
            # back as float32 in [-1, 0]. An ALIGN tif is binary by
            # construction, so never let the heuristic decide.
            graph.add_node(node_name, image=img, compression="label", metadata=metadata)
            added += 1

            if verbose and (i <= 5 or i % 10 == 0 or i == len(files_to_add)):
                print(f"  [{i:2d}/{len(files_to_add)}] {node_name}: {img.shape}")

            # Incremental save
            if save_every > 0 and output_path and added % save_every == 0:
                if verbose:
                    print(f"  Saving checkpoint ({added} added)...")
                save_graph(graph, output_path, verbose=False)

        except Exception as e:
            if verbose:
                print(f"  [ERR] {fpath.name}: {e}")

    if verbose:
        print(f"\nAdded {added} subslices")
        print(f"Total nodes: {len(graph.nodes)}")

    return graph


def add_raw_channels_to_graph(
    graph: ca.Graph,
    hyb_dir: Union[str, Path],
    channels: Iterable[str],
    save_every: int = 10,
    output_path: Optional[Union[str, Path]] = None,
    orientation_code: Optional[str] = None,
    slices: Optional[Iterable[int]] = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> ca.Graph:
    """
    Add BARseq raw fluorescence nodes and wire them to the ALIGN renders.

    Topology: the FIRST channel listed is the section's hub. Every other node
    for that section — each cutoff's ALIGN render, and every other raw channel
    — gets one `castalign.base.Identity` edge to the hub. Star, not mesh: no
    cycles, one hop to the hub, two between any two ALIGN renders. All the
    images involved come off the same `round(orig / DOWNSAMPLE_XY)` target
    shape, computed once from the cellmask by downsample_subslices_cellmask.py,
    so Identity is the literal relationship between them, not an approximation.

    What that buys: a fit on ANY node of the section reaches every other node of
    it through Identity, so the raw images inherit the alignment made on an
    ALIGN render. What it costs: two independent fits on two renders of the same
    section become contradictory, and BFS resolves by path length, not by which
    was made last. Fit one node per section.

    Nodes are stored with `compression="label"`, castalign's lossless gzip
    branch (utils.compress_image:153). These are uint16 fluorescence images —
    the heuristic would send them to the lossy JPEG path, which clips at the
    99.9th percentile, i.e. the brightest somas.

    Only sections that already have an ALIGN node are considered; a raw node
    with no sibling would have no path to in-vivo.
    """
    channels = [c.upper() for c in channels]
    if not channels:
        return graph

    hyb_dir = Path(hyb_dir)
    if not hyb_dir.is_dir():
        raise FileNotFoundError(
            f"Raw channel folder not found:\n"
            f"  {hyb_dir}\n"
            f"SUBSLICE_RAW_CHANNELS is set, so downsample_subslices_cellmask.py\n"
            f"must have run — it writes every channel there. Leave\n"
            f"SUBSLICE_RAW_CHANNELS empty to skip raw channels."
        )

    by_slice = align_nodes_by_slice(graph)
    if slices is not None:
        wanted = {int(n) for n in slices}
        by_slice = {k: v for k, v in by_slice.items() if k in wanted}
    if not by_slice:
        print("  No ALIGN nodes in the graph — nothing to attach raw channels to.")
        return graph

    spacing = subslice_spacing_zyx(SCOPE)
    added = edges_added = missing = 0
    present_pairs = []          # (tif path, node name) for the shape guard
    # Nodes that exist, plus the ones this run would create -- so a dry run
    # reports the Identity edges too instead of stopping at the node list.
    known = set(graph.nodes)

    for slice_no, align_nodes in by_slice.items():
        hub = None
        for channel in channels:
            node_name = raw_channel_node_name(slice_no, channel)
            tif = raw_channel_tif(hyb_dir, slice_no, channel)

            if node_name in graph.nodes:
                if tif.exists():
                    present_pairs.append((tif, node_name))
            elif not tif.exists():
                if verbose:
                    print(f"  slice {slice_no}: no {channel} TIF — skipped ({tif.name})")
                missing += 1
                continue
            elif dry_run:
                print(f"  would ADD: {node_name}")
                known.add(node_name)
                added += 1
            else:
                img = load_single_subslice(tif)
                graph.add_node(
                    node_name, image=img, compression="label",
                    metadata={
                        'spacing': spacing,
                        'orientation': orientation_code,
                        'shape': tuple(int(n) for n in img.shape),
                    },
                )
                added += 1
                known.add(node_name)
                if verbose:
                    print(f"  Added node: {node_name}  shape {img.shape}  lossless")
                if save_every > 0 and output_path and added % save_every == 0:
                    if verbose:
                        print(f"  Saving checkpoint ({added} added)...")
                    save_graph(graph, output_path, verbose=False)

            if hub is None:
                hub = node_name

        if hub is None or hub not in known:
            continue

        # Identity to every sibling of this section: the other raw channels and
        # every ALIGN render. Idempotent — an existing edge is left alone.
        siblings = [raw_channel_node_name(slice_no, c) for c in channels[1:]]
        for other in siblings + align_nodes:
            if other not in known or other == hub:
                continue
            if other in graph.edges.get(hub, {}):
                continue
            if dry_run:
                print(f"  would LINK: {hub} <-> {other}  (Identity)")
            else:
                graph.add_edge(hub, other, ca.Identity())
            edges_added += 1

    assert_stored_shapes_match(graph, present_pairs, verbose=verbose)

    if verbose:
        print(f"\n{'Would add' if dry_run else 'Added'} {added} raw channel node(s), "
              f"{edges_added} Identity edge(s)")
        if missing:
            print(f"  {missing} section/channel combination(s) had no TIF")
    return graph


def save_graph(
    graph: ca.Graph,
    output_path: Union[str, Path],
    verbose: bool = True
) -> Path:
    """Save graph to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Saving graph to: {output_path}")

    # Handle overwrite logic
    if output_path.exists() and str(getattr(graph, "filename", "")) == str(output_path):
        graph.save()
    else:
        graph.save(str(output_path))

    if verbose and output_path.exists():
        size_mb = output_path.stat().st_size / 1e6
        print(f"  Saved: {size_mb:.1f} MB")

    return output_path


def load_graph(path: Union[str, Path], verbose: bool = True) -> ca.Graph:
    """Load existing graph from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph not found: {path}")

    if verbose:
        print(f"Loading graph: {path.name}")

    graph = ca.Graph.load(str(path))

    if verbose:
        print(f"  Loaded: {len(graph.nodes)} nodes")

    return graph


# ============================================
# High-Level Pipeline
# ============================================

def _derive_graph_path(
    block_red_path: Optional[Path],
    invivo_red_path: Optional[Path],
) -> Path:
    """
    Derive a default GRAPH_PATH when not set in local_config.py.

    Rule 1 (preferred): ``<ANALYSIS_ROOT>/alignment/<ANALYSIS_ROOT name>_graph.db``
    Rule 2 (no ANALYSIS_ROOT): ``<parent_of_data_file>/alignment/<parent_folder_name>_graph.db``

    Anchored on the red-channel paths (red is canonical, green is sibling).
    Prefers BLOCK_STACK_PATH_RED, falls back to INVIVO_PATH_RED. Raises
    ValueError if neither is set, or if both are set but live in different
    parent folders.
    """
    analysis_alignment = analysis_subdir(ALIGNMENT_SUBDIR)
    if analysis_alignment is not None:
        return analysis_alignment / f"{subject_name()}_graph.db"

    if block_red_path is None and invivo_red_path is None:
        raise ValueError(
            "Cannot auto-derive GRAPH_PATH: neither BLOCK_STACK_PATH_RED nor\n"
            "INVIVO_PATH_RED is set. If you're running a BARseq-only graph\n"
            "(SUBSLICE_DIR only), please set GRAPH_PATH manually in\n"
            "local_config.py."
        )

    if block_red_path is not None and invivo_red_path is not None:
        if block_red_path.parent != invivo_red_path.parent:
            raise ValueError(
                "Cannot auto-derive GRAPH_PATH: BLOCK_STACK_PATH_RED and\n"
                "INVIVO_PATH_RED live in different folders:\n"
                f"  BLOCK_STACK_PATH_RED parent: {block_red_path.parent}\n"
                f"  INVIVO_PATH_RED parent:      {invivo_red_path.parent}\n"
                "Please set GRAPH_PATH manually in local_config.py."
            )

    # Prefer block, fall back to invivo
    anchor = block_red_path if block_red_path is not None else invivo_red_path
    parent = anchor.parent
    # NOT `subject_name`: that name is imported at the top of this module, and
    # assigning it anywhere in this function makes it function-local for the
    # WHOLE function -- which broke the ANALYSIS_ROOT branch above with
    # UnboundLocalError on every config that has ANALYSIS_ROOT set.
    subject = parent.name
    return parent / "alignment" / f"{subject}_graph.db"


def build_subslice_graph(
    force_rebuild: bool = False,
    save_every: int = 10,
    subslice_dirs: Optional[Iterable[Union[str, Path]]] = None,
    slices: Optional[Iterable[int]] = None,
    raw_channels: Optional[Iterable[str]] = None,
    dry_run: bool = False,
) -> Path:
    """
    Build alignment graph from whatever is configured in local_config.py.

    Reads INVIVO_PATH, BLOCK_STACK_PATH, SUBSLICE_DIR from config. For each:
    - blank (or missing attribute) → skip that node type
    - set but file/dir doesn't exist → hard error
    - set and exists → add to graph if not already a node

    Re-running after editing config augments the existing graph. Use
    force_rebuild=True to wipe and start over.

    Parameters
    ----------
    force_rebuild : bool
        If True, delete existing graph before building
    save_every : int
        Save subslice checkpoint every N subslices
    subslice_dirs : iterable of str or Path, optional
        Render folders to ingest, overriding config's SUBSLICE_DIR. Each is
        resolved the same way (relative names resolve under the preprocessing
        output roots), and each becomes its own node set because
        `subslice_node_name` appends the folder — so naming two cutoffs here
        puts both in one graph in a single run.
    slices : iterable of int, optional
        Restrict the add to these section numbers. None means every file.
    raw_channels : iterable of str, optional
        BARseq raw fluorescence channels to carry in beside the ALIGN renders
        (`["MSCARLET"]`), overriding config's SUBSLICE_RAW_CHANNELS. They attach
        to sections already holding an ALIGN node, so this can run against an
        existing graph with no SUBSLICE_DIR set. Pass `[]` to skip them.
    dry_run : bool
        Report what would be added and write nothing.

    Returns
    -------
    Path
        Path to saved graph (GRAPH_PATH from config)
    """
    # -------------------------------------------------------------
    # Resolve + validate config
    # -------------------------------------------------------------
    # Fail on a blank/unknown SCOPE now, not after loading a multi-GB stack.
    get_profile(SCOPE)

    invivo_red_path = Path(INVIVO_PATH_RED) if INVIVO_PATH_RED else None
    invivo_green_path = Path(INVIVO_PATH_GREEN) if INVIVO_PATH_GREEN else None
    block_red_path = Path(BLOCK_STACK_PATH_RED) if BLOCK_STACK_PATH_RED else None
    block_green_path = Path(BLOCK_STACK_PATH_GREEN) if BLOCK_STACK_PATH_GREEN else None
    # Relative SUBSLICE_DIR inherits preprocessing's overlay output dir and
    # names only the threshold folder; absolute is used verbatim; blank skips.
    # `subslice_dirs` overrides config so several cutoffs can be ingested in
    # one run; each folder is its own node set, so they coexist.
    requested = list(subslice_dirs) if subslice_dirs else [SUBSLICE_DIR]
    subslice_paths = [d for d in (resolve_subslice_dir(v) for v in requested) if d]
    channels = [c.upper() for c in (
        SUBSLICE_RAW_CHANNELS if raw_channels is None else raw_channels)]

    # Green-without-red on the same modality would dangle the Identity edge.
    if invivo_green_path and not invivo_red_path:
        raise ValueError(
            "INVIVO_PATH_GREEN is set but INVIVO_PATH_RED is blank.\n"
            "Green-without-red would dangle the Identity edge. Either set\n"
            "INVIVO_PATH_RED or leave INVIVO_PATH_GREEN blank."
        )
    if block_green_path and not block_red_path:
        raise ValueError(
            "BLOCK_STACK_PATH_GREEN is set but BLOCK_STACK_PATH_RED is blank.\n"
            "Green-without-red would dangle the Identity edge. Either set\n"
            "BLOCK_STACK_PATH_RED or leave BLOCK_STACK_PATH_GREEN blank."
        )

    # Configured-but-missing = hard error (catches typos)
    for label, p in [
        ("INVIVO_PATH_RED",       invivo_red_path),
        ("INVIVO_PATH_GREEN",     invivo_green_path),
        ("BLOCK_STACK_PATH_RED",  block_red_path),
        ("BLOCK_STACK_PATH_GREEN", block_green_path),
    ]:
        if p and not p.exists():
            raise FileNotFoundError(
                f"{label} is set in local_config.py but the file does not exist:\n"
                f"  {p}\n"
                f"Fix the path or leave {label} blank to skip that node."
            )
    for d in subslice_paths:
        if not d.is_dir():
            raise FileNotFoundError(
                f"Subslice folder is not a directory:\n"
                f"  {d}\n"
                f"Fix the path, or leave SUBSLICE_DIR blank to skip subslices."
            )

    any_invivo = bool(invivo_red_path or invivo_green_path)
    any_block = bool(block_red_path or block_green_path)
    if not (any_invivo or any_block or subslice_paths or channels):
        raise ValueError(
            "No inputs configured in local_config.py — set at least one of:\n"
            "  INVIVO_PATH_RED / INVIVO_PATH_GREEN          (in vivo 2P stack)\n"
            "  BLOCK_STACK_PATH_RED / BLOCK_STACK_PATH_GREEN (ex vivo block)\n"
            "  SUBSLICE_DIR                                  (BARseq subslices)"
        )

    # -------------------------------------------------------------
    # Resolve GRAPH_PATH (config value, or auto-derive from data)
    # -------------------------------------------------------------
    if GRAPH_PATH:
        output_path = Path(GRAPH_PATH)
        graph_path_source = "config"
    else:
        output_path = _derive_graph_path(block_red_path, invivo_red_path)
        graph_path_source = "derived"

    # -------------------------------------------------------------
    # Detect whether the alignment folder already exists, then create
    # -------------------------------------------------------------
    alignment_folder_existed = output_path.parent.exists()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GRAPH BUILDER")
    print("=" * 60)
    print(f"Graph: {output_path}")
    if graph_path_source == "derived":
        print(f"       (GRAPH_PATH not set in local_config.py — auto-derived.)")
        print(f"       (Set GRAPH_PATH manually to pin to a different location.)")
    if alignment_folder_existed:
        print(f"Alignment folder: exists")
    else:
        print(f"Alignment folder: created (did not exist)")
    print()
    print(f"Inputs:")
    print(f"  invivo  red:   {invivo_red_path   if invivo_red_path   else '(not set, skipping)'}")
    print(f"  invivo  green: {invivo_green_path if invivo_green_path else '(not set, skipping)'}")
    print(f"  block   red:   {block_red_path    if block_red_path    else '(not set, skipping)'}")
    print(f"  block   green: {block_green_path  if block_green_path  else '(not set, skipping)'}")
    if subslice_paths:
        for i, d in enumerate(subslice_paths):
            print(f"  subslices:     {d}" if i == 0 else f"                 {d}")
    else:
        print(f"  subslices:     (not set, skipping)")
    print(f"  raw channels:  "
          f"{', '.join(channels) + f'  (hub: {channels[0]})' if channels else '(none, skipping)'}")
    if dry_run:
        print()
        print("  DRY RUN — nothing will be written")
    print()

    # -------------------------------------------------------------
    # Orientation codes (optional) — read before anything large is
    # loaded, so a mirror between two modalities stops the run early.
    # -------------------------------------------------------------
    orientation_codes = load_orientation_codes()
    configured_modalities = (
        (["invivo"] if any_invivo else [])
        + (["block_stack"] if any_block else [])
        + ([SUBSLICE_MODALITY] if subslice_paths else [])
    )
    check_orientation_handedness(orientation_codes, configured_modalities)
    print()

    # -------------------------------------------------------------
    # Load or create graph
    # -------------------------------------------------------------
    if force_rebuild and output_path.exists():
        if dry_run:
            print("Force rebuild — WOULD DELETE the existing graph")
        else:
            print("Force rebuild — deleting existing graph")
            output_path.unlink()

    if output_path.exists():
        print("Existing graph found — augmenting with any missing nodes")
        g = load_graph(output_path)
    else:
        print("Creating new graph")
        g = create_subslice_graph()

    # -------------------------------------------------------------
    # Migration guard: legacy non-suffixed nodes from the pre-multi-channel
    # schema must be migrated explicitly so the rigid fits aren't silently
    # orphaned by the rename.
    # -------------------------------------------------------------
    legacy_present = LEGACY_NODE_NAMES & set(g.nodes)
    if legacy_present:
        raise ValueError(
            "Loaded graph contains node names this builder no longer writes:\n"
            f"  {sorted(legacy_present)}\n\n"
            "Either the pre-multi-channel schema (no _red / _green suffix)\n"
            "or the pre-2026-08-30 2P names, which were invivo_ref_* and\n"
            "ex_vivo_block_* before they were renamed to match\n"
            "local_config: invivo_* and block_stack_*.\n\n"
            "Rebuild with `force_rebuild=True` to migrate. Fits on the\n"
            "renamed nodes must be redone (minutes via Mode D in the\n"
            "notebook); subslice fits go too.\n\n"
            "    build_subslice_graph(force_rebuild=True)"
        )

    existing_nodes = set(g.nodes)

    # -------------------------------------------------------------
    # Add in-vivo channels
    # -------------------------------------------------------------
    if any_invivo:
        print("\n1. Adding in-vivo channels")
        print("-" * 60)

        red_stack = green_stack = None
        red_spacing = green_spacing = None

        if invivo_red_path:
            if "invivo_red" in existing_nodes:
                print(f"  invivo_red already in graph — skipping")
            elif dry_run:
                print(f"  would ADD: invivo_red")
            else:
                red_stack = load_invivo_stack(invivo_red_path)
                red_spacing, _ = spacing_for_tiff(invivo_red_path)

        if invivo_green_path:
            if "invivo_green" in existing_nodes:
                print(f"  invivo_green already in graph — skipping")
            elif dry_run:
                print(f"  would ADD: invivo_green")
            else:
                green_stack = load_invivo_stack(invivo_green_path)
                green_spacing, _ = spacing_for_tiff(invivo_green_path)

        if red_stack is not None or green_stack is not None:
            add_invivo_to_graph(
                g,
                red_stack=red_stack, green_stack=green_stack,
                red_spacing=red_spacing, green_spacing=green_spacing,
                orientation_code=orientation_codes.get("invivo"),
            )
            del red_stack, green_stack
            save_graph(g, output_path, verbose=False)

    # -------------------------------------------------------------
    # Add ex-vivo block channels
    # -------------------------------------------------------------
    if any_block:
        print("\n2. Adding ex-vivo block channels")
        print("-" * 60)

        red_stack = green_stack = None
        red_spacing = green_spacing = None

        if block_red_path:
            if "block_stack_red" in existing_nodes:
                print(f"  block_stack_red already in graph — skipping")
            elif dry_run:
                print(f"  would ADD: block_stack_red")
            else:
                red_stack = load_block_stack(block_red_path)
                red_spacing, _ = spacing_for_tiff(block_red_path)

        if block_green_path:
            if "block_stack_green" in existing_nodes:
                print(f"  block_stack_green already in graph — skipping")
            elif dry_run:
                print(f"  would ADD: block_stack_green")
            else:
                green_stack = load_block_stack(block_green_path)
                green_spacing, _ = spacing_for_tiff(block_green_path)

        if red_stack is not None or green_stack is not None:
            add_block_to_graph(
                g,
                red_stack=red_stack, green_stack=green_stack,
                red_spacing=red_spacing, green_spacing=green_spacing,
                orientation_code=orientation_codes.get("block_stack"),
            )
            del red_stack, green_stack
            save_graph(g, output_path, verbose=False)

    for d in subslice_paths:
        print("\n3. Adding BARseq subslices")
        print("-" * 60)
        print(f"  From: {d}")
        add_subslices_to_graph(
            g,
            subslice_dir=d,
            save_every=save_every,
            output_path=None if dry_run else output_path,
            orientation_code=orientation_codes.get(SUBSLICE_MODALITY),
            slices=slices,
            dry_run=dry_run,
        )

    if channels:
        print("\n4. Adding BARseq raw channels")
        print("-" * 60)
        hyb_dir = hyb_downsampled_dir()
        print(f"  From: {hyb_dir}")
        add_raw_channels_to_graph(
            g,
            hyb_dir=hyb_dir,
            channels=channels,
            save_every=save_every,
            output_path=None if dry_run else output_path,
            orientation_code=orientation_codes.get(SUBSLICE_MODALITY),
            slices=slices,
            dry_run=dry_run,
        )

    # -------------------------------------------------------------
    # Final save + summary
    # -------------------------------------------------------------
    if dry_run:
        print("\nDry run — nothing written")
        print("-" * 60)
    else:
        print("\nSaving final graph")
        print("-" * 60)
        save_graph(g, output_path)

    print("\n" + "=" * 60)
    print("GRAPH BUILD COMPLETE")
    print("=" * 60)
    print(f"Graph: {output_path}")
    print(f"Total nodes: {len(g.nodes)}")
    n_anchor = 0
    for label, present in [
        ("invivo_red",      "invivo_red" in g.nodes),
        ("invivo_green",    "invivo_green" in g.nodes),
        ("block_stack_red",   "block_stack_red" in g.nodes),
        ("block_stack_green", "block_stack_green" in g.nodes),
    ]:
        if present:
            print(f"  - {label}")
            n_anchor += 1
    n_align = sum(len(v) for v in align_nodes_by_slice(g).values())
    if n_align:
        print(f"  - {n_align} ex vivo subslices")
    n_raw = len(g.nodes) - n_anchor - n_align
    if n_raw:
        print(f"  - {n_raw} BARseq raw channel nodes")
    print(f"\nReady for alignment in CASTalign!")

    return output_path


# ============================================
# Usage
# ============================================

def main():
    ap = argparse.ArgumentParser(
        description="Build or augment the alignment graph from local_config.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Default (no flags) AUGMENTS: every node already in the graph is left alone and
anything missing is added, reading SUBSLICE_DIR from local_config.py.

A subslice node is named `{stem}_{render folder}`, so each rolony cutoff is its
own node set and several coexist in one graph. Naming folders here overrides
the config value and can ingest more than one in a single run:

    python alignment/subslice_graph_builder.py                       # config's SUBSLICE_DIR
    python alignment/subslice_graph_builder.py -d qc20_5_ge3 -d qc20_5_ge5
    python alignment/subslice_graph_builder.py -d qc20_5_ge5 --slices 1 22
    python alignment/subslice_graph_builder.py --raw-channels MSCARLET
    python alignment/subslice_graph_builder.py --dry-run             # report, write nothing

--force-rebuild WIPES the .db and every fitted edge, including the hand-made
block_stack_red -> invivo_red fit, which exists nowhere else. Back it up first.
""")
    ap.add_argument("-d", "--subslice-dir", action="append", default=None,
                    metavar="DIR",
                    help="render folder to ingest (repeatable); relative names "
                         "resolve like SUBSLICE_DIR. Default: config's value")
    ap.add_argument("--slices", type=int, nargs="+", default=None, metavar="N",
                    help="only these section numbers (default: every file)")
    ap.add_argument("--raw-channels", nargs="+", default=None, metavar="NAME",
                    help="BARseq raw channels to carry in (MSCARLET GCAMP DAPI); "
                         "the first is the section's Identity hub. "
                         "Default: config's SUBSLICE_RAW_CHANNELS")
    ap.add_argument("--no-raw-channels", action="store_true",
                    help="skip raw channels even when config asks for them")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be added; write nothing")
    ap.add_argument("--force-rebuild", action="store_true",
                    help="DESTRUCTIVE: delete the graph and every fitted edge first")
    ap.add_argument("--save-every", type=int, default=10, metavar="N",
                    help="checkpoint every N subslices (default: 10)")
    args = ap.parse_args()

    build_subslice_graph(
        force_rebuild=args.force_rebuild,
        save_every=args.save_every,
        subslice_dirs=args.subslice_dir,
        slices=args.slices,
        raw_channels=[] if args.no_raw_channels else args.raw_channels,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
