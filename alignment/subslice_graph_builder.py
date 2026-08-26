"""
Subslice Graph Builder - LineStuffUp Alignment Graph for BARseq Subslices

Builds the alignment graph from the 2P volumes named in local_config.py plus the
downsampled BARseq subslices produced by preprocessing/.

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

import sys
from pathlib import Path
from typing import Optional, Union, List

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

# The microscope that acquired this subject's data — the source of truth for
# every pixel size below. Validated at first use, not at import, so a module
# import (or --help) does not require a complete config.
SCOPE = getattr(local_config, "SCOPE", "")

# Legacy node names from the pre-multi-channel schema. Presence of either in a
# loaded graph triggers the migration guard in build_subslice_graph().
LEGACY_NODE_NAMES = {"invivo_ref", "ex_vivo_block"}

import castalign as ca
import numpy as np
import imageio.v2 as imageio
from utilities.image_io import get_tiff_resolution
import orientation
from analysis_paths import (
    analysis_subdir,
    resolve_subslice_dir,
    subject_name,
    ALIGNMENT_SUBDIR,
)
from scope_profiles import (
    MICROSCOPE_PROFILES,
    MAX_PLAUSIBLE_XY_UM_PER_PX,
    assert_matches_metadata,
    get_profile,
    identify_scope,
    is_plausible_xy,
    spacing_zyx,
    subslice_spacing_zyx,
    unknown_scope_message,
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
    # preprocessing/generate_alignment_tif.py. The step 4 overlay is the older
    # source and stays as a fallback -- it is an RGB display figure, so
    # load_single_subslice() collapses it by BT.601 luminance, which on BY95
    # renders the median marker cell DARKER than the mask field behind it.
    files = sorted(directory.glob("slice*_subslice_ALIGN.tif"))
    if files:
        print(f"Found {len(files)} BARseq alignment TIFs")
        return files

    files = sorted(directory.glob("slice*_subslice_mScarlet_cellmask.tif"))
    if files:
        print(f"Found {len(files)} BARseq subslice files (step 4 overlays)")
        print("  No slice*_subslice_ALIGN.tif here. These are RGB display")
        print("  overlays and fit worse; run preprocessing/generate_alignment_tif.py")
        print("  and point SUBSLICE_DIR at its output to use the intended image.")
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


def detect_spacing(tiff_path: Union[str, Path], tolerance: float = 0.05) -> tuple:
    """
    Read (Z, Y, X) spacing from a TIFF's own resolution metadata.

    Kept for inspecting what a file claims. It is NOT how node spacing is
    decided — that is `spacing_for_tiff`, which uses SCOPE and only compares
    against this. Retained because `validate_mnn.py` and the notebook use it to
    report a file's declared pitch.

    Returns
    -------
    tuple
        ((z, y, x) spacing in µm/px, microscope_name)

    Raises
    ------
    ValueError
        - "No XY resolution metadata ..." when the TIFF has no XY calibration
        - "Uncalibrated XY resolution ..." when XY exceeds
          MAX_PLAUSIBLE_XY_UM_PER_PX (a DPI default, not a real scope)
        - "Could not identify microscope ..." when XY is plausible but matches
          no profile (a genuine new scope to add)
    """
    res = get_tiff_resolution(tiff_path)

    if res['xy_um_per_px'] is None:
        raise ValueError(
            f"No XY resolution metadata in: {Path(tiff_path).name}\n"
            "Add resolution in ImageJ (Image > Properties) or set spacing manually."
        )

    xy = res['xy_um_per_px']
    z = res['z_um_per_px']

    if not is_plausible_xy(xy):
        raise ValueError(
            f"Uncalibrated XY resolution in {Path(tiff_path).name}: detected "
            f"{xy:.4f} µm/px (> {MAX_PLAUSIBLE_XY_UM_PER_PX} µm/px plausibility "
            f"ceiling). This is almost certainly a screen/print DPI default "
            f"(e.g. 72 DPI → 352.78 µm/px), not a real microscope calibration."
        )

    name = identify_scope(xy, tolerance)
    if name is None:
        raise ValueError(unknown_scope_message(xy, Path(tiff_path).name))

    if z is None:
        z = MICROSCOPE_PROFILES[name]['z_um_per_px']
    return (z, xy, xy), name


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
    base_name: str = "invivo_ref",
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
    base_name: str = "ex_vivo_block",
) -> ca.Graph:
    """Add ex-vivo block red/green nodes joined by Identity. See `_add_volume_channels`."""
    print(f"Adding ex-vivo block channels under base '{base_name}'...")
    return _add_volume_channels(
        graph, base_name,
        red_stack=red_stack, green_stack=green_stack,
        red_spacing=red_spacing, green_spacing=green_spacing,
        orientation_code=orientation_code,
    )


def add_subslices_to_graph(
    graph: ca.Graph,
    subslice_dir: Union[str, Path],
    save_every: int = 10,
    output_path: Optional[Union[str, Path]] = None,
    orientation_code: Optional[str] = None,
    verbose: bool = True
) -> ca.Graph:
    """
    Add all downsampled BARseq subslices to graph.

    Subslice spacing: (Z, Y, X) = (20.0, xy, xy) µm/px, where xy is SCOPE's
    in-plane pitch — the same pitch preprocessing resampled the images to.
    Z is the physical section thickness, not a pixel size.
    """
    spacing = subslice_spacing_zyx(SCOPE)

    files = discover_subslices(subslice_dir)

    # Check which are already in graph
    existing_nodes = set(graph.nodes)
    files_to_add = []

    for f in files:
        node_name = f.stem  # e.g., "slice10_subslice_mScarlet_overlay_DAPI"
        if node_name not in existing_nodes:
            files_to_add.append((f, node_name))

    if verbose:
        print(f"\nSubslice Loading:")
        print(f"  Total files: {len(files)}")
        print(f"  Already in graph: {len(files) - len(files_to_add)}")
        print(f"  To add: {len(files_to_add)}")

    if len(files_to_add) == 0:
        print("All subslices already in graph!")
        return graph

    # (Z, Y, X) µm/px, plus the one orientation every subslice shares.
    metadata = {'spacing': spacing, 'orientation': orientation_code}

    added = 0
    for i, (fpath, node_name) in enumerate(files_to_add, 1):
        try:
            img = load_single_subslice(fpath)
            graph.add_node(node_name, image=img, compression="normal", metadata=metadata)
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
    subject_name = parent.name
    return parent / "alignment" / f"{subject_name}_graph.db"


def build_subslice_graph(
    force_rebuild: bool = False,
    save_every: int = 10
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
    subslice_dir = resolve_subslice_dir(SUBSLICE_DIR)

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
    if subslice_dir and not subslice_dir.is_dir():
        raise FileNotFoundError(
            f"SUBSLICE_DIR resolved to something that is not a directory:\n"
            f"  {subslice_dir}\n"
            f"Fix the path or leave SUBSLICE_DIR blank to skip subslices."
        )

    any_invivo = bool(invivo_red_path or invivo_green_path)
    any_block = bool(block_red_path or block_green_path)
    if not (any_invivo or any_block or subslice_dir):
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
    print(f"  subslices:     {subslice_dir      if subslice_dir      else '(not set, skipping)'}")
    print()

    # -------------------------------------------------------------
    # Orientation codes (optional) — read before anything large is
    # loaded, so a mirror between two modalities stops the run early.
    # -------------------------------------------------------------
    orientation_codes = load_orientation_codes()
    configured_modalities = (
        (["invivo_ref"] if any_invivo else [])
        + (["ex_vivo_block"] if any_block else [])
        + ([SUBSLICE_MODALITY] if subslice_dir else [])
    )
    check_orientation_handedness(orientation_codes, configured_modalities)
    print()

    # -------------------------------------------------------------
    # Load or create graph
    # -------------------------------------------------------------
    if force_rebuild and output_path.exists():
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
            "Loaded graph contains legacy non-suffixed nodes from the\n"
            "pre-multi-channel schema:\n"
            f"  {sorted(legacy_present)}\n\n"
            "Rebuild with `force_rebuild=True` to migrate to the new\n"
            "_red / _green naming. Rigid fits will need to be redone\n"
            "(typically minutes via Mode D in the notebook).\n\n"
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
            if "invivo_ref_red" in existing_nodes:
                print(f"  invivo_ref_red already in graph — skipping")
            else:
                red_stack = load_invivo_stack(invivo_red_path)
                red_spacing, _ = spacing_for_tiff(invivo_red_path)

        if invivo_green_path:
            if "invivo_ref_green" in existing_nodes:
                print(f"  invivo_ref_green already in graph — skipping")
            else:
                green_stack = load_invivo_stack(invivo_green_path)
                green_spacing, _ = spacing_for_tiff(invivo_green_path)

        if red_stack is not None or green_stack is not None:
            add_invivo_to_graph(
                g,
                red_stack=red_stack, green_stack=green_stack,
                red_spacing=red_spacing, green_spacing=green_spacing,
                orientation_code=orientation_codes.get("invivo_ref"),
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
            if "ex_vivo_block_red" in existing_nodes:
                print(f"  ex_vivo_block_red already in graph — skipping")
            else:
                red_stack = load_block_stack(block_red_path)
                red_spacing, _ = spacing_for_tiff(block_red_path)

        if block_green_path:
            if "ex_vivo_block_green" in existing_nodes:
                print(f"  ex_vivo_block_green already in graph — skipping")
            else:
                green_stack = load_block_stack(block_green_path)
                green_spacing, _ = spacing_for_tiff(block_green_path)

        if red_stack is not None or green_stack is not None:
            add_block_to_graph(
                g,
                red_stack=red_stack, green_stack=green_stack,
                red_spacing=red_spacing, green_spacing=green_spacing,
                orientation_code=orientation_codes.get("ex_vivo_block"),
            )
            del red_stack, green_stack
            save_graph(g, output_path, verbose=False)

    if subslice_dir:
        print("\n3. Adding BARseq subslices")
        print("-" * 60)
        add_subslices_to_graph(
            g,
            subslice_dir=subslice_dir,
            save_every=save_every,
            output_path=output_path,
            orientation_code=orientation_codes.get(SUBSLICE_MODALITY),
        )

    # -------------------------------------------------------------
    # Final save + summary
    # -------------------------------------------------------------
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
        ("invivo_ref_red",      "invivo_ref_red" in g.nodes),
        ("invivo_ref_green",    "invivo_ref_green" in g.nodes),
        ("ex_vivo_block_red",   "ex_vivo_block_red" in g.nodes),
        ("ex_vivo_block_green", "ex_vivo_block_green" in g.nodes),
    ]:
        if present:
            print(f"  - {label}")
            n_anchor += 1
    if subslice_dir:
        print(f"  - {len(g.nodes) - n_anchor} ex vivo subslices")
    print(f"\nReady for alignment in CASTalign!")

    return output_path


# ============================================
# Usage
# ============================================

if __name__ == "__main__":
    build_subslice_graph()
