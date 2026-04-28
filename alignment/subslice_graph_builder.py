"""
Subslice Graph Builder - LineStuffUp Alignment Graph for Anisotropic Subslices

Builds alignment graph using anisotropically-downsampled ex vivo coronal subslices
to match in vivo 2-photon stack resolution.

Key Resolution Matching:
- Ex vivo subslices: Anisotropic scaling to match in vivo physically
  - X dimension: 2.34 µm/px (matches in vivo X/Y)
  - Y dimension: 1.0 µm/px (matches in vivo Z)

- In vivo stack:
  - X/Y: 2.34 µm/px (512 px / 1200 µm)
  - Z: 1.0 µm/px (399 px / 399 µm)

After xrotate=90:
- Ex vivo Y (Dorsal-Ventral) → In vivo Z
- Ex vivo X (Lateral-Medial) → In vivo X

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

# Legacy node names from the pre-multi-channel schema. Presence of either in a
# loaded graph triggers the migration guard in build_subslice_graph().
LEGACY_NODE_NAMES = {"invivo_ref", "ex_vivo_block"}

import castalign as ca
import numpy as np
import imageio.v2 as imageio
from utilities.image_io import get_tiff_resolution


# ============================================
# Configuration
# ============================================

# Resolution parameters
INVIVO_XY_UM_PER_PX = 2.34   # In vivo X/Y resolution
INVIVO_Z_UM_PER_PX = 1.0     # In vivo Z resolution
EXVIVO_X_UM_PER_PX = 2.34    # Anisotropic ex vivo X (matches in vivo X/Y)
EXVIVO_Y_UM_PER_PX = 1.0     # Anisotropic ex vivo Y (matches in vivo Z)

# Known microscope profiles for resolution autodetection.
# XY resolution identifies the microscope; Z comes from metadata or falls back to default.
# To add a new microscope, add an entry here.
MICROSCOPE_PROFILES = {
    'li_lab': {
        'xy_um_per_px': 2.34,       # 512 px / 1200 µm FOV
        'z_um_per_px': 1.0,
        'description': 'Li lab 2P (1200 µm FOV)',
    },
    'huang_lab': {
        'xy_um_per_px': 1.1055,     # 512 px / 566.08 µm FOV
        'z_um_per_px': 2.0,
        'description': 'Huang lab 2P (566.08 µm FOV)',
    },
}


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


def discover_aniso_subslices(
    directory: Union[str, Path]
) -> List[Path]:
    """
    Discover available anisotropic subslice files.

    Returns
    -------
    List[Path]
        Sorted list of subslice file paths
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Anisotropic subslice directory not found: {directory}")

    files = sorted(directory.glob("slice*_subslice_mScarlet_cellmask.tif"))
    print(f"Found {len(files)} anisotropic subslice files")

    return files


def load_single_subslice(path: Union[str, Path]) -> np.ndarray:
    """
    Load a single anisotropic subslice.

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


def detect_spacing(
    tiff_path: Union[str, Path],
    tolerance: float = 0.05
) -> tuple:
    """
    Autodetect microscope from TIFF metadata and return spacing.

    Reads XY resolution from TIFF tags, matches against MICROSCOPE_PROFILES.

    Parameters
    ----------
    tiff_path : Path
        Path to TIFF stack
    tolerance : float
        Relative tolerance for XY matching (default 5%)

    Returns
    -------
    tuple
        ((z, y, x) spacing in µm/px, microscope_name)

    Raises
    ------
    ValueError
        If no resolution metadata or no matching microscope profile
    """
    res = get_tiff_resolution(tiff_path)

    if res['xy_um_per_px'] is None:
        raise ValueError(
            f"No XY resolution metadata in: {Path(tiff_path).name}\n"
            "Add resolution in ImageJ (Image > Properties) or set spacing manually."
        )

    xy = res['xy_um_per_px']
    z = res['z_um_per_px']

    for name, profile in MICROSCOPE_PROFILES.items():
        expected_xy = profile['xy_um_per_px']
        if abs(xy - expected_xy) / expected_xy < tolerance:
            if z is None:
                z = profile['z_um_per_px']
                print(f"  Z spacing not in metadata, using {name} default: {z} µm/px")
            print(f"  Detected microscope: {name} ({profile['description']})")
            print(f"  Spacing (Z, Y, X): ({z}, {xy:.4f}, {xy:.4f}) µm/px")
            return (z, xy, xy), name

    profile_lines = "\n".join(
        f"    '{name}': XY = {p['xy_um_per_px']} µm/px ({p['description']})"
        for name, p in MICROSCOPE_PROFILES.items()
    )
    raise ValueError(
        f"\n{'='*60}\n"
        f"Could not identify microscope from image resolution.\n\n"
        f"  Your image:  {Path(tiff_path).name}\n"
        f"  XY detected: {xy:.4f} µm/px\n\n"
        f"This doesn't match any known microscope:\n"
        f"{profile_lines}\n\n"
        f"To fix, add your microscope to MICROSCOPE_PROFILES\n"
        f"in {Path(__file__).name}:\n\n"
        f"    'your_scope_name': {{\n"
        f"        'xy_um_per_px': {xy:.4f},\n"
        f"        'z_um_per_px': <your Z spacing in µm>,\n"
        f"        'description': '<scope name> (<FOV size> FOV)',\n"
        f"    }},\n"
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
    default_spacing = (INVIVO_Z_UM_PER_PX, INVIVO_XY_UM_PER_PX, INVIVO_XY_UM_PER_PX)

    if red_stack is not None and red_node not in graph.nodes:
        spacing = red_spacing if red_spacing is not None else default_spacing
        graph.add_node(
            red_node, image=red_stack, compression="high",
            metadata={'spacing': spacing},
        )
        print(f"  Added node: {red_node}  shape {red_stack.shape}  spacing {spacing} µm/px (Z, Y, X)")

    if green_stack is not None and green_node not in graph.nodes:
        spacing = green_spacing if green_spacing is not None else default_spacing
        graph.add_node(
            green_node, image=green_stack, compression="high",
            metadata={'spacing': spacing},
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
    base_name: str = "invivo_ref",
) -> ca.Graph:
    """Add in-vivo red/green nodes joined by Identity. See `_add_volume_channels`."""
    print(f"Adding in-vivo channels under base '{base_name}'...")
    return _add_volume_channels(
        graph, base_name,
        red_stack=red_stack, green_stack=green_stack,
        red_spacing=red_spacing, green_spacing=green_spacing,
    )


def add_block_to_graph(
    graph: ca.Graph,
    *,
    red_stack: Optional[np.ndarray] = None,
    green_stack: Optional[np.ndarray] = None,
    red_spacing: Optional[tuple] = None,
    green_spacing: Optional[tuple] = None,
    base_name: str = "ex_vivo_block",
) -> ca.Graph:
    """Add ex-vivo block red/green nodes joined by Identity. See `_add_volume_channels`."""
    print(f"Adding ex-vivo block channels under base '{base_name}'...")
    return _add_volume_channels(
        graph, base_name,
        red_stack=red_stack, green_stack=green_stack,
        red_spacing=red_spacing, green_spacing=green_spacing,
    )


def _assert_spacing_match(
    spacing_a: tuple,
    spacing_b: tuple,
    label_a: str,
    label_b: str,
    tolerance: float = 0.02,
) -> None:
    """Raise if two (Z, Y, X) spacings differ by more than `tolerance` (relative)."""
    a = np.asarray(spacing_a, dtype=float)
    b = np.asarray(spacing_b, dtype=float)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError(f"Expected (Z, Y, X) spacing tuples, got {spacing_a!r} / {spacing_b!r}")
    rel_diff = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
    if np.any(rel_diff > tolerance):
        raise ValueError(
            f"Spacing mismatch between {label_a} and {label_b} (>{tolerance:.0%} relative):\n"
            f"  {label_a}: {tuple(a)} µm/px\n"
            f"  {label_b}: {tuple(b)} µm/px\n"
            "Red and green channels of one volume must share a voxel grid. "
            "Check that the two TIFFs come from the same acquisition."
        )


def add_subslices_to_graph(
    graph: ca.Graph,
    subslice_dir: Union[str, Path],
    save_every: int = 10,
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> ca.Graph:
    """
    Add all anisotropic subslices to graph.

    Subslice spacing: (Z, Y, X) = (20.0, 1.0, 2.34) µm/px
    - Z: 20 µm physical slice thickness
    - Y: 1.0 µm/px (matches in vivo Z)
    - X: 2.34 µm/px (matches in vivo X/Y)

    NOTE: The subslices are CORONAL (ex vivo), so after xrotate=90:
    - Ex vivo Y → In vivo Z
    - Ex vivo X → In vivo X
    """
    files = discover_aniso_subslices(subslice_dir)

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

    # Subslice spacing: anisotropic to match in vivo after rotation
    # (Z, Y, X) where Z is slice thickness, Y/X are the pixel spacings
    metadata = {
        'spacing': (20.0, EXVIVO_Y_UM_PER_PX, EXVIVO_X_UM_PER_PX),  # (Z, Y, X) µm/px
        'anisotropic': True,
        'note': 'Y matches in_vivo Z, X matches in_vivo XY'
    }

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

    Rule: ``<parent_of_data_file>/alignment/<parent_folder_name>_graph.db``

    Anchored on the red-channel paths (red is canonical, green is sibling).
    Prefers BLOCK_STACK_PATH_RED, falls back to INVIVO_PATH_RED. Raises
    ValueError if neither is set, or if both are set but live in different
    parent folders.
    """
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
    # Resolve + validate data paths
    # -------------------------------------------------------------
    invivo_red_path = Path(INVIVO_PATH_RED) if INVIVO_PATH_RED else None
    invivo_green_path = Path(INVIVO_PATH_GREEN) if INVIVO_PATH_GREEN else None
    block_red_path = Path(BLOCK_STACK_PATH_RED) if BLOCK_STACK_PATH_RED else None
    block_green_path = Path(BLOCK_STACK_PATH_GREEN) if BLOCK_STACK_PATH_GREEN else None
    subslice_dir = Path(SUBSLICE_DIR) if SUBSLICE_DIR else None

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
            f"SUBSLICE_DIR is set in local_config.py but is not a directory:\n"
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
                red_spacing, _ = detect_spacing(invivo_red_path)

        if invivo_green_path:
            if "invivo_ref_green" in existing_nodes:
                print(f"  invivo_ref_green already in graph — skipping")
            else:
                green_stack = load_invivo_stack(invivo_green_path)
                green_spacing, _ = detect_spacing(invivo_green_path)
                # Sanity check: green must share the red channel's voxel grid.
                # When red was just loaded, compare to red_spacing; when red is
                # already a graph node, re-detect from the red TIFF (cheap —
                # metadata only).
                ref_spacing = red_spacing
                if ref_spacing is None and invivo_red_path:
                    ref_spacing, _ = detect_spacing(invivo_red_path)
                if ref_spacing is not None:
                    _assert_spacing_match(ref_spacing, green_spacing,
                                          "INVIVO_PATH_RED", "INVIVO_PATH_GREEN")

        if red_stack is not None or green_stack is not None:
            add_invivo_to_graph(
                g,
                red_stack=red_stack, green_stack=green_stack,
                red_spacing=red_spacing, green_spacing=green_spacing,
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
                red_spacing, _ = detect_spacing(block_red_path)

        if block_green_path:
            if "ex_vivo_block_green" in existing_nodes:
                print(f"  ex_vivo_block_green already in graph — skipping")
            else:
                green_stack = load_block_stack(block_green_path)
                green_spacing, _ = detect_spacing(block_green_path)
                ref_spacing = red_spacing
                if ref_spacing is None and block_red_path:
                    ref_spacing, _ = detect_spacing(block_red_path)
                if ref_spacing is not None:
                    _assert_spacing_match(ref_spacing, green_spacing,
                                          "BLOCK_STACK_PATH_RED", "BLOCK_STACK_PATH_GREEN")

        if red_stack is not None or green_stack is not None:
            add_block_to_graph(
                g,
                red_stack=red_stack, green_stack=green_stack,
                red_spacing=red_spacing, green_spacing=green_spacing,
            )
            del red_stack, green_stack
            save_graph(g, output_path, verbose=False)

    if subslice_dir:
        print("\n3. Adding anisotropic subslices")
        print("-" * 60)
        add_subslices_to_graph(
            g,
            subslice_dir=subslice_dir,
            save_every=save_every,
            output_path=output_path,
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
