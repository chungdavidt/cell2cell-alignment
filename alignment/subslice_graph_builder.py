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

# Optional data inputs — blank string or missing attribute = skip that node
INVIVO_PATH = getattr(local_config, "INVIVO_PATH", "")
BLOCK_STACK_PATH = getattr(local_config, "BLOCK_STACK_PATH", "")
SUBSLICE_DIR = getattr(local_config, "SUBSLICE_DIR", "")

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


def add_invivo_to_graph(
    graph: ca.Graph,
    invivo_stack: np.ndarray,
    node_name: str = "invivo_ref",
    spacing: Optional[tuple] = None
) -> ca.Graph:
    """
    Add in vivo reference to graph with correct spacing metadata.

    Parameters
    ----------
    spacing : tuple, optional
        (Z, Y, X) in µm/px. If None, uses hardcoded Li lab defaults.
    """
    print(f"Adding in vivo reference '{node_name}' to graph...")

    if spacing is None:
        spacing = (INVIVO_Z_UM_PER_PX, INVIVO_XY_UM_PER_PX, INVIVO_XY_UM_PER_PX)
    metadata = {'spacing': spacing}

    graph.add_node(node_name, image=invivo_stack, compression="high", metadata=metadata)

    print(f"  Added: {invivo_stack.shape}")
    print(f"  Spacing: {metadata['spacing']} µm/px (Z, Y, X)")

    return graph


def add_block_to_graph(
    graph: ca.Graph,
    block_stack: np.ndarray,
    node_name: str = "ex_vivo_block",
    spacing: Optional[tuple] = None
) -> ca.Graph:
    """
    Add ex-vivo block volume to graph.

    Parameters
    ----------
    spacing : tuple, optional
        (Z, Y, X) in µm/px. If None, uses hardcoded Li lab defaults.
    """
    print(f"Adding ex-vivo block '{node_name}' to graph...")

    if spacing is None:
        spacing = (INVIVO_Z_UM_PER_PX, INVIVO_XY_UM_PER_PX, INVIVO_XY_UM_PER_PX)
    metadata = {'spacing': spacing}

    graph.add_node(node_name, image=block_stack, compression="high", metadata=metadata)

    print(f"  Added: {block_stack.shape}")
    print(f"  Spacing: {metadata['spacing']} µm/px (Z, Y, X)")

    return graph


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
    block_path: Optional[Path],
    invivo_path: Optional[Path],
) -> Path:
    """
    Derive a default GRAPH_PATH when not set in local_config.py.

    Rule: ``<parent_of_data_file>/alignment/<parent_folder_name>_graph.db``

    Prefers BLOCK_STACK_PATH, falls back to INVIVO_PATH. Raises ValueError
    if neither is set, or if both are set but live in different parent
    folders (which would make the "one graph per subject" convention
    ambiguous — in that case, set GRAPH_PATH manually in local_config.py).
    """
    if block_path is None and invivo_path is None:
        raise ValueError(
            "Cannot auto-derive GRAPH_PATH: neither BLOCK_STACK_PATH nor\n"
            "INVIVO_PATH is set. If you're running a BARseq-only graph\n"
            "(SUBSLICE_DIR only), please set GRAPH_PATH manually in\n"
            "local_config.py."
        )

    if block_path is not None and invivo_path is not None:
        if block_path.parent != invivo_path.parent:
            raise ValueError(
                "Cannot auto-derive GRAPH_PATH: BLOCK_STACK_PATH and\n"
                "INVIVO_PATH live in different folders:\n"
                f"  BLOCK_STACK_PATH parent: {block_path.parent}\n"
                f"  INVIVO_PATH parent:      {invivo_path.parent}\n"
                "Please set GRAPH_PATH manually in local_config.py."
            )

    # Prefer block, fall back to invivo
    anchor = block_path if block_path is not None else invivo_path
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
    invivo_path = Path(INVIVO_PATH) if INVIVO_PATH else None
    block_path = Path(BLOCK_STACK_PATH) if BLOCK_STACK_PATH else None
    subslice_dir = Path(SUBSLICE_DIR) if SUBSLICE_DIR else None

    # Configured-but-missing = hard error (catches typos)
    if invivo_path and not invivo_path.exists():
        raise FileNotFoundError(
            f"INVIVO_PATH is set in local_config.py but the file does not exist:\n"
            f"  {invivo_path}\n"
            f"Fix the path or leave INVIVO_PATH blank to skip the in vivo node."
        )
    if block_path and not block_path.exists():
        raise FileNotFoundError(
            f"BLOCK_STACK_PATH is set in local_config.py but the file does not exist:\n"
            f"  {block_path}\n"
            f"Fix the path or leave BLOCK_STACK_PATH blank to skip the block node."
        )
    if subslice_dir and not subslice_dir.is_dir():
        raise FileNotFoundError(
            f"SUBSLICE_DIR is set in local_config.py but is not a directory:\n"
            f"  {subslice_dir}\n"
            f"Fix the path or leave SUBSLICE_DIR blank to skip subslices."
        )

    if not (invivo_path or block_path or subslice_dir):
        raise ValueError(
            "No inputs configured in local_config.py — set at least one of:\n"
            "  INVIVO_PATH         (in vivo 2P stack)\n"
            "  BLOCK_STACK_PATH    (ex vivo block)\n"
            "  SUBSLICE_DIR        (BARseq anisotropic subslices)"
        )

    # -------------------------------------------------------------
    # Resolve GRAPH_PATH (config value, or auto-derive from data)
    # -------------------------------------------------------------
    if GRAPH_PATH:
        output_path = Path(GRAPH_PATH)
        graph_path_source = "config"
    else:
        output_path = _derive_graph_path(block_path, invivo_path)
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
    print(f"  invivo:    {invivo_path if invivo_path else '(not set, skipping)'}")
    print(f"  block:     {block_path if block_path else '(not set, skipping)'}")
    print(f"  subslices: {subslice_dir if subslice_dir else '(not set, skipping)'}")
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

    existing_nodes = set(g.nodes)

    # -------------------------------------------------------------
    # Add nodes (idempotent: skip any already present)
    # -------------------------------------------------------------
    if invivo_path:
        if "invivo_ref" in existing_nodes:
            print("\nIn vivo node already in graph — skipping")
        else:
            print("\n1. Adding in vivo reference")
            print("-" * 60)
            invivo = load_invivo_stack(invivo_path)
            invivo_spacing, _ = detect_spacing(invivo_path)
            add_invivo_to_graph(g, invivo, spacing=invivo_spacing)
            del invivo
            save_graph(g, output_path, verbose=False)

    if block_path:
        if "ex_vivo_block" in existing_nodes:
            print("\nEx vivo block already in graph — skipping")
        else:
            print("\n2. Adding ex vivo block")
            print("-" * 60)
            block_stack = load_block_stack(block_path)
            block_spacing, _ = detect_spacing(block_path)
            add_block_to_graph(g, block_stack, spacing=block_spacing)
            del block_stack
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
    if invivo_path:
        print(f"  - 1 in vivo reference")
    if block_path:
        print(f"  - 1 ex vivo block")
    if subslice_dir:
        n_fixed = int(bool(invivo_path)) + int(bool(block_path))
        print(f"  - {len(g.nodes) - n_fixed} ex vivo subslices")
    print(f"\nReady for alignment in CASTalign!")

    return output_path


# ============================================
# Usage
# ============================================

if __name__ == "__main__":
    build_subslice_graph()
