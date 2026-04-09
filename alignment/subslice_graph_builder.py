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

# Add castalign to path
CASTALIGN_ROOT = Path("/home/dtc/lab/programs/castalign")
if str(CASTALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(CASTALIGN_ROOT))

import castalign as ca
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from typing import Optional, Union, List


# ============================================
# Configuration
# ============================================

OUTPUT_ROOT = Path("/home/dtc/lab/output")
INVIVO_PATH = OUTPUT_ROOT / "in_vivo_flip_corrected" / "JH302_1x_ch2_flipped.tiff"

# Anisotropic subslices directory (Python-generated cellmask overlays)
ANISO_SUBSLICE_DIR = OUTPUT_ROOT / "mScarlet_cellmask_subslice" / "threshold_0.30_cellmask_0.50_anisotropic"

# Resolution parameters
INVIVO_XY_UM_PER_PX = 2.34   # In vivo X/Y resolution
INVIVO_Z_UM_PER_PX = 1.0     # In vivo Z resolution
EXVIVO_X_UM_PER_PX = 2.34    # Anisotropic ex vivo X (matches in vivo X/Y)
EXVIVO_Y_UM_PER_PX = 1.0     # Anisotropic ex vivo Y (matches in vivo Z)


# ============================================
# Data Loading Functions
# ============================================

def load_invivo_stack(path: Union[str, Path] = INVIVO_PATH) -> np.ndarray:
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
    print(f"  Resolution: {INVIVO_Z_UM_PER_PX} µm/px (Z), {INVIVO_XY_UM_PER_PX} µm/px (Y/X)")

    return stack


def discover_aniso_subslices(
    directory: Union[str, Path] = ANISO_SUBSLICE_DIR
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
    node_name: str = "invivo_ref"
) -> ca.Graph:
    """
    Add in vivo reference to graph with correct spacing metadata.

    Spacing: (Z, Y, X) = (1.0, 2.34, 2.34) µm/px
    """
    print(f"Adding in vivo reference '{node_name}' to graph...")

    # In vivo spacing: (Z, Y, X) in µm/pixel
    metadata = {'spacing': (INVIVO_Z_UM_PER_PX, INVIVO_XY_UM_PER_PX, INVIVO_XY_UM_PER_PX)}

    graph.add_node(node_name, image=invivo_stack, compression="high", metadata=metadata)

    print(f"  Added: {invivo_stack.shape}")
    print(f"  Spacing: {metadata['spacing']} µm/px (Z, Y, X)")

    return graph


def add_subslices_to_graph(
    graph: ca.Graph,
    subslice_dir: Union[str, Path] = ANISO_SUBSLICE_DIR,
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

def build_subslice_graph(
    output_path: Optional[Union[str, Path]] = None,
    force_rebuild: bool = False,
    save_every: int = 10
) -> Path:
    """
    Build complete subslice alignment graph.

    Parameters
    ----------
    output_path : Path, optional
        Where to save graph (default: linestuffup_output/JH302_subslice_graph.db)
    force_rebuild : bool
        If True, rebuild even if graph exists
    save_every : int
        Save checkpoint every N subslices

    Returns
    -------
    Path
        Path to saved graph
    """
    if output_path is None:
        output_path = OUTPUT_ROOT / "linestuffup_output" / "castalign_test.db"
    output_path = Path(output_path)

    print("=" * 60)
    print("SUBSLICE GRAPH BUILDER")
    print("=" * 60)
    print(f"\nResolution Configuration:")
    print(f"  In vivo:  Z={INVIVO_Z_UM_PER_PX} µm/px, Y/X={INVIVO_XY_UM_PER_PX} µm/px")
    print(f"  Ex vivo:  Y={EXVIVO_Y_UM_PER_PX} µm/px, X={EXVIVO_X_UM_PER_PX} µm/px")
    print(f"  After xrotate=90: Ex vivo Y → In vivo Z (matched!)")
    print()

    # Check if graph exists
    if output_path.exists() and not force_rebuild:
        print(f"Graph already exists: {output_path.name}")
        print(f"  Size: {output_path.stat().st_size/1e6:.1f} MB")
        print("\nTo rebuild, use force_rebuild=True")
        return output_path

    if force_rebuild and output_path.exists():
        print("Force rebuild - recreating graph...")
        output_path.unlink()

    # Check for partial graph (resume)
    if output_path.exists():
        print("Partial graph found - resuming...")
        g = load_graph(output_path)
    else:
        # Start fresh
        print("\n1. Loading in vivo stack...")
        print("-" * 60)
        invivo = load_invivo_stack()

        print("\n2. Creating graph...")
        print("-" * 60)
        g = create_subslice_graph()
        add_invivo_to_graph(g, invivo)

        del invivo  # Free memory

        print("\nSaving initial checkpoint...")
        save_graph(g, output_path, verbose=False)

    # Add subslices
    print("\n3. Adding anisotropic subslices...")
    print("-" * 60)
    add_subslices_to_graph(g, save_every=save_every, output_path=output_path)

    # Final save
    print("\n4. Saving final graph...")
    print("-" * 60)
    save_graph(g, output_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUBSLICE GRAPH COMPLETE")
    print("=" * 60)
    print(f"Graph: {output_path}")
    print(f"Nodes: {len(g.nodes)}")
    print(f"  - 1 in vivo reference")
    print(f"  - {len(g.nodes) - 1} ex vivo subslices")
    print(f"\nReady for alignment in LineStuffUp!")

    return output_path


# ============================================
# Usage
# ============================================

if __name__ == "__main__":
    # Build graph
    graph_path = build_subslice_graph()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Open JH302_alignment_workflow.ipynb")
    print("2. Update GRAPH_SAVE_PATH to:", graph_path)
    print("3. Launch alignment GUI")
    print("4. Use xrotate=90 to rotate coronal slices")
    print("5. Align subslices to in vivo reference")
