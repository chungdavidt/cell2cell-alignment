"""
Graph utilities for FOV clustering and connectivity analysis.

Provides:
- 8-connectivity adjacency matrix construction
- Connected component detection using BFS
- Bridge FOV calculation for diagonal connections
- FOV name parsing

These replicate the graph algorithms from identify_mscarlet_subslices.m.
"""

import numpy as np
import re
from collections import deque
from typing import List, Tuple, Optional


def parse_fov_grid_positions(fov_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse FOV names to extract grid positions.

    FOV naming convention: MAX_Pos{N}_{row}_{col}
    Example: 'MAX_Pos5_012_034' -> row=12, col=34

    Args:
        fov_names: List of FOV name strings

    Returns:
        positions: (N, 2) array of (row, col) positions
        valid_mask: (N,) boolean array indicating valid parses

    Note:
        Row and column values in FOV names may be zero-padded.
    """
    n = len(fov_names)
    positions = np.full((n, 2), np.nan)
    valid_mask = np.zeros(n, dtype=bool)

    # Pattern: MAX_Pos{N}_{row}_{col} or variations
    # Extract the last two underscore-separated numeric parts
    for i, name in enumerate(fov_names):
        parts = name.split('_')
        if len(parts) >= 4:
            try:
                # parts[-2] is row, parts[-1] is col
                # But original MATLAB uses parts{3} and parts{4} (1-indexed)
                # which maps to parts[2] and parts[3] in Python (0-indexed)
                row = int(parts[2])
                col = int(parts[3])
                positions[i] = [row, col]
                valid_mask[i] = True
            except (ValueError, IndexError):
                pass

    return positions, valid_mask


def build_adjacency_8connected(positions: np.ndarray) -> np.ndarray:
    """
    Build adjacency matrix for 8-connectivity (edge or corner neighbors).

    Two FOVs are adjacent if they differ by at most 1 in both row and column
    (excluding the same position).

    Args:
        positions: (N, 2) array of (row, col) positions

    Returns:
        (N, N) boolean adjacency matrix where adj[i,j] = True means i and j are neighbors
    """
    n = positions.shape[0]
    adj = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            row_diff = abs(positions[i, 0] - positions[j, 0])
            col_diff = abs(positions[i, 1] - positions[j, 1])

            # 8-connected: immediate neighbors (edge or corner)
            # Both differences must be <= 1, and at least one must be > 0
            if row_diff <= 1 and col_diff <= 1 and (row_diff + col_diff) > 0:
                adj[i, j] = True
                adj[j, i] = True

    return adj


def find_connected_components(adj: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Find connected components using BFS.

    Args:
        adj: (N, N) adjacency matrix

    Returns:
        components: (N,) array where components[i] is the component ID for node i
                   Component IDs are 1-indexed (0 means unvisited)
        num_components: Total number of components found
    """
    n = adj.shape[0]
    components = np.zeros(n, dtype=int)
    num_components = 0
    visited = np.zeros(n, dtype=bool)

    for start in range(n):
        if visited[start]:
            continue

        num_components += 1

        # BFS from start node
        queue = deque([start])
        visited[start] = True
        components[start] = num_components

        while queue:
            current = queue.popleft()

            # Find all unvisited neighbors
            neighbors = np.where(adj[current, :])[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    components[neighbor] = num_components
                    queue.append(neighbor)

    return components, num_components


def add_bridge_fovs(
    fov_list: List[str],
    positions: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """
    Add bridge FOVs to connect diagonal-only neighbors.

    For each pair of FOVs that are diagonal neighbors (both row and column
    differ by 1), add a bridge FOV at one of the two possible positions
    to ensure 4-connectivity through the bridge.

    Tiebreaker: Prefer topmost (smallest row), then leftmost (smallest col).

    Args:
        fov_list: List of FOV names
        positions: (N, 2) array of (row, col) positions

    Returns:
        bridge_fovs: List of bridge FOV names
        bridge_positions: (M, 2) array of bridge positions
    """
    bridge_fovs = []
    bridge_positions_list = []
    n = len(fov_list)

    # Extract prefix from existing FOV name
    # Format: MAX_Pos5_012_034 -> MAX and Pos5
    if n == 0:
        return [], np.zeros((0, 2))

    parts = fov_list[0].split('_')
    max_prefix = parts[0]  # 'MAX'
    pos_prefix = parts[1] if len(parts) > 1 else 'Pos0'  # 'Pos5'

    # Find all diagonal pairs
    for i in range(n):
        for j in range(i + 1, n):
            r1, c1 = positions[i]
            r2, c2 = positions[j]

            row_diff = abs(r1 - r2)
            col_diff = abs(c1 - c2)

            # Check if diagonal neighbors (both diff = 1)
            if row_diff == 1 and col_diff == 1:
                # Two possible bridge positions:
                # Option A: (r1, c2)
                # Option B: (r2, c1)

                # Tiebreaker: prefer smallest row, then smallest col
                if r1 < r2:
                    bridge_row = int(r1)
                    bridge_col = int(c2)
                elif r2 < r1:
                    bridge_row = int(r2)
                    bridge_col = int(c1)
                else:  # r1 == r2 (shouldn't happen for diagonal)
                    bridge_row = int(r1)
                    bridge_col = int(min(c1, c2))

                bridge_pos = np.array([bridge_row, bridge_col])

                # Check if this bridge already exists
                already_exists = False

                # Check in existing positions
                for pos in positions:
                    if np.array_equal(pos, bridge_pos):
                        already_exists = True
                        break

                # Check in already-added bridges
                for bp in bridge_positions_list:
                    if np.array_equal(bp, bridge_pos):
                        already_exists = True
                        break

                if not already_exists:
                    # Construct FOV name from position
                    bridge_name = f"{max_prefix}_{pos_prefix}_{bridge_row:03d}_{bridge_col:03d}"
                    bridge_fovs.append(bridge_name)
                    bridge_positions_list.append(bridge_pos)

    if bridge_positions_list:
        bridge_positions = np.array(bridge_positions_list)
    else:
        bridge_positions = np.zeros((0, 2))

    return bridge_fovs, bridge_positions


def get_largest_component(
    components: np.ndarray,
    num_components: int
) -> Tuple[np.ndarray, int]:
    """
    Find the largest connected component.

    Args:
        components: Component assignment for each node (1-indexed)
        num_components: Total number of components

    Returns:
        mask: Boolean mask for nodes in largest component
        largest_idx: Index of largest component (1-indexed)
    """
    if num_components == 0:
        return np.zeros(len(components), dtype=bool), 0

    # Count size of each component
    component_sizes = np.array([
        np.sum(components == c) for c in range(1, num_components + 1)
    ])

    largest_idx = np.argmax(component_sizes) + 1  # 1-indexed
    mask = components == largest_idx

    return mask, largest_idx


def extract_fov_prefix(fov_name: str) -> Tuple[str, str]:
    """
    Extract MAX and Pos prefixes from FOV name.

    Args:
        fov_name: FOV name like 'MAX_Pos5_012_034'

    Returns:
        Tuple of (max_prefix, pos_prefix) like ('MAX', 'Pos5')
    """
    parts = fov_name.split('_')
    max_prefix = parts[0] if len(parts) > 0 else 'MAX'
    pos_prefix = parts[1] if len(parts) > 1 else 'Pos0'
    return max_prefix, pos_prefix


def construct_fov_name(max_prefix: str, pos_prefix: str, row: int, col: int) -> str:
    """
    Construct FOV name from components.

    Args:
        max_prefix: Usually 'MAX'
        pos_prefix: Usually 'Pos{N}'
        row: Grid row
        col: Grid column

    Returns:
        FOV name like 'MAX_Pos5_012_034'
    """
    return f"{max_prefix}_{pos_prefix}_{row:03d}_{col:03d}"
