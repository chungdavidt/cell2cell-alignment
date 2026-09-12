#!/usr/bin/env python3
"""
Identify marker+ Subslices for LineStuffUp Alignment.

Finds contiguous regions of FOVs with marker-expressing cells per slice.
Creates subslice definitions (FOV lists) for downstream stitching and overlay.

Marker-parameterized 2026-09-11 (was identify_mscarlet_subslices.py). The
mScarlet pass is unchanged and still writes subslice_definitions.mat, which
steps 2-7 read; every other marker writes its own
subslice_definitions_{marker}.mat and feeds nothing downstream. run_pipeline.py
does not pass --marker: the runner is the mScarlet chain, and another marker's
pass is run by hand, the same arrangement generate_marker_cellmask_subslice.py
has.

The marker+ test is `> 0` -- any expression at all, no rolony threshold. That
is deliberate and coarse: FOV selection is a first pass over ~1 mm tiles, and
which sections are actually used is chosen by hand afterwards.

Usage:
    python identify_marker_subslices.py
    python identify_marker_subslices.py --marker gcamp
    python identify_marker_subslices.py --test  # Test mode (slice 22 only)
    python identify_marker_subslices.py --slice 22  # Specific slice

Output: the marker's definitions file (see --marker), containing:
    - subslice_info: struct array with fields:
        .slice_id: Slice number
        .fov_list: Cell array of FOV names in subslice
        .fov_grid_positions: [N x 2] array of (row, col) grid positions
        .marker_fovs: FOVs with marker+ cells (original, before bridges)
        .bridge_fovs: FOVs added to ensure edge-connectivity
        .num_marker_cells: Total marker+ QC-passing cells in subslice
        .marker: Which marker produced this entry ('mscarlet', 'gcamp')
      A file written before parameterization carries .mscarlet_fovs /
      .num_mscarlet_cells instead; those are read through LEGACY_FIELD_ALIASES
      and are never written back, so one fact never has two names on disk.
      scipy writes this as a 1xN CELL of 1x1 structs, so MATLAB indexes it
      `subslice_info{i}`, not `subslice_info(i)`.

Algorithm:
    1. Apply QC filter: reads >= QC_MIN_READS, genes >= QC_MIN_GENES
    2. Find FOVs with marker+ cells (mScarlet column 113, GCaMP 111 in Python;
       index-only, never resolved by gene name)
    3. Build 8-connectivity graph (edge or corner neighbors)
    4. Find largest connected component per slice
    5. Add bridge FOVs for diagonal-only connections
    6. Generate diagnostic visualizations
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import sparse

from preprocessing_config import (
    FILT_NEURONS_PATH,
    SUBSLICE_DEFINITIONS_DIR,
    SUBSLICE_DEFINITIONS_FILE,
    QC_MIN_READS,
    QC_MIN_GENES,
)
# Column and label only. get_marker() is not used here on purpose: it validates
# the floor/ceiling pair, which sets count -> colour and has no bearing on which
# FOVs hold a marker+ cell.
from marker_profiles import MARKERS, marker_names
from utilities.mat_io import load_filt_neurons, load_mat, save_mat, sparse_to_dense, get_expression_column, resolve_marker_column
from utilities.graph_utils import (
    parse_fov_grid_positions,
    build_adjacency_8connected,
    find_connected_components,
    add_bridge_fovs,
    get_largest_component,
)
from utilities.visualization import visualize_subslice


# The marker the rest of the pipeline is built on. This is NOT merely the
# argparse default: it also decides which run writes the file steps 2-7 read,
# whose diagnostics land in the parent directory, and whose "next steps" text is
# printed. Changing it repoints the pipeline; it is not a preference.
PIPELINE_MARKER = 'mscarlet'

SUBSLICE_FIELDS = (
    'slice_id', 'fov_list', 'fov_grid_positions',
    'marker_fovs', 'bridge_fovs', 'num_marker_cells', 'marker',
)

# What those two fields were called before this script was parameterized. The
# subslice_definitions.mat already on disk carries only these, and a partial run
# merges this run's slices with the ones already in the file -- so read through
# the old names rather than dropping every existing entry as malformed.
LEGACY_FIELD_ALIASES = {
    'marker_fovs': 'mscarlet_fovs',
    'num_marker_cells': 'num_mscarlet_cells',
}

# Fields a pre-parameterization entry cannot have under any name. A file written
# before the rename is necessarily the pipeline marker's, which is what makes
# this default safe rather than a guess.
LEGACY_FIELD_DEFAULTS = {
    'marker': PIPELINE_MARKER,
}


def definitions_path(marker):
    """Where `marker`'s definitions live.

    mScarlet keeps the unsuffixed name because steps 2-7 read it; any other
    marker gets its own file so a run cannot repoint the pipeline.
    """
    if marker == PIPELINE_MARKER:
        return SUBSLICE_DEFINITIONS_FILE
    return str(Path(SUBSLICE_DEFINITIONS_DIR) / f"subslice_definitions_{marker}.mat")


# There is deliberately no _with_legacy_aliases() writer. Writing both
# `marker_fovs` and `mscarlet_fovs` meant two names for one fact and two writers
# of it: edit_subslice_definitions.py mutated the legacy pair, the merge loader
# below preferred the neutral pair, and a partial run then silently reverted the
# hand edit. The aliases are a READ path for files written before the rename,
# nothing more -- every writer emits the neutral names only.


def _entry_field(entry, name):
    """One field off an entry, whether it came from this run (dict) or from
    load_mat (a scipy mat_struct)."""
    return entry[name] if isinstance(entry, dict) else getattr(entry, name)


def _load_existing_definitions(definitions_file):
    """Entries already on disk, normalised to plain dicts. [] when absent.

    Mixed dict/mat_struct lists do not round-trip through save_mat, so loaded
    entries are converted before they are merged with this run's.
    """
    path = Path(definitions_file)
    if not path.exists():
        return []
    entries = load_mat(definitions_file).get('subslice_info')
    if entries is None:
        return []
    if isinstance(entries, np.ndarray):
        entries = list(entries.flatten())
    elif not isinstance(entries, list):
        entries = [entries]

    out = []
    for entry in entries:
        fields = {}
        try:
            for name in SUBSLICE_FIELDS:
                try:
                    fields[name] = _entry_field(entry, name)
                except (KeyError, AttributeError):
                    alias = LEGACY_FIELD_ALIASES.get(name)
                    if alias is not None:
                        fields[name] = _entry_field(entry, alias)
                    elif name in LEGACY_FIELD_DEFAULTS:
                        fields[name] = LEGACY_FIELD_DEFAULTS[name]
                    else:
                        raise
        except (KeyError, AttributeError):
            continue        # malformed entry: drop rather than propagate
        out.append(_strip_fov_names(fields))
    return out


def _strip_fov_names(entry):
    """Trailing-space padding off the FOV name fields.

    savemat stores a list of names as an N-by-L char matrix, padding the short
    ones to the longest with SPACES, and loadmat hands those spaces back. The
    names go straight into a filesystem path
    (`utilities/image_io.py`: `hyb_root / fov_name / 'alignedn2vhyb01.tif'`), so
    a padded name is a path that cannot exist and the FOV drops out silently.
    Invisible on a dataset whose names are all one length -- BY95's are, at 16
    characters -- and live the moment one has ten or more slide positions
    (`MAX_Pos9_003_005` is 16, `MAX_Pos10_003_006` is 17). Measured, not assumed.
    """
    for field in ('fov_list', 'marker_fovs', 'bridge_fovs'):
        value = entry.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            entry[field] = value.strip()
        elif isinstance(value, np.ndarray):
            entry[field] = np.array([str(v).strip() for v in value.ravel()])
        elif isinstance(value, (list, tuple)):
            entry[field] = [str(v).strip() for v in value]
    return entry


def identify_marker_subslices(marker: str = PIPELINE_MARKER,
                              test_mode: bool = False,
                              target_slice: int = None):
    """
    Main function to identify marker+ subslices.

    Args:
        marker: marker_profiles key -- 'mscarlet' (the pipeline's) or 'gcamp'
        test_mode: If True, process only slice 22 (known good slice)
        target_slice: If specified, process only this slice
    """
    TEST_SLICE = 22  # Known-good mScarlet slice; not re-picked per marker

    if marker not in MARKERS:
        raise ValueError(f"--marker {marker}: not one of {sorted(MARKERS)}")
    profile = MARKERS[marker]
    label = profile['label']
    definitions_file = definitions_path(marker)

    print("=" * 40)
    print(f"IDENTIFY {label} SUBSLICES")
    print("=" * 40)
    if test_mode:
        print(f"Mode: TEST (slice {TEST_SLICE} only)")
    elif target_slice is not None:
        print(f"Mode: SINGLE SLICE ({target_slice})")
    else:
        print("Mode: FULL (all slices)")
    print()

    # Create output directory. mScarlet's diagnostics stay where they have always
    # been; another marker gets a subdirectory so its per-slice PNGs, which carry
    # no marker in their filenames, cannot overwrite mScarlet's.
    output_dir = Path(SUBSLICE_DEFINITIONS_DIR)
    if marker != PIPELINE_MARKER:
        output_dir = output_dir / marker
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load filt_neurons
    print("Loading filt_neurons...")
    filt_neurons = load_filt_neurons(FILT_NEURONS_PATH)

    expmat = filt_neurons['expmat']
    n_cells = expmat.shape[0]
    print(f"  Total cells in dataset: {n_cells}")

    # FOV names should already be normalized by load_filt_neurons
    fov_names = filt_neurons['fov']
    print("  FOV names normalized")

    # Apply QC filter
    print("\nApplying QC filter...")
    print(f"  QC criteria: reads >= {QC_MIN_READS} AND genes >= {QC_MIN_GENES}")

    # Calculate total reads and genes per cell
    if sparse.issparse(expmat):
        total_reads = np.asarray(expmat.sum(axis=1)).flatten()
        total_genes = np.asarray((expmat > 0).sum(axis=1)).flatten()
    else:
        total_reads = np.sum(expmat, axis=1)
        total_genes = np.sum(expmat > 0, axis=1)

    pass_qc = (total_reads >= QC_MIN_READS) & (total_genes >= QC_MIN_GENES)
    print(f"  Cells passing QC: {np.sum(pass_qc)} / {n_cells} ({100*np.sum(pass_qc)/n_cells:.1f}%)")

    # Find marker+ cells. gene_name is blank for every marker in the table --
    # this panel labels its readout slots with stale names -- so the resolve
    # falls through to the index, which is the only trustworthy handle.
    print(f"\nFinding {label}+ cells...")
    marker_col = resolve_marker_column(
        filt_neurons, profile['gene_name'], profile['column'])
    marker_expression = get_expression_column(expmat, marker_col)
    marker_positive = marker_expression > 0

    print(f"  {label}+ cells (any expression): {np.sum(marker_positive)} / {n_cells} "
          f"({100*np.sum(marker_positive)/n_cells:.1f}%)")

    # Combined filter
    marker_qc_pass = pass_qc & marker_positive
    print(f"  {label}+ cells passing QC: {np.sum(marker_qc_pass)} "
          f"({100*np.sum(marker_qc_pass)/np.sum(pass_qc):.1f}%)")
    print()

    # Get unique slices
    slice_ids = np.asarray(filt_neurons['slice']).flatten()
    unique_slices = np.unique(slice_ids[~np.isnan(slice_ids)])
    unique_slices = unique_slices.astype(int)

    if test_mode:
        if TEST_SLICE not in unique_slices:
            raise ValueError(f"Test slice {TEST_SLICE} not found in dataset")
        unique_slices = np.array([TEST_SLICE])
        print(f"Processing test slice: {TEST_SLICE}\n")
    elif target_slice is not None:
        if target_slice not in unique_slices:
            raise ValueError(f"Slice {target_slice} not found in dataset")
        unique_slices = np.array([target_slice])
        print(f"Processing slice: {target_slice}\n")
    else:
        print(f"Found {len(unique_slices)} unique slices\n")

    # Process each slice
    subslice_info_list = []

    for s_idx, slice_id in enumerate(unique_slices):
        print("=" * 40)
        print(f"[{s_idx+1}/{len(unique_slices)}] Processing slice {slice_id}")
        print("=" * 40)

        # Get cells in this slice
        in_slice = slice_ids == slice_id
        slice_marker_qc = in_slice & marker_qc_pass

        print(f"  Cells in slice: {np.sum(in_slice)}")
        print(f"  {label}+ QC-passing cells: {np.sum(slice_marker_qc)}")

        if np.sum(slice_marker_qc) == 0:
            print(f"  WARNING: No {label}+ cells in slice, skipping\n")
            continue

        # Get unique FOVs with marker+ cells
        slice_fov_names = np.array(fov_names)[slice_marker_qc]
        slice_fovs = list(np.unique(slice_fov_names))
        print(f"  FOVs with {label}+ cells: {len(slice_fovs)}")

        # Parse FOV names to get grid positions
        fov_positions, valid_mask = parse_fov_grid_positions(slice_fovs)

        if np.sum(valid_mask) == 0:
            print("  WARNING: No valid FOV names parsed, skipping\n")
            continue

        if np.sum(~valid_mask) > 0:
            print(f"  WARNING: {np.sum(~valid_mask)} FOVs could not be parsed")

        # Keep only valid FOVs
        valid_indices = np.where(valid_mask)[0]
        slice_fovs = [slice_fovs[i] for i in valid_indices]
        fov_positions = fov_positions[valid_mask]

        print(f"  Valid FOVs for clustering: {len(slice_fovs)}")

        # Build adjacency matrix (8-connectivity)
        adj_matrix = build_adjacency_8connected(fov_positions)

        # Find connected components
        components, num_components = find_connected_components(adj_matrix)
        print(f"  Connected components found: {num_components}")

        if num_components == 0:
            print("  WARNING: No connected components, skipping\n")
            continue

        # Find largest component
        component_sizes = [np.sum(components == c) for c in range(1, num_components + 1)]
        print(f"  Component sizes: {component_sizes}")

        largest_mask, largest_idx = get_largest_component(components, num_components)
        print(f"  Largest component: #{largest_idx} with {np.sum(largest_mask)} FOVs")

        # Get FOVs in largest component
        marker_fovs = [slice_fovs[i] for i in range(len(slice_fovs)) if largest_mask[i]]
        marker_positions = fov_positions[largest_mask]

        # Add bridge FOVs for diagonal connections
        print("  Adding bridge FOVs for diagonal connections...")
        bridge_fovs, bridge_positions = add_bridge_fovs(marker_fovs, marker_positions)
        print(f"  Bridge FOVs added: {len(bridge_fovs)}")

        # Combine marker and bridge FOVs
        final_fov_list = marker_fovs + bridge_fovs
        if len(bridge_positions) > 0:
            final_positions = np.vstack([marker_positions, bridge_positions])
        else:
            final_positions = marker_positions

        # Count marker+ cells in final subslice
        fov_names_array = np.array(fov_names)
        in_subslice_fovs = np.isin(fov_names_array, final_fov_list)
        subslice_marker_cells = in_slice & in_subslice_fovs & marker_qc_pass
        num_marker_cells = int(np.sum(subslice_marker_cells))

        print(f"  Total FOVs in subslice: {len(final_fov_list)} "
              f"({len(marker_fovs)} {label} + {len(bridge_fovs)} bridge)")
        print(f"  Total {label}+ cells in subslice: {num_marker_cells}")

        # Save subslice info
        subslice_info = {
            'slice_id': int(slice_id),
            'fov_list': final_fov_list,
            'fov_grid_positions': final_positions,
            'marker_fovs': marker_fovs,
            'bridge_fovs': bridge_fovs,
            'num_marker_cells': num_marker_cells,
            'marker': marker,
        }
        subslice_info_list.append(subslice_info)

        # Generate diagnostic visualization
        viz_path = visualize_subslice(
            slice_id, marker_fovs, marker_positions,
            bridge_fovs, bridge_positions, output_dir,
            marker_label=label,
        )
        print(f"  Saved diagnostic plot: {viz_path}")
        print()

    # Save results
    print("=" * 40)
    print("SAVING RESULTS")
    print("=" * 40)
    print(f"Total slices with subslices: {len(subslice_info_list)}")
    print(f"Output file: {definitions_file}")

    # A PARTIAL run must not delete the slices it did not look at. This file is
    # the only record of subslice membership and step 2 reads all of it, so
    # writing just this run's slices would silently discard the rest.
    # A FULL run legitimately replaces everything -- if a slice stops qualifying
    # (e.g. after a QC threshold change) its entry SHOULD disappear.
    partial = test_mode or target_slice is not None
    to_save = subslice_info_list
    if partial:
        processed = {int(info['slice_id']) for info in subslice_info_list}
        kept = [e for e in _load_existing_definitions(definitions_file)
                if int(_entry_field(e, 'slice_id')) not in processed]
        to_save = sorted(kept + subslice_info_list,
                         key=lambda e: int(_entry_field(e, 'slice_id')))
        print(f"Partial run: merging with {len(kept)} slice(s) already on file")
        print(f"Total slices in file after merge: {len(to_save)}")

    # scipy writes a Python list of dicts as a 1xN CELL of 1x1 structs, not a
    # struct array -- MATLAB readers need `subslice_info{i}`, not `(i)`.
    save_mat(definitions_file, {'subslice_info': to_save}, format='5')
    print("Saved subslice definitions\n")

    # Summary
    print("=" * 40)
    print("SUMMARY")
    print("=" * 40)
    for info in subslice_info_list:
        print(f"Slice {info['slice_id']}: {len(info['fov_list'])} FOVs "
              f"({len(info['marker_fovs'])} {label} + {len(info['bridge_fovs'])} bridge), "
              f"{info['num_marker_cells']} cells")

    print("\nNext steps:")
    print(f"  1. Review diagnostic plots in: {output_dir}")
    if marker == PIPELINE_MARKER:
        print("  2. (Optional) Edit subslice definitions using edit_subslice_definitions.py")
        print("  3. Run stitch_subslices.py to create stitched subslice images")
    else:
        print(f"  2. {label} definitions feed nothing downstream -- the pipeline")
        print(f"     reads {SUBSLICE_DEFINITIONS_FILE}.")
        print(f"     matlab/gen_marker_plots_dtc.m finds this file on its own with")
        print(f"     MARKER = '{marker}' and CROP_TO_SUBSLICE = true.")
    print()

    return subslice_info_list


def main():
    parser = argparse.ArgumentParser(
        description="Identify marker+ subslices for alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--marker', '-m',
        choices=marker_names(),
        default=PIPELINE_MARKER,
        help="Which marker's connected region to find. Selects the expmat "
             "column and the output file. mscarlet writes the "
             "subslice_definitions.mat steps 2-7 read; anything else writes "
             "subslice_definitions_{marker}.mat and feeds nothing downstream "
             f"(default: {PIPELINE_MARKER})"
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test mode: process only slice 22'
    )
    parser.add_argument(
        '--slice', '-s',
        type=int,
        default=None,
        help='Process specific slice only'
    )

    args = parser.parse_args()

    identify_marker_subslices(
        marker=args.marker,
        test_mode=args.test,
        target_slice=args.slice
    )


if __name__ == '__main__':
    main()
