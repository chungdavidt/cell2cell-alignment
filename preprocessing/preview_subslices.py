#!/usr/bin/env python3
"""
Report what identify_mscarlet_subslices.py would emit, without writing anything.

Runs step 1's selection per slice using the same utility functions, then prints
a per-slice table and a summary of which slices would yield a subslice.

Usage:
    python preview_subslices.py                       # paths from preprocessing_config
    python preview_subslices.py --filt-neurons PATH   # no local_config needed
    python preview_subslices.py --min-cells 20 --min-fovs 2   # report-only cut
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utilities.mat_io import (
    load_filt_neurons,
    get_expression_column,
    resolve_marker_column,
)
from utilities.graph_utils import (
    parse_fov_grid_positions,
    build_adjacency_8connected,
    find_connected_components,
    add_bridge_fovs,
    get_largest_component,
)


def preview(filt_neurons_path, marker_col, marker_name, min_reads, min_genes,
            min_cells, min_fovs):
    print(f"filt_neurons: {filt_neurons_path}")
    filt_neurons = load_filt_neurons(filt_neurons_path)

    expmat = filt_neurons['expmat']
    n_cells = expmat.shape[0]
    fov_names = np.array(filt_neurons['fov'])

    if sparse.issparse(expmat):
        total_reads = np.asarray(expmat.sum(axis=1)).flatten()
        total_genes = np.asarray((expmat > 0).sum(axis=1)).flatten()
    else:
        total_reads = np.sum(expmat, axis=1)
        total_genes = np.sum(expmat > 0, axis=1)

    pass_qc = (total_reads >= min_reads) & (total_genes >= min_genes)

    col = resolve_marker_column(filt_neurons, marker_name, marker_col)
    marker_positive = get_expression_column(expmat, col) > 0
    marker_qc = pass_qc & marker_positive

    slice_ids = np.asarray(filt_neurons['slice']).flatten()
    has_slice = ~np.isnan(slice_ids)
    unique_slices = np.unique(slice_ids[has_slice]).astype(int)

    print(f"cells: {n_cells}")
    print(f"  QC pass (reads>={min_reads}, genes>={min_genes}): {pass_qc.sum()} "
          f"({100*pass_qc.sum()/n_cells:.1f}%)")
    print(f"  marker column {col} > 0: {marker_positive.sum()} "
          f"({100*marker_positive.sum()/n_cells:.1f}%)")
    print(f"  QC pass AND marker+: {marker_qc.sum()}")
    print(f"  slice assigned: {has_slice.sum()} ({100*has_slice.sum()/n_cells:.1f}%)")
    print(f"  QC pass AND marker+ AND slice assigned: {(marker_qc & has_slice).sum()}")
    print(f"slices: {len(unique_slices)}")
    print()

    header = (f"{'slice':>5} {'cells':>7} {'QC':>7} {'QC+mSc':>7} {'FOVs':>5} "
              f"{'comps':>5} {'largest':>7} {'bridge':>6} {'final':>5} {'sub_cells':>9}  note")
    print(header)
    print("-" * len(header))

    rows = []
    for slice_id in unique_slices:
        in_slice = slice_ids == slice_id
        n_in_slice = int(in_slice.sum())
        n_qc = int((in_slice & pass_qc).sum())
        sel = in_slice & marker_qc
        n_sel = int(sel.sum())

        row = dict(slice_id=int(slice_id), n_in_slice=n_in_slice, n_qc=n_qc,
                   n_sel=n_sel, n_fovs=0, n_comps=0, n_largest=0, n_bridge=0,
                   n_final=0, sub_cells=0, note="")

        if n_sel == 0:
            row['note'] = "SKIP: no marker+ QC cells"
            rows.append(row)
            continue

        slice_fovs = list(np.unique(fov_names[sel]))
        row['n_fovs'] = len(slice_fovs)

        positions, valid_mask = parse_fov_grid_positions(slice_fovs)
        if valid_mask.sum() == 0:
            row['note'] = "SKIP: no parseable FOV names"
            rows.append(row)
            continue
        if (~valid_mask).sum() > 0:
            row['note'] = f"{(~valid_mask).sum()} unparseable FOV; "

        slice_fovs = [slice_fovs[i] for i in np.where(valid_mask)[0]]
        positions = positions[valid_mask]

        components, num_components = find_connected_components(
            build_adjacency_8connected(positions))
        row['n_comps'] = int(num_components)
        if num_components == 0:
            row['note'] += "SKIP: no components"
            rows.append(row)
            continue

        largest_mask, _ = get_largest_component(components, num_components)
        marker_fovs = [slice_fovs[i] for i in range(len(slice_fovs)) if largest_mask[i]]
        marker_positions = positions[largest_mask]
        row['n_largest'] = len(marker_fovs)

        bridge_fovs, bridge_positions = add_bridge_fovs(marker_fovs, marker_positions)
        row['n_bridge'] = len(bridge_fovs)
        row['n_final'] = len(marker_fovs) + len(bridge_fovs)

        final_list = marker_fovs + bridge_fovs
        row['sub_cells'] = int((in_slice & np.isin(fov_names, final_list) & marker_qc).sum())

        dropped = row['n_fovs'] - row['n_largest']
        if dropped > 0:
            row['note'] += f"{dropped} FOV dropped w/ largest-component; "
        if row['sub_cells'] < min_cells or row['n_largest'] < min_fovs:
            row['note'] += f"below cut ({min_cells} cells / {min_fovs} FOVs)"
        rows.append(row)

    for r in rows:
        print(f"{r['slice_id']:>5} {r['n_in_slice']:>7} {r['n_qc']:>7} {r['n_sel']:>7} "
              f"{r['n_fovs']:>5} {r['n_comps']:>5} {r['n_largest']:>7} {r['n_bridge']:>6} "
              f"{r['n_final']:>5} {r['sub_cells']:>9}  {r['note']}")

    emitted = [r for r in rows if r['n_final'] > 0]
    skipped = [r for r in rows if r['n_final'] == 0]
    below = [r for r in emitted if r['sub_cells'] < min_cells or r['n_largest'] < min_fovs]

    print()
    print("=" * 40)
    print(f"slices examined:        {len(rows)}")
    print(f"would emit a subslice:  {len(emitted)}")
    print(f"skipped by step 1:      {len(skipped)}  {[r['slice_id'] for r in skipped]}")
    print(f"below the report cut:   {len(below)}  {[r['slice_id'] for r in below]}")
    if emitted:
        cells = np.array([r['sub_cells'] for r in emitted])
        fovs = np.array([r['n_final'] for r in emitted])
        print(f"subslice marker cells:  min {cells.min()}  median {int(np.median(cells))}  max {cells.max()}")
        print(f"subslice FOVs:          min {fovs.min()}  median {int(np.median(fovs))}  max {fovs.max()}")
        print(f"single-FOV subslices:   {int((fovs == 1).sum())}")
    print("Nothing was written.")

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--filt-neurons', default=None,
                   help='Path to filt_neurons.mat (default: FILT_NEURONS_PATH from preprocessing_config)')
    p.add_argument('--marker-column', type=int, default=None,
                   help='0-indexed marker column (default: MSCARLET_COLUMN_INDEX)')
    p.add_argument('--marker-name', default=None,
                   help='Marker gene name; blank means trust the column index')
    p.add_argument('--min-reads', type=int, default=None)
    p.add_argument('--min-genes', type=int, default=None)
    p.add_argument('--min-cells', type=int, default=20,
                   help='Report-only: flag subslices below this many marker+ QC cells')
    p.add_argument('--min-fovs', type=int, default=2,
                   help='Report-only: flag subslices whose largest component is below this many FOVs')
    args = p.parse_args()

    path = args.filt_neurons
    marker_col, marker_name = args.marker_column, args.marker_name
    min_reads, min_genes = args.min_reads, args.min_genes

    if path is None or marker_col is None or marker_name is None \
            or min_reads is None or min_genes is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from preprocessing_config import (
            FILT_NEURONS_PATH, MSCARLET_COLUMN_INDEX, MSCARLET_GENE_NAME,
            QC_MIN_READS, QC_MIN_GENES,
        )
        path = path or FILT_NEURONS_PATH
        marker_col = MSCARLET_COLUMN_INDEX if marker_col is None else marker_col
        marker_name = MSCARLET_GENE_NAME if marker_name is None else marker_name
        min_reads = QC_MIN_READS if min_reads is None else min_reads
        min_genes = QC_MIN_GENES if min_genes is None else min_genes

    preview(path, marker_col, marker_name, min_reads, min_genes,
            args.min_cells, args.min_fovs)


if __name__ == '__main__':
    main()
