"""
Read-only cross-tab of empty cells vs unassigned slices in a filt_neurons.mat.

BY95 has a median of 0 reads/cell and 40.2% NaN `slice` values. If those are
the same population, one drop-mask at the top of step 1 handles both. This
prints the 2x2 contingency table, what each group costs in marker+ cells, and
the QC pass rate before and after dropping. Writes nothing.

Usage:
    python preprocessing/check_empty_cells.py <DATA_ROOT_or_filt_neurons.mat>
    python preprocessing/check_empty_cells.py        # uses local_config.DATA_ROOT
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

MSCARLET_COLUMN_INDEX = 113
GCAMP_COLUMN_INDEX = 111
QC_MIN_READS = 20
QC_MIN_GENES = 5


def _column(expmat, idx):
    import scipy.sparse as sp

    if idx >= expmat.shape[1]:
        return None
    col = expmat[:, idx]
    return np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()


def _pct(n, total):
    return f"{n:>8d}  ({100 * n / total:5.1f}%)"


def main():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_dir():
            p = p / "filt_neurons.mat"
    else:
        import local_config

        p = Path(local_config.DATA_ROOT) / "filt_neurons.mat"

    print(f"filt_neurons: {p}")
    from utilities.mat_io import load_filt_neurons

    fn = load_filt_neurons(p)
    expmat = fn["expmat"]
    n = expmat.shape[0]

    reads = np.asarray(expmat.sum(axis=1)).ravel()
    ngene = np.asarray((expmat > 0).sum(axis=1)).ravel()
    empty = reads == 0
    nan = np.isnan(np.asarray(fn["slice"]).ravel().astype(float))

    print(f"\ncells: {n}")
    print(f"reads/cell   median {np.median(reads):.0f}  mean {reads.mean():.2f}  max {reads.max():.0f}")
    print(f"genes/cell   median {np.median(ngene):.0f}  mean {ngene.mean():.2f}  max {ngene.max():.0f}")

    print("\n=== zero reads vs NaN slice ===")
    print(f"{'':22}{'slice NaN':>18}{'slice assigned':>18}")
    for label, mask in (("reads == 0", empty), ("reads > 0", ~empty)):
        print(f"{label:22}{int((mask & nan).sum()):>18d}{int((mask & ~nan).sum()):>18d}")
    print(f"\nzero-read {_pct(int(empty.sum()), n)}")
    print(f"NaN slice {_pct(int(nan.sum()), n)}")
    if empty.sum():
        print(f"agreement: {100 * float((empty == nan).mean()):.1f}% of cells fall on the diagonal")
        if (empty & ~nan).sum() == 0 and (~empty & nan).sum() == 0:
            print("  -> identical populations; one drop-mask covers both")
        else:
            print(f"  -> NOT identical: {int((empty & ~nan).sum())} empty cells carry a slice, "
                  f"{int((~empty & nan).sum())} cells with reads have no slice")

    print("\n=== what each mask would drop ===")
    for label, keep in (("keep reads > 0", ~empty),
                        ("keep slice assigned", ~nan),
                        ("keep both", ~empty & ~nan)):
        k = int(keep.sum())
        qc = int(((reads >= QC_MIN_READS) & (ngene >= QC_MIN_GENES) & keep).sum())
        print(f"{label:22} keeps {_pct(k, n)}   "
              f"QC pass within kept: {100 * qc / max(k, 1):5.1f}%")

    print("\n=== marker+ cells by group ===")
    for marker, idx in (("mScarlet", MSCARLET_COLUMN_INDEX), ("GCaMP", GCAMP_COLUMN_INDEX)):
        col = _column(expmat, idx)
        if col is None:
            continue
        pos = col > 0
        print(f"{marker} (col {idx}): {int(pos.sum())} positive, max {col.max():.0f}, "
              f"median-of-positive {np.median(col[pos]) if pos.any() else 0:.0f}")
        print(f"    lost to a NaN-slice filter: {int((pos & nan).sum())}")
        print(f"    lost to a zero-read filter: {int((pos & empty).sum())}")
        for thr in (1, 2, 3, 5):
            keep = (col >= thr) & ~nan & ~empty
            print(f"    >= {thr} counts and kept by both filters: {int(keep.sum())}")

    print("\n=== slice ids among kept cells ===")
    sl = np.asarray(fn["slice"]).ravel().astype(float)
    kept = ~empty & ~nan
    uniq = np.unique(sl[kept])
    print(f"{uniq.size} slices, range {uniq.min():.0f}..{uniq.max():.0f}")
    counts = [(int(s), int((sl[kept] == s).sum())) for s in uniq]
    print(f"cells per slice: min {min(c for _, c in counts)}, "
          f"max {max(c for _, c in counts)}, median {int(np.median([c for _, c in counts]))}")
    empties = [s for s, c in counts if c == 0]
    if empties:
        print(f"slices with no kept cells: {empties}")


if __name__ == "__main__":
    main()
