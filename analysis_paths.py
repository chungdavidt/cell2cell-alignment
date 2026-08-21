"""Resolution of the per-subject analysis tree.

Lives at the project root, next to local_config.py, and imports nothing beyond
the stdlib — every consumer (preprocessing, alignment, the cellpose venv) can
import it without pulling in the numpy stack that ``utilities/__init__`` loads.

All derived outputs for a subject live under one root, declared as
``ANALYSIS_ROOT`` in ``local_config.py``::

    <ANALYSIS_ROOT>/          e.g. .../cell_type/analysis/BY95
        preprocessing/        BARseq preprocessing outputs (OUTPUT_ROOT)
        alignment/            <subject>_graph.db
        cellpose/             *_seg.npy, sweep_in_vivo_*, combo_in_vivo_*

Blank or absent ``ANALYSIS_ROOT`` keeps every consumer on its previous
data-tree-adjacent default, so existing subjects need no config change.
"""

from pathlib import Path
from typing import Optional
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent

PREPROCESSING_SUBDIR = "preprocessing"
ALIGNMENT_SUBDIR = "alignment"
CELLPOSE_SUBDIR = "cellpose"


def get_analysis_root() -> Optional[Path]:
    """Return ANALYSIS_ROOT from local_config.py, or None when unset.

    Raises ValueError if set to a relative path (it would resolve against the
    caller's cwd, which differs between run_pipeline.py steps and notebooks).
    """
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    try:
        import local_config
    except ImportError:
        return None

    root = getattr(local_config, "ANALYSIS_ROOT", "")
    if not root:
        return None
    path = Path(root).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"ANALYSIS_ROOT in local_config.py must be absolute: {root!r}"
        )
    return path


def analysis_subdir(name: str, create: bool = False) -> Optional[Path]:
    """Return ``<ANALYSIS_ROOT>/<name>``, or None when ANALYSIS_ROOT is unset."""
    root = get_analysis_root()
    if root is None:
        return None
    path = root / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def cellpose_dir(create: bool = False) -> Optional[Path]:
    """Return the analysis-tree cellpose dir, or None when ANALYSIS_ROOT is unset."""
    return analysis_subdir(CELLPOSE_SUBDIR, create=create)


def subject_name() -> Optional[str]:
    """Subject label implied by ANALYSIS_ROOT's folder name (e.g. 'BY95')."""
    root = get_analysis_root()
    return None if root is None else root.name
