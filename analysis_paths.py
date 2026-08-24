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


def _mscarlet_cellmask_root() -> Path:
    """``MSCARLET_CELLMASK_DIR`` from preprocessing_config, imported lazily.

    Imported here rather than at module scope because preprocessing_config
    validates the whole preprocessing config at import — it raises when
    DATA_ROOT or SCOPE is unset — and a 2P-only run has no reason to satisfy
    that. Only a relative SUBSLICE_DIR needs it.
    """
    preprocessing = _PROJECT_ROOT / "preprocessing"
    if str(preprocessing) not in sys.path:
        sys.path.insert(0, str(preprocessing))
    try:
        import preprocessing_config
    except Exception as e:
        raise ValueError(
            f"SUBSLICE_DIR is relative, so it is resolved against "
            f"preprocessing's output tree — but preprocessing_config could not "
            f"be loaded:\n"
            f"  {type(e).__name__}: {e}\n\n"
            f"Either fix that (it needs DATA_ROOT and SCOPE), or set "
            f"SUBSLICE_DIR to an absolute path."
        )
    return Path(preprocessing_config.MSCARLET_CELLMASK_DIR)


def resolve_subslice_dir(value: Optional[str] = None) -> Optional[Path]:
    """Resolve ``SUBSLICE_DIR`` to an absolute directory.

    The BARseq subslices the graph builder reads are step 4's output, so their
    parent directory is already known to preprocessing_config. A relative
    SUBSLICE_DIR inherits it and only names the threshold folder::

        SUBSLICE_DIR = "threshold_0.00_cellmask_0.50"

    which survives any later rename of the directories above it. Blank means
    skip subslices, and an absolute path is used verbatim — pointing at a tree
    preprocessing did not write stays possible.

    `value` defaults to local_config's. Raises ValueError when a relative value
    does not name an existing folder, listing what is there instead.
    """
    if value is None:
        if str(_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT))
        try:
            import local_config
        except ImportError:
            return None
        value = getattr(local_config, "SUBSLICE_DIR", "")

    if not value:
        return None

    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    root = _mscarlet_cellmask_root()
    resolved = root / path
    if resolved.is_dir():
        return resolved

    available = sorted(p.name for p in root.glob("threshold_*") if p.is_dir()) \
        if root.is_dir() else []
    listing = ("\n".join(f"    {name}" for name in available) if available
               else "    (none — run preprocessing steps 3-5 first)")
    raise ValueError(
        f"\n{'='*60}\n"
        f"SUBSLICE_DIR = {value!r} is relative, so it resolves under\n"
        f"preprocessing's overlay output:\n"
        f"  {root}\n"
        f"but that folder does not exist there.\n\n"
        f"Available:\n{listing}\n\n"
        f"Set SUBSLICE_DIR to one of those names, to an absolute path, or\n"
        f"blank to skip subslices.\n"
        f"{'='*60}"
    )
