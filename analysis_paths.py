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


def _preprocessing_roots() -> "tuple[Path, Path]":
    """(overlay root, alignment-tif root) from preprocessing_config.

    A relative SUBSLICE_DIR is resolved against both: step 4's overlays live
    under the first, generate_alignment_tif.py's output under the second. Only
    the second holds anything the graph builder will ingest — it takes ALIGN
    tifs and nothing else — but the lookup stays symmetric so a folder name
    under either root resolves instead of silently missing.
    """
    cfg = _preprocessing_config()
    return (Path(cfg.MSCARLET_CELLMASK_DIR), Path(cfg.SUBSLICE_ALIGN_DIR))


def _preprocessing_config(reason: str = None):
    """preprocessing_config, imported lazily.

    Imported here rather than at module scope because preprocessing_config
    validates the whole preprocessing config at import — it raises when
    DATA_ROOT or SCOPE is unset — and a 2P-only run has no reason to satisfy
    that. Only a relative SUBSLICE_DIR and the raw-channel lookup need it.

    `reason` names what asked for it, so the error says which config line to
    fix rather than always blaming SUBSLICE_DIR.
    """
    preprocessing = _PROJECT_ROOT / "preprocessing"
    if str(preprocessing) not in sys.path:
        sys.path.insert(0, str(preprocessing))
    try:
        import preprocessing_config
    except Exception as e:
        why = reason or (
            "SUBSLICE_DIR is relative, so it is resolved against "
            "preprocessing's output tree"
        )
        raise ValueError(
            f"{why} — but preprocessing_config could not be loaded:\n"
            f"  {type(e).__name__}: {e}\n\n"
            f"Either fix that (it needs DATA_ROOT and SCOPE), or set "
            f"SUBSLICE_DIR to an absolute path."
        )
    return preprocessing_config


def hyb_downsampled_dir() -> Path:
    """Where downsample_subslices_cellmask.py writes every channel.

    One folder holds `slice{N}_subslice_{DAPI,GCAMP,MSCARLET}.tif` beside
    `slice{N}_subslice_CELLMASK.h5`, all resampled to the same target shape
    computed once from the cellmask — so a raw channel image is pixel-for-pixel
    on the grid the ALIGN tif is painted on. That is what makes them Identity
    siblings in the graph rather than something needing registration.
    """
    cfg = _preprocessing_config(
        reason="The BARseq raw channels are preprocessing output, so their "
               "folder comes from preprocessing_config"
    )
    return Path(cfg.HYB_DOWNSAMPLED_DIR)


def resolve_subslice_dir(value: Optional[str] = None) -> Optional[Path]:
    """Resolve ``SUBSLICE_DIR`` to an absolute directory.

    The BARseq images the graph builder reads are preprocessing output, so
    their parent directory is already known to preprocessing_config. A relative
    SUBSLICE_DIR inherits it and names only the trailing folder — resolved
    against BOTH output roots::

        SUBSLICE_DIR = "qc20_5_ge1"                    # alignment TIFs
        SUBSLICE_DIR = "threshold_0.00_cellmask_0.50"  # step 4 overlays

    which survives any later rename of the directories above it. Resolving an
    overlay folder is not the same as the graph builder accepting one: it
    ingests ALIGN tifs only and raises on a folder holding none. A name that
    exists under both roots raises rather than guessing. Blank means
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

    roots = _preprocessing_roots()
    hits = [r / path for r in roots if (r / path).is_dir()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(
            f"\n{'='*60}\n"
            f"SUBSLICE_DIR = {value!r} names a folder under more than one\n"
            f"preprocessing output root:\n"
            + "".join(f"  {h}\n" for h in hits) +
            f"\nUse an absolute path to say which one.\n"
            f"{'='*60}"
        )

    available = []
    for r in roots:
        if r.is_dir():
            available += [f"    {r.name}/{p.name}" for p in sorted(r.iterdir())
                          if p.is_dir()]
    listing = "\n".join(available) if available else \
        "    (none — run the preprocessing pipeline first)"
    raise ValueError(
        f"\n{'='*60}\n"
        f"SUBSLICE_DIR = {value!r} is relative, so it resolves under\n"
        f"preprocessing's output roots:\n"
        + "".join(f"  {r}\n" for r in roots) +
        f"but no such folder exists in either.\n\n"
        f"Available:\n{listing}\n\n"
        f"Set SUBSLICE_DIR to one of those names (the trailing folder only), to\n"
        f"an absolute path, or blank to skip subslices.\n"
        f"{'='*60}"
    )
