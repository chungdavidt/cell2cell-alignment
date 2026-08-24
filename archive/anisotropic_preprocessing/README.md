# Anisotropic preprocessing — frozen, do not run

These four files are the preprocessing steps 3–5 and their config **as they were
at commit `fee3bbd`**, the last commit before the resample became isotropic. They
are kept because every BARseq output currently sitting on disk under the old
`*_anisotropic` directory names was produced by this code, and nothing else
explains those files' physical scale.

**Frozen.** They import `preprocessing_config` names that no longer exist and
write to directory names the pipeline no longer uses. Do not run them, do not
import them, do not port constants out of them. The live pipeline is
`preprocessing/`.

## What made them anisotropic

Two separate resample factors, from `preprocessing_config.py` here:

    DOWNSAMPLE_X = INVIVO_XY_UM_PER_PX / EXVIVO_UM_PER_PX   # 7.3125
    DOWNSAMPLE_Y = INVIVO_Z_UM_PER_PX  / EXVIVO_UM_PER_PX   # 3.125

They assume the BARseq sections were cut across the 2P imaging plane, so one
in-plane BARseq axis maps to 2P XY and the other to 2P Z. Both numbers also came
from Li lab optics (2.34 µm/px XY, 1.0 µm Z) belonging to the retired JH302
dataset.

Sections are cut **axial** — the same plane the 2P images — so both BARseq
in-plane axes map to 2P XY and the two factors collapse to one:

    DOWNSAMPLE_XY = xy_um_per_px / 0.32

with `xy_um_per_px` coming from the scope named as `SCOPE` in `local_config.py`
(`scope_profiles.py`). Commits `ea8fb57` and `2486117`.

## File map

| Archived (frozen) | Live equivalent |
|---|---|
| `downsample_subslices_cellmask_anisotropic.py` | `preprocessing/downsample_subslices_cellmask.py` |
| `generate_mscarlet_cellmask_subslice_anisotropic.py` | `preprocessing/generate_mscarlet_cellmask_subslice.py` |
| `interactive_mscarlet_threshold_cellmask_subslice_anisotropic.py` | `preprocessing/interactive_mscarlet_threshold_cellmask_subslice.py` |
| `preprocessing_config.py` | `preprocessing/preprocessing_config.py` |

Output directories they wrote, all at the superseded scale:

    HYB_subslice_stitched_tif_downsampled_micronwise_anisotropic/
    mScarlet_cellmask_interactive_subslice_anisotropic/
    mScarlet_cellmask_subslice/threshold_*_cellmask_*_anisotropic/

The live pipeline writes the same three without the `_anisotropic` suffix. Those
old directories are still on the data drive under the old names; they are stale
at the wrong scale and are what P11 regenerates. Nothing reads them.
