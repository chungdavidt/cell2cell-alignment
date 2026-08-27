# Unused preprocessing scripts — frozen 2026-08-27

Two scripts moved out of `preprocessing/` after a full-codebase review found that
neither is reachable, neither is consumed, and neither currently runs. They are kept
rather than deleted because the *ideas* in them are still live; the code is not.

**Frozen.** Do not run, do not import, do not port constants out of them.

---

## refine_subslices_by_threshold.py

**What it did.** Step 1 (`identify_mscarlet_subslices.py`) picks FOVs on a binary
test — does this FOV hold *any* QC-passing mScarlet+ cell — then takes the largest
8-connected cluster and adds bridge FOVs for contiguity. That is deliberately
permissive, so the footprint carries FOVs holding one or two faint cells. This script
re-ran the selection with two extra conditions: a cell counted only above a normalized
`--threshold`, and an FOV survived only with `>= --min-cells` such cells (default 3).
It then repaired connectivity — `--bridge-strategy adaptive|keep|remove` plus a fresh
8-connected pass — and wrote `subslice_definitions_thresh_{t:.2f}.mat` beside a
per-slice before/after comparison PNG, never touching the canonical definitions.

**Why it is unused.**
- Nothing dispatches it. Its only reference was `run_pipeline.py`'s `OPTIONAL_STEPS`,
  a table that was never read (removed 2026-08-27 along with this move).
- Nothing consumes its output. `subslice_definitions_thresh_*.mat` appeared exactly
  twice in the repo: its own docstring and its own write line. Step 2 reads the
  unsuffixed `SUBSLICE_DEFINITIONS_FILE`.
- It could not run. `:247` `np.asarray(filt_neurons['expmat'])` with no `issparse`
  guard — `utilities/mat_io.py` keeps `expmat` sparse above 1e6 elements, and
  `np.asarray` on a `csr_matrix` gives a 0-d object array, so `np.sum(..., axis=1)`
  raises. It was the only preprocessing script without that guard; fourteen others
  have it. The 2026-08-21 marker-column sweep fixed the line below and missed this one
  — what you would expect of a file nobody exercises.

**Its function survives, split in two and done better.** QC 20/5 now gates step 1
itself, so ~74% of rows are gone before an FOV is considered. Marker brightness moved
to `ALIGN_MIN_ROLONIES` at step 6, which cuts in **absolute rolony counts** into its
own output folder — better than a normalized `count / max_expr` fraction, which means
something different in every brain because it depends on the single brightest cell in
the dataset (see `check_rolony_cutoff.py`'s docstring).

**The one thing nothing else does: prune FOVs.** Shrinking the footprint is still a
live idea for the step 4 performance problem (per-cell full-canvas scan, ~8 min/slice,
~8 h for step 4 alone). If it comes back, re-derive it against absolute counts, and
use `utilities/graph_utils.py` rather than the local copies here — this file's
`add_bridge_fovs` takes a third `all_fov_names` argument the shared one does not, so
the two can already disagree about which bridges to add.

---

## create_ex_vivo_volume.py

**What it did.** Port of `create_ex_vivo_volume_dtc.m`: rotate/shift 2D stitched
sections per an alignment CSV (`Slide,Slice,X_px_,Y_px_,Degrees`) and stack them into
a 3D ex-vivo volume, writing `aligned_stack_DAPI.tif` + `alignment_metadata.mat`.

**Why it is unused.**
- It cannot be imported. `:40-41` does `from utilities.mat_io import ...` with no
  `sys.path` bootstrap and — alone among preprocessing scripts — no
  `preprocessing_config` import, which is what puts the repo root on the path for its
  siblings. The repo has no `setup.py`, `pyproject.toml`, `setup.cfg` or `conftest.py`.
  Python puts the *script's* directory on `sys.path[0]`, never the cwd, so `utilities`
  is unreachable from any invocation.
- Same dead `OPTIONAL_STEPS` entry as above.
- Keyed to the retired JH302 dataset: its section-name parser expects
  `JH302_SLIDE_SLICE` (`:15`, `:234`).
- Oldest of the group — last substantive commit 2026-04-10.

**The idea is still open.** `project_todo_masterlist.md` notes that now the planes
match and sections stack at 20 µm, stacking ex vivo into a volume could matter again.
Archiving this parks a design rather than discarding a tool — but restore it by
rewriting against the current config surface, not by moving the file back.
