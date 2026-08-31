# cell2cell-alignment

Python pipeline for aligning ex vivo brain slice images to in vivo 2-photon imaging volumes, using the castalign / LineStuffUp GUI for interactive registration.

The pipeline is config-driven: you fill in `local_config.py` with whatever data you have (in vivo stack, ex vivo block, BARseq subslices — any subset), the graph builder adds a node for each one, and you align them in the notebook. BARseq users run an optional preprocessing pipeline first to generate the subslice overlays.

---

## 1. Prerequisites

- Python 3.9+
- Two separate Python venvs, because castalign and cellpose have incompatible dependencies (cellpose pins older numpy, plus pulls in the torch/torchvision ML stack that you don't want bloating the alignment env).

| Env | Used for | Install |
|---|---|---|
| `.castalign-venv/` | Alignment graph builder + `castalign_testground.ipynb`. Preprocessing pipeline likely also runs here (TBD — see note). | `pip install -r requirements-castalign.txt` |
| `.cellpose-venv/` | Cellpose 3D segmentation for alignment validation (code pending). | `pip install -r requirements-cellpose.txt` |

Both venvs live at project root and are gitignored.

```bash
# castalign env
python3 -m venv .castalign-venv
source .castalign-venv/bin/activate        # Linux/Mac
# .castalign-venv\Scripts\activate         # Windows (PowerShell/cmd)
pip install -r requirements-castalign.txt

# cellpose env (separate terminal recommended)
python3 -m venv .cellpose-venv
source .cellpose-venv/bin/activate         # Linux/Mac
# .cellpose-venv\Scripts\activate          # Windows (PowerShell/cmd)
pip install -r requirements-cellpose.txt
```

**Preprocessing env — TBD.** The `preprocessing/` scripts use numpy/scipy/h5py/tifffile/scikit-image and don't touch castalign or cellpose directly. Currently assumed to run in `.castalign-venv/`. Needs review — may warrant its own `.preprocessing-venv/` if version conflicts emerge.

---

## 2. Initial setup (one-time per machine)

### Step 1 — Clone this repo and create the envs

Clone this repo, then follow the env-creation commands in Section 1. castalign installs as a pip dependency — no separate clone needed.

### Step 2 — Create your local config

```bash
cd cell2cell-alignment
cp local_config.example.py local_config.py
```

`local_config.py` is gitignored — each person's paths stay local. Don't commit it.

### Step 3 — Fill in the config

Open `local_config.py` and edit the paths. See the next section for what each variable means.

---

## 3. The config file — `local_config.py`

This is the only file you need to edit to get running. It holds all machine-specific paths.

| Variable | Required? | Used by | Purpose |
|---|---|---|---|
| `GRAPH_PATH` | **Yes** | Graph builder + notebook | Where the alignment graph `.db` file is saved / loaded |
| `INVIVO_PATH_RED` | Optional | Graph builder | In vivo 2P TIFF — red (alignment) channel |
| `INVIVO_PATH_GREEN` | Optional | Graph builder | In vivo 2P TIFF — green (signal) channel |
| `BLOCK_STACK_PATH_RED` | Optional | Graph builder | Ex vivo block TIFF — red (alignment) channel |
| `BLOCK_STACK_PATH_GREEN` | Optional | Graph builder | Ex vivo block TIFF — green (signal) channel |
| `SUBSLICE_DIR` | Optional | Graph builder | Directory of BARseq subslice overlays |
| `DATA_ROOT` | BARseq only | Preprocessing pipeline | Raw BARseq data root |
| `OUTPUT_ROOT` | BARseq only | Preprocessing pipeline | Where preprocessing writes its outputs |
| `SCOPE` | **Yes** | Everything | The microscope that acquired this subject's data — the source of truth for every pixel size |

### Rules for the optional data inputs

The graph builder turns each path into a node. The rules are intentionally strict:

- **Blank (`""`)** → skip that node, don't add it to the graph
- **Set but the file/directory doesn't exist** → hard error (catches typos)
- **Set and exists** → add to the graph
- **GREEN set without RED for the same volume** → hard error (would dangle the Identity edge)

You need **at least one** of the four 2P paths or `SUBSLICE_DIR` set, or the builder raises a `ValueError` ("nothing to build").

### Multi-channel volumes — red + green

Each 2P volume is acquired in two channels: red (sparsely-labelled, used for alignment fitting and Cellpose segmentation) and green (signal of interest). The two channels share a voxel grid in hardware, so they enter the graph as sibling nodes joined by `castalign.base.Identity()`. Rigid + nonlinear edges are fitted only on the `_red ↔ _red` pair; queries between green channels (or red↔green) compose through Identity automatically and cost nothing.

Per-volume node names:
- `invivo_red`, `invivo_green`
- `block_stack_red`, `block_stack_green`

If you only have a red channel today, leave the green path blank — the green node and Identity edge are simply skipped, and you can add green later by setting the path and re-running the builder (idempotent).

### What each variable holds

**`GRAPH_PATH`**
Where the alignment graph file lives. Used by both the graph builder (writes here) and the notebook (reads from here). Typically ends in `.db`. Put it somewhere with enough disk space — graphs with full BARseq subslices can run hundreds of MB.
```python
GRAPH_PATH = "/Users/yourname/data/linestuffup_output/my_experiment.db"
```

**`INVIVO_PATH_RED` / `INVIVO_PATH_GREEN`**
Multi-page TIFFs of the in vivo 2P volume, one per channel. Both channels must come from the same acquisition (same voxel grid). The builder asserts this from TIFF metadata.
- Li lab data: typically the flipped output from `prepare_invivo_volume.py` (Y-axis mirror correction).
- Huang lab data: typically the raw max-projection TIFFs — no preprocessing needed.

**`BLOCK_STACK_PATH_RED` / `BLOCK_STACK_PATH_GREEN`**
Multi-page TIFFs of the ex vivo tissue block (2P volume imaged before slicing), one per channel.

**`SUBSLICE_DIR`, `DATA_ROOT`, `OUTPUT_ROOT`** — BARseq preprocessing variables
All three are used by the BARseq preprocessing stage (Section 4) and the subslice nodes it produces:
- `DATA_ROOT` — where the raw BARseq data lives
- `OUTPUT_ROOT` — where preprocessing writes its intermediate outputs
- `SUBSLICE_DIR` — the specific subfolder under `OUTPUT_ROOT` containing the final `slice*_subslice_mScarlet_cellmask.tif` overlays that the graph builder consumes

Leave all three blank if you aren't running BARseq preprocessing right now. You can fill them in and re-run the graph builder any time later — it will pick up the subslices then.

**`SCOPE`** — see the next section.

### Microscope resolution — `SCOPE`

You name the microscope once:

```python
SCOPE = "huang_lab"
```

and every pixel size in the project follows from it. The profiles live in `scope_profiles.py` at the project root — stdlib-only, so preprocessing, the graph builder and the cellpose venv all read the same table:

| Profile key | XY (µm/px) | Z (µm/px) | FOV | Used by |
|---|---|---|---|---|
| `li_lab` | 2.34 | 1.0 | 1200 µm | JH302 (retired) |
| `huang_lab` | 1.1000 | 1.0 | 563.2 µm | BY95 |

`huang_lab` read 0.3910 µm/px over a 200.19 µm field until 2026-08-30, when the stacks' own `XResolution` tag was read: `909090/1000000` with ImageJ unit µm, i.e. **1.1000 µm/px over 563.2 µm**. The declaration had never been checked against a file, because `get_tiff_resolution` recognised only the literal string `micron` and these files say `µm` — so the guard had nothing to compare and returned silently.

`huang_lab_566um` (1.1055 µm/px, 566.08 µm, 2.0 µm Z — by84, by94, by89) was deleted the same day. 563.2 and 566.08 µm are 0.51% apart, so the "one scanner at two zooms ~2.83× apart" story was itself an artifact of the wrong 200.19, and the two keys collided inside the 5% tolerance `identify_scope` matches on. Those three subjects now hard-error until their pitch is re-derived from their own tags — which is also the measurement that settles whether the two profiles were ever different.

What reads it:

| Consumer | Uses |
|---|---|
| Preprocessing | `DOWNSAMPLE_XY = xy / 0.32` — the BARseq resample factor (1.1000 → 3.4375×, 2.34 → 7.3125×) |
| Graph builder | `(z, xy, xy)` on every 2P node, `(20.0, xy, xy)` on every BARseq subslice node |
| `validate_mnn.py` | µm scale for MNN centroid distances |

`SCOPE` is authoritative, not a fallback. Where a TIFF carries `XResolution` metadata it is still read, but only to **check** the declaration: a file whose pixel size disagrees stops the run rather than silently overriding `SCOPE`. Files with no metadata, or with the 72-DPI placeholder the raw BARseq TIFFs carry, are covered by `SCOPE` with no complaint.

To add a microscope, add an entry to `MICROSCOPE_PROFILES` in `scope_profiles.py` — nothing else needs editing. An unrecognized pixel size errors with a copy-pasteable template.

---

## 4. BARseq preprocessing

Generates the subslice overlays that `SUBSLICE_DIR` points at. Requires raw BARseq data laid out under `DATA_ROOT`.

```bash
cd preprocessing/
python run_pipeline.py                # all slices
python run_pipeline.py --test         # slice 22 only, quick sanity check
python run_pipeline.py --start-from 3 # resume from step N
```

The outputs land under `OUTPUT_ROOT`. Point `SUBSLICE_DIR` in `local_config.py` at the resulting `mScarlet_cellmask_subslice/threshold_*/` folder.

---

## 5. Build the graph

```bash
python alignment/subslice_graph_builder.py
```

The builder reads the red/green 2P paths and `SUBSLICE_DIR` from your config and adds nodes for whichever are set. Sibling red/green channels of one volume are joined by `castalign.base.Identity`. It saves to `GRAPH_PATH`.

**Re-runs are idempotent.** If the graph already exists, it loads and adds any configured nodes that aren't already in it. So if you later fill in another path, just re-run. To wipe and rebuild, edit the `__main__` call to pass `force_rebuild=True`.

---

## 6. Align in the notebook

Open `alignment/castalign_testground.ipynb`. Run cells 0–6 (imports + load graph). From there, use whichever modes match the nodes you put in your graph:

| Mode | What it aligns | Requires |
|---|---|---|
| A / B / C | BARseq subslice → ex vivo block | `SUBSLICE_DIR` + `BLOCK_STACK_PATH_RED` |
| D | Ex vivo block → in vivo | `BLOCK_STACK_PATH_RED` + `INVIVO_PATH_RED` |

A typical session: run Mode D first (it's the one-time 3D → 3D registration), then Modes A/B/C per slice if you have BARseq subslices. Use `R` (affine) followed by `V` (3D triangulation) for the best results. Press `q` in the GUI to save and quit.

After aligning, the notebook has sections for verification (Pearson r, napari overlay), nonlinear refinement, and warped-image export.

---

## 7. Project layout

```
cell2cell-alignment/
├── README.md                      ← this file
├── local_config.py                ← your paths (gitignored)
├── local_config.example.py        ← template (tracked)
│
├── alignment/                     ← graph builder + alignment notebook
│   ├── subslice_graph_builder.py  ← builds the alignment graph from config
│   └── castalign_testground.ipynb ← interactive alignment GUI
│
├── preprocessing/                 ← optional, BARseq only
│   ├── run_pipeline.py            ← 5-step orchestrator
│   └── ...                        ← individual steps + optional utilities
│
└── utilities/                     ← shared helpers (I/O, graph ops, plotting)
```

---

## 8. Common issues

**`ImportError: local_config.py not found`**
You haven't copied the template yet. Run `cp local_config.example.py local_config.py`.

**`FileNotFoundError: INVIVO_PATH_RED is set ... but the file does not exist`**
Typo in the path. The builder refuses to silently skip configured-but-broken paths — fix the path or leave it blank to skip the node.

**`ValueError: No inputs configured`**
All four 2P paths and `SUBSLICE_DIR` are blank. Set at least one.

**`ValueError: ..._GREEN is set but ..._RED is blank`**
Green-without-red would dangle the Identity edge in the graph. Either set the matching `_RED` path or leave the `_GREEN` path blank.

**`ValueError: Loaded graph contains legacy non-suffixed nodes`**
The graph predates the multi-channel refactor. Rebuild with `force_rebuild=True` to migrate to the new `_red` / `_green` naming. Rigid fits will need to be redone (typically minutes via Mode D).

**`ValueError: Could not identify microscope from image resolution`**
Your TIFF's XY resolution doesn't match any profile in `MICROSCOPE_PROFILES`. The error message prints a copy-pasteable block — paste it into `scope_profiles.py` and re-run.

**Graph exists but I want to rebuild from scratch**
Either delete the `.db` file at `GRAPH_PATH`, or edit `__main__` in `subslice_graph_builder.py` to pass `force_rebuild=True`.
