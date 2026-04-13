# cell2cell-alignment

Python pipeline for aligning ex vivo brain slice images to in vivo 2-photon imaging volumes, using the castalign / LineStuffUp GUI for interactive registration. Supports two workflows:

- **2P-only alignment** — align an ex vivo block to an in vivo stack directly. No BARseq data needed.
- **BARseq + 2P alignment** — preprocess BARseq coronal slices (FOV stitching, mScarlet cell detection, anisotropic downsampling), then align to both an ex vivo block and the in vivo volume via a two-hop graph.

---

## 1. Prerequisites

- Python 3.9+ (a conda env is recommended)
- A local clone of the castalign repo (separate from this one) — you'll point this repo at it via `CASTALIGN_ROOT`
- Python packages: `numpy`, `scipy`, `matplotlib`, `tifffile`, `h5py`, `scikit-image`, `Pillow`, `pandas`, `imageio`

There is no `requirements.txt` yet. Install manually:

```bash
conda create -n castalign python=3.10
conda activate castalign
pip install numpy scipy matplotlib tifffile h5py scikit-image pillow pandas imageio
# castalign itself is imported by path (see CASTALIGN_ROOT below), not pip-installed
```

---

## 2. Initial setup (one-time per machine)

### Step 1 — Clone both repos

Clone this repo and the castalign repo to wherever you want them on your disk. They don't need to live in the same folder — you'll connect them via `CASTALIGN_ROOT` in the next step.

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
| `CASTALIGN_ROOT` | **Yes** | Graph builder + notebook | Path to your local castalign repo clone |
| `GRAPH_PATH` | **Yes** | Graph builder + notebook | Where the alignment graph `.db` file is saved / loaded |
| `INVIVO_PATH` | Optional | Graph builder | Path to in vivo 2P TIFF stack |
| `BLOCK_STACK_PATH` | Optional | Graph builder | Path to ex vivo block TIFF stack |
| `SUBSLICE_DIR` | Optional | Graph builder | Directory of BARseq anisotropic subslice overlays |
| `DATA_ROOT` | BARseq only | Preprocessing pipeline | Raw BARseq data root |
| `OUTPUT_ROOT` | BARseq only | Preprocessing pipeline | Where preprocessing writes its outputs |

### Rules for the three optional data inputs

`INVIVO_PATH`, `BLOCK_STACK_PATH`, and `SUBSLICE_DIR` are the three things the graph builder can add as nodes. The rules are intentionally strict:

- **Blank (`""`)** → skip that node type, don't add it to the graph
- **Set but the file/directory doesn't exist** → hard error (catches typos)
- **Set and exists** → add to the graph

You need **at least one** set, or the builder raises a `ValueError` ("nothing to build").

### What each variable holds

**`CASTALIGN_ROOT`**
Path to the folder that *contains* the `castalign/` package folder. Not the package itself.
```python
CASTALIGN_ROOT = "/Users/yourname/code/castalign"
# where castalign/castalign/base.py exists
```

**`GRAPH_PATH`**
Where the alignment graph file lives. Used by both the graph builder (writes here) and the notebook (reads from here). Typically ends in `.db`. Put it somewhere with enough disk space — graphs with full BARseq subslices can run hundreds of MB.
```python
GRAPH_PATH = "/Users/yourname/data/linestuffup_output/my_experiment.db"
```

**`INVIVO_PATH`**
A single multi-page TIFF of the in vivo 2P volume.
- Li lab data: typically the flipped output from `prepare_invivo_volume.py` (Y-axis mirror correction).
- Huang lab data: typically the raw max-projection TIFF — no preprocessing needed.

**`BLOCK_STACK_PATH`**
A single multi-page TIFF of the ex vivo tissue block (2P volume imaged before slicing).

**`SUBSLICE_DIR`**
Only used for BARseq. Directory containing the anisotropic `slice*_subslice_mScarlet_cellmask.tif` overlays that come out of the preprocessing pipeline. Leave blank if you don't have BARseq data.

**`DATA_ROOT` / `OUTPUT_ROOT`**
Only used by the preprocessing pipeline under `preprocessing/`. Leave blank if you're doing 2P-only alignment.

### Microscope resolution

You don't set pixel size anywhere in the config — the graph builder reads it from each TIFF's metadata and matches against `MICROSCOPE_PROFILES` in `alignment/subslice_graph_builder.py`. Currently supports:

| Scope | XY (µm/px) | Z (µm/px) | FOV |
|---|---|---|---|
| Li lab 2P | 2.34 | 1.0 | 1200 µm |
| Huang lab 2P | 1.1055 | 2.0 | 566.08 µm |

If your microscope isn't in the list, the builder errors with a copy-pasteable template telling you exactly what to add to `MICROSCOPE_PROFILES`.

---

## 4. Running the alignment (2P-only workflow)

Minimal setup. Skip all of `preprocessing/`.

1. Fill in `local_config.py`: `CASTALIGN_ROOT`, `INVIVO_PATH`, `BLOCK_STACK_PATH`, `GRAPH_PATH`. Leave `SUBSLICE_DIR`, `DATA_ROOT`, `OUTPUT_ROOT` blank.
2. Build the graph:
   ```bash
   python alignment/subslice_graph_builder.py
   ```
   This adds one `invivo_ref` node and one `ex_vivo_block` node, then saves to `GRAPH_PATH`.
3. Open the notebook:
   ```
   alignment/castalign_testground.ipynb
   ```
4. Run cells 0–6 (imports + load graph). Skip the slice-selector and Modes A/B/C (those are for BARseq subslices you don't have). Jump to **Section 9 — Mode D: Block → In-Vivo Alignment**.
5. Align (`R` for affine, then `V` for 3D triangulation). Press `q` to save + quit.
6. Run the verification + export cells (Sections 11–12).

### Re-running the builder

The builder is idempotent — if the graph already exists, it loads it and adds whatever configured nodes are missing. So if you later decide to also add a `SUBSLICE_DIR`, just fill in the config and re-run. Use `force_rebuild=True` (edit the `__main__` call) to wipe and start over.

---

## 5. Running the alignment (BARseq + 2P workflow)

Full pipeline. Requires the raw BARseq output layout under `DATA_ROOT`.

1. Fill in **all** of `local_config.py` (all 7 variables, including `SUBSLICE_DIR` and `DATA_ROOT` / `OUTPUT_ROOT`).
2. Run preprocessing:
   ```bash
   cd preprocessing/
   python run_pipeline.py               # all slices
   python run_pipeline.py --test        # slice 22 only, for a quick check
   python run_pipeline.py --start-from 3 # resume from step N
   ```
   This generates stitched subslices, anisotropic downsampling, and mScarlet overlays under `OUTPUT_ROOT`. Point `SUBSLICE_DIR` at the `mScarlet_cellmask_subslice/threshold_*_anisotropic/` folder it produces.
3. Build the graph:
   ```bash
   python alignment/subslice_graph_builder.py
   ```
4. Open `alignment/castalign_testground.ipynb` and work through all modes (A/B/C for per-slice alignment to the block, D for block → in vivo).

---

## 6. Project layout

```
cell2cell-alignment/
├── README.md                      ← this file
├── local_config.py                ← your paths (gitignored)
├── local_config.example.py        ← template (tracked)
│
├── alignment/                     ← for 2P-only or post-preprocessing
│   ├── subslice_graph_builder.py  ← builds the alignment graph from config
│   └── castalign_testground.ipynb ← interactive alignment GUI
│
├── preprocessing/                 ← BARseq only — skip for 2P-only workflow
│   ├── run_pipeline.py            ← 5-step orchestrator
│   └── ...                        ← individual steps + optional utilities
│
└── utilities/                     ← shared helpers (I/O, graph ops, plotting)
```

---

## 7. Common issues

**`ImportError: local_config.py not found`**
You haven't copied the template yet. Run `cp local_config.example.py local_config.py`.

**`FileNotFoundError: INVIVO_PATH is set ... but the file does not exist`**
Typo in the path. The builder refuses to silently skip configured-but-broken paths — fix the path or leave it blank to skip the node.

**`ValueError: No inputs configured`**
All three of `INVIVO_PATH`, `BLOCK_STACK_PATH`, `SUBSLICE_DIR` are blank. Set at least one.

**`ValueError: Could not identify microscope from image resolution`**
Your TIFF's XY resolution doesn't match any profile in `MICROSCOPE_PROFILES`. The error message prints a copy-pasteable block — paste it into `alignment/subslice_graph_builder.py` and re-run.

**Graph exists but I want to rebuild from scratch**
Either delete the `.db` file at `GRAPH_PATH`, or edit `__main__` in `subslice_graph_builder.py` to pass `force_rebuild=True`.
