# Cardinal-direction labeling system — discussion summary (parked)

**Parked 2026-05-30.** Triggered by the cross-modality alignment work — now that we're aligning BARseq slices to ex-vivo 2P to in-vivo 2P, we can no longer be orientation-agnostic. Need a way to assign D/V, M/L (or R/L), A/P to each volume's array axes. Discussion stopped at an open decision; re-enter from "Open decision" below when this becomes urgent.

---

## What castalign provides (verified at commit `1360831`, v0.2.0)

- **No anatomical/cardinal-direction system at all.** Coordinates are raw `(z, y, x)` numpy/napari indices. Zero mentions of dorsal/ventral/medial/lateral/anterior/posterior/coronal/sagittal/axial in source or docs.
- **Only orientation primitives are geometric, not anatomical:** `Flip` / `FlipFixed` (flip along an array axis — used by the GUI's "Flip along z axis (f)" button), plus `rotation_matrix` in utils. Operate on axes z/y/x, not on anatomy.
- **The one useful hook:** `Graph.add_node(name, image=None, compression="normal", metadata=None)` (graph.py:240) stores free-form per-node metadata in `self.node_metadata[name]`. Persisted via `repr()` in both SQLite and HDF5 save paths (graph.py:114–115, 184–185, 208–213) and reloaded via `eval()`. Graph-level `self.metadata` also exists (graph.py:31). Any repr-able Python object rides along with the graph for free — natural storage for orientation labels.

## The proposed labeling scheme (general, future-proof)

Borrow the neuroimaging "orientation string" convention (what nibabel/ITK use):

- **Per-volume 3-letter code** for `(z, y, x)` = anatomical direction of *increasing* index. e.g. `PVL` means `+z=Posterior, +y=Ventral, +x=Lateral`.
- Letters from `{D/V, M/L (or R/L), A/P}`. **M/L is hemisphere-relative; R/L is absolute** — R/L catches mirror-flips (chirality changes) explicitly, M/L silently flips chirality across the midline. For a single-hemisphere ALM crop, M/L is fine; a general system probably wants R/L.
- **Pick one canonical target code** (e.g. matching the in-vivo volume's native orientation, since in-vivo anatomical is the never-deformed ground truth for the project).
- **Generic algorithm**: given `(source_code, target_code)`, compute the permutation + flips that take source → target. This is exactly nibabel's `aff2axcodes` / `ornt_transform` (~20 lines, fully general). Future horizontal/sagittal/oblique-but-axis-aligned cuts just get a different *label* — the function is untouched.

### For 2D / 1-voxel-thick slices

The 3-letter code still applies, but `z` is degenerate (thickness 1) and just encodes the **slice-normal anatomical direction** = the cutting plane:
- coronal → z-normal = A/P
- sagittal → z-normal = M/L
- horizontal → z-normal = D/V

The two in-plane letters carry which way is dorsal / medial within the 2D image. Same scheme handles 3D volumes and 2D slices.

**One thing the label cannot do:** recover out-of-plane tilt — a single 1-voxel plane under-constrains pitch/roll (see `project_nonlinear_fit_direction.md`). The label fixes only the cardinal assignment; any real out-of-plane tilt has to come from which in-vivo slice the BARseq plane is matched to.

## Image-quality / degradation analysis — castalign vs preprocessing

Verified mechanism: `castalign.base.transform_image` resamples with

```python
scipy.ndimage.map_coordinates(img, disp, prefilter=False, order=(0 if labels else 1))
```

- Intensity images → `order=1` (linear) — **a low-pass filter; every pass blurs (attenuates high frequencies)**.
- Label/segmentation images → `order=0` (nearest) — exact, no blur.
- `thickness==1` hack at `base.py:198` so 1-voxel slices don't vanish in `map_coordinates` — castalign **can** rotate flat slices in-plane.

**Governing principle:** resample the original image exactly **once**. Each extra interpolating pass is permanent blur, compounded and irreversible.

### Two reorientations behave completely differently

| Reorientation | Interpolates? | Degradation |
|---|---|---|
| **90° axis permutation + flip** (cardinal axis assignment) | No — pure reindex | **Lossless**, anywhere |
| **Arbitrary in-plane rotation** | Yes — linear (order=1) | Blurs once per pass |

### Counting resamples

- **Bake an in-plane rotation into a preprocessing TIFF:** preprocessing rotates (resample #1, blur) → castalign alignment warp (resample #2, blur) = **two passes, compounded blur**. Image degraded before castalign sees it, then degraded again.
- **Express the same rotation as a castalign edge:** composes with the alignment transform; castalign applies the single composed transform to the original image = **one resample, minimal blur**. This is exactly what castalign's compose-and-resample-once architecture is built for.

### Compression — not a discriminator

castalign default `compression="normal"` is **lossy**:
- volumes (when `min(shape) > 10`) → **VP9 video, 20 Mbit/s** (utils.py:164,182)
- thin stacks → **JPEG quality 90** (utils.py:190)
- only `level="label"` → lossless gzip (utils.py:153)

The raw uncompressed array is kept in memory during a session (`node_images[name] = image`); the lossy hit only bites **on save/reload** — the `.db` persists only the compressed blob. Applies to both approaches equally, so it doesn't favor one over the other. For nodes where exact voxel values matter (e.g. Cellpose-centroid MNN validation), pass `compression="label"` at a large disk cost.

## Conclusion / proposed split

**Split by whether the operation interpolates, not by "known vs unknown":**

- **Lossless reindex** (90° transpose + flip — the cardinal axis assignment): do it in **preprocessing**. No degradation either way, and the GUI shows volumes right-way-up so cross-modality landmark picking is anatomically intuitive. Record the orientation code in `node_metadata` for the resulting node.
- **Any true rotation / sub-voxel shift**: keep it **inside castalign** as a composed transform. Never bake it into a preprocessing TIFF — that forces a second resample and compounds the blur.

The trap to avoid: "correcting" a non-90° angle in preprocessing for nicer display. That's an interpolating pass that costs a permanent blur on top of the alignment resample. Non-90° rotations belong in the graph so they fold into the one final resample.

## Open decision — where the discussion stopped

For the BARseq slice in-plane reorientation specifically: **is it constrained to 90° steps (→ preprocessing, lossless reindex) or a true arbitrary angle (→ castalign edge, single composed resample)?**

If 90° steps: build the label system + a preprocessing reindex function (nibabel-style permutation/flip from source→target code); store the label in `node_metadata`.

If arbitrary angle: still build the label system + lossless preprocessing reindex for cardinal axis assignment (so things display sensibly), but express any non-cardinal residual as a castalign edge so it composes with the alignment and only resamples once.

### Decisions to make when re-entering

1. **Canonical target frame** — probably the in-vivo volume's native orientation (since in-vivo anatomical is the project's ground truth for everything; canonical = no reindex needed for in-vivo). Confirm.
2. **M/L (hemisphere-relative) or R/L (absolute)** — decides whether the system catches mirror-flips. Single-hemisphere ALM work tolerates M/L; multi-hemisphere or chirality-sensitive work needs R/L.
3. **Per-modality source orientation codes** — need to be determined empirically for BARseq slice (currently coronal, will vary), ex-vivo 2P block, in-vivo 2P.
4. **Where to store** — `node_metadata` on the graph (rides along, queryable) is the obvious answer; possibly also a preprocessing config sidecar so the source code is recorded outside the graph too.

## Related memory files

- `feedback_deterministic_preprocessing_over_notebook_transforms.md` — known geometric transforms belong in preprocessing
- `project_future_orientation_preprocessing.md` — bake dataset-specific orientation transforms into preprocessing scripts
- `project_nonlinear_fit_direction.md` — 1-voxel-slice failure mode (under-constrained pitch/roll)
- `project_canonical_invivo_node_planned.md` — a related but distinct idea (canonical *rotated* in-vivo node for axis-pinned operations)
- `project_invivo_anatomical_ground_truth.md` — in-vivo anatomical is the universal ground-truth frame
