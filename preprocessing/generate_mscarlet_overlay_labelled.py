#!/usr/bin/env python3
"""
Generate mScarlet Overlay with FOV Labels.

Creates mScarlet overlay images with FOV labels by stitching individual
FOVs, adding mScarlet cells above threshold, and labeling each FOV.

This is a Python port of generate_mscarlet_overlay_labelled.m with exact fidelity.

Uses the SAME FOV placement logic as generate_stitched_FOVs_labelled_FIXED
to ensure labels are positioned correctly.

Usage:
    python generate_mscarlet_overlay_labelled.py
    python generate_mscarlet_overlay_labelled.py --threshold 0.3
    python generate_mscarlet_overlay_labelled.py --threshold 0.3 --slice 22
    python generate_mscarlet_overlay_labelled.py --threshold 0 --slices 1 2 3 4 5 6 7 8 9 10

Output: mScarlet_overlay_dapi_labelled/threshold_X_downsampled/*.png
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import sparse
import time

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("WARNING: PIL not installed. Text labels will not be added.")

try:
    from skimage.transform import resize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    from scipy.ndimage import zoom

from config import (
    FILT_NEURONS_PATH,
    HYB_ROOT,
    MSCARLET_LABELLED_DIR,
    MSCARLET_COLUMN_INDEX,
    QC_MIN_READS,
    QC_MIN_GENES,
    DAPI_BRIGHTNESS,
    RED_OPACITY,
    LABEL_FONT_SIZE,
    LABEL_COLOR,
    LABEL_TEXT_COLOR,
    get_threshold_folder,
)
from utils.mat_io import load_filt_neurons, load_mat, get_expression_column
from utils.regression import calculate_fov_offset


def imresize_nearest(image: np.ndarray, output_shape: tuple) -> np.ndarray:
    """Resize image using nearest-neighbor interpolation."""
    if HAS_SKIMAGE:
        return resize(
            image,
            output_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(image.dtype)
    else:
        zoom_factors = (output_shape[0] / image.shape[0],
                        output_shape[1] / image.shape[1])
        return zoom(image, zoom_factors, order=0)


def add_text_label(image: np.ndarray, position: tuple, text: str) -> np.ndarray:
    """
    Add text label to image using PIL.

    Args:
        image: RGB image array (float 0-1 or uint8)
        position: (x, y) center position
        text: Label text

    Returns:
        Image with label added
    """
    if not HAS_PIL:
        return image

    # Convert to uint8 if needed
    if image.dtype == np.float64 or image.dtype == np.float32:
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)

    # Create PIL image
    pil_img = Image.fromarray(image_uint8)
    draw = ImageDraw.Draw(pil_img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", LABEL_FONT_SIZE)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("arial.ttf", LABEL_FONT_SIZE)
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Get text size for centering
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x, y = position
    text_x = x - text_width // 2
    text_y = y - text_height // 2

    # Draw background box
    padding = 4
    box_coords = [
        text_x - padding,
        text_y - padding,
        text_x + text_width + padding,
        text_y + text_height + padding
    ]
    draw.rectangle(box_coords, fill=LABEL_COLOR)

    # Draw text
    draw.text((text_x, text_y), text, fill=LABEL_TEXT_COLOR, font=font)

    # Convert back to numpy
    return np.array(pil_img).astype(image.dtype) / 255.0 if image.dtype in (np.float32, np.float64) else np.array(pil_img)


def generate_mscarlet_overlay_labelled(
    threshold: float = 0.0,
    slice_selection=None,
):
    """
    Generate mScarlet overlays with FOV labels.

    Args:
        threshold: Normalized expression threshold (0-1)
        slice_selection: Slice(s) to process (int, list, or None for all)
    """
    MAX_PIXELS = 1e7  # Downsample large canvases

    hyb_root = Path(HYB_ROOT)
    threshold_folder = get_threshold_folder(threshold)
    output_dir = Path(MSCARLET_LABELLED_DIR) / threshold_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 40)
    print("GENERATE MSCARLET OVERLAY (LABELLED)")
    print("=" * 40)
    print(f"Threshold: {threshold:.2f}")
    print(f"Output: {output_dir}")
    print()

    # Load filt_neurons
    print("Loading filt_neurons...")
    filt_neurons = load_filt_neurons(FILT_NEURONS_PATH)
    expmat = filt_neurons['expmat']

    # Apply QC filters
    if sparse.issparse(expmat):
        total_reads = np.asarray(expmat.sum(axis=1)).flatten()
        total_genes = np.asarray((expmat > 0).sum(axis=1)).flatten()
    else:
        total_reads = np.sum(expmat, axis=1)
        total_genes = np.sum(expmat > 0, axis=1)

    pass_qc = (total_reads >= QC_MIN_READS) & (total_genes >= QC_MIN_GENES)
    print(f"QC filtering: {np.sum(pass_qc)} / {len(pass_qc)} cells pass ({100*np.sum(pass_qc)/len(pass_qc):.1f}%)")

    # Get mScarlet expression
    mscarlet_expression = get_expression_column(expmat, MSCARLET_COLUMN_INDEX)
    max_expr = np.max(mscarlet_expression[pass_qc])
    print(f"Global max mScarlet (QC-passed): {max_expr} transcripts")

    mscarlet_normalized = mscarlet_expression / max_expr
    mscarlet_positive = pass_qc & (mscarlet_expression > 0)

    print(f"mScarlet+ cells: {np.sum(mscarlet_positive)} / {np.sum(pass_qc)} QC-passed cells "
          f"({100*np.sum(mscarlet_positive)/np.sum(pass_qc):.1f}%)")

    # Determine slices to process
    slice_ids = np.asarray(filt_neurons['slice']).flatten()
    fov_names = filt_neurons['fov']
    pos = np.asarray(filt_neurons['pos'])
    pos40x = np.asarray(filt_neurons['pos40x'])

    unique_slices = np.unique(slice_ids[~np.isnan(slice_ids)]).astype(int)

    if slice_selection is None:
        slices_to_process = unique_slices
        print(f"Processing ALL slices ({len(unique_slices)} total)\n")
    elif isinstance(slice_selection, int):
        slices_to_process = [slice_selection]
        print(f"Processing slice {slice_selection}\n")
    else:
        slices_to_process = list(slice_selection)
        print(f"Processing {len(slices_to_process)} specified slice(s)\n")

    # Process each slice
    for s_idx, slice_id in enumerate(slices_to_process):
        print("=" * 40)
        print(f"Slice {slice_id} ({s_idx + 1} of {len(slices_to_process)})")
        print("=" * 40)

        in_slice = slice_ids == slice_id
        if not np.any(in_slice):
            print("  Skip (no neurons)\n")
            continue

        slice_fov_names = np.array(fov_names)[in_slice]
        unique_fovs = list(np.unique(slice_fov_names))
        print(f"  Found {len(unique_fovs)} unique FOVs")

        slice_mscarlet_cells = in_slice & mscarlet_positive
        print(f"  mScarlet+ cells: {np.sum(slice_mscarlet_cells)}")

        # Compute FOV placements
        print("  Computing FOV placements...")
        placements = []
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf

        for fov_name in unique_fovs:
            fov_mask = (np.array(fov_names) == fov_name) & in_slice
            if np.sum(fov_mask) < 3:
                continue

            fov_pos = pos[fov_mask]
            fov_pos40x = pos40x[fov_mask]

            try:
                x_off, y_off = calculate_fov_offset(fov_pos, fov_pos40x, scale_factor=2.0)
            except Exception:
                continue

            placements.append({
                'fov': fov_name,
                'x_off': x_off,
                'y_off': y_off,
            })

            x_min = min(x_min, x_off + 1)
            y_min = min(y_min, y_off + 1)
            x_max = max(x_max, x_off + 3200)
            y_max = max(y_max, y_off + 3200)

        if not placements:
            print("  Skip (no valid placements)\n")
            continue

        print(f"  Computed {len(placements)} placements")

        # Calculate canvas size
        x_shift = max(0, 1 - x_min)
        y_shift = max(0, 1 - y_min)
        H = int(y_max + y_shift)
        W = int(x_max + x_shift)

        # Determine scaling
        scale = 1.0
        if H * W > MAX_PIXELS:
            scale = np.sqrt(MAX_PIXELS / (H * W))
            Hs = max(1, round(H * scale))
            Ws = max(1, round(W * scale))
            print(f"  Canvas: {H}x{W} -> {Hs}x{Ws} (scale={scale:.3f})")
        else:
            Hs, Ws = H, W
            print(f"  Canvas: {H}x{W} (no scaling)")

        # Place FOV cellmasks with UNIQUE IDs
        print("  Placing FOV cellmasks with unique IDs...")
        tic_cellmasks = time.time()
        canvas = np.zeros((Hs, Ws), dtype=np.uint32)
        fov_loaded = 0
        fov_skipped = 0
        next_cell_id = 1

        for placement in placements:
            fov_name = placement['fov']
            mask_path = hyb_root / fov_name / 'cellmask.mat'
            if not mask_path.exists():
                fov_skipped += 1
                continue

            # Load cellmask
            try:
                tmp = load_mat(mask_path)
            except Exception:
                fov_skipped += 1
                continue

            cellmask = None
            possible_names = ['maski', 'cellmask', 'mask', 'segmentation', 'seg']
            for name in possible_names:
                if name in tmp:
                    cellmask = np.asarray(tmp[name])
                    break

            if cellmask is None:
                # Try first numeric array
                for key, value in tmp.items():
                    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                        cellmask = np.asarray(value)
                        break

            if cellmask is None:
                fov_skipped += 1
                continue

            # Downsample cellmask first
            xs = round((placement['x_off'] + x_shift + 1) * scale)
            ys = round((placement['y_off'] + y_shift + 1) * scale)
            target_h = max(1, round(cellmask.shape[0] * scale))
            target_w = max(1, round(cellmask.shape[1] * scale))

            cellmask_small = imresize_nearest(cellmask, (target_h, target_w))

            # Remap cell IDs to unique global IDs
            unique_ids = np.unique(cellmask_small)
            unique_ids = unique_ids[unique_ids > 0]

            remapped_cellmask = np.zeros(cellmask_small.shape, dtype=np.uint32)
            for old_id in unique_ids:
                remapped_cellmask[cellmask_small == old_id] = next_cell_id
                next_cell_id += 1

            xe = xs + remapped_cellmask.shape[1] - 1
            ye = ys + remapped_cellmask.shape[0] - 1

            # Clip to canvas bounds
            ox1 = max(xs, 1) - 1  # 0-indexed
            oy1 = max(ys, 1) - 1
            ox2 = min(xe, Ws)
            oy2 = min(ye, Hs)

            if ox2 <= ox1 or oy2 <= oy1:
                continue

            fov_x1 = ox1 - (xs - 1)
            fov_x2 = fov_x1 + (ox2 - ox1)
            fov_y1 = oy1 - (ys - 1)
            fov_y2 = fov_y1 + (oy2 - oy1)

            # Place on canvas (first FOV wins)
            mask_to_place = remapped_cellmask[fov_y1:fov_y2, fov_x1:fov_x2]
            canvas_region = canvas[oy1:oy2, ox1:ox2]
            empty_mask = canvas_region == 0
            canvas_region[empty_mask] = mask_to_place[empty_mask]
            canvas[oy1:oy2, ox1:ox2] = canvas_region

            fov_loaded += 1

        print(f"  Loaded {fov_loaded} FOVs ({fov_skipped} skipped) ({time.time()-tic_cellmasks:.1f}s)")
        print(f"  Created canvas with {next_cell_id - 1} unique cell IDs")

        # Create RGB overlay with DAPI background
        dapi_norm = (canvas > 0).astype(float) * DAPI_BRIGHTNESS
        overlay_rgb = np.stack([dapi_norm, dapi_norm, dapi_norm], axis=2)

        # Add mScarlet overlay
        print("  Adding mScarlet cells...")
        tic_overlay = time.time()

        # Build lookup table: cell_id -> mScarlet intensity
        cell_id_to_intensity = {}

        slice_cell_indices = np.where(slice_mscarlet_cells)[0]
        cells_added = 0
        cells_below_thresh = 0
        cells_out_of_bounds = 0

        for cell_idx in slice_cell_indices:
            red_intensity = mscarlet_normalized[cell_idx]

            if red_intensity < threshold:
                cells_below_thresh += 1
                continue

            # Get cell position in stitched canvas
            x_stitched = round(pos[cell_idx, 0] * 2 * scale) + round(x_shift * scale)
            y_stitched = round(pos[cell_idx, 1] * 2 * scale) + round(y_shift * scale)

            # Convert to 0-indexed
            x_stitched = int(x_stitched) - 1
            y_stitched = int(y_stitched) - 1

            # Check bounds
            if x_stitched < 0 or x_stitched >= Ws or y_stitched < 0 or y_stitched >= Hs:
                cells_out_of_bounds += 1
                continue

            # Get unique cellmask ID at this position
            cell_id = canvas[y_stitched, x_stitched]
            if cell_id == 0:
                cells_out_of_bounds += 1
                continue

            # Store in lookup table
            if cell_id in cell_id_to_intensity:
                cell_id_to_intensity[cell_id] = max(cell_id_to_intensity[cell_id], red_intensity)
            else:
                cell_id_to_intensity[cell_id] = red_intensity
            cells_added += 1

        print(f"  Built lookup table for {cells_added} cells "
              f"({cells_below_thresh} below threshold, {cells_out_of_bounds} out of bounds)")

        # Apply red overlay using vectorized lookup
        if cells_added > 0:
            print("  Applying red overlay (single-pass lookup)...")

            # Create lookup array
            max_cell_id = next_cell_id - 1
            lookup = np.zeros(max_cell_id + 1, dtype=np.float32)
            for cell_id, intensity in cell_id_to_intensity.items():
                lookup[cell_id] = intensity

            # Vectorized lookup
            intensity_map = np.zeros((Hs, Ws), dtype=np.float32)
            nonzero_mask = canvas > 0
            intensity_map[nonzero_mask] = lookup[canvas[nonzero_mask]]

            # Create mask of mScarlet+ pixels
            mscarlet_mask = intensity_map > 0

            # Vectorized RGB blending
            overlay_rgb[:, :, 0] = overlay_rgb[:, :, 0] * (1 - RED_OPACITY * mscarlet_mask) + intensity_map * RED_OPACITY
            overlay_rgb[:, :, 1] = overlay_rgb[:, :, 1] * (1 - RED_OPACITY * mscarlet_mask)
            overlay_rgb[:, :, 2] = overlay_rgb[:, :, 2] * (1 - RED_OPACITY * mscarlet_mask)

            print(f"  Applied red overlay to {cells_added} cells (single-pass, {time.time()-tic_overlay:.1f}s)")
        else:
            print("  (No mScarlet+ cells to overlay)")

        # Add FOV labels
        if HAS_PIL:
            print("  Adding FOV labels...")
            tic_labels = time.time()
            for placement in placements:
                fov_center_x = round((placement['x_off'] + x_shift + 3200 / 2) * scale)
                fov_center_y = round((placement['y_off'] + y_shift + 3200 / 2) * scale)

                if 1 <= fov_center_x <= Ws and 1 <= fov_center_y <= Hs:
                    overlay_rgb = add_text_label(overlay_rgb, (fov_center_x, fov_center_y), placement['fov'])

            print(f"  Added {len(placements)} labels ({time.time()-tic_labels:.1f}s)")
        else:
            print("  WARNING: Skipping labels (PIL not available)")

        # Save output
        base_name = f"slice_{slice_id:02d}_labelled"
        png_path = output_dir / f"{base_name}.png"

        # Convert to uint8 and save
        overlay_uint8 = (np.clip(overlay_rgb, 0, 1) * 255).astype(np.uint8)
        if HAS_PIL:
            Image.fromarray(overlay_uint8).save(png_path)
        else:
            import matplotlib.pyplot as plt
            plt.imsave(png_path, overlay_uint8)

        print(f"  Saved PNG: {png_path}")
        print()

    print("=" * 40)
    print("COMPLETE")
    print("=" * 40)
    print(f"Outputs saved to: {output_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate mScarlet overlays with FOV labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--threshold', '-th',
        type=float,
        default=0.0,
        help='Normalized expression threshold (0-1), default: 0'
    )
    parser.add_argument(
        '--slice', '-s',
        type=int,
        default=None,
        help='Process single slice'
    )
    parser.add_argument(
        '--slices',
        type=int,
        nargs='+',
        default=None,
        help='Process multiple specific slices'
    )

    args = parser.parse_args()

    # Determine slice selection
    if args.slices is not None:
        slice_selection = args.slices
    elif args.slice is not None:
        slice_selection = args.slice
    else:
        slice_selection = None  # All

    generate_mscarlet_overlay_labelled(
        threshold=args.threshold,
        slice_selection=slice_selection,
    )


if __name__ == '__main__':
    main()
