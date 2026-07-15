#!/usr/bin/env python3
"""
visualize_clahe_sweep.py
========================
Show 4 Swaroopa X-rays processed with a grid of CLAHE parameters:
  rows  — clipLimit  (1, 2, 4, 8, 16)
  cols  — tileGridSize (4, 8, 16, 32)

Each cell shows: [original inverted | CLAHE result]
alongside the EPnP DRR for reference.

Layout per frame block:
  left  : original inverted X-ray
  middle: CLAHE-processed X-ray
  right : EPnP DRR

Usage:
    python scripts/visualize_clahe_sweep.py
    python scripts/visualize_clahe_sweep.py --frames ap_000 ap_010 lat_000 lat_010
"""
import argparse
import json
import sys
from pathlib import Path
from itertools import product

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CLAHE parameter grid
# ---------------------------------------------------------------------------
CLIP_LIMITS  = [1.0, 2.0, 4.0, 8.0, 16.0]
TILE_SIZES   = [4, 8, 16, 32]

# Default 4 frames to visualise
DEFAULT_FRAMES = ['ap_010', 'ap_020', 'lat_000', 'lat_025']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gray_float(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img.astype(np.float32) / 255.0


def apply_clahe(gray_float: np.ndarray, clip_limit: float, tile: int) -> np.ndarray:
    """Invert then apply CLAHE, return float32 [0,1]."""
    inverted = 1.0 - gray_float
    u8 = (inverted * 255).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    out = clahe.apply(u8)
    return out.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='CLAHE parameter sweep for Swaroopa X-rays')
    p.add_argument('--poses_json', type=Path,
                   default=ROOT / 'results/swaroopa_epnp_poses_diffdrr.json')
    p.add_argument('--drr_dir',    type=Path,
                   default=ROOT / 'results/swaroopa_epnp_drrs_diffdrr')
    p.add_argument('--xray_dir',   type=Path,
                   default=ROOT / 'data/swaroopa_labelled')
    p.add_argument('--frames',     nargs='+', default=DEFAULT_FRAMES,
                   help='Exactly 4 proj_keys to display')
    p.add_argument('--output',     type=Path,
                   default=ROOT / 'results/figures/swaroopa_clahe_sweep.png')
    p.add_argument('--dpi',        type=int, default=150)
    p.add_argument('--thumb',      type=int, default=128,
                   help='Thumbnail size in pixels')
    return p.parse_args()


def main():
    args = parse_args()

    # --- Load poses JSON ---
    with open(args.poses_json) as f:
        meta = json.load(f)
    frame_records = meta['frames']  # rel_path -> dict

    # Build key->record mapping
    key_to_rec = {v['proj_key']: (k, v) for k, v in frame_records.items()}

    frames = args.frames
    n_frames = len(frames)

    n_clips = len(CLIP_LIMITS)
    n_tiles = len(TILE_SIZES)

    # -----------------------------------------------------------------------
    # Figure layout:
    #   - Outer rows = clip limits  (n_clips)
    #   - Outer cols = tile sizes   (n_tiles)
    #   - Each cell contains a 1×(n_frames*2+1) sub-strip:
    #       for each frame: [inverted | clahe] ... then [DRR(s)] in a separate strip
    #
    # Simpler layout:
    #   rows = n_clips * n_frames,  cols = n_tiles * 3  (orig | clahe | drr)
    #   Group by frame within each clip×tile block.
    #
    # Actually cleanest: one page per frame (4 pages), each page shows clip×tile grid
    # with original | clahe | drr trio. → 4 output PNGs.
    # But user asked for visualisation (singular), so let's do:
    #
    # rows = clip limits,  cols = tile sizes
    # each cell = small strip [inv_xray | clahe | drr] for the 4 frames side by side
    # -----------------------------------------------------------------------

    thumb = args.thumb

    # We'll make 4 separate figures, one per frame
    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for frame_key in frames:
        if frame_key not in key_to_rec:
            print(f"[WARN] {frame_key} not in poses JSON, skipping")
            continue

        rel_path, rec = key_to_rec[frame_key]

        # Load X-ray
        xray_path = args.xray_dir / rec['xray_relative_path']
        xray_raw = load_gray_float(xray_path)
        xray_inv = 1.0 - xray_raw

        # Load DRR
        drr_path = args.drr_dir / rec['drr_relative_path']
        drr_img = load_gray_float(drr_path)

        # Resize to thumb
        def thumb_img(im):
            return cv2.resize(im, (thumb, thumb), interpolation=cv2.INTER_AREA)

        xray_t = thumb_img(xray_inv)
        drr_t  = thumb_img(drr_img)

        # ---- Build figure: rows=clipLimits, cols=tileSizes
        # Each cell: [orig_inv | clahe | drr] side by side
        cell_w = 3  # sub-images per cell
        fig_w = n_tiles * cell_w
        fig_h = n_clips

        fig, axes = plt.subplots(
            fig_h, fig_w,
            figsize=(fig_w * thumb / args.dpi + 1.5,
                     fig_h * thumb / args.dpi + 1.5),
            squeeze=False,
        )
        fig.suptitle(
            f'CLAHE sweep  —  {frame_key}  '
            f'(reproj={rec["reproj_error_px"]:.1f}px)\n'
            f'rows=clipLimit, cols=tileGridSize',
            fontsize=9, y=0.995
        )

        # Column headers (tile sizes)
        for ci, tile in enumerate(TILE_SIZES):
            col_base = ci * cell_w + 1  # centre cell
            axes[0, col_base].set_title(f'tile={tile}', fontsize=7, pad=2)

        for ri, clip in enumerate(CLIP_LIMITS):
            # Row label
            axes[ri, 0].set_ylabel(f'clip={clip:.0f}', fontsize=7, rotation=0,
                                   labelpad=28, va='center')
            for ci, tile in enumerate(TILE_SIZES):
                col_base = ci * cell_w

                clahe_t = thumb_img(apply_clahe(xray_raw, clip, tile))

                for sub, im in enumerate([xray_t, clahe_t, drr_t]):
                    ax = axes[ri, col_base + sub]
                    ax.imshow(im, cmap='gray', vmin=0, vmax=1, aspect='equal',
                              interpolation='nearest')
                    ax.axis('off')
                    if ri == 0 and sub == 0:
                        ax.set_title('orig', fontsize=5, pad=1)
                    elif ri == 0 and sub == 1:
                        ax.set_title('CLAHE', fontsize=5, pad=1)
                    elif ri == 0 and sub == 2:
                        ax.set_title('DRR', fontsize=5, pad=1)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = args.output.parent / f'{args.output.stem}_{frame_key}.png'
        fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        output_paths.append(out_path)
        print(f"Saved: {out_path}")

    print(f"Done. {len(output_paths)} figures written.")


if __name__ == '__main__':
    main()
