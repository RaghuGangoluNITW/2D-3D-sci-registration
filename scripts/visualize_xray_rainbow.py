#!/usr/bin/env python3
"""Show 4 Swaroopa X-rays with a rainbow colormap + shared colorbar."""
import json, sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parent.parent

FRAMES       = ['ap_010', 'ap_020', 'lat_000', 'lat_025']
POSES_JSON   = ROOT / 'results/swaroopa_epnp_poses_diffdrr.json'
XRAY_ROOT    = ROOT / 'data/swaroopa_labelled'
OUTPUT       = ROOT / 'results/figures/swaroopa_xray_rainbow.png'
CMAP         = 'gray'

# ---------------------------------------------------------------------------

def load_gray_float(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def suppress_highlights(
    img: np.ndarray,
    threshold: float = 0.9,
    min_blob_px: int  = 500,
    feather_sigma: int = 31,
    darken: float      = 0.5,
) -> np.ndarray:
    """
    1. Threshold pixels > `threshold` → binary mask.
    2. Remove connected components smaller than `min_blob_px` pixels.
    3. Feather the surviving mask with a Gaussian blur (sigma=feather_sigma).
    4. Reduce brightness of the masked region by `darken`
       (0 = black, 1 = unchanged, 0.5 = half as bright).
    """
    u8 = (img * 255).clip(0, 255).astype(np.uint8)
    # --- 1. binary mask ---
    _, hard_mask = cv2.threshold(u8, int(threshold * 255), 255, cv2.THRESH_BINARY)

    # --- 2. remove small blobs ---
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hard_mask, connectivity=8)
    clean_mask = np.zeros_like(hard_mask)
    for lbl in range(1, n_labels):                       # skip background (0)
        if stats[lbl, cv2.CC_STAT_AREA] >= min_blob_px:
            clean_mask[labels == lbl] = 255

    # --- 3. feather edges with Gaussian blur ---
    k = feather_sigma | 1                                 # must be odd
    soft_mask = cv2.GaussianBlur(clean_mask.astype(np.float32),
                                 (k * 4 + 1, k * 4 + 1), k)
    soft_mask = (soft_mask / soft_mask.max()).clip(0, 1) if soft_mask.max() > 0 else soft_mask

    # --- 4. reduce brightness in masked area ---
    # blend: out = img * (1 - soft_mask*(1-darken))
    scale = 1.0 - soft_mask * (1.0 - darken)
    return (img * scale).clip(0, 1).astype(np.float32)

THRESHOLD     = 0.6
DARKEN_LEVELS = [None, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]  # None = original inverted

def normalise(img: np.ndarray) -> np.ndarray:
    """Stretch image to full [0, 1] range."""
    mn, mx = img.min(), img.max()
    if mx > mn:
        return (img - mn) / (mx - mn)
    return img.copy()

def main():
    with open(POSES_JSON) as f:
        meta = json.load(f)
    key_to_rec = {v['proj_key']: (k, v) for k, v in meta['frames'].items()}

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    norm_color = mcolors.Normalize(vmin=0, vmax=1)
    cmap_rainbow = matplotlib.colormaps[CMAP]

    n_rows = len(DARKEN_LEVELS)
    n_cols = len(FRAMES)

    # rows = darken levels, cols = frames, last row = colorbar
    fig = plt.figure(figsize=(n_cols * 3.8, n_rows * 3.8 + 0.7))
    gs = fig.add_gridspec(n_rows + 1, n_cols,
                          height_ratios=[20] * n_rows + [1],
                          hspace=0.06, wspace=0.04)

    # pre-load raw images once
    raws = {}
    for fkey in FRAMES:
        if fkey not in key_to_rec:
            print(f"[WARN] {fkey} not found, skipping")
            continue
        _, rec = key_to_rec[fkey]
        xray_path = XRAY_ROOT / rec['xray_relative_path']
        raws[fkey] = (load_gray_float(xray_path), rec)

    for ri, darken in enumerate(DARKEN_LEVELS):
        for ci, fkey in enumerate(FRAMES):
            if fkey not in raws:
                continue
            gray, rec = raws[fkey]
            if darken is None:
                processed = normalise(1.0 - gray)
                row_label = 'original\n(invert only)'
            else:
                inverted = suppress_highlights(1.0 - gray,
                                               threshold=THRESHOLD,
                                               min_blob_px=500,
                                               feather_sigma=31,
                                               darken=darken)
                processed = normalise(inverted)
                row_label = f'darken={darken:.1f}'
            ax = fig.add_subplot(gs[ri, ci])
            ax.imshow(processed, cmap=cmap_rainbow, norm=norm_color,
                      aspect='equal', interpolation='lanczos')
            ax.axis('off')
            if ri == 0:
                ax.set_title(f'{fkey}\n(reproj={rec["reproj_error_px"]:.1f} px)',
                             fontsize=9)
            if ci == 0:
                ax.set_ylabel(row_label, fontsize=9,
                              rotation=0, labelpad=52, va='center')

    # shared colorbar
    cbar_ax = fig.add_subplot(gs[n_rows, :])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm_color, cmap=cmap_rainbow),
                      cax=cbar_ax, orientation='horizontal')
    cb.set_label(f'Normalised intensity  (threshold={THRESHOLD}, feather σ=31)', fontsize=10)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle(f'Swaroopa X-rays — darken sweep  (threshold={THRESHOLD})',
                 fontsize=13, y=1.002)
    fig.savefig(OUTPUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == '__main__':
    main()
