#!/usr/bin/env python3
"""
visualize_before_after_epnp.py
================================
3-column grid for 10 Swaroopa X-rays:
  Col 1  — Before  : inverted X-ray (no processing)
  Col 2  — EPnP    : DRR at EPnP init pose
  Col 3  — After   : highlight-suppressed + normalised X-ray
                     (threshold=0.6, darken=0.8, feather σ=31)

Usage:
    python scripts/visualize_before_after_epnp.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT         = Path(__file__).resolve().parent.parent
POSES_JSON   = ROOT / 'results/swaroopa_epnp_poses_diffdrr.json'
DRR_DIR      = ROOT / 'results/swaroopa_epnp_drrs_diffdrr'
XRAY_ROOT    = ROOT / 'data/swaroopa_labelled'
OUTPUT       = ROOT / 'results/figures/swaroopa_before_epnp_after.png'

THRESHOLD    = 0.6
DARKEN       = 0.8
FEATHER_SIG  = 31
MIN_BLOB_PX  = 500
N_FRAMES     = 10
THUMB        = 300   # pixels per thumbnail
DPI          = 150

# ---------------------------------------------------------------------------

def load_gray_float(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def suppress_highlights(img, threshold, min_blob_px, feather_sigma, darken):
    u8 = (img * 255).clip(0, 255).astype(np.uint8)
    _, hard_mask = cv2.threshold(u8, int(threshold * 255), 255, cv2.THRESH_BINARY)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hard_mask, connectivity=8)
    clean_mask = np.zeros_like(hard_mask)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_blob_px:
            clean_mask[labels == lbl] = 255
    k = feather_sigma | 1
    soft_mask = cv2.GaussianBlur(clean_mask.astype(np.float32),
                                 (k * 4 + 1, k * 4 + 1), k)
    if soft_mask.max() > 0:
        soft_mask = (soft_mask / soft_mask.max()).clip(0, 1)
    scale = 1.0 - soft_mask * (1.0 - darken)
    return (img * scale).clip(0, 1).astype(np.float32)


def normalise(img: np.ndarray) -> np.ndarray:
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn) if mx > mn else img.copy()


def thumb(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

# ---------------------------------------------------------------------------

def main():
    with open(POSES_JSON) as f:
        meta = json.load(f)

    # sort frames for reproducibility; pick first N_FRAMES
    records = sorted(meta['frames'].items(),
                     key=lambda kv: kv[1]['proj_key'])[:N_FRAMES]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        N_FRAMES, 3,
        figsize=(3 * THUMB / DPI + 1.2, N_FRAMES * THUMB / DPI + 0.6),
        squeeze=False,
    )

    col_titles = ['Before\n(inverted)', 'EPnP DRR', 'After\n(suppressed + norm)']
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=10, pad=6)

    for ri, (rel_path, rec) in enumerate(records):
        proj_key  = rec['proj_key']
        xray_path = XRAY_ROOT / rec['xray_relative_path']
        drr_path  = DRR_DIR   / rec['drr_relative_path']

        gray = load_gray_float(xray_path)

        # --- Before ---
        before = thumb(normalise(1.0 - gray), THUMB)

        # --- EPnP DRR ---
        drr = thumb(normalise(load_gray_float(drr_path)), THUMB)

        # --- After ---
        suppressed = suppress_highlights(1.0 - gray,
                                         threshold=THRESHOLD,
                                         min_blob_px=MIN_BLOB_PX,
                                         feather_sigma=FEATHER_SIG,
                                         darken=DARKEN)
        after = thumb(normalise(suppressed), THUMB)

        for ci, im in enumerate([before, drr, after]):
            ax = axes[ri, ci]
            ax.imshow(im, cmap='gray', vmin=0, vmax=1,
                      aspect='equal', interpolation='lanczos')
            ax.axis('off')

        # row label on the left
        axes[ri, 0].set_ylabel(
            f'{proj_key}\n{rec["reproj_error_px"]:.1f} px',
            fontsize=7, rotation=0, labelpad=58, va='center'
        )

    fig.suptitle(
        f'Before / EPnP DRR / After  —  threshold={THRESHOLD}, darken={DARKEN}',
        fontsize=11, y=1.002
    )
    plt.tight_layout()
    fig.savefig(OUTPUT, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == '__main__':
    main()
