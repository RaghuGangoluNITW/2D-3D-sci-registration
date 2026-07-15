#!/usr/bin/env python3
"""
visualize_spine_crop_heuristic.py
================================
Heuristic, image-only spine cropping for Swaroopa X-rays.
No landmarks are used.

For each frame:
- invert X-ray so bone is bright
- build a coarse body mask from thresholding
- enhance local contrast with CLAHE
- compute a vertical-column score from intensity, gradients, and local variance
- pick the best central column as the spine center
- crop a fixed-width ROI around that column inside the body bbox

Outputs a figure with three columns per frame:
    left   = full X-ray with ROI box
    middle = cropped spine ROI
    right  = per-column spine score with chosen center/crop bounds
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from swaroopa_loader import SwaroLoader, SWARO_IMG_SIZE


def to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def preprocess_xray(raw: np.ndarray) -> np.ndarray:
    inv = 1.0 - raw.astype(np.float32)
    u8 = to_u8(inv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = clahe.apply(u8).astype(np.float32) / 255.0
    return out


def body_mask(inv_img: np.ndarray) -> np.ndarray:
    u8 = to_u8(inv_img)
    thr, mask = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thr < 8:
        _, mask = cv2.threshold(u8, 8, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean[labels == largest] = 255
    return (clean > 0).astype(np.uint8)


def local_std(img: np.ndarray, ksize: int = 21) -> np.ndarray:
    mu = cv2.GaussianBlur(img, (0, 0), ksize / 6.0)
    mu2 = cv2.GaussianBlur(img * img, (0, 0), ksize / 6.0)
    var = np.maximum(mu2 - mu * mu, 0.0)
    return np.sqrt(var)


def estimate_spine_bbox(raw: np.ndarray, view: Optional[str] = None):
    proc = preprocess_xray(raw)
    mask = body_mask(proc)
    h, w = proc.shape

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        bbox = (w // 3, h // 6, 2 * w // 3, 5 * h // 6)
        debug = {
            'x_lo': 0,
            'x_hi': w,
            'col_score_full': np.zeros(w, dtype=np.float32),
            'x_center': (bbox[0] + bbox[2]) // 2,
        }
        return bbox, proc, mask, debug

    x0_body, x1_body = int(xs.min()), int(xs.max())
    y0_body, y1_body = int(ys.min()), int(ys.max())

    gx = cv2.Sobel(proc, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(proc, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    texture = local_std(proc, ksize=25)
    score_map = (0.45 * proc + 0.35 * grad + 0.20 * texture) * mask.astype(np.float32)

    x_lo = max(x0_body, int(0.18 * w))
    x_hi = min(x1_body, int(0.82 * w))
    col_score = score_map[:, x_lo:x_hi].sum(axis=0)
    col_score = cv2.GaussianBlur(col_score[np.newaxis, :], (0, 0), 9).ravel()
    x_center = int(x_lo + np.argmax(col_score))

    body_h = y1_body - y0_body + 1
    body_w = x1_body - x0_body + 1

    if view == 'lat':
        crop_w = int(0.6*w)
    else:
        crop_w = int(0.6*w)
    crop_h = int(max(0.55 * h, min(0.82 * h, 0.88 * body_h)))

    y_center = int((y0_body + y1_body) / 2)
    x0 = max(0, x_center - crop_w // 2)
    x1 = min(w, x_center + crop_w // 2)
    y0 = 0
    y1 = h

    if (x1 - x0) < 20 or (y1 - y0) < 20:
        x0, x1 = max(0, w // 3), min(w, 2 * w // 3)
        y0, y1 = max(0, h // 6), min(h, 5 * h // 6)

    col_score_full = np.zeros(w, dtype=np.float32)
    col_score_full[x_lo:x_hi] = col_score
    debug = {
        'x_lo': x_lo,
        'x_hi': x_hi,
        'col_score_full': col_score_full,
        'x_center': x_center,
    }
    return (x0, y0, x1, y1), proc, mask, debug


def draw_box(img: np.ndarray, bbox):
    x0, y0, x1, y1 = bbox
    out = np.dstack([img, img, img])
    cv2.rectangle(out, (x0, y0), (x1, y1), (0.2, 1.0, 0.2), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description='Heuristic spine crop without landmarks')
    ap.add_argument('--frames', nargs='+', default=['ap_002', 'ap_006', 'ap_010', 'lat_000', 'lat_003', 'lat_021'])
    ap.add_argument('--output', default='results/figures/swaroopa_spine_crop_heuristic.png')
    ap.add_argument('--dpi', type=int, default=140)
    args = ap.parse_args()

    loader = SwaroLoader()
    spec = loader.load(frames=args.frames, verbose=False)

    n = len(spec.projections)
    fig, axes = plt.subplots(n, 3, figsize=(11.2, 2.7 * n))
    fig.patch.set_facecolor('#111111')
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, proj in enumerate(spec.projections):
        view = 'lat' if proj.proj_key.startswith('lat') else 'ap'
        bbox, proc, mask, debug = estimate_spine_bbox(proj.image_raw, view=view)
        x0, y0, x1, y1 = bbox
        crop = proc[y0:y1, x0:x1]

        ax = axes[row, 0]
        ax.imshow(draw_box(proc, bbox), interpolation='bilinear')
        ax.set_title(f'{proj.proj_key} — heuristic ROI', color='white', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')

        ax = axes[row, 1]
        ax.imshow(crop, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax.set_title(f'crop  {crop.shape[1]}×{crop.shape[0]}', color='white', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')

        ax = axes[row, 2]
        # Recompute the same score map used to pick columns, then compute
        # per-row (horizontal) score sums inside the cropped ROI (y0:y1, x0:x1).
        gx = cv2.Sobel(proc, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(proc, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        texture = local_std(proc, ksize=25)
        score_map = (0.45 * proc + 0.35 * grad + 0.20 * texture) * mask.astype(np.float32)

        # Sum along the x-axis inside the crop to get per-row scores
        if x1 > x0 and y1 > y0:
            row_sums = score_map[y0:y1, x0:x1].sum(axis=1)
        else:
            row_sums = np.array([], dtype=np.float32)

        # Place the crop-row sums into a full-height vector for plotting
        full_h = proc.shape[0]
        score_full = np.zeros(full_h, dtype=np.float32)
        if row_sums.size > 0:
            score_full[y0:y1] = row_sums

        y = np.arange(full_h)
        if score_full.max() > 0:
            score_plot = score_full / score_full.max()
        else:
            score_plot = score_full

        # Plot normalized crop-row sums horizontally (score on x, row on y)
        ax.plot(score_plot, y, color='#66ccff', linewidth=1.4)
        y_center = int((y0 + y1) / 2)
        ax.axhline(y_center, color='lime', linewidth=1.4, linestyle='-')
        ax.axhspan(y0, y1, color='lime', alpha=0.16)
        ax.set_ylim(full_h - 1, 0)
        ax.set_xlim(0, max(1.02, float(score_plot.max()) + 0.05))
        ax.set_title(f'row crop-sum  center_row={y_center}', color='white', fontsize=8)
        ax.set_xlabel('norm crop-sum', color='#cccccc', fontsize=7)
        ax.set_ylabel('image row', color='#cccccc', fontsize=7)
        ax.tick_params(colors='#bbbbbb', labelsize=6)
        ax.set_facecolor('#161616')
        ax.grid(True, axis='x', color='#333333', alpha=0.5, linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=args.dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
