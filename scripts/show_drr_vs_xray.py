#!/usr/bin/env python3
"""
show_drr_vs_xray.py
===================
For each of the 4 smoke-test frames, render the DRR at the EPnP pose and
show it alongside the X-ray with diagnostic overlays:

  Col 1: X-ray (inverted, 1-raw)
  Col 2: DRR @ EPnP (full-res 320px)
  Col 3: Overlay  R=DRR  G=xray_matched  B=0
  Col 4: Gradient magnitude (DRR)
  Col 5: Gradient magnitude (X-ray inv)
  Col 6: GO mask  (pixels that pass both-gradient threshold)

Printed per frame:
  - NCC(drr, xray_matched)
  - GO_cost(drr, xray_matched)
  - DRR coverage (fraction of pixels > 0.01)
  - GO mask fraction (N / total)

Saved: results/figures/swaroopa_drr_vs_xray.png
"""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from swaroopa_loader import SwaroLoader, project_world_swaro, SWARO_PIX_MM, SWARO_IMG_SIZE
from deepfluoro_drr import DeepFluoroDRR
from similarity import normalized_cross_correlation, go_cost, gradient_orientation_similarity

FRAMES      = ['ap_013', 'ap_031', 'lat_003', 'lat_026']
RENDER_SIZE = 320
OUT         = ROOT / 'results/figures/swaroopa_drr_vs_xray.png'
LM_COLOURS  = {'L1':'#ff4444','L2':'#ff9900','L3':'#ffee00','L4':'#44ff44','L5':'#44ddff'}
SIGMA       = 2.0

loader  = SwaroLoader()
spec    = loader.load(frames=FRAMES, verbose=True)
drr_gen = DeepFluoroDRR(spec, hu_threshold=0.0)

lm_names = sorted(spec.landmarks_3d.keys())
pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
pix_r    = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)
scale_lm = RENDER_SIZE / SWARO_IMG_SIZE

# ── Gradient helpers ─────────────────────────────────────────────────────────
def grad_mag(img, sigma=SIGMA):
    s = gaussian_filter(img.astype(np.float32), sigma=sigma)
    gx = np.gradient(s, axis=1)
    gy = np.gradient(s, axis=0)
    return np.sqrt(gx**2 + gy**2)

def go_mask(drr, xray, sigma=SIGMA):
    dm = grad_mag(drr,  sigma)
    xm = grad_mag(xray, sigma)
    return (dm > np.median(dm)) & (xm > np.median(xm))

# ── Figure layout ─────────────────────────────────────────────────────────────
n_cols = 6
n_rows = len(FRAMES)
col_titles = [
    'X-ray (polarity-matched)',
    'DRR @ EPnP',
    'Overlay R=DRR G=xray',
    'Grad mag — DRR',
    'Grad mag — X-ray inv',
    'GO mask (both grads > median)',
]

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.2, n_rows*3.4))
fig.patch.set_facecolor('#111111')
for ci, ct in enumerate(col_titles):
    axes[0, ci].set_title(ct, color='white', fontsize=8, pad=4)

print("Rendering ...")
for ri, proj in enumerate(spec.projections):
    key  = proj.proj_key
    R_gt = proj.R_proj
    t_gt = proj.t_proj

    drr      = drr_gen.generate_from_extrinsic(R_gt, t_gt, RENDER_SIZE, pix_r, 120)
    xray_raw = cv2.resize(proj.image_raw, (RENDER_SIZE, RENDER_SIZE), interpolation=cv2.INTER_AREA)
    # Auto-detect polarity: match DRR (bone=bright) convention
    if float(proj.image_raw.mean()) < 0.5:
        xray_matched = 1.0 - xray_raw   # LAT: invert
        polarity_label = 'inv (LAT)'
    else:
        xray_matched = xray_raw          # AP: already bone-bright
        polarity_label = 'raw (AP)'

    # ── Metrics ──────────────────────────────────────────────────────────────
    coverage = float(np.count_nonzero(drr > 0.01)) / drr.size
    ncc_val  = normalized_cross_correlation(drr, xray_matched)
    go_val   = go_cost(drr, xray_matched)
    go_sim   = gradient_orientation_similarity(drr, xray_matched)
    mask     = go_mask(drr, xray_matched)
    mask_frac = float(mask.sum()) / drr.size

    print(f"  [{key}] ({polarity_label})  coverage={coverage:.2%}  NCC={ncc_val:+.4f}  "
          f"GO_cost={go_val:.4f}  GO_sim={go_sim:.4f}  mask={mask_frac:.2%}")

    # ── Projected landmarks ───────────────────────────────────────────────────
    uv = project_world_swaro(pts3d, R_gt, t_gt) * scale_lm

    # ── Images to display ────────────────────────────────────────────────────
    gm_drr  = grad_mag(drr)
    gm_xray = grad_mag(xray_matched)

    overlay = np.zeros((RENDER_SIZE, RENDER_SIZE, 3), dtype=np.float32)
    overlay[..., 0] = drr
    overlay[..., 1] = xray_matched
    overlay = np.clip(overlay, 0, 1)

    mask_img = mask.astype(np.float32)

    # Normalise grad mags for display
    def norm_disp(x):
        mx = x.max()
        return x / mx if mx > 0 else x

    imgs  = [xray_matched, drr, overlay, norm_disp(gm_drr), norm_disp(gm_xray), mask_img]
    cmaps = ['gray', 'gray', None, 'hot', 'hot', 'gray']

    for ci, (img, cmap) in enumerate(zip(imgs, cmaps)):
        ax = axes[ri, ci]
        ax.set_facecolor('#0a0a0a')
        kw = dict(interpolation='bilinear')
        if cmap == 'gray':
            ax.imshow(img, cmap='gray', vmin=0, vmax=1, **kw)
        elif cmap is None:
            ax.imshow(img, vmin=0, vmax=1, **kw)
        else:
            ax.imshow(img, cmap=cmap, **kw)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#444'); sp.set_linewidth(0.5)

        if ci == 0:
            ax.set_ylabel(key, color='white', fontsize=9)
            # annotate GT landmarks (stars)
            if hasattr(proj, 'gt_landmarks_2d') and proj.gt_landmarks_2d:
                for n, (u, v) in proj.gt_landmarks_2d.items():
                    ax.plot(u*scale_lm, v*scale_lm, '*',
                            color=LM_COLOURS.get(n, 'w'), ms=7, mew=0.8, mec='white', zorder=6)

        # Plot projected 3D landmarks on DRR and overlay columns
        if ci in (1, 2):
            for ni, (u, v) in enumerate(uv):
                if 0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE:
                    ax.plot(u, v, 'o', color=LM_COLOURS.get(lm_names[ni], 'w'),
                            ms=5, mew=0.8, mec='white', zorder=6)

    # Row annotation: metrics in title of DRR col
    axes[ri, 1].set_title(
        f"cov={coverage:.0%}  NCC={ncc_val:+.3f}\nGO_cost={go_val:.3f}  mask={mask_frac:.0%}",
        color='#aaffaa', fontsize=7, pad=3
    )

plt.tight_layout(pad=0.4)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved → {OUT}")
