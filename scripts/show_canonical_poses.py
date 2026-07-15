#!/usr/bin/env python3
"""Show DRR at EPnP (canonical) pose vs X-ray for 4 frames."""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from swaroopa_loader import SwaroLoader, project_world_swaro, SWARO_PIX_MM, SWARO_IMG_SIZE
from deepfluoro_drr import DeepFluoroDRR

FRAMES      = ['ap_013', 'ap_031', 'lat_003', 'lat_026']
RENDER_SIZE = 320
OUT         = ROOT / 'results/figures/swaroopa_canonical_poses.png'
LM_COLOURS  = {'L1':'#ff4444','L2':'#ff9900','L3':'#ffee00','L4':'#44ff44','L5':'#44ddff'}

loader  = SwaroLoader()
spec    = loader.load(frames=FRAMES, verbose=True)
drr_gen = DeepFluoroDRR(spec, hu_threshold=150.0)

lm_names = sorted(spec.landmarks_3d.keys())
pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
pix_r    = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)
scale_lm = RENDER_SIZE / SWARO_IMG_SIZE

print("Rendering canonical DRRs ...")

fig, axes = plt.subplots(len(FRAMES), 4, figsize=(16, len(FRAMES)*3.6))
fig.patch.set_facecolor('#111111')

col_titles = ['X-ray (raw)', 'X-ray (inverted)', 'DRR @ EPnP pose', 'Overlay  DRR(R) · X-ray-inv(G)']
for ci, ct in enumerate(col_titles):
    axes[0, ci].set_title(ct, color='white', fontsize=9, pad=5)

for ri, proj in enumerate(spec.projections):
    key  = proj.proj_key
    R_gt = proj.R_proj
    t_gt = proj.t_proj

    print(f"  {key}: EPnP reproj={proj.reproj_error_px:.2f}px")

    drr = drr_gen.generate_from_extrinsic(R_gt, t_gt, RENDER_SIZE, pix_r, 120)
    uv  = project_world_swaro(pts3d, R_gt, t_gt) * scale_lm

    xray_raw = cv2.resize(proj.image_raw,  (RENDER_SIZE, RENDER_SIZE), interpolation=cv2.INTER_AREA)
    xray_inv = 1.0 - xray_raw

    overlay      = np.zeros((RENDER_SIZE, RENDER_SIZE, 3), dtype=np.float32)
    overlay[...,0] = drr       # red  = DRR
    overlay[...,1] = xray_inv  # green = inverted xray
    overlay        = np.clip(overlay, 0, 1)

    imgs = [xray_raw, xray_inv, drr, overlay]
    cmaps = ['gray', 'gray', 'gray', None]

    for ci, (img, cmap) in enumerate(zip(imgs, cmaps)):
        ax = axes[ri, ci]
        ax.set_facecolor('#0a0a0a')
        kw = dict(vmin=0, vmax=1, interpolation='bilinear')
        if cmap:
            ax.imshow(img, cmap=cmap, **kw)
        else:
            ax.imshow(img, **kw)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#444'); sp.set_linewidth(0.5)

        # Row label
        if ci == 0:
            ax.set_ylabel(key, color='white', fontsize=9)
            # GT 2D landmarks (stars)
            if hasattr(proj, 'gt_landmarks_2d') and proj.gt_landmarks_2d:
                for n, (u, v) in proj.gt_landmarks_2d.items():
                    ax.plot(u*scale_lm, v*scale_lm, '*',
                            color=LM_COLOURS.get(n,'w'), ms=8, mew=1, mec='white', zorder=6)

        # Projected 3D landmarks on DRR and overlay
        if ci in (2, 3):
            for j, n in enumerate(lm_names):
                u, v = uv[j]
                if 0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE:
                    ax.plot(u, v, 'o', color=LM_COLOURS[n], ms=5, mew=1.2, mec='white', zorder=6)

        # Reproj error annotation on DRR col
        if ci == 2:
            ax.set_xlabel(f"EPnP reproj = {proj.reproj_error_px:.2f} px",
                          color='#aaaaaa', fontsize=7)

# Legend
lm_h = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
                markersize=6, label=n, mec='white') for n,c in LM_COLOURS.items()]
gt_h  = Line2D([0],[0], marker='*', color='w', markerfacecolor='white',
               markersize=8, label='GT annot (★)', mec='grey')
fig.legend(handles=lm_h+[gt_h], loc='lower center', ncol=6,
           facecolor='#1e1e1e', edgecolor='#555', labelcolor='white',
           fontsize=8, bbox_to_anchor=(0.5, 0.0))

plt.suptitle('Canonical (EPnP) Pose — DRR vs X-ray', color='white', fontsize=12, y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 1.0])
plt.subplots_adjust(hspace=0.06, wspace=0.04)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved → {OUT}")
