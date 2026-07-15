#!/usr/bin/env python3
"""
show_fallback_poses.py
======================
Shows what the DRR looks like at the anatomy (fallback) pose — i.e., the
pose that would be used if there were no 2D landmark annotations to run EPnP.

Layout: 4 rows (frames) × 5 cols
  [X-ray raw | DRR @ EPnP | DRR @ fallback AP | DRR @ fallback LAT | overlay fallback]
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from swaroopa_loader import (
    SwaroLoader, _anatomy_pose, project_world_swaro,
    SWARO_PIX_MM, SWARO_IMG_SIZE,
)
from deepfluoro_drr import DeepFluoroDRR

FRAMES      = ['ap_013', 'ap_031', 'lat_003', 'lat_026']
RENDER_SIZE = 256
OUT         = ROOT / 'results/figures/swaroopa_fallback_poses_ccw.png'
LM_COLOURS  = {'L1':'#ff4444','L2':'#ff9900','L3':'#ffee00','L4':'#44ff44','L5':'#44ddff'}

loader  = SwaroLoader()
spec    = loader.load(frames=FRAMES, verbose=True)
drr_gen = DeepFluoroDRR(spec, hu_threshold=150.0)

lm_names = sorted(spec.landmarks_3d.keys())
pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
pix_r    = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)
scale_lm = RENDER_SIZE / SWARO_IMG_SIZE

# 90° CCW in-plane rotation matrix (around camera z / optical axis)
# P_cam_new = Rz @ P_cam_old  =>  R_new = Rz @ R_old,  t_new = Rz @ t_old
Rz90_ccw = np.array([[ 0., 1., 0.],
                      [-1., 0., 0.],
                      [ 0., 0., 1.]], dtype=np.float64)

# Precompute both anatomy fallback poses, then rotate 90° CCW
_R_ap,  _t_ap  = _anatomy_pose(spec.landmarks_3d, azimuth_deg=0.0)
_R_lat, _t_lat = _anatomy_pose(spec.landmarks_3d, azimuth_deg=90.0)
R_fb_ap,  t_fb_ap  = Rz90_ccw @ _R_ap,  Rz90_ccw @ _t_ap
R_fb_lat, t_fb_lat = Rz90_ccw @ _R_lat, Rz90_ccw @ _t_lat

print("Rendering fallback DRRs ...")
for k, (R, t) in [('AP fallback',  (R_fb_ap, t_fb_ap)),
                   ('LAT fallback', (R_fb_lat, t_fb_lat))]:
    uv = project_world_swaro(pts3d, R, t)
    print(f"  {k}: projected LMs at {uv.round(1).tolist()}")

# ── Figure ────────────────────────────────────────────────────────────────────
# Cols:  X-ray | EPnP DRR | Fallback DRR (correct view) | Overlay fallback
N   = len(FRAMES)
fig, axes = plt.subplots(N, 4, figsize=(16, N * 3.4))
fig.patch.set_facecolor('#111111')

col_titles = [
    'X-ray (inverted)',
    'DRR @ EPnP pose\n(with landmarks)',
    'DRR @ anatomy fallback\n(90° CCW, no landmarks)',
    'Overlay  DRR(R) · X-ray(G)\n(fallback)',
]
for ci, ct in enumerate(col_titles):
    axes[0, ci].set_title(ct, color='white', fontsize=8.5, pad=5)

for ri, proj in enumerate(spec.projections):
    key      = proj.proj_key
    view_tag = key.split('_')[0]   # 'ap' or 'lat'

    # Pick the matching anatomy fallback for this view
    R_fb = R_fb_ap  if view_tag == 'ap'  else R_fb_lat
    t_fb = t_fb_ap  if view_tag == 'ap'  else t_fb_lat

    drr_epnp = drr_gen.generate_from_extrinsic(proj.R_proj, proj.t_proj, RENDER_SIZE, pix_r, 120)
    drr_fb   = drr_gen.generate_from_extrinsic(R_fb,        t_fb,        RENDER_SIZE, pix_r, 120)

    uv_epnp = project_world_swaro(pts3d, proj.R_proj, proj.t_proj) * scale_lm
    uv_fb   = project_world_swaro(pts3d, R_fb, t_fb) * scale_lm

    xray_inv = cv2.resize(1.0 - proj.image_raw, (RENDER_SIZE, RENDER_SIZE), interpolation=cv2.INTER_AREA)

    overlay = np.zeros((RENDER_SIZE, RENDER_SIZE, 3), dtype=np.float32)
    overlay[..., 0] = drr_fb
    overlay[..., 1] = xray_inv
    overlay = np.clip(overlay, 0, 1)

    col_data = [
        (xray_inv, 'gray', None),
        (drr_epnp, 'gray', uv_epnp),
        (drr_fb,   'gray', uv_fb),
        (overlay,  None,   uv_fb),
    ]

    for ci, (img, cmap, uv) in enumerate(col_data):
        ax = axes[ri, ci]
        ax.set_facecolor('#0a0a0a')
        kw = dict(vmin=0, vmax=1, interpolation='bilinear')
        ax.imshow(img, cmap=cmap, **kw) if cmap else ax.imshow(img, **kw)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#444'); sp.set_linewidth(0.5)

        if ci == 0:
            ax.set_ylabel(key, color='white', fontsize=9)
            # GT 2D landmarks
            if hasattr(proj, 'gt_landmarks_2d') and proj.gt_landmarks_2d:
                for n, (u, v) in proj.gt_landmarks_2d.items():
                    ax.plot(u * scale_lm, v * scale_lm, '*',
                            color=LM_COLOURS.get(n, 'w'), ms=8, mew=1, mec='white', zorder=6)

        if uv is not None:
            for j, n in enumerate(lm_names):
                u, v = uv[j]
                if 0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE:
                    ax.plot(u, v, 'o', color=LM_COLOURS[n], ms=5, mew=1.2, mec='white', zorder=6)

        if ci == 1:
            ax.set_xlabel(f"reproj = {proj.reproj_error_px:.2f} px", color='#aaaaaa', fontsize=7)
        if ci == 2:
            ax.set_xlabel(
                f"az = {'0°  (AP)' if view_tag == 'ap' else '90° (LAT)'}  — no landmarks",
                color='#aaaaff', fontsize=7)

# Legend
lm_h = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
                markersize=6, label=n, mec='white') for n, c in LM_COLOURS.items()]
gt_h  = Line2D([0],[0], marker='*', color='w', markerfacecolor='white',
               markersize=8, label='GT annot (★)', mec='grey')
fig.legend(handles=lm_h + [gt_h], loc='lower center', ncol=6,
           facecolor='#1e1e1e', edgecolor='#555', labelcolor='white',
           fontsize=8, bbox_to_anchor=(0.5, 0.0))

plt.suptitle(
    'Fallback anatomy pose (90° CCW)  vs  EPnP pose\n'
    'Fallback = centroid-centred, azimuth 0° (AP) / 90° (LAT), then rotated 90° CCW in-plane',
    color='white', fontsize=11, y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 1.0])
plt.subplots_adjust(hspace=0.06, wspace=0.04)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved → {OUT}")
