#!/usr/bin/env python3
"""
show_initial_poses.py
=====================
For every Swaroopa X-ray frame render a DRR at the EPnP-initialised pose
and show it side-by-side with the X-ray.  No optimisation is run.

Layout : N_frames rows × 2 cols  [X-ray | Initial-pose DRR]
         Landmark centroids overlaid on both columns.

Output : results/figures/swaroopa_initial_poses.png
"""

import sys, cv2
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from swaroopa_loader import SwaroLoader, SWARO_PIX_MM, SWARO_IMG_SIZE
from run_swaroopa_diffdrr import build_subject, DiffDRRGenerator
import torch

# ── Config ────────────────────────────────────────────────────────────────────
RENDER_SIZE = 224          # px — fast to render
OUT = Path('results/figures/swaroopa_initial_poses.png')
OUT.parent.mkdir(parents=True, exist_ok=True)

LM_COLOURS = {'L1': '#ff4444', 'L2': '#ff9900', 'L3': '#ffee00',
              'L4': '#44ff44', 'L5': '#44ddff'}
BG = '#111111'
FG = '#dddddd'
matplotlib.rcParams.update({
    'text.color': FG, 'axes.labelcolor': FG,
    'figure.facecolor': BG, 'axes.facecolor': BG,
})

# ── Load ALL frames ───────────────────────────────────────────────────────────
print('Loading Swaroopa specimen (all frames) ...')
loader  = SwaroLoader()
spec    = loader.load(verbose=True)          # all 35 frames
projs   = spec.projections

print('Building DiffDRR subject ...')
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
subject = build_subject(spec)
drr_gen = DiffDRRGenerator(subject, device, ct_origin_lps=spec.ct_origin)
print(f'DiffDRR ready  (device={device})')

lm_names = sorted(spec.landmarks_3d.keys())
pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
pix      = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)

# ── Figure ────────────────────────────────────────────────────────────────────
N = len(projs)
fig, axes = plt.subplots(N, 2,
                         figsize=(7, N * 2.2),
                         gridspec_kw={'hspace': 0.04, 'wspace': 0.02})
fig.patch.set_facecolor(BG)

axes[0, 0].set_title('X-ray  +  GT landmarks', fontsize=9, pad=6, color=FG)
axes[0, 1].set_title('Initial pose DRR  +  projected lm', fontsize=9, pad=6, color=FG)

for row_i, proj in enumerate(projs):
    key = proj.proj_key
    print(f'  [{row_i+1}/{N}] {key} ...')

    xray = cv2.resize(proj.image_raw, (RENDER_SIZE, RENDER_SIZE),
                      interpolation=cv2.INTER_AREA)

    if proj.R_proj is not None:
        drr = drr_gen.generate_from_extrinsic(
            proj.R_proj, proj.t_proj, RENDER_SIZE, pix)
        uv_drr = drr_gen.project_pts(
            proj.R_proj, proj.t_proj, pts3d, RENDER_SIZE, pix)
        uv_xray = uv_drr   # same projection for both columns
        reproj_str = f'reproj={proj.reproj_error_px:.1f}px'
    else:
        drr = np.zeros((RENDER_SIZE, RENDER_SIZE), dtype=np.float32)
        uv_drr = np.full((len(lm_names), 2), -1.0)
        uv_xray = uv_drr
        reproj_str = 'no EPnP'

    for ci, (img, uv, subtitle) in enumerate([
        (xray, uv_xray, key),
        (drr,  uv_drr,  reproj_str),
    ]):
        ax = axes[row_i, ci]
        ax.set_facecolor(BG)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, -0.005, subtitle, transform=ax.transAxes,
                fontsize=6.5, ha='center', va='top', color=FG)
        if ci == 0:
            view = 'AP' if key.startswith('ap') else 'LAT'
            ax.set_ylabel(f'{view}\n{key}', fontsize=7, rotation=90,
                          labelpad=3, color=FG)

        # Landmark overlay
        for j, name in enumerate(lm_names):
            u, v = uv[j]
            if 0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE:
                ax.plot(u, v, 'o',
                        color=LM_COLOURS.get(name, 'white'),
                        markersize=5, markeredgewidth=0.6,
                        markeredgecolor='white', zorder=5)
                ax.text(u + 3, v - 3, name,
                        color=LM_COLOURS.get(name, 'white'),
                        fontsize=5, fontweight='bold', zorder=6)

plt.savefig(OUT, dpi=130, bbox_inches='tight', facecolor=BG)
print(f'\nSaved → {OUT}')
