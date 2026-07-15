#!/usr/bin/env python3
"""
visualize_diffdrr_grid.py
=========================
Registration grid for swaroopa_diffdrr_results.json.

Layout: 8 rows × 3 cols
  • 2 best AP  success frames  (lowest final PDE, success=True)
  • 2 worst AP fail frames     (highest final PDE, success=False or PDE got worse)
  • 2 best LAT success frames
  • 2 worst LAT fail frames

Each row: [X-ray + GT landmarks] | [Initial pose DRR + lm] | [Final pose DRR + lm]

Renders DRRs via DiffDRRGenerator (same renderer as the registration run).
Output: results/figures/diffdrr_registration_grid.png
"""

import sys, json, cv2
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from swaroopa_loader import (
    SwaroLoader,
    SWARO_PIX_MM, SWARO_IMG_SIZE,
)
from deepfluoro_loader import perturb_extrinsic
from run_swaroopa_diffdrr import build_subject, DiffDRRGenerator

import torch

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS     = Path('results/swaroopa_diffdrr_results.json')
OUT         = Path('results/figures/diffdrr_registration_grid.png')
OUT.parent.mkdir(parents=True, exist_ok=True)
RENDER_SIZE = 224
LM_COLOURS  = {'L1': '#ff4444', 'L2': '#ff9900', 'L3': '#ffee00',
               'L4': '#44ff44', 'L5': '#44ddff'}
BG          = '#111111'
FG          = '#dddddd'

matplotlib.rcParams.update({
    'text.color': FG, 'axes.labelcolor': FG, 'figure.facecolor': BG,
    'axes.facecolor': BG,
})

# ── Load results ──────────────────────────────────────────────────────────────
with open(RESULTS) as f:
    data = json.load(f)
pp = data['swaroopa']['per_projection']

def pick_frames(prefix, n_best=2, n_worst=2):
    """Return (best_success, worst_fail) frame keys for given prefix."""
    sub = {k: v for k, v in pp.items() if k.startswith(prefix)}
    succ = {k: v for k, v in sub.items() if v['success'] == 'True'}
    fail = {k: v for k, v in sub.items() if v['success'] != 'True'}
    # best success = lowest final PDE (best actual accuracy)
    best = sorted(succ, key=lambda k: succ[k]['final_pde_mm'])[:n_best]
    # worst fail = highest final PDE
    worst = sorted(fail, key=lambda k: fail[k]['final_pde_mm'], reverse=True)[:n_worst]
    return best, worst

ap_best,  ap_worst  = pick_frames('ap')
lat_best, lat_worst = pick_frames('lat')

ROW_FRAMES = ap_best + ap_worst + lat_best + lat_worst
ROW_LABELS = (
    [f'AP success ✓ — {k}' for k in ap_best]  +
    [f'AP fail    ✗ — {k}' for k in ap_worst] +
    [f'LAT success ✓ — {k}' for k in lat_best]  +
    [f'LAT fail   ✗ — {k}' for k in lat_worst]
)
N_ROWS = len(ROW_FRAMES)   # 8

print(f'Selected frames: {ROW_FRAMES}')

# ── Load specimen ─────────────────────────────────────────────────────────────
print('Loading specimen ...')
loader  = SwaroLoader()
spec    = loader.load(frames=ROW_FRAMES, verbose=True)

print('Building diffdrr subject ...')
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
subject = build_subject(spec)
drr_gen = DiffDRRGenerator(subject, device, ct_origin_lps=spec.ct_origin)

lm_names = sorted(spec.landmarks_3d.keys())
pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
scale    = RENDER_SIZE / SWARO_IMG_SIZE
pix      = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)


def render_pose(proj, delta_6):
    R, t = perturb_extrinsic(proj.R_proj, proj.t_proj,
                              np.array(delta_6[:3]), np.array(delta_6[3:]))
    drr = drr_gen.generate_from_extrinsic(R, t, RENDER_SIZE, pix)
    # Use diffdrr's native perspective_projection for pixel-accurate landmark overlay
    uv  = drr_gen.project_pts(R, t, pts3d, RENDER_SIZE, pix)
    return drr, uv, R, t


def overlay_lm(ax, uv, marker='o', ms=6):
    for j, name in enumerate(lm_names):
        u, v = uv[j]
        if not (0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE):
            continue
        ax.plot(u, v, marker, color=LM_COLOURS.get(name, 'white'),
                markersize=ms, markeredgewidth=0.8, markeredgecolor='white', zorder=5)
        ax.text(u + 4, v - 4, name, color=LM_COLOURS.get(name, 'white'),
                fontsize=6, fontweight='bold', zorder=6)


# ── Build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(N_ROWS, 3,
                         figsize=(14, N_ROWS * 3.0),
                         gridspec_kw={'hspace': 0.05, 'wspace': 0.02},
                         squeeze=False)
fig.patch.set_facecolor(BG)

col_titles = ['X-ray  +  GT landmarks',
              'Initial pose DRR  +  projected lm',
              'Final pose DRR  +  projected lm']
for ci, ct in enumerate(col_titles):
    axes[0, ci].set_title(ct, fontsize=10, pad=8, color=FG)

for row_i, (fk, row_lbl) in enumerate(zip(ROW_FRAMES, ROW_LABELS)):
    proj = next(p for p in spec.projections if p.proj_key == fk)
    res  = pp[fk]
    delta = res['best_pose_delta']

    print(f'  Rendering {fk} ...')
    drr_init, uv_init, _, _ = render_pose(proj, [0, 0, 0, 0, 0, 0])
    drr_final, uv_final, _, _ = render_pose(proj, delta)

    xray_s = cv2.resize(proj.image_raw, (RENDER_SIZE, RENDER_SIZE),
                        interpolation=cv2.INTER_AREA)

    # GT 2D landmarks (in 1024px space) scaled down
    # GT overlay on x-ray: project from 3D using diffdrr, same as DRR columns
    gt_uv   = drr_gen.project_pts(proj.R_proj, proj.t_proj, pts3d, RENDER_SIZE, pix)
    gt_names = lm_names

    init_pde  = float(res['initial_pde_mm'])
    final_pde = float(res['final_pde_mm'])
    init_go   = float(res['initial_go'])
    final_go  = float(res['final_go'])
    ok_str    = '✓' if res['success'] == 'True' else '✗'
    ok_col    = '#44dd88' if res['success'] == 'True' else '#ee4444'

    imgs_data = [
        (xray_s,    'X-ray  +  diffdrr projected lm', gt_uv,   lm_names, 'D', 7),
        (drr_init,  f'Initial  PDE={init_pde:.0f}mm  GO={init_go:.3f}',
                    uv_init,  lm_names, 'o', 5),
        (drr_final, f'Final {ok_str}  PDE={final_pde:.0f}mm  GO={final_go:.3f}',
                    uv_final, lm_names, 'o', 5),
    ]

    for ci, (img, subtitle, uv, names, mk, ms) in enumerate(imgs_data):
        ax = axes[row_i, ci]
        ax.set_facecolor(BG)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])

        # Draw landmarks
        for j, name in enumerate(names):
            u, v = uv[j]
            if 0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE:
                ax.plot(u, v, mk, color=LM_COLOURS.get(name, 'white'),
                        markersize=ms, markeredgewidth=0.8,
                        markeredgecolor='white', zorder=5)
                ax.text(u + 4, v - 4, name,
                        color=LM_COLOURS.get(name, 'white'),
                        fontsize=5.5, fontweight='bold', zorder=6)

        # Subtitle
        sc = ok_col if ci == 2 else FG
        ax.text(0.5, -0.01, subtitle, transform=ax.transAxes,
                fontsize=7, ha='center', va='top', color=sc)

        # Row label on left column
        if ci == 0:
            ax.set_ylabel(row_lbl, fontsize=7.5, rotation=90,
                          labelpad=4, color=FG)

plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'\nSaved → {OUT}')
