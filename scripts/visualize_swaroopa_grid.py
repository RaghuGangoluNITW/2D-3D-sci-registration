#!/usr/bin/env python3
"""
visualize_swaroopa_grid.py
==========================
12-frame grid: 3 AP success, 3 AP fail, 3 lat success, 3 lat fail.
Each row: [X-ray + GT lm] | [Initial DRR + proj lm] | [Final DRR + proj lm]
"""

import sys, json, cv2
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from swaroopa_loader import (
    SwaroLoader, project_world_swaro,
    SWARO_PIX_MM, SWARO_IMG_SIZE,
)
from deepfluoro_loader import perturb_extrinsic
from deepfluoro_drr import DeepFluoroDRR

# ── Frame selection ───────────────────────────────────────────────────────────
FRAMES_AP_SUCC  = ['ap_013', 'ap_029', 'ap_031']
FRAMES_AP_FAIL  = ['ap_002', 'ap_009', 'ap_015']
FRAMES_LAT_SUCC = ['lat_003', 'lat_026', 'lat_027']
FRAMES_LAT_FAIL = ['lat_000', 'lat_022', 'lat_035']

# Ordered rows: AP success, AP fail, Lat success, Lat fail
ROW_FRAMES = FRAMES_AP_SUCC + FRAMES_AP_FAIL + FRAMES_LAT_SUCC + FRAMES_LAT_FAIL
N_ROWS = len(ROW_FRAMES)   # 12

RESULTS     = Path('results/swaroopa_results_new_ct.json')
OUT         = Path('results/figures/swaroopa_registration_grid.png')
RENDER_SIZE = 192          # smaller for grid readability
LM_COLOURS  = {'L1': '#ff4444', 'L2': '#ff9900', 'L3': '#ffee00',
               'L4': '#44ff44', 'L5': '#44ddff'}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading specimen ...")
loader  = SwaroLoader()
spec    = loader.load(frames=ROW_FRAMES, verbose=True)
drr_gen = DeepFluoroDRR(spec, hu_threshold=150.0)

with open(RESULTS) as f:
    pp = json.load(f)['swaroopa']['per_projection']

lm_names = sorted(spec.landmarks_3d.keys())
pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
scale    = RENDER_SIZE / SWARO_IMG_SIZE
pix      = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)


def render_at(proj, delta_6):
    R, t = perturb_extrinsic(proj.R_proj, proj.t_proj,
                              np.array(delta_6[:3]), np.array(delta_6[3:]))
    drr = drr_gen.generate_from_extrinsic(R, t, RENDER_SIZE, pix, 120)
    uv  = project_world_swaro(pts3d, R, t) * scale
    return drr, uv


def overlay_lm(ax, uv, visible_only=True):
    for j, name in enumerate(lm_names):
        u, v = uv[j]
        if visible_only and not (0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE):
            continue
        ax.plot(u, v, 'o', color=LM_COLOURS[name], markersize=5,
                markeredgewidth=1, markeredgecolor='white', zorder=5)
        ax.text(u + 4, v - 4, name, color=LM_COLOURS[name],
                fontsize=6, fontweight='bold', zorder=6)


# ── Build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(N_ROWS, 3, figsize=(11, N_ROWS * 2.6))
fig.patch.set_facecolor('#111111')

# Section divider y-positions (in axes-fraction coords per row)
section_labels = {
    0:  ('AP',  'SUCCESS', '#22cc44'),
    3:  ('AP',  'FAIL',    '#cc2222'),
    6:  ('LAT', 'SUCCESS', '#22cc44'),
    9:  ('LAT', 'FAIL',    '#cc2222'),
}

col_titles = ['X-ray  (GT landmarks)', 'Initial DRR', 'Final DRR']
for ci, ct in enumerate(col_titles):
    axes[0, ci].set_title(ct, color='white', fontsize=9, pad=4)

for row_i, proj_key in enumerate(ROW_FRAMES):
    proj = next(p for p in spec.projections if p.proj_key == proj_key)
    r    = pp[proj_key]
    delta = r['best_pose_delta']
    success = r['success']

    print(f"  Rendering {proj_key}  ΔGO={r['go_delta']:+.3f}  {'✓' if success else '✗'}")

    drr_init, uv_init   = render_at(proj, [0]*6)
    drr_final, uv_final = render_at(proj, delta)
    xray_small = cv2.resize(proj.image_raw, (RENDER_SIZE, RENDER_SIZE),
                            interpolation=cv2.INTER_AREA)

    imgs_uvs = [
        (xray_small, None),
        (drr_init,   uv_init),
        (drr_final,  uv_final),
    ]

    # Section label on first row of each block (col 0)
    if row_i in section_labels:
        view, outcome, colour = section_labels[row_i]
        axes[row_i, 0].set_ylabel(
            f'{view}\n{outcome}',
            color=colour, fontsize=8, fontweight='bold', rotation=0,
            labelpad=38, va='center'
        )

    for col_i, (img, uv) in enumerate(imgs_uvs):
        ax = axes[row_i, col_i]
        ax.set_facecolor('#111111')
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#333')

        # Overlay landmarks
        if col_i == 0:
            for j, name in enumerate(lm_names):
                if name in proj.gt_landmarks_2d:
                    u, v = np.array(proj.gt_landmarks_2d[name]) * scale
                    ax.plot(u, v, 'o', color=LM_COLOURS[name], markersize=5,
                            markeredgewidth=1, markeredgecolor='white', zorder=5)
                    ax.text(u + 4, v - 4, name, color=LM_COLOURS[name],
                            fontsize=6, fontweight='bold', zorder=6)
        else:
            overlay_lm(ax, uv)

        # Row annotation on right side of col 2
        if col_i == 2:
            dgo   = r['go_delta']
            fg    = r['final_go']
            tick  = '✓' if success else '✗'
            c     = '#44ff44' if success else '#ff4444'
            ax.text(1.01, 0.5,
                    f"{proj_key}\nGO {r['initial_go']:.3f}→{fg:.3f}\nΔGO {dgo:+.3f}  {tick}",
                    transform=ax.transAxes, color=c, fontsize=6.5,
                    va='center', ha='left', linespacing=1.6)

    # Horizontal separator between sections
    if row_i in (2, 5, 8):
        for ci in range(3):
            axes[row_i, ci].spines['bottom'].set_edgecolor('#666')
            axes[row_i, ci].spines['bottom'].set_linewidth(1.5)

# ── Legend ────────────────────────────────────────────────────────────────────
lm_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                     markersize=7, label=n, markeredgecolor='white')
              for n, c in LM_COLOURS.items()]
fig.legend(handles=lm_handles, loc='lower center', ncol=5,
           facecolor='#1e1e1e', edgecolor='#555', labelcolor='white',
           fontsize=8, title='Vertebral centroids', title_fontsize=8,
           bbox_to_anchor=(0.5, 0.005))

plt.suptitle(
    'Swaroopa 2D/3D Registration Results  —  Initial vs Final Pose\n'
    'CT: 18520000 (309 slices, 0.412 mm)  |  Camera: Fx=3646 px, pix=0.288 mm, SID=1050 mm\n'
    'Success criterion: ΔGO > 0.05  and  final GO < 0.60',
    color='white', fontsize=10, y=0.995
)

plt.tight_layout(rect=[0.07, 0.03, 1.0, 0.975])
plt.subplots_adjust(hspace=0.06, wspace=0.04)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved: {OUT}")
