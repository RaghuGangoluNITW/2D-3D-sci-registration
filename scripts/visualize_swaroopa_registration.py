#!/usr/bin/env python3
"""
visualize_swaroopa_registration.py
===================================
Shows initial pose DRR vs final pose DRR vs real X-ray for one AP and one
lateral Swaroopa frame side-by-side, with vertebral centroids overlaid.

Layout (2 rows × 3 cols):
  Row 0 — AP  frame:  [X-ray + GT lm] | [Initial DRR + proj lm] | [Final DRR + proj lm]
  Row 1 — Lat frame:  [X-ray + GT lm] | [Initial DRR + proj lm] | [Final DRR + proj lm]
"""

import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from swaroopa_loader import (
    SwaroLoader, project_world_swaro,
    SWARO_PIX_MM, SWARO_IMG_SIZE,
)
from deepfluoro_loader import perturb_extrinsic
from deepfluoro_drr import DeepFluoroDRR

# ── Config ────────────────────────────────────────────────────────────────────
AP_FRAME  = 'ap_013'
LAT_FRAME = 'lat_026'
RESULTS   = Path('results/swaroopa_results_new_ct.json')
OUT       = Path('results/figures/swaroopa_registration_comparison.png')

RENDER_SIZE = 256
LM_COLOURS  = {'L1': 'red', 'L2': 'orange', 'L3': 'yellow',
                'L4': 'lime', 'L5': 'cyan'}

# ── Load specimen ─────────────────────────────────────────────────────────────
print("Loading specimen ...")
loader = SwaroLoader()
spec   = loader.load(frames=[AP_FRAME, LAT_FRAME], verbose=True)
drr_gen = DeepFluoroDRR(spec, hu_threshold=150.0)

# ── Load per-frame results ────────────────────────────────────────────────────
with open(RESULTS) as f:
    res = json.load(f)
pp = res['swaroopa']['per_projection']

# ── Helper: render DRR at 256px for a given pose delta ───────────────────────
def render(proj, delta_x):
    pix = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)
    R, t = perturb_extrinsic(proj.R_proj, proj.t_proj,
                              np.array(delta_x[:3]), np.array(delta_x[3:]))
    return (R, t), drr_gen.generate_from_extrinsic(R, t, RENDER_SIZE, pix, 120)

# ── Helper: project landmarks onto DRR pixel coords ──────────────────────────
def proj_lm(pts3d, lm_names, R, t, render_size):
    uv_full = project_world_swaro(pts3d, R, t)   # in 1024px space
    scale   = render_size / SWARO_IMG_SIZE
    return uv_full * scale                        # scaled to render_size

# ── Build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.patch.set_facecolor('#1a1a1a')

col_titles = ['Real X-ray + GT landmarks',
              'Initial pose DRR + projected landmarks',
              'Final pose DRR + projected landmarks']
row_labels  = [f'AP  ({AP_FRAME})', f'Lateral  ({LAT_FRAME})']

lm_names = sorted(spec.landmarks_3d.keys())
pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])

for row_i, (proj_key, frame_label) in enumerate([(AP_FRAME,  row_labels[0]),
                                                   (LAT_FRAME, row_labels[1])]):
    proj = next(p for p in spec.projections if p.proj_key == proj_key)
    delta = pp[proj_key]['best_pose_delta']   # 6-DOF result from optimiser

    # ── Render initial and final DRRs ─────────────────────────────────────
    print(f"Rendering {proj_key} initial DRR ...")
    (R_init, t_init), drr_init = render(proj, [0, 0, 0, 0, 0, 0])

    print(f"Rendering {proj_key} final DRR ...")
    (R_final, t_final), drr_final = render(proj, delta)

    # Resize xray to match render size
    import cv2
    xray_small = cv2.resize(proj.image_raw, (RENDER_SIZE, RENDER_SIZE),
                            interpolation=cv2.INTER_AREA)

    # ── Landmark projections ───────────────────────────────────────────────
    uv_init  = proj_lm(pts3d, lm_names, R_init,  t_init,  RENDER_SIZE)
    uv_final = proj_lm(pts3d, lm_names, R_final, t_final, RENDER_SIZE)
    # GT 2D landmarks (already in 1024px space) → scale down
    scale = RENDER_SIZE / SWARO_IMG_SIZE

    # ── Retrieve metrics ───────────────────────────────────────────────────
    r = pp[proj_key]
    init_go  = r['initial_go']
    final_go = r['final_go']
    d_go     = r['go_delta']
    success  = r['success']

    # ── Plot ───────────────────────────────────────────────────────────────
    data_cols = [
        (xray_small,  'X-ray',        None,       None),
        (drr_init,    f'Initial  GO={init_go:.3f}',   uv_init,   None),
        (drr_final,   f'Final  GO={final_go:.3f}  ΔGO={d_go:+.3f}  {"✓" if success else "✗"}',
                                       uv_final,  None),
    ]

    for col_i, (img, subtitle, uv_proj, _) in enumerate(data_cols):
        ax = axes[row_i, col_i]
        ax.set_facecolor('#1a1a1a')
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')

        # GT landmarks on xray column; projected landmarks on DRR columns
        if col_i == 0:
            for j, name in enumerate(lm_names):
                if name in proj.gt_landmarks_2d:
                    u, v = proj.gt_landmarks_2d[name]
                    u_s, v_s = u * scale, v * scale
                    ax.plot(u_s, v_s, 'o', color=LM_COLOURS[name],
                            markersize=7, markeredgewidth=1.5,
                            markeredgecolor='white', zorder=5)
                    ax.annotate(name, (u_s, v_s),
                                xytext=(6, -6), textcoords='offset points',
                                fontsize=8, color=LM_COLOURS[name], fontweight='bold')
        else:
            for j, name in enumerate(lm_names):
                u, v = uv_proj[j]
                if 0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE:
                    ax.plot(u, v, 'o', color=LM_COLOURS[name],
                            markersize=7, markeredgewidth=1.5,
                            markeredgecolor='white', zorder=5)
                    ax.annotate(name, (u, v),
                                xytext=(6, -6), textcoords='offset points',
                                fontsize=8, color=LM_COLOURS[name], fontweight='bold')

        # Titles / labels
        if row_i == 0:
            ax.set_title(col_titles[col_i], color='white', fontsize=10, pad=6)
        ax.set_xlabel(subtitle,
                      color=('lime' if success and col_i == 2 else
                             'tomato' if not success and col_i == 2 else 'white'),
                      fontsize=9)
        ax.set_ylabel(frame_label if col_i == 0 else '',
                      color='white', fontsize=10)
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')

# ── Legend ────────────────────────────────────────────────────────────────────
legend_patches = [mpatches.Patch(color=c, label=n) for n, c in LM_COLOURS.items()]
fig.legend(handles=legend_patches, loc='lower center', ncol=5,
           facecolor='#2a2a2a', edgecolor='white', labelcolor='white',
           fontsize=10, title='Vertebral centroids', title_fontsize=10)

plt.suptitle('Swaroopa 2D/3D Registration — Initial vs Final Pose\n'
             f'CT: 18520000 (309 slices, 0.412mm)  |  Camera: Fx=3646px, pix=0.288mm, SID=1050mm',
             color='white', fontsize=12, y=0.99)

plt.tight_layout(rect=[0, 0.06, 1, 0.97])
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved: {OUT}")
