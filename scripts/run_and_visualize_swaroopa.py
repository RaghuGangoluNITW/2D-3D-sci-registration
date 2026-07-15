#!/usr/bin/env python3
"""
run_and_visualize_swaroopa.py
==============================
Runs the full 3-phase CMA-ES registration pipeline on ONE AP and ONE LAT
Swaroopa frame, then produces a side-by-side visualisation:

  [X-ray + GT landmarks] | [Initial DRR + projected LM] | [Final DRR + projected LM]
  (one row per view type)

Usage:
  python scripts/run_and_visualize_swaroopa.py
  python scripts/run_and_visualize_swaroopa.py --ap ap_006 --lat lat_003
"""

import argparse, sys, time
from pathlib import Path

import numpy as np
import cv2
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

# reuse the registration logic from run_swaroopa
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_swaroopa import run_registration, _make_objectives

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--ap',  default='ap_002',  help='AP frame key')
p.add_argument('--lat', default='lat_000', help='LAT frame key')
p.add_argument('--perturb_rot',   type=float, default=15.0)
p.add_argument('--perturb_trans', type=float, default=30.0)
p.add_argument('--n_starts',      type=int,   default=16)
p.add_argument('--out', default='results/figures/swaroopa_init_vs_final.png')
args = p.parse_args()

AP_KEY  = args.ap
LAT_KEY = args.lat
OUT     = Path(args.out)

RENDER_SIZE = 256
LM_COLOURS  = {'L1': 'red', 'L2': 'orange', 'L3': 'yellow',
                'L4': 'lime', 'L5': 'cyan'}

# ── Load specimen ──────────────────────────────────────────────────────────────
print(f"Loading specimen for frames: {AP_KEY}, {LAT_KEY} ...")
loader = SwaroLoader()
spec   = loader.load(frames=[AP_KEY, LAT_KEY], verbose=True)
drr_gen = DeepFluoroDRR(spec, hu_threshold=0.0)

lm_names = sorted(spec.landmarks_3d.keys())
pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])

# ── Run registration for both frames ─────────────────────────────────────────
results = {}
for proj_key in [AP_KEY, LAT_KEY]:
    proj = next(p for p in spec.projections if p.proj_key == proj_key)
    print(f"\n{'='*55}")
    print(f"Registering {proj_key} ...")
    t0 = time.time()
    res = run_registration(
        drr_gen, proj, spec,
        perturb_rot_deg=args.perturb_rot,
        perturb_trans_mm=args.perturb_trans,
        n_starts=args.n_starts,
        verbose=True,
    )
    res.runtime_s = time.time() - t0
    results[proj_key] = res
    go_delta = res.initial_go - res.final_go
    pde_str = (f"  PDE: {res.initial_pde_mm:.1f}→{res.final_pde_mm:.1f} mm"
               if not np.isnan(res.initial_pde_mm) else "")
    print(f"  → GO: {res.initial_go:.4f} → {res.final_go:.4f}  "
          f"ΔGO={go_delta:+.4f}{pde_str}  "
          f"{'[SUCCESS]' if res.success else '[FAIL]'}  ({res.runtime_s:.1f}s)")

# ── Helpers ───────────────────────────────────────────────────────────────────
def render_drr(proj, delta_x):
    pix = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)
    R, t = perturb_extrinsic(proj.R_proj, proj.t_proj,
                              np.array(delta_x[:3]), np.array(delta_x[3:]))
    drr = drr_gen.generate_from_extrinsic(R, t, RENDER_SIZE, pix, 120)
    return (R, t), drr

def proj_lm_scaled(R, t, render_size):
    uv_full = project_world_swaro(pts3d, R, t)
    scale   = render_size / SWARO_IMG_SIZE
    return uv_full * scale

# ── Build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.patch.set_facecolor('#1a1a1a')

col_titles = ['Real X-ray  (GT landmarks)',
              'Initial pose DRR  (projected landmarks)',
              'Final pose DRR  (projected landmarks)']
row_labels  = [f'AP  ({AP_KEY})', f'Lateral  ({LAT_KEY})']

for row_i, (proj_key, row_label) in enumerate([(AP_KEY, row_labels[0]),
                                                (LAT_KEY, row_labels[1])]):
    proj = next(p for p in spec.projections if p.proj_key == proj_key)
    res  = results[proj_key]
    delta = res.best_pose_delta

    print(f"\nRendering DRRs for {proj_key} ...")
    (R_init,  t_init),  drr_init  = render_drr(proj, np.zeros(6))
    (R_final, t_final), drr_final = render_drr(proj, delta)

    xray_small = cv2.resize(proj.image_raw, (RENDER_SIZE, RENDER_SIZE),
                            interpolation=cv2.INTER_AREA)

    uv_init  = proj_lm_scaled(R_init,  t_init,  RENDER_SIZE)
    uv_final = proj_lm_scaled(R_final, t_final, RENDER_SIZE)

    scale_gt = RENDER_SIZE / SWARO_IMG_SIZE
    init_go  = res.initial_go
    final_go = res.final_go
    d_go     = init_go - final_go
    success  = res.success
    pde_init = f"{res.initial_pde_mm:.1f}mm" if not np.isnan(res.initial_pde_mm) else "n/a"
    pde_final= f"{res.final_pde_mm:.1f}mm"   if not np.isnan(res.final_pde_mm)   else "n/a"

    panels = [
        (xray_small,  'X-ray',                                          None,      'gt'),
        (drr_init,    f'Initial   GO={init_go:.3f}   PDE={pde_init}',   uv_init,   'init'),
        (drr_final,   f'Final   GO={final_go:.3f}  ΔGO={d_go:+.3f}  PDE={pde_final}  '
                      f'{"✓ SUCCESS" if success else "✗ FAIL"}',        uv_final,  'final'),
    ]

    for col_i, (img, subtitle, uv_proj, mode) in enumerate(panels):
        ax = axes[row_i, col_i]
        ax.set_facecolor('#1a1a1a')
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')

        if mode == 'gt':
            # GT 2D landmarks annotated on the real X-ray
            for j, name in enumerate(lm_names):
                if name in proj.gt_landmarks_2d:
                    u, v = proj.gt_landmarks_2d[name]
                    u_s, v_s = u * scale_gt, v * scale_gt
                    ax.plot(u_s, v_s, 'o', color=LM_COLOURS.get(name, 'white'),
                            markersize=8, markeredgewidth=1.5,
                            markeredgecolor='white', zorder=5)
                    ax.annotate(name, (u_s, v_s), xytext=(6, -6),
                                textcoords='offset points', fontsize=8,
                                color=LM_COLOURS.get(name, 'white'), fontweight='bold')
        else:
            # Projected 3D landmarks onto DRR
            for j, name in enumerate(lm_names):
                u, v = uv_proj[j]
                if 0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE:
                    ax.plot(u, v, 'o', color=LM_COLOURS.get(name, 'white'),
                            markersize=8, markeredgewidth=1.5,
                            markeredgecolor='white', zorder=5)
                    ax.annotate(name, (u, v), xytext=(6, -6),
                                textcoords='offset points', fontsize=8,
                                color=LM_COLOURS.get(name, 'white'), fontweight='bold')

        if row_i == 0:
            ax.set_title(col_titles[col_i], color='white', fontsize=10, pad=6)

        label_color = ('lime' if success and col_i == 2 else
                       'tomato' if not success and col_i == 2 else 'white')
        ax.set_xlabel(subtitle, color=label_color, fontsize=9)
        ax.set_ylabel(row_label if col_i == 0 else '', color='white', fontsize=10)
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_patches = [mpatches.Patch(color=c, label=n) for n, c in LM_COLOURS.items()]
fig.legend(handles=legend_patches, loc='lower center', ncol=5,
           facecolor='#2a2a2a', edgecolor='white', labelcolor='white',
           fontsize=10, title='Vertebral centroids', title_fontsize=10)

ap_res  = results[AP_KEY]
lat_res = results[LAT_KEY]
plt.suptitle(
    f'Swaroopa 2D/3D Registration — Initial vs Final Pose\n'
    f'AP  GO: {ap_res.initial_go:.3f}→{ap_res.final_go:.3f}  '
    f'PDE: {ap_res.initial_pde_mm:.1f}→{ap_res.final_pde_mm:.1f}mm  '
    f'{"✓" if ap_res.success else "✗"}    |    '
    f'LAT  GO: {lat_res.initial_go:.3f}→{lat_res.final_go:.3f}  '
    f'PDE: {lat_res.initial_pde_mm:.1f}→{lat_res.final_pde_mm:.1f}mm  '
    f'{"✓" if lat_res.success else "✗"}',
    color='white', fontsize=11, y=0.99)

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved: {OUT}")
