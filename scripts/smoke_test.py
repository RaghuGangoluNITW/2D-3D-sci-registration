#!/usr/bin/env python3
"""
smoke_test.py
=============
Runs run_swaroopa.run_registration() on 2 AP + 2 lateral frames using
the updated objective (xray inverted at preprocessing, standard NCC/GO)
and saves a visualisation to results/figures/swaroopa_smoke_test.png.
"""

import sys, time
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from swaroopa_loader import (
    SwaroLoader, project_world_swaro,
    SWARO_PIX_MM, SWARO_IMG_SIZE,
)
from deepfluoro_loader import perturb_extrinsic
from deepfluoro_drr import DeepFluoroDRR

# Import the updated pipeline from run_swaroopa
sys.path.insert(0, str(ROOT))
from run_swaroopa import run_registration

TEST_FRAMES  = ['ap_013', 'ap_031', 'lat_003', 'lat_026']
RENDER_SIZE  = 256
OUT          = ROOT / 'results/figures/swaroopa_smoke_test.png'
LM_COLOURS   = {'L1':'#ff4444','L2':'#ff9900','L3':'#ffee00','L4':'#44ff44','L5':'#44ddff'}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading specimen ...")
loader  = SwaroLoader()
spec    = loader.load(frames=TEST_FRAMES, verbose=True)
drr_gen = DeepFluoroDRR(spec, hu_threshold=150.0)

lm_names = sorted(spec.landmarks_3d.keys())
pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
scale_lm = RENDER_SIZE / SWARO_IMG_SIZE

# ── Register ──────────────────────────────────────────────────────────────────
print(f"\nRegistering {len(spec.projections)} frames ...")
results = {}
for proj in spec.projections:
    key = proj.proj_key
    print(f"\n  [{key}]")
    t0 = time.time()
    res = run_registration(drr_gen, proj, spec, verbose=True)
    res.runtime_s = time.time() - t0
    results[key] = res
    print(f"    done in {res.runtime_s:.0f}s")

# ── Helpers ───────────────────────────────────────────────────────────────────
pix_r = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)

def render(R, t):
    return drr_gen.generate_from_extrinsic(R, t, RENDER_SIZE, pix_r, 120)

def proj_lm(R, t):
    return project_world_swaro(pts3d, R, t) * scale_lm

def draw_lm(ax, uv, color_map=None, marker='o', ms=5):
    for j, n in enumerate(lm_names):
        u, v = uv[j]
        if 0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE:
            c = LM_COLOURS[n] if color_map is None else color_map[n]
            ax.plot(u, v, marker, color=c, ms=ms, mew=1.2, mec='white', zorder=6)

def draw_gt_lm(ax, proj):
    """Draw GT 2D landmark annotations (from JSON)."""
    if not hasattr(proj, 'gt_landmarks_2d') or proj.gt_landmarks_2d is None:
        return
    for n, (u, v) in proj.gt_landmarks_2d.items():
        u_s, v_s = u * scale_lm, v * scale_lm
        ax.plot(u_s, v_s, '*', color=LM_COLOURS.get(n,'white'),
                ms=7, mew=1, mec='white', zorder=7)

# ── Figure: 4 rows × 4 cols ───────────────────────────────────────────────────
# Cols: [X-ray (inverted display) | init DRR | final DRR | overlay]
N = len(TEST_FRAMES)
fig, axes = plt.subplots(N, 4, figsize=(15, N * 3.4))
fig.patch.set_facecolor('#111111')

col_titles = [
    'X-ray\n(inverted for registration)',
    'Initial DRR\n(EPnP + perturbation)',
    'Final DRR\n(after CMA-ES)',
    'Overlay\nDRR(R)  X-ray(G)',
]
for ci, ct in enumerate(col_titles):
    axes[0, ci].set_title(ct, color='white', fontsize=8.5, pad=5)

for ri, proj in enumerate(spec.projections):
    key = proj.proj_key
    res = results[key]

    # Raw xray inverted (as used by the objective)
    xray_inv = 1.0 - proj.image_raw
    xray_s   = cv2.resize(xray_inv, (RENDER_SIZE, RENDER_SIZE),
                          interpolation=cv2.INTER_AREA)

    # Initial pose: re-derive with the same seed used inside run_registration
    rng_init = np.random.default_rng(42 + proj.proj_index)
    dr = rng_init.uniform(-15., 15., 3)
    dt = rng_init.uniform(-30., 30., 3)
    R_init, t_init = perturb_extrinsic(proj.R_proj, proj.t_proj, dr, dt)

    # Final pose
    R_f, t_f = perturb_extrinsic(proj.R_proj, proj.t_proj,
                        res.best_pose_delta[:3], res.best_pose_delta[3:])

    drr_init  = render(R_init, t_init)
    drr_final = render(R_f, t_f)
    uv_init   = proj_lm(R_init, t_init)
    uv_final  = proj_lm(R_f,   t_f)

    # Overlay: DRR → red channel, xray_inv → green channel
    overlay = np.zeros((RENDER_SIZE, RENDER_SIZE, 3), dtype=np.float32)
    overlay[..., 0] = drr_final          # R = DRR
    overlay[..., 1] = xray_s             # G = xray_inv
    overlay = np.clip(overlay, 0, 1)

    imgs_meta = [
        (xray_s,   'gray', None,     True),   # x-ray
        (drr_init, 'gray', uv_init,  False),  # init DRR
        (drr_final,'gray', uv_final, False),  # final DRR
        (overlay,  None,   uv_final, False),  # overlay
    ]

    dgo = res.initial_go - res.final_go
    succ = res.success

    for ci, (img, cmap, uv, show_gt) in enumerate(imgs_meta):
        ax = axes[ri, ci]
        ax.set_facecolor('#0a0a0a')
        if cmap:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')
        else:
            ax.imshow(img, vmin=0, vmax=1, interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#44ff44' if succ else '#ff4444')
            sp.set_linewidth(2 if ci == 3 else 0.5)

        if ci == 0:
            ax.set_ylabel(key, color='white', fontsize=9, labelpad=4)
            draw_gt_lm(ax, proj)
        elif uv is not None:
            draw_lm(ax, uv)

        # Annotate last column with scores
        if ci == 3:
            col = '#44ff44' if succ else '#ff4444'
            tick = '✓ SUCCESS' if succ else '✗ FAIL'
            label = (f"{tick}\n"
                     f"GO {res.initial_go:.3f} → {res.final_go:.3f}  "
                     f"(ΔGO={dgo:+.3f})\n"
                     f"PDE {res.initial_pde_mm:.1f} → {res.final_pde_mm:.1f} mm")
            ax.set_xlabel(label, color=col, fontsize=7, labelpad=3)

# ── Legend ────────────────────────────────────────────────────────────────────
lm_h = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
               markersize=6, label=n, mec='white')
        for n, c in LM_COLOURS.items()]
gt_h = Line2D([0],[0], marker='*', color='w', markerfacecolor='white',
              markersize=8, label='GT (★)', mec='grey')
fig.legend(handles=lm_h + [gt_h],
           loc='lower center', ncol=6,
           facecolor='#1e1e1e', edgecolor='#555',
           labelcolor='white', fontsize=8,
           bbox_to_anchor=(0.5, 0.0))

n_succ = sum(1 for r in results.values() if r.success)
plt.suptitle(
    f'Swaroopa Smoke Test — inverted xray preprocessing\n'
    f'NCC(drr, 1−xray) + GO(drr, 1−xray)    '
    f'{n_succ}/{N} frames succeeded',
    color='white', fontsize=11, y=1.01
)
plt.tight_layout(rect=[0, 0.05, 1, 1.0])
plt.subplots_adjust(hspace=0.08, wspace=0.04)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved → {OUT}")
