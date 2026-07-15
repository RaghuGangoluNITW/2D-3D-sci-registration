#!/usr/bin/env python3
"""
smoke_test_no_perturb.py
========================
Runs registration on 4 frames starting directly from the EPnP pose (no perturbation).
Side-by-side with the perturbed run to show the difference.
Saves: results/figures/swaroopa_no_perturb.png
"""
import sys, time
from pathlib import Path
import numpy as np
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from swaroopa_loader import SwaroLoader, project_world_swaro, SWARO_PIX_MM, SWARO_IMG_SIZE
from deepfluoro_loader import perturb_extrinsic
from deepfluoro_drr import DeepFluoroDRR
from run_swaroopa import run_registration

FRAMES      = ['ap_013', 'ap_031', 'lat_003', 'lat_026']
RENDER_SIZE = 256
OUT         = ROOT / 'results/figures/swaroopa_no_perturb.png'
LM_COLOURS  = {'L1':'#ff4444','L2':'#ff9900','L3':'#ffee00','L4':'#44ff44','L5':'#44ddff'}

print("Loading specimen ...")
loader  = SwaroLoader()
spec    = loader.load(frames=FRAMES, verbose=True)
drr_gen = DeepFluoroDRR(spec, hu_threshold=0.0)

lm_names = sorted(spec.landmarks_3d.keys())
pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
pix_r    = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)
scale_lm = RENDER_SIZE / SWARO_IMG_SIZE

# ── Run no-perturb registration ───────────────────────────────────────────────
print("\nRegistering (no perturbation — start from EPnP) ...")
results = {}
for proj in spec.projections:
    key = proj.proj_key
    print(f"\n  [{key}]")
    t0  = time.time()
    res = run_registration(drr_gen, proj, spec, verbose=True, no_perturb=True,
                           perturb_rot_deg=5.0, perturb_trans_mm=10.0,
                           n_starts=5, fast=True)
    res.runtime_s = time.time() - t0
    results[key] = res
    print(f"    done in {res.runtime_s:.0f}s")

# ── Helpers ───────────────────────────────────────────────────────────────────
def render(R, t):
    return drr_gen.generate_from_extrinsic(R, t, RENDER_SIZE, pix_r, 120)

def proj_lm(R, t):
    return project_world_swaro(pts3d, R, t) * scale_lm

def draw_lm(ax, uv):
    for j, n in enumerate(lm_names):
        u, v = uv[j]
        if 0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE:
            ax.plot(u, v, 'o', color=LM_COLOURS[n], ms=5, mew=1.2, mec='white', zorder=6)

def draw_gt(ax, proj):
    if hasattr(proj, 'gt_landmarks_2d') and proj.gt_landmarks_2d:
        for n, (u, v) in proj.gt_landmarks_2d.items():
            ax.plot(u * scale_lm, v * scale_lm, '*',
                    color=LM_COLOURS.get(n, 'w'), ms=8, mew=1, mec='white', zorder=7)

# ── Figure: 4 rows × 4 cols ───────────────────────────────────────────────────
# Cols: [X-ray (inv) | EPnP DRR (init=final start) | Final DRR | Overlay]
N   = len(FRAMES)
fig, axes = plt.subplots(N, 4, figsize=(15, N * 3.4))
fig.patch.set_facecolor('#111111')

col_titles = [
    'X-ray (inverted)',
    'EPnP DRR\n(start = no perturbation)',
    'Final DRR\n(after CMA-ES from EPnP)',
    'Overlay  DRR(R) · X-ray(G)',
]
for ci, ct in enumerate(col_titles):
    axes[0, ci].set_title(ct, color='white', fontsize=8.5, pad=5)

for ri, proj in enumerate(spec.projections):
    key = proj.proj_key
    res = results[key]

    xray_inv = cv2.resize(1.0 - proj.image_raw, (RENDER_SIZE, RENDER_SIZE),
                          interpolation=cv2.INTER_AREA)

    drr_epnp  = render(proj.R_proj, proj.t_proj)
    uv_epnp   = proj_lm(proj.R_proj, proj.t_proj)

    R_f, t_f  = perturb_extrinsic(proj.R_proj, proj.t_proj,
                                   res.best_pose_delta[:3], res.best_pose_delta[3:])
    drr_final = render(R_f, t_f)
    uv_final  = proj_lm(R_f, t_f)

    overlay = np.zeros((RENDER_SIZE, RENDER_SIZE, 3), dtype=np.float32)
    overlay[..., 0] = drr_final
    overlay[..., 1] = xray_inv
    overlay = np.clip(overlay, 0, 1)

    dgo  = res.initial_go - res.final_go
    succ = res.success

    for ci, (img, cmap, uv, is_xray) in enumerate([
        (xray_inv,  'gray', None,      True),
        (drr_epnp,  'gray', uv_epnp,   False),
        (drr_final, 'gray', uv_final,  False),
        (overlay,   None,   uv_final,  False),
    ]):
        ax = axes[ri, ci]
        ax.set_facecolor('#0a0a0a')
        kw = dict(vmin=0, vmax=1, interpolation='bilinear')
        ax.imshow(img, cmap=cmap, **kw) if cmap else ax.imshow(img, **kw)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#44ff44' if succ else '#ff4444')
            sp.set_linewidth(2 if ci == 3 else 0.5)

        if ci == 0:
            ax.set_ylabel(key, color='white', fontsize=9)
            draw_gt(ax, proj)
        elif uv is not None:
            draw_lm(ax, uv)

        if ci == 1:
            ax.set_xlabel(f"EPnP reproj={proj.reproj_error_px:.2f}px  GO={res.initial_go:.3f}",
                          color='#aaaaaa', fontsize=7)
        if ci == 3:
            col  = '#44ff44' if succ else '#ff4444'
            tick = '✓ SUCCESS' if succ else '✗ FAIL'
            ax.set_xlabel(
                f"{tick}   GO {res.initial_go:.3f}→{res.final_go:.3f}  ΔGO={dgo:+.3f}\n"
                f"PDE {res.initial_pde_mm:.1f}→{res.final_pde_mm:.1f} mm  ({res.runtime_s:.0f}s)",
                color=col, fontsize=7)

# Legend
lm_h = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
                markersize=6, label=n, mec='white') for n, c in LM_COLOURS.items()]
gt_h  = Line2D([0],[0], marker='*', color='w', markerfacecolor='white',
               markersize=8, label='GT annot (★)', mec='grey')
fig.legend(handles=lm_h + [gt_h], loc='lower center', ncol=6,
           facecolor='#1e1e1e', edgecolor='#555', labelcolor='white',
           fontsize=8, bbox_to_anchor=(0.5, 0.0))

n_succ = sum(1 for r in results.values() if r.success)
plt.suptitle(
    f'No-Perturbation Smoke Test — CMA-ES starts at EPnP pose (delta=0)\n'
    f'Search ±15°/±30 mm centred on EPnP    {n_succ}/{N} frames succeeded',
    color='white', fontsize=11, y=1.01)
plt.tight_layout(rect=[0, 0.05, 1, 1.0])
plt.subplots_adjust(hspace=0.08, wspace=0.04)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved → {OUT}")
