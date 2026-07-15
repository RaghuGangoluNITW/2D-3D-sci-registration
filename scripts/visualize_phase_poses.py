#!/usr/bin/env python3
"""
visualize_phase_poses.py
========================
For every frame that has a phase_log in the results JSON, render a DRR at
each optimisation phase (EPnP GT → perturbed → grid → p1_coarse → p2_opt).

One PNG per frame is written to results/figures/swaroopa_phase_poses_<frame>.png
A combined multi-row figure is written to results/figures/swaroopa_phase_poses_all.png
"""

import sys, json, cv2
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from swaroopa_loader import SwaroLoader, SWARO_PIX_MM, SWARO_IMG_SIZE
from run_swaroopa_diffdrr import build_subject_masked, DiffDRRGenerator, perturb_extrinsic
import torch

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS_JSON = Path('results/swaroopa_diffdrr_results.json')
RENDER_SIZE  = 256
CYLINDER_R   = 40       # match --cylinder_r used during registration
OUT_DIR      = Path('results/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

LM_COLOURS = {'L1': '#ff4444', 'L2': '#ff9900', 'L3': '#ffee00',
              'L4': '#44ff44', 'L5': '#44ddff'}
BG, FG = '#111111', '#dddddd'
matplotlib.rcParams.update({'text.color': FG, 'axes.labelcolor': FG,
                             'figure.facecolor': BG, 'axes.facecolor': BG})

# ── Load results JSON ──────────────────────────────────────────────────────────
with open(RESULTS_JSON) as f:
    data = json.load(f)

per_proj = data['swaroopa']['per_projection']

# All frames that have a non-empty phase_log
frame_keys = [k for k, v in per_proj.items() if v.get('phase_log')]
if not frame_keys:
    raise RuntimeError('No frames with phase_log found in results JSON. '
                       'Re-run registration with the updated code.')

print(f'Frames with phase_log: {frame_keys}')

# ── Load all frames at once ────────────────────────────────────────────────────
print('Loading specimen ...')
loader = SwaroLoader()
spec   = loader.load(frames=frame_keys, verbose=False)

lm_names = sorted(spec.landmarks_3d.keys())
pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])

# ── Build DRR generator (shared across frames) ────────────────────────────────
print('Building DiffDRR subject ...')
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
subject = build_subject_masked(spec, cylinder_r_mm=CYLINDER_R)
drr_gen = DiffDRRGenerator(subject, device, ct_origin_lps=spec.ct_origin)
pix     = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)

# ── Helpers ────────────────────────────────────────────────────────────────────
def render_delta(proj, delta):
    R_c, t_c = perturb_extrinsic(proj.R_proj, proj.t_proj,
                                  np.array(delta[:3]), np.array(delta[3:]))
    drr = drr_gen.generate_from_extrinsic(R_c, t_c, RENDER_SIZE, pix)
    uv  = drr_gen.project_pts(R_c, t_c, pts3d, RENDER_SIZE, pix)
    return drr, uv

def draw_row(axes_row, proj, res, frame_key):
    phase_log = res['phase_log']
    panels = [{'phase': 'EPnP (GT)', 'cost': res['initial_go'],
                'pde_mm': res['initial_pde_mm'], 'delta': [0.0]*6}] + phase_log

    xray = 1.0 - cv2.resize(proj.image_raw, (RENDER_SIZE, RENDER_SIZE),
                             interpolation=cv2.INTER_AREA)
    ok_str = '✓' if str(res.get('success', '')).lower() in ('true', '1') else '✗'

    # Col 0: X-ray
    ax = axes_row[0]
    ax.set_facecolor(BG)
    ax.imshow(xray, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'X-ray\n{frame_key}  {ok_str}', fontsize=8, color=FG, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for j, name in enumerate(lm_names):
        if name in proj.gt_landmarks_2d:
            gt = proj.gt_landmarks_2d[name]
            u = gt[0] * RENDER_SIZE / SWARO_IMG_SIZE
            v = gt[1] * RENDER_SIZE / SWARO_IMG_SIZE
            ax.plot(u, v, 'D', color=LM_COLOURS.get(name, 'white'),
                    markersize=5, markeredgewidth=0.5, markeredgecolor='white', zorder=5)
            ax.text(u+3, v-3, name, color=LM_COLOURS.get(name, 'white'),
                    fontsize=5, fontweight='bold', zorder=6)

    # Phase columns
    for ci, entry in enumerate(panels):
        ax = axes_row[ci + 1]
        ax.set_facecolor(BG)
        drr, uv = render_delta(proj, entry['delta'])
        ax.imshow(drr, cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])

        cost_str = f'cost={float(entry["cost"]):.3f}'
        pde_val  = entry['pde_mm']
        pde_str  = (f'PDE={float(pde_val):.1f}mm'
                    if pde_val is not None and str(pde_val).lower() not in ('none', 'nan')
                    else 'PDE=N/A')
        ax.set_title(f'{entry["phase"]}\n{cost_str}  {pde_str}', fontsize=8, color=FG, pad=4)

        for j, name in enumerate(lm_names):
            u, v = uv[j]
            if 0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE:
                ax.plot(u, v, 'o', color=LM_COLOURS.get(name, 'white'),
                        markersize=5, markeredgewidth=0.5, markeredgecolor='white', zorder=5)
                ax.text(u+3, v-3, name, color=LM_COLOURS.get(name, 'white'),
                        fontsize=5, fontweight='bold', zorder=6)

# ── Figure: one row per frame ─────────────────────────────────────────────────
# Determine max number of phase panels across all frames
max_phases = max(len(per_proj[k]['phase_log']) for k in frame_keys)
N_COLS = 1 + 1 + max_phases   # X-ray + EPnP GT + phase_log entries
N_ROWS = len(frame_keys)

fig, axes = plt.subplots(N_ROWS, N_COLS,
                          figsize=(3.2 * N_COLS, 3.6 * N_ROWS),
                          gridspec_kw={'wspace': 0.03, 'hspace': 0.25},
                          squeeze=False)
fig.patch.set_facecolor(BG)

proj_map = {p.proj_key: p for p in spec.projections}

for row_i, fk in enumerate(frame_keys):
    res  = per_proj[fk]
    proj = proj_map[fk]
    print(f'  Rendering {fk} ({len(res["phase_log"])} phases) ...')
    # pad axes_row to N_COLS by hiding extra axes
    axes_row = axes[row_i]
    for ax in axes_row:
        ax.set_visible(False)
    n_used = 1 + 1 + len(res['phase_log'])
    for ax in axes_row[:n_used]:
        ax.set_visible(True)
    draw_row(axes_row[:n_used], proj, res, fk)

fig.suptitle('Registration phases — all frames', fontsize=12,
             color=FG, fontweight='bold', y=1.01)

out_all = OUT_DIR / 'swaroopa_phase_poses_all.png'
plt.savefig(out_all, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'\nSaved → {out_all}')
