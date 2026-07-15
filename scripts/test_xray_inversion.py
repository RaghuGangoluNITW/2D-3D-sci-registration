#!/usr/bin/env python3
"""
test_xray_inversion.py
======================
Compares two registration strategies on 2 AP + 2 lateral Swaroopa frames:

  NORMAL  : target = xray,       objective = NCC(1-drr, xray)   [current]
  INVERTED: target = 1-xray,     objective = NCC(drr,   1-xray) [new]

Saves side-by-side visualisation:
  results/figures/swaroopa_xray_inversion_test.png
"""

import sys, time
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from swaroopa_loader import (
    SwaroLoader, project_world_swaro,
    SWARO_PIX_MM, SWARO_IMG_SIZE,
)
from deepfluoro_loader import perturb_extrinsic
from deepfluoro_drr import DeepFluoroDRR
from similarity import go_cost as compute_go_cost, normalized_cross_correlation
from optimizer import run_cmaes_single

# ── Config ────────────────────────────────────────────────────────────────────
TEST_FRAMES  = ['ap_013', 'ap_031', 'lat_003', 'lat_026']
RENDER_SIZE  = 192
OUT          = Path('results/figures/swaroopa_xray_inversion_test.png')
LM_COLOURS   = {'L1':'#ff4444','L2':'#ff9900','L3':'#ffee00','L4':'#44ff44','L5':'#44ddff'}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading specimen ...")
loader  = SwaroLoader()
spec    = loader.load(frames=TEST_FRAMES, verbose=True)
drr_gen = DeepFluoroDRR(spec, hu_threshold=150.0)

lm_names = sorted(spec.landmarks_3d.keys())
pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
scale_lm = RENDER_SIZE / SWARO_IMG_SIZE

# ── Registration helper ───────────────────────────────────────────────────────
def run_reg(proj, invert_xray: bool):
    """3-phase CMA-ES starting from EPnP, with normal or inverted xray target."""
    R_gt = proj.R_proj.copy()
    t_gt = proj.t_proj.copy()

    COARSE, OPT, FINE = 64, 180, 256
    pix = lambda sz: SWARO_PIX_MM * (SWARO_IMG_SIZE / sz)

    raw = proj.image_raw
    if invert_xray:
        raw = 1.0 - raw   # ← invert

    tgt_c = cv2.resize(raw, (COARSE, COARSE), interpolation=cv2.INTER_AREA)
    tgt_o = cv2.resize(raw, (OPT,    OPT),    interpolation=cv2.INTER_AREA)
    tgt_f = cv2.resize(raw, (FINE,   FINE),   interpolation=cv2.INTER_AREA)

    def _ncc_cost(drr, tgt, invert_drr):
        cov = np.count_nonzero(drr > 0.01) / drr.size
        if cov < 0.08:
            return 1.0
        src = (1.0 - drr) if invert_drr else drr
        return float(1.0 - normalized_cross_correlation(src, tgt))

    def _go(drr, tgt):
        cov = np.count_nonzero(drr > 0.01) / drr.size
        if cov < 0.08:
            return 1.0
        return float(compute_go_cost(drr, tgt))

    def _ncc_go(drr, tgt, invert_drr):
        return 0.5 * _ncc_cost(drr, tgt, invert_drr) + 0.5 * _go(drr, tgt)

    # When xray is inverted we compare DRR directly (no inversion)
    # When xray is normal   we compare 1-DRR (flip DRR polarity)
    inv_drr = not invert_xray

    def obj(cost_fn, tgt, sz, steps):
        def f(x):
            R, t = perturb_extrinsic(R_gt, t_gt, x[:3], x[3:])
            drr = drr_gen.generate_from_extrinsic(R, t, sz, pix(sz), steps)
            return cost_fn(drr, tgt)
        return f

    obj_c = obj(lambda d,t: _ncc_cost(d, t, inv_drr), tgt_c, COARSE, 40)
    obj_o = obj(lambda d,t: _ncc_go(d, t, inv_drr),   tgt_o, OPT,    80)
    obj_f = obj(lambda d,t: _ncc_cost(d, t, inv_drr), tgt_f, FINE,   120)

    def go_eval(R, t):
        drr = drr_gen.generate_from_extrinsic(R, t, OPT, pix(OPT), 80)
        return float(_go(drr, tgt_o))

    search = np.array([10.]*3 + [20.]*3)

    # Phase 1: small grid around EPnP
    rng = np.random.default_rng(42)
    grid = (rng.random((15, 6))*2 - 1) * search
    grid = np.vstack([np.zeros(6), grid])
    costs = np.array([obj_c(x) for x in grid])
    seeds = grid[np.argsort(costs)[:8]]

    # Phase 2: CMA-ES NCC+GO at 180px
    best_pose, best_cost = np.zeros(6), obj_o(np.zeros(6))
    for x0 in seeds:
        r = run_cmaes_single(obj_o, x0, search/3, bounds_center=np.zeros(6),
                             popsize=16, tolx=0.2, maxiter=300)
        if r.cost < best_cost:
            best_cost, best_pose = r.cost, r.pose.copy()

    # Phase 3: fine NCC at 256px
    r3 = run_cmaes_single(obj_f, best_pose, search/8, bounds_center=np.zeros(6),
                          popsize=16, tolx=0.05, maxiter=300)
    best_x = r3.pose if r3.cost < obj_f(best_pose) else best_pose

    R_f, t_f = perturb_extrinsic(R_gt, t_gt, best_x[:3], best_x[3:])

    init_go  = go_eval(R_gt, t_gt)
    final_go = go_eval(R_f,  t_f)
    return R_f, t_f, best_x, init_go, final_go


# ── Run both strategies for all 4 frames ─────────────────────────────────────
results = {}
for proj in spec.projections:
    key = proj.proj_key
    print(f"\n{'='*50}")
    for inv in [False, True]:
        label = 'INVERTED' if inv else 'NORMAL'
        print(f"  [{key}] {label} xray ...")
        t0 = time.time()
        R_f, t_f, delta, ig, fg = run_reg(proj, invert_xray=inv)
        dt = time.time() - t0
        dgo = ig - fg
        succ = dgo > 0.05 and fg < 0.6
        print(f"    GO {ig:.4f}→{fg:.4f}  ΔGO={dgo:+.4f}  {'✓' if succ else '✗'}  ({dt:.0f}s)")
        results[(key, inv)] = dict(R_f=R_f, t_f=t_f, delta=delta,
                                   init_go=ig, final_go=fg, dgo=dgo, success=succ)


# ── Render final DRRs + build figure ─────────────────────────────────────────
pix_r = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)

def render(R, t):
    return drr_gen.generate_from_extrinsic(R, t, RENDER_SIZE, pix_r, 120)

def proj_lm(R, t):
    return project_world_swaro(pts3d, R, t) * scale_lm

def add_lm(ax, uv, gt_lm=None, proj_key=None):
    if gt_lm is not None:
        for j, n in enumerate(lm_names):
            if n in gt_lm:
                u, v = np.array(gt_lm[n]) * scale_lm
                ax.plot(u, v, 'o', color=LM_COLOURS[n], ms=4, mew=1, mec='white', zorder=5)
    else:
        for j, n in enumerate(lm_names):
            u, v = uv[j]
            if 0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE:
                ax.plot(u, v, 'o', color=LM_COLOURS[n], ms=4, mew=1, mec='white', zorder=5)

print("\nRendering final DRRs ...")

# Layout: 4 frames (rows) × 5 cols
# Col: [X-ray raw] [X-ray inv] [Init DRR] [Final NORMAL] [Final INVERTED]
N_ROWS = len(TEST_FRAMES)
fig, axes = plt.subplots(N_ROWS, 5, figsize=(17, N_ROWS * 3.0))
fig.patch.set_facecolor('#111111')

col_titles = [
    'X-ray (original)',
    'X-ray (inverted)',
    'Initial DRR (EPnP)',
    'Final — Normal xray\nNCC(1−DRR, xray)',
    'Final — Inverted xray\nNCC(DRR, 1−xray)',
]
for ci, ct in enumerate(col_titles):
    axes[0, ci].set_title(ct, color='white', fontsize=8, pad=4)

for ri, proj in enumerate(spec.projections):
    key = proj.proj_key
    xray     = proj.image_raw
    xray_inv = 1.0 - xray
    xray_s   = cv2.resize(xray,     (RENDER_SIZE, RENDER_SIZE), interpolation=cv2.INTER_AREA)
    xray_inv_s = cv2.resize(xray_inv,(RENDER_SIZE, RENDER_SIZE), interpolation=cv2.INTER_AREA)

    # Initial DRR at EPnP pose
    drr_init = render(proj.R_proj, proj.t_proj)
    uv_init  = proj_lm(proj.R_proj, proj.t_proj)

    rN = results[(key, False)]
    rI = results[(key, True)]

    drr_N = render(rN['R_f'], rN['t_f'])
    drr_I = render(rI['R_f'], rI['t_f'])
    uv_N  = proj_lm(rN['R_f'], rN['t_f'])
    uv_I  = proj_lm(rI['R_f'], rI['t_f'])

    imgs = [
        (xray_s,    None,    None),
        (xray_inv_s,None,    None),
        (drr_init,  uv_init, None),
        (drr_N,     uv_N,    rN),
        (drr_I,     uv_I,    rI),
    ]

    for ci, (img, uv, res) in enumerate(imgs):
        ax = axes[ri, ci]
        ax.set_facecolor('#111111')
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#333')

        if ci == 0:
            ax.set_ylabel(key, color='white', fontsize=9)
            add_lm(ax, None, gt_lm=proj.gt_landmarks_2d)
        elif uv is not None:
            add_lm(ax, uv)

        if res is not None:
            dgo  = res['dgo']
            fg   = res['final_go']
            succ = res['success']
            col  = '#44ff44' if succ else '#ff4444'
            tick = '✓' if succ else '✗'
            ax.set_xlabel(f"GO {res['init_go']:.3f}→{fg:.3f}  ΔGO={dgo:+.3f} {tick}",
                          color=col, fontsize=7)
        elif ci == 2:
            ig_n = results[(key,False)]['init_go']
            ax.set_xlabel(f"Init GO={ig_n:.3f}", color='#aaaaaa', fontsize=7)

# Legend
lm_h = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
               markersize=6, label=n, mec='white')
        for n,c in LM_COLOURS.items()]
fig.legend(handles=lm_h, loc='lower center', ncol=5,
           facecolor='#1e1e1e', edgecolor='#555', labelcolor='white',
           fontsize=8, title='Vertebral centroids', title_fontsize=8,
           bbox_to_anchor=(0.5, 0.0))

plt.suptitle(
    'Swaroopa: Effect of X-ray Inversion on Registration\n'
    'Normal: NCC(1−DRR, xray)   |   Inverted: NCC(DRR, 1−xray)',
    color='white', fontsize=11, y=1.0
)
plt.tight_layout(rect=[0, 0.04, 1, 0.98])
plt.subplots_adjust(hspace=0.12, wspace=0.04)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved: {OUT}")
