#!/usr/bin/env python3
"""
visualize_arjun_ms.py — msLevelCheck Registration Visualisation (Arjun)
========================================================================
Renders a multi-panel figure for each frame in arjun_ms_results.json.
DRRs are produced with DeepFluoroDRR using the same masked-CT strategy
as the registration, so each panel shows what the optimiser was seeing.

  Row 1 (per frame):
    [Real X-ray]  [EPnP init]  [S1-Phase1]  [S1-Phase2]  [S1-Phase3]
    (Stage-1 masked full-spine CT)

  Row 2 (per frame):
    [Best S2 group — masked to its 3-vertebra cylinder]
    [S3-L1 — masked]  [S3-L2]  …  [S3-L5]

Usage:
    python scripts/visualize_arjun_ms.py
    python scripts/visualize_arjun_ms.py --results results/arjun_ms_results.json
    python scripts/visualize_arjun_ms.py --out_dir results/figures/arjun_ms
"""

import argparse
import copy
import json
import sys
import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')
matplotlib.rcParams.update({
    'text.color':       '#dddddd',
    'axes.labelcolor':  '#dddddd',
    'xtick.color':      '#dddddd',
    'ytick.color':      '#dddddd',
    'axes.edgecolor':   '#444',
    'axes.facecolor':   '#1e1e1e',
    'figure.facecolor': '#111111',
    'grid.color':       '#333',
    'grid.linestyle':   '--',
    'grid.alpha':       0.6,
})

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from arjun_loader import (
    ArjunLoader,
    ArjunProjection,
    pixel_spacing_mm,
    _make_K,
    ARJUN_REF_PIX_MM,
    ARJUN_REF_SIZE,
)
from deepfluoro_loader import DeepFluoroSpecimen, perturb_extrinsic
from deepfluoro_drr import DeepFluoroDRR
from similarity import go_cost

from run_arjun_ms import (
    masked_specimen,
    build_stage_groups,
    compute_spine_axis,
    CYLINDER_RADIUS_MM,
    STAGE2_GROUP_SIZE,
)

RENDER_SIZE = 256
OUT_DIR     = Path('results/figures/arjun_ms')

# Pixel spacing at RENDER_SIZE
RENDER_PIX  = ARJUN_REF_PIX_MM * (ARJUN_REF_SIZE / RENDER_SIZE)
RENDER_STEPS = 120


# ---------------------------------------------------------------------------
# Build a dict of DeepFluoroDRR generators — one per stage group + s1
# ---------------------------------------------------------------------------

def _build_drr_generators(spec, lm_names):
    gens = {}

    print('  Building DRR: Stage-1 full CT (hu_threshold=150) …', flush=True)
    gens['s1'] = DeepFluoroDRR(spec, hu_threshold=150.0)

    s2_groups = build_stage_groups(lm_names, group_size=STAGE2_GROUP_SIZE)
    s3_verts  = [[lm] for lm in lm_names]

    for group in s2_groups + s3_verts:
        key = '+'.join(group) if len(group) > 1 else group[0]
        print(f'  Building DRR: {key} …', flush=True)
        gens[key] = DeepFluoroDRR(masked_specimen(spec, group), hu_threshold=0.0)

    return gens


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _snap_to_drr(generators, gen_key, snap, proj):
    gen = generators.get(gen_key, generators['s1'])
    R   = np.array(snap['R'], dtype=np.float64)
    t   = np.array(snap['t'], dtype=np.float64)
    return gen.generate_from_extrinsic(R, t, RENDER_SIZE, RENDER_PIX, RENDER_STEPS)


def _proj_pts(spec, proj, R, t):
    """Project all 3D landmarks → pixel coords using per-frame Arjun intrinsics."""
    lm_names  = sorted(spec.landmarks_3d.keys())
    pts3d     = np.array([spec.landmarks_3d[n] for n in lm_names])
    fx, fy, cx, cy = _make_K(proj.img_w, proj.img_h)

    from deepfluoro_loader import xzy
    P_cam = (R @ xzy(pts3d).T).T + t
    u = fx * P_cam[:, 0] / P_cam[:, 2] + cx
    v = fy * P_cam[:, 1] / P_cam[:, 2] + cy
    uv = np.stack([u, v], axis=1)
    return {n: uv[i] for i, n in enumerate(lm_names)}


def _mean_pde(proj_obj, pred_uv):
    pix_mm = pixel_spacing_mm(proj_obj.img_w, proj_obj.img_h)
    errs = []
    for name, uv in pred_uv.items():
        if name in proj_obj.gt_landmarks_2d:
            gt = proj_obj.gt_landmarks_2d[name]
            if 0 < gt[0] < proj_obj.img_w and 0 < gt[1] < proj_obj.img_h:
                errs.append(np.linalg.norm(np.array(uv) - np.array(gt)) * pix_mm)
    return float(np.mean(errs)) if errs else float('nan')


def _overlay_landmarks(ax, image, pts2d_gt, pts2d_pred=None, title='', go=None, pde=None):
    scale = RENDER_SIZE / max(1, len(image))   # image is already RENDER_SIZE square
    ax.imshow(image, cmap='gray', vmin=0, vmax=1, aspect='equal',
              extent=[0, RENDER_SIZE, RENDER_SIZE, 0])

    if pts2d_gt:
        for name, uv in pts2d_gt.items():
            sx = float(uv[0]) / max(1, next(iter([p.img_w for p in []])) if False else 1)
            # Scale GT coords from original image space → RENDER_SIZE
            pass  # handled below with explicit scale_gt
    # Re-do with correct scaling
    ax.cla()
    ax.set_facecolor('#1e1e1e')
    ax.imshow(image, cmap='gray', vmin=0, vmax=1, aspect='equal',
              extent=[0, RENDER_SIZE, RENDER_SIZE, 0])

    if pts2d_gt:
        for name, uv in pts2d_gt.items():
            ax.plot(float(uv[0]), float(uv[1]), 'g+', ms=8, mew=1.5)
            ax.text(float(uv[0]) + 3, float(uv[1]) - 3, name,
                    color='#88ff88', fontsize=5.5)
    if pts2d_pred:
        for name, uv in pts2d_pred.items():
            ax.plot(float(uv[0]), float(uv[1]), 'r+', ms=8, mew=1.5)

    label_parts = [title]
    if go  is not None: label_parts.append(f'GO={go:.3f}')
    if pde is not None: label_parts.append(f'PDE={pde:.1f}mm')
    ax.set_title('\n'.join(label_parts), fontsize=6.5, pad=2, color='#dddddd')
    ax.axis('off')


# ---------------------------------------------------------------------------
# Per-frame figure
# ---------------------------------------------------------------------------

def render_frame_figure(frame_key, res, spec, generators, proj_obj):
    poses    = res.get('poses', {})
    s1_snaps = poses.get('stage1', {})
    s2_snaps = poses.get('stage2', {})
    s3_snaps = poses.get('stage3', {})
    lm_names = sorted(spec.landmarks_3d.keys())

    # Scale GT landmarks from original image space to RENDER_SIZE
    img_scale = RENDER_SIZE / proj_obj.img_w   # same scale used for both axes
    gt_uv_render = {n: [proj_obj.gt_landmarks_2d[n][0] * img_scale,
                         proj_obj.gt_landmarks_2d[n][1] * img_scale]
                    for n in lm_names if n in proj_obj.gt_landmarks_2d}

    def _pred_render(snap):
        R, t = np.array(snap['R']), np.array(snap['t'])
        pred = _proj_pts(spec, proj_obj, R, t)
        # Scale predicted UV (in original image pixels) to RENDER_SIZE
        return {n: [uv[0] * img_scale, uv[1] * img_scale] for n, uv in pred.items()}

    # ── Row 1: Stage-1 snapshots ──────────────────────────────────────────
    row1 = []
    xray = cv2.resize(proj_obj.image_raw, (RENDER_SIZE, RENDER_SIZE),
                      interpolation=cv2.INTER_AREA)
    row1.append(('X-ray', xray, gt_uv_render, None, None, None))

    for snap_key, label in [('epnp',   'EPnP init'),
                             ('phase1', 'S1-P1\n(64px)'),
                             ('phase2', 'S1-P2\n(180px)'),
                             ('phase3', 'S1-P3\n(256px)')]:
        if snap_key not in s1_snaps:
            continue
        snap = s1_snaps[snap_key]
        drr  = _snap_to_drr(generators, 's1', snap, proj_obj)
        pred = _pred_render(snap)
        pde  = snap.get('pde_mm', _mean_pde(proj_obj, {n: _proj_pts(spec, proj_obj,
                         np.array(snap['R']), np.array(snap['t']))[n] for n in lm_names}))
        row1.append((label, drr, gt_uv_render, pred, snap.get('go'), pde))

    # ── Row 2: Stage-2 best + Stage-3 per vertebra ───────────────────────
    row2 = []

    best_s2_key = min(res.get('stage2_go', {'_': 999}).items(), key=lambda x: x[1])[0]
    if best_s2_key and best_s2_key in s2_snaps:
        snap    = s2_snaps[best_s2_key].get('phase3', {})
        if snap:
            drr  = _snap_to_drr(generators, best_s2_key, snap, proj_obj)
            pred = _pred_render(snap)
            grp_lms = best_s2_key.split('+')
            pde  = snap.get('pde_mm', _mean_pde(proj_obj, pred))
            row2.append((f'S2 best\n{best_s2_key}', drr,
                         {n: gt_uv_render[n] for n in grp_lms if n in gt_uv_render},
                         {n: pred[n] for n in grp_lms if n in pred},
                         snap.get('go'), pde))

    for lm in lm_names:
        if lm not in s3_snaps:
            continue
        snap = s3_snaps[lm].get('phase3', {})
        if not snap:
            continue
        drr  = _snap_to_drr(generators, lm, snap, proj_obj)
        pred = _pred_render(snap)
        pde  = snap.get('pde_mm', _mean_pde(proj_obj, pred))
        row2.append((f'S3 {lm}', drr,
                     {lm: gt_uv_render[lm]} if lm in gt_uv_render else {},
                     {lm: pred[lm]} if lm in pred else {},
                     snap.get('go'), pde))

    # ── Build figure ──────────────────────────────────────────────────────
    ncols = max(len(row1), len(row2))
    cell  = RENDER_SIZE / 72 + 0.5
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * cell, 2 * cell + 1.0))
    fig.patch.set_facecolor('#111111')
    if ncols == 1:
        axes = axes[:, np.newaxis]

    ok  = res.get('success', False)
    col = '#44dd88' if ok else '#ee4444'
    fig.suptitle(
        f'{frame_key}  |  GO: {res["initial_go"]:.4f}→{res["final_go"]:.4f}  '
        f'ΔGO={res["go_delta"]:+.4f}  |  '
        f'PDE: {res["initial_pde_mm"]:.1f}→{res["final_pde_mm"]:.1f}mm  '
        f'[{"SUCCESS" if ok else "FAIL"}]',
        fontsize=9, fontweight='bold', color=col, y=1.01,
    )

    for ci in range(ncols):
        for ri, row in enumerate([row1, row2]):
            ax = axes[ri, ci]
            ax.set_facecolor('#1e1e1e')
            if ci < len(row):
                title, img, gt, pred, go, pde = row[ci]
                _overlay_landmarks(ax, img, gt, pred, title, go, pde)
            else:
                ax.axis('off')

    axes[0, 0].set_ylabel('Stage 1  (masked CT)', fontsize=7, color='#aaaaaa',
                           rotation=90, labelpad=4)
    axes[1, 0].set_ylabel('Stage 2-3  (masked CT)', fontsize=7, color='#aaaaaa',
                           rotation=90, labelpad=4)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


# ---------------------------------------------------------------------------
# Summary dashboard
# ---------------------------------------------------------------------------

def render_summary(all_keys, all_res):
    n = len(all_keys)
    x = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(max(12, n * 1.6), 10))
    fig.patch.set_facecolor('#111111')

    lm_names = sorted(next(iter(all_res.values()))['poses']['stage3'].keys())

    # GO progression
    ax = axes[0]
    ax.set_facecolor('#1e1e1e')
    ax.set_title('GO cost progression through msLevelCheck stages', fontsize=10)
    ax.set_ylabel('GO cost  (lower = better)', fontsize=9)

    go_epnp, go_s1, go_s2, go_s3 = [], [], [], []
    for k in all_keys:
        r = all_res[k]
        go_epnp.append(r['initial_go'])
        go_s1.append(r['stage1_go'])
        go_s2.append(np.mean(list(r['stage2_go'].values())))
        go_s3.append(r['final_go'])

    for label, vals, color, marker, ls in [
        ('EPnP',   go_epnp, '#aaaaaa', 'o', '--'),
        ('Stage1', go_s1,   '#66aaff', 's', '-'),
        ('Stage2', go_s2,   '#ffaa44', '^', '-'),
        ('Stage3', go_s3,   '#44dd88', 'D', '-'),
    ]:
        ax.plot(x, vals, color=color, marker=marker, linestyle=ls,
                linewidth=1.5, markersize=6, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(all_keys, rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(axis='y')
    for xi, k in enumerate(all_keys):
        ok = all_res[k].get('success', False)
        ax.axvspan(xi - 0.4, xi + 0.4,
                   color='#44dd88' if ok else '#ee4444', alpha=0.06)

    # PDE per landmark
    ax2 = axes[1]
    ax2.set_facecolor('#1e1e1e')
    ax2.set_title('Per-landmark PDE (mm) — Stage 3 final', fontsize=10)
    ax2.set_ylabel('PDE (mm)', fontsize=9)
    bar_w     = 0.15
    lm_colors = ['#4499ff', '#ff9944', '#44dd88', '#ff4466', '#cc88ff']
    for li, lm in enumerate(lm_names):
        pde_vals = [all_res[k]['pde_per_lm'].get(lm, float('nan')) for k in all_keys]
        offsets  = x + (li - len(lm_names) / 2 + 0.5) * bar_w
        bars     = ax2.bar(offsets, pde_vals, width=bar_w,
                           color=lm_colors[li % len(lm_colors)],
                           label=lm, alpha=0.85, edgecolor='#555')
        for bar, v in zip(bars, pde_vals):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.5,
                         f'{v:.0f}', ha='center', va='bottom',
                         fontsize=5.5, color='#cccccc')

    mean_pde_s1 = [all_res[k]['stage1_pde_mm'] for k in all_keys]
    mean_pde_ms = [all_res[k]['final_pde_mm']   for k in all_keys]
    ax2.plot(x, mean_pde_s1, 'w--', linewidth=1.2, alpha=0.5, label='Stage1 mean')
    ax2.plot(x, mean_pde_ms, 'w-',  linewidth=1.8, alpha=0.8, label='MS final mean')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_keys, rotation=30, ha='right', fontsize=8)
    ax2.legend(fontsize=8, framealpha=0.3, ncol=len(lm_names) + 2)
    ax2.grid(axis='y')

    go_epnp_m = np.mean(go_epnp)
    go_s3_m   = np.mean(go_s3)
    pde_s1_m  = np.nanmean(mean_pde_s1)
    pde_ms_m  = np.nanmean(mean_pde_ms)
    fig.suptitle(
        f'msLevelCheck (Arjun) Summary — {n} frames  |  '
        f'Mean GO: {go_epnp_m:.4f}→{go_s3_m:.4f}  |  '
        f'Mean PDE: {pde_s1_m:.1f}→{pde_ms_m:.1f}mm',
        fontsize=11, fontweight='bold', y=1.005,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results/arjun_ms_results.json')
    p.add_argument('--out_dir', default='results/figures/arjun_ms')
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results) as f:
        data = json.load(f)
    pp         = data.get('per_projection') or data
    frame_keys = sorted(pp.keys())
    print(f'Loaded {len(frame_keys)} frames: {frame_keys}')

    loader   = ArjunLoader()
    spec     = loader.load(frames=frame_keys, verbose=False)
    proj_map = {p.proj_key: p for p in spec.projections}
    lm_names = sorted(spec.landmarks_3d.keys())

    print('Building DRR generators …')
    generators = _build_drr_generators(spec, lm_names)
    print(f'  Ready: {sorted(generators.keys())}')

    for frame_key in frame_keys:
        if frame_key not in pp or frame_key not in proj_map:
            print(f'  Skipping {frame_key} (missing data)')
            continue
        print(f'  Rendering {frame_key} ...', flush=True)
        fig = render_frame_figure(frame_key, pp[frame_key], spec,
                                   generators, proj_map[frame_key])
        out = out_dir / f'ms_{frame_key}.png'
        fig.savefig(out, dpi=130, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'    Saved: {out}')

    print('  Rendering summary dashboard ...')
    fig_sum = render_summary(frame_keys, pp)
    out_sum = out_dir / 'ms_summary.png'
    fig_sum.savefig(out_sum, dpi=130, bbox_inches='tight',
                    facecolor=fig_sum.get_facecolor())
    plt.close(fig_sum)
    print(f'  Saved: {out_sum}')

    print('\nAll done.')
    for fk in frame_keys:
        print(f'  {out_dir}/ms_{fk}.png')
    print(f'  {out_dir}/ms_summary.png')


if __name__ == '__main__':
    main()
