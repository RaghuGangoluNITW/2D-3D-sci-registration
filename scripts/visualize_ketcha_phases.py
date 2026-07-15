#!/usr/bin/env python3
"""
visualize_ketcha_phases.py
==========================
Visualise the Ketcha 2017 registration phase log stored in
results/swaroopa_ketcha_results.json.

For each registered frame, renders a DRR at every logged phase:
  ground_truth | perturbed | phase1_multistart | phase2_refine

One PNG per frame + a combined figure are saved to results/figures/.

Usage
-----
    python scripts/visualize_ketcha_phases.py
    python scripts/visualize_ketcha_phases.py --results results/swaroopa_ketcha_results.json
    python scripts/visualize_ketcha_phases.py --frames ap_039 lat_023
"""

import argparse
import sys
import json
import cv2
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from swaroopa_loader import SwaroLoader, SWARO_PIX_MM, SWARO_IMG_SIZE
from run_swaroopa_diffdrr import build_subject_masked, build_subject, DiffDRRGenerator

# ── Style ──────────────────────────────────────────────────────────────────────
BG, FG = '#111111', '#dddddd'
LM_COLOURS = {'L1': '#ff4444', 'L2': '#ff9900', 'L3': '#ffee00',
               'L4': '#44ff44', 'L5': '#44ddff'}
matplotlib.rcParams.update({'text.color': FG, 'axes.labelcolor': FG,
                             'figure.facecolor': BG, 'axes.facecolor': BG})

RENDER_SIZE = 256


# ── Phase ordering ─────────────────────────────────────────────────────────────
PHASE_ORDER = ['ground_truth', 'perturbed', 'phase1_multistart', 'phase2_refine']
PHASE_LABELS = {
    'ground_truth':      'GT (EPnP)',
    'perturbed':         'Perturbed\n(init)',
    'phase1_multistart': 'Phase 1\n(multi-start)',
    'phase2_refine':     'Phase 2\n(refined)',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results/swaroopa_ketcha_results.json')
    p.add_argument('--frames', nargs='+', default=None,
                   help='Subset of frame keys to visualise (default: all in JSON)')
    p.add_argument('--cylinder_r', type=float, default=40.0,
                   help='Cylinder mask radius used during registration (default: 40)')
    p.add_argument('--no_cylinder', action='store_true',
                   help='Build subject without cylinder mask')
    p.add_argument('--hu_min', type=float, default=0.0,
                   help='Lower HU clip inside the cylinder (default: 0)')
    p.add_argument('--out_dir', default='results/figures')
    return p.parse_args()


def render_from_extrinsic(drr_gen, R, t, pts3d, size, pix_mm):
    """Render DRR and project landmarks from stored extrinsic matrices."""
    R_np = np.array(R, dtype=np.float64)
    t_np = np.array(t, dtype=np.float64).flatten()
    drr  = drr_gen.generate_from_extrinsic(R_np, t_np, size, pix_mm)
    uv   = drr_gen.project_pts(R_np, t_np, pts3d, size, pix_mm)
    return drr, uv


def draw_landmarks(ax, uv, lm_names, size):
    for j, name in enumerate(lm_names):
        u, v = uv[j]
        if 0 <= u < size and 0 <= v < size:
            ax.plot(u, v, 'o', color=LM_COLOURS.get(name, 'white'),
                    markersize=5, markeredgewidth=0.5, markeredgecolor='white', zorder=5)
            ax.text(u + 3, v - 3, name, color=LM_COLOURS.get(name, 'white'),
                    fontsize=5, fontweight='bold', zorder=6)


def draw_gt_landmarks_2d(ax, proj, size):
    """Overlay calibrated 2D GT landmark positions on the x-ray panel."""
    for name, gt in proj.gt_landmarks_2d.items():
        u = gt[0] * size / SWARO_IMG_SIZE
        v = gt[1] * size / SWARO_IMG_SIZE
        ax.plot(u, v, 'D', color=LM_COLOURS.get(name, 'white'),
                markersize=5, markeredgewidth=0.5, markeredgecolor='white', zorder=5)
        ax.text(u + 3, v - 3, name, color=LM_COLOURS.get(name, 'white'),
                fontsize=5, fontweight='bold', zorder=6)


def make_frame_figure(proj, res, drr_gen, lm_names, pts3d, frame_key):
    """Return a matplotlib Figure with one column per phase + leftmost x-ray."""
    phase_log = res['phase_log']

    # Sort phase_log entries into the canonical order
    phase_map = {entry['phase']: entry for entry in phase_log}
    phases = [p for p in PHASE_ORDER if p in phase_map]

    pix_mm = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)
    xray   = 1.0 - cv2.resize(proj.image_raw, (RENDER_SIZE, RENDER_SIZE),
                               interpolation=cv2.INTER_AREA)

    n_cols  = 1 + len(phases)   # x-ray + one per phase
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(3.2 * n_cols, 4.0),
                             gridspec_kw={'wspace': 0.03})
    fig.patch.set_facecolor(BG)
    ok_str = '✓' if res.get('success') else '✗'

    # ── Col 0: X-ray ─────────────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor(BG)
    ax0.imshow(xray, cmap='gray', vmin=0, vmax=1)
    ax0.set_title(f'X-ray\n{frame_key}  {ok_str}', fontsize=8, color=FG, pad=4)
    ax0.set_xticks([]); ax0.set_yticks([])
    draw_gt_landmarks_2d(ax0, proj, RENDER_SIZE)

    # ── Phase columns ─────────────────────────────────────────────────────────
    for ci, phase_key in enumerate(phases):
        entry = phase_map[phase_key]
        ax = axes[ci + 1]
        ax.set_facecolor(BG)

        # Render from stored R_proj / t_proj matrices
        drr, uv = render_from_extrinsic(
            drr_gen,
            entry['R_proj'], entry['t_proj'],
            pts3d, RENDER_SIZE, pix_mm,
        )
        ax.imshow(drr, cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])

        cost_val = entry.get('cost')
        pde_val  = entry.get('pde_mm')
        cost_str = f'cost={cost_val:.3f}' if cost_val is not None else 'cost=GT'
        pde_str  = (f'PDE={float(pde_val):.1f}mm'
                    if pde_val is not None and str(pde_val).lower() not in ('none', 'nan', 'inf')
                    else 'PDE=0.0mm' if phase_key == 'ground_truth' else 'PDE=N/A')

        label = PHASE_LABELS.get(phase_key, phase_key)
        ax.set_title(f'{label}\n{cost_str}  {pde_str}', fontsize=8, color=FG, pad=4)

        draw_landmarks(ax, uv, lm_names, RENDER_SIZE)

    fig.suptitle(
        f'{frame_key} — Ketcha 2017 registration phases  '
        f'(init PDE={res["initial_pde_mm"]:.1f}mm → final PDE={res["final_pde_mm"]:.1f}mm)',
        fontsize=10, color=FG, fontweight='bold', y=1.03,
    )
    return fig


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load results ──────────────────────────────────────────────────────────
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(
            f'Results JSON not found: {results_path}\n'
            'Run:  python run_swaroopa_ketcha.py --frames ap_039 lat_023 --fast')
    with open(results_path) as f:
        data = json.load(f)

    per_proj = data['swaroopa']['per_projection']
    frame_keys = args.frames or [k for k, v in per_proj.items() if v.get('phase_log')]
    # Keep only frames actually present in JSON
    frame_keys = [k for k in frame_keys if k in per_proj and per_proj[k].get('phase_log')]

    if not frame_keys:
        raise RuntimeError('No frames with phase_log found. Re-run registration first.')

    print(f'Frames to visualise: {frame_keys}')

    # ── Load specimen (only the frames we need) ───────────────────────────────
    print('Loading specimen ...')
    loader = SwaroLoader()
    spec   = loader.load(frames=frame_keys, verbose=False)

    lm_names = sorted(spec.landmarks_3d.keys())
    pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])

    # ── Build DRR generator ───────────────────────────────────────────────────
    print('Building DiffDRR subject ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.no_cylinder:
        subject = build_subject(spec)
    else:
        subject = build_subject_masked(spec, cylinder_r_mm=args.cylinder_r,
                                       hu_min=args.hu_min)
    drr_gen = DiffDRRGenerator(subject, device, ct_origin_lps=spec.ct_origin)
    print(f'DRR generator ready  (device={device})')

    proj_map = {p.proj_key: p for p in spec.projections}

    figs_all = []
    for fk in frame_keys:
        print(f'\nRendering {fk} ...')
        res  = per_proj[fk]
        proj = proj_map[fk]

        fig = make_frame_figure(proj, res, drr_gen, lm_names, pts3d, fk)

        # Save per-frame PNG
        out_frame = out_dir / f'ketcha_phases_{fk}.png'
        fig.savefig(out_frame, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f'  Saved → {out_frame}')
        figs_all.append((fk, fig))

    # ── Combined figure (stacked rows) ────────────────────────────────────────
    if len(frame_keys) > 1:
        print('\nBuilding combined figure ...')
        # Determine max phases across frames
        max_phases = max(
            len([p for p in PHASE_ORDER if p in {e['phase'] for e in per_proj[k]['phase_log']}])
            for k in frame_keys
        )
        n_cols = 1 + max_phases
        n_rows = len(frame_keys)
        pix_mm = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)

        fig_all, axes_all = plt.subplots(
            n_rows, n_cols,
            figsize=(3.2 * n_cols, 4.0 * n_rows),
            gridspec_kw={'wspace': 0.03, 'hspace': 0.28},
            squeeze=False,
        )
        fig_all.patch.set_facecolor(BG)

        for row_i, fk in enumerate(frame_keys):
            res  = per_proj[fk]
            proj = proj_map[fk]
            phase_log = res['phase_log']
            phase_map_f = {entry['phase']: entry for entry in phase_log}
            phases = [p for p in PHASE_ORDER if p in phase_map_f]

            xray = 1.0 - cv2.resize(proj.image_raw, (RENDER_SIZE, RENDER_SIZE),
                                     interpolation=cv2.INTER_AREA)
            ok_str = '✓' if res.get('success') else '✗'

            # Hide unused columns
            for ax in axes_all[row_i]:
                ax.set_visible(False)
            n_used = 1 + len(phases)
            for ax in axes_all[row_i, :n_used]:
                ax.set_visible(True)

            ax0 = axes_all[row_i, 0]
            ax0.set_facecolor(BG)
            ax0.imshow(xray, cmap='gray', vmin=0, vmax=1)
            ax0.set_title(f'X-ray  {fk}  {ok_str}', fontsize=8, color=FG, pad=4)
            ax0.set_xticks([]); ax0.set_yticks([])
            draw_gt_landmarks_2d(ax0, proj, RENDER_SIZE)

            for ci, phase_key in enumerate(phases):
                entry = phase_map_f[phase_key]
                ax = axes_all[row_i, ci + 1]
                ax.set_facecolor(BG)
                drr, uv = render_from_extrinsic(
                    drr_gen, entry['R_proj'], entry['t_proj'],
                    pts3d, RENDER_SIZE, pix_mm,
                )
                ax.imshow(drr, cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                cost_val = entry.get('cost')
                pde_val  = entry.get('pde_mm')
                cost_str = f'cost={cost_val:.3f}' if cost_val is not None else 'cost=GT'
                pde_str  = (f'PDE={float(pde_val):.1f}mm'
                            if pde_val is not None and str(pde_val).lower() not in ('none', 'nan')
                            else 'PDE=0.0mm' if phase_key == 'ground_truth' else 'PDE=N/A')
                label = PHASE_LABELS.get(phase_key, phase_key)
                ax.set_title(f'{label}\n{cost_str}  {pde_str}', fontsize=8, color=FG, pad=4)
                draw_landmarks(ax, uv, lm_names, RENDER_SIZE)

        fig_all.suptitle('Ketcha 2017 — registration phases', fontsize=12,
                         color=FG, fontweight='bold', y=1.01)
        out_all = out_dir / 'ketcha_phases_all.png'
        fig_all.savefig(out_all, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f'Saved combined → {out_all}')

    plt.close('all')
    print('\nDone.')


if __name__ == '__main__':
    main()
