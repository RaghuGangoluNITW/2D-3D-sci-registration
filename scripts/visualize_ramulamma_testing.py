#!/usr/bin/env python3
"""
visualize_ramulamma_testing.py — Visualise Ramulamma testing registration results
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ramulamma_testing_loader import (
    RamuTestLoader,
    RamuTestProjection,
    compute_pde_ramu_test,
    RAMU_TEST_PIX_MM,
    RAMU_TEST_IMG_SIZE,
)
from deepfluoro_drr import DeepFluoroDRR
from deepfluoro_loader import perturb_extrinsic

BASE   = Path(__file__).parent.parent
FIGS   = BASE / 'results' / 'figures'
FIGS.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = BASE / 'results' / 'ramulamma_testing_results.json'


def load_results():
    with open(RESULTS_JSON) as f:
        return json.load(f)


def render_drr(drr_gen, proj, best_x, size=256):
    R_gt = proj.R_proj
    t_gt = proj.t_proj
    pix  = RAMU_TEST_PIX_MM * (RAMU_TEST_IMG_SIZE / size)
    R_f, t_f = perturb_extrinsic(R_gt, t_gt, best_x[:3], best_x[3:])
    drr_init  = drr_gen.generate_from_extrinsic(R_gt, t_gt, size, pix, 120)
    drr_final = drr_gen.generate_from_extrinsic(R_f,  t_f,  size, pix, 120)
    return drr_init, drr_final


# ── Figure 1: Overview grid (X-ray | DRR@EPnP | DRR@final | overlay) ───────

def make_overview(spec, drr_gen, results_data):
    per_proj = results_data['ramulamma_test']['per_projection']
    projs    = spec.projections
    n        = len(projs)
    size     = 200

    fig, axes = plt.subplots(n, 4, figsize=(14, n * 3.0))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle('Ramulamma Testing — Registration Overview\n'
                 'X-ray  |  DRR@EPnP  |  DRR@Final  |  Overlay',
                 fontsize=13, fontweight='bold')

    col_titles = ['X-ray (C-arm)', 'DRR @ EPnP init', 'DRR @ Final pose', 'Overlay']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight='bold')

    lm_names = sorted(spec.landmarks_3d.keys())
    pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])

    for row, proj in enumerate(projs):
        key  = proj.proj_key
        res  = per_proj[key]
        best_x = np.array(res['best_pose_delta'])

        xray = proj.image_raw
        xray_small = cv2.resize(xray, (size, size), interpolation=cv2.INTER_AREA)
        drr_init, drr_final = render_drr(drr_gen, proj, best_x, size)

        xray_n = (xray_small - xray_small.min()) / (xray_small.max() - xray_small.min() + 1e-8)
        drr_i_n = (drr_init  - drr_init.min())  / (drr_init.max()  - drr_init.min()  + 1e-8)
        drr_f_n = (drr_final - drr_final.min()) / (drr_final.max() - drr_final.min() + 1e-8)

        # Overlay: X-ray in red channel, DRR@final in green
        overlay = np.zeros((size, size, 3), dtype=np.float32)
        overlay[:, :, 0] = xray_n
        overlay[:, :, 1] = drr_f_n

        for col_idx, img in enumerate([xray_n, drr_i_n, drr_f_n, overlay]):
            ax = axes[row, col_idx]
            ax.imshow(img if col_idx == 3 else img, cmap=None if col_idx == 3 else 'gray',
                      vmin=0, vmax=1)
            ax.axis('off')

        pde_i = res['initial_pde_mm']
        pde_f = res['final_pde_mm']
        status = '✓' if res['success'] else '✗'
        axes[row, 0].set_ylabel(
            f"Frame {key}\n{status} EPnP={pde_i:.1f}mm → {pde_f:.1f}mm",
            fontsize=8, rotation=0, labelpad=70, va='center')

    plt.tight_layout(rect=[0.12, 0, 1, 0.95])
    out = FIGS / 'ramulamma_testing_overview.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Figure 2: PDE bar chart ──────────────────────────────────────────────────

def make_summary_chart(results_data):
    per_proj = results_data['ramulamma_test']['per_projection']
    keys     = list(per_proj.keys())
    init_pde = [per_proj[k]['initial_pde_mm'] for k in keys]
    final_pde = [per_proj[k]['final_pde_mm']  for k in keys]
    go_init  = [per_proj[k]['initial_go']     for k in keys]
    go_final = [per_proj[k]['final_go']        for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Ramulamma Testing — Registration Metrics per Frame',
                 fontsize=13, fontweight='bold')

    x = np.arange(len(keys))
    w = 0.35
    ax1.bar(x - w/2, init_pde,  w, label='EPnP init', color='#5B9BD5', alpha=0.9)
    ax1.bar(x + w/2, final_pde, w, label='Final',      color='#70AD47', alpha=0.9)
    ax1.axhline(5.0,  color='green',  linestyle='--', linewidth=1.2, label='5mm clinical target')
    ax1.axhline(15.0, color='orange', linestyle='--', linewidth=1.2, label='15mm success threshold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Frame\n{k}' for k in keys], fontsize=8)
    ax1.set_ylabel('PDE (mm)')
    ax1.set_title('Point Distance Error')
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, max(max(final_pde) * 1.3, 16))

    for i, (vi, vf) in enumerate(zip(init_pde, final_pde)):
        ax1.text(i - w/2, vi + 0.3, f'{vi:.1f}', ha='center', va='bottom', fontsize=7)
        ax1.text(i + w/2, vf + 0.3, f'{vf:.1f}', ha='center', va='bottom', fontsize=7)

    ax2.bar(x - w/2, go_init,  w, label='EPnP init', color='#5B9BD5', alpha=0.9)
    ax2.bar(x + w/2, go_final, w, label='Final',      color='#70AD47', alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Frame\n{k}' for k in keys], fontsize=8)
    ax2.set_ylabel('GO cost (lower = better alignment)')
    ax2.set_title('Gradient Orientation Cost')
    ax2.legend(fontsize=8)

    # Summary box
    mean_i = np.mean(init_pde)
    mean_f = np.mean(final_pde)
    success_n = sum(1 for k in keys if per_proj[k]['success'])
    txt = (f'n={len(keys)} frames\n'
           f'EPnP init PDE: {mean_i:.2f} mm\n'
           f'Final PDE:     {mean_f:.2f} mm\n'
           f'Success (PDE<15mm): {success_n}/{len(keys)} = {success_n/len(keys)*100:.0f}%')
    ax1.text(0.98, 0.97, txt, transform=ax1.transAxes,
             fontsize=8, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    out = FIGS / 'ramulamma_testing_summary.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Figure 3: Landmark reprojection accuracy ─────────────────────────────────

def make_landmarks_figure(spec, results_data):
    per_proj = results_data['ramulamma_test']['per_projection']
    projs    = spec.projections
    n        = len(projs)

    lm_names = sorted(spec.landmarks_3d.keys())
    pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
    colors   = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    fig.suptitle('Ramulamma Testing — Landmark Reprojection\n'
                 'Annotated positions (●) vs Pipeline projections (✕)',
                 fontsize=13, fontweight='bold')

    for idx, proj in enumerate(projs):
        ax  = axes[idx]
        key = proj.proj_key
        res = per_proj[key]
        best_x = np.array(res['best_pose_delta'])

        xray = proj.image_raw
        ax.imshow(xray, cmap='gray', vmin=0, vmax=1, aspect='auto')

        R_f, t_f = perturb_extrinsic(proj.R_proj, proj.t_proj, best_x[:3], best_x[3:])
        tmp = RamuTestProjection.__new__(RamuTestProjection)
        tmp.R_proj = R_f
        tmp.t_proj = t_f
        uv_final = tmp.project(pts3d)

        for i, (name, col) in enumerate(zip(lm_names, colors)):
            if name not in proj.gt_landmarks_2d:
                continue
            gt = proj.gt_landmarks_2d[name]
            ax.plot(gt[0], gt[1], 'o', color=col, ms=8, markeredgecolor='white', mew=1.5)

            x_f, y_f = uv_final[i, 0], uv_final[i, 1]
            if 0 <= x_f <= xray.shape[1] - 1 and 0 <= y_f <= xray.shape[0] - 1:
                ax.plot(x_f, y_f, 'x', color=col, ms=9, mew=2.5)

        ax.set_title(f'Frame {key} | PDE: {res["initial_pde_mm"]:.1f}→{res["final_pde_mm"]:.1f}mm',
                     fontsize=8)
        ax.axis('off')

    # Hide unused axes
    for idx in range(len(projs), len(axes)):
        axes[idx].axis('off')

    # Legend
    used_names = [name for name in lm_names if name in projs[0].gt_landmarks_2d]
    used_colors = [colors[lm_names.index(name)] for name in used_names]
    legend_elements = [
        mpatches.Patch(color=used_colors[i], label=used_names[i]) for i in range(len(used_names))
    ] + [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', ms=8, label='Annotated GT'),
        plt.Line2D([0], [0], marker='x', color='gray', ms=8, mew=2, label='Pipeline projection'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=min(6, len(legend_elements)), fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = FIGS / 'ramulamma_testing_landmarks.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('[Visualise] Loading results ...')
    results_data = load_results()
    per_proj = results_data['ramulamma_test']['per_projection']

    print('[Visualise] Loading specimen ...')
    loader = RamuTestLoader()
    spec   = loader.load(verbose=False)
    drr_gen = DeepFluoroDRR(spec, hu_threshold=150.0)

    print('[Visualise] Figure 1: overview ...')
    make_overview(spec, drr_gen, results_data)

    print('[Visualise] Figure 2: PDE/GO summary ...')
    make_summary_chart(results_data)

    print('[Visualise] Figure 3: landmark reprojection ...')
    make_landmarks_figure(spec, results_data)

    # Print summary to console
    keys = list(per_proj.keys())
    print(f'\n{"="*55}')
    print(f'RAMULAMMA TESTING  —  Final Results')
    print(f'{"="*55}')
    print(f'{"Frame":<10}  {"EPnP PDE":>10}  {"Final PDE":>10}  {"Status":>8}')
    print(f'{"-"*55}')
    for k in keys:
        r = per_proj[k]
        s = 'SUCCESS' if r['success'] else 'FAIL'
        print(f'{k:<10}  {r["initial_pde_mm"]:>9.2f}mm  {r["final_pde_mm"]:>9.2f}mm  {s:>8}')
    mean_i = np.mean([per_proj[k]['initial_pde_mm'] for k in keys])
    mean_f = np.mean([per_proj[k]['final_pde_mm']   for k in keys])
    sn     = sum(1 for k in keys if per_proj[k]['success'])
    print(f'{"-"*55}')
    print(f'{"Mean":<10}  {mean_i:>9.2f}mm  {mean_f:>9.2f}mm  {sn}/{len(keys)}={sn/len(keys)*100:.0f}%')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
