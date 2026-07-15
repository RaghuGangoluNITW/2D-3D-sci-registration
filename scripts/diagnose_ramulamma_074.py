#!/usr/bin/env python3
"""
Create a clean diagnostic figure for Ramulamma frame 074 only.
Shows:
  1) X-ray
  2) DRR at accepted pose
  3) Overlay
  4) Annotated GT landmarks vs projected landmarks (L2-L5 only)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src'))

from ramulamma_testing_loader import RamuTestLoader, RamuTestProjection, RAMU_TEST_IMG_SIZE, RAMU_TEST_PIX_MM
from deepfluoro_loader import perturb_extrinsic
from deepfluoro_drr import DeepFluoroDRR

OUT = BASE / 'results' / 'figures' / 'ramulamma_074_diagnostic.png'
RESULTS = BASE / 'results' / 'ramulamma_testing_results.json'


def main():
    with open(RESULTS) as f:
        results = json.load(f)

    loader = RamuTestLoader()
    spec = loader.load(frames=['074'], verbose=False)
    proj = spec.projections[0]
    res = results['ramulamma_test']['per_projection']['074']

    best_x = np.array(res['best_pose_delta'], dtype=np.float64)
    R_final, t_final = perturb_extrinsic(
        proj.R_proj, proj.t_proj,
        best_x[:3], best_x[3:]
    )

    drr_gen = DeepFluoroDRR(spec, hu_threshold=150.0)
    render_size = 256
    render_pix = RAMU_TEST_PIX_MM * (RAMU_TEST_IMG_SIZE / render_size)
    drr = drr_gen.generate_from_extrinsic(R_final, t_final, render_size, render_pix, 120)

    xray = proj.image_raw
    xray_small = xray

    drr_resized = np.array(
        plt.matplotlib.image.imsave if False else drr
    )
    # resize with matplotlib-free numpy path via nearest-like indexing
    import cv2
    drr_resized = cv2.resize(drr, (xray.shape[1], xray.shape[0]), interpolation=cv2.INTER_LINEAR)

    xray_n = (xray_small - xray_small.min()) / (xray_small.max() - xray_small.min() + 1e-8)
    drr_n = (drr_resized - drr_resized.min()) / (drr_resized.max() - drr_resized.min() + 1e-8)

    overlay = np.zeros((xray.shape[0], xray.shape[1], 3), dtype=np.float32)
    overlay[:, :, 0] = xray_n
    overlay[:, :, 1] = drr_n

    lm_names = ['L2', 'L3', 'L4', 'L5']
    pts3d = np.array([spec.landmarks_3d[name] for name in sorted(spec.landmarks_3d.keys())])
    all_names = sorted(spec.landmarks_3d.keys())

    tmp = RamuTestProjection.__new__(RamuTestProjection)
    tmp.R_proj = R_final
    tmp.t_proj = t_final
    uv = tmp.project(pts3d)
    uv_map = {name: uv[i] for i, name in enumerate(all_names)}

    colors = {
        'L2': '#f39c12',
        'L3': '#2ecc71',
        'L4': '#3498db',
        'L5': '#e74c3c',
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(
        f'Ramulamma Frame 074 — Registration Diagnostic\n'
        f'Accepted pose PDE: {res["final_pde_mm"]:.2f} mm',
        fontsize=14,
        fontweight='bold'
    )

    axes[0, 0].imshow(xray_n, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('X-ray (input)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(drr_n, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('DRR at accepted pose')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(overlay, vmin=0, vmax=1)
    axes[1, 0].set_title('Overlay: X-ray (red) + DRR (green)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(xray_n, cmap='gray', vmin=0, vmax=1)
    for name in lm_names:
        gt = proj.gt_landmarks_2d[name]
        pr = uv_map[name]
        axes[1, 1].plot(gt[0], gt[1], 'o', color=colors[name], ms=8, markeredgecolor='white', mew=1.5)
        axes[1, 1].plot(pr[0], pr[1], 'x', color=colors[name], ms=10, mew=2.5)
        axes[1, 1].text(gt[0] + 10, gt[1] - 10, name, color=colors[name], fontsize=10, weight='bold')
    axes[1, 1].set_title('Landmarks used: GT dots vs projected crosses')
    axes[1, 1].axis('off')

    handles = []
    from matplotlib.lines import Line2D
    for name in lm_names:
        handles.append(Line2D([0], [0], color=colors[name], marker='o', linestyle='None', label=f'{name} GT'))
    handles.append(Line2D([0], [0], color='black', marker='x', linestyle='None', label='Projected'))
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
