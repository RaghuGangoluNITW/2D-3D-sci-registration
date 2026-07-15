#!/usr/bin/env python3
"""Diagnostic figure for scene-only Ramulamma registration."""

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src'))

from deepfluoro_drr import DeepFluoroDRR
from deepfluoro_loader import perturb_extrinsic
from ramulamma_scene_loader import RamuSceneLoader, RamuSceneProjection

RESULTS = BASE / 'results' / 'ramulamma_scene_results.json'
OUT = BASE / 'results' / 'figures' / 'ramulamma_scene_diagnostic.png'


def main():
    with open(RESULTS) as f:
        results = json.load(f)['ramulamma_scene']

    loader = RamuSceneLoader()
    spec = loader.load(verbose=False)
    proj = spec.projections[0]

    best_x = np.array(results['best_pose_delta'], dtype=np.float64)
    R_final, t_final = perturb_extrinsic(proj.R_proj, proj.t_proj, best_x[:3], best_x[3:])

    drr_gen = DeepFluoroDRR(spec, hu_threshold=150.0)
    drr = drr_gen.generate_from_extrinsic(R_final, t_final, 256, 0.297656 * (1024 / 256), 120)
    drr = cv2.resize(drr, (proj.image_raw.shape[1], proj.image_raw.shape[0]), interpolation=cv2.INTER_LINEAR)

    xray = proj.image_raw
    xray_n = (xray - xray.min()) / (xray.max() - xray.min() + 1e-8)
    drr_n = (drr - drr.min()) / (drr.max() - drr.min() + 1e-8)
    overlay = np.zeros((xray.shape[0], xray.shape[1], 3), dtype=np.float32)
    overlay[:, :, 0] = xray_n
    overlay[:, :, 1] = drr_n

    all_names = sorted(spec.landmarks_3d.keys())
    pts3d = np.array([spec.landmarks_3d[n] for n in all_names])
    tmp = RamuSceneProjection.__new__(RamuSceneProjection)
    tmp.R_proj = R_final
    tmp.t_proj = t_final
    uv = tmp.project(pts3d)
    uv_map = {name: uv[i] for i, name in enumerate(all_names)}

    used = results['used_landmarks']
    colors = {'L3': '#2ecc71', 'L4': '#3498db', 'L5': '#e74c3c'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Ramulamma Scene-only Registration\nActive volume + active markup from 2026-02-25-Scene.mrml', fontsize=14, fontweight='bold')

    axes[0, 0].imshow(xray_n, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Active scene frame (1.nrrd, z=6)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(drr_n, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('DRR at accepted pose')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(overlay, vmin=0, vmax=1)
    axes[1, 0].set_title('Overlay: X-ray red + DRR green')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(xray_n, cmap='gray', vmin=0, vmax=1)
    for name in used:
        gt = proj.gt_landmarks_2d[name]
        pr = uv_map[name]
        c = colors[name]
        axes[1, 1].plot(gt[0], gt[1], 'o', color=c, ms=8, markeredgecolor='white', mew=1.5)
        axes[1, 1].plot(pr[0], pr[1], 'x', color=c, ms=10, mew=2.5)
        axes[1, 1].text(gt[0] + 10, gt[1] - 10, name, color=c, fontsize=10, weight='bold')
    axes[1, 1].set_title('Used landmarks only: GT dots vs projected crosses')
    axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
