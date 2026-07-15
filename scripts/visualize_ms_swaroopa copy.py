#!/usr/bin/env python3
"""
visualize_ms_swaroopa.py — msLevelCheck Registration Visualisation (DiffDRR)
=============================================================================
Renders a multi-panel figure for each frame in swaroopa_ms_diffdrr_results.json.
DRRs are produced with **diffdrr** (Siddon ray-caster) using the *same*
cylindrical-mask logic that the registration used, so each panel shows
exactly what the optimiser was seeing.

    Row 1 (per frame):
    [Real X-ray]  [EPnP init]  [S1-Phase1]  [S1-Phase2]  [S1-Phase3]
        (all rendered from the Stage-1 masked full-spine CT)

  Row 2 (per frame):
    [Best S2 group — masked to its 3-vertebra cylinder]
    [S3-L1 — masked]  [S3-L2]  [S3-L3]  [S3-L4]  [S3-L5]

Usage:
    python scripts/visualize_ms_swaroopa.py
    python scripts/visualize_ms_swaroopa.py --results results/swaroopa_ms_diffdrr_results.json
"""

import argparse
import copy
import json
import sys
import tempfile
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

from swaroopa_loader import (
    SwaroLoader,
    project_world_swaro,
    SWARO_PIX_MM,
    SWARO_IMG_SIZE,
)
from deepfluoro_loader import DeepFluoroSpecimen

from run_swaroopa_diffdrr import (
    build_subject,
    DiffDRRGenerator,
    _X0_MM,
    _Y0_MM,
)

from run_ms_swaroopa import (
    compute_spine_axis,
    build_stage_groups,
    CYLINDER_RADIUS_MM,
)
from run_swaroopa_diffdrr import build_subject_masked

RENDER_SIZE = 256
OUT_DIR     = Path('results/figures')


# ---------------------------------------------------------------------------
# Build DiffDRRGenerator dict
# ---------------------------------------------------------------------------

def _build_drr_generators(spec, lm_names, spine_axis, device,
                           cylinder_r_mm=CYLINDER_RADIUS_MM,
                           hu_min=0.0, hu_max=1500.0, z_pad_mm=30.0):
    """
    Returns dict: 's1' + one key per S2 group + one per S3 vertebra.

    Masking mirrors build_subject_masked() in run_swaroopa_diffdrr.py:
      - XY-centroid cylinder of radius cylinder_r_mm around the group landmarks
      - Z-slab from group z_min−z_pad_mm to z_max+z_pad_mm
      - Outside voxels set to −1000 HU (air)
      - Inside voxels clipped to [hu_min, hu_max]
    """
    import SimpleITK as sitk
    from diffdrr.data import read as diffdrr_read

    def _subject_from_group(lm_group):
        lm_pts = np.array([spec.landmarks_3d[lm] for lm in lm_group])
        cx, cy = lm_pts[:, 0].mean(), lm_pts[:, 1].mean()
        z_lo   = lm_pts[:, 2].min() - z_pad_mm
        z_hi   = lm_pts[:, 2].max() + z_pad_mm

        nz, ny, nx = spec.ct_volume.shape
        ox, oy, oz = spec.ct_origin
        sx, sy, sz = spec.ct_spacing
        xi = ox + np.arange(nx) * sx
        yi = oy + np.arange(ny) * sy
        zi = oz + np.arange(nz) * sz

        XX, YY = np.meshgrid(xi, yi, indexing='ij')
        dist2  = (XX - cx) ** 2 + (YY - cy) ** 2
        in_xy  = (dist2 <= cylinder_r_mm ** 2).T               # (nz_Y, nx) → transposed to (ny, nx)
        in_z   = (zi >= z_lo) & (zi <= z_hi)                   # (nz,)
        mask   = in_xy[np.newaxis, :, :] & in_z[:, np.newaxis, np.newaxis]   # (nz, ny, nx)

        ct_m = np.where(mask, np.clip(spec.ct_volume, hu_min, hu_max), -1000).astype(np.int16)

        pct = mask.sum() / mask.size * 100
        print(f'      mask: r={cylinder_r_mm:.0f}mm  z=[{z_lo:.0f},{z_hi:.0f}]mm  '
              f'HU=[{hu_min:.0f},{hu_max:.0f}]  {pct:.1f}% voxels', flush=True)

        sitk_img = sitk.GetImageFromArray(ct_m)
        sitk_img.SetOrigin([float(v) for v in spec.ct_origin])
        sitk_img.SetSpacing([float(v) for v in spec.ct_spacing])
        sitk_img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

        with tempfile.NamedTemporaryFile(suffix='.nrrd', delete=False) as f:
            tmp = f.name
        sitk.WriteImage(sitk_img, tmp)
        subj = diffdrr_read(tmp, orientation=None, center_volume=False,
                            bone_attenuation_multiplier=1.0)
        os.unlink(tmp)
        return subj

    gens = {}

    print('    Building DiffDRR: Stage-1 full-spine mask …', flush=True)
    gens['s1'] = DiffDRRGenerator(
        _subject_from_group(lm_names), device, ct_origin_lps=spec.ct_origin)

    s2_groups = build_stage_groups(lm_names, group_size=3)
    s3_verts  = [[lm] for lm in lm_names]

    for group in s2_groups + s3_verts:
        key = '+'.join(group) if len(group) > 1 else group[0]
        print(f'    Building DiffDRR: {key} …', flush=True)
        gens[key] = DiffDRRGenerator(
            _subject_from_group(group), device, ct_origin_lps=spec.ct_origin)

    return gens


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _snap_to_drr(generators, gen_key, snap, sz=RENDER_SIZE):
    gen = generators.get(gen_key, generators['s1'])
    R   = np.array(snap['R'], dtype=np.float64)
    t   = np.array(snap['t'], dtype=np.float64)
    pix = SWARO_PIX_MM * (SWARO_IMG_SIZE / sz)
    return gen.generate_from_extrinsic(R, t, sz, pix)


def _overlay_landmarks(ax, image, pts2d_gt, pts2d_pred=None,
                        title='', go=None, pde=None):
    scale = RENDER_SIZE / SWARO_IMG_SIZE
    ax.imshow(image, cmap='gray', vmin=0, vmax=1, aspect='equal',
              extent=[0, RENDER_SIZE, RENDER_SIZE, 0])
    if pts2d_gt:
        for name, uv in pts2d_gt.items():
            x, y = float(uv[0]) * scale, float(uv[1]) * scale
            ax.plot(x, y, 'g+', ms=8, mew=1.5)
            ax.text(x + 3, y - 3, name, color='#88ff88', fontsize=5.5)
    if pts2d_pred:
        for name, uv in pts2d_pred.items():
            x, y = float(uv[0]) * scale, float(uv[1]) * scale
            ax.plot(x, y, 'r+', ms=8, mew=1.5)
    label_parts = [title]
    if go  is not None: label_parts.append(f'GO={go:.3f}')
    if pde is not None: label_parts.append(f'PDE={pde:.1f}mm')
    ax.set_title('\n'.join(label_parts), fontsize=6.5, pad=2, color='#dddddd')
    ax.axis('off')


def _proj_pts(spec, R, t):
    lm_names = sorted(spec.landmarks_3d.keys())
    pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
    uv       = project_world_swaro(pts3d, R, t)
    return {n: uv[i] for i, n in enumerate(lm_names)}


def _mean_pde(proj_obj, pred_uv):
    errs = []
    for name, uv in pred_uv.items():
        if name in proj_obj.gt_landmarks_2d:
            gt = proj_obj.gt_landmarks_2d[name]
            if 0 < gt[0] < SWARO_IMG_SIZE and 0 < gt[1] < SWARO_IMG_SIZE:
                errs.append(np.linalg.norm(np.array(uv) - np.array(gt)) * SWARO_PIX_MM)
    return float(np.mean(errs)) if errs else float('nan')


# ---------------------------------------------------------------------------
# Per-frame figure
# ---------------------------------------------------------------------------

def render_frame_figure(frame_key, res, spec, generators, proj_obj):
    poses    = res.get('poses', {})
    s1_snaps = poses.get('stage1', {})
    s2_snaps = poses.get('stage2', {})
    s3_snaps = poses.get('stage3', {})
    lm_names = sorted(spec.landmarks_3d.keys())

    gt_uv = {n: proj_obj.gt_landmarks_2d[n]
              for n in lm_names if n in proj_obj.gt_landmarks_2d}

    # Row 1 — Stage-1 masked CT
    row1 = []
    xray = cv2.resize(proj_obj.image_raw, (RENDER_SIZE, RENDER_SIZE),
                      interpolation=cv2.INTER_AREA)
    row1.append(('X-ray', xray, gt_uv, None, None, None))

    for snap_key, label in [('epnp',   'EPnP init'),
                             ('phase1', 'S1-P1\n(64px)'),
                             ('phase2', 'S1-P2\n(180px)'),
                             ('phase3', 'S1-P3\n(256px)')]:
        if snap_key not in s1_snaps:
            continue
        snap = s1_snaps[snap_key]
        drr  = _snap_to_drr(generators, 's1', snap)
        R, t = np.array(snap['R']), np.array(snap['t'])
        pred = _proj_pts(spec, R, t)
        row1.append((label, drr, gt_uv, pred, snap.get('go'), _mean_pde(proj_obj, pred)))

    # Row 2 — masked CT per group/vertebra
    row2 = []

    # Best S2 group
    best_s2_key = min(res.get('stage2_go', {'_':999}).items(), key=lambda x: x[1])[0]
    if best_s2_key and best_s2_key in s2_snaps:
        snap     = s2_snaps[best_s2_key]['phase3']
        drr      = _snap_to_drr(generators, best_s2_key, snap)
        R, t     = np.array(snap['R']), np.array(snap['t'])
        pred     = _proj_pts(spec, R, t)
        grp_lms  = best_s2_key.split('+')
        row2.append((f'S2 best\n{best_s2_key}', drr,
                     {n: gt_uv[n]  for n in grp_lms if n in gt_uv},
                     {n: pred[n]   for n in grp_lms if n in pred},
                     snap.get('go'), _mean_pde(proj_obj, pred)))

    for lm in lm_names:
        if lm not in s3_snaps:
            continue
        snap = s3_snaps[lm].get('phase3', {})
        if not snap:
            continue
        drr  = _snap_to_drr(generators, lm, snap)
        R, t = np.array(snap['R']), np.array(snap['t'])
        uv   = project_world_swaro(spec.landmarks_3d[lm][np.newaxis], R, t)[0]
        pde  = (float(np.linalg.norm(uv - np.array(proj_obj.gt_landmarks_2d[lm])) * SWARO_PIX_MM)
                if lm in proj_obj.gt_landmarks_2d else None)
        row2.append((f'S3 {lm}', drr,
                     {lm: gt_uv[lm]} if lm in gt_uv else {},
                     {lm: uv},
                     snap.get('go', res['stage3_go'].get(lm)), pde))

    # Build figure
    ncols = max(len(row1), len(row2))
    cell  = RENDER_SIZE / 72 + 0.5
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * cell, 2 * cell + 1.0))
    fig.patch.set_facecolor('#111111')
    if ncols == 1:
        axes = axes[:, np.newaxis]

    ok  = res.get('success', False)
    ok  = bool(ok) if not isinstance(ok, str) else ok.upper() not in ('FALSE', 'FAIL', 'NO')
    col = '#44dd88' if ok else '#ee4444'
    fig.suptitle(
        f'{frame_key}  |  GO: {res["initial_go"]:.4f}→{res["final_go"]:.4f}  '
        f'ΔGO={res["go_delta"]:+.4f}  |  '
        f'PDE: {res["initial_pde_mm"]:.1f}→{res["final_pde_mm"]:.1f}mm  '
        f'[{"SUCCESS" if ok else "FAIL"}]',
        fontsize=9, fontweight='bold', color=col, y=1.01,
    )

    def _fill(panels, row_idx):
        for ci in range(ncols):
            ax = axes[row_idx, ci]
            ax.set_facecolor('#1e1e1e')
            if ci < len(panels):
                _overlay_landmarks(ax, *panels[ci][1:])
                ax.set_title('\n'.join(
                    [panels[ci][0]] +
                    ([f'GO={panels[ci][4]:.3f}'] if panels[ci][4] is not None else []) +
                    ([f'PDE={panels[ci][5]:.1f}mm'] if panels[ci][5] is not None else [])
                ), fontsize=6.5, pad=2, color='#dddddd')
            else:
                ax.axis('off')

    # Use the overlay helper directly for cleaner code
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
    fig, axes = plt.subplots(2, 1, figsize=(max(14, n * 1.4), 10))
    fig.patch.set_facecolor('#111111')

    lm_names = sorted(next(iter(all_res.values()))['poses']['stage3'].keys())

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

    ax.set_xticks(x); ax.set_xticklabels(all_keys, rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=8, framealpha=0.3); ax.grid(axis='y')
    for xi, k in enumerate(all_keys):
        ok = all_res[k].get('success', False)
        ok = bool(ok) if not isinstance(ok, str) else ok.upper() not in ('FALSE', 'FAIL', 'NO')
        ax.axvspan(xi - 0.4, xi + 0.4, color='#44dd88' if ok else '#ee4444', alpha=0.06)

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
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f'{v:.0f}', ha='center', va='bottom',
                         fontsize=5.5, color='#cccccc')

    mean_pde_s1 = [all_res[k]['stage1_pde_mm'] for k in all_keys]
    mean_pde_ms = [all_res[k]['final_pde_mm']   for k in all_keys]
    ax2.plot(x, mean_pde_s1, 'w--', linewidth=1.2, alpha=0.5, label='Stage1 mean')
    ax2.plot(x, mean_pde_ms, 'w-',  linewidth=1.8, alpha=0.8, label='MS final mean')
    ax2.set_xticks(x); ax2.set_xticklabels(all_keys, rotation=30, ha='right', fontsize=8)
    ax2.legend(fontsize=8, framealpha=0.3, ncol=len(lm_names) + 2); ax2.grid(axis='y')

    fig.suptitle(
        f'msLevelCheck Summary — {n} frames  |  '
        f'Mean GO: {np.mean(go_epnp):.4f}→{np.mean(go_s3):.4f}  |  '
        f'Mean PDE: {np.nanmean(mean_pde_s1):.1f}→{np.nanmean(mean_pde_ms):.1f}mm',
        fontsize=11, fontweight='bold', y=1.005,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results/swaroopa_ms_diffdrr_results.json')
    p.add_argument('--out_dir', default='results/figures')
    return p.parse_args()


def main():
    import torch

    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results) as f:
        payload = json.load(f)
    ms_cfg = payload.get('ms_config', {}) if isinstance(payload, dict) else {}
    pp = payload.get('per_projection') if isinstance(payload, dict) else None
    if pp is None:
        pp = payload.get('swaroopa', {}).get('per_projection') if isinstance(payload, dict) else None
    pp = pp or payload
    frame_keys = sorted(pp.keys())
    print(f'Loaded {len(frame_keys)} frames: {frame_keys}')

    loader = SwaroLoader()
    spec   = loader.load(frames=frame_keys, verbose=False)
    proj_map   = {p.proj_key: p for p in spec.projections}
    lm_names   = sorted(spec.landmarks_3d.keys())
    spine_axis = compute_spine_axis(spec.landmarks_3d)

    cyl_r = ms_cfg.get('ms_cylinder_radius')
    if cyl_r is None:
        cyl_r = ms_cfg.get('cylinder_radius_mm')
    if cyl_r is None:
        cyl_r = ms_cfg.get('cylinder_radius')
    if cyl_r is None:
        cyl_r = CYLINDER_RADIUS_MM
    hu_min = ms_cfg.get('min_hu', 0.0)
    if hu_min is None:
        hu_min = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Building DiffDRR generators on {device} …')
    generators = _build_drr_generators(
        spec,
        lm_names,
        spine_axis,
        device,
        cylinder_r_mm=float(cyl_r),
        hu_min=float(hu_min),
    )
    print(f'  Ready: {sorted(generators.keys())}')

    for frame_key in frame_keys:
        if frame_key not in pp or frame_key not in proj_map:
            continue
        print(f'  Rendering {frame_key} ...')
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
        print(f'  results/figures/ms_{fk}.png')
    print('  results/figures/ms_summary.png')


if __name__ == '__main__':
    main()
