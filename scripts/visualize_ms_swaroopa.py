#!/usr/bin/env python3
"""
visualize_ms_swaroopa.py — 5-column AP/LAT msLevelCheck visualization
=====================================================================
Produces four output files:
  ms_ap_5col.png   — all AP frames, one row each, 5 columns
  ms_lat_5col.png  — all LAT frames, one row each, 5 columns
  ms_ap_summary.png
  ms_lat_summary.png

Column layout per row:
  1  Inverted X-ray (1 − raw)
  2  EPnP pose DRR              (Stage-1 full-spine masked CT)
  3  Starting perturbed pose    (Stage-1 full-spine masked CT)
  4  Stage-1 best pose          (Stage-1 full-spine masked CT)
  5  Stage-3 vertebra composite (each vertebra rendered with its own
       cylinder whose height = avg inter-vertebral distance, then tinted
       with a unique colour and blended into one RGB image)

Usage:
    python scripts/visualize_ms_swaroopa.py
    python scripts/visualize_ms_swaroopa.py \\
        --results results/swaroopa_ms_diffdrr_results.json
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from swaroopa_loader import (
    SwaroLoader,
    project_world_swaro,
    SWARO_PIX_MM,
    SWARO_IMG_SIZE,
)
from run_ms_swaroopa import CYLINDER_RADIUS_MM
from run_swaroopa_diffdrr import DiffDRRGenerator, landmark_crop_bbox

RENDER_SIZE = 256
PIX_MM = SWARO_PIX_MM * (SWARO_IMG_SIZE / RENDER_SIZE)

# Tint palette for Stage-3 vertebrae (R, G, B ∈ [0,1])
_VERT_COLORS = np.array([
    [1.00, 0.35, 0.35],   # red-ish
    [1.00, 0.68, 0.25],   # orange
    [0.40, 0.95, 0.45],   # green
    [0.35, 0.78, 1.00],   # cyan-blue
    [0.88, 0.55, 1.00],   # purple
], dtype=np.float32)

COL_TITLES = [
    'Inverted X-ray',
    'EPnP pose',
    'Perturbed start',
    'Stage-1 best',
    'Stage-3 combined',
]

matplotlib.rcParams.update({
    'text.color':       '#dddddd',
    'axes.labelcolor':  '#dddddd',
    'xtick.color':      '#dddddd',
    'ytick.color':      '#dddddd',
    'axes.edgecolor':   '#444',
    'axes.facecolor':   '#1e1e1e',
    'figure.facecolor': '#111111',
})


# ---------------------------------------------------------------------------
# CT masking helpers
# ---------------------------------------------------------------------------

def _subject_from_group(spec, lm_group, cylinder_r_mm, hu_min, hu_max, z_pad_mm):
    """Build a TorchIO Subject with a cylinder-and-slab CT mask."""
    import SimpleITK as sitk
    from diffdrr.data import read as diffdrr_read

    lm_pts = np.array([spec.landmarks_3d[lm] for lm in lm_group], dtype=np.float64)
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
    in_xy  = (dist2 <= cylinder_r_mm ** 2).T                         # (ny, nx)
    in_z   = (zi >= z_lo) & (zi <= z_hi)                             # (nz,)
    mask   = in_xy[np.newaxis, :, :] & in_z[:, np.newaxis, np.newaxis]  # (nz, ny, nx)

    ct_m = np.where(mask, np.clip(spec.ct_volume, hu_min, hu_max), -1000).astype(np.int16)
    pct  = mask.mean() * 100
    print(f'      [{"+".join(lm_group)}] r={cylinder_r_mm:.0f}mm '
          f'z=[{z_lo:.0f},{z_hi:.0f}]mm  HU=[{hu_min:.0f},{hu_max:.0f}]  {pct:.1f}%',
          flush=True)

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


def _avg_ivd(spec, lm_names):
    """Average inter-vertebral distance in mm (along z-axis centroids)."""
    pts = np.array([spec.landmarks_3d[n] for n in lm_names], dtype=np.float64)
    if len(pts) < 2:
        return 60.0
    return float(np.mean(np.abs(pts[1:, 2] - pts[:-1, 2])))


def _build_generators(spec, lm_names, device, cylinder_r_mm, hu_min, hu_max=1500.0):
    """Return dict with 's1' (full spine) and one key per vertebra."""
    ivd     = _avg_ivd(spec, lm_names)
    z_pad_s3 = max(ivd / 2.0, 5.0)   # half IVD above/below → full IVD height
    print(f'  Avg IVD={ivd:.1f}mm → Stage-3 cylinder half-height={z_pad_s3:.1f}mm')

    gens = {}

    print('  Building S1 generator (full spine) …', flush=True)
    gens['s1'] = DiffDRRGenerator(
        _subject_from_group(spec, lm_names,
                             cylinder_r_mm=cylinder_r_mm,
                             hu_min=hu_min, hu_max=hu_max, z_pad_mm=30.0),
        device, ct_origin_lps=spec.ct_origin)

    for lm in lm_names:
        print(f'  Building S3 generator [{lm}] …', flush=True)
        gens[lm] = DiffDRRGenerator(
            _subject_from_group(spec, [lm],
                                 cylinder_r_mm=cylinder_r_mm,
                                 hu_min=hu_min, hu_max=hu_max, z_pad_mm=z_pad_s3),
            device, ct_origin_lps=spec.ct_origin)

    return gens


# ---------------------------------------------------------------------------
# DRR rendering helpers
# ---------------------------------------------------------------------------

def _render(gen, snap, sz=RENDER_SIZE):
    """Render a single DRR from a pose snapshot dict {R, t}."""
    if snap is None:
        return np.zeros((sz, sz), dtype=np.float32)
    R   = np.array(snap['R'], dtype=np.float64)
    t   = np.array(snap['t'], dtype=np.float64)
    pix = SWARO_PIX_MM * (SWARO_IMG_SIZE / sz)
    return gen.generate_from_extrinsic(R, t, sz, pix)


def _snap(s1_snaps, keys):
    """Return the first matching snapshot from stage-1 snaps."""
    for k in keys:
        if k in s1_snaps:
            return s1_snaps[k]
    return None


def _stage3_composite(generators, s3_snaps, lm_names, sz=RENDER_SIZE):
    """
    Render each vertebra at its Stage-3 best pose using its own cylinder-masked
    CT, tint with a unique colour, then blend into a single RGB image.
    """
    rgb = np.zeros((sz, sz, 3), dtype=np.float32)
    rendered = []
    for i, lm in enumerate(lm_names):
        entry = s3_snaps.get(lm, {})
        # prefer phase3, fall back to phase2/phase1/root
        snap = (entry.get('phase3') or entry.get('phase2') or
                entry.get('phase1') or
                (entry if ('R' in entry and 't' in entry) else None))
        if snap is None:
            continue
        gen = generators.get(lm, generators['s1'])
        drr = _render(gen, snap, sz)
        rendered.append(drr)
        c   = _VERT_COLORS[i % len(_VERT_COLORS)]
        rgb += drr[..., None] * c[None, None, :]

    if not rendered:
        return np.zeros((sz, sz, 3), dtype=np.float32)

    # Normalise accumulated colour image
    if rgb.max() > 0:
        rgb = rgb / rgb.max()

    # Mix: 30% mean-grey base + 80% tinted (slight bone-grey retention)
    base = np.mean(np.stack(rendered), axis=0)[..., None].repeat(3, axis=2)
    out  = np.clip(0.30 * base + 0.80 * rgb, 0.0, 1.0)
    return out


def _drr_edge_overlay(xray_inv: np.ndarray, drr, color=(0.25, 0.85, 1.0), edge_alpha=0.45):
    """Blend Canny edges from a DRR (or RGB composite) on top of the X-ray.

    Returns a float32 RGB image in [0, 1].
    """
    # Convert DRR to uint8 grayscale for Canny
    if drr.ndim == 3:
        gray_drr = (np.mean(drr, axis=2) * 255).astype(np.uint8)
    else:
        gray_drr = (np.clip(drr, 0, 1) * 255).astype(np.uint8)

    # Canny edges on DRR
    blurred = cv2.GaussianBlur(gray_drr, (3, 3), 0)
    edges   = cv2.Canny(blurred, threshold1=20, threshold2=60).astype(np.float32) / 255.0

    # Slightly dilate for visibility at 256 px
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges  = cv2.dilate(edges, kernel)

    # Build RGB: xray_inv as gray background
    xray_f = np.clip(xray_inv, 0, 1).astype(np.float32)
    rgb    = np.stack([xray_f, xray_f, xray_f], axis=2)

    # Overlay edges in colour
    c = np.array(color, dtype=np.float32)
    rgb = rgb * (1.0 - edge_alpha * edges[..., None]) + edge_alpha * edges[..., None] * c
    return np.clip(rgb, 0, 1)


def _drr_edge_overlay_s3(xray_inv: np.ndarray, generators, s3_snaps, lm_names,
                         sz=RENDER_SIZE, edge_alpha=0.45):
    """Stage-3 edge overlay: each vertebra rendered individually with its own
    colour, edges accumulated on the X-ray background."""
    xray_f = np.clip(xray_inv, 0, 1).astype(np.float32)
    rgb    = np.stack([xray_f, xray_f, xray_f], axis=2).copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    for i, lm in enumerate(lm_names):
        entry = s3_snaps.get(lm, {})
        snap  = (entry.get('phase3') or entry.get('phase2') or
                 entry.get('phase1') or
                 (entry if ('R' in entry and 't' in entry) else None))
        if snap is None:
            continue
        gen = generators.get(lm, generators['s1'])
        drr = _render(gen, snap, sz)
        gray = (np.clip(drr, 0, 1) * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges   = cv2.Canny(blurred, threshold1=20, threshold2=60).astype(np.float32) / 255.0
        edges   = cv2.dilate(edges, kernel)
        c = _VERT_COLORS[i % len(_VERT_COLORS)]
        rgb = rgb * (1.0 - edge_alpha * edges[..., None]) + edge_alpha * edges[..., None] * c

    return np.clip(rgb, 0, 1)


def _proj_pts(spec, snap):
    if snap is None:
        return {}
    R = np.array(snap['R'], dtype=np.float64)
    t = np.array(snap['t'], dtype=np.float64)
    lm_names = sorted(spec.landmarks_3d.keys())
    pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
    uv       = project_world_swaro(pts3d, R, t)
    return {n: uv[i] for i, n in enumerate(lm_names)}


def _proj_pts_s3(spec, s3_snaps):
    """Project each vertebra centroid using its own per-vertebra Stage-3 pose."""
    result = {}
    for lm_name, snaps in s3_snaps.items():
        if lm_name not in spec.landmarks_3d:
            continue
        snap = snaps.get('phase3') or snaps.get('phase2') or snaps.get('phase1')
        if snap is None:
            continue
        R = np.array(snap['R'], dtype=np.float64)
        t = np.array(snap['t'], dtype=np.float64)
        pt = np.array([spec.landmarks_3d[lm_name]])
        uv = project_world_swaro(pt, R, t)
        result[lm_name] = uv[0]
    return result


def _mean_pde_from_snap(proj_obj, snap):
    if snap is None:
        return float('nan')
    R = np.array(snap['R'], dtype=np.float64)
    t = np.array(snap['t'], dtype=np.float64)
    errs = []
    lm_names = sorted(proj_obj.gt_landmarks_2d.keys())
    for n in lm_names:
        gt = proj_obj.gt_landmarks_2d[n]
        if 0 < gt[0] < SWARO_IMG_SIZE and 0 < gt[1] < SWARO_IMG_SIZE:
            pts3d = np.array(proj_obj.spec_landmarks_3d[n])[None] if hasattr(proj_obj, 'spec_landmarks_3d') else None
            if pts3d is not None:
                uv = project_world_swaro(pts3d, R, t)[0]
                errs.append(np.linalg.norm(uv - np.array(gt)) * SWARO_PIX_MM)
    return float(np.mean(errs)) if errs else float('nan')


def _overlay_lm(ax, image, lm_gt_uv, lm_pred_uv=None, is_rgb=False):
    scale = RENDER_SIZE / SWARO_IMG_SIZE
    if is_rgb:
        ax.imshow(np.clip(image, 0, 1),
                  extent=[0, RENDER_SIZE, RENDER_SIZE, 0], aspect='equal')
    else:
        ax.imshow(image, cmap='gray', vmin=0, vmax=1,
                  extent=[0, RENDER_SIZE, RENDER_SIZE, 0], aspect='equal')
    for name, uv in lm_gt_uv.items():
        x, y = float(uv[0]) * scale, float(uv[1]) * scale
        ax.plot(x, y, 'g+', ms=8, mew=1.5)
        ax.text(x + 3, y - 3, name, color='#88ff88', fontsize=5.5)
    if lm_pred_uv:
        for name, uv in lm_pred_uv.items():
            x, y = float(uv[0]) * scale, float(uv[1]) * scale
            ax.plot(x, y, 'rx', ms=6, mew=1.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor('#1e1e1e')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')


# ---------------------------------------------------------------------------
# Build one row of 5 cells for a frame
# ---------------------------------------------------------------------------

def _build_row_images(res, proj_obj, spec, generators):
    """Return list of 5 items: (img, is_rgb, title_lines, pred_uv_or_None)."""
    lm_names   = sorted(spec.landmarks_3d.keys())
    gt_uv      = {n: proj_obj.gt_landmarks_2d[n]
                  for n in lm_names if n in proj_obj.gt_landmarks_2d}
    s1_snaps   = res.get('poses', {}).get('stage1', {})
    s3_snaps   = res.get('poses', {}).get('stage3', {})

    raw   = cv2.resize(proj_obj.image_raw, (RENDER_SIZE, RENDER_SIZE), cv2.INTER_AREA)
    inv   = 1.0 - raw.astype(np.float32)

    snap_epnp  = _snap(s1_snaps, ['epnp'])
    snap_start = _snap(s1_snaps, ['perturbed', 'init'])
    snap_s1    = _snap(s1_snaps, ['phase3', 'phase2', 'phase1'])

    gen_s1 = generators['s1']

    # Per-vertebra projected centroids from final Stage-3 poses
    s3_proj = _proj_pts_s3(spec, s3_snaps)

    # Pre-render DRRs for edge overlay
    drr_epnp  = _render(gen_s1, snap_epnp)
    drr_start = _render(gen_s1, snap_start)
    drr_s1    = _render(gen_s1, snap_s1)

    def _pde_label(snap):
        if snap is None: return ''
        go  = snap.get('go')
        return (f'GO={go:.3f}' if go is not None else '')

    # Edge-overlay colours per column
    _COL_COLORS = [
        (0.25, 0.85, 1.00),   # EPnP     — cyan
        (1.00, 0.75, 0.20),   # perturbed — amber
        (0.40, 1.00, 0.55),   # Stage-1  — green
    ]

    rows = [
        (inv,                                                    False, ['Inverted X-ray'],          s3_proj),
        (_drr_edge_overlay(inv, drr_epnp,  _COL_COLORS[0]),      True,  ['EPnP pose',
                                                                           _pde_label(snap_epnp)],   _proj_pts(spec, snap_epnp)),
        (_drr_edge_overlay(inv, drr_start, _COL_COLORS[1]),      True,  ['Perturbed start',
                                                                           _pde_label(snap_start)],  _proj_pts(spec, snap_start)),
        (_drr_edge_overlay(inv, drr_s1,    _COL_COLORS[2]),      True,  ['Stage-1 best',
                                                                           _pde_label(snap_s1)],     _proj_pts(spec, snap_s1)),
        (_drr_edge_overlay_s3(inv, generators, s3_snaps, lm_names), True,
         ['Stage-3 combined',
          f'PDE={res.get("final_pde_mm", float("nan")):.1f}mm'], s3_proj),
    ]
    return rows, gt_uv


# ---------------------------------------------------------------------------
# Combined grid figure (all frames of one view in one figure)
# ---------------------------------------------------------------------------

def _render_grid(view_label, frame_keys, pp, proj_map, spec, generators, out_path, landmark_crop=False):
    if not frame_keys:
        print(f'  No {view_label} frames — skipping {out_path.name}')
        return

    nrows = len(frame_keys)
    ncols = 5
    cell  = RENDER_SIZE / 100.0           # inches per cell
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * (cell + 0.3), nrows * (cell + 0.55)),
                              squeeze=False)
    fig.patch.set_facecolor('#111111')

    # Column headers on first row
    for ci, title in enumerate(COL_TITLES):
        axes[0, ci].set_title(title, fontsize=9, color='#e8e8e8', pad=4)

    for ri, key in enumerate(frame_keys):
        res      = pp[key]
        proj_obj = proj_map[key]
        row_imgs, gt_uv = _build_row_images(res, proj_obj, spec, generators)

        ok  = res.get('success', False)
        ok  = ok if isinstance(ok, bool) else str(ok).upper() not in ('FALSE', 'FAIL', 'NO')
        row_color = '#44dd88' if ok else '#ee4444'

        for ci, (img, is_rgb, title_lines, pred_uv) in enumerate(row_imgs):
            ax = axes[ri, ci]
            _overlay_lm(ax, img, gt_uv if ci > 0 else {}, pred_uv, is_rgb=is_rgb)
            # Draw landmark bounding-box on the X-ray panel when requested
            if ci == 0 and landmark_crop:
                lm_pts = np.array(list(gt_uv.values()), dtype=np.float32) if gt_uv else None
                if lm_pts is not None and len(lm_pts) > 0:
                    bx0, by0, bx1, by1 = landmark_crop_bbox(lm_pts)
                    scale = RENDER_SIZE / SWARO_IMG_SIZE
                    from matplotlib.patches import Rectangle
                    rect = Rectangle(
                        (bx0 * scale, by0 * scale),
                        (bx1 - bx0) * scale, (by1 - by0) * scale,
                        linewidth=1.2, edgecolor='#ffdd44', facecolor='none', linestyle='--',
                    )
                    ax.add_patch(rect)
            sub = '\n'.join(t for t in title_lines if t)
            ax.set_title(sub, fontsize=6.5, color='#cccccc', pad=2)

        # Row label on leftmost axis
        pde_i = res.get('initial_pde_mm', float('nan'))
        pde_f = res.get('final_pde_mm', float('nan'))
        go_i  = res.get('initial_go', float('nan'))
        go_f  = res.get('final_go', float('nan'))
        axes[ri, 0].set_ylabel(
            f'{key}\nGO {go_i:.3f}→{go_f:.3f}\nPDE {pde_i:.1f}→{pde_f:.1f}mm',
            fontsize=7, color=row_color, rotation=90, labelpad=8,
        )

    fig.suptitle(
        f'{view_label.upper()} — msLevelCheck 5-column grid  '
        f'({nrows} frames)',
        fontsize=11, color='white', y=1.01,
    )
    plt.tight_layout(rect=[0.04, 0, 1, 0.99])
    fig.savefig(out_path, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def _render_summary(view_label, frame_keys, pp, lm_names, out_path):
    if not frame_keys:
        return

    n = len(frame_keys)
    x = np.arange(n)
    lm_colors = ['#4499ff', '#ff9944', '#44dd88', '#ff4466', '#cc88ff']

    fig, axes = plt.subplots(2, 1, figsize=(max(10, n * 1.4), 10))
    fig.patch.set_facecolor('#111111')

    # ── GO cost progression ──────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('#1e1e1e')
    ax.set_title(f'GO cost — {view_label.upper()}', fontsize=10)
    ax.set_ylabel('GO cost  (lower = better)', fontsize=9)

    go_epnp = [pp[k]['initial_go']  for k in frame_keys]
    go_s1   = [pp[k]['stage1_go']   for k in frame_keys]
    go_s3   = [pp[k]['final_go']    for k in frame_keys]

    for label, vals, color, marker, ls in [
        ('EPnP',   go_epnp, '#aaaaaa', 'o', '--'),
        ('Stage1', go_s1,   '#66aaff', 's', '-'),
        ('Stage3', go_s3,   '#44dd88', 'D', '-'),
    ]:
        ax.plot(x, vals, color=color, marker=marker, linestyle=ls,
                linewidth=1.5, markersize=6, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(frame_keys, rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(axis='y')
    for xi, k in enumerate(frame_keys):
        ok = pp[k].get('success', False)
        ok = ok if isinstance(ok, bool) else str(ok).upper() not in ('FALSE', 'FAIL', 'NO')
        ax.axvspan(xi - 0.4, xi + 0.4, color='#44dd88' if ok else '#ee4444', alpha=0.07)

    # ── Per-landmark PDE bars ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#1e1e1e')
    ax2.set_title(f'Per-landmark PDE (mm) Stage-3 — {view_label.upper()}', fontsize=10)
    ax2.set_ylabel('PDE (mm)', fontsize=9)
    bar_w = 0.14
    for li, lm in enumerate(lm_names):
        pde_vals = [pp[k]['pde_per_lm'].get(lm, float('nan')) for k in frame_keys]
        offs     = x + (li - len(lm_names) / 2 + 0.5) * bar_w
        bars     = ax2.bar(offs, pde_vals, width=bar_w,
                           color=lm_colors[li % len(lm_colors)],
                           label=lm, alpha=0.85, edgecolor='#555')
        for bar, v in zip(bars, pde_vals):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.5, f'{v:.0f}',
                         ha='center', va='bottom', fontsize=5.5, color='#cccccc')

    pde_s1  = [pp[k]['stage1_pde_mm'] for k in frame_keys]
    pde_ms  = [pp[k]['final_pde_mm']  for k in frame_keys]
    ax2.plot(x, pde_s1, 'w--', lw=1.2, alpha=0.5, label='Stage1 mean')
    ax2.plot(x, pde_ms, 'w-',  lw=1.8, alpha=0.8, label='Stage3 mean')
    ax2.set_xticks(x)
    ax2.set_xticklabels(frame_keys, rotation=30, ha='right', fontsize=8)
    ax2.legend(fontsize=8, framealpha=0.3, ncol=len(lm_names) + 2)
    ax2.grid(axis='y')

    fig.suptitle(
        f'msLevelCheck Summary ({view_label.upper()}) — {n} frames  |  '
        f'Mean GO {np.mean(go_epnp):.4f}→{np.mean(go_s3):.4f}  |  '
        f'Mean PDE {np.nanmean(pde_s1):.1f}→{np.nanmean(pde_ms):.1f}mm',
        fontsize=11, fontweight='bold', y=1.005,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results/swaroopa_ms_diffdrr_results.json')
    p.add_argument('--out_dir', default='results/figures')
    p.add_argument('--landmark_crop', action='store_true',
                   help='Draw the GT-landmark bounding box on the X-ray panel')
    return p.parse_args()


def main():
    import torch

    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load results ─────────────────────────────────────────────────────────
    with open(args.results) as f:
        payload = json.load(f)
    ms_cfg = payload.get('ms_config', {}) if isinstance(payload, dict) else {}
    pp = (payload.get('per_projection') or
          (payload.get('swaroopa') or {}).get('per_projection') or
          payload)
    frame_keys = sorted(pp.keys())
    print(f'Loaded {len(frame_keys)} frames: {frame_keys}')

    # ── CT/geometry ──────────────────────────────────────────────────────────
    loader   = SwaroLoader()
    spec     = loader.load(frames=frame_keys, verbose=False)
    proj_map = {p.proj_key: p for p in spec.projections}
    lm_names = sorted(spec.landmarks_3d.keys())

    # resolve cylinder radius with fallback chain
    cyl_r = (ms_cfg.get('ms_cylinder_radius') or
             ms_cfg.get('cylinder_radius_mm') or
             ms_cfg.get('cylinder_radius') or
             CYLINDER_RADIUS_MM)
    hu_min = float(ms_cfg.get('min_hu') or 0.0)

    # ── DiffDRR generators ────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Building DiffDRR generators on {device}  '
          f'(cyl_r={float(cyl_r):.1f}mm  hu_min={hu_min:.0f}) …')
    generators = _build_generators(spec, lm_names, device,
                                    cylinder_r_mm=float(cyl_r), hu_min=hu_min)

    # ── Split AP / LAT ────────────────────────────────────────────────────────
    ap_keys  = [k for k in frame_keys if k.startswith('ap_')  and k in proj_map]
    lat_keys = [k for k in frame_keys if k.startswith('lat_') and k in proj_map]

    # ── 5-column grids ────────────────────────────────────────────────────────
    _render_grid('ap',  ap_keys,  pp, proj_map, spec, generators,
                 out_dir / 'ms_ap_5col.png',  landmark_crop=args.landmark_crop)
    _render_grid('lat', lat_keys, pp, proj_map, spec, generators,
                 out_dir / 'ms_lat_5col.png', landmark_crop=args.landmark_crop)

    # ── Summaries ─────────────────────────────────────────────────────────────
    _render_summary('ap',  ap_keys,  pp, lm_names, out_dir / 'ms_ap_summary.png')
    _render_summary('lat', lat_keys, pp, lm_names, out_dir / 'ms_lat_summary.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
