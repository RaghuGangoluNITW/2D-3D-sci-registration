#!/usr/bin/env python3
"""
visualize_loss_landscape.py — 2D slices of the loss landscape around the EPnP pose
====================================================================================
For each X-ray frame, sweeps the translation component of the search space in a
±range grid (default ±1 mm, configurable) around the EPnP (GT) pose, fixing
rotation at GT.  Produces three 2D heatmaps per frame:
  - X-Y plane  (Δtx vs Δty, Δtz=0)
  - Y-Z plane  (Δty vs Δtz, Δtx=0)
  - Z-X plane  (Δtz vs Δtx, Δty=0)

Output: a PNG grid with 3 columns × N rows (one row per frame).

Usage:
    python scripts/visualize_loss_landscape.py \
        --frames ap_002 ap_006 \
        --loss lncc --range_mm 1.0 --n_pts 25 \
        --output results/figures/loss_landscape.png
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from swaroopa_loader import (
    SwaroLoader,
    SWARO_PIX_MM,
    SWARO_IMG_SIZE,
)
from deepfluoro_loader import perturb_extrinsic
from run_ms_swaroopa import (
    CYLINDER_RADIUS_MM,
    compute_spine_axis,
)
from run_swaroopa_diffdrr import (
    build_subject,
    build_subject_hu_clipped,
    build_subject_masked,
    DiffDRRGenerator,
    DeepFluoroDRRAdapter,
    landmark_crop_bbox,
    _resize_crop_mask,
    _suppress_highlights,
    _pattern_intensity,
    _local_ncc,
    _ngi,
)

matplotlib.rcParams.update({
    'text.color':      '#dddddd',
    'axes.labelcolor': '#dddddd',
    'xtick.color':     '#dddddd',
    'ytick.color':     '#dddddd',
    'axes.edgecolor':  '#555555',
    'figure.facecolor':'#111111',
    'axes.facecolor':  '#1a1a1a',
})

# ---------------------------------------------------------------------------
# DRR generator builder (mirrors run_ms_swaroopa_diffdrr._make_drr_gen)
# ---------------------------------------------------------------------------

def _make_drr_gen(specimen, args, device_str):
    if args.drr_backend == 'deepfluoro':
        return DeepFluoroDRRAdapter(specimen, device=device_str)
    if args.cylinder_radius is not None:
        hu_min = args.min_hu if args.min_hu is not None else 0.0
        subject = build_subject_masked(specimen, cylinder_r_mm=args.cylinder_radius, hu_min=hu_min)
    elif args.min_hu is not None:
        subject = build_subject_hu_clipped(specimen, hu_min=args.min_hu)
    else:
        subject = build_subject(specimen)
    return DiffDRRGenerator(subject, torch.device(device_str), ct_origin_lps=specimen.ct_origin)


# ---------------------------------------------------------------------------
# Loss evaluator
# ---------------------------------------------------------------------------

_DRR_THR  = 0.05
_BODY_THR = 0.05
EVAL_SIZE = 128   # render size for landscape evaluation (balance speed/quality)


def _eval_loss(drr_gen, R_gt, t_gt, proj, loss: str,
               crop_bbox, delta_rot, delta_trans):
    """Evaluate loss at GT + (delta_rot, delta_trans)."""
    R_c, t_c = perturb_extrinsic(R_gt, t_gt, delta_rot, delta_trans)
    pix = SWARO_PIX_MM * (SWARO_IMG_SIZE / EVAL_SIZE)
    drr = drr_gen.generate_from_extrinsic(R_c, t_c, EVAL_SIZE, pix)

    raw  = cv2.resize(proj.image_raw, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_AREA)
    tgt  = 1.0 - raw.astype(np.float32)
    if crop_bbox is not None:
        mask = _resize_crop_mask(crop_bbox, EVAL_SIZE)
    else:
        mask = None

    if loss == 'pi':
        if mask is not None:
            drr = drr * mask; tgt = tgt * mask
        return float(_pattern_intensity(drr, tgt))
    elif loss == 'lncc':
        if mask is not None:
            drr = drr * mask; tgt = tgt * mask
        return float(_local_ncc(drr, tgt))
    elif loss == 'ngi':
        if mask is not None:
            drr = drr * mask; tgt = tgt * mask
        return float(_ngi(tgt, drr))
    else:  # ncc
        m = (drr > _DRR_THR) & (raw > _BODY_THR)
        if mask is not None:
            m &= mask > 0.5
        if m.sum() < 50:
            return 1.0
        d = drr[m] - drr[m].mean()
        x = tgt[m] - tgt[m].mean()
        denom = np.linalg.norm(d) * np.linalg.norm(x)
        return float(1.0 - np.dot(d, x) / denom) if denom > 1e-8 else 1.0


# ---------------------------------------------------------------------------
# Sweep one 2D plane
# ---------------------------------------------------------------------------

def _sweep_plane(drr_gen, R_gt, t_gt, proj, loss, crop_bbox,
                 axis_i, axis_j, axis_labels, grid, n_pts):
    """
    Sweep translation axes axis_i and axis_j over [-range, +range].
    Returns (loss_grid [n_pts x n_pts], axis_labels [2]).
    """
    costs = np.zeros((n_pts, n_pts), dtype=np.float32)
    for ri, vi in enumerate(grid):
        for ci, vj in enumerate(grid):
            dt = np.zeros(3)
            dt[axis_i] = vi
            dt[axis_j] = vj
            costs[ri, ci] = _eval_loss(drr_gen, R_gt, t_gt, proj, loss,
                                       crop_bbox, np.zeros(3), dt)
    return costs


# ---------------------------------------------------------------------------
# Main landscape computation per frame
# ---------------------------------------------------------------------------

def compute_landscape(drr_gen, proj, lm_names, pts3d, args):
    """Returns dict with keys 'xy', 'yz', 'zx', each a (n_pts, n_pts) array."""
    R_gt = proj.R_proj.copy()
    t_gt = proj.t_proj.copy()

    # Build crop bbox if requested
    crop_bbox = None
    if args.landmark_crop:
        gt_uv = {n: proj.gt_landmarks_2d[n] for n in lm_names if n in proj.gt_landmarks_2d}
        if gt_uv:
            pts2d = np.array(list(gt_uv.values()), dtype=np.float32)
            crop_bbox = landmark_crop_bbox(pts2d)

    grid = np.linspace(-args.range_mm, args.range_mm, args.n_pts)

    print(f"  Sweeping XY plane …", flush=True)
    xy = _sweep_plane(drr_gen, R_gt, t_gt, proj, args.loss, crop_bbox,
                      0, 1, ['tx', 'ty'], grid, args.n_pts)
    print(f"  Sweeping YZ plane …", flush=True)
    yz = _sweep_plane(drr_gen, R_gt, t_gt, proj, args.loss, crop_bbox,
                      1, 2, ['ty', 'tz'], grid, args.n_pts)
    print(f"  Sweeping ZX plane …", flush=True)
    zx = _sweep_plane(drr_gen, R_gt, t_gt, proj, args.loss, crop_bbox,
                      2, 0, ['tz', 'tx'], grid, args.n_pts)

    return {'xy': xy, 'yz': yz, 'zx': zx, 'grid': grid}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_surface(ax, costs, grid, xlabel, ylabel, title):
    """3-D surface plot on a 3-D axis."""
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers projection)
    X, Y = np.meshgrid(grid, grid)

    vmin, vmax = costs.min(), costs.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    surf = ax.plot_surface(X, Y, costs,
                           cmap='plasma', norm=norm,
                           linewidth=0, antialiased=True, alpha=0.88)

    # Mark GT centre (0, 0) as a vertical line
    gt_loss = costs[len(grid) // 2, len(grid) // 2]
    ax.plot([0], [0], [gt_loss], 'w+', ms=10, mew=1.5, zorder=5)

    # Mark minimum
    idx   = np.unravel_index(np.argmin(costs), costs.shape)
    mx, my = grid[idx[1]], grid[idx[0]]
    ax.scatter([mx], [my], [costs.min()], color='cyan', s=40, zorder=6,
               label=f'min ({mx:.2f},{my:.2f})')

    ax.set_xlabel(f'{xlabel} (mm)', fontsize=6, labelpad=2)
    ax.set_ylabel(f'{ylabel} (mm)', fontsize=6, labelpad=2)
    ax.set_zlabel('loss', fontsize=6, labelpad=2)
    ax.set_title(title, fontsize=7, pad=4)
    ax.tick_params(labelsize=5)
    ax.set_facecolor('#1a1a1a')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333')
    ax.yaxis.pane.set_edgecolor('#333')
    ax.zaxis.pane.set_edgecolor('#333')
    ax.legend(fontsize=5.5, framealpha=0.2, loc='upper right')
    plt.colorbar(surf, ax=ax, fraction=0.03, pad=0.08, shrink=0.6).ax.tick_params(labelsize=5)


def render_figure(frame_results, out_path, loss_name, range_mm):
    n_rows = len(frame_results)
    fig = plt.figure(figsize=(13, 4.0 * n_rows), facecolor='#111111')
    fig.patch.set_facecolor('#111111')

    planes = [
        ('xy', 'tx', 'ty'),
        ('yz', 'ty', 'tz'),
        ('zx', 'tz', 'tx'),
    ]

    for ri, (frame_key, res) in enumerate(frame_results.items()):
        grid = res['grid']
        for ci, (plane, xl, yl) in enumerate(planes):
            ax = fig.add_subplot(n_rows, 3, ri * 3 + ci + 1, projection='3d')
            costs = res[plane]
            _plot_surface(ax, costs, grid, xl, yl,
                          f'{frame_key}  |  {plane.upper()} plane  ({loss_name})')

    fig.suptitle(
        f'Translation loss landscape around GT/EPnP pose  —  {loss_name.upper()}  ±{range_mm:.1f} mm',
        fontsize=10, fontweight='bold', y=1.005,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'\nSaved: {out_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Visualise the 2D translation loss landscape around the GT/EPnP pose.')

    # Frame selection (mirrors run_ms_swaroopa_diffdrr.py)
    p.add_argument('--frames', nargs='+', type=str, default=None,
                   help='Projection keys to process (e.g. ap_002 lat_000). '
                        'Defaults to all available frames.')

    # CT / DRR settings (mirrors run_ms_swaroopa_diffdrr.py)
    p.add_argument('--cylinder_radius', type=float, default=None)
    p.add_argument('--min_hu', type=float, default=None)
    p.add_argument('--drr_backend', default='diffdrr', choices=['diffdrr', 'deepfluoro'])
    p.add_argument('--suppress_highlights', action='store_true')

    # Loss
    p.add_argument('--loss', default='ncc', choices=['ncc', 'pi', 'lncc', 'ngi'])
    p.add_argument('--landmark_crop', action='store_true',
                   help='Restrict loss to landmark bounding box (AP pad 100/150, LAT 150/100)')

    # Landscape sweep settings
    p.add_argument('--range_mm', type=float, default=1.0,
                   help='Half-range for translation sweep in mm (default 1.0)')
    p.add_argument('--n_pts', type=int, default=25,
                   help='Number of sample points per axis (default 25)')

    # Output
    p.add_argument('--output', default='results/figures/loss_landscape.png')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device_str}')
    print(f'Loss: {args.loss}  |  range: ±{args.range_mm} mm  |  grid: {args.n_pts}×{args.n_pts}')

    # Load specimen
    loader = SwaroLoader()
    frame_keys = args.frames
    spec = loader.load(frames=frame_keys, verbose=False)
    proj_map = {p.proj_key: p for p in spec.projections}

    if frame_keys is None:
        frame_keys = sorted(proj_map.keys())
    else:
        frame_keys = [k for k in frame_keys if k in proj_map]

    if not frame_keys:
        print('No valid frames found.'); return

    print(f'Frames: {frame_keys}')
    lm_names = sorted(spec.landmarks_3d.keys())
    pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])

    # Build DRR generator (shared across all frames)
    print('Building DRR generator …')
    drr_gen = _make_drr_gen(spec, args, device_str)

    # Compute landscape per frame
    frame_results = {}
    for fk in frame_keys:
        proj = proj_map[fk]
        print(f'\n[{fk}]  GT PDE should be ~0 mm at centre of landscape')
        t0 = time.time()
        res = compute_landscape(drr_gen, proj, lm_names, pts3d, args)
        elapsed = time.time() - t0
        gt_loss = _eval_loss(drr_gen, proj.R_proj, proj.t_proj, proj, args.loss,
                             None, np.zeros(3), np.zeros(3))
        print(f'  GT loss (delta=0): {gt_loss:.4f}  |  min(XY)={res["xy"].min():.4f}  '
              f'elapsed={elapsed:.1f}s')
        frame_results[fk] = res

    # Render
    render_figure(frame_results, Path(args.output), args.loss, args.range_mm)


if __name__ == '__main__':
    main()
