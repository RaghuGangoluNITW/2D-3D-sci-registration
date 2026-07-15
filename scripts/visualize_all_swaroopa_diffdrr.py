#!/usr/bin/env python3
"""
visualize_all_swaroopa_diffdrr.py
==================================
Renders a DiffDRR visualisation for every registered frame in
results/swaroopa_results_all.json.

Layout per page:
  One row per frame (AP frames first, then LAT), 3 columns:
    [Real X-ray + GT lm] | [Initial DRR + proj lm] | [Final DRR + proj lm]

Two output files:
  results/figures/swaroopa_all_ap_diffdrr.png   (AP frames, 4 per page)
  results/figures/swaroopa_all_lat_diffdrr.png  (LAT frames, 4 per page)

Single summary dashboard also saved:
  results/figures/swaroopa_all_summary.png
"""

import sys, json, os, tempfile, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))
from swaroopa_loader import (SwaroLoader, SWARO_PIX_MM, SWARO_IMG_SIZE,
                              SWARO_SID_MM, SWARO_CX, SWARO_CY)
from deepfluoro_loader import perturb_extrinsic, xzy_inv
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
from diffdrr.data import read as diffdrr_read
import SimpleITK as sitk
from run_swaroopa_diffdrr import build_subject, build_subject_masked, build_subject_hu_clipped, _suppress_highlights

# ── CLI args ──────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser()
_ap.add_argument('--results',     default='results/swaroopa_results_go_5ap5lat_merged.json')
_ap.add_argument('--out_ap',      default='results/figures/swaroopa_all_ap_diffdrr.png')
_ap.add_argument('--out_lat',     default='results/figures/swaroopa_all_lat_diffdrr.png')
_ap.add_argument('--out_summary', default='results/figures/swaroopa_all_summary.png')
_ap.add_argument('--cylinder_radius', type=float, default=None,
                 help='Cylinder mask radius (mm) around spine centroid (default: no mask)')
_ap.add_argument('--min_hu', type=float, default=None,
                 help='Lower HU clip inside cylinder (default: 0.0)')
_ap.add_argument('--suppress_highlights', action='store_true',
                 help='Invert X-ray and darken bright metal highlights before display')
_args = _ap.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS     = Path(_args.results)
OUT_AP      = Path(_args.out_ap)
OUT_LAT     = Path(_args.out_lat)
OUT_SUMMARY = Path(_args.out_summary)
RENDER_SIZE = 192          # smaller so all frames fit on one page
LM_COLOURS  = {'L1':'red','L2':'orange','L3':'yellow','L4':'lime','L5':'cyan'}
ROWS_PER_PAGE = 6          # frames per output image

_X0_MM = (SWARO_CX - SWARO_IMG_SIZE / 2.0) * SWARO_PIX_MM
_Y0_MM = (SWARO_CY - SWARO_IMG_SIZE / 2.0) * SWARO_PIX_MM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load results ──────────────────────────────────────────────────────────────
with open(RESULTS) as f:
    res = json.load(f)
pp = res.get('per_projection') or res.get('swaroopa', {}).get('per_projection') or res
all_frames = sorted(pp.keys())
ap_frames  = [k for k in all_frames if k.startswith('ap_')]
lat_frames = [k for k in all_frames if k.startswith('lat_')]
print(f"Loaded {len(all_frames)} frames  (AP={len(ap_frames)}, LAT={len(lat_frames)})")

# ── Load specimen ─────────────────────────────────────────────────────────────
print("Loading specimen ...")
loader = SwaroLoader()
spec   = loader.load(frames=all_frames, verbose=False)

lm_names = sorted(spec.landmarks_3d.keys())
pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])

# ── Build diffdrr subject ─────────────────────────────────────────────────────
print("Building diffdrr subject ...")
if _args.cylinder_radius is not None:
    hu_min = _args.min_hu if _args.min_hu is not None else 0.0
    print(f"  Using cylinder mask: r={_args.cylinder_radius}mm, min_hu={hu_min}")
    subject = build_subject_masked(spec, cylinder_r_mm=_args.cylinder_radius, hu_min=hu_min)
elif _args.min_hu is not None:
    print(f"  HU clip (no cylinder mask): min_hu={_args.min_hu}")
    subject = build_subject_hu_clipped(spec, hu_min=_args.min_hu)
else:
    subject = build_subject(spec)

# LPS→RAS origin offset
expected_ras      = np.asarray(spec.ct_origin, dtype=np.float64) * np.array([-1.,-1.,1.])
torchio_ras       = subject.volume.affine[:3, 3].astype(np.float64)
lps_to_ras_offset = torchio_ras - expected_ras
print(f"  LPS→RAS origin offset: {lps_to_ras_offset.round(3)} mm")

# ── DRR module cache ──────────────────────────────────────────────────────────
_drr_cache = {}
def get_drr(size, pix_mm):
    key = (size, round(pix_mm, 6))
    if key not in _drr_cache:
        _drr_cache[key] = DRR(subject, sdd=SWARO_SID_MM, height=size, width=size,
                               delx=pix_mm, dely=pix_mm, x0=_X0_MM, y0=_Y0_MM,
                               renderer="siddon", reverse_x_axis=False).to(device)
    return _drr_cache[key]

# ── Pose / render helpers ─────────────────────────────────────────────────────
L2R = np.array([-1., -1., 1.])

def pose_from_extrinsic(R, t):
    right = xzy_inv(R.T @ np.array([1.,0.,0.])).flatten() * L2R
    up    = xzy_inv(R.T @ np.array([0.,1.,0.])).flatten() * L2R
    pa    = xzy_inv(R.T @ np.array([0.,0.,1.])).flatten() * L2R
    src   = xzy_inv(-R.T @ t).flatten()                   * L2R
    up    = -up
    mat   = np.eye(4, dtype=np.float32)
    mat[:3,:3] = np.stack([right, up, pa], axis=1).astype(np.float32)
    mat[:3, 3] = src.astype(np.float32)
    return RigidTransform(torch.tensor(mat, dtype=torch.float32, device=device))

@torch.no_grad()
def render(R, t, size):
    pix = SWARO_PIX_MM * (SWARO_IMG_SIZE / size)
    drr_mod = get_drr(size, pix)
    pose    = pose_from_extrinsic(R, t)
    img     = drr_mod(pose).squeeze().cpu().numpy()
    mn, mx  = img.min(), img.max()
    return (img - mn) / (mx - mn) if mx > mn else img

@torch.no_grad()
def proj_lm(R, t, pts3d_lps, size):
    pix     = SWARO_PIX_MM * (SWARO_IMG_SIZE / size)
    drr_mod = get_drr(size, pix)
    pose    = pose_from_extrinsic(R, t)
    pts_ras = pts3d_lps * np.array([-1.,-1.,1.]) + lps_to_ras_offset
    pts_t   = torch.tensor(pts_ras, dtype=torch.float32, device=device).unsqueeze(0)
    uv      = drr_mod.perspective_projection(pose, pts_t)
    return uv[0].cpu().numpy().astype(np.float32)   # (N,2) col,row

def compute_pde_mm(R, t, proj, pts3d_lps, size):
    """Compute mean 3D PDE (mm) between projected pts3d and GT landmarks."""
    uv = proj_lm(R, t, pts3d_lps, size)   # (N,2) in render-size pixels
    scale = SWARO_IMG_SIZE / size
    pde_vals = []
    for j, name in enumerate(lm_names):
        if name in proj.gt_landmarks_2d:
            u_gt, v_gt = proj.gt_landmarks_2d[name]
            u_pr, v_pr = uv[j]
            d_px = np.hypot(u_gt - u_pr * scale, v_gt - v_pr * scale)
            pde_vals.append(d_px * SWARO_PIX_MM)
    return float(np.mean(pde_vals)) if pde_vals else float('nan')

# ── Per-frame rendering data ──────────────────────────────────────────────────
def render_frame(proj_key):
    proj  = next(p for p in spec.projections if p.proj_key == proj_key)
    r     = pp[proj_key]
    delta_final = r['best_pose_delta']
    # Perturbed (initial) delta from phase log
    phase0 = next((p for p in r['phase_log'] if p['phase'] == 'perturbed'), None)
    delta_perturbed = phase0['delta'] if phase0 else [0]*6

    # Three poses: EPnP (delta=0), Perturbed, Final
    R_epnp,  t_epnp  = proj.R_proj.copy(), proj.t_proj.copy()
    R_pert,  t_pert  = perturb_extrinsic(proj.R_proj, proj.t_proj,
                                          np.array(delta_perturbed[:3]),
                                          np.array(delta_perturbed[3:]))
    R_final, t_final = perturb_extrinsic(proj.R_proj, proj.t_proj,
                                          np.array(delta_final[:3]),
                                          np.array(delta_final[3:]))

    drr_epnp = render(R_epnp,  t_epnp,  RENDER_SIZE)
    drr_pert = render(R_pert,  t_pert,  RENDER_SIZE)
    drr_f    = render(R_final, t_final, RENDER_SIZE)

    xray = cv2.resize(proj.image_raw, (RENDER_SIZE, RENDER_SIZE),
                       interpolation=cv2.INTER_AREA)
    xray = 1.0 - xray   # invert: dark-bone → bright-bone
    if _args.suppress_highlights:
        xray = _suppress_highlights(xray)

    uv_epnp = proj_lm(R_epnp,  t_epnp,  pts3d, RENDER_SIZE)
    uv_pert = proj_lm(R_pert,  t_pert,  pts3d, RENDER_SIZE)
    uv_f    = proj_lm(R_final, t_final, pts3d, RENDER_SIZE)

    pde_epnp = compute_pde_mm(R_epnp, t_epnp, proj, pts3d, RENDER_SIZE)
    pde_pert = phase0['pde_mm'] if phase0 else r['initial_pde_mm']

    return dict(proj=proj, r=r, xray=xray,
                drr_epnp=drr_epnp, drr_pert=drr_pert, drr_f=drr_f,
                uv_epnp=uv_epnp, uv_pert=uv_pert, uv_f=uv_f,
                pde_epnp=pde_epnp, pde_pert=pde_pert,
                R_epnp=R_epnp, t_epnp=t_epnp,
                R_pert=R_pert,  t_pert=t_pert,
                R_final=R_final, t_final=t_final)

# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_frame_row(axes_row, fdata, frame_label, scale):
    proj  = fdata['proj']
    r     = fdata['r']
    s     = r['final_pde_mm'] < r['initial_pde_mm']
    fg, dg = r['final_go'], r['go_delta']
    fp     = r['final_pde_mm']
    ep     = fdata['pde_epnp']
    pp_mm  = fdata['pde_pert']

    data_cols = [
        (fdata['xray'],     'X-ray + GT lm',                                  None,            True),
        (fdata['drr_epnp'], f'EPnP  PDE={ep:.1f}mm',                           fdata['uv_epnp'], False),
        (fdata['drr_pert'], f'Perturbed  PDE={pp_mm:.1f}mm',                   fdata['uv_pert'], False),
        (fdata['drr_f'],    f'Final  GO={fg:.3f}  ΔGO={dg:+.3f}  PDE={fp:.1f}mm  {"✓" if s else "✗"}',
                                                                                fdata['uv_f'],    False),
    ]

    for col_i, (img, subtitle, uv_proj, is_xray) in enumerate(data_cols):
        ax = axes_row[col_i]
        ax.set_facecolor('#1a1a1a')
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#444')

        if is_xray:
            for j, name in enumerate(lm_names):
                if name in proj.gt_landmarks_2d:
                    u, v = proj.gt_landmarks_2d[name]
                    ax.plot(u*scale, v*scale, 'o', color=LM_COLOURS[name],
                            markersize=5, markeredgewidth=1, markeredgecolor='white', zorder=5)
            ax.set_ylabel(frame_label, color='white', fontsize=8, labelpad=3)
        else:
            for j, name in enumerate(lm_names):
                u, v = uv_proj[j]
                if 0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE:
                    ax.plot(u, v, 'o', color=LM_COLOURS[name],
                            markersize=5, markeredgewidth=1, markeredgecolor='white', zorder=5)

        if col_i == 3:
            sc = 'lime' if s else 'tomato'
        else:
            sc = '#aaaaaa'
        ax.set_xlabel(subtitle, color=sc, fontsize=7, labelpad=2)

# ── Save one multi-frame figure for a list of frames ─────────────────────────
def save_multiframe_figure(frame_list, out_path, view_label):
    n      = len(frame_list)
    ncols  = 4
    scale  = RENDER_SIZE / SWARO_IMG_SIZE

    col_titles = ['Real X-ray + GT lm',
                  'EPnP pose (DiffDRR)',
                  'Perturbed pose (DiffDRR)',
                  'Final registered (DiffDRR)']

    fig, axes = plt.subplots(n, ncols,
                             figsize=(ncols * (RENDER_SIZE/72 + 0.4),
                                      n     * (RENDER_SIZE/72 + 0.55) + 1.2))
    fig.patch.set_facecolor('#1a1a1a')

    if n == 1:
        axes = axes[np.newaxis, :]   # keep 2D indexing

    success_list = [pp[k]['final_pde_mm'] < pp[k]['initial_pde_mm'] for k in frame_list]
    n_ok = sum(success_list)
    init_pdes  = [pp[k]['initial_pde_mm'] for k in frame_list]
    final_pdes = [pp[k]['final_pde_mm']   for k in frame_list]

    fig.suptitle(
        f'Swaroopa — {view_label} ({n} frames)  |  '
        f'Success {n_ok}/{n}  |  '
        f'Mean PDE {np.mean(init_pdes):.1f}→{np.mean(final_pdes):.1f} mm\n'
        f'DiffDRR renderer  |  Fx=3646px  pix=0.288mm  SID=1050mm',
        color='white', fontsize=10, y=0.995)

    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, color='white', fontsize=9, pad=4)

    for row_i, fkey in enumerate(frame_list):
        print(f"  Rendering {fkey} ...")
        fdata = render_frame(fkey)
        plot_frame_row(axes[row_i], fdata, fkey, scale)

    legend_patches = [mpatches.Patch(color=c, label=n) for n, c in LM_COLOURS.items()]
    fig.legend(handles=legend_patches, loc='lower center', ncol=5,
               facecolor='#2a2a2a', edgecolor='white', labelcolor='white',
               fontsize=8, title='Vertebral centroids', title_fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ── Summary dashboard ─────────────────────────────────────────────────────────
def save_summary(frame_list, out_path):
    n         = len(frame_list)
    init_pde  = np.array([pp[k]['initial_pde_mm'] for k in frame_list])
    final_pde = np.array([pp[k]['final_pde_mm']   for k in frame_list])
    init_go   = np.array([pp[k]['initial_go']      for k in frame_list])
    final_go  = np.array([pp[k]['final_go']        for k in frame_list])
    success   = np.array([pp[k]['final_pde_mm'] < pp[k]['initial_pde_mm'] for k in frame_list])
    x         = np.arange(n)
    ap_mask   = np.array([k.startswith('ap') for k in frame_list])

    C_SUCC, C_FAIL, C_INIT = '#44dd88', '#ee4444', '#aaaaaa'
    C_AP, C_LAT = '#66aaff', '#ffaa44'
    bar_colors = [C_SUCC if s else C_FAIL for s in success]

    matplotlib.rcParams.update({
        'text.color': '#dddddd', 'axes.labelcolor': '#dddddd',
        'xtick.color': '#dddddd', 'ytick.color': '#dddddd',
        'axes.edgecolor': '#444', 'axes.facecolor': '#1e1e1e',
        'figure.facecolor': '#111111', 'grid.color': '#333',
        'grid.linestyle': '--', 'grid.alpha': 0.6,
    })

    fig, axes = plt.subplots(3, 1, figsize=(max(18, n*0.55), 14))
    fig.suptitle(
        f'Swaroopa CMA-ES Registration — All {n} frames  |  '
        f'Success {success.sum()}/{n} = {success.mean()*100:.0f}%  |  '
        f'Mean PDE {init_pde.mean():.1f}→{final_pde.mean():.1f} mm',
        fontsize=13, fontweight='bold', y=0.998)

    def _setup(ax, title, ylabel):
        ax.set_facecolor('#1e1e1e')
        ax.set_title(title, fontsize=11, pad=5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(frame_list, rotation=55, ha='right', fontsize=7.5)
        ax.grid(axis='y')
        for i, k in enumerate(frame_list):
            c = '#002244' if k.startswith('ap') else '#442200'
            ax.axvspan(i-0.5, i+0.5, color=c, alpha=0.25, zorder=0)

    # Panel 1: PDE
    ax = axes[0]
    ax.bar(x-0.2, init_pde,  width=0.38, color=C_INIT, alpha=0.6, label='Initial PDE')
    ax.bar(x+0.2, final_pde, width=0.38, color=bar_colors, alpha=0.9)
    ax.axhline(30, color='yellow', lw=0.8, ls=':', alpha=0.7, label='30 mm')
    _setup(ax, 'Per-frame PDE: initial (grey) vs final (green=success / red=fail)', 'PDE (mm)')
    ax.legend(handles=[
        mpatches.Patch(color=C_INIT,   alpha=0.6, label='Initial PDE'),
        mpatches.Patch(color=C_SUCC,   alpha=0.9, label='Final — success'),
        mpatches.Patch(color=C_FAIL,   alpha=0.9, label='Final — fail'),
        mpatches.Patch(color='#002244',alpha=0.5, label='AP'),
        mpatches.Patch(color='#442200',alpha=0.5, label='LAT'),
    ], fontsize=8, loc='upper right', ncol=3)

    # Panel 2: GO cost
    ax = axes[1]
    ax.bar(x-0.2, init_go,  width=0.38, color=C_INIT, alpha=0.6)
    ax.bar(x+0.2, final_go, width=0.38, color=bar_colors, alpha=0.9)
    ax.axhline(0.6, color='yellow', lw=0.8, ls=':', alpha=0.7)
    _setup(ax, 'Per-frame GO cost: initial (grey) vs final', 'GO cost')

    # Panel 3: PDE improvement
    delta_pde = init_pde - final_pde
    sort_idx = np.argsort(delta_pde)[::-1]
    ax = axes[2]
    ax.bar(np.arange(n), delta_pde[sort_idx],
           color=[bar_colors[i] for i in sort_idx], alpha=0.85)
    ax.axhline(0, color='#dddddd', lw=0.7, alpha=0.5)
    ax.set_facecolor('#1e1e1e')
    ax.set_title('PDE improvement (Δ = initial − final), sorted descending', fontsize=11, pad=5)
    ax.set_ylabel('ΔPDE (mm)', fontsize=9)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([frame_list[i] for i in sort_idx], rotation=55, ha='right', fontsize=7.5)
    ax.grid(axis='y')
    for xi, val in enumerate(delta_pde[sort_idx]):
        ax.text(xi, val + (2 if val >= 0 else -4), f'{val:+.0f}',
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=6.5, color='#dddddd')

    plt.tight_layout(rect=[0, 0, 1, 0.997])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Print console summary
    print(f"\n{'='*60}")
    print(f"  Frame          Init PDE   Final PDE    ΔGO     OK?")
    print(f"  {'-'*58}")
    for k in all_frames:
        r = pp[k]
        s = r['final_pde_mm'] < r['initial_pde_mm']
        print(f"  {k:<14}  {r['initial_pde_mm']:>8.1f}mm  {r['final_pde_mm']:>8.1f}mm  "
              f"{r['go_delta']:>+7.4f}  {'✓' if s else '✗'}")
    print(f"{'='*60}\n")

    print("Rendering AP frames ...")
    save_multiframe_figure(ap_frames,  OUT_AP,  'AP frames')

    print("Rendering LAT frames ...")
    save_multiframe_figure(lat_frames, OUT_LAT, 'LAT frames')

    print("Saving summary dashboard ...")
    save_summary(all_frames, OUT_SUMMARY)

    print("\nAll done.")
