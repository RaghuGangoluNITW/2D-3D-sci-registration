"""
generate_manuscript_figures.py
================================
Generates ALL publication figures for main.tex.
Run:
  cd /home/supermicro/Documents/2D_3D_Raghu
  python3 scripts/generate_manuscript_figures.py

Outputs → manuscript/figures/
  figA_ct_mip.pdf/png           -- CT coronal MIP with L1-L5 centroids
  figB_arjun_frame_<a-e>.pdf    -- per-frame 4-panel: Xray|DRR_init|DRR_final|Overlay
  figC_arjun_all5.pdf           -- all 5 Arjun frames in one grid
  figD_ramulamma_frame_<k>.pdf  -- per-frame 4-panel for all 8 Ramulamma frames
  figE_ramulamma_all8.pdf       -- all 8 Ramulamma frames in one grid
  figF_go_convergence_arjun.pdf -- GO bar + ΔGO bar, Arjun
  figG_go_convergence_ramu.pdf  -- GO bar + ΔGO bar, Ramulamma
  figH_go_violin.pdf            -- violin/box comparison of ΔGO distributions
  figI_pde_arjun.pdf            -- per-landmark PDE chart Arjun
  figJ_success_rate.pdf         -- success rate grouped bar
  figK_runtime.pdf              -- runtime per projection
  figL_checkerboard_all.pdf     -- checkerboard overlay grid (both patients)
"""

import sys, os, json
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import cv2

from arjun_loader import (
    ArjunLoader, ArjunProjection, _make_K,
    ARJUN_REF_PIX_MM, ARJUN_REF_SIZE,
)
from ramulamma_loader import (RamulamaLoader, RAMU_PIX_MM, RAMU_IMG_SIZE,
                              RAMU_FX, RAMU_FY, RAMU_CX, RAMU_CY,
                              project_world_ramu)
from deepfluoro_loader import perturb_extrinsic
from deepfluoro_drr import DeepFluoroDRR

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.join(os.path.dirname(__file__), '..')
OUT     = os.path.join(BASE, 'manuscript', 'figures')
os.makedirs(OUT, exist_ok=True)

ARJUN_JSON = os.path.join(BASE, 'results', 'arjun_results.json')
RAMU_JSON  = os.path.join(BASE, 'results', 'ramulamma_results_clean.json')

# ── render sizes ──────────────────────────────────────────────────────────────
A_VIS, A_PIX_MM, A_STEPS = 384, ARJUN_REF_PIX_MM*(ARJUN_REF_SIZE/384), 256
R_VIS, R_PIX_MM, R_STEPS = 256, RAMU_PIX_MM*(RAMU_IMG_SIZE/256), 180

PERTURB_ROT_DEG  = 30.0
PERTURB_TRANS_MM = 60.0

C_OK   = '#2ecc71'
C_FAIL = '#e74c3c'
C_INIT = '#95a5a6'
C_BLUE = '#2980b9'
C_ORG  = '#e67e22'

# ── helpers ───────────────────────────────────────────────────────────────────

def save(fig, name):
    pdf_path = os.path.join(OUT, f'{name}.pdf')
    png_path = os.path.join(OUT, f'{name}.png')
    if os.path.exists(pdf_path) and os.path.exists(png_path):
        plt.close(fig)
        print(f'  Skipped {name} (already exists)')
        return
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {name}')


def norm_img(img):
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)


def render_drr(drr_gen, R, t, vis, pix, steps, invert=True):
    d = drr_gen.generate_from_extrinsic(R, t, vis, pix, steps)
    return (1.0 - d) if invert else d


def checkerboard(a, b, tiles=8):
    h, w = a.shape
    th, tw = max(1, h // tiles), max(1, w // tiles)
    mask = np.zeros((h, w), dtype=np.float32)
    for i in range(tiles):
        for j in range(tiles):
            if (i + j) % 2 == 0:
                mask[i*th:(i+1)*th, j*tw:(j+1)*tw] = 1.0
    return np.where(mask > 0.5, a, b)


def project_lm(proj, pts3d, R, t, vis_size):
    from arjun_loader import xzy
    fx, fy, cx, cy = _make_K(proj.img_w, proj.img_h)
    pts_xzy = xzy(pts3d)
    P = (R @ pts_xzy.T).T + t
    u = fx * P[:, 0] / P[:, 2] + cx
    v = fy * P[:, 1] / P[:, 2] + cy
    return u * vis_size / proj.img_w, v * vis_size / proj.img_h


def project_lm_ramu(pts3d, R, t, vis_size):
    """Project 3D landmarks using Ramulamma intrinsics, scale to vis_size."""
    uv = project_world_ramu(pts3d, R, t)   # (N,2) in original 1024×1024 frame
    return uv[:, 0] * vis_size / RAMU_IMG_SIZE, uv[:, 1] * vis_size / RAMU_IMG_SIZE


def gt_to_vis(proj, lm_name, vis_size):
    uv = proj.gt_landmarks_2d[lm_name]
    return uv[0]*vis_size/proj.img_w, uv[1]*vis_size/proj.img_h


pe_stroke = lambda lw, clr: [pe.withStroke(linewidth=lw, foreground=clr)]

# ── load data ─────────────────────────────────────────────────────────────────
print('Loading Arjun data...')
a_loader = ArjunLoader()
a_spec   = a_loader.load(verbose=True)
a_drr    = DeepFluoroDRR(a_spec)

print('Loading Ramulamma data...')
RAMU_FRAMES = [6, 7, 8, 9, 10, 15, 16, 21]
r_loader = RamulamaLoader()
# The PREOP CT was stored under data/testing/ – patch the paths before loading
_RAMU_PREOP_DIR = Path('/home/supermicro/Documents/2D_3D_Raghu/data/testing') \
    / 'RAMULAMMA PREOP' / 'RAMULAMMA PREOP'
_RAMU_CT_FALLBACK  = _RAMU_PREOP_DIR / '4 L_Spine  1.0  B60s.nrrd'
_RAMU_LM_FALLBACK  = _RAMU_PREOP_DIR / 'centroids.mrk.json'
if not r_loader.ct_nrrd.exists() and _RAMU_CT_FALLBACK.exists():
    print(f'  [patch] CT path → {_RAMU_CT_FALLBACK}')
    r_loader.ct_nrrd = _RAMU_CT_FALLBACK
if not r_loader.lm_3d_json.exists() and _RAMU_LM_FALLBACK.exists():
    print(f'  [patch] LM path → {_RAMU_LM_FALLBACK}')
    r_loader.lm_3d_json = _RAMU_LM_FALLBACK
r_spec   = r_loader.load(frames=RAMU_FRAMES, verbose=True)
r_drr    = DeepFluoroDRR(r_spec)

with open(ARJUN_JSON)  as f: ar = json.load(f)['arjun']
with open(RAMU_JSON)   as f: rr = json.load(f)['ramulamma']

a_pp = ar['per_projection']
r_pp = rr['per_projection']

a_proj_map = {p.proj_key: p for p in a_spec.projections}
r_proj_map = {p.proj_key: p for p in r_spec.projections}

lm_names_a = sorted(a_spec.landmarks_3d.keys())
pts3d_a    = np.array([a_spec.landmarks_3d[n] for n in lm_names_a])
lm_names_r = sorted(r_spec.landmarks_3d.keys())
pts3d_r    = np.array([r_spec.landmarks_3d[n] for n in lm_names_r])
LM_CMAP    = plt.cm.tab10(np.linspace(0, 1, 5))


def annotate_lm(ax, lm_names, u_proj, v_proj, vis, proj_obj=None,
                annotated=None, show_gt=False, gt_vis=None):
    for i, lname in enumerate(lm_names):
        ux, vy = u_proj[i], v_proj[i]
        if not (0 <= ux < vis and 0 <= vy < vis):
            continue
        c = LM_CMAP[i % len(LM_CMAP)]
        ax.plot(ux, vy, '+', color=c, ms=14, mew=2.5,
                path_effects=pe_stroke(3, 'black'), zorder=6)
        ax.text(ux+4, vy+4, lname, color=c, fontsize=8, fontweight='bold',
                path_effects=pe_stroke(1.8, 'black'), zorder=7)
        if show_gt and annotated and lname in annotated and proj_obj and gt_vis:
            gx, gy = gt_to_vis(proj_obj, lname, gt_vis)
            ax.plot(gx, gy, 'o', color='lime', ms=9,
                    mec='black', mew=1, alpha=0.8, zorder=5)
            ax.plot([gx, ux], [gy, vy], '-', color=c, lw=1.2, alpha=0.6, zorder=4)


def make_4panel(xray_vis, drr_init, drr_final, init_go, final_go,
                success, lm_names, pts3d, proj_obj, R_init, t_init,
                R_final, t_final, vis_size, title, pde_dict=None):
    """Returns (fig) with 4 panels: Xray | DRR_init | DRR_final | Checkerboard overlay."""
    ovl = checkerboard(norm_img(xray_vis), drr_final)
    panels = [
        (xray_vis,   f'X-ray (real)\nFrame size {proj_obj.img_w}×{proj_obj.img_h}'),
        (drr_init,   f'DRR @ init pose\nGO = {init_go:.4f}'),
        (drr_final,  f'DRR @ final pose\nGO = {final_go:.4f}  {"✓ SUCCESS" if success else "✗ FAIL"}'),
        (ovl,        'Checkerboard Overlay\n(X-ray | DRR final)'),
    ]
    border = [C_BLUE, C_ORG,
              C_OK if success else C_FAIL,
              C_OK if success else C_FAIL]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    annotated = list(proj_obj.gt_landmarks_2d.keys()) if hasattr(proj_obj, 'gt_landmarks_2d') else []

    for ax, (img, subtitle), bclr in zip(axes, panels, border):
        ax.imshow(norm_img(img) if img.dtype != np.float32 else img,
                  cmap='gray', vmin=0, vmax=1)
        is_xray = subtitle.startswith('X-ray')
        is_drr  = 'DRR' in subtitle or 'Overlay' in subtitle
        R_use = R_init if 'init' in subtitle else R_final
        t_use = t_init if 'init' in subtitle else t_final
        u_proj, v_proj = project_lm(proj_obj, pts3d, R_use, t_use, vis_size)
        if is_xray:
            # show GT green dots on X-ray
            for lname in annotated:
                gx, gy = gt_to_vis(proj_obj, lname, vis_size)
                ax.plot(gx, gy, 'o', color='lime', ms=11,
                        mec='black', mew=1.2, zorder=7)
                ax.text(gx+4, gy-5, lname, color='lime', fontsize=9, fontweight='bold',
                        path_effects=pe_stroke(2, 'black'), zorder=8)
        annotate_lm(ax, lm_names, u_proj, v_proj, vis_size,
                    proj_obj if is_drr else None,
                    annotated if is_drr else None,
                    show_gt=is_drr, gt_vis=vis_size)
        ax.set_xlim(0, vis_size); ax.set_ylim(vis_size, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(subtitle, fontsize=9.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(bclr); sp.set_linewidth(3)

    # PDE annotation
    if pde_dict:
        txt = 'PDE per landmark:\n' + '\n'.join(f'  {k}: {v:.2f} mm' for k, v in sorted(pde_dict.items()))
        axes[-1].text(0.98, 0.02, txt, transform=axes[-1].transAxes,
                      fontsize=8, va='bottom', ha='right', fontfamily='monospace',
                      bbox=dict(boxstyle='round', fc='#f8f8f8', alpha=0.85))

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    gt_m = plt.Line2D([0],[0], marker='o', color='lime', ls='None', ms=9, mec='black', label='GT annotation')
    pr_m = plt.Line2D([0],[0], marker='+', color='white', ls='None', ms=9, mew=2, label='3D reprojection')
    axes[2].legend(handles=[gt_m, pr_m], fontsize=8.5, loc='lower right', framealpha=0.8)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# FIGURE A: CT MIP + L1-L5 centroids
# ═══════════════════════════════════════════════════════════════════════
print('\nFigure A – CT MIP...')
import SimpleITK as sitk

ct_img  = sitk.ReadImage(str(a_loader.ct_nrrd))
ct_arr  = sitk.GetArrayFromImage(ct_img).astype(np.float32)  # (z, y, x)
spacing = ct_img.GetSpacing()  # (sx, sy, sz)

# Coronal MIP (collapse along y axis → view: x-z plane)
mip_cor = np.max(ct_arr, axis=1)  # (z, x)
# Sagittal MIP (collapse x)
mip_sag = np.max(ct_arr, axis=2)  # (z, y)
# Axial MIP (collapse z)
mip_axi = np.max(ct_arr, axis=0)  # (y, x)

# Window/level
def wl(img, ww=2000, wc=300):
    lo, hi = wc - ww/2, wc + ww/2
    return np.clip((img - lo) / (hi - lo), 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
views = [
    (mip_cor, 'Coronal MIP',  'x (L→R)', 'z (S→I)'),
    (mip_sag, 'Sagittal MIP', 'y (A→P)', 'z (S→I)'),
    (mip_axi, 'Axial MIP',    'x (L→R)', 'y (A→P)'),
]
for ax, (mip, title, xlabel, ylabel) in zip(axes, views):
    ax.imshow(wl(mip), cmap='bone', origin='upper', aspect='auto')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

# Overlay landmarks on coronal (x vs z)
origin = np.array(ct_img.GetOrigin())   # (ox, oy, oz) in LPS mm
sx, sy, sz = spacing
lm_plot_cmap = plt.cm.tab10(np.linspace(0, 1, len(lm_names_a)))
for i, lname in enumerate(lm_names_a):
    pt = a_spec.landmarks_3d[lname]  # LPS mm
    # to voxel index: col=x, row=z
    col_idx = (pt[0] - origin[0]) / sx
    row_idx = (pt[2] - origin[2]) / sz
    axes[0].plot(col_idx, row_idx, 'o', color=lm_plot_cmap[i], ms=12,
                 mec='white', mew=1.5, zorder=5)
    axes[0].text(col_idx+4, row_idx-3, lname, color=lm_plot_cmap[i],
                 fontsize=10, fontweight='bold',
                 path_effects=pe_stroke(2, 'black'), zorder=6)
    # sagittal: col=y, row=z
    col_sag = (pt[1] - origin[1]) / sy
    axes[1].plot(col_sag, row_idx, 'o', color=lm_plot_cmap[i], ms=12,
                 mec='white', mew=1.5, zorder=5)
    axes[1].text(col_sag+4, row_idx-3, lname, color=lm_plot_cmap[i],
                 fontsize=10, fontweight='bold',
                 path_effects=pe_stroke(2, 'black'), zorder=6)
    # axial: col=x, row=y
    col_axi = (pt[0] - origin[0]) / sx
    row_axi = (pt[1] - origin[1]) / sy
    axes[2].plot(col_axi, row_axi, 'o', color=lm_plot_cmap[i], ms=12,
                 mec='white', mew=1.5, zorder=5)
    axes[2].text(col_axi+4, row_axi-3, lname, color=lm_plot_cmap[i],
                 fontsize=10, fontweight='bold',
                 path_effects=pe_stroke(2, 'black'), zorder=6)

fig.suptitle('Preoperative CT – Maximum Intensity Projections with L1–L5 Centroid Landmarks (Arjun)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
save(fig, 'figA_ct_mip')

# ═══════════════════════════════════════════════════════════════════════
# FIGURES B1-B5: Arjun per-frame 4-panel
# ═══════════════════════════════════════════════════════════════════════
print('\nFigures B – Arjun per-frame panels...')
for key in sorted(a_pp.keys()):
    proj   = a_proj_map[key]
    pdata  = a_pp[key]
    bd     = np.array(pdata.get('best_pose_delta', [0]*6))
    R_epnp = proj.R_proj.copy()
    t_epnp = proj.t_proj.copy()
    R_fin, t_fin = perturb_extrinsic(R_epnp, t_epnp, bd[:3], bd[3:])

    xray_vis = cv2.resize(proj.image_raw, (A_VIS, A_VIS), interpolation=cv2.INTER_AREA)
    drr_i    = render_drr(a_drr, R_epnp, t_epnp, A_VIS, A_PIX_MM, A_STEPS)
    drr_f    = render_drr(a_drr, R_fin,  t_fin,  A_VIS, A_PIX_MM, A_STEPS)

    u_i, v_i = project_lm(proj, pts3d_a, R_epnp, t_epnp, A_VIS)
    u_f, v_f = project_lm(proj, pts3d_a, R_fin,  t_fin,  A_VIS)

    fig = make_4panel(
        xray_vis, drr_i, drr_f,
        pdata['initial_go'], pdata['final_go'], pdata['success'],
        lm_names_a, pts3d_a, proj,
        R_epnp, t_epnp, R_fin, t_fin, A_VIS,
        f'Arjun – Frame {key.upper()} | '
        f'GO: {pdata["initial_go"]:.4f}→{pdata["final_go"]:.4f} '
        f'(Δ={pdata["go_delta"]:+.4f}) | '
        f'PDE: {pdata["initial_pde_mm"]:.2f}→{pdata["final_pde_mm"]:.2f} mm',
        pdata.get('pde_per_landmark'),
    )
    save(fig, f'figB_arjun_frame_{key}')

# ═══════════════════════════════════════════════════════════════════════
# FIGURE C: All 5 Arjun frames in a 5-row grid
# ═══════════════════════════════════════════════════════════════════════
print('\nFigure C – Arjun all-5 grid...')
keys_a = sorted(a_pp.keys())
fig, axes = plt.subplots(5, 4, figsize=(22, 28))
col_titles = ['X-ray (real) + GT landmarks',
              'DRR @ EPnP init',
              'DRR @ CMA-ES final',
              'Checkerboard overlay']
for ax, t in zip(axes[0], col_titles):
    ax.set_title(t, fontsize=11, fontweight='bold', pad=8)

for row, key in enumerate(keys_a):
    proj  = a_proj_map[key]
    pd    = a_pp[key]
    bd    = np.array(pd.get('best_pose_delta', [0]*6))
    R_e, t_e = proj.R_proj.copy(), proj.t_proj.copy()
    R_f, t_f = perturb_extrinsic(R_e, t_e, bd[:3], bd[3:])

    xray_v = cv2.resize(proj.image_raw, (A_VIS, A_VIS), interpolation=cv2.INTER_AREA)
    drr_i  = render_drr(a_drr, R_e, t_e, A_VIS, A_PIX_MM, A_STEPS)
    drr_f  = render_drr(a_drr, R_f, t_f, A_VIS, A_PIX_MM, A_STEPS)
    ovl    = checkerboard(norm_img(xray_v), drr_f)

    imgs    = [xray_v, drr_i, drr_f, ovl]
    R_list  = [R_e, R_e, R_f, R_f]
    t_list  = [t_e, t_e, t_f, t_f]
    bclrs   = [C_BLUE, C_ORG, C_OK if pd['success'] else C_FAIL,
                C_OK if pd['success'] else C_FAIL]
    subs    = [
        f'Frame {key.upper()}  |  {proj.img_w}×{proj.img_h}',
        f'Init GO={pd["initial_go"]:.4f}  PDE={pd["initial_pde_mm"]:.2f}mm',
        f'Final GO={pd["final_go"]:.4f}  PDE={pd["final_pde_mm"]:.2f}mm  '
        f'{"✓" if pd["success"] else "✗"}',
        f'Δ GO={pd["go_delta"]:+.4f}',
    ]
    annotated = list(proj.gt_landmarks_2d.keys())
    for col, (ax, img, R_u, t_u, bclr, sub) in enumerate(
            zip(axes[row], imgs, R_list, t_list, bclrs, subs)):
        ax.imshow(norm_img(img), cmap='gray', vmin=0, vmax=1)
        u_p, v_p = project_lm(proj, pts3d_a, R_u, t_u, A_VIS)
        if col == 0:
            for lname in annotated:
                gx, gy = gt_to_vis(proj, lname, A_VIS)
                ax.plot(gx, gy, 'o', color='lime', ms=10, mec='black', mew=1.2, zorder=7)
                ax.text(gx+4, gy-4, lname, color='lime', fontsize=8, fontweight='bold',
                        path_effects=pe_stroke(2, 'black'), zorder=8)
        for i, lname in enumerate(lm_names_a):
            ux, vy = u_p[i], v_p[i]
            if 0 <= ux < A_VIS and 0 <= vy < A_VIS:
                c = LM_CMAP[i % len(LM_CMAP)]
                ax.plot(ux, vy, '+', color=c, ms=13, mew=2.5,
                        path_effects=pe_stroke(3, 'black'), zorder=6)
        ax.set_xlim(0, A_VIS); ax.set_ylim(A_VIS, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(sub, fontsize=8.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(bclr); sp.set_linewidth(3)

fig.suptitle('Arjun – All 5 Projections: X-ray | DRR(init) | DRR(final) | Checkerboard Overlay',
             fontsize=14, fontweight='bold', y=1.005)
fig.tight_layout()
save(fig, 'figC_arjun_all5')

# ═══════════════════════════════════════════════════════════════════════
# FIGURES D1-D8: Ramulamma per-frame 4-panel
# ═══════════════════════════════════════════════════════════════════════
print('\nFigures D – Ramulamma per-frame panels...')
for key in sorted(r_pp.keys()):
    proj  = r_proj_map.get(key)
    if proj is None:
        continue
    pd = r_pp[key]
    rng = np.random.default_rng(42 + proj.proj_index)
    dr  = rng.uniform(-PERTURB_ROT_DEG,   PERTURB_ROT_DEG,   3)
    dt  = rng.uniform(-PERTURB_TRANS_MM,  PERTURB_TRANS_MM,  3)
    R_gt, t_gt   = proj.R_proj.copy(), proj.t_proj.copy()
    R_in, t_in   = perturb_extrinsic(R_gt, t_gt, dr, dt)
    bd            = np.array(pd.get('best_pose_delta', [0]*6))
    R_fn, t_fn    = perturb_extrinsic(R_gt, t_gt, bd[:3], bd[3:])

    xray_v = cv2.resize(proj.image_raw, (R_VIS, R_VIS), interpolation=cv2.INTER_AREA)
    drr_i  = render_drr(r_drr, R_in, t_in, R_VIS, R_PIX_MM, R_STEPS)
    drr_f  = render_drr(r_drr, R_fn, t_fn, R_VIS, R_PIX_MM, R_STEPS)

    # Ramulamma uses simple centroid projection (no gt_landmarks_2d)
    u_i, v_i = project_lm_ramu(pts3d_r, R_in, t_in, R_VIS)
    u_f, v_f = project_lm_ramu(pts3d_r, R_fn, t_fn, R_VIS)
    ovl       = checkerboard(norm_img(xray_v), drr_f)

    panels = [
        (xray_v,  f'X-ray  Frame {int(key)}'),
        (drr_i,   f'DRR @ init\nGO={pd["initial_go"]:.4f}'),
        (drr_f,   f'DRR @ final\nGO={pd["final_go"]:.4f}  {"✓" if pd["success"] else "✗"}'),
        (ovl,     f'Checkerboard\nΔGO={pd["go_delta"]:+.4f}'),
    ]
    R_ls = [R_in, R_in, R_fn, R_fn]
    t_ls = [t_in, t_in, t_fn, t_fn]
    bclrs = [C_BLUE, C_ORG,
             C_OK if pd['success'] else C_FAIL,
             C_OK if pd['success'] else C_FAIL]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, (img, subtitle), R_u, t_u, bclr in zip(axes, panels, R_ls, t_ls, bclrs):
        ax.imshow(norm_img(img), cmap='gray', vmin=0, vmax=1)
        u_p, v_p = project_lm_ramu(pts3d_r, R_u, t_u, R_VIS)
        for i, lname in enumerate(lm_names_r):
            ux, vy = u_p[i], v_p[i]
            if 0 <= ux < R_VIS and 0 <= vy < R_VIS:
                c = LM_CMAP[i % len(LM_CMAP)]
                ax.plot(ux, vy, '+', color=c, ms=12, mew=2.5,
                        path_effects=pe_stroke(3, 'black'), zorder=6)
                ax.text(ux+3, vy+3, lname, color=c, fontsize=8.5, fontweight='bold',
                        path_effects=pe_stroke(1.8, 'black'), zorder=7)
        ax.set_xlim(0, R_VIS); ax.set_ylim(R_VIS, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(subtitle, fontsize=9.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(bclr); sp.set_linewidth(3)
    fig.suptitle(
        f'Ramulamma – Frame {int(key)} | '
        f'GO: {pd["initial_go"]:.4f}→{pd["final_go"]:.4f} (Δ={pd["go_delta"]:+.4f}) | '
        f'{"SUCCESS" if pd["success"] else "FAIL"}',
        fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, f'figD_ramulamma_frame_{key}')

# ═══════════════════════════════════════════════════════════════════════
# FIGURE E: All 8 Ramulamma frames in a grid
# ═══════════════════════════════════════════════════════════════════════
print('\nFigure E – Ramulamma all-8 grid...')
keys_r = sorted(r_pp.keys())
fig, axes = plt.subplots(8, 4, figsize=(22, 44))
col_titles = ['X-ray (real)', 'DRR @ init pose', 'DRR @ final pose', 'Checkerboard overlay']
for ax, t in zip(axes[0], col_titles):
    ax.set_title(t, fontsize=12, fontweight='bold', pad=8)

for row, key in enumerate(keys_r):
    proj = r_proj_map.get(key)
    if proj is None:
        for ax in axes[row]: ax.axis('off')
        continue
    pd  = r_pp[key]
    rng = np.random.default_rng(42 + proj.proj_index)
    dr  = rng.uniform(-PERTURB_ROT_DEG,   PERTURB_ROT_DEG,   3)
    dt  = rng.uniform(-PERTURB_TRANS_MM,  PERTURB_TRANS_MM,  3)
    R_gt, t_gt = proj.R_proj.copy(), proj.t_proj.copy()
    R_in, t_in = perturb_extrinsic(R_gt, t_gt, dr, dt)
    bd         = np.array(pd.get('best_pose_delta', [0]*6))
    R_fn, t_fn = perturb_extrinsic(R_gt, t_gt, bd[:3], bd[3:])

    xray_v = cv2.resize(proj.image_raw, (R_VIS, R_VIS), interpolation=cv2.INTER_AREA)
    drr_i  = render_drr(r_drr, R_in, t_in, R_VIS, R_PIX_MM, R_STEPS)
    drr_f  = render_drr(r_drr, R_fn, t_fn, R_VIS, R_PIX_MM, R_STEPS)
    ovl    = checkerboard(norm_img(xray_v), drr_f)

    imgs   = [xray_v, drr_i, drr_f, ovl]
    R_ls   = [R_in, R_in, R_fn, R_fn]
    t_ls   = [t_in, t_in, t_fn, t_fn]
    bclrs  = [C_BLUE, C_ORG,
              C_OK if pd['success'] else C_FAIL,
              C_OK if pd['success'] else C_FAIL]
    subs   = [
        f'Frame {int(key)}',
        f'Init GO={pd["initial_go"]:.4f}',
        f'Final GO={pd["final_go"]:.4f}  {"✓" if pd["success"] else "✗"}',
        f'Δ GO={pd["go_delta"]:+.4f}',
    ]
    for col, (ax, img, R_u, t_u, bclr, sub) in enumerate(
            zip(axes[row], imgs, R_ls, t_ls, bclrs, subs)):
        ax.imshow(norm_img(img), cmap='gray', vmin=0, vmax=1)
        u_p, v_p = project_lm_ramu(pts3d_r, R_u, t_u, R_VIS)
        for i, lname in enumerate(lm_names_r):
            ux, vy = u_p[i], v_p[i]
            if 0 <= ux < R_VIS and 0 <= vy < R_VIS:
                c = LM_CMAP[i % len(LM_CMAP)]
                ax.plot(ux, vy, '+', color=c, ms=11, mew=2,
                        path_effects=pe_stroke(2.5, 'black'), zorder=6)
        ax.set_xlim(0, R_VIS); ax.set_ylim(R_VIS, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(sub, fontsize=8.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(bclr); sp.set_linewidth(2.5)

fig.suptitle('Ramulamma – All 8 Projections: X-ray | DRR(init) | DRR(final) | Checkerboard',
             fontsize=14, fontweight='bold', y=1.002)
fig.tight_layout()
save(fig, 'figE_ramulamma_all8')

# ═══════════════════════════════════════════════════════════════════════
# FIGURE F: GO convergence – Arjun (bar + ΔGO)
# ═══════════════════════════════════════════════════════════════════════
print('\nFigure F – Arjun GO convergence...')
keys = sorted(a_pp.keys())
x = np.arange(len(keys)); w = 0.32

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
# Left: initial vs final bars
ax = axes[0]
i_go = [a_pp[k]['initial_go']  for k in keys]
f_go = [a_pp[k]['final_go']    for k in keys]
succ = [a_pp[k]['success']     for k in keys]
b1 = ax.bar(x - w/2, i_go, w, color=C_INIT, label='Initial GO', alpha=0.9, edgecolor='k', lw=0.7)
b2 = ax.bar(x + w/2, f_go, w,
            color=[C_OK if s else C_FAIL for s in succ],
            alpha=0.9, edgecolor='k', lw=0.7, label='Final GO')
ax.axhline(0.60, color='darkorange', ls='--', lw=1.5, label='Threshold (GO<0.60)')
ax.axhline(0.50, color='grey', ls=':', lw=1, label='Random baseline (~0.50)')
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
            f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#555')
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
            f'{bar.get_height():.3f}', ha='center', fontsize=8,
            color=C_OK if bar.get_height() < 0.60 else C_FAIL, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([f'Frame {k.upper()}' for k in keys], fontsize=10)
ax.set_ylabel('GO Dissimilarity  (↓ = better)', fontsize=11)
ax.set_title('Arjun: Initial vs Final GO Score', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.08); ax.grid(axis='y', alpha=0.35)
ax.legend(fontsize=9)

# Right: ΔGO
ax2 = axes[1]
dgo = [a_pp[k]['go_delta'] for k in keys]
bars = ax2.bar(x, dgo, color=[C_OK if s else C_FAIL for s in succ],
               alpha=0.9, edgecolor='k', lw=0.7)
ax2.axhline(0.05, color='darkorange', ls='--', lw=1.5, label='Min ΔGO threshold (0.05)')
ax2.axhline(0, color='black', lw=0.8)
for bar, val in zip(bars, dgo):
    yp = val + 0.003 if val >= 0 else val - 0.008
    ax2.text(bar.get_x()+bar.get_width()/2, yp, f'{val:+.4f}',
             ha='center', fontsize=8.5, fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels([f'Frame {k.upper()}' for k in keys], fontsize=10)
ax2.set_ylabel('ΔGO  (Init−Final, ↑ = better)', fontsize=11)
ax2.set_title('Arjun: GO Improvement per Frame', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.35)
ax2.legend(fontsize=9)
ok_p = mpatches.Patch(color=C_OK,   label='SUCCESS'); fl_p = mpatches.Patch(color=C_FAIL, label='FAIL')
ax2.legend(handles=[ok_p, fl_p], fontsize=9, loc='upper right')
fig.suptitle('Arjun – GO Convergence Analysis (5/5 successful)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
save(fig, 'figF_go_convergence_arjun')

# ═══════════════════════════════════════════════════════════════════════
# FIGURE G: GO convergence – Ramulamma
# ═══════════════════════════════════════════════════════════════════════
print('\nFigure G – Ramulamma GO convergence...')
keys_r2 = sorted(r_pp.keys())
xr = np.arange(len(keys_r2))

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
ax = axes[0]
i_go_r = [r_pp[k]['initial_go'] for k in keys_r2]
f_go_r = [r_pp[k]['final_go']   for k in keys_r2]
succ_r = [r_pp[k]['success']    for k in keys_r2]
b1 = ax.bar(xr - w/2, i_go_r, w, color=C_INIT, alpha=0.9, edgecolor='k', lw=0.7, label='Initial GO')
b2 = ax.bar(xr + w/2, f_go_r, w,
            color=[C_OK if s else C_FAIL for s in succ_r],
            alpha=0.9, edgecolor='k', lw=0.7, label='Final GO')
ax.axhline(0.60, color='darkorange', ls='--', lw=1.5, label='Threshold (GO<0.60)')
ax.axhline(0.50, color='grey', ls=':', lw=1, label='Random baseline')
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
            f'{bar.get_height():.3f}', ha='center', fontsize=7.5, color='#555')
for bar, s in zip(b2, succ_r):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
            f'{bar.get_height():.3f}', ha='center', fontsize=7.5,
            color=C_OK if s else C_FAIL, fontweight='bold')
ax.set_xticks(xr); ax.set_xticklabels([f'Frame\n{int(k)}' for k in keys_r2], fontsize=9)
ax.set_ylabel('GO Dissimilarity  (↓ = better)', fontsize=11)
ax.set_title('Ramulamma: Initial vs Final GO Score', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.15); ax.grid(axis='y', alpha=0.35)
ax.legend(fontsize=9)

ax2 = axes[1]
dgo_r = [r_pp[k]['go_delta'] for k in keys_r2]
bars  = ax2.bar(xr, dgo_r, color=[C_OK if s else C_FAIL for s in succ_r],
                alpha=0.9, edgecolor='k', lw=0.7)
ax2.axhline(0.05, color='darkorange', ls='--', lw=1.5, label='Min ΔGO threshold (0.05)')
ax2.axhline(0, color='black', lw=0.8)
for bar, val in zip(bars, dgo_r):
    yp = val + 0.008 if val >= 0 else val - 0.022
    ax2.text(bar.get_x()+bar.get_width()/2, yp, f'{val:+.3f}',
             ha='center', fontsize=8, fontweight='bold')
ax2.set_xticks(xr); ax2.set_xticklabels([f'Frame\n{int(k)}' for k in keys_r2], fontsize=9)
ax2.set_ylabel('ΔGO  (Init−Final, ↑ = better)', fontsize=11)
ax2.set_title('Ramulamma: GO Improvement per Frame', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.35)
ok_p = mpatches.Patch(color=C_OK, label='SUCCESS'); fl_p = mpatches.Patch(color=C_FAIL, label='FAIL')
ax2.legend(handles=[ok_p, fl_p], fontsize=9)
fig.suptitle('Ramulamma – GO Convergence Analysis (5/8 successful)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
save(fig, 'figG_go_convergence_ramu')

# ═══════════════════════════════════════════════════════════════════════
# FIGURE H: ΔGO violin + scatter
# ═══════════════════════════════════════════════════════════════════════
print('\nFigure H – ΔGO violin...')
a_dg = [a_pp[k]['go_delta'] for k in a_pp]
r_dg = [r_pp[k]['go_delta'] for k in r_pp]
fig, ax = plt.subplots(figsize=(8, 6))
vp = ax.violinplot([a_dg, r_dg], positions=[1, 2], showmedians=True, showextrema=True)
for pc, c in zip(vp['bodies'], [C_BLUE, C_ORG]):
    pc.set_facecolor(c); pc.set_alpha(0.6)
for key in ('cmedians','cmaxes','cmins','cbars'):
    vp[key].set_color('black')
ax.scatter(np.ones(len(a_dg)) + np.random.normal(0, 0.03, len(a_dg)), a_dg,
           color=C_BLUE, s=80, zorder=6, edgecolors='k', lw=0.8, label='Arjun frames')
ax.scatter(np.ones(len(r_dg))*2 + np.random.normal(0, 0.03, len(r_dg)), r_dg,
           color=C_ORG,  s=80, zorder=6, edgecolors='k', lw=0.8, label='Ramulamma frames')
for i, (k, v) in enumerate(a_pp.items()):
    ax.text(1 + 0.15, v['go_delta'], k.upper(), fontsize=8, va='center', color=C_BLUE)
for i, (k, v) in enumerate(r_pp.items()):
    ax.text(2 + 0.12, v['go_delta'], str(int(k)), fontsize=8, va='center', color=C_ORG)
ax.axhline(0.05, color='darkorange', ls='--', lw=1.5, label='Success threshold (ΔGO=0.05)')
ax.axhline(0,    color='black',      ls='-',  lw=0.8)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Arjun\n(n=5, 100% success)', 'Ramulamma\n(n=8, 62.5% success)'], fontsize=12)
ax.set_ylabel('ΔGO  (Init − Final,  ↑ = better)', fontsize=12)
ax.set_title('Distribution of GO Improvement per Patient', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.35)
fig.tight_layout()
save(fig, 'figH_go_delta_violin')

# ═══════════════════════════════════════════════════════════════════════
# FIGURE I: PDE per landmark – Arjun
# ═══════════════════════════════════════════════════════════════════════
print('\nFigure I – PDE per landmark...')
keys_a2 = sorted(a_pp.keys())
lm_all  = sorted({lm for k in keys_a2 for lm in a_pp[k].get('pde_per_landmark', {})})
xa2 = np.arange(len(keys_a2))
bar_w = 0.14
pal = ['#2980b9','#27ae60','#e67e22','#e74c3c','#8e44ad']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
for i, lm in enumerate(lm_all):
    vals_i = [a_pp[k]['pde_per_landmark'].get(lm, np.nan) for k in keys_a2]
    offset = (i - len(lm_all)/2 + 0.5) * bar_w
    ax.bar(xa2 + offset, vals_i, bar_w, label=lm, color=pal[i],
           edgecolor='k', lw=0.5, alpha=0.9, zorder=3)
ax.axhline(5.0, color='red', ls='--', lw=1.5, label='5 mm clinical target')
ax.set_xticks(xa2); ax.set_xticklabels([f'Frame {k.upper()}' for k in keys_a2], fontsize=10)
ax.set_ylabel('PDE (mm)', fontsize=11)
ax.set_title('Per-Landmark PDE at Final Pose', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.35); ax.set_ylim(0, 17)

# Right: mean PDE init vs final per frame
ax2 = axes[1]
i_pde = [a_pp[k]['initial_pde_mm'] for k in keys_a2]
f_pde = [a_pp[k]['final_pde_mm']   for k in keys_a2]
ax2.bar(xa2 - 0.2, i_pde, 0.38, color='#95a5a6', edgecolor='k', lw=0.7, label='Initial mean PDE', alpha=0.9)
ax2.bar(xa2 + 0.2, f_pde, 0.38,
        color=[C_OK if f <= i else C_FAIL for i, f in zip(i_pde, f_pde)],
        edgecolor='k', lw=0.7, label='Final mean PDE', alpha=0.9)
ax2.axhline(5.0, color='red', ls='--', lw=1.5, label='5 mm clinical target')
for xi, (ip, fp) in enumerate(zip(i_pde, f_pde)):
    ax2.text(xi - 0.2, ip + 0.3, f'{ip:.2f}', ha='center', fontsize=8, color='#555')
    ax2.text(xi + 0.2, fp + 0.3, f'{fp:.2f}', ha='center', fontsize=8, fontweight='bold',
             color=C_OK if fp <= ip else C_FAIL)
ax2.set_xticks(xa2); ax2.set_xticklabels([f'Frame {k.upper()}' for k in keys_a2], fontsize=10)
ax2.set_ylabel('Mean PDE (mm)', fontsize=11)
ax2.set_title('Mean PDE: Initial vs Final', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.35); ax2.set_ylim(0, 17)
fig.suptitle('Arjun – Projection Distance Error (PDE) Analysis',
             fontsize=13, fontweight='bold')
fig.tight_layout()
save(fig, 'figI_pde_arjun')

# ═══════════════════════════════════════════════════════════════════════
# FIGURE J: Success rate summary
# ═══════════════════════════════════════════════════════════════════════
print('\nFigure J – Success rate...')
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
cats   = ['Arjun\n(5 frames)', 'Ramulamma\n(8 frames)', 'Pooled\n(13 frames)']
sr     = [5/5*100, 5/8*100, 10/13*100]
counts = ['5/5','5/8','10/13']
colors = [C_BLUE, C_ORG, '#27ae60']
ax = axes[0]
bars = ax.bar(cats, sr, color=colors, width=0.5, edgecolor='k', lw=0.8, zorder=3)
for bar, pct, cnt in zip(bars, sr, counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
            f'{pct:.1f}%\n({cnt})', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_ylim(0, 118); ax.grid(axis='y', alpha=0.35)
ax.set_title('Registration Success Rate', fontsize=12, fontweight='bold')

# Right: final GO distribution dot plot
ax2 = axes[1]
a_fg = [a_pp[k]['final_go'] for k in a_pp]
r_fg = [r_pp[k]['final_go'] for k in r_pp]
a_succ = [a_pp[k]['success'] for k in a_pp]
r_succ = [r_pp[k]['success'] for k in r_pp]
jit_a = np.random.default_rng(0).normal(0, 0.05, len(a_fg))
jit_r = np.random.default_rng(1).normal(0, 0.05, len(r_fg))
ax2.scatter(np.ones(len(a_fg)) + jit_a, a_fg,
            c=[C_OK if s else C_FAIL for s in a_succ], s=120, zorder=5, edgecolors='k', lw=0.8)
ax2.scatter(np.ones(len(r_fg))*2 + jit_r, r_fg,
            c=[C_OK if s else C_FAIL for s in r_succ], s=120, zorder=5, edgecolors='k', lw=0.8)
ax2.axhline(0.60, color='darkorange', ls='--', lw=1.5, label='Success threshold (GO<0.60)')
ax2.set_xticks([1, 2]); ax2.set_xticklabels(['Arjun (n=5)', 'Ramulamma (n=8)'], fontsize=11)
ax2.set_ylabel('Final GO Score', fontsize=11)
ax2.set_title('Final GO Distribution', fontsize=12, fontweight='bold')
ok_p = mpatches.Patch(color=C_OK, label='SUCCESS'); fl_p = mpatches.Patch(color=C_FAIL, label='FAIL')
ax2.legend(handles=[ok_p, fl_p], fontsize=9); ax2.grid(axis='y', alpha=0.35)
fig.suptitle('Registration Performance Summary', fontsize=13, fontweight='bold')
fig.tight_layout()
save(fig, 'figJ_success_rate')

# ═══════════════════════════════════════════════════════════════════════
# FIGURE K: Runtime
# ═══════════════════════════════════════════════════════════════════════
print('\nFigure K – Runtime...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (lbl, pp, keys_u, color, hw) in zip(axes, [
        ('Arjun (GPU – RTX 2080 Ti)', a_pp, sorted(a_pp.keys()), C_BLUE, 'GPU'),
        ('Ramulamma (CPU only)',       r_pp, sorted(r_pp.keys()), C_ORG,  'CPU')]):
    rts  = [pp[k]['runtime_s'] for k in keys_u]
    succ = [pp[k]['success']   for k in keys_u]
    xpos = np.arange(len(keys_u))
    bars = ax.bar(xpos, rts, color=[C_OK if s else C_FAIL for s in succ],
                  edgecolor='k', lw=0.7, alpha=0.9, zorder=3)
    for bar, val in zip(bars, rts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f'{val:.0f}s', ha='center', fontsize=9, fontweight='bold')
    ax.axhline(np.mean(rts), color='navy', ls='--', lw=1.5,
               label=f'Mean: {np.mean(rts):.1f}s')
    ax.set_xticks(xpos)
    ax.set_xticklabels(
        [f'Frame {k.upper()}' if len(k) == 1 else f'Frame {int(k)}' for k in keys_u],
        fontsize=9)
    ax.set_ylabel('Runtime (seconds)', fontsize=11)
    ax.set_title(lbl, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.35); ax.legend(fontsize=9)
    ok_p = mpatches.Patch(color=C_OK, label='SUCCESS')
    fl_p = mpatches.Patch(color=C_FAIL, label='FAIL')
    ax.legend(handles=[ok_p, fl_p, plt.Line2D([0],[0],color='navy',ls='--',
              label=f'Mean {np.mean(rts):.1f}s')], fontsize=8.5)
fig.suptitle('Per-Projection Runtime  (Arjun: GPU 20× faster than Ramulamma: CPU)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
save(fig, 'figK_runtime')

# ═══════════════════════════════════════════════════════════════════════
# FIGURE L: Checkerboard overlay collage
# ═══════════════════════════════════════════════════════════════════════
print('\nFigure L – Checkerboard collage...')
fig, axes = plt.subplots(3, 5, figsize=(24, 15))
axes = axes.flat
idx = 0
for key in sorted(a_pp.keys()):
    proj = a_proj_map[key]
    pd   = a_pp[key]
    bd   = np.array(pd.get('best_pose_delta', [0]*6))
    R_e, t_e = proj.R_proj.copy(), proj.t_proj.copy()
    R_f, t_f = perturb_extrinsic(R_e, t_e, bd[:3], bd[3:])
    xray_v = cv2.resize(proj.image_raw, (A_VIS, A_VIS), interpolation=cv2.INTER_AREA)
    drr_f  = render_drr(a_drr, R_f, t_f, A_VIS, A_PIX_MM, A_STEPS)
    ovl    = checkerboard(norm_img(xray_v), drr_f)
    ax = axes[idx]; idx += 1
    ax.imshow(ovl, cmap='gray', vmin=0, vmax=1)
    u_f, v_f = project_lm(proj, pts3d_a, R_f, t_f, A_VIS)
    for i, lname in enumerate(lm_names_a):
        ux, vy = u_f[i], v_f[i]
        if 0 <= ux < A_VIS and 0 <= vy < A_VIS:
            c = LM_CMAP[i % len(LM_CMAP)]
            ax.plot(ux, vy, '+', color=c, ms=14, mew=2.5,
                    path_effects=pe_stroke(3, 'black'), zorder=6)
            ax.text(ux+3, vy+3, lname, color=c, fontsize=8.5, fontweight='bold',
                    path_effects=pe_stroke(1.8, 'black'), zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'Arjun – {key.upper()}', fontsize=10, fontweight='bold')
    ax.set_xlabel(f'GO={pd["final_go"]:.4f}  {"✓" if pd["success"] else "✗"}', fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_OK if pd['success'] else C_FAIL); sp.set_linewidth(3)

for key in sorted(r_pp.keys()):
    proj = r_proj_map.get(key)
    if proj is None: continue
    pd  = r_pp[key]
    rng = np.random.default_rng(42 + proj.proj_index)
    dr  = rng.uniform(-PERTURB_ROT_DEG, PERTURB_ROT_DEG, 3)
    dt  = rng.uniform(-PERTURB_TRANS_MM, PERTURB_TRANS_MM, 3)
    R_gt, t_gt = proj.R_proj.copy(), proj.t_proj.copy()
    bd = np.array(pd.get('best_pose_delta', [0]*6))
    R_fn, t_fn = perturb_extrinsic(R_gt, t_gt, bd[:3], bd[3:])
    xray_v = cv2.resize(proj.image_raw, (R_VIS, R_VIS), interpolation=cv2.INTER_AREA)
    drr_f  = render_drr(r_drr, R_fn, t_fn, R_VIS, R_PIX_MM, R_STEPS)
    ovl    = checkerboard(norm_img(xray_v), drr_f)
    ax = axes[idx]; idx += 1
    ax.imshow(ovl, cmap='gray', vmin=0, vmax=1)
    u_f, v_f = project_lm_ramu(pts3d_r, R_fn, t_fn, R_VIS)
    for i, lname in enumerate(lm_names_r):
        ux, vy = u_f[i], v_f[i]
        if 0 <= ux < R_VIS and 0 <= vy < R_VIS:
            c = LM_CMAP[i % len(LM_CMAP)]
            ax.plot(ux, vy, '+', color=c, ms=12, mew=2.5,
                    path_effects=pe_stroke(2.5, 'black'), zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'Ramulamma – {int(key)}', fontsize=10, fontweight='bold')
    ax.set_xlabel(f'GO={pd["final_go"]:.4f}  {"✓" if pd["success"] else "✗"}', fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_OK if pd['success'] else C_FAIL); sp.set_linewidth(3)

while idx < len(list(axes)):
    axes[idx].axis('off'); idx += 1

fig.suptitle('Checkerboard Overlay: X-ray ↔ DRR at Final Pose — All 13 Projections',
             fontsize=14, fontweight='bold')
fig.tight_layout()
save(fig, 'figL_checkerboard_all')

print('\n✓ All manuscript figures generated in:', OUT)
