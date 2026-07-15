"""
visualize_arjun_drr_centroids.py
=================================
Renders a DRR for one AP frame (b) and one oblique/lateral-like frame (e)
for Arjun, overlaying the projected L1–L5 vertebral centroids and GT 2D
annotations on both the original X-ray and the DRR.

Output: results/figures/arjun_drr_centroids.png
"""

import sys, json
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from arjun_loader import (
    _load_mrk_json_3d, _make_K_mat, _make_K, pixel_spacing_mm,
    LM_3D_JSON, XRAY_DIR, ARJUN_SID_MM,
)
from deepfluoro_loader import DeepFluoroSpecimen, xzy
from deepfluoro_drr import DeepFluoroDRR
import SimpleITK as sitk

# ── Config ─────────────────────────────────────────────────────────────────
FRAMES = {
    'AP (b)':              'b',
    'Oblique / Lat-like (e)': 'e',
}
LM_ORDER = ['L1', 'L2', 'L3', 'L4', 'L5']
COLORS   = {'L1':'#FF4444','L2':'#FF8800','L3':'#FFEE00','L4':'#44CC44','L5':'#4488FF'}

CT_NRRD  = ROOT / 'data/testing/ARJUN PREOP/3 L_Spine  1.0  B60s_3.nrrd'

# ── Load CT ────────────────────────────────────────────────────────────────
print("Loading CT ...")
img_itk = sitk.ReadImage(str(CT_NRRD))
ct_vol  = sitk.GetArrayFromImage(img_itk).astype(np.float32)
spacing = np.array(img_itk.GetSpacing(), dtype=np.float64)
origin  = np.array(img_itk.GetOrigin(),  dtype=np.float64)
print(f"  shape={ct_vol.shape}  spacing={spacing.round(3)}")

# ── 3D landmarks ───────────────────────────────────────────────────────────
lm_3d = _load_mrk_json_3d(LM_3D_JSON)
print(f"  3D landmarks: {sorted(lm_3d.keys())}")

# ── Build DRR renderer ─────────────────────────────────────────────────────
print("Building DRR renderer ...")
spec = DeepFluoroSpecimen(
    specimen_id  = 'arjun',
    ct_volume    = ct_vol,
    ct_spacing   = spacing,
    ct_origin    = origin,
    landmarks_3d = lm_3d,
    projections  = [],
)
renderer = DeepFluoroDRR(spec)

# ── EPnP helper ────────────────────────────────────────────────────────────
def solve_epnp(pts3d_world, pts2d, K):
    n = len(pts3d_world)
    flag = cv2.SOLVEPNP_SQPNP if n == 3 else cv2.SOLVEPNP_EPNP
    _, rvec, tvec = cv2.solvePnP(
        xzy(pts3d_world).astype(np.float64),
        pts2d.astype(np.float64),
        K, np.zeros(4), flags=flag,
    )
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.flatten().astype(np.float64)

def project_pts(pts3d_world, R, t, fx, fy, cx, cy):
    pts_xzy = xzy(np.atleast_2d(pts3d_world))
    P = (R @ pts_xzy.T).T + t
    u = fx * P[:,0] / P[:,2] + cx
    v = fy * P[:,1] / P[:,2] + cy
    return np.stack([u, v], axis=1)

def render_drr(R, t, img_w, img_h):
    """Render at output size matching the X-ray, with correct pixel spacing."""
    pix_mm = pixel_spacing_mm(img_w, img_h)
    out_size = max(img_w, img_h)
    torch.cuda.empty_cache()
    drr = renderer.generate_from_extrinsic(
        R_proj           = R,
        t_proj           = t,
        output_size      = out_size,
        pixel_spacing_mm = pix_mm,
        sdd_mm           = ARJUN_SID_MM,
        n_steps          = 400,
    )
    # Crop/resize to actual image dimensions
    drr = cv2.resize(drr.astype(np.float32), (img_w, img_h),
                     interpolation=cv2.INTER_LINEAR)
    lo, hi = drr.min(), drr.max()
    if hi > lo:
        drr = (drr - lo) / (hi - lo)
    return drr

# ── Build figure: 2 rows (frames) × 2 cols (X-ray | DRR) ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
fig.patch.set_facecolor('#111111')

for row, (frame_label, frame_id) in enumerate(FRAMES.items()):
    jpg_path  = XRAY_DIR / f'{frame_id}.jpg'
    json_path = XRAY_DIR / f'{frame_id}.json'

    xray = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img_h, img_w = xray.shape

    # Load 2D annotations
    ann = json.load(open(json_path))
    lm_2d = {s['label']: np.array(s['points'][0])
              for s in ann.get('shapes', []) if s['shape_type'] == 'point'}

    common = [l for l in LM_ORDER if l in lm_3d and l in lm_2d]
    pts3d  = np.array([lm_3d[l] for l in common])
    pts2d  = np.array([lm_2d[l] for l in common])
    K = _make_K_mat(img_w, img_h)
    fx, fy, cx, cy = _make_K(img_w, img_h)
    pix_mm = pixel_spacing_mm(img_w, img_h)

    R, t = solve_epnp(pts3d, pts2d, K)
    uv_pred = project_pts(pts3d, R, t, fx, fy, cx, cy)
    reproj_px = float(np.sqrt(((uv_pred - pts2d)**2).sum(axis=1)).mean())
    reproj_mm = reproj_px * pix_mm

    all_pts3d = np.array([lm_3d[l] for l in LM_ORDER if l in lm_3d])
    all_names = [l for l in LM_ORDER if l in lm_3d]
    all_uv    = project_pts(all_pts3d, R, t, fx, fy, cx, cy)

    print(f"\n{frame_label} ({frame_id})  {img_w}x{img_h}  "
          f"pix={pix_mm:.3f}mm  Fx={fx:.0f}px")
    print(f"  EPnP reproj: {reproj_px:.2f}px ({reproj_mm:.2f}mm)  "
          f"common lm: {common}")

    # ── Render DRR ─────────────────────────────────────────────────────────
    print(f"  Rendering DRR ...")
    drr = render_drr(R, t, img_w, img_h)

    for col, (img_data, title) in enumerate([
        (xray, f'X-ray  ({frame_id})'),
        (drr,  f'DRR  pix={pix_mm:.3f}mm  Fx={fx:.0f}px\n'
               f'EPnP reproj: {reproj_px:.1f}px = {reproj_mm:.2f}mm'),
    ]):
        ax = axes[row, col]
        ax.set_facecolor('#111111')
        ax.imshow(img_data, cmap='gray', vmin=0, vmax=1,
                  origin='upper', interpolation='bilinear',
                  extent=[0, img_w, img_h, 0])

        # All projected centroids (circles)
        for i, lm_name in enumerate(all_names):
            u, v = all_uv[i]
            in_frame = 0 <= u < img_w and 0 <= v < img_h
            marker = 'o' if lm_name in common else 's'
            ax.plot(u, v, marker,
                    color=COLORS[lm_name],
                    markersize=11, markeredgecolor='white', markeredgewidth=1.5,
                    zorder=5, alpha=1.0 if in_frame else 0.3)
            if in_frame:
                ax.annotate(lm_name, (u, v),
                            xytext=(10, 0), textcoords='offset points',
                            color=COLORS[lm_name], fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.15', fc='#000000cc', lw=0))

        # GT annotation crosses + error lines
        for lm_name, uv_gt, uv_p in zip(common, pts2d, uv_pred):
            ax.plot(*uv_gt, '+', color=COLORS[lm_name],
                    markersize=13, markeredgewidth=2.2, zorder=6)
            ax.plot([uv_gt[0], uv_p[0]], [uv_gt[1], uv_p[1]], '-',
                    color=COLORS[lm_name], linewidth=1.3, alpha=0.75, zorder=4)

        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.set_title(f'{frame_label}\n{title}',
                     color='white', fontsize=9, pad=6)
        ax.axis('off')

# ── Legend ─────────────────────────────────────────────────────────────────
lm_patches = [mpatches.Patch(facecolor=COLORS[l], edgecolor='white', label=l)
              for l in LM_ORDER if l in lm_3d]
sym_handles = [
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#888',
               markeredgecolor='white', markersize=9, label='● Projected centroid (annotated lm)'),
    plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='#888',
               markeredgecolor='white', markersize=9, label='■ Projected centroid (no 2D label)'),
    plt.Line2D([0],[0], marker='+', color='#aaa', markersize=12,
               markeredgewidth=2, linestyle='none', label='+ GT 2D annotation'),
    plt.Line2D([0],[0], color='#aaa', linewidth=1.5, label='— Reprojection error'),
]
fig.legend(handles=lm_patches + sym_handles,
           loc='lower center', ncol=9, fontsize=8,
           framealpha=0.8, facecolor='#222222', labelcolor='white',
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle('Arjun — DRR with Projected L1–L5 Vertebral Centroids\n'
             '● = EPnP-projected  |  + = GT annotation  |  lines = reprojection error',
             color='white', fontsize=12, fontweight='bold')

out_path = ROOT / 'results/figures/arjun_drr_centroids.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#111111')
plt.close(fig)
print(f"\nSaved: {out_path}")
