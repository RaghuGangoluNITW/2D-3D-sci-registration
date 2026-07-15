"""
visualize_swaroopa_default_pose.py
====================================
Renders one AP DRR and one Lateral DRR for Swaroopa using the EPnP
initial pose (solved from 2D landmark annotations), overlays L1–L5
projected centroids and the GT 2D annotations from the corresponding
X-ray frames (ap_013 for AP, lat_003 for lateral).

Output: results/figures/swaroopa_initial_pose_drr.png
"""

import sys
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
from swaroopa_loader import (
    _load_mrk_json_3d, _load_landmarks_2d, _load_png,
    _anatomy_pose, _solve_pnp_swaro, _reproj_error_swaro,
    LM_3D_JSON, XRAY_DIR_AP, XRAY_DIR_LAT, LM_2D_JSON,
    CT_DICOM_DIR,
    SWARO_PIX_MM, SWARO_FX, SWARO_FY, SWARO_CX, SWARO_CY,
    SWARO_SID_MM, SWARO_IMG_SIZE,
    project_world_swaro,
)
from deepfluoro_loader import DeepFluoroSpecimen
from deepfluoro_drr import DeepFluoroDRR
import SimpleITK as sitk

# ── Frames to show ─────────────────────────────────────────────────────────
FRAMES = [
    dict(view='AP',      ap_lat='ap',  frame_idx='013', azimuth=0.,  az_label='AP (az=0°)'),
    dict(view='Lateral', ap_lat='lat', frame_idx='003', azimuth=90., az_label='Lateral (az=90°)'),
]

LM_ORDER = ['L1', 'L2', 'L3', 'L4', 'L5']
COLORS   = {'L1':'#FF4444','L2':'#FF8800','L3':'#FFEE00','L4':'#44CC44','L5':'#4488FF'}

# ── Load CT ────────────────────────────────────────────────────────────────
print("Loading CT (DICOM) ...")
reader = sitk.ImageSeriesReader()
reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(CT_DICOM_DIR)))
img_itk = reader.Execute()
ct_vol  = sitk.GetArrayFromImage(img_itk).astype(np.float32)
spacing = np.array(img_itk.GetSpacing(), dtype=np.float64)
origin  = np.array(img_itk.GetOrigin(),  dtype=np.float64)
print(f"  shape={ct_vol.shape}  spacing={spacing.round(3)}")

# ── Landmarks ──────────────────────────────────────────────────────────────
lm_3d     = _load_mrk_json_3d(LM_3D_JSON)
lm_2d_all = _load_landmarks_2d(LM_2D_JSON)
print(f"  3D landmarks: {sorted(lm_3d.keys())}")

# ── DRR renderer ───────────────────────────────────────────────────────────
print("Building DRR renderer ...")
spec = DeepFluoroSpecimen(
    specimen_id='swaroopa', ct_volume=ct_vol, ct_spacing=spacing,
    ct_origin=origin, landmarks_3d=lm_3d, projections=[],
)
renderer = DeepFluoroDRR(spec)

def render_drr(R, t):
    torch.cuda.empty_cache()
    drr = renderer.generate_from_extrinsic(
        R_proj=R, t_proj=t,
        output_size=SWARO_IMG_SIZE,
        pixel_spacing_mm=SWARO_PIX_MM,
        sdd_mm=SWARO_SID_MM,
        n_steps=400,
    )
    drr = drr.astype(np.float32)
    lo, hi = drr.min(), drr.max()
    return (drr - lo) / (hi - lo) if hi > lo else drr

# ── Figure: 2 rows × 2 cols (X-ray | DRR) ─────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
fig.patch.set_facecolor('#111111')

for row, cfg in enumerate(FRAMES):
    view      = cfg['view']
    frame_idx = cfg['frame_idx']
    azimuth   = cfg['azimuth']
    az_label  = cfg['az_label']
    ap_lat    = cfg['ap_lat']

    xray_dir = XRAY_DIR_AP if ap_lat == 'ap' else XRAY_DIR_LAT
    png_path = xray_dir / f'frame_{frame_idx}_z000.png'
    json_key = f'frame_{frame_idx}_z00'

    xray  = _load_png(png_path)                     # float32 [0,1]
    lm_2d = lm_2d_all.get(json_key, {})

    # EPnP initial pose from GT 2D annotations
    common    = [l for l in LM_ORDER if l in lm_3d and l in lm_2d]
    pts3d_epnp = np.array([lm_3d[l] for l in common])
    pts2d_epnp = np.array([lm_2d[l] for l in common])
    R, t = _solve_pnp_swaro(pts3d_epnp, pts2d_epnp)

    # Project all 5 landmarks
    all_pts3d = np.array([lm_3d[l] for l in LM_ORDER if l in lm_3d])
    all_names = [l for l in LM_ORDER if l in lm_3d]
    all_uv    = project_world_swaro(all_pts3d, R, t)

    # GT 2D annotations available for this frame (same as EPnP landmarks)
    annotated = common
    pts2d_gt  = np.array([lm_2d[l] for l in annotated]) if annotated else np.zeros((0,2))
    pts2d_pr  = project_world_swaro(
        np.array([lm_3d[l] for l in annotated]), R, t
    ) if annotated else np.zeros((0,2))

    reproj_px_vals = np.linalg.norm(pts2d_pr - pts2d_gt, axis=1) if len(annotated) else []
    reproj_px = float(np.mean(reproj_px_vals)) if len(reproj_px_vals) else 0.
    reproj_mm = reproj_px * SWARO_PIX_MM

    print(f"\n{az_label}  frame_{frame_idx}  EPnP "
          f"reproj={reproj_px:.1f}px ({reproj_mm:.1f}mm)  "
          f"lm used={annotated}")

    # Render DRR
    print(f"  Rendering DRR ...")
    drr = render_drr(R, t)

    def draw_panel(ax, img, title):
        ax.set_facecolor('#111111')
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='upper',
                  interpolation='bilinear',
                  extent=[0, SWARO_IMG_SIZE, SWARO_IMG_SIZE, 0])

        # Projected centroids
        for i, lm_name in enumerate(all_names):
            u, v = all_uv[i]
            in_f = 0 <= u < SWARO_IMG_SIZE and 0 <= v < SWARO_IMG_SIZE
            mk   = 'o' if lm_name in annotated else 's'
            ax.plot(u, v, mk, color=COLORS[lm_name],
                    markersize=11, markeredgecolor='white', markeredgewidth=1.5,
                    zorder=5, alpha=1.0 if in_f else 0.25)
            if in_f:
                ax.annotate(lm_name, (u, v),
                            xytext=(10, 0), textcoords='offset points',
                            color=COLORS[lm_name], fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.15', fc='#000000cc', lw=0))

        # GT annotation crosses + reprojection error lines
        for lm_name, uv_gt, uv_p in zip(annotated, pts2d_gt, pts2d_pr):
            ax.plot(*uv_gt, '+', color=COLORS[lm_name],
                    markersize=13, markeredgewidth=2.2, zorder=6)
            ax.plot([uv_gt[0], uv_p[0]], [uv_gt[1], uv_p[1]], '-',
                    color=COLORS[lm_name], linewidth=1.4, alpha=0.8, zorder=4)

        ax.set_xlim(0, SWARO_IMG_SIZE); ax.set_ylim(SWARO_IMG_SIZE, 0)
        ax.set_title(title, color='white', fontsize=9, pad=6)
        ax.axis('off')

    draw_panel(axes[row, 0], xray,
               f'{az_label}  —  X-ray (frame_{frame_idx})\n'
               f'{len(annotated)} GT annotations  |  pix={SWARO_PIX_MM:.3f}mm')
    draw_panel(axes[row, 1], drr,
               f'{az_label}  —  DRR (EPnP initial pose)\n'
               f'Reproj: {reproj_px:.1f}px = {reproj_mm:.1f}mm  |  Fx={SWARO_FX:.0f}px')

# ── Legend ─────────────────────────────────────────────────────────────────
lm_patches = [mpatches.Patch(facecolor=COLORS[l], edgecolor='white', label=l)
              for l in LM_ORDER if l in lm_3d]
sym = [
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#888',
               markeredgecolor='white', markersize=9, linestyle='none',
               label='● Projected centroid (annotated)'),
    plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='#888',
               markeredgecolor='white', markersize=9, linestyle='none',
               label='■ Projected centroid (no 2D label)'),
    plt.Line2D([0],[0], marker='+', color='#aaa', markersize=12,
               markeredgewidth=2, linestyle='none', label='+ GT annotation'),
    plt.Line2D([0],[0], color='#aaa', linewidth=1.5, label='— Reprojection error'),
]
fig.legend(handles=lm_patches + sym,
           loc='lower center', ncol=9, fontsize=8,
           framealpha=0.8, facecolor='#222222', labelcolor='white',
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle(
    'Swaroopa — AP & Lateral DRR (EPnP initial pose)  |  '
    f'pix={SWARO_PIX_MM:.3f}mm  Fx={SWARO_FX:.0f}px  SID={SWARO_SID_MM:.0f}mm\n'
    '● projected centroid  |  + GT 2D annotation  |  lines = reprojection error',
    color='white', fontsize=11, fontweight='bold',
)

out = ROOT / 'results/figures/swaroopa_initial_pose_drr.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#111111')
plt.close(fig)
print(f"\nSaved: {out}")
