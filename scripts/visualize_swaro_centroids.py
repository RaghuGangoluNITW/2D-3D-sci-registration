"""
visualize_swaro_centroids.py
============================
Renders one AP frame (ap_013) and one lateral frame (lat_003) as DRRs,
then projects the 5 vertebral centroids (L1–L5) using BOTH pixel spacings
(0.205 mm  vs  0.288 mm) and overlays them on the DRR + original X-ray.

Output: results/figures/swaro_centroid_comparison.png
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

# ── Import loader machinery (but override pixel spacing ourselves) ─────────
import swaroopa_loader as sl
from swaroopa_loader import (
    CT_NRRD, CT_DICOM_DIR, LM_3D_JSON,
    XRAY_DIR_AP, XRAY_DIR_LAT, LM_2D_JSON,
    SWARO_SID_MM, SWARO_IMG_SIZE,
    _load_mrk_json_3d, _load_landmarks_2d, _load_png,
    _solve_pnp_swaro, _reproj_error_swaro,
    xzy,
)
from deepfluoro_drr import DeepFluoroDRR
from deepfluoro_loader import DeepFluoroSpecimen
import SimpleITK as sitk

# ── Camera configs to compare ──────────────────────────────────────────────
CONFIGS = {
    '0.205 mm\n(old assumed)': 0.205,
    '0.288 mm\n(DICOM true)':  0.288,
}

FRAMES = {
    'AP  (ap_013)':  ('ap',  '013', XRAY_DIR_AP),
    'Lat (lat_003)': ('lat', '003', XRAY_DIR_LAT),
}

COLORS = {
    'L1': '#FF4444',
    'L2': '#FF8800',
    'L3': '#FFDD00',
    'L4': '#44CC44',
    'L5': '#4488FF',
}
LM_ORDER = ['L1', 'L2', 'L3', 'L4', 'L5']

# ── Load CT + landmarks once ───────────────────────────────────────────────
print("Loading CT ...")
if CT_NRRD.exists():
    try:
        img = sitk.ReadImage(str(CT_NRRD))
        ct_vol  = sitk.GetArrayFromImage(img).astype(np.float32)
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        origin  = np.array(img.GetOrigin(),  dtype=np.float64)
        print(f"  NRRD loaded: {ct_vol.shape}")
    except RuntimeError:
        img = None

if img is None or not CT_NRRD.exists():
    reader = sitk.ImageSeriesReader()
    names  = reader.GetGDCMSeriesFileNames(str(CT_DICOM_DIR))
    reader.SetFileNames(names)
    img     = reader.Execute()
    ct_vol  = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing = np.array(img.GetSpacing(), dtype=np.float64)
    origin  = np.array(img.GetOrigin(),  dtype=np.float64)
    print(f"  DICOM loaded: {ct_vol.shape}")

try:
    reader = sitk.ImageSeriesReader()
    names  = reader.GetGDCMSeriesFileNames(str(CT_DICOM_DIR))
    reader.SetFileNames(names)
    img_itk = reader.Execute()
    ct_vol  = sitk.GetArrayFromImage(img_itk).astype(np.float32)
    spacing = np.array(img_itk.GetSpacing(), dtype=np.float64)
    origin  = np.array(img_itk.GetOrigin(),  dtype=np.float64)
    print(f"  DICOM CT: shape={ct_vol.shape}  spacing={spacing.round(3)}")
except Exception as e:
    print(f"  CT load error: {e}")
    sys.exit(1)

lm_3d   = _load_mrk_json_3d(LM_3D_JSON)
lm_2d_all = _load_landmarks_2d(LM_2D_JSON)
print(f"  3D landmarks: {sorted(lm_3d.keys())}")

# ── Build DRR renderer ─────────────────────────────────────────────────────
print("Building DRR renderer ...")
spec_for_drr = DeepFluoroSpecimen(
    specimen_id  = 'swaroopa',
    ct_volume    = ct_vol,
    ct_spacing   = spacing,
    ct_origin    = origin,
    landmarks_3d = lm_3d,
    projections  = [],
)
drr_renderer = DeepFluoroDRR(spec_for_drr)

# ── Helper: build K matrix for a given pixel spacing ──────────────────────
def make_K(pix_mm: float) -> np.ndarray:
    fx = fy = SWARO_SID_MM / pix_mm
    cx = cy = (SWARO_IMG_SIZE - 1) / 2.0
    return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float64)

# ── Helper: project 3D → 2D given R, t, pix_mm ───────────────────────────
def project_pts(pts3d_world: np.ndarray, R: np.ndarray, t: np.ndarray,
                pix_mm: float) -> np.ndarray:
    fx = fy = SWARO_SID_MM / pix_mm
    cx = cy = (SWARO_IMG_SIZE - 1) / 2.0
    pts_xzy = xzy(np.atleast_2d(pts3d_world))
    P_cam = (R @ pts_xzy.T).T + t
    u = fx * P_cam[:, 0] / P_cam[:, 2] + cx
    v = fy * P_cam[:, 1] / P_cam[:, 2] + cy
    return np.stack([u, v], axis=1)

# ── Helper: EPnP with a given K ───────────────────────────────────────────
def solve_pnp_with_K(pts3d_world, pts2d, K):
    """EPnP/SQPNP with arbitrary camera matrix K."""
    pts_xzy = xzy(pts3d_world).astype(np.float64)
    n = len(pts_xzy)
    flag = cv2.SOLVEPNP_SQPNP if n == 3 else cv2.SOLVEPNP_EPNP
    _, rvec, tvec = cv2.solvePnP(
        pts_xzy, pts2d.astype(np.float64),
        K, np.zeros(4), flags=flag,
    )
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.flatten().astype(np.float64)

# ── Render DRR for a given (R, t, pix_mm) ────────────────────────────────
def render_drr(R: np.ndarray, t: np.ndarray, pix_mm: float) -> np.ndarray:
    """Returns float32 [0,1] DRR image rendered at native 1024×1024."""
    drr = drr_renderer.generate_from_extrinsic(
        R_proj           = R,
        t_proj           = t,
        output_size      = SWARO_IMG_SIZE,
        pixel_spacing_mm = pix_mm,
        sdd_mm           = SWARO_SID_MM,
        n_steps          = 400,
    )
    drr = drr.astype(np.float32)
    lo, hi = drr.min(), drr.max()
    if hi > lo:
        drr = (drr - lo) / (hi - lo)
    return drr

# ── Build figure ───────────────────────────────────────────────────────────
# Layout: rows = AP / Lat,   cols = [X-ray] [DRR+0.205] [DRR+0.288]
# Each cell also gets the 2D annotation dots + projected centroids

n_rows = len(FRAMES)          # 2
n_cols = 1 + len(CONFIGS)     # 3  (xray | old pix | new pix)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * 5, n_rows * 5.2),
                         constrained_layout=True)

pix_labels  = list(CONFIGS.keys())
frame_labels = list(FRAMES.keys())

for row_idx, (frame_label, (view_tag, frame_idx, subdir)) in enumerate(FRAMES.items()):
    proj_key  = f"{view_tag}_{frame_idx}"
    png_path  = subdir / f"frame_{frame_idx}_z000.png"
    json_key  = f"frame_{frame_idx}_z00"

    xray_img = _load_png(png_path)   # float32 [0,1]
    lm_2d    = lm_2d_all.get(json_key, {})

    common_labels = [l for l in LM_ORDER if l in lm_3d and l in lm_2d]
    pts3d = np.array([lm_3d[l] for l in common_labels])
    pts2d = np.array([lm_2d[l] for l in common_labels])

    print(f"\n{'='*60}")
    print(f"Frame {proj_key}: {len(common_labels)} common landmarks: {common_labels}")

    # ── Column 0: original X-ray with GT 2D annotations ──────────────────
    ax_xray = axes[row_idx, 0]
    ax_xray.imshow(xray_img, cmap='gray', vmin=0, vmax=1,
                   origin='upper', interpolation='bilinear')
    for lm_name in common_labels:
        u, v = lm_2d[lm_name]
        ax_xray.plot(u, v, 'o', color=COLORS[lm_name],
                     markersize=9, markeredgecolor='white', markeredgewidth=1.5,
                     label=lm_name, zorder=5)
        ax_xray.text(u + 12, v, lm_name, color=COLORS[lm_name],
                     fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.55))
    ax_xray.set_title(f"{frame_label}\nX-ray + GT annotations", fontsize=10)
    ax_xray.axis('off')

    # ── Columns 1–2: DRR + projected centroids for each pixel spacing ─────
    for col_offset, (config_label, pix_mm) in enumerate(CONFIGS.items()):
        col_idx = col_offset + 1
        ax = axes[row_idx, col_idx]

        K = make_K(pix_mm)
        fx = K[0, 0]
        print(f"\n  pix_mm={pix_mm:.3f}  Fx={fx:.1f}px")

        # Solve EPnP with this pixel spacing's K
        try:
            R, t = solve_pnp_with_K(pts3d, pts2d, K)
            uv_pred = project_pts(pts3d, R, t, pix_mm)
            reproj_px = float(np.sqrt(((uv_pred - pts2d)**2).sum(axis=1)).mean())
            reproj_mm = reproj_px * pix_mm
            print(f"    EPnP reproj: {reproj_px:.2f}px  ({reproj_mm:.2f}mm)")
        except Exception as e:
            print(f"    EPnP failed: {e}")
            R, t = sl._anatomy_pose(lm_3d, azimuth_deg=0. if view_tag=='ap' else 90.)
            reproj_px = reproj_mm = 0.

        # Render DRR
        print(f"    Rendering DRR ...")
        try:
            drr = render_drr(R, t, pix_mm)
        except Exception as e:
            print(f"    DRR render error: {e}")
            drr = np.zeros((SWARO_IMG_SIZE, SWARO_IMG_SIZE), dtype=np.float32)

        ax.imshow(drr, cmap='gray', vmin=0, vmax=1,
                  origin='upper', interpolation='bilinear')

        # Project ALL 5 centroids (including those without 2D annotations)
        all_pts3d = np.array([lm_3d[l] for l in LM_ORDER if l in lm_3d])
        all_names = [l for l in LM_ORDER if l in lm_3d]
        all_uv    = project_pts(all_pts3d, R, t, pix_mm)

        for i, lm_name in enumerate(all_names):
            u, v = all_uv[i]
            in_frame = 0 <= u < SWARO_IMG_SIZE and 0 <= v < SWARO_IMG_SIZE
            marker = 'o' if lm_name in common_labels else 's'
            ax.plot(u, v, marker, color=COLORS[lm_name],
                    markersize=10, markeredgecolor='white', markeredgewidth=1.5,
                    zorder=5, alpha=1.0 if in_frame else 0.4)
            if in_frame:
                ax.text(u + 12, v, lm_name, color=COLORS[lm_name],
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.55))

        # Also draw GT 2D dots for used landmarks (for comparison)
        for lm_name in common_labels:
            u_gt, v_gt = lm_2d[lm_name]
            ax.plot(u_gt, v_gt, '+', color=COLORS[lm_name],
                    markersize=11, markeredgewidth=2.0, zorder=6)

        # Draw lines between GT (+) and projected (o) for used lms
        for lm_name, (u_gt, v_gt), (u_p, v_p) in zip(
                common_labels, pts2d, uv_pred):
            ax.plot([u_gt, u_p], [v_gt, v_p], '-',
                    color=COLORS[lm_name], linewidth=1.2, alpha=0.8, zorder=4)

        ax.set_xlim(0, SWARO_IMG_SIZE)
        ax.set_ylim(SWARO_IMG_SIZE, 0)

        pix_clean = config_label.replace('\n', ' ')
        ax.set_title(
            f"DRR  pix={pix_mm:.3f}mm  Fx={fx:.0f}px\n"
            f"EPnP reproj: {reproj_px:.1f}px = {reproj_mm:.1f}mm",
            fontsize=9,
        )
        ax.axis('off')

# ── Legend (circles = projected, + = GT annotation, squares = no annotation) ─
legend_handles = [
    mpatches.Patch(facecolor=COLORS[l], edgecolor='white', label=l)
    for l in LM_ORDER if l in lm_3d
]
legend_handles += [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='white', markersize=9,
               label='● Projected centroid (annotated)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markeredgecolor='white', markersize=9,
               label='■ Projected centroid (no 2D label)'),
    plt.Line2D([0], [0], marker='+', color='gray', markersize=11,
               markeredgewidth=2, label='+ GT 2D annotation'),
]
fig.legend(handles=legend_handles, loc='lower center',
           ncol=len(legend_handles), fontsize=8,
           framealpha=0.85, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Swaroopa — Vertebral Centroid Projection: Old vs Corrected Pixel Spacing\n"
    "Circles = EPnP-projected centroids  |  + = GT 2D annotation  |  Lines = reprojection error",
    fontsize=11, fontweight='bold',
)

out_dir = ROOT / 'results' / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'swaro_centroid_comparison.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.close(fig)
print(f"\nSaved: {out_path}")
