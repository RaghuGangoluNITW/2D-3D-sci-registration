#!/usr/bin/env python3
"""
visualize_pose_rotations.py
============================
6×5 grid: for 6 X-ray frames show the inverted X-ray and DRRs at 4 poses:
  Col 0 — Inverted X-ray
  Col 1 — EPnP pose              (HU≥50, no cylinder mask)
  Col 2 — EPnP + Rx(180°)        (flip around X axis)
  Col 3 — EPnP + Ry(180°)        (flip around Y axis)
  Col 4 — EPnP + Rz(180°)        (flip around Z axis)

Usage:
    python scripts/visualize_pose_rotations.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from swaroopa_loader import SwaroLoader, SWARO_IMG_SIZE, SWARO_PIX_MM, SWARO_SID_MM, SWARO_CX, SWARO_CY
from run_swaroopa_diffdrr import build_subject_hu_clipped, build_subject, xzy_inv
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--frames', nargs='+',
               default=['ap_002', 'ap_006', 'ap_010', 'lat_000', 'lat_003', 'lat_021'])
p.add_argument('--hu_min',       type=float, default=50.0)
p.add_argument('--render_size',  type=int,   default=192)
p.add_argument('--output', type=Path,
               default=Path('results/figures/pose_rotations_grid.png'))
p.add_argument('--dpi', type=int, default=130)
args = p.parse_args()

FRAMES = args.frames
HU_MIN = args.hu_min
SZ     = args.render_size
OUT    = args.output
PIX_MM = SWARO_PIX_MM * (SWARO_IMG_SIZE / SZ)
_X0_MM = (SWARO_CX - SWARO_IMG_SIZE / 2.0) * SWARO_PIX_MM
_Y0_MM = (SWARO_CY - SWARO_IMG_SIZE / 2.0) * SWARO_PIX_MM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
L2R    = np.array([-1., -1., 1.])

# 180° rotation matrices around each axis
Rx180 = np.array([[1., 0., 0.],
                   [0.,-1., 0.],
                   [0., 0.,-1.]])
Ry180 = np.array([[-1., 0., 0.],
                   [ 0., 1., 0.],
                   [ 0., 0.,-1.]])
Rz180 = np.array([[-1., 0., 0.],
                   [ 0.,-1., 0.],
                   [ 0., 0., 1.]])

POSE_VARIANTS = [
    ('EPnP',       np.eye(3)),
    ('Rx 180°',    Rx180),
    ('Ry 180°',    Ry180),
    ('Rz 180°',    Rz180),
]

# ---------------------------------------------------------------------------
# Load specimen & build subject
# ---------------------------------------------------------------------------
print("Loading specimen ...")
loader = SwaroLoader()
spec   = loader.load(frames=FRAMES, verbose=False)

print(f"Building DRR subject (HU≥{HU_MIN:.0f}, no cylinder mask) ...")
subj = build_subject_hu_clipped(spec, hu_min=HU_MIN)

# LPS→RAS offset
ref_subj      = build_subject(spec)
expected_ras  = np.asarray(spec.ct_origin, dtype=np.float64) * np.array([-1.,-1.,1.])
torchio_ras   = ref_subj.volume.affine[:3, 3].astype(np.float64)
lps_to_ras_offset = torchio_ras - expected_ras
del ref_subj

drr_mod = DRR(subj, sdd=SWARO_SID_MM, height=SZ, width=SZ,
              delx=PIX_MM, dely=PIX_MM, x0=_X0_MM, y0=_Y0_MM,
              renderer="siddon", reverse_x_axis=False).to(device)

# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
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
def render(R, t):
    pose = pose_from_extrinsic(R, t)
    img  = drr_mod(pose).squeeze().cpu().numpy().astype(np.float32)
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn) if mx > mn else img

def load_xray(proj):
    raw = cv2.resize(proj.image_raw, (SZ, SZ), interpolation=cv2.INTER_AREA)
    return 1.0 - raw.astype(np.float32)

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
n_rows = len(FRAMES)
n_cols = 1 + len(POSE_VARIANTS)   # col 0 = xray, cols 1-4 = poses

col_labels = ['X-ray (inv.)'] + [label for label, _ in POSE_VARIANTS]

print(f"\nRendering {n_rows}×{n_cols} grid ...")
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(n_cols * (SZ/72 + 0.3),
                                   n_rows * (SZ/72 + 0.45) + 0.8))
fig.patch.set_facecolor('#111111')
if n_rows == 1:
    axes = axes[np.newaxis, :]

for ci, lbl in enumerate(col_labels):
    axes[0, ci].set_title(lbl, color='white', fontsize=9, pad=3)

for ri, fkey in enumerate(FRAMES):
    proj = next(p for p in spec.projections if p.proj_key == fkey)
    R0, t0 = proj.R_proj.copy(), proj.t_proj.copy()
    reproj  = getattr(proj, 'reproj_error_px', float('nan'))

    print(f"  {fkey}  (reproj={reproj:.2f}px)")

    # Col 0: inverted X-ray
    ax = axes[ri, 0]
    ax.imshow(load_xray(proj), cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.set_ylabel(fkey, color='white', fontsize=8, labelpad=3)
    ax.set_xlabel(f'reproj={reproj:.1f}px', color='#aaaaaa', fontsize=6.5, labelpad=2)

    # Cols 1-4: rotated DRRs
    for ci, (label, R_rot) in enumerate(POSE_VARIANTS):
        R_var = R_rot @ R0   # apply rotation to extrinsic R
        drr_img = render(R_var, t0)
        ax = axes[ri, 1 + ci]
        ax.imshow(drr_img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#444')
        col_clr = 'lime' if ci == 0 else '#aaaaaa'
        ax.set_xlabel(label, color=col_clr, fontsize=7, labelpad=2)

fig.suptitle(
    f'EPnP pose + 180° axis rotations  |  HU≥{HU_MIN:.0f}  |  full CT  |  DiffDRR',
    color='white', fontsize=10, y=1.002)

plt.tight_layout(rect=[0, 0, 1, 0.998])
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=args.dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved: {OUT}")
