#!/usr/bin/env python3
"""
visualize_hu_threshold_grid.py
==============================
Render a 6×7 grid showing the effect of min_hu threshold on DRR appearance.

Rows  : 6 selected X-ray frames (ap_002, ap_006, ap_010, lat_000, lat_003, lat_021)
Cols  : 7 HU thresholds  (0, 50, 100, 150, 200, 250, 300)

Each cell shows the DRR at the EPnP pose with a cylinder mask (r=50mm) and the
given min_hu lower clip.  The inverted X-ray is placed in a prepended col 0.

Usage:
    python scripts/visualize_hu_threshold_grid.py
    python scripts/visualize_hu_threshold_grid.py --frames ap_002 lat_000 \
        --hu_steps 0 100 200 300 --cylinder_radius 50 \
        --output results/figures/hu_threshold_grid.png
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
from deepfluoro_loader import DeepFluoroSpecimen
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--frames', nargs='+',
               default=['ap_002', 'ap_006', 'ap_010', 'lat_000', 'lat_003', 'lat_021'])
p.add_argument('--hu_steps', nargs='+', type=float,
               default=[0, 50, 100, 150, 200, 250, 300])
p.add_argument('--render_size', type=int, default=192)
p.add_argument('--output', type=Path,
               default=Path('results/figures/hu_threshold_grid.png'))
p.add_argument('--dpi', type=int, default=130)
p.add_argument('--suppress_highlights', action='store_true')
args = p.parse_args()

FRAMES      = args.frames
HU_STEPS    = args.hu_steps
SZ          = args.render_size
OUT         = args.output
PIX_MM      = SWARO_PIX_MM * (SWARO_IMG_SIZE / SZ)
_X0_MM      = (SWARO_CX - SWARO_IMG_SIZE / 2.0) * SWARO_PIX_MM
_Y0_MM      = (SWARO_CY - SWARO_IMG_SIZE / 2.0) * SWARO_PIX_MM
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

L2R = np.array([-1., -1., 1.])

# ---------------------------------------------------------------------------
# Load specimen
# ---------------------------------------------------------------------------
print("Loading specimen ...")
loader = SwaroLoader()
spec   = loader.load(frames=FRAMES, verbose=False)

# LPS→RAS offset
_subject0 = build_subject(spec)
expected_ras      = np.asarray(spec.ct_origin, dtype=np.float64) * np.array([-1.,-1.,1.])
torchio_ras       = _subject0.volume.affine[:3, 3].astype(np.float64)
lps_to_ras_offset = torchio_ras - expected_ras
del _subject0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_drr_module(subject):
    return DRR(subject, sdd=SWARO_SID_MM, height=SZ, width=SZ,
               delx=PIX_MM, dely=PIX_MM, x0=_X0_MM, y0=_Y0_MM,
               renderer="siddon", reverse_x_axis=False).to(device)

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
def render(drr_mod, R, t):
    pose = pose_from_extrinsic(R, t)
    img  = drr_mod(pose).squeeze().cpu().numpy().astype(np.float32)
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn) if mx > mn else img

def suppress_highlights(img, thr=0.6, darken=0.8, sigma=31, min_blob=500):
    from scipy.ndimage import gaussian_filter, label as ndi_label
    bright = (img > thr).astype(np.uint8)
    labeled, n = ndi_label(bright)
    mask = np.zeros_like(bright)
    for lbl in range(1, n+1):
        if (labeled == lbl).sum() >= min_blob:
            mask |= (labeled == lbl).astype(np.uint8)
    weight = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    weight = np.clip(weight, 0, 1)
    return img * (1.0 - (1.0 - darken) * weight)

def load_xray(proj):
    raw = cv2.resize(proj.image_raw, (SZ, SZ), interpolation=cv2.INTER_AREA)
    xray = 1.0 - raw.astype(np.float32)
    if args.suppress_highlights:
        xray = suppress_highlights(xray)
    return xray

# ---------------------------------------------------------------------------
# Build one DRR module per HU threshold
# ---------------------------------------------------------------------------
print(f"Building {len(HU_STEPS)} DRR subjects (no cylinder mask) ...")
drr_mods = {}
for hu in HU_STEPS:
    print(f"  min_hu={hu:.0f} ...", end=' ', flush=True)
    subj = build_subject_hu_clipped(spec, hu_min=hu)
    drr_mods[hu] = make_drr_module(subj)
    print("ready")

# ---------------------------------------------------------------------------
# Render grid: rows=frames, cols=[xray] + [HU0..HU6]
# ---------------------------------------------------------------------------
n_rows = len(FRAMES)
n_cols = 1 + len(HU_STEPS)   # col 0 = X-ray, cols 1..7 = DRRs

col_labels = ['X-ray (inv.)'] + [f'HU≥{int(h)}' for h in HU_STEPS]

print(f"\nRendering {n_rows}×{n_cols} grid ...")
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(n_cols * (SZ/72 + 0.3),
                                   n_rows * (SZ/72 + 0.45) + 0.8))
fig.patch.set_facecolor('#111111')

if n_rows == 1:
    axes = axes[np.newaxis, :]

# Column headers
for ci, lbl in enumerate(col_labels):
    axes[0, ci].set_title(lbl, color='white', fontsize=8, pad=3)

for ri, fkey in enumerate(FRAMES):
    proj = next(p for p in spec.projections if p.proj_key == fkey)
    R, t = proj.R_proj.copy(), proj.t_proj.copy()
    reproj = getattr(proj, 'reproj_error_px', float('nan'))

    print(f"  {fkey}  (reproj={reproj:.2f}px)")

    # Col 0: inverted X-ray
    ax = axes[ri, 0]
    ax.imshow(load_xray(proj), cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.set_ylabel(fkey, color='white', fontsize=8, labelpad=3)
    ax.set_xlabel(f'reproj={reproj:.1f}px', color='#aaaaaa', fontsize=6.5, labelpad=2)

    # Cols 1..: DRRs at each HU threshold
    for ci, hu in enumerate(HU_STEPS):
        drr_img = render(drr_mods[hu], R, t)
        ax = axes[ri, 1 + ci]
        ax.imshow(drr_img, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#444')
        ax.set_xlabel(f'HU≥{int(hu)}', color='#aaaaaa', fontsize=6.5, labelpad=2)

fig.suptitle(
    f'DRR HU threshold sweep  |  full CT (no cylinder mask)  |  '
    f'{len(FRAMES)} frames × {len(HU_STEPS)} thresholds  |  DiffDRR',
    color='white', fontsize=10, y=1.002)

plt.tight_layout(rect=[0, 0, 1, 0.998])
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=args.dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved: {OUT}")
