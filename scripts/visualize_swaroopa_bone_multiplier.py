#!/usr/bin/env python3
"""
visualize_swaroopa_bone_multiplier.py
======================================
Renders a grid comparing the inverted X-ray against EPnP-pose DiffDRRs
generated with a sweep of bone_attenuation_multiplier values.

Layout: rows = frames, cols = [Inverted X-ray | mult=v0 | mult=v1 | ...]

The EPnP poses are read from the saved poses JSON produced by
export_swaroopa_epnp_drrs.py, so no re-registration is needed.

Usage:
    python scripts/visualize_swaroopa_bone_multiplier.py
    python scripts/visualize_swaroopa_bone_multiplier.py \\
        --frames ap_002 ap_013 lat_000 lat_026 \\
        --multipliers 0.5 1.0 2.0 4.0 8.0 \\
        --drr_size 256 \\
        --output results/figures/swaroopa_bone_mult_sweep.png
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torchio as tio
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from swaroopa_loader import (
    SwaroLoader,
    SWARO_IMG_SIZE, SWARO_PIX_MM, SWARO_SID_MM, SWARO_CX, SWARO_CY,
)
from deepfluoro_loader import DeepFluoroSpecimen, xzy_inv
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
from diffdrr.data import read as diffdrr_read

_X0_MM = (SWARO_CX - SWARO_IMG_SIZE / 2.0) * SWARO_PIX_MM
_Y0_MM = (SWARO_CY - SWARO_IMG_SIZE / 2.0) * SWARO_PIX_MM
_L2R   = np.array([-1., -1., 1.])


# ---------------------------------------------------------------------------
# DiffDRR helpers (identical to export_swaroopa_epnp_drrs.py)
# ---------------------------------------------------------------------------

def build_subject(spec: DeepFluoroSpecimen,
                  bone_attenuation_multiplier: float = 1.0) -> tio.Subject:
    sitk_img = sitk.GetImageFromArray(spec.ct_volume.astype(np.int16))
    sitk_img.SetOrigin([float(v) for v in spec.ct_origin])
    sitk_img.SetSpacing([float(v) for v in spec.ct_spacing])
    sitk_img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    with tempfile.NamedTemporaryFile(suffix='.nrrd', delete=False) as f:
        tmp = f.name
    sitk.WriteImage(sitk_img, tmp)
    subj = diffdrr_read(tmp, orientation=None, center_volume=False,
                        bone_attenuation_multiplier=bone_attenuation_multiplier)
    os.unlink(tmp)
    return subj


def pose_from_extrinsic(R: np.ndarray, t: np.ndarray,
                        device: torch.device) -> RigidTransform:
    right = xzy_inv(R.T @ np.array([1., 0., 0.])).flatten() * _L2R
    up    = xzy_inv(R.T @ np.array([0., 1., 0.])).flatten() * _L2R
    pa    = xzy_inv(R.T @ np.array([0., 0., 1.])).flatten() * _L2R
    src   = xzy_inv(-R.T @ t).flatten()                     * _L2R
    up    = -up
    R_pose = np.stack([right, up, pa], axis=1).astype(np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R_pose
    mat[:3,  3] = src.astype(np.float32)
    return RigidTransform(torch.tensor(mat, dtype=torch.float32, device=device))


@torch.no_grad()
def render_one(subject: tio.Subject, R: np.ndarray, t: np.ndarray,
               size: int, pix_mm: float, device: torch.device) -> np.ndarray:
    drr_mod = DRR(subject, sdd=SWARO_SID_MM,
                  height=size, width=size, delx=pix_mm, dely=pix_mm,
                  x0=_X0_MM, y0=_Y0_MM,
                  renderer='siddon', reverse_x_axis=False).to(device)
    pose = pose_from_extrinsic(R, t, device)
    img  = drr_mod(pose).squeeze().cpu().numpy()
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    del drr_mod
    torch.cuda.empty_cache()
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Bone-multiplier sweep: inverted X-ray vs DiffDRRs'
    )
    p.add_argument('--frames', nargs='+', default=['ap_002', 'ap_013', 'lat_000', 'lat_026'],
                   help='proj_keys to include (default: ap_002 ap_013 lat_000 lat_026)')
    p.add_argument('--multipliers', nargs='+', type=float,
                   default=[0.5, 1.0, 2.0, 4.0, 8.0],
                   help='bone_attenuation_multiplier values to sweep')
    p.add_argument('--poses_json', type=Path,
                   default=Path('results/swaroopa_epnp_poses_diffdrr.json'),
                   help='Poses JSON from export_swaroopa_epnp_drrs.py')
    p.add_argument('--xray_dir', type=Path,
                   default=Path('data/swaroopa_labelled'),
                   help='Root of X-ray data (contains ap/ and lateral/)')
    p.add_argument('--output', type=Path,
                   default=Path('results/figures/swaroopa_bone_mult_sweep.png'))
    p.add_argument('--drr_size', type=int, default=256)
    p.add_argument('--thumb_size', type=int, default=220,
                   help='Display size per image in the grid (px)')
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load poses JSON ───────────────────────────────────────────────────
    with open(args.poses_json) as f:
        payload = json.load(f)
    frame_records = payload['frames']   # key = relative path

    # Build proj_key → record lookup
    by_proj: Dict[str, dict] = {v['proj_key']: v for v in frame_records.values()}

    requested = args.frames
    missing   = [k for k in requested if k not in by_proj]
    if missing:
        print(f"WARNING: proj_keys not found in poses JSON: {missing}")
    frames = [k for k in requested if k in by_proj]
    if not frames:
        raise SystemExit("No valid frames found.")

    # ── Load specimen (for X-ray images + landmark positions) ─────────────
    print("Loading specimen ...")
    loader = SwaroLoader()
    spec   = loader.load(frames=frames, verbose=False)
    proj_by_key = {p.proj_key: p for p in spec.projections}

    pix_mm = SWARO_PIX_MM * (SWARO_IMG_SIZE / args.drr_size)

    # ── Pre-build one Subject per multiplier ──────────────────────────────
    multipliers: List[float] = args.multipliers
    print(f"Building {len(multipliers)} DiffDRR subjects "
          f"(bone_attenuation_multiplier = {multipliers}) ...")
    subjects = {}
    for m in multipliers:
        print(f"  mult={m:.2f} ...")
        subjects[m] = build_subject(spec, bone_attenuation_multiplier=m)

    # ── Figure layout ─────────────────────────────────────────────────────
    NCOLS  = 1 + len(multipliers)   # X-ray + one per multiplier
    NROWS  = len(frames)
    TS     = args.thumb_size

    fig_w = NCOLS * (TS / args.dpi) * 1.08
    fig_h = NROWS * (TS / args.dpi) * 1.40
    fig, axes = plt.subplots(NROWS, NCOLS,
                             figsize=(max(fig_w, 6), max(fig_h, 4)),
                             squeeze=False)
    fig.patch.set_facecolor('#111111')

    for ax in axes.flat:
        ax.axis('off')
        ax.set_facecolor('#111111')

    LM_COLOURS = {'L1': '#ff4444', 'L2': '#ff9900',
                  'L3': '#ffee00', 'L4': '#44ff44', 'L5': '#44ccff'}

    # Column header row (top of each column)
    col_labels = ['X-ray\n(inverted)'] + [f'mult={m:.1f}' for m in multipliers]

    # ── Render ────────────────────────────────────────────────────────────
    for row_i, proj_key in enumerate(frames):
        rec  = by_proj[proj_key]
        proj = proj_by_key.get(proj_key)

        # --- X-ray column ------------------------------------------------
        xray_rel  = rec['xray_relative_path']
        xray_path = args.xray_dir / xray_rel
        try:
            xray_raw = cv2.imread(str(xray_path), cv2.IMREAD_GRAYSCALE)
            if xray_raw is None:
                raise FileNotFoundError
            xray_raw = xray_raw.astype(np.float32) / 255.0
        except FileNotFoundError:
            xray_raw = np.zeros((TS, TS), dtype=np.float32)

        xray_inv = 1.0 - cv2.resize(xray_raw, (TS, TS), interpolation=cv2.INTER_AREA)

        ax_x = axes[row_i, 0]
        ax_x.imshow(xray_inv, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax_x.axis('off')

        # Overlay GT 2D landmarks
        if proj is not None and proj.gt_landmarks_2d:
            scale = TS / SWARO_IMG_SIZE
            for lm_name, uv in proj.gt_landmarks_2d.items():
                c = LM_COLOURS.get(lm_name, 'white')
                ax_x.plot(uv[0] * scale, uv[1] * scale,
                          '+', color=c, markersize=9, markeredgewidth=1.8)
                ax_x.text(uv[0] * scale + 3, uv[1] * scale - 3,
                          lm_name, color=c, fontsize=5.5, va='bottom')

        reproj = rec['reproj_error_px']
        reproj_str = f'{reproj:.1f}px' if reproj > 0 else 'SQPNP'
        ax_x.set_title(f'{proj_key}  (reproj={reproj_str})',
                       color='#dddddd', fontsize=7, pad=3)

        R = np.array(rec['R_proj'])
        t = np.array(rec['t_proj'])

        # --- DRR columns (one per multiplier) ----------------------------
        for col_i, m in enumerate(multipliers):
            print(f"  Rendering {proj_key}  mult={m:.2f} ...")
            drr_img = render_one(subjects[m], R, t, args.drr_size, pix_mm, device)
            drr_th  = cv2.resize(drr_img, (TS, TS), interpolation=cv2.INTER_AREA)

            ax_d = axes[row_i, col_i + 1]
            ax_d.imshow(drr_th, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
            ax_d.axis('off')

            # Column header only on first row
            title = col_labels[col_i + 1] if row_i == 0 else f'mult={m:.1f}'
            ax_d.set_title(title, color='#aaaaaa', fontsize=7, pad=3)

    plt.suptitle(
        'Swaroopa EPnP DRR — bone_attenuation_multiplier sweep  '
        f'(DiffDRR Siddon, {args.drr_size}×{args.drr_size})',
        color='white', fontsize=10, y=1.01,
    )
    plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(args.output), dpi=args.dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
