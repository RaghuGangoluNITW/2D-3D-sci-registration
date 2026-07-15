#!/usr/bin/env python3
"""
visualize_nnunet_centroids_vs_swaroopa.py
==========================================
Side-by-side grid of Swaroopa X-ray vs live DRR (rendered from nnunet CT),
with:
  - X-ray panel : 2D landmark annotations from Swaroopa labels  (+  cross)
  - DRR panel   : HU-weighted lumbar centroids from nnunet seg, pushed 15 mm (◆ diamond)

Pose for each DRR comes from EPnP on the same Swaroopa landmark set.

Layout: N rows × 2 cols   (one row per Swaroopa frame)
  Col 0 — Processed X-ray  with 2D centroid annotations
  Col 1 — Live DRR          with projected nnunet centroid overlay

Usage:
    python scripts/visualize_nnunet_centroids_vs_swaroopa.py
    python scripts/visualize_nnunet_centroids_vs_swaroopa.py \
        --nnunet_dir  spine_segmentation_nnunet_v2 \
        --output      results/figures/nnunet_centroids_vs_swaroopa.png \
        --n_cases     5 \
        --seed        42 \
        --push_mm     15 \
        --xray_proc   none \
        --cols        4
"""

import argparse
import csv
import os
import random
import sys
import tempfile
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import torchio as tio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from diffdrr.drr import DRR
from diffdrr.data import read as diffdrr_read
from diffdrr.pose import RigidTransform

from swaroopa_loader import (
    SwaroLoader,
    SWARO_IMG_SIZE,
    SWARO_PIX_MM,
    SWARO_SID_MM,
    _solve_pnp_swaro,
    _load_mrk_json_3d,
    _load_landmarks_2d,
    _load_png,
    LM_3D_JSON, LM_2D_JSON, XRAY_DIR_AP, XRAY_DIR_LAT,
)
from deepfluoro_loader import xzy, xzy_inv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LUMBAR_LABELS = {"L1": 20, "L2": 21, "L3": 22, "L4": 23, "L5": 24}
LM_COLOURS    = {"L1": "#ff4444", "L2": "#ff9900",
                 "L3": "#ffee00", "L4": "#44ff44", "L5": "#44ccff"}

# DRR rendering parameters — Swaroopa SID, pixel spacing scaled to 256px
SDD_MM   = SWARO_SID_MM                          # 1050 mm
DET_SIZE = 256                                   # px
PIX_MM   = SWARO_PIX_MM * (SWARO_IMG_SIZE / 256) # 1.152 mm/px  (same FOV as Swaroopa 1024)

CACHE_CSV = "spine_segmentation_nnunet_v2/valid_lumbar_cases.csv"

# ---------------------------------------------------------------------------
# X-ray processing (copied verbatim from visualize_swaroopa_epnp_drrs.py)
# ---------------------------------------------------------------------------

ALL_PROCS = ['none', 'clahe', 'blur', 'histmatch', 'gamma', 'percentile']
PROC_LABELS = {
    'none':       'Inverted (no processing)',
    'clahe':      'Inverted + CLAHE',
    'blur':       'Inverted + Gaussian blur (σ=1.5)',
    'histmatch':  'Inverted + histogram match to DRR',
    'gamma':      'Inverted + gamma correction (γ=0.7)',
    'percentile': 'Percentile stretch [1–99%] + invert',
}


def process_xray(xray_raw: np.ndarray, mode: str, drr_ref=None) -> np.ndarray:
    if mode == 'none':
        return np.clip(1.0 - xray_raw, 0.0, 1.0)
    if mode == 'percentile':
        lo, hi = np.percentile(xray_raw, [1, 99])
        stretched = np.clip((xray_raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return np.clip(1.0 - stretched, 0.0, 1.0)
    inv = np.clip(1.0 - xray_raw, 0.0, 1.0)
    if mode == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        u8 = (inv * 255.0).clip(0, 255).astype(np.uint8)
        return clahe.apply(u8).astype(np.float32) / 255.0
    if mode == 'blur':
        return cv2.GaussianBlur(inv, (0, 0), sigmaX=1.5)
    if mode == 'gamma':
        return np.power(inv.clip(0.0, 1.0), 0.7).astype(np.float32)
    if mode == 'histmatch':
        if drr_ref is None:
            return inv
        src_u8 = (inv * 255.0).clip(0, 255).astype(np.uint8).ravel()
        ref_u8 = (drr_ref * 255.0).clip(0, 255).astype(np.uint8).ravel()
        src_hist, _ = np.histogram(src_u8, 256, [0, 256])
        ref_hist, _ = np.histogram(ref_u8, 256, [0, 256])
        src_cdf = src_hist.cumsum().astype(np.float64)
        ref_cdf = ref_hist.cumsum().astype(np.float64)
        src_cdf /= src_cdf[-1]; ref_cdf /= ref_cdf[-1]
        lut = np.zeros(256, dtype=np.uint8)
        j = 0
        for i in range(256):
            while j < 255 and ref_cdf[j] < src_cdf[i]:
                j += 1
            lut[i] = j
        return lut[(inv * 255.0).clip(0, 255).astype(np.uint8)].astype(np.float32) / 255.0
    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# nnunet helpers
# ---------------------------------------------------------------------------

def get_cases_with_all_lumbar(seg_dir: Path, vol_dir: Path) -> list:
    cache_path = Path(CACHE_CSV)
    if cache_path.exists():
        print(f"  Loading valid cases from cache: {cache_path}")
        cases = []
        with open(cache_path, newline="") as f:
            for row in csv.DictReader(f):
                vp, sp = Path(row["volume_path"]), Path(row["seg_path"])
                if vp.exists() and sp.exists():
                    cases.append((vp, sp))
        print(f"  {len(cases)} cases loaded.")
        return cases

    print("  Scanning segmentations …")
    cases = []
    for i, seg_path in enumerate(sorted(seg_dir.glob("*.nii*")), 1):
        stem = seg_path.name.split(".")[0]
        vol_path = vol_dir / seg_path.name
        if not vol_path.exists():
            vol_path = vol_dir / (stem + ".nii.gz")
        if not vol_path.exists():
            continue
        data = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
        if all(np.any(data == lbl) for lbl in LUMBAR_LABELS.values()):
            cases.append((vol_path, seg_path))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["volume_path", "seg_path"])
        w.writeheader()
        for vp, sp in cases:
            w.writerow({"volume_path": str(vp), "seg_path": str(sp)})
    print(f"  Cached {len(cases)} valid cases → {cache_path}")
    return cases


def compute_geometric_centroids(seg_path: Path):
    seg_nib  = nib.load(str(seg_path))
    seg_data = seg_nib.get_fdata(dtype=np.float32)
    affine   = seg_nib.affine
    centroids = {}
    for name, lbl in LUMBAR_LABELS.items():
        vox_coords = np.array(np.where(seg_data == lbl)).T
        vox_com    = vox_coords.mean(axis=0)
        centroids[name] = (affine @ np.append(vox_com, 1.0))[:3]
    return centroids, seg_data, affine


def compute_weighted_centroids(seg_data, affine, vol_path):
    hu = nib.load(str(vol_path)).get_fdata(dtype=np.float32)
    centroids = {}
    for name, lbl in LUMBAR_LABELS.items():
        mask       = seg_data == lbl
        vox_coords = np.array(np.where(mask)).T
        weights    = np.clip(hu[mask].astype(np.float64), 1.0, None)
        weights   /= weights.sum()
        vox_com    = (vox_coords * weights[:, None]).sum(axis=0)
        centroids[name] = (affine @ np.append(vox_com, 1.0))[:3]
    return centroids


def pushed_centroids(geom_cents, wt_cents_raw, push_mm=15.0):
    """Return HU-weighted centroids pushed push_mm away from geom towards wt."""
    pts_geom = np.stack([geom_cents[k]    for k in LUMBAR_LABELS], axis=0)
    pts_wt   = np.stack([wt_cents_raw[k]  for k in LUMBAR_LABELS], axis=0)
    dirs     = pts_geom - pts_wt                             # geom ← wt direction
    norms    = np.linalg.norm(dirs, axis=1, keepdims=True)
    unit_dirs = np.where(norms > 1e-6, dirs / norms, 0.0)
    return pts_wt + unit_dirs * push_mm                     # (5,3) LPS mm


# ---------------------------------------------------------------------------
# DRR rendering helpers (same convention as visualize_lumbar_centroids_drr.py)
# ---------------------------------------------------------------------------

def build_sitk_from_nib(vol_path: Path):
    nib_img = nib.load(str(vol_path))
    data    = nib_img.get_fdata(dtype=np.float32).astype(np.int16)
    affine  = nib_img.affine
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).tolist()
    origin  = affine[:3, 3].tolist()
    data_zyx = data.transpose(2, 1, 0)
    sitk_img = sitk.GetImageFromArray(data_zyx)
    sitk_img.SetSpacing([float(s) for s in spacing])
    sitk_img.SetOrigin([float(o) for o in origin])
    sitk_img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    return sitk_img, spacing, origin


def build_subject(sitk_img, tmp_dir, bone_mult: float = 4.0, min_hu: float = None):
    if min_hu is not None:
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        arr = np.where(arr < min_hu, min_hu, arr)
        clipped = sitk.GetImageFromArray(arr)
        clipped.CopyInformation(sitk_img)
        sitk_img = clipped
    tmp_path = os.path.join(tmp_dir, "vol.nrrd")
    sitk.WriteImage(sitk_img, tmp_path)
    return diffdrr_read(tmp_path, orientation=None, center_volume=False,
                        bone_attenuation_multiplier=bone_mult)


def make_pose_from_epnp(R: np.ndarray, t: np.ndarray,
                        spine_centre_lps: np.ndarray,
                        device) -> tuple:
    """
    Recentre EPnP pose (R, t) so the X-ray source sits SDD_MM behind the
    *spine centroid* of the target CT rather than behind its bounding-box
    centre.  This guarantees that the spine appears in the middle of the DRR
    and that project_centroids_on_drr is perfectly consistent with rendering.

    Convention: P_cam = R @ xzy(pts_lps) + t   (Swaroopa LPS-xzy calibration)
    Returns (RigidTransform, R, t_new).
    """
    # Keep spine in LPS — R is calibrated in xzy-LPS so no flip needed here
    spine_xzy = xzy(spine_centre_lps.reshape(1, 3)).flatten()   # (3,) xzy-LPS

    # Principal axis in xzy-LPS space (camera z towards scene)
    pa_xzy = R.T @ np.array([0., 0., 1.])
    pa_xzy /= np.linalg.norm(pa_xzy)

    # Source SDD_MM behind spine centre along principal axis
    src_xzy = spine_xzy - pa_xzy * SDD_MM
    t_new   = -(R @ src_xzy)

    # ── Build RigidTransform (mirrors pose_from_extrinsic) ────────────────
    # Vectors are in xzy-LPS; convert to xyz-RAS for diffdrr (L2R + xzy_inv)
    L2R   = np.array([-1., -1., 1.])
    right = xzy_inv(R.T @ np.array([1., 0., 0.])).flatten() * L2R
    up    = xzy_inv(R.T @ np.array([0., 1., 0.])).flatten() * L2R
    pa    = xzy_inv(R.T @ np.array([0., 0., 1.])).flatten() * L2R
    src   = xzy_inv(-R.T @ t_new).flatten()                  * L2R
    up    = -up

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.stack([right, up, pa], axis=1).astype(np.float32)
    mat[:3,  3] = src.astype(np.float32)
    pose = RigidTransform(torch.tensor(mat, dtype=torch.float32, device=device))
    return pose, R, t_new


def project_centroids_on_drr(pts_lps: np.ndarray,
                              R: np.ndarray, t: np.ndarray,
                              pix_mm: float = None) -> np.ndarray:
    """
    Project LPS centroid points onto the DRR pixel grid.
    R is calibrated via _solve_pnp_swaro on LPS landmarks:
        P_cam = R @ xzy(pts_lps) + t
    Returns (N, 2) pixel coords on (DET_SIZE × DET_SIZE) image.
    """
    if pix_mm is None:
        pix_mm = PIX_MM
    P_cam   = (R @ xzy(pts_lps).T).T + t
    fx      = SDD_MM / pix_mm
    cx      = DET_SIZE / 2.0
    u       = fx * P_cam[:, 0] / P_cam[:, 2] + cx
    v       = fx * P_cam[:, 1] / P_cam[:, 2] + cx
    return np.stack([u, v], axis=1)


def drr_to_float(drr_tensor) -> np.ndarray:
    img = drr_tensor.squeeze().cpu().float().numpy()
    img = np.clip(img, 0, None)
    if img.max() > 0:
        img /= img.max()
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Swaroopa frame loader
# ---------------------------------------------------------------------------

def load_swaroopa_frames(view: str):
    """
    Return list of dicts, one per annotated frame in the given view ('ap'/'lat').
    Each dict: xray_raw, pts2d (N,2), labels, R_epnp, t_epnp, fname.
    """
    xray_dir = XRAY_DIR_AP if view == 'ap' else XRAY_DIR_LAT
    lm_3d     = _load_mrk_json_3d(LM_3D_JSON)     # RAS mm (Slicer mrk.json)
    lm_2d_all = _load_landmarks_2d(LM_2D_JSON)

    frames = []
    for png_path in sorted(xray_dir.glob("frame_*_z000.png")):
        frame_num = png_path.stem.split("_")[1]
        json_key  = f"frame_{frame_num}_z00"
        lm_2d     = lm_2d_all.get(json_key, {})
        common    = [l for l in sorted(lm_2d) if l in lm_3d]
        if len(common) < 3:
            continue
        pts3d = np.array([lm_3d[c]  for c in common])
        pts2d = np.array([lm_2d[c]  for c in common])
        try:
            R, t = _solve_pnp_swaro(pts3d, pts2d)
        except Exception:
            continue
        frames.append(dict(
            xray_raw = _load_png(png_path),
            pts2d    = pts2d,
            labels   = common,
            R        = R,
            t        = t,
            fname    = png_path.name,
            view     = view,
        ))
    return frames


# ---------------------------------------------------------------------------
# Grid renderer
# ---------------------------------------------------------------------------

def render_grid(swaroopa_frames, nnunet_cases, push_mm,
                xray_proc, output, cols, thumb_size, dpi, device, delx=None,
                bone_mult=4.0, min_hu=None):
    """
    Render comparison grid: X-ray (Swaroopa annots) | DRR (nnunet centroids).
    Each Swaroopa frame is paired with a random nnunet CT.
    """
    if delx is None:
        delx = SWARO_PIX_MM * (SWARO_IMG_SIZE / DET_SIZE)
    pix_mm = delx
    N            = len(swaroopa_frames)
    TS           = thumb_size
    NCOLS_FRAMES = cols
    NCOLS_IMGS   = NCOLS_FRAMES * 2
    NROWS        = (N + NCOLS_FRAMES - 1) // NCOLS_FRAMES

    fig_w = NCOLS_IMGS * (TS / dpi) * 1.05
    fig_h = NROWS      * (TS / dpi) * 1.55
    fig, axes = plt.subplots(NROWS, NCOLS_IMGS,
                             figsize=(max(fig_w, 8), max(fig_h, 4)),
                             squeeze=False)
    fig.patch.set_facecolor('#111111')
    for ax in axes.flat:
        ax.axis('off')
        ax.set_facecolor('#111111')

    for idx, ep in enumerate(swaroopa_frames):
        # Pick nnunet CT for this frame
        vol_path, seg_path = nnunet_cases[idx % len(nnunet_cases)]

        print(f"  [{idx+1}/{N}] {ep['fname']}  ↔  {seg_path.stem} …", end=" ", flush=True)

        # ── Compute centroids ──────────────────────────────────────────────
        geom_cents, seg_data, affine = compute_geometric_centroids(seg_path)
        wt_cents   = compute_weighted_centroids(seg_data, affine, vol_path)
        pts_lps    = pushed_centroids(geom_cents, wt_cents, push_mm=push_mm)  # (5,3) LPS

        # ── CT geometry ───────────────────────────────────────────────────
        nib_img       = nib.load(str(vol_path))
        vol_data      = nib_img.get_fdata(dtype=np.float32)
        affine_nib    = nib_img.affine
        spacing       = np.sqrt((affine_nib[:3, :3] ** 2).sum(axis=0)).tolist()
        origin        = affine_nib[:3, 3].tolist()
        vol_shape_zyx = (vol_data.shape[2], vol_data.shape[1], vol_data.shape[0])

        # ── Build DRR pose: source centred on spine centroid ────────────────
        spine_centre_lps = pts_lps.mean(axis=0)   # mean of L1-L5 centroids
        pose, R, t_new = make_pose_from_epnp(
            ep['R'], ep['t'], spine_centre_lps, device)

        # ── Render DRR ────────────────────────────────────────────────────
        sitk_img, _, _ = build_sitk_from_nib(vol_path)
        with tempfile.TemporaryDirectory() as tmp:
            subject = build_subject(sitk_img, tmp, bone_mult=bone_mult, min_hu=min_hu)
            drr_mod = DRR(subject, sdd=SDD_MM, height=DET_SIZE, width=DET_SIZE,
                          delx=pix_mm, dely=pix_mm, x0=0.0, y0=0.0,
                          renderer="siddon", reverse_x_axis=False).to(device)
            with torch.no_grad():
                drr_tensor = drr_mod(pose)
            del drr_mod
        torch.cuda.empty_cache() if device.type == "cuda" else None
        drr_float = drr_to_float(drr_tensor)   # float32 [0,1]
        print("done.")

        # ── Project centroids onto DRR ────────────────────────────────────────
        proj_px = project_centroids_on_drr(pts_lps, R, t_new, pix_mm)  # (5,2)

        # ── Thumbnail scaling ─────────────────────────────────────────────
        drr_th    = cv2.resize(drr_float, (TS, TS), interpolation=cv2.INTER_AREA)
        xray_small = cv2.resize(ep['xray_raw'], (TS, TS), interpolation=cv2.INTER_AREA)
        xray_disp  = process_xray(xray_small, xray_proc,
                                  drr_ref=drr_th if xray_proc == 'histmatch' else None)

        # Grid position
        row  = idx // NCOLS_FRAMES
        col0 = (idx %  NCOLS_FRAMES) * 2
        col1 = col0 + 1

        scale_x = TS / SWARO_IMG_SIZE
        scale_d = TS / DET_SIZE

        # ── X-ray panel ───────────────────────────────────────────────────
        ax_x = axes[row, col0]
        ax_x.imshow(xray_disp, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax_x.axis('off')
        for i, lbl in enumerate(ep['labels']):
            c = LM_COLOURS.get(lbl, 'white')
            u, v = ep['pts2d'][i] * scale_x
            ax_x.plot(u, v, '+', color=c, markersize=8, markeredgewidth=1.5)
            ax_x.text(u + 3, v - 3, lbl, color=c, fontsize=5, va='bottom')
        ax_x.set_title(f"{ep['fname']}  ({ep['view']})\nX-ray",
                       color='#cccccc', fontsize=6, pad=2)

        # ── DRR panel ─────────────────────────────────────────────────────
        ax_d = axes[row, col1]
        ax_d.imshow(drr_th, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax_d.axis('off')
        for i, lbl in enumerate(LUMBAR_LABELS):
            c    = LM_COLOURS.get(lbl, 'white')
            u, v = proj_px[i] * scale_d
            if 0 <= u < TS and 0 <= v < TS:
                ax_d.plot(u, v, 'D', color=c, markersize=6,
                          markeredgecolor='white', markeredgewidth=0.6)
                ax_d.text(u + 3, v - 3, lbl, color=c, fontsize=5, va='bottom')
        ax_d.set_title(f"{seg_path.stem}\nDRR  push={push_mm:.0f}mm",
                       color='#aaaaaa', fontsize=6, pad=2)

    proc_label = PROC_LABELS[xray_proc]
    plt.suptitle(
        f'Swaroopa X-ray vs nnunet DRR centroids  [push={push_mm:.0f}mm, {proc_label}]',
        color='white', fontsize=11, y=1.01,
    )
    plt.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output), dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved: {output}  ({N} frames, grid {NROWS}×{NCOLS_FRAMES})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Swaroopa X-ray vs live nnunet DRR centroid overlay'
    )
    p.add_argument('--nnunet_dir',  type=Path,
                   default=Path('spine_segmentation_nnunet_v2'),
                   help='Root folder with volumes/ and segmentations/')
    p.add_argument('--output',      type=Path,
                   default=Path('results/figures/nnunet_centroids_vs_swaroopa.png'))
    p.add_argument('--n_cases',     type=int, default=5,
                   help='Number of random nnunet CTs to cycle over (default: 5)')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--push_mm',     type=float, default=15.0,
                   help='Centroid push distance in mm (default: 15)')
    p.add_argument('--views',       nargs='+', default=['ap', 'lat'],
                   choices=['ap', 'lat'],
                   help='Which Swaroopa views to include (default: ap lat)')
    p.add_argument('--cols',        type=int, default=4)
    p.add_argument('--thumb_size',  type=int, default=200)
    p.add_argument('--dpi',         type=int, default=150)
    p.add_argument('--xray_proc',   type=str, default='none',
                   choices=ALL_PROCS)
    p.add_argument('--delx',        type=float,
                   default=SWARO_PIX_MM * (SWARO_IMG_SIZE / 256),
                   help='DRR pixel spacing in mm (default: %(default).4f — '
                        'same FOV as Swaroopa at 256px). '
                        'Smaller value = larger / zoomed-in vertebrae.')
    p.add_argument('--bone_mult',     type=float, default=4.0,
                   help='bone_attenuation_multiplier for DiffDRR (default: 4.0)')
    p.add_argument('--min_hu',        type=float, default=None,
                   help='HU threshold: voxels below this value are clamped to it '
                        '(e.g. 0 removes air/soft tissue, 200 keeps only bone)')
    p.add_argument('--rebuild_cache', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    if args.rebuild_cache and Path(CACHE_CSV).exists():
        Path(CACHE_CSV).unlink()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── nnunet CTs ──────────────────────────────────────────────────────────
    root    = args.nnunet_dir
    seg_dir = root / 'segmentations'
    vol_dir = root / 'volumes'
    print("Scanning nnunet cases …")
    valid = get_cases_with_all_lumbar(seg_dir, vol_dir)
    if len(valid) < args.n_cases:
        raise RuntimeError(f"Need {args.n_cases} valid cases, found {len(valid)}")
    chosen_cts = random.sample(valid, args.n_cases)

    # ── Swaroopa frames ─────────────────────────────────────────────────────
    print("Loading Swaroopa frames …")
    all_frames = []
    for view in args.views:
        frames = load_swaroopa_frames(view)
        print(f"  {view}: {len(frames)} annotated frames")
        all_frames.extend(frames)
    if not all_frames:
        raise RuntimeError("No Swaroopa frames found.")

    # ── Render ──────────────────────────────────────────────────────────────
    render_grid(
        swaroopa_frames = all_frames,
        nnunet_cases    = chosen_cts,
        push_mm         = args.push_mm,
        xray_proc       = args.xray_proc,
        output          = args.output,
        cols            = args.cols,
        thumb_size      = args.thumb_size,
        dpi             = args.dpi,
        device          = device,
        delx            = args.delx,
        bone_mult       = args.bone_mult,
        min_hu          = args.min_hu,
    )


if __name__ == '__main__':
    main()
