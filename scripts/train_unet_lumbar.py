#!/usr/bin/env python3
"""
train_unet_lumbar.py
====================
Train a 2D U-Net to predict L1-L5 vertebra centroid heatmaps from AP DRRs.

Training:
  Input  : AP DRR rendered from nnunet CT        (1 × DET_SIZE × DET_SIZE)
  Target : 5 × DET_SIZE × DET_SIZE Gaussian heatmaps, one per L1-L5 vertebra,
           peak at the projected centroid position

Valid case selection: only CTs that have all of L1-L5 AND a considerable
  number of voxels (>HU threshold) below the inferior boundary of L5
  (i.e. sacrum/pelvis region is present in the scan).

Validation: Swaroopa stored AP X-rays are used as inputs. The validation
  metric is the mean Euclidean distance (px) from each predicted heatmap peak
  to the corresponding annotated 2D landmark.  The model is saved whenever
  validation distance improves.

Debug / dry-run:
  --dry_run   generates 5 (DRR, heatmap) pairs, saves visualisations, and exits.

Usage:
    # dry run — verify training data quality:
    python scripts/train_unet_lumbar.py --dry_run

    # full training:
    python scripts/train_unet_lumbar.py \\
        --n_cases 30 --n_train 300 --n_iters 5000 \\
        --delx 0.8 --bone_mult 16.0 --min_hu 0 --gauss_sigma 8
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
import torch.nn as nn
import torch.optim as optim
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from diffdrr.drr import DRR
from diffdrr.data import read as diffdrr_read
from diffdrr.pose import RigidTransform

from swaroopa_loader import (
    SWARO_IMG_SIZE,
    SWARO_PIX_MM,
    SWARO_SID_MM,
    _solve_pnp_swaro,
    _load_mrk_json_3d,
    _load_landmarks_2d,
    _load_png,
    LM_3D_JSON, LM_2D_JSON, XRAY_DIR_AP,
)
from deepfluoro_loader import xzy, xzy_inv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LUMBAR_LABELS   = {"L1": 20, "L2": 21, "L3": 22, "L4": 23, "L5": 24}
LM_COLOURS      = {"L1": "#ff4444", "L2": "#ff9900",
                   "L3": "#ffee00", "L4": "#44ff44", "L5": "#44ccff"}
SDD_MM          = SWARO_SID_MM
DET_SIZE        = 256
PIX_MM_DEFAULT  = SWARO_PIX_MM * (SWARO_IMG_SIZE / 256)   # 1.152 mm/px
CACHE_CSV       = "spine_segmentation_nnunet_v2/valid_lumbar_below_l5.csv"

# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    """Lightweight U-Net: 1-channel input → 5-channel sigmoid heatmap output."""

    def __init__(self, in_ch: int = 1, out_ch: int = 5,
                 features: tuple = (32, 64, 128, 256)):
        super().__init__()
        self.pool     = nn.MaxPool2d(2)
        self.encoders = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.encoders.append(_ConvBlock(ch, f))
            ch = f
        self.bottleneck = _ConvBlock(ch, ch * 2)
        ch = ch * 2
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.decoders.append(_ConvBlock(f * 2, f))
            ch = f
        self.head = nn.Conv2d(ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x = enc(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:],
                                               mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        return torch.sigmoid(self.head(x))


class UNetSingle(UNet):
    """
    Same architecture as UNet but with a single output channel.
    All 5 Gaussian blobs are merged into one heatmap.
    During inference the 5 local maxima are found and sorted by v (row)
    to assign L1…L5 labels (top-most peak = L1).
    """
    def __init__(self, features: tuple = (32, 64, 128, 256)):
        super().__init__(in_ch=1, out_ch=1, features=features)


def build_model(model_type: str) -> nn.Module:
    """Return a fresh model of the requested type."""
    if model_type == 'single':
        return UNetSingle()
    return UNet(in_ch=1, out_ch=5)



def get_valid_cases(seg_dir: Path, vol_dir: Path,
                    min_below_voxels: int = 1000,
                    min_below_hu: float = 100.0,
                    rebuild: bool = False) -> list:
    """
    Return list of (vol_path, seg_path) for CTs that have all L1-L5 labels
    AND at least `min_below_voxels` voxels with HU > `min_below_hu` below
    the inferior boundary of L5 (sacrum / pelvis region).
    """
    cache = Path(CACHE_CSV)
    if cache.exists() and not rebuild:
        print(f"  Loading valid cases from cache: {cache}")
        cases = []
        with open(cache, newline="") as f:
            for row in csv.DictReader(f):
                vp, sp = Path(row["volume_path"]), Path(row["seg_path"])
                if vp.exists() and sp.exists():
                    cases.append((vp, sp))
        print(f"  {len(cases)} cases loaded.")
        return cases

    print("  Scanning CTs for L1-L5 + below-L5 data …")
    cases = []
    for seg_path in sorted(seg_dir.glob("*.nii*")):
        stem     = seg_path.name.split(".")[0]
        vol_path = vol_dir / seg_path.name
        if not vol_path.exists():
            vol_path = vol_dir / (stem + ".nii.gz")
        if not vol_path.exists():
            continue
        seg_nib  = nib.load(str(seg_path))
        seg_data = seg_nib.get_fdata(dtype=np.float32)
        if not all(np.any(seg_data == lbl) for lbl in LUMBAR_LABELS.values()):
            continue
        # inferior boundary of L5 in voxel z-axis
        l5_vox_z = np.where(seg_data == 24)[2]
        l5_z_min = int(l5_vox_z.min())
        if l5_z_min < 5:
            continue                      # L5 already at bottom of scan
        hu_data = nib.load(str(vol_path)).get_fdata(dtype=np.float32)
        below_count = int(np.sum(hu_data[:, :, :l5_z_min] > min_below_hu))
        if below_count >= min_below_voxels:
            cases.append((vol_path, seg_path))
            print(f"    ✓ {seg_path.stem}  (below-L5 voxels = {below_count})")
        else:
            print(f"    ✗ {seg_path.stem}  (below-L5 voxels = {below_count} < {min_below_voxels})")

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["volume_path", "seg_path"])
        w.writeheader()
        for vp, sp in cases:
            w.writerow({"volume_path": str(vp), "seg_path": str(sp)})
    print(f"  Cached {len(cases)} valid cases → {cache}")
    return cases


# ---------------------------------------------------------------------------
# Centroid helpers  (verbatim from viz script)
# ---------------------------------------------------------------------------

def compute_geometric_centroids(seg_path: Path):
    seg_nib  = nib.load(str(seg_path))
    seg_data = seg_nib.get_fdata(dtype=np.float32)
    affine   = seg_nib.affine
    centroids = {}
    for name, lbl in LUMBAR_LABELS.items():
        vox     = np.array(np.where(seg_data == lbl)).T
        vox_com = vox.mean(axis=0)
        centroids[name] = (affine @ np.append(vox_com, 1.0))[:3]
    return centroids, seg_data, affine


def compute_weighted_centroids(seg_data: np.ndarray, affine: np.ndarray,
                               vol_path: Path) -> dict:
    hu = nib.load(str(vol_path)).get_fdata(dtype=np.float32)
    centroids = {}
    for name, lbl in LUMBAR_LABELS.items():
        mask    = seg_data == lbl
        vox     = np.array(np.where(mask)).T
        weights = np.clip(hu[mask].astype(np.float64), 1.0, None)
        weights /= weights.sum()
        vox_com = (vox * weights[:, None]).sum(axis=0)
        centroids[name] = (affine @ np.append(vox_com, 1.0))[:3]
    return centroids


def pushed_centroids(geom_cents: dict, wt_cents: dict, push_mm: float = 15.0) -> np.ndarray:
    pts_geom = np.stack([geom_cents[k] for k in LUMBAR_LABELS])
    pts_wt   = np.stack([wt_cents[k]   for k in LUMBAR_LABELS])
    dirs     = pts_geom - pts_wt
    norms    = np.linalg.norm(dirs, axis=1, keepdims=True)
    unit_dirs = np.where(norms > 1e-6, dirs / norms, 0.0)
    return (pts_wt + unit_dirs * push_mm).astype(np.float32)   # (5,3) LPS mm


# ---------------------------------------------------------------------------
# Pose / projection helpers  (verbatim from viz script)
# ---------------------------------------------------------------------------

def make_pose_from_epnp(R: np.ndarray, t: np.ndarray,
                        spine_centre_lps: np.ndarray, device) -> tuple:
    spine_xzy = xzy(spine_centre_lps.reshape(1, 3)).flatten()
    pa_xzy    = R.T @ np.array([0., 0., 1.])
    pa_xzy   /= np.linalg.norm(pa_xzy)
    src_xzy   = spine_xzy - pa_xzy * SDD_MM
    t_new     = -(R @ src_xzy)
    L2R   = np.array([-1., -1., 1.])
    right = xzy_inv(R.T @ np.array([1., 0., 0.])).flatten() * L2R
    up    = xzy_inv(R.T @ np.array([0., 1., 0.])).flatten() * L2R
    pa    = xzy_inv(R.T @ np.array([0., 0., 1.])).flatten() * L2R
    src   = xzy_inv(-R.T @ t_new).flatten()                  * L2R
    up    = -up
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.stack([right, up, pa], axis=1)
    mat[:3,  3] = src
    pose = RigidTransform(torch.tensor(mat, dtype=torch.float32, device=device))
    return pose, R, t_new


def project_centroids_on_drr(pts_lps: np.ndarray, R: np.ndarray,
                              t: np.ndarray, pix_mm: float) -> np.ndarray:
    P_cam = (R @ xzy(pts_lps).T).T + t
    fx    = SDD_MM / pix_mm
    cx    = DET_SIZE / 2.0
    u     = fx * P_cam[:, 0] / P_cam[:, 2] + cx
    v     = fx * P_cam[:, 1] / P_cam[:, 2] + cx
    return np.stack([u, v], axis=1)   # (5,2) pixel coords


# ---------------------------------------------------------------------------
# Subject builder  (verbatim from viz script)
# ---------------------------------------------------------------------------

def _build_sitk_from_nib(vol_path: Path):
    nib_img  = nib.load(str(vol_path))
    data     = nib_img.get_fdata(dtype=np.float32).astype(np.int16)
    affine   = nib_img.affine
    spacing  = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).tolist()
    origin   = affine[:3, 3].tolist()
    data_zyx = data.transpose(2, 1, 0)
    img = sitk.GetImageFromArray(data_zyx)
    img.SetSpacing([float(s) for s in spacing])
    img.SetOrigin([float(o) for o in origin])
    img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    return img


def _build_subject(sitk_img, tmp_dir: str,
                   bone_mult: float = 4.0, min_hu: float = None):
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


# ---------------------------------------------------------------------------
# Gaussian heatmap
# ---------------------------------------------------------------------------

def make_gaussian_heatmap(u: float, v: float,
                          size: int, sigma: float = 10.0) -> np.ndarray:
    """Return (size × size) float32 Gaussian centred at pixel (u, v)."""
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma ** 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# Swaroopa AP frame loader
# ---------------------------------------------------------------------------

def load_swaroopa_frames_ap() -> list:
    lm_3d     = _load_mrk_json_3d(LM_3D_JSON)
    lm_2d_all = _load_landmarks_2d(LM_2D_JSON)
    frames    = []
    for png_path in sorted(XRAY_DIR_AP.glob("frame_*_z000.png")):
        frame_num = png_path.stem.split("_")[1]
        json_key  = f"frame_{frame_num}_z00"
        lm_2d     = lm_2d_all.get(json_key, {})
        common    = [l for l in sorted(lm_2d) if l in lm_3d]
        if len(common) < 3:
            continue
        pts3d = np.array([lm_3d[c] for c in common])
        pts2d = np.array([lm_2d[c] for c in common])
        try:
            R, t = _solve_pnp_swaro(pts3d, pts2d)
        except Exception:
            continue
        frames.append(dict(
            xray_raw=_load_png(png_path),
            pts2d=pts2d, labels=common,
            R=R, t=t, fname=png_path.name,
        ))
    return frames


# ---------------------------------------------------------------------------
# Single training-sample generator
# ---------------------------------------------------------------------------

def generate_sample(vol_path: Path, seg_path: Path,
                    R: np.ndarray, t: np.ndarray, device,
                    push_mm: float, pix_mm: float,
                    bone_mult: float, min_hu, gauss_sigma: float,
                    model_type: str = 'multi') -> tuple:
    """
    Render one AP DRR for `vol_path` using EPnP pose (R, t) anchored at the
    spine centroid.  Returns:
        drr_np   : (H, W) float32 in [0, 1]
        heatmaps : (5, H, W) float32 Gaussian targets
        proj_px  : (5, 2) projected centroid pixel coords
    """
    geom_cents, seg_data, affine = compute_geometric_centroids(seg_path)
    wt_cents  = compute_weighted_centroids(seg_data, affine, vol_path)
    pts_lps   = pushed_centroids(geom_cents, wt_cents, push_mm=push_mm)  # (5,3)
    spine_ctr = pts_lps.mean(axis=0)

    pose, R_new, t_new = make_pose_from_epnp(R, t, spine_ctr, device)

    sitk_img = _build_sitk_from_nib(vol_path)
    with tempfile.TemporaryDirectory() as tmp:
        subject = _build_subject(sitk_img, tmp, bone_mult=bone_mult, min_hu=min_hu)
        drr_mod = DRR(subject, sdd=SDD_MM, height=DET_SIZE, width=DET_SIZE,
                      delx=pix_mm, dely=pix_mm, x0=0.0, y0=0.0,
                      renderer="siddon", reverse_x_axis=False).to(device)
        with torch.no_grad():
            drr_tensor = drr_mod(pose)
        del drr_mod
    torch.cuda.empty_cache() if device.type == "cuda" else None

    drr_np = drr_tensor.squeeze().cpu().float().numpy()
    drr_np = np.clip(drr_np, 0, None)
    if drr_np.max() > 0:
        drr_np /= drr_np.max()

    proj_px  = project_centroids_on_drr(pts_lps, R_new, t_new, pix_mm)  # (5,2)
    per_ch = np.stack([
        make_gaussian_heatmap(proj_px[i, 0], proj_px[i, 1],
                              DET_SIZE, sigma=gauss_sigma)
        for i in range(5)
    ], axis=0)   # (5, H, W)

    if model_type == 'single':
        heatmaps = per_ch.max(axis=0, keepdims=True)   # (1, H, W) merged
    else:
        heatmaps = per_ch                              # (5, H, W)

    return drr_np.astype(np.float32), heatmaps, proj_px


# ---------------------------------------------------------------------------
# Training-data cache  (float16 .npz  ~134 MB / 1000 samples)
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path) -> Path:
    return cache_dir / 'train_cache.npz'


def load_cache(cache_dir: Path, n_required: int):
    """
    Load cached arrays if the cache exists.
    Returns (drrs, heatmaps, proj_pxs) as float32, or None if cache missing.
    If cache has fewer than n_required samples, returns what exists so the
    caller can top up.
    """
    cp = _cache_path(cache_dir)
    if not cp.exists():
        return None
    data = np.load(str(cp))
    n = int(data['drrs'].shape[0])
    if n < n_required:
        print(f"  Cache has {n} samples (need {n_required}) — will generate {n_required - n} more.")
    else:
        print(f"  Cache hit: {n} samples in {cp}  (using first {n_required})")
    return (data['drrs'][:n_required].astype(np.float32),
            data['heatmaps'][:n_required].astype(np.float32),
            data['proj_pxs'][:n_required].astype(np.float32))


def save_cache(cache_dir: Path,
               drrs: np.ndarray,
               heatmaps: np.ndarray,
               proj_pxs: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(cache_dir)
    np.savez_compressed(str(cp),
                        drrs=drrs.astype(np.float16),
                        heatmaps=heatmaps.astype(np.float16),
                        proj_pxs=proj_pxs.astype(np.float32))
    mb = cp.stat().st_size / 1e6
    print(f"  Cache saved → {cp}  ({mb:.1f} MB,  {drrs.shape[0]} samples)")


# ---------------------------------------------------------------------------
# Validation  (Swaroopa AP X-rays → heatmap peaks → distance to annotations)
# ---------------------------------------------------------------------------

# Bounding-box mask in original 1024×1024 pixel coords
MASK_X0, MASK_X1 = 300,700   # column range (inclusive)
MASK_Y0, MASK_Y1 = 0,   800   # row range    (inclusive)


def find_top5_maxima(heatmap: np.ndarray,
                     min_distance: int = 10) -> np.ndarray:
    """
    Find the 5 highest local maxima in a 2-D heatmap via non-maximum
    suppression with a square window of `min_distance` pixels.
    Returns (5, 2) array of (u, v) = (col, row) sorted by v ascending
    (top of image first → L1 … L5 assignment).
    If fewer than 5 peaks are found, duplicates the global maximum.
    """
    h, w   = heatmap.shape
    padded = np.pad(heatmap, min_distance, mode='edge')
    peaks  = []
    work   = heatmap.copy()
    for _ in range(5):
        idx      = int(work.argmax())
        v, u     = divmod(idx, w)
        peaks.append((u, v))
        # suppress neighbourhood
        r0 = max(0, v - min_distance); r1 = min(h, v + min_distance + 1)
        c0 = max(0, u - min_distance); c1 = min(w, u + min_distance + 1)
        work[r0:r1, c0:c1] = 0.0
    peaks_arr = np.array(peaks, dtype=np.float32)          # (5,2) u,v
    order     = np.argsort(peaks_arr[:, 1])                # sort by v (row)
    return peaks_arr[order]                                 # L1…L5 top→bottom


def apply_xray_mask(img: np.ndarray) -> np.ndarray:
    """
    Keep pixels inside (x: MASK_X0–MASK_X1, y: MASK_Y0–MASK_Y1) at their
    original value; replace the rest with the image mean.
    `img` should already be at DET_SIZE resolution.
    """
    h, w   = img.shape[:2]
    sx     = w / SWARO_IMG_SIZE   # scale from 1024 → DET_SIZE
    sy     = h / SWARO_IMG_SIZE
    x0     = max(0, int(round(MASK_X0 * sx)))
    x1     = min(w, int(round(MASK_X1 * sx)))
    y0     = max(0, int(round(MASK_Y0 * sy)))
    y1     = min(h, int(round(MASK_Y1 * sy)))
    out    = np.full_like(img, img.mean())
    out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    return out

def validate(model: nn.Module, val_frames: list, device, pix_mm: float,
             model_type: str = 'multi') -> float:
    """
    Mean Euclidean distance (px) from predicted heatmap peak to annotated
    2D landmark, evaluated on Swaroopa AP X-rays resized to DET_SIZE.
    For 'multi': argmax per channel → L1…L5.
    For 'single': find_top5_maxima on merged map, sort by v → L1…L5.
    """
    model.eval()
    label_order = list(LUMBAR_LABELS.keys())   # L1 … L5
    dists = []
    with torch.no_grad():
        for fr in val_frames:
            xray_sm  = cv2.resize(fr['xray_raw'], (DET_SIZE, DET_SIZE),
                                  interpolation=cv2.INTER_AREA)
            xray_inv = np.clip(1.0 - xray_sm, 0.0, 1.0).astype(np.float32)
            xray_inv = apply_xray_mask(xray_inv)
            inp  = torch.tensor(xray_inv[None, None], dtype=torch.float32, device=device)
            pred = model(inp).squeeze(0).cpu().numpy()   # (5,H,W) or (1,H,W)
            scale = DET_SIZE / SWARO_IMG_SIZE
            if model_type == 'single':
                peaks = find_top5_maxima(pred[0])   # (5,2) u,v sorted by v
                for j, lbl in enumerate(label_order):
                    if lbl not in fr['labels']:
                        continue
                    li    = fr['labels'].index(lbl)
                    u_ann = fr['pts2d'][li, 0] * scale
                    v_ann = fr['pts2d'][li, 1] * scale
                    u_pred, v_pred = peaks[j]
                    dists.append(float(np.hypot(u_pred - u_ann, v_pred - v_ann)))
            else:
                for j, lbl in enumerate(label_order):
                    if lbl not in fr['labels']:
                        continue
                    li    = fr['labels'].index(lbl)
                    u_ann = fr['pts2d'][li, 0] * scale
                    v_ann = fr['pts2d'][li, 1] * scale
                    flat_idx      = int(pred[j].argmax())
                    v_pred, u_pred = divmod(flat_idx, DET_SIZE)
                    dists.append(float(np.hypot(u_pred - u_ann, v_pred - v_ann)))
    model.train()
    return float(np.mean(dists)) if dists else float('inf')


# ---------------------------------------------------------------------------
# Dry-run: save (DRR | heatmap × 5) figure for each sample
# ---------------------------------------------------------------------------

def save_dry_run(samples: list, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_order = list(LUMBAR_LABELS.keys())
    for i, (drr_np, heatmaps, proj_px) in enumerate(samples):
        fig, axes = plt.subplots(1, 6, figsize=(20, 3.5))
        fig.patch.set_facecolor('#111111')

        # DRR
        axes[0].imshow(drr_np, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('DRR input', color='white', fontsize=8)
        for j in range(5):
            u, v = proj_px[j]
            if 0 <= u < DET_SIZE and 0 <= v < DET_SIZE:
                axes[0].plot(u, v, marker='D', color=LM_COLOURS[label_order[j]],
                             markersize=5, markeredgecolor='white', markeredgewidth=0.5)

        # Heatmaps
        for j, lbl in enumerate(label_order):
            axes[j + 1].imshow(heatmaps[j], cmap='hot', vmin=0, vmax=1)
            axes[j + 1].set_title(f'{lbl}  target', color='white', fontsize=8)
            u, v = proj_px[j]
            if 0 <= u < DET_SIZE and 0 <= v < DET_SIZE:
                axes[j + 1].plot(u, v, 'c+', markersize=10, markeredgewidth=2)

        for ax in axes:
            ax.axis('off')
            ax.set_facecolor('#111111')

        plt.tight_layout(pad=0.3)
        out = output_dir / f"dry_run_{i:02d}.png"
        plt.savefig(str(out), dpi=120, bbox_inches='tight', facecolor='#111111')
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Post-training plots
# ---------------------------------------------------------------------------

def save_loss_curves(train_history: list, val_history: list,
                     output_dir: Path) -> None:
    """Save training-loss and validation-distance curves to one PNG."""
    if not train_history:
        return
    train_iters, train_losses = zip(*train_history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#111111')

    ax1.set_facecolor('#1a1a1a')
    ax1.plot(train_iters, train_losses, color='#44aaff', linewidth=1.2)
    ax1.set_xlabel('Iteration', color='#cccccc')
    ax1.set_ylabel('MSE Loss', color='#cccccc')
    ax1.set_title('Training Loss', color='white')
    ax1.tick_params(colors='#888888')
    for sp in ax1.spines.values(): sp.set_color('#444444')

    ax2.set_facecolor('#1a1a1a')
    if val_history:
        val_iters, val_dists = zip(*val_history)
        ax2.plot(val_iters, val_dists, color='#ff8844', linewidth=1.5, marker='o',
                 markersize=4)
        ax2.axhline(min(val_dists), color='#ff4444', linestyle='--',
                    linewidth=0.8, label=f'best={min(val_dists):.2f}px')
        ax2.legend(facecolor='#222222', labelcolor='#cccccc', fontsize=8)
    ax2.set_xlabel('Iteration', color='#cccccc')
    ax2.set_ylabel('Mean dist to annotation (px)', color='#cccccc')
    ax2.set_title('Validation Distance', color='white')
    ax2.tick_params(colors='#888888')
    for sp in ax2.spines.values(): sp.set_color('#444444')

    plt.tight_layout(pad=1.0)
    out = output_dir / 'loss_curves.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close(fig)
    print(f"  Saved loss curves → {out}")


def save_centroid_overlay_grid(model: nn.Module, val_frames: list,
                               device, output_dir: Path,
                               thumb_size: int = 256,
                               model_type: str = 'multi') -> None:
    """
    Run the best model on every Swaroopa AP X-ray, overlay:
      ◆  predicted centroid  (filled diamond, label colour)
      +  annotated centroid  (cross, same colour)
    and save all frames as a single grid PNG.
    """
    model.eval()
    label_order = list(LUMBAR_LABELS.keys())
    scale       = DET_SIZE / SWARO_IMG_SIZE
    TS          = thumb_size
    N           = len(val_frames)
    NCOLS       = min(6, N)
    NROWS       = (N + NCOLS - 1) // NCOLS

    fig, axes = plt.subplots(NROWS, NCOLS,
                             figsize=(NCOLS * 2.2, NROWS * 2.6),
                             squeeze=False)
    fig.patch.set_facecolor('#111111')
    for ax in axes.flat:
        ax.axis('off'); ax.set_facecolor('#111111')

    with torch.no_grad():
        for idx, fr in enumerate(val_frames):
            row, col = divmod(idx, NCOLS)
            ax = axes[row][col]

            # prepare input
            xray_sm  = cv2.resize(fr['xray_raw'], (DET_SIZE, DET_SIZE),
                                  interpolation=cv2.INTER_AREA)
            xray_inv = np.clip(1.0 - xray_sm, 0.0, 1.0).astype(np.float32)
            xray_inv = apply_xray_mask(xray_inv)
            inp  = torch.tensor(xray_inv[None, None], dtype=torch.float32,
                                device=device)
            pred = model(inp).squeeze(0).cpu().numpy()   # (5,H,W) or (1,H,W)

            if model_type == 'single':
                peaks_uv = find_top5_maxima(pred[0])   # (5,2) sorted by v
            else:
                peaks_uv = None

            # display — show the masked image so the user sees what the model sees
            disp = cv2.resize(xray_inv, (TS, TS), interpolation=cv2.INTER_AREA)
            ax.imshow(disp, cmap='gray', vmin=0, vmax=1)
            sc = TS / DET_SIZE

            for j, lbl in enumerate(label_order):
                c = LM_COLOURS.get(lbl, 'white')

                # predicted peak
                if model_type == 'single':
                    u_pred, v_pred = peaks_uv[j]
                else:
                    flat_idx       = int(pred[j].argmax())
                    v_pred, u_pred = divmod(flat_idx, DET_SIZE)
                ax.plot(u_pred * sc, v_pred * sc, 'D',
                        color=c, markersize=5,
                        markeredgecolor='white', markeredgewidth=0.5)

                # annotated ground truth (if available)
                if lbl in fr['labels']:
                    li    = fr['labels'].index(lbl)
                    u_ann = fr['pts2d'][li, 0] * scale * sc
                    v_ann = fr['pts2d'][li, 1] * scale * sc
                    ax.plot(u_ann, v_ann, '+',
                            color=c, markersize=9, markeredgewidth=1.5)

            ax.set_title(fr['fname'], color='#aaaaaa', fontsize=5, pad=1)

    plt.suptitle(
        'Predicted centroids (◆) vs annotations (+)  —  best model',
        color='white', fontsize=9, y=1.01,
    )
    plt.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.2)
    out = output_dir / 'centroid_overlay_grid.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close(fig)
    print(f"  Saved centroid overlay grid → {out}")
    model.train()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Train U-Net for L1-L5 centroid detection from AP DRRs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- same flags as visualize_nnunet_centroids_vs_swaroopa.py ---
    p.add_argument('--nnunet_dir',    type=Path, default=Path('spine_segmentation_nnunet_v2'),
                   help='Root folder with volumes/ and segmentations/')
    p.add_argument('--n_cases',       type=int,  default=20,
                   help='Number of nnunet CTs to use for training')
    p.add_argument('--seed',          type=int,  default=42)
    p.add_argument('--push_mm',       type=float, default=15.0,
                   help='Centroid push distance from HU-weighted to geometric (mm)')
    p.add_argument('--delx',          type=float, default=PIX_MM_DEFAULT,
                   help='DRR pixel spacing (mm). Smaller = zoomed in.')
    p.add_argument('--bone_mult',     type=float, default=4.0,
                   help='bone_attenuation_multiplier for DiffDRR')
    p.add_argument('--min_hu',        type=float, default=None,
                   help='HU floor: voxels below this are clamped (None = no clamp)')
    p.add_argument('--rebuild_cache', action='store_true',
                   help='Ignore cached valid-case list and rescan from scratch')
    # --- training params ---
    p.add_argument('--n_iters',       type=int,  default=2000,
                   help='Number of gradient-update iterations')
    p.add_argument('--n_train',       type=int,  default=100,
                   help='Number of (DRR, heatmap) pairs to pre-generate')
    p.add_argument('--batch_size',    type=int,  default=4)
    p.add_argument('--lr',            type=float, default=1e-3,
                   help='Adam learning rate')
    p.add_argument('--gauss_sigma',   type=float, default=10.0,
                   help='Gaussian heatmap sigma (px)')
    p.add_argument('--model_type',    type=str, default='multi',
                   choices=['multi', 'single'],
                   help='multi: 5-channel heatmap output (one channel per vertebra). '
                        'single: 1-channel merged heatmap, peaks sorted by row → L1…L5.')
    p.add_argument('--output_dir',    type=Path, default=Path('results/unet_lumbar'),
                   help='Directory for checkpoints and logs')
    p.add_argument('--cache_dir',     type=Path, default=Path('results/unet_lumbar/drr_cache'),
                   help='Directory for cached DRR training samples (.npz). '
                        'Reused if enough samples exist; topped up otherwise.')
    # --- debug ---
    p.add_argument('--dry_run',       action='store_true',
                   help='Generate 5 training examples, save visualisations, and exit')
    p.add_argument('--val_only',      action='store_true',
                   help='Skip training: load checkpoint, run validation + overlay grid, and exit')
    p.add_argument('--checkpoint',    type=Path, default=None,
                   help='Path to .pt checkpoint to load (default: output_dir/unet_lumbar_best.pt)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"DET_SIZE={DET_SIZE}  delx={args.delx:.4f} mm/px  "
          f"bone_mult={args.bone_mult}  min_hu={args.min_hu}  "
          f"gauss_sigma={args.gauss_sigma}px")

    # ── Val-only mode ─────────────────────────────────────────────────────────
    if args.val_only:
        ckpt_path = args.checkpoint or (args.output_dir / 'unet_lumbar_best.pt')
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"\nVal-only mode — loading {ckpt_path}")
        swaro_frames = load_swaroopa_frames_ap()
        print(f"  {len(swaro_frames)} Swaroopa AP frames loaded.")
        model = build_model(args.model_type).to(device)
        ckpt  = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt['model_state'])
        val_dist = validate(model, swaro_frames, device, args.delx, args.model_type)
        print(f"  Validation distance : {val_dist:.2f} px")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_centroid_overlay_grid(model, swaro_frames, device, args.output_dir,
                                   model_type=args.model_type)
        return

    # ── Valid CTs ────────────────────────────────────────────────────────────
    seg_dir = args.nnunet_dir / 'segmentations'
    vol_dir = args.nnunet_dir / 'volumes'
    print("\nFinding valid cases (L1-L5 + below-L5 data) …")
    valid = get_valid_cases(seg_dir, vol_dir, rebuild=args.rebuild_cache)
    if not valid:
        raise RuntimeError("No valid cases found. Try --rebuild_cache.")
    chosen = random.sample(valid, min(args.n_cases, len(valid)))
    print(f"Using {len(chosen)} CTs.")

    # ── Swaroopa AP frames ───────────────────────────────────────────────────
    print("\nLoading Swaroopa AP frames …")
    swaro_frames = load_swaroopa_frames_ap()
    if not swaro_frames:
        raise RuntimeError("No Swaroopa AP frames found.")
    print(f"  {len(swaro_frames)} frames loaded.")

    # ── Pre-generate training data (with disk cache) ────────────────────────
    n_gen = 5 if args.dry_run else args.n_train
    print(f"\nPre-generating {n_gen} training samples …")
    dry_samples = []

    # Try loading from cache (skip for dry_run)
    cached = None if args.dry_run else load_cache(args.cache_dir, n_gen)
    if cached is not None:
        c_drrs, c_hm, c_px = cached
        n_have = int(c_drrs.shape[0])
    else:
        c_drrs = c_hm = c_px = None
        n_have = 0

    n_need = n_gen - n_have
    new_drrs, new_hm, new_px = [], [], []

    for i in range(n_need):
        vol_path, seg_path = chosen[(n_have + i) % len(chosen)]
        ep = swaro_frames[(n_have + i) % len(swaro_frames)]
        print(f"  [{n_have+i+1:3d}/{n_gen}] {seg_path.stem[:30]} + {ep['fname']} …",
              end=" ", flush=True)
        try:
            drr_np, hm, proj_px = generate_sample(
                vol_path, seg_path, ep['R'], ep['t'], device,
                args.push_mm, args.delx, args.bone_mult, args.min_hu,
                args.gauss_sigma, args.model_type,
            )
            new_drrs.append(drr_np)
            new_hm.append(hm)
            new_px.append(proj_px)
            print("done.")
        except Exception as exc:
            print(f"SKIP ({exc})")

    # Merge cached + newly generated into full arrays
    parts_d  = ([c_drrs]             if c_drrs is not None else []) + \
               ([np.stack(new_drrs)] if new_drrs else [])
    parts_h  = ([c_hm]               if c_hm   is not None else []) + \
               ([np.stack(new_hm)]   if new_hm  else [])
    parts_px = ([c_px]               if c_px   is not None else []) + \
               ([np.stack(new_px)]   if new_px  else [])

    if not parts_d:
        raise RuntimeError("No training samples generated.")

    train_drrs_arr     = np.concatenate(parts_d,  axis=0)[:n_gen]
    train_heatmaps_arr = np.concatenate(parts_h,  axis=0)[:n_gen]
    train_px_arr       = np.concatenate(parts_px, axis=0)[:n_gen]

    # Save updated cache if new samples were generated (not dry_run)
    if not args.dry_run and new_drrs:
        save_cache(args.cache_dir, train_drrs_arr, train_heatmaps_arr, train_px_arr)

    # Build dry_run vis list from merged arrays
    if args.dry_run:
        for i in range(min(5, len(train_drrs_arr))):
            dry_samples.append((train_drrs_arr[i], train_heatmaps_arr[i],
                                train_px_arr[i]))

    train_drrs     = list(train_drrs_arr)
    train_heatmaps = list(train_heatmaps_arr)

    # ── Dry-run exit ─────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\nDry run: saving {len(dry_samples)} example(s) …")
        save_dry_run(dry_samples, args.output_dir / 'dry_run')
        print("Done. Exiting (--dry_run).")
        return

    if not train_drrs:
        raise RuntimeError("No training samples generated.")

    # ── Tensors ──────────────────────────────────────────────────────────────
    X = torch.tensor(np.stack(train_drrs)[:, None],    dtype=torch.float32)  # (N,1,H,W)
    Y = torch.tensor(np.stack(train_heatmaps),         dtype=torch.float32)  # (N,5,H,W)
    print(f"\nDataset: {X.shape[0]} samples  X={tuple(X.shape)}  Y={tuple(Y.shape)}")

    # ── Model / optimiser ────────────────────────────────────────────────────
    model     = build_model(args.model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = nn.MSELoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_dist = float('inf')
    best_ckpt     = args.output_dir / 'unet_lumbar_best.pt'
    val_interval  = max(1, args.n_iters // 20)   # ~20 validation checkpoints

    # history for plots
    train_loss_history: list  = []   # (iter, loss)
    val_dist_history:   list  = []   # (iter, dist_px)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net: {n_params:,} parameters")
    print(f"Training for {args.n_iters} iters  "
          f"(batch={args.batch_size}, lr={args.lr}, val every {val_interval} iters)\n")

    # ── Training loop ────────────────────────────────────────────────────────
    for it in range(1, args.n_iters + 1):
        idx  = torch.randint(0, len(X), (args.batch_size,))
        xb   = X[idx].to(device)
        yb   = Y[idx].to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 50 == 0 or it == 1:
            print(f"  iter {it:5d}/{args.n_iters}  train_loss={loss.item():.6f}", end="")
            train_loss_history.append((it, float(loss.item())))

        if it % val_interval == 0 or it == args.n_iters:
            val_dist = validate(model, swaro_frames, device, args.delx, args.model_type)
            val_dist_history.append((it, val_dist))
            print(f"  val_dist={val_dist:.2f}px", end="")
            if val_dist < best_val_dist:
                best_val_dist = val_dist
                torch.save({
                    'iter':         it,
                    'model_state':  model.state_dict(),
                    'val_dist_px':  val_dist,
                    'args':         vars(args),
                }, str(best_ckpt))
                print(f"  ✓ best saved ({best_val_dist:.2f}px)", end="")

        if it % 50 == 0 or it == args.n_iters:
            print()

    print(f"\nTraining complete.")
    print(f"  Best validation distance : {best_val_dist:.2f} px")
    print(f"  Checkpoint               : {best_ckpt}")

    # ── Loss curves ──────────────────────────────────────────────────────────
    save_loss_curves(train_loss_history, val_dist_history, args.output_dir)

    # ── Centroid overlay grid on best model ──────────────────────────────────
    ckpt = torch.load(str(best_ckpt), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    save_centroid_overlay_grid(model, swaro_frames, device, args.output_dir,
                               model_type=args.model_type)


if __name__ == '__main__':
    main()
