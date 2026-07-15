#!/usr/bin/env python3
"""
visualize_lumbar_centroids_drr.py
==================================
Selects 5 random CT volumes from spine_segmentation_nnunet_v2 that have
all L1-L5 vertebrae segmented, computes their centroids, generates AP and
LAT DRRs using diffdrr, overlays the projected centroids, and saves a
5x2 visualisation grid.

Usage:
    python scripts/visualize_lumbar_centroids_drr.py \
        [--data_dir spine_segmentation_nnunet_v2] \
        [--out results/figures/lumbar_centroids_drr.png] \
        [--seed 42]
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import random
from pathlib import Path

import cv2
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import SimpleITK as sitk
import torchio as tio

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diffdrr.drr import DRR
from diffdrr.data import read as diffdrr_read
from diffdrr.pose import RigidTransform
from swaroopa_loader import (
    _solve_pnp_swaro,
    _load_mrk_json_3d,
    _load_landmarks_2d,
    _load_png,
    SWARO_K,
    SWARO_FX, SWARO_FY, SWARO_CX, SWARO_CY,
    SWARO_PIX_MM, SWARO_IMG_SIZE, SWARO_SID_MM,
    LM_3D_JSON, LM_2D_JSON, XRAY_DIR_AP, XRAY_DIR_LAT,
)
from deepfluoro_loader import xzy, xzy_inv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# nnUNet TotalSegmentator vertebra label IDs
LUMBAR_LABELS = {
    "L1": 20,
    "L2": 21,
    "L3": 22,
    "L4": 23,
    "L5": 24,
}

LUMBAR_COLORS = {
    "L1": "#e74c3c",
    "L2": "#e67e22",
    "L3": "#f1c40f",
    "L4": "#2ecc71",
    "L5": "#3498db",
}

# DRR parameters
SDD_MM   = 1020.0   # source-to-detector distance
DET_SIZE = 256      # pixels
PIX_MM   = 1.5      # pixel spacing in mm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CACHE_CSV = "spine_segmentation_nnunet_v2/valid_lumbar_cases.csv"


def get_cases_with_all_lumbar(seg_dir: Path, vol_dir: Path) -> list:
    """Return list of (volume_path, seg_path) for cases with L1-L5.

    Results are cached in CACHE_CSV so the expensive label scan only runs once.
    """
    cache_path = Path(CACHE_CSV)

    # ── Load from cache if available ────────────────────────────────────────
    if cache_path.exists():
        print(f"  Loading valid cases from cache: {cache_path}")
        cases = []
        with open(cache_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vp = Path(row["volume_path"])
                sp = Path(row["seg_path"])
                if vp.exists() and sp.exists():
                    cases.append((vp, sp))
        print(f"  {len(cases)} cases loaded from cache.")
        return cases

    # ── Full scan and save cache ─────────────────────────────────────────────
    print("  Cache not found — scanning segmentations (this may take a while)…")
    cases = []
    all_segs = sorted(seg_dir.glob("*.nii*"))
    total = len(all_segs)
    for i, seg_path in enumerate(all_segs, 1):
        print(f"  [{i}/{total}] {seg_path.name}", end="\r", flush=True)
        stem = seg_path.name.split(".")[0]
        vol_path = vol_dir / seg_path.name
        if not vol_path.exists():
            vol_path = vol_dir / (stem + ".nii.gz")
        if not vol_path.exists():
            continue

        seg = nib.load(str(seg_path))
        data = seg.get_fdata(dtype=np.float32)
        has_all = all(np.any(data == lbl) for lbl in LUMBAR_LABELS.values())
        if has_all:
            cases.append((vol_path, seg_path))

    print()  # newline after \r progress
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["volume_path", "seg_path"])
        writer.writeheader()
        for vp, sp in cases:
            writer.writerow({"volume_path": str(vp), "seg_path": str(sp)})
    print(f"  Cached {len(cases)} valid cases → {cache_path}")
    return cases


def compute_geometric_centroids(seg_path: Path):
    """Geometric centre of mass (unit mass per voxel).
    Returns dict {name: np.array([x,y,z])} in world (LPS mm).
    """
    seg_nib  = nib.load(str(seg_path))
    seg_data = seg_nib.get_fdata(dtype=np.float32)
    affine   = seg_nib.affine

    centroids = {}
    for name, lbl in LUMBAR_LABELS.items():
        vox_coords = np.array(np.where(seg_data == lbl)).T   # (N,3)
        vox_com    = vox_coords.mean(axis=0)
        world      = (affine @ np.append(vox_com, 1.0))[:3]
        centroids[name] = world
    return centroids, seg_data, affine


def compute_weighted_centroids(seg_data: np.ndarray, affine: np.ndarray,
                               vol_path: Path):
    """CT HU-intensity-weighted centre of mass.
    Returns dict {name: np.array([x,y,z])} in world (LPS mm).
    Voxels with HU <= 0 are clamped to 1 to keep the weighting meaningful.
    """
    vol_nib = nib.load(str(vol_path))
    hu      = vol_nib.get_fdata(dtype=np.float32)

    centroids = {}
    for name, lbl in LUMBAR_LABELS.items():
        mask       = seg_data == lbl
        vox_coords = np.array(np.where(mask)).T          # (N,3)
        weights    = np.clip(hu[mask].astype(np.float64), 1.0, None)
        weights   /= weights.sum()
        vox_com    = (vox_coords * weights[:, None]).sum(axis=0)
        world      = (affine @ np.append(vox_com, 1.0))[:3]
        centroids[name] = world
    return centroids


def load_epnp_poses(seed: int):
    """
    Pick one random AP and one random LAT X-ray from the Swaroopa labelled
    dataset, run EPnP with the annotated 2D/3D landmarks to get (R, t), and
    return a dict with 'ap' and 'lat' entries, each containing:
        R, t         : EPnP pose
        xray         : float32 [0,1] grayscale image (SWARO_IMG_SIZE)
        pts2d        : (N,2) annotation pixel coords on the original xray
        labels       : list of landmark names corresponding to pts2d
    """
    rng = random.Random(seed)
    lm_3d     = _load_mrk_json_3d(LM_3D_JSON)
    lm_2d_all = _load_landmarks_2d(LM_2D_JSON)

    result = {}
    for view, xray_dir in [("ap", XRAY_DIR_AP), ("lat", XRAY_DIR_LAT)]:
        frames = sorted(xray_dir.glob("frame_*_z000.png"))
        if not frames:
            raise FileNotFoundError(f"No X-rays found in {xray_dir}")
        chosen_png = rng.choice(frames)
        frame_num  = chosen_png.stem.split("_")[1]
        json_key   = f"frame_{frame_num}_z00"

        lm_2d  = lm_2d_all.get(json_key, {})
        common = [l for l in sorted(lm_2d) if l in lm_3d]
        if len(common) < 3:
            raise RuntimeError(
                f"EPnP needs ≥3 shared landmarks for {chosen_png.name}; "
                f"got {len(common)}: {common}")

        pts3d = np.array([lm_3d[l]  for l in common])
        pts2d = np.array([lm_2d[l]  for l in common])   # (N,2) on SWARO_IMG_SIZE
        R, t  = _solve_pnp_swaro(pts3d, pts2d)
        xray  = _load_png(chosen_png)                    # float32 [0,1]
        print(f"  EPnP {view}: {chosen_png.name}  ({len(common)} lm: {common})")
        result[view] = dict(R=R, t=t, xray=xray, pts2d=pts2d, labels=common,
                            fname=chosen_png.name)
    return result


def project_epnp_to_drr(world_pts_lps: np.ndarray,
                        R: np.ndarray, t: np.ndarray,
                        det_size: int, pix_mm: float,
                        sid_mm: float) -> np.ndarray:
    """
    Project LPS world points to DRR pixel coords.
    R and t are in straight LPS space (output of make_pose_from_epnp),
    so NO xzy reorder is applied here.
    Returns (N,2) pixel coordinates on a (det_size × det_size) image.
    """
    pts = np.atleast_2d(world_pts_lps)
    P_cam = (R @ pts.T).T + t                            # (N,3) in camera space
    fx = sid_mm / pix_mm
    fy = sid_mm / pix_mm
    cx = (det_size - 1) / 2.0
    cy = (det_size - 1) / 2.0
    u  = fx * P_cam[:, 0] / P_cam[:, 2] + cx
    v  = fy * P_cam[:, 1] / P_cam[:, 2] + cy
    return np.stack([u, v], axis=1)


def solve_per_ct_pose(pts_wt_lps: np.ndarray, label_order: list,
                      pts2d_xray: np.ndarray, xray_labels: list) -> tuple:
    """
    Solve PnP using the nnunet CT centroids (pts3d) and the xray 2D
    annotations (pts2d).  Only uses labels present in both sets.

    pts_wt_lps  : (5,3) array for L1-L5 in LUMBAR_LABELS order
    label_order : list of label names matching pts_wt_lps rows
    pts2d_xray  : (N,2) annotation coords in original SWARO_IMG_SIZE space
    xray_labels : list of label names matching pts2d_xray rows

    Returns (R, t) in Swaroopa xzy-LPS convention, or None if <3 matches.
    """
    common = [l for l in xray_labels if l in label_order]
    if len(common) < 3:
        return None, None
    idx3d = [label_order.index(l) for l in common]
    idx2d = [xray_labels.index(l)  for l in common]
    # Convert LPS → RAS: _solve_pnp_swaro expects RAS world coords
    pts3d_ras = pts_wt_lps[idx3d] * np.array([-1., -1., 1.])
    pts2d     = pts2d_xray[idx2d]
    try:
        R, t = _solve_pnp_swaro(pts3d_ras, pts2d)
        return R, t
    except Exception:
        return None, None


def make_pose_from_epnp(R: np.ndarray, t: np.ndarray,
                        ct_spacing, ct_origin, vol_shape_zyx,
                        device: torch.device) -> RigidTransform:
    """
    Recentre a Swaroopa EPnP pose (R, t) to look at a different CT volume.

    Convention: P_cam = R @ xzy(P_world_ras) + t
    (Swaroopa landmarks and CT centroids are in RAS mm.)

    Returns (RigidTransform, R, t_new) where t_new recentres the source
    to SDD_MM behind the target CT volume centre.
    """
    nz, ny, nx = vol_shape_zyx
    sx, sy, sz = ct_spacing
    ox, oy, oz = ct_origin                  # LPS from NIfTI affine
    cx_lps = ox + (nx - 1) * sx / 2.0
    cy_lps = oy + (ny - 1) * sy / 2.0
    cz_lps = oz + (nz - 1) * sz / 2.0

    # Convert CT centre LPS → RAS (TorchIO/diffdrr world frame)
    centre_ras = np.array([-cx_lps, -cy_lps, cz_lps])

    # Principal axis in world RAS from EPnP orientation
    pa_xzy = R.T @ np.array([0., 0., 1.])   # xzy-RAS space
    pa_ras = np.array([pa_xzy[0], pa_xzy[2], pa_xzy[1]])  # → xyz-RAS
    pa_ras = pa_ras / np.linalg.norm(pa_ras)

    # Source SDD_MM behind the CT centre
    source_ras = centre_ras - pa_ras * SDD_MM

    # New t: recentre source for this CT (keep EPnP orientation)
    src_xzy = np.array([source_ras[0], source_ras[2], source_ras[1]])
    t_new   = -(R @ src_xzy)

    # Convert to diffdrr RigidTransform using pose_from_extrinsic logic
    # (mirrors run_swaroopa_diffdrr.py::pose_from_extrinsic)
    L2R   = np.array([-1., -1., 1.])
    right = xzy_inv(R.T @ np.array([1., 0., 0.])).flatten() * L2R
    up    = xzy_inv(R.T @ np.array([0., 1., 0.])).flatten() * L2R
    pa    = xzy_inv(R.T @ np.array([0., 0., 1.])).flatten() * L2R
    src   = xzy_inv(-R.T @ t_new).flatten()                 * L2R
    up    = -up

    R_pose = np.stack([right, up, pa], axis=1).astype(np.float32)
    t_pose = src.astype(np.float32)
    mat    = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R_pose
    mat[:3,  3] = t_pose
    pose = RigidTransform(torch.tensor(mat, dtype=torch.float32, device=device))

    return pose, R, t_new


def build_sitk_from_nib(vol_path: Path):
    """Load a NIfTI volume and return a SimpleITK image with correct geometry."""
    nib_img  = nib.load(str(vol_path))
    data     = nib_img.get_fdata(dtype=np.float32).astype(np.int16)
    affine   = nib_img.affine

    # Extract spacing and origin from affine (assume diagonal or sform/qform)
    # spacing = voxel sizes (mm)
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).tolist()
    origin  = affine[:3, 3].tolist()

    # NIfTI data layout: (X, Y, Z) → sitk expects Z,Y,X when GetImageFromArray
    # nib returns (i, j, k); we need to transpose to (k, j, i) = (Z, Y, X)
    data_zyx = data.transpose(2, 1, 0)

    sitk_img = sitk.GetImageFromArray(data_zyx)
    sitk_img.SetSpacing([float(s) for s in spacing])
    sitk_img.SetOrigin([float(o) for o in origin])
    sitk_img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    return sitk_img, spacing, origin


def build_subject(sitk_img: sitk.Image, tmp_dir: str) -> tio.Subject:
    """Write sitk image to temp NRRD and load as diffdrr Subject."""
    tmp_path = os.path.join(tmp_dir, "vol.nrrd")
    sitk.WriteImage(sitk_img, tmp_path)
    subject = diffdrr_read(
        tmp_path,
        orientation=None,
        center_volume=False,
        bone_attenuation_multiplier=4.0,
    )
    return subject


def build_drr_module(subject: tio.Subject, device: torch.device) -> DRR:
    drr = DRR(
        subject,
        sdd=SDD_MM,
        height=DET_SIZE,
        width=DET_SIZE,
        delx=PIX_MM,
        dely=PIX_MM,
        x0=0.0,
        y0=0.0,
        renderer="siddon",
        reverse_x_axis=False,
    ).to(device)
    return drr


def make_pose(R: np.ndarray, t: np.ndarray, device: torch.device) -> RigidTransform:
    """
    Convert a 3x3 rotation matrix R and translation vector t (LPS world)
    to a diffdrr RigidTransform (4×4 SE(3) matrix).

    Convention: P_cam = R @ P_world_lps + t
    diffdrr world frame is RAS, so we apply LPS→RAS flip (negate X, Y).

    Follows the same pattern as run_swaroopa_diffdrr.py:
        R_pose columns = [right | up | principal]  (world vectors)
        t_pose         = source position in world
    """
    L2R = np.array([-1.0, -1.0, 1.0])

    right = (R.T @ np.array([1., 0., 0.])) * L2R
    up    = (R.T @ np.array([0., 1., 0.])) * L2R
    pa    = (R.T @ np.array([0., 0., 1.])) * L2R
    src   = (-R.T @ t) * L2R

    # Negate up to reconcile detector vertical-flip convention (same as reference)
    up = -up

    R_pose = np.stack([right, up, pa], axis=1).astype(np.float32)  # columns
    t_pose = src.astype(np.float32)

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R_pose
    mat[:3,  3] = t_pose

    return RigidTransform(torch.tensor(mat, dtype=torch.float32, device=device))


def drr_to_image(drr_tensor: torch.Tensor) -> np.ndarray:
    """Convert DRR output tensor to uint8 display image.
    DiffDRR output is a ray integral: higher value = more attenuation = bone.
    We normalize and display directly so bone appears bright.
    """
    img = drr_tensor.squeeze().cpu().float().numpy()
    img = np.clip(img, 0, None)
    if img.max() > 0:
        img = img / img.max()
    return (img * 255).astype(np.uint8)


def project_world_to_detector(world_pts: np.ndarray,
                               R: np.ndarray, t: np.ndarray,
                               sdd: float, pix_mm: float,
                               det_size: int) -> np.ndarray:
    """
    Project world LPS points onto the detector plane.
    Returns pixel coordinates (u, v) on the (det_size x det_size) image.

    Pinhole model:  p_cam = R @ p_world + t
    Principal point is at (det_size/2, det_size/2).
    """
    pts_cam = (R @ world_pts.T).T + t   # (N, 3)
    # x_cam, y_cam, z_cam  (z_cam is depth along principal axis)
    z = pts_cam[:, 2]
    # perspective divide
    u_mm = pts_cam[:, 0] * sdd / z
    v_mm = pts_cam[:, 1] * sdd / z

    cx = det_size / 2.0
    cy = det_size / 2.0

    u_px = cx + u_mm / pix_mm
    v_px = cy + v_mm / pix_mm
    return np.stack([u_px, v_px], axis=1)


def make_ap_pose(centroids_lps: np.ndarray, spacing, origin, vol_shape_zyx):
    """
    AP view: X-ray source along +Y (anterior), detector at -Y.
    Returns (R, t) in LPS world.
    """
    # Volume centre in world
    nz, ny, nx = vol_shape_zyx
    sx, sy, sz = spacing
    ox, oy, oz = origin
    cx = ox + (nx - 1) * sx / 2.0
    cy = oy + (ny - 1) * sy / 2.0
    cz = oz + (nz - 1) * sz / 2.0
    centre = np.array([cx, cy, cz])

    # Source: along +Y from centre
    source = centre + np.array([0.0, SDD_MM / 2.0, 0.0])

    # Camera axes for AP view:
    # principal: from source toward centre → -Y
    principal = np.array([0., -1., 0.])
    # right: +X (Left → right on detector)
    right = np.array([1., 0., 0.])
    # up = [0,0,-1] (Inferior world direction = downward on detector)
    # This means Superior/L1 → small row index → top of image
    up = np.array([0., 0., -1.])

    # R such that P_cam = R @ P_world + t  (R = [right|up|principal]^T)
    R = np.stack([right, up, principal], axis=0)
    t = -R @ source
    return R, t, source


def make_lat_pose(centroids_lps: np.ndarray, spacing, origin, vol_shape_zyx):
    """
    LAT view: X-ray source along +X (left lateral), detector at -X.
    Returns (R, t) in LPS world.
    """
    nz, ny, nx = vol_shape_zyx
    sx, sy, sz = spacing
    ox, oy, oz = origin
    cx = ox + (nx - 1) * sx / 2.0
    cy = oy + (ny - 1) * sy / 2.0
    cz = oz + (nz - 1) * sz / 2.0
    centre = np.array([cx, cy, cz])

    source = centre + np.array([SDD_MM / 2.0, 0.0, 0.0])

    # principal: from source toward centre → -X
    principal = np.array([-1., 0., 0.])
    # up = [0,0,-1] (Inferior world direction = downward on detector)
    # This means Superior/L1 → top of image
    up = np.array([0., 0., -1.])
    # right = cross(up, principal) to form orthonormal frame
    right = np.cross(up, principal)       # = [0,-1,0] × [-1,0,0] → [0,1,0]? let's compute
    right = right / np.linalg.norm(right)

    R = np.stack([right, up, principal], axis=0)
    t = -R @ source
    return R, t, source


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="spine_segmentation_nnunet_v2",
                        help="Root folder containing volumes/ and segmentations/")
    parser.add_argument("--out", default="results/figures/lumbar_centroids_drr.png",
                        help="Output figure path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_mm", type=float, default=5.0,
                        help="Extra displacement (mm) applied to the HU-weighted CoM "
                             "along the geom→weighted direction (default: 5.0)")
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force rescan segmentations even if cache CSV exists")
    args = parser.parse_args()

    if args.rebuild_cache and Path(CACHE_CSV).exists():
        Path(CACHE_CSV).unlink()
        print("Cache deleted – will rescan.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    root     = Path(args.data_dir)
    seg_dir  = root / "segmentations"
    vol_dir  = root / "volumes"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Collect valid cases ──────────────────────────────────────────────────
    print("Scanning for cases with full L1-L5 segmentations…")
    valid = get_cases_with_all_lumbar(seg_dir, vol_dir)
    print(f"  Found {len(valid)} valid cases.")
    if len(valid) < 5:
        raise RuntimeError(f"Need at least 5 valid cases, found {len(valid)}.")

    chosen = random.sample(valid, 5)

    # ── EPnP poses from random Swaroopa X-rays ───────────────────────────────
    print("\nComputing EPnP poses from random Swaroopa X-rays…")
    epnp_poses = load_epnp_poses(args.seed)

    # ── Figure layout: 5 rows × 4 cols  [AP xray | AP DRR | LAT xray | LAT DRR] ─
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    fig.suptitle("Lumbar Vertebrae Centroids (L1–L5)  —  EPnP X-ray & DRR",
                 fontsize=14, fontweight="bold", y=1.005)
    # Column headers
    for col, title in enumerate(["AP X-ray (EPnP)", "AP DRR", "LAT X-ray (EPnP)", "LAT DRR"]):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold", pad=12)

    from matplotlib.lines import Line2D
    legend_patches = [
        mpatches.Patch(color=LUMBAR_COLORS[k], label=k) for k in LUMBAR_LABELS
        if k in [ep['labels'] for ep in epnp_poses.values()][0] or True
    ] + [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='gray', markersize=8, label='EPnP annotation (xray)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
               markeredgecolor='gray', markersize=7,
               label=f'Pushed HU-weighted CoM (+{args.push_mm:.0f}mm, DRR)'),
    ]

    for row, (vol_path, seg_path) in enumerate(chosen):
        case_name = seg_path.stem
        print(f"\n[{row+1}/5] Processing {case_name}…")

        # 1. Centroids in world (LPS mm)
        geom_cents, seg_data_cached, affine_cached = compute_geometric_centroids(seg_path)
        wt_cents   = compute_weighted_centroids(seg_data_cached, affine_cached, vol_path)
        pts_geom   = np.stack([geom_cents[k] for k in LUMBAR_LABELS], axis=0)  # (5,3)
        pts_wt_raw = np.stack([wt_cents[k]   for k in LUMBAR_LABELS], axis=0)

        # Push the HU-weighted CoM further along the geom→weighted vector
        PUSH_MM = args.push_mm
        # directions = pts_wt_raw - pts_geom                          # (5,3)
        directions = - pts_wt_raw + pts_geom                          # (5,3)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        # where geom and weighted coincide, keep point as-is
        safe = (norms > 1e-6).flatten()
        unit_dirs = np.where(norms > 1e-6, directions / norms, 0.0)
        pts_wt = pts_wt_raw + unit_dirs * PUSH_MM

        # use geometric centroids for pose (spine centre estimate)
        pts_world  = pts_geom

        # 2. Load volume geometry
        nib_img  = nib.load(str(vol_path))
        vol_data = nib_img.get_fdata(dtype=np.float32)   # (X,Y,Z) for nib
        affine   = nib_img.affine
        spacing  = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).tolist()
        origin   = affine[:3, 3].tolist()
        # nib shape is (i,j,k) ~ (X,Y,Z); we need (Z,Y,X) shape for zyx
        vol_shape_zyx = (vol_data.shape[2], vol_data.shape[1], vol_data.shape[0])

        # 3. Build SimpleITK image and diffdrr subject
        sitk_img, _, _ = build_sitk_from_nib(vol_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            subject = build_subject(sitk_img, tmp_dir)

            drr_module = build_drr_module(subject, device)

            for view_idx, view in enumerate(["AP", "LAT"]):
                epnp_key = "ap" if view == "AP" else "lat"
                ep       = epnp_poses[epnp_key]
                xray_img   = ep['xray']        # float32 [0,1], SWARO_IMG_SIZE
                pts2d_orig = ep['pts2d']       # (N,2) on SWARO_IMG_SIZE, unflipped
                lm_labels  = ep['labels']

                xray_col = view_idx * 2
                drr_col  = view_idx * 2 + 1

                # ── X-ray panel ────────────────────────────────────────────
                ax_xray = axes[row, xray_col]
                ax_xray.imshow(xray_img, cmap="gray", origin="upper")
                for i, lbl in enumerate(lm_labels):
                    color = LUMBAR_COLORS.get(lbl, "white")
                    u, v  = pts2d_orig[i]
                    ax_xray.plot(u, v, "o", color=color, markersize=8,
                                 markeredgecolor="white", markeredgewidth=0.8,
                                 zorder=5)
                    ax_xray.text(u + 8, v - 8, lbl, color=color,
                                 fontsize=7, fontweight="bold",
                                 bbox=dict(boxstyle="round,pad=0.1",
                                           fc="black", alpha=0.4, lw=0))
                ax_xray.set_title(f"{ep['fname']}  {view}", fontsize=8)
                ax_xray.axis("off")

                # ── Per-CT PnP: find pose that maps nnunet centroids → xray pts ─
                label_order = list(LUMBAR_LABELS.keys())  # L1..L5
                R_ct, t_ct  = solve_per_ct_pose(
                    pts_wt, label_order, pts2d_orig, lm_labels)
                if R_ct is None:
                    # fall back to shared EPnP orientation
                    R_ct, t_ct = ep['R'], ep['t']
                    print(f"  [WARN] per-CT PnP failed for {view}, using shared EPnP pose")

                # ── DRR panel ───────────────────────────────────────────────
                print(f"  Rendering {view}…", end=" ", flush=True)

                pose, R, t = make_pose_from_epnp(
                    R_ct, t_ct, spacing, origin, vol_shape_zyx, device)

                with torch.no_grad():
                    drr_tensor = drr_module(pose)

                img_u8 = drr_to_image(drr_tensor)
                print("done.")

                # Project centroids onto DRR.
                # R, t are from make_pose_from_epnp: P_cam = R @ xzy(P_ras) + t
                # pts_wt are in LPS (NIfTI) → convert to RAS first.
                pts_ras = pts_wt * np.array([-1., -1., 1.])
                P_cam   = (R @ xzy(pts_ras).T).T + t
                fx_drr  = SDD_MM / PIX_MM
                cx_drr  = DET_SIZE / 2.0
                u_drr   = fx_drr * P_cam[:, 0] / P_cam[:, 2] + cx_drr
                v_drr   = fx_drr * P_cam[:, 1] / P_cam[:, 2] + cx_drr
                proj_wt_px = np.stack([u_drr, v_drr], axis=1)

                ax_drr = axes[row, drr_col]
                ax_drr.imshow(img_u8, cmap="gray", origin="upper",
                              extent=[0, DET_SIZE, DET_SIZE, 0])

                for i, (name, color) in enumerate(LUMBAR_COLORS.items()):
                    uw, vw = proj_wt_px[i]
                    if 0 <= uw < DET_SIZE and 0 <= vw < DET_SIZE:
                        ax_drr.plot(uw, vw, "D", color=color, markersize=8,
                                    markeredgecolor="white", markeredgewidth=0.8,
                                    zorder=5)
                        ax_drr.text(uw + 4, vw - 4, name, color=color,
                                    fontsize=7, fontweight="bold",
                                    bbox=dict(boxstyle="round,pad=0.1",
                                              fc="black", alpha=0.4, lw=0))

                ax_drr.set_title(f"{case_name}  {view} DRR", fontsize=8)
                ax_drr.axis("off")

        del drr_module
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Legend ────────────────────────────────────────────────────────────────
    fig.legend(handles=legend_patches, loc="lower center", ncol=7,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved figure → {out_path}")


if __name__ == "__main__":
    main()
