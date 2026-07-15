#!/usr/bin/env python3
"""
Export Swaroopa DRRs using EPnP initialization pose per X-ray (DiffDRR renderer).

For each selected projection, this script:
1) loads Swaroopa data via SwaroLoader (which computes EPnP init pose),
2) renders a DRR at that init pose using diffdrr (Siddon ray-casting),
3) saves the DRR with the corresponding X-ray filename,
4) stores all init poses in a JSON file.

Coordinate convention (mirrors run_swaroopa_diffdrr.py):
  - Our calibrated extrinsic: P_cam = R_proj @ xzy(P_world_lps) + t_proj
  - DiffDRR expects poses in TorchIO-RAS world; pose_from_extrinsic() applies
    the LPS→RAS sign-flip (X→−X, Y→−Y, Z unchanged) + up-axis negation.
  - CT is loaded with center_volume=False so world-space source coords stay valid.
"""

import argparse
import json
import sys
import tempfile
import os
# Reduce GPU memory fragmentation for DiffDRR Siddon (must be set before torch import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torchio as tio
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from swaroopa_loader import (
    SwaroLoader,
    SWARO_IMG_SIZE,
    SWARO_PIX_MM,
    SWARO_SID_MM,
    SWARO_FX,
    SWARO_FY,
    SWARO_CX,
    SWARO_CY,
)
from deepfluoro_loader import DeepFluoroSpecimen, xzy_inv
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
from diffdrr.data import read as diffdrr_read

# Principal-point offset in mm (matches run_swaroopa_diffdrr.py)
_X0_MM = (SWARO_CX - SWARO_IMG_SIZE / 2.0) * SWARO_PIX_MM
_Y0_MM = (SWARO_CY - SWARO_IMG_SIZE / 2.0) * SWARO_PIX_MM

# LPS → RAS sign flip
_L2R = np.array([-1., -1., 1.])


# ---------------------------------------------------------------------------
# DiffDRR helpers (ported from run_swaroopa_diffdrr.py)
# ---------------------------------------------------------------------------

def build_subject(spec: DeepFluoroSpecimen) -> tio.Subject:
    """Build a TorchIO Subject from the specimen CT (center_volume=False)."""
    sitk_img = sitk.GetImageFromArray(spec.ct_volume.astype(np.int16))
    sitk_img.SetOrigin([float(v) for v in spec.ct_origin])
    sitk_img.SetSpacing([float(v) for v in spec.ct_spacing])
    sitk_img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

    with tempfile.NamedTemporaryFile(suffix='.nrrd', delete=False) as f:
        tmp_path = f.name
    sitk.WriteImage(sitk_img, tmp_path)

    subject = diffdrr_read(
        tmp_path,
        orientation=None,
        center_volume=False,
        bone_attenuation_multiplier=4.0,
    )
    os.unlink(tmp_path)
    return subject


def pose_from_extrinsic(R: np.ndarray, t: np.ndarray,
                        device: torch.device) -> RigidTransform:
    """Convert our (R_proj, t_proj) → diffdrr RigidTransform (LPS→RAS aware).

    Matches pose_from_extrinsic() in run_swaroopa_diffdrr.py exactly.
    """
    right = xzy_inv(R.T @ np.array([1., 0., 0.])).flatten() * _L2R
    up    = xzy_inv(R.T @ np.array([0., 1., 0.])).flatten() * _L2R
    pa    = xzy_inv(R.T @ np.array([0., 0., 1.])).flatten() * _L2R
    src   = xzy_inv(-R.T @ t).flatten() * _L2R

    # Negate up to reconcile diffdrr's upward-row convention with ours
    up = -up

    R_pose = np.stack([right, up, pa], axis=1).astype(np.float32)
    t_pose = src.astype(np.float32)

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R_pose
    mat[:3,  3] = t_pose
    return RigidTransform(torch.tensor(mat, dtype=torch.float32, device=device))


@torch.no_grad()
def render_drr(drr_mod: DRR,
              R: np.ndarray,
              t: np.ndarray,
              device: torch.device) -> np.ndarray:
    """Render one DRR via a pre-built DRR module; return float32 [0,1] array."""
    pose = pose_from_extrinsic(R, t, device)
    img = drr_mod(pose)                          # (1, 1, H, W)
    img = img.squeeze().cpu().numpy()            # (H, W)
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    return img.astype(np.float32)


def build_drr_module(subject: tio.Subject,
                     output_size: int,
                     pix_mm: float,
                     device: torch.device) -> DRR:
    """Build and return a DiffDRR module at the given resolution."""
    return DRR(
        subject,
        sdd=SWARO_SID_MM,
        height=output_size,
        width=output_size,
        delx=pix_mm,
        dely=pix_mm,
        x0=_X0_MM,
        y0=_Y0_MM,
        renderer="siddon",
        reverse_x_axis=False,
    ).to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Swaroopa DRRs at EPnP init pose and save poses JSON"
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        default=None,
        help="Optional projection keys, e.g. ap_001 lat_000 (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/swaroopa_epnp_drrs"),
        help="Directory where DRRs are saved (view-wise subfolders are created)",
    )
    parser.add_argument(
        "--poses_json",
        type=Path,
        default=Path("results/swaroopa_epnp_poses.json"),
        help="Output JSON path for EPnP init poses",
    )
    parser.add_argument(
        "--drr_size",
        type=int,
        default=512,
        help="Rendered DRR size in pixels (square, default 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for DRR generation ('cuda' or 'cpu'). Default: auto",
    )
    return parser.parse_args()


def proj_key_to_xray_info(proj_key: str) -> Tuple[str, str]:
    """Map proj_key ('ap_001'/'lat_001') to view subfolder + xray filename."""
    view_tag, frame_idx = proj_key.split("_", 1)
    if view_tag == "ap":
        view_dir = "ap"
    elif view_tag == "lat":
        view_dir = "lateral"
    else:
        raise ValueError(f"Unknown view tag in proj_key: {proj_key}")
    xray_name = f"frame_{frame_idx}_z000.png"
    return view_dir, xray_name


def save_drr_png(drr: np.ndarray, out_path: Path) -> None:
    """Save float DRR [0,1] as uint8 PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    drr_u8 = np.clip(drr, 0.0, 1.0)
    drr_u8 = (drr_u8 * 255.0).astype(np.uint8)
    ok = cv2.imwrite(str(out_path), drr_u8)
    if not ok:
        raise RuntimeError(f"Failed to save DRR: {out_path}")


def main() -> None:
    args = parse_args()

    loader = SwaroLoader()
    spec = loader.load(frames=args.frames, verbose=True)

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    print("  Building DiffDRR subject from CT ...")
    subject = build_subject(spec)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.poses_json.parent.mkdir(parents=True, exist_ok=True)

    pixel_spacing_mm = SWARO_PIX_MM * (SWARO_IMG_SIZE / float(args.drr_size))

    print(f"  Building DRR module ({args.drr_size}×{args.drr_size}, "
          f"{pixel_spacing_mm:.4f} mm/px) ...")
    drr_mod = build_drr_module(subject, args.drr_size, pixel_spacing_mm, device)

    frame_records: Dict[str, Dict] = {}
    for proj in spec.projections:
        view_dir, xray_name = proj_key_to_xray_info(proj.proj_key)
        drr_rel = Path(view_dir) / xray_name
        drr_path = args.output_dir / drr_rel

        drr = render_drr(drr_mod, proj.R_proj, proj.t_proj, device)
        save_drr_png(drr, drr_path)
        torch.cuda.empty_cache()

        common_landmarks = [
            name
            for name in sorted(proj.gt_landmarks_2d.keys())
            if name in spec.landmarks_3d
        ]

        if len(common_landmarks) >= 3 and np.isfinite(proj.reproj_error_px) and proj.reproj_error_px > 0.0:
            init_method = "epnp"
        else:
            init_method = "anatomy_fallback_or_unknown"

        key = str(drr_rel)
        frame_records[key] = {
            "proj_key": proj.proj_key,
            "xray_name": xray_name,
            "xray_relative_path": str(Path(view_dir) / xray_name),
            "drr_relative_path": str(drr_rel),
            "init_method": init_method,
            "num_common_landmarks": len(common_landmarks),
            "reproj_error_px": float(proj.reproj_error_px),
            "R_proj": proj.R_proj.tolist(),
            "t_proj": proj.t_proj.tolist(),
        }

        print(f"Saved DRR: {drr_path}")

    payload = {
        "dataset": "swaroopa",
        "generator": {
            "renderer": "diffdrr_siddon",
            "bone_attenuation_multiplier": 4.0,
            "drr_size": int(args.drr_size),
            "pixel_spacing_mm": float(pixel_spacing_mm),
            "sid_mm": float(SWARO_SID_MM),
            "intrinsics": {
                "fx": float(SWARO_FX),
                "fy": float(SWARO_FY),
                "cx": float(SWARO_CX),
                "cy": float(SWARO_CY),
            },
        },
        "output_dir": str(args.output_dir),
        "num_frames": len(frame_records),
        "frames": frame_records,
    }

    with open(args.poses_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nSaved poses JSON: {args.poses_json}")
    print(f"Saved DRRs root: {args.output_dir}")
    print(f"Total frames: {len(frame_records)}")


if __name__ == "__main__":
    main()
