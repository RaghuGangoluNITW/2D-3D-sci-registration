"""
arjun_loader.py — Real Patient Loader for Arjun Dataset
=========================================================
Loads Arjun's pre-op CT (NRRD) + intra-op C-arm JPEG images and wraps
them into the DeepFluoroSpecimen / DeepFluoroProjection interface.

KEY DIFFERENCE vs Ramulamma:
  Arjun has 2D landmark annotations (labelme JSON) alongside each X-ray.
  This allows EPnP initialization → much more accurate initial pose than
  the anatomy-centroid guess used for Ramulamma.

Data layout:
  CT (pre-op NRRD):
    data/testing/ARJUN PREOP/3 L_Spine  1.0  B60s_3.nrrd
      512×512×406, spacing 0.3613×0.3613×0.70 mm

  3D landmarks (vertebral centroids, Slicer mrk.json):
    data/testing/ARJUN PREOP/centroids.mrk.json
      Labels: L1, L2, L3, L4, L5, D12, D11 (RAS mm)

  Intra-op X-rays + 2D annotations (labelme format):
    data/testing/ARJUN INTRAOP AP/a.jpg  (+ a.json)
    data/testing/ARJUN INTRAOP AP/b.jpg  (+ b.json)
    ...e.jpg  (+ e.json)
    Each JSON has pixel-coord landmarks {L2..L5} or {L1..L5}
    Image dimensions vary per frame (745–1166 pixels wide/tall).

Camera model:
  Since these are JPEG captures (not DICOM), exact SID/pixel spacing is
  unknown. We assume Ziehm Vision FD geometry (same as Ramulamma):
    SID = 1110 mm, pixel spacing = 0.2 mm per pixel at 1024 px
  We then rescale intrinsics to match each JPEG's actual image dimensions
  using a "virtual" pixel spacing so the FOV stays consistent.

  For an image of size W×H, we compute:
    cx = (W-1)/2
    cy = (H-1)/2
    scale = min(W, H) / 1024.0    # relative to Ramulamma 1024-px reference
    fx = fy = RAMU_FX * scale      # scale focal length with image size

  This is equivalent to assuming the same angular FOV as Ramulamma.

EPnP initialization:
  For frames with ≥4 annotated landmarks:
    cv2.solvePnP with SOLVEPNP_EPNP
    Uses per-frame intrinsic K (since image sizes vary)
  Fall back to anatomy centroid pose if too few landmarks.

PDE evaluation:
  Since 2D annotations are available, PDE (mm) can be computed
  directly as reprojection error × pixel_spacing_mm.
"""

import os
import sys
import json
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np
import SimpleITK as sitk
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from deepfluoro_loader import (
    DeepFluoroSpecimen,
    DeepFluoroProjection,
    xzy,
    perturb_extrinsic,
)

# ---------------------------------------------------------------------------
# Arjun-specific camera constants
# ---------------------------------------------------------------------------

# Reference: Ziehm Vision FD (same device as Ramulamma)
ARJUN_SID_MM: float  = 1110.0
ARJUN_REF_PIX_MM: float = 0.2      # pixel spacing at reference size 1024 px
ARJUN_REF_SIZE: int  = 1024
ARJUN_REF_FX: float  = ARJUN_SID_MM / ARJUN_REF_PIX_MM   # 5550 px


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE = Path('/home/supermicro/Documents/2D_3D_Raghu/data/testing')

CT_NRRD = _BASE / 'ARJUN PREOP' / '3 L_Spine  1.0  B60s_3.nrrd'
LM_3D_JSON = _BASE / 'ARJUN PREOP' / 'centroids.mrk.json'
XRAY_DIR = _BASE / 'ARJUN INTRAOP AP'


# ---------------------------------------------------------------------------
# Per-frame intrinsics (depends on image size)
# ---------------------------------------------------------------------------

def _make_K(img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    Return (fx, fy, cx, cy) for an image of size img_w × img_h.
    Scale focal length proportionally to min(W, H) / 1024.
    """
    scale = min(img_w, img_h) / ARJUN_REF_SIZE
    fx = ARJUN_REF_FX * scale
    fy = ARJUN_REF_FX * scale
    cx = (img_w - 1) / 2.0
    cy = (img_h - 1) / 2.0
    return fx, fy, cx, cy


def _make_K_mat(img_w: int, img_h: int) -> np.ndarray:
    fx, fy, cx, cy = _make_K(img_w, img_h)
    return np.array([
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.],
    ], dtype=np.float64)


def pixel_spacing_mm(img_w: int, img_h: int) -> float:
    """Effective pixel spacing for this image size (mm/px)."""
    scale = min(img_w, img_h) / ARJUN_REF_SIZE
    return ARJUN_REF_PIX_MM / scale   # larger image → smaller pixels


# ---------------------------------------------------------------------------
# ArjunProjection
# ---------------------------------------------------------------------------

class ArjunProjection(DeepFluoroProjection):
    """
    Arjun X-ray projection.  Stores per-frame intrinsics so that
    project() uses the correct K for each image size.
    """

    def __init__(self, img_w: int, img_h: int, **kwargs):
        super().__init__(**kwargs)
        self.img_w = img_w
        self.img_h = img_h
        self._fx, self._fy, self._cx, self._cy = _make_K(img_w, img_h)

    def project(self, pts3d_world: np.ndarray) -> np.ndarray:
        """Project world XYZ → pixel (u, v) using per-frame intrinsics."""
        pts = np.atleast_2d(pts3d_world)
        P_cam = (self.R_proj @ xzy(pts).T).T + self.t_proj
        u = self._fx * P_cam[:, 0] / P_cam[:, 2] + self._cx
        v = self._fy * P_cam[:, 1] / P_cam[:, 2] + self._cy
        return np.stack([u, v], axis=1)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def compute_pde_arjun(proj: ArjunProjection,
                      R_cand: np.ndarray,
                      t_cand: np.ndarray,
                      pts3d: np.ndarray,
                      lm_names: List[str]) -> Dict[str, float]:
    """PDE (mm) between candidate pose projection and GT 2D annotations."""
    tmp = ArjunProjection.__new__(ArjunProjection)
    tmp.img_w = proj.img_w
    tmp.img_h = proj.img_h
    tmp._fx, tmp._fy, tmp._cx, tmp._cy = _make_K(proj.img_w, proj.img_h)
    tmp.R_proj = R_cand
    tmp.t_proj = t_cand
    uv_pred = tmp.project(pts3d)

    pix_mm = pixel_spacing_mm(proj.img_w, proj.img_h)
    pde: Dict[str, float] = {}
    for i, name in enumerate(lm_names):
        if name in proj.gt_landmarks_2d:
            gt = proj.gt_landmarks_2d[name]
            err_px = float(np.linalg.norm(uv_pred[i] - gt))
            pde[name] = err_px * pix_mm
    return pde


def mean_pde_arjun(proj, R, t, pts3d, lm_names) -> float:
    d = compute_pde_arjun(proj, R, t, pts3d, lm_names)
    return float(np.mean(list(d.values()))) if d else float('inf')


# ---------------------------------------------------------------------------
# Helpers — file I/O
# ---------------------------------------------------------------------------

def _load_mrk_json_3d(json_path: Path) -> Dict[str, np.ndarray]:
    """Load Slicer mrk.json → {label: (3,) RAS mm}.
    Same format as Ramulamma — raw RAS used as-is.
    """
    with open(json_path) as f:
        d = json.load(f)
    cps = d['markups'][0]['controlPoints']
    return {cp['label'].strip(): np.array(cp['position'], dtype=np.float64) for cp in cps}


def _load_labelme_json(json_path: Path) -> Dict[str, np.ndarray]:
    """Load labelme-format JSON → {label: (2,) [x, y] pixel coords}."""
    with open(json_path) as f:
        d = json.load(f)
    result = {}
    for shape in d.get('shapes', []):
        label = shape['label'].strip()
        pts = shape['points']
        if pts:
            result[label] = np.array(pts[0], dtype=np.float64)  # [x, y]
    return result


def _load_jpeg(jpeg_path: Path) -> Tuple[np.ndarray, int, int]:
    """Load JPEG → (float32 [0,1], width, height).

    Arjun X-rays are standard radiographs: bone is BRIGHT (high pixel value).
    DRRs from DeepFluoroDRR are also bone-bright.
    The NCC objective uses NCC(1-drr, tgt) — the (1-drr) flips the DRR polarity
    so both inputs share the same contrast orientation (bone-bright).
    We therefore do NOT invert the JPEG here.
    """
    img = cv2.imread(str(jpeg_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {jpeg_path}")
    h, w = img.shape
    arr = img.astype(np.float32) / 255.0
    # Do NOT invert: keep bone-bright as in the original JPEG
    return arr, w, h


# ---------------------------------------------------------------------------
# EPnP solver with per-frame intrinsics
# ---------------------------------------------------------------------------

def _solve_pnp_arjun(pts3d_world: np.ndarray,
                     pts2d: np.ndarray,
                     img_w: int,
                     img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """EPnP with Arjun per-frame camera intrinsics."""
    K = _make_K_mat(img_w, img_h)
    pts_xzy = xzy(pts3d_world).astype(np.float64)
    pts2d_f = pts2d.astype(np.float64)
    _, rvec, tvec = cv2.solvePnP(
        pts_xzy, pts2d_f, K, np.zeros(4),
        flags=cv2.SOLVEPNP_EPNP,
    )
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.flatten().astype(np.float64)


def _reproj_error_arjun(pts3d: np.ndarray,
                        pts2d: np.ndarray,
                        R: np.ndarray, t: np.ndarray,
                        img_w: int, img_h: int) -> float:
    """Mean reprojection error in pixels."""
    K = _make_K_mat(img_w, img_h)
    pts_xzy = xzy(pts3d).astype(np.float64)
    P_cam = (R @ pts_xzy.T).T + t
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u_pred = fx * P_cam[:, 0] / P_cam[:, 2] + cx
    v_pred = fy * P_cam[:, 1] / P_cam[:, 2] + cy
    uv_pred = np.stack([u_pred, v_pred], axis=1)
    return float(np.sqrt(((uv_pred - pts2d) ** 2).sum(axis=1)).mean())


# ---------------------------------------------------------------------------
# Anatomy pose fallback (same logic as Ramulamma)
# ---------------------------------------------------------------------------

def _anatomy_pose(lm_3d: Dict[str, np.ndarray],
                  sid_mm: float = ARJUN_SID_MM,
                  azimuth_deg: float = 0.0,
                  elevation_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Build an anatomy-centred (R, t) pose as EPnP fallback."""
    centroid = np.mean(np.array(list(lm_3d.values()), dtype=np.float64), axis=0)
    az  = np.deg2rad(azimuth_deg)
    el  = np.deg2rad(elevation_deg)
    src_dir = np.array([
        np.sin(az) * np.cos(el),
        np.cos(az) * np.cos(el),
        np.sin(el),
    ], dtype=np.float64)
    src_world = centroid + sid_mm * src_dir

    src_xzy  = src_world[[0, 2, 1]]
    cent_xzy = centroid[[0, 2, 1]]
    z_cam = cent_xzy - src_xzy
    z_cam /= np.linalg.norm(z_cam)
    world_up = np.array([0., 0., -1.])
    if abs(z_cam.dot(world_up)) > 0.9:
        world_up = np.array([1., 0., 0.])
    x_cam = np.cross(world_up, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)

    R = np.stack([x_cam, y_cam, z_cam], axis=0)
    t = -R @ src_xzy
    return R.astype(np.float64), t.astype(np.float64)


# ---------------------------------------------------------------------------
# ArjunLoader
# ---------------------------------------------------------------------------

class ArjunLoader:
    """
    Load Arjun's pre-op CT + intra-op JPEG X-rays with 2D annotations.

    Returns a DeepFluoroSpecimen with ArjunProjection objects.

    Usage::
        loader = ArjunLoader()
        spec   = loader.load()           # all 5 frames
        spec   = loader.load(frames=['a', 'c'])   # specific frames
    """

    # Frame names in order
    FRAME_NAMES = ['a', 'b', 'c', 'd', 'e']

    def __init__(self,
                 ct_nrrd: Path = CT_NRRD,
                 lm_3d_json: Path = LM_3D_JSON,
                 xray_dir: Path = XRAY_DIR):
        self.ct_nrrd    = Path(ct_nrrd)
        self.lm_3d_json = Path(lm_3d_json)
        self.xray_dir   = Path(xray_dir)

    def _load_ct(self, verbose: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if verbose:
            print(f"  Loading CT: {self.ct_nrrd.name}")
        img = sitk.ReadImage(str(self.ct_nrrd))
        arr = sitk.GetArrayFromImage(img)   # (Z, Y, X) int16 — keep this order for DRR generator
        ct_vol = arr.astype(np.float32)
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        origin  = np.array(img.GetOrigin(),  dtype=np.float64)
        if verbose:
            Z, Y, X = ct_vol.shape
            print(f"    Shape (Z,Y,X): ({Z},{Y},{X})  "
                  f"spacing={spacing.round(3)} mm  "
                  f"HU range=[{ct_vol.min():.0f}, {ct_vol.max():.0f}]")
        return ct_vol, spacing, origin

    def _load_3d_landmarks(self, verbose: bool) -> Dict[str, np.ndarray]:
        lm = _load_mrk_json_3d(self.lm_3d_json)
        if verbose:
            print(f"  3D landmarks ({len(lm)}): {sorted(lm.keys())}")
        return lm

    def load(self,
             frames: Optional[List[str]] = None,
             verbose: bool = True) -> DeepFluoroSpecimen:
        """
        Load Arjun dataset as a DeepFluoroSpecimen.

        Args:
            frames  : list of frame names, e.g. ['a', 'b', 'c'].
                      None = all 5 frames.
            verbose : print progress.

        Returns:
            DeepFluoroSpecimen with ArjunProjection objects.
        """
        if verbose:
            print("[Arjun] Loading specimen ...")

        ct_vol, spacing, origin = self._load_ct(verbose)
        lm_3d = self._load_3d_landmarks(verbose)

        selected = frames if frames is not None else self.FRAME_NAMES

        projections: List[ArjunProjection] = []
        for idx, fname in enumerate(selected):
            jpeg_path = self.xray_dir / f"{fname}.jpg"
            json_path = self.xray_dir / f"{fname}.json"

            if not jpeg_path.exists():
                if verbose:
                    print(f"  [SKIP] {fname}: JPEG not found")
                continue

            # Load X-ray image
            image_raw, img_w, img_h = _load_jpeg(jpeg_path)

            # Load 2D annotations
            lm_2d: Dict[str, np.ndarray] = {}
            if json_path.exists():
                lm_2d = _load_labelme_json(json_path)
                if verbose:
                    print(f"  frame {fname}: {img_w}×{img_h}  "
                          f"2D annotations: {sorted(lm_2d.keys())}")

            # Build 3D↔2D correspondences for EPnP
            common_labels = [l for l in sorted(lm_2d.keys()) if l in lm_3d]
            R_init, t_init = None, None
            reproj_err = 0.0

            if len(common_labels) >= 4:
                pts3d = np.array([lm_3d[l] for l in common_labels])
                pts2d = np.array([lm_2d[l] for l in common_labels])
                try:
                    R_init, t_init = _solve_pnp_arjun(pts3d, pts2d, img_w, img_h)
                    reproj_err = _reproj_error_arjun(pts3d, pts2d, R_init, t_init, img_w, img_h)
                    if verbose:
                        pix_mm = pixel_spacing_mm(img_w, img_h)
                        pde_mm = reproj_err * pix_mm
                        print(f"    EPnP reproj: {reproj_err:.2f} px  ({pde_mm:.2f} mm)  "
                              f"K: fx={_make_K(img_w,img_h)[0]:.0f}px  pix={pix_mm:.4f}mm")
                except Exception as e:
                    if verbose:
                        print(f"    EPnP failed: {e} — using anatomy pose")
                    R_init, t_init = None, None

            if R_init is None:
                R_init, t_init = _anatomy_pose(lm_3d, azimuth_deg=0.0)
                reproj_err = 0.0
                if verbose:
                    print(f"    Using anatomy pose fallback")

            proj = ArjunProjection(
                img_w=img_w,
                img_h=img_h,
                specimen_id     = 'arjun',
                proj_index      = idx,
                proj_key        = fname,
                image_raw       = image_raw,
                image_display   = image_raw,
                R_proj          = R_init,
                t_proj          = t_init,
                gt_landmarks_2d = lm_2d,
                rot_180_for_up  = False,
                reproj_error_px = reproj_err,
            )
            projections.append(proj)

        spec = DeepFluoroSpecimen(
            specimen_id  = 'arjun',
            ct_volume    = ct_vol,
            ct_spacing   = spacing,
            ct_origin    = origin,
            landmarks_3d = lm_3d,
            projections  = projections,
        )

        if verbose:
            print(f"\n  Loaded: {spec}")

        return spec
