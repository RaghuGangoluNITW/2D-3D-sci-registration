"""
ramulamma_testing_loader.py — Loader for Ramulamma Testing Dataset
===================================================================
Loads Ramulamma's pre-op CT (NRRD) + intra-op C-arm NRRD images
from the data/testing folder, using the newly uploaded 2D landmark
annotations (CENTROIDS.mrk.json) to enable EPnP initialisation —
exactly the same pipeline used for Arjun.

Data layout:
  CT (pre-op NRRD):
    data/testing/RAMULAMMA PREOP/RAMULAMMA PREOP/4 L_Spine  1.0  B60s.nrrd
      512×512×395, spacing 0.668×0.668×0.700 mm

  3D landmarks (vertebral centroids, Slicer mrk.json):
    data/testing/RAMULAMMA PREOP/RAMULAMMA PREOP/centroids.mrk.json
      Labels: L1, L2, L3, L4, L5  (LPS mm)

  Intra-op X-rays (NRRD, from Ziehm C-arm):
    data/testing/RAMULAMMA INTRAOP/Data/
      00000058.TIF.nrrd, 00000058.TIF_1.nrrd,
      00000074.TIF.nrrd,
      00000089.TIF.nrrd, 00000089.TIF_1.nrrd,
      00000103.TIF.nrrd, 00000105.TIF.nrrd, 00000107.TIF.nrrd
    All 1024×1024×1, pixel spacing = 0.29765624 mm, uint16.

    2D landmarks (vertebral centroids annotated in Slicer on 1.nrrd stack):
    data/testing/RAMULAMMA INTRAOP/Data/CENTROIDS.mrk.json
      Labels: L2, L3, L4, L5  (pixel coords col, row — spacing=1 NRRD)
    data/testing/RAMULAMMA INTRAOP/Data/CENTROIDS-2.mrk.json
      Labels: L3, L4, L5  (pixel coords col, row)

Coordinate notes:
  The annotations were placed in Slicer on the 1.nrrd stack (spacing=1, origin=0,
  ijkToRAS = -1 0 0 / 0 -1 0 / 0 0 1).  The LPS position (x, y, z) in the JSON
  directly gives the pixel column (x) and row (y) in the 1024×1024 image, since
  LPS_x = voxel_col × 1mm/px.  These pixel coords apply equally to the individual
  TIF NRRDs (same 1024×1024 detector, same patient position).

Camera model:
  Ziehm Vision FD — identical to the main Ramulamma dataset:
    SID = 1110 mm
    Pixel spacing = 0.29765624 mm/px  (from NRRD header)
    Fx = Fy = SID / pixel_spacing = 1110 / 0.29765624 ≈ 3729.9 px
    Cx = Cy = (1024 - 1) / 2 = 511.5

  NOTE: the testing NRRDs use 0.29765624 mm/px (vs 0.2 mm/px in the original
  DICOM loader).  We use the NRRD header spacing directly.

EPnP initialisation:
    Same as arjun_loader.py — cv2.solvePnP(SOLVEPNP_EPNP) using the
    matched 3D (from preop CT centroids.mrk.json) ↔ 2D (pixel coords) pairs.

IMPORTANT:
    The uploaded annotations do NOT cover all exported NRRD frames. `CENTROIDS.mrk.json`
    corresponds to the AP frame exported as `00000074.TIF.nrrd` (4 landmarks, valid for EPnP).
    `CENTROIDS-2.mrk.json` contains only 3 landmarks and is therefore not sufficient for a
    defensible EPnP validation run. We therefore validate only the annotated AP frame by default.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import nrrd
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))

from deepfluoro_loader import (
    DeepFluoroProjection,
    DeepFluoroSpecimen,
    perturb_extrinsic,
    xzy,
)

# ---------------------------------------------------------------------------
# Camera constants
# ---------------------------------------------------------------------------

RAMU_TEST_SID_MM: float = 1110.0
RAMU_TEST_PIX_MM: float = 0.29765624   # from NRRD header
RAMU_TEST_IMG_SIZE: int = 1024

RAMU_TEST_FX: float = RAMU_TEST_SID_MM / RAMU_TEST_PIX_MM   # ≈ 3729.9 px
RAMU_TEST_FY: float = RAMU_TEST_FX
RAMU_TEST_CX: float = (RAMU_TEST_IMG_SIZE - 1) / 2.0        # 511.5
RAMU_TEST_CY: float = RAMU_TEST_CX

RAMU_TEST_K: np.ndarray = np.array([
    [RAMU_TEST_FX, 0.,           RAMU_TEST_CX],
    [0.,           RAMU_TEST_FY, RAMU_TEST_CY],
    [0.,           0.,           1.          ],
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE = Path('/home/supermicro/Documents/2D_3D_Raghu/data/testing')

CT_NRRD = _BASE / 'RAMULAMMA PREOP' / 'RAMULAMMA PREOP' / '4 L_Spine  1.0  B60s.nrrd'
LM_3D_JSON = _BASE / 'RAMULAMMA PREOP' / 'RAMULAMMA PREOP' / 'centroids.mrk.json'
XRAY_DIR = _BASE / 'RAMULAMMA INTRAOP' / 'Data'

# Primary annotation file (4 landmarks L2-L5)
LM_2D_JSON_PRIMARY = XRAY_DIR / 'CENTROIDS.mrk.json'
# Secondary annotation file (3 landmarks L3-L5, different C-arm pose)
LM_2D_JSON_SECONDARY = XRAY_DIR / 'CENTROIDS-2.mrk.json'


# ---------------------------------------------------------------------------
# RamuTestProjection — fixed 1024×1024 Ziehm geometry
# ---------------------------------------------------------------------------

class RamuTestProjection(DeepFluoroProjection):
    """Ramulamma testing X-ray projection — fixed 1024×1024 Ziehm geometry."""

    def project(self, pts3d_world: np.ndarray) -> np.ndarray:
        """Project world XYZ → pixel (u, v)."""
        pts = np.atleast_2d(pts3d_world)
        P_cam = (self.R_proj @ xzy(pts).T).T + self.t_proj
        u = RAMU_TEST_FX * P_cam[:, 0] / P_cam[:, 2] + RAMU_TEST_CX
        v = RAMU_TEST_FY * P_cam[:, 1] / P_cam[:, 2] + RAMU_TEST_CY
        return np.stack([u, v], axis=1)


# ---------------------------------------------------------------------------
# PDE helpers
# ---------------------------------------------------------------------------

def compute_pde_ramu_test(proj: RamuTestProjection,
                          R: np.ndarray,
                          t: np.ndarray,
                          pts3d: np.ndarray,
                          lm_names: List[str]) -> Dict[str, float]:
    """PDE (mm) = reprojection error × pixel_spacing."""
    tmp = RamuTestProjection.__new__(RamuTestProjection)
    tmp.R_proj = R
    tmp.t_proj = t
    uv_pred = tmp.project(pts3d)
    pde: Dict[str, float] = {}
    for i, name in enumerate(lm_names):
        if name in proj.gt_landmarks_2d:
            gt = proj.gt_landmarks_2d[name]
            err_px = float(np.linalg.norm(uv_pred[i] - gt))
            pde[name] = err_px * RAMU_TEST_PIX_MM
    return pde


def mean_pde_ramu_test(proj, R, t, pts3d, lm_names) -> float:
    d = compute_pde_ramu_test(proj, R, t, pts3d, lm_names)
    return float(np.mean(list(d.values()))) if d else float('inf')


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _load_mrk_json_3d(path: Path) -> Dict[str, np.ndarray]:
    """Load Slicer mrk.json → {label: (3,) LPS mm} for 3D CT landmarks."""
    with open(path) as f:
        d = json.load(f)
    cps = d['markups'][0]['controlPoints']
    return {cp['label'].strip(): np.array(cp['position'], dtype=np.float64)
            for cp in cps}


def _load_mrk_json_2d(path: Path) -> Dict[str, np.ndarray]:
    """
    Load Slicer mrk.json → {label: (2,) [col, row] pixel coords}.

    The annotations were placed on 1.nrrd (spacing=1mm, origin=0, ijkToRAS=-1 0 0 / 0 -1 0 / 0 0 1).
    In that volume, LPS (x, y, z) = (col, row, frame_index).
    We extract (col, row) = LPS (x, y) directly as pixel coordinates
    applicable to any 1024×1024 frame from the same acquisition.
    """
    with open(path) as f:
        d = json.load(f)
    cps = d['markups'][0]['controlPoints']
    result: Dict[str, np.ndarray] = {}
    for cp in cps:
        label = cp['label'].strip()
        pos = cp['position']        # [LPS_x, LPS_y, LPS_z]
        col = pos[0]                # pixel column
        row = pos[1]                # pixel row
        result[label] = np.array([col, row], dtype=np.float64)
    return result


def _load_nrrd_xray(path: Path) -> np.ndarray:
    """
    Load 1024×1024×1 NRRD C-arm image → float32 [0, 1].

    Ramulamma X-rays are stored as uint16 with bone-bright convention
    (high pixel value = bone / high attenuation).  DRRs from
    DeepFluoroDRR are bone-bright.  We normalise to [0, 1] without inversion.
    """
    data, _ = nrrd.read(str(path))
    if data.ndim == 3:
        arr = data[:, :, 0]          # (W, H) after removing slice dim
    else:
        arr = data
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    # NRRD axis 0 = x (col), axis 1 = y (row).
    # image_raw is expected as (H, W) by the optimizer/DRR generator.
    return arr.T                     # → (H=1024, W=1024)


# ---------------------------------------------------------------------------
# EPnP solver
# ---------------------------------------------------------------------------

def _solve_pnp(pts3d_world: np.ndarray,
               pts2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """EPnP using the fixed Ziehm camera matrix."""
    pts_xzy = xzy(pts3d_world).astype(np.float64)
    _, rvec, tvec = cv2.solvePnP(
        pts_xzy, pts2d.astype(np.float64), RAMU_TEST_K, np.zeros(4),
        flags=cv2.SOLVEPNP_EPNP,
    )
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.flatten().astype(np.float64)


def _reproj_error_px(pts3d: np.ndarray,
                     pts2d: np.ndarray,
                     R: np.ndarray,
                     t: np.ndarray) -> float:
    """Mean reprojection error in pixels."""
    pts_xzy = xzy(pts3d).astype(np.float64)
    P_cam = (R @ pts_xzy.T).T + t
    u = RAMU_TEST_FX * P_cam[:, 0] / P_cam[:, 2] + RAMU_TEST_CX
    v = RAMU_TEST_FY * P_cam[:, 1] / P_cam[:, 2] + RAMU_TEST_CY
    uv_pred = np.stack([u, v], axis=1)
    return float(np.sqrt(((uv_pred - pts2d) ** 2).sum(axis=1)).mean())


# ---------------------------------------------------------------------------
# Anatomy fallback pose (same as arjun_loader.py)
# ---------------------------------------------------------------------------

def _anatomy_pose(lm_3d: Dict[str, np.ndarray],
                  sid_mm: float = RAMU_TEST_SID_MM,
                  azimuth_deg: float = 0.0,
                  elevation_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(np.array(list(lm_3d.values()), dtype=np.float64), axis=0)
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    src_dir = np.array([np.sin(az) * np.cos(el),
                        np.cos(az) * np.cos(el),
                        np.sin(el)], dtype=np.float64)
    src_world = centroid + sid_mm * src_dir
    src_xzy   = src_world[[0, 2, 1]]
    cent_xzy  = centroid[[0, 2, 1]]
    z_cam = cent_xzy - src_xzy;  z_cam /= np.linalg.norm(z_cam)
    up = np.array([0., 0., -1.])
    if abs(z_cam.dot(up)) > 0.9:
        up = np.array([1., 0., 0.])
    x_cam = np.cross(up, z_cam);   x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    R = np.stack([x_cam, y_cam, z_cam], axis=0)
    t = -R @ src_xzy
    return R.astype(np.float64), t.astype(np.float64)


# ---------------------------------------------------------------------------
# RamuTestLoader
# ---------------------------------------------------------------------------

# Frame name → NRRD filename mapping
FRAME_FILES = {
    '058':   '00000058.TIF.nrrd',
    '058_1': '00000058.TIF_1.nrrd',
    '074':   '00000074.TIF.nrrd',
    '089':   '00000089.TIF.nrrd',
    '089_1': '00000089.TIF_1.nrrd',
    '103':   '00000103.TIF.nrrd',
    '105':   '00000105.TIF.nrrd',
    '107':   '00000107.TIF.nrrd',
}

# Only `074` has 4-point AP annotations suitable for EPnP validation.
FRAME_TO_ANNOTATION = {
    '074': 'primary',
}


class RamuTestLoader:
    """
    Load Ramulamma testing data (NRRD X-rays + 2D annotations + pre-op CT).

    Each intraop NRRD frame is given the same 2D landmark annotations
    (CENTROIDS.mrk.json, L2-L5) since they all come from the same C-arm
    acquisition with the patient in a fixed position.

    Usage::
        loader = RamuTestLoader()
        spec   = loader.load()            # all 8 frames
        spec   = loader.load(frames=['074', '058'])
    """

    def __init__(self,
                 ct_nrrd: Path = CT_NRRD,
                 lm_3d_json: Path = LM_3D_JSON,
                 lm_2d_primary: Path = LM_2D_JSON_PRIMARY,
                 lm_2d_secondary: Path = LM_2D_JSON_SECONDARY,
                 xray_dir: Path = XRAY_DIR):
        self.ct_nrrd         = Path(ct_nrrd)
        self.lm_3d_json      = Path(lm_3d_json)
        self.lm_2d_primary   = Path(lm_2d_primary)
        self.lm_2d_secondary = Path(lm_2d_secondary)
        self.xray_dir        = Path(xray_dir)

    def _load_ct(self, verbose: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if verbose:
            print(f'  Loading CT: {self.ct_nrrd.name}')
        img = sitk.ReadImage(str(self.ct_nrrd))
        arr = sitk.GetArrayFromImage(img)   # (Z, Y, X) int16 — keep this order for DRR generator
        ct_vol = arr.astype(np.float32)
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        origin  = np.array(img.GetOrigin(),  dtype=np.float64)
        if verbose:
            Z, Y, X = ct_vol.shape
            print(f'    Shape (Z,Y,X): ({Z},{Y},{X})  spacing={spacing.round(3)} mm  '
                  f'HU range=[{ct_vol.min():.0f}, {ct_vol.max():.0f}]')
        return ct_vol, spacing, origin

    def load(self,
             frames: Optional[List[str]] = None,
             verbose: bool = True) -> DeepFluoroSpecimen:
        """
        Load Ramulamma testing dataset as a DeepFluoroSpecimen.

        Args:
            frames  : list of frame keys to load (e.g. ['074', '089']).
                      None = only defensibly annotated frames.
            verbose : print progress.

        Returns:
            DeepFluoroSpecimen with RamuTestProjection objects.
        """
        if verbose:
            print('[RamuTest] Loading Ramulamma testing specimen ...')

        ct_vol, spacing, origin = self._load_ct(verbose)

        # 3D landmarks from pre-op CT (L1-L5 in LPS mm)
        lm_3d = _load_mrk_json_3d(self.lm_3d_json)
        if verbose:
            print(f'  3D landmarks ({len(lm_3d)}): {sorted(lm_3d.keys())}')

        # 2D landmark annotations (pixel col, row)
        lm_2d_primary: Dict[str, np.ndarray] = {}
        if self.lm_2d_primary.exists():
            lm_2d_primary.update(_load_mrk_json_2d(self.lm_2d_primary))
        if verbose:
            print(f'  2D primary annotations ({len(lm_2d_primary)}): {sorted(lm_2d_primary.keys())}')

        selected = frames if frames is not None else list(FRAME_TO_ANNOTATION.keys())

        projections: List[RamuTestProjection] = []
        for idx, frame_key in enumerate(selected):
            if frame_key not in FRAME_FILES:
                if verbose:
                    print(f'  [SKIP] {frame_key}: unknown frame key')
                continue

            nrrd_path = self.xray_dir / FRAME_FILES[frame_key]
            if not nrrd_path.exists():
                if verbose:
                    print(f'  [SKIP] {frame_key}: file not found ({nrrd_path.name})')
                continue

            image_raw = _load_nrrd_xray(nrrd_path)   # (H, W) float32
            H, W = image_raw.shape

            annotation_key = FRAME_TO_ANNOTATION.get(frame_key)
            lm_2d_all = lm_2d_primary if annotation_key == 'primary' else {}

            if verbose:
                ann_desc = sorted(lm_2d_all.keys()) if lm_2d_all else 'none'
                print(f'  frame {frame_key}: {W}×{H}  2D annotations: {ann_desc}')

            # Build 3D ↔ 2D correspondences
            common = [l for l in sorted(lm_2d_all.keys()) if l in lm_3d]
            R_init, t_init = None, None
            reproj_err = 0.0

            if len(common) >= 4:
                pts3d = np.array([lm_3d[l]      for l in common])
                pts2d = np.array([lm_2d_all[l]  for l in common])
                try:
                    R_init, t_init = _solve_pnp(pts3d, pts2d)
                    reproj_err = _reproj_error_px(pts3d, pts2d, R_init, t_init)
                    if verbose:
                        pde_mm = reproj_err * RAMU_TEST_PIX_MM
                        print(f'    EPnP reproj: {reproj_err:.2f} px  ({pde_mm:.2f} mm)  '
                              f'Fx={RAMU_TEST_FX:.0f} px  pix={RAMU_TEST_PIX_MM:.6f} mm')
                except Exception as e:
                    if verbose:
                        print(f'    EPnP failed: {e} — using anatomy pose')
                    R_init, t_init = None, None

            if R_init is None:
                R_init, t_init = _anatomy_pose(lm_3d, azimuth_deg=0.0)
                reproj_err = 0.0
                if verbose:
                    if lm_2d_all:
                        print('    Using anatomy pose fallback')
                    else:
                        print('    No valid 2D annotations for this frame — anatomy pose only')

            proj = RamuTestProjection(
                specimen_id      = 'ramulamma_test',
                proj_index       = idx,
                proj_key         = frame_key,
                image_raw        = image_raw,
                image_display    = image_raw,
                R_proj           = R_init,
                t_proj           = t_init,
                gt_landmarks_2d  = lm_2d_all,
                rot_180_for_up   = False,
                reproj_error_px  = reproj_err,
            )
            projections.append(proj)

        spec = DeepFluoroSpecimen(
            specimen_id  = 'ramulamma_test',
            ct_volume    = ct_vol,
            ct_spacing   = spacing,
            ct_origin    = origin,
            landmarks_3d = lm_3d,
            projections  = projections,
        )

        if verbose:
            print(f'\n  Loaded: {spec}')
        return spec
