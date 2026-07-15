"""
new_swaroopa_loader.py — Patient Loader for New Swaroop Dataset (ProcessedData)
================================================================================
Loads new Swaroop pre-op CT (NRRD) + intra-op C-arm NRRD images (LAT only)
from the ProcessedData folder, wrapping them into the DeepFluoroSpecimen /
DeepFluoroProjection interface.

Data layout:
  CT (pre-op NRRD):
    Swaroop/CT label/2 Unnamed Series.nrrd
      512×512×257, spacing 0.660×0.660×1.000 mm

  3D landmarks (Slicer mrk.json, LPS):
    Swaroop/CT label/centroid_3.mrk.json
      Labels: L1, L2, L3, L4, L5

  Intra-op X-rays (NRRD, 1024×1024) + 2D annotations (mrk.json):
    Swaroop/XRAY Label/XRAY-LAT/LAT-SET-1/  → 0 XA - ...nrrd  (L5,L4,L3,L2,L1) ← FULL
    Swaroop/XRAY Label/XRAY-LAT/LAT-SET-2/  → 0 XA - ...nrrd  (L5,L4,L3,L2,L1) ← FULL
    Swaroop/XRAY Label/XRAY-LAT/LAT-SET-3/  → 0 XA - ...nrrd  (L5,L4,L3,L2,L1) ← FULL
    (AP sets are missing images — LAT only)

  2D landmark format: XRCENTROID.mrk.json with position=(u, v, 0).

Camera model (assumed Ziehm Vision FD — same as SARKHI/Arjun, no DICOM metadata):
  SID     = 1110 mm
  Pixel spacing = 0.2 mm/px at 1024 px
  Fx = Fy = 5550 px  Cx = Cy = 511.5
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
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
# Camera constants (Ziehm Vision FD assumed — same as SARKHI)
# ---------------------------------------------------------------------------

NSWARO_SID_MM:   float = 1110.0
NSWARO_PIX_MM:   float = 0.2
NSWARO_IMG_SIZE: int   = 1024

NSWARO_FX: float = NSWARO_SID_MM / NSWARO_PIX_MM    # 5550 px
NSWARO_FY: float = NSWARO_SID_MM / NSWARO_PIX_MM
NSWARO_CX: float = (NSWARO_IMG_SIZE - 1) / 2.0       # 511.5
NSWARO_CY: float = (NSWARO_IMG_SIZE - 1) / 2.0

NSWARO_K: np.ndarray = np.array([
    [NSWARO_FX, 0.,        NSWARO_CX],
    [0.,        NSWARO_FY, NSWARO_CY],
    [0.,        0.,        1.        ],
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Data root
# ---------------------------------------------------------------------------

_PROC_BASE = Path('/home/supermicro/Documents/2D_3D_Raghu/data/ProcessedData/Processed Data 2D-3D/Swaroop')

CT_NRRD    = _PROC_BASE / 'CT label' / '2 Unnamed Series.nrrd'
LM_3D_JSON = _PROC_BASE / 'CT label' / 'centroid_3.mrk.json'
XRAY_BASE  = _PROC_BASE / 'XRAY Label'


# ---------------------------------------------------------------------------
# NewSwaroProjection
# ---------------------------------------------------------------------------

class NewSwaroProjection(DeepFluoroProjection):
    """New Swaroop projection using fixed 1024×1024 Ziehm intrinsics."""

    def project(self, pts3d_world: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts3d_world)
        P_cam = (self.R_proj @ xzy(pts).T).T + self.t_proj
        u = NSWARO_FX * P_cam[:, 0] / P_cam[:, 2] + NSWARO_CX
        v = NSWARO_FY * P_cam[:, 1] / P_cam[:, 2] + NSWARO_CY
        return np.stack([u, v], axis=1)


# ---------------------------------------------------------------------------
# PDE helpers
# ---------------------------------------------------------------------------

def project_world_nswaro(pts3d_world: np.ndarray,
                          R: np.ndarray,
                          t: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts3d_world)
    P_cam = (R @ xzy(pts).T).T + t
    u = NSWARO_FX * P_cam[:, 0] / P_cam[:, 2] + NSWARO_CX
    v = NSWARO_FY * P_cam[:, 1] / P_cam[:, 2] + NSWARO_CY
    return np.stack([u, v], axis=1)


def compute_pde_nswaro(proj: NewSwaroProjection,
                        R_cand: np.ndarray,
                        t_cand: np.ndarray,
                        pts3d: np.ndarray,
                        lm_names: List[str]) -> Dict[str, float]:
    """PDE (mm) between candidate pose and GT 2D annotations."""
    uv_pred = project_world_nswaro(pts3d, R_cand, t_cand)
    pde: Dict[str, float] = {}
    for i, name in enumerate(lm_names):
        if name in proj.gt_landmarks_2d:
            gt = proj.gt_landmarks_2d[name]
            if 0 <= gt[0] < NSWARO_IMG_SIZE and 0 <= gt[1] < NSWARO_IMG_SIZE:
                err_px = float(np.linalg.norm(uv_pred[i] - gt))
                pde[name] = err_px * NSWARO_PIX_MM
    return pde


def mean_pde_nswaro(proj, R, t, pts3d, lm_names) -> float:
    d = compute_pde_nswaro(proj, R, t, pts3d, lm_names)
    return float(np.mean(list(d.values()))) if d else float('inf')


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _load_mrk_json_3d(json_path: Path) -> Dict[str, np.ndarray]:
    """Load Slicer mrk.json → {label: (3,) LPS mm}."""
    with open(json_path) as f:
        d = json.load(f)
    cps = d['markups'][0]['controlPoints']
    return {
        cp['label'].strip(): np.array(cp['position'], dtype=np.float64)
        for cp in cps
    }


def _load_mrk_json_2d(json_path: Path) -> Dict[str, np.ndarray]:
    """
    Load Slicer mrk.json with 2D pixel coords → {label: (2,) [u, v]}.
    position=(u, v, 0) where u,v are pixel coordinates.
    Works for both CENTROID.mrk.json and XRCENTROID.mrk.json.
    """
    with open(json_path) as f:
        d = json.load(f)
    cps = d['markups'][0]['controlPoints']
    result: Dict[str, np.ndarray] = {}
    for cp in cps:
        label = cp['label'].strip()
        pos = cp['position']
        if cp.get('positionStatus', 'defined') == 'defined':
            result[label] = np.array([pos[0], pos[1]], dtype=np.float64)
    return result


def _load_nrrd_xray(nrrd_path: Path) -> np.ndarray:
    """Load NRRD X-ray → float32 [0,1], shape (H, W)."""
    img = sitk.ReadImage(str(nrrd_path))
    arr = sitk.GetArrayFromImage(img).squeeze()
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


# ---------------------------------------------------------------------------
# EPnP and anatomy pose helpers
# ---------------------------------------------------------------------------

def _solve_pnp_nswaro(pts3d_world: np.ndarray,
                       pts2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """EPnP using New Swaroop camera intrinsics. pts3d_world in LPS mm."""
    pts_xzy = xzy(pts3d_world).astype(np.float64)
    n = len(pts_xzy)
    flag = cv2.SOLVEPNP_SQPNP if n == 3 else cv2.SOLVEPNP_EPNP
    _, rvec, tvec = cv2.solvePnP(
        pts_xzy, pts2d.astype(np.float64),
        NSWARO_K, np.zeros(4),
        flags=flag,
    )
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.flatten().astype(np.float64)


def _reproj_error_nswaro(pts3d: np.ndarray, pts2d: np.ndarray,
                          R: np.ndarray, t: np.ndarray) -> float:
    uv_pred = project_world_nswaro(pts3d, R, t)
    return float(np.sqrt(((uv_pred - pts2d) ** 2).sum(axis=1)).mean())


def _anatomy_pose(lm_3d: Dict[str, np.ndarray],
                  azimuth_deg: float = 90.0) -> Tuple[np.ndarray, np.ndarray]:
    """Anatomy-centred pose. For LAT, azimuth=90°."""
    centroid = np.mean(np.array(list(lm_3d.values()), dtype=np.float64), axis=0)
    az  = np.deg2rad(azimuth_deg)
    src_dir = np.array([np.sin(az), np.cos(az), 0.], dtype=np.float64)
    src_world = centroid + NSWARO_SID_MM * src_dir

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
# NewSwaroLoader
# ---------------------------------------------------------------------------

class NewSwaroLoader:
    """
    Load new Swaroop pre-op CT + intra-op NRRD X-rays with 2D mrk.json labels.

    Only LAT sets are available (AP images are missing from this dataset).

    Usage::
        loader = NewSwaroLoader()
        spec   = loader.load()                           # all LAT sets
        spec   = loader.load(sets=['LAT-SET-1'])         # specific sets
    """

    def __init__(self,
                 ct_nrrd:    Path = CT_NRRD,
                 lm_3d_json: Path = LM_3D_JSON,
                 xray_base:  Path = XRAY_BASE):
        self.ct_nrrd    = Path(ct_nrrd)
        self.lm_3d_json = Path(lm_3d_json)
        self.xray_base  = Path(xray_base)

    def _load_ct(self, verbose: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if verbose:
            print(f"  Loading CT: {self.ct_nrrd.name}")
        img = sitk.ReadImage(str(self.ct_nrrd))
        ct_vol  = sitk.GetArrayFromImage(img).astype(np.float32)
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        origin  = np.array(img.GetOrigin(),  dtype=np.float64)
        if verbose:
            Z, Y, X = ct_vol.shape
            print(f"    Shape (Z,Y,X): ({Z},{Y},{X})  "
                  f"spacing={spacing.round(3)} mm  "
                  f"HU=[{ct_vol.min():.0f},{ct_vol.max():.0f}]")
        return ct_vol, spacing, origin

    def _load_3d_landmarks(self, verbose: bool) -> Dict[str, np.ndarray]:
        lm = _load_mrk_json_3d(self.lm_3d_json)
        if verbose:
            print(f"  3D landmarks ({len(lm)}): {sorted(lm.keys())}")
        return lm

    def _discover_sets(self) -> List[Tuple[str, Path, Path, str]]:
        """
        Find all (set_name, nrrd_path, json_path, view_tag) tuples.
        Searches both XRAY-AP and XRAY-LAT but returns only sets with
        both image and centroid JSON present.
        """
        found = []
        for view_tag, subfolder in [('AP', 'XRAY-AP'), ('LAT', 'XRAY-LAT')]:
            view_dir = self.xray_base / subfolder
            if not view_dir.exists():
                continue
            for set_dir in sorted(view_dir.iterdir()):
                nrrd_files = list(set_dir.glob('*.nrrd'))
                json_files = list(set_dir.glob('*.mrk.json'))
                if nrrd_files and json_files:
                    set_name = set_dir.name   # e.g. AP-SET-2, LAT-SET-1
                    found.append((set_name, nrrd_files[0], json_files[0], view_tag))
        return found

    def load(self,
             sets: Optional[List[str]] = None,
             verbose: bool = True) -> DeepFluoroSpecimen:
        """
        Load new Swaroop dataset as a DeepFluoroSpecimen.

        Args:
            sets    : list of set names, e.g. ['LAT-SET-1', 'LAT-SET-2'].
                      None = all sets where image + labels exist.
            verbose : print progress.
        """
        if verbose:
            print("[NewSwaroop] Loading specimen ...")

        ct_vol, spacing, origin = self._load_ct(verbose)
        lm_3d = self._load_3d_landmarks(verbose)

        all_sets = self._discover_sets()
        if verbose:
            print(f"  Sets discovered: {[s[0] for s in all_sets]}")

        if sets is not None:
            all_sets = [s for s in all_sets if s[0] in sets]

        projections: List[NewSwaroProjection] = []
        for idx, (set_name, nrrd_path, json_path, view_tag) in enumerate(all_sets):
            az_fallback = 0.0 if view_tag == 'AP' else 90.0

            image_raw = _load_nrrd_xray(nrrd_path)
            lm_2d = _load_mrk_json_2d(json_path)
            lumbar_2d = {k: v for k, v in lm_2d.items() if k in ['L1','L2','L3','L4','L5']}

            if verbose:
                print(f"  {set_name}: img={image_raw.shape}  "
                      f"2D labels={sorted(lumbar_2d.keys())}")

            # EPnP initialization
            common_labels = [l for l in sorted(lumbar_2d.keys()) if l in lm_3d]
            R_init, t_init = None, None
            reproj_err = 0.0

            if len(common_labels) >= 4:
                pts3d = np.array([lm_3d[l] for l in common_labels])
                pts2d = np.array([lumbar_2d[l] for l in common_labels])
                try:
                    R_init, t_init = _solve_pnp_nswaro(pts3d, pts2d)
                    reproj_err = _reproj_error_nswaro(pts3d, pts2d, R_init, t_init)
                    if verbose:
                        pde_mm = reproj_err * NSWARO_PIX_MM
                        print(f"    EPnP reproj: {reproj_err:.2f}px ({pde_mm:.2f}mm)  "
                              f"labels={common_labels}")
                except Exception as e:
                    if verbose:
                        print(f"    EPnP failed: {e} — using anatomy pose")
                    R_init, t_init = None, None

            if R_init is None:
                R_init, t_init = _anatomy_pose(lm_3d, azimuth_deg=az_fallback)
                reproj_err = 0.0
                if verbose:
                    print(f"    Anatomy pose fallback (az={az_fallback:.0f}°)")

            proj = NewSwaroProjection(
                specimen_id     = 'new_swaroopa',
                proj_index      = idx,
                proj_key        = set_name,
                image_raw       = image_raw,
                image_display   = image_raw,
                R_proj          = R_init,
                t_proj          = t_init,
                gt_landmarks_2d = lumbar_2d,
                rot_180_for_up  = False,
                reproj_error_px = reproj_err,
            )
            projections.append(proj)

        if not projections:
            raise RuntimeError("No projections loaded — check NRRD/JSON paths.")

        spec = DeepFluoroSpecimen(
            specimen_id  = 'new_swaroopa',
            ct_volume    = ct_vol,
            ct_spacing   = spacing,
            ct_origin    = origin,
            landmarks_3d = lm_3d,
            projections  = projections,
        )

        if verbose:
            print(f"\n  Loaded: {spec}")

        return spec
