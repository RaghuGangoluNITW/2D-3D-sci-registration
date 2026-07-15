"""
generic_patient_loader.py — Unified Loader for All New Patient Datasets
=======================================================================
Handles any patient following the canonical folder layout:

    <patient_root>/
        CT label/
            <CT>.nrrd                 — pre-op CT volume
            CENTROID.mrk.json         — 3D vertebral centroids (LPS mm)
        XRAY Label/
            XRAY-AP/
                AP-SET-1/
                    <image>.nrrd      — 1024×1024 C-arm image
                    CENTROID.mrk.json — 2D pixel coords (u,v,0)
                AP-SET-2/ ...
            XRAY-LAT/
                LAT-SET-1/ ...

Works for: MANGTA, NARSIMHA, SAMRAJYAM, SARKHI, Swaroop (and future patients).

Camera model (Ziehm Vision FD assumed — no DICOM metadata):
    SID = 1110 mm,  pixel spacing = 0.2 mm/px at 1024 px,  Fx = Fy = 5550 px

EPnP initialization:
    ≥4 common landmarks → EPNP,  3 → SQPNP,  <3 → anatomy fallback
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
# Camera constants
# ---------------------------------------------------------------------------
CAM_SID_MM:   float = 1110.0
CAM_PIX_MM:   float = 0.2
CAM_IMG_SIZE: int   = 1024
CAM_FX: float = CAM_SID_MM / CAM_PIX_MM   # 5550 px
CAM_FY: float = CAM_SID_MM / CAM_PIX_MM
CAM_CX: float = (CAM_IMG_SIZE - 1) / 2.0
CAM_CY: float = (CAM_IMG_SIZE - 1) / 2.0
CAM_K: np.ndarray = np.array([
    [CAM_FX, 0.,     CAM_CX],
    [0.,     CAM_FY, CAM_CY],
    [0.,     0.,     1.    ],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# GenericProjection
# ---------------------------------------------------------------------------
class GenericProjection(DeepFluoroProjection):
    """Generic C-arm projection with fixed 1024×1024 Ziehm intrinsics."""
    def project(self, pts3d_world: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts3d_world)
        P_cam = (self.R_proj @ xzy(pts).T).T + self.t_proj
        u = CAM_FX * P_cam[:, 0] / P_cam[:, 2] + CAM_CX
        v = CAM_FY * P_cam[:, 1] / P_cam[:, 2] + CAM_CY
        return np.stack([u, v], axis=1)


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------
def project_world(pts3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts3d)
    P   = (R @ xzy(pts).T).T + t
    u   = CAM_FX * P[:, 0] / P[:, 2] + CAM_CX
    v   = CAM_FY * P[:, 1] / P[:, 2] + CAM_CY
    return np.stack([u, v], axis=1)


def compute_pde(proj: GenericProjection,
                R: np.ndarray, t: np.ndarray,
                pts3d: np.ndarray, lm_names: List[str]) -> Dict[str, float]:
    uv = project_world(pts3d, R, t)
    pde: Dict[str, float] = {}
    for i, name in enumerate(lm_names):
        if name in proj.gt_landmarks_2d:
            gt = proj.gt_landmarks_2d[name]
            if 0 <= gt[0] < CAM_IMG_SIZE and 0 <= gt[1] < CAM_IMG_SIZE:
                pde[name] = float(np.linalg.norm(uv[i] - gt)) * CAM_PIX_MM
    return pde


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def _load_mrk_3d(path: Path) -> Dict[str, np.ndarray]:
    d = json.loads(path.read_text())
    return {cp['label'].strip(): np.array(cp['position'], dtype=np.float64)
            for cp in d['markups'][0]['controlPoints']
            if cp.get('positionStatus', 'defined') == 'defined'}


def _load_mrk_2d(path: Path) -> Dict[str, np.ndarray]:
    """mrk.json with pixel coords stored as (u, v, 0)."""
    d = json.loads(path.read_text())
    result: Dict[str, np.ndarray] = {}
    for cp in d['markups'][0]['controlPoints']:
        if cp.get('positionStatus', 'defined') == 'defined':
            pos = cp['position']
            result[cp['label'].strip()] = np.array([pos[0], pos[1]], dtype=np.float64)
    return result


def _load_nrrd(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).squeeze().astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


# ---------------------------------------------------------------------------
# EPnP / anatomy pose
# ---------------------------------------------------------------------------
def _solve_pnp(pts3d: np.ndarray, pts2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(pts3d)
    flag = cv2.SOLVEPNP_SQPNP if n == 3 else cv2.SOLVEPNP_EPNP
    _, rvec, tvec = cv2.solvePnP(
        xzy(pts3d).astype(np.float64), pts2d.astype(np.float64),
        CAM_K, np.zeros(4), flags=flag)
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.flatten().astype(np.float64)


def _reproj_err(pts3d: np.ndarray, pts2d: np.ndarray,
                R: np.ndarray, t: np.ndarray) -> float:
    uv = project_world(pts3d, R, t)
    return float(np.sqrt(((uv - pts2d)**2).sum(axis=1)).mean())


def _anatomy_pose(lm_3d: Dict[str, np.ndarray],
                  azimuth_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(np.array(list(lm_3d.values()), dtype=np.float64), axis=0)
    az = np.deg2rad(azimuth_deg)
    src_dir   = np.array([np.sin(az), np.cos(az), 0.], dtype=np.float64)
    src_world = centroid + CAM_SID_MM * src_dir
    src_xzy   = src_world[[0, 2, 1]]
    cnt_xzy   = centroid[[0, 2, 1]]
    z_cam = (cnt_xzy - src_xzy) / np.linalg.norm(cnt_xzy - src_xzy)
    up    = np.array([0., 0., -1.])
    if abs(z_cam.dot(up)) > 0.9:
        up = np.array([1., 0., 0.])
    x_cam = np.cross(up, z_cam); x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    R = np.stack([x_cam, y_cam, z_cam], axis=0)
    return R.astype(np.float64), (-R @ src_xzy).astype(np.float64)


# ---------------------------------------------------------------------------
# GenericPatientLoader
# ---------------------------------------------------------------------------
LUMBAR = {'L1', 'L2', 'L3', 'L4', 'L5'}


class GenericPatientLoader:
    """
    Load any patient in the newdata canonical folder format.

    Usage::
        loader = GenericPatientLoader('/path/to/SARKHI')
        spec   = loader.load()
        spec   = loader.load(min_labels=4)   # only sets with ≥4 common labels
    """

    def __init__(self, patient_root: Path):
        self.root   = Path(patient_root)
        self.pid    = self.root.name

    # ── private ──────────────────────────────────────────────────────────────

    def _get_ct(self) -> Tuple[Path, Path]:
        ct_dir   = self.root / 'CT label'
        ct_nrrds = list(ct_dir.glob('*.nrrd'))
        ct_jsons = [f for f in ct_dir.glob('*.mrk.json')]
        if not ct_nrrds: raise FileNotFoundError(f"[{self.pid}] CT NRRD missing in {ct_dir}")
        if not ct_jsons: raise FileNotFoundError(f"[{self.pid}] CT centroid JSON missing in {ct_dir}")
        return ct_nrrds[0], ct_jsons[0]

    def _discover_sets(self, min_labels: int) -> List[Tuple[str, Path, Path, str]]:
        """
        Returns (set_name, nrrd_path, json_path, view_tag) for all sets
        where: image exists AND json exists AND ≥min_labels common labels with CT.
        """
        ct_lm_3d = _load_mrk_3d(self._get_ct()[1])
        found = []
        for view_tag, subfolder in [('AP', 'XRAY-AP'), ('LAT', 'XRAY-LAT')]:
            vdir = self.root / 'XRAY Label' / subfolder
            if not vdir.exists(): continue
            for sdir in sorted(vdir.iterdir()):
                nrrds = list(sdir.glob('*.nrrd'))
                jsons = list(sdir.glob('*.mrk.json'))
                if not nrrds or not jsons: continue
                lm_2d = _load_mrk_2d(jsons[0])
                lumbar_2d = {k: v for k, v in lm_2d.items() if k in LUMBAR}
                common = [l for l in lumbar_2d if l in ct_lm_3d]
                if len(common) >= min_labels:
                    found.append((sdir.name, nrrds[0], jsons[0], view_tag))
        return found

    # ── public ───────────────────────────────────────────────────────────────

    def load(self, min_labels: int = 3,
             verbose: bool = True) -> DeepFluoroSpecimen:
        ct_nrrd, ct_json = self._get_ct()

        if verbose:
            print(f"[{self.pid}] Loading CT: {ct_nrrd.name}", flush=True)
        img    = sitk.ReadImage(str(ct_nrrd))
        ct_vol = sitk.GetArrayFromImage(img).astype(np.float32)
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        origin  = np.array(img.GetOrigin(),  dtype=np.float64)
        lm_3d   = _load_mrk_3d(ct_json)
        # Keep only lumbar labels that exist
        lm_3d = {k: v for k, v in lm_3d.items() if k in LUMBAR | {'D12', 'D11'}}
        if verbose:
            Z, Y, X = ct_vol.shape
            print(f"  Shape=({Z},{Y},{X}) sp={spacing.round(3)} HU=[{ct_vol.min():.0f},{ct_vol.max():.0f}]")
            print(f"  3D labels: {sorted(lm_3d.keys())}")

        sets = self._discover_sets(min_labels)
        if verbose:
            print(f"  Runnable sets (≥{min_labels} common labels): {[s[0] for s in sets]}")

        projections: List[GenericProjection] = []
        for idx, (set_name, nrrd_path, json_path, view_tag) in enumerate(sets):
            image_raw  = _load_nrrd(nrrd_path)
            lm_2d_all  = _load_mrk_2d(json_path)
            lumbar_2d  = {k: v for k, v in lm_2d_all.items() if k in LUMBAR}
            common     = sorted(l for l in lumbar_2d if l in lm_3d)
            az_fallback = 0.0 if view_tag == 'AP' else 90.0

            R_init, t_init, reproj_err = None, None, 0.0
            if len(common) >= 3:
                pts3d = np.array([lm_3d[l] for l in common])
                pts2d = np.array([lumbar_2d[l] for l in common])
                try:
                    R_init, t_init = _solve_pnp(pts3d, pts2d)
                    reproj_err     = _reproj_err(pts3d, pts2d, R_init, t_init)
                    if verbose:
                        print(f"  {set_name}: EPnP reproj={reproj_err:.2f}px "
                              f"({reproj_err*CAM_PIX_MM:.2f}mm) labels={common}")
                except Exception as e:
                    if verbose: print(f"  {set_name}: EPnP failed ({e}) → anatomy")
                    R_init = None

            if R_init is None:
                R_init, t_init = _anatomy_pose(lm_3d, az_fallback)
                if verbose: print(f"  {set_name}: anatomy pose (az={az_fallback}°)")

            projections.append(GenericProjection(
                specimen_id     = self.pid,
                proj_index      = idx,
                proj_key        = set_name,
                image_raw       = image_raw,
                image_display   = image_raw,
                R_proj          = R_init,
                t_proj          = t_init,
                gt_landmarks_2d = lumbar_2d,
                rot_180_for_up  = False,
                reproj_error_px = reproj_err,
            ))

        if not projections:
            raise RuntimeError(f"[{self.pid}] No runnable sets found (min_labels={min_labels})")

        spec = DeepFluoroSpecimen(
            specimen_id  = self.pid,
            ct_volume    = ct_vol,
            ct_spacing   = spacing,
            ct_origin    = origin,
            landmarks_3d = lm_3d,
            projections  = projections,
        )
        if verbose: print(f"  Loaded: {spec}\n")
        return spec
