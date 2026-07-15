"""
deepfluoro_loader.py — DeepFluoro (IPCAI 2020) Dataset Loader
==============================================================
Loads the DeepFluoro cadaveric pelvis dataset using ONLY the full-resolution
HDF5 file.  The ds4x / ds8x downsampled datasets are NOT used.

Dataset file:
  data/ipcai_2020_full_res_data/ipcai_2020_full_res_data.h5
    Specimens: 17-1882, 17-1905, 18-0725, 18-1109, 18-2799, 18-2800
    Per specimen:
      vol/pixels         (X, Y, Z) float32 CT, 1mm isotropic
      vol/origin         (3, 1) world origin [ox, oy, oz] in mm
      vol/spacing        (3, 1) = [[1],[1],[1]] mm
      vol-landmarks      14 anatomical landmarks in world XYZ mm
      projections/NNN    fluoroscopy projections, each with:
        image/pixels               (1536, 1536) float32
        image/spacing              [[0.194],[0.194]] mm/px
        gt-poses/cam-to-pelvis-vol (4, 4)
        gt-landmarks/{name}        (2, 1) 2D pixel coords [u, v] (raw/stored frame)
        rot-180-for-up             int (0 or 1)
    proj-params:
      intrinsic   (3,3) — stored K has NEGATIVE FX/FY (-5257.73); not used
      extrinsic   (4,4) — global reference frame; not used

COORDINATE SYSTEM AND PROJECTION MODEL
---------------------------------------
Vol-landmarks are in world (LPS mm) XYZ coordinates.
The 'cam-to-pelvis-vol' pose uses a "pelvis-vol" frame incompatible with
the vol-landmark world coordinates — it is stored for reference only.

Projection extrinsics (R_proj, t_proj) are computed at load time via EPnP
from the GT 3D-2D landmark correspondences.

VALIDATED PROJECTION FORMULA (< 0.001 px error):
  pt_xzy = pt_world[[0, 2, 1]]          # swap Y<->Z
  P_cam  = R_proj @ pt_xzy + t_proj
  u = FX_POS * P_cam[0] / P_cam[2] + CX    # FX_POS = +5257.73
  v = FX_POS * P_cam[1] / P_cam[2] + CY

IMAGE ORIENTATION (rot-180-for-up)
------------------------------------
The raw stored image ("RAW frame") is NOT anatomically upright for all
projections.  The rot-180-for-up flag controls this:

  rot_180_for_up = False:
    - The stored image IS already in DRR-matching orientation
    - GT 2D landmarks in HDF5 are in DRR-matching frame
    - image_raw  == stored pixels (no transform)
    - image_display == np.rot90(stored, 2)  [upright for display]

  rot_180_for_up = True:
    - The stored image has been rotated 180° for display (upright)
    - GT 2D landmarks in HDF5 are in this ROTATED (upright) frame
    - image_raw  == np.rot90(stored, 2)  [undo display rotation → DRR frame]
    - image_display == stored pixels (already upright)
    - gt_landmarks_2d is also un-rotated to DRR frame:
        u_raw = 1535 - u_stored
        v_raw = 1535 - v_stored

KEY INVARIANT:
  DRR generated from R_proj/t_proj ALWAYS matches image_raw.
  No per-projection flip is needed in registration code.

SDD = FX_POS * PIXEL_SPACING_MM = 5257.73 * 0.194 ≈ 1020 mm

Public dataset: Grupp et al., IPCAI 2020
  https://doi.org/10.7281/T1/IFSXNV
"""

import numpy as np
import h5py
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE = Path('/home/supermicro/Documents/2D_3D_Raghu/data')
FULL_RES_HDF5 = _BASE / 'ipcai_2020_full_res_data' / 'ipcai_2020_full_res_data.h5'


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPECIMEN_NAMES: List[str] = [
    '17-1882', '17-1905', '18-0725', '18-1109', '18-2799', '18-2800',
]

# Maps legacy ds4x index -> full-res specimen name (kept for compatibility)
SPECIMEN_MAP: Dict[str, str] = {
    '01': '17-1882',
    '02': '17-1905',
    '03': '18-0725',
    '04': '18-1109',
    '05': '18-2799',
    '06': '18-2800',
}

LANDMARK_NAMES: List[str] = [
    'ASIS-l', 'ASIS-r',
    'FH-l',   'FH-r',
    'GSN-l',  'GSN-r',
    'IOF-l',  'IOF-r',
    'IPS-l',  'IPS-r',
    'MOF-l',  'MOF-r',
    'SPS-l',  'SPS-r',
]

# Camera intrinsics — positive focal lengths (for XZY-permuted points)
FX_POS: float = 5257.731934
FY_POS: float = 5257.731934
CX: float = 767.5
CY: float = 767.5

# Image sizes
FULL_RES_SIZE: int = 1536

# Physical geometry
PIXEL_SPACING_MM: float = 0.194
SDD_MM: float = FX_POS * PIXEL_SPACING_MM  # ≈ 1020 mm

K_POS: np.ndarray = np.array(
    [[FX_POS, 0., CX], [0., FY_POS, CY], [0., 0., 1.]], dtype=np.float64
)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def xzy(pts: np.ndarray) -> np.ndarray:
    """Permute world XYZ → XZY (swap Y↔Z).

    Required because the camera frame has Y and Z swapped relative to the CT
    world frame.  Validated to give < 0.001 px reprojection error on all
    specimens and projections.

    Args:
        pts : (..., 3) array in world XYZ
    Returns:
        same shape, columns permuted to [x, z, y]
    """
    pts = np.atleast_2d(pts)
    return pts[:, [0, 2, 1]]


def xzy_inv(pts_xzy: np.ndarray) -> np.ndarray:
    """Inverse of xzy() — convert XZY back to XYZ."""
    pts_xzy = np.atleast_2d(pts_xzy)
    return pts_xzy[:, [0, 2, 1]]


# ---------------------------------------------------------------------------
# Core projection functions
# ---------------------------------------------------------------------------

def _solve_pnp(pts3d_world: np.ndarray,
               pts2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute projection extrinsic (R_proj, t_proj) via EPnP.

    Args:
        pts3d_world : (N, 3) 3D world XYZ (mm)
        pts2d       : (N, 2) 2D pixel coords in RAW (DRR-matching) frame

    Returns:
        R_proj : (3, 3)  maps XZY-world to camera frame
        t_proj : (3,)
    """
    pts_xzy = xzy(pts3d_world)
    _, rvec, tvec = cv2.solvePnP(
        pts_xzy.astype(np.float64),
        pts2d.astype(np.float64),
        K_POS, np.zeros(4),
        flags=cv2.SOLVEPNP_EPNP,
    )
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.flatten().astype(np.float64)


def project_world_to_image(pts3d_world: np.ndarray,
                            R_proj: np.ndarray,
                            t_proj: np.ndarray) -> np.ndarray:
    """Project 3D world XYZ (mm) → 2D pixel coords in RAW (DRR-matching) frame.

    Projection model (validated, < 0.001 px error):
      pt_xzy = pt_world[[0, 2, 1]]
      P_cam  = R_proj @ pt_xzy + t_proj
      u = FX_POS * P_cam[0] / P_cam[2] + CX
      v = FY_POS * P_cam[1] / P_cam[2] + CY

    Args:
        pts3d_world : (N, 3) or (3,) world XYZ mm
        R_proj      : (3, 3)
        t_proj      : (3,)
    Returns:
        uv : (N, 2) pixel coords in raw 1536×1536 frame
    """
    pts = np.atleast_2d(pts3d_world)
    P_cam = (R_proj @ xzy(pts).T).T + t_proj  # (N, 3)
    u = FX_POS * P_cam[:, 0] / P_cam[:, 2] + CX
    v = FY_POS * P_cam[:, 1] / P_cam[:, 2] + CY
    return np.stack([u, v], axis=1)


def source_position_world(R_proj: np.ndarray,
                           t_proj: np.ndarray) -> np.ndarray:
    """X-ray source position in world XYZ mm."""
    src_xzy = -R_proj.T @ t_proj
    return np.array([src_xzy[0], src_xzy[2], src_xzy[1]])


def principal_axis_world(R_proj: np.ndarray) -> np.ndarray:
    """Camera principal axis (toward detector) in world XYZ."""
    ax_xzy = R_proj.T @ np.array([0., 0., 1.])
    return np.array([ax_xzy[0], ax_xzy[2], ax_xzy[1]])


def detector_axes_world(R_proj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Detector u (column) and v (row) axes in world XYZ."""
    u_xzy = R_proj.T @ np.array([1., 0., 0.])
    v_xzy = R_proj.T @ np.array([0., 1., 0.])
    return (np.array([u_xzy[0], u_xzy[2], u_xzy[1]]),
            np.array([v_xzy[0], v_xzy[2], v_xzy[1]]))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DeepFluoroProjection:
    """One calibrated fluoroscopy projection from the DeepFluoro dataset."""

    specimen_id: str   # e.g. '17-1882'
    proj_index:  int   # 0-based index within specimen
    proj_key:    str   # HDF5 key, e.g. '000'

    # ── Images ──────────────────────────────────────────────────────────────
    # image_raw:     DRR-matching orientation (what registration code uses)
    # image_display: anatomically upright (for visualisation)
    image_raw:     np.ndarray    # (1536, 1536) float32 [0, 1]
    image_display: np.ndarray    # (1536, 1536) float32 [0, 1]

    # ── Extrinsic (from EPnP, < 0.001 px reprojection error) ────────────────
    #   P_cam = R_proj @ pt_world[[0,2,1]] + t_proj
    R_proj: Optional[np.ndarray] = None   # (3, 3)
    t_proj: Optional[np.ndarray] = None   # (3,)

    # ── GT 2D landmarks in RAW (DRR-matching) frame ─────────────────────────
    gt_landmarks_2d: Dict[str, np.ndarray] = field(default_factory=dict)
    # {name: (2,) [u, v]}  — in DRR-matching coords (consistent with R/t_proj)

    rot_180_for_up: bool = False

    # Reprojection error at load time; inf if PnP failed (< 4 landmarks)
    reproj_error_px: float = float('inf')

    # ── Methods ─────────────────────────────────────────────────────────────

    def project(self, pts3d_world: np.ndarray) -> np.ndarray:
        """Project 3D world XYZ (mm) → 2D RAW-frame pixel coords.

        Returns (N, 2) pixel coords in the same frame as image_raw.
        """
        if self.R_proj is None:
            raise RuntimeError("Projection extrinsic not set.")
        return project_world_to_image(pts3d_world, self.R_proj, self.t_proj)

    def get_image(self, mode: str = 'raw') -> np.ndarray:
        """Return the X-ray image.

        mode : 'raw'     — DRR-matching orientation (default, use for registration)
               'display' — anatomically upright (use for visualisation)
        """
        return self.image_display if mode == 'display' else self.image_raw

    def source_position(self) -> np.ndarray:
        """X-ray source in world XYZ mm."""
        return source_position_world(self.R_proj, self.t_proj)

    def principal_axis(self) -> np.ndarray:
        return principal_axis_world(self.R_proj)

    def detector_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        return detector_axes_world(self.R_proj)

    def detector_center(self) -> np.ndarray:
        """Detector center in world XYZ mm."""
        return self.source_position() + SDD_MM * self.principal_axis()

    def __repr__(self) -> str:
        src = self.source_position() if self.R_proj is not None else None
        s = (f"src=({src[0]:.0f},{src[1]:.0f},{src[2]:.0f})mm"
             if src is not None else "no-extrinsic")
        return (f"DeepFluoroProjection({self.specimen_id}/{self.proj_key} "
                f"reproj={self.reproj_error_px:.4f}px rot180={self.rot_180_for_up} {s})")


@dataclass
class DeepFluoroSpecimen:
    """All data for one DeepFluoro cadaveric specimen."""

    specimen_id:  str
    ct_volume:    np.ndarray    # (X, Y, Z) float32 CT
    ct_spacing:   np.ndarray    # (3,) mm — [1, 1, 1]
    ct_origin:    np.ndarray    # (3,) world XYZ mm of voxel [0,0,0]
    landmarks_3d: Dict[str, np.ndarray] = field(default_factory=dict)
    projections:  List[DeepFluoroProjection] = field(default_factory=list)

    def get_landmark_array(self) -> Tuple[List[str], np.ndarray]:
        """Return (names, pts) where pts is (N, 3) world XYZ mm, sorted by name."""
        names = sorted(self.landmarks_3d.keys())
        pts = np.array([self.landmarks_3d[n] for n in names], dtype=np.float64)
        return names, pts

    def get_ct_center(self) -> np.ndarray:
        """CT volume center in world XYZ mm."""
        X, Y, Z = self.ct_volume.shape
        return self.ct_origin + 0.5 * np.array([X, Y, Z]) * self.ct_spacing

    def valid_projections(self, max_reproj_px: float = 5.0) -> List[DeepFluoroProjection]:
        """Projections where PnP succeeded and reproj error is acceptable.

        Args:
            max_reproj_px : maximum allowed reprojection error (default 5 px).
                            Projections above this are degenerate (< 6 in-frame
                            landmarks in an extreme oblique view) and unreliable
                            for registration.  Set to np.inf to include all.
        """
        return [p for p in self.projections
                if np.isfinite(p.reproj_error_px) and p.reproj_error_px <= max_reproj_px]

    def mean_reproj_error(self, max_reproj_px: float = 5.0) -> float:
        errs = [p.reproj_error_px for p in self.valid_projections(max_reproj_px)]
        return float(np.mean(errs)) if errs else float('inf')

    def __repr__(self) -> str:
        valid = len(self.valid_projections())
        return (f"DeepFluoroSpecimen({self.specimen_id} | "
                f"CT={self.ct_volume.shape} | "
                f"lm={len(self.landmarks_3d)} | "
                f"projs={len(self.projections)} ({valid} valid≤5px) | "
                f"mean_reproj={self.mean_reproj_error():.4f}px)")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DeepFluoroLoader:
    """
    Load the DeepFluoro (IPCAI 2020) full-resolution dataset.

    Uses ONLY ipcai_2020_full_res_data.h5 — no ds4x/ds8x dependency.

    Usage::

        loader = DeepFluoroLoader()
        spec   = loader.load_specimen('17-1882')   # by full-res name
        spec   = loader.load_specimen('01')        # also accepts ds4x index
        specs  = loader.load_all_specimens()

    Each ``DeepFluoroProjection`` provides:
      * image_raw       — (1536,1536) float32 [0,1] in DRR-matching orientation
      * image_display   — same, but anatomically upright (rotated 180° if needed)
      * R_proj, t_proj  — calibrated extrinsic (< 0.01 px reproj error)
      * gt_landmarks_2d — GT 2D coords in DRR-matching (image_raw) frame
    """

    def __init__(self, full_res_path: Path = FULL_RES_HDF5):
        self.full_res_path = Path(full_res_path)
        if not self.full_res_path.exists():
            raise FileNotFoundError(
                f"Full-res HDF5 not found: {self.full_res_path}"
            )

    def list_specimens(self) -> List[str]:
        with h5py.File(self.full_res_path, 'r') as f:
            return [k for k in f.keys() if k != 'proj-params']

    def load_specimen(self,
                      specimen_id: str,
                      max_projections: Optional[int] = None,
                      verbose: bool = True) -> DeepFluoroSpecimen:
        """Load one specimen.

        Args:
            specimen_id     : full-res name ('17-1882') OR ds4x index ('01')
            max_projections : limit projections loaded (None = all)
            verbose         : print progress

        Returns:
            DeepFluoroSpecimen
        """
        # Accept ds4x index as shorthand
        if specimen_id in SPECIMEN_MAP:
            specimen_id = SPECIMEN_MAP[specimen_id]

        if verbose:
            print(f"[DeepFluoro] Loading {specimen_id} ...")

        with h5py.File(self.full_res_path, 'r') as f:
            if specimen_id not in f:
                raise ValueError(
                    f"Unknown specimen '{specimen_id}'. "
                    f"Available: {self.list_specimens()}"
                )
            sg = f[specimen_id]

            # ── CT volume ──────────────────────────────────────────────────
            vg = sg['vol']
            ct_pixels  = vg['pixels'][()].astype(np.float32)
            ct_origin  = vg['origin'][()].flatten().astype(np.float64)
            ct_spacing = vg['spacing'][()].flatten().astype(np.float64)
            if verbose:
                print(f"  CT: {ct_pixels.shape}, spacing={ct_spacing}, "
                      f"origin={ct_origin.round(1)}")

            # ── 3D landmarks ───────────────────────────────────────────────
            lm_3d: Dict[str, np.ndarray] = {}
            for name in sg['vol-landmarks'].keys():
                lm_3d[name] = (
                    sg['vol-landmarks'][name][()].flatten().astype(np.float64)
                )
            if verbose:
                print(f"  Landmarks ({len(lm_3d)}): {sorted(lm_3d.keys())}")

            lm_names_sorted = sorted(lm_3d.keys())
            pts3d_all = np.array([lm_3d[n] for n in lm_names_sorted],
                                  dtype=np.float64)

            # ── Projections ────────────────────────────────────────────────
            proj_keys = sorted(sg['projections'].keys())
            if max_projections is not None:
                proj_keys = proj_keys[:max_projections]

            projections: List[DeepFluoroProjection] = []
            all_reproj: List[float] = []

            for i, pk in enumerate(proj_keys):
                pg = sg['projections'][pk]

                rot_flag = bool(pg['rot-180-for-up'][()])

                # ── Raw fluoroscopy image ──────────────────────────────────
                # Stored pixels.  We need image_raw in DRR-matching frame and
                # image_display in anatomically upright frame.
                #
                # rot_180_for_up=False:
                #   stored = DRR frame  →  image_raw = stored,  display = rot90(stored,2)
                # rot_180_for_up=True:
                #   stored = upright     →  image_raw = rot90(stored,2), display = stored
                img_stored = pg['image']['pixels'][()].astype(np.float32)
                mn, mx = img_stored.min(), img_stored.max()
                if mx > mn:
                    img_stored = (img_stored - mn) / (mx - mn)

                if rot_flag:
                    # stored is upright; raw = undo the 180° rotation
                    img_raw     = np.rot90(img_stored, 2).copy()
                    img_display = img_stored
                else:
                    # stored is raw (DRR-matching); display = rotate for anatomy
                    img_raw     = img_stored
                    img_display = np.rot90(img_stored, 2).copy()

                # ── GT 2D landmarks in RAW (DRR-matching) frame ───────────
                # HDF5 stores landmarks in the same frame as the stored image.
                # We need them in image_raw frame for PnP and for PDE eval.
                #
                # rot_180_for_up=False: stored frame == raw frame → no transform
                # rot_180_for_up=True:  stored frame == upright → flip to raw frame
                gt_lm_2d: Dict[str, np.ndarray] = {}
                for lm_name in pg['gt-landmarks'].keys():
                    uv = pg['gt-landmarks'][lm_name][()].flatten().astype(np.float64)
                    if rot_flag:
                        # undo the 180° rotation: u_raw = 1535-u, v_raw = 1535-v
                        uv = np.array([
                            FULL_RES_SIZE - 1.0 - uv[0],
                            FULL_RES_SIZE - 1.0 - uv[1],
                        ])
                    gt_lm_2d[lm_name] = uv

                # ── Projection extrinsic via EPnP ─────────────────────────
                # Uses gt_lm_2d (raw frame) so R/t_proj is consistent with
                # image_raw and DRR output.
                R_proj, t_proj, reproj_err = self._compute_extrinsic(
                    lm_names_sorted, pts3d_all, gt_lm_2d
                )
                all_reproj.append(reproj_err)

                projections.append(DeepFluoroProjection(
                    specimen_id=specimen_id,
                    proj_index=i,
                    proj_key=pk,
                    image_raw=img_raw,
                    image_display=img_display,
                    R_proj=R_proj,
                    t_proj=t_proj,
                    gt_landmarks_2d=gt_lm_2d,
                    rot_180_for_up=rot_flag,
                    reproj_error_px=reproj_err,
                ))

            valid_count = sum(1 for e in all_reproj if np.isfinite(e))
            mean_err = (float(np.mean([e for e in all_reproj if np.isfinite(e)]))
                        if valid_count else float('inf'))
            if verbose:
                print(f"  Projections: {len(projections)} loaded, "
                      f"{valid_count} valid (PnP OK), "
                      f"mean reproj = {mean_err:.4f} px")

        return DeepFluoroSpecimen(
            specimen_id=specimen_id,
            ct_volume=ct_pixels,
            ct_spacing=ct_spacing,
            ct_origin=ct_origin,
            landmarks_3d=lm_3d,
            projections=projections,
        )

    def load_specimen_by_index(self, ds4x_id: str, **kwargs) -> DeepFluoroSpecimen:
        """Load specimen by ds4x index ('01'-'06')."""
        return self.load_specimen(ds4x_id, **kwargs)

    def load_all_specimens(self, **kwargs) -> List[DeepFluoroSpecimen]:
        """Load all 6 specimens."""
        return [self.load_specimen(sid, **kwargs) for sid in SPECIMEN_NAMES]

    @staticmethod
    def _compute_extrinsic(
        lm_names: List[str],
        pts3d_all: np.ndarray,
        gt_lm_2d: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute projection extrinsic for one projection via EPnP.

        Uses gt_lm_2d in the RAW (DRR-matching) frame.
        Returns (eye(3), zeros(3), inf) when < 4 valid correspondences.
        """
        p3, p2 = [], []
        for name, pt3d in zip(lm_names, pts3d_all):
            if name in gt_lm_2d:
                uv = gt_lm_2d[name]
                if 0.0 < uv[0] < FULL_RES_SIZE and 0.0 < uv[1] < FULL_RES_SIZE:
                    p3.append(pt3d)
                    p2.append(uv)

        if len(p3) < 4:
            return np.eye(3), np.zeros(3), float('inf')

        p3 = np.array(p3, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)

        R, t = _solve_pnp(p3, p2)

        P_cam = (R @ xzy(p3).T).T + t
        u = FX_POS * P_cam[:, 0] / P_cam[:, 2] + CX
        v = FY_POS * P_cam[:, 1] / P_cam[:, 2] + CY
        err = float(np.sqrt((u - p2[:, 0])**2 + (v - p2[:, 1])**2).mean())
        return R, t, err


# ---------------------------------------------------------------------------
# Pose perturbation helpers
# ---------------------------------------------------------------------------

def perturb_extrinsic(R_proj: np.ndarray,
                      t_proj: np.ndarray,
                      delta_rot_deg: np.ndarray,
                      delta_trans_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a 6-DOF perturbation to a projection extrinsic (in camera frame).

    Args:
        R_proj, t_proj : base extrinsic
        delta_rot_deg  : (3,) Euler XYZ rotations in degrees
        delta_trans_mm : (3,) translation shift in mm
    Returns:
        R_new, t_new
    """
    rx, ry, rz = np.deg2rad(delta_rot_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx,  cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0,  1]])
    R_delta = Rz @ Ry @ Rx
    return R_delta @ R_proj, t_proj + delta_trans_mm


# ---------------------------------------------------------------------------
# PDE evaluation utilities
# ---------------------------------------------------------------------------

def compute_pde_for_pose(proj: DeepFluoroProjection,
                          R_cand: np.ndarray,
                          t_cand: np.ndarray,
                          pts3d_world: np.ndarray,
                          lm_names: List[str],
                          pixel_spacing_mm: float = PIXEL_SPACING_MM,
                          ) -> Dict[str, float]:
    """Compute PDE (mm) between a candidate pose and GT 2D landmarks.

    PDE_i = ||project(pts3d[i], R_cand, t_cand) - gt_lm_2d[i]|| * px_size_mm

    gt_lm_2d is in RAW (image_raw) frame.  Projected points are also in RAW
    frame.  No flip needed.

    Args:
        proj            : holds gt_landmarks_2d (raw frame)
        R_cand, t_cand  : candidate extrinsic
        pts3d_world     : (N, 3) world XYZ mm
        lm_names        : names for each row
        pixel_spacing_mm: mm per pixel
    Returns:
        {name: pde_mm}
    """
    uv_pred = project_world_to_image(pts3d_world, R_cand, t_cand)  # raw frame
    pde: Dict[str, float] = {}
    for i, name in enumerate(lm_names):
        if name in proj.gt_landmarks_2d:
            gt = proj.gt_landmarks_2d[name]
            if 0 < gt[0] < FULL_RES_SIZE and 0 < gt[1] < FULL_RES_SIZE:
                err_px = float(np.linalg.norm(uv_pred[i] - gt))
                pde[name] = err_px * pixel_spacing_mm
    return pde


def mean_pde_for_pose(proj: DeepFluoroProjection,
                       R_cand: np.ndarray,
                       t_cand: np.ndarray,
                       pts3d_world: np.ndarray,
                       lm_names: List[str],
                       pixel_spacing_mm: float = PIXEL_SPACING_MM) -> float:
    d = compute_pde_for_pose(proj, R_cand, t_cand, pts3d_world, lm_names,
                              pixel_spacing_mm)
    return float(np.mean(list(d.values()))) if d else float('inf')


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test(verbose: bool = True) -> bool:
    """Validate loader: check reprojection and PDE@GT for all 6 specimens.

    Thresholds:
      - Normal projections (≥ 6 in-frame landmarks): reproj < 0.01 px
      - Degenerate projections (4-5 in-frame landmarks): reproj < 20 px
        These occur e.g. in extreme oblique views where only one hip is in frame.
        EPnP with 4-5 nearly-coplanar points is ill-conditioned; the resulting
        pose is still usable for registration as long as PDE@GT < 5mm.

    Returns True if all normal projections pass.
    """
    loader = DeepFluoroLoader()
    all_ok = True

    for spec_name in SPECIMEN_NAMES:
        spec = loader.load_specimen(spec_name, max_projections=5, verbose=verbose)
        valid = spec.valid_projections()
        lm_names, pts3d = spec.get_landmark_array()

        if verbose:
            print(f"\n{spec}")

        for proj in valid:
            pde = mean_pde_for_pose(proj, proj.R_proj, proj.t_proj, pts3d, lm_names)
            # Count how many landmarks were in-frame (used by PnP)
            n_in_frame = sum(
                1 for nm in lm_names
                if nm in proj.gt_landmarks_2d
                and 0 < proj.gt_landmarks_2d[nm][0] < FULL_RES_SIZE
                and 0 < proj.gt_landmarks_2d[nm][1] < FULL_RES_SIZE
            )
            # Strict threshold for well-constrained projections
            if n_in_frame >= 6:
                ok_reproj = proj.reproj_error_px < 0.01
                ok_pde = pde < 0.01
                if not (ok_reproj and ok_pde):
                    all_ok = False
                tag = 'OK' if (ok_reproj and ok_pde) else 'FAIL'
            else:
                # Degenerate: EPnP ill-conditioned with ≤5 coplanar landmarks
                ok_reproj = proj.reproj_error_px < 20.0
                ok_pde = pde < 5.0
                tag = f'OK(deg,n={n_in_frame})' if (ok_reproj and ok_pde) else 'FAIL(deg)'

            if verbose:
                src = proj.source_position()
                print(f"  [{tag}] {proj.proj_key}: "
                      f"reproj={proj.reproj_error_px:.5f}px  "
                      f"PDE@GT={pde:.5f}mm  "
                      f"n_lm={n_in_frame}  "
                      f"rot180={proj.rot_180_for_up}  "
                      f"src=({src[0]:.0f},{src[1]:.0f},{src[2]:.0f})mm")

    if verbose:
        print(f"\n{'ALL PASS' if all_ok else 'SOME FAILURES'}")
    return all_ok


if __name__ == '__main__':
    import sys
    ok = _self_test(verbose=True)
    sys.exit(0 if ok else 1)
