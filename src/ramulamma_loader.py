"""
ramulamma_loader.py — Real Patient Loader for Ramulamma
=======================================================
Loads Ramulamma's pre-op CT (NRRD) + intra-op C-arm DICOMs and wraps
them into the exact same DeepFluoroSpecimen / DeepFluoroProjection
interface that the existing optimiser and DRR generator already use.

NO 2D LANDMARK ANNOTATIONS REQUIRED.
Initial pose is estimated from the 3D anatomy centroid and a set of
canonical C-arm view angles (AP + lateral), so the CMA-ES optimizer
can run directly on the real X-ray images.

Data layout expected:
  CT (pre-op NRRD):
    data/RAMULAMMA PREOP-*/RAMULAMMA PREOP/4 L_Spine  1.0  B60s.nrrd
      512×512×395, spacing 0.668×0.668×0.700 mm

  3D landmarks (vertebral centroids, Slicer mrk.json):
    data/RAMULAMMA PREOP-*/RAMULAMMA PREOP/centroids.mrk.json
      Markups → controlPoints → label, position (RAS mm)
      Labels: L1, L2, L3, L4, L5

  C-arm DICOMs (Ziehm Vision FD, 1024×1024 16-bit):
    data/Ramulamma intra op Dicom images/20251011/0/1 .. 107
      DistanceSourceToDetector = 1110 mm
      Pixel spacing = 0.2 mm (Ziehm private tag 0019,1014)

Camera model (identical to DeepFluoro):
  u = Fx * P[0]/P[2] + Cx
  v = Fy * P[1]/P[2] + Cy
  where P = R @ xzy(pt_world) + t
  Fx = Fy = SID / pixel_spacing = 1110 / 0.2 = 5550  (px)
  Cx = Cy = (1024 - 1) / 2 = 511.5
  Image size: 1024×1024
  Pixel spacing: 0.2 mm/px (from Ziehm tag; full FOV = 204.8 mm)

Key difference vs DeepFluoro:
  - Different Fx/Fy/Cx/Cy (Ziehm geometry vs DeepFluoro geometry)
  - Different image size (1024 vs 1536)
  - Different pixel spacing (0.2 vs 0.194 mm)
  - Anatomy: lumbar spine (L1-L5) vs pelvis (ASIS/FH/etc.)
  - No 2D annotations needed — pose initialised from 3D anatomy
  - DRR generator uses the same DeepFluoroDRR class — no change needed
"""

import os
import sys
import json
import glob
import zipfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np
import SimpleITK as sitk
import pydicom
import cv2

sys.path.insert(0, str(Path(__file__).parent))

# Re-use all the data structures and math from deepfluoro_loader
from deepfluoro_loader import (
    DeepFluoroSpecimen,
    DeepFluoroProjection,
    _solve_pnp,
    project_world_to_image,
    xzy,
    compute_pde_for_pose,
    mean_pde_for_pose,
    perturb_extrinsic,
)

# ---------------------------------------------------------------------------
# Ramulamma-specific camera constants
# ---------------------------------------------------------------------------

# Ziehm Vision FD geometry
RAMU_SID_MM: float = 1110.0          # Source-to-Image Distance (DICOM tag 0018,1110)
RAMU_PIX_MM: float = 0.2             # mm per pixel (Ziehm private tag 0019,1014)
RAMU_IMG_SIZE: int = 1024            # detector pixels (square)

# Focal lengths in pixels
RAMU_FX: float = RAMU_SID_MM / RAMU_PIX_MM       # 5550.0 px
RAMU_FY: float = RAMU_SID_MM / RAMU_PIX_MM       # 5550.0 px
RAMU_CX: float = (RAMU_IMG_SIZE - 1) / 2.0       # 511.5
RAMU_CY: float = (RAMU_IMG_SIZE - 1) / 2.0       # 511.5

# Camera matrix (3×3)
import numpy as _np
RAMU_K: _np.ndarray = _np.array([
    [RAMU_FX, 0.,      RAMU_CX],
    [0.,      RAMU_FY, RAMU_CY],
    [0.,      0.,      1.     ],
], dtype=_np.float64)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE = Path('/home/supermicro/Documents/2D_3D_Raghu/data')

CT_NRRD = (
    _BASE
    / 'testing'
    / 'RAMULAMMA PREOP'
    / 'RAMULAMMA PREOP'
    / '4 L_Spine  1.0  B60s.nrrd'
)

LM_3D_JSON = (
    _BASE
    / 'testing'
    / 'RAMULAMMA PREOP'
    / 'RAMULAMMA PREOP'
    / 'centroids.mrk.json'
)

DICOM_DIR          = _BASE / 'Ramulamma intra op Dicom images' / '20251011' / '0'
DICOM_DIR_CLEAN    = _BASE / 'Ramulamma intra op Dicom images' / '20251011' / 'without_instruments'
DICOM_DIR_INSTRUM  = _BASE / 'Ramulamma intra op Dicom images' / '20251011' / 'with_instruments'
LM_2D_DIR = DICOM_DIR.parent / 'landmarks_2d'
MRB_CENTROIDS = _BASE / 'RAMULAMMA DICOM CENTROIDS' / '2026-02-25-Scene.mrb'


# ---------------------------------------------------------------------------
# EPnP using Ramulamma intrinsics
# ---------------------------------------------------------------------------

def _solve_pnp_ramu(pts3d_world: np.ndarray,
                    pts2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """EPnP using Ramulamma camera intrinsics (K = RAMU_K)."""
    pts_xzy = xzy(pts3d_world)
    _, rvec, tvec = cv2.solvePnP(
        pts_xzy.astype(np.float64),
        pts2d.astype(np.float64),
        RAMU_K, np.zeros(4),
        flags=cv2.SOLVEPNP_EPNP,
    )
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.flatten().astype(np.float64)


def project_world_ramu(pts3d_world: np.ndarray,
                       R_proj: np.ndarray,
                       t_proj: np.ndarray) -> np.ndarray:
    """Project 3D world XYZ → 2D pixel (Ramulamma intrinsics)."""
    pts = np.atleast_2d(pts3d_world)
    P_cam = (R_proj @ xzy(pts).T).T + t_proj
    u = RAMU_FX * P_cam[:, 0] / P_cam[:, 2] + RAMU_CX
    v = RAMU_FY * P_cam[:, 1] / P_cam[:, 2] + RAMU_CY
    return np.stack([u, v], axis=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mrk_json_3d(json_path: Path) -> Dict[str, np.ndarray]:
    """Load Slicer mrk.json → {label: (3,) LPS mm}.

    Slicer mrk.json stores positions in RAS mm, but the CT NRRD loaded via
    SimpleITK uses LPS. For this dataset the Slicer RAS values already match
    the SimpleITK LPS frame (verified: raw positions fall inside CT bbox),
    so we use them as-is without sign conversion.
    """
    with open(json_path) as f:
        d = json.load(f)
    cps = d['markups'][0]['controlPoints']
    landmarks = {}
    for cp in cps:
        label = cp['label'].strip()
        pos = np.array(cp['position'], dtype=np.float64)
        landmarks[label] = pos.copy()
    return landmarks


def _load_mrk_json_2d(json_path: Path) -> Dict[str, np.ndarray]:
    """Load Slicer mrk.json for 2D annotations → {label: (2,) [u, v]}."""
    with open(json_path) as f:
        d = json.load(f)
    cps = d['markups'][0]['controlPoints']
    landmarks = {}
    for cp in cps:
        label = cp['label'].strip()
        pos = np.array(cp['position'], dtype=np.float64)
        landmarks[label] = pos[:2]   # (u, v)
    return landmarks


def _load_mrk_json_2d_from_text(raw_text: str) -> Dict[str, np.ndarray]:
    """Load Slicer mrk.json text for 2D annotations → {label: (2,) [u, v]}."""
    d = json.loads(raw_text)
    cps = d['markups'][0]['controlPoints']
    landmarks = {}
    for cp in cps:
        label = cp['label'].strip()
        pos = np.array(cp['position'], dtype=np.float64)
        landmarks[label] = pos[:2]
    return landmarks


def _frame_number_from_mrk_text(raw_text: str) -> Optional[int]:
    """Infer frame number from mrk control-point z coordinate (stored as slice/frame id)."""
    d = json.loads(raw_text)
    cps = d.get('markups', [{}])[0].get('controlPoints', [])
    if not cps:
        return None
    z_vals = np.array([float(cp.get('position', [0.0, 0.0, np.nan])[2]) for cp in cps], dtype=np.float64)
    z_vals = z_vals[np.isfinite(z_vals)]
    if z_vals.size == 0:
        return None
    return int(round(float(np.median(z_vals))))


def _load_dicom_image(dcm_path: Path) -> np.ndarray:
    """Load a Ramulamma DICOM → float32 [0, 1], shape (1024, 1024)."""
    ds = pydicom.dcmread(str(dcm_path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    # Normalise: Ziehm stores 16-bit linear intensity
    arr = arr / arr.max() if arr.max() > 0 else arr
    # C-arm images are inverted vs DRR (bone = bright in DICOM, dark in DRR)
    # Keep as-is; the optimiser handles polarity via 1-drr inversion
    return arr


def _reproj_error(pts3d: np.ndarray, pts2d: np.ndarray,
                  R: np.ndarray, t: np.ndarray) -> float:
    """Mean reprojection error in pixels (Ramulamma intrinsics)."""
    uv_pred = project_world_ramu(pts3d, R, t)
    return float(np.sqrt(((uv_pred - pts2d) ** 2).sum(axis=1)).mean())


# ---------------------------------------------------------------------------
# RamuSpecimen — thin subclass that overrides projection math
# ---------------------------------------------------------------------------

class RamuProjection(DeepFluoroProjection):
    """Ramulamma projection.  Inherits all fields from DeepFluoroProjection
    but overrides .project() to use Ramulamma's camera intrinsics."""

    def project(self, pts3d_world: np.ndarray) -> np.ndarray:
        return project_world_ramu(pts3d_world, self.R_proj, self.t_proj)


# ---------------------------------------------------------------------------
# Public helpers that replace the deepfluoro_loader equivalents
# (used by run_ramulamma.py)
# ---------------------------------------------------------------------------

def compute_pde_ramu(proj: RamuProjection,
                     R_cand: np.ndarray,
                     t_cand: np.ndarray,
                     pts3d: np.ndarray,
                     lm_names: List[str]) -> Dict[str, float]:
    """PDE (mm) between candidate pose and GT 2D landmarks (Ramulamma frame)."""
    uv_pred = project_world_ramu(pts3d, R_cand, t_cand)
    pde: Dict[str, float] = {}
    for i, name in enumerate(lm_names):
        if name in proj.gt_landmarks_2d:
            gt = proj.gt_landmarks_2d[name]
            if 0 < gt[0] < RAMU_IMG_SIZE and 0 < gt[1] < RAMU_IMG_SIZE:
                err_px = float(np.linalg.norm(uv_pred[i] - gt))
                pde[name] = err_px * RAMU_PIX_MM
    return pde


def mean_pde_ramu(proj, R, t, pts3d, lm_names) -> float:
    d = compute_pde_ramu(proj, R, t, pts3d, lm_names)
    return float(np.mean(list(d.values()))) if d else float('inf')


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class RamulamaLoader:
    """
    Load Ramulamma pre-op CT + intra-op C-arm data.

    Returns a DeepFluoroSpecimen with RamuProjection objects so that the
    existing optimiser (run_deepfluoro.py / run_ramulamma.py) works
    without modification.

    Usage::

        loader = RamulamaLoader()
        spec   = loader.load()           # all annotated frames
        spec   = loader.load(frames=[50, 60, 70])   # specific frames
    """

    def __init__(self,
                 ct_nrrd: Path = CT_NRRD,
                 lm_3d_json: Path = LM_3D_JSON,
                 dicom_dir: Path = DICOM_DIR,
                 lm_2d_dir: Path = LM_2D_DIR):
        self.ct_nrrd    = Path(ct_nrrd)
        self.lm_3d_json = Path(lm_3d_json)
        self.dicom_dir  = Path(dicom_dir)
        self.lm_2d_dir  = Path(lm_2d_dir)

    # ── private ──────────────────────────────────────────────────────────────

    def _load_ct(self, verbose: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (ct_volume, ct_spacing, ct_origin) in mm."""
        if verbose:
            print(f"  Loading CT: {self.ct_nrrd.name}")
        img = sitk.ReadImage(str(self.ct_nrrd))
        arr = sitk.GetArrayFromImage(img)      # (Z, Y, X) int16 — keep this order for DRR generator
        # Convert HU: NRRD from Slicer stores raw HU directly
        ct_vol = arr.astype(np.float32)        # keep HU values
        spacing = np.array(img.GetSpacing(), dtype=np.float64)   # (sx, sy, sz) mm
        origin  = np.array(img.GetOrigin(),  dtype=np.float64)   # LPS mm
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

    def _find_dicom_files(self) -> Dict[int, Path]:
        """Return {instance_number: path} for all valid Ramulamma DICOMs."""
        files = {}
        for p in sorted(self.dicom_dir.iterdir()):
            if p.is_file():
                try:
                    ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                    inst = int(getattr(ds, 'InstanceNumber', -1))
                    if inst > 0:
                        files[inst] = p
                except Exception:
                    pass
        return files

    def _load_2d_landmark_files(self) -> Dict[int, Path]:
        """Return {frame_number: path} for all annotated 2D landmark JSONs."""
        if not self.lm_2d_dir.exists():
            return {}
        result = {}
        for p in sorted(self.lm_2d_dir.glob('frame_*.mrk.json')):
            try:
                # filename: frame_50.mrk.json → 50
                num = int(p.stem.split('_')[1])
                result[num] = p
            except Exception:
                pass
        return result

    def _load_2d_landmarks_from_mrb(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Return {frame_number: {label: [u,v]}} from Slicer MRB centroid archive."""
        if not MRB_CENTROIDS.exists() or not zipfile.is_zipfile(str(MRB_CENTROIDS)):
            return {}

        result: Dict[int, Dict[str, np.ndarray]] = {}
        with zipfile.ZipFile(str(MRB_CENTROIDS), 'r') as zf:
            for name in zf.namelist():
                low = name.lower()
                if not low.endswith('.mrk.json'):
                    continue
                if 'centroid' not in low:
                    continue
                raw = zf.read(name).decode('utf-8')
                frame_num = _frame_number_from_mrk_text(raw)
                if frame_num is None:
                    continue
                lm2d = _load_mrk_json_2d_from_text(raw)
                if lm2d:
                    result[frame_num] = lm2d

        return result

    # ── public ───────────────────────────────────────────────────────────────

    def _anatomy_pose(self,
                      lm_3d: Dict[str, np.ndarray],
                      azimuth_deg: float = 0.0,
                      elevation_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build an (R, t) pose by placing the X-ray source above the 3D
        anatomy centroid at distance SID mm, looking toward the centroid.

        azimuth_deg   : rotation around the patient's Z (SI) axis
                        0° = posterior→anterior (PA / AP view)
                       90° = right→left (lateral)
        elevation_deg : tilt up/down from the horizontal plane
        """
        centroid = np.mean(
            np.array(list(lm_3d.values()), dtype=np.float64), axis=0
        )  # LPS mm  (X, Y, Z)

        # Source position: SID mm from centroid in the chosen view direction
        az  = np.deg2rad(azimuth_deg)
        el  = np.deg2rad(elevation_deg)
        # Unit vector from centroid → source  (in LPS world)
        # AP view: source is anterior (+Y in LPS), so direction = (0, +1, 0)
        src_dir = np.array([
            np.sin(az) * np.cos(el),
             np.cos(az) * np.cos(el),
            np.sin(el)
        ], dtype=np.float64)
        src_world = centroid + RAMU_SID_MM * src_dir  # LPS world mm

        # Build camera coordinate frame
        # Camera looks from source toward centroid  → -z direction in camera
        # src_world and centroid are in LPS; DRR uses XZY permutation:
        src_xzy  = src_world[[0, 2, 1]]
        cent_xzy = centroid[[0, 2, 1]]

        z_cam = cent_xzy - src_xzy; z_cam /= np.linalg.norm(z_cam)
        # Up vector: try world Y (LPS), then fall back to Z
        world_up = np.array([0., 0., -1.])  # superior direction in XZY
        if abs(z_cam.dot(world_up)) > 0.9:
            world_up = np.array([1., 0., 0.])
        x_cam = np.cross(world_up, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)

        R = np.stack([x_cam, y_cam, z_cam], axis=0)   # (3,3)
        t = -R @ src_xzy                               # (3,)
        return R.astype(np.float64), t.astype(np.float64)

    def load(self,
             frames: Optional[List[int]] = None,
             verbose: bool = True) -> DeepFluoroSpecimen:
        """
        Load Ramulamma as a DeepFluoroSpecimen.

        No 2D annotations required.  Initial pose for each frame is derived
        from the 3D anatomy centroid at a canonical view angle.

        Args:
            frames  : list of DICOM InstanceNumber values to include.
                      None = evenly-spaced sample of 8 frames from all 107.
            verbose : print progress

        Returns:
            DeepFluoroSpecimen with RamuProjection objects.
            spec.landmarks_3d  = {L1..L5: (3,) LPS mm}
            spec.projections   = list of RamuProjection (one per selected frame)
        """
        if verbose:
            print("[Ramulamma] Loading specimen ...")

        ct_vol, spacing, origin = self._load_ct(verbose)
        lm_3d = self._load_3d_landmarks(verbose)

        dicom_files = self._find_dicom_files()
        lm_2d_files = self._load_2d_landmark_files()
        lm_2d_from_mrb = self._load_2d_landmarks_from_mrb()

        if verbose:
            print(f"  DICOM frames found: {len(dicom_files)}")
            print(f"  2D landmark JSONs found: {len(lm_2d_files)}")
            print(f"  2D centroids from MRB: {len(lm_2d_from_mrb)}")

        # Which frames to use
        all_frames = sorted(dicom_files.keys())
        if frames is not None:
            selected = sorted(set(frames) & set(all_frames))
        else:
            # Default: 8 evenly-spaced frames
            n = len(all_frames)
            step = max(1, n // 8)
            selected = all_frames[::step][:8]

        if not selected:
            raise RuntimeError(f"No matching DICOM frames found in {self.dicom_dir}")

        if verbose:
            print(f"  Selected frames: {selected}")

        # Canonical poses: AP view and slight oblique variations
        # The optimizer will correct from these starting poses
        # azimuth 0 = AP (posterior→anterior)
        canonical_azimuths = [0.0, 15.0, -15.0, 30.0]
        R_ap, t_ap = self._anatomy_pose(lm_3d, azimuth_deg=0.0, elevation_deg=0.0)

        # Build projections — one per selected frame
        projections: List[RamuProjection] = []
        for idx, frame_num in enumerate(selected):
            dcm_path = dicom_files[frame_num]

            # Load image
            image_raw = _load_dicom_image(dcm_path)

            # Default pose from anatomy centroid; override with EPnP when 2D landmarks exist.
            R_init, t_init = self._anatomy_pose(
                lm_3d,
                azimuth_deg=canonical_azimuths[idx % len(canonical_azimuths)],
                elevation_deg=0.0,
            )

            gt_landmarks_2d: Dict[str, np.ndarray] = {}
            if frame_num in lm_2d_files:
                gt_landmarks_2d = _load_mrk_json_2d(lm_2d_files[frame_num])
            elif frame_num in lm_2d_from_mrb:
                gt_landmarks_2d = lm_2d_from_mrb[frame_num]

            reproj_error_px = 0.0
            common = sorted(set(lm_3d.keys()) & set(gt_landmarks_2d.keys()))
            if len(common) >= 4:
                pts3d = np.array([lm_3d[n] for n in common], dtype=np.float64)
                pts2d = np.array([gt_landmarks_2d[n] for n in common], dtype=np.float64)
                try:
                    R_init, t_init = _solve_pnp_ramu(pts3d, pts2d)
                    reproj_error_px = _reproj_error(pts3d, pts2d, R_init, t_init)
                except Exception:
                    pass

            proj = RamuProjection(
                specimen_id      = 'ramulamma',
                proj_index       = idx,
                proj_key         = str(frame_num).zfill(3),
                image_raw        = image_raw,
                image_display    = image_raw,
                R_proj           = R_init,
                t_proj           = t_init,
                gt_landmarks_2d  = gt_landmarks_2d,
                rot_180_for_up   = False,
                reproj_error_px  = reproj_error_px,
            )
            projections.append(proj)

            if verbose:
                centroid = np.mean(
                    np.array(list(lm_3d.values()), dtype=np.float64), axis=0
                )
                ann = len(gt_landmarks_2d)
                if ann > 0:
                    print(f"  frame {frame_num:>3}: init=2D landmarks ({ann})  reproj={reproj_error_px:.2f}px")
                else:
                    print(f"  frame {frame_num:>3}: init pose az={canonical_azimuths[idx % len(canonical_azimuths)]:.0f}°  "
                          f"anatomy centroid=[{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]mm")

        # Assemble DeepFluoroSpecimen
        spec = DeepFluoroSpecimen(
            specimen_id  = 'ramulamma',
            ct_volume    = ct_vol,
            ct_spacing   = spacing,
            ct_origin    = origin,
            landmarks_3d = lm_3d,
            projections  = projections,
        )

        if verbose:
            print(f"\n  Loaded: {spec}")

        return spec
