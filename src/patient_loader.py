"""
patient_loader.py — Generic Patient Data Loader
================================================
A single, unified loader that handles any patient dataset that follows the
canonical  data/patients/<id>/  folder layout (or any custom paths via config).

Replaces the individual ramulamma_loader / arjun_loader / swaroopa_loader files.
Those files are kept for backwards-compatibility but are no longer needed for
new work.

Quick start
-----------
    from patient_loader import load_patient, load_patients, PATIENT_REGISTRY

    # Load one patient
    spec = load_patient('ramulamma')
    spec = load_patient('arjun', frames=['a', 'b'])
    spec = load_patient('swaroopa', frames=['ap_002', 'lat_000'])

    # Load all registered patients
    specs = load_patients()                          # all
    specs = load_patients(['ramulamma', 'swaroopa']) # subset

Adding a new patient
--------------------
Either call register_patient() at runtime, or add an entry to PATIENT_REGISTRY
below.  The minimum required structure is::

    data/patients/<id>/
        preop/ct.nrrd
        preop/centroids.mrk.json
        intraop/                  ← put images here (DICOM files, *.jpg, *.png)

Supported image formats
-----------------------
  'dicom'   — Ziehm 16-bit DICOM files (Ramulamma)
  'jpeg'    — JPEG C-arm captures (Arjun)
  'png'     — PNG C-arm captures (Swaroopa)

Supported 2D landmark formats
-------------------------------
  'none'        — no 2D annotations  (Ramulamma)
  'labelme'     — per-frame labelme JSON alongside each image (Arjun)
  'shared_json' — single JSON: {frame_key: {label: [u,v]}}  (Swaroopa)

Pose initialisation
-------------------
  EPnP   — if ≥3 common 3D↔2D landmark pairs are available (Arjun, Swaroopa)
  anatomy — anatomy-centroid canonical view (Ramulamma, or EPnP fallback)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
# Data root
# ---------------------------------------------------------------------------

_DATA_ROOT = Path(__file__).parent.parent / 'data'


# ============================================================================
# Config dataclasses
# ============================================================================

@dataclass
class CameraConfig:
    """
    Camera intrinsic model.

    If `scale_with_frame_size` is True the focal length is scaled
    proportionally to min(W,H) / ref_img_size for each frame (Arjun-style).
    Otherwise a fixed K is used for all frames.
    """
    sid_mm:              float       # source-to-image distance (mm)
    pix_mm:              float       # mm per pixel at ref_img_size
    img_size:            int         # reference square image size (px)
    scale_with_frame_size: bool = False  # per-frame scaling (variable image sizes)

    # Derived properties (computed in __post_init__)
    fx: float = field(init=False)
    fy: float = field(init=False)
    cx: float = field(init=False)
    cy: float = field(init=False)

    def __post_init__(self):
        self.fx = self.sid_mm / self.pix_mm
        self.fy = self.sid_mm / self.pix_mm
        self.cx = (self.img_size - 1) / 2.0
        self.cy = (self.img_size - 1) / 2.0

    def for_frame(self, w: int, h: int) -> Tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy) for a frame of size w×h."""
        if self.scale_with_frame_size:
            scale = min(w, h) / self.img_size
            return self.fx * scale, self.fy * scale, (w - 1) / 2.0, (h - 1) / 2.0
        return self.fx, self.fy, (w - 1) / 2.0, (h - 1) / 2.0

    def K(self, w: int = 0, h: int = 0) -> np.ndarray:
        """Return (3×3) camera matrix."""
        w = w or self.img_size
        h = h or self.img_size
        fx, fy, cx, cy = self.for_frame(w, h)
        return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float64)

    def pix_mm_for_frame(self, w: int, h: int) -> float:
        """Effective pixel spacing (mm/px) for a frame of size w×h."""
        if self.scale_with_frame_size:
            scale = min(w, h) / self.img_size
            return self.pix_mm / scale
        return self.pix_mm


@dataclass
class PatientConfig:
    """
    Full description of a patient dataset.

    Paths may be absolute or relative to `data/patients/<patient_id>/`.
    """

    patient_id: str

    # ── CT ───────────────────────────────────────────────────────────────────
    ct_nrrd:     Path           # pre-op CT volume
    lm_3d_json:  Path           # Slicer mrk.json with 3D centroids (RAS mm)

    # ── Camera ───────────────────────────────────────────────────────────────
    camera: CameraConfig

    # ── Intra-op X-ray images ────────────────────────────────────────────────
    # List of (directory, view_tag) pairs.
    # view_tag is used as a prefix in proj_key: 'ap_002', 'lat_000', etc.
    # For single-view datasets (DICOM or JPEG) use view_tag='' (empty).
    xray_sources: List[Tuple[Path, str]]
    xray_format:  str           # 'dicom' | 'jpeg' | 'png'

    # ── 2D Landmarks ─────────────────────────────────────────────────────────
    lm_2d_format: str                   # 'none' | 'labelme' | 'shared_json'
    lm_2d_json:   Optional[Path] = None # used when lm_2d_format='shared_json'

    # ── Frame selection defaults ──────────────────────────────────────────────
    default_n_frames: Optional[int] = 8  # None = all frames; int = subsample

    # ── Canonical view azimuths for anatomy-centred pose init ─────────────────
    # One per entry in xray_sources (index wraps); 0°=AP, 90°=lateral.
    canonical_azimuths: List[float] = field(default_factory=lambda: [0.0])

    def __post_init__(self):
        self.ct_nrrd    = Path(self.ct_nrrd)
        self.lm_3d_json = Path(self.lm_3d_json)
        self.xray_sources = [(Path(d), t) for d, t in self.xray_sources]
        if self.lm_2d_json:
            self.lm_2d_json = Path(self.lm_2d_json)


# ============================================================================
# Built-in patient registry
# ============================================================================

def _p(patient_id: str, *parts) -> Path:
    """Shorthand: resolve a path under data/patients/<patient_id>/."""
    return _DATA_ROOT / 'patients' / patient_id / Path(*parts)


PATIENT_REGISTRY: Dict[str, PatientConfig] = {

    # ── Ramulamma ─────────────────────────────────────────────────────────────
    # DICOM intra-op, no 2D annotations, anatomy centroid pose init.
    'ramulamma': PatientConfig(
        patient_id   = 'ramulamma',
        ct_nrrd      = _p('ramulamma', 'preop', 'ct.nrrd'),
        lm_3d_json   = _p('ramulamma', 'preop', 'centroids.mrk.json'),
        camera       = CameraConfig(
            sid_mm   = 1110.0,
            pix_mm   = 0.2,
            img_size = 1024,
        ),
        xray_sources = [(_p('ramulamma', 'intraop', 'dicoms'), '')],
        xray_format  = 'dicom',
        lm_2d_format = 'none',
        default_n_frames = 8,
        canonical_azimuths = [0.0, 15.0, -15.0, 30.0],
    ),

    # ── Arjun ─────────────────────────────────────────────────────────────────
    # JPEG intra-op, labelme 2D annotations per frame, EPnP + variable K.
    'arjun': PatientConfig(
        patient_id   = 'arjun',
        ct_nrrd      = _p('arjun', 'preop', 'ct.nrrd'),
        lm_3d_json   = _p('arjun', 'preop', 'centroids.mrk.json'),
        camera       = CameraConfig(
            sid_mm               = 1110.0,
            pix_mm               = 0.2,
            img_size             = 1024,
            scale_with_frame_size= True,   # K scales per-frame
        ),
        xray_sources = [(_p('arjun', 'intraop'), '')],
        xray_format  = 'jpeg',
        lm_2d_format = 'labelme',
        default_n_frames = None,           # all 5 frames
        canonical_azimuths = [0.0],
    ),

    # ── Swaroopa ──────────────────────────────────────────────────────────────
    # PNG intra-op (AP + lateral), shared landmarks_2d JSON, EPnP init.
    'swaroopa': PatientConfig(
        patient_id   = 'swaroopa',
        ct_nrrd      = _p('swaroopa', 'preop', 'ct.nrrd'),
        lm_3d_json   = _p('swaroopa', 'preop', 'centroids.mrk.json'),
        camera       = CameraConfig(
            sid_mm   = 1050.0,
            pix_mm   = 0.288,
            img_size = 1024,
        ),
        xray_sources = [
            (_p('swaroopa', 'intraop', 'ap'),      'ap'),
            (_p('swaroopa', 'intraop', 'lateral'),  'lat'),
        ],
        xray_format  = 'png',
        lm_2d_format = 'shared_json',
        lm_2d_json   = _p('swaroopa', 'landmarks_2d.json'),
        default_n_frames = None,           # all annotated frames
        canonical_azimuths = [0.0, 90.0],  # ap=0°, lat=90°
    ),
}


def register_patient(cfg: PatientConfig) -> None:
    """Register a new patient config at runtime."""
    PATIENT_REGISTRY[cfg.patient_id] = cfg


# ============================================================================
# Generic Projection class
# ============================================================================

class PatientProjection(DeepFluoroProjection):
    """
    One C-arm frame for any patient.
    Stores per-frame camera intrinsics so project() is always correct.
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float, **kwargs):
        # pop PatientProjection-specific keys before passing to parent
        img_w = kwargs.pop('img_w', int(cx * 2 + 1))
        img_h = kwargs.pop('img_h', int(cy * 2 + 1))
        super().__init__(**kwargs)
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        # Image dimensions (stored for PDE computation)
        self.img_w: int = img_w
        self.img_h: int = img_h

    def project(self, pts3d_world: np.ndarray) -> np.ndarray:
        """Project 3D world XYZ → 2D pixel (u, v) using per-frame intrinsics."""
        pts = np.atleast_2d(pts3d_world)
        P_cam = (self.R_proj @ xzy(pts).T).T + self.t_proj
        u = self._fx * P_cam[:, 0] / P_cam[:, 2] + self._cx
        v = self._fy * P_cam[:, 1] / P_cam[:, 2] + self._cy
        return np.stack([u, v], axis=1)


# ============================================================================
# PDE helpers (work with PatientProjection)
# ============================================================================

def compute_pde(proj: PatientProjection,
                R_cand: np.ndarray,
                t_cand: np.ndarray,
                pts3d: np.ndarray,
                lm_names: List[str],
                pix_mm: float) -> Dict[str, float]:
    """
    PDE (mm) between candidate pose reprojection and GT 2D landmarks.

    Args:
        proj      : projection with gt_landmarks_2d populated
        R_cand    : candidate rotation matrix
        t_cand    : candidate translation vector
        pts3d     : (N,3) 3D landmark positions
        lm_names  : labels matching pts3d rows
        pix_mm    : pixel spacing in mm

    Returns:
        dict {label: pde_mm}
    """
    tmp = PatientProjection.__new__(PatientProjection)
    tmp._fx, tmp._fy, tmp._cx, tmp._cy = proj._fx, proj._fy, proj._cx, proj._cy
    tmp.R_proj = R_cand
    tmp.t_proj = t_cand
    uv_pred = tmp.project(pts3d)

    pde: Dict[str, float] = {}
    for i, name in enumerate(lm_names):
        if name in proj.gt_landmarks_2d:
            gt = np.asarray(proj.gt_landmarks_2d[name])
            err_px = float(np.linalg.norm(uv_pred[i] - gt))
            pde[name] = err_px * pix_mm
    return pde


def mean_pde(proj: PatientProjection,
             R: np.ndarray,
             t: np.ndarray,
             pts3d: np.ndarray,
             lm_names: List[str],
             pix_mm: float) -> float:
    d = compute_pde(proj, R, t, pts3d, lm_names, pix_mm)
    return float(np.mean(list(d.values()))) if d else float('nan')


# ============================================================================
# File I/O helpers
# ============================================================================

def _load_mrk_json_3d(json_path: Path) -> Dict[str, np.ndarray]:
    """Load Slicer mrk.json → {label: (3,) position as-is (RAS mm)}."""
    with open(json_path) as f:
        d = json.load(f)
    cps = d['markups'][0]['controlPoints']
    return {
        cp['label'].strip(): np.array(cp['position'], dtype=np.float64)
        for cp in cps
    }


def _load_ct(ct_path: Path, verbose: bool = True
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (ct_volume float32 HU, spacing (3,), origin (3,))."""
    if not ct_path.exists():
        raise FileNotFoundError(
            f"CT NRRD not found: {ct_path}\n"
            f"Run the data organisation step first or check the patient config."
        )
    if verbose:
        print(f"  Loading CT: {ct_path.name}")
    img     = sitk.ReadImage(str(ct_path))
    ct_vol  = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing = np.array(img.GetSpacing(),  dtype=np.float64)
    origin  = np.array(img.GetOrigin(),   dtype=np.float64)
    if verbose:
        Z, Y, X = ct_vol.shape
        print(f"    Shape (Z,Y,X): ({Z},{Y},{X})  "
              f"spacing={spacing.round(3)} mm  "
              f"HU=[{ct_vol.min():.0f}, {ct_vol.max():.0f}]")
    return ct_vol, spacing, origin


def _load_dicom(dcm_path: Path) -> np.ndarray:
    """Load DICOM → float32 [0,1]."""
    import pydicom
    ds  = pydicom.dcmread(str(dcm_path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    mx  = arr.max()
    return arr / mx if mx > 0 else arr


def _load_jpeg(jpeg_path: Path) -> Tuple[np.ndarray, int, int]:
    """Load JPEG → (float32 [0,1], w, h)."""
    img = cv2.imread(str(jpeg_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {jpeg_path}")
    h, w = img.shape
    return img.astype(np.float32) / 255.0, w, h


def _load_png(png_path: Path) -> np.ndarray:
    """Load PNG → float32 [0,1]."""
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {png_path}")
    return img.astype(np.float32) / 255.0


def _load_labelme_json(json_path: Path) -> Dict[str, np.ndarray]:
    """Load labelme JSON → {label: (2,) [x, y] pixel coords}."""
    with open(json_path) as f:
        d = json.load(f)
    result: Dict[str, np.ndarray] = {}
    for shape in d.get('shapes', []):
        label = shape['label'].strip()
        pts   = shape['points']
        if pts:
            result[label] = np.array(pts[0], dtype=np.float64)
    return result


def _load_shared_lm_json(json_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load shared landmarks JSON → {frame_key: {label: (2,) [u,v]}}."""
    with open(json_path) as f:
        raw = json.load(f)
    return {
        fk: {lbl: np.array(coords, dtype=np.float64) for lbl, coords in lm.items()}
        for fk, lm in raw.items()
    }


# ============================================================================
# Pose initialisation helpers
# ============================================================================

def _anatomy_pose(lm_3d: Dict[str, np.ndarray],
                  sid_mm: float,
                  azimuth_deg: float = 0.0,
                  elevation_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an anatomy-centred (R, t) pose.

    azimuth_deg   : 0° = AP (posterior→anterior), 90° = lateral
    elevation_deg : tilt up/down
    """
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
    z_cam    = cent_xzy - src_xzy
    z_cam   /= np.linalg.norm(z_cam)
    world_up = np.array([0., 0., -1.])
    if abs(z_cam.dot(world_up)) > 0.9:
        world_up = np.array([1., 0., 0.])
    x_cam  = np.cross(world_up, z_cam);  x_cam /= np.linalg.norm(x_cam)
    y_cam  = np.cross(z_cam, x_cam)

    R = np.stack([x_cam, y_cam, z_cam], axis=0)
    t = -R @ src_xzy
    return R.astype(np.float64), t.astype(np.float64)


def _epnp(pts3d: np.ndarray, pts2d: np.ndarray,
          K: np.ndarray,
          n_pts: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    EPnP (or SQPNP for 3 pts).  pts3d is in world XYZ; xzy permutation applied here.
    """
    n_pts = n_pts or len(pts3d)
    flag  = cv2.SOLVEPNP_SQPNP if n_pts == 3 else cv2.SOLVEPNP_EPNP
    _, rvec, tvec = cv2.solvePnP(
        xzy(pts3d).astype(np.float64),
        pts2d.astype(np.float64),
        K, np.zeros(4),
        flags=flag,
    )
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.flatten().astype(np.float64)


def _reproj_err(pts3d: np.ndarray, pts2d: np.ndarray,
                R: np.ndarray, t: np.ndarray,
                fx: float, fy: float, cx: float, cy: float) -> float:
    P   = (R @ xzy(pts3d).T).T + t
    u   = fx * P[:, 0] / P[:, 2] + cx
    v   = fy * P[:, 1] / P[:, 2] + cy
    return float(np.sqrt(((np.stack([u, v], 1) - pts2d) ** 2).sum(1)).mean())


# ============================================================================
# Frame enumerators
# ============================================================================

def _enum_dicom_frames(directory: Path) -> Dict[str, Path]:
    """
    Enumerate DICOM files in a directory.
    Returns {instance_str: path} sorted by InstanceNumber.
    Falls back to alphabetical if InstanceNumber is not present.
    """
    import pydicom

    result: Dict[int, Path] = {}
    fallback: List[Path] = []

    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        try:
            ds   = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            inst = int(getattr(ds, 'InstanceNumber', -1))
            if inst > 0:
                result[inst] = p
            else:
                fallback.append(p)
        except Exception:
            fallback.append(p)

    if result:
        return {str(k): v for k, v in sorted(result.items())}
    # fallback: use filename as key
    return {p.stem: p for p in sorted(fallback)}


def _enum_jpeg_frames(directory: Path) -> Dict[str, Path]:
    """Enumerate JPEG frames.  Key = stem (e.g. 'a', 'b')."""
    return {
        p.stem: p
        for p in sorted(directory.glob('*.jpg'))
    }


def _enum_png_frames(directory: Path, view_tag: str) -> Dict[str, Path]:
    """
    Enumerate PNG frames.
    Key = '<view_tag>_<NNN>' (e.g. 'ap_002', 'lat_000').
    For view_tag='' key = stem.
    """
    result: Dict[str, Path] = {}
    for p in sorted(directory.glob('frame_*_z000.png')):
        parts = p.stem.split('_')
        if len(parts) >= 2:
            num = parts[1]
            key = f"{view_tag}_{num}" if view_tag else p.stem
            result[key] = p
    return result


# ============================================================================
# Main loader
# ============================================================================

class PatientLoader:
    """
    Generic loader for any patient following the canonical folder layout.

    Usage::
        loader = PatientLoader(cfg)
        spec   = loader.load()                    # default frames
        spec   = loader.load(frames=['ap_002'])   # specific frames
        spec   = loader.load(n_frames=4)          # subsample N frames

    The returned DeepFluoroSpecimen is identical in structure to what the
    existing run_*.py scripts expect.
    """

    def __init__(self, cfg: PatientConfig, verbose: bool = True):
        self.cfg     = cfg
        self.verbose = verbose

    # ── private ──────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _load_lm_2d_all(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Load all 2D landmarks depending on format."""
        fmt = self.cfg.lm_2d_format
        if fmt == 'none':
            return {}
        if fmt == 'shared_json':
            return _load_shared_lm_json(self.cfg.lm_2d_json)
        # 'labelme' is loaded per-frame in _collect_frames
        return {}

    def _collect_frames(self,
                        lm_2d_all: Dict[str, Dict[str, np.ndarray]]
                        ) -> Dict[str, dict]:
        """
        Collect all frames from all xray_sources.

        Returns a dict keyed by proj_key with fields:
            path, view_tag, img_format, lm_2d
        """
        frames_info: Dict[str, dict] = {}
        fmt = self.cfg.xray_format

        for src_dir, view_tag in self.cfg.xray_sources:
            if not src_dir.exists():
                self._log(f"  [WARN] X-ray directory not found: {src_dir}")
                continue

            if fmt == 'dicom':
                enum = _enum_dicom_frames(src_dir)
                for key, path in enum.items():
                    proj_key = f"{view_tag}_{key}" if view_tag else key
                    frames_info[proj_key] = dict(
                        path=path, view_tag=view_tag,
                        img_format='dicom', lm_2d={},
                    )

            elif fmt == 'jpeg':
                enum = _enum_jpeg_frames(src_dir)
                for key, path in enum.items():
                    proj_key = f"{view_tag}_{key}" if view_tag else key
                    # labelme: JSON alongside image
                    lm_2d: Dict[str, np.ndarray] = {}
                    if self.cfg.lm_2d_format == 'labelme':
                        json_path = path.with_suffix('.json')
                        if json_path.exists():
                            lm_2d = _load_labelme_json(json_path)
                    frames_info[proj_key] = dict(
                        path=path, view_tag=view_tag,
                        img_format='jpeg', lm_2d=lm_2d,
                    )

            elif fmt == 'png':
                enum = _enum_png_frames(src_dir, view_tag)
                for proj_key, path in enum.items():
                    # shared_json: match by frame index
                    lm_2d = {}
                    if self.cfg.lm_2d_format == 'shared_json' and lm_2d_all:
                        frame_idx = proj_key.split('_', 1)[1] if '_' in proj_key else proj_key
                        json_key  = f"frame_{frame_idx}_z00"
                        lm_2d     = lm_2d_all.get(json_key, {})
                    frames_info[proj_key] = dict(
                        path=path, view_tag=view_tag,
                        img_format='png', lm_2d=lm_2d,
                    )

        return frames_info

    def _select_frames(self,
                       frames_info: Dict[str, dict],
                       requested: Optional[List[str]],
                       n_frames: Optional[int]) -> List[str]:
        """
        Return an ordered list of proj_keys to process.

        Priority: explicit `requested` list > subsample to `n_frames` > all.
        """
        all_keys = list(frames_info.keys())

        if requested is not None:
            missing = [k for k in requested if k not in frames_info]
            if missing:
                self._log(f"  [WARN] Requested frames not found: {missing}")
            return [k for k in requested if k in frames_info]

        n = n_frames if n_frames is not None else self.cfg.default_n_frames
        if n is None or n >= len(all_keys):
            return all_keys

        # Evenly-spaced subsample
        step = max(1, len(all_keys) // n)
        return all_keys[::step][:n]

    def _init_pose(self,
                   lm_3d: Dict[str, np.ndarray],
                   lm_2d: Dict[str, np.ndarray],
                   cam: CameraConfig,
                   w: int, h: int,
                   view_tag: str,
                   frame_key: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute initial (R, t) for one frame.
        Returns (R, t, reproj_err_px).
        """
        fx, fy, cx, cy = cam.for_frame(w, h)
        K = cam.K(w, h)

        common = [l for l in sorted(lm_2d.keys()) if l in lm_3d]
        R, t   = None, None
        rerr   = 0.0

        if len(common) >= 3:
            pts3d = np.array([lm_3d[l] for l in common])
            pts2d = np.array([lm_2d[l] for l in common])
            try:
                R, t = _epnp(pts3d, pts2d, K, n_pts=len(common))
                rerr = _reproj_err(pts3d, pts2d, R, t, fx, fy, cx, cy)
                if self.verbose:
                    pix = cam.pix_mm_for_frame(w, h)
                    self._log(f"    EPnP  n={len(common)}  "
                              f"reproj={rerr:.2f}px  ({rerr*pix:.2f}mm)")
            except Exception as e:
                self._log(f"    EPnP failed ({e}) → anatomy pose")
                R, t = None, None

        if R is None:
            # Pick canonical azimuth based on view_tag
            view_idx = [t for _, t in self.cfg.xray_sources].index(view_tag) \
                       if view_tag in [t for _, t in self.cfg.xray_sources] else 0
            az = self.cfg.canonical_azimuths[view_idx % len(self.cfg.canonical_azimuths)]
            R, t = _anatomy_pose(lm_3d, cam.sid_mm, azimuth_deg=az)
            rerr = 0.0
            self._log(f"    Anatomy pose  az={az:.0f}°")

        return R, t, rerr

    # ── public ───────────────────────────────────────────────────────────────

    def load(self,
             frames: Optional[List[str]] = None,
             n_frames: Optional[int] = None,
             ) -> DeepFluoroSpecimen:
        """
        Load this patient as a DeepFluoroSpecimen.

        Args:
            frames   : explicit list of proj_keys to include.
                       For DICOM: instance number strings ('1', '50', ...)
                       For JPEG: file stems ('a', 'b', ...)
                       For PNG:  view-prefixed keys ('ap_002', 'lat_000', ...)
                       None = use default_n_frames or all frames.
            n_frames : override the number of evenly-spaced frames to sample.
                       Only used when `frames` is None.

        Returns:
            DeepFluoroSpecimen with PatientProjection objects.
        """
        pid = self.cfg.patient_id
        self._log(f"[{pid}] Loading patient ...")

        ct_vol, spacing, origin = _load_ct(self.cfg.ct_nrrd, self.verbose)
        lm_3d   = _load_mrk_json_3d(self.cfg.lm_3d_json)
        self._log(f"  3D landmarks ({len(lm_3d)}): {sorted(lm_3d.keys())}")

        lm_2d_all  = self._load_lm_2d_all()
        frames_info = self._collect_frames(lm_2d_all)
        self._log(f"  Frames discovered: {len(frames_info)}")

        selected = self._select_frames(frames_info, frames, n_frames)
        self._log(f"  Selected ({len(selected)}): {selected}")

        cam = self.cfg.camera
        projections: List[PatientProjection] = []

        for idx, proj_key in enumerate(selected):
            info = frames_info[proj_key]
            path, view_tag, fmt, lm_2d = (
                info['path'], info['view_tag'],
                info['img_format'], info['lm_2d'],
            )

            # Load image
            try:
                if fmt == 'dicom':
                    img = _load_dicom(path)
                    w = h = img.shape[0]
                elif fmt == 'jpeg':
                    img, w, h = _load_jpeg(path)
                elif fmt == 'png':
                    img = _load_png(path)
                    h, w = img.shape if img.ndim == 2 else img.shape[:2]
                else:
                    raise ValueError(f"Unknown image format: {fmt}")
            except Exception as e:
                self._log(f"  [SKIP] {proj_key}: {e}")
                continue

            self._log(f"  [{idx}] {proj_key}  {w}×{h}  "
                      f"2D lm={sorted(lm_2d.keys()) if lm_2d else '—'}")

            R, t, rerr = self._init_pose(lm_3d, lm_2d, cam, w, h, view_tag, proj_key)
            fx, fy, cx, cy = cam.for_frame(w, h)

            proj = PatientProjection(
                fx=fx, fy=fy, cx=cx, cy=cy,
                img_w           = w,
                img_h           = h,
                specimen_id     = pid,
                proj_index      = idx,
                proj_key        = proj_key,
                image_raw       = img,
                image_display   = img,
                R_proj          = R,
                t_proj          = t,
                gt_landmarks_2d = lm_2d,
                rot_180_for_up  = False,
                reproj_error_px = rerr,
            )
            projections.append(proj)

        if not projections:
            raise RuntimeError(
                f"[{pid}] No projections loaded — check xray_sources paths."
            )

        spec = DeepFluoroSpecimen(
            specimen_id  = pid,
            ct_volume    = ct_vol,
            ct_spacing   = spacing,
            ct_origin    = origin,
            landmarks_3d = lm_3d,
            projections  = projections,
        )
        self._log(f"\n  Loaded: {spec}")
        return spec


# ============================================================================
# Convenience functions
# ============================================================================

def load_patient(patient_id: str,
                 frames: Optional[List[str]] = None,
                 n_frames: Optional[int] = None,
                 verbose: bool = True) -> DeepFluoroSpecimen:
    """
    Load a single patient by ID.

    Args:
        patient_id : key in PATIENT_REGISTRY (e.g. 'ramulamma', 'arjun', 'swaroopa')
        frames     : explicit list of frame keys (None = default)
        n_frames   : how many frames to subsample (None = all or config default)
        verbose    : print progress

    Returns:
        DeepFluoroSpecimen ready for the CMA-ES registration pipeline.
    """
    if patient_id not in PATIENT_REGISTRY:
        raise KeyError(
            f"Unknown patient '{patient_id}'. "
            f"Available: {sorted(PATIENT_REGISTRY.keys())}"
        )
    cfg    = PATIENT_REGISTRY[patient_id]
    loader = PatientLoader(cfg, verbose=verbose)
    return loader.load(frames=frames, n_frames=n_frames)


def load_patients(patient_ids: Optional[List[str]] = None,
                  verbose: bool = True) -> Dict[str, DeepFluoroSpecimen]:
    """
    Load multiple patients.

    Args:
        patient_ids : list of IDs to load. None = load all registered patients.
        verbose     : print progress.

    Returns:
        Dict {patient_id: DeepFluoroSpecimen}
    """
    ids  = patient_ids if patient_ids is not None else sorted(PATIENT_REGISTRY.keys())
    out: Dict[str, DeepFluoroSpecimen] = {}
    for pid in ids:
        try:
            out[pid] = load_patient(pid, verbose=verbose)
        except Exception as e:
            print(f"[ERROR] Failed to load patient '{pid}': {e}")
    return out


def list_patients() -> None:
    """Print a summary of all registered patients."""
    print(f"{'ID':<15} {'CT exists':<12} {'DICOM/img dir':<50} {'2D lm format'}")
    print("-" * 95)
    for pid, cfg in sorted(PATIENT_REGISTRY.items()):
        ct_ok  = "✓" if cfg.ct_nrrd.exists() else "✗ MISSING"
        dirs   = ", ".join(str(d) for d, _ in cfg.xray_sources)
        print(f"{pid:<15} {ct_ok:<12} {dirs:<50} {cfg.lm_2d_format}")
