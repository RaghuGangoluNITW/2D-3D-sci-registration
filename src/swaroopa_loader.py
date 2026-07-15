"""
swaroopa_loader.py — Real Patient Loader for Swaroopa
=====================================================
Loads Swaroopa's pre-op CT (NRRD) + intra-op C-arm PNG images with
2D landmark annotations and wraps them into the DeepFluoroSpecimen /
DeepFluoroProjection interface used by the existing optimiser.

Data layout:
  CT (pre-op NRRD, exported from Slicer):
    data/swaroopa-.../swaroopa/PREOP CENTROID/
        TempWrite2 Unnamed Series/2 Unnamed Series.nrrd

  3D landmarks (vertebral centroids, Slicer mrk.json):
    data/swaroopa-.../swaroopa/PREOP CENTROID/CENTROIDS.mrk.json
      Labels: L1, L2, L3, L4, L5 (RAS mm)

  Intra-op X-rays (PNG, 1024×1024, uint8) — split by view:
    data/swaroopa_labelled/ap/frame_NNN_z000.png      (25 AP frames)
    data/swaroopa_labelled/lateral/frame_NNN_z000.png (13 lateral frames)

  2D landmark annotations (shared JSON at top level):
    data/swaroopa_labelled/landmarks_output.json
    Format: { "frame_NNN_z00": {"L1": [u, v], ...}, ... }
    Key: frame_NNN_z00  (note: _z00 suffix, PNG filename uses _z000)

Camera model (Ziehm Vision FD):
  SID     = 1050 mm  (DICOM DistanceSourceToDetector)
  pixels  = 1024 × 1024
  Pixel spacing = 0.288 mm → Fx = Fy = 1050 / 0.288 ≈ 3646 px  (from Ziehm private DICOM tags 0019,1014)
  Cx = Cy = (1024 - 1) / 2 = 511.5

EPnP initialization:
  For frames with ≥ 3 annotated 2D landmarks → cv2.solvePnP (EPNP/SQPNP).
  Fallback: azimuth=0° (AP) or azimuth=90° (lateral) anatomy pose.
  proj_key encodes view: 'ap_NNN' or 'lat_NNN'.

PDE evaluation:
  2D annotations available → PDE (mm) computed as reprojection error
  × pixel_spacing_mm after optimisation.
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

SWARO_SID_MM:   float = 1050.0          # DICOM DistanceSourceToDetector (mm)
SWARO_PIX_MM:   float = 0.288           # true pixel spacing from Ziehm DICOM (0019,1014) / (0019,1213) (mm/px)
SWARO_IMG_SIZE: int   = 1024            # detector pixels (square)

SWARO_FX: float = SWARO_SID_MM / SWARO_PIX_MM    # ≈ 3646 px
SWARO_FY: float = SWARO_SID_MM / SWARO_PIX_MM
SWARO_CX: float = (SWARO_IMG_SIZE - 1) / 2.0     # 511.5
SWARO_CY: float = (SWARO_IMG_SIZE - 1) / 2.0

SWARO_K: np.ndarray = np.array([
    [SWARO_FX, 0.,       SWARO_CX],
    [0.,       SWARO_FY, SWARO_CY],
    [0.,       0.,       1.      ],
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE = Path('/home/supermicro/Documents/2D_3D_Raghu/data')

CT_NRRD = _BASE / 'swaroopa_labelled' / 'preop_ct_new' / 'swaroopa_ct_new.nrrd'

LM_3D_JSON = _BASE / 'swaroopa_labelled' / 'preop_ct_new' / 'centroid.mrk.json'

XRAY_DIR        = _BASE / 'swaroopa_labelled'
XRAY_DIR_AP     = _BASE / 'swaroopa_labelled' / 'ap'
XRAY_DIR_LAT    = _BASE / 'swaroopa_labelled' / 'lateral'
LM_2D_JSON      = _BASE / 'swaroopa_labelled' / 'landmarks_output.json'

# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def project_world_swaro(pts3d_world: np.ndarray,
                        R_proj: np.ndarray,
                        t_proj: np.ndarray) -> np.ndarray:
    """Project 3D world XYZ → 2D pixel using Swaroopa intrinsics."""
    pts = np.atleast_2d(pts3d_world)
    P_cam = (R_proj @ xzy(pts).T).T + t_proj
    u = SWARO_FX * P_cam[:, 0] / P_cam[:, 2] + SWARO_CX
    v = SWARO_FY * P_cam[:, 1] / P_cam[:, 2] + SWARO_CY
    return np.stack([u, v], axis=1)



# ---------------------------------------------------------------------------
# SwaroProjection
# ---------------------------------------------------------------------------

class SwaroProjection(DeepFluoroProjection):
    """Swaroopa projection with correct camera intrinsics."""

    def project(self, pts3d_world: np.ndarray) -> np.ndarray:
        return project_world_swaro(pts3d_world, self.R_proj, self.t_proj)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def compute_pde_swaro(proj: SwaroProjection,
                      R_cand: np.ndarray,
                      t_cand: np.ndarray,
                      pts3d: np.ndarray,
                      lm_names: List[str]) -> Dict[str, float]:
    """PDE (mm) between candidate pose and GT 2D landmarks (Swaroopa frame)."""
    uv_pred = project_world_swaro(pts3d, R_cand, t_cand)
    pde: Dict[str, float] = {}
    for i, name in enumerate(lm_names):
        if name in proj.gt_landmarks_2d:
            gt = proj.gt_landmarks_2d[name]
            if 0 < gt[0] < SWARO_IMG_SIZE and 0 < gt[1] < SWARO_IMG_SIZE:
                err_px = float(np.linalg.norm(uv_pred[i] - gt))
                pde[name] = err_px * SWARO_PIX_MM
    return pde


def mean_pde_swaro(proj, R, t, pts3d, lm_names) -> float:
    d = compute_pde_swaro(proj, R, t, pts3d, lm_names)
    return float(np.mean(list(d.values()))) if d else float('inf')


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _load_mrk_json_3d(json_path: Path) -> Dict[str, np.ndarray]:
    """Load Slicer mrk.json → {label: (3,) RAS mm}."""
    with open(json_path) as f:
        d = json.load(f)
    cps = d['markups'][0]['controlPoints']
    return {
        cp['label'].strip(): np.array(cp['position'], dtype=np.float64)
        for cp in cps
    }


def _load_landmarks_2d(json_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load landmarks_output.json → {frame_key: {label: (2,) [u, v]}}.

    JSON keys are "frame_NNN_z00"; PNG filenames are "frame_NNN_z000.png".
    The index NNN is used to match them.
    """
    with open(json_path) as f:
        raw = json.load(f)
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for frame_key, lm_dict in raw.items():
        result[frame_key] = {
            label: np.array(coords, dtype=np.float64)
            for label, coords in lm_dict.items()
        }
    return result


def _load_png(png_path: Path) -> np.ndarray:
    """Load PNG → float32 [0, 1], shape (1024, 1024)."""
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {png_path}")
    return img.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# EPnP + anatomy pose helpers
# ---------------------------------------------------------------------------

def _solve_pnp_swaro(pts3d_world: np.ndarray,
                     pts2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """EPnP (≥4 pts) or SQPNP (3 pts) using Swaroopa camera intrinsics."""
    pts_xzy = xzy(pts3d_world).astype(np.float64)
    n = len(pts_xzy)
    flag = cv2.SOLVEPNP_SQPNP if n == 3 else cv2.SOLVEPNP_EPNP
    _, rvec, tvec = cv2.solvePnP(
        pts_xzy, pts2d.astype(np.float64),
        SWARO_K, np.zeros(4),
        flags=flag,
    )
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.flatten().astype(np.float64)


def _reproj_error_swaro(pts3d: np.ndarray, pts2d: np.ndarray,
                        R: np.ndarray, t: np.ndarray) -> float:
    """Mean reprojection error in pixels."""
    uv_pred = project_world_swaro(pts3d, R, t)
    return float(np.sqrt(((uv_pred - pts2d) ** 2).sum(axis=1)).mean())


def _anatomy_pose(lm_3d: Dict[str, np.ndarray],
                  azimuth_deg: float = 0.0,
                  elevation_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Anatomy-centred (R, t) pose. azimuth_deg=0 → AP (posterior→anterior)."""
    centroid = np.mean(np.array(list(lm_3d.values()), dtype=np.float64), axis=0)
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    src_dir = np.array([
        np.sin(az) * np.cos(el),
        np.cos(az) * np.cos(el),
        np.sin(el),
    ], dtype=np.float64)
    src_world = centroid + SWARO_SID_MM * src_dir

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
# SwaroLoader
# ---------------------------------------------------------------------------

class SwaroLoader:
    """
    Load Swaroopa pre-op CT (NRRD) + intra-op PNG X-rays with 2D labels.

    Returns a DeepFluoroSpecimen with SwaroProjection objects.
    EPnP initialisation is used for frames with ≥3 annotated landmarks.

    Usage::
        loader = SwaroLoader()
        spec   = loader.load()                      # all annotated frames
        spec   = loader.load(frames=['000','002'])  # specific frame indices
    """

    def __init__(self,
                 ct_nrrd:      Path = CT_NRRD,
                 lm_3d_json:   Path = LM_3D_JSON,
                 xray_dir_ap:  Path = XRAY_DIR_AP,
                 xray_dir_lat: Path = XRAY_DIR_LAT,
                 lm_2d_json:   Path = LM_2D_JSON):
        self.ct_nrrd      = Path(ct_nrrd)
        self.lm_3d_json   = Path(lm_3d_json)
        self.xray_dir_ap  = Path(xray_dir_ap)
        self.xray_dir_lat = Path(xray_dir_lat)
        self.lm_2d_json   = Path(lm_2d_json)

    # ── private ──────────────────────────────────────────────────────────────

    def _load_ct(self, verbose: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.ct_nrrd.exists():
            raise FileNotFoundError(
                f"CT NRRD not found: {self.ct_nrrd}\n"
                f"Expected: data/swaroopa_labelled/preop_ct_new/swaroopa_ct_new.nrrd"
            )
        if verbose:
            print(f"  Loading CT (NRRD): {self.ct_nrrd}")
        img = sitk.ReadImage(str(self.ct_nrrd))
        ct_vol  = sitk.GetArrayFromImage(img).astype(np.float32)
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

    # ── public ───────────────────────────────────────────────────────────────

    def load(self,
             frames: Optional[List[str]] = None,
             verbose: bool = True) -> DeepFluoroSpecimen:
        """
        Load Swaroopa dataset as a DeepFluoroSpecimen.

        Scans both ap/ and lateral/ subfolders.  proj_key is 'ap_NNN' or
        'lat_NNN' so AP and lateral frames are distinguishable in results.

        Args:
            frames  : list of proj_keys to register, e.g. ['ap_001','lat_000'].
                      None = all frames found in both subfolders.
            verbose : print progress.

        Returns:
            DeepFluoroSpecimen with SwaroProjection objects.
        """
        if verbose:
            print("[Swaroopa] Loading specimen ...")

        ct_vol, spacing, origin = self._load_ct(verbose)
        lm_3d = self._load_3d_landmarks(verbose)

        # ── Landmarks are kept at their true Slicer LPS positions ─────────────
        # Previously an isocenter-centroid shift was applied here, but that
        # was an artificial displacement that made centroids project to wrong
        # anatomy positions in the DRR.  The proper LPS → TorchIO-RAS
        # coordinate correction (notebook step 3.3) is applied at projection
        # time inside DiffDRRGenerator.project_pts, using the TorchIO affine
        # and the CT origin from SimpleITK.
        if verbose and lm_3d:
            lm_centroid = np.mean(np.array(list(lm_3d.values()), dtype=np.float64), axis=0)
            print(f"  Landmark centroid (LPS mm): {lm_centroid.round(1)}")
            print(f"  CT origin        (LPS mm): {origin.round(1)}")

        lm_2d_all = _load_landmarks_2d(self.lm_2d_json)

        # Collect PNGs from both subfolders: proj_key → (png_path, view_tag)
        # view_tag: 'ap' or 'lat'; azimuth for anatomy fallback: 0° / 90°
        all_views: Dict[str, Tuple[Path, str]] = {}
        for view_tag, subdir, az in [
            ('ap',  self.xray_dir_ap,  0.0),
            ('lat', self.xray_dir_lat, 90.0),
        ]:
            if not subdir.exists():
                continue
            for p in sorted(subdir.glob('frame_*_z000.png')):
                parts = p.stem.split('_')
                if len(parts) >= 2:
                    proj_key = f"{view_tag}_{parts[1]}"
                    all_views[proj_key] = (p, view_tag)

        if verbose:
            n_ap  = sum(1 for _, (_, v) in all_views.items() if v == 'ap')
            n_lat = sum(1 for _, (_, v) in all_views.items() if v == 'lat')
            print(f"  PNG frames found: {len(all_views)}  (AP={n_ap}, lateral={n_lat})")
            print(f"  Annotated frames in JSON: {len(lm_2d_all)}")

        selected_keys = list(frames) if frames is not None else sorted(all_views.keys())

        projections: List[SwaroProjection] = []
        for idx, proj_key in enumerate(selected_keys):
            if proj_key not in all_views:
                if verbose:
                    print(f"  [SKIP] {proj_key}: PNG not found")
                continue

            png_path, view_tag = all_views[proj_key]
            frame_idx = proj_key.split('_', 1)[1]   # e.g. 'ap_001' → '001'

            # JSON key uses frame_NNN_z00 (no view prefix)
            json_key = f"frame_{frame_idx}_z00"
            lm_2d = lm_2d_all.get(json_key, {})

            image_raw = _load_png(png_path)
            az_fallback = 0.0 if view_tag == 'ap' else 90.0

            # EPnP initialisation
            common_labels = [l for l in sorted(lm_2d.keys()) if l in lm_3d]
            R_init, t_init = None, None
            reproj_err = 0.0

            if len(common_labels) >= 3:
                pts3d = np.array([lm_3d[l] for l in common_labels])
                pts2d = np.array([lm_2d[l] for l in common_labels])
                try:
                    R_init, t_init = _solve_pnp_swaro(pts3d, pts2d)
                    reproj_err = _reproj_error_swaro(pts3d, pts2d, R_init, t_init)
                    if verbose:
                        pde_mm = reproj_err * SWARO_PIX_MM
                        print(f"  [{view_tag}] frame {frame_idx}: {len(common_labels)} lm  "
                              f"EPnP reproj={reproj_err:.2f}px ({pde_mm:.2f}mm)")
                except Exception as e:
                    if verbose:
                        print(f"  [{view_tag}] frame {frame_idx}: EPnP failed → anatomy pose")
                    R_init, t_init = None, None

            if R_init is None:
                R_init, t_init = _anatomy_pose(lm_3d, azimuth_deg=az_fallback)
                reproj_err = 0.0
                if verbose:
                    print(f"  [{view_tag}] frame {frame_idx}: anatomy pose "
                          f"(az={az_fallback:.0f}°)  labels={sorted(lm_2d.keys())}")

            proj = SwaroProjection(
                specimen_id     = 'swaroopa',
                proj_index      = idx,
                proj_key        = proj_key,
                image_raw       = image_raw,
                image_display   = image_raw,
                R_proj          = R_init,
                t_proj          = t_init,
                gt_landmarks_2d = lm_2d,
                rot_180_for_up  = False,
                reproj_error_px = reproj_err,
            )
            projections.append(proj)

        if not projections:
            raise RuntimeError("No projections loaded — check PNG/JSON paths.")

        spec = DeepFluoroSpecimen(
            specimen_id  = 'swaroopa',
            ct_volume    = ct_vol,
            ct_spacing   = spacing,
            ct_origin    = origin,
            landmarks_3d = lm_3d,
            projections  = projections,
        )

        if verbose:
            print(f"\n  Loaded: {spec}")

        return spec
