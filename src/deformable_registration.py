"""
deformable_registration.py
==========================
Per-vertebra deformable refinement for 2D/3D spine registration.

Two strategies are provided:

  1. compute_deformation()        — fast closed-form backprojection (original).
     After a global rigid pose (R, t) is found, each vertebra's 3D position
     is corrected via backprojection of its observed 2D centroid.

  2. compute_deformation_cmaes()  — full per-vertebra CMA-ES (Ketcha 2017b).
     For each labeled vertebra, a small independent CMA-ES is run starting
     from the global rigid pose, optimising DRR similarity in a local ROI
     around that vertebra.  This finds the true best pose per vertebra rather
     than inferring it geometrically — critical for patients with large prone/
     supine changes or scoliosis (>10° curvature change).

     Expected benefit (from Ketcha 2017b):
       rigid LevelCheck: max PDE 22.4 mm under strong deformation
       per-vertebra CMA-ES: max PDE 3.9 mm  (same deformation)

Coordinate convention (must match generic_patient_loader.py)
============================================================
Forward projection:
    P_cam = R @ xzy(p_world) + t
    u = FX * P_cam[0] / P_cam[2] + CX
    v = FY * P_cam[1] / P_cam[2] + CY
where xzy(p) swaps indices 1 and 2:  [x,y,z] → [x,z,y]
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_DEFORM_MM: float = 15.0   # maximum plausible per-vertebra deformation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _xzy(v: np.ndarray) -> np.ndarray:
    """Swap indices 1 and 2 of a 1-D or (N,3) array."""
    v = np.atleast_2d(v)
    return v[:, [0, 2, 1]].squeeze()


def _camera_centre(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Camera centre in world coordinates.
    Satisfies R @ xzy(C) + t = 0
    → C = xzy(-R^T @ t)
    """
    t_flat = np.asarray(t).ravel()
    return _xzy(-R.T @ t_flat)


def _backproject_ray(R: np.ndarray, t: np.ndarray,
                     u: float, v: float,
                     fx: float, fy: float,
                     cx: float, cy: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (camera_centre_world, unit_ray_direction_world) for pixel (u, v).
    """
    C = _camera_centre(R, t)
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    d_world = _xzy(R.T @ d_cam)
    d_world = d_world / np.linalg.norm(d_world)
    return C, d_world


def _closest_point_on_ray(C: np.ndarray, d_hat: np.ndarray,
                           p: np.ndarray) -> np.ndarray:
    """
    Closest point on ray r(λ)=C+λ·d_hat to point p, with λ ≥ 0.
    """
    lam = float(np.dot(p - C, d_hat))
    lam = max(0.0, lam)
    return C + lam * d_hat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class VertebralDeformation:
    """Holds the deformation result for one vertebra in one X-ray set."""
    def __init__(self,
                 label: str,
                 p3d_ct: np.ndarray,       # original CT centroid (mm, world)
                 p3d_intraop: np.ndarray,   # corrected intraop position (mm)
                 delta_mm: np.ndarray,      # deformation vector (mm)
                 clamped: bool):            # True if constrained to MAX_DEFORM_MM
        self.label        = label
        self.p3d_ct       = p3d_ct
        self.p3d_intraop  = p3d_intraop
        self.delta_mm     = delta_mm
        self.magnitude_mm = float(np.linalg.norm(delta_mm))
        self.clamped      = clamped

    def to_dict(self) -> dict:
        return dict(
            label         = self.label,
            magnitude_mm  = round(self.magnitude_mm, 4),
            delta_mm      = [round(float(v), 4) for v in self.delta_mm],
            p3d_ct        = [round(float(v), 4) for v in self.p3d_ct],
            p3d_intraop   = [round(float(v), 4) for v in self.p3d_intraop],
            clamped       = self.clamped,
        )


def compute_deformation(
    R: np.ndarray,
    t: np.ndarray,
    pts3d: np.ndarray,
    lm_names: List[str],
    gt_2d_pixels: Dict[str, np.ndarray],
    fx: float, fy: float,
    cx: float, cy: float,
    pix_mm: float = 0.2,
    max_deform_mm: float = MAX_DEFORM_MM,
) -> Dict[str, VertebralDeformation]:
    """
    Compute per-vertebra deformation from a rigid camera pose.

    Parameters
    ----------
    R           : (3,3) rotation from EPnP/CMA-ES
    t           : (3,) or (3,1) translation from EPnP/CMA-ES
    pts3d       : (N,3) CT centroid positions in world (LPS mm)
    lm_names    : list of N landmark names
    gt_2d_pixels: {name: (u,v)} observed 2D centroids in pixels
    fx,fy,cx,cy : camera intrinsics (pixels)
    pix_mm      : pixel spacing (mm/px) — used for PDE check only
    max_deform_mm: biomechanical clamp (mm)

    Returns
    -------
    dict {name: VertebralDeformation}  — one entry per labeled vertebra
    """
    results: Dict[str, VertebralDeformation] = {}

    for i, name in enumerate(lm_names):
        if name not in gt_2d_pixels:
            continue
        uv = gt_2d_pixels[name]
        u, v = float(uv[0]), float(uv[1])
        p3d  = np.asarray(pts3d[i], dtype=np.float64)

        C, d_hat = _backproject_ray(R, t, u, v, fx, fy, cx, cy)
        p3d_deformed = _closest_point_on_ray(C, d_hat, p3d)
        delta = p3d_deformed - p3d
        mag   = float(np.linalg.norm(delta))

        clamped = False
        if mag > max_deform_mm:
            delta     = delta * (max_deform_mm / mag)
            p3d_deformed = p3d + delta
            clamped   = True

        results[name] = VertebralDeformation(
            label       = name,
            p3d_ct      = p3d.copy(),
            p3d_intraop = p3d_deformed,
            delta_mm    = delta,
            clamped     = clamped,
        )

    return results


def deformable_pde(
    R: np.ndarray,
    t: np.ndarray,
    deformations: Dict[str, VertebralDeformation],
    all_pts3d: np.ndarray,
    all_lm_names: List[str],
    gt_2d_pixels: Dict[str, np.ndarray],
    fx: float, fy: float, cx: float, cy: float,
    pix_mm: float = 0.2,
) -> Dict[str, float]:
    """
    PDE after applying per-vertebra deformation corrections.

    For labeled vertebrae: the deformed 3D position is on the backprojection
    ray → reprojection is (near) 0 px → PDE ≈ 0 mm.
    For unlabeled vertebrae: linearly interpolate the deformation from the
    two nearest labeled neighbours (by vertebra index along the spine).
    """
    # Build deformed positions for all landmarks
    deformed_pts: Dict[str, np.ndarray] = {}
    labeled_indices: List[Tuple[int, str]] = []

    for i, name in enumerate(all_lm_names):
        if name in deformations:
            deformed_pts[name] = deformations[name].p3d_intraop
            labeled_indices.append((i, name))
        else:
            deformed_pts[name] = all_pts3d[i].copy()   # original, will interpolate

    # Interpolate deformation for unlabeled vertebrae
    if labeled_indices:
        for i, name in enumerate(all_lm_names):
            if name in deformations:
                continue  # already handled
            # find nearest labeled neighbours by index distance
            dists  = [(abs(i - li), li, ln, deformations[ln].delta_mm)
                      for li, ln in labeled_indices]
            dists.sort(key=lambda x: x[0])
            if len(dists) == 1:
                delta_interp = dists[0][3].copy()
            else:
                # weighted average of two nearest
                d0, _, _, v0 = dists[0]
                d1, _, _, v1 = dists[1]
                total = d0 + d1 + 1e-9
                w0 = d1 / total
                w1 = d0 / total
                delta_interp = w0 * v0 + w1 * v1
            deformed_pts[name] = all_pts3d[i] + delta_interp

    # Reproject deformed 3D positions and compute PDE
    pde_out: Dict[str, float] = {}
    for i, name in enumerate(all_lm_names):
        if name not in gt_2d_pixels:
            continue
        p = np.asarray(deformed_pts[name], dtype=np.float64)
        # Forward project through current rigid pose (apply xzy axis swap)
        p_xzy = p[[0, 2, 1]]                 # [x,y,z] → [x,z,y]
        P_cam = R @ p_xzy + np.asarray(t).ravel()
        if P_cam[2] <= 0:
            continue
        u_proj = fx * P_cam[0] / P_cam[2] + cx
        v_proj = fy * P_cam[1] / P_cam[2] + cy
        gt  = gt_2d_pixels[name]
        pde_out[name] = float(np.linalg.norm(
            np.array([u_proj, v_proj]) - gt)) * pix_mm

    return pde_out


def deformation_summary(deformations_per_set: Dict[str, Dict[str, VertebralDeformation]]
                         ) -> Dict[str, dict]:
    """
    Summarize deformation statistics across multiple X-ray sets for one patient.
    Returns {vertebra_label: {mean_mm, std_mm, max_mm, n}}
    """
    per_label: Dict[str, List[float]] = {}
    for sid, deforms in deformations_per_set.items():
        for name, vd in deforms.items():
            per_label.setdefault(name, []).append(vd.magnitude_mm)

    summary = {}
    for name, mags in per_label.items():
        summary[name] = dict(
            mean_mm = round(float(np.mean(mags)), 3),
            std_mm  = round(float(np.std(mags)), 3),
            max_mm  = round(float(np.max(mags)), 3),
            n       = len(mags),
        )
    return summary


def cross_view_consistency(
    deformations_ap:  Dict[str, VertebralDeformation],
    deformations_lat: Dict[str, VertebralDeformation],
) -> Dict[str, float]:
    """
    Compare the component of the deformation vectors that each view is
    sensitive to.

    AP view  is sensitive to left-right (X) and cranio-caudal (Z) motion.
    LAT view is sensitive to anterior-posterior (Y) and cranio-caudal (Z) motion.
    The cranio-caudal (Z) component should be consistent between AP and LAT.

    Returns {vertebra: abs_diff_Z_mm}
    """
    consistency: Dict[str, float] = {}
    common = set(deformations_ap.keys()) & set(deformations_lat.keys())
    for name in common:
        dz_ap  = deformations_ap[name].delta_mm[2]
        dz_lat = deformations_lat[name].delta_mm[2]
        consistency[name] = abs(float(dz_ap) - float(dz_lat))
    return consistency


# ---------------------------------------------------------------------------
# Per-vertebra CMA-ES deformation  (Ketcha 2017b)
# ---------------------------------------------------------------------------
def compute_deformation_cmaes(
    R_global: np.ndarray,
    t_global: np.ndarray,
    pts3d: np.ndarray,
    lm_names: List[str],
    gt_2d_pixels: Dict[str, np.ndarray],
    drr_generator,               # DeepFluoroDRR instance
    xray_image: np.ndarray,      # full-resolution X-ray (H, W) float32
    fx: float, fy: float,
    cx: float, cy: float,
    pix_mm: float = 0.2,
    max_deform_mm: float = MAX_DEFORM_MM,
    roi_px: int = 180,           # half-side of the ROI window in pixels
    render_size: int = 64,       # DRR render size for per-vertebra opt
    n_starts: int = 6,           # CMA-ES restarts per vertebra
    max_rot_deg: float = 5.0,    # search range ± rotation (degrees)
    max_trans_mm: float = 12.0,  # search range ± translation (mm)
    verbose: bool = False,
) -> Dict[str, VertebralDeformation]:
    """
    Per-vertebra CMA-ES deformation refinement (Ketcha 2017b).

    For each labeled vertebra:
      1. Project CT centroid using global pose → approximate pixel location.
      2. Crop an (roi_px × roi_px) window in the X-ray around that pixel.
      3. Run a small CMA-ES (6-DOF) starting from the global pose, computing
         DRR similarity ONLY inside the cropped ROI.  Search bounds are tight
         (±max_rot_deg, ±max_trans_mm) to stay near the global solution.
      4. The best local pose is used to backproject the vertebra centroid to
         its corrected intraop 3D position.

    Compared to the simple geometric backprojection, this finds the true
    best-fit pose for each vertebra independently, which handles large
    deformations (scoliosis, prone–supine curvature change) correctly.

    Parameters
    ----------
    drr_generator : DeepFluoroDRR instance with generate_from_extrinsic().
    xray_image    : full-res X-ray used to crop ROI targets.
    roi_px        : ROI half-width (pixels at native X-ray resolution).
    render_size   : DRR render resolution for per-vertebra optimisation.
    n_starts      : number of CMA-ES multi-starts per vertebra.
    max_rot_deg   : ±rotation search range (degrees).
    max_trans_mm  : ±translation search range (mm).

    Returns
    -------
    dict {name: VertebralDeformation}
    """
    # Lazy imports — avoids circular dependency at module load time
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(__file__))
    from optimizer import run_cmaes_single
    from deepfluoro_loader import perturb_extrinsic
    import cv2

    H, W = xray_image.shape[:2]

    results: Dict[str, VertebralDeformation] = {}

    for i, name in enumerate(lm_names):
        if name not in gt_2d_pixels:
            continue

        p3d   = np.asarray(pts3d[i], dtype=np.float64)
        uv_gt = gt_2d_pixels[name]
        u_gt, v_gt = float(uv_gt[0]), float(uv_gt[1])

        # ── Step 1: build ROI target from X-ray ──────────────────────────
        u0 = int(np.clip(u_gt - roi_px, 0, W - 1))
        u1 = int(np.clip(u_gt + roi_px, 0, W))
        v0 = int(np.clip(v_gt - roi_px, 0, H - 1))
        v1 = int(np.clip(v_gt + roi_px, 0, H))
        roi_xray = xray_image[v0:v1, u0:u1].astype(np.float32)

        # Fall back to geometric backprojection if ROI is too small
        def _backproj_fallback():
            C, d_hat = _backproject_ray(R_global, t_global,
                                        u_gt, v_gt, fx, fy, cx, cy)
            p3d_def = _closest_point_on_ray(C, d_hat, p3d)
            delta   = p3d_def - p3d
            mag     = float(np.linalg.norm(delta))
            clamped = mag > max_deform_mm
            if clamped:
                delta = delta * (max_deform_mm / mag)
                p3d_def = p3d + delta
            return VertebralDeformation(
                label=name, p3d_ct=p3d.copy(), p3d_intraop=p3d_def,
                delta_mm=delta, clamped=clamped)

        if roi_xray.size < 100:
            results[name] = _backproj_fallback()
            continue

        roi_tgt = cv2.resize(roi_xray, (render_size, render_size),
                             interpolation=cv2.INTER_AREA)

        # Compute approximate mm/px at render resolution
        render_pix_mm = pix_mm * (max(H, W) / render_size)
        # Scale factor: roi in render coords
        sc = render_size * 4 / max(H, W)
        du0, du1 = int(u0 * sc), int(u1 * sc)
        dv0, dv1 = int(v0 * sc), int(v1 * sc)

        # ── Step 2: per-vertebra cost function ───────────────────────────
        def _cost(x, _R=R_global, _t=t_global, _tgt=roi_tgt,
                  _du0=du0, _du1=du1, _dv0=dv0, _dv1=dv1,
                  _rpx=render_pix_mm, _rn=render_size * 4):
            Rc, tc = perturb_extrinsic(_R, _t, x[:3], x[3:])
            try:
                drr_full = drr_generator.generate_from_extrinsic(
                    Rc, tc, _rn, _rpx / 4, 40)
            except Exception:
                return 1.0
            drr_roi = drr_full[_dv0:_dv1, _du0:_du1]
            if drr_roi.size < 4:
                return 1.0
            drr_rs = cv2.resize(drr_roi.astype(np.float32),
                                (render_size, render_size),
                                interpolation=cv2.INTER_AREA)
            if np.count_nonzero(drr_rs > 0.01) / drr_rs.size < 0.05:
                return 1.0
            m1, m2 = drr_rs.mean(), _tgt.mean()
            s1, s2 = drr_rs.std(),  _tgt.std()
            if s1 < 1e-8 or s2 < 1e-8:
                return 1.0
            ncc = ((drr_rs - m1) * (_tgt - m2)).mean() / (s1 * s2)
            return float(1.0 - max(-1.0, min(1.0, ncc)))

        # ── Step 3: multi-start CMA-ES ────────────────────────────────────
        rot_rad = float(np.deg2rad(max_rot_deg))
        sigma_r = np.array([rot_rad, rot_rad, rot_rad,
                            max_trans_mm, max_trans_mm, max_trans_mm])

        best_cost = float('inf')
        best_x    = np.zeros(6)

        rng = np.random.default_rng()
        for _ in range(n_starts):
            x0 = rng.uniform(-0.3, 0.3, 6) * sigma_r
            try:
                res = run_cmaes_single(_cost, x0, sigma_r / 3.0,
                                       bounds_center=np.zeros(6),
                                       popsize=8, tolx=0.05, maxiter=150)
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_x    = res.pose.copy()
            except Exception:
                pass

        # ── Step 4: backproject from refined pose ─────────────────────────
        R_loc, t_loc = perturb_extrinsic(R_global, t_global,
                                         best_x[:3], best_x[3:])
        C, d_hat = _backproject_ray(R_loc, t_loc, u_gt, v_gt, fx, fy, cx, cy)
        p3d_def  = _closest_point_on_ray(C, d_hat, p3d)
        delta    = p3d_def - p3d
        mag      = float(np.linalg.norm(delta))
        clamped  = mag > max_deform_mm
        if clamped:
            delta   = delta * (max_deform_mm / mag)
            p3d_def = p3d + delta

        if verbose:
            print(f"    [{name}] cost={best_cost:.4f}  "
                  f"deform={mag:.1f}mm {'(clamped)' if clamped else ''}")

        results[name] = VertebralDeformation(
            label=name, p3d_ct=p3d.copy(), p3d_intraop=p3d_def,
            delta_mm=delta, clamped=clamped)

    return results
