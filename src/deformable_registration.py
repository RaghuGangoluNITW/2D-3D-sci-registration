"""
deformable_registration.py
==========================
Per-vertebra deformable refinement for 2D/3D spine registration.

After a global rigid pose (R, t) is found (EPnP + CMA-ES), each vertebra
is treated independently.  A closed-form backprojection finds the 3D
position on the camera ray through the observed 2D centroid that is
closest to the CT centroid.  The difference is the deformation vector —
the amount the vertebra moved between the preop CT and the intraop position.

Coordinate convention (must match generic_patient_loader.py)
============================================================
Forward projection:
    P_cam = R @ xzy(p_world) + t
    u = FX * P_cam[0] / P_cam[2] + CX
    v = FY * P_cam[1] / P_cam[2] + CY
where xzy(p) swaps indices 1 and 2:  [x,y,z] → [x,z,y]

Backprojection (derived from above):
    Camera centre in world:  C = xzy(-R^T @ t)
    Ray direction from pixel (u,v):
        d_cam  = [(u-CX)/FX, (v-CY)/FY, 1.0]   (camera space)
        d_world = xzy(R^T @ d_cam / |d_cam|)      (world space, unit)
    Point on ray: r(λ) = C + λ * d_world,  λ > 0
    Closest point to CT centroid p3d:
        λ*    = dot(p3d - C, d_world)
        p3d'  = C + λ* * d_world                  (deformed position)
        Δ     = p3d' - p3d                         (deformation vector)

Biomechanical constraint: |Δ| ≤ MAX_DEFORM_MM (default 15 mm)
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
