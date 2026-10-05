"""
spine_ssm.py
============
Statistical Shape Model (SSM) for 2D lumbar spine landmark positions.

The lumbar spine (D12–L5) follows highly constrained anatomy: vertebrae
are nearly collinear in AP views and follow a lordotic curve in LAT views.
This SSM captures that prior from training data and uses it to:

  1. Complete partial predictions (e.g. CNN finds L3/L4/L5, SSM infers D12/L1/L2)
  2. Snap noisy predictions onto the anatomically valid manifold
  3. Provide a robust initialisation even when the CNN fails entirely (mean shape)

Theory
------
  - Collect N training shape vectors: s_i = [x_D12, y_D12, x_L1, y_L1, ..., x_L5, y_L5] ∈ ℝ¹²
  - Align shapes: translate to centroid, normalise scale by inter-vertebra span
  - PCA → mean shape s̄, modes P, variances λ
  - Any valid shape: s = s̄ + P b,  |b_k| ≤ 3√λ_k
  - Fitting given partial noisy observations: minimise ‖M(s̄ + Pb + t) - obs‖²
    where M selects observed channels, t is 2D translation, b is capped shape params

Usage::
    from spine_ssm import SpineSSM
    ssm = SpineSSM.build_from_manifest('data/landmark_manifest_combined.csv')
    ssm.save('checkpoints/spine_ssm.pkl')

    ssm = SpineSSM.load('checkpoints/spine_ssm.pkl')
    completed = ssm.fit_and_complete({'L4': [420,780], 'L5': [405,990]},
                                      img_hw=(1024,1024))
    # completed = {'D12':[x,y], 'L1':[x,y], ..., 'L5':[x,y]}
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

CLASS_NAMES = ['D12', 'L1', 'L2', 'L3', 'L4', 'L5']
N_CLASSES   = len(CLASS_NAMES)          # 6
SHAPE_DIM   = N_CLASSES * 2            # 12


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shape_to_vec(coords_dict: Dict[str, List[float]]) -> np.ndarray:
    """Convert {name: [x,y]} → (12,) vector.  Missing classes → NaN."""
    v = np.full(SHAPE_DIM, np.nan)
    for i, name in enumerate(CLASS_NAMES):
        if name in coords_dict:
            v[2*i]   = coords_dict[name][0]
            v[2*i+1] = coords_dict[name][1]
    return v


def _vec_to_dict(v: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert (12,) vector → {name: np.array([x,y])}."""
    return {name: np.array([v[2*i], v[2*i+1]])
            for i, name in enumerate(CLASS_NAMES)}


def _centroid(v: np.ndarray) -> np.ndarray:
    """Mean of all valid (x,y) pairs."""
    valid = v.reshape(-1, 2)
    valid = valid[~np.isnan(valid).any(axis=1)]
    return valid.mean(axis=0)


def _scale(v: np.ndarray) -> float:
    """Scale = span from lowest to highest valid y-coordinate."""
    valid = v.reshape(-1, 2)
    valid = valid[~np.isnan(valid).any(axis=1)]
    return float(valid[:, 1].max() - valid[:, 1].min()) + 1e-6


def _align(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Translate to centroid, normalise scale.  Returns (aligned, centroid, scale)."""
    c = _centroid(v)
    s = _scale(v)
    a = v.copy()
    a[0::2] = (a[0::2] - c[0]) / s
    a[1::2] = (a[1::2] - c[1]) / s
    return a, c, s


def _unalign(v_norm: np.ndarray, centroid: np.ndarray, scale: float) -> np.ndarray:
    v = v_norm.copy()
    v[0::2] = v[0::2] * scale + centroid[0]
    v[1::2] = v[1::2] * scale + centroid[1]
    return v


# ── Main class ────────────────────────────────────────────────────────────────

class SpineSSM:
    """
    Statistical Shape Model for 2D lumbar spine (D12–L5).

    Parameters
    ----------
    mean_shape : (12,) normalised mean shape vector
    modes      : (12, n_modes) PCA eigenvectors
    variances  : (n_modes,) eigenvalues
    train_scales : (N,) per-sample scale factors (for un-normalising at test)
    """

    def __init__(self,
                 mean_shape:    np.ndarray,
                 modes:         np.ndarray,
                 variances:     np.ndarray,
                 mean_centroid: np.ndarray,
                 mean_scale:    float,
                 n_modes:       int = 5):
        self.mean_shape    = mean_shape          # (12,)
        self.modes         = modes[:, :n_modes]  # (12, n_modes)
        self.variances     = variances[:n_modes]  # (n_modes,)
        self.mean_centroid = mean_centroid        # (2,) in original pixel space
        self.mean_scale    = mean_scale           # float
        self.n_modes       = n_modes

    # ── Build ─────────────────────────────────────────────────────────────────
    @classmethod
    def build_from_manifest(cls,
                            csv_path: str,
                            n_modes:  int = 5,
                            view:     Optional[str] = None) -> 'SpineSSM':
        """
        Build SSM from landmark_manifest_combined.csv.

        Uses ALL real samples. For samples missing some vertebrae, those
        are imputed from the PCA mean iteratively (EM-PCA style):
          1. Initialise missing coords from neighbour interpolation
          2. Align all shapes and compute PCA
          3. Re-project each shape onto PCA to fill its missing coords
          4. Repeat until convergence
        """
        import csv as _csv

        # ── Load samples ──────────────────────────────────────────────────────
        shapes_raw: List[np.ndarray] = []
        with open(csv_path, newline='') as f:
            for row in _csv.DictReader(f):
                # Accept real NRRDs and BUU JPGs; skip synthetic PNGs
                img_p = row['img_path']
                if not (img_p.endswith('.nrrd') or img_p.endswith('.jpg')):
                    continue
                if view and row['view'] != view:
                    continue
                lbls = row['labels'].split('|')
                crds = [[float(x) for x in c.split(',')]
                        for c in row['coords_px'].split(';')]
                d = dict(zip(lbls, crds))
                v = _shape_to_vec(d)
                # Skip samples with fewer than 2 valid vertebrae
                n_valid = (~np.isnan(v)).sum() // 2
                if n_valid < 2:
                    continue
                shapes_raw.append(v)

        print(f'  SSM: {len(shapes_raw)} samples loaded (view={view or "all"})')

        # ── EM-PCA: iteratively refine missing values ─────────────────────────
        # Step 0: neighbour imputation as starting point
        shapes = np.stack([_impute(v) for v in shapes_raw])   # (N, 12)

        for em_iter in range(10):
            # Align
            aligned, centroids, scales = [], [], []
            for v in shapes:
                a, c, s = _align(v)
                aligned.append(a); centroids.append(c); scales.append(s)
            aligned_np = np.stack(aligned)

            mean_shape = aligned_np.mean(axis=0)
            centred    = aligned_np - mean_shape
            cov        = centred.T @ centred / max(len(aligned_np) - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx      = np.argsort(eigvals)[::-1]
            eigvals  = eigvals[idx];  eigvecs = eigvecs[:, idx]
            modes_k  = eigvecs[:, :n_modes]

            # Re-project missing coords from PCA reconstruction
            prev_shapes = shapes.copy()
            for i, (v_raw, v_aligned, c, s) in enumerate(
                    zip(shapes_raw, aligned_np, centroids, scales)):
                nan_mask = np.isnan(v_raw)
                if not nan_mask.any():
                    continue
                # Project observed channels onto PCA subspace
                obs_mask_i = ~nan_mask
                obs_vals   = v_aligned[obs_mask_i]
                # Solve for b: modes_k[obs] @ b ≈ (v_aligned - mean)[obs]
                A = modes_k[obs_mask_i, :]
                b_obs = v_aligned[obs_mask_i] - mean_shape[obs_mask_i]
                b, _, _, _ = np.linalg.lstsq(A, b_obs, rcond=None)
                # Reconstruct full shape and fill missing
                full_aligned = mean_shape + modes_k @ b
                # Un-align
                full_px = _unalign(full_aligned, c, s)
                # Fill only missing coords
                shapes[i][nan_mask] = full_px[nan_mask]

            delta = np.abs(shapes - prev_shapes).max()
            if delta < 0.1:
                break

        # Final PCA on completed shapes
        aligned, centroids, scales = [], [], []
        for v in shapes:
            a, c, s = _align(v)
            aligned.append(a); centroids.append(c); scales.append(s)
        aligned_np    = np.stack(aligned)
        mean_shape    = aligned_np.mean(axis=0)
        mean_centroid = np.stack(centroids).mean(axis=0)
        mean_scale    = float(np.mean(scales))

        centred  = aligned_np - mean_shape
        cov      = centred.T @ centred / max(len(aligned_np) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx      = np.argsort(eigvals)[::-1]
        eigvals  = eigvals[idx];  eigvecs = eigvecs[:, idx]

        var_explained = eigvals[:n_modes].sum() / (eigvals.sum() + 1e-9)
        print(f'  SSM: {n_modes} modes explain {100*var_explained:.1f}% variance')
        print(f'  SSM: mean scale={mean_scale:.1f}px  centroid={mean_centroid}')

        return cls(mean_shape, eigvecs, eigvals, mean_centroid, mean_scale, n_modes)

    # ── Save / load ───────────────────────────────────────────────────────────
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f'  SSM saved → {path}')

    @classmethod
    def load(cls, path: str) -> 'SpineSSM':
        with open(path, 'rb') as f:
            d = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(d)
        return obj

    # ── Mean shape in pixel space ─────────────────────────────────────────────
    def mean_shape_px(self,
                      centroid: Optional[np.ndarray] = None,
                      scale:    Optional[float]      = None) -> Dict[str, np.ndarray]:
        """Return mean shape in pixel coords (use stored mean if not given)."""
        c = centroid if centroid is not None else self.mean_centroid
        s = scale    if scale    is not None else self.mean_scale
        return _vec_to_dict(_unalign(self.mean_shape, c, s))

    # ── Fit to partial / noisy observations ──────────────────────────────────
    def fit_and_complete(self,
                         observations:  Dict[str, List[float]],
                         img_hw:        Tuple[int, int],
                         n_modes_fit:   int = 5,
                         b_constraint:  float = 3.0,
                         ) -> Dict[str, np.ndarray]:
        """
        Fit SSM to partial/noisy CNN predictions and return complete spine.

        Parameters
        ----------
        observations : dict of detected landmarks {name: [x_px, y_px]}
                       Can be empty (returns mean shape centred on image).
        img_hw       : (H, W) of original image, used to set default centroid.
        n_modes_fit  : number of PCA modes to optimise (default 5).
        b_constraint : |b_k| ≤ b_constraint × sqrt(λ_k)

        Returns
        -------
        Dict[str, np.ndarray] — complete 6-vertebra positions in pixel space.
        """
        H, W = img_hw
        n_modes = min(n_modes_fit, self.n_modes)
        modes   = self.modes[:, :n_modes]
        vars    = self.variances[:n_modes]

        # Observed channels mask
        obs_vec   = _shape_to_vec(observations)  # (12,) with NaN for missing
        obs_mask  = ~np.isnan(obs_vec)           # (12,) bool

        if obs_mask.sum() == 0:
            # No observations — return mean shape centred on image
            c = np.array([W / 2.0, H / 2.0])
            return self.mean_shape_px(c, self.mean_scale)

        # Observed pixel coords as vector
        obs_vals = obs_vec[obs_mask]

        # Initial guess: translate centroid to match observed centroid
        obs_pts   = obs_vec.reshape(-1, 2)
        obs_valid = obs_pts[obs_mask.reshape(-1, 2).all(axis=1)]
        init_c    = obs_valid.mean(axis=0)

        # Estimate initial scale from observed span (fall back to mean_scale)
        if len(obs_valid) >= 2:
            span_obs = float(obs_valid[:, 1].max() - obs_valid[:, 1].min())
            # Mean span of the observed vertebrae in the mean shape
            obs_idx  = [i for i, n in enumerate(CLASS_NAMES) if n in observations]
            mean_pts = self.mean_shape.reshape(-1, 2)
            if len(obs_idx) >= 2:
                mean_obs_pts  = mean_pts[obs_idx]
                span_mean_obs = float(mean_obs_pts[:, 1].max() -
                                      mean_obs_pts[:, 1].min()) + 1e-6
                init_s = span_obs / span_mean_obs
            else:
                init_s = self.mean_scale
        else:
            init_s = self.mean_scale

        # params: [b_0..b_{n-1}, tx, ty, log_scale]
        x0 = np.zeros(n_modes + 3)
        x0[n_modes]     = init_c[0]
        x0[n_modes + 1] = init_c[1]
        x0[n_modes + 2] = float(np.log(init_s + 1e-9))

        def _predict(params):
            b       = params[:n_modes]
            tx, ty  = params[n_modes], params[n_modes + 1]
            scale   = float(np.exp(params[n_modes + 2]))
            shape_n = self.mean_shape + modes @ b   # (12,) normalised
            shape_n[0::2] = shape_n[0::2] * scale + tx
            shape_n[1::2] = shape_n[1::2] * scale + ty
            return shape_n

        def _loss(params):
            pred      = _predict(params)
            obs_err   = (pred[obs_mask] - obs_vals) ** 2
            # Shape parameter regularisation: (b/σ)²
            b = params[:n_modes]
            reg = np.sum((b / (np.sqrt(vars) + 1e-9)) ** 2)
            return obs_err.mean() + 0.1 * reg

        # Bounds: b_k in [-3σ, 3σ], tx/ty unconstrained, log_scale loosely bounded
        bounds = []
        for k in range(n_modes):
            lim = b_constraint * float(np.sqrt(vars[k]))
            bounds.append((-lim, lim))
        bounds += [(0, W), (0, H), (np.log(50), np.log(3000))]

        res = minimize(_loss, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-10})

        pred_vec = _predict(res.x)
        result   = _vec_to_dict(pred_vec)

        # Clip to image bounds
        for name in result:
            result[name][0] = float(np.clip(result[name][0], 0, W))
            result[name][1] = float(np.clip(result[name][1], 0, H))

        return result

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def print_summary(self):
        print(f'SpineSSM: {self.n_modes} modes, mean_scale={self.mean_scale:.1f}px')
        total_var = self.variances.sum()
        cumvar = 0.0
        for i, v in enumerate(self.variances):
            cumvar += v
            print(f'  Mode {i+1}: λ={v:.4f}  cumvar={100*cumvar/total_var:.1f}%')


# ── Missing vertebra imputation ───────────────────────────────────────────────

# Average inter-vertebra vertical spacing in normalised (scale=1) coords
# Derived from the coverage stats: mean spacing ~180px, mean full span ~900px
# → normalised spacing ~0.18 each
_NORM_SPACING = {
    ('D12', 'L1'): np.array([ 0.003, -0.160]),  # D12 is above L1 (lower y)
    ('L1',  'L2'): np.array([ 0.005, -0.170]),
    ('L2',  'L3'): np.array([ 0.006, -0.180]),
    ('L3',  'L4'): np.array([ 0.006, -0.185]),
    ('L4',  'L5'): np.array([ 0.005, -0.178]),
}


def _impute(v: np.ndarray) -> np.ndarray:
    """
    Fill NaN vertebrae using neighbouring known positions + average spacing.
    Iterates until stable (handles chains of missing vertebrae).
    """
    v = v.copy()
    pts = v.reshape(-1, 2)   # (6, 2), may contain NaN

    for _ in range(10):      # iterate until stable
        changed = False
        for i, name in enumerate(CLASS_NAMES):
            if not np.isnan(pts[i]).any():
                continue     # already known

            # Try to infer from the neighbour above (lower index = higher vertebra)
            if i > 0 and not np.isnan(pts[i-1]).any():
                prev  = CLASS_NAMES[i-1]
                delta = _NORM_SPACING.get((prev, name))
                if delta is not None:
                    # Use approximate scale from known points
                    known = pts[~np.isnan(pts).any(axis=1)]
                    s = float(known[:, 1].max() - known[:, 1].min()) + 1e-6
                    pts[i] = pts[i-1] - delta * s   # delta in normalised → px
                    changed = True
                    continue

            # Try to infer from the neighbour below (higher index = lower vertebra)
            if i < N_CLASSES - 1 and not np.isnan(pts[i+1]).any():
                nxt   = CLASS_NAMES[i+1]
                delta = _NORM_SPACING.get((name, nxt))
                if delta is not None:
                    known = pts[~np.isnan(pts).any(axis=1)]
                    s = float(known[:, 1].max() - known[:, 1].min()) + 1e-6
                    pts[i] = pts[i+1] + delta * s
                    changed = True

        if not changed:
            break

    return pts.reshape(-1)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, argparse as _ap
    sys.path.insert(0, str(Path(__file__).parent))

    _p = _ap.ArgumentParser()
    _p.add_argument('--manifest', default='data/landmark_manifest_combined.csv')
    _p.add_argument('--out',      default='checkpoints/spine_ssm.pkl')
    _p.add_argument('--n_modes',  type=int, default=5)
    _args = _p.parse_args()

    print(f'Building SSM from {_args.manifest} ...')
    ssm = SpineSSM.build_from_manifest(_args.manifest, n_modes=_args.n_modes)
    ssm.print_summary()
    ssm.save(_args.out)

    # Test: give only L4+L5, ask for full spine
    print('\nTest: observed = {L4: (420,780), L5: (405,990)}')
    result = ssm.fit_and_complete(
        {'L4': [420, 780], 'L5': [405, 990]},
        img_hw=(1024, 1024)
    )
    for name, xy in result.items():
        print(f'  {name}: ({xy[0]:.0f}, {xy[1]:.0f})')

    # Test: empty observations → mean shape
    print('\nTest: empty observations (mean shape)')
    result2 = ssm.fit_and_complete({}, img_hw=(1024, 1024))
    for name, xy in result2.items():
        print(f'  {name}: ({xy[0]:.0f}, {xy[1]:.0f})')
