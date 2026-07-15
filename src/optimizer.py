"""
optimizer.py — CMA-ES Multi-Start Registration Optimizer
=========================================================
Implements the optimization framework from Ketcha 2017:

1. Multi-start CMA-ES with kD-tree space-partitioning initialization (§2.3)
2. Two-phase convergence: weak TolX=1 for all starts, then tight TolX=0.1
3. Per-stage search ranges (SR) from Table 1

6DOF pose vector: [xr, yr, zr, eta, theta, phi]
  xr, yr, zr  in mm  (translation of CT center)
  eta, theta, phi in degrees (Euler rotations)
"""

import numpy as np
import cma
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Search range presets (Table 1 from paper)
# ---------------------------------------------------------------------------

# Stage 1: full global rigid registration
# [xr_mm, yr_mm, zr_mm, eta_deg, theta_deg, phi_deg]
# CT is already roughly centered (L=0). C-arm isocenter is inside patient body.
# phi (in-plane rotation) is tightly bounded — spine rarely tilts > 10° in AP view
SR_STAGE1 = np.array([60., 60., 50., 15., 15., 10.])

# Stages 2-4: reduced
SR_STAGE2 = np.array([30., 30., 20., 10., 10., 7.])
SR_STAGE3 = np.array([15., 15., 10., 7.,  7.,  5.])
SR_STAGE4 = np.array([10., 10., 10., 5.,  5.,  5.])


def compute_adaptive_sr(iters_poses: List[np.ndarray],
                        label_projs: List[np.ndarray],
                        f_k: float,
                        z_r: float,
                        sdd: float = 1000.0) -> np.ndarray:
    """
    Compute adaptive search range for x,y per Eq. (5) of Ketcha 2017.

    Eq. (5):  SR_{x,y}(f_k) = (z_r/SDD) * (f_k * IVD + D_a)

    where:
        IVD = mean inter-vertebral distance on detector (from prev stage)
        D_a = std of projected label positions across initialization poses

    Args:
        iters_poses   : list of (6,) pose vectors from previous stage initializations
        label_projs   : list of (N, 2) projected label positions for each pose
        f_k           : IVD fraction (Stage2=0.4, Stage3=0.2, Stage4=0.15)
        z_r           : current z-translation estimate (depth, mm)

    Returns:
        sr_xy : scalar adaptive search range for x and y (mm)
    """
    if len(iters_poses) == 0 or len(label_projs) == 0:
        return 50.0  # fallback

    # IVD: mean distance between consecutive vertebral projections
    ivd_vals = []
    for projs in label_projs:
        if len(projs) > 1:
            diffs = np.diff(projs, axis=0)
            ivd_vals.extend(np.linalg.norm(diffs, axis=1).tolist())
    IVD = np.mean(ivd_vals) if ivd_vals else 30.0

    # D_a: spread of label projections across initialization poses
    if len(label_projs) > 1:
        projs_stack = np.stack(label_projs, axis=0)    # (n_init, N_labels, 2)
        mean_proj   = projs_stack.mean(axis=0)          # (N_labels, 2)
        deviations  = projs_stack - mean_proj           # (n_init, N_labels, 2)
        da_vals     = np.linalg.norm(deviations, axis=-1)  # (n_init, N_labels)
        D_a         = float(da_vals.mean())
    else:
        D_a = 5.0

    sr_xy = (abs(z_r) / sdd) * (f_k * IVD + D_a)
    return float(np.clip(sr_xy, 5.0, 200.0))


# ---------------------------------------------------------------------------
# kD-tree partitioned multi-start initialization
# ---------------------------------------------------------------------------

def kdtree_multistart_init(search_ranges: np.ndarray,
                           n_starts: int,
                           center: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate n_starts initialization points using kD-tree plane-splitting
    (Bentley 1975) as described in paper §2.3.

    Each sub-region is initialized at its center.

    Args:
        search_ranges : (6,) half-range for each DOF [x, y, z, eta, theta, phi]
        n_starts      : number of multi-start initializations
        center        : (6,) center of search space (default: zeros)

    Returns:
        inits : (n_starts, 6) initialization points
    """
    if center is None:
        center = np.zeros(6)

    n_dof = len(search_ranges)
    # Start with one box = full search range
    lo = center - search_ranges
    hi = center + search_ranges
    boxes = [(lo.copy(), hi.copy())]

    while len(boxes) < n_starts:
        # Find largest box (by range of its longest dimension)
        sizes = [np.max((b[1] - b[0]) / (search_ranges + 1e-8)) for b in boxes]
        idx = int(np.argmax(sizes))
        blo, bhi = boxes.pop(idx)
        # Split along the dimension with largest normalized range
        norms = (bhi - blo) / (search_ranges + 1e-8)
        dim = int(np.argmax(norms))
        mid = (blo[dim] + bhi[dim]) / 2.0
        blo1, bhi1 = blo.copy(), bhi.copy()
        blo2, bhi2 = blo.copy(), bhi.copy()
        bhi1[dim] = mid
        blo2[dim] = mid
        boxes.extend([(blo1, bhi1), (blo2, bhi2)])

    # Each start at box center
    inits = np.array([(b[0] + b[1]) / 2.0 for b in boxes[:n_starts]])
    return inits


# ---------------------------------------------------------------------------
# Single CMA-ES run
# ---------------------------------------------------------------------------

@dataclass
class OptResult:
    pose: np.ndarray    # (6,) best pose [xr,yr,zr,eta,theta,phi]
    cost: float         # best GO cost (lower = better)
    n_evals: int        # number of function evaluations


def run_cmaes_single(cost_fn: Callable[[np.ndarray], float],
                     x0: np.ndarray,
                     search_ranges: np.ndarray,
                     bounds_center: Optional[np.ndarray] = None,
                     popsize: int = 50,
                     tolx: float = 1.0,
                     maxiter: int = 1000,
                     ) -> OptResult:
    """
    Run a single CMA-ES optimization.

    Args:
        cost_fn       : function(pose_6dof) -> float, lower = better
        x0            : (6,) initial point
        search_ranges : (6,) half-range — used to set initial sigma AND bounds
        bounds_center : (6,) center for bounds box (default: x0).
                        Set to a fixed anchor (e.g. the known perturbation) so
                        GT is always inside the box even when x0 is a grid seed
                        far from GT.
        popsize       : population size λ
        tolx          : convergence tolerance (TolX)
        maxiter       : max iterations

    Returns:
        OptResult with best pose and cost found.
    """
    # Initial sigma: 1/4 of the search range (paper heuristic)
    sigma0 = float(np.mean(search_ranges) / 4.0)

    # Bounds: centered on bounds_center (defaults to x0 for backward compat.)
    bc = bounds_center if bounds_center is not None else x0

    opts = cma.CMAOptions()
    opts['popsize']   = popsize
    opts['maxiter']   = maxiter
    opts['tolx']      = tolx
    opts['tolconditioncov'] = 1e14
    opts['verbose']   = -9  # silent
    opts['bounds']    = [list(bc - search_ranges), list(bc + search_ranges)]

    try:
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
        n_evals = 0
        while not es.stop():
            solutions = es.ask()
            fitnesses = [cost_fn(np.array(s)) for s in solutions]
            es.tell(solutions, fitnesses)
            n_evals += len(solutions)
        result = es.result
        best_pose = np.array(result.xbest)
        best_cost = float(result.fbest)
    except Exception as e:
        best_pose = x0.copy()
        best_cost = float(cost_fn(x0))
        n_evals = 1

    return OptResult(pose=best_pose, cost=best_cost, n_evals=n_evals)


# ---------------------------------------------------------------------------
# Multi-start CMA-ES (Phase 1 + Phase 2)
# ---------------------------------------------------------------------------

def multistart_cmaes(cost_fn: Callable[[np.ndarray], float],
                     search_ranges: np.ndarray,
                     center: Optional[np.ndarray] = None,
                     n_starts: int = 50,
                     popsize: int = 125,
                     tolx_phase1: float = 1.0,
                     tolx_phase2: float = 0.1,
                     verbose: bool = True,
                     ) -> OptResult:
    """
    Full multi-start CMA-ES as in Ketcha 2017 §2.3.

    Phase 1: n_starts independent runs with weak convergence (TolX=1)
    Phase 2: Best solution from Phase 1 re-optimized with tight convergence

    Args:
        cost_fn        : function(pose_6dof) -> float
        search_ranges  : (6,) half-ranges
        center         : (6,) center of search space
        n_starts       : number of Phase 1 initializations (paper: 50 for Stage 1)
        popsize        : CMA-ES population size (paper: 125 for Stage 1)
        tolx_phase1    : weak tolerance for Phase 1
        tolx_phase2    : tight tolerance for Phase 2

    Returns:
        OptResult (best pose overall)
    """
    if center is None:
        center = np.zeros(6)

    # Phase 1: distributed multi-start
    inits = kdtree_multistart_init(search_ranges, n_starts, center)

    best_result = None
    total_evals = 0

    if verbose:
        print(f"  Phase 1: {n_starts} starts, λ={popsize}, TolX={tolx_phase1}")

    for i, x0 in enumerate(inits):
        result = run_cmaes_single(
            cost_fn, x0, search_ranges,
            popsize=popsize, tolx=tolx_phase1
        )
        total_evals += result.n_evals

        if best_result is None or result.cost < best_result.cost:
            best_result = result

        if verbose and (i % max(1, n_starts//5) == 0):
            print(f"    start {i+1}/{n_starts}: cost={result.cost:.4f} "
                  f"(best so far: {best_result.cost:.4f})")

    if verbose:
        print(f"  Phase 1 done. Best cost={best_result.cost:.4f}, "
              f"pose={best_result.pose.round(2)}, evals={total_evals}")

    # Phase 2: re-optimize best solution with tight convergence
    if verbose:
        print(f"  Phase 2: single restart from best, TolX={tolx_phase2}")

    # Tighter search around Phase 1 best
    tight_ranges = search_ranges / 4.0
    result2 = run_cmaes_single(
        cost_fn, best_result.pose, tight_ranges,
        popsize=popsize, tolx=tolx_phase2, maxiter=500
    )
    total_evals += result2.n_evals

    if result2.cost < best_result.cost:
        best_result = result2

    if verbose:
        print(f"  Phase 2 done. Best cost={best_result.cost:.4f}, evals={total_evals}")

    best_result.n_evals = total_evals
    return best_result


# ---------------------------------------------------------------------------
# Quaternion averaging (paper Eq. 4)
# ---------------------------------------------------------------------------

def euler_to_quaternion(eta: float, theta: float, phi: float) -> np.ndarray:
    """Convert Euler angles (deg) to unit quaternion (w, x, y, z)."""
    er, tr, pr = np.deg2rad([eta, theta, phi])
    # Half angles
    cy, sy = np.cos(pr/2), np.sin(pr/2)
    cp, sp = np.cos(tr/2), np.sin(tr/2)
    cr, sr = np.cos(er/2), np.sin(er/2)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion (w,x,y,z) to Euler angles (eta,theta,phi) in degrees."""
    w, x, y, z = q / (np.linalg.norm(q) + 1e-10)

    # Roll (eta) = rotation about X
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    eta = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (theta) = rotation about Y
    sinp = 2*(w*y - z*x)
    theta = np.arcsin(np.clip(sinp, -1, 1))

    # Yaw (phi) = rotation about Z
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    phi = np.arctan2(siny_cosp, cosy_cosp)

    return np.rad2deg(np.array([eta, theta, phi]))


def average_poses(poses: List[np.ndarray]) -> np.ndarray:
    """
    Average multiple 6DOF poses using Eq. (3) and (4) from Ketcha 2017.
    Translation: arithmetic mean.
    Rotation: quaternion eigen-decomposition mean (Markley 2007).

    Args:
        poses : list of (6,) arrays [xr, yr, zr, eta, theta, phi]

    Returns:
        (6,) averaged pose
    """
    if len(poses) == 0:
        return np.zeros(6)
    if len(poses) == 1:
        return poses[0].copy()

    poses_arr = np.array(poses)

    # Average translation
    avg_t = poses_arr[:, :3].mean(axis=0)

    # Average rotation via quaternion eigen-method
    quats = np.array([euler_to_quaternion(*p[3:]) for p in poses_arr])
    # Ensure consistent hemisphere (all quaternions should be in same half-sphere)
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]

    # Accumulate M = sum_i q_i q_i^T
    M = quats.T @ quats   # (4, 4)
    eigvals, eigvecs = np.linalg.eigh(M)
    avg_q = eigvecs[:, -1]  # eigenvector of largest eigenvalue

    avg_rot = quaternion_to_euler(avg_q)

    return np.concatenate([avg_t, avg_rot])


if __name__ == '__main__':
    # Test: find minimum of a simple quadratic (should recover zero)
    import numpy as np

    true_params = np.array([5., -10., 3., 2., -1., 4.])

    def toy_cost(p):
        return float(np.sum((p - true_params)**2))

    sr = SR_STAGE1.copy()
    result = multistart_cmaes(
        cost_fn=toy_cost,
        search_ranges=sr,
        center=np.zeros(6),
        n_starts=10,
        popsize=30,
        verbose=True,
    )
    print(f"\nTrue:  {true_params}")
    print(f"Found: {result.pose.round(3)}")
    print(f"Error: {np.abs(result.pose - true_params).max():.4f}")
    print(f"Total evals: {result.n_evals}")
