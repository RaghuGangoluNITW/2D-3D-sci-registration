"""
registration.py — Multi-Stage msLevelCheck 3D-2D Registration
==============================================================
Implements Algorithm 1 from Ketcha 2017: Phys. Med. Biol. 62 (2017) 4604–4622.

Stage framework: {All, 5, 3, 1}  (4 stages, "Framework 6" from paper)

Key design:
  - Stage 1: Full global rigid registration of entire CT vs radiograph
  - Stage 2-4: Progressively localized sub-volume registrations
  - Each stage initialized from previous stage's result(s)
  - Labels projected using per-vertebra rigid transforms from final stage

This module registers:
  - PREOP CT (3D) + PREOP 3D labels -> C-arm intraoperative image (2D target)

The 2D target is now the actual intraoperative C-arm image.
PDE is computed as the distance (mm, in detector space) between:
  - The PREOP label projected at the found pose  (predicted position)
  - The POSTOP label projected at zero pose using POSTOP CT origin
    NOTE: Since POSTOP CT and C-arm are in different coordinate frames,
    PDE is reported in normalized detector units (pixel distance) rather
    than absolute mm. The paper computes PDE against manually annotated
    2D reference points; without those annotations, we report the optimizer
    convergence quality (cost) as the primary metric.
"""

import numpy as np
import time
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

from drr import DRRGenerator, normalize_image, preprocess_topogram, preprocess_carm
from similarity import go_cost, normalized_cross_correlation, mutual_information
from optimizer import (
    multistart_cmaes, average_poses, compute_adaptive_sr,
    SR_STAGE1, SR_STAGE2, SR_STAGE3, SR_STAGE4
)
from data_loader import PatientSession, Centroid


# ---------------------------------------------------------------------------
# Registration result
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage: int
    subvol_labels: List[str]                  # labels used in this sub-volume
    pose: np.ndarray                          # (6,) [xr,yr,zr,eta,theta,phi]
    cost: float
    proj_2d: Optional[np.ndarray] = None      # (N,2) projected positions (mm)
    n_evals: int = 0


@dataclass
class RegistrationResult:
    patient_name: str
    preop_labels: List[str]
    target_image: np.ndarray              # 2D target used (C-arm or topogram, H×W)
    target_source: str = 'unknown'        # 'carm_lateral', 'carm_ap', 'topogram'

    # Per-stage results
    stage_results: List[StageResult] = field(default_factory=list)

    # Final per-label poses and 2D projections
    label_poses: Dict[str, np.ndarray] = field(default_factory=dict)  # label -> (6,)
    label_proj_2d: Dict[str, np.ndarray] = field(default_factory=dict)  # label -> (2,)

    # Metrics (PDE against POSTOP CT centroids re-projected — approximate)
    pde_per_label: Optional[Dict[str, float]] = None
    mean_pde: Optional[float] = None
    max_pde: Optional[float] = None
    failure_rate: Optional[float] = None   # fraction of labels with PDE > 20 mm

    # Stage 1 rigid-only result (for comparison)
    rigid_label_proj_2d: Optional[Dict[str, np.ndarray]] = None
    rigid_pde_per_label: Optional[Dict[str, float]] = None
    rigid_mean_pde: Optional[float] = None
    rigid_max_pde: Optional[float] = None

    # Best GO similarity achieved (primary metric when no GT 2D annotations)
    best_go_cost: Optional[float] = None

    runtime_seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Registration: {self.patient_name}",
            f"Target: {self.target_source}  shape={self.target_image.shape}",
            f"Labels: {self.preop_labels}",
        ]
        if self.best_go_cost is not None:
            lines.append(f"\nBest GO cost (lower=better): {self.best_go_cost:.6f}")

        if self.rigid_mean_pde is not None:
            lines.append(f"\nStage-1 RIGID only (approx PDE vs POSTOP centroids):")
            lines.append(f"  Mean PDE: {self.rigid_mean_pde:.2f} mm")
            lines.append(f"  Max PDE:  {self.rigid_max_pde:.2f} mm")
            if self.rigid_pde_per_label:
                for lbl, pde in self.rigid_pde_per_label.items():
                    lines.append(f"    {lbl}: {pde:.2f} mm")

        if self.mean_pde is not None:
            lines.append(f"\nmsLevelCheck (multi-stage, approx PDE):")
            lines.append(f"  Mean PDE:     {self.mean_pde:.2f} mm")
            lines.append(f"  Max PDE:      {self.max_pde:.2f} mm")
            lines.append(f"  Failure rate: {self.failure_rate*100:.1f}%  (> 20 mm)")
            if self.pde_per_label:
                for lbl, pde in self.pde_per_label.items():
                    lines.append(f"    {lbl}: {pde:.2f} mm")
        else:
            lines.append("\n[No GT 2D annotations — showing 2D projections (mm from det center)]")
            if self.label_proj_2d:
                for lbl, uv in self.label_proj_2d.items():
                    lines.append(f"  {lbl}: ({uv[0]:.1f}, {uv[1]:.1f}) mm")

        lines.append(f"\nRuntime: {self.runtime_seconds:.1f} s")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Binary volume mask (paper §2.1, r=50 mm)
# ---------------------------------------------------------------------------

def apply_spine_mask(ct_volume: np.ndarray,
                     ct_spacing: np.ndarray,
                     ct_origin: np.ndarray,
                     label_positions_lps: np.ndarray,
                     radius_mm: float = 50.0) -> np.ndarray:
    """
    Apply binary cylindrical mask around the spine axis (paper §2.1).
    Sets voxels > radius_mm from the interpolated spine line to 0.

    Args:
        ct_volume    : (Z, Y, X) HU array
        ct_spacing   : (sx, sy, sz) mm
        ct_origin    : LPS mm of [0,0,0] voxel
        label_positions_lps: (N, 3) sorted LPS vertebral centroids
        radius_mm    : mask radius (paper: 50 mm)
    """
    Z, Y, X = ct_volume.shape
    sx, sy, sz = ct_spacing

    # Build voxel grid in LPS world (X, Y, Z coordinates)
    xi = np.arange(X) * sx + ct_origin[0]   # L (mm)
    yi = np.arange(Y) * sy + ct_origin[1]   # P (mm)
    zi = np.arange(Z) * sz + ct_origin[2]   # S (mm)

    # For each voxel, compute min distance to the spine line segments
    # The spine axis: linear interpolation through label Z positions
    # Parameterize by S-axis (Z), interpolate X and Y
    lps = label_positions_lps
    # Sort by Z (Superior axis) = lps[:,2]
    order = np.argsort(lps[:, 2])
    lps_sorted = lps[order]

    s_vals = lps_sorted[:, 2]   # Z(S) positions
    l_vals = lps_sorted[:, 0]   # L positions
    p_vals = lps_sorted[:, 1]   # P positions

    from scipy.interpolate import interp1d
    if len(s_vals) > 1:
        fl = interp1d(s_vals, l_vals, kind='linear', fill_value='extrapolate')
        fp = interp1d(s_vals, p_vals, kind='linear', fill_value='extrapolate')
    else:
        fl = lambda z: l_vals[0]
        fp = lambda z: p_vals[0]

    # Compute mask in 3D — process slice by slice to save memory
    masked = ct_volume.copy()
    for iz in range(Z):
        z_world = zi[iz]
        spine_l = float(fl(z_world))
        spine_p = float(fp(z_world))

        # Distance of each pixel in this slice from spine
        dist_x = (xi - spine_l)**2    # (X,)
        dist_y = (yi - spine_p)**2    # (Y,)
        dist_2d = np.sqrt(dist_x[None, :] + dist_y[:, None])  # (Y, X)
        mask_slice = dist_2d <= radius_mm
        masked[iz] *= mask_slice

    return masked


# ---------------------------------------------------------------------------
# Adaptive histogram equalization (applied after Stage 1, paper §2.4)
# ---------------------------------------------------------------------------

def adaptive_histeq(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE (adaptive histogram equalization) to enhance local contrast."""
    from skimage.exposure import equalize_adapthist
    img_norm = np.clip(image, 0, 1).astype(np.float32)
    return equalize_adapthist(img_norm, clip_limit=0.02).astype(np.float32)


# ---------------------------------------------------------------------------
# Cost function factory
# ---------------------------------------------------------------------------

def make_cost_fn(gen: DRRGenerator,
                 target_2d: np.ndarray,
                 det_h: int, det_w: int,
                 pix_mm: float,
                 n_steps: int,
                 sigma: float,
                 metric: str = 'go',
                 ) -> callable:
    """
    Create a cost function: pose (6,) -> float.
    Lower = better alignment.
    """
    def cost_fn(pose: np.ndarray) -> float:
        xr, yr, zr, eta, theta, phi = pose
        try:
            drr = gen.generate(xr, yr, zr, eta, theta, phi,
                               det_h=det_h, det_w=det_w,
                               pix_mm=pix_mm, n_steps=n_steps)
            drr_n = normalize_image(drr)
            if metric == 'go':
                return go_cost(drr_n, target_2d, sigma=sigma)
            elif metric == 'ncc':
                return 1.0 - normalized_cross_correlation(drr_n, target_2d)
            else:
                return go_cost(drr_n, target_2d, sigma=sigma)
        except Exception:
            return 1.0
    return cost_fn


# ---------------------------------------------------------------------------
# Main msLevelCheck Registration
# ---------------------------------------------------------------------------

class MSLevelCheck:
    """
    Multi-stage 3D–2D registration: msLevelCheck (Ketcha 2017).

    Implements 4-stage framework: {All, 5, 3, 1}
    """

    # Stage config: (n_subvol, pix_mm, sigma, SR, ms, lambda, f_k)
    STAGES = [
        # Stage 1: All vertebrae, coarse resolution
        dict(n=None,  pix_mm=2.00, sigma=2.00, sr=SR_STAGE1, ms=30,  lam=80,  fk=None),
        # Stage 2: 5 adjacent vertebrae
        dict(n=5,     pix_mm=1.75, sigma=1.50, sr=SR_STAGE2, ms=15,  lam=50,  fk=0.40),
        # Stage 3: 3 adjacent vertebrae
        dict(n=3,     pix_mm=1.75, sigma=1.50, sr=SR_STAGE3, ms=15,  lam=50,  fk=0.20),
        # Stage 4: 1 vertebra (per-label)
        dict(n=1,     pix_mm=1.50, sigma=1.25, sr=SR_STAGE4, ms=15,  lam=50,  fk=0.15),
    ]
    # NOTE: Paper uses MS=50/25, λ=125/100, but reduced here for speed given
    # we're on CPU-fallback for some steps. Increase for production runs.

    def __init__(self,
                 preop: PatientSession,
                 postop: PatientSession,
                 verbose: bool = True):
        self.preop   = preop
        self.postop  = postop
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def register(self,
                 stages_to_run: int = 4,
                 use_spine_mask: bool = True,
                 view: str = 'lateral',
                 det_size: int = 256,
                 ) -> RegistrationResult:
        """
        Run multi-stage registration.

        Args:
            stages_to_run : 1=rigid only, 4=full msLevelCheck
            use_spine_mask: apply 50mm spine mask to CT
            view          : 'lateral' or 'ap' — which C-arm view to use
            det_size      : DRR detector size in pixels (square)
        """
        t_start = time.time()
        preop, postop = self.preop, self.postop

        self._log(f"\n{'='*60}")
        self._log(f"msLevelCheck: {preop.patient_name} PREOP CT -> C-arm {view.upper()}")
        self._log(f"  PREOP CT: {preop.ct_volume.shape}, labels: {[c.label for c in preop.centroids]}")

        # ----------------------------------------------------------------
        # Select 2D target: C-arm image (preferred) or topogram (fallback)
        # ----------------------------------------------------------------
        target_raw = None
        target_source = 'none'

        if view == 'lateral' and preop.carm_lateral is not None:
            target_raw    = preop.carm_lateral.image
            target_source = 'carm_lateral'
            self._log(f"  2D target: C-arm LATERAL {target_raw.shape} from {preop.carm_lateral.source_path.name}")
        elif view == 'ap' and preop.carm_ap is not None:
            target_raw    = preop.carm_ap.image
            target_source = 'carm_ap'
            self._log(f"  2D target: C-arm AP {target_raw.shape} from {preop.carm_ap.source_path.name}")
        elif preop.topogram is not None:
            target_raw    = preop.topogram
            target_source = 'topogram'
            self._log(f"  2D target: Topogram (fallback) {target_raw.shape}")
        else:
            raise ValueError(f"No 2D target found for {preop.patient_name} {view} view.")

        # Preprocess: C-arm -> invert + resize + CLAHE
        if 'carm' in target_source:
            target_2d = preprocess_carm(target_raw, target_size=det_size, invert=True)
        else:
            target_2d = preprocess_topogram(target_raw, target_pix_mm=2.0)
            from skimage.transform import resize
            target_2d = resize(target_2d, (det_size, det_size), anti_aliasing=True).astype(np.float32)

        self._log(f"  Preprocessed target: {target_2d.shape}  range=[{target_2d.min():.3f},{target_2d.max():.3f}]")

        # ----------------------------------------------------------------
        # Sort labels by S-axis (Superior = Z in LPS)
        # ----------------------------------------------------------------
        labels_sorted = sorted(preop.centroids, key=lambda c: c.position[2])
        label_names   = [c.label for c in labels_sorted]
        label_lps     = np.array([c.position for c in labels_sorted])  # (N,3)
        N_labels      = len(labels_sorted)
        self._log(f"  Labels ({N_labels}): {label_names}")

        # ----------------------------------------------------------------
        # Apply spine mask
        # ----------------------------------------------------------------
        if use_spine_mask:
            self._log("  Applying spine mask (r=50 mm)...")
            ct_masked = apply_spine_mask(
                preop.ct_volume, preop.ct_spacing, preop.ct_origin, label_lps
            )
        else:
            ct_masked = preop.ct_volume

        # ----------------------------------------------------------------
        # Compute adaptive pix_mm based on label span
        # so the DRR FOV tightly covers the vertebral column
        # ----------------------------------------------------------------
        label_si_span = label_lps[:, 2].max() - label_lps[:, 2].min()  # S-I mm
        label_lr_span = label_lps[:, 0].max() - label_lps[:, 0].min()  # L-R mm
        fov_needed    = max(label_si_span, label_lr_span) * 1.5         # 50% margin
        auto_pix_mm   = fov_needed / det_size
        auto_pix_mm   = float(np.clip(auto_pix_mm, 0.4, 2.0))
        self._log(f"  Label span: SI={label_si_span:.1f}mm LR={label_lr_span:.1f}mm "
                  f"-> auto pix_mm={auto_pix_mm:.3f} (FOV={auto_pix_mm*det_size:.0f}mm)")

        # Override Stage pix_mm values with auto-computed value
        # (keeps ratio between stages; Stage 4 uses tighter FOV)
        # Use a local copy to avoid mutating the class-level STAGES dict
        stage_cfgs = []
        for stg in self.STAGES:
            cfg = dict(stg)
            cfg['pix_mm'] = float(np.clip(auto_pix_mm * (stg['pix_mm'] / 2.0), 0.3, 2.0))
            stage_cfgs.append(cfg)

        # ----------------------------------------------------------------
        # Build DRR generator (view matches C-arm view)
        # ----------------------------------------------------------------
        gen = DRRGenerator(ct_masked, preop.ct_spacing, preop.ct_origin,
                           view=view)

        # ----------------------------------------------------------------
        # Stage 1: Full rigid registration
        # ----------------------------------------------------------------
        self._log(f"\n[Stage 1] Full rigid registration ({N_labels} labels)")
        s1 = stage_cfgs[0]

        # Initialize search center: zero translation in X/Y, but offset Z
        # so the label cluster is roughly centered on the detector at identity
        label_ctr_lps = label_lps.mean(axis=0)           # centroid of all labels
        ct_ctr_lps    = gen.ctr_lps                       # CT geometric center
        z_init        = label_ctr_lps[2] - ct_ctr_lps[2] # S-axis offset
        x0_s1         = np.array([0., 0., z_init, 0., 0., 0.])
        self._log(f"  Stage 1 init: z_offset={z_init:.1f}mm (labels ctr={label_ctr_lps.round(1)})")

        cost_fn_s1 = make_cost_fn(
            gen, target_2d,
            det_h=det_size, det_w=det_size,
            pix_mm=s1['pix_mm'], n_steps=200,
            sigma=s1['sigma'],
        )

        result_s1 = multistart_cmaes(
            cost_fn_s1, s1['sr'], center=x0_s1,
            n_starts=s1['ms'], popsize=s1['lam'],
            verbose=self.verbose,
        )
        self._log(f"  Stage 1 best: cost={result_s1.cost:.6f}, pose={result_s1.pose.round(2)}")

        stage_results = [StageResult(
            stage=1, subvol_labels=label_names,
            pose=result_s1.pose.copy(), cost=result_s1.cost,
            n_evals=result_s1.n_evals
        )]

        # Save Stage 1 (rigid) projections for comparison
        rigid_projs = gen.project_labels(label_lps, *result_s1.pose)

        result = RegistrationResult(
            patient_name=preop.patient_name,
            preop_labels=label_names,
            target_image=target_2d,
            target_source=target_source,
            stage_results=stage_results,
            best_go_cost=result_s1.cost,
        )

        # Store rigid-only result
        result.rigid_label_proj_2d = {
            lbl: rigid_projs[i] for i, lbl in enumerate(label_names)
        }

        if stages_to_run == 1:
            # Rigid-only mode
            for i, lbl in enumerate(label_names):
                result.label_proj_2d[lbl] = rigid_projs[i]
                result.label_poses[lbl]   = result_s1.pose.copy()
            result.runtime_seconds = time.time() - t_start
            return result

        # ----------------------------------------------------------------
        # After Stage 1: apply adaptive histogram equalization to target
        # (paper §2.4 — enhances local bone structure)
        # ----------------------------------------------------------------
        target_eq = adaptive_histeq(target_2d)

        # ----------------------------------------------------------------
        # Stage 2: 5-vertebra sub-images
        # ----------------------------------------------------------------
        stage2_results: List[StageResult] = []
        s2 = stage_cfgs[1]

        if N_labels >= 2 and stages_to_run >= 2:
            window_size_2 = min(s2['n'], N_labels)
            windows_2 = self._get_windows(label_names, window_size_2)
            self._log(f"\n[Stage 2] {len(windows_2)} sub-volumes of size {window_size_2}")

            for wlabels in windows_2:
                widx = [label_names.index(l) for l in wlabels]
                sub_lps = label_lps[widx]

                ct_sub = apply_spine_mask(
                    preop.ct_volume, preop.ct_spacing, preop.ct_origin,
                    sub_lps, radius_mm=50.0
                )
                gen_sub = DRRGenerator(ct_sub, preop.ct_spacing, preop.ct_origin,
                                       view=view)

                sr2 = s2['sr'].copy()
                cost_fn_s2 = make_cost_fn(
                    gen_sub, target_eq,
                    det_h=det_size, det_w=det_size,
                    pix_mm=s2['pix_mm'], n_steps=200,
                    sigma=s2['sigma'],
                )

                sub_res = multistart_cmaes(
                    cost_fn_s2, sr2, center=result_s1.pose.copy(),
                    n_starts=s2['ms'], popsize=s2['lam'],
                    verbose=False,
                )
                self._log(f"  Stage2 {wlabels}: cost={sub_res.cost:.6f}")
                stage2_results.append(StageResult(
                    stage=2, subvol_labels=wlabels,
                    pose=sub_res.pose.copy(), cost=sub_res.cost,
                    n_evals=sub_res.n_evals
                ))

        # ----------------------------------------------------------------
        # Stage 3: 3-vertebra sub-images
        # ----------------------------------------------------------------
        stage3_results: List[StageResult] = []
        s3 = stage_cfgs[2]

        if N_labels >= 2 and stages_to_run >= 3 and len(stage2_results) > 0:
            window_size_3 = min(s3['n'], N_labels)
            windows_3 = self._get_windows(label_names, window_size_3)
            self._log(f"\n[Stage 3] {len(windows_3)} sub-volumes of size {window_size_3}")

            for wlabels in windows_3:
                init_pose = self._get_init_pose(wlabels, stage2_results, result_s1.pose)

                widx = [label_names.index(l) for l in wlabels]
                sub_lps = label_lps[widx]
                ct_sub = apply_spine_mask(
                    preop.ct_volume, preop.ct_spacing, preop.ct_origin,
                    sub_lps, radius_mm=50.0
                )
                gen_sub = DRRGenerator(ct_sub, preop.ct_spacing, preop.ct_origin,
                                       view=view)

                sr3 = s3['sr'].copy()
                cost_fn_s3 = make_cost_fn(
                    gen_sub, target_eq,
                    det_h=det_size, det_w=det_size,
                    pix_mm=s3['pix_mm'], n_steps=200,
                    sigma=s3['sigma'],
                )

                sub_res = multistart_cmaes(
                    cost_fn_s3, sr3, center=init_pose,
                    n_starts=s3['ms'], popsize=s3['lam'],
                    verbose=False,
                )
                self._log(f"  Stage3 {wlabels}: cost={sub_res.cost:.6f}")
                stage3_results.append(StageResult(
                    stage=3, subvol_labels=wlabels,
                    pose=sub_res.pose.copy(), cost=sub_res.cost,
                    n_evals=sub_res.n_evals
                ))

        # ----------------------------------------------------------------
        # Stage 4: Per-vertebra (1-label) registration
        # ----------------------------------------------------------------
        stage4_results: List[StageResult] = []
        s4 = stage_cfgs[3]
        prev_stage_results = stage3_results if stage3_results else stage2_results
        prev_stage_results = prev_stage_results if prev_stage_results else stage_results

        if stages_to_run >= 4:
            self._log(f"\n[Stage 4] Per-vertebra registration ({N_labels} labels)")

            for lbl in label_names:
                init_pose = self._get_init_pose([lbl], prev_stage_results, result_s1.pose)

                lidx = label_names.index(lbl)
                sub_lps = label_lps[[lidx]]
                ct_sub = apply_spine_mask(
                    preop.ct_volume, preop.ct_spacing, preop.ct_origin,
                    sub_lps, radius_mm=50.0
                )
                gen_sub = DRRGenerator(ct_sub, preop.ct_spacing, preop.ct_origin,
                                       view=view)

                sr4 = s4['sr'].copy()
                cost_fn_s4 = make_cost_fn(
                    gen_sub, target_eq,
                    det_h=det_size, det_w=det_size,
                    pix_mm=s4['pix_mm'], n_steps=200,
                    sigma=s4['sigma'],
                )

                sub_res = multistart_cmaes(
                    cost_fn_s4, sr4, center=init_pose,
                    n_starts=s4['ms'], popsize=s4['lam'],
                    verbose=False,
                )
                self._log(f"  Stage4 {lbl}: cost={sub_res.cost:.6f}")
                stage4_results.append(StageResult(
                    stage=4, subvol_labels=[lbl],
                    pose=sub_res.pose.copy(), cost=sub_res.cost,
                    n_evals=sub_res.n_evals,
                ))

        result.stage_results.extend(stage2_results + stage3_results + stage4_results)

        # ----------------------------------------------------------------
        # Final: use Stage 4 per-label poses to project labels
        # ----------------------------------------------------------------
        if stage4_results:
            final_stage = stage4_results
        elif stage3_results:
            final_stage = stage3_results
        elif stage2_results:
            final_stage = stage2_results
        else:
            final_stage = stage_results

        # Track best GO cost achieved across all stages
        all_costs = [sr.cost for sr in result.stage_results]
        result.best_go_cost = min(all_costs) if all_costs else result_s1.cost

        # Map each label to its best pose
        label_to_pose = {}
        for sr in final_stage:
            for lbl in sr.subvol_labels:
                if lbl not in label_to_pose or sr.cost < label_to_pose[lbl][1]:
                    label_to_pose[lbl] = (sr.pose.copy(), sr.cost)

        # Project labels using their individual poses
        for i, lbl in enumerate(label_names):
            if lbl in label_to_pose:
                pose = label_to_pose[lbl][0]
            else:
                pose = result_s1.pose
            result.label_poses[lbl] = pose
            proj = gen.project_labels(label_lps[[i]], *pose)[0]
            result.label_proj_2d[lbl] = proj

        result.runtime_seconds = time.time() - t_start
        self._log(f"\nTotal runtime: {result.runtime_seconds:.1f} s")
        return result

    def _get_windows(self, label_names: List[str], window_size: int) -> List[List[str]]:
        """Get all sliding windows of adjacent labels."""
        n = len(label_names)
        if window_size >= n:
            return [label_names]
        return [label_names[i:i+window_size] for i in range(n - window_size + 1)]

    def _get_init_pose(self,
                       target_labels: List[str],
                       prev_results: List[StageResult],
                       fallback_pose: np.ndarray) -> np.ndarray:
        """
        Get initialization pose for a set of target labels by averaging
        poses from previous stage results that overlap with target_labels.
        (Paper Eq. 3 & 4)
        """
        overlapping = []
        for sr in prev_results:
            if any(l in sr.subvol_labels for l in target_labels):
                overlapping.append(sr.pose.copy())

        if not overlapping:
            return fallback_pose.copy()
        return average_poses(overlapping)


# ---------------------------------------------------------------------------
# PDE Evaluation
# ---------------------------------------------------------------------------

def compute_pde(result: RegistrationResult,
                postop: PatientSession,
                gen_preop: DRRGenerator) -> RegistrationResult:
    """
    Compute Projection Distance Error (PDE) for each label.

    The PDE is defined as the 2D Euclidean distance (mm, in detector space)
    between:
      - PREDICTED: PREOP 3D label projected at the found registration pose
      - REFERENCE: POSTOP 3D label projected at zero pose using the SAME
                   PREOP DRR generator (i.e., we assume the POSTOP label
                   position is the "ground truth" intraoperative position,
                   and we compute where it lands in the PREOP CT's detector).

    IMPORTANT NOTE: This is an approximate PDE because:
      1. POSTOP labels are in POSTOP CT coordinates, not PREOP CT
      2. The true PDE (paper Eq. 7) requires manually annotated 2D positions
         in the actual intraoperative C-arm image.
      3. Here we use the difference in detector-space projections as a proxy.

    For a physically meaningful PDE:
      - A human annotator would mark vertebral centroids in the C-arm image
      - PDE = distance between found projection and the annotated position

    Args:
        result     : RegistrationResult with label_proj_2d filled
        postop     : POSTOP session (has centroids as GT in 3D)
        gen_preop  : DRRGenerator built from PREOP CT (for consistent projection)
    """
    # Map POSTOP label positions by name
    postop_label_dict = {c.label: c.position for c in postop.centroids}

    # Project POSTOP labels with zero pose using PREOP generator
    # This simulates where the POSTOP anatomy projects if it were in the
    # PREOP CT coordinate space (not physically correct but consistent)
    gt_2d: Dict[str, np.ndarray] = {}
    common_labels = [l for l in result.preop_labels if l in postop_label_dict]

    if not common_labels:
        print("  [warn] No common labels between PREOP and POSTOP for PDE")
        return result

    postop_lps = np.array([postop_label_dict[l] for l in common_labels])
    gt_projs = gen_preop.project_labels(postop_lps, 0, 0, 0, 0, 0, 0)
    for lbl, proj in zip(common_labels, gt_projs):
        gt_2d[lbl] = proj

    # Compute PDE for labels that exist in both
    pde_dict = {}
    for lbl in common_labels:
        if lbl not in result.label_proj_2d:
            continue
        predicted_uv = result.label_proj_2d[lbl]
        gt_uv        = gt_2d[lbl]
        pde = float(np.linalg.norm(predicted_uv - gt_uv))
        pde_dict[lbl] = pde

    # Rigid-only PDE
    if result.rigid_label_proj_2d:
        rigid_pde: Dict[str, float] = {}
        for lbl in common_labels:
            if lbl in result.rigid_label_proj_2d:
                pde = float(np.linalg.norm(result.rigid_label_proj_2d[lbl] - gt_2d[lbl]))
                rigid_pde[lbl] = pde
        result.rigid_pde_per_label = rigid_pde
        vals = list(rigid_pde.values())
        result.rigid_mean_pde = float(np.mean(vals)) if vals else None
        result.rigid_max_pde  = float(np.max(vals))  if vals else None

    # msLevelCheck PDE
    result.pde_per_label = pde_dict
    vals = list(pde_dict.values())
    if vals:
        result.mean_pde    = float(np.mean(vals))
        result.max_pde     = float(np.max(vals))
        result.failure_rate = float(sum(v > 20.0 for v in vals) / len(vals))

    return result


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import discover_sessions

    sessions  = discover_sessions(Path('/home/supermicro/Documents/2D_3D_Raghu/data'))
    pairs     = [(sessions[p]['PREOP'], sessions[p]['POSTOP'], p)
                 for p in sessions if 'PREOP' in sessions[p] and 'POSTOP' in sessions[p]]

    for preop, postop, pname in pairs[:1]:   # test on first patient
        print(f"\nTesting on {pname}...")
        print(f"  C-arm lateral: {preop.carm_lateral}")
        reg = MSLevelCheck(preop, postop, verbose=True)
        result = reg.register(stages_to_run=1)   # Quick test: Stage 1 only

        gen_eval = DRRGenerator(preop.ct_volume, preop.ct_spacing, preop.ct_origin)
        result = compute_pde(result, postop, gen_eval)

        print(result.summary())
