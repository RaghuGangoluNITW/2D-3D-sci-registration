#!/usr/bin/env python3
"""
run_new_patients.py — Run 2D/3D Registration on All New Patients
=================================================================
Processes all patients in data/newdata/ProcessedNew/Processed Data 2D-3D/
Runs EPnP-initialised CMA-ES registration on every runnable set.
Saves per-patient JSON results + overlay PNG visualisations.
Prints a cross-patient comparison table at the end.

Usage:
    python run_new_patients.py                          # all patients
    python run_new_patients.py --patients SARKHI Swaroop
    python run_new_patients.py --geometry_only          # DRR check only
    python run_new_patients.py --verbose

Outputs:
    results/newdata/<PATIENT>/results.json
    results/newdata/<PATIENT>/<SET>_overlay.png
    results/newdata/all_results.json
    results/newdata/comparison_table.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from generic_patient_loader import (
    GenericPatientLoader,
    GenericProjection,
    compute_pde,
    project_world,
    _solve_pnp,
    _reproj_err,
    _anatomy_pose,
    CAM_SID_MM, CAM_PIX_MM, CAM_FX, CAM_FY, CAM_CX, CAM_CY, CAM_IMG_SIZE, CAM_K,
)
from deepfluoro_loader import DeepFluoroSpecimen, perturb_extrinsic
from deepfluoro_drr import DeepFluoroDRR
from similarity import go_cost as compute_go_cost, normalized_cross_correlation
from optimizer import run_cmaes_single
from deformable_registration import (
    compute_deformation,
    compute_deformation_cmaes,
    deformable_pde as compute_deformable_pde,
)

# ---------------------------------------------------------------------------
# Data root
# ---------------------------------------------------------------------------
NEW_DATA_ROOT = Path('data/patients')
RESULTS_ROOT  = Path('results')

# ---------------------------------------------------------------------------
# Render sizes
# ---------------------------------------------------------------------------
COARSE_SIZE = 64
OPT_SIZE    = 180
FINE_SIZE   = 256
COARSE_PIX  = CAM_PIX_MM * (CAM_IMG_SIZE / COARSE_SIZE)
OPT_PIX     = CAM_PIX_MM * (CAM_IMG_SIZE / OPT_SIZE)
FINE_PIX    = CAM_PIX_MM * (CAM_IMG_SIZE / FINE_SIZE)

# Label colours (BGR) for overlay images
LABEL_COLORS = {
    'L1': (0,   255, 0),    # green
    'L2': (255, 128, 0),    # orange
    'L3': (0,   200, 255),  # cyan
    'L4': (0,   0,   255),  # red
    'L5': (255, 0,   255),  # magenta
    'D12':(128, 255, 128),
    'D11':(200, 200, 200),
}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def make_overlay_image(
        xray: np.ndarray,
        drr: np.ndarray,
        proj: GenericProjection,
        pts3d_all: np.ndarray,
        lm_names: List[str],
        R_init: np.ndarray, t_init: np.ndarray,
        R_final: np.ndarray, t_final: np.ndarray,
        init_pde: Dict[str, float],
        final_pde: Dict[str, float],
        patient_id: str,
        set_name: str,
) -> np.ndarray:
    """
    3-panel image:
      Left   : X-ray + EPnP projected labels (initial pose)
      Centre : X-ray + final optimised labels
      Right  : Final DRR with projected labels
    """
    H, W = xray.shape
    def to_bgr(img):
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

    def draw_labels(canvas, R, t, color_key='final', gt=True):
        uv = project_world(pts3d_all, R, t)
        for i, name in enumerate(lm_names):
            col = LABEL_COLORS.get(name, (255, 255, 255))
            pu, pv = int(round(uv[i, 0])), int(round(uv[i, 1]))
            if 0 <= pu < W and 0 <= pv < H:
                cv2.circle(canvas, (pu, pv), 10, col, -1)
                cv2.circle(canvas, (pu, pv), 10, (0, 0, 0), 2)
                cv2.putText(canvas, name, (pu + 12, pv + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            # Draw GT 2D position if available
            if gt and name in proj.gt_landmarks_2d:
                gu, gv = proj.gt_landmarks_2d[name].astype(int)
                if 0 <= gu < W and 0 <= gv < H:
                    cv2.drawMarker(canvas, (gu, gv), (0, 255, 255),
                                   cv2.MARKER_CROSS, 18, 2)

    # Scale DRR to xray size
    drr_rs = cv2.resize(drr, (W, H), interpolation=cv2.INTER_LINEAR)

    left   = to_bgr(xray)
    centre = to_bgr(xray)
    right  = to_bgr(drr_rs)

    draw_labels(left,   R_init,  t_init)
    draw_labels(centre, R_final, t_final)
    draw_labels(right,  R_final, t_final, gt=False)

    def pde_str(pde):
        if pde: return f"{np.mean(list(pde.values())):.1f}mm"
        return "N/A"

    # Header bars
    for img, title in [(left,   f"INITIAL (EPnP)  PDE={pde_str(init_pde)}"),
                       (centre, f"FINAL (CMA-ES)  PDE={pde_str(final_pde)}"),
                       (right,  f"DRR @ final pose")]:
        cv2.rectangle(img, (0, 0), (W, 32), (30, 30, 30), -1)
        cv2.putText(img, f"{patient_id} | {set_name} | {title}",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Legend on left panel
    y = H - 20
    for name, col in list(LABEL_COLORS.items())[:5]:
        cv2.circle(left, (14, y), 6, col, -1)
        cv2.putText(left, name, (24, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        y -= 22
    cv2.drawMarker(left, (14, y), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
    cv2.putText(left, "GT", (24, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return np.hstack([left, centre, right])


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
def _make_objectives(drr_gen, R_gt, t_gt, proj, specimen, lm_weight_scale=1.0):
    from metal_suppression import suppress_metal
    xray = proj.image_raw

    # Compute metal mask once — reused in every cost function call
    _, metal_mask = suppress_metal(xray)
    # Resize masks to each rendering resolution
    def _resize_mask(sz):
        if metal_mask.max() == 0:
            return None   # no metal detected — skip masking
        return cv2.resize(metal_mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
    mmask_c = _resize_mask(COARSE_SIZE)
    mmask_o = _resize_mask(OPT_SIZE)
    mmask_f = _resize_mask(FINE_SIZE)

    tgt_c = cv2.resize(xray, (COARSE_SIZE, COARSE_SIZE), interpolation=cv2.INTER_AREA)
    tgt_o = cv2.resize(xray, (OPT_SIZE,    OPT_SIZE),    interpolation=cv2.INTER_AREA)
    tgt_f = cv2.resize(xray, (FINE_SIZE,   FINE_SIZE),   interpolation=cv2.INTER_AREA)

    lm_names  = sorted(specimen.landmarks_3d.keys())
    pts3d_all = np.array([specimen.landmarks_3d[n] for n in lm_names])
    ann_names = [n for n in lm_names if n in proj.gt_landmarks_2d]
    ann_idx   = [lm_names.index(n) for n in ann_names]
    pts3d_ann = pts3d_all[ann_idx]
    pts2d_ann = np.array([proj.gt_landmarks_2d[n] for n in ann_names])
    diag = float(np.sqrt(2)) * CAM_IMG_SIZE

    def _lm(R, t):
        if len(pts3d_ann) == 0: return 0.0
        uv = project_world(pts3d_ann, R, t)
        return float(np.clip(np.mean(np.linalg.norm(uv - pts2d_ann, axis=1)) / diag, 0, 1))

    def _ncc(drr, tgt, mmask=None):
        if np.count_nonzero(drr > 0.01) / drr.size < 0.08: return 1.0
        return float(1.0 - normalized_cross_correlation(1.0 - drr, tgt, metal_mask=mmask))

    def _ng(drr, tgt, mmask=None):
        if np.count_nonzero(drr > 0.01) / drr.size < 0.08: return 1.0
        return 0.5 * float(1.0 - normalized_cross_correlation(1.0 - drr, tgt, metal_mask=mmask)) \
             + 0.5 * float(compute_go_cost(drr, tgt, metal_mask=mmask))

    def obj(cf, tgt, sz, px, ns, lw, mmask):
        def f(x):
            Rc, tc = perturb_extrinsic(R_gt, t_gt, x[:3], x[3:])
            d = drr_gen.generate_from_extrinsic(Rc, tc, sz, px, ns)
            return (1 - lw) * cf(d, tgt, mmask) + lw * _lm(Rc, tc)
        return f

    lws = lm_weight_scale
    oc = obj(_ncc, tgt_c, COARSE_SIZE, COARSE_PIX, 40,  0.4*lws, mmask_c)
    op = obj(_ng,  tgt_o, OPT_SIZE,    OPT_PIX,    80,  0.3*lws, mmask_o)
    of = obj(_ncc, tgt_f, FINE_SIZE,   FINE_PIX,   120, 0.2*lws, mmask_f)

    def go_eval(R, t):
        d = drr_gen.generate_from_extrinsic(R, t, OPT_SIZE, OPT_PIX, 80)
        return float(compute_go_cost(d, tgt_o, metal_mask=mmask_o))

    return oc, op, of, go_eval


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def run_registration(drr_gen, proj: GenericProjection,
                     specimen: DeepFluoroSpecimen,
                     rot_deg=10.0, trans_mm=20.0,
                     n_starts=16, verbose=False,
                     deform_cmaes=False,
                     lm_weight_scale=1.0) -> dict:

    R_gt = proj.R_proj.copy()
    t_gt = proj.t_proj.copy()
    lm_names = sorted(specimen.landmarks_3d.keys())
    pts3d    = np.array([specimen.landmarks_3d[n] for n in lm_names])

    pde_init = compute_pde(proj, R_gt, t_gt, pts3d, lm_names)
    init_pde_mean = float(np.mean(list(pde_init.values()))) if pde_init else float('nan')

    oc, op, of, go_eval = _make_objectives(drr_gen, R_gt, t_gt, proj, specimen, lm_weight_scale=lm_weight_scale)
    init_go = go_eval(R_gt, t_gt)

    if verbose: print(f"    init PDE={init_pde_mean:.2f}mm  GO={init_go:.4f}")

    sr = np.array([rot_deg]*3 + [trans_mm]*3)
    rng = np.random.default_rng(7 + proj.proj_index)
    grid = np.vstack([np.zeros(6),
                      (rng.random((n_starts-1, 6))*2 - 1) * sr])
    gc = np.array([oc(x) for x in grid])
    top = grid[np.argsort(gc)[:n_starts]]

    best_pose, best_cost = np.zeros(6), op(np.zeros(6))
    for x0 in top:
        r = run_cmaes_single(op, x0, sr/3, bounds_center=np.zeros(6),
                             popsize=16, tolx=0.2, maxiter=300)
        if r.cost < best_cost:
            best_cost, best_pose = r.cost, r.pose.copy()

    r3 = run_cmaes_single(of, best_pose, sr/8, bounds_center=np.zeros(6),
                          popsize=16, tolx=0.05, maxiter=300)
    best_x = r3.pose if r3.cost < of(best_pose) else best_pose

    # Safety net
    Rc, tc = perturb_extrinsic(R_gt, t_gt, best_x[:3], best_x[3:])
    pde_c  = compute_pde(proj, Rc, tc, pts3d, lm_names)
    mean_c = float(np.mean(list(pde_c.values()))) if pde_c else float('inf')
    thr    = (init_pde_mean * 1.5) if (not np.isnan(init_pde_mean) and init_pde_mean > 0.1) else 3.0
    if mean_c > thr:
        if verbose: print(f"    safety net: {mean_c:.1f}mm>{thr:.1f}mm → EPnP")
        best_x = np.zeros(6)

    R_fin, t_fin = perturb_extrinsic(R_gt, t_gt, best_x[:3], best_x[3:])
    final_go  = go_eval(R_fin, t_fin)
    pde_final = compute_pde(proj, R_fin, t_fin, pts3d, lm_names)
    final_pde_mean = float(np.mean(list(pde_final.values()))) if pde_final else float('nan')
    success = final_pde_mean < 15.0

    # ── Deformable stage ────────────────────────────────────────────────────
    # For each labeled vertebra, backproject its observed 2D centroid through
    # the final rigid camera pose to find its corrected 3D intraop position.
    # The offset (deformation vector) quantifies per-vertebra spinal deformation
    # between preop CT and intraop position.
    if deform_cmaes and hasattr(proj, 'drr_generator') and proj.drr_generator is not None:
        deformations = compute_deformation_cmaes(
            R_fin, t_fin,
            pts3d, lm_names,
            proj.gt_landmarks_2d,
            drr_generator=proj.drr_generator,
            xray_image=proj.image_raw,
            fx=CAM_FX, fy=CAM_FY, cx=CAM_CX, cy=CAM_CY,
            pix_mm=CAM_PIX_MM,
            verbose=verbose,
        )
    else:
        deformations = compute_deformation(
            R_fin, t_fin,
            pts3d, lm_names,
            proj.gt_landmarks_2d,
            CAM_FX, CAM_FY, CAM_CX, CAM_CY,
            pix_mm=CAM_PIX_MM,
        )
    deform_pde_dict = compute_deformable_pde(
        R_fin, t_fin,
        deformations,
        pts3d, lm_names,
        proj.gt_landmarks_2d,
        CAM_FX, CAM_FY, CAM_CX, CAM_CY,
        pix_mm=CAM_PIX_MM,
    )
    deform_pde_mean = (
        float(np.mean(list(deform_pde_dict.values()))) if deform_pde_dict else float('nan')
    )
    deform_magnitudes = {k: round(v.magnitude_mm, 3) for k, v in deformations.items()}
    deform_vectors    = {k: v.to_dict() for k, v in deformations.items()}

    if verbose and deform_magnitudes:
        mag_str = '  '.join(f"{k}:{v:.1f}mm" for k, v in sorted(deform_magnitudes.items()))
        print(f"    deform: {mag_str}  → deformable PDE={deform_pde_mean:.3f}mm")

        print(f"    final PDE={final_pde_mean:.2f}mm  GO={final_go:.4f}  "
              f"ΔGO={init_go-final_go:+.4f}  {'OK' if success else 'FAIL'}")

    return dict(
        proj_key=proj.proj_key,
        initial_pde_mm=init_pde_mean,
        final_pde_mm=final_pde_mean,
        deform_pde_mm=deform_pde_mean,
        initial_go=init_go,
        final_go=final_go,
        go_delta=init_go - final_go,
        success=success,
        pde_per_landmark=pde_final,
        best_pose_delta=best_x.tolist(),
        deform_magnitudes=deform_magnitudes,
        deform_vectors=deform_vectors,
        R_init=R_gt, t_init=t_gt,
        R_final=R_fin, t_final=t_fin,
        pts3d=pts3d,
        lm_names=lm_names,
        pde_init_dict=pde_init,
        pde_final_dict=pde_final,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--patients', nargs='+', default=None)
    p.add_argument('--geometry_only', action='store_true')
    p.add_argument('--rot',     type=float, default=10.0)
    p.add_argument('--trans',   type=float, default=20.0)
    p.add_argument('--starts',  type=int,   default=16)
    p.add_argument('--min_labels', type=int, default=3)
    p.add_argument('--camera',  type=str,   default='ziehm_vision_fd',
                   help='Camera config name from configs/cameras/. '
                        'Default: ziehm_vision_fd. '
                        'Other options: siemens_cios_spin, ge_oec_9900')
    p.add_argument('--deform_cmaes', action='store_true',
                   help='Use per-vertebra CMA-ES deformation (Ketcha 2017b) '
                        'instead of geometric backprojection.')
    p.add_argument('--auto_landmarks', action='store_true',
                   help='Use the trained CNN to predict 2D landmarks instead of '
                        'reading CENTROID.mrk.json. Falls back to file labels '
                        'if detector confidence is low.')
    p.add_argument('--lm_conf', type=float, default=0.15,
                   help='Min heatmap confidence to accept a CNN prediction (default 0.15).')
    p.add_argument('--ssm_landmarks', action='store_true',
                   help='Use SSM mean shape (no CNN) to initialise 2D landmarks. '
                        'Works fully annotation-free: mean spine centred on image.')
    p.add_argument('--ssm_reg', action='store_true',
                   help='Use trained SSM regressor to predict 2D landmarks. '
                        'Fully annotation-free: CNN regresses 8 SSM params per image.')
    p.add_argument('--ssm_path', type=str, default=None,
                   help='Path to SSM .pkl used by --ssm_reg (default: spine_ssm_buu.pkl if exists, else spine_ssm.pkl).')
    p.add_argument('--ckpt_dir', type=str, default=None,
                   help='Checkpoint directory for SSM regressor AP folds '
                        '(default: checkpoints/ssm_regressor_ap/). '
                        'Use --lat_ckpt_dir to set the LAT directory separately.')
    p.add_argument('--lat_ckpt_dir', type=str, default=None,
                   help='Checkpoint directory for SSM regressor LAT folds '
                        '(default: checkpoints/ssm_regressor_lat/).')
    p.add_argument('--heatmap_reg', action='store_true',
                   help='Use the trained heatmap-based LandmarkHeatmapNet '
                        '(scripts/train_landmark_detector.py) to predict raw '
                        '2D landmark peaks, per-patient/per-view fold '
                        'checkpoints. Fully annotation-free alternative to '
                        '--ssm_reg. Uses raw CNN argmax peaks (no SSM '
                        'completion layer) so this tests the detector itself.')
    p.add_argument('--heatmap_ckpt_dir', type=str, default=None,
                   help='Checkpoint directory for heatmap detector AP folds '
                        '(default: checkpoints/landmark_detector_ap/).')
    p.add_argument('--heatmap_lat_ckpt_dir', type=str, default=None,
                   help='Checkpoint directory for heatmap detector LAT folds '
                        '(default: checkpoints/landmark_detector_lat/).')
    p.add_argument('--pose_reg', action='store_true',
                   help='Use the patient-specific direct 6-DOF PoseRegressor '
                        '(scripts/train_pose_regressor.py, checkpoints/'
                        'pose_regressor/<PATIENT>.pth) to predict the initial '
                        'pose directly from the X-ray image -- no 2D '
                        'landmarks, no EPnP. Built to sidestep the proven '
                        'EPnP-collinearity instability of --ssm_reg/'
                        '--heatmap_reg. Requires a checkpoint already trained '
                        'for this specific patient.')
    p.add_argument('--pose_reg_ckpt_dir', type=str, default=None,
                   help='Checkpoint directory for PoseRegressor '
                        '(default: checkpoints/pose_regressor/).')
    p.add_argument('--anatomy_pose', action='store_true',
                   help='Seed the initial pose from a coarse anatomy-based '
                        'estimate (view-angle heuristic, no CNN landmarks '
                        'needed) instead of CNN->EPnP. Test of whether a '
                        'stable-but-coarse start beats an unstable-but-'
                        'precise-looking EPnP pose from near-collinear '
                        'vertebral centroids. Overrides --ssm_reg/--heatmap_reg '
                        'seeding if both are given.')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Discover patients
    patient_dirs = sorted([p for p in NEW_DATA_ROOT.iterdir() if p.is_dir()])
    if args.patients:
        patient_dirs = [p for p in patient_dirs if p.name in args.patients]

    from camera_config import load_camera_config, list_cameras
    cam = load_camera_config(args.camera)

    # Load SSM regressor once if --ssm_reg requested
    _ssm_reg_detectors = {}  # patient -> SSMRegressorDetector (loaded per-fold lazily)
    if args.ssm_reg:
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent / 'src'))
            from landmark_detector import SSMRegressorDetector
            print('  SSM-reg: SSMRegressorDetector ready (fold checkpoints will load per patient)')
        except Exception as e:
            print(f'  [WARNING] SSMRegressorDetector load failed: {e}')
            args.ssm_reg = False

    # Load SSM once if --ssm_landmarks requested
    _ssm = None
    if args.ssm_landmarks:
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent / 'src'))
            from spine_ssm import SpineSSM
            _ssm_path = Path(__file__).parent / 'checkpoints' / 'spine_ssm.pkl'
            _ssm = SpineSSM.load(str(_ssm_path))
            print(f'  SSM-landmarks: loaded {_ssm_path.name}  ({_ssm.n_modes} modes)')
        except Exception as e:
            print(f'  [WARNING] SSM load failed: {e}  — falling back to file labels.')

    # Load landmark detector once if requested
    _detector = None
    if args.auto_landmarks:
        try:
            from landmark_detector import LandmarkDetector
            _detector = LandmarkDetector(conf_threshold=args.lm_conf)
            print(f'  Auto-landmarks: CNN detector loaded (conf≥{args.lm_conf})')
        except FileNotFoundError as e:
            print(f'  [WARNING] {e}')
            print('  Falling back to CENTROID.mrk.json labels for all sets.')

    print(f"\n{'='*65}")
    print(f"  NEW PATIENT PIPELINE  —  {len(patient_dirs)} patients")
    print(f"  Data  : {NEW_DATA_ROOT}")
    print(f"  Out   : {RESULTS_ROOT}")
    print(f"  Camera: {cam}")
    print(f"{'='*65}")

    all_patient_results = {}
    t0_global = time.time()

    for patient_dir in patient_dirs:
        pid = patient_dir.name
        print(f"\n{'─'*65}")
        print(f"  PATIENT: {pid}")
        print(f"{'─'*65}")

        out_dir = RESULTS_ROOT / pid
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            loader = GenericPatientLoader(patient_dir, camera=args.camera)
            spec   = loader.load(min_labels=args.min_labels, verbose=True)
        except Exception as e:
            print(f"  ❌ Load failed: {e}")
            continue

        drr_gen = DeepFluoroDRR(spec, hu_threshold=150.0)

        if args.geometry_only:
            lm_names = sorted(spec.landmarks_3d.keys())
            pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
            for proj in spec.projections:
                pde = compute_pde(proj, proj.R_proj, proj.t_proj, pts3d, lm_names)
                m   = np.mean(list(pde.values())) if pde else float('inf')
                drr = drr_gen.generate_from_extrinsic(proj.R_proj, proj.t_proj, 64, COARSE_PIX, 40)
                ok  = drr.max() > 0.01
                print(f"  {proj.proj_key}: reproj={proj.reproj_error_px:.2f}px  "
                      f"PDE@init={m:.2f}mm  DRR={'OK' if ok else 'EMPTY'}")
            continue

        patient_results = []

        # ── SSM regressor landmark prediction (Option A) ──────────────────
        # Uses VIEW-SPECIFIC SSM + checkpoint per projection (AP vs LAT),
        # since a single mixed AP+LAT SSM/regressor was found to corrupt the
        # mean_scale calibration (see checkpoints/spine_ssm_ap.pkl vs
        # spine_ssm_lat.pkl, built to fix this). proj.proj_key (the set
        # name, e.g. 'AP-SET-1' / 'LAT-SET-1') is used to pick the view.
        #
        # IMPORTANT: this computes a genuine CNN-seeded initial pose via
        # EPnP on the CNN's 2D predictions, and overwrites proj.R_proj /
        # proj.t_proj (the anchor pose that run_registration() perturbs
        # around) with it. proj.gt_landmarks_2d (the TRUE annotated 2D
        # labels) is left untouched, since compute_pde() always scores
        # against gt_landmarks_2d regardless of R_proj/t_proj -- this is
        # what makes init_pde/final_pde a genuine measure of automatic
        # pipeline accuracy (CNN -> EPnP -> CMA-ES) instead of accidentally
        # scoring the already-correct file-label pose against itself.
        if args.ssm_reg:
            try:
                from landmark_detector import SSMRegressorDetector

                def _load_detector(view: str) -> Optional['SSMRegressorDetector']:
                    """Load the view-specific SSM regressor for this patient,
                    or None if no fold checkpoint exists for this patient+view."""
                    default_dir = (Path(__file__).parent / 'checkpoints' /
                                   f'ssm_regressor_{view.lower()}')
                    override = args.ckpt_dir if view == 'AP' else args.lat_ckpt_dir
                    ckpt_dir = Path(override) if override else default_dir
                    if not ckpt_dir.is_absolute():
                        ckpt_dir = Path(__file__).parent / ckpt_dir
                    fold_ckpt = ckpt_dir / f'fold_{pid}.pth'
                    if not fold_ckpt.exists():
                        return None
                    default_ssm = (Path(__file__).parent / 'checkpoints' /
                                   f'spine_ssm_{view.lower()}.pkl')
                    ssm_path = args.ssm_path or (str(default_ssm) if default_ssm.exists() else None)
                    return SSMRegressorDetector(
                        checkpoint=str(fold_ckpt), ssm_path=ssm_path,
                        patient=pid, view=view, device='cuda')

                _detectors_by_view = {}
                n_seeded  = 0
                n_skipped = 0
                for proj in spec.projections:
                    view = 'AP' if 'AP' in proj.proj_key.upper() else 'LAT'
                    if view not in _detectors_by_view:
                        try:
                            _detectors_by_view[view] = _load_detector(view)
                        except Exception as e:
                            print(f'  [WARNING] SSM-reg: failed to load {view} '
                                  f'detector for {pid}: {e}')
                            _detectors_by_view[view] = None
                    det_ssm = _detectors_by_view[view]
                    if det_ssm is None:
                        print(f'    {proj.proj_key}: no {view} fold checkpoint '
                              f'for {pid} — keeping file-label EPnP anchor')
                        n_skipped += 1
                        continue
                    try:
                        pred, _ = det_ssm.predict(proj.image_raw, return_conf=True)
                        # Solve EPnP from the CNN's 2D guesses + known 3D
                        # centroids -- this is the actual "automatic" initial
                        # pose a real deployment would compute, since no
                        # manual/true 2D labels are available at inference
                        # time. gt_landmarks_2d is NOT touched here.
                        common = sorted(n for n in pred if n in spec.landmarks_3d)
                        if len(common) >= 3:
                            pts3d_cnn = np.array([spec.landmarks_3d[n] for n in common])
                            pts2d_cnn = np.array([pred[n] for n in common])
                            R_cnn, t_cnn = _solve_pnp(pts3d_cnn, pts2d_cnn, CAM_K)
                            proj.R_proj = R_cnn
                            proj.t_proj = t_cnn
                            proj.reproj_error_px = _reproj_err(
                                pts3d_cnn, pts2d_cnn, R_cnn, t_cnn)
                            n_seeded += 1
                        else:
                            print(f'    {proj.proj_key}: CNN gave <3 usable '
                                  f'landmarks ({len(common)}) — cannot seed EPnP, '
                                  f'keeping file-label anchor')
                    except Exception as e:
                        print(f'    {proj.proj_key}: SSM-reg failed ({e})')
                print(f'  SSM-reg: seeded {n_seeded} CNN->EPnP initial pose(s) '
                      f'across {len(spec.projections)} projections  '
                      f'({n_skipped} skipped, no checkpoint)')
            except Exception as e:
                print(f'  [WARNING] SSM-reg: {e} — falling back to file-label anchor')

        # ── Heatmap detector landmark prediction (Option B) ────────────────
        # Same CNN->EPnP seeding pattern as --ssm_reg above, but using raw
        # LandmarkHeatmapNet argmax peaks (scripts/train_landmark_detector.py)
        # instead of SSM-parameter regression. No SSM completion layer here
        # -- only the raw peaks with confidence above --lm_conf are used, so
        # this measures the heatmap detector itself, matching how it was
        # evaluated during training (evaluate() in the training script).
        if args.heatmap_reg:
            try:
                from landmark_model import LandmarkHeatmapNet
                from landmark_dataset import CLASS_NAMES, HMAP_SIZE

                def _load_heatmap_detector(view: str):
                    default_dir = (Path(__file__).parent / 'checkpoints' /
                                   f'landmark_detector_{view.lower()}')
                    override = (args.heatmap_ckpt_dir if view == 'AP'
                               else args.heatmap_lat_ckpt_dir)
                    ckpt_dir = Path(override) if override else default_dir
                    if not ckpt_dir.is_absolute():
                        ckpt_dir = Path(__file__).parent / ckpt_dir
                    fold_ckpt = ckpt_dir / f'fold_{pid}.pth'
                    if not fold_ckpt.exists():
                        return None
                    ckpt = torch.load(str(fold_ckpt), map_location='cuda',
                                      weights_only=False)
                    m = LandmarkHeatmapNet(pretrained=False, dropout=0.0).to('cuda')
                    m.load_state_dict(ckpt['model_state'])
                    m.eval()
                    val_err = ckpt.get('val_err_mm', float('nan'))
                    print(f'  [HeatmapDetector] {fold_ckpt.name}  '
                          f'(val_err={val_err:.1f}mm)')
                    return m

                @torch.no_grad()
                def _predict_heatmap(m, xray):
                    orig_h, orig_w = xray.shape[:2]
                    img = xray.astype(np.float32)
                    lo, hi = img.min(), img.max()
                    if hi > lo:
                        img = (img - lo) / (hi - lo)
                    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
                        img_u8).astype(np.float32) / 255.0
                    img_rs = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                    t = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to('cuda')
                    logits = m(t)
                    hmap = logits.sigmoid()[0].cpu().numpy()
                    sx, sy = orig_w / HMAP_SIZE, orig_h / HMAP_SIZE
                    coords, confs = {}, {}
                    for i, name in enumerate(CLASS_NAMES):
                        ch = hmap[i]
                        peak = float(ch.max())
                        confs[name] = peak
                        if peak < args.lm_conf:
                            continue
                        flat_idx = int(ch.argmax())
                        hx = (flat_idx % HMAP_SIZE + 0.5) * sx
                        hy = (flat_idx // HMAP_SIZE + 0.5) * sy
                        coords[name] = np.array([hx, hy], dtype=np.float64)
                    return coords, confs

                _hm_detectors_by_view = {}
                n_seeded  = 0
                n_skipped = 0
                for proj in spec.projections:
                    view = 'AP' if 'AP' in proj.proj_key.upper() else 'LAT'
                    if view not in _hm_detectors_by_view:
                        try:
                            _hm_detectors_by_view[view] = _load_heatmap_detector(view)
                        except Exception as e:
                            print(f'  [WARNING] heatmap-reg: failed to load {view} '
                                  f'detector for {pid}: {e}')
                            _hm_detectors_by_view[view] = None
                    m = _hm_detectors_by_view[view]
                    if m is None:
                        print(f'    {proj.proj_key}: no {view} fold checkpoint '
                              f'for {pid} — keeping file-label EPnP anchor')
                        n_skipped += 1
                        continue
                    try:
                        pred, confs = _predict_heatmap(m, proj.image_raw)
                        common = sorted(n for n in pred if n in spec.landmarks_3d)
                        if len(common) >= 3:
                            pts3d_cnn = np.array([spec.landmarks_3d[n] for n in common])
                            pts2d_cnn = np.array([pred[n] for n in common])
                            R_cnn, t_cnn = _solve_pnp(pts3d_cnn, pts2d_cnn, CAM_K)
                            proj.R_proj = R_cnn
                            proj.t_proj = t_cnn
                            proj.reproj_error_px = _reproj_err(
                                pts3d_cnn, pts2d_cnn, R_cnn, t_cnn)
                            n_seeded += 1
                            if args.verbose:
                                conf_str = '  '.join(
                                    f'{k}={confs.get(k,0):.2f}' for k in pred)
                                print(f'    {proj.proj_key}: heatmap-reg '
                                      f'({len(pred)} peaks above conf) [{conf_str}]')
                        else:
                            print(f'    {proj.proj_key}: heatmap detector gave <3 '
                                  f'confident landmarks ({len(common)}) — cannot '
                                  f'seed EPnP, keeping file-label anchor')
                    except Exception as e:
                        print(f'    {proj.proj_key}: heatmap-reg failed ({e})')
                print(f'  heatmap-reg: seeded {n_seeded} CNN->EPnP initial pose(s) '
                      f'across {len(spec.projections)} projections  '
                      f'({n_skipped} skipped, no checkpoint)')
            except Exception as e:
                print(f'  [WARNING] heatmap-reg: {e} — falling back to file-label anchor')

        # ── Direct pose regression (Option E) ──────────────────────────────
        # Patient-specific 6-DOF PoseRegressor (src/pose_regressor.py):
        # predicts a pose PERTURBATION directly from the image, relative to
        # the same anatomy-pose reference frame used during training
        # (see scripts/train_pose_regressor.py -- VIEW_AZIMUTHS/
        # _anatomy_pose_random, which is numerically identical to
        # generic_patient_loader._anatomy_pose already imported above).
        # No 2D landmarks, no EPnP -- this is the approach built to sidestep
        # the proven EPnP-collinearity instability of --ssm_reg/--heatmap_reg.
        if args.pose_reg:
            try:
                from pose_regressor import PoseRegressor, decode_pose

                def _load_pose_model():
                    default_dir = Path(__file__).parent / 'checkpoints' / 'pose_regressor'
                    ckpt_dir = Path(args.pose_reg_ckpt_dir) if args.pose_reg_ckpt_dir else default_dir
                    if not ckpt_dir.is_absolute():
                        ckpt_dir = Path(__file__).parent / ckpt_dir
                    ckpt_path = ckpt_dir / f'{pid}.pth'
                    if not ckpt_path.exists():
                        return None
                    ckpt = torch.load(str(ckpt_path), map_location='cuda', weights_only=False)
                    m = PoseRegressor(backbone_name=ckpt.get('backbone_name', 'resnet34'),
                                      pretrained=False).to('cuda')
                    m.load_state_dict(ckpt['model_state'])
                    m.eval()
                    val_pde = ckpt.get('val_pde_mm', float('nan'))
                    print(f'  [PoseRegressor] {ckpt_path.name}  (val_pde={val_pde:.1f}mm, '
                          f'synthetic-only)')
                    return m

                @torch.no_grad()
                def _predict_pose(m, xray):
                    img = xray.astype(np.float32)
                    lo, hi = img.min(), img.max()
                    if hi > lo:
                        img = (img - lo) / (hi - lo)
                    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
                        img_u8).astype(np.float32) / 255.0
                    img_rs = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                    t = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to('cuda')
                    pred_norm = m(t)[0].cpu().numpy()
                    return decode_pose(pred_norm)

                pose_model = _load_pose_model()
                n_seeded = 0
                if pose_model is None:
                    print(f'  [WARNING] pose-reg: no checkpoint for {pid} at '
                          f'checkpoints/pose_regressor/{pid}.pth — falling back '
                          f'to file-label anchor')
                else:
                    for proj in spec.projections:
                        try:
                            # decode_pose() now returns the FULL absolute
                            # (R, t) directly -- no reference-pose delta
                            # involved (see pose_regressor.py docstring for
                            # why the earlier delta-from-one-reference
                            # parameterisation was replaced).
                            R_pred, t_pred = _predict_pose(pose_model, proj.image_raw)
                            proj.R_proj = R_pred
                            proj.t_proj = t_pred
                            proj.reproj_error_px = float('inf')  # not meaningful (no PnP)
                            n_seeded += 1
                            if args.verbose:
                                print(f'    {proj.proj_key}: pose-reg pred '
                                      f't={t_pred.round(1)}mm')
                        except Exception as e:
                            print(f'    {proj.proj_key}: pose-reg failed ({e})')
                    print(f'  pose-reg: seeded {n_seeded} direct-pose initial pose(s) '
                          f'across {len(spec.projections)} projections (no EPnP)')
            except Exception as e:
                print(f'  [WARNING] pose-reg: {e} — falling back to file-label anchor')

        # ── Coarse anatomy-pose seeding (Option D) ─────────────────────────
        # No CNN landmarks needed at all -- uses only the view tag (AP/LAT)
        # to pick a viewing azimuth and centers the pose on the 3D landmark
        # centroid. Deliberately coarse/generic, but geometrically STABLE
        # (does not depend on solving an ill-conditioned PnP from
        # near-collinear vertebral centroids the way CNN->EPnP does).
        # Test: does a stable-but-coarse seed let CMA-ES converge better
        # than an unstable-but-precise-looking CNN->EPnP seed?
        if args.anatomy_pose:
            n_seeded = 0
            for proj in spec.projections:
                view = 'AP' if 'AP' in proj.proj_key.upper() else 'LAT'
                az = 0.0 if view == 'AP' else 90.0
                R_a, t_a = _anatomy_pose(spec.landmarks_3d, az)
                proj.R_proj = R_a
                proj.t_proj = t_a
                proj.reproj_error_px = float('inf')  # not meaningful here
                n_seeded += 1
            print(f'  anatomy-pose: seeded {n_seeded} coarse initial pose(s) '
                  f'across {len(spec.projections)} projections (no CNN)')

        # ── SSM mean-shape landmark initialisation (Option C) ─────────────
        if args.ssm_landmarks and _ssm is not None:
            n_ssm_filled = 0
            for proj in spec.projections:
                try:
                    h, w = proj.image_raw.shape[:2]
                    ssm_lms = _ssm.fit_and_complete({}, img_hw=(h, w))
                    for name, uv in ssm_lms.items():
                        proj.gt_landmarks_2d[name] = np.array(uv, dtype=np.float64)
                        n_ssm_filled += 1
                except Exception as e:
                    print(f'    {proj.proj_key}: SSM landmark init failed ({e})')
            print(f'  SSM-landmarks: set {n_ssm_filled} landmark(s) '
                  f'across {len(spec.projections)} projections (mean shape)')

        # ── Auto-landmark detection ────────────────────────────────────────
        if args.auto_landmarks:
            if _detector is None:
                print('  [WARNING] --auto_landmarks: detector not loaded, '
                      'falling back to file labels.')
            else:
                n_replaced = 0
                for proj in spec.projections:
                    try:
                        pred, confs = _detector.predict(
                            proj.image_raw, return_conf=True)
                        # Replace GT landmarks with CNN predictions where confident
                        for name, uv in pred.items():
                            if confs.get(name, 0) >= args.lm_conf:
                                proj.gt_landmarks_2d[name] = uv
                                n_replaced += 1
                        if args.verbose:
                            conf_str = '  '.join(
                                f'{k}={confs.get(k,0):.2f}' for k in pred)
                            print(f'    {proj.proj_key}: CNN landmarks '
                                  f'({len(pred)} detected) [{conf_str}]')
                    except Exception as e:
                        print(f'    {proj.proj_key}: CNN detection failed '
                              f'({e}), using file labels')
                if n_replaced:
                    print(f'  Auto-landmarks: replaced {n_replaced} '
                          f'landmark(s) with CNN predictions')

        for i, proj in enumerate(spec.projections):
            print(f"\n  [{i+1}/{len(spec.projections)}] {proj.proj_key} ...", flush=True)
            t0 = time.time()
            res = run_registration(drr_gen, proj, spec,
                                   rot_deg=args.rot, trans_mm=args.trans,
                                   n_starts=args.starts, verbose=args.verbose,
                                   deform_cmaes=args.deform_cmaes,
                                   lm_weight_scale=0.0 if (args.ssm_landmarks or args.ssm_reg or args.heatmap_reg or args.anatomy_pose or args.pose_reg) else 1.0)
            res['runtime_s'] = time.time() - t0

            print(f"  → PDE: {res['initial_pde_mm']:.2f}→{res['final_pde_mm']:.2f}mm  "
                  f"(deform={res['deform_pde_mm']:.3f}mm)  "
                  f"GO: {res['initial_go']:.4f}→{res['final_go']:.4f}  "
                  f"ΔGO={res['go_delta']:+.4f}  "
                  f"{'[OK]' if res['success'] else '[FAIL]'}  ({res['runtime_s']:.1f}s)")

            # Generate DRR at final pose for visualisation
            drr_vis = drr_gen.generate_from_extrinsic(
                res['R_final'], res['t_final'], OPT_SIZE, OPT_PIX, 80)

            overlay = make_overlay_image(
                proj.image_raw, drr_vis, proj,
                res['pts3d'], res['lm_names'],
                res['R_init'], res['t_init'],
                res['R_final'], res['t_final'],
                res['pde_init_dict'], res['pde_final_dict'],
                pid, proj.proj_key,
            )
            img_path = out_dir / f"{proj.proj_key}_overlay.png"
            cv2.imwrite(str(img_path), overlay)
            print(f"  Saved overlay → {img_path.relative_to(Path('.'))}")

            patient_results.append(res)

        # Per-patient summary
        ipdes  = [r['initial_pde_mm'] for r in patient_results if not np.isnan(r['initial_pde_mm'])]
        fpdes  = [r['final_pde_mm']   for r in patient_results if not np.isnan(r['final_pde_mm'])]
        igos   = [r['initial_go']     for r in patient_results]
        fgos   = [r['final_go']       for r in patient_results]
        succ   = sum(r['success'] for r in patient_results)

        print(f"\n  {pid} SUMMARY: "
              f"PDE {np.mean(ipdes):.2f}→{np.mean(fpdes):.2f}mm  "
              f"GO {np.mean(igos):.4f}→{np.mean(fgos):.4f}  "
              f"Success {succ}/{len(patient_results)}")

        # Save per-patient JSON
        per_proj_json = {}
        for r in patient_results:
            per_proj_json[r['proj_key']] = {
                k: v for k, v in r.items()
                if k not in ('R_init','t_init','R_final','t_final','pts3d',
                             'lm_names','pde_init_dict','pde_final_dict')
            }

        # Deformation summary across sets for this patient
        all_deform_mags: dict = {}
        for r in patient_results:
            for lm, mag in r.get('deform_magnitudes', {}).items():
                all_deform_mags.setdefault(lm, []).append(mag)
        deform_summary_pat = {
            lm: dict(mean=round(float(np.mean(v)), 3),
                     std=round(float(np.std(v)), 3),
                     max=round(float(np.max(v)), 3))
            for lm, v in all_deform_mags.items()
        }
        mean_deform_pde = float(np.mean(
            [r['deform_pde_mm'] for r in patient_results
             if not np.isnan(r['deform_pde_mm'])]
        ))

        if deform_summary_pat:
            mag_str = '  '.join(
                f"{k}={v['mean']:.1f}±{v['std']:.1f}mm"
                for k, v in sorted(deform_summary_pat.items())
            )
            print(f"  Deformation: {mag_str}")
            print(f"  Mean deformable PDE: {mean_deform_pde:.4f}mm (expected ≈0 for labeled verts)")

        patient_json = {
            'specimen_id':         pid,
            'n_projections':       len(patient_results),
            'mean_initial_pde_mm': float(np.mean(ipdes)) if ipdes else None,
            'mean_final_pde_mm':   float(np.mean(fpdes)) if fpdes else None,
            'mean_initial_go':     float(np.mean(igos))  if igos  else None,
            'mean_final_go':       float(np.mean(fgos))  if fgos  else None,
            'mean_go_delta':       float(np.mean(igos)-np.mean(fgos)) if igos else None,
            'success_rate':        float(succ / len(patient_results)) if patient_results else 0.0,
            'per_projection':      per_proj_json,
        }
        (out_dir / 'results.json').write_text(json.dumps(patient_json, indent=2, default=str))
        all_patient_results[pid] = patient_json

    if args.geometry_only:
        return

    # ── Load Arjun baseline ──────────────────────────────────────────────────
    arjun_json = Path('results/arjun_results.json')
    if arjun_json.exists():
        arjun_data = json.loads(arjun_json.read_text())
        all_patient_results['ARJUN (baseline)'] = arjun_data['arjun']

    # ── Save combined JSON ────────────────────────────────────────────────────
    (RESULTS_ROOT / 'all_results.json').write_text(
        json.dumps(all_patient_results, indent=2, default=str))

    # ── Print comparison table ────────────────────────────────────────────────
    lines = []
    lines.append(f"\n{'='*75}")
    lines.append(f"  CROSS-PATIENT RESULTS COMPARISON")
    lines.append(f"{'='*75}")
    lines.append(f"  {'Patient':<22} {'N':>3}  {'Init PDE':>10} {'Final PDE':>10} "
                 f"{'Init GO':>9} {'Final GO':>9} {'ΔGO':>8}  {'Rate':>6}")
    lines.append(f"  {'-'*72}")

    all_fpde = []; all_igo = []; all_fgo = []
    for pid, d in all_patient_results.items():
        n      = d['n_projections']
        ipde   = d['mean_initial_pde_mm']
        fpde   = d['mean_final_pde_mm']
        igo    = d['mean_initial_go']
        fgo    = d['mean_final_go']
        god    = d['mean_go_delta']
        sr     = d['success_rate'] * 100
        ipde_s = f"{ipde:.2f}mm" if ipde else "N/A"
        fpde_s = f"{fpde:.2f}mm" if fpde else "N/A"
        god_s  = f"{god:+.4f}"  if god  else "N/A"
        lines.append(f"  {pid:<22} {n:>3}  {ipde_s:>10} {fpde_s:>10} "
                     f"{igo:>9.4f} {fgo:>9.4f} {god_s:>8}  {sr:>5.0f}%")
        if fpde: all_fpde.append(fpde)
        if igo:  all_igo.append(igo)
        if fgo:  all_fgo.append(fgo)

    lines.append(f"  {'-'*72}")
    if all_fpde:
        lines.append(f"  {'COMBINED':<22} {sum(d['n_projections'] for d in all_patient_results.values()):>3}  "
                     f"{'':>10} {np.mean(all_fpde):>9.2f}mm "
                     f"{np.mean(all_igo):>9.4f} {np.mean(all_fgo):>9.4f} "
                     f"{np.mean(all_igo)-np.mean(all_fgo):>+8.4f}")

    lines.append(f"\n  CONSISTENCY CHECK:")
    if len(all_fpde) >= 2:
        pde_std = np.std(all_fpde)
        go_std  = np.std(all_fgo)
        lines.append(f"    Final PDE std across patients: {pde_std:.2f}mm  "
                     f"{'✅ CONSISTENT' if pde_std < 5 else '⚠️  VARIABLE'}")
        lines.append(f"    Final GO  std across patients: {go_std:.4f}  "
                     f"{'✅ CONSISTENT' if go_std < 0.05 else '⚠️  VARIABLE'}")

    lines.append(f"{'='*75}")
    report = '\n'.join(lines)
    print(report)

    (RESULTS_ROOT / 'comparison_table.txt').write_text(report, encoding='utf-8')
    print(f"\n  All outputs saved to: {RESULTS_ROOT}/")
    print(f"  Total runtime: {time.time()-t0_global:.1f}s")


if __name__ == '__main__':
    main()
