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

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from generic_patient_loader import (
    GenericPatientLoader,
    GenericProjection,
    compute_pde,
    project_world,
    CAM_SID_MM, CAM_PIX_MM, CAM_FX, CAM_FY, CAM_CX, CAM_CY, CAM_IMG_SIZE,
)
from deepfluoro_loader import DeepFluoroSpecimen, perturb_extrinsic
from deepfluoro_drr import DeepFluoroDRR
from similarity import go_cost as compute_go_cost, normalized_cross_correlation
from optimizer import run_cmaes_single
from deformable_registration import (
    compute_deformation,
    deformable_pde as compute_deformable_pde,
)

# ---------------------------------------------------------------------------
# Data root
# ---------------------------------------------------------------------------
NEW_DATA_ROOT = Path('data/ProcessedData/Processed Data 2D-3D')
RESULTS_ROOT  = Path('results/newdata')

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
def _make_objectives(drr_gen, R_gt, t_gt, proj, specimen):
    xray = proj.image_raw
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

    def _ncc(drr, tgt):
        if np.count_nonzero(drr > 0.01) / drr.size < 0.08: return 1.0
        return float(1.0 - normalized_cross_correlation(1.0 - drr, tgt))

    def _ng(drr, tgt):
        if np.count_nonzero(drr > 0.01) / drr.size < 0.08: return 1.0
        return 0.5 * float(1.0 - normalized_cross_correlation(1.0 - drr, tgt)) \
             + 0.5 * float(compute_go_cost(drr, tgt))

    def obj(cf, tgt, sz, px, ns, lw):
        def f(x):
            Rc, tc = perturb_extrinsic(R_gt, t_gt, x[:3], x[3:])
            d = drr_gen.generate_from_extrinsic(Rc, tc, sz, px, ns)
            return (1 - lw) * cf(d, tgt) + lw * _lm(Rc, tc)
        return f

    oc = obj(_ncc, tgt_c, COARSE_SIZE, COARSE_PIX, 40,  0.4)
    op = obj(_ng,  tgt_o, OPT_SIZE,    OPT_PIX,    80,  0.3)
    of = obj(_ncc, tgt_f, FINE_SIZE,   FINE_PIX,   120, 0.2)

    def go_eval(R, t):
        d = drr_gen.generate_from_extrinsic(R, t, OPT_SIZE, OPT_PIX, 80)
        return float(compute_go_cost(d, tgt_o))

    return oc, op, of, go_eval


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def run_registration(drr_gen, proj: GenericProjection,
                     specimen: DeepFluoroSpecimen,
                     rot_deg=10.0, trans_mm=20.0,
                     n_starts=16, verbose=False) -> dict:

    R_gt = proj.R_proj.copy()
    t_gt = proj.t_proj.copy()
    lm_names = sorted(specimen.landmarks_3d.keys())
    pts3d    = np.array([specimen.landmarks_3d[n] for n in lm_names])

    pde_init = compute_pde(proj, R_gt, t_gt, pts3d, lm_names)
    init_pde_mean = float(np.mean(list(pde_init.values()))) if pde_init else float('nan')

    oc, op, of, go_eval = _make_objectives(drr_gen, R_gt, t_gt, proj, specimen)
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
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Discover patients
    patient_dirs = sorted([p for p in NEW_DATA_ROOT.iterdir() if p.is_dir()])
    if args.patients:
        patient_dirs = [p for p in patient_dirs if p.name in args.patients]

    print(f"\n{'='*65}")
    print(f"  NEW PATIENT PIPELINE  —  {len(patient_dirs)} patients")
    print(f"  Data: {NEW_DATA_ROOT}")
    print(f"  Out : {RESULTS_ROOT}")
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
            loader = GenericPatientLoader(patient_dir)
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
        for i, proj in enumerate(spec.projections):
            print(f"\n  [{i+1}/{len(spec.projections)}] {proj.proj_key} ...", flush=True)
            t0 = time.time()
            res = run_registration(drr_gen, proj, spec,
                                   rot_deg=args.rot, trans_mm=args.trans,
                                   n_starts=args.starts, verbose=args.verbose)
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

    (RESULTS_ROOT / 'comparison_table.txt').write_text(report)
    print(f"\n  All outputs saved to: {RESULTS_ROOT}/")
    print(f"  Total runtime: {time.time()-t0_global:.1f}s")


if __name__ == '__main__':
    main()
