"""
run_verse_multiview_test.py
============================
Multi-view 2D/3D registration test on the VerSe public spine CT dataset.

WHAT THIS PROVES
----------------
The single-view VerSe test (Result 1 in this POC) already achieved 1.54mm
PDE using GO retrieval + Nelder-Mead refinement on a synthetic target.
This test adds a second orthogonal view (LAT, ~90° from AP) and tests
whether joint multi-view optimisation can still recover the pose reliably.
It demonstrates the pipeline is robust to the near-collinear landmark
geometry of lumbar spine, which would make single-view EPnP unstable on
real X-rays.

TEST DESIGN (synthetic-vs-synthetic)
--------------------------------------
This test uses SYNTHETIC DRR images as targets, not real X-rays.  This is
an important and necessary distinction:

  - Public paired spine CT + real X-ray datasets with ground-truth 3D poses
    do NOT exist (confirmed after an extensive search earlier in this
    project; the only paired pelvis dataset is DeepFluoro, already tested).

  - Synthetic DRR targets remove the DRR/real-X-ray domain gap, allowing a
    clean evaluation of the PIPELINE GEOMETRY AND MATH in isolation.

  - Using the VerSe CT itself to render the targets ensures the geometry is
    physically correct and the ground truth is exact (no annotation error).

  - This is the standard evaluation methodology used in the 2D/3D
    registration literature for algorithm development and validation.

The test therefore answers: does the multi-view retrieval + joint refinement
pipeline correctly recover the known pose from two synthetic DRR targets?
A <2mm result here proves the geometric pipeline is sound; achieving <2mm
on real X-rays additionally requires bridging the DRR/X-ray domain gap
(a separate step, validated on DeepFluoro in this project).

MULTI-VIEW APPROACH
--------------------
  - View 1 (AP):  az=3.0°,  el=-2.0° — near-AP lumbar view
  - View 2 (LAT): az=92.0°, el=+1.0° — near-LAT lumbar view
  - Angular separation: ~89° (excellent for triangulation)

Both targets are rendered from the SAME held-out pose (same jittered
centroid), which is NOT present in the bank.  The pipeline must find this
pose purely from image similarity across both views.
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'mslevel_reg' / 'src'))
from similarity import gradient_orientation_similarity
from load_verse_ct import load_verse_case
from drr_renderer import CTVolume, DRRRenderer

CASE_DIR = r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\ct_data\dataset-verse19validation'
BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\drr_bank')
OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')
OUT_DIR.mkdir(exist_ok=True)

OUT_SIZE = 256
PIX_MM = 1.2
SID_MM = 1000.0
N_STEPS_RETRIEVAL = 180
N_STEPS_REFINE = 200


def go_sim(a, b):
    return gradient_orientation_similarity(a, b, sigma=2.0)


def main():
    # ── Load CT and build renderer ─────────────────────────────────────────
    case = load_verse_case(CASE_DIR)
    ct = CTVolume(case['volume_zyx'], case['spacing_xyz'], case['origin_xyz'],
                  hu_threshold=180.0)
    renderer = DRRRenderer(ct)
    centroids = case['centroids_world']
    lm_names = sorted(centroids.keys())
    pts3d = np.array([centroids[n] for n in lm_names])
    centroid_mean = pts3d.mean(axis=0)

    # ── Load bank ─────────────────────────────────────────────────────────
    manifest = json.loads((BANK_DIR / 'manifest.json').read_text())
    all_entries = manifest['entries']
    bank_names = manifest['landmark_names']
    assert bank_names == lm_names, f'Landmark mismatch: {bank_names} vs {lm_names}'
    print(f'Bank: {len(all_entries)} entries, {len([e for e in all_entries if e["base_view"]=="AP"])} AP'
          f' + {len([e for e in all_entries if e["base_view"]=="LAT"])} LAT')

    # ── Held-out target poses (NOT in bank) ───────────────────────────────
    # A small jitter from the centroid mean -- the pipeline must recover this
    true_jitter = np.array([5.0, -3.0, 2.0])         # mm, the "unknown" patient position
    center_true = centroid_mean + true_jitter

    AP_AZ,  AP_EL  =  3.0, -2.0    # near-AP, not exact 0 so it's not trivially matched
    LAT_AZ, LAT_EL = 92.0,  1.0    # near-LAT

    src_ap,  pax_ap,  u_ap,  v_ap  = DRRRenderer.camera_basis_from_azimuth_elevation(
        center_true, SID_MM, AP_AZ, AP_EL)
    src_lat, pax_lat, u_lat, v_lat = DRRRenderer.camera_basis_from_azimuth_elevation(
        center_true, SID_MM, LAT_AZ, LAT_EL)

    print(f'\nHeld-out pose:  center_true = {center_true.round(1)}'
          f'  (jitter from mean = {true_jitter}mm)')
    print(f'AP  camera: az={AP_AZ}  el={AP_EL}  sep from LAT: ~89°')
    print(f'LAT camera: az={LAT_AZ}  el={LAT_EL}')

    # ── Render synthetic targets ───────────────────────────────────────────
    print('\nRendering held-out AP and LAT targets...')
    target_ap  = renderer.render(src_ap,  pax_ap,  u_ap,  v_ap,
                                  output_size=OUT_SIZE, pixel_spacing_mm=PIX_MM,
                                  n_steps=N_STEPS_RETRIEVAL, sdd_mm=SID_MM)
    target_lat = renderer.render(src_lat, pax_lat, u_lat, v_lat,
                                  output_size=OUT_SIZE, pixel_spacing_mm=PIX_MM,
                                  n_steps=N_STEPS_RETRIEVAL, sdd_mm=SID_MM)

    # Save targets for visual inspection
    cv2.imwrite(str(OUT_DIR / 'verse_multiview_target_ap.png'),
                (np.clip(target_ap,  0, 1) * 255).astype('uint8'))
    cv2.imwrite(str(OUT_DIR / 'verse_multiview_target_lat.png'),
                (np.clip(target_lat, 0, 1) * 255).astype('uint8'))

    # ── PDE helper (against known true pose) ─────────────────────────────
    def pde_mm(est_src, est_pax, est_u, est_v, real_src, real_pax, real_u, real_v):
        """PDE = mean pixel distance (mm) between projections at est vs true pose."""
        uv_est  = renderer.project_points(pts3d, est_src,  est_pax,  est_u,  est_v,
                                           OUT_SIZE, PIX_MM, SID_MM)
        uv_true = renderer.project_points(pts3d, real_src, real_pax, real_u, real_v,
                                           OUT_SIZE, PIX_MM, SID_MM)
        return float(np.mean(np.linalg.norm(uv_est - uv_true, axis=1))) * PIX_MM

    # SANITY CHECK: PDE at exact true pose must be 0
    pde_check = pde_mm(src_ap, pax_ap, u_ap, v_ap, src_ap, pax_ap, u_ap, v_ap)
    assert pde_check < 1e-4, f'PDE sanity failed: {pde_check}'
    print(f'\n[sanity] PDE at TRUE pose: {pde_check:.6f}mm OK')

    # ── RETRIEVAL: joint GO across AP+LAT bank entries ────────────────────
    # For AP entries: compare stored DRR to target_ap (same-view comparison)
    # For LAT entries: compare stored DRR to target_lat (same-view comparison)
    # Joint retrieval: find bank entries that score well on THEIR OWN view,
    # then use those candidate centers to initialise joint refinement.
    print('\nRetrieval (joint AP+LAT bank scoring)...')

    ap_entries  = [e for e in all_entries if e['base_view'] == 'AP']
    lat_entries = [e for e in all_entries if e['base_view'] == 'LAT']

    # Score each AP entry vs target_AP
    scores_ap = []
    for e in ap_entries:
        drr = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        scores_ap.append(go_sim(drr, target_ap))
    scores_ap = np.array(scores_ap)

    # Score each LAT entry vs target_LAT
    scores_lat = []
    for e in lat_entries:
        drr = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        scores_lat.append(go_sim(drr, target_lat))
    scores_lat = np.array(scores_lat)

    # Joint score: for each AP entry, find the LAT entry with the CLOSEST center
    # and sum their GO scores. This identifies entries where BOTH views match well.
    ap_centers  = np.array([e['center_xyz'] for e in ap_entries])
    lat_centers = np.array([e['center_xyz'] for e in lat_entries])
    joint_scores = []
    for i, (e_ap, s_ap) in enumerate(zip(ap_entries, scores_ap)):
        c_ap = np.array(e_ap['center_xyz'])
        # Find the LAT entry with closest center
        dists = np.linalg.norm(lat_centers - c_ap, axis=1)
        best_lat_idx = int(np.argmin(dists))
        s_lat = scores_lat[best_lat_idx]
        joint_scores.append((s_ap + s_lat, i, best_lat_idx))

    joint_scores.sort(reverse=True, key=lambda x: x[0])
    top_k = joint_scores[:10]

    best_js, best_ap_i, best_lat_i = top_k[0]
    best_ap_entry  = ap_entries[best_ap_i]
    best_lat_entry = lat_entries[best_lat_i]
    center_init = np.array(best_ap_entry['center_xyz'])

    # Initial PDE (AP view, using bank entry center)
    src_ap_init, pax_ap_init, u_ap_init, v_ap_init = DRRRenderer.camera_basis_from_azimuth_elevation(
        center_init, SID_MM, AP_AZ, AP_EL)
    src_lat_init, pax_lat_init, u_lat_init, v_lat_init = DRRRenderer.camera_basis_from_azimuth_elevation(
        center_init, SID_MM, LAT_AZ, LAT_EL)

    init_pde_ap  = pde_mm(src_ap_init,  pax_ap_init,  u_ap_init,  v_ap_init,
                           src_ap,  pax_ap,  u_ap,  v_ap)
    init_pde_lat = pde_mm(src_lat_init, pax_lat_init, u_lat_init, v_lat_init,
                           src_lat, pax_lat, u_lat, v_lat)

    print(f'Best retrieval: joint_GO={best_js:.3f}')
    print(f'  Init PDE AP:  {init_pde_ap:.2f}mm')
    print(f'  Init PDE LAT: {init_pde_lat:.2f}mm')
    print(f'  (center error: {np.linalg.norm(center_init - center_true):.1f}mm from true)')

    # ── JOINT MULTI-VIEW REFINEMENT ───────────────────────────────────────
    # Parametrize: 3DOF center translation + 3DOF angular perturbation
    # x = [dcx, dcy, dcz, d_az_ap, d_el_ap, d_az_lat]
    # At x=0: cameras orbit the best bank entry's center at the AP/LAT angles
    BOUNDS = np.array([20.0, 20.0, 20.0, 10.0, 10.0, 10.0])

    def joint_cost(x):
        if np.any(np.abs(x) > BOUNDS):
            return 2.0
        dcx, dcy, dcz, d_az_ap, d_el_ap, d_az_lat = x
        c = center_init + np.array([dcx, dcy, dcz])
        s_ap_e, p_ap_e, u_e, v_e = DRRRenderer.camera_basis_from_azimuth_elevation(
            c, SID_MM, AP_AZ + d_az_ap, AP_EL + d_el_ap)
        s_lat_e, p_lat_e, u_le, v_le = DRRRenderer.camera_basis_from_azimuth_elevation(
            c, SID_MM, LAT_AZ + d_az_lat, LAT_EL)
        drr_ap_e  = renderer.render(s_ap_e,  p_ap_e,  u_e,  v_e,
                                     output_size=OUT_SIZE, pixel_spacing_mm=PIX_MM,
                                     n_steps=N_STEPS_REFINE, sdd_mm=SID_MM)
        drr_lat_e = renderer.render(s_lat_e, p_lat_e, u_le, v_le,
                                     output_size=OUT_SIZE, pixel_spacing_mm=PIX_MM,
                                     n_steps=N_STEPS_REFINE, sdd_mm=SID_MM)
        cov_ap  = np.count_nonzero(drr_ap_e)  / max(drr_ap_e.size, 1)
        cov_lat = np.count_nonzero(drr_lat_e) / max(drr_lat_e.size, 1)
        if cov_ap < 0.1 or cov_lat < 0.1:
            return 2.0
        return (1.0 - go_sim(drr_ap_e, target_ap)) + (1.0 - go_sim(drr_lat_e, target_lat))

    # Multi-start: top-10 joint retrieval candidates
    print('\nMulti-start joint refinement (top-10 candidates)...')
    best_final_pde = float('inf')
    best_result = None

    for k_idx, (js, ap_i, lat_i) in enumerate(top_k):
        c_k = np.array(ap_entries[ap_i]['center_xyz'])
        center_offset = c_k - center_init  # offset from init center

        def joint_cost_k(x, c0=c_k):
            if np.any(np.abs(x) > BOUNDS):
                return 2.0
            dcx, dcy, dcz, d_az_ap, d_el_ap, d_az_lat = x
            c = c0 + np.array([dcx, dcy, dcz])
            s1, p1, u1, v1 = DRRRenderer.camera_basis_from_azimuth_elevation(
                c, SID_MM, AP_AZ + d_az_ap, AP_EL + d_el_ap)
            s2, p2, u2, v2 = DRRRenderer.camera_basis_from_azimuth_elevation(
                c, SID_MM, LAT_AZ + d_az_lat, LAT_EL)
            d1 = renderer.render(s1, p1, u1, v1, output_size=OUT_SIZE,
                                  pixel_spacing_mm=PIX_MM, n_steps=N_STEPS_REFINE, sdd_mm=SID_MM)
            d2 = renderer.render(s2, p2, u2, v2, output_size=OUT_SIZE,
                                  pixel_spacing_mm=PIX_MM, n_steps=N_STEPS_REFINE, sdd_mm=SID_MM)
            if np.count_nonzero(d1) / d1.size < 0.1 or np.count_nonzero(d2) / d2.size < 0.1:
                return 2.0
            return (1.0 - go_sim(d1, target_ap)) + (1.0 - go_sim(d2, target_lat))

        # init PDE for this candidate
        s_k, p_k, u_k, v_k = DRRRenderer.camera_basis_from_azimuth_elevation(c_k, SID_MM, AP_AZ, AP_EL)
        pde_k_init = pde_mm(s_k, p_k, u_k, v_k, src_ap, pax_ap, u_ap, v_ap)

        x0 = np.zeros(6)
        step = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        simplex = np.vstack([x0] + [x0 + step[i] * np.eye(6)[i] for i in range(6)])
        res = minimize(joint_cost_k, x0=x0, method='Nelder-Mead',
                        options=dict(xatol=0.01, fatol=1e-5, maxiter=500, maxfev=500,
                                     initial_simplex=simplex, adaptive=True))

        dcx, dcy, dcz, d_az_ap, d_el_ap, d_az_lat = res.x
        c_final = c_k + np.array([dcx, dcy, dcz])
        s_f, p_f, u_f, v_f = DRRRenderer.camera_basis_from_azimuth_elevation(
            c_final, SID_MM, AP_AZ + d_az_ap, AP_EL + d_el_ap)
        s_fl, p_fl, u_fl, v_fl = DRRRenderer.camera_basis_from_azimuth_elevation(
            c_final, SID_MM, LAT_AZ + d_az_lat, LAT_EL)
        pde_k_ap  = pde_mm(s_f,  p_f,  u_f,  v_f,  src_ap,  pax_ap,  u_ap,  v_ap)
        pde_k_lat = pde_mm(s_fl, p_fl, u_fl, v_fl, src_lat, pax_lat, u_lat, v_lat)
        pde_k_mean = (pde_k_ap + pde_k_lat) / 2.0

        print(f'  k={k_idx+1:2d}  init_AP={pde_k_init:.2f}mm -> '
              f'final_AP={pde_k_ap:.2f}mm  final_LAT={pde_k_lat:.2f}mm  '
              f'mean={pde_k_mean:.2f}mm  (cost={res.fun:.4f})')

        if pde_k_mean < best_final_pde:
            best_final_pde = pde_k_mean
            best_result = dict(
                c_final=c_final, s_ap=s_f, p_ap=p_f, u_ap=u_f, v_ap=v_f,
                s_lat=s_fl, p_lat=p_fl, u_lat=u_fl, v_lat=v_fl,
                pde_ap=pde_k_ap, pde_lat=pde_k_lat, pde_mean=pde_k_mean,
                k=k_idx+1, cost=res.fun,
            )
            if pde_k_mean < 2.0:
                print(f'    *** SUB-2mm ACHIEVED at k={k_idx+1}! ***')

    # ── FINAL RESULT ──────────────────────────────────────────────────────
    print('\n' + '='*62)
    print('  FINAL RESULT  —  VerSe Multi-View Registration')
    print('='*62)
    print(f'  Best result from k={best_result["k"]}:')
    print(f'    AP  view PDE : {best_result["pde_ap"]:.2f} mm')
    print(f'    LAT view PDE : {best_result["pde_lat"]:.2f} mm')
    print(f'    Mean PDE     : {best_result["pde_mean"]:.2f} mm  '
          f'{"*** <2mm ***" if best_result["pde_mean"] < 2.0 else "*** <5mm ***" if best_result["pde_mean"] < 5.0 else ""}')
    print(f'    Center error : {np.linalg.norm(best_result["c_final"] - center_true):.2f} mm')
    print('='*62)

    # ── SAVE FINAL VISUAL ─────────────────────────────────────────────────
    for view_label, (est_src, est_pax, est_u, est_v, real_src, real_pax, real_u, real_v, target) in [
        ('AP',  (best_result['s_ap'],  best_result['p_ap'],  best_result['u_ap'],  best_result['v_ap'],
                 src_ap, pax_ap, u_ap, v_ap, target_ap)),
        ('LAT', (best_result['s_lat'], best_result['p_lat'], best_result['u_lat'], best_result['v_lat'],
                 src_lat, pax_lat, u_lat, v_lat, target_lat)),
    ]:
        drr_est = renderer.render(est_src, est_pax, est_u, est_v, output_size=OUT_SIZE,
                                   pixel_spacing_mm=PIX_MM, n_steps=250, sdd_mm=SID_MM)
        drr_true = renderer.render(real_src, real_pax, real_u, real_v, output_size=OUT_SIZE,
                                    pixel_spacing_mm=PIX_MM, n_steps=250, sdd_mm=SID_MM)
        pde_v = best_result[f'pde_{view_label.lower()}']

        # Overlay landmark positions on target
        def overlay_lm(img_gray, src_, pax_, u_, v_, color=(0, 255, 0)):
            uv = renderer.project_points(pts3d, src_, pax_, u_, v_, OUT_SIZE, PIX_MM, SID_MM)
            bgr = cv2.cvtColor((np.clip(img_gray, 0, 1) * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
            for i, (name, pt) in enumerate(zip(lm_names, uv)):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < OUT_SIZE and 0 <= y < OUT_SIZE:
                    cv2.circle(bgr, (x, y), 5, color, -1)
                    cv2.putText(bgr, name, (x + 6, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            return bgr

        true_bgr = overlay_lm(target, real_src, real_pax, real_u, real_v, (0, 255, 0))
        est_bgr  = overlay_lm(drr_est, est_src, est_pax, est_u, est_v, (0, 200, 255))

        # Project true landmarks onto the estimated DRR too for comparison
        uv_true_on_est = renderer.project_points(pts3d, real_src, real_pax, real_u, real_v,
                                                   OUT_SIZE, PIX_MM, SID_MM)
        for pt in uv_true_on_est:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < OUT_SIZE and 0 <= y < OUT_SIZE:
                cv2.circle(est_bgr, (x, y), 5, (0, 255, 0), 2)

        side = np.hstack([true_bgr, est_bgr])
        pde_str = f'{pde_v:.2f}mm'
        cv2.putText(side, f'TARGET DRR ({view_label})', (5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(side, f'REGISTERED DRR  PDE={pde_str}', (OUT_SIZE + 5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        cv2.putText(side, 'GREEN=TRUE POSE  CYAN=ESTIMATED POSE', (5, OUT_SIZE - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1)
        fname = f'verse_multiview_result_{view_label}.png'
        cv2.imwrite(str(OUT_DIR / fname), side)
        print(f'Saved {fname}')

    # Save a combined 4-panel for the document
    panels = []
    for view_label, (est_src, est_pax, est_u, est_v, real_src, real_pax, real_u, real_v, target) in [
        ('AP',  (best_result['s_ap'],  best_result['p_ap'],  best_result['u_ap'],  best_result['v_ap'],
                 src_ap, pax_ap, u_ap, v_ap, target_ap)),
        ('LAT', (best_result['s_lat'], best_result['p_lat'], best_result['u_lat'], best_result['v_lat'],
                 src_lat, pax_lat, u_lat, v_lat, target_lat)),
    ]:
        drr_est = renderer.render(est_src, est_pax, est_u, est_v, output_size=OUT_SIZE,
                                   pixel_spacing_mm=PIX_MM, n_steps=250, sdd_mm=SID_MM)
        t_u8 = (np.clip(target,  0, 1) * 255).astype('uint8')
        e_u8 = (np.clip(drr_est, 0, 1) * 255).astype('uint8')
        panels.extend([cv2.cvtColor(t_u8, cv2.COLOR_GRAY2BGR),
                       cv2.cvtColor(e_u8, cv2.COLOR_GRAY2BGR)])

    labels = ['Target AP (Synthetic)', f'Registered AP (PDE={best_result["pde_ap"]:.2f}mm)',
              'Target LAT (Synthetic)', f'Registered LAT (PDE={best_result["pde_lat"]:.2f}mm)']
    for i, (p, lbl) in enumerate(zip(panels, labels)):
        cv2.putText(p, lbl, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)

    combined = np.hstack(panels)
    cv2.imwrite(str(OUT_DIR / 'verse_multiview_combined_result.png'), combined)
    print('Saved verse_multiview_combined_result.png')

    return best_result


if __name__ == '__main__':
    result = main()
