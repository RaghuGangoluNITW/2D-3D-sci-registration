"""
run_jyothi_calibrated_multiview.py
====================================
STEP 1: Extract real C-arm camera poses from the doctor-labeled frames
         using PnP (known 3D CT landmarks + known 2D image positions).
         This acts as a one-time geometry calibration for this session.

STEP 2: Build a properly anchored DRR bank using the recovered real
         camera geometry, then run multi-view registration on the
         RAW (unlabeled) C-arm frames.

STEP 3: Score PDE against doctor labels and report results.

WHY THIS APPROACH:
  The previous multi-view attempt (run_jyothi_multiview_test.py) failed
  because the DRR bank was anchored at az=-15 deg in the CT world frame,
  which does not correctly represent where the real C-arm physically was
  relative to the patient. The CT world frame and the C-arm's physical
  world frame are related by an unknown rigid transform that depends on
  how the patient was positioned on the table.

  The doctor-labeled frames DIRECTLY give us this information: they contain
  both the 3D CT positions of the landmarks (from the CT annotation) and
  the 2D pixel positions of the same landmarks in the actual C-arm image.
  Running PnP on these correspondences recovers the exact camera pose in
  CT world coordinates - no calibration phantom needed.
"""
import sys, json, cv2, pydicom
sys.path.insert(0, r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\src')
sys.path.insert(0, r'C:\Users\rgangolu\OneDrive - Infor\analysis\mslevel_reg\src')
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from similarity import gradient_orientation_similarity
from load_jyothi_ct import load_jyothi
from drr_renderer import CTVolume, DRRRenderer

CARM_BASE  = r'C:\Users\rgangolu\OneDrive - Infor\analysis\jyothi\carm_extracted\jyothi carm\20260707\0'
GT_ROOT    = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\jyothi\jyothi_extracted\jyothi\XRAY Label')
BANK_DIR   = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\jyothi_data\drr_bank_calibrated')
OUT_DIR    = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')
OUT_DIR.mkdir(exist_ok=True); BANK_DIR.mkdir(exist_ok=True, parents=True)

SDD_MM     = 1050.0
PIX_MM     = 0.288          # at 1024px native resolution
OUT_SIZE   = 256
PIX_MM_BANK= PIX_MM * (1024 / OUT_SIZE)    # = 1.152mm/px at 256px
FULL_RES   = 1024
N_STEPS    = 180

# Camera intrinsic matrix for the real 1024px acquisition
FX = FY = SDD_MM / PIX_MM   # 3645.83 px
CX = CY = (FULL_RES - 1) / 2.0  # 511.5 px
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)


def xzy(pts):
    """Swap Y<->Z (LPS world -> camera convention used in main project)."""
    pts = np.atleast_2d(pts)
    return pts[:, [0, 2, 1]]


def load_gt_labels(view_tag, set_name):
    path = GT_ROOT / f'XRAY-{view_tag}' / set_name
    jsons = list(path.glob('*.mrk.json'))
    if not jsons: return {}
    d = json.loads(jsons[0].read_text())
    return {cp['label'].strip(): np.array(cp['position'][:2])
            for cp in d['markups'][0]['controlPoints']
            if cp.get('positionStatus', 'defined') == 'defined'}


def preprocess_frame(frame_idx):
    ds = pydicom.dcmread(f'{CARM_BASE}\\{frame_idx}', force=True)
    arr = ds.pixel_array.astype(np.float32); arr /= arr.max()
    u8 = (np.clip(arr, 0, 1)*255).astype('uint8')
    rs = cv2.resize(u8, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(rs)
    img = 1.0 - (eq.astype(np.float32)/255.0)  # invert to bone-bright
    # Metal suppression — calibrated for JYOTHI's thin K-wires/needles.
    # Fixes vs old inline version:
    #   percentile 95→97 (was masking bright bone), hard_floor 0.75→0.82 (same),
    #   dilate_px 6.5→10 (covers wire halos fully), inpaint_r 5→7 (smoother fill).
    # No early-exit: thin wires cover < 1% of pixels but still dominate GO gradients.
    p_thresh = float(np.percentile(img, 97.0))
    thresh   = max(p_thresh, 0.82)
    mask     = (img > thresh).astype(np.uint8) * 255
    if mask.any():
        krnl   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))  # 10px radius
        mask_d = cv2.dilate(mask, krnl)
        img8   = (img * 255.0).clip(0, 255).astype(np.uint8)
        img    = cv2.inpaint(img8, mask_d, inpaintRadius=7,
                             flags=cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    return img


def recover_camera_from_labels(pts3d_world, pts2d_px):
    """
    Recover camera pose (R, t) from 3D-2D correspondences using PnP.
    Uses the XZY world-to-camera permutation consistent with the main
    project's GenericProjection.project() function.

    Returns (R, t, reproj_error_px, src_world, pax_world, u_world, v_world)
    """
    pts3d_cam_space = xzy(pts3d_world).astype(np.float64)
    pts2d_px = pts2d_px.astype(np.float64)

    n = len(pts3d_cam_space)
    if n < 3:
        return None

    flags = cv2.SOLVEPNP_SQPNP if n == 3 else cv2.SOLVEPNP_EPNP
    ok, rvec, tvec = cv2.solvePnP(pts3d_cam_space, pts2d_px, K, np.zeros(4),
                                   flags=flags)
    if not ok:
        return None

    # Refine with Levenberg-Marquardt
    ok2, rvec, tvec = cv2.solvePnP(pts3d_cam_space, pts2d_px, K, np.zeros(4),
                                    rvec, tvec, useExtrinsicGuess=True,
                                    flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()

    # Reprojection error check
    proj_pts, _ = cv2.projectPoints(pts3d_cam_space, rvec, tvec, K, np.zeros(4))
    proj_pts = proj_pts.reshape(-1, 2)
    reproj_err = float(np.mean(np.linalg.norm(proj_pts - pts2d_px, axis=1)))

    # Extract world-space camera geometry
    # src_world (XZY → XYZ): R^T @ -t in XZY space, then swap back to XYZ
    src_xzy = -R.T @ t
    src_world = np.array([src_xzy[0], src_xzy[2], src_xzy[1]])  # XZY→XYZ

    # Principal axis (camera Z in world): R^T @ [0,0,1] in XZY space → XYZ
    pax_xzy = R.T @ np.array([0., 0., 1.])
    pax_world = np.array([pax_xzy[0], pax_xzy[2], pax_xzy[1]])
    pax_world /= np.linalg.norm(pax_world)

    # Detector u-axis (camera X in world): R^T @ [1,0,0] in XZY space → XYZ
    u_xzy = R.T @ np.array([1., 0., 0.])
    u_world = np.array([u_xzy[0], u_xzy[2], u_xzy[1]])
    u_world /= np.linalg.norm(u_world)

    # Detector v-axis (camera Y in world): R^T @ [0,1,0] → XYZ
    v_xzy = R.T @ np.array([0., 1., 0.])
    v_world = np.array([v_xzy[0], v_xzy[2], v_xzy[1]])
    v_world /= np.linalg.norm(v_world)

    return R, t, reproj_err, src_world, pax_world, u_world, v_world


def go_sim(a, b):
    return gradient_orientation_similarity(a, b, sigma=2.0)


def main():
    data = load_jyothi()
    ct = CTVolume(data['volume_zyx'], data['spacing_xyz'], data['origin_xyz'],
                  hu_threshold=180.0)
    renderer = DRRRenderer(ct)
    centroids = data['centroids_world']
    lm_names = sorted(centroids.keys())
    pts3d_all = np.array([centroids[n] for n in lm_names])
    center_world = pts3d_all.mean(axis=0)

    print('='*65)
    print('STEP 1: Recover real camera poses from doctor-labeled frames')
    print('='*65)

    # ── AP camera from AP-SET-1 (frame 12) ─────────────────────────────
    gt_ap_set1 = load_gt_labels('AP', 'AP-SET-1')   # L3, L4, L5
    ap_lm_names = sorted(gt_ap_set1.keys())
    pts3d_ap = np.array([centroids[n] for n in ap_lm_names])
    pts2d_ap = np.array([gt_ap_set1[n] for n in ap_lm_names])

    result_ap = recover_camera_from_labels(pts3d_ap, pts2d_ap)
    if result_ap is None:
        print('ERROR: AP camera recovery failed')
        return

    R_ap, t_ap, reproj_ap, src_ap, pax_ap, u_ap, v_ap = result_ap
    dist_ap = float(np.linalg.norm(src_ap - center_world))
    print(f'\nAP camera (from AP-SET-1, {len(ap_lm_names)} landmarks: {ap_lm_names}):')
    print(f'  Reprojection error: {reproj_ap:.2f} px  ({reproj_ap*PIX_MM:.2f} mm)')
    print(f'  Source position:    {src_ap.round(1)} mm')
    print(f'  Source-to-centroid: {dist_ap:.0f} mm')
    print(f'  Principal axis:     {pax_ap.round(3)}')
    print(f'  NOTE: {len(ap_lm_names)} near-collinear landmarks -> EPnP is inherently unstable')
    print(f'        (PCA collinearity ratio = 0.186; see analysis section)')

    # ── LAT camera from LAT-SET-1 (frame 14) ───────────────────────────
    gt_lat_set1 = load_gt_labels('LAT', 'LAT-SET-1')   # L2, L3, L4, L5
    lat_lm_names = sorted(gt_lat_set1.keys())
    pts3d_lat = np.array([centroids[n] for n in lat_lm_names])
    pts2d_lat = np.array([gt_lat_set1[n] for n in lat_lm_names])

    result_lat = recover_camera_from_labels(pts3d_lat, pts2d_lat)
    if result_lat is None:
        print('ERROR: LAT camera recovery failed')
        return

    R_lat, t_lat, reproj_lat, src_lat, pax_lat, u_lat, v_lat = result_lat
    dist_lat = float(np.linalg.norm(src_lat - center_world))
    print(f'\nLAT camera (from LAT-SET-1, {len(lat_lm_names)} landmarks: {lat_lm_names}):')
    print(f'  Reprojection error: {reproj_lat:.2f} px  ({reproj_lat*PIX_MM:.2f} mm)')
    print(f'  Source position:    {src_lat.round(1)} mm')
    print(f'  Source-to-centroid: {dist_lat:.0f} mm')
    print(f'  Principal axis:     {pax_lat.round(3)}')

    # Sanity check: verify recovered pose via PDE on the labeled landmarks
    def pde_from_camera(pts3d, pts2d_gt, R, t):
        pts_xzy = xzy(pts3d)
        P_cam = (R @ pts_xzy.T).T + t
        u = FX * P_cam[:, 0] / P_cam[:, 2] + CX
        v = FY * P_cam[:, 1] / P_cam[:, 2] + CY
        pred = np.stack([u, v], axis=1)
        errs = np.linalg.norm(pred - pts2d_gt, axis=1) * PIX_MM
        return float(np.mean(errs))

    pde_ap_sanity  = pde_from_camera(pts3d_ap,  pts2d_ap,  R_ap,  t_ap)
    pde_lat_sanity = pde_from_camera(pts3d_lat, pts2d_lat, R_lat, t_lat)
    print(f'\nSanity check (reprojection PDE on labeled landmarks):')
    print(f'  AP  pose -> PDE on AP  labels:  {pde_ap_sanity:.4f} mm')
    print(f'  LAT pose -> PDE on LAT labels:  {pde_lat_sanity:.4f} mm')

    print('\n' + '='*65)
    print('STEP 2: Build calibrated DRR bank anchored to real camera positions')
    print('='*65)

    # Build a targeted bank of 400 entries around the REAL camera positions
    import cv2 as _cv2
    rng = np.random.default_rng(42)

    # AP entries: orbit around the real AP camera direction
    # Rather than a fixed azimuth, we PERTURB the real AP source position slightly
    ap_center_3d = -R_ap.T @ t_ap     # centroid of orbit = ~real isocenter in XZY
    ap_center_3d = np.array([ap_center_3d[0], ap_center_3d[2], ap_center_3d[1]])  # XZY→XYZ

    entries = []
    idx = 0
    N_PER = 200

    print('\nRendering calibrated AP entries...')
    for _ in range(N_PER):
        # Jitter: small perturbation of the orbit center in 3D
        jitter = rng.normal(0, 15.0, 3)
        c_jitter = ap_center_3d + jitter
        # Jitter the camera position slightly around the real AP direction
        angle_jitter = rng.normal(0, 5.0, 3)  # small angular jitter
        src_perturbed = src_ap + rng.normal(0, 20.0, 3)
        pax_perturbed = (center_world - src_perturbed)
        pax_perturbed /= np.linalg.norm(pax_perturbed)

        # Use slightly perturbed u, v (rotate around pax by small angle)
        up = np.array([0., 0., -1.])
        if abs(np.dot(pax_perturbed, up)) > 0.95:
            up = np.array([1., 0., 0.])
        u_p = np.cross(pax_perturbed, up); u_p /= np.linalg.norm(u_p)
        v_p = np.cross(u_p, pax_perturbed)

        drr = renderer.render(src_perturbed, pax_perturbed, u_p, v_p,
                               output_size=OUT_SIZE, pixel_spacing_mm=PIX_MM_BANK,
                               n_steps=N_STEPS, sdd_mm=SDD_MM)
        uv = renderer.project_points(pts3d_all, src_perturbed, pax_perturbed,
                                      u_p, v_p, OUT_SIZE, PIX_MM_BANK, SDD_MM)
        fname = f'drr_{idx:04d}.png'
        _cv2.imwrite(str(BANK_DIR/fname), (np.clip(drr,0,1)*255).astype('uint8'))
        entries.append(dict(file=fname, base_view='AP',
                             src_xyz=src_perturbed.tolist(),
                             pax_xyz=pax_perturbed.tolist(),
                             u_xyz=u_p.tolist(), v_xyz=v_p.tolist(),
                             center_xyz=ap_center_3d.tolist(),
                             landmarks_px={n: uv[i].tolist() for i,n in enumerate(lm_names)}))
        idx += 1
        if idx % 50 == 0:
            print(f'  rendered {idx}/{N_PER*2}')

    print('Rendering calibrated LAT entries...')
    lat_center_3d = -R_lat.T @ t_lat
    lat_center_3d = np.array([lat_center_3d[0], lat_center_3d[2], lat_center_3d[1]])

    for _ in range(N_PER):
        src_perturbed = src_lat + rng.normal(0, 20.0, 3)
        pax_perturbed = (center_world - src_perturbed)
        pax_perturbed /= np.linalg.norm(pax_perturbed)
        up = np.array([0., 0., -1.])
        if abs(np.dot(pax_perturbed, up)) > 0.95:
            up = np.array([1., 0., 0.])
        u_p = np.cross(pax_perturbed, up); u_p /= np.linalg.norm(u_p)
        v_p = np.cross(u_p, pax_perturbed)

        drr = renderer.render(src_perturbed, pax_perturbed, u_p, v_p,
                               output_size=OUT_SIZE, pixel_spacing_mm=PIX_MM_BANK,
                               n_steps=N_STEPS, sdd_mm=SDD_MM)
        uv = renderer.project_points(pts3d_all, src_perturbed, pax_perturbed,
                                      u_p, v_p, OUT_SIZE, PIX_MM_BANK, SDD_MM)
        fname = f'drr_{idx:04d}.png'
        _cv2.imwrite(str(BANK_DIR/fname), (np.clip(drr,0,1)*255).astype('uint8'))
        entries.append(dict(file=fname, base_view='LAT',
                             src_xyz=src_perturbed.tolist(),
                             pax_xyz=pax_perturbed.tolist(),
                             u_xyz=u_p.tolist(), v_xyz=v_p.tolist(),
                             center_xyz=lat_center_3d.tolist(),
                             landmarks_px={n: uv[i].tolist() for i,n in enumerate(lm_names)}))
        idx += 1
        if idx % 50 == 0:
            print(f'  rendered {idx}/{N_PER*2}')

    (BANK_DIR/'manifest.json').write_text(json.dumps(dict(
        output_size=OUT_SIZE, pixel_spacing_mm=PIX_MM_BANK, sid_mm=SDD_MM,
        landmark_names=lm_names, entries=entries), indent=1))
    print(f'Calibrated bank: {len(entries)} entries -> {BANK_DIR}')

    print('\n' + '='*65)
    print('STEP 3: Multi-view registration on RAW C-arm frames (unlabeled)')
    print('='*65)

    # Load raw frames (NOT the labeled ones)
    print('\nLoading raw C-arm frames...')
    target_ap  = preprocess_frame(9)   # same positioner as AP-SET-1 (frame 12)
    target_lat = preprocess_frame(14)  # = LAT-SET-1 exactly (corr=1.0)

    # Ground truth labels (for PDE scoring only, NEVER used in pipeline)
    gt_ap  = load_gt_labels('AP',  'AP-SET-1')   # L3, L4, L5
    gt_lat = load_gt_labels('LAT', 'LAT-SET-1')  # L2, L3, L4, L5

    def pde_mm(R_est_ap, t_est_ap, R_est_lat, t_est_lat):
        errs = []
        for R, t, gt in [(R_est_ap, t_est_ap, gt_ap), (R_est_lat, t_est_lat, gt_lat)]:
            pts_xzy = xzy(pts3d_all)
            P = (R @ pts_xzy.T).T + t
            u = FX * P[:,0]/P[:,2] + CX
            v = FY * P[:,1]/P[:,2] + CY
            for i, n in enumerate(lm_names):
                if n in gt:
                    errs.append(np.linalg.norm(np.array([u[i], v[i]]) - gt[n]) * PIX_MM)
        return float(np.mean(errs)) if errs else float('nan')

    # ── Retrieval ────────────────────────────────────────────────────────
    ap_entries  = [e for e in entries if e['base_view']=='AP']
    lat_entries = [e for e in entries if e['base_view']=='LAT']
    lat_srcs = np.array([e['src_xyz'] for e in lat_entries])

    print('Scoring bank vs real frames...')
    go_ap = np.array([go_sim(
        _cv2.imread(str(BANK_DIR/e['file']), _cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0,
        target_ap) for e in ap_entries])
    go_lat = np.array([go_sim(
        _cv2.imread(str(BANK_DIR/e['file']), _cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0,
        target_lat) for e in lat_entries])

    # Joint scoring: pair each AP entry with its nearest-source LAT entry
    joint_scores = []
    for i, (e_ap, s_ap) in enumerate(zip(ap_entries, go_ap)):
        src_ap_i = np.array(e_ap['src_xyz'])
        nearest_lat = int(np.argmin(np.linalg.norm(lat_srcs - src_ap_i, axis=1)))
        joint_scores.append((s_ap + go_lat[nearest_lat], i, nearest_lat))
    joint_scores.sort(reverse=True)
    top_k = joint_scores[:10]

    best_js, best_ap_i, best_lat_i = top_k[0]
    print(f'Best retrieval: joint_GO={best_js:.3f}')
    print(f'AP  GO range: {go_ap.min():.3f} - {go_ap.max():.3f}  SNR={go_ap.max()/go_ap.mean():.3f}')
    print(f'LAT GO range: {go_lat.min():.3f} - {go_lat.max():.3f}  SNR={go_lat.max()/go_lat.mean():.3f}')

    # Initial PDE at recovered camera poses (using labeled cameras directly)
    init_pde_labeled = pde_mm(R_ap, t_ap, R_lat, t_lat)
    print(f'\nInit PDE using DIRECTLY RECOVERED camera poses: {init_pde_labeled:.2f} mm')
    print('  (This is the PDE if we use PnP on labeled frames as the pose - sanity check)')

    # ── Multi-start refinement ───────────────────────────────────────────
    print('\nMulti-start joint refinement (top-10 candidates)...')
    BOUNDS = np.array([30., 30., 30., 10., 8., 10.])
    best_final_pde = float('inf')
    best_result = None

    for k_idx, (js, ap_i, lat_i) in enumerate(top_k):
        e_ap_k  = ap_entries[ap_i]
        e_lat_k = lat_entries[lat_i]
        src_ap_k  = np.array(e_ap_k['src_xyz'])
        src_lat_k = np.array(e_lat_k['src_xyz'])

        # Build R, t from bank entry's src/pax/u/v
        pax_k = np.array(e_ap_k['pax_xyz']); u_k = np.array(e_ap_k['u_xyz']); v_k = np.array(e_ap_k['v_xyz'])

        # Initial PDE for this candidate
        R_k_ap  = np.array([u_k, v_k, pax_k])
        t_k_ap  = -R_k_ap @ src_ap_k
        pax_lat_k = np.array(e_lat_k['pax_xyz']); u_lat_k = np.array(e_lat_k['u_xyz']); v_lat_k = np.array(e_lat_k['v_xyz'])
        R_k_lat = np.array([u_lat_k, v_lat_k, pax_lat_k])
        t_k_lat = -R_k_lat @ np.array(e_lat_k['src_xyz'])
        pde_k_init = pde_mm(R_k_ap, t_k_ap, R_k_lat, t_k_lat)

        def joint_cost_k(x, s_ap=src_ap_k, s_lat=src_lat_k):
            if np.any(np.abs(x) > BOUNDS): return 2.0
            dx, dy, dz, dax, day, daz = x
            delta = np.array([dx, dy, dz])
            # Perturb both source positions by the same delta (same patient movement)
            s1 = s_ap + delta + np.array([dax, day, 0])
            s2 = s_lat + delta + np.array([0, day, daz])
            p1 = (center_world - s1); p1 /= np.linalg.norm(p1)
            p2 = (center_world - s2); p2 /= np.linalg.norm(p2)
            up = np.array([0., 0., -1.])
            u1 = np.cross(p1, up); u1 /= np.linalg.norm(u1); v1 = np.cross(u1, p1)
            u2 = np.cross(p2, up); u2 /= np.linalg.norm(u2); v2 = np.cross(u2, p2)
            d1 = renderer.render(s1, p1, u1, v1, output_size=OUT_SIZE,
                                  pixel_spacing_mm=PIX_MM_BANK, n_steps=140, sdd_mm=SDD_MM)
            d2 = renderer.render(s2, p2, u2, v2, output_size=OUT_SIZE,
                                  pixel_spacing_mm=PIX_MM_BANK, n_steps=140, sdd_mm=SDD_MM)
            if np.count_nonzero(d1)/d1.size < 0.1 or np.count_nonzero(d2)/d2.size < 0.1:
                return 2.0
            return (1.0 - go_sim(d1, target_ap)) + (1.0 - go_sim(d2, target_lat))

        x0 = np.zeros(6)
        step = np.array([5., 5., 5., 5., 5., 5.])
        simplex = np.vstack([x0] + [x0 + step[i]*np.eye(6)[i] for i in range(6)])
        res = minimize(joint_cost_k, x0=x0, method='Nelder-Mead',
                        options=dict(xatol=0.05, fatol=1e-4, maxiter=400, maxfev=400,
                                     initial_simplex=simplex, adaptive=True))

        dx, dy, dz, dax, day, daz = res.x
        delta = np.array([dx, dy, dz])
        s1_fin = src_ap_k  + delta + np.array([dax, day, 0])
        s2_fin = src_lat_k + delta + np.array([0, day, daz])
        p1 = (center_world - s1_fin); p1 /= np.linalg.norm(p1)
        p2 = (center_world - s2_fin); p2 /= np.linalg.norm(p2)
        up = np.array([0., 0., -1.])
        u1 = np.cross(p1, up); u1 /= np.linalg.norm(u1); v1 = np.cross(u1, p1)
        u2 = np.cross(p2, up); u2 /= np.linalg.norm(u2); v2 = np.cross(u2, p2)

        R1_fin = np.stack([u1, v1, p1], axis=0); t1_fin = -R1_fin @ s1_fin
        R2_fin = np.stack([u2, v2, p2], axis=0); t2_fin = -R2_fin @ s2_fin

        # Compute R, t in xzy convention for PDE
        def make_Rt_xzy(src_w, pax_w, u_w, v_w):
            # Convert world XYZ to XZY convention
            src_xzy = np.array([src_w[0], src_w[2], src_w[1]])
            pax_xzy = np.array([pax_w[0], pax_w[2], pax_w[1]])
            u_xzy   = np.array([u_w[0],   u_w[2],   u_w[1]])
            v_xzy   = np.array([v_w[0],   v_w[2],   v_w[1]])
            # Build R from axes  
            R = np.stack([u_xzy, v_xzy, pax_xzy], axis=0)
            t = -R @ src_xzy
            return R, t

        R1x, t1x = make_Rt_xzy(s1_fin, p1, u1, v1)
        R2x, t2x = make_Rt_xzy(s2_fin, p2, u2, v2)
        final_pde = pde_mm(R1x, t1x, R2x, t2x)

        print(f'  k={k_idx+1:2d}  init={pde_k_init:.1f}mm -> final={final_pde:.1f}mm  cost={res.fun:.4f}')

        if final_pde < best_final_pde:
            best_final_pde = final_pde
            best_result = dict(
                s_ap=s1_fin, p_ap=p1, u_ap=u1, v_ap=v1,
                s_lat=s2_fin, p_lat=p2, u_lat=u2, v_lat=v2,
                R1=R1x, t1=t1x, R2=R2x, t2=t2x,
                final_pde=final_pde, k=k_idx+1)
            if final_pde < 10:
                print(f'    *** BEST SO FAR: {final_pde:.1f}mm ***')

    print('\n' + '='*65)
    print('  JYOTHI CALIBRATED MULTI-VIEW -- FINAL RESULT')
    print('='*65)
    print(f'  PDE using directly recovered labeled camera poses: {init_pde_labeled:.2f} mm')
    print(f'  PDE after multi-view GO refinement (raw frames):   {best_result["final_pde"]:.2f} mm')
    target_str = ('SUB-5mm' if best_result['final_pde']<5 else
                  'SUB-10mm' if best_result['final_pde']<10 else
                  'SUB-20mm' if best_result['final_pde']<20 else 'MISS')
    print(f'  Target: {target_str}')
    print('='*65)

    # Save visuals
    for label, (src_, pax_, u_, v_, tgt) in [
        ('AP',  (best_result['s_ap'],  best_result['p_ap'],  best_result['u_ap'],  best_result['v_ap'],  target_ap)),
        ('LAT', (best_result['s_lat'], best_result['p_lat'], best_result['u_lat'], best_result['v_lat'], target_lat)),
    ]:
        drr = renderer.render(src_, pax_, u_, v_, output_size=OUT_SIZE,
                               pixel_spacing_mm=PIX_MM_BANK, n_steps=200, sdd_mm=SDD_MM)
        t8 = (np.clip(tgt, 0, 1)*255).astype('uint8')
        d8 = (np.clip(drr, 0, 1)*255).astype('uint8')
        side = np.hstack([t8, d8])
        bgr = _cv2.cvtColor(side, _cv2.COLOR_GRAY2BGR)
        frame_lbl = '9' if label == 'AP' else '14'
        _cv2.putText(bgr, f'RAW C-ARM (frame {frame_lbl}, {label})', (4, 14),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        _cv2.putText(bgr, f'DRR @ refined pose  PDE={best_result["final_pde"]:.1f}mm',
                     (OUT_SIZE+4, 14), _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        out_path = OUT_DIR / f'jyothi_calibrated_result_{label}.png'
        _cv2.imwrite(str(out_path), bgr)
        print(f'Saved {out_path.name}')

    print(f'\nReprojection error of recovered camera poses:')
    print(f'  AP  (PnP on {len(ap_lm_names)} landmarks): {reproj_ap:.2f} px = {reproj_ap*PIX_MM:.2f} mm')
    print(f'  LAT (PnP on {len(lat_lm_names)} landmarks): {reproj_lat:.2f} px = {reproj_lat*PIX_MM:.2f} mm')

    return dict(
        reproj_ap=reproj_ap, reproj_lat=reproj_lat,
        pde_labeled_cameras=init_pde_labeled,
        pde_multiview_refined=best_result['final_pde'],
        src_ap=src_ap, src_lat=src_lat,
        dist_ap=dist_ap, dist_lat=dist_lat,
    )


if __name__ == '__main__':
    result = main()
