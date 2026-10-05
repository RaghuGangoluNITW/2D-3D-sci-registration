"""
run_full_pipeline.py
======================
Step 2 of the POC: take the retrieved 2D landmark positions from the
DRR-bank match (test_retrieval_synthetic.py's output) and run them
through EPnP + a local refinement (gradient-free Nelder-Mead on
image similarity, standing in for CMA-ES) to get a final 3D pose and
a real Projection Distance Error (PDE) in mm against the TRUE held-out
pose. This is the number that actually answers "does retrieval-based
registration achieve <2mm" -- not just "is the matched DRR a reasonable
nearest-neighbor" (already shown in test_retrieval_synthetic.py).

Run on a CLEAN target (synthetic DRR standing in for a hypothetical
clean pre-op-style X-ray) first, to establish the best case before any
real-X-ray domain gap is introduced.
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from load_verse_ct import load_verse_case
from drr_renderer import CTVolume, DRRRenderer

CASE_DIR = r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\ct_data\dataset-verse19validation'
BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\drr_bank')
OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')


def ncc(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(a * b))


def solve_pnp(pts3d, pts2d, K):
    """Standard EPnP via OpenCV. pts3d: (N,3) world mm, pts2d: (N,2) px."""
    n = len(pts3d)
    flag = cv2.SOLVEPNP_EPNP if n >= 4 else cv2.SOLVEPNP_SQPNP
    ok, rvec, tvec = cv2.solvePnP(pts3d.astype(np.float64), pts2d.astype(np.float64),
                                   K, np.zeros(4), flags=flag)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.flatten()


def project(pts3d, R, t, K):
    """Simple pinhole projection using R,t,K (camera-space convention:
    cam_pt = R @ world_pt + t, then apply K)."""
    cam = (R @ pts3d.T).T + t
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv


def main():
    manifest = json.loads((BANK_DIR / 'manifest.json').read_text())
    entries = manifest['entries']
    names = manifest['landmark_names']
    out_size = manifest['output_size']
    pix_mm = manifest['pixel_spacing_mm']
    sid_mm = manifest['sid_mm']

    case = load_verse_case(CASE_DIR)
    ct = CTVolume(case['volume_zyx'], case['spacing_xyz'], case['origin_xyz'], hu_threshold=180.0)
    renderer = DRRRenderer(ct)
    centroids = case['centroids_world']
    pts3d = np.array([centroids[n] for n in names])

    rng = np.random.default_rng(123)
    center = pts3d.mean(axis=0)
    true_az, true_el = 4.0, -3.0
    true_center = center + rng.normal(0, 10.0, size=3)
    src_t, pax_t, u_t, v_t = DRRRenderer.camera_basis_from_azimuth_elevation(
        true_center, sid_mm, true_az, true_el)
    target = renderer.render(src_t, pax_t, u_t, v_t, output_size=out_size,
                              pixel_spacing_mm=pix_mm, n_steps=180)
    true_uv = renderer.project_points(pts3d, src_t, pax_t, u_t, v_t, out_size, pix_mm, sid_mm)

    # ---- Step A: retrieval ----
    scores = []
    for e in entries:
        drr_img = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        scores.append(ncc(drr_img, target))
    best_i = int(np.argmax(scores))
    best = entries[best_i]
    retrieved_uv = np.array([best['landmarks_px'][n] for n in names])
    print(f'Retrieved match: {best["file"]}  az={best["azimuth_deg"]:.1f} el={best["elevation_deg"]:.1f}  '
          f'NCC={scores[best_i]:.4f}')

    # ---- Step B: EPnP from retrieved 2D positions ----
    # Simple pinhole intrinsics consistent with our renderer convention
    # (principal point at image center, "focal length" = sid_mm / pix_mm
    # since our renderer scales by sid/depth already in mm then /pix_mm).
    f_px = sid_mm / pix_mm
    cx = cy = (out_size - 1) / 2.0
    K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]], dtype=np.float64)

    # pts3d must be expressed relative to a consistent "camera-space"
    # convention for solvePnP; our renderer works directly in world mm
    # with explicit src/pax/u/v basis, so we instead solve PnP in that
    # same "detector-plane" convention: treat (u_comp, v_comp, depth)
    # as the standard camera frame (x=u, y=v, z=depth along pax).
    # Build a rotation matrix from world->camera using the RETRIEVED
    # entry's own basis as the EPnP's camera frame reference, i.e. we
    # solve for the pose correction needed, in that local frame.
    R_bank = np.stack([np.array(best['u_xyz']), np.array(best['v_xyz']), np.array(best['pax_xyz'])], axis=0)
    t_bank = -R_bank @ np.array(best['src_xyz'])

    R_pnp, t_pnp = solve_pnp(pts3d, retrieved_uv, K)
    uv_check = project(pts3d, R_pnp, t_pnp, K)
    reproj_err = np.linalg.norm(uv_check - retrieved_uv, axis=1).mean()
    print(f'EPnP self-reprojection error on retrieved points: {reproj_err:.2f}px')

    def pose_to_basis(R, t):
        src = -R.T @ t
        pax = R[2]
        u = R[0]
        v = R[1]
        return src, pax, u, v

    def render_from_Rt(R, t):
        src, pax, u, v = pose_to_basis(R, t)
        return renderer.render(src, pax, u, v, output_size=out_size,
                                pixel_spacing_mm=pix_mm, n_steps=120)

    def pde(R, t):
        uv_pred = project(pts3d, R, t, K)
        return float(np.mean(np.linalg.norm(uv_pred - true_uv, axis=1)) * pix_mm)

    init_pde = pde(R_pnp, t_pnp)
    print(f'Initial PDE (EPnP from retrieval, before refinement): {init_pde:.2f}mm')

    # ---- Step C: local refinement via image similarity (NCC), small search ----
    def rt_from_delta(x0):
        drx, dry, drz, dtx, dty, dtz = x0
        Rdelta = cv2.Rodrigues(np.array([drx, dry, drz]) * np.pi / 180.0)[0]
        R2 = Rdelta @ R_pnp
        t2 = t_pnp + np.array([dtx, dty, dtz])
        return R2, t2

    def cost(x0):
        R2, t2 = rt_from_delta(x0)
        drr2 = render_from_Rt(R2, t2)
        return 1.0 - ncc(drr2, target)

    # Nelder-Mead's default initial simplex step is far too small for this
    # problem's natural scale (degrees / mm) -- it converges in a handful of
    # evaluations without actually exploring the space (found via direct
    # debugging: nit=1, nfev=7 with default settings). Build an explicit
    # initial simplex with step sizes matched to each parameter's scale
    # (3 deg for rotation axes, 5mm for translation axes).
    x0 = np.zeros(6)
    step = np.array([3.0, 3.0, 3.0, 5.0, 5.0, 5.0])
    simplex = np.vstack([x0] + [x0 + step[i] * np.eye(6)[i] for i in range(6)])

    best_cost0 = cost(x0)
    res = minimize(cost, x0=x0, method='Nelder-Mead',
                    options=dict(xatol=0.02, fatol=1e-5, maxiter=400, maxfev=400,
                                 initial_simplex=simplex, adaptive=True))
    R_final, t_final = rt_from_delta(res.x)
    final_pde = pde(R_final, t_final)
    print(f'Refinement: nit={res.nit} nfev={res.nfev} cost {best_cost0:.4f} -> {res.fun:.4f}')
    print(f'Final PDE after refinement: {final_pde:.2f}mm')

    # visualize
    drr_final = render_from_Rt(R_final, t_final)
    tgt_u8 = (np.clip(target, 0, 1) * 255).astype('uint8')
    fin_u8 = (np.clip(drr_final, 0, 1) * 255).astype('uint8')
    side = np.hstack([tgt_u8, fin_u8])
    side_bgr = cv2.cvtColor(side, cv2.COLOR_GRAY2BGR)
    cv2.putText(side_bgr, 'TARGET', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(side_bgr, f'FINAL (PDE={final_pde:.2f}mm)', (out_size + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    OUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT_DIR / 'full_pipeline_synthetic_result.png'), side_bgr)

    print('\n=== SUMMARY (clean synthetic target) ===')
    print(f'  Retrieval NCC          : {scores[best_i]:.4f}')
    print(f'  Init PDE (EPnP only)   : {init_pde:.2f}mm')
    print(f'  Final PDE (refined)    : {final_pde:.2f}mm')
    print(f'  Target <2mm?           : {"YES" if final_pde < 2.0 else "NO"}')


if __name__ == '__main__':
    main()
