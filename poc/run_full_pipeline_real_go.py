"""
run_full_pipeline_real_go.py
===============================
Fixed version of run_full_pipeline_real.py: uses GO (Gradient
Orientation) similarity instead of raw-intensity NCC for BOTH the
retrieval/filter step AND the refinement step. GO compares gradient
DIRECTION, not brightness, which is why it should (and did, in
test_retrieval_metrics_real.py) survive the DRR-vs-real-X-ray domain
gap much better than NCC.

Still cannot compute a true numeric PDE here (no shared ground-truth
CT for this specific real-world patient -- same caveat as before), but
we CAN now report:
  1. Whether the retrieved pose is visually/structurally more plausible
     than the NCC-based retrieval.
  2. Whether local GO-based refinement, started from that retrieval,
     converges to visually better alignment (same mechanism as the
     project's existing GO-CMAES pipeline, now applied end-to-end here).
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
from run_full_pipeline_real import preprocess_real_xray, REAL_XRAY_PATH

CASE_DIR = r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\ct_data\dataset-verse19validation'
BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\drr_bank')
OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')


def go_sim(a, b):
    return gradient_orientation_similarity(a, b, sigma=2.0)


def main():
    manifest = json.loads((BANK_DIR / 'manifest.json').read_text())
    entries = [e for e in manifest['entries'] if e['base_view'] == 'AP']
    names = manifest['landmark_names']
    out_size = manifest['output_size']
    pix_mm = manifest['pixel_spacing_mm']
    sid_mm = manifest['sid_mm']

    case = load_verse_case(CASE_DIR)
    ct = CTVolume(case['volume_zyx'], case['spacing_xyz'], case['origin_xyz'], hu_threshold=180.0)
    renderer = DRRRenderer(ct)
    centroids = case['centroids_world']
    pts3d = np.array([centroids[n] for n in names])

    target = preprocess_real_xray(REAL_XRAY_PATH, out_size)

    # ---- Step A: retrieval using GO instead of NCC ----
    scores = []
    for e in entries:
        drr_img = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        scores.append(go_sim(drr_img, target))
    best_i = int(np.argmax(scores))
    best = entries[best_i]
    retrieved_uv = np.array([best['landmarks_px'][n] for n in names])
    print(f'Retrieved match (GO): {best["file"]}  az={best["azimuth_deg"]:.1f} el={best["elevation_deg"]:.1f}  '
          f'GO={scores[best_i]:.4f}')

    # ---- Step B: EPnP from retrieved 2D positions ----
    f_px = sid_mm / pix_mm
    cx = cy = (out_size - 1) / 2.0
    K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(pts3d.astype(np.float64), retrieved_uv.astype(np.float64),
                                   K, np.zeros(4), flags=cv2.SOLVEPNP_EPNP)
    R_pnp, _ = cv2.Rodrigues(rvec)
    t_pnp = tvec.flatten()

    def pose_to_basis(R, t):
        return -R.T @ t, R[2], R[0], R[1]

    def render_from_Rt(R, t, n_steps=120):
        src, pax, u, v = pose_to_basis(R, t)
        return renderer.render(src, pax, u, v, output_size=out_size,
                                pixel_spacing_mm=pix_mm, n_steps=n_steps)

    drr_init = render_from_Rt(R_pnp, t_pnp)
    init_go = go_sim(drr_init, target)
    print(f'Initial GO similarity (EPnP pose vs real target): {init_go:.4f}')

    # ---- Step C: local refinement via GO similarity ----
    def rt_from_delta(x0):
        drx, dry, drz, dtx, dty, dtz = x0
        Rdelta = cv2.Rodrigues(np.array([drx, dry, drz]) * np.pi / 180.0)[0]
        return Rdelta @ R_pnp, t_pnp + np.array([dtx, dty, dtz])

    def cost(x0):
        R2, t2 = rt_from_delta(x0)
        drr2 = render_from_Rt(R2, t2)
        return 1.0 - go_sim(drr2, target)

    x0 = np.zeros(6)
    step = np.array([3.0, 3.0, 3.0, 5.0, 5.0, 5.0])
    simplex = np.vstack([x0] + [x0 + step[i] * np.eye(6)[i] for i in range(6)])
    res = minimize(cost, x0=x0, method='Nelder-Mead',
                    options=dict(xatol=0.02, fatol=1e-5, maxiter=400, maxfev=400,
                                 initial_simplex=simplex, adaptive=True))
    R_final, t_final = rt_from_delta(res.x)
    final_go = 1.0 - res.fun
    print(f'Refinement: nit={res.nit} nfev={res.nfev}  GO {init_go:.4f} -> {final_go:.4f}')

    drr_final = render_from_Rt(R_final, t_final, n_steps=180)
    tgt_u8 = (np.clip(target, 0, 1) * 255).astype('uint8')
    init_u8 = (np.clip(drr_init, 0, 1) * 255).astype('uint8')
    fin_u8 = (np.clip(drr_final, 0, 1) * 255).astype('uint8')
    row = np.hstack([tgt_u8, init_u8, fin_u8])
    row_bgr = cv2.cvtColor(row, cv2.COLOR_GRAY2BGR)
    for i, lbl in enumerate(['REAL TARGET', f'INIT (GO={init_go:.3f})', f'FINAL (GO={final_go:.3f})']):
        cv2.putText(row_bgr, lbl, (i * out_size + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
    OUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT_DIR / 'full_pipeline_real_go_result.png'), row_bgr)
    print(f'\nSaved -> {OUT_DIR / "full_pipeline_real_go_result.png"}')

    print('\n=== SUMMARY (real public X-ray, GO metric) ===')
    print(f'  Retrieval GO            : {scores[best_i]:.4f}')
    print(f'  Init GO (EPnP pose)     : {init_go:.4f}')
    print(f'  Final GO (refined)      : {final_go:.4f}')
    print(f'  (No ground-truth CT for this real patient -> no numeric PDE possible.')
    print(f'   This measures whether the pipeline CONVERGES to a visually/structurally')
    print(f'   plausible pose on real data, using a domain-gap-robust metric.)')


if __name__ == '__main__':
    main()
