"""
run_deepfluoro_go_test.py
=============================
The real numeric test on DeepFluoro: run retrieval (GO metric) -> pose
init from the retrieved DRR's own known camera pose -> GO-based local
refinement, against REAL cadaveric fluoroscopy, and compute a TRUE PDE
against DeepFluoro's real, INDEPENDENT 2D ground-truth landmarks (from
the original 2020 researcher annotation -- never touched by this
pipeline's retrieval or refinement steps, used ONLY here to score the
final result after the fact).

No manual intervention anywhere in the pipeline being tested:
  1. CT 3D landmarks: DeepFluoro's own 2020 annotation (given, not
     auto-detected -- same caveat as every other POC in this project).
  2. DRR bank: rendered from known synthetic poses sampled across the
     REAL measured pose range of this specimen's actual acquisitions
     (see build_drr_bank_deepfluoro.py) -- each DRR's 2D landmark
     positions are computed for free from the known pose, never
     measured by eye.
  3. Retrieval: GO (Gradient Orientation) similarity between each real
     X-ray and every bank DRR -- the same metric fix found necessary on
     the Radiopaedia/JYOTHI tests.
  4. Pose init: the retrieved DRR's own exact known camera pose (not
     EPnP derived from projected points) -- sidesteps any landmark
     collinearity issue (moot here anyway: this dataset's 14 pelvis
     landmarks are NOT collinear, PCA ratio 0.72/0.52 vs JYOTHI's
     0.11-0.13).
  5. Refinement: Nelder-Mead on GO similarity alone, bounded to the
     bank's own sampled neighborhood, with a coverage guard against
     degenerate near-blank poses (same fixes as the JYOTHI test).
  6. Scoring: PDE in mm against gt_landmarks_2d -- real, independent,
     never used anywhere above.
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
from deepfluoro_loader import DeepFluoroLoader, PIXEL_SPACING_MM
from load_deepfluoro import H5_PATH
from drr_renderer import CTVolume, DRRRenderer

SPECIMEN_ID = '17-1882'
BANK_DIR = Path(rf'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\deepfluoro_data\{SPECIMEN_ID}\drr_bank')
OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')
OUT_DIR.mkdir(exist_ok=True)


def go_sim(a, b):
    return gradient_orientation_similarity(a, b, sigma=2.0)


def preprocess_xray(xray_raw, out_size):
    """Resize + CLAHE-normalize a real fluoroscopy image (already [0,1],
    bone-bright convention per deepfluoro_loader's log-transform
    preprocessing) to bank resolution. No metal suppression needed --
    these are cadaveric specimens, no surgical instruments present."""
    u8 = (np.clip(xray_raw, 0, 1) * 255).astype('uint8')
    resized = cv2.resize(u8, (out_size, out_size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(resized)
    return eq.astype(np.float32) / 255.0


def main():
    manifest = json.loads((BANK_DIR / 'manifest.json').read_text())
    all_entries = manifest['entries']
    names = manifest['landmark_names']
    out_size = manifest['output_size']
    pix_mm_bank = manifest['pixel_spacing_mm']
    sdd_mm = manifest['sdd_mm']
    real_pix_mm = PIXEL_SPACING_MM   # 0.194mm/px, DeepFluoro's real calibrated value

    loader = DeepFluoroLoader(full_res_path=H5_PATH)
    spec = loader.load_specimen(SPECIMEN_ID, verbose=False)
    ct = CTVolume(spec.ct_volume, spec.ct_spacing, spec.ct_origin, hu_threshold=150.0)
    renderer = DRRRenderer(ct)
    pts3d_all = np.array([spec.landmarks_3d[n] for n in names])

    def pose_to_basis(R, t):
        return -R.T @ t, R[2], R[0], R[1]

    def render_from_Rt(R, t, n_steps=140):
        src, pax, u, v = pose_to_basis(R, t)
        return renderer.render(src, pax, u, v, output_size=out_size,
                                pixel_spacing_mm=pix_mm_bank, n_steps=n_steps,
                                sdd_mm=sdd_mm)

    def project_world(pts3d, R, t, px_size_mm):
        """Project world XYZ points to pixel coords at the given pixel
        size, using drr_renderer's own (src,pax,u,v)-basis camera
        convention -- the SAME one the bank/renderer use internally.

        IMPORTANT: this does NOT need DeepFluoro's XZY axis swap. That
        swap is specific to deepfluoro_loader.py's R_proj/t_proj
        parameterization; it is not a universal requirement, and this
        POC's (src,pax,u,v) basis is a different, self-consistent
        convention. Verified directly: calling drr_renderer.project_points
        with the TRUE ground-truth (src,pax,u,v) reproduces
        gt_landmarks_2d to <0.1px (see session sanity check) -- so this
        same math, with the OPTIMIZED (R,t) plugged in via pose_to_basis,
        is the correct and only thing needed to score PDE here. An
        earlier version of this function used deepfluoro_loader's K/R/t
        convention directly without the XZY swap it actually requires,
        which gave a nonsense ~640mm PDE even for the TRUE pose -- caught
        by sanity-checking against the true pose before trusting any
        result, not assumed to be fine.
        """
        src, pax, u, v = pose_to_basis(R, t)
        return renderer.project_points(pts3d, src, pax, u, v,
                                        FULL_RES_SIZE, px_size_mm, sdd_mm)

    FULL_RES_SIZE = 1536

    # FOV FILTER: this dataset's own metadata (gt-poses/left-femur-good-fov,
    # right-femur-good-fov) flags each real projection as intentionally
    # collimated/cropped to ONE hip for per-bone (femur) registration, or
    # left wide for whole-pelvis registration. Confirmed by direct visual
    # inspection (see POC/results/deepfluoro_proj008_full.png): a flagged
    # "good fov" hip view shows only ONE femoral head/acetabulum filling
    # most of the frame, cropping out the rest of the pelvis -- a real,
    # intentional FOV difference the DRR bank's fixed whole-pelvis framing
    # cannot match (same category of bug as the FOV mismatch found and
    # fixed in the JYOTHI test, now caught here too). Evaluating only on
    # projections where NEITHER femur flag is set keeps this test
    # consistent with what the bank was actually built to represent --
    # whole-pelvis framing -- rather than silently testing against a
    # different acquisition protocol the bank was never designed for.
    import h5py
    with h5py.File(H5_PATH, 'r') as f:
        pelvis_proj_keys = set()
        for pk, pg in f[SPECIMEN_ID]['projections'].items():
            lf = int(pg['gt-poses']['left-femur-good-fov'][()])
            rf = int(pg['gt-poses']['right-femur-good-fov'][()])
            if lf == 0 and rf == 0:
                pelvis_proj_keys.add(pk)

    results = []
    all_projs = [p for p in spec.valid_projections(max_reproj_px=2.0)
                 if p.proj_key in pelvis_proj_keys]
    eval_projs = all_projs[:14]
    print(f'Evaluating {len(eval_projs)} whole-pelvis-FOV real projections '
          f'(of {len(pelvis_proj_keys)} total flagged whole-pelvis, '
          f'{sum(1 for p in spec.valid_projections(max_reproj_px=2.0))} valid overall)')

    # Sanity check: PDE at the TRUE ground-truth pose must be ~0. Catches
    # any convention mismatch before trusting any downstream number.
    #
    # IMPORTANT: proj.R_proj/proj.t_proj are in DeepFluoro's OWN
    # XZY-swapped convention (deepfluoro_loader.py) -- they are NOT
    # directly compatible with this script's pose_to_basis()/project_world
    # (which expect R's rows to BE the u,v,pax detector/principal-axis
    # basis vectors directly, no swap). Feeding R_proj/t_proj into
    # pose_to_basis() was a second instance of the same convention
    # mismatch bug (first caught via the raw K-matrix approach, now
    # caught again here) -- get src/pax/u/v from proj's OWN validated
    # accessor methods instead, which already return world-XYZ vectors
    # directly usable by drr_renderer's project_points().
    _check = eval_projs[0]
    _src = _check.source_position()
    _pax = _check.principal_axis()
    _u, _v = _check.detector_axes()
    _uv_true = renderer.project_points(pts3d_all, _src, _pax, _u, _v,
                                        FULL_RES_SIZE, real_pix_mm, sdd_mm)
    _errs = [np.linalg.norm(_uv_true[i] - _check.gt_landmarks_2d[n]) * real_pix_mm
             for i, n in enumerate(names) if n in _check.gt_landmarks_2d]
    _sanity_pde = float(np.mean(_errs))
    print(f'[sanity check] PDE at TRUE pose ({_check.proj_key}): {_sanity_pde:.4f}mm '
          f'(must be ~0)')
    assert _sanity_pde < 0.5, f'Projection convention bug: true-pose PDE={_sanity_pde:.1f}mm, expected ~0'

    for proj in eval_projs:
        target = preprocess_xray(proj.image_raw, out_size)

        # Retrieval via GO over the WHOLE bank (no AP/LAT split needed --
        # this dataset has one continuous pose family, not two discrete
        # views like the spine patients).
        scores = []
        for e in all_entries:
            drr_img = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            scores.append(go_sim(drr_img, target))
        best_i = int(np.argmax(scores))
        best = all_entries[best_i]

        # Pose init: retrieved DRR's own known camera pose (no EPnP
        # needed -- see module docstring).
        src0 = np.array(best['src_xyz'])
        pax0 = np.array(best['pax_xyz'])
        u0 = np.array(best['u_xyz'])
        v0 = np.array(best['v_xyz'])
        R_init = np.stack([u0, v0, pax0], axis=0)
        t_init = -R_init @ src0

        def pde_mm(R, t):
            # Project directly into the real 1536px / 0.194mm-per-px frame
            # (same frame gt_landmarks_2d is stored in) -- no separate
            # rescale step needed.
            uv_pred = project_world(pts3d_all, R, t, real_pix_mm)
            errs = []
            for i, n in enumerate(names):
                if n in proj.gt_landmarks_2d:
                    gt = proj.gt_landmarks_2d[n]
                    errs.append(np.linalg.norm(uv_pred[i] - gt) * real_pix_mm)
            return float(np.mean(errs)) if errs else float('nan')

        init_pde = pde_mm(R_init, t_init)

        def rt_from_delta(x0, R0=R_init, t0=t_init):
            drx, dry, drz, dtx, dty, dtz = x0
            Rdelta = cv2.Rodrigues(np.array([drx, dry, drz]) * np.pi / 180.0)[0]
            return Rdelta @ R0, t0 + np.array([dtx, dty, dtz])

        # Same bounded-refinement + coverage-guard fixes validated on the
        # JYOTHI test: refinement must not wander outside the bank's own
        # sampled neighborhood, and must not reward a near-blank DRR.
        BOUNDS = np.array([15.0, 15.0, 15.0, 25.0, 25.0, 25.0])

        def cost(x0):
            if np.any(np.abs(x0) > BOUNDS):
                return 1.0
            R2, t2 = rt_from_delta(x0)
            drr2 = render_from_Rt(R2, t2)
            coverage = np.count_nonzero(drr2) / max(drr2.size, 1)
            if coverage < 0.1:
                return 1.0
            return 1.0 - go_sim(drr2, target)

        x0 = np.zeros(6)
        step = np.array([3.0, 3.0, 3.0, 5.0, 5.0, 5.0])
        simplex = np.vstack([x0] + [x0 + step[i] * np.eye(6)[i] for i in range(6)])
        res = minimize(cost, x0=x0, method='Nelder-Mead',
                        options=dict(xatol=0.02, fatol=1e-5, maxiter=400, maxfev=400,
                                     initial_simplex=simplex, adaptive=True))
        R_final, t_final = rt_from_delta(res.x)
        final_pde = pde_mm(R_final, t_final)

        print(f'{proj.proj_key:<8}  retrieval GO={scores[best_i]:.3f}  '
              f'init_PDE={init_pde:.1f}mm  final_PDE={final_pde:.1f}mm  nit={res.nit}')
        results.append(dict(proj_key=proj.proj_key, init_pde=init_pde, final_pde=final_pde))

        if proj.proj_key in (eval_projs[0].proj_key, eval_projs[len(eval_projs)//2].proj_key):
            drr_final = render_from_Rt(R_final, t_final, n_steps=200)
            tgt_u8 = (np.clip(target, 0, 1) * 255).astype('uint8')
            fin_u8 = (np.clip(drr_final, 0, 1) * 255).astype('uint8')
            side = np.hstack([tgt_u8, fin_u8])
            side_bgr = cv2.cvtColor(side, cv2.COLOR_GRAY2BGR)
            cv2.putText(side_bgr, 'REAL FLUOROSCOPY', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(side_bgr, f'FINAL (PDE={final_pde:.1f}mm)', (out_size + 10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imwrite(str(OUT_DIR / f'deepfluoro_go_{proj.proj_key}_result.png'), side_bgr)

    print('\n=== SUMMARY: DeepFluoro real fluoroscopy, GO-based retrieval pipeline ===')
    for r in results:
        print(f'  {r["proj_key"]:<8}: {r["init_pde"]:.1f}mm -> {r["final_pde"]:.1f}mm')
    init_pdes = [r['init_pde'] for r in results]
    final_pdes = [r['final_pde'] for r in results]
    print(f'\n  Mean init PDE : {np.mean(init_pdes):.1f}mm')
    print(f'  Mean final PDE: {np.mean(final_pdes):.1f}mm')
    print(f'  Sets under 2mm (final): {sum(1 for p in final_pdes if p < 2.0)}/{len(final_pdes)}')
    print(f'  Sets under 5mm (final): {sum(1 for p in final_pdes if p < 5.0)}/{len(final_pdes)}')

    (OUT_DIR / 'deepfluoro_go_results.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
