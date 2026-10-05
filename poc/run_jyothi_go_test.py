"""
run_jyothi_go_test.py
========================
The real numeric test: run retrieval (GO metric) -> EPnP -> GO-based
refinement against JYOTHI's REAL, doctor-verified, labeled X-rays, and
compute a TRUE PDE against her real ground truth (unlike the Radiopaedia
test, which had no shared ground truth at all).

This directly tests whether the fix found in run_full_pipeline_real_go.py
(GO instead of NCC for cross-modal retrieval) produces a real, numeric
improvement on real clinical data, compared to this project's past
NCC/CNN-landmark-based attempts (which never beat ~17-70mm PDE on real
X-rays, see project history).
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
from load_jyothi_ct import load_jyothi
from drr_renderer import CTVolume, DRRRenderer

BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\jyothi_data\drr_bank')
OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')
OUT_DIR.mkdir(exist_ok=True)


def go_sim(a, b):
    return gradient_orientation_similarity(a, b, sigma=2.0)


def preprocess_xray(xray_arr, out_size):
    """Resize + CLAHE-normalize a real X-ray (already [0,1]) to bank res,
    THEN suppress bright surgical instruments (needles/K-wires/screwdrivers).

    All six of JYOTHI's labeled sets are intraoperative frames with metal
    instruments crossing the spine (confirmed visually -- see POC/results/
    carm_full_contact_sheet.png). The main project's own metal_suppression.py
    exists specifically because these bright, high-gradient instrument
    pixels dominate GO similarity and drag retrieval/refinement toward a
    pose that aligns the *instrument*, not the bone. This was never applied
    in the first two passes of this test -- a real omission, not yet fixed.

    POLARITY NOTE: JYOTHI's raw Ziehm frames use the OPPOSITE intensity
    convention from the DRR renderer. In the raw image, direct-beam
    background (no attenuation) is near-white, bone is mid-gray, and
    metal instruments are the DARKEST pixels -- the reverse of the DRR's
    "bone-bright" convention (drr = 1 - transmission). suppress_metal()
    assumes metal is the brightest ~3% of pixels, so applying it directly
    to the raw-polarity image masks bright background clutter instead of
    the actual wires (verified visually -- see POC/results/
    metal_suppress_debug_ap1.png vs ..._inverted.png). We must invert to
    bone-bright FIRST, suppress metal (now correctly the brightest
    pixels), then keep that orientation (it also matches the DRR bank).
    """
    u8 = (np.clip(xray_arr, 0, 1) * 255).astype('uint8')
    resized = cv2.resize(u8, (out_size, out_size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(resized)
    img = 1.0 - (eq.astype(np.float32) / 255.0)   # raw-polarity -> bone-bright
    clean = suppress_thin_instruments(img, hard_floor=0.75, percentile=95.0,
                                       dilate_px=6, inpaint_r=5)
    return clean


def suppress_thin_instruments(img, hard_floor=0.75, percentile=95.0,
                               dilate_px=6, inpaint_r=5):
    """Metal suppression tailored to JYOTHI's thin K-wires/needles.

    suppress_metal() in the main project has a built-in early-exit: it skips
    suppression entirely when the detected mask covers < 10% of pixels
    (designed for images with large screw/rod implant areas). A thin needle
    or K-wire covers a much smaller fraction (~5%), so that guard always
    fired and silently returned the image unchanged on both earlier passes
    of this test -- the real reason metal suppression had zero effect
    before, not a logic error in the polarity fix itself. This local copy
    skips that guard (verified correct on this image via
    POC/results/metal_suppress_debug_ap1_inverted.png).
    """
    p_thresh = float(np.percentile(img, percentile))
    thresh = max(p_thresh, hard_floor)
    mask = (img > thresh).astype(np.uint8) * 255
    if mask.mean() < 1e-6:
        return img.copy()
    ksize = dilate_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask_d = cv2.dilate(mask, kernel)
    img8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    clean8 = cv2.inpaint(img8, mask_d, inpaintRadius=inpaint_r, flags=cv2.INPAINT_TELEA)
    return clean8.astype(np.float32) / 255.0


def main():
    manifest = json.loads((BANK_DIR / 'manifest.json').read_text())
    all_entries = manifest['entries']
    names = manifest['landmark_names']
    out_size = manifest['output_size']
    pix_mm_bank = manifest['pixel_spacing_mm']
    sid_mm = manifest['sid_mm']
    # Real, device-reported pixel spacing for THIS patient's C-arm session,
    # read from DICOM tag (0019,1014) on the raw frames each labeled set was
    # pixel-matched to (correlation=1.0000) -- not the generic 0.2mm/px
    # project default. See build_drr_bank_jyothi.py for the full story.
    real_pix_mm = 0.288

    data = load_jyothi()
    ct = CTVolume(data['volume_zyx'], data['spacing_xyz'], data['origin_xyz'], hu_threshold=180.0)
    renderer = DRRRenderer(ct)
    centroids = data['centroids_world']
    pts3d_all = np.array([centroids[n] for n in names])

    f_px = sid_mm / pix_mm_bank
    cx = cy = (out_size - 1) / 2.0
    K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]], dtype=np.float64)

    def pose_to_basis(R, t):
        return -R.T @ t, R[2], R[0], R[1]

    def render_from_Rt(R, t, n_steps=140):
        src, pax, u, v = pose_to_basis(R, t)
        # Must pass sdd_mm=sid_mm explicitly -- otherwise render() auto-picks
        # its own detector distance from the CT bounding box, which does NOT
        # match sid_mm (the value baked into K and into the bank's landmark
        # positions). See build_drr_bank_jyothi.py for the full explanation;
        # this is the same bug, hit a second time at refinement time.
        return renderer.render(src, pax, u, v, output_size=out_size,
                                pixel_spacing_mm=pix_mm_bank, n_steps=n_steps,
                                sdd_mm=sid_mm)

    def project(pts3d, R, t):
        cam = (R @ pts3d.T).T + t
        uv = (K @ cam.T).T
        return uv[:, :2] / uv[:, 2:3]

    results = []
    for s in data['xray_sets']:
        view = s['view']
        entries = [e for e in all_entries if e['base_view'] == view]
        target = preprocess_xray(s['image'], out_size)

        # Retrieval via GO
        scores = []
        for e in entries:
            drr_img = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            scores.append(go_sim(drr_img, target))
        best_i = int(np.argmax(scores))
        best = entries[best_i]
        retrieved_uv = np.array([best['landmarks_px'][n] for n in names])

        # Initialization: use the RETRIEVED bank entry's own known camera
        # pose directly, instead of re-deriving a pose via EPnP from the
        # reprojected landmarks.
        #
        # JYOTHI's vertebral centroids are nearly collinear in 3D (PCA of
        # the 6 centroids: 2nd/1st singular-value ratio = 0.13, max
        # perpendicular deviation from the best-fit line = 10mm). The main
        # project's own analysis (build_paper_discussion_report.py) found
        # this exact point layout makes EPnP catastrophically sensitive:
        # ~1mm of 2D landmark noise -> ~15 deg rotation / ~340mm translation
        # error. That instability, not the retrieval quality, is what was
        # driving refinement to blow UP instead of improving init_PDE in
        # 2/3 AP sets on every previous pass of this test.
        #
        # But EPnP is not actually necessary here: every DRR bank entry was
        # rendered from a known, exact camera pose (not reconstructed from
        # 2D points), and that exact pose is stored in the manifest. Using
        # it directly as the initial pose sidesteps the collinear-landmark
        # instability completely -- EPnP was only ever a detour.
        src0 = np.array(best['src_xyz'])
        pax0 = np.array(best['pax_xyz'])
        u0 = np.array(best['u_xyz'])
        v0 = np.array(best['v_xyz'])
        # Camera basis (u, v, pax as rows) is the rotation CT-world -> camera
        R_pnp = np.stack([u0, v0, pax0], axis=0)
        t_pnp = -R_pnp @ src0

        # Ground-truth 2D landmarks (REAL, doctor-labeled) in the ORIGINAL
        # 1024px image's pixel coordinates -- rescale our render-space (uv)
        # predictions to that frame for a true PDE in mm.
        scale_to_1024 = 1024.0 / out_size

        def pde_mm(R, t):
            uv_pred = project(pts3d_all, R, t) * scale_to_1024
            errs = []
            for i, n in enumerate(names):
                if n in s['landmarks_2d']:
                    gt = s['landmarks_2d'][n]
                    errs.append(np.linalg.norm(uv_pred[i] - gt) * real_pix_mm)
            return float(np.mean(errs)) if errs else float('nan')

        init_pde = pde_mm(R_pnp, t_pnp)

        # Refinement via GO
        def rt_from_delta(x0):
            drx, dry, drz, dtx, dty, dtz = x0
            Rdelta = cv2.Rodrigues(np.array([drx, dry, drz]) * np.pi / 180.0)[0]
            return Rdelta @ R_pnp, t_pnp + np.array([dtx, dty, dtz])

        # Hard bounds on refinement, in the SAME units as x0 (deg, mm). The
        # retrieval step already found the best-matching bank entry out of
        # a +/-10deg / +/-20mm jitter range around the true pose (that's
        # the whole premise of the bank); refinement only needs to correct
        # the residual gap between the discretely-sampled bank entry and
        # the true continuous pose, which should be well within that same
        # neighborhood. A coverage-fraction guard alone was NOT robust: a
        # degenerate "CT drifted almost off detector" pose can still clear
        # a coverage floor by chance (confirmed: one divergent case scored
        # coverage=0.032, just over a 0.03 floor, yet was a thin
        # meaningless sliver -- see POC/results/final_pose_check2.png) and
        # still match noise in the target by fluke, driving GO cost to
        # near 0. Hard bounds are a direct, physically-grounded fix rather
        # than another threshold to re-tune.
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

        print(f'{s["name"]:<12} ({view})  retrieval GO={scores[best_i]:.3f}  '
              f'init_PDE={init_pde:.1f}mm  final_PDE={final_pde:.1f}mm  '
              f'nit={res.nit}')
        results.append(dict(name=s['name'], view=view, init_pde=init_pde, final_pde=final_pde))

        # Save a visual for the first set of each view
        if s['name'] in ('AP-SET-1', 'LAT-SET-1'):
            drr_final = render_from_Rt(R_final, t_final, n_steps=180)
            tgt_u8 = (np.clip(target, 0, 1) * 255).astype('uint8')
            fin_u8 = (np.clip(drr_final, 0, 1) * 255).astype('uint8')
            side = np.hstack([tgt_u8, fin_u8])
            side_bgr = cv2.cvtColor(side, cv2.COLOR_GRAY2BGR)
            cv2.putText(side_bgr, 'JYOTHI REAL X-RAY', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(side_bgr, f'FINAL (PDE={final_pde:.1f}mm)', (out_size + 10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imwrite(str(OUT_DIR / f'jyothi_go_{s["name"]}_result.png'), side_bgr)

    print('\n=== SUMMARY: JYOTHI real X-rays, GO-based retrieval pipeline ===')
    for r in results:
        print(f'  {r["name"]:<12} ({r["view"]}): {r["init_pde"]:.1f}mm -> {r["final_pde"]:.1f}mm')
    pdes = [r['final_pde'] for r in results]
    print(f'\n  Mean final PDE: {np.mean(pdes):.1f}mm')
    print(f'  Sets under 2mm: {sum(1 for p in pdes if p < 2.0)}/{len(pdes)}')


if __name__ == '__main__':
    main()
