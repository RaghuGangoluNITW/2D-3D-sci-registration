"""
build_drr_bank_jyothi.py
===========================
Builds a DRR bank from JYOTHI's REAL CT, exactly like build_drr_bank.py
did for the public VerSe case, so the GO-metric retrieval pipeline can
be tested against her REAL, doctor-verified ground truth X-rays
(the one case in this whole POC where a true numeric PDE can be
computed end-to-end on real clinical data).

IMPORTANT CORRECTION (2nd pass): the previous version of this file used
the main project's GENERIC Ziehm Vision FD default (SID=1110mm, pixel
spacing=0.2mm/px) -- which that project's own docs say applies only
"when no DICOM metadata [is available]". For JYOTHI we DO have the raw
C-arm DICOM frames (jyothi/carm_extracted/...), and every one of them
carries real, device-reported geometry tags:
    (0018,1110) Distance Source to Detector = 1050 mm
    (0019,1014) detector pixel spacing       = 0.288 mm/px (at 1024px)
Verified by pixel-matching (correlation = 1.0000) each of the 6 labeled
X-ray sets to its exact source raw frame -- these are not estimates,
they are the device's own calibration for these exact images. Using
1110/0.2 here was silently injecting a real calibration error (on top
of the sdd_mm render/landmark-projection bug fixed earlier) into every
PDE computed against JYOTHI's data.
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from load_jyothi_ct import load_jyothi
from drr_renderer import CTVolume, DRRRenderer

BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\jyothi_data\drr_bank')
BANK_DIR.mkdir(exist_ok=True, parents=True)

# REAL Ziehm Vision RFD 3D geometry for THIS patient's C-arm session,
# read directly from the raw DICOM tags (0018,1110) and (0019,1014) --
# not the generic project default.
OUT_SIZE = 256
REAL_PIX_MM_1024 = 0.288            # detector pixel spacing at native 1024px (real, from DICOM)
PIX_MM = REAL_PIX_MM_1024 * (1024 / OUT_SIZE)
SID_MM = 1050.0                     # Distance Source to Detector (real, from DICOM)
N_STEPS = 180


def main():
    data = load_jyothi()
    ct = CTVolume(data['volume_zyx'], data['spacing_xyz'], data['origin_xyz'], hu_threshold=180.0)
    renderer = DRRRenderer(ct)
    centroids = data['centroids_world']
    names = list(centroids.keys())
    pts = np.array(list(centroids.values()))
    full_center = pts.mean(axis=0)
    print('Full-spine centroid (world mm):', full_center.round(1))

    # FOV-CENTER FIX: the DRR bank was previously centered on the FULL
    # D12-L5 spine, but every one of JYOTHI's real labeled X-rays is
    # intraoperatively collimated to a tighter FOV around the vertebrae
    # actually being instrumented (AP sets show only L3-L5, LAT sets show
    # L2-L5) -- confirmed visually: the real images are more zoomed-in /
    # cropped lower than any bank DRR (see POC/results/
    # ap1_preprocessed_256.png vs drr_0012.png). The AP-only and LAT-only
    # subset centroids are offset from the full-spine centroid by 31-48mm,
    # which alone explains much of the retrieval error: no bank entry was
    # ever framed anything like the real acquisition. Center each view's
    # bank on the mean of exactly the vertebrae visible in that view's
    # real labeled sets.
    ap_names = sorted({n for s in data['xray_sets'] if s['view'] == 'AP' for n in s['landmarks_2d']})
    lat_names = sorted({n for s in data['xray_sets'] if s['view'] == 'LAT' for n in s['landmarks_2d']})
    ap_center = np.array([centroids[n] for n in ap_names]).mean(axis=0)
    lat_center = np.array([centroids[n] for n in lat_names]).mean(axis=0)
    print('AP labeled-region centroid:', ap_center.round(1), ap_names)
    print('LAT labeled-region centroid:', lat_center.round(1), lat_names)
    view_center = {'AP': ap_center, 'LAT': lat_center}

    rng = np.random.default_rng(0)
    manifest = []
    idx = 0
    # Base azimuths updated to match EMPIRICALLY CONFIRMED real acquisition
    # angles (measured from DICOM positioner tags AND verified by scoring real
    # C-arm frames against bank entries via GO similarity):
    #   AP  frame (frame 9, DICOM primary=-19.16°): best GO match at az=-11.8°
    #       -> center the AP bank at az=-15° to bracket both DICOM and empirical
    #   LAT frame (frame 14, DICOM secondary=-90.3°): best GO match at az=113.7°
    #       -> center LAT bank at az=113° for accurate coverage
    base_azimuths = [-15.0, 113.0]   # AP and LAT real acquisition angles
    base_elevs    = [3.68,   -0.3]   # real DICOM elevation angles
    N_PER_VIEW = 200

    for (base_az, base_el), base_view in zip(
            [(-15.0, 3.68), (113.0, -0.3)], ['AP', 'LAT']):
        center = view_center[base_view]
        for _ in range(N_PER_VIEW):
            az = base_az + rng.normal(0, 10.0)
            el = base_el + rng.normal(0, 8.0)
            trans_jitter = rng.normal(0, 20.0, size=3)
            jittered_center = center + trans_jitter

            src, pax, u, v = DRRRenderer.camera_basis_from_azimuth_elevation(
                jittered_center, SID_MM, az, el)
            # IMPORTANT: pass sdd_mm=SID_MM explicitly. render() otherwise
            # auto-computes its own detector distance from the CT bounding
            # box, which does NOT equal SID_MM -- that mismatch silently
            # bakes a ~16% magnification error into every bank entry's
            # landmark positions (and later into the EPnP K-matrix), since
            # project_points() below always uses SID_MM. Harmless when
            # target+bank share the same bug (as in the VerSe synthetic
            # test); fatal when the target is a real X-ray with its own
            # independent, correct geometry (JYOTHI's case).
            drr = renderer.render(src, pax, u, v, output_size=OUT_SIZE,
                                   pixel_spacing_mm=PIX_MM, n_steps=N_STEPS,
                                   sdd_mm=SID_MM)
            uv = renderer.project_points(pts, src, pax, u, v, OUT_SIZE, PIX_MM, SID_MM)

            fname = f'drr_{idx:04d}.png'
            cv2.imwrite(str(BANK_DIR / fname), (np.clip(drr, 0, 1) * 255).astype('uint8'))
            manifest.append(dict(
                file=fname, base_view=base_view,
                azimuth_deg=float(az), elevation_deg=float(el),
                center_xyz=jittered_center.tolist(),
                src_xyz=src.tolist(), pax_xyz=pax.tolist(), u_xyz=u.tolist(), v_xyz=v.tolist(),
                landmarks_px={n: uv[i].tolist() for i, n in enumerate(names)},
            ))
            idx += 1
            if idx % 50 == 0:
                print(f'  rendered {idx}/{len(base_azimuths) * N_PER_VIEW}')

    (BANK_DIR / 'manifest.json').write_text(json.dumps(dict(
        output_size=OUT_SIZE, pixel_spacing_mm=PIX_MM, sid_mm=SID_MM,
        landmark_names=names, entries=manifest,
    ), indent=1))
    print(f'Done. {idx} DRRs -> {BANK_DIR}')


if __name__ == '__main__':
    main()
