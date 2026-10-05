"""
build_drr_bank_deepfluoro.py
================================
Builds a DRR bank from a real DeepFluoro cadaveric pelvis CT, exactly
like build_drr_bank.py (VerSe) and build_drr_bank_jyothi.py did, so the
GO-metric retrieval pipeline can be tested end-to-end on real cadaveric
fluoroscopy with real, independent ground truth.

Camera geometry: DeepFluoro's own calibrated SDD (source-to-detector
distance, ~1020mm) and pixel spacing (0.194mm/px at 1536px native res),
read directly from mslevel_reg's deepfluoro_loader.py constants.

IMPORTANT distinction (got this wrong on the first pass of this file,
fixed before running): camera_basis_from_azimuth_elevation()'s `sid_mm`
argument is actually the SOURCE-TO-ORBIT-CENTER distance (how far the
camera sits from the pelvis landmark centroid we orbit around) -- NOT
the same thing as SDD (source-to-detector distance), which is a fixed
real device property. The pelvis sits BETWEEN the source and detector,
so these two distances are genuinely different numbers:
  - source-to-pelvis-center: measured directly from this specimen's own
    110 real projections = 696-898mm (varies per acquisition).
  - SDD (source-to-detector): fixed real calibrated value = ~1020mm.
Treating them as the same number (as earlier POC bank-builders did, for
VerSe/JYOTHI, where it didn't matter because target+bank shared the same
simplified convention) would have been wrong here, since we have real,
independently-measured values for both and want the bank's magnification
to actually match the real acquisitions.

No manual landmark marking anywhere in this script: the 14 3D landmarks
come from the DeepFluoro dataset's own 2020 annotation (done once by the
original researchers for their published paper), used here only to
compute each DRR's "free" 2D landmark positions via known math (the
camera projection), exactly as in every other POC bank-builder.
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'mslevel_reg' / 'src'))
from deepfluoro_loader import DeepFluoroLoader, FX_POS, PIXEL_SPACING_MM, FULL_RES_SIZE
from load_deepfluoro import H5_PATH
from drr_renderer import CTVolume, DRRRenderer

OUT_SIZE = 256
# Scale real 0.194mm/px (at 1536px) to our render resolution
PIX_MM = PIXEL_SPACING_MM * (FULL_RES_SIZE / OUT_SIZE)
SDD_MM = FX_POS * PIXEL_SPACING_MM   # ~1020mm, DeepFluoro's real calibrated SDD
N_STEPS = 200

SPECIMEN_ID = '17-1882'
BANK_DIR = Path(rf'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\deepfluoro_data\{SPECIMEN_ID}\drr_bank')
BANK_DIR.mkdir(exist_ok=True, parents=True)


def main():
    loader = DeepFluoroLoader(full_res_path=H5_PATH)
    spec = loader.load_specimen(SPECIMEN_ID, verbose=True)
    ct = CTVolume(spec.ct_volume, spec.ct_spacing, spec.ct_origin, hu_threshold=150.0)
    renderer = DRRRenderer(ct)
    names, pts = spec.get_landmark_array()
    center = pts.mean(axis=0)
    print('Pelvis landmark centroid (world mm):', center.round(1))

    rng = np.random.default_rng(0)
    manifest = []
    idx = 0
    # Measured the REAL pose spread across all 110 valid real projections
    # of this specimen first (checked separately before writing this):
    #   azimuth: -18.1 to +29.3 deg
    #   elevation: -6.5 to +36.1 deg
    #   source-to-pelvis-center distance: 696 to 898mm
    # Sample uniformly across that measured range (not a narrow jitter
    # around a single guessed base pose) -- this is exactly the "bank
    # doesn't cover the true pose" failure mode that broke the JYOTHI
    # test, now avoided by measuring the real range first.
    N_TOTAL = 500

    for _ in range(N_TOTAL):
        az = rng.uniform(-25.0, 35.0)
        el = rng.uniform(-12.0, 42.0)
        src_to_center_mm = rng.uniform(680.0, 920.0)
        trans_jitter = rng.normal(0, 15.0, size=3)
        jittered_center = center + trans_jitter

        src, pax, u, v = DRRRenderer.camera_basis_from_azimuth_elevation(
            jittered_center, src_to_center_mm, az, el)
        drr = renderer.render(src, pax, u, v, output_size=OUT_SIZE,
                               pixel_spacing_mm=PIX_MM, n_steps=N_STEPS,
                               sdd_mm=SDD_MM)
        uv = renderer.project_points(pts, src, pax, u, v, OUT_SIZE, PIX_MM, SDD_MM)

        fname = f'drr_{idx:04d}.png'
        cv2.imwrite(str(BANK_DIR / fname), (np.clip(drr, 0, 1) * 255).astype('uint8'))
        manifest.append(dict(
            file=fname, azimuth_deg=float(az), elevation_deg=float(el),
            src_to_center_mm=float(src_to_center_mm),
            center_xyz=jittered_center.tolist(),
            src_xyz=src.tolist(), pax_xyz=pax.tolist(), u_xyz=u.tolist(), v_xyz=v.tolist(),
            landmarks_px={n: uv[i].tolist() for i, n in enumerate(names)},
        ))
        idx += 1
        if idx % 50 == 0:
            print(f'  rendered {idx}/{N_TOTAL}')

    (BANK_DIR / 'manifest.json').write_text(json.dumps(dict(
        output_size=OUT_SIZE, pixel_spacing_mm=PIX_MM, sdd_mm=SDD_MM,
        landmark_names=names, entries=manifest,
    ), indent=1))
    print(f'Done. {idx} DRRs -> {BANK_DIR}')


if __name__ == '__main__':
    main()
