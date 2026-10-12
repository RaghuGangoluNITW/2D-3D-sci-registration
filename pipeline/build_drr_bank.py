"""
build_drr_bank.py
===================
Generates the "DRR bank" -- many synthetic X-rays rendered from the CT at
different camera angles, each one tagged with its own 2D landmark pixel
positions (computed for free from the known pose, not measured by hand).

This is the step the user described as: "cut the video into multiple
slices of frames from every angle where I also have the window of the
kitchen seen clearly too" -- except, importantly, the per-DRR landmark
positions are a computed byproduct of the known render pose, not
something visually read off the image.

Covers a pose range around a true AP view and a true LAT view (azimuth
0 deg and 90 deg respectively), with a modest spread in azimuth,
elevation, and translation -- wide enough to require a real retrieval
step to find the best match, not so wide that no DRR is ever close.
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from load_verse_ct import load_verse_case
from drr_renderer import CTVolume, DRRRenderer

CASE_DIR = r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\ct_data\dataset-verse19validation'
BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\drr_bank')
BANK_DIR.mkdir(exist_ok=True)

OUT_SIZE = 256
PIX_MM = 1.2
SID_MM = 1000.0
N_STEPS = 180


def main():
    case = load_verse_case(CASE_DIR)
    ct = CTVolume(case['volume_zyx'], case['spacing_xyz'], case['origin_xyz'], hu_threshold=180.0)
    renderer = DRRRenderer(ct)
    centroids = case['centroids_world']
    names = list(centroids.keys())
    pts = np.array(list(centroids.values()))
    center = pts.mean(axis=0)

    rng = np.random.default_rng(0)
    manifest = []
    idx = 0

    # Sample poses around AP (az~0) and LAT (az~90), with a realistic
    # spread (not a single fixed point, but not unboundedly wide either --
    # this models "C-arm roughly positioned for an AP or LAT shot, with
    # normal day-to-day positioning variation").
    base_azimuths = [0.0, 90.0]
    N_PER_VIEW = 150

    for base_az in base_azimuths:
        for _ in range(N_PER_VIEW):
            az = base_az + rng.normal(0, 8.0)      # +/- a few deg around AP/LAT
            el = rng.normal(0, 8.0)
            trans_jitter = rng.normal(0, 15.0, size=3)  # mm, small positioning jitter
            jittered_center = center + trans_jitter

            src, pax, u, v = DRRRenderer.camera_basis_from_azimuth_elevation(
                jittered_center, SID_MM, az, el)
            drr = renderer.render(src, pax, u, v, output_size=OUT_SIZE,
                                   pixel_spacing_mm=PIX_MM, n_steps=N_STEPS)
            uv = renderer.project_points(pts, src, pax, u, v, OUT_SIZE, PIX_MM, SID_MM)

            fname = f'drr_{idx:04d}.png'
            cv2.imwrite(str(BANK_DIR / fname), (np.clip(drr, 0, 1) * 255).astype('uint8'))

            manifest.append(dict(
                file=fname,
                base_view='AP' if base_az == 0.0 else 'LAT',
                azimuth_deg=float(az), elevation_deg=float(el),
                center_xyz=jittered_center.tolist(),
                src_xyz=src.tolist(), pax_xyz=pax.tolist(),
                u_xyz=u.tolist(), v_xyz=v.tolist(),
                landmarks_px={n: uv[i].tolist() for i, n in enumerate(names)},
            ))
            idx += 1
            if idx % 50 == 0:
                print(f'  rendered {idx}/{len(base_azimuths)*N_PER_VIEW}')

    (BANK_DIR / 'manifest.json').write_text(json.dumps(dict(
        output_size=OUT_SIZE, pixel_spacing_mm=PIX_MM, sid_mm=SID_MM,
        landmark_names=names, entries=manifest,
    ), indent=1))
    print(f'Done. {idx} DRRs written to {BANK_DIR}')


if __name__ == '__main__':
    main()
