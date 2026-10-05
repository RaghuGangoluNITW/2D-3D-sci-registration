"""
sanity_check_render.py
Quick visual check: render one DRR at an AP-like pose and overlay the
projected lumbar centroids, to confirm the renderer + projection math
are correct before building the full DRR bank.
"""
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from load_verse_ct import load_verse_case
from drr_renderer import CTVolume, DRRRenderer

CASE_DIR = r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\ct_data\dataset-verse19validation'
OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')
OUT_DIR.mkdir(exist_ok=True)

case = load_verse_case(CASE_DIR)
ct = CTVolume(case['volume_zyx'], case['spacing_xyz'], case['origin_xyz'], hu_threshold=180.0)
renderer = DRRRenderer(ct)

centroids = case['centroids_world']
center = np.mean(list(centroids.values()), axis=0)
print('spine centroid (world mm):', center.round(1))

SID_MM = 1000.0
OUT_SIZE = 384
PIX_MM = 1.0

for az, tag in [(0.0, 'AP'), (90.0, 'LAT')]:
    src, pax, u, v = DRRRenderer.camera_basis_from_azimuth_elevation(center, SID_MM, az)
    drr = renderer.render(src, pax, u, v, output_size=OUT_SIZE, pixel_spacing_mm=PIX_MM, n_steps=250)

    pts = np.array(list(centroids.values()))
    names = list(centroids.keys())
    uv = renderer.project_points(pts, src, pax, u, v, OUT_SIZE, PIX_MM, SID_MM)

    img = (np.clip(drr, 0, 1) * 255).astype('uint8')
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (px, py), name in zip(uv, names):
        if 0 <= px < OUT_SIZE and 0 <= py < OUT_SIZE:
            cv2.drawMarker(img_bgr, (int(px), int(py)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(img_bgr, name, (int(px) + 8, int(py)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out_path = OUT_DIR / f'sanity_{tag}.png'
    cv2.imwrite(str(out_path), img_bgr)
    print(f'{tag}: nonzero_frac={np.mean(drr > 0.01):.3f}  saved {out_path}')
