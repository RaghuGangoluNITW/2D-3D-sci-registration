"""
test_retrieval_synthetic.py
=============================
Step 1 of the POC: prove the retrieval/filter mechanism itself works
correctly in the BEST case -- matching a held-out synthetic DRR (treated
as if it were the unknown "target" photo) against the DRR bank.

This isolates the retrieval logic from the real-X-ray domain-gap problem:
if this fails, the matching logic itself is broken. If this succeeds
(which it should, since DRR-vs-DRR has no domain gap), the mechanism is
sound and any later failure on a REAL X-ray is specifically a domain-gap
problem, not a logic problem.
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
OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')


def ncc(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(a * b))


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
    pts = np.array([centroids[n] for n in names])

    rng = np.random.default_rng(123)  # different seed from the bank build

    # Hold out a NEW pose (not in the bank) as the "unknown target"
    center = pts.mean(axis=0)
    true_az = 4.0   # a few deg off true AP, within the bank's AP spread
    true_el = -3.0
    true_center = center + rng.normal(0, 10.0, size=3)
    src, pax, u, v = DRRRenderer.camera_basis_from_azimuth_elevation(
        true_center, sid_mm, true_az, true_el)
    target = renderer.render(src, pax, u, v, output_size=out_size,
                              pixel_spacing_mm=pix_mm, n_steps=180)
    true_uv = renderer.project_points(pts, src, pax, u, v, out_size, pix_mm, sid_mm)

    # Retrieval: score every bank DRR against the target by NCC
    scores = []
    for e in entries:
        drr_img = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        s = ncc(drr_img, target)
        scores.append(s)
    scores = np.array(scores)
    order = np.argsort(-scores)  # best (highest NCC) first

    print('Top-5 matches by NCC:')
    for rank, i in enumerate(order[:5]):
        e = entries[i]
        print(f'  #{rank+1}: {e["file"]}  NCC={scores[i]:.4f}  '
              f'view={e["base_view"]}  az={e["azimuth_deg"]:.1f} el={e["elevation_deg"]:.1f}')
    print(f'\nTrue target pose: az={true_az:.1f} el={true_el:.1f} (not in bank)')

    best = entries[order[0]]
    best_uv = {n: np.array(v) for n, v in best['landmarks_px'].items()}

    # Compare retrieved landmark px positions vs the TRUE target's px positions
    px_errs = []
    for i, n in enumerate(names):
        err_px = np.linalg.norm(best_uv[n] - true_uv[i])
        err_mm = err_px * pix_mm
        px_errs.append(err_mm)
        print(f'  {n}: retrieved=({best_uv[n][0]:.1f},{best_uv[n][1]:.1f}) '
              f'true=({true_uv[i][0]:.1f},{true_uv[i][1]:.1f})  error={err_mm:.2f}mm')
    print(f'\nMean 2D landmark error (retrieved vs true, best-match DRR): '
          f'{np.mean(px_errs):.2f}mm')

    # Save visual: target vs best match side by side
    best_img = cv2.imread(str(BANK_DIR / best['file']), cv2.IMREAD_GRAYSCALE)
    tgt_u8 = (np.clip(target, 0, 1) * 255).astype('uint8')
    side = np.hstack([tgt_u8, best_img])
    side_bgr = cv2.cvtColor(side, cv2.COLOR_GRAY2BGR)
    cv2.putText(side_bgr, 'TARGET (held-out)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(side_bgr, 'BEST MATCH (bank)', (out_size + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    OUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT_DIR / 'retrieval_synthetic_target_vs_match.png'), side_bgr)
    print(f'\nSaved comparison -> {OUT_DIR / "retrieval_synthetic_target_vs_match.png"}')


if __name__ == '__main__':
    main()
