"""
test_retrieval_metrics_real.py
=================================
Fix attempt for the domain-gap problem found in run_full_pipeline_real.py:
raw pixel-intensity NCC compares BRIGHTNESS patterns, which have nothing
in common between a clean DRR and a real X-ray (different absolute
intensity statistics, noise, soft tissue). What SHOULD transfer across
the domain gap is EDGE STRUCTURE -- where the bone boundaries are and
which direction they point -- regardless of absolute brightness.

Tests three metrics built for exactly this cross-modal problem (already
implemented in the main project's src/similarity.py, used by its proven
1.4mm manual pipeline for DRR-vs-real-Xray comparison):
  1. NGF  (Normalized Gradient Field)      -- compares gradient DIRECTION
  2. GO   (Gradient Orientation)            -- compares gradient DIRECTION
  3. Gradient-magnitude NCC                 -- compares edge STRENGTH maps
all evaluated on the SAME real X-ray target that broke raw NCC, to see
whether the retrieval step can be fixed by changing the metric alone,
with everything else (DRR bank, image preprocessing) held constant.
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'mslevel_reg' / 'src'))
from similarity import (
    gradient_orientation_similarity, ngf_cost, gradient_ncc,
)
from run_full_pipeline_real import preprocess_real_xray, ncc, REAL_XRAY_PATH

BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\drr_bank')
OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')


def main():
    manifest = json.loads((BANK_DIR / 'manifest.json').read_text())
    entries = [e for e in manifest['entries'] if e['base_view'] == 'AP']
    out_size = manifest['output_size']

    target = preprocess_real_xray(REAL_XRAY_PATH, out_size)

    results = {'ncc': [], 'go': [], 'ngf_inv': [], 'grad_ncc_inv': []}
    for e in entries:
        drr_img = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        results['ncc'].append(ncc(drr_img, target))
        results['go'].append(gradient_orientation_similarity(drr_img, target, sigma=2.0))
        results['ngf_inv'].append(1.0 - ngf_cost(drr_img, target, sigma=1.5, eps=0.01))
        results['grad_ncc_inv'].append(1.0 - gradient_ncc(drr_img, target, sigma=1.5) * 2)  # back to [-1,1]-ish

    print(f'{"metric":<14}{"best_score":>12}{"best_file":>14}{"az":>8}{"el":>8}')
    print('-' * 60)
    best_summaries = {}
    for metric, vals in results.items():
        vals = np.array(vals)
        best_i = int(np.argmax(vals))
        e = entries[best_i]
        print(f'{metric:<14}{vals[best_i]:>12.4f}{e["file"]:>14}{e["azimuth_deg"]:>8.1f}{e["elevation_deg"]:>8.1f}')
        best_summaries[metric] = (best_i, vals[best_i])

    # Save a montage: target + best match per metric
    tgt_u8 = (np.clip(target, 0, 1) * 255).astype('uint8')
    imgs = [tgt_u8]
    labels = ['TARGET']
    for metric, (idx, score) in best_summaries.items():
        img = cv2.imread(str(BANK_DIR / entries[idx]['file']), cv2.IMREAD_GRAYSCALE)
        imgs.append(img)
        labels.append(f'{metric}={score:.3f}')
    montage = np.hstack(imgs)
    montage_bgr = cv2.cvtColor(montage, cv2.COLOR_GRAY2BGR)
    for i, lbl in enumerate(labels):
        cv2.putText(montage_bgr, lbl, (i * out_size + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    OUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT_DIR / 'metric_comparison_real.png'), montage_bgr)
    print(f'\nSaved -> {OUT_DIR / "metric_comparison_real.png"}')

    # Also rank ALL candidates by each metric and check how many of the
    # top-10 are near az~0 (true AP) -- a proxy for "is this metric
    # actually finding the right REGION of pose-space, not just one lucky
    # match". Real pose for this X-ray is unknown, but it IS a standard,
    # well-positioned AP film, so true az should be close to 0.
    print('\nTop-10 azimuth values per metric (closer to 0 = more plausible AP):')
    for metric, vals in results.items():
        vals = np.array(vals)
        order = np.argsort(-vals)[:10]
        azs = [entries[i]['azimuth_deg'] for i in order]
        print(f'  {metric:<14}: {[round(a,1) for a in azs]}  (mean |az|={np.mean(np.abs(azs)):.1f})')


if __name__ == '__main__':
    main()
