"""
run_full_pipeline_real_masked.py
===================================
Follow-up check on run_full_pipeline_real.py's finding: the raw NCC score
collapsed (0.89 -> 0.16) when matching a real X-ray, apparently because
NCC is computed over the WHOLE image, and the real X-ray's background
(ribs, bowel gas, soft tissue) has no equivalent in the clean DRR.

Fair test: does restricting the similarity comparison to a central
spine-column mask (present in both the DRR and the real X-ray) recover
a meaningfully better score, or does the real-vs-synthetic texture gap
persist even within the spine region itself? This matters for the
project's conclusion -- if masking fixes it, the issue is "DRRs don't
model the body outline" (fixable via masking); if it doesn't, the issue
is deeper (DRRs don't model real bone TEXTURE/noise even where anatomy
overlaps).
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from run_full_pipeline_real import preprocess_real_xray, ncc, REAL_XRAY_PATH

BANK_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\drr_bank')
OUT_DIR = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results')


def main():
    manifest = json.loads((BANK_DIR / 'manifest.json').read_text())
    entries = [e for e in manifest['entries'] if e['base_view'] == 'AP']
    out_size = manifest['output_size']

    target = preprocess_real_xray(REAL_XRAY_PATH, out_size)

    # Central spine-column mask: a vertical band covering roughly the
    # middle third of the image width, full height -- present in both
    # the DRR (bone signal) and the real X-ray (spine + overlapping soft
    # tissue), rather than the full frame (ribs/background included).
    mask = np.zeros((out_size, out_size), dtype=bool)
    x0, x1 = int(out_size * 0.35), int(out_size * 0.65)
    mask[:, x0:x1] = True

    def ncc_masked(a, b, m):
        a = a[m].astype(np.float64)
        b = b[m].astype(np.float64)
        a = (a - a.mean()) / (a.std() + 1e-8)
        b = (b - b.mean()) / (b.std() + 1e-8)
        return float(np.mean(a * b))

    scores_full, scores_masked = [], []
    for e in entries:
        drr_img = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        scores_full.append(ncc(drr_img, target))
        scores_masked.append(ncc_masked(drr_img, target, mask))
    scores_full = np.array(scores_full)
    scores_masked = np.array(scores_masked)

    order_full = np.argsort(-scores_full)
    order_masked = np.argsort(-scores_masked)

    print('=== FULL-FRAME NCC (baseline from previous script) ===')
    print(f'  best: {entries[order_full[0]]["file"]}  NCC={scores_full[order_full[0]]:.4f}')
    print('=== SPINE-COLUMN-MASKED NCC ===')
    for rank, i in enumerate(order_masked[:5]):
        e = entries[i]
        print(f'  #{rank+1}: {e["file"]}  masked_NCC={scores_masked[i]:.4f}  '
              f'(full_NCC was {scores_full[i]:.4f})  az={e["azimuth_deg"]:.1f}')

    print(f'\nBest masked NCC = {scores_masked[order_masked[0]]:.4f}  '
          f'vs clean-synthetic reference 0.8917')

    best = entries[order_masked[0]]
    best_img = cv2.imread(str(BANK_DIR / best['file']), cv2.IMREAD_GRAYSCALE)
    tgt_u8 = (np.clip(target, 0, 1) * 255).astype('uint8')
    tgt_viz = cv2.cvtColor(tgt_u8, cv2.COLOR_GRAY2BGR)
    best_viz = cv2.cvtColor(best_img, cv2.COLOR_GRAY2BGR)
    for img in (tgt_viz, best_viz):
        cv2.rectangle(img, (x0, 0), (x1, out_size - 1), (0, 255, 255), 1)
    side = np.hstack([tgt_viz, best_viz])
    cv2.putText(side, 'REAL (mask shown)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(side, f'BEST masked (NCC={scores_masked[order_masked[0]]:.3f})',
                (out_size + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.imwrite(str(OUT_DIR / 'retrieval_real_masked_comparison.png'), side)
    print(f'Saved -> {OUT_DIR / "retrieval_real_masked_comparison.png"}')


if __name__ == '__main__':
    main()
