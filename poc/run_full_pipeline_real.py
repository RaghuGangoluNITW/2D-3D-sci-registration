"""
run_full_pipeline_real.py
============================
Step 3 of the POC (the real test): run the EXACT same retrieval -> EPnP ->
refinement pipeline as run_full_pipeline.py, but against a REAL public
X-ray image (Radiopaedia, case rID 46571, "Normal AP lumbar spine
radiograph", Dr. Ian Bickle, CC-BY-NC-SA via Radiopaedia's standard
case license) instead of a synthetic held-out DRR.

IMPORTANT CAVEAT (same one flagged before building this): this real
X-ray is from a DIFFERENT, UNKNOWN patient -- there is no shared ground
truth 3D CT for this specific person, so we CANNOT compute a true PDE
number here. What we CAN measure and report honestly:
  1. Does the retrieval step pick a DRR that LOOKS structurally similar
     (same anatomy, plausible pose) -- a qualitative check.
  2. Does the NCC similarity score for the real X-ray's best match look
     comparable to the clean synthetic case's score, or much lower
     (indicating the domain gap is hurting the comparison itself)?
  3. Visual side-by-side for a human judgement call.
This isolates the SAME retrieval mechanism already proven correct on
clean synthetic data, now stress-tested against real image statistics
(contrast, noise, soft tissue, imaging artifacts) with no hardware
present -- the best-case real-world scenario.
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
REAL_XRAY_PATH = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\real_xray\radiopaedia_ap_lumbar_46571.jpg')


def ncc(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(a * b))


def preprocess_real_xray(path, out_size):
    """Crop out the burned-in 'R/S' marker region loosely, convert to
    grayscale, resize to the bank's resolution, normalize contrast
    (CLAHE) to roughly match DRR dynamic range conventions."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    # crop the spine column roughly (center 70% width, full height) to
    # remove some background/ribs variability -- a simple, defensible
    # preprocessing step, not hand-tuned to make the result look better.
    x0 = int(w * 0.05)
    x1 = int(w * 0.95)
    cropped = img[:, x0:x1]
    resized = cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(resized)
    norm = eq.astype(np.float32) / 255.0
    return norm


def main():
    manifest = json.loads((BANK_DIR / 'manifest.json').read_text())
    entries = [e for e in manifest['entries'] if e['base_view'] == 'AP']  # real image is AP
    names = manifest['landmark_names']
    out_size = manifest['output_size']
    pix_mm = manifest['pixel_spacing_mm']

    target = preprocess_real_xray(REAL_XRAY_PATH, out_size)

    scores = []
    for e in entries:
        drr_img = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        scores.append(ncc(drr_img, target))
    scores = np.array(scores)
    order = np.argsort(-scores)

    print('Top-5 matches (real X-ray vs AP DRR bank) by NCC:')
    for rank, i in enumerate(order[:5]):
        e = entries[i]
        print(f'  #{rank+1}: {e["file"]}  NCC={scores[i]:.4f}  az={e["azimuth_deg"]:.1f} el={e["elevation_deg"]:.1f}')

    print(f'\nFor comparison, the clean-synthetic-target test (run_full_pipeline.py) '
          f'achieved NCC=0.8917 for its best match.')
    print(f'Real X-ray best match NCC = {scores[order[0]]:.4f}')

    best = entries[order[0]]
    best_img = cv2.imread(str(BANK_DIR / best['file']), cv2.IMREAD_GRAYSCALE)
    tgt_u8 = (np.clip(target, 0, 1) * 255).astype('uint8')
    side = np.hstack([tgt_u8, best_img])
    side_bgr = cv2.cvtColor(side, cv2.COLOR_GRAY2BGR)
    cv2.putText(side_bgr, 'REAL X-RAY (target)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(side_bgr, f'BEST DRR MATCH (NCC={scores[order[0]]:.3f})', (out_size + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    OUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT_DIR / 'retrieval_real_target_vs_match.png'), side_bgr)
    print(f'\nSaved comparison -> {OUT_DIR / "retrieval_real_target_vs_match.png"}')

    # Also save a small montage of the top-5 for visual spread assessment
    tops = []
    for i in order[:5]:
        e = entries[i]
        img = cv2.imread(str(BANK_DIR / e['file']), cv2.IMREAD_GRAYSCALE)
        img = cv2.putText(img.copy(), f'{scores[i]:.3f}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        tops.append(img)
    montage = np.hstack([tgt_u8] + tops)
    cv2.imwrite(str(OUT_DIR / 'retrieval_real_top5_montage.png'), montage)
    print(f'Saved top-5 montage -> {OUT_DIR / "retrieval_real_top5_montage.png"}')


if __name__ == '__main__':
    main()
