#!/usr/bin/env python3
"""
eval_ssm_regressor.py
=====================
Evaluate the new view-specific SSMRegressorDetector (AP + LAT) on each LOPO fold.

For every patient:
  - Load fold-specific AP checkpoint  → predict on AP images
  - Load fold-specific LAT checkpoint → predict on LAT images
  - Compute per-vertebra Euclidean error in mm
  - Report per-vertebra mean ± std, and overall spine error

Falls back to ssm_regressor_ln/ (combined) if AP/LAT checkpoint not found.

Usage:
    python scripts/eval_ssm_regressor.py
    python scripts/eval_ssm_regressor.py --pixel_mm 0.2 --device cuda
"""
import sys, argparse, csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from landmark_detector import SSMRegressorDetector
from landmark_dataset import CLASS_NAMES, _load_xray, IMG_SIZE

MANIFEST  = ROOT / 'data' / 'landmark_manifest_combined.csv'
PIXEL_MM  = 0.2
AP_DIR    = ROOT / 'checkpoints' / 'ssm_regressor_ap'
LAT_DIR   = ROOT / 'checkpoints' / 'ssm_regressor_lat'
LN_DIR    = ROOT / 'checkpoints' / 'ssm_regressor_ln'
SSM_PATH  = ROOT / 'checkpoints' / 'spine_ssm_buu.pkl'


def load_manifest_real(csv_path):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row['img_path'].endswith('.nrrd'):
                rows.append(row)
    return rows


def parse_coords(row):
    labels = row['labels'].split('|')
    coords = row['coords_px'].split(';')
    out = {}
    for lbl, c in zip(labels, coords):
        u, v = map(float, c.strip().split(','))
        out[lbl.strip()] = np.array([u, v])
    return out


def get_ckpt(patient, view, ckpt_dir_ap, ckpt_dir_lat, ckpt_dir_ln):
    """Return (ckpt_path, source_label) for a patient+view combo."""
    if view == 'XRAY-AP':
        p = ckpt_dir_ap / f'fold_{patient}.pth'
        if p.exists():
            return p, 'AP'
    elif view == 'XRAY-LAT':
        p = ckpt_dir_lat / f'fold_{patient}.pth'
        if p.exists():
            return p, 'LAT'
    # fallback to combined LN
    p = ckpt_dir_ln / f'fold_{patient}.pth'
    if p.exists():
        return p, 'LN(fallback)'
    return None, 'MISSING'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pixel_mm', type=float, default=PIXEL_MM)
    p.add_argument('--device',   default='cuda')
    args = p.parse_args()

    rows = load_manifest_real(MANIFEST)

    # Group by patient
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r['patient']].append(r)

    patients = sorted(by_patient.keys())
    print(f'\nEvaluating {len(patients)} LOPO folds  '
          f'(pixel_mm={args.pixel_mm}, SSMRegressorDetector AP+LAT)\n')

    all_errs   = []
    vert_errs  = defaultdict(list)
    pat_errs   = {}
    detectors  = {}   # cache (patient, view) → detector

    bar = '=' * 62
    print(bar)

    for patient in patients:
        p_rows   = by_patient[patient]
        p_errs   = []
        p_detail = []

        for row in p_rows:
            view     = row['view']      # 'XRAY-AP' or 'XRAY-LAT'
            set_name = row['set_name']
            gt_coords = parse_coords(row)
            H_orig, W_orig = None, None

            # Load image to get original size
            img = _load_xray(row['img_path'])
            H_orig, W_orig = img.shape[:2]

            # Get detector (cached per patient+view)
            key = (patient, view)
            if key not in detectors:
                ckpt_path, src = get_ckpt(patient, view,
                                          AP_DIR, LAT_DIR, LN_DIR)
                if ckpt_path is None:
                    detectors[key] = (None, 'MISSING')
                else:
                    view_short = 'AP' if 'AP' in view else 'LAT'
                    det = SSMRegressorDetector(
                        checkpoint=str(ckpt_path),
                        ssm_path=str(SSM_PATH),
                        view=view_short,
                        device=args.device,
                    )
                    detectors[key] = (det, src)

            det, src = detectors[key]
            if det is None:
                p_detail.append(f'  {set_name:12s}  [NO CHECKPOINT]')
                continue

            # Predict — returns coords already in original image space
            pred_coords = det.predict(img)

            # Compute errors
            row_errs = []
            row_str  = f'  {set_name:12s}  [{src}]  '
            for lbl, gt in gt_coords.items():
                if lbl in pred_coords:
                    err_px  = float(np.linalg.norm(pred_coords[lbl] - gt))
                    err_mm  = err_px * args.pixel_mm
                    row_str += f'{lbl}={err_mm:.1f}mm  '
                    p_errs.append(err_mm)
                    all_errs.append(err_mm)
                    vert_errs[lbl].append(err_mm)
                    row_errs.append(err_mm)
            p_detail.append(row_str)

        mean_p = float(np.mean(p_errs)) if p_errs else float('nan')
        pat_errs[patient] = (mean_p, len(p_errs))

        print(f'\n┌─ {patient}  ({len(p_rows)} images)')
        for line in p_detail:
            print('│ ' + line)
        print(f'└─ Mean spine error: {mean_p:.1f} mm  over {len(p_errs)} landmarks')

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f'\n{bar}')
    print('  GLOBAL SSMRegressor LOPO EVALUATION (AP+LAT view-specific)')
    print(bar)
    print(f'  {"Patient":<22}  {"Mean mm":>8}  {"N lm":>6}')
    print(f'  {"-"*22}  {"-"*8}  {"-"*6}')
    for pat in patients:
        mean_e, n = pat_errs[pat]
        bar_s = '█' * max(0, int(50 - mean_e / 3))
        print(f'  {pat:<22}  {mean_e:8.1f}  {n:6d}  {bar_s}')

    overall = float(np.mean(all_errs)) if all_errs else float('nan')
    median  = float(np.median(all_errs)) if all_errs else float('nan')
    print(f'\n  Overall mean: {overall:.1f} mm  (median={median:.1f})  '
          f'N={len(all_errs)} landmarks')

    print(f'\n  Per-vertebra global:')
    for v in CLASS_NAMES:
        errs = vert_errs[v]
        if not errs:
            continue
        pct10 = 100 * sum(1 for e in errs if e < 10) / len(errs)
        pct5  = 100 * sum(1 for e in errs if e < 5)  / len(errs)
        print(f'  {v}:  {np.mean(errs):5.1f} ± {np.std(errs):.1f} mm'
              f'    <10mm:{pct10:.0f}%  <5mm:{pct5:.0f}%  (N={len(errs)})')
    print(bar)


if __name__ == '__main__':
    main()
