#!/usr/bin/env python3
"""
merge_manifests.py
==================
Merge BUU-LSPINE manifest(s) with our existing real patient manifest,
producing a combined CSV for training the SSM regressor and heatmap CNN.

Usage:
    python scripts/merge_manifests.py

Output:
    data/landmark_manifest_buu_combined.csv   ← real (67) + BUU samples
"""

import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent

MANIFESTS = {
    'real':     ROOT / 'data' / 'landmark_manifest_real.csv',   # or combined
    'buu_400':  ROOT / 'data' / 'buu_lspine_manifest_400.csv',
    'buu_full': ROOT / 'data' / 'buu_lspine_manifest_full.csv',
}

# Original real manifest (NRRDs only, no synthetics)
REAL_MANIFEST    = ROOT / 'data' / 'landmark_manifest_combined.csv'
OUTPUT_MANIFEST  = ROOT / 'data' / 'landmark_manifest_buu_combined.csv'

FIELDNAMES = ['patient', 'set_name', 'view', 'img_path',
              'lm_path', 'labels', 'coords_px']


def load_real_rows(csv_path: Path) -> list:
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r['img_path'].endswith('.nrrd'):  # real only, no synthetics
                rows.append(r)
    return rows


def load_buu_rows(csv_path: Path) -> list:
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main():
    # Load real rows
    real_rows = load_real_rows(REAL_MANIFEST)
    print(f'Real patients: {len(real_rows)} images')

    # Load BUU rows — prefer full if available, else sample
    buu_rows = load_buu_rows(MANIFESTS['buu_full'])
    if not buu_rows:
        buu_rows = load_buu_rows(MANIFESTS['buu_400'])
    print(f'BUU-LSPINE:    {len(buu_rows)} images')

    if not buu_rows:
        print('\n⚠  No BUU-LSPINE manifest found.')
        print('   Run: python scripts/download_buu_lspine.py  first')
        return

    combined = real_rows + buu_rows
    print(f'Combined:      {len(combined)} images')

    # Stats
    n_ap  = sum(1 for r in combined if 'AP' in r.get('view','').upper())
    n_lat = sum(1 for r in combined if 'LAT' in r.get('view','').upper() or 'LA' in r.get('view','').upper())
    n_buu = sum(1 for r in combined if r['patient'].startswith('BUU_'))
    print(f'  AP: {n_ap}  LAT/LA: {n_lat}  BUU patients: {n_buu//2}')

    with open(OUTPUT_MANIFEST, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(combined)

    print(f'\n✅ Merged manifest → {OUTPUT_MANIFEST}')
    print(f'\nRetrain SSM regressor with:')
    print(f'  python scripts/train_ssm_regressor.py \\')
    print(f'    --manifest data/landmark_manifest_buu_combined.csv \\')
    print(f'    --epochs 100 --batch 16 --lr 3e-4 --device cuda')
    print(f'\nRebuild SSM from combined data:')
    print(f'  python src/spine_ssm.py --manifest data/landmark_manifest_buu_combined.csv')


if __name__ == '__main__':
    main()
