#!/usr/bin/env python3
"""
build_buu_manifest.py
=====================
Convert BUU-LSPINE dataset into our landmark manifest CSV format.

CSV format (per image, no header):
    x1, y1, x2, y2, 0      ← one vertebral endplate line (left + right endpoints)

Each vertebra has 2 consecutive rows (superior + inferior endplate).
Vertebra centroid = mean of all 4 endpoints.

AP (10 rows): rows 0-1=L1, 2-3=L2, 4-5=L3, 6-7=L4, 8-9=L5
LA (11 rows): rows 0-1=L1, 2-3=L2, 4-5=L3, 6-7=L4, 8-9=L5, row 10=S1 (skipped)

Output columns match landmark_manifest_combined.csv:
    patient, set_name, view, img_path, lm_path, labels, coords_px

Usage:
    python scripts/build_buu_manifest.py [--data_dir data/buu_lspine/sample400]
"""

import argparse, csv
from pathlib import Path

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / 'data' / 'buu_lspine' / 'sample400' / 'BUU-LSPINE_400'
OUT_CSV   = ROOT / 'data' / 'buu_lspine_manifest_400.csv'

# Row-pair index → vertebra label (same for AP and LA)
VERT_MAP = {0: 'L1', 1: 'L2', 2: 'L3', 3: 'L4', 4: 'L5'}


def parse_buu_csv(csv_path: Path) -> dict:
    """
    Parse BUU endplate CSV → {vertebra_name: [cx_px, cy_px]}.
    Returns centroids for all fully-annotated vertebrae.
    """
    lines = csv_path.read_text().strip().split('\n')
    rows  = []
    for line in lines:
        vals = [v.strip() for v in line.split(',')]
        try:
            rows.append([float(v) for v in vals[:4]])
        except ValueError:
            continue

    coords = {}
    n_verts = len(rows) // 2   # ignore last row if odd (e.g. S1 sup in LA)
    for vi in range(min(n_verts, 5)):   # max 5 vertebrae (L1-L5)
        vname = VERT_MAP.get(vi)
        if vname is None:
            continue
        sup = rows[vi * 2]      # [x1, y1, x2, y2] superior endplate
        inf = rows[vi * 2 + 1]  # inferior endplate
        cx  = (sup[0] + sup[2] + inf[0] + inf[2]) / 4.0
        cy  = (sup[1] + sup[3] + inf[1] + inf[3]) / 4.0
        coords[vname] = [cx, cy]

    return coords


def build_manifest(data_dir: Path, out_csv: Path):
    ap_dir = data_dir / 'AP'
    la_dir = data_dir / 'LA'

    if not ap_dir.exists() or not la_dir.exists():
        raise FileNotFoundError(f'Expected AP/ and LA/ under {data_dir}')

    ap_csvs = sorted(ap_dir.glob('*.csv'))
    la_csvs = sorted(la_dir.glob('*.csv'))
    print(f'AP CSVs: {len(ap_csvs)}   LA CSVs: {len(la_csvs)}')

    rows_out = []
    skipped  = 0

    for csv_path in ap_csvs + la_csvs:
        img_path = csv_path.with_suffix('.jpg')
        if not img_path.exists():
            skipped += 1
            continue

        stem  = csv_path.stem               # e.g. "0005-F-025Y0"
        parts = stem.split('-')
        pid   = parts[0]                    # "0005"
        view  = 'XRAY-AP' if stem.endswith('0') else 'XRAY-LAT'

        coords = parse_buu_csv(csv_path)
        if len(coords) < 2:
            skipped += 1
            continue

        labels     = list(coords.keys())
        coords_str = ';'.join(f'{v[0]:.2f},{v[1]:.2f}' for v in coords.values())

        rows_out.append({
            'patient':   f'BUU_{pid}',
            'set_name':  stem,
            'view':      view,
            'img_path':  str(img_path.relative_to(ROOT)),
            'lm_path':   str(csv_path.relative_to(ROOT)),
            'labels':    '|'.join(labels),
            'coords_px': coords_str,
        })

    print(f'Valid samples: {len(rows_out)}   Skipped: {skipped}')

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'patient','set_name','view','img_path','lm_path','labels','coords_px'])
        w.writeheader()
        w.writerows(rows_out)

    print(f'Manifest → {out_csv}')
    return rows_out


def verify_sample(data_dir: Path, n: int = 3):
    """Draw centroids on a few images to visually verify correctness."""
    import cv2, numpy as np
    out_dir = ROOT / 'data' / 'buu_lspine' / 'verify'
    out_dir.mkdir(parents=True, exist_ok=True)

    COLORS = {'L1':(255,80,80),'L2':(255,180,0),'L3':(0,220,80),
              'L4':(0,160,255),'L5':(180,0,255)}

    for view_dir, suffix in [(data_dir/'AP','0'), (data_dir/'LA','1')]:
        for csv_path in sorted(view_dir.glob(f'*{suffix}.csv'))[:n]:
            img_path = csv_path.with_suffix('.jpg')
            if not img_path.exists():
                continue
            img    = cv2.imread(str(img_path))
            coords = parse_buu_csv(csv_path)
            for vname, (cx, cy) in coords.items():
                col = COLORS.get(vname, (255,255,255))
                cv2.circle(img, (int(cx), int(cy)), 25, col, -1)
                cv2.putText(img, vname, (int(cx)+30, int(cy)+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, col, 4)
            # Downscale for saving
            scale = 512 / max(img.shape[:2])
            small = cv2.resize(img, None, fx=scale, fy=scale)
            out   = out_dir / f'{csv_path.stem}_verify.jpg'
            cv2.imwrite(str(out), small)
            print(f'  Saved → {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=str(DATA_DIR))
    p.add_argument('--out',      default=str(OUT_CSV))
    p.add_argument('--verify',   action='store_true',
                   help='Save overlay images to verify centroid accuracy')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_csv  = Path(args.out)

    rows = build_manifest(data_dir, out_csv)

    if args.verify:
        print('\nGenerating verification overlays ...')
        verify_sample(data_dir, n=3)

    # Quick stats
    n_ap  = sum(1 for r in rows if r['view'] == 'XRAY-AP')
    n_lat = sum(1 for r in rows if r['view'] == 'XRAY-LAT')
    from collections import Counter
    lbl_counts = Counter(
        lbl for r in rows for lbl in r['labels'].split('|'))
    print(f'\nSummary:  AP={n_ap}  LAT={n_lat}  total={len(rows)}')
    print(f'Label counts: {dict(lbl_counts)}')
    print(f'\nNext: python scripts/merge_manifests.py')


if __name__ == '__main__':
    main()
