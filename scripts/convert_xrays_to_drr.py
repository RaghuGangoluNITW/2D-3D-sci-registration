#!/usr/bin/env python3
"""
convert_xrays_to_drr.py
========================
Stage 2b of the automatic PDE <2mm pipeline.

Applies the trained CycleGAN G_AB (X-ray → DRR) to all real X-ray images
and saves the converted images alongside the originals.

The converted images are then used to augment the SSM regressor training
dataset — real X-rays that now look like DRRs, giving the regressor more
domain-matched training examples.

Output layout
-------------
data/converted_xrays/
    <PATIENT>/
        XRAY-AP/
            <SET>/
                <img>_converted.png    # X-ray converted to DRR style
        XRAY-LAT/
            ...
data/converted_xray_manifest.csv       # manifest in same format as
                                        # landmark_manifest_buu_combined.csv

Usage
-----
python scripts/convert_xrays_to_drr.py
python scripts/convert_xrays_to_drr.py --ckpt checkpoints/cyclegan/best.pth
python scripts/convert_xrays_to_drr.py --patients SARKHI Swaroop
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from cyclegan import CycleGAN

CKPT_DIR  = ROOT / 'checkpoints' / 'cyclegan'
OUT_ROOT  = ROOT / 'data' / 'converted_xrays'
MANIFEST  = ROOT / 'data' / 'converted_xray_manifest.csv'
SRC_MANIFEST = ROOT / 'data' / 'landmark_manifest_buu_combined.csv'

SAVE_SIZE = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_nrrd(path: str) -> np.ndarray:
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).squeeze().astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def _clahe(img: np.ndarray) -> np.ndarray:
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
    return cl.astype(np.float32) / 255.0


@torch.no_grad()
def convert_image(model: CycleGAN, img: np.ndarray,
                  device: str, size: int = SAVE_SIZE) -> np.ndarray:
    """
    Convert a single grayscale float32 [0,1] X-ray to DRR style.

    The image is resized to `size` for the network, then the output is
    returned at the same `size`. We keep the network-output size (512)
    rather than the original 1024 — the SSM regressor trains at 512 anyway.
    """
    # CLAHE + resize to network size
    img_cl = _clahe(img)
    img_rs = cv2.resize(img_cl, (size, size), interpolation=cv2.INTER_LINEAR)

    t = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    out = model.xray_to_drr(t)                                          # (1,1,H,W)
    out_np = out[0, 0].cpu().numpy()                                    # (H,W)
    return np.clip(out_np, 0, 1).astype(np.float32)


def _scale_coords(coords_str: str, orig_size: int, save_size: int) -> str:
    """Scale u,v coordinates from original image size to save_size."""
    scale  = save_size / orig_size
    parts  = coords_str.split(';')
    scaled = []
    for part in parts:
        u, v = map(float, part.strip().split(','))
        scaled.append(f'{u*scale:.2f},{v*scale:.2f}')
    return ';'.join(scaled)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=None,
                   help='CycleGAN checkpoint (default: best.pth, then latest.pth)')
    p.add_argument('--patients', nargs='+', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--size', type=int, default=SAVE_SIZE,
                   help=f'Output image size (default {SAVE_SIZE})')
    p.add_argument('--src_manifest', default=str(SRC_MANIFEST),
                   help='Source real-image manifest to convert')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load model ────────────────────────────────────────────────────────
    ckpt_path = args.ckpt
    if ckpt_path is None:
        for cand in ['best.pth', 'latest.pth']:
            p = CKPT_DIR / cand
            if p.exists():
                ckpt_path = str(p)
                break
    if ckpt_path is None:
        print('ERROR: No CycleGAN checkpoint found.')
        print(f'Run: python scripts/train_cyclegan.py')
        return

    print(f'Loading CycleGAN from: {ckpt_path}')
    model = CycleGAN.load(ckpt_path, device=args.device)
    model.eval()
    print(f'Model loaded (ngf={model.ngf})')

    # ── Load source manifest ──────────────────────────────────────────────
    src_rows = []
    with open(args.src_manifest, newline='') as f:
        for row in csv.DictReader(f):
            if row['img_path'].endswith('.nrrd'):
                src_rows.append(row)

    if args.patients:
        src_rows = [r for r in src_rows if r['patient'] in args.patients]

    print(f'Converting {len(src_rows)} real X-ray images...')

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    n_ok = 0
    n_err = 0

    for row in src_rows:
        patient   = row['patient']
        set_name  = row['set_name']
        view      = row['view']          # 'XRAY-AP' or 'XRAY-LAT'
        img_path  = ROOT / row['img_path']

        # ── Load & convert ────────────────────────────────────────────────
        try:
            img     = _load_nrrd(str(img_path))
            orig_h, orig_w = img.shape[:2]
            converted = convert_image(model, img, args.device, args.size)
        except Exception as e:
            print(f'  WARN: {patient}/{set_name} — {e}')
            n_err += 1
            continue

        # ── Save ──────────────────────────────────────────────────────────
        out_dir = OUT_ROOT / patient / view / set_name
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = img_path.stem
        out_path = out_dir / f'{stem}_converted.png'
        img8 = (converted * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out_path), img8)

        # ── Build manifest row ────────────────────────────────────────────
        # Scale coordinates from original image size to SAVE_SIZE
        orig_size = max(orig_h, orig_w)
        coords_scaled = _scale_coords(row['coords_px'], orig_size, args.size)

        manifest_rows.append({
            'patient':   f'CONV_{patient}',
            'set_name':  set_name,
            'view':      view,
            'img_path':  str(out_path.relative_to(ROOT)).replace('\\', '/'),
            'lm_path':   row['lm_path'],
            'labels':    row['labels'],
            'coords_px': coords_scaled,
        })
        n_ok += 1

    # ── Write manifest ────────────────────────────────────────────────────
    fieldnames = ['patient', 'set_name', 'view', 'img_path',
                  'lm_path', 'labels', 'coords_px']
    with open(MANIFEST, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)

    print(f'\n  Converted  : {n_ok}  images')
    print(f'  Errors     : {n_err}')
    print(f'  Output     : {OUT_ROOT}')
    print(f'  Manifest   : {MANIFEST}')


if __name__ == '__main__':
    main()
