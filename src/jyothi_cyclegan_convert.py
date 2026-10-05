"""
jyothi_cyclegan_convert.py
==========================
Run CycleGAN G_AB (X-ray → DRR style) on JYOTHI's raw C-arm DICOM frames
and save the converted images as PNGs alongside a manifest.

The converted images can then be fed to the heatmap landmark detector in
place of the raw frames, reducing the synthetic-DRR/real-X-ray domain gap.

Usage
-----
    cd mslevel_reg
    python src/jyothi_cyclegan_convert.py

Outputs
-------
    POC/jyothi_data/converted/
        frame_<N>_raw.png       -- CLAHE-preprocessed raw frame (for reference)
        frame_<N>_converted.png -- CycleGAN G_AB output
    POC/jyothi_data/converted/jyothi_converted_manifest.csv
"""

import sys
import csv
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch

ROOT = Path(__file__).parent.parent           # mslevel_reg/
sys.path.insert(0, str(ROOT / 'src'))

from cyclegan import CycleGAN

# ── Paths ──────────────────────────────────────────────────────────────────
CKPT_PATH  = ROOT / 'checkpoints' / 'cyclegan' / 'best.pth'
CARM_DIR   = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\jyothi\carm_extracted\jyothi carm\20260707\0')
OUT_DIR    = Path(r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\jyothi_data\converted')
MANIFEST   = OUT_DIR / 'jyothi_converted_manifest.csv'

SAVE_SIZE  = 512   # matches landmark_dataset.py IMG_SIZE

# Frames of particular interest: 9=AP, 14=LAT (used in earlier tests)
# We convert ALL 100 frames so the full sequence is available.
# ──────────────────────────────────────────────────────────────────────────


def load_dicom_frame(path: Path) -> np.ndarray:
    """Load a raw DICOM frame (no extension) and return float32 [0,1]."""
    ds  = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def clahe_normalize(img: np.ndarray) -> np.ndarray:
    u8  = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    cl  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
    return cl.astype(np.float32) / 255.0


@torch.no_grad()
def convert_frame(model: CycleGAN, img: np.ndarray, device: str) -> np.ndarray:
    """CLAHE → resize to SAVE_SIZE → G_AB → float32 [0,1]."""
    img_cl = clahe_normalize(img)
    img_rs = cv2.resize(img_cl, (SAVE_SIZE, SAVE_SIZE),
                        interpolation=cv2.INTER_LINEAR)
    t   = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to(device)
    out = model.xray_to_drr(t)
    return out[0, 0].cpu().numpy().clip(0, 1).astype(np.float32)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Load CycleGAN
    print(f'Loading CycleGAN from: {CKPT_PATH}')
    model = CycleGAN.load(str(CKPT_PATH), device=device)
    model.eval()
    print(f'  ngf={model.ngf}, n_res={model.G_AB.n_res}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Enumerate raw DICOM frames (files named 1,2,...,100 — no extension)
    frame_paths = sorted(
        [p for p in CARM_DIR.iterdir() if p.is_file() and p.stem.isdigit()],
        key=lambda p: int(p.stem)
    )
    print(f'Found {len(frame_paths)} C-arm frames in {CARM_DIR}')

    manifest_rows = []
    n_ok = n_err = 0

    for fp in frame_paths:
        fnum = int(fp.stem)
        try:
            raw = load_dicom_frame(fp)
        except Exception as e:
            print(f'  WARN frame {fnum}: {e}')
            n_err += 1
            continue

        # Save CLAHE-preprocessed raw (for reference / side-by-side)
        raw_cl   = clahe_normalize(raw)
        raw_rs   = cv2.resize(raw_cl, (SAVE_SIZE, SAVE_SIZE),
                              interpolation=cv2.INTER_LINEAR)
        raw_path = OUT_DIR / f'frame_{fnum:03d}_raw.png'
        cv2.imwrite(str(raw_path), (raw_rs * 255).astype(np.uint8))

        # Convert through CycleGAN
        converted  = convert_frame(model, raw, device)
        conv_path  = OUT_DIR / f'frame_{fnum:03d}_converted.png'
        cv2.imwrite(str(conv_path), (converted * 255).astype(np.uint8))

        manifest_rows.append({
            'frame_num':    fnum,
            'raw_path':     str(raw_path),
            'converted_path': str(conv_path),
        })
        n_ok += 1
        if fnum % 10 == 0:
            print(f'  Processed {fnum}/{len(frame_paths)}...')

    # Write manifest
    with open(MANIFEST, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame_num', 'raw_path', 'converted_path'])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f'\nDone. {n_ok} frames converted, {n_err} errors.')
    print(f'Output: {OUT_DIR}')
    print(f'Manifest: {MANIFEST}')
    print()
    print('Key frames saved:')
    for fnum in [9, 14]:
        rp = OUT_DIR / f'frame_{fnum:03d}_raw.png'
        cp = OUT_DIR / f'frame_{fnum:03d}_converted.png'
        print(f'  Frame {fnum}  raw={rp.exists()}  converted={cp.exists()}')


if __name__ == '__main__':
    main()
