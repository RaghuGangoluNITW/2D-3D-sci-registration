#!/usr/bin/env python3
"""
extract_extra_xrays.py
=======================
Extracts real (unlabeled) X-ray / C-arm frames from the "full data" export
into a flat folder of PNG images, for use as extra CycleGAN training data
(Stage 2 domain adaptation). These images do NOT need manual landmark
annotation -- CycleGAN only needs a pool of realistic X-ray-style images
to learn the domain mapping, not paired labels.

Sources (all confirmed genuine intraoperative/diagnostic X-ray images,
verified by file signature / pixel content inspection):
  1. Ramulamma C arm images intra op/*.TIF   -- 85 frames, 1024x1024 16-bit,
     confirmed real Ziehm C-arm fluoroscopy (burned-in exposure overlay).
  2. jyothi carm/.../*                        -- 100 raw DICOM frames
     (confirmed DICM magic bytes), no file extension.
  3. BIKNA-XRAY/.../*                         -- 119 raw DICOM frames
     (confirmed DICM magic bytes), no file extension.
  4. swaroopa/XRAY DICOM/*.seq.nrrd           -- 173-frame multi-frame NRRD
     sequence (already Slicer-converted).
  5. muthakka/*.seq.nrrd                      -- 224-frame multi-frame NRRD
     sequence (already Slicer-converted).

Output: data/extra_xrays/<source>/<idx>.png  (512x512 uint8, normalised)
        data/extra_xrays_manifest.csv          (no labels -- just paths)

Usage:
    python scripts/extract_extra_xrays.py
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk

ROOT     = Path(__file__).parent.parent
FULLDATA = ROOT / 'data' / 'full data' / 'extracted'
OUT_ROOT = ROOT / 'data' / 'extra_xrays'
MANIFEST = ROOT / 'data' / 'extra_xrays_manifest.csv'
SAVE_SIZE = 512

# Skip frames with too little content (near-blank / collimator-only frames)
MIN_NONZERO_FRAC = 0.15


def _normalize_u8(arr: np.ndarray) -> np.ndarray:
    """Normalise a 2D array to uint8 [0,255] using percentile clipping
    (robust to outlier bright/dark pixels e.g. burned-in text overlays)."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def _save(img_u8: np.ndarray, out_dir: Path, idx: int) -> bool:
    """Save a normalised frame if it has enough non-trivial content."""
    if img_u8.shape[:2] != (SAVE_SIZE, SAVE_SIZE):
        img_u8 = cv2.resize(img_u8, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    nonzero_frac = float(np.count_nonzero(img_u8 > 10)) / img_u8.size
    if nonzero_frac < MIN_NONZERO_FRAC:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f'{idx:05d}.png'), img_u8)
    return True


def extract_tiff_folder(src_dir: Path, out_dir: Path) -> int:
    """Extract .TIF frames (16-bit single images)."""
    n_ok = 0
    # Windows filesystems are case-insensitive: glob('*.TIF') and
    # glob('*.tif') would double-match the same files, so dedupe by
    # resolved path.
    tiff_paths = sorted({p.resolve() for p in
                         list(src_dir.glob('*.TIF')) + list(src_dir.glob('*.tif'))})
    for i, p in enumerate(tiff_paths):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 2:
            continue
        img_u8 = _normalize_u8(img)
        if _save(img_u8, out_dir, n_ok):
            n_ok += 1
    return n_ok


def extract_dicom_folder(src_dir: Path, out_dir: Path) -> int:
    """Extract raw extension-less DICOM frames (recursively)."""
    n_ok = 0
    files = sorted(src_dir.rglob('*'))
    for p in files:
        if not p.is_file():
            continue
        try:
            with open(p, 'rb') as f:
                header = f.read(132)
            if header[128:132] != b'DICM':
                continue
            ds = pydicom.dcmread(str(p), force=True)
            arr = ds.pixel_array
        except Exception:
            continue
        if arr.ndim == 3:
            # Multi-frame DICOM: save each frame separately
            for frame in arr:
                img_u8 = _normalize_u8(frame)
                if _save(img_u8, out_dir, n_ok):
                    n_ok += 1
        elif arr.ndim == 2:
            img_u8 = _normalize_u8(arr)
            if _save(img_u8, out_dir, n_ok):
                n_ok += 1
    return n_ok


def extract_nrrd_sequence(nrrd_path: Path, out_dir: Path) -> int:
    """Extract frames from a multi-frame .seq.nrrd volume."""
    img = sitk.ReadImage(str(nrrd_path))
    arr = sitk.GetArrayFromImage(img)  # (frames, H, W) or (H, W, frames) etc.
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        # Heuristic: frame axis is whichever dim isn't square-ish H/W
        # Most Slicer sequence exports use (frames, H, W)
        pass
    elif arr.ndim == 4:
        # (frames, H, W, channels) -- squeeze channel dim
        arr = arr.squeeze(-1) if arr.shape[-1] == 1 else arr[..., 0]

    n_ok = 0
    for frame in arr:
        if frame.ndim != 2:
            continue
        img_u8 = _normalize_u8(frame)
        if _save(img_u8, out_dir, n_ok):
            n_ok += 1
    return n_ok


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sources_done = []

    # 1. Ramulamma TIFF frames
    src = FULLDATA / 'Ramulamma C arm images intra op'
    if src.exists():
        n = extract_tiff_folder(src, OUT_ROOT / 'ramulamma_carm')
        print(f'Ramulamma C-arm TIFFs: {n} frames extracted')
        sources_done.append(('ramulamma_carm', n))

    # 2. jyothi carm DICOM frames
    src = FULLDATA / 'jyothi carm'
    if src.exists():
        n = extract_dicom_folder(src, OUT_ROOT / 'jyothi_carm')
        print(f'Jyothi C-arm DICOMs: {n} frames extracted')
        sources_done.append(('jyothi_carm', n))

    # 3. BIKNA-XRAY DICOM frames
    src = FULLDATA / 'BIKNA-XRAY'
    if src.exists():
        n = extract_dicom_folder(src, OUT_ROOT / 'bikna_xray')
        print(f'BIKNA-XRAY DICOMs: {n} frames extracted')
        sources_done.append(('bikna_xray', n))

    # 4. swaroopa XRAY DICOM sequence NRRD
    src = FULLDATA / 'swaroopa' / 'XRAY DICOM'
    if src.exists():
        seq_files = list(src.glob('*.seq.nrrd'))
        if seq_files:
            n = extract_nrrd_sequence(seq_files[0], OUT_ROOT / 'swaroopa_xray')
            print(f'Swaroopa XRAY sequence: {n} frames extracted')
            sources_done.append(('swaroopa_xray', n))

    # 5. muthakka XA sequence NRRD
    src = FULLDATA / 'muthakka'
    if src.exists():
        seq_files = list(src.glob('*.seq.nrrd'))
        if seq_files:
            n = extract_nrrd_sequence(seq_files[0], OUT_ROOT / 'muthakka_xray')
            print(f'Muthakka XA sequence: {n} frames extracted')
            sources_done.append(('muthakka_xray', n))

    # -- Write manifest ------------------------------------------------------
    rows = []
    for source_name, _ in sources_done:
        src_dir = OUT_ROOT / source_name
        for png in sorted(src_dir.glob('*.png')):
            rows.append({
                'source': source_name,
                'img_path': str(png.relative_to(ROOT)).replace('\\', '/'),
            })

    with open(MANIFEST, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['source', 'img_path'])
        w.writeheader()
        w.writerows(rows)

    print(f'\nTotal extra X-ray frames: {len(rows)}')
    print(f'Manifest written: {MANIFEST}')


if __name__ == '__main__':
    main()
