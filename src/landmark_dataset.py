"""
landmark_dataset.py
====================
PyTorch Dataset for 2D vertebra landmark detection on lumbar X-rays.

Each sample is one X-ray image with 0–6 vertebra centroids annotated in
Slicer's CENTROID.mrk.json format.

Label convention (CLASS_NAMES order = channel index):
    0: D12   1: L1   2: L2   3: L3   4: L4   5: L5

Input  : 1-channel float32 image, resized to IMG_SIZE × IMG_SIZE
Output : NUM_CLASSES × HMAP_SIZE × HMAP_SIZE  Gaussian heatmaps (float32, 0–1)

Usage::
    from landmark_dataset import LandmarkDataset, build_manifest
    manifest = build_manifest('data/ProcessedData/Processed Data 2D-3D')
    ds = LandmarkDataset(manifest, augment=True)
    img, hmap, meta = ds[0]
"""

from __future__ import annotations
import json, csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import nrrd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# ── Constants ────────────────────────────────────────────────────────────────
# CENTROID classes (D12, L1-L5) plus LEFT_PED/RIGHT_PED pairs, extracted
# automatically from each patient's existing CT via
# scripts/extract_transverse_process_tips.py -- added to give EPnP a
# NON-COLLINEAR point set. A direct noise-injection test proved that
# solving pose from centroids alone (nearly collinear along the spine's
# own axis) is unstable even with near-perfect 2D detection (median 15deg
# / 340mm 3D pose error from just 1mm of 2D noise); adding the bilateral
# pedicle points (spread left/right of the spine axis) was shown to
# reduce that same-test error by roughly 2.5x-47x depending on noise
# level (see project history / JYOTHI verification). See
# scripts/extract_transverse_process_tips.py for the full derivation and
# scripts/build_augmented_landmark_manifest.py for how these are merged
# into the training manifest.
CLASS_NAMES = [
    'D12', 'L1', 'L2', 'L3', 'L4', 'L5',
    'D12_LEFT_PED', 'D12_RIGHT_PED',
    'L1_LEFT_PED',  'L1_RIGHT_PED',
    'L2_LEFT_PED',  'L2_RIGHT_PED',
    'L3_LEFT_PED',  'L3_RIGHT_PED',
    'L4_LEFT_PED',  'L4_RIGHT_PED',
    'L5_LEFT_PED',  'L5_RIGHT_PED',
]
NUM_CLASSES  = len(CLASS_NAMES)
CLS2IDX      = {c: i for i, c in enumerate(CLASS_NAMES)}

IMG_SIZE  = 512   # network input resolution (pixels)
HMAP_SIZE = 128   # heatmap output resolution  (1/4 of IMG_SIZE)
SIGMA_PX  = 16    # Gaussian sigma in heatmap pixels (~16px = ~26mm at orig scale)
                  # Broad targets are critical for generalization with <60 training images.


# ── Manifest builder ─────────────────────────────────────────────────────────
def build_manifest(data_root: str | Path,
                   csv_path:  str | Path | None = None) -> List[dict]:
    """
    Walk ``data_root`` and collect every set that has both a *.nrrd X-ray and
    a CENTROID.mrk.json landmark file.

    Returns a list of dicts:
        patient, set_name, view, img_path, lm_path, labels, coords_px

    Optionally writes a CSV to ``csv_path``.
    """
    root = Path(data_root)
    samples: List[dict] = []

    for pat_dir in sorted(root.iterdir()):
        xray_root = pat_dir / 'XRAY Label'
        if not xray_root.exists():
            continue
        for view_dir in sorted(xray_root.iterdir()):
            view = view_dir.name          # e.g. XRAY-AP or XRAY-LAT
            for set_dir in sorted(view_dir.iterdir()):
                nrrds = sorted(set_dir.glob('*.nrrd'))
                mrks  = list(set_dir.glob('CENTROID.mrk.json'))
                if not nrrds or not mrks:
                    continue
                lm_data  = json.load(open(mrks[0]))
                ctrl_pts = lm_data['markups'][0]['controlPoints']
                labels   = [cp['label']    for cp in ctrl_pts]
                coords   = [cp['position'] for cp in ctrl_pts]   # [x, y, 0]
                samples.append({
                    'patient':   pat_dir.name,
                    'set_name':  set_dir.name,
                    'view':      view,
                    'img_path':  str(nrrds[0]),
                    'lm_path':   str(mrks[0]),
                    'labels':    labels,
                    'coords_px': [[c[0], c[1]] for c in coords],
                })

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['patient', 'set_name', 'view', 'img_path', 'lm_path',
                        'labels', 'coords_px'])
            for s in samples:
                w.writerow([s['patient'], s['set_name'], s['view'],
                            s['img_path'], s['lm_path'],
                            '|'.join(s['labels']),
                            ';'.join(f"{c[0]:.2f},{c[1]:.2f}"
                                     for c in s['coords_px'])])
    return samples


# ── Heatmap generation ────────────────────────────────────────────────────────
def _make_heatmaps(labels:    List[str],
                   coords_px: List[List[float]],
                   orig_hw:   Tuple[int, int],
                   sigma:     float = SIGMA_PX) -> np.ndarray:
    """
    Build a (NUM_CLASSES, HMAP_SIZE, HMAP_SIZE) float32 heatmap tensor.

    ``coords_px`` are in the *original* image space (orig_hw = (H, W)).
    We scale them to HMAP_SIZE.
    """
    H, W   = orig_hw
    sx     = HMAP_SIZE / W
    sy     = HMAP_SIZE / H
    hmap   = np.zeros((NUM_CLASSES, HMAP_SIZE, HMAP_SIZE), dtype=np.float32)

    ys = np.arange(HMAP_SIZE, dtype=np.float32)
    xs = np.arange(HMAP_SIZE, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)   # (HMAP_SIZE, HMAP_SIZE)

    for lbl, (cx, cy) in zip(labels, coords_px):
        idx = CLS2IDX.get(lbl)
        if idx is None:
            continue
        cx_h = cx * sx
        cy_h = cy * sy
        blob = np.exp(-((grid_x - cx_h)**2 + (grid_y - cy_h)**2)
                       / (2 * sigma**2))
        hmap[idx] = np.maximum(hmap[idx], blob)

    return hmap


# ── Image loading ─────────────────────────────────────────────────────────────
def _load_xray(path: str) -> np.ndarray:
    """Load a *.nrrd or *.jpg X-ray, squeeze to (H, W), normalise to [0, 1] float32."""
    path = str(path)
    if path.lower().endswith('.jpg') or path.lower().endswith('.png'):
        import cv2 as _cv2
        data = _cv2.imread(path, _cv2.IMREAD_GRAYSCALE)
        if data is None:
            raise FileNotFoundError(f'Could not read image: {path}')
        data = data.astype(np.float32)
    else:
        data, _ = nrrd.read(path)
        if data.ndim == 3:
            if data.shape[2] == 1:
                data = data[:, :, 0]
            elif data.shape[0] == 1:
                data = data[0]
        data = data.astype(np.float32)
    lo, hi = data.min(), data.max()
    if hi > lo:
        data = (data - lo) / (hi - lo)
    return data   # (H, W) in [0, 1]


# ── Augmentation helpers ──────────────────────────────────────────────────────
def _augment(img: np.ndarray,
             hmap: np.ndarray,
             coords: List[List[float]],
             orig_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random augmentation to the (H, W) image and regenerate heatmaps.

    Augmentations applied:
        - Intensity jitter (contrast, brightness, gamma)
        - Gaussian blur (p=0.4)
        - Gaussian noise
        - Random rotation ±15°, scale 0.80–1.20×
        - Horizontal flip (p=0.5)
        - Random translation ±5% of image size
    """
    H, W = orig_hw
    rng  = np.random.default_rng()

    # ── Intensity jitter ────────────────────────────────────────────────────
    alpha = rng.uniform(0.6, 1.5)    # contrast
    beta  = rng.uniform(-0.2, 0.2)   # brightness
    img   = np.clip(alpha * img + beta, 0, 1)

    # ── Gamma ────────────────────────────────────────────────────────────────
    gamma = rng.uniform(0.6, 1.8)
    img   = np.clip(img ** gamma, 0, 1)

    # ── Gaussian blur (simulates detector blur / motion) ─────────────────────
    if rng.random() < 0.4:
        ksize = rng.choice([3, 5, 7])
        img   = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # ── Gaussian noise ───────────────────────────────────────────────────────
    noise = rng.normal(0, rng.uniform(0.0, 0.06), img.shape).astype(np.float32)
    img   = np.clip(img + noise, 0, 1)

    # ── Affine: rotation + scale + translation ────────────────────────────────
    angle  = rng.uniform(-15, 15)
    scale  = rng.uniform(0.80, 1.20)
    tx     = rng.uniform(-0.05 * W, 0.05 * W)
    ty     = rng.uniform(-0.05 * H, 0.05 * H)
    cx_img, cy_img = W / 2, H / 2
    M = cv2.getRotationMatrix2D((cx_img, cy_img), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    # Transform landmark coords
    new_coords = []
    for (x, y) in coords:
        pt = M @ np.array([x, y, 1.0])
        new_coords.append([float(pt[0]), float(pt[1])])

    # ── Horizontal flip ───────────────────────────────────────────────────────
    if rng.random() < 0.5:
        img        = img[:, ::-1].copy()
        new_coords = [[W - c[0], c[1]] for c in new_coords]

    # ── Recompute heatmaps from new coords ────────────────────────────────────
    return img, new_coords


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE to normalise appearance across DRRs and real X-rays."""
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out    = clahe.apply(img_u8).astype(np.float32) / 255.0
    return out


def _load_image(path: str) -> np.ndarray:
    """Load an X-ray image from .nrrd or .png/.jpg — returns (H,W) float32 [0,1],
    with CLAHE applied to normalise appearance across DRRs and real X-rays."""
    p = path.lower()
    if p.endswith('.nrrd'):
        img = _load_xray(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'Cannot read image: {path}')
        img = img.astype(np.float32) / 255.0
    return _apply_clahe(img)

class LandmarkDataset(Dataset):
    """
    Parameters
    ----------
    samples : list of dicts from build_manifest()
    augment : apply random augmentation (use True for training only)
    img_size : network input size (default IMG_SIZE=512)
    """

    def __init__(self,
                 samples:  List[dict],
                 augment:  bool = False,
                 img_size: int  = IMG_SIZE):
        self.samples  = samples
        self.augment  = augment
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # Load raw image (handles both NRRD and PNG)
        img     = _load_image(s['img_path'])      # (H, W) float32
        orig_hw = img.shape[:2]
        labels  = s['labels']
        coords  = [list(c) for c in s['coords_px']]  # copy

        # Augment (modifies coords in-place semantically)
        if self.augment:
            img, coords = _augment(img, None, coords, orig_hw)

        # Build heatmaps at original resolution then they're made at HMAP_SIZE
        hmap = _make_heatmaps(labels, coords, orig_hw)

        # Resize image to IMG_SIZE × IMG_SIZE
        img_rs = cv2.resize(img, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_LINEAR)

        # To tensors
        img_t  = torch.from_numpy(img_rs).unsqueeze(0)   # (1, H, W)
        hmap_t = torch.from_numpy(hmap)                  # (C, HMAP_SIZE, HMAP_SIZE)

        meta = {
            'patient':  s['patient'],
            'set_name': s['set_name'],
            'view':     s['view'],
            'labels':   labels,
            'coords_px': coords,
            'orig_hw':  orig_hw,
        }
        return img_t, hmap_t, meta


# ── Coord extraction from heatmap ─────────────────────────────────────────────
def soft_argmax(hmap: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft-argmax over a (C, H, W) heatmap.

    Returns (C, 2) tensor of (x, y) coords in heatmap pixel space.
    """
    C, H, W = hmap.shape
    flat = hmap.view(C, -1)
    flat = torch.softmax(flat * 100, dim=-1)     # temperature sharpening

    ys = torch.arange(H, dtype=torch.float32, device=hmap.device)
    xs = torch.arange(W, dtype=torch.float32, device=hmap.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)

    pred_x = (flat * grid_x).sum(dim=-1)
    pred_y = (flat * grid_y).sum(dim=-1)
    return torch.stack([pred_x, pred_y], dim=-1)  # (C, 2)


def hmap_to_coords(hmap: np.ndarray,
                   conf_threshold: float = 0.05) -> Dict[str, Optional[np.ndarray]]:
    """
    Convert a (C, HMAP_SIZE, HMAP_SIZE) numpy heatmap (from inference)
    to a dict {vertebra_name: [x_px, y_px]} in *heatmap* pixel space.

    Returns None for channels whose peak confidence < conf_threshold.
    """
    result: Dict[str, Optional[np.ndarray]] = {}
    for i, name in enumerate(CLASS_NAMES):
        ch  = hmap[i]
        peak = float(ch.max())
        if peak < conf_threshold:
            result[name] = None
            continue
        flat_idx = int(ch.argmax())
        y = flat_idx // HMAP_SIZE
        x = flat_idx  % HMAP_SIZE
        result[name] = np.array([float(x), float(y)])
    return result


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else \
           'data/ProcessedData/Processed Data 2D-3D'
    manifest = build_manifest(root, csv_path='data/landmark_manifest.csv')
    print(f'Found {len(manifest)} labelled images')

    ds = LandmarkDataset(manifest, augment=True)
    img, hmap, meta = ds[0]
    print(f'img shape : {img.shape}   dtype={img.dtype}')
    print(f'hmap shape: {hmap.shape}  max={hmap.max():.3f}')
    print(f'meta      : {meta["patient"]} / {meta["set_name"]}  '
          f'labels={meta["labels"]}')
    coords = hmap_to_coords(hmap.numpy())
    print('Predicted heatmap peaks:', {k: v for k, v in coords.items() if v is not None})
