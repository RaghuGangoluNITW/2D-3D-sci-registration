"""
train_landmark_detector.py
===========================
LOPO training of LandmarkHeatmapNet (MobileNetV3-Small + UNet decoder) --
the heatmap-based 2D vertebra detector, as an alternative to the direct
SSM-parameter regression approach (train_ssm_regressor.py).

Why try this: the SSM regressor asks the CNN to predict 8 abstract global
shape scalars per image (a coarse, whole-image task). The heatmap detector
instead asks it to find each vertebra independently as a local, mostly
translation-invariant pattern -- a task that tends to generalise better
from small real datasets (~68 labeled images here) because each output
channel only needs to learn "what does an L3 centroid look like nearby",
not "what is the global shape code of this whole spine".

For each fold (leave-one-patient-out):
  - Train on all other patients' real images + synthetic DRR / converted
    X-ray background rows (same augmentation manifests as the SSM
    regressor, for a fair comparison)
  - Target: NUM_CLASSES (6) Gaussian heatmaps
  - Evaluate: mean mm error on real val images (checkpoint selection
    criterion), plus PCK@10mm as a secondary/reported metric

Usage:
    python scripts/train_landmark_detector.py \
        --manifest data/landmark_manifest_real.csv \
        --epochs 200 --batch 4 --lr 3e-4 --view AP \
        --synthetic_manifest data/synthetic_drr_manifest_ap.csv \
        --converted_manifest data/converted_xray_manifest_ap.csv \
        --ckpt_dir checkpoints/landmark_detector_ap
"""

import sys, argparse, csv, time, random
from pathlib import Path
from collections import defaultdict

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from landmark_model import LandmarkHeatmapNet, HeatmapLoss
from landmark_dataset import (
    CLASS_NAMES, NUM_CLASSES, CLS2IDX,
    IMG_SIZE, HMAP_SIZE, _load_xray,
)

CKPT_DIR = ROOT / 'checkpoints' / 'landmark_detector'
PIXEL_MM = 0.2


# ── Dataset ───────────────────────────────────────────────────────────────────

class HeatmapDataset(Dataset):
    """
    Each sample: CLAHE X-ray image + NUM_CLASSES Gaussian heatmaps.

    Same hybrid loading strategy as SSMRegressionDataset:
      NRRD (real X-rays, ~68)        -> pre-load into RAM
      PNG  (synthetic DRRs, 1000s)   -> lazy-load per batch
    """

    def __init__(self, rows: list, augment: bool = False, oversample: int = 1):
        self.augment = augment

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        n_preload = sum(1 for r in rows
                        if not r['img_path'].lower().endswith('.png'))
        n_lazy    = len(rows) - n_preload
        print(f'  Pre-loading {n_preload} NRRD/JPG images; '
              f'{n_lazy} PNGs will load lazily ...', flush=True)

        self.imgs      = []   # np.ndarray if pre-loaded else None
        self.img_paths = []   # path if lazy else None
        self.labels    = []
        self.coords    = []
        self.hw        = []

        for row in rows:
            is_png = row['img_path'].lower().endswith('.png')
            labels      = row['labels'].split('|')
            coords_strs = row['coords_px'].split(';')
            coords_px   = [[float(x) for x in c.strip().split(',')]
                           for c in coords_strs]

            if is_png:
                # Synthetic DRRs are always saved at 512x512 (SAVE_SIZE in
                # generate_synthetic_drrs.py), matching IMG_SIZE here.
                self.imgs.append(None)
                self.img_paths.append(row['img_path'])
                self.labels.append(labels)
                self.coords.append(coords_px)
                self.hw.append((IMG_SIZE, IMG_SIZE))
            else:
                img = _load_xray(row['img_path'])
                H, W = img.shape[:2]
                img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                img_cl = clahe.apply(img_u8).astype(np.float32) / 255.0
                self.imgs.append(img_cl)
                self.img_paths.append(None)
                self.labels.append(labels)
                self.coords.append(coords_px)
                self.hw.append((H, W))

        self.indices = list(range(len(rows))) * oversample

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        labels = list(self.labels[i])
        coords = [list(c) for c in self.coords[i]]
        H, W   = self.hw[i]

        if self.imgs[i] is not None:
            img = self.imgs[i].copy()
        else:
            raw = cv2.imread(self.img_paths[i], cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raw = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            img = raw.astype(np.float32) / 255.0
            if img.shape[0] != H or img.shape[1] != W:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            img, coords = _augment(img, coords, (H, W))

        hmap = _make_heatmaps(labels, coords, (H, W))

        img_rs = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_LINEAR)
        img_t  = torch.from_numpy(img_rs).unsqueeze(0)
        hmap_t = torch.from_numpy(hmap)
        valid  = torch.zeros(NUM_CLASSES, dtype=torch.bool)
        for lbl in labels:
            idx2 = CLS2IDX.get(lbl)
            if idx2 is not None:
                valid[idx2] = True
        return img_t, hmap_t, valid


def _make_heatmaps(labels, coords_px, orig_hw, sigma: float = 16.0):
    """Build (NUM_CLASSES, HMAP_SIZE, HMAP_SIZE) Gaussian heatmaps."""
    H, W = orig_hw
    sx = HMAP_SIZE / W
    sy = HMAP_SIZE / H
    hmap = np.zeros((NUM_CLASSES, HMAP_SIZE, HMAP_SIZE), dtype=np.float32)
    ys = np.arange(HMAP_SIZE, dtype=np.float32)
    xs = np.arange(HMAP_SIZE, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    for lbl, (cx, cy) in zip(labels, coords_px):
        idx = CLS2IDX.get(lbl.strip() if isinstance(lbl, str) else lbl)
        if idx is None:
            continue
        cx_h, cy_h = cx * sx, cy * sy
        blob = np.exp(-((grid_x - cx_h)**2 + (grid_y - cy_h)**2) / (2 * sigma**2))
        hmap[idx] = np.maximum(hmap[idx], blob)
    return hmap


def _augment(img: np.ndarray, coords: list, orig_hw: tuple):
    """Random augmentation: intensity, blur, noise, affine, flip, cutout."""
    H, W = orig_hw
    rng  = np.random.default_rng()

    alpha = rng.uniform(0.6, 1.5)
    beta  = rng.uniform(-0.2, 0.2)
    img   = np.clip(alpha * img + beta, 0, 1)
    gamma = rng.uniform(0.6, 1.8)
    img   = np.clip(img ** gamma, 0, 1)
    if rng.random() < 0.4:
        ksize = int(rng.choice([3, 5, 7]))
        img   = cv2.GaussianBlur(img, (ksize, ksize), 0)
    noise = rng.normal(0, rng.uniform(0.0, 0.06), img.shape).astype(np.float32)
    img   = np.clip(img + noise, 0, 1)

    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.80, 1.20)
    tx    = rng.uniform(-0.05 * W, 0.05 * W)
    ty    = rng.uniform(-0.05 * H, 0.05 * H)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += tx; M[1, 2] += ty
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    new_coords = []
    for (x, y) in coords:
        pt = M @ np.array([x, y, 1.0])
        new_coords.append([float(pt[0]), float(pt[1])])

    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
        new_coords = [[W - c[0], c[1]] for c in new_coords]

    # Random cutout (simulate metal implants / occlusion)
    if rng.random() < 0.3:
        ew = int(rng.uniform(0.05, 0.20) * W)
        eh = int(rng.uniform(0.05, 0.20) * H)
        ex = rng.integers(0, max(1, W - ew))
        ey = rng.integers(0, max(1, H - eh))
        img[ey:ey+eh, ex:ex+ew] = float(rng.uniform(0.0, 0.4))

    return img, new_coords


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, rows, device, pixel_mm=PIXEL_MM, pck_thresh_mm=10.0):
    """Return (mean_err_mm, pck_at_thresh, per_vertebra_errs)."""
    model.eval()
    all_errs  = []
    vert_errs = defaultdict(list)
    n_correct = 0
    n_total   = 0

    for row in rows:
        img = _load_xray(row['img_path'])
        H, W = img.shape[:2]
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img_cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_u8).astype(np.float32) / 255.0
        img_rs = cv2.resize(img_cl, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to(device)

        logits = model(t)
        hmap   = logits.sigmoid()[0].cpu().numpy()
        sx, sy = W / HMAP_SIZE, H / HMAP_SIZE

        labels      = row['labels'].split('|')
        coords_strs = row['coords_px'].split(';')
        gt = {}
        for lbl, cs in zip(labels, coords_strs):
            u, v = map(float, cs.strip().split(','))
            gt[lbl.strip()] = np.array([u, v])

        for i, name in enumerate(CLASS_NAMES):
            if name not in gt:
                continue
            ch = hmap[i]
            flat_idx = int(ch.argmax())
            hx = (flat_idx % HMAP_SIZE + 0.5) * sx
            hy = (flat_idx // HMAP_SIZE + 0.5) * sy
            err = float(np.linalg.norm(np.array([hx, hy]) - gt[name])) * pixel_mm
            all_errs.append(err)
            vert_errs[name].append(err)
            n_total += 1
            if err <= pck_thresh_mm:
                n_correct += 1

    mean_err = float(np.mean(all_errs)) if all_errs else float('nan')
    pck      = (n_correct / n_total) if n_total else float('nan')
    return mean_err, pck, vert_errs


# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(train_rows, val_rows, args, fold_name):
    device = torch.device(args.device)

    max_per_epoch = 1500
    if len(train_rows) > max_per_epoch:
        real_in_train  = [r for r in train_rows if r['img_path'].endswith('.nrrd')]
        other_in_train = [r for r in train_rows if not r['img_path'].endswith('.nrrd')]
        n_sample = max(0, max_per_epoch - len(real_in_train))
        sampled  = random.sample(other_in_train, min(n_sample, len(other_in_train)))
        train_rows = real_in_train + sampled
        print(f'  Epoch dataset capped: {len(real_in_train)} real + '
              f'{len(sampled)} synthetic/converted = {len(train_rows)} total')

    oversample = max(1, 200 // max(len(train_rows), 1))

    ds_train = HeatmapDataset(train_rows, augment=True, oversample=oversample)
    ds_val   = HeatmapDataset(val_rows,   augment=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch,
                          shuffle=True, num_workers=args.workers,
                          pin_memory=True,
                          persistent_workers=(args.workers > 0),
                          prefetch_factor=4 if args.workers > 0 else None)

    # Suppress timm's "Unexpected keys found while loading pretrained weights" warning.
    # timm writes this to stderr when adapting a pretrained backbone to a new head,
    # which is expected behaviour here. On Windows/PowerShell with $ErrorActionPreference
    # set to 'Stop', any stderr output from a native command is treated as a fatal error
    # and kills the training process. Redirecting stderr during model construction avoids
    # this without hiding any real errors (which are raised as exceptions, not stderr text).
    import contextlib, io as _io
    with contextlib.redirect_stderr(_io.StringIO()):
        model = LandmarkHeatmapNet(pretrained=args.pretrain is None, dropout=0.3).to(device)
    if args.pretrain:
        ckpt_pre = torch.load(args.pretrain, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt_pre['model_state'], strict=False)
        print(f'  Loaded pre-trained weights from {Path(args.pretrain).name} '
              f'(missing={len(missing)} unexpected={len(unexpected)})')

    # Freeze the earliest backbone blocks (low-level edge/texture filters
    # transfer fine from ImageNet even for X-rays); train the rest + decoder.
    for name, p in model._backbone.named_parameters():
        if 'blocks.0' in name or 'conv_stem' in name or 'bn1' in name:
            p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)

    warmup_steps = max(1, int(args.epochs * 0.10))
    def _lr_lambda(ep):
        if ep < warmup_steps:
            return float(ep + 1) / float(warmup_steps + 1)
        progress = (ep - warmup_steps) / max(1, args.epochs - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    criterion = HeatmapLoss(pos_weight=100.0).to(device)

    best_err  = float('inf')
    best_pck  = -1.0
    best_ep   = 0
    patience  = 50
    out_dir   = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f'fold_{fold_name}.pth'

    print(f'  Oversampled: {len(train_rows)} x {oversample} = '
          f'{len(train_rows)*oversample} samples/epoch')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'  Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M total')

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for imgs, hmaps, valid in dl_train:
            imgs, hmaps, valid = imgs.to(device), hmaps.to(device), valid.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, hmaps, valid_mask=valid)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        scheduler.step()
        train_loss /= len(ds_train)

        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            val_err, val_pck, verts = evaluate(model, val_rows, device)
            elapsed = time.time() - t0
            vert_str = '  '.join(
                f'{n}={np.mean(verts[n]):.0f}' if verts[n] else f'{n}=---'
                for n in CLASS_NAMES)

            # Select "best" by mean mm error, not PCK@10mm -- with so few
            # training images the 10mm threshold is too tight to move off
            # 0.000 for many epochs even while the actual error is falling
            # steeply, which would otherwise freeze checkpoint selection at
            # epoch 1 and throw away real improvement.
            improved = val_err < best_err
            if improved:
                best_err = val_err
                best_pck = val_pck
                best_ep  = ep
                torch.save({'model_state': model.state_dict(),
                            'val_pck':     best_pck,
                            'val_err_mm':  best_err,
                            'epoch':       ep,
                            'fold':        fold_name}, ckpt_path)

            marker = ' [BEST]' if improved else ''
            print(f'  ep {ep:3d}/{args.epochs}  loss={train_loss:.4f}  '
                  f'val_err={val_err:.1f}mm  val_pck@10mm={val_pck:.3f}  '
                  f'[{vert_str}]{marker}  ({elapsed:.1f}s)')

            if ep - best_ep >= patience:
                print(f'  Early stop at ep {ep} (best ep={best_ep})')
                break

    print(f'  -> Best  val_err={best_err:.1f}mm  val_pck@10mm={best_pck:.3f}  (ep {best_ep})')
    return best_err


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='data/landmark_manifest_real.csv')
    p.add_argument('--epochs',   type=int,   default=200)
    p.add_argument('--batch',    type=int,   default=4)
    p.add_argument('--lr',       type=float, default=3e-4)
    p.add_argument('--device',   default='cuda')
    p.add_argument('--pretrain', default=None)
    p.add_argument('--only', nargs='+', default=None)
    p.add_argument('--ckpt_dir', default=None)
    p.add_argument('--view', default=None, choices=['AP', 'LAT'])
    p.add_argument('--synthetic_manifest', default=None)
    p.add_argument('--converted_manifest', default=None)
    p.add_argument('--workers', type=int, default=8,
                    help='DataLoader worker processes (0=synchronous main-process loading)')
    return p.parse_args()


def main():
    args = parse_args()

    real_rows = []
    with open(ROOT / args.manifest) as f:
        for row in csv.DictReader(f):
            if args.view and args.view not in row.get('view', row.get('set_name', '')):
                continue
            if row['img_path'].endswith('.nrrd'):
                real_rows.append(row)

    synthetic_rows = []
    if args.synthetic_manifest:
        syn_path = Path(args.synthetic_manifest)
        if not syn_path.is_absolute():
            syn_path = ROOT / syn_path
        if syn_path.exists():
            with open(syn_path) as f:
                for row in csv.DictReader(f):
                    if args.view and args.view not in row.get('view', ''):
                        continue
                    synthetic_rows.append(row)
            print(f'Synthetic DRR rows : {len(synthetic_rows)}  (from {syn_path.name})')
        else:
            print(f'WARNING: synthetic_manifest not found: {syn_path}')

    converted_rows = []
    if args.converted_manifest:
        conv_path = Path(args.converted_manifest)
        if not conv_path.is_absolute():
            conv_path = ROOT / conv_path
        if conv_path.exists():
            with open(conv_path) as f:
                for row in csv.DictReader(f):
                    if args.view and args.view not in row.get('view', ''):
                        continue
                    converted_rows.append(row)
            print(f'Converted X-ray rows: {len(converted_rows)}  (from {conv_path.name})')
        else:
            print(f'WARNING: converted_manifest not found: {conv_path}')

    print(f'Real images : {len(real_rows)}  view={args.view or "all"}')

    by_patient = defaultdict(list)
    for r in real_rows:
        by_patient[r['patient']].append(r)
    patients = sorted(by_patient.keys())

    print(f'\nLOPO folds : {len(patients)} patients')
    print(f'Background : synthetic={len(synthetic_rows)}  converted={len(converted_rows)}')
    print('=' * 62)

    fold_results = {}
    for patient in patients:
        if args.only and patient not in args.only:
            continue
        val_rows   = by_patient[patient]
        other_real = [r for p, rs in by_patient.items() if p != patient for r in rs]
        conv_filtered = [r for r in converted_rows if r['patient'] != f'CONV_{patient}']
        train_rows = other_real + synthetic_rows + conv_filtered

        print(f'\n{"-"*62}')
        print(f'  FOLD: val={patient}  other_real={len(other_real)}  '
              f'synthetic={len(synthetic_rows)}  converted={len(conv_filtered)}  '
              f'total_train={len(train_rows)}')
        print(f'  Heatmap detector training  ({args.epochs} epochs)')
        print(f'{"-"*62}')

        best = train_fold(train_rows, val_rows, args, fold_name=patient)
        fold_results[patient] = best

    print('\n' + '=' * 62)
    print('  LOPO SUMMARY  (best val mean error, mm, on real val images)')
    print('=' * 62)
    for patient in patients:
        err = fold_results.get(patient, float('nan'))
        if np.isnan(err):
            continue
        bar = '#' * max(0, int(30 - err / 3))
        print(f'  {patient:<22}  {err:5.1f}mm  {bar}')
    valid = [v for v in fold_results.values() if not np.isnan(v)]
    if valid:
        print(f'  {"MEAN":<22}  {np.mean(valid):5.1f}mm')
    print('=' * 62)


if __name__ == '__main__':
    main()
