"""
train_ssm_regressor.py
======================
LOPO training of the SSMRegressor on real X-ray images.

For each fold (leave-one-patient-out):
  - Train on all other patients' real images (67 total → ~60 train, ~6 val)
  - Target: 8 normalised SSM parameters per image
  - Evaluate: mean spine error in mm on val images

Usage:
    python scripts/train_ssm_regressor.py \
        --manifest data/landmark_manifest_combined.csv \
        --epochs 200 --batch 8 --lr 3e-4 --device cuda
"""

import sys, argparse, csv, time
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

from ssm_regressor import SSMRegressor, SSMRegressionLoss, encode_params, decode_params
from spine_ssm import SpineSSM, CLASS_NAMES
from landmark_dataset import IMG_SIZE, _load_xray, CLS2IDX

SSM_PATH  = ROOT / 'checkpoints' / 'spine_ssm.pkl'
CKPT_DIR  = ROOT / 'checkpoints' / 'ssm_regressor'
PIXEL_MM  = 0.2


# ── Dataset ───────────────────────────────────────────────────────────────────

class SSMRegressionDataset(Dataset):
    """
    Each sample: CLAHE X-ray image  +  normalised SSM parameter vector.
    Images are pre-loaded, CLAHE-processed and resized into RAM at init time.
    Only augmentation (fast, numpy-only) happens in __getitem__.
    """

    def __init__(self, rows: list, ssm: SpineSSM,
                 augment: bool = False, oversample: int = 1):
        self.ssm      = ssm
        self.augment  = augment
        self.rows_base = rows

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Hybrid loading strategy:
        #   NRRD files (real X-rays, ~68)  → pre-load into RAM (slow per-file)
        #   PNG files (synthetic DRRs, ~6000) → store path, load lazily per batch
        #   JPG files (BUU)                → pre-load into RAM
        self.imgs        = []   # None = lazy (PNG), np.ndarray = pre-loaded
        self.imgs_rs     = []   # same
        self.img_paths   = []   # path for lazy-loaded rows
        self.coords_list = []
        self.hw_list     = []

        n_preload = sum(1 for r in rows
                        if not r['img_path'].lower().endswith('.png'))
        n_lazy    = len(rows) - n_preload
        print(f'  Pre-loading {n_preload} NRRD/JPG images; '
              f'{n_lazy} PNGs will load lazily ...', flush=True)

        for row in rows:
            is_png = row['img_path'].lower().endswith('.png')
            labels      = row['labels'].split('|')
            coords_strs = row['coords_px'].split(';')

            if is_png:
                # For PNGs, peek at the size from the manifest coords to get scale
                # PNGs are always saved at SAVE_SIZE (512) in generate_synthetic_drrs
                coords_dict = {}
                for lbl, cs in zip(labels, coords_strs):
                    u, v = map(float, cs.strip().split(','))
                    coords_dict[lbl.strip()] = [u, v]   # already in IMG_SIZE space if 512==IMG_SIZE
                # If SAVE_SIZE (512) != IMG_SIZE, scale coords
                png_save_size = 512   # matches SAVE_SIZE in generate_synthetic_drrs.py
                sx = IMG_SIZE / png_save_size
                if sx != 1.0:
                    coords_dict = {k: [u*sx, v*sx] for k,(u,v) in coords_dict.items()}
                self.imgs.append(None)
                self.imgs_rs.append(None)
                self.img_paths.append(row['img_path'])
                self.coords_list.append(coords_dict)
                self.hw_list.append((IMG_SIZE, IMG_SIZE))
            else:
                img = _load_xray(row['img_path'])
                H, W = img.shape[:2]
                img_u8  = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                img_cl  = clahe.apply(img_u8).astype(np.float32) / 255.0
                img_rs  = cv2.resize(img_cl, (IMG_SIZE, IMG_SIZE),
                                     interpolation=cv2.INTER_LINEAR)
                coords_dict = {}
                sx, sy = IMG_SIZE / W, IMG_SIZE / H
                for lbl, cs in zip(labels, coords_strs):
                    u, v = map(float, cs.strip().split(','))
                    coords_dict[lbl.strip()] = [u * sx, v * sy]
                self.imgs.append(img_rs)
                self.imgs_rs.append(img_rs)
                self.img_paths.append(None)
                self.coords_list.append(coords_dict)
                self.hw_list.append((IMG_SIZE, IMG_SIZE))

        # Pre-encode SSM targets for all base images (fast numpy lstsq, no scipy)
        self.targets = []
        for coords_dict, (H, W) in zip(self.coords_list, self.hw_list):
            p = _fast_encode_init(coords_dict, (H, W), ssm)
            self.targets.append(p)

        # Oversample by repeating indices
        self.indices = list(range(len(rows))) * oversample

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        i           = self.indices[idx]
        coords_dict = {k: list(v) for k, v in self.coords_list[i].items()}
        H, W        = self.hw_list[i]

        # Load image: pre-loaded (NRRD/JPG) or lazy (PNG)
        if self.imgs[i] is not None:
            img = self.imgs[i].copy()
        else:
            # Lazy load PNG synthetic DRR
            raw = cv2.imread(self.img_paths[i], cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raw = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            img = raw.astype(np.float32) / 255.0
            if img.shape[0] != IMG_SIZE:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                                 interpolation=cv2.INTER_LINEAR)

        # Augment on the pre-loaded CLAHE image (fast: already 512×512)
        if self.augment:
            img, coords_dict = _augment_ssm(img, coords_dict)
            H, W = img.shape[:2]
            img_rs = img   # already IMG_SIZE
        else:
            img_rs = img

        tensor = torch.from_numpy(img_rs).unsqueeze(0)   # (1, H, W)

        # Encode target — fast update when augmented (no scipy optimization)
        if self.augment:
            params = _fast_encode_augmented(coords_dict, (H, W), self.ssm, self.targets[i])
        else:
            params = self.targets[i]

        return tensor, torch.from_numpy(params.copy())


def _fast_encode_augmented(coords_dict: dict, img_hw: tuple,
                            ssm, base_params: np.ndarray) -> np.ndarray:
    """
    Fast target re-encoding for augmented samples.
    Instead of running scipy L-BFGS-B (100ms/sample), we update only
    tx/ty/scale from the transformed coords; shape params b stay the same.
    This is correct because affine transforms don't change spine shape.
    """
    H, W = img_hw
    n_modes = ssm.n_modes
    params = base_params.copy()
    if len(coords_dict) < 2:
        return params
    pts = np.array(list(coords_dict.values()), dtype=np.float32)
    centroid = pts.mean(axis=0)
    span = float(pts[:, 1].max() - pts[:, 1].min()) + 1e-6
    params[n_modes]     = float(centroid[0] / W)
    params[n_modes + 1] = float(centroid[1] / H)
    params[n_modes + 2] = float(np.log(span / (ssm.mean_scale + 1e-9) + 1e-9))
    return params


def _fast_encode_init(coords_dict: dict, img_hw: tuple, ssm) -> np.ndarray:
    """
    Fast SSM parameter encoding for init-time target computation.
    Uses numpy lstsq (microseconds) instead of scipy L-BFGS-B (100ms).
    Computes b via partial-observation least-squares; tx/ty/scale from centroids.
    """
    from spine_ssm import CLASS_NAMES
    H, W     = img_hw
    n_modes  = ssm.n_modes
    params   = np.zeros(n_modes + 3, dtype=np.float32)
    if len(coords_dict) < 2:
        return params

    pts      = np.array(list(coords_dict.values()), dtype=np.float32)
    centroid = pts.mean(axis=0)
    span     = float(pts[:, 1].max() - pts[:, 1].min()) + 1e-6

    # Position / scale params
    params[n_modes]     = float(centroid[0] / W)
    params[n_modes + 1] = float(centroid[1] / H)
    params[n_modes + 2] = float(np.log(span / (ssm.mean_scale + 1e-9) + 1e-9))

    # Shape params b via partial-observation least squares
    obs_idx = [i for i, n in enumerate(CLASS_NAMES) if n in coords_dict]
    if len(obs_idx) >= 2:
        obs_norm = np.zeros(len(CLASS_NAMES) * 2, dtype=np.float64)
        obs_mask = np.zeros(len(CLASS_NAMES) * 2, dtype=bool)
        for i in obs_idx:
            u, v = coords_dict[CLASS_NAMES[i]]
            obs_norm[2 * i]     = (float(u) - centroid[0]) / span
            obs_norm[2 * i + 1] = (float(v) - centroid[1]) / span
            obs_mask[2 * i]     = True
            obs_mask[2 * i + 1] = True
        M       = ssm.modes[obs_mask, :n_modes].astype(np.float64)
        r       = obs_norm[obs_mask] - ssm.mean_shape[obs_mask].astype(np.float64)
        b_raw, _, _, _ = np.linalg.lstsq(M, r, rcond=None)
        b_norm  = b_raw / (np.sqrt(ssm.variances[:n_modes]) + 1e-9)
        params[:n_modes] = b_norm.astype(np.float32)
    return params


def _augment_ssm(img: np.ndarray, coords_dict: dict):
    """Augment image + coords. Returns (img, new_coords_dict)."""
    H, W  = img.shape[:2]
    rng   = np.random.default_rng()

    # ── Intensity ─────────────────────────────────────────────────────────
    alpha = rng.uniform(0.5, 1.6)
    beta  = rng.uniform(-0.25, 0.25)
    img   = np.clip(alpha * img + beta, 0, 1)
    gamma = rng.uniform(0.5, 2.0)
    img   = np.clip(img ** gamma, 0, 1)
    if rng.random() < 0.5:
        ksize = int(rng.choice([3, 5, 7]))
        img   = cv2.GaussianBlur(img, (ksize, ksize), 0)
    noise = rng.normal(0, rng.uniform(0.0, 0.08), img.shape).astype(np.float32)
    img   = np.clip(img + noise, 0, 1)

    # ── Affine ────────────────────────────────────────────────────────────
    angle = rng.uniform(-20, 20)
    scale = rng.uniform(0.65, 1.45)
    tx    = rng.uniform(-0.30 * W, 0.30 * W)
    ty    = rng.uniform(-0.30 * H, 0.30 * H)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += tx;  M[1, 2] += ty
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    new_coords = {}
    for name, (u, v) in coords_dict.items():
        pt  = M @ np.array([u, v, 1.0])
        nu, nv = float(pt[0]), float(pt[1])
        if 0 <= nu < W and 0 <= nv < H:
            new_coords[name] = [nu, nv]
    if len(new_coords) < 2:
        new_coords = {k: list(v) for k, v in coords_dict.items()}

    # ── Elastic deformation (simulates spine flexibility / C-arm angle) ───
    if rng.random() < 0.5:
        alpha_e = rng.uniform(8, 24)
        sigma_e = rng.uniform(4, 8)
        dx = cv2.GaussianBlur(
            rng.uniform(-1, 1, (H, W)).astype(np.float32), (0, 0), sigma_e) * alpha_e
        dy = cv2.GaussianBlur(
            rng.uniform(-1, 1, (H, W)).astype(np.float32), (0, 0), sigma_e) * alpha_e
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        map_x  = (xs + dx).astype(np.float32)
        map_y  = (ys + dy).astype(np.float32)
        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
        deformed = {}
        for name, (u, v) in new_coords.items():
            ui = int(np.clip(round(u), 0, W - 1))
            vi = int(np.clip(round(v), 0, H - 1))
            nu = float(u + dx[vi, ui])
            nv = float(v + dy[vi, ui])
            if 0 <= nu < W and 0 <= nv < H:
                deformed[name] = [nu, nv]
        if len(deformed) >= 2:
            new_coords = deformed

    # ── Random cutout (simulate metal implants / occlusion) ───────────────
    if rng.random() < 0.4:
        ew = int(rng.uniform(0.05, 0.22) * W)
        eh = int(rng.uniform(0.05, 0.22) * H)
        ex = rng.integers(0, max(1, W - ew))
        ey = rng.integers(0, max(1, H - eh))
        img[ey:ey + eh, ex:ex + ew] = float(rng.uniform(0.0, 0.4))

    # ── Horizontal flip ───────────────────────────────────────────────────
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
        new_coords = {n: [W - 1 - u, v] for n, (u, v) in new_coords.items()}

    return img, new_coords if len(new_coords) >= 2 else coords_dict


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(model, ssm, rows, device, pixel_mm=PIXEL_MM):
    """Return mean spine error in mm, per-vertebra errors."""
    model.eval()
    all_errs   = []
    vert_errs  = defaultdict(list)

    with torch.no_grad():
        for row in rows:
            img = _load_xray(row['img_path'])
            H, W = img.shape[:2]

            img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            img_cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_u8).astype(np.float32) / 255.0
            img_rs = cv2.resize(img_cl, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            t      = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to(device)

            params_pred = model(t)[0].cpu().numpy()
            pred_coords = decode_params(params_pred, (H, W), ssm)

            labels      = row['labels'].split('|')
            coords_strs = row['coords_px'].split(';')
            for lbl, cs in zip(labels, coords_strs):
                lbl = lbl.strip()
                u, v = map(float, cs.strip().split(','))
                gt = np.array([u, v])
                if lbl in pred_coords:
                    err = float(np.linalg.norm(pred_coords[lbl] - gt)) * pixel_mm
                    all_errs.append(err)
                    vert_errs[lbl].append(err)

    mean_err = float(np.mean(all_errs)) if all_errs else float('nan')
    return mean_err, vert_errs


# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(train_rows, val_rows, ssm, args, fold_name):
    import random as _random
    device = torch.device(args.device)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    oversample = max(1, 200 // max(len(train_rows), 1))   # ×1 when large dataset present
    # Cap dataset size per epoch to keep training time reasonable on small GPU
    # With 6000+ rows, one epoch = 1500+ batches = too slow. Sample 1000 per epoch.
    max_per_epoch   = 1500
    train_rows_full = train_rows
    import random as _random

    if len(train_rows) > max_per_epoch:
        real_in_train  = [r for r in train_rows if r['img_path'].endswith('.nrrd')]
        other_in_train = [r for r in train_rows if not r['img_path'].endswith('.nrrd')]
        n_sample       = max(0, max_per_epoch - len(real_in_train))
        sampled        = _random.sample(other_in_train, min(n_sample, len(other_in_train)))
        train_rows     = real_in_train + sampled
        print(f'  Epoch dataset capped: {len(real_in_train)} real + '
              f'{len(sampled)} synthetic/converted = {len(train_rows)} total')
        oversample = max(1, 200 // max(len(train_rows), 1))

    ds_train = SSMRegressionDataset(train_rows, ssm, augment=True, oversample=oversample)
    ds_val   = SSMRegressionDataset(val_rows,   ssm, augment=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch,
                          shuffle=True, num_workers=0, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch,
                          shuffle=False, num_workers=0)

    backbone_name = getattr(args, 'backbone', 'resnet50')
    model = SSMRegressor(
        n_modes=ssm.n_modes, dropout=0.3,
        pretrained=args.pretrain is None,
        backbone_name=backbone_name,
    ).to(device)
    if args.pretrain:
        ckpt_pre = torch.load(args.pretrain, map_location=device, weights_only=False)
        # Support loading from a checkpoint with a different backbone gracefully
        missing, unexpected = model.load_state_dict(ckpt_pre['model_state'], strict=False)
        if unexpected:
            print(f'  [pretrain] Ignoring {len(unexpected)} unexpected keys: {unexpected[:3]}...')
        if missing:
            print(f'  [pretrain] {len(missing)} keys not in checkpoint (new layers): {missing[:3]}...')
        print(f'  Loaded pre-trained weights from {Path(args.pretrain).name}')

    # ── Differential LR: deep backbone layers get lower LR, head gets full LR ──
    # ResNet-50:  layer1(frozen) | layer2(0.01×) | layer3(0.1×) | layer4(0.5×) | head(1×)
    # MobileNet:  blocks 0-2(frozen) | blocks 3-4(0.05×) | head(1×)
    def _param_groups(model, base_lr):
        bb = backbone_name
        if 'resnet' in bb:
            # Freeze layer1; tiered LR for layer2→layer4
            for name, p in model.backbone.named_parameters():
                if name.startswith('layer1') or name.startswith('conv1') or name.startswith('bn1'):
                    p.requires_grad_(False)
            groups = [
                {'params': [p for n, p in model.backbone.named_parameters()
                            if p.requires_grad and 'layer2' in n],
                 'lr': base_lr * 0.01, 'weight_decay': 1e-4},
                {'params': [p for n, p in model.backbone.named_parameters()
                            if p.requires_grad and 'layer3' in n],
                 'lr': base_lr * 0.1,  'weight_decay': 1e-4},
                {'params': [p for n, p in model.backbone.named_parameters()
                            if p.requires_grad and 'layer4' in n],
                 'lr': base_lr * 0.5,  'weight_decay': 1e-4},
                {'params': list(model.head.parameters()),
                 'lr': base_lr, 'weight_decay': 5e-3},
            ]
            # Filter empty groups
            groups = [g for g in groups if g['params']]
        elif 'efficientnet' in bb:
            for name, p in model.backbone.named_parameters():
                if any(f'blocks.{i}' in name for i in range(3)):
                    p.requires_grad_(False)
            groups = [
                {'params': [p for n, p in model.backbone.named_parameters()
                            if p.requires_grad], 'lr': base_lr * 0.1, 'weight_decay': 1e-4},
                {'params': list(model.head.parameters()),
                 'lr': base_lr, 'weight_decay': 5e-3},
            ]
        else:  # mobilenet / fallback
            for name, p in model.backbone.named_parameters():
                if any(tag in name for tag in ['blocks.4', 'blocks.3', 'conv_head']):
                    pass  # train
                else:
                    p.requires_grad_(False)
            groups = [
                {'params': [p for p in model.backbone.parameters() if p.requires_grad],
                 'lr': base_lr * 0.05, 'weight_decay': 1e-3},
                {'params': list(model.head.parameters()),
                 'lr': base_lr, 'weight_decay': 5e-3},
            ]
        return [g for g in groups if len(g['params']) > 0]

    optimizer = torch.optim.AdamW(_param_groups(model, args.lr))

    # Cosine schedule with linear warmup (10% of epochs)
    warmup_steps = max(1, int(args.epochs * 0.10))
    def _lr_lambda(ep):
        if ep < warmup_steps:
            return float(ep + 1) / float(warmup_steps + 1)
        progress = (ep - warmup_steps) / max(1, args.epochs - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    criterion = SSMRegressionLoss(n_modes=ssm.n_modes).to(device)

    best_err   = float('inf')
    best_ep    = 0
    patience   = 50
    out_dir    = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path  = out_dir / f'fold_{fold_name}.pth'

    print(f'  Oversampled: {len(train_rows)} x {oversample} = '
          f'{len(train_rows)*oversample} samples/epoch')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'  Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M total')

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for imgs, targets in dl_train:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        scheduler.step()
        train_loss /= len(ds_train)

        # Validate every 5 epochs
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            val_err, verts = evaluate(model, ssm, val_rows, device)
            elapsed = time.time() - t0

            vert_str = '  '.join(
                f'{n}={np.mean(verts[n]):.0f}' if verts[n] else f'{n}=---'
                for n in CLASS_NAMES)

            improved = val_err < best_err
            if improved:
                best_err = val_err
                best_ep  = ep
                torch.save({'model_state':   model.state_dict(),
                            'val_err_mm':    best_err,
                            'epoch':         ep,
                            'fold':          fold_name,
                            'backbone_name': backbone_name}, ckpt_path)

            marker = ' [BEST]' if improved else ''
            print(f'  ep {ep:3d}/{args.epochs}  loss={train_loss:.4f}  '
                  f'val_err={val_err:.1f}mm  '
                  f'[{vert_str}]{marker}  ({elapsed:.1f}s)')

            # Early stop
            if ep - best_ep >= patience:
                print(f'  Early stop at ep {ep} (best ep={best_ep})')
                break

    print(f'  -> Best  val_err={best_err:.1f}mm  (ep {best_ep})')
    return best_err


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='data/landmark_manifest_combined.csv')
    p.add_argument('--epochs',   type=int,   default=200)
    p.add_argument('--batch',    type=int,   default=8)
    p.add_argument('--lr',       type=float, default=1e-4)
    p.add_argument('--device',   default='cuda')
    p.add_argument('--pretrain', default=None,
                   help='Path to BUU pre-trained checkpoint to fine-tune from.')
    p.add_argument('--ssm',      default=str(SSM_PATH))
    p.add_argument('--only', nargs='+', default=None,
                   help='Only run these patient folds (e.g. --only Swaroop VIJAYALAXMI).')
    p.add_argument('--ckpt_dir', default=None,
                   help='Directory for fold checkpoints (default: checkpoints/ssm_regressor/).')
    p.add_argument('--view', default=None, choices=['AP', 'LAT'],
                   help='Train on AP-only or LAT-only images. Default: all views.')
    p.add_argument('--backbone', default='resnet50',
                   help='timm backbone name. Default: resnet50. '
                        'Options: resnet50, efficientnet_b3, mobilenetv3_small_100.')
    # ── Stage 2 augmentation manifests ───────────────────────────────────────
    p.add_argument('--synthetic_manifest', default=None,
                   help='Path to synthetic DRR manifest (data/synthetic_drr_manifest.csv). '
                        'Added to EVERY training fold — large dataset of free labels from CT.')
    p.add_argument('--converted_manifest', default=None,
                   help='Path to CycleGAN-converted X-ray manifest '
                        '(data/converted_xray_manifest.csv). '
                        'Added to training folds — real X-rays in DRR style.')
    return p.parse_args()


def main():
    args = parse_args()

    # Load SSM
    ssm = SpineSSM.load(args.ssm)
    print(f'SSM loaded: {ssm.n_modes} modes, mean_scale={ssm.mean_scale:.1f}px')

    # Load real rows (NRRD) for LOPO folds, and BUU rows (JPG) as fixed background
    real_rows = []
    buu_rows  = []
    with open(ROOT / args.manifest) as f:
        for row in csv.DictReader(f):
            # View filter: 'AP' matches XRAY-AP, 'LAT' matches XRAY-LAT
            if args.view and args.view not in row.get('view', row.get('set_name', '')):
                continue
            if row['img_path'].endswith('.nrrd'):
                real_rows.append(row)
            elif row['img_path'].endswith('.jpg'):
                buu_rows.append(row)

    # ── Stage 2: load synthetic DRR rows (added to every fold) ───────────────
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
            print(f'Synthetic DRR rows : {len(synthetic_rows)}  '
                  f'(from {syn_path.name})')
        else:
            print(f'WARNING: synthetic_manifest not found: {syn_path}')
    else:
        # Auto-detect: use default path if it exists
        _auto = ROOT / 'data' / 'synthetic_drr_manifest.csv'
        if _auto.exists():
            with open(_auto) as f:
                for row in csv.DictReader(f):
                    if args.view and args.view not in row.get('view', ''):
                        continue
                    synthetic_rows.append(row)
            print(f'Synthetic DRR rows : {len(synthetic_rows)}  '
                  f'(auto-detected {_auto.name})')

    # ── Stage 2: load CycleGAN-converted X-ray rows ───────────────────────────
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
            print(f'Converted X-ray rows: {len(converted_rows)}  '
                  f'(from {conv_path.name})')
        else:
            print(f'WARNING: converted_manifest not found: {conv_path}')
    else:
        # Auto-detect
        _auto = ROOT / 'data' / 'converted_xray_manifest.csv'
        if _auto.exists():
            with open(_auto) as f:
                for row in csv.DictReader(f):
                    if args.view and args.view not in row.get('view', ''):
                        continue
                    converted_rows.append(row)
            print(f'Converted X-ray rows: {len(converted_rows)}  '
                  f'(auto-detected {_auto.name})')

    print(f'Real images : {len(real_rows)}  BUU images: {len(buu_rows)}  '
          f'view={args.view or "all"}')
    rows = real_rows  # LOPO uses real rows only

    # Group by patient
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r['patient']].append(r)
    patients = sorted(by_patient.keys())

    # Background rows added to every training fold
    # Priority: synthetic DRRs first (most data), then BUU, then converted
    background_rows = synthetic_rows + buu_rows + converted_rows

    print(f'\nLOPO folds : {len(patients)} patients')
    print(f'Background : synthetic={len(synthetic_rows)}  '
          f'buu={len(buu_rows)}  converted={len(converted_rows)}  '
          f'total={len(background_rows)}')
    print('=' * 62)

    fold_results = {}
    for patient in patients:
        if args.only and patient not in args.only:
            continue
        val_rows   = by_patient[patient]
        # Real train = all other real patients + ALL background rows
        other_real = [r for p, rs in by_patient.items()
                      if p != patient for r in rs]
        # For converted rows: exclude rows belonging to the val patient
        # (those images were generated from val patient's X-rays — would leak)
        conv_filtered = [r for r in converted_rows
                         if r['patient'] != f'CONV_{patient}']
        train_rows = other_real + synthetic_rows + buu_rows + conv_filtered

        print(f'\n{"-"*62}')
        print(f'  FOLD: val={patient}  '
              f'other_real={len(other_real)}  '
              f'synthetic={len(synthetic_rows)}  '
              f'buu={len(buu_rows)}  '
              f'converted={len(conv_filtered)}  '
              f'total_train={len(train_rows)}')
        print(f'  SSM regression  ({args.epochs} epochs)')
        print(f'{"-"*62}')

        best = train_fold(train_rows, val_rows, ssm, args, fold_name=patient)
        fold_results[patient] = best

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 62)
    print('  LOPO SUMMARY  (mean spine error mm on real val images)')
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
