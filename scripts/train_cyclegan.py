
#!/usr/bin/env python3
"""
train_cyclegan.py
=================
Stage 2 of the automatic PDE <2mm pipeline.

Trains an unpaired X-ray ↔ DRR CycleGAN so that real intraoperative
X-rays can be converted to the DRR domain before landmark detection.

Domain A = real X-ray   (69 labeled images from data/patients/ +
                         ~700 unlabeled real X-ray/C-arm frames from
                         data/extra_xrays/, see extract_extra_xrays.py)
Domain B = synthetic DRR (11,800+ images from data/synthetic_drrs/)

The CycleGAN learns the *style* mapping between the two domains without
needing paired (same patient, same pose) images.

After training, G_AB (X-ray → DRR) is used to convert real X-rays before
feeding them to the SSM regressor, eliminating the domain gap.

Usage
-----
python scripts/train_cyclegan.py                     # default settings
python scripts/train_cyclegan.py --epochs 100 --view AP
python scripts/train_cyclegan.py --resume checkpoints/cyclegan/latest.pth

Output
------
checkpoints/cyclegan/
    latest.pth            # saved every epoch
    best.pth              # best generator loss
    epoch_NNN.pth         # periodic save every 10 epochs
logs/cyclegan_train.csv   # loss log
"""

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from cyclegan import CycleGAN
from landmark_dataset import IMG_SIZE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CKPT_DIR  = ROOT / 'checkpoints' / 'cyclegan'
LOG_DIR   = ROOT / 'logs'
XRAY_ROOT = ROOT / 'data' / 'patients'
DRR_ROOT  = Path(os.environ.get('DRR_CACHE', str(ROOT / 'data' / 'synthetic_drrs')))
EXTRA_XRAY_ROOT     = ROOT / 'data' / 'extra_xrays'
EXTRA_XRAY_MANIFEST = ROOT / 'data' / 'extra_xrays_manifest.csv'

# Training image size — use 256 for speed on 4GB GPU
TRAIN_SIZE = 256

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _clahe(img_f32: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a float32 [0,1] grayscale image."""
    u8 = (np.clip(img_f32, 0, 1) * 255).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
    return cl.astype(np.float32) / 255.0


def _load_nrrd(path: str) -> np.ndarray:
    """Load NRRD as float32 normalised [0,1]."""
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).squeeze().astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def _augment(img: np.ndarray, size: int) -> np.ndarray:
    """Fast augmentation: crop, flip, gamma, noise."""
    rng = np.random.default_rng()
    H, W = img.shape

    # Random crop to size
    if H > size and W > size:
        y0 = rng.integers(0, H - size)
        x0 = rng.integers(0, W - size)
        img = img[y0:y0+size, x0:x0+size]
    else:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    # Horizontal flip
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()

    # Gamma
    gamma = rng.uniform(0.7, 1.4)
    img   = np.clip(img ** gamma, 0, 1)

    # Brightness / contrast
    alpha = rng.uniform(0.8, 1.2)
    beta  = rng.uniform(-0.1, 0.1)
    img   = np.clip(alpha * img + beta, 0, 1)

    # Gaussian noise
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0, 0.03, img.shape).astype(np.float32), 0, 1)

    return img.astype(np.float32)


class XRayDataset(Dataset):
    """
    Domain A: real X-ray images from all patients.
    Loads all NRRD X-ray files (labeled, 69 images), applies CLAHE,
    augments on the fly. Optionally also loads the ~700 extra unlabeled
    real X-ray/C-arm frames from data/extra_xrays/ (see
    extract_extra_xrays.py) -- CycleGAN only needs unpaired style
    examples, not landmark labels, so these are safe to add here even
    though they cannot be used by the SSM regressor.
    """

    def __init__(self, xray_root: Path, view: str = 'both',
                 size: int = TRAIN_SIZE, augment: bool = True,
                 extra_root: Path = None, extra_manifest: Path = None):
        self.size    = size
        self.augment = augment
        self.paths   = []       # NRRD paths (labeled patients)
        self.extra_paths = []   # PNG paths (unlabeled extra X-rays)

        view_dirs = []
        if view in ('both', 'AP'):
            view_dirs.append('XRAY-AP')
        if view in ('both', 'LAT'):
            view_dirs.append('XRAY-LAT')

        for patient_dir in sorted(xray_root.iterdir()):
            if not patient_dir.is_dir():
                continue
            xray_label = patient_dir / 'XRAY Label'
            if not xray_label.exists():
                continue
            for vd in view_dirs:
                vdir = xray_label / vd
                if not vdir.exists():
                    continue
                for set_dir in sorted(vdir.iterdir()):
                    for nrrd in sorted(set_dir.glob('*.nrrd')):
                        self.paths.append(nrrd)

        print(f'  XRayDataset: {len(self.paths)} labeled real X-ray images  (view={view})')

        # Extra unlabeled real X-ray/C-arm frames (no view split -- raw
        # clinical frames, mixed AP/LAT/oblique; fine for CycleGAN style
        # matching which is unpaired and view-agnostic).
        if extra_manifest is not None and Path(extra_manifest).exists():
            with open(extra_manifest, newline='') as f:
                for row in csv.DictReader(f):
                    p = ROOT / row['img_path']
                    if p.exists():
                        self.extra_paths.append(p)
        elif extra_root is not None and Path(extra_root).exists():
            for src_dir in sorted(Path(extra_root).iterdir()):
                if src_dir.is_dir():
                    self.extra_paths.extend(sorted(src_dir.glob('*.png')))

        print(f'  XRayDataset: {len(self.extra_paths)} extra unlabeled X-ray frames')

        total = len(self.paths) + len(self.extra_paths)
        print(f'  XRayDataset: {total} total real X-ray images  (view={view})')

        # Pre-load all X-rays into RAM (eliminates repeated OneDrive reads)
        print(f'  Pre-loading {total} X-rays into RAM...', flush=True)
        self._cache = []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for p in self.paths:
            img = _load_nrrd(str(p))
            img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            img_cl = clahe.apply(img_u8).astype(np.float32) / 255.0
            img_rs = cv2.resize(img_cl, (size, size), interpolation=cv2.INTER_LINEAR)
            self._cache.append(img_rs)
        for p in self.extra_paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_u8 = clahe.apply(img).astype(np.float32) / 255.0
            img_rs = cv2.resize(img_u8, (size, size), interpolation=cv2.INTER_LINEAR)
            self._cache.append(img_rs)
        print(f'  X-ray cache ready ({len(self._cache)} images)', flush=True)

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self._cache[idx % len(self._cache)].copy()
        if self.augment:
            img = _augment(img, self.size)
        return torch.from_numpy(img).unsqueeze(0)


class DRRDataset(Dataset):
    """
    Domain B: synthetic DRR PNGs from data/synthetic_drrs/.
    """

    # VIJAYALAXMI has an unresolved low-bone-contrast rendering bug
    # (corrupted-looking curved pattern, not the moiré/jitter issue) --
    # always excluded regardless of what's left over in any local disk
    # cache, so a stale/partial cache copy can't leak bad DRRs in.
    EXCLUDED_PATIENTS = {'VIJAYALAXMI'}

    def __init__(self, drr_root: Path, view: str = 'both',
                 size: int = TRAIN_SIZE, augment: bool = True):
        self.size    = size
        self.augment = augment
        self.paths   = []

        view_dirs = []
        if view in ('both', 'AP'):
            view_dirs.append('AP')
        if view in ('both', 'LAT'):
            view_dirs.append('LAT')

        for patient_dir in sorted(drr_root.iterdir()):
            if not patient_dir.is_dir():
                continue
            if patient_dir.name in self.EXCLUDED_PATIENTS:
                continue
            for vd in view_dirs:
                vdir = patient_dir / vd
                if not vdir.exists():
                    continue
                for png in sorted(vdir.glob('*.png')):
                    self.paths.append(png)

        print(f'  DRRDataset:  {len(self.paths)} synthetic DRRs      (view={view})')

        # Pre-load all DRRs into RAM (745MB fits in 31GB RAM, eliminates disk I/O)
        print(f'  Pre-loading {len(self.paths)} DRRs into RAM...', flush=True)
        self._cache = []
        for i, p in enumerate(self.paths):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((size, size), dtype=np.uint8)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
            self._cache.append(img.astype(np.float32) / 255.0)
            if (i + 1) % 1000 == 0:
                print(f'    {i+1}/{len(self.paths)} loaded...', flush=True)
        print(f'  DRR cache ready ({len(self._cache)} images)', flush=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self._cache[idx].copy()
        if self.augment:
            img = _augment(img, self.size)
        return torch.from_numpy(img).unsqueeze(0)


class InfiniteSampler:
    """Wraps a DataLoader to yield batches indefinitely (for unequal domain sizes)."""
    def __init__(self, loader):
        self._loader = loader
        self._iter   = iter(loader)

    def __next__(self) -> torch.Tensor:
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            return next(self._iter)


# ---------------------------------------------------------------------------
# Visualisation: save a grid of real/fake/cycle images each epoch
# ---------------------------------------------------------------------------

def _save_sample_grid(model: CycleGAN,
                      real_A: torch.Tensor,
                      real_B: torch.Tensor,
                      path: Path,
                      device: str):
    """Save a 2×3 grid: [real_A, fake_B, rec_A] / [real_B, fake_A, rec_B]."""
    model.eval()
    with torch.no_grad():
        a = real_A[:1].to(device)
        b = real_B[:1].to(device)
        fake_b = model.G_AB(a)
        rec_a  = model.G_BA(fake_b)
        fake_a = model.G_BA(b)
        rec_b  = model.G_AB(fake_a)

    def _t2np(t):
        return (t[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    row1 = np.hstack([_t2np(a), _t2np(fake_b), _t2np(rec_a)])
    row2 = np.hstack([_t2np(b), _t2np(fake_a), _t2np(rec_b)])
    grid = np.vstack([row1, row2])

    # Add labels
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
    H = grid.shape[0] // 2
    W = grid.shape[1] // 3
    for i, label in enumerate(['Real A', 'Fake B', 'Rec A',
                                'Real B', 'Fake A', 'Rec B']):
        row = (i // 3) * H
        col = (i %  3) * W
        cv2.putText(grid_bgr, label, (col + 5, row + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid_bgr)
    model.train()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = args.device
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    samples_dir = ROOT / 'results' / 'cyclegan_samples'
    samples_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────
    extra_manifest = None if args.no_extra_xrays else EXTRA_XRAY_MANIFEST
    ds_A = XRayDataset(XRAY_ROOT, view=args.view, size=args.size,
                       extra_manifest=extra_manifest)
    ds_B = DRRDataset(DRR_ROOT,  view=args.view, size=args.size)

    if len(ds_A) == 0:
        print('ERROR: No X-ray images found. Check data/patients/ structure.')
        return
    if len(ds_B) == 0:
        print('ERROR: No DRR images found. Run generate_synthetic_drrs.py first.')
        return

    dl_A = DataLoader(ds_A, batch_size=1, shuffle=True,
                      num_workers=0, pin_memory=True, drop_last=True)
    dl_B = DataLoader(ds_B, batch_size=1, shuffle=True,
                      num_workers=0, pin_memory=True, drop_last=True)

    # Wrap the smaller domain with infinite sampler so both always have data
    sampler_A = InfiniteSampler(dl_A)
    sampler_B = InfiniteSampler(dl_B)

    # Number of steps per epoch — cap it so training isn't too slow
    # Default: 500 steps/epoch (cycles through X-rays ~7x, DRRs sampled randomly)
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch else min(500, max(len(dl_A), len(dl_B)))
    print(f'  Steps/epoch: {steps_per_epoch}  '
          f'(A={len(dl_A)}, B={len(dl_B)})')

    # ── Model ─────────────────────────────────────────────────────────────
    model  = CycleGAN(ngf=args.ngf, ndf=args.ndf, n_res=args.n_res,
                      lambda_cycle=args.lambda_cycle,
                      lambda_ident=args.lambda_ident).to(device)

    # Separate optimizers for G and D (standard CycleGAN setup)
    opt_G = torch.optim.Adam(
        list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
        lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()),
        lr=args.lr, betas=(0.5, 0.999))

    # Linear LR decay: keep LR flat for first half, then decay to 0
    def _lr_lambda(epoch):
        decay_start = args.epochs // 2
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / max(1, args.epochs - decay_start))

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, _lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, _lr_lambda)

    # Mixed precision
    use_amp = (device == 'cuda')
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    start_epoch = 1
    best_g_loss = float('inf')

    # ── Resume ────────────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.G_AB.load_state_dict(ckpt['G_AB'])
        model.G_BA.load_state_dict(ckpt['G_BA'])
        model.D_A.load_state_dict(ckpt['D_A'])
        model.D_B.load_state_dict(ckpt['D_B'])
        if 'opt_G' in ckpt: opt_G.load_state_dict(ckpt['opt_G'])
        if 'opt_D' in ckpt: opt_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f'  Resumed from {args.resume}  (epoch {start_epoch})')

    # ── Log file ──────────────────────────────────────────────────────────
    log_path  = LOG_DIR / 'cyclegan_train.csv'
    log_fields = ['epoch', 'G_total', 'G_adv_AB', 'G_adv_BA',
                  'G_cycle', 'G_ident', 'D_total', 'lr', 'time_s']
    log_mode  = 'a' if args.resume else 'w'
    log_file  = open(log_path, log_mode, newline='')
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if log_mode == 'w':
        log_writer.writeheader()

    print(f'\n{"="*60}')
    print(f'  CycleGAN Training')
    print(f'  Size     : {args.size}x{args.size}')
    print(f'  ngf/ndf  : {args.ngf}/{args.ndf}  n_res={args.n_res}')
    print(f'  lambda_cycle : {args.lambda_cycle}  lambda_ident={args.lambda_ident}')
    print(f'  AMP      : {use_amp}')
    print(f'  Device   : {device}')
    print(f'{"="*60}')

    # Keep fixed samples for visualisation
    _sample_A = next(sampler_A).to(device)
    _sample_B = next(sampler_B).to(device)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        epoch_losses = {k: 0.0 for k in
                        ['G/total','G/adv_AB','G/adv_BA',
                         'G/cycle','G/ident','D/total']}

        model.train()
        for step in range(steps_per_epoch):
            real_A = next(sampler_A).to(device)
            real_B = next(sampler_B).to(device)

            losses = model.train_step(real_A, real_B, opt_G, opt_D, scaler)

            for k in epoch_losses:
                epoch_losses[k] += losses.get(k, 0.0)

        # Average over steps
        for k in epoch_losses:
            epoch_losses[k] /= steps_per_epoch

        sched_G.step()
        sched_D.step()

        elapsed = time.time() - t0
        lr_now  = opt_G.param_groups[0]['lr']

        print(f'  ep {epoch:3d}/{args.epochs}  '
              f'G={epoch_losses["G/total"]:.3f}  '
              f'(adv={epoch_losses["G/adv_AB"]:.3f}+{epoch_losses["G/adv_BA"]:.3f}  '
              f'cyc={epoch_losses["G/cycle"]:.3f}  '
              f'idt={epoch_losses["G/ident"]:.3f})  '
              f'D={epoch_losses["D/total"]:.3f}  '
              f'lr={lr_now:.5f}  ({elapsed:.0f}s)')

        # Log
        log_writer.writerow({
            'epoch':    epoch,
            'G_total':  round(epoch_losses['G/total'], 5),
            'G_adv_AB': round(epoch_losses['G/adv_AB'], 5),
            'G_adv_BA': round(epoch_losses['G/adv_BA'], 5),
            'G_cycle':  round(epoch_losses['G/cycle'], 5),
            'G_ident':  round(epoch_losses['G/ident'], 5),
            'D_total':  round(epoch_losses['D/total'], 5),
            'lr':       round(lr_now, 7),
            'time_s':   round(elapsed, 1),
        })
        log_file.flush()

        # Save latest
        model.save(CKPT_DIR / 'latest.pth', epoch, opt_G, opt_D)

        # Save best
        if epoch_losses['G/total'] < best_g_loss:
            best_g_loss = epoch_losses['G/total']
            model.save(CKPT_DIR / 'best.pth', epoch)
            print(f'    -> New best G_loss={best_g_loss:.4f}  saved best.pth')

        # Periodic save every 10 epochs
        if epoch % 10 == 0:
            model.save(CKPT_DIR / f'epoch_{epoch:03d}.pth', epoch)

        # Save visualisation sample every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            _save_sample_grid(
                model, _sample_A, _sample_B,
                samples_dir / f'epoch_{epoch:03d}.png',
                device)

    log_file.close()
    print(f'\n  Training complete.')
    print(f'  Best G loss : {best_g_loss:.4f}')
    print(f'  Checkpoints : {CKPT_DIR}')
    print(f'  Samples     : {samples_dir}')
    print(f'  Log         : {log_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',        type=int,   default=150,
                   help='Total epochs (default 150; first half constant LR, '
                        'second half linear decay)')
    p.add_argument('--lr',            type=float, default=2e-4,
                   help='Initial learning rate (default 2e-4)')
    p.add_argument('--size',          type=int,   default=TRAIN_SIZE,
                   help=f'Training image size (default {TRAIN_SIZE})')
    p.add_argument('--ngf',           type=int,   default=32,
                   help='Generator base filters (default 32, 4GB safe)')
    p.add_argument('--ndf',           type=int,   default=32,
                   help='Discriminator base filters (default 32)')
    p.add_argument('--n_res',         type=int,   default=4,
                   help='Residual blocks in generator (default 4)')
    p.add_argument('--lambda_cycle',  type=float, default=10.0)
    p.add_argument('--lambda_ident',  type=float, default=5.0)
    p.add_argument('--view',          default='both',
                   choices=['AP', 'LAT', 'both'],
                   help='Train on AP, LAT, or both views (default: both)')
    p.add_argument('--device',        default='cuda')
    p.add_argument('--resume',        default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--steps_per_epoch', type=int, default=None,
                   help='Max steps per epoch (default: min(500, dataset_size)). '
                        'Keeps training time predictable regardless of dataset size.')
    p.add_argument('--no_extra_xrays', action='store_true',
                   help='Disable loading the ~700 extra unlabeled real X-ray/C-arm '
                        'frames from data/extra_xrays_manifest.csv (default: included).')

                    
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
