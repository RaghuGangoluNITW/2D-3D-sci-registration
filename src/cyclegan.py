"""
cyclegan.py
===========
Lightweight CycleGAN for unpaired X-ray ↔ DRR style transfer.

Designed specifically for RTX 2050 (4 GB VRAM):
  - Generator: U-Net with 32 base filters (not 64) + 4 residual blocks
  - Discriminator: PatchGAN 70×70 with 3 conv layers
  - Mixed precision (torch.autocast) cuts VRAM ~40%
  - Batch size 1 (standard for CycleGAN)

Domain convention
-----------------
  Domain A = real X-ray  (intraoperative C-arm fluoroscopy)
  Domain B = synthetic DRR (rendered from CT)

Generators:
  G_AB : X-ray  → DRR style   (used at inference to convert real X-rays)
  G_BA : DRR    → X-ray style (used for cycle consistency)

Loss terms (standard CycleGAN, Zhu et al. 2017):
  L_adv    : adversarial (LSGAN — more stable than BCE)
  L_cycle  : cycle consistency ‖G_BA(G_AB(a)) - a‖₁ + ‖G_AB(G_BA(b)) - b‖₁
  L_ident  : identity        ‖G_AB(b) - b‖₁     + ‖G_BA(a) - a‖₁
             (prevents hue/contrast shifts when input is already target domain)

Weights: λ_cycle=10, λ_ident=5  (paper defaults)
"""

from __future__ import annotations

import itertools
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _norm(channels: int, norm_type: str = 'instance') -> nn.Module:
    if norm_type == 'instance':
        return nn.InstanceNorm2d(channels, affine=False, track_running_stats=False)
    elif norm_type == 'batch':
        return nn.BatchNorm2d(channels)
    else:
        return nn.Identity()


class ResBlock(nn.Module):
    """Residual block with optional dropout for generator converter."""
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            _norm(channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            _norm(channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Generator (U-Net with skip connections + residual bottleneck)
# ---------------------------------------------------------------------------

class UNetGenerator(nn.Module):
    """
    U-Net style generator for CycleGAN.

    Architecture (ngf=32, 4 residual blocks):
      Encoder:   3× downsampling (stride-2 conv)  [32, 64, 128]
      Bottleneck: 4× ResBlock(128)
      Decoder:   3× upsampling (bilinear + conv)  [64, 32, 1]
      Skip connections from encoder to decoder

    Input / output: (B, 1, H, W) float32, values in [0, 1].
    Output is tanh-activated then mapped to [0, 1].

    Total params at ngf=32: ~1.4M  (fits easily in 4GB)
    """

    def __init__(self, ngf: int = 32, n_res: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.ngf   = ngf
        self.n_res = n_res

        # ── Encoder ──────────────────────────────────────────────────────
        # e1: 1   → ngf      (stride 1, reflection pad)
        # e2: ngf → ngf*2    (stride 2 downsample)
        # e3: ngf*2 → ngf*4  (stride 2 downsample)
        self.e1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, ngf, 7, bias=False),
            _norm(ngf), nn.ReLU(inplace=True),
        )
        self.e2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=False),
            _norm(ngf * 2), nn.ReLU(inplace=True),
        )
        self.e3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=False),
            _norm(ngf * 4), nn.ReLU(inplace=True),
        )

        # ── Bottleneck ────────────────────────────────────────────────────
        self.res = nn.Sequential(*[ResBlock(ngf * 4, dropout) for _ in range(n_res)])

        # ── Decoder ───────────────────────────────────────────────────────
        # Skip connections come from encoder (f1, f2, f3), not bottleneck.
        # After upsample, concat with encoder feature of same spatial size:
        #   d3: upsample(res) → concat with f2 (ngf*2) → ngf*4+ngf*2 → ngf*2
        #   d2: upsample(d3)  → concat with f1 (ngf)   → ngf*2+ngf   → ngf
        #   d1: upsample(d2)  → concat with x  (1)     → ngf+1       → 1
        self.d3 = nn.Sequential(
            nn.Conv2d(ngf * 4 + ngf * 2, ngf * 2, 3, padding=1, bias=False),
            _norm(ngf * 2), nn.ReLU(inplace=True),
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(ngf * 2 + ngf, ngf, 3, padding=1, bias=False),
            _norm(ngf), nn.ReLU(inplace=True),
        )
        self.d1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf + 1, 1, 7),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder (save feature maps for skip connections)
        f1 = self.e1(x)                         # (B, ngf,   H,   W)
        f2 = self.e2(f1)                         # (B, ngf*2, H/2, W/2)
        f3 = self.e3(f2)                         # (B, ngf*4, H/4, W/4)

        # Bottleneck
        r  = self.res(f3)                        # (B, ngf*4, H/4, W/4)

        # Decoder with skip connections (bilinear upsample + concat + conv)
        # Skip connections: encoder features f1/f2 (NOT f3 which is bottleneck input)
        u3 = F.interpolate(r,  size=f2.shape[2:], mode='bilinear',
                            align_corners=False)
        u3 = self.d3(torch.cat([u3, f2], dim=1)) # (B, ngf*2, H/2, W/2)

        u2 = F.interpolate(u3, size=f1.shape[2:], mode='bilinear',
                            align_corners=False)
        u2 = self.d2(torch.cat([u2, f1], dim=1)) # (B, ngf,   H,   W)

        u1 = F.interpolate(u2, size=x.shape[2:],  mode='bilinear',
                            align_corners=False)
        out = self.d1(torch.cat([u1, x], dim=1))  # (B, 1,     H,   W)

        # Map tanh output [-1,1] → [0,1]
        return (out + 1.0) * 0.5


# ---------------------------------------------------------------------------
# Discriminator (PatchGAN)
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    4-layer architecture (ndf=32 base filters):
      Conv(1→32, k4s2p1)  → LeakyReLU
      Conv(32→64, k4s2p1) → IN → LeakyReLU
      Conv(64→128,k4s2p1) → IN → LeakyReLU
      Conv(128→1, k4s1p1)         [patch score map]

    Input: (B, 1, H, W) float32
    Output: (B, 1, H', W') real/fake score map — no sigmoid (LSGAN uses MSE)
    """

    def __init__(self, ndf: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1 — no norm
            nn.Conv2d(1, ndf,      4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(ndf,   ndf*2, 4, stride=2, padding=1, bias=False),
            _norm(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1, bias=False),
            _norm(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            # Output patch score
            nn.Conv2d(ndf*4, 1,     4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer  (stabilises GAN training)
# ---------------------------------------------------------------------------

class ImageBuffer:
    """
    Stores a history of 50 generated images and returns a random mix of
    history and newly generated images when sampled.  Prevents the
    discriminator from overfitting to the most recent generator outputs.
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        """
        For each image in the batch, randomly return either the new image or
        a stored one, then store the new image.
        """
        result = []
        for img in images.detach().unbind(0):
            img = img.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(img)
                result.append(img)
            else:
                if torch.rand(1).item() > 0.5:
                    idx = int(torch.randint(len(self.buffer), (1,)).item())
                    result.append(self.buffer[idx].clone())
                    self.buffer[idx] = img
                else:
                    result.append(img)
        return torch.cat(result, dim=0)


# ---------------------------------------------------------------------------
# Full CycleGAN model
# ---------------------------------------------------------------------------

class CycleGAN(nn.Module):
    """
    Full CycleGAN (A=X-ray, B=DRR).

    Wraps both generators and both discriminators.
    Provides a single train_step() method for the main training loop.

    Parameters
    ----------
    ngf     : generator base filters (default 32 for 4GB VRAM)
    ndf     : discriminator base filters (default 32)
    n_res   : residual blocks in generator bottleneck (default 4)
    lambda_cycle : cycle consistency weight (default 10)
    lambda_ident : identity loss weight (default 5)
    """

    def __init__(self,
                 ngf:           int   = 32,
                 ndf:           int   = 32,
                 n_res:         int   = 4,
                 lambda_cycle:  float = 10.0,
                 lambda_ident:  float = 5.0,
                 dropout:       float = 0.0):
        super().__init__()

        self.lambda_cycle = lambda_cycle
        self.lambda_ident = lambda_ident

        # Generators
        self.G_AB = UNetGenerator(ngf, n_res, dropout)   # X-ray → DRR
        self.G_BA = UNetGenerator(ngf, n_res, dropout)   # DRR   → X-ray

        # Discriminators
        self.D_A  = PatchDiscriminator(ndf)   # real vs fake X-ray
        self.D_B  = PatchDiscriminator(ndf)   # real vs fake DRR

        # Replay buffers
        self.buf_A = ImageBuffer()
        self.buf_B = ImageBuffer()

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for conv layers, orthogonal for BN/IN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def xray_to_drr(self, xray: torch.Tensor) -> torch.Tensor:
        """Convert X-ray image(s) to DRR style. Input/output: [0,1]."""
        return self.G_AB(xray)

    @torch.no_grad()
    def drr_to_xray(self, drr: torch.Tensor) -> torch.Tensor:
        """Convert DRR image(s) to X-ray style. Input/output: [0,1]."""
        return self.G_BA(drr)

    # ── Loss helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _lsgan_loss_real(pred: torch.Tensor) -> torch.Tensor:
        """LSGAN: (D(real) - 1)² / 2"""
        return F.mse_loss(pred, torch.ones_like(pred))

    @staticmethod
    def _lsgan_loss_fake(pred: torch.Tensor) -> torch.Tensor:
        """LSGAN: D(fake)² / 2"""
        return F.mse_loss(pred, torch.zeros_like(pred))

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        opt_G:  torch.optim.Optimizer,
        opt_D:  torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        """
        One training step: update generators then discriminators.

        Parameters
        ----------
        real_A : (B, 1, H, W) real X-ray batch
        real_B : (B, 1, H, W) real DRR batch
        opt_G  : optimizer for G_AB + G_BA
        opt_D  : optimizer for D_A + D_B
        scaler : GradScaler for mixed precision (pass None to disable)

        Returns
        -------
        dict of scalar loss values for logging
        """

        # ── Step 1: Update Generators ─────────────────────────────────────
        # Freeze discriminators during generator update
        for p in self.D_A.parameters(): p.requires_grad_(False)
        for p in self.D_B.parameters(): p.requires_grad_(False)

        opt_G.zero_grad(set_to_none=True)

        def _forward_G():
            # Forward pass through both generators
            fake_B = self.G_AB(real_A)   # A → B (X-ray → DRR)
            fake_A = self.G_BA(real_B)   # B → A (DRR → X-ray)

            # Cycle
            rec_A  = self.G_BA(fake_B)   # A → B → A
            rec_B  = self.G_AB(fake_A)   # B → A → B

            # Identity
            idt_A  = self.G_BA(real_A)   # G_BA(A) should ≈ A
            idt_B  = self.G_AB(real_B)   # G_AB(B) should ≈ B

            # Adversarial: generators want discriminators to output 1 (real)
            l_adv_AB = self._lsgan_loss_real(self.D_B(fake_B))
            l_adv_BA = self._lsgan_loss_real(self.D_A(fake_A))

            # Cycle consistency
            l_cyc_A = F.l1_loss(rec_A, real_A)
            l_cyc_B = F.l1_loss(rec_B, real_B)
            l_cycle = self.lambda_cycle * (l_cyc_A + l_cyc_B)

            # Identity
            l_idt_A = F.l1_loss(idt_A, real_A)
            l_idt_B = F.l1_loss(idt_B, real_B)
            l_ident = self.lambda_ident * (l_idt_A + l_idt_B)

            l_G = l_adv_AB + l_adv_BA + l_cycle + l_ident
            return l_G, fake_A, fake_B, {
                'G/adv_AB':  l_adv_AB.item(),
                'G/adv_BA':  l_adv_BA.item(),
                'G/cycle':   l_cycle.item(),
                'G/ident':   l_ident.item(),
                'G/total':   l_G.item(),
            }

        if scaler is not None:
            with torch.autocast('cuda'):
                l_G, fake_A, fake_B, g_losses = _forward_G()
            scaler.scale(l_G).backward()
            scaler.unscale_(opt_G)
            nn.utils.clip_grad_norm_(
                list(self.G_AB.parameters()) + list(self.G_BA.parameters()), 1.0)
            scaler.step(opt_G)
        else:
            l_G, fake_A, fake_B, g_losses = _forward_G()
            l_G.backward()
            nn.utils.clip_grad_norm_(
                list(self.G_AB.parameters()) + list(self.G_BA.parameters()), 1.0)
            opt_G.step()

        # ── Step 2: Update Discriminators ────────────────────────────────
        for p in self.D_A.parameters(): p.requires_grad_(True)
        for p in self.D_B.parameters(): p.requires_grad_(True)

        opt_D.zero_grad(set_to_none=True)

        # Pull from replay buffer (detached fakes)
        fake_A_buf = self.buf_A.push_and_pop(fake_A.detach())
        fake_B_buf = self.buf_B.push_and_pop(fake_B.detach())

        def _forward_D():
            # D_A: distinguish real A from fake A
            l_DA_real = self._lsgan_loss_real(self.D_A(real_A))
            l_DA_fake = self._lsgan_loss_fake(self.D_A(fake_A_buf))
            l_DA = (l_DA_real + l_DA_fake) * 0.5

            # D_B: distinguish real B from fake B
            l_DB_real = self._lsgan_loss_real(self.D_B(real_B))
            l_DB_fake = self._lsgan_loss_fake(self.D_B(fake_B_buf))
            l_DB = (l_DB_real + l_DB_fake) * 0.5

            l_D = l_DA + l_DB
            return l_D, {
                'D/DA': l_DA.item(),
                'D/DB': l_DB.item(),
                'D/total': l_D.item(),
            }

        if scaler is not None:
            with torch.autocast('cuda'):
                l_D, d_losses = _forward_D()
            scaler.scale(l_D).backward()
            scaler.unscale_(opt_D)
            nn.utils.clip_grad_norm_(
                list(self.D_A.parameters()) + list(self.D_B.parameters()), 1.0)
            scaler.step(opt_D)
            scaler.update()
        else:
            l_D, d_losses = _forward_D()
            l_D.backward()
            nn.utils.clip_grad_norm_(
                list(self.D_A.parameters()) + list(self.D_B.parameters()), 1.0)
            opt_D.step()

        return {**g_losses, **d_losses}

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save(self, path: str | Path,
             epoch: int = 0,
             opt_G: Optional[torch.optim.Optimizer] = None,
             opt_D: Optional[torch.optim.Optimizer] = None):
        ckpt = {
            'epoch':      epoch,
            'G_AB':       self.G_AB.state_dict(),
            'G_BA':       self.G_BA.state_dict(),
            'D_A':        self.D_A.state_dict(),
            'D_B':        self.D_B.state_dict(),
            'ngf':        self.ngf,
        }
        if opt_G: ckpt['opt_G'] = opt_G.state_dict()
        if opt_D: ckpt['opt_D'] = opt_D.state_dict()
        torch.save(ckpt, str(path))

    @classmethod
    def load(cls, path: str | Path,
             device: str = 'cpu') -> 'CycleGAN':
        ckpt   = torch.load(str(path), map_location=device, weights_only=False)
        ngf    = ckpt.get('ngf', 32)
        model  = cls(ngf=ngf)
        model.G_AB.load_state_dict(ckpt['G_AB'])
        model.G_BA.load_state_dict(ckpt['G_BA'])
        model.D_A.load_state_dict(ckpt['D_A'])
        model.D_B.load_state_dict(ckpt['D_B'])
        return model.to(device)

    @property
    def ngf(self) -> int:
        return self.G_AB.ngf


# ---------------------------------------------------------------------------
# Quick parameter count check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import torch

    model = CycleGAN(ngf=32, ndf=32, n_res=4)

    def _count(m): return sum(p.numel() for p in m.parameters()) / 1e6

    print(f'G_AB params : {_count(model.G_AB):.2f}M')
    print(f'G_BA params : {_count(model.G_BA):.2f}M')
    print(f'D_A  params : {_count(model.D_A):.2f}M')
    print(f'D_B  params : {_count(model.D_B):.2f}M')
    print(f'Total       : {_count(model):.2f}M')

    # VRAM estimate at 512×512 batch=1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = model.to(device)
    x = torch.randn(1, 1, 512, 512, device=device)
    with torch.autocast('cuda' if device == 'cuda' else 'cpu'):
        y = model.G_AB(x)
    print(f'G_AB forward OK: {x.shape} → {y.shape}')
    if device == 'cuda':
        print(f'VRAM after forward: {torch.cuda.memory_allocated()/1024**2:.0f} MB')
