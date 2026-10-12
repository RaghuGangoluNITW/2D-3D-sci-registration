"""
landmark_model.py
==================
MobileNetV3-Small encoder + lightweight UNet decoder for vertebra heatmap
regression on lumbar X-rays.

Architecture
------------
Encoder : MobileNetV3-Small (pretrained ImageNet, 1st conv adapted to 1-ch input)
          → 4 skip connections at strides 2, 4, 8, 16
          → bottleneck at stride 32  (576 ch)

Decoder : 4× UNet up-blocks (ConvTranspose2d + skip concat + 2×Conv + Dropout2d)
          → final 1×1 conv → NUM_CLASSES channels at 1/4 input resolution

Input  : (B, 1, 512, 512)  float32, normalised to [0, 1]
Output : (B, NUM_CLASSES, 128, 128)  raw logits

~1.5M total params (vs 5.6M for EfficientNet-B0) — much better for small datasets.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from landmark_dataset import NUM_CLASSES

# ── Helper blocks ─────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    """Upsample × 2, concat skip, two ConvBnRelu + optional spatial dropout."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.0):
        super().__init__()
        self.up      = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv    = nn.Sequential(
            ConvBnRelu(in_ch // 2 + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims differ by 1 px (odd input size)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                               align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.drop(self.conv(x))


# ── Main model ────────────────────────────────────────────────────────────────

class LandmarkHeatmapNet(nn.Module):
    """
    MobileNetV3-Small encoder → 4-level UNet decoder → heatmap head.

    Parameters
    ----------
    pretrained : load ImageNet weights for the encoder (recommended)
    num_classes: number of vertebra channels (default NUM_CLASSES = 6)
    """

    # EfficientNet-B0 feature channel counts at each stride level
    # stride 2→16, stride 4→24, stride 8→40, stride 16→112, stride 32→1280
    _ENC_CHS = [16, 24, 40, 112, 1280]

class LandmarkHeatmapNet(nn.Module):
    """
    MobileNetV3-Small encoder → 4-level UNet decoder → heatmap head.

    MobileNetV3-Small has ~1.5M total params vs EfficientNet-B0's 5.6M,
    making it far better suited to fine-tuning on small real-image datasets.

    Parameters
    ----------
    pretrained : load ImageNet weights for the encoder (recommended)
    num_classes: number of vertebra channels (default NUM_CLASSES = 6)
    dropout    : spatial dropout probability in decoder UpBlocks (default 0.3)
    """

    def __init__(self, pretrained: bool = True, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes

        # ── Encoder via timm ──────────────────────────────────────────────────
        self._backbone = timm.create_model(
            'mobilenetv3_small_100',
            pretrained=pretrained,
            features_only=True,
            in_chans=1,
            out_indices=(0, 1, 2, 3, 4),
        )
        enc_chs = self._backbone.feature_info.channels()  # [16, 16, 24, 48, 576]

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up4 = UpBlock(enc_chs[4], enc_chs[3], 128, dropout=dropout)
        self.up3 = UpBlock(128,        enc_chs[2],  64, dropout=dropout)
        self.up2 = UpBlock( 64,        enc_chs[1],  32, dropout=dropout)
        self.up1 = UpBlock( 32,        enc_chs[0],  32, dropout=0.0)

        # ── Per-class independent heads (3×3 conv → 1×1 conv per vertebra) ───
        # Separate heads prevent channel collapse: each class specialises
        # independently rather than sharing a single 1×1 projection.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(16,  1, kernel_size=1),
            )
            for _ in range(num_classes)
        ])
        self._init_heads()

    def _init_heads(self):
        for head in self.heads:
            nn.init.normal_(head[0].weight, std=0.01)
            nn.init.normal_(head[2].weight, std=0.01)
            nn.init.constant_(head[2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 512, 512)
        feats = self._backbone(x)  # list of 5 tensors at strides 2,4,8,16,32
        s0, s1, s2, s3, s4 = feats

        d = self.up4(s4, s3)   # stride 16
        d = self.up3(d,  s2)   # stride 8
        d = self.up2(d,  s1)   # stride 4 → 128×128 for 512 input
        d = self.up1(d,  s0)   # stride 2 → 256×256
        # Downsample back to HMAP_SIZE=128
        d = F.avg_pool2d(d, 2) # → 128×128

        # Each class predicts independently
        return torch.cat([h(d) for h in self.heads], dim=1)  # (B, num_classes, 128, 128)


# ── Loss ──────────────────────────────────────────────────────────────────────

class HeatmapLoss(nn.Module):
    """
    Weighted MSE loss for sparse Gaussian heatmaps.

    The positive region (gt > threshold) is up-weighted by ``pos_weight`` to
    counter the severe class imbalance (only ~3% of pixels are positive).

    loss = mean over positive pixels  * pos_weight
         + mean over negative pixels
    """

    def __init__(self, pos_weight: float = 100.0, threshold: float = 0.01):
        super().__init__()
        self.pos_weight = pos_weight
        self.threshold  = threshold

    def forward(self, logits: torch.Tensor,
                gt:     torch.Tensor,
                valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        pred = logits.sigmoid()
        mse  = (pred - gt) ** 2

        pos = (gt >= self.threshold).float()
        neg = 1.0 - pos

        if valid_mask is not None:
            chan_mask = valid_mask[:, :, None, None].float()
            pos = pos * chan_mask
            neg = neg * chan_mask

        pos_count = pos.sum().clamp(min=1)
        neg_count = neg.sum().clamp(min=1)

        loss = (mse * pos).sum() / pos_count * self.pos_weight \
             + (mse * neg).sum() / neg_count
        return loss


def make_valid_mask(meta_list: list) -> torch.Tensor:
    """
    Build a (B, NUM_CLASSES) bool tensor indicating which vertebra channels
    have a ground-truth annotation in each batch element.
    """
    from landmark_dataset import CLS2IDX
    B = len(meta_list)
    mask = torch.zeros(B, NUM_CLASSES, dtype=torch.bool)
    for b, meta in enumerate(meta_list):
        for lbl in meta['labels']:
            idx = CLS2IDX.get(lbl)
            if idx is not None:
                mask[b, idx] = True
    return mask


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model  = LandmarkHeatmapNet(pretrained=False)
    x      = torch.randn(2, 1, 512, 512)
    logits = model(x)
    print(f'Input : {x.shape}')
    print(f'Output: {logits.shape}')  # expect (2, 6, 128, 128)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params / 1e6:.1f} M')
