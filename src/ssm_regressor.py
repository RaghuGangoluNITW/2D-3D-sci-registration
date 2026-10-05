"""
ssm_regressor.py
================
CNN → SSM parameter regression for fully automatic 2D spine localisation.

Instead of predicting 6×heatmaps (hard, needs dense spatial supervision),
we directly regress the 8 SSM parameters that fully describe the spine shape
in an image:

    params = [b_0, b_1, b_2, b_3, b_4,   ← 5 PCA shape coefficients
              tx_norm, ty_norm,             ← spine centroid in [0,1] image coords
              log_scale_norm]               ← log(scale / mean_scale)

  spine_px = SSM.decode(b, tx*W, ty*H, exp(log_scale_norm)*mean_scale)

Architecture
------------
  Backbone (default: ResNet-50, 25M params, ImageNet pretrained for 1-ch input)
  → Global Average Pool  → (2048,)
  → MLP: 2048 → 512 → 256 → 128 → 8
  with LayerNorm + ReLU + Dropout between layers

  Also supports: mobilenetv3_small_100 (legacy, 576→256→128→8)

Training target normalisation
------------------------------
  b_k          : normalised by sqrt(λ_k) so target ≈ N(0,1)
  tx_norm      : tx_px / W  ∈ (0, 1)
  ty_norm      : ty_px / H  ∈ (0, 1)
  log_scale_n  : log(scale / mean_scale)  ≈ N(0, ~0.3)

Loss
----
  MSE with per-group weights:
    b:     weight = 1.0
    tx/ty: weight = 3.0  (position most important for EPnP init)
    scale: weight = 1.5
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

try:
    import timm
    _TIMM = True
except ImportError:
    _TIMM = False


# ── Model ─────────────────────────────────────────────────────────────────────

class SSMRegressor(nn.Module):
    """
    CNN backbone → GAP → MLP → 8 SSM parameters.

    Parameters
    ----------
    n_modes       : number of SSM shape modes (default 5)
    dropout       : dropout probability in MLP (default 0.3)
    pretrained    : load ImageNet weights for backbone
    backbone_name : timm model name. Default 'resnet50' (2048-dim, 25M params).
                    Legacy: 'mobilenetv3_small_100' (576-dim).
    """

    N_EXTRA = 3   # tx_norm, ty_norm, log_scale_norm

    # Feature dimensions for known backbones
    _FEAT_DIMS = {
        'resnet50':               2048,
        'resnet34':               512,
        'efficientnet_b3':        1536,
        'efficientnet_b2':        1408,
        'mobilenetv3_small_100':  576,
    }

    def __init__(self,
                 n_modes:       int   = 5,
                 dropout:       float = 0.3,
                 pretrained:    bool  = True,
                 backbone_name: str   = 'resnet50'):
        super().__init__()
        self.n_modes       = n_modes
        self.n_params      = n_modes + self.N_EXTRA   # 8
        self.backbone_name = backbone_name

        if not _TIMM:
            raise ImportError('timm is required: pip install timm')

        import os
        backbone_init = os.environ.get('BACKBONE_INIT_PTH', '')

        # If we have a local backbone init file, skip the HuggingFace download
        # by creating the model without pretrained weights first
        _pretrained = pretrained and not backbone_init

        self.backbone = timm.create_model(
            backbone_name,
            pretrained  = _pretrained,
            in_chans    = 1,
            num_classes = 0,
            global_pool = 'avg',
        )

        # Load backbone-only init if available (bypasses HuggingFace download)
        if backbone_init and os.path.exists(backbone_init):
            ckpt = torch.load(backbone_init, map_location='cpu', weights_only=False)
            if 'backbone_state' in ckpt:
                missing, unexpected = self.backbone.load_state_dict(
                    ckpt['backbone_state'], strict=False)
                donor = ckpt.get('donor_fold', backbone_init)
                print(f'  [backbone] Loaded from {donor}  '
                      f'(missing={len(missing)} unexpected={len(unexpected)})')

        # Auto-detect feature dim (safe for any backbone)
        with torch.no_grad():
            _dummy   = torch.zeros(1, 1, 64, 64)
            feat_dim = self.backbone(_dummy).shape[-1]

        # MLP head — wider for larger backbones
        if feat_dim >= 1024:
            # ResNet-50 / EfficientNet-B3: 2048/1536 → 512 → 256 → 128 → n
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2),
                nn.Linear(128, self.n_params),
            )
        else:
            # MobileNetV3-Small (576): 576 → 256 → 128 → n  (legacy)
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2),
                nn.Linear(128, self.n_params),
            )

        # Zero-init final layer → outputs ≈ 0 at start
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W) float32 in [0, 1]

        Returns
        -------
        (B, n_params) raw predictions in normalised parameter space
        """
        feats = self.backbone(x)   # (B, feat_dim)
        return self.head(feats)    # (B, 8)


# ── Loss ──────────────────────────────────────────────────────────────────────

class SSMRegressionLoss(nn.Module):
    """
    Weighted MSE over normalised SSM parameters.

    Weights: shape coefficients=1.0, position=3.0, scale=1.5
    """

    def __init__(self, n_modes: int = 5,
                 w_shape: float = 1.0,
                 w_pos:   float = 3.0,
                 w_scale: float = 1.5):
        super().__init__()
        self.n_modes = n_modes
        weights = (
            [w_shape] * n_modes +   # b_0..b_4
            [w_pos, w_pos] +        # tx, ty
            [w_scale]               # log_scale
        )
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : (B, n_params)
        Returns scalar loss.
        """
        sq = (pred - target) ** 2                   # (B, n_params)
        return (sq * self.weights).mean()


# ── Parameter encoding / decoding ─────────────────────────────────────────────

def encode_params(coords_dict:  Dict[str, list],
                  img_hw:       Tuple[int, int],
                  ssm,
                  ) -> Optional[np.ndarray]:
    """
    Convert ground-truth pixel coords to normalised SSM parameter vector.

    Fits the SSM to all available landmarks (SSM.fit_and_complete),
    then extracts and normalises [b, tx, ty, log_scale].

    Returns None if fewer than 2 landmarks available.
    """
    from spine_ssm import _align, _vec_to_dict, _shape_to_vec, CLASS_NAMES

    H, W = img_hw
    if len(coords_dict) < 2:
        return None

    # Convert coords to numpy
    obs = {k: [float(v[0]), float(v[1])] for k, v in coords_dict.items()}

    # Fit SSM to get (b, centroid, scale) — use internal fitting
    from spine_ssm import _shape_to_vec
    obs_vec  = _shape_to_vec(obs)
    obs_mask = ~np.isnan(obs_vec)

    # Get the aligned GT shape (project onto SSM manifold)
    result = ssm.fit_and_complete(obs, img_hw=(H, W))

    # Reconstruct the normalised shape vector
    full_vec = np.array([[result[n][0], result[n][1]] for n in CLASS_NAMES]).flatten()

    # Compute centroid and scale from reconstructed shape
    valid_pts = full_vec.reshape(-1, 2)
    centroid  = valid_pts.mean(axis=0)
    span      = float(valid_pts[:, 1].max() - valid_pts[:, 1].min()) + 1e-6

    # Infer b by inverting: full_norm = mean_shape + modes @ b
    full_norm = full_vec.copy()
    full_norm[0::2] = (full_norm[0::2] - centroid[0]) / span
    full_norm[1::2] = (full_norm[1::2] - centroid[1]) / span

    # Least-squares b
    modes_k = ssm.modes[:, :ssm.n_modes]
    b_raw   = np.linalg.lstsq(modes_k, full_norm - ssm.mean_shape, rcond=None)[0]

    # Normalise
    b_norm       = b_raw / (np.sqrt(ssm.variances[:ssm.n_modes]) + 1e-9)
    tx_norm      = centroid[0] / W
    ty_norm      = centroid[1] / H
    log_scale_n  = float(np.log(span / (ssm.mean_scale + 1e-9) + 1e-9))

    params = np.concatenate([b_norm, [tx_norm, ty_norm, log_scale_n]]).astype(np.float32)
    return params


def decode_params(params:   np.ndarray,
                  img_hw:   Tuple[int, int],
                  ssm,
                  ) -> Dict[str, np.ndarray]:
    """
    Convert normalised SSM parameter vector to pixel coords dict.

    Parameters
    ----------
    params : (n_modes+3,) normalised SSM parameters from network output
    img_hw : (H, W) of original image
    ssm    : SpineSSM instance
    """
    H, W     = img_hw
    n_modes  = ssm.n_modes
    b_norm   = params[:n_modes]
    tx_norm  = float(params[n_modes])
    ty_norm  = float(params[n_modes + 1])
    log_s_n  = float(params[n_modes + 2])

    # De-normalise
    b       = b_norm * (np.sqrt(ssm.variances[:n_modes]) + 1e-9)
    tx      = tx_norm * W
    ty      = ty_norm * H
    scale   = float(np.exp(log_s_n)) * ssm.mean_scale

    # Reconstruct shape in pixel space
    modes_k  = ssm.modes[:, :n_modes]
    shape_n  = ssm.mean_shape + modes_k @ b
    shape_px = shape_n.copy()
    shape_px[0::2] = shape_px[0::2] * scale + tx
    shape_px[1::2] = shape_px[1::2] * scale + ty

    # Clip to image
    shape_px[0::2] = np.clip(shape_px[0::2], 0, W)
    shape_px[1::2] = np.clip(shape_px[1::2], 0, H)

    from spine_ssm import CLASS_NAMES
    return {name: np.array([shape_px[2*i], shape_px[2*i+1]])
            for i, name in enumerate(CLASS_NAMES)}


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    for bb in ['resnet50', 'mobilenetv3_small_100']:
        model = SSMRegressor(n_modes=5, pretrained=False, backbone_name=bb)
        x     = torch.randn(2, 1, 512, 512)
        out   = model(x)
        total = sum(p.numel() for p in model.parameters())
        print(f'{bb}: output={out.shape}  params={total/1e6:.2f}M')
