"""
pose_regressor.py
==================
Patient-specific 6-DOF pose regression -- a NEW, independent approach to
automatic 2D/3D registration, built to sidestep a proven failure mode of
the CNN -> 2D-landmarks -> EPnP pipeline (ssm_regressor.py / landmark_
detector.py).

Why this exists
----------------
This session's earlier work (SSM-parameter regression, then heatmap-based
landmark detection) both funnel through the same final step: solve EPnP
from a handful of predicted 2D vertebra centroids to get a 3D pose. A
direct noise-injection test proved that step is fundamentally unstable for
this anatomy: the D12/L1-L5 centroids sit nearly collinear along the
spine's own axis, so EPnP is ill-conditioned there -- even ~1mm of 2D
landmark noise (better than either trained detector achieved) produced a
median 3D pose error of 15 degrees rotation / 340mm translation, far
outside CMA-ES's real capture range. More landmark accuracy cannot fix an
ill-conditioned linear system.

This module skips landmarks and EPnP entirely. Instead:

    X-ray-like image  ->  CNN  ->  full 6-DOF pose directly (rotation +
                                    translation, see parameterisation below)

There is no PnP step to be unstable, because no 2D point correspondences
are used at inference time at all.

Why this is feasible per-patient (and wasn't for cross-patient LOPO)
----------------------------------------------------------------------
Direct pose regression from a single 2D image is a much easier learning
problem when the network only ever has to recognise ONE known, fixed 3D
shape (this patient's own CT) from many angles, instead of generalising
across every patient's anatomy. We can render unlimited labeled training
images for that one patient (each synthetic DRR already carries its exact
generating pose "for free" -- see generate_synthetic_drrs.py, which
already renders these images but previously discarded the pose label,
keeping only the projected 2D landmarks).

Clinical workflow this targets (per user/clinical-guide framing)
-------------------------------------------------------------------
  1. Patient's pre-op CT is available before surgery (already required).
  2. BEFORE the operation: generate a large synthetic-DRR training set from
     just this patient's CT (many simulated C-arm poses, AP + LAT, with
     and without simulated instrument occlusion), train a small
     patient-specific PoseRegressor CNN.
  3. DURING the operation: for every X-ray the surgeon takes (repeatedly,
     not just once), feed it to this patient's trained model to get a
     6-DOF pose estimate directly -- no landmarks, no EPnP. Hand that pose
     to the existing CMA-ES refinement stage (optimizer.py, already
     validated at 1.4mm PDE from a good starting pose) as the seed.

This module does NOT attempt cross-patient generalisation and is not a
replacement for ssm_regressor.py / landmark_detector.py in that role.

Pose parameterisation (REVISED -- second iteration)
-------------------------------------------------------
Two earlier parameterisations were tried and rejected this session:

  1. A small Euler-angle PERTURBATION away from one fixed anatomical
     reference pose. Broke down because real acquisitions of the SAME
     patient during the SAME operation differ by up to ~150 degrees /
     2000+mm from any single reference -- far too large for a "small
     delta" representation.
  2. A spherical "orbit around the patient, always look at the spine
     centroid" model (azimuth/elevation/roll/distance). This was found
     to be geometrically WRONG, not just under-ranged: a direct
     numerical fit (allowing the look-at point, angles, AND distance to
     vary freely) still could not reproduce the real camera poses to
     better than ~130-220px reprojection error. The reason: a C-arm is
     aimed at whatever vertebral level the surgeon needs, which shifts
     between shots -- confirmed directly by checking which landmarks
     fall inside vs. outside the image frame per real projection (e.g.
     D12/L1/L2 project OFF-FRAME entirely in the AP sets; only L3/L4/L5
     are ever visible). There is no single fixed 3D point that all real
     acquisitions are centred on, so a "look-at-one-point" model cannot
     represent the real data regardless of how its parameters are tuned.

This version predicts the FULL, UNCONSTRAINED 6-DOF pose directly, with
no look-at assumption and no small-delta assumption:

  - Rotation: the 6D continuous rotation representation of Zhou et al.,
    "On the Continuity of Rotation Representations in Neural Networks"
    (CVPR 2019) -- the network predicts two 3-vectors, which are
    Gram-Schmidt orthonormalised into a full, always-valid rotation
    matrix (via rotation_6d_to_matrix / matrix_to_rotation_6d below).
    This avoids the singularities and discontinuities of Euler angles
    or axis-angle representations, and unlike a 3x3 matrix or
    quaternion, a raw 6-vector requires no unit-norm constraint on the
    network's output to always decode to a valid rotation.
  - Translation: predicted directly in world mm (X, Y, Z), linearly
    normalised by the actual observed range of real translation vectors
    (see TRANS_MM_RANGE below) -- no orbit/distance-from-target
    assumption, just the raw 3D position.

    predicted (r1x,r1y,r1z, r2x,r2y,r2z, tx_norm,ty_norm,tz_norm)
    -> R (via rotation_6d_to_matrix), t (via decode_translation)
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import timm
    _TIMM = True
except ImportError:
    _TIMM = False


# ---------------------------------------------------------------------------
# Full 6-DOF pose parameterisation: 6D continuous rotation + direct
# world-mm translation. No look-at / orbit / small-delta assumption.
# ---------------------------------------------------------------------------
# Measured real-world translation range (JYOTHI, 6 real X-ray sets, EPnP-
# verified reprojection error): t_x in [-118, 1347]mm, t_y in [-1096, -271]
# mm, t_z in [-275, 2330]mm. Also cross-checked against 5000 samples from
# the training-data orbit sampler (scripts/train_pose_regressor.py's
# _sample_random_pose, which aims at random landmark subsets across the
# full azimuth/elevation/roll/distance ranges): that sampler produces
# t_x/t_y up to +-1580mm and t_z from -712 to +2809mm. Ranges below cover
# both with margin. These are WORLD-frame translation components (same
# convention as DeepFluoroProjection.t_proj), not distance-from-target --
# there is no look-at assumption in how they are used here.
TRANS_MM_MIN = np.array([-2000., -2000., -1000.], dtype=np.float64)
TRANS_MM_MAX = np.array([ 2000.,  2000.,  3200.], dtype=np.float64)
TRANS_MM_MEAN = (TRANS_MM_MIN + TRANS_MM_MAX) / 2.0
TRANS_MM_HALF = (TRANS_MM_MAX - TRANS_MM_MIN) / 2.0

# Output dimensionality of the network: 6 (rotation, 6D repr) + 3 (translation).
POSE_VEC_DIM = 9


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Convert a 6D rotation representation (Zhou et al., CVPR 2019) to a
    3x3 rotation matrix via Gram-Schmidt orthonormalisation.

    Any real-valued 6-vector decodes to SOME valid rotation matrix (no
    unit-norm or orthogonality constraint needed on the raw network
    output), which avoids the discontinuities of Euler angles / axis-
    angle and the extra unit-norm constraint quaternions need.

    Args:
        d6 : (6,) array, [a1(3), a2(3)]
    Returns:
        (3, 3) rotation matrix, rows = [x_cam, y_cam, z_cam]
    """
    a1 = np.asarray(d6[:3], dtype=np.float64)
    a2 = np.asarray(d6[3:], dtype=np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    a2_proj = a2 - np.dot(b1, a2) * b1
    b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def matrix_to_rotation_6d(R: np.ndarray) -> np.ndarray:
    """Inverse-ish of rotation_6d_to_matrix: just take the first two rows
    of R as the 6D representation (any valid rotation matrix already has
    orthonormal rows, so this round-trips exactly through
    rotation_6d_to_matrix)."""
    return np.concatenate([R[0, :], R[1, :]]).astype(np.float32)


def encode_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Encode a full (R, t) camera extrinsic as a (9,) training target:
        [r1x,r1y,r1z, r2x,r2y,r2z, tx_norm,ty_norm,tz_norm]

    R's first two rows are used directly as the 6D rotation
    representation (matrix_to_rotation_6d). Translation is linearly
    normalised per-axis by the measured real-world range
    (TRANS_MM_MIN/MAX) to roughly [-1, 1].
    """
    rot6 = matrix_to_rotation_6d(R)
    t_norm = (np.asarray(t, dtype=np.float64) - TRANS_MM_MEAN) / TRANS_MM_HALF
    return np.concatenate([rot6, t_norm.astype(np.float32)])


def decode_pose(pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of encode_pose(): (9,) network output -> (R, t)."""
    R = rotation_6d_to_matrix(pred[:6])
    t = pred[6:9] * TRANS_MM_HALF + TRANS_MM_MEAN
    return R, t.astype(np.float64)


class PoseRegressor(nn.Module):
    """
    CNN backbone -> GAP -> MLP -> 9-D pose encoding: 6D continuous
    rotation representation + normalised world-mm translation
    (see encode_pose / decode_pose above).

    Same backbone choices as ssm_regressor.SSMRegressor (timm, 1-channel
    input) but a smaller, purely-numeric regression head since the output
    space is only 9 scalars with no shape/SSM structure to respect.
    """

    def __init__(self,
                 backbone_name: str = 'resnet34',
                 pretrained:    bool = True,
                 dropout:       float = 0.3):
        super().__init__()
        if not _TIMM:
            raise ImportError('timm is required for PoseRegressor backbones.')

        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            in_chans=1, num_classes=0,  # feature extractor, no head
        )
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.LayerNorm(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, POSE_VEC_DIM),
        )
        nn.init.zeros_(self.head[-1].bias)
        nn.init.normal_(self.head[-1].weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> (B, POSE_VEC_DIM) pose encoding."""
        feats = self.backbone(x)
        return self.head(feats)


class PoseRegressionLoss(nn.Module):
    """
    Weighted MSE between predicted and target pose encodings
    (see encode_pose()). Indices: [0:6]=6D rotation representation,
    [6:9]=normalised translation (x, y, z).

    depth_weight upweights the z-translation term specifically (index 8).
    Depth/distance-from-camera is the hardest quantity to judge from a
    single 2D image (a classic monocular depth-ambiguity: moving closer
    while rotating less can look almost identical to staying farther
    while rotating more) -- upweighting gives the network more gradient
    signal to actually learn it rather than defaulting to the safe
    "always predict the mean" answer.
    """

    def __init__(self, rot_weight: float = 1.0, trans_weight: float = 1.0,
                 depth_weight: float = 2.0):
        super().__init__()
        self.rot_weight   = rot_weight
        self.trans_weight = trans_weight
        self.depth_weight = depth_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = (pred - target) ** 2
        rot_loss   = err[:, :6].mean()
        xy_loss    = err[:, 6:8].mean()
        depth_loss = err[:, 8].mean()
        return (self.rot_weight * rot_loss
                + self.trans_weight * xy_loss
                + self.depth_weight * depth_loss)


if __name__ == '__main__':
    model = PoseRegressor(backbone_name='resnet34', pretrained=False)
    x = torch.randn(2, 1, 512, 512)
    out = model(x)
    print(f'Input : {x.shape}')
    print(f'Output: {out.shape}')  # expect (2, 9)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params / 1e6:.2f} M')

    # Round-trip sanity check: random valid rotation matrix + translation
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    t_true = np.array([500.0, -800.0, 1200.0])
    enc = encode_pose(Q, t_true)
    R2, t2 = decode_pose(enc)
    print(f'Round-trip rotation max abs diff: {np.max(np.abs(Q - R2)):.6f}')
    print(f'Round-trip translation: {t_true} -> {t2.round(3)}')
