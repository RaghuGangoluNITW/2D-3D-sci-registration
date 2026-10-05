#!/usr/bin/env python3
"""
train_pose_regressor.py
========================
Trains a PATIENT-SPECIFIC 6-DOF pose regression CNN (src/pose_regressor.py)
-- the new "no landmarks, no EPnP" approach to automatic 2D/3D registration.

Clinical framing (per user requirement, this session)
--------------------------------------------------------
One patient, fully independent of every other patient. Pre-op CT is
available before the operation; this script represents the "prepare
before the operation" step: given only this one patient's CT, generate a
large synthetic-DRR training set (many simulated C-arm poses, with a
fraction simulating instrument occlusion) and train a small CNN to
predict the C-arm pose directly from an X-ray-like image of THIS patient.

Why not landmarks + EPnP (the approach used earlier this session)
----------------------------------------------------------------------
A direct test (see project history / status report) proved that solving
EPnP from the D12/L1-L5 vertebra centroids is fundamentally unstable --
those points are nearly collinear along the spine axis, so even near-
perfect 2D landmark accuracy still produces wild 3D pose swings (median
15 degrees / 340mm at 1mm 2D error in one measured case). This approach
sidesteps that entirely: the CNN predicts the pose directly, with no
point-based linear-algebra step in between.

What this script does
-----------------------
  1. Load ONE patient's CT + 3D landmarks.
  2. Render a large set of synthetic DRRs at KNOWN random poses. Poses
     are sampled by aiming at a random subset of the patient's own
     vertebra landmarks (matching the real observation that different
     acquisitions centre on different vertebral levels) from a random
     azimuth/elevation/roll/distance -- see _sample_random_pose(). The
     resulting ABSOLUTE (R, t) is encoded directly via
     pose_regressor.encode_pose() (6D rotation + normalised world-mm
     translation) as the training target -- NOT a delta from one fixed
     reference pose (an earlier version used a small-delta
     parameterisation and was found to badly under-cover the real range
     of intraoperative C-arm positions; see pose_regressor.py docstring
     for the full history).
  3. A held-out validation split of freshly-sampled synthetic poses (NOT
     other patients -- this is intentionally NOT LOPO, matching the
     per-patient clinical framing) is used to track generalisation.
  4. Trains PoseRegressor, saves checkpoint to
     checkpoints/pose_regressor/<PATIENT>.pth

Usage
-----
    python scripts/train_pose_regressor.py --patient JYOTHI \
        --n_train 4000 --n_val 400 --epochs 100 --instrument_frac 0.3
"""

import argparse
import sys
import time
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import SimpleITK as sitk

from generic_patient_loader import (
    GenericPatientLoader, project_world, compute_pde,
    CAM_PIX_MM, CAM_IMG_SIZE, CAM_K, _load_mrk_3d, _solve_pnp,
)
from deepfluoro_loader import DeepFluoroSpecimen
from deepfluoro_drr import DeepFluoroDRR
from pose_regressor import (
    PoseRegressor, PoseRegressionLoss, encode_pose, decode_pose,
    TRANS_MM_MIN, TRANS_MM_MAX,
)
from cyclegan import CycleGAN

DATA_ROOT    = ROOT / 'data' / 'patients'
CT_ONLY_ROOT = ROOT / 'data' / 'patients_ct_only'
CKPT_DIR     = ROOT / 'checkpoints' / 'pose_regressor'
LUMBAR       = ['D12', 'L1', 'L2', 'L3', 'L4', 'L5']

RENDER_SIZE = 256      # DRR render resolution (fast)
TRAIN_SIZE  = 512      # network input resolution
N_STEPS     = 200      # ray-marching steps (verified clean, see generate_synthetic_drrs.py)

# ---------------------------------------------------------------------------
# Training pose SAMPLING ranges (generator only -- these do NOT need to
# exactly explain the real data; they just need to produce diverse,
# plausible camera views so that whatever real pose shows up at inference
# time falls inside the training distribution somewhere).
#
# A "look-at a fixed spine centroid, orbit at some az/el/roll/distance"
# model was tested as an ANALYTIC fit to real poses and found to be
# geometrically wrong (see pose_regressor.py docstring) -- real
# acquisitions are not all centred on one fixed 3D point. But used only
# as a SAMPLING GENERATOR (aim point varies randomly per sample, exact
# framing is verified post-hoc against the real image frame, and the
# resulting absolute (R, t) is encoded directly via encode_pose() with NO
# reference-pose/delta involved), this same orbit construction is still a
# convenient way to generate diverse, anatomically-plausible camera poses
# -- it just isn't used to claim any single real pose matches one exact
# az/el/roll/distance tuple relative to one fixed point.
#
# Ranges set with margin around the measured real spread on JYOTHI's 6
# real X-ray sets: azimuth -23 to +176 deg, elevation -46 to +51 deg,
# roll -6 to +82 deg, source-to-centroid distance ~850-1200mm.
AZIMUTH_RANGE_DEG   = 180.0   # sample uniformly in [-180, 180)
ELEVATION_RANGE_DEG = 65.0    # sample uniformly in [-65, 65]
ROLL_RANGE_DEG      = 100.0   # sample uniformly in [-100, 100]
DIST_MIN_MM         = 700.0
DIST_MAX_MM         = 1400.0


# ---------------------------------------------------------------------------
# Patient loading (reuses same CT-only fallback pattern as
# generate_synthetic_drrs.py -- kept as a local copy so this script has no
# hidden dependency on that script's internals).
# ---------------------------------------------------------------------------

def _load_specimen(patient_dir: Path):
    has_xray_label = (patient_dir / 'XRAY Label').exists()
    if has_xray_label:
        loader = GenericPatientLoader(patient_dir)
        spec = loader.load(min_labels=3, verbose=False)
    else:
        ct_dir = patient_dir / 'CT label'
        ct_nrrds = list(ct_dir.glob('*.nrrd'))
        ct_jsons = list(ct_dir.glob('*.mrk.json'))
        if not ct_nrrds or not ct_jsons:
            raise FileNotFoundError(f'CT or centroid JSON missing in {ct_dir}')
        img = sitk.ReadImage(str(ct_nrrds[0]))
        ct_vol = sitk.GetArrayFromImage(img).astype(np.float32)
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        origin = np.array(img.GetOrigin(), dtype=np.float64)
        lm_3d = _load_mrk_3d(ct_jsons[0])
        keep = set(LUMBAR) | {'D11', 'S1'}
        lm_3d = {k: v for k, v in lm_3d.items() if k in keep}
        spec = DeepFluoroSpecimen(
            specimen_id=patient_dir.name, ct_volume=ct_vol,
            ct_spacing=spacing, ct_origin=origin,
            landmarks_3d=lm_3d, projections=[])
    return spec


def _find_patient_dir(patient: str) -> Path:
    for root in (DATA_ROOT, CT_ONLY_ROOT):
        d = root / patient
        if d.exists():
            return d
    raise FileNotFoundError(f'Patient {patient!r} not found in {DATA_ROOT} or {CT_ONLY_ROOT}')


def _orbit_pose(aim_point: np.ndarray, azimuth_deg: float, elevation_deg: float,
                roll_deg: float, distance_mm: float) -> tuple:
    """
    Build a camera extrinsic (R, t) for a source positioned on a sphere of
    radius `distance_mm` around `aim_point`, at the given azimuth/
    elevation, looking at `aim_point`, with an explicit roll applied
    around the viewing axis.

    Used ONLY as a training-data SAMPLING generator (see module-level
    comment above AZIMUTH_RANGE_DEG) -- aim_point varies randomly per
    sample (see PatientPoseDataset), and the resulting (R, t) is encoded
    directly via pose_regressor.encode_pose() with no reference-pose
    delta involved. This is not claimed to be an exact analytic model of
    any real acquisition.
    """
    az, el = np.deg2rad([azimuth_deg, elevation_deg])
    src_dir = np.array([
        np.cos(el) * np.sin(az),
        np.cos(el) * np.cos(az),
        np.sin(el),
    ], dtype=np.float64)
    src_world = aim_point + distance_mm * src_dir

    src_xzy = src_world[[0, 2, 1]]
    cnt_xzy = aim_point[[0, 2, 1]]
    z_cam = (cnt_xzy - src_xzy); z_cam /= np.linalg.norm(z_cam)
    up = np.array([0., 0., -1.])
    if abs(z_cam.dot(up)) > 0.9:
        up = np.array([1., 0., 0.])
    x_cam = np.cross(up, z_cam); x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)

    roll = np.deg2rad(roll_deg)
    cr, sr = np.cos(roll), np.sin(roll)
    x_cam_r = cr * x_cam + sr * y_cam
    y_cam_r = -sr * x_cam + cr * y_cam

    R = np.stack([x_cam_r, y_cam_r, z_cam], axis=0).astype(np.float64)
    t = (-R @ src_xzy).astype(np.float64)
    return R, t


def _sample_random_pose(lm_3d: dict, rng: np.random.Generator) -> tuple:
    """
    Sample one random, plausible camera pose for training data generation.

    The aim point is chosen as the centroid of a RANDOM SUBSET of the
    patient's vertebra landmarks (2 to all of them) rather than always all
    landmarks -- this matches the real observation that different
    intraoperative acquisitions of the same patient centre on different
    vertebral levels (e.g. only L3/L4/L5 visible in some real AP sets,
    with D12/L1/L2 off-frame entirely). Azimuth/elevation/roll/distance
    are then sampled uniformly over the ranges set above.
    """
    names = list(lm_3d.keys())
    n_pick = rng.integers(2, len(names) + 1)
    picked = rng.choice(names, size=n_pick, replace=False)
    aim_point = np.mean([lm_3d[n] for n in picked], axis=0)

    az   = rng.uniform(-AZIMUTH_RANGE_DEG, AZIMUTH_RANGE_DEG)
    el   = rng.uniform(-ELEVATION_RANGE_DEG, ELEVATION_RANGE_DEG)
    roll = rng.uniform(-ROLL_RANGE_DEG, ROLL_RANGE_DEG)
    dist = rng.uniform(DIST_MIN_MM, DIST_MAX_MM)

    return _orbit_pose(aim_point, az, el, roll, dist)


# ---------------------------------------------------------------------------
# Synthetic dataset: renders on-the-fly (no disk cache -- pose regression
# needs far fewer images per patient than the landmark-manifest approach,
# and every image is patient-specific anyway so there's no reuse benefit
# from caching to disk across runs the way SYN_* DRRs were shared).
# ---------------------------------------------------------------------------

class PatientPoseDataset(Dataset):
    """
    PRE-RENDERED synthetic DRR + pose-perturbation-target dataset for ONE
    patient. All `n_samples` DRRs are rendered ONCE at construction time
    and cached in RAM (matching the pattern used everywhere else in this
    codebase -- XRayDataset/DRRDataset in train_cyclegan.py,
    SSMRegressionDataset in train_ssm_regressor.py -- rather than
    re-rendering a fresh DRR on every __getitem__ call, which would make
    every epoch as slow as the first and waste most of the epoch budget
    on GPU ray-marching instead of network training).

    Each cached sample:
      1. Picks a view (AP/LAT), gets that view's anatomy-pose reference.
      2. Samples a random 6-DOF perturbation (rot_deg, trans_mm).
      3. Renders a DRR at the perturbed pose (resampling on blank/near-
         blank renders -- wide perturbation ranges can push the pose far
         enough to miss the CT volume entirely).
      4. Target = encode_pose(rot_deg, trans_mm) -- the SAME perturbation
         used to render, not a separately-computed landmark-based target.

    Instrument occlusion and other cheap image-space augmentation are
    applied fresh on every __getitem__ call (not baked into the cache),
    the same way CLAHE + affine + noise augmentation is applied on top of
    a fixed pre-loaded image cache elsewhere in this project.
    """

    # Minimum fraction of nonzero pixels for a DRR to be considered a valid
    # (non-blank) render.
    MIN_BONE_COVER = 0.05
    MAX_TRIES = 20

    # Anatomy-framing constraint (in addition to MIN_BONE_COVER): requires
    # at least MIN_VISIBLE_LM vertebra landmarks to project inside the
    # image with a margin -- i.e. the spine must be reasonably centred /
    # framed, not just "some bone visible in a corner". Matches the same
    # constraint generate_synthetic_drrs.py already enforces
    # (MIN_VISIBLE_LM/MIN_MARGIN_PX there). Found necessary because without
    # it, wide pose sampling produces badly-cropped renders that (a) no
    # real C-arm operator would capture and (b) are far enough outside the
    # CycleGAN's own training distribution to produce corrupted style-
    # transfer output when passed through G_BA/G_AB.
    MIN_VISIBLE_LM = 3
    MIN_MARGIN_PX  = 30

    def __init__(self, spec, gen: DeepFluoroDRR, n_samples: int,
                 instrument_frac: float = 0.3, seed: int = 0,
                 render_size: int = RENDER_SIZE, train_size: int = TRAIN_SIZE,
                 augment: bool = True,
                 cyclegan: Optional[CycleGAN] = None,
                 xray_style_frac: float = 0.5,
                 device: str = 'cuda'):
        """
        cyclegan / xray_style_frac : DOMAIN-GAP FIX. A first end-to-end test
            (see project history) found the pose regressor trained purely on
            clean synthetic DRRs saturates at the edge of its training range
            on real X-rays -- real images are far enough outside the
            synthetic-only training distribution that the network doesn't
            recognise them at all. If a trained CycleGAN checkpoint is
            provided, a fraction (xray_style_frac) of training images are
            passed through G_BA (DRR -> X-ray style) before being shown to
            the pose regressor, using the SAME known pose label either way
            -- this exposes the network to real-X-ray-like appearance
            (noise, contrast, texture) during training without needing any
            real paired pose-labeled X-rays, which don't exist.
        """
        self.instr_frac  = instrument_frac
        self.render_size = render_size
        self.train_size  = train_size
        self.augment     = augment
        self.cyclegan    = cyclegan
        self.xray_style_frac = xray_style_frac if cyclegan is not None else 0.0
        self.device      = device
        self.rng = np.random.default_rng(seed)
        self.lm_3d = {k: v for k, v in spec.landmarks_3d.items() if k in LUMBAR}

        print(f'  Pre-rendering {n_samples} DRRs (seed={seed})...', flush=True)
        t0 = time.time()
        self._imgs    = []   # list of (render_size, render_size) float32 [0,1] CLAHE'd
        self._targets = []   # list of (9,) float32 -- absolute pose, encode_pose()
        pix_mm = CAM_PIX_MM * (CAM_IMG_SIZE / render_size)
        lm_names_check = [n for n in LUMBAR if n in spec.landmarks_3d]
        pts3d_check = np.array([spec.landmarks_3d[n] for n in lm_names_check])

        for i in range(n_samples):
            for _try in range(self.MAX_TRIES):
                R, t = _sample_random_pose(self.lm_3d, self.rng)

                # Anatomy-framing check (cheap, before rendering): reject
                # poses where the spine projects too close to the frame
                # edge / mostly out of frame, BEFORE spending GPU time on
                # a full DRR render.
                uv = project_world(pts3d_check, R, t)
                m = self.MIN_MARGIN_PX
                n_visible = int(np.sum(
                    (uv[:, 0] >= m) & (uv[:, 0] < CAM_IMG_SIZE - m) &
                    (uv[:, 1] >= m) & (uv[:, 1] < CAM_IMG_SIZE - m)))
                if n_visible < self.MIN_VISIBLE_LM:
                    continue

                drr = gen.generate_from_extrinsic(R, t, render_size, pix_mm, N_STEPS)
                if float(np.count_nonzero(drr > 0.02)) / drr.size >= self.MIN_BONE_COVER:
                    break
            else:
                R, t = _sample_random_pose(self.lm_3d, self.rng)
                drr = gen.generate_from_extrinsic(R, t, render_size, pix_mm, N_STEPS)

            # Domain-gap fix: pass a fraction of RAW (pre-CLAHE) DRRs
            # through CycleGAN's DRR->X-ray generator, keeping the SAME
            # pose label (style transfer doesn't move the anatomy, only
            # changes appearance). IMPORTANT: must feed the raw [0,1] DRR
            # here, NOT the CLAHE'd version -- train_cyclegan.py's
            # DRRDataset (domain B) never applies CLAHE, so G_BA was only
            # ever trained on raw DRR statistics. Feeding it a CLAHE'd
            # image was found to produce badly corrupted output (comb/grid
            # artifacts) since that input is out-of-distribution for G_BA.
            if self.cyclegan is not None and self.rng.random() < self.xray_style_frac:
                with torch.no_grad():
                    d256 = cv2.resize(drr.astype(np.float32), (256, 256),
                                      interpolation=cv2.INTER_LINEAR)
                    t_in = torch.from_numpy(d256).unsqueeze(0).unsqueeze(0).to(self.device)
                    styled = self.cyclegan.drr_to_xray(t_in)[0, 0].cpu().numpy()
                styled = np.clip(cv2.resize(styled, (render_size, render_size),
                                            interpolation=cv2.INTER_LINEAR), 0, 1)
                # CLAHE applied AFTER styling here, since the pose
                # regressor's own preprocessing (both training and
                # inference) always applies CLAHE -- this must match.
                img_final = self._clahe(styled)
            else:
                img_final = self._clahe(drr)

            self._imgs.append(img_final)
            self._targets.append(encode_pose(R, t))
            if (i + 1) % 500 == 0:
                print(f'    {i+1}/{n_samples} rendered...', flush=True)
        print(f'  Pre-render done: {n_samples} images in {time.time()-t0:.0f}s  '
              f'(xray_style_frac={self.xray_style_frac})', flush=True)

    def __len__(self):
        return len(self._imgs)

    def _augment_instrument(self, img: np.ndarray) -> np.ndarray:
        """Simulate a metal instrument (screw/rod) as a dark occluding
        rectangle -- cheap proxy for real hardware without needing any
        actual post-op/instrumented CT data for this patient."""
        H, W = img.shape[:2]
        n_instr = self.rng.integers(1, 3)
        img = img.copy()
        for _ in range(n_instr):
            ew = int(self.rng.uniform(0.03, 0.12) * W)
            eh = int(self.rng.uniform(0.15, 0.45) * H)
            ex = int(self.rng.uniform(0, max(1, W - ew)))
            ey = int(self.rng.uniform(0, max(1, H - eh)))
            angle = self.rng.uniform(-25, 25)
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[ey:ey+eh, ex:ex+ew] = 255
            M = cv2.getRotationMatrix2D((ex + ew/2, ey + eh/2), angle, 1.0)
            mask = cv2.warpAffine(mask, M, (W, H))
            dark_val = float(self.rng.uniform(0.0, 0.15))
            img[mask > 0] = dark_val
        return img

    @staticmethod
    def _clahe(img: np.ndarray) -> np.ndarray:
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
        return cl.astype(np.float32) / 255.0

    def __getitem__(self, idx: int):
        img = self._imgs[idx].copy()
        target = self._targets[idx]

        if self.augment and self.rng.random() < self.instr_frac:
            img = self._augment_instrument(img)
        if self.augment:
            # Intensity-only jitter (no affine -- that would invalidate the
            # pose target since it geometrically moves the anatomy without
            # updating the label). Wider ranges than the first version of
            # this augmentation -- narrow jitter wasn't enough to prevent
            # the model from over-fitting to the exact synthetic rendering
            # style; widened to match the intensity augmentation strength
            # already proven to help domain-gap in landmark_dataset.py /
            # train_cyclegan.py's _augment().
            alpha = self.rng.uniform(0.6, 1.5)
            beta  = self.rng.uniform(-0.15, 0.15)
            img = np.clip(alpha * img + beta, 0, 1)
            gamma = self.rng.uniform(0.6, 1.7)
            img = np.clip(img ** gamma, 0, 1)
            if self.rng.random() < 0.4:
                ksize = int(self.rng.choice([3, 5, 7]))
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            noise = self.rng.normal(0, self.rng.uniform(0.0, 0.06), img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1).astype(np.float32)

        img_rs = cv2.resize(img, (self.train_size, self.train_size),
                            interpolation=cv2.INTER_LINEAR)
        return (torch.from_numpy(img_rs).unsqueeze(0),
                torch.from_numpy(target))


# ---------------------------------------------------------------------------
# Evaluation: decode predicted pose, compute PDE against this patient's
# own 3D landmarks re-projected at the TRUE (rendering) pose -- i.e. same
# PDE definition used everywhere else in this project, just computed from
# the known synthetic ground-truth pose instead of a file-label projection.
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, spec, gen, device, n_eval: int = 40,
            instrument_frac: float = 0.3, seed: int = 999,
            render_size: int = RENDER_SIZE, train_size: int = TRAIN_SIZE,
            pixel_mm: float = 0.2):
    model.eval()
    rng = np.random.default_rng(seed)
    lm_names = sorted(n for n in spec.landmarks_3d if n in LUMBAR)
    pts3d = np.array([spec.landmarks_3d[n] for n in lm_names])
    lm_3d = {k: v for k, v in spec.landmarks_3d.items() if k in LUMBAR}

    pde_errs = []
    for i in range(n_eval):
        # Resample on blank/near-blank OR badly-framed renders (same
        # anatomy-framing + bone-coverage reasoning as
        # PatientPoseDataset.__init__).
        pix_mm = CAM_PIX_MM * (CAM_IMG_SIZE / render_size)
        for _try in range(20):
            R_true, t_true = _sample_random_pose(lm_3d, rng)
            uv_check = project_world(pts3d, R_true, t_true)
            m = 30
            n_visible = int(np.sum(
                (uv_check[:, 0] >= m) & (uv_check[:, 0] < CAM_IMG_SIZE - m) &
                (uv_check[:, 1] >= m) & (uv_check[:, 1] < CAM_IMG_SIZE - m)))
            if n_visible < 3:
                continue
            drr = gen.generate_from_extrinsic(R_true, t_true, render_size, pix_mm, N_STEPS)
            if float(np.count_nonzero(drr > 0.02)) / drr.size >= 0.05:
                break
        else:
            R_true, t_true = _sample_random_pose(lm_3d, rng)
            drr = gen.generate_from_extrinsic(R_true, t_true, render_size, pix_mm, N_STEPS)

        u8 = (np.clip(drr, 0, 1) * 255).astype(np.uint8)
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8).astype(np.float32) / 255.0
        if rng.random() < instrument_frac:
            H, W = img.shape
            ew, eh = int(0.08*W), int(0.3*H)
            ex, ey = rng.integers(0, W-ew), rng.integers(0, H-eh)
            img[ey:ey+eh, ex:ex+ew] = 0.05
        img_rs = cv2.resize(img, (train_size, train_size), interpolation=cv2.INTER_LINEAR)
        t_in = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to(device)

        pred_norm = model(t_in)[0].cpu().numpy()
        R_pred, t_pred = decode_pose(pred_norm)

        uv_true = project_world(pts3d, R_true, t_true)
        uv_pred = project_world(pts3d, R_pred, t_pred)
        err = float(np.mean(np.linalg.norm(uv_pred - uv_true, axis=1))) * pixel_mm
        pde_errs.append(err)

    return float(np.mean(pde_errs)), float(np.median(pde_errs))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--patient', required=True)
    p.add_argument('--n_train', type=int, default=4000,
                   help='Synthetic training samples per epoch pass (rendered on the fly).')
    p.add_argument('--n_val',   type=int, default=200,
                   help='Held-out synthetic validation samples (fixed seed, same patient).')
    p.add_argument('--epochs',  type=int, default=60)
    p.add_argument('--batch',   type=int, default=8)
    p.add_argument('--lr',      type=float, default=3e-4)
    p.add_argument('--backbone', default='resnet34')
    p.add_argument('--instrument_frac', type=float, default=0.3,
                   help='Fraction of training images with a simulated instrument occlusion.')
    p.add_argument('--cyclegan_ckpt', default=None,
                   help='Path to a trained CycleGAN checkpoint (e.g. '
                        'checkpoints/cyclegan/best.pth). If given, a '
                        'fraction of training DRRs are passed through '
                        'G_BA (DRR->X-ray style) to close the synthetic-'
                        'vs-real domain gap -- see PatientPoseDataset '
                        'docstring. Default: None (no domain adaptation, '
                        'the setting that produced the saturating-output '
                        'bug on real X-rays in the first end-to-end test).')
    p.add_argument('--xray_style_frac', type=float, default=0.5,
                   help='Fraction of training images passed through '
                        'CycleGAN G_BA when --cyclegan_ckpt is given.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--ckpt_dir', default=None)
    p.add_argument('--steps_per_epoch', type=int, default=200,
                   help='Batches per epoch (dataset is rendered on the fly, so this '
                        'controls epoch wall-clock time independent of --n_train).')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    patient_dir = _find_patient_dir(args.patient)
    print(f'Loading patient: {args.patient}  ({patient_dir})')
    spec = _load_specimen(patient_dir)
    if not spec.landmarks_3d:
        print('ERROR: no 3D landmarks found for this patient.')
        return
    print(f'  3D landmarks: {sorted(spec.landmarks_3d.keys())}')

    gen = DeepFluoroDRR(spec, hu_threshold=150.0, device=args.device)

    cyclegan = None
    if args.cyclegan_ckpt:
        print(f'Loading CycleGAN for domain adaptation: {args.cyclegan_ckpt}')
        cyclegan = CycleGAN.load(args.cyclegan_ckpt, device=args.device)
        cyclegan.eval()
        print(f'  xray_style_frac={args.xray_style_frac} '
              f'(fraction of training images styled as real X-rays)')

    ds_train = PatientPoseDataset(spec, gen, n_samples=args.n_train,
                                  instrument_frac=args.instrument_frac, seed=0,
                                  augment=True, cyclegan=cyclegan,
                                  xray_style_frac=args.xray_style_frac,
                                  device=args.device)
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)

    model = PoseRegressor(backbone_name=args.backbone, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_steps = max(1, int(args.epochs * 0.10))
    def _lr_lambda(ep):
        if ep < warmup_steps:
            return float(ep + 1) / float(warmup_steps + 1)
        progress = (ep - warmup_steps) / max(1, args.epochs - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    criterion = PoseRegressionLoss().to(device)

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'{args.patient}.pth'

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable/1e6:.2f}M')
    print(f'Steps/epoch: {args.steps_per_epoch}  batch={args.batch}  '
          f'instrument_frac={args.instrument_frac}')

    best_pde = float('inf')
    best_ep  = 0
    patience = 20

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_steps = 0
        it = iter(dl_train)
        for step in range(args.steps_per_epoch):
            try:
                imgs, targets = next(it)
            except StopIteration:
                it = iter(dl_train)
                imgs, targets = next(it)
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_steps += 1
        scheduler.step()
        train_loss /= max(n_steps, 1)

        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            mean_pde, med_pde = evaluate(
                model, spec, gen, device,
                n_eval=args.n_val, instrument_frac=args.instrument_frac)
            elapsed = time.time() - t0
            improved = mean_pde < best_pde
            if improved:
                best_pde = mean_pde
                best_ep  = ep
                torch.save({'model_state': model.state_dict(),
                            'backbone_name': args.backbone,
                            'patient': args.patient,
                            'val_pde_mm': best_pde,
                            'epoch': ep}, ckpt_path)
            marker = ' [BEST]' if improved else ''
            print(f'  ep {ep:3d}/{args.epochs}  loss={train_loss:.4f}  '
                  f'val_pde={mean_pde:.2f}mm  val_pde_median={med_pde:.2f}mm'
                  f'{marker}  ({elapsed:.1f}s)')
            if ep - best_ep >= patience:
                print(f'  Early stop at ep {ep} (best ep={best_ep})')
                break

    print(f'\n-> Best val_pde={best_pde:.2f}mm  (ep {best_ep})')
    print(f'   Checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()
