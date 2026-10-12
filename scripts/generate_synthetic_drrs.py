#!/usr/bin/env python3
"""
generate_synthetic_drrs.py
==========================
Stage 1 of the automatic PDE <2mm pipeline.

For each patient CT, sample N random C-arm poses and render a DRR at
each pose.  Project the 3D vertebral centroids through the same pose to get
free 2D landmark labels — no manual annotation required.

Only poses where ALL visible landmarks fall inside the image AND the DRR
is non-trivial (enough bone coverage) are kept.

Output layout
-------------
data/synthetic_drrs/
    <PATIENT>/
        AP/
            <idx>.png          # 512×512 uint8 DRR
            <idx>_labels.csv   # L1,u,v  L2,u,v ...
        LAT/
            ...
data/synthetic_drr_manifest.csv   # combined manifest in same format as
                                   # landmark_manifest_buu_combined.csv

Usage
-----
# All patients, 500 poses each view:
python scripts/generate_synthetic_drrs.py

# Single patient, 100 poses (quick test):
python scripts/generate_synthetic_drrs.py --patients PATIENT_001 --n_per_view 100

# All patients, full run (recommended):
python scripts/generate_synthetic_drrs.py --n_per_view 500 --render_size 512
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import SimpleITK as sitk

from generic_patient_loader import (
    GenericPatientLoader,
    project_world,
    CAM_PIX_MM,
    CAM_IMG_SIZE,
    CAM_FX, CAM_FY, CAM_CX, CAM_CY,
    _load_mrk_3d,
)
from deepfluoro_drr import DeepFluoroDRR
from deepfluoro_loader import perturb_extrinsic, DeepFluoroSpecimen
from camera_config import load_camera_config
from landmark_dataset import CLASS_NAMES as ALL_LANDMARK_NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT    = ROOT / 'data' / 'patients'
CT_ONLY_ROOT = ROOT / 'data' / 'patients_ct_only'
OUT_ROOT    = ROOT / 'data' / 'synthetic_drrs'
MANIFEST    = ROOT / 'data' / 'synthetic_drr_manifest.csv'

# Patients excluded due to confirmed DRR rendering artifacts (curved
# moire/ripple pattern overwhelming the anatomy signal) that increasing
# n_steps (tested up to 800) did not resolve — likely a pose/geometry
# calibration issue specific to this patient, out of scope for this fix.
EXCLUDED_PATIENTS: set[str] = set()  # add patient IDs to skip

# Landmark names to project: centroids (D12 + lumbar) plus the extracted
# non-collinear pedicle points (LEFT_PED/RIGHT_PED per level) -- see
# src/landmark_dataset.py CLASS_NAMES for the full derivation. Both sets
# are projected identically (just 3D points through the same synthetic
# pose), so adding pedicles costs nothing extra in rendering, only in the
# 3D point lookup below.
LUMBAR = ['D12', 'L1', 'L2', 'L3', 'L4', 'L5']
ALL_LANDMARKS = list(ALL_LANDMARK_NAMES)  # centroids + pedicle points

# Image size for saved PNGs (regressor training resolution)
SAVE_SIZE = 512

# DRR render size (small → fast; upsampled to SAVE_SIZE on disk)
# IMPORTANT: n_steps must be high enough to avoid ray-marching aliasing
# artifacts on larger CT volumes (confirmed bug: n_steps=80 produces pure
# noise/diagonal-stripe images on volumes with >450 Z-slices, e.g. certain patients with CT artifacts).
# n_steps=200 gives clean images across all tested patients (PATIENT_001, PATIENT_001)
# at 256px render size — verified visually. Still fits comfortably in 4GB VRAM.
RENDER_SIZE = 256
N_STEPS     = 200

# Pose perturbation ranges relative to existing calibrated projections
# We sample around the real poses found per patient — this keeps the
# X-ray source inside the patient body (anatomically valid).
ROT_RANGE_DEG   = 25.0   # ± degrees per axis
TRANS_RANGE_MM  = 50.0   # ± mm per axis

# Quality thresholds
MIN_VISIBLE_LM  = 3      # minimum landmarks that must be in-frame
MIN_BONE_COVER  = 0.05   # DRR must have >5% nonzero pixels
MIN_MARGIN_PX   = 30     # landmarks must be this far from image edge

# AP azimuth range around 0° (frontal), LAT around 90°
VIEW_AZIMUTHS = {
    'AP':  (  0.0, 30.0),   # (centre_deg, half_range_deg)
    'LAT': ( 90.0, 30.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _anatomy_pose_random(lm_3d: dict, azimuth_deg: float,
                          rng: np.random.Generator,
                          sid_mm: float = 1110.0) -> tuple:
    """
    Build a camera pose looking at the spine centroid from a given azimuth,
    then add a small random perturbation.

    Returns (R, t) for the camera extrinsic.
    """
    centroid = np.mean(np.array(list(lm_3d.values())), axis=0)
    az = np.deg2rad(azimuth_deg)

    # Source direction in XY plane (XZY world convention used by DRR)
    src_dir   = np.array([np.sin(az), np.cos(az), 0.], dtype=np.float64)
    src_world = centroid + sid_mm * src_dir

    # Convert to XZY (camera convention: swap Y↔Z)
    src_xzy = src_world[[0, 2, 1]]
    cnt_xzy = centroid[[0, 2, 1]]

    z_cam = (cnt_xzy - src_xzy)
    z_cam /= np.linalg.norm(z_cam)
    up = np.array([0., 0., -1.])
    if abs(z_cam.dot(up)) > 0.9:
        up = np.array([1., 0., 0.])
    x_cam = np.cross(up, z_cam); x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)

    R = np.stack([x_cam, y_cam, z_cam], axis=0).astype(np.float64)
    t = (-R @ src_xzy).astype(np.float64)
    return R, t


def _random_perturbation(rng: np.random.Generator) -> tuple:
    """Sample a 6-DOF perturbation: (delta_rot_deg, delta_trans_mm)."""
    rot   = rng.uniform(-ROT_RANGE_DEG,  ROT_RANGE_DEG,  3)
    trans = rng.uniform(-TRANS_RANGE_MM, TRANS_RANGE_MM, 3)
    return rot, trans


def _project_landmarks(lm_3d: dict, R: np.ndarray, t: np.ndarray,
                        img_size: int) -> dict:
    """
    Project 3D landmarks through (R, t) and return {name: (u, v)} for
    landmarks that are within image bounds (with margin).

    lm_3d may contain both centroid names (D12/L1-L5) and pedicle names
    (<LEVEL>_LEFT_PED / <LEVEL>_RIGHT_PED) -- both are projected exactly
    the same way, since this function only needs a name->3D-point dict.
    """
    names  = [n for n in ALL_LANDMARKS if n in lm_3d]
    pts3d  = np.array([lm_3d[n] for n in names])
    uv     = project_world(pts3d, R, t)

    margin = MIN_MARGIN_PX
    result = {}
    for name, (u, v) in zip(names, uv):
        if margin <= u < img_size - margin and margin <= v < img_size - margin:
            result[name] = (float(u), float(v))
    return result


def _drr_quality_ok(drr: np.ndarray) -> bool:
    """
    Return True if the DRR has enough bone content.

    NOTE: We rely on N_STEPS being set high enough (verified: 200 steps at
    256px render size) to avoid ray-marching aliasing artifacts on large CT
    volumes, rather than trying to statistically detect corrupted images
    post-hoc. A Laplacian/variance-based aliasing detector was tried and
    found ineffective — the artifacts are low-frequency correlated blobs,
    not high-frequency noise, so they don't show up in edge-energy ratios.
    """
    return float(np.count_nonzero(drr > 0.05)) / drr.size >= MIN_BONE_COVER


def _save_drr_png(drr: np.ndarray, path: Path):
    """Save float32 DRR [0,1] as uint8 PNG at SAVE_SIZE."""
    # Upscale from render size to save size if needed
    if drr.shape[0] != SAVE_SIZE:
        drr = cv2.resize(drr, (SAVE_SIZE, SAVE_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    img8 = (np.clip(drr, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img8)


def _scale_coords(coords: dict, render_size: int, save_size: int) -> dict:
    """Scale landmark pixel coords from render space to save space."""
    scale = save_size / render_size
    return {k: (u * scale, v * scale) for k, (u, v) in coords.items()}


def _coords_to_save_space(coords: dict, cam_img_size: int,
                           save_size: int) -> dict:
    """Scale landmark pixel coords from native camera space to save_size."""
    scale = save_size / cam_img_size
    return {k: (u * scale, v * scale) for k, (u, v) in coords.items()}


# ---------------------------------------------------------------------------
# Per-patient generation
# ---------------------------------------------------------------------------

def _load_extra_pedicle_landmarks(ct_dir: Path) -> dict:
    """
    Load the auto-extracted, non-collinear pedicle points (see
    scripts/extract_transverse_process_tips.py) for one patient, if they
    exist. Returns {} if EXTRA_LANDMARKS.mrk.json is missing (e.g. this
    patient wasn't processed, or extraction found no usable candidates --
    both are non-fatal, generation just proceeds with centroids only for
    that patient).
    """
    extra_path = ct_dir / 'EXTRA_LANDMARKS.mrk.json'
    if not extra_path.exists():
        return {}
    try:
        return _load_mrk_3d(extra_path)
    except Exception as e:
        print(f'  [WARNING] Failed to load {extra_path.name}: {e}')
        return {}


def _load_ct_only_specimen(patient_dir: Path):
    """
    Load a minimal DeepFluoroSpecimen from a CT-only patient folder
    (has 'CT label/<ct>.nrrd' + 'CENTROID.mrk.json' but NO 'XRAY Label/'
    subfolder — i.e. no manually-annotated X-ray sets, just CT + 3D
    centroids). Used for patients sourced from the "full data" export that
    have 3D landmarks but no paired 2D X-ray annotation.

    Returns a DeepFluoroSpecimen with an empty projections list — this is
    sufficient for DeepFluoroDRR rendering, since poses are generated
    fresh by the anatomy-pose fallback rather than read from real
    projections.
    """
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
    keep_labels = set(LUMBAR) | {'D12', 'D11', 'S1'}
    lm_3d = {k: v for k, v in lm_3d.items() if k in keep_labels}
    lm_3d.update(_load_extra_pedicle_landmarks(ct_dir))

    return DeepFluoroSpecimen(
        specimen_id=patient_dir.name,
        ct_volume=ct_vol,
        ct_spacing=spacing,
        ct_origin=origin,
        landmarks_3d=lm_3d,
        projections=[],
    )


def generate_patient(patient_dir: Path, n_per_view: int,
                     render_size: int, save_size: int,
                     hu_threshold: float, device: str,
                     seed: int = 42) -> list:
    """
    Generate synthetic DRRs for one patient.

    Returns list of manifest row dicts.
    """
    pid  = patient_dir.name
    rng  = np.random.default_rng(seed + hash(pid) % (2**31))

    print(f'\n{"-"*60}')
    print(f'  Patient: {pid}')
    print(f'{"-"*60}')

    # -- Load CT + landmarks -----------------------------------------------
    # Try the standard loader first (requires XRAY Label/ with real sets);
    # fall back to a CT-only minimal specimen if no XRAY Label/ exists
    # (used for the new "full data" patients that have 3D centroids only).
    has_xray_label = (patient_dir / 'XRAY Label').exists()
    try:
        if has_xray_label:
            loader = GenericPatientLoader(patient_dir)
            spec = loader.load(min_labels=3, verbose=False)
        else:
            spec = _load_ct_only_specimen(patient_dir)
            print(f'  (CT-only patient — no XRAY Label/, using anatomy-pose seeding)')
    except Exception as e:
        print(f'  [FAILED] Load failed: {e}')
        return []

    if not spec.landmarks_3d:
        print(f'  [FAILED] No 3D landmarks found')
        return []

    lm_3d = {k: v for k, v in spec.landmarks_3d.items() if k in LUMBAR}
    n_centroids = len(lm_3d)
    lm_3d.update(_load_extra_pedicle_landmarks(patient_dir / 'CT label'))
    n_pedicles = len(lm_3d) - n_centroids
    print(f'  3D landmarks: {n_centroids} centroids + {n_pedicles} pedicle points  '
          f'{sorted(lm_3d.keys())}')

    # -- Init DRR generator ------------------------------------------------
    gen = DeepFluoroDRR(spec, hu_threshold=hu_threshold, device=device)

    # -- Collect seed poses from existing real projections -----------------
    # We anchor random poses around the real calibrated poses — this keeps
    # the geometry anatomically valid (source outside body).
    seed_poses_ap  = [(p.R_proj, p.t_proj) for p in spec.projections
                      if 'AP' in p.proj_key.upper() and p.R_proj is not None]
    seed_poses_lat = [(p.R_proj, p.t_proj) for p in spec.projections
                      if 'LAT' in p.proj_key.upper() and p.R_proj is not None]

    # Fall back to anatomy pose if no real poses available for a view
    az_ap, az_range_ap   = VIEW_AZIMUTHS['AP']
    az_lat, az_range_lat = VIEW_AZIMUTHS['LAT']

    if not seed_poses_ap:
        az = rng.uniform(az_ap - az_range_ap, az_ap + az_range_ap)
        R0, t0 = _anatomy_pose_random(lm_3d, az, rng)
        seed_poses_ap = [(R0, t0)]
    if not seed_poses_lat:
        az = rng.uniform(az_lat - az_range_lat, az_lat + az_range_lat)
        R0, t0 = _anatomy_pose_random(lm_3d, az, rng)
        seed_poses_lat = [(R0, t0)]

    manifest_rows = []

    for view, seed_poses in [('AP', seed_poses_ap), ('LAT', seed_poses_lat)]:
        out_dir = OUT_ROOT / pid / view
        out_dir.mkdir(parents=True, exist_ok=True)

        n_ok    = 0
        n_tried = 0
        idx     = 0
        # Max attempts = 5× requested to handle quality rejects
        max_tries = n_per_view * 5

        t0_view = time.time()
        print(f'  Generating {view}: target={n_per_view} ...', flush=True)

        while n_ok < n_per_view and n_tried < max_tries:
            n_tried += 1

            # -- Sample pose -----------------------------------------------
            # Pick a random seed pose, then perturb it
            R_seed, t_seed = seed_poses[rng.integers(len(seed_poses))]
            rot_delta, trans_delta = _random_perturbation(rng)
            R, t = perturb_extrinsic(R_seed, t_seed, rot_delta, trans_delta)

            # -- Project landmarks at NATIVE camera resolution -------------
            coords_native = _project_landmarks(lm_3d, R, t, CAM_IMG_SIZE)
            if len(coords_native) < MIN_VISIBLE_LM:
                continue

            # -- Render DRR ------------------------------------------------
            pix_mm = CAM_PIX_MM * (CAM_IMG_SIZE / render_size)
            try:
                drr = gen.generate_from_extrinsic(R, t, render_size,
                                                   pix_mm, N_STEPS)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise

            if not _drr_quality_ok(drr):
                continue

            # -- Scale coords to save_size space --------------------------
            coords_save = _coords_to_save_space(coords_native,
                                                 CAM_IMG_SIZE, save_size)

            # -- Save DRR PNG ----------------------------------------------
            img_path = out_dir / f'{idx:05d}.png'
            _save_drr_png(drr, img_path)

            # -- Save label CSV --------------------------------------------
            csv_path = out_dir / f'{idx:05d}_labels.csv'
            with open(csv_path, 'w', newline='') as cf:
                w = csv.writer(cf)
                w.writerow(['label', 'u_px', 'v_px'])
                for name, (u, v) in sorted(coords_save.items()):
                    w.writerow([name, f'{u:.2f}', f'{v:.2f}'])

            # -- Build manifest row ----------------------------------------
            labels_str = '|'.join(sorted(coords_save.keys()))
            # coords_px format: "u,v;u,v;..." sorted by label
            coords_str = ';'.join(
                f'{u:.2f},{v:.2f}'
                for _, (u, v) in sorted(coords_save.items())
            )
            view_tag = f'XRAY-{view}'
            manifest_rows.append({
                'patient':    f'SYN_{pid}',
                'set_name':   f'{view}_{idx:05d}',
                'view':       view_tag,
                'img_path':   str(img_path.relative_to(ROOT)).replace('\\', '/'),
                'lm_path':    str(csv_path.relative_to(ROOT)).replace('\\', '/'),
                'labels':     labels_str,
                'coords_px':  coords_str,
            })

            n_ok  += 1
            idx   += 1

            if n_ok % 50 == 0:
                elapsed = time.time() - t0_view
                rate    = n_ok / elapsed
                eta     = (n_per_view - n_ok) / max(rate, 1e-6)
                print(f'    {n_ok}/{n_per_view}  '
                      f'({rate:.1f} DRR/s, ETA {eta:.0f}s)  '
                      f'reject_rate={(n_tried-n_ok)/n_tried*100:.0f}%',
                      flush=True)

        elapsed = time.time() - t0_view
        print(f'  {view}: generated {n_ok}/{n_per_view} '
              f'in {elapsed:.0f}s  ({n_ok/elapsed:.1f} DRR/s)  '
              f'tried={n_tried}')

        torch.cuda.empty_cache()

    return manifest_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate synthetic DRRs from patient CTs for SSM regressor training.')
    p.add_argument('--patients', nargs='+', default=None,
                   help='Patient names to process. Default: all.')
    p.add_argument('--n_per_view', type=int, default=500,
                   help='DRRs to generate per view (AP + LAT) per patient. Default: 500.')
    p.add_argument('--render_size', type=int, default=RENDER_SIZE,
                   help=f'DRR render resolution (default: {RENDER_SIZE}). '
                        'Saved PNGs are always 512px.')
    p.add_argument('--save_size', type=int, default=SAVE_SIZE,
                   help=f'Saved PNG resolution (default: {SAVE_SIZE}).')
    p.add_argument('--hu_threshold', type=float, default=150.0,
                   help='HU threshold for bone-focused DRR (default: 150). '
                        'Lower = more soft tissue included. 150 verified to '
                        'give clean bone contrast without washing out spine detail.')
    p.add_argument('--device', default='cuda',
                   help='PyTorch device (default: cuda).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--append', action='store_true',
                   help='Append to existing manifest instead of overwriting.')
    return p.parse_args()


def main():
    args = parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Discover patients from both the original annotated-patients folder
    # AND the new CT-only patients folder (3D centroids, no X-ray
    # annotation — sourced from the "full data" export). Excludes patients
    # with confirmed unresolvable DRR rendering artifacts (see
    # EXCLUDED_PATIENTS module constant).
    patient_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()
                           and d.name not in EXCLUDED_PATIENTS])
    if CT_ONLY_ROOT.exists():
        patient_dirs += sorted([d for d in CT_ONLY_ROOT.iterdir() if d.is_dir()
                                and d.name not in EXCLUDED_PATIENTS])
    if args.patients:
        patient_dirs = [d for d in patient_dirs if d.name in args.patients]

    if not patient_dirs:
        print('No patient directories found.')
        return

    print(f'{"="*60}')
    print(f'  SYNTHETIC DRR GENERATION')
    print(f'  Patients : {[d.name for d in patient_dirs]}')
    print(f'  Per view : {args.n_per_view}')
    print(f'  Render   : {args.render_size}px -> saved at {args.save_size}px')
    print(f'  HU thr   : {args.hu_threshold}')
    print(f'  Device   : {args.device}')
    print(f'  Output   : {OUT_ROOT}')
    print(f'{"="*60}')

    all_rows  = []
    t0_global = time.time()

    for patient_dir in patient_dirs:
        rows = generate_patient(
            patient_dir,
            n_per_view    = args.n_per_view,
            render_size   = args.render_size,
            save_size     = args.save_size,
            hu_threshold  = args.hu_threshold,
            device        = args.device,
            seed          = args.seed,
        )
        all_rows.extend(rows)

    # -- Write manifest ----------------------------------------------------
    mode = 'a' if args.append and MANIFEST.exists() else 'w'
    fieldnames = ['patient', 'set_name', 'view', 'img_path',
                  'lm_path', 'labels', 'coords_px']

    existing_rows = []
    if mode == 'a':
        with open(MANIFEST, newline='') as f:
            existing_rows = list(csv.DictReader(f))

    with open(MANIFEST, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in existing_rows + all_rows:
            w.writerow(row)

    total_time = time.time() - t0_global
    n_ap  = sum(1 for r in all_rows if 'AP'  in r['view'])
    n_lat = sum(1 for r in all_rows if 'LAT' in r['view'])

    print(f'\n{"="*60}')
    print(f'  DONE')
    print(f'  Total DRRs generated : {len(all_rows)}  (AP={n_ap}  LAT={n_lat})')
    print(f'  Manifest written to  : {MANIFEST}')
    print(f'  Total time           : {total_time:.0f}s  '
          f'({len(all_rows)/total_time:.1f} DRR/s overall)')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()

