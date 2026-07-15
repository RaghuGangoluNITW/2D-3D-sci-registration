#!/usr/bin/env python3
"""
sample_loss_pde.py — Multi-Start Loss vs PDE Correlation Sampler
=================================================================
Samples N random poses around the EPnP-initialised extrinsic for each
X-ray projection, evaluates 13 loss functions and PDE at every pose.

Supports two datasets:
  --dataset deepfluoro   — cadaveric pelvis benchmark
  --dataset swaroopa     (default) — real Swaroopa patient C-arm images

Two output CSVs:
  --out      <path>  Random-pose samples (main correlation dataset)
  --epnp_out <path>  EPnP-pose row per projection (loss at initialization)

Usage:
  # Swaroopa — all annotated frames, 50 random poses each (default)
  python sample_loss_pde.py --n_poses 50

  # Swaroopa AP frames only
  python sample_loss_pde.py --frames ap --n_poses 100

  # DeepFluoro specimen 01, 3 poses (quick test)
  python sample_loss_pde.py --dataset deepfluoro --specimen 01 --n_poses 3 --max_proj 2

Output CSV columns (random poses):
  specimen, proj_key, sample_idx,
  rot_delta_{x,y,z}   (degrees, applied to EPnP pose),
  trans_delta_{x,y,z} (mm, applied to EPnP pose),
  pde_mean_mm, pde_max_mm,
  ncc_cost, go_cost, lncc_cost, ms_ncc_cost, grad_ncc_cost,
  mi_cost, nmi_cost, eod_cost, cr_cost, src_cost,
  gd_cost, pi_cost, grad_ms_ncc_cost,
  drr_coverage, [pde_{lm_name} per landmark]

EPnP CSV columns (one row per projection, delta=0):
  specimen, proj_key, reproj_error_px,
  pde_mean_mm, pde_max_mm, <same 13 loss columns>,
  drr_coverage, [pde_{lm_name} per landmark]
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from deepfluoro_loader import (
    DeepFluoroLoader,
    DeepFluoroSpecimen,
    DeepFluoroProjection,
    SPECIMEN_NAMES,
    SPECIMEN_MAP,
    PIXEL_SPACING_MM    as _DF_PIX_MM,
    FULL_RES_SIZE       as _DF_FULL_SIZE,
    perturb_extrinsic,
    compute_pde_for_pose,
)
from deepfluoro_drr import DeepFluoroDRR
from similarity import (
    go_cost as _go_cost,
    normalized_cross_correlation,
    mutual_information,
    local_ncc,
    multiscale_ncc,
    gradient_ncc,
    normalised_mutual_information,
    entropy_of_difference,
    correlation_ratio,
    stochastic_rank_correlation,
    gradient_difference,
    pattern_intensity,
    gradient_multiscale_ncc,
    normalised_gradient_information,
    local_gradient_ncc,
)
from swaroopa_loader import (
    SwaroLoader,
    SwaroProjection,
    compute_pde_swaro,
    SWARO_PIX_MM,
    SWARO_IMG_SIZE,
)
# DiffDRR renderer (Siddon ray-casting) — used for Swaroopa
from run_swaroopa_diffdrr import (
    DiffDRRGenerator,
    build_subject,
    build_subject_masked,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_SIZE  = 180
EVAL_STEPS = 80


# ---------------------------------------------------------------------------
# Dataset adapter — abstracts PDE computation and pixel spacing
# ---------------------------------------------------------------------------

class _DatasetAdapter:
    """Thin shim so sample_projection() is dataset-agnostic."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        if dataset == 'swaroopa':
            self.pix_mm    = SWARO_PIX_MM
            self.full_size = SWARO_IMG_SIZE
        else:
            self.pix_mm    = _DF_PIX_MM
            self.full_size = _DF_FULL_SIZE

    def compute_pde(self, proj, R, t, pts3d, lm_names) -> Dict[str, float]:
        if self.dataset == 'swaroopa':
            return compute_pde_swaro(proj, R, t, pts3d, lm_names)
        else:
            return compute_pde_for_pose(proj, R, t, pts3d, lm_names)


# All loss function names (in CSV column order)
ALL_LOSS_COLS = [
    'ncc_cost',
    'go_cost',
    'lncc_cost',
    'ms_ncc_cost',
    'grad_ncc_cost',
    'mi_cost',
    'nmi_cost',
    'eod_cost',
    'cr_cost',
    'src_cost',
    'gd_cost',
    'pi_cost',
    'grad_ms_ncc_cost',
    'ngi_cost',
    'lgncc_cost',
]

_SENTINEL = {col: float('nan') for col in ALL_LOSS_COLS}

# Mutable config set by main() before sampling starts
_CFG = {
    'suppress_highlights': False,
    'hl_threshold':   0.6,
    'hl_darken':      0.8,
    'hl_feather_sigma': 31,
    'hl_min_blob_px': 500,
    'clahe':          False,
    'clahe_clip':     2.0,
    'clahe_grid':     8,
}


def _suppress_highlights(img: np.ndarray) -> np.ndarray:
    """Darken bright-highlight blobs in a float32 [0,1] grayscale image.

    Mirrors scripts/visualize_before_after_epnp.py::suppress_highlights().
    """
    threshold    = _CFG['hl_threshold']
    darken       = _CFG['hl_darken']
    feather_sigma = _CFG['hl_feather_sigma']
    min_blob_px  = _CFG['hl_min_blob_px']

    u8 = (img * 255).clip(0, 255).astype(np.uint8)
    _, hard_mask = cv2.threshold(u8, int(threshold * 255), 255, cv2.THRESH_BINARY)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        hard_mask, connectivity=8)
    clean_mask = np.zeros_like(hard_mask)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_blob_px:
            clean_mask[labels == lbl] = 255
    k = feather_sigma | 1          # ensure odd
    soft_mask = cv2.GaussianBlur(
        clean_mask.astype(np.float32),
        (k * 4 + 1, k * 4 + 1), k)
    if soft_mask.max() > 0:
        soft_mask = (soft_mask / soft_mask.max()).clip(0, 1)
    scale = 1.0 - soft_mask * (1.0 - darken)
    return (img * scale).clip(0, 1).astype(np.float32)


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to a
    float32 [0,1] grayscale image. Returns float32 [0,1]."""
    clip  = _CFG['clahe_clip']
    grid  = _CFG['clahe_grid']
    u8    = (img * 255).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    eq    = clahe.apply(u8)
    return eq.astype(np.float32) / 255.0


def _eval_metrics(drr_gen: DeepFluoroDRR,
                  R_cand: np.ndarray,
                  t_cand: np.ndarray,
                  target: np.ndarray,
                  pix_mm_full: float,
                  full_size: int) -> dict:
    """Render DRR at (R_cand, t_cand) and return all loss function costs."""
    pix_mm = pix_mm_full * (full_size / EVAL_SIZE)
    drr = drr_gen.generate_from_extrinsic(R_cand, t_cand, EVAL_SIZE, pix_mm, EVAL_STEPS)

    cov = float(np.count_nonzero(drr > 0.01)) / drr.size
    if cov < 0.05:
        return {**_SENTINEL, 'drr_coverage': cov}

    d   = drr
    # Swaroopa X-rays are stored dark-bone (bright = air); DiffDRR is bright-bone.
    # Invert the target so both have bright = bone, matching the DRR polarity.
    # (mirrors run_swaroopa_diffdrr.py lines 420-422: tgt = 1.0 - raw)
    tgt = 1.0 - target
    if _CFG['suppress_highlights']:
        tgt = _suppress_highlights(tgt)
    if _CFG['clahe']:
        tgt = _apply_clahe(tgt)

    ncc_v   = float(normalized_cross_correlation(d, tgt))
    go_v    = float(_go_cost(d, tgt))
    lncc_v  = float(local_ncc(d, tgt))
    ms_ncc  = float(multiscale_ncc(d, tgt))
    g_ncc   = float(gradient_ncc(d, tgt))
    mi_raw  = float(mutual_information(d, tgt))
    mi_v    = float(np.clip(1.0 - mi_raw / (np.log(32) + 1e-10), 0.0, 1.0))
    nmi_v   = float(normalised_mutual_information(d, tgt))
    eod_v   = float(entropy_of_difference(d, tgt))
    cr_v    = float(correlation_ratio(d, tgt))
    src_v   = float(stochastic_rank_correlation(d, tgt))
    gd_v    = float(gradient_difference(d, tgt))
    pi_v    = float(pattern_intensity(d, tgt))
    gms_ncc = float(gradient_multiscale_ncc(d, tgt))
    ngi_v   = float(normalised_gradient_information(d, tgt))
    lgncc_v = float(local_gradient_ncc(d, tgt))

    return {
        'ncc_cost':         float(1.0 - np.clip(ncc_v, -1.0, 1.0)),
        'go_cost':          go_v,
        'lncc_cost':        lncc_v,
        'ms_ncc_cost':      ms_ncc,
        'grad_ncc_cost':    g_ncc,
        'mi_cost':          mi_v,
        'nmi_cost':         nmi_v,
        'eod_cost':         eod_v,
        'cr_cost':          cr_v,
        'src_cost':         src_v,
        'gd_cost':          gd_v,
        'pi_cost':          pi_v,
        'grad_ms_ncc_cost': gms_ncc,
        'ngi_cost':         ngi_v,
        'lgncc_cost':       lgncc_v,
        'drr_coverage':     cov,
    }


# ---------------------------------------------------------------------------
# Sample poses for one projection
# ---------------------------------------------------------------------------

def sample_projection(
        drr_gen:        DeepFluoroDRR,
        proj:           DeepFluoroProjection,
        lm_names:       List[str],
        pts3d:          np.ndarray,
        n_poses:        int,
        rot_sigma_deg:  float,
        trans_sigma_mm: float,
        rng:            np.random.Generator,
        adapter:        _DatasetAdapter,
) -> List[dict]:
    """
    Draw n_poses random perturbations from the EPnP pose of `proj` and
    evaluate losses against that same X-ray only.
    Returns n_poses row-dicts.
    """
    target      = cv2.resize(proj.image_raw, (EVAL_SIZE, EVAL_SIZE),
                             interpolation=cv2.INTER_AREA)
    lm_name_set = sorted(lm_names)
    rows        = []

    for i in range(n_poses):
        delta_rot   = rng.uniform(-rot_sigma_deg,   rot_sigma_deg,   3)
        delta_trans = rng.uniform(-trans_sigma_mm,  trans_sigma_mm,  3)

        R_cand, t_cand = perturb_extrinsic(
            proj.R_proj, proj.t_proj, delta_rot, delta_trans
        )

        pde_dict = adapter.compute_pde(proj, R_cand, t_cand, pts3d, lm_names)
        pde_vals = list(pde_dict.values())
        pde_mean = float(np.mean(pde_vals)) if pde_vals else float('nan')
        pde_max  = float(np.max(pde_vals))  if pde_vals else float('nan')

        metrics = _eval_metrics(drr_gen, R_cand, t_cand, target,
                                adapter.pix_mm, adapter.full_size)

        row = {
            'specimen':      proj.specimen_id,
            'proj_key':      proj.proj_key,
            'sample_idx':    i,
            'rot_delta_x':   float(delta_rot[0]),
            'rot_delta_y':   float(delta_rot[1]),
            'rot_delta_z':   float(delta_rot[2]),
            'trans_delta_x': float(delta_trans[0]),
            'trans_delta_y': float(delta_trans[1]),
            'trans_delta_z': float(delta_trans[2]),
            'pde_mean_mm':   pde_mean,
            'pde_max_mm':    pde_max,
            **{col: metrics[col] for col in ALL_LOSS_COLS},
            'drr_coverage':  metrics['drr_coverage'],
        }
        for nm in lm_name_set:
            row[f'pde_{nm}'] = pde_dict.get(nm, float('nan'))
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Evaluate losses at the EPnP pose (delta = 0)
# ---------------------------------------------------------------------------

def eval_epnp_pose(
        drr_gen:  DeepFluoroDRR,
        proj:     DeepFluoroProjection,
        lm_names: List[str],
        pts3d:    np.ndarray,
        adapter:  _DatasetAdapter,
) -> dict:
    """Evaluate all 13 losses at the EPnP pose against the same X-ray."""
    target   = cv2.resize(proj.image_raw, (EVAL_SIZE, EVAL_SIZE),
                          interpolation=cv2.INTER_AREA)
    pde_dict = adapter.compute_pde(proj, proj.R_proj, proj.t_proj, pts3d, lm_names)
    pde_vals = list(pde_dict.values())
    pde_mean = float(np.mean(pde_vals)) if pde_vals else float('nan')
    pde_max  = float(np.max(pde_vals))  if pde_vals else float('nan')
    metrics  = _eval_metrics(drr_gen, proj.R_proj, proj.t_proj, target,
                             adapter.pix_mm, adapter.full_size)
    row = {
        'specimen':        proj.specimen_id,
        'proj_key':        proj.proj_key,
        'reproj_error_px': float(proj.reproj_error_px),
        'pde_mean_mm':     pde_mean,
        'pde_max_mm':      pde_max,
        **{col: metrics[col] for col in ALL_LOSS_COLS},
        'drr_coverage':    metrics['drr_coverage'],
    }
    for nm in sorted(lm_names):
        row[f'pde_{nm}'] = pde_dict.get(nm, float('nan'))
    return row


# ---------------------------------------------------------------------------
# Dataset loader helper
# ---------------------------------------------------------------------------

def _load_dataset(args):
    """
    Returns list of (specimen, valid_projs, lm_names, pts3d, adapter).
    """
    adapter = _DatasetAdapter(args.dataset)
    items   = []

    if args.dataset == 'swaroopa':
        print('[Swaroopa] Loading specimen ...')
        loader = SwaroLoader()

        # Optional frame filter
        frames_filter = None
        if args.frames:
            if args.frames in ('ap', 'lat'):
                # Load all first, then filter by prefix
                tmp = loader.load(verbose=False)
                frames_filter = [p.proj_key for p in tmp.projections
                                 if p.proj_key.startswith(args.frames + '_')]
            else:
                frames_filter = args.frames.split(',')

        specimen    = loader.load(frames=frames_filter, verbose=True)
        valid_projs = [p for p in specimen.projections
                       if p.R_proj is not None][:args.max_proj]
        lm_names    = sorted(specimen.landmarks_3d.keys())
        pts3d       = np.array([specimen.landmarks_3d[n] for n in lm_names])
        items.append((specimen, valid_projs, lm_names, pts3d, adapter))

    else:  # deepfluoro
        loader = DeepFluoroLoader()
        if args.all:
            spec_list = SPECIMEN_NAMES
        else:
            key = args.specimen
            spec_list = [SPECIMEN_MAP.get(key, key)]

        for spec_name in spec_list:
            print(f'\n[DeepFluoro] Loading specimen: {spec_name}')
            specimen    = loader.load_specimen(
                spec_name, max_projections=args.max_proj, verbose=False
            )
            valid_projs = specimen.valid_projections()[:args.max_proj]
            lm_names, pts3d = specimen.get_landmark_array()
            items.append((specimen, valid_projs, lm_names, pts3d, adapter))

    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Sample loss vs PDE for correlation analysis'
    )
    parser.add_argument('--dataset',     default='swaroopa',
                        choices=['deepfluoro', 'swaroopa'],
                        help='Dataset to use (default: swaroopa)')
    # DeepFluoro options
    parser.add_argument('--specimen',    default='01',
                        help='[deepfluoro] Specimen index 01-06 or name')
    parser.add_argument('--all',         action='store_true',
                        help='[deepfluoro] Run all 6 specimens')
    # Swaroopa options
    parser.add_argument('--frames',      default=None,
                        help='[swaroopa] Filter: ap / lat / comma-separated proj_keys')
    # Shared
    parser.add_argument('--max_proj',    type=int, default=999,
                        help='Max projections (default: all)')
    parser.add_argument('--n_poses',     type=int, default=50,
                        help='Random poses per projection (default 50)')
    parser.add_argument('--rot_sigma',   type=float, default=15.0,
                        help='Rotation perturbation ± degrees (default 15)')
    parser.add_argument('--trans_sigma', type=float, default=30.0,
                        help='Translation perturbation ± mm (default 30)')
    parser.add_argument('--seed',        type=int, default=0,
                        help='RNG seed (default 0)')
    parser.add_argument('--out',         default='results/loss_pde_samples.csv',
                        help='Output CSV for random-pose samples')
    parser.add_argument('--epnp_out',    default='results/loss_pde_epnp.csv',
                        help='Output CSV for EPnP-pose baselines')
    parser.add_argument('--drr_backend',  default='diffdrr',
                        choices=['diffdrr', 'deepfluoro'],
                        help='DRR renderer backend: diffdrr (Siddon, default) '
                             'or deepfluoro (original C-arm model)')
    parser.add_argument('--suppress_highlights', action='store_true',
                        help='Apply highlight suppression to X-ray before '
                             'computing losses (threshold=0.6, darken=0.8)')
    parser.add_argument('--hl_threshold',    type=float, default=0.6,
                        help='Highlight suppression brightness threshold (default 0.6)')
    parser.add_argument('--hl_darken',       type=float, default=0.8,
                        help='Darken factor for highlights (default 0.8)')
    parser.add_argument('--hl_feather_sigma', type=int,  default=31,
                        help='Gaussian feather sigma for highlight mask (default 31)')
    parser.add_argument('--hl_min_blob_px',  type=int,  default=500,
                        help='Min highlight blob size in pixels (default 500)')
    parser.add_argument('--clahe', action='store_true',
                        help='Apply CLAHE to X-ray after highlight suppression')
    parser.add_argument('--clahe_clip', type=float, default=2.0,
                        help='CLAHE clip limit (default 2.0)')
    parser.add_argument('--clahe_grid', type=int,   default=8,
                        help='CLAHE tile grid size (default 8)')
    parser.add_argument('--device',      default=None,
                        help='cuda / cpu (default: auto)')
    parser.add_argument('--min_hu',        type=float, default=None,
                        help='[diffdrr] Lower HU clip inside cylinder mask '
                             '(requires --cylinder_radius; default: None = disabled)')
    parser.add_argument('--cylinder_radius', type=float, default=None,
                        help='[diffdrr] Cylinder mask radius (mm) around '
                             'spine centroid; enables build_subject_masked '
                             '(default: None = plain build_subject)')

    # Compute timestamped defaults before parsing so they appear in --help
    _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser.set_defaults(
        out      =f'results/loss_pde_samples_{_ts}.csv',
        epnp_out =f'results/loss_pde_epnp_{_ts}.csv',
    )

    args = parser.parse_args()

    out_path      = Path(args.out)
    epnp_out_path = Path(args.epnp_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epnp_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure highlight suppression
    _CFG['suppress_highlights'] = args.suppress_highlights
    _CFG['hl_threshold']        = args.hl_threshold
    _CFG['hl_darken']           = args.hl_darken
    _CFG['hl_feather_sigma']    = args.hl_feather_sigma
    _CFG['hl_min_blob_px']      = args.hl_min_blob_px
    _CFG['clahe']               = args.clahe
    _CFG['clahe_clip']          = args.clahe_clip
    _CFG['clahe_grid']          = args.clahe_grid
    if args.suppress_highlights:
        print(f'[highlight suppression ON]  threshold={args.hl_threshold}  '
              f'darken={args.hl_darken}  feather_sigma={args.hl_feather_sigma}  '
              f'min_blob_px={args.hl_min_blob_px}')
    if args.clahe:
        print(f'[CLAHE ON]  clip={args.clahe_clip}  grid={args.clahe_grid}')

    rng           = np.random.default_rng(args.seed)
    dataset_items = _load_dataset(args)

    if not dataset_items:
        print('No data loaded. Exiting.')
        return

    # Build fieldnames from first specimen's landmarks
    _, _, lm_names_0, _, _ = dataset_items[0]
    lm_cols = [f'pde_{nm}' for nm in sorted(lm_names_0)]

    sample_fields = (
        ['specimen', 'proj_key', 'sample_idx',
         'rot_delta_x', 'rot_delta_y', 'rot_delta_z',
         'trans_delta_x', 'trans_delta_y', 'trans_delta_z',
         'pde_mean_mm', 'pde_max_mm']
        + ALL_LOSS_COLS + ['drr_coverage'] + lm_cols
    )
    epnp_fields = (
        ['specimen', 'proj_key', 'reproj_error_px',
         'pde_mean_mm', 'pde_max_mm']
        + ALL_LOSS_COLS + ['drr_coverage'] + lm_cols
    )

    t0_global  = time.time()
    total_rows = 0
    total_epnp = 0

    fh_s = open(out_path,      'w', newline='')
    fh_e = open(epnp_out_path, 'w', newline='')
    try:
        writer_s = csv.DictWriter(fh_s, fieldnames=sample_fields, extrasaction='ignore')
        writer_e = csv.DictWriter(fh_e, fieldnames=epnp_fields,   extrasaction='ignore')
        writer_s.writeheader()
        writer_e.writeheader()

        for specimen, valid_projs, lm_names, pts3d, adapter in dataset_items:
            print(f"\n{'='*60}")
            print(f"Specimen : {specimen.specimen_id}  "
                  f"({len(valid_projs)} projs, {len(lm_names)} landmarks)")
            t0_spec = time.time()

            if args.drr_backend == 'diffdrr':
                import torch
                _dev   = torch.device(args.device if args.device else
                                      ('cuda' if torch.cuda.is_available() else 'cpu'))
                if args.cylinder_radius is not None:
                    hu_min = args.min_hu if args.min_hu is not None else 0.0
                    print(f"  Building DiffDRR subject with cylinder mask "
                          f"(r={args.cylinder_radius}mm, min_hu={hu_min}) ...")
                    _subj = build_subject_masked(
                        specimen,
                        cylinder_r_mm=args.cylinder_radius,
                        hu_min=hu_min,
                    )
                else:
                    print(f"  Building DiffDRR (Siddon) subject from CT ...")
                    _subj = build_subject(specimen)
                drr_gen = DiffDRRGenerator(_subj, _dev,
                                           ct_origin_lps=specimen.ct_origin)
            else:
                print(f"  Building DeepFluoro DRR renderer ...")
                drr_gen = DeepFluoroDRR(specimen, device=args.device)

            for proj_idx, proj in enumerate(valid_projs):
                t0_proj = time.time()

                # ── EPnP baseline row ──────────────────────────────────────
                print(f"  [{proj_idx+1}/{len(valid_projs)}] {proj.proj_key}  "
                      f"reproj={proj.reproj_error_px:.2f}px  "
                      f"EPnP eval...", end=' ', flush=True)
                epnp_row = eval_epnp_pose(drr_gen, proj, lm_names, pts3d, adapter)
                writer_e.writerow(epnp_row)
                fh_e.flush()
                total_epnp += 1
                print(f"PDE={epnp_row['pde_mean_mm']:.1f}mm  "
                      f"go={epnp_row['go_cost']:.4f}  "
                      f"sampling {args.n_poses}...", end=' ', flush=True)

                # ── Random-pose samples ────────────────────────────────────
                rows = sample_projection(
                    drr_gen, proj, lm_names, pts3d,
                    n_poses=args.n_poses,
                    rot_sigma_deg=args.rot_sigma,
                    trans_sigma_mm=args.trans_sigma,
                    rng=rng,
                    adapter=adapter,
                )
                writer_s.writerows(rows)
                fh_s.flush()

                elapsed     = time.time() - t0_proj
                total_rows += len(rows)
                rate        = len(rows) / max(elapsed, 1e-3)
                print(f"done  [{elapsed:.1f}s  {rate:.1f} samp/s]")

            print(f"  Specimen done in {time.time()-t0_spec:.1f}s")

    finally:
        fh_s.close()
        fh_e.close()

    total_time = time.time() - t0_global
    print(f"\n{'='*60}")
    print(f"Random-pose samples : {total_rows} rows  → {out_path}")
    print(f"EPnP baselines      : {total_epnp} rows  → {epnp_out_path}")
    print(f"Total time          : {total_time:.1f}s  "
          f"({total_rows/max(total_time,1e-3):.1f} samp/s overall)")


if __name__ == '__main__':
    main()
