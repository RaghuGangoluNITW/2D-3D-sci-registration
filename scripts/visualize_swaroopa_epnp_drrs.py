#!/usr/bin/env python3
"""
visualize_swaroopa_epnp_drrs.py
================================
Side-by-side grid of inverted X-ray vs EPnP-pose DRR for every Swaroopa frame.

Reads DRRs and pose JSON produced by export_swaroopa_epnp_drrs.py (DiffDRR).

Layout: N rows × 2 cols
  Col 0 — Inverted X-ray (1 – image_raw, so bone appears bright)
  Col 1 — EPnP DRR (as saved; normalised to [0,1])

Each row is labelled with the proj_key, reprojection error, and init method.

Usage:
    python scripts/visualize_swaroopa_epnp_drrs.py
    python scripts/visualize_swaroopa_epnp_drrs.py \
        --drr_dir  results/swaroopa_epnp_drrs_diffdrr \
        --poses_json results/swaroopa_epnp_poses_diffdrr.json \
        --output   results/figures/swaroopa_epnp_drr_comparison.png \
        --cols 4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from swaroopa_loader import SwaroLoader, SWARO_IMG_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gray_float(path: Path) -> np.ndarray:
    """Load a grayscale PNG as float32 [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# X-ray processing pipeline
# ---------------------------------------------------------------------------

ALL_PROCS = ['none', 'clahe', 'blur', 'histmatch', 'gamma', 'percentile']

PROC_LABELS = {
    'none':       'Inverted (no processing)',
    'clahe':      'Inverted + CLAHE',
    'blur':       'Inverted + Gaussian blur (σ=1.5)',
    'histmatch':  'Inverted + histogram match to DRR',
    'gamma':      'Inverted + gamma correction (γ=0.7)',
    'percentile': 'Percentile stretch [1–99%] + invert',
}


def process_xray(xray_raw: np.ndarray,
                 mode: str,
                 drr_ref: np.ndarray = None) -> np.ndarray:
    """
    Apply *mode* processing to a float32 [0,1] raw X-ray (bone dark)
    and return a float32 [0,1] result ready for display (bone bright).
    """
    if mode == 'none':
        return np.clip(1.0 - xray_raw, 0.0, 1.0)

    if mode == 'percentile':
        lo, hi = np.percentile(xray_raw, [1, 99])
        stretched = np.clip((xray_raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return np.clip(1.0 - stretched, 0.0, 1.0)

    # All remaining modes start from simple inversion
    inv = np.clip(1.0 - xray_raw, 0.0, 1.0)

    if mode == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        u8 = (inv * 255.0).clip(0, 255).astype(np.uint8)
        return clahe.apply(u8).astype(np.float32) / 255.0

    if mode == 'blur':
        return cv2.GaussianBlur(inv, (0, 0), sigmaX=1.5)

    if mode == 'gamma':
        return np.power(inv.clip(0.0, 1.0), 0.7).astype(np.float32)

    if mode == 'histmatch':
        if drr_ref is None:
            return inv
        src_u8 = (inv     * 255.0).clip(0, 255).astype(np.uint8).ravel()
        ref_u8 = (drr_ref * 255.0).clip(0, 255).astype(np.uint8).ravel()
        src_hist, _ = np.histogram(src_u8, 256, [0, 256])
        ref_hist, _ = np.histogram(ref_u8, 256, [0, 256])
        src_cdf = src_hist.cumsum().astype(np.float64)
        ref_cdf = ref_hist.cumsum().astype(np.float64)
        src_cdf /= src_cdf[-1]
        ref_cdf /= ref_cdf[-1]
        lut = np.zeros(256, dtype=np.uint8)
        j = 0
        for i in range(256):
            while j < 255 and ref_cdf[j] < src_cdf[i]:
                j += 1
            lut[i] = j
        matched = lut[(inv * 255.0).clip(0, 255).astype(np.uint8)]
        return matched.astype(np.float32) / 255.0

    raise ValueError(f"Unknown xray_proc mode: {mode}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Visualise processed X-ray vs EPnP DRR for all Swaroopa frames'
    )
    p.add_argument('--drr_dir',    type=Path,
                   default=Path('results/swaroopa_epnp_drrs_diffdrr'),
                   help='Root folder of saved DRRs')
    p.add_argument('--poses_json', type=Path,
                   default=Path('results/swaroopa_epnp_poses_diffdrr.json'),
                   help='Poses JSON written by export_swaroopa_epnp_drrs.py')
    p.add_argument('--xray_dir',   type=Path,
                   default=Path('data/swaroopa_labelled'),
                   help='Root of X-ray data (contains ap/ and lateral/)')
    p.add_argument('--output',     type=Path,
                   default=Path('results/figures/swaroopa_epnp_drr_comparison.png'),
                   help='Output PNG path (ignored when --all_procs is set)')
    p.add_argument('--frames',     nargs='+', default=None,
                   help='Subset of proj_keys to visualise (default: all)')
    p.add_argument('--cols',       type=int, default=4,
                   help='Number of frame columns in the grid (default: 4)')
    p.add_argument('--thumb_size', type=int, default=200,
                   help='Thumbnail size in pixels for each image (default: 200)')
    p.add_argument('--dpi',        type=int, default=150)
    p.add_argument('--xray_proc',  type=str, default='none',
                   choices=ALL_PROCS,
                   help='X-ray processing mode (default: none)')
    p.add_argument('--all_procs',  action='store_true',
                   help='Generate one image per processing mode (6 total)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core grid renderer
# ---------------------------------------------------------------------------

def render_grid(all_keys, frame_records, proj_by_key,
                drr_dir, xray_dir,
                xray_proc, output, cols, thumb_size, dpi):
    """Render and save one comparison grid PNG."""
    N  = len(all_keys)
    TS = thumb_size
    NCOLS_FRAMES = cols
    NCOLS_IMGS   = NCOLS_FRAMES * 2
    NROWS        = (N + NCOLS_FRAMES - 1) // NCOLS_FRAMES

    fig_w = NCOLS_IMGS * (TS / dpi) * 1.05
    fig_h = NROWS      * (TS / dpi) * 1.35
    fig, axes = plt.subplots(NROWS, NCOLS_IMGS,
                             figsize=(max(fig_w, 8), max(fig_h, 4)),
                             squeeze=False)
    fig.patch.set_facecolor('#111111')

    for ax in axes.flat:
        ax.axis('off')
        ax.set_facecolor('#111111')

    LM_COLOURS = {'L1': '#ff4444', 'L2': '#ff9900',
                  'L3': '#ffee00', 'L4': '#44ff44', 'L5': '#44ccff'}

    for idx, rel_key in enumerate(all_keys):
        rec        = frame_records[rel_key]
        proj_key   = rec['proj_key']
        reproj_err = rec['reproj_error_px']
        init_meth  = rec['init_method']
        xray_rel   = rec['xray_relative_path']

        xray_path = xray_dir / xray_rel
        drr_path  = drr_dir  / rel_key

        try:
            xray_raw = load_gray_float(xray_path)
        except FileNotFoundError:
            xray_raw = np.zeros((TS, TS), dtype=np.float32)

        try:
            drr_img = load_gray_float(drr_path)
        except FileNotFoundError:
            drr_img = np.zeros((TS, TS), dtype=np.float32)

        # Resize DRR to thumb first (needed as reference for histmatch)
        drr_th = cv2.resize(drr_img, (TS, TS), interpolation=cv2.INTER_AREA)

        # Resize raw xray to thumb, then apply processing
        xray_small = cv2.resize(xray_raw, (TS, TS), interpolation=cv2.INTER_AREA)
        xray_proc_img = process_xray(xray_small, xray_proc,
                                     drr_ref=drr_th if xray_proc == 'histmatch' else None)

        row  = idx // NCOLS_FRAMES
        col0 = (idx %  NCOLS_FRAMES) * 2
        col1 = col0 + 1

        ax_x = axes[row, col0]
        ax_d = axes[row, col1]

        # ── X-ray panel ───────────────────────────────────────────────────
        ax_x.imshow(xray_proc_img, cmap='gray', vmin=0, vmax=1,
                    interpolation='bilinear')
        ax_x.axis('off')

        proj = proj_by_key.get(proj_key)
        if proj is not None and proj.gt_landmarks_2d:
            scale = TS / SWARO_IMG_SIZE
            for lm_name, uv in proj.gt_landmarks_2d.items():
                c = LM_COLOURS.get(lm_name, 'white')
                ax_x.plot(uv[0] * scale, uv[1] * scale,
                          '+', color=c, markersize=8, markeredgewidth=1.5)
                ax_x.text(uv[0] * scale + 3, uv[1] * scale - 3,
                          lm_name, color=c, fontsize=5, va='bottom')

        ax_x.set_title(f'{proj_key}\nX-ray', color='#cccccc', fontsize=6, pad=2)

        # ── DRR panel ─────────────────────────────────────────────────────
        ax_d.imshow(drr_th, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax_d.axis('off')
        meth_str   = 'EPnP' if init_meth == 'epnp' else 'anat.'
        reproj_str = f'{reproj_err:.1f}px' if reproj_err > 0 else 'SQPNP'
        ax_d.set_title(f'DRR ({meth_str})\nreproj={reproj_str}',
                       color='#aaaaaa', fontsize=6, pad=2)

    proc_label = PROC_LABELS[xray_proc]
    plt.suptitle(
        f'Swaroopa EPnP DRR vs X-ray  [{proc_label}]',
        color='white', fontsize=11, y=1.01,
    )
    plt.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output), dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output}  ({N} frames, grid {NROWS}×{NCOLS_FRAMES},  proc={xray_proc})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    with open(args.poses_json) as f:
        payload = json.load(f)
    frame_records = payload['frames']

    all_keys = sorted(frame_records.keys())
    if args.frames:
        all_keys = [k for k in all_keys
                    if frame_records[k]['proj_key'] in args.frames]
    if not all_keys:
        print("No frames to visualise.")
        return

    print("Loading Swaroopa specimen for landmark overlay ...")
    loader = SwaroLoader()
    spec   = loader.load(
        frames=[frame_records[k]['proj_key'] for k in all_keys],
        verbose=False,
    )
    proj_by_key = {p.proj_key: p for p in spec.projections}

    modes = ALL_PROCS if args.all_procs else [args.xray_proc]

    for mode in modes:
        if args.all_procs:
            out = args.output.parent / f'swaroopa_epnp_drr_{mode}.png'
        else:
            out = args.output

        render_grid(
            all_keys       = all_keys,
            frame_records  = frame_records,
            proj_by_key    = proj_by_key,
            drr_dir        = args.drr_dir,
            xray_dir       = args.xray_dir,
            xray_proc      = mode,
            output         = out,
            cols           = args.cols,
            thumb_size     = args.thumb_size,
            dpi            = args.dpi,
        )

if __name__ == '__main__':
    main()
