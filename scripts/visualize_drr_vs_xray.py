"""
DRR vs X-ray visualisation for all 24 DeepFluoro projections.

For each projection this generates 4 panels:
  1. X-ray  (target image, DRR-matching orientation)
  2. DRR @ GT pose  (what perfect alignment looks like)
  3. DRR @ initial perturbed pose  (starting point of registration)
  4. Checkerboard blend  X-ray / DRR@GT  (alignment quality at GT)

Coloured landmark dots are overlaid on every panel.
Each panel is annotated with initial PDE → final PDE and SUCCESS / FAIL.

Outputs  (saved to  results/figures/):
  drr_xray_17-1882.png        — one figure per specimen (4-panel grid per proj)
  drr_xray_17-1905.png
  drr_xray_18-0725.png
  drr_xray_18-1109.png
  drr_xray_18-2799.png
  drr_xray_18-2800.png
  drr_xray_all_overview.png   — compact 1-panel-per-proj overview
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import cv2

from src.deepfluoro_loader import (
    DeepFluoroLoader,
    perturb_extrinsic,
    project_world_to_image,
    PIXEL_SPACING_MM,
    FULL_RES_SIZE,
)
from src.deepfluoro_drr import DeepFluoroDRR

# ── Constants ────────────────────────────────────────────────────────────────
H5_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data',
                         'ipcai_2020_full_res_data',
                         'ipcai_2020_full_res_data.h5')
JSON_PATH = os.path.join(os.path.dirname(__file__), '..', 'results',
                         'deepfluoro_results.json')
OUT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')

# DRR render resolution (same as optimisation Phase-2)
VIS_SIZE   = 180                           # pixels
VIS_PIX_MM = FULL_RES_SIZE * PIXEL_SPACING_MM / VIS_SIZE   # ≈ 1.657 mm/px
VIS_STEPS  = 150                           # ray-march steps (quality)

# Perturbation parameters — must match run_deepfluoro.py exactly
PERTURB_ROT_DEG = 3.0
PERTURB_TRANS_MM = 10.0

# Landmark dot colour palette
LM_CMAP = plt.cm.Set1
LANDMARK_COLORS: dict = {}   # filled lazily from lm_names order

SPECIMEN_IDS = ['17-1882', '17-1905', '18-0725', '18-1109', '18-2799', '18-2800']

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_results() -> dict:
    with open(JSON_PATH) as f:
        return json.load(f)


def render_drr(drr_gen: DeepFluoroDRR, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Render a DRR and invert so bright = bone, dark = air (matching X-ray polarity)."""
    drr = drr_gen.generate_from_extrinsic(R, t, VIS_SIZE, VIS_PIX_MM, VIS_STEPS)
    return 1.0 - drr   # invert: DRR is bone-bright by default


def resize_xray(image_raw: np.ndarray) -> np.ndarray:
    """Resize 1536×1536 X-ray to VIS_SIZE using area interpolation."""
    return cv2.resize(image_raw, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_AREA)


def checkerboard_blend(xray_vis: np.ndarray, drr_vis: np.ndarray,
                       tile: int = 18) -> np.ndarray:
    """Tile checkerboard: odd tiles = X-ray, even tiles = DRR."""
    y_idx, x_idx = np.indices((VIS_SIZE, VIS_SIZE))
    mask = ((y_idx // tile) + (x_idx // tile)) % 2 == 0
    blend = np.where(mask, xray_vis, drr_vis)
    return blend.astype(np.float32)


def project_to_vis(pts3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project 3D landmarks → 2D coords in VIS_SIZE pixel space."""
    uv_full = project_world_to_image(pts3d, R, t)   # (N,2) in 1536px space
    return uv_full * (VIS_SIZE / FULL_RES_SIZE)      # (N,2) in VIS_SIZE space


def draw_landmarks(ax, uv: np.ndarray, lm_names, pde_dict: dict = None,
                   marker_size: float = 6, show_names: bool = False):
    """Overlay landmark dots on a matplotlib axes."""
    n = len(lm_names)
    for i, name in enumerate(lm_names):
        col = LM_CMAP(i / max(n - 1, 1))
        u, v = uv[i]
        if 0 <= u < VIS_SIZE and 0 <= v < VIS_SIZE:
            ax.plot(u, v, 'o', color=col, markersize=marker_size,
                    markeredgewidth=0.8, markeredgecolor='white', zorder=5)
            if show_names and pde_dict:
                pde_val = pde_dict.get(name, None)
                label = f"{name.split('-')[0]}" + (f"\n{pde_val:.1f}" if pde_val else "")
                ax.text(u + 3, v - 3, label, color=col, fontsize=4,
                        fontweight='bold', zorder=6,
                        path_effects=[
                            __import__('matplotlib.patheffects', fromlist=['withStroke'])
                            .withStroke(linewidth=1, foreground='black')
                        ])


def status_colour(success: bool) -> str:
    return '#2ecc71' if success else '#e74c3c'


# ── Per-specimen figure ───────────────────────────────────────────────────────

def make_specimen_figure(spec_id: str, results: dict):
    """Generate the 4-panel × N-projections figure for one specimen."""
    print(f"\n{'='*60}")
    print(f"  Specimen {spec_id}")
    print(f"{'='*60}")

    loader = DeepFluoroLoader(H5_PATH)
    spec   = loader.load_specimen(spec_id)
    projs  = spec.valid_projections(max_reproj_px=5.0)
    drr_gen = DeepFluoroDRR(spec)
    lm_names, pts3d = spec.get_landmark_array()

    # Build global colour map (first specimen wins — same landmarks across all)
    global LANDMARK_COLORS
    if not LANDMARK_COLORS:
        n = len(lm_names)
        LANDMARK_COLORS = {name: LM_CMAP(i / max(n - 1, 1))
                           for i, name in enumerate(lm_names)}

    spec_res = results.get(spec_id, {})
    per_proj = spec_res.get('per_projection', {})

    # Filter to only projections that were actually registered (present in JSON)
    if per_proj:
        projs = [p for p in projs if p.proj_key in per_proj]

    n_proj  = len(projs)
    n_cols  = 4   # X-ray | DRR@GT | DRR@init | Checkerboard
    col_labels = ['X-ray (target)', 'DRR @ GT pose', 'DRR @ initial pose',
                  'Checkerboard (X-ray / DRR@GT)']

    fig_h = 2.8 * n_proj + 1.0
    fig_w = 2.8 * n_cols + 1.2
    fig, axes = plt.subplots(n_proj, n_cols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)
    fig.patch.set_facecolor('#1a1a2e')

    # Column headers
    for c, lbl in enumerate(col_labels):
        axes[0, c].set_title(lbl, color='white', fontsize=9, fontweight='bold', pad=4)

    for row, proj in enumerate(projs):
        key = proj.proj_key
        proj_res  = per_proj.get(key, {})
        init_pde  = proj_res.get('initial_pde_mm', float('nan'))
        final_pde = proj_res.get('final_pde_mm',   float('nan'))
        success   = proj_res.get('success', False)
        pde_lm    = proj_res.get('pde_per_lm', {})

        print(f"  proj {key}: init={init_pde:.1f}mm  final={final_pde:.1f}mm  "
              f"{'SUCCESS' if success else 'FAIL'}")

        # ── Build initial perturbed pose (same seed as run_deepfluoro.py) ──
        rng = np.random.default_rng(42 + proj.proj_index)
        dr  = rng.uniform(-PERTURB_ROT_DEG,   PERTURB_ROT_DEG,   3)
        dt  = rng.uniform(-PERTURB_TRANS_MM,  PERTURB_TRANS_MM,  3)
        R_init, t_init = perturb_extrinsic(proj.R_proj, proj.t_proj, dr, dt)

        # ── Render ──────────────────────────────────────────────────────────
        print(f"    rendering DRR@GT ...")
        drr_gt   = render_drr(drr_gen, proj.R_proj, proj.t_proj)
        print(f"    rendering DRR@init ...")
        drr_init = render_drr(drr_gen, R_init, t_init)
        xray_vis = resize_xray(proj.image_raw)
        checker  = checkerboard_blend(xray_vis, drr_gt)

        # ── Project landmarks ────────────────────────────────────────────────
        uv_gt   = project_to_vis(pts3d, proj.R_proj, proj.t_proj)
        uv_init = project_to_vis(pts3d, R_init, t_init)

        panels   = [xray_vis, drr_gt, drr_init, checker]
        uv_list  = [uv_gt,    uv_gt,  uv_init,  uv_gt  ]
        show_pde = [True,     True,   False,     True   ]

        for col, (img, uv, show_p) in enumerate(zip(panels, uv_list, show_pde)):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1,
                      origin='upper', interpolation='nearest')
            draw_landmarks(ax, uv, lm_names,
                           pde_dict=pde_lm if (show_p and col == 0) else None,
                           marker_size=5, show_names=(col == 0))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(status_colour(success) if col == 0 else '#444466')
                spine.set_linewidth(2.5 if col == 0 else 1)
            ax.set_facecolor('#0d0d1a')

        # Row annotation (left of first panel)
        sc = status_colour(success)
        tag = 'SUCCESS' if success else 'FAIL'
        row_label = (f"proj {key}\n"
                     f"init  {init_pde:.1f}mm\n"
                     f"final {final_pde:.1f}mm\n"
                     f"[{tag}]")
        axes[row, 0].set_ylabel(row_label, color=sc, fontsize=7.5,
                                fontweight='bold', labelpad=6, rotation=0,
                                ha='right', va='center')

    # Legend for landmarks
    handles = [mpatches.Patch(color=LM_CMAP(i / max(len(lm_names) - 1, 1)),
                               label=name)
               for i, name in enumerate(lm_names)]
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(lm_names), 8),
               fontsize=7, framealpha=0.3,
               facecolor='#1a1a2e', edgecolor='#444466',
               labelcolor='white',
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(f"Specimen {spec_id} — DRR vs X-ray registration results\n"
                 f"({n_proj} projections  |  success rate "
                 f"{spec_res.get('success_rate', 0)*100:.0f}%  |  "
                 f"mean final PDE {spec_res.get('mean_final_pde', 0):.2f} mm)",
                 color='white', fontsize=11, fontweight='bold', y=0.999)

    fig.tight_layout(rect=[0.06, 0.04, 1.0, 0.995])

    out_path = os.path.join(OUT_DIR, f'drr_xray_{spec_id}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → saved {out_path}")
    return out_path


# ── Compact overview grid (one panel per projection) ─────────────────────────

def make_overview_figure(results: dict):
    """One checkerboard blend per projection, arranged as 6-specimen rows."""
    print("\n" + "="*60)
    print("  Building overview figure ...")
    print("="*60)

    loader = DeepFluoroLoader(H5_PATH)

    # Collect cells: list of (img, label, success)
    cells = []
    for spec_id in SPECIMEN_IDS:
        spec    = loader.load_specimen(spec_id)
        projs   = spec.valid_projections(max_reproj_px=5.0)
        drr_gen = DeepFluoroDRR(spec)
        lm_names, pts3d = spec.get_landmark_array()
        per_proj = results.get(spec_id, {}).get('per_projection', {})

        # Only include projections that were actually registered
        if per_proj:
            projs = [p for p in projs if p.proj_key in per_proj]

        for proj in projs:
            key      = proj.proj_key
            proj_res = per_proj.get(key, {})
            success  = proj_res.get('success', False)
            final_pde = proj_res.get('final_pde_mm', float('nan'))

            print(f"  {spec_id}/{key}  final={final_pde:.1f}mm  {'OK' if success else 'FAIL'}")
            drr_gt   = render_drr(drr_gen, proj.R_proj, proj.t_proj)
            xray_vis = resize_xray(proj.image_raw)
            blend    = checkerboard_blend(xray_vis, drr_gt)

            uv_gt = project_to_vis(pts3d, proj.R_proj, proj.t_proj)

            label = (f"{spec_id}/{key}\n"
                     f"{final_pde:.1f}mm  "
                     f"{'✓' if success else '✗'}")
            cells.append((blend, uv_gt, lm_names, label, success))

    # Layout
    n_cells = len(cells)
    n_cols  = 5   # max per row
    n_rows  = (n_cells + n_cols - 1) // n_cols
    cell_px = VIS_SIZE / 130   # figure inches per cell (at 130 dpi)
    pad = 0.3
    fig_w = n_cols * (cell_px + pad) + 0.6
    fig_h = n_rows * (cell_px + pad) + 0.8

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)
    fig.patch.set_facecolor('#1a1a2e')

    for idx, (blend, uv, lm_names_cell, label, success) in enumerate(cells):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.imshow(blend, cmap='gray', vmin=0, vmax=1,
                  origin='upper', interpolation='nearest')
        draw_landmarks(ax, uv, lm_names_cell, marker_size=3.5)
        ax.set_xticks([]); ax.set_yticks([])
        sc = status_colour(success)
        for spine in ax.spines.values():
            spine.set_edgecolor(sc)
            spine.set_linewidth(2)
        ax.set_title(label, color=sc, fontsize=5.5, pad=2, fontweight='bold')
        ax.set_facecolor('#0d0d1a')

    # Hide unused axes
    for idx in range(len(cells), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis('off')

    # Landmark legend
    n = len(lm_names)
    handles = [mpatches.Patch(color=LM_CMAP(i / max(n - 1, 1)), label=nm)
               for i, nm in enumerate(lm_names)]
    fig.legend(handles=handles, loc='lower center',
               ncol=min(n, 8), fontsize=6.5,
               framealpha=0.3, facecolor='#1a1a2e', edgecolor='#444466',
               labelcolor='white', bbox_to_anchor=(0.5, 0.0))

    success_count = sum(1 for _, _, _, _, s in cells if s)
    fig.suptitle(
        f"DeepFluoro 2D/3D registration overview  —  {success_count}/{n_cells} SUCCESS\n"
        f"Checkerboard: X-ray (dark tiles) / DRR@GT (light tiles) + landmarks",
        color='white', fontsize=9, fontweight='bold', y=1.001)

    fig.tight_layout(rect=[0, 0.06, 1, 0.998])

    out_path = os.path.join(OUT_DIR, 'drr_xray_all_overview.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  → overview saved: {out_path}")
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualise DRR vs X-ray results')
    parser.add_argument('--specimen', '-s', nargs='+', default=SPECIMEN_IDS,
                        help='Specimen IDs to process (default: all 6)')
    parser.add_argument('--overview-only', action='store_true',
                        help='Only generate the compact overview figure')
    parser.add_argument('--no-overview', action='store_true',
                        help='Skip the overview figure')
    args = parser.parse_args()

    results = load_results()
    print(f"Loaded results for {len(results)} specimens.")

    if not args.overview_only:
        for spec_id in args.specimen:
            make_specimen_figure(spec_id, results)

    if not args.no_overview:
        make_overview_figure(results)

    print("\nAll figures saved to", OUT_DIR)


if __name__ == '__main__':
    main()
