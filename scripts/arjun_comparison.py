"""
arjun_comparison.py — Side-by-side X-ray / DRR(EPnP) / DRR(final) for each Arjun frame
=========================================================================================
Produces one figure per frame:
    results/figures/arjun_comparison_<key>.png
and one combined overview:
    results/figures/arjun_comparison_overview.png

Each figure has 3 columns (X-ray | DRR @ EPnP pose | DRR @ final pose)
with annotated 2D landmarks overlaid on each panel.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

from arjun_loader import (
    ArjunLoader,
    ArjunProjection,
    _make_K,
    ARJUN_REF_PIX_MM,
    ARJUN_REF_SIZE,
)
from deepfluoro_loader import perturb_extrinsic
from deepfluoro_drr import DeepFluoroDRR

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS_JSON = os.path.join(os.path.dirname(__file__), '..', 'results', 'arjun_results.json')
OUT_DIR      = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

VIS_SIZE   = 384          # pixels for DRR render (larger = cleaner visuals)
VIS_PIX_MM = ARJUN_REF_PIX_MM * (ARJUN_REF_SIZE / VIS_SIZE)
VIS_STEPS  = 200          # more steps = sharper bone edges


# ── Helpers ────────────────────────────────────────────────────────────────────

def render_drr(drr_gen, R, t, invert=True):
    drr = drr_gen.generate_from_extrinsic(R, t, VIS_SIZE, VIS_PIX_MM, VIS_STEPS)
    return (1.0 - drr) if invert else drr


def project_lm_to_vis(proj, pts3d, lm_names, R, t):
    """Project 3D landmarks → pixel coords in VIS_SIZE image space."""
    from arjun_loader import xzy
    fx, fy, cx, cy = _make_K(proj.img_w, proj.img_h)
    pts_xzy = xzy(pts3d)
    P_cam = (R @ pts_xzy.T).T + t
    # project
    u = fx * P_cam[:, 0] / P_cam[:, 2] + cx
    v = fy * P_cam[:, 1] / P_cam[:, 2] + cy
    # scale to VIS_SIZE space (original image is img_w × img_h)
    u_vis = u * VIS_SIZE / proj.img_w
    v_vis = v * VIS_SIZE / proj.img_h
    return np.stack([u_vis, v_vis], axis=1)


def gt_lm_to_vis(proj, lm_name):
    """Scale GT 2D annotation pixel coord to VIS_SIZE."""
    uv = proj.gt_landmarks_2d[lm_name]
    u_vis = uv[0] * VIS_SIZE / proj.img_w
    v_vis = uv[1] * VIS_SIZE / proj.img_h
    return u_vis, v_vis


# ── Per-frame figure ───────────────────────────────────────────────────────────

def make_frame_figure(proj, spec, drr_gen, pdata, out_path):
    """
    3-panel side-by-side: X-ray | DRR@EPnP | DRR@final
    Landmarks are overlaid on all three panels.
    GT annotations shown as green circles; reprojected 3D landmarks as red crosses.
    """
    R_epnp = proj.R_proj.copy()
    t_epnp = proj.t_proj.copy()

    best_delta = np.array(pdata.get('best_pose_delta', [0] * 6))
    R_final, t_final = perturb_extrinsic(R_epnp, t_epnp, best_delta[:3], best_delta[3:])

    init_pde  = pdata.get('initial_pde_mm', float('nan'))
    final_pde = pdata.get('final_pde_mm',   float('nan'))
    init_go   = pdata.get('initial_go',     float('nan'))
    final_go  = pdata.get('final_go',       float('nan'))
    success   = pdata.get('success', False)

    # ---- Images ----
    xray_vis = cv2.resize(proj.image_raw, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_AREA)
    drr_epnp  = render_drr(drr_gen, R_epnp,  t_epnp)
    drr_final = render_drr(drr_gen, R_final, t_final)

    # ---- Landmark coords ----
    lm_names = sorted(spec.landmarks_3d.keys())
    pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])
    annotated = list(proj.gt_landmarks_2d.keys())

    uv_epnp  = project_lm_to_vis(proj, pts3d, lm_names, R_epnp,  t_epnp)
    uv_final = project_lm_to_vis(proj, pts3d, lm_names, R_final, t_final)

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    panels = [
        (xray_vis,  uv_epnp,  'X-ray (real)\n(GT annotations shown)',
         f'Frame {proj.proj_key.upper()}  ({proj.img_w}×{proj.img_h})',  '#3498db'),
        (drr_epnp,  uv_epnp,  'DRR @ EPnP initial pose',
         f'PDE={init_pde:.2f} mm   GO={init_go:.4f}',                    '#e67e22'),
        (drr_final, uv_final, 'DRR @ final (optimised) pose',
         f'PDE={final_pde:.2f} mm   GO={final_go:.4f}   '
         f'{"✓ SUCCESS" if success else "✗ FAIL"}',
         '#2ecc71' if success else '#e74c3c'),
    ]

    lm_cmap   = plt.cm.tab10(np.linspace(0, 1, len(lm_names)))

    for ax, (img, uv_pred, col_title, subtitle, border_clr) in zip(axes, panels):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='upper')

        # GT 2D annotations (green circles) — only on first panel
        if col_title.startswith('X-ray'):
            for lname in annotated:
                ux, vy = gt_lm_to_vis(proj, lname)
                ax.plot(ux, vy, 'o', color='lime', markersize=11,
                        markeredgecolor='black', markeredgewidth=1.2, zorder=6)
                ax.text(ux + 5, vy - 5, lname, color='lime',
                        fontsize=9, fontweight='bold', zorder=7,
                        path_effects=[
                            __import__('matplotlib.patheffects', fromlist=['withStroke'])
                            .withStroke(linewidth=2, foreground='black')
                        ])

        # Reprojected 3D landmarks (coloured crosses on all panels)
        for i, (lname, (ux, vy)) in enumerate(zip(lm_names, uv_pred)):
            clr = lm_cmap[i]
            in_frame = (0 <= ux < VIS_SIZE) and (0 <= vy < VIS_SIZE)
            if not in_frame:
                continue
            ax.plot(ux, vy, '+', color=clr, markersize=14,
                    markeredgewidth=2.5, zorder=5)
            ax.text(ux + 4, vy + 4, lname, color=clr, fontsize=8,
                    fontweight='bold', zorder=6,
                    path_effects=[
                        __import__('matplotlib.patheffects', fromlist=['withStroke'])
                        .withStroke(linewidth=1.5, foreground='black')
                    ])

            # Draw line from GT to reprojected (on DRR panels)
            if not col_title.startswith('X-ray') and lname in annotated:
                gx, gy = gt_lm_to_vis(proj, lname)
                ax.plot([gx, ux], [gy, vy], '-', color=clr,
                        linewidth=1.2, alpha=0.7, zorder=4)
                # Ghost GT dot on DRR panels for reference
                ax.plot(gx, gy, 'o', color='lime', markersize=7,
                        markeredgecolor='black', markeredgewidth=0.8,
                        alpha=0.6, zorder=5)

        ax.set_xlim(0, VIS_SIZE); ax.set_ylim(VIS_SIZE, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(col_title, fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel(subtitle, fontsize=9)

        # Coloured border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_clr)
            spine.set_linewidth(3)

    # Legend
    gt_marker   = plt.Line2D([0],[0], marker='o', color='lime', linestyle='None',
                              markersize=9, markeredgecolor='black', label='GT 2D annotation')
    pred_marker = plt.Line2D([0],[0], marker='+', color='white', linestyle='None',
                              markersize=9, markeredgewidth=2, label='3D→2D reprojection')
    axes[2].legend(handles=[gt_marker, pred_marker], fontsize=9,
                   loc='lower right', framealpha=0.7)

    # Per-landmark PDE table on right margin
    pde_lm = pdata.get('pde_per_landmark', {})
    if pde_lm:
        pde_lines = ['Per-landmark PDE (final):']
        for lm, val in sorted(pde_lm.items()):
            pde_lines.append(f'  {lm}: {val:.2f} mm')
        fig.text(0.995, 0.5, '\n'.join(pde_lines),
                 ha='right', va='center', fontsize=8.5,
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    go_delta = init_go - final_go
    fig.suptitle(
        f'Arjun Frame {proj.proj_key.upper()} — 2D/3D Registration\n'
        f'EPnP→CMA-ES  |  PDE: {init_pde:.2f}→{final_pde:.2f} mm  '
        f'|  GO: {init_go:.4f}→{final_go:.4f}  (Δ={go_delta:+.4f})',
        fontsize=12, fontweight='bold', y=1.02
    )

    fig.tight_layout(rect=[0, 0, 0.98, 1.0])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ── Combined overview (all frames in one figure) ──────────────────────────────

def make_overview(spec, drr_gen, results_json, out_path):
    per_proj   = results_json['arjun']['per_projection']
    frame_keys = sorted(per_proj.keys())
    proj_map   = {p.proj_key: p for p in spec.projections}

    n_rows = len(frame_keys)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    col_labels = ['X-ray (real)', 'DRR @ EPnP initial pose', 'DRR @ final pose']
    for ax, lbl in zip(axes[0], col_labels):
        ax.set_title(lbl, fontsize=12, fontweight='bold', pad=8)

    lm_names_all = sorted(spec.landmarks_3d.keys())
    pts3d_all    = np.array([spec.landmarks_3d[n] for n in lm_names_all])
    lm_cmap      = plt.cm.tab10(np.linspace(0, 1, len(lm_names_all)))

    for row, key in enumerate(frame_keys):
        if key not in proj_map:
            for ax in axes[row]:
                ax.set_visible(False)
            continue

        proj   = proj_map[key]
        pdata  = per_proj[key]

        R_epnp = proj.R_proj.copy()
        t_epnp = proj.t_proj.copy()
        best_delta = np.array(pdata.get('best_pose_delta', [0] * 6))
        R_final, t_final = perturb_extrinsic(R_epnp, t_epnp, best_delta[:3], best_delta[3:])

        init_pde  = pdata.get('initial_pde_mm', float('nan'))
        final_pde = pdata.get('final_pde_mm',   float('nan'))
        success   = pdata.get('success', False)
        annotated = list(proj.gt_landmarks_2d.keys())

        xray_vis  = cv2.resize(proj.image_raw, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_AREA)
        drr_epnp  = render_drr(drr_gen, R_epnp,  t_epnp)
        drr_final = render_drr(drr_gen, R_final, t_final)

        uv_epnp  = project_lm_to_vis(proj, pts3d_all, lm_names_all, R_epnp,  t_epnp)
        uv_final = project_lm_to_vis(proj, pts3d_all, lm_names_all, R_final, t_final)

        border = '#2ecc71' if success else '#e74c3c'

        for col, (ax, img, uv_pred, subtitle) in enumerate(zip(
            axes[row],
            [xray_vis, drr_epnp, drr_final],
            [uv_epnp,  uv_epnp,  uv_final],
            [f'Frame {key.upper()}  {proj.img_w}×{proj.img_h}',
             f'EPnP PDE={init_pde:.1f} mm',
             f'Final PDE={final_pde:.1f} mm  {"✓" if success else "✗"}'],
        )):
            ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='upper')

            # GT dots on X-ray
            if col == 0:
                for lname in annotated:
                    ux, vy = gt_lm_to_vis(proj, lname)
                    ax.plot(ux, vy, 'o', color='lime', markersize=9,
                            markeredgecolor='black', markeredgewidth=1, zorder=6)
                    ax.text(ux + 4, vy - 3, lname, color='lime',
                            fontsize=7.5, fontweight='bold', zorder=7)

            # Reprojected crosses
            for i, (lname, (ux, vy)) in enumerate(zip(lm_names_all, uv_pred)):
                if not (0 <= ux < VIS_SIZE and 0 <= vy < VIS_SIZE):
                    continue
                ax.plot(ux, vy, '+', color=lm_cmap[i], markersize=11,
                        markeredgewidth=2, zorder=5)

                # Line from GT to reprojected on DRR panels
                if col > 0 and lname in annotated:
                    gx, gy = gt_lm_to_vis(proj, lname)
                    ax.plot([gx, ux], [gy, vy], '-', color=lm_cmap[i],
                            linewidth=1, alpha=0.6, zorder=4)
                    ax.plot(gx, gy, 'o', color='lime', markersize=6,
                            markeredgecolor='black', markeredgewidth=0.8,
                            alpha=0.55, zorder=5)

            ax.set_xlim(0, VIS_SIZE); ax.set_ylim(VIS_SIZE, 0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(subtitle, fontsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(2.5)

    n_succ = sum(p['success'] for p in per_proj.values())
    r = results_json['arjun']
    fig.suptitle(
        f'Arjun Lumbar Spine — 2D/3D Registration Comparison\n'
        f'{n_succ}/{len(per_proj)} success  |  '
        f'Mean PDE: {r["mean_initial_pde_mm"]:.2f}→{r["mean_final_pde_mm"]:.2f} mm  |  '
        f'Mean GO: {r["mean_initial_go"]:.4f}→{r["mean_final_go"]:.4f}',
        fontsize=13, fontweight='bold', y=1.01
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print('Loading Arjun data ...')
    results_json = json.load(open(RESULTS_JSON))
    spec    = ArjunLoader().load(verbose=False)
    drr_gen = DeepFluoroDRR(spec, hu_threshold=-200.0)  # include soft tissue for realistic DRR

    per_proj = results_json['arjun']['per_projection']
    proj_map = {p.proj_key: p for p in spec.projections}

    print(f'\nGenerating per-frame comparison figures ({len(per_proj)} frames) ...')
    for key in sorted(per_proj.keys()):
        if key not in proj_map:
            print(f'  [skip] frame {key} not in loaded projections')
            continue
        proj  = proj_map[key]
        pdata = per_proj[key]
        out   = os.path.join(OUT_DIR, f'arjun_comparison_{key}.png')
        make_frame_figure(proj, spec, drr_gen, pdata, out)

    print('\nGenerating combined overview figure ...')
    overview_out = os.path.join(OUT_DIR, 'arjun_comparison_overview.png')
    make_overview(spec, drr_gen, results_json, overview_out)

    print('\nDone. All figures saved to results/figures/')


if __name__ == '__main__':
    main()
