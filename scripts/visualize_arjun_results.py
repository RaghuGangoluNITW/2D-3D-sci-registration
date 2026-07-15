"""
Visualise Arjun 2D/3D registration results.

Generates:
  results/figures/arjun_overview.png   — X-ray | DRR@EPnP | DRR@final | overlay
  results/figures/arjun_summary.png    — PDE bar chart + GO improvement
  results/figures/arjun_landmarks.png  — 2D landmark reprojection check per frame
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

from arjun_loader import (
    ArjunLoader,
    ArjunProjection,
    pixel_spacing_mm,
    _make_K,
    ARJUN_REF_PIX_MM,
    ARJUN_REF_SIZE,
)
from deepfluoro_loader import perturb_extrinsic
from deepfluoro_drr import DeepFluoroDRR

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_JSON = os.path.join(os.path.dirname(__file__), '..', 'results', 'arjun_results.json')
OUT_DIR      = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')

VIS_SIZE   = 180
VIS_PIX_MM = ARJUN_REF_PIX_MM * (ARJUN_REF_SIZE / VIS_SIZE)   # ~1.138 mm/px
VIS_STEPS  = 120

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_drr(drr_gen, R, t, invert=True):
    drr = drr_gen.generate_from_extrinsic(R, t, VIS_SIZE, VIS_PIX_MM, VIS_STEPS)
    return 1.0 - drr if invert else drr


def resize_xray(img):
    return cv2.resize(img, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_AREA)


def checkerboard(a, b, tiles=8):
    h, w = a.shape
    th, tw = h // tiles, max(1, w // tiles)
    mask = np.zeros((h, w), dtype=np.float32)
    for i in range(tiles):
        for j in range(tiles):
            if (i + j) % 2 == 0:
                mask[i*th:(i+1)*th, j*tw:(j+1)*tw] = 1.0
    return np.where(mask > 0.5, a, b)


def project_landmarks(proj, pts3d, lm_names, R, t):
    """Project 3D landmarks using given R, t and per-frame K."""
    from arjun_loader import xzy
    fx, fy, cx, cy = _make_K(proj.img_w, proj.img_h)
    pts_xzy = xzy(pts3d)
    P_cam = (R @ pts_xzy.T).T + t
    u = fx * P_cam[:, 0] / P_cam[:, 2] + cx
    v = fy * P_cam[:, 1] / P_cam[:, 2] + cy
    return np.stack([u, v], axis=1)


# ── Overview figure ───────────────────────────────────────────────────────────

def make_overview(spec, drr_gen, results_json, out_name='arjun_overview.png'):
    per_proj   = results_json['arjun']['per_projection']
    frame_keys = sorted(per_proj.keys())   # 'a', 'b', 'c', 'd', 'e'
    proj_map   = {p.proj_key: p for p in spec.projections}

    n_rows = len(frame_keys)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = ['X-ray (real)', 'DRR @ EPnP pose', 'DRR @ final pose', 'Overlay (final)']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

    for row, key in enumerate(frame_keys):
        if key not in proj_map:
            for ax in axes[row]:
                ax.axis('off')
            continue

        proj    = proj_map[key]
        pdata   = per_proj[key]
        success = pdata['success']
        init_go = pdata.get('initial_go', float('nan'))
        final_go = pdata.get('final_go', float('nan'))
        init_pde  = pdata.get('initial_pde_mm', float('nan'))
        final_pde = pdata.get('final_pde_mm', float('nan'))

        # EPnP initial pose (stored in proj.R_proj, proj.t_proj)
        R_epnp = proj.R_proj.copy()
        t_epnp = proj.t_proj.copy()

        # Final pose from saved delta
        best_delta = np.array(pdata.get('best_pose_delta', [0]*6))
        R_final, t_final = perturb_extrinsic(R_epnp, t_epnp, best_delta[:3], best_delta[3:])

        # Images
        xray_small = resize_xray(proj.image_raw)
        drr_epnp   = render_drr(drr_gen, R_epnp,  t_epnp,  invert=True)
        drr_final  = render_drr(drr_gen, R_final,  t_final, invert=True)
        overlay    = checkerboard(xray_small, drr_final)

        images    = [xray_small, drr_epnp, drr_final, overlay]
        subtitles = [
            f'Frame {key.upper()}  ({proj.img_w}×{proj.img_h})',
            f'EPnP PDE={init_pde:.1f}mm  GO={init_go:.3f}',
            f'Final PDE={final_pde:.1f}mm  GO={final_go:.3f}',
            f'{"✓ SUCCESS" if success else "✗ FAIL"}',
        ]
        border_color = '#2ecc71' if success else '#e74c3c'

        for col, (ax, img, sub) in enumerate(zip(axes[row], images, subtitles)):
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_xlabel(sub, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)

    n_succ = sum(p['success'] for p in per_proj.values())
    mean_init_pde = results_json['arjun'].get('mean_initial_pde_mm')
    mean_final_pde = results_json['arjun'].get('mean_final_pde_mm')
    pde_str = (f'  PDE: {mean_init_pde:.1f}→{mean_final_pde:.1f}mm'
               if mean_init_pde is not None else '')
    title = (f'Arjun Lumbar Spine Registration — EPnP + CMA-ES\n'
             f'{n_succ}/{len(per_proj)} success{pde_str}  '
             f'GO: {results_json["arjun"].get("mean_initial_go", 0):.3f}→'
             f'{results_json["arjun"].get("mean_final_go", 0):.3f}')
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Summary chart ─────────────────────────────────────────────────────────────

def make_summary(results_json, out_name='arjun_summary.png'):
    per_proj = results_json['arjun']['per_projection']
    keys     = sorted(per_proj.keys())
    init_pde  = [per_proj[k].get('initial_pde_mm', float('nan')) for k in keys]
    final_pde = [per_proj[k].get('final_pde_mm',   float('nan')) for k in keys]
    init_go   = [per_proj[k].get('initial_go', float('nan'))     for k in keys]
    final_go  = [per_proj[k].get('final_go',   float('nan'))     for k in keys]
    success   = [per_proj[k]['success'] for k in keys]

    x  = np.arange(len(keys))
    w  = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: PDE before/after
    ax = axes[0]
    has_pde = any(not np.isnan(p) for p in init_pde)
    if has_pde:
        ax.bar(x - w/2, init_pde,  w, label='EPnP PDE',   color='#95a5a6', alpha=0.85)
        ax.bar(x + w/2, final_pde, w, label='Final PDE',
               color=['#2ecc71' if s else '#e74c3c' for s in success], alpha=0.9)
        ax.axhline(15.0, color='orange', linestyle='--', linewidth=1.2, label='15 mm threshold')
        ax.set_ylabel('PDE (mm)  — lower is better', fontsize=10)
        ax.set_title('EPnP vs Final PDE per Frame', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        for bar in ax.patches[len(keys):]:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{h:.1f}',
                        ha='center', va='bottom', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No PDE data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('PDE (no data)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Frame {k.upper()}' for k in keys], fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: GO before/after
    ax2 = axes[1]
    ax2.bar(x - w/2, init_go,  w, label='Initial GO',  color='#95a5a6', alpha=0.85)
    ax2.bar(x + w/2, final_go, w, label='Final GO',
            color=['#2ecc71' if s else '#e74c3c' for s in success], alpha=0.85)
    ax2.axhline(0.60, color='orange', linestyle='--', linewidth=1.2, label='GO<0.60 threshold')
    ax2.set_xticks(x); ax2.set_xticklabels([f'Frame {k.upper()}' for k in keys], fontsize=10)
    ax2.set_ylabel('GO Score  (lower = better)', fontsize=10)
    ax2.set_ylim(0, 1.1)
    ax2.set_title('GO Score per Frame', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    for bar in ax2.patches[len(keys):]:
        h = bar.get_height()
        if not np.isnan(h):
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.015, f'{h:.3f}',
                     ha='center', va='bottom', fontsize=8)

    # Panel 3: ΔGO improvement
    ax3 = axes[2]
    delta_go = [per_proj[k].get('go_delta', float('nan')) for k in keys]
    colors = ['#2ecc71' if s else '#e74c3c' for s in success]
    bars_d = ax3.bar(x, delta_go, color=colors, alpha=0.85, edgecolor='white')
    ax3.axhline(0.03, color='orange', linestyle='--', linewidth=1.2, label='Min ΔGO>0.03')
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_xticks(x); ax3.set_xticklabels([f'Frame {k.upper()}' for k in keys], fontsize=10)
    ax3.set_ylabel('ΔGO  (initial − final,  higher = better)', fontsize=10)
    ax3.set_title('GO Score Improvement per Frame', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    for bar, d in zip(bars_d, delta_go):
        if not np.isnan(d):
            h = bar.get_height()
            ypos = h + 0.005 if h >= 0 else h - 0.025
            ax3.text(bar.get_x() + bar.get_width()/2, ypos, f'{d:+.3f}',
                     ha='center', va='bottom', fontsize=8)

    success_patch = mpatches.Patch(color='#2ecc71', label='SUCCESS')
    fail_patch    = mpatches.Patch(color='#e74c3c', label='FAIL')
    ax3.legend(handles=[success_patch, fail_patch], fontsize=9)

    n_success = sum(success)
    r = results_json['arjun']
    pde_str = ''
    if r.get('mean_initial_pde_mm') is not None:
        pde_str = (f'  |  PDE: {r["mean_initial_pde_mm"]:.1f}→'
                   f'{r["mean_final_pde_mm"]:.1f}mm')
    fig.suptitle(
        f'Arjun Registration Summary\n'
        f'{n_success}/{len(keys)} success{pde_str}  '
        f'|  GO: {r.get("mean_initial_go", 0):.3f}→{r.get("mean_final_go", 0):.3f}  '
        f'(Δ={r.get("mean_go_delta", 0):+.3f})',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Landmark reprojection figure ──────────────────────────────────────────────

def make_landmarks_figure(spec, results_json, out_name='arjun_landmarks.png'):
    """Show X-ray with GT 2D dots and reprojected 3D landmarks overlaid."""
    per_proj  = results_json['arjun']['per_projection']
    proj_map  = {p.proj_key: p for p in spec.projections}
    keys      = sorted(per_proj.keys())

    lm_names = sorted(spec.landmarks_3d.keys())
    pts3d    = np.array([spec.landmarks_3d[n] for n in lm_names])

    n_rows = len(keys)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = ['EPnP Initial Pose Reprojection', 'Final Pose Reprojection']
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=12, fontweight='bold', pad=6)

    colors_lm = plt.cm.tab10(np.linspace(0, 1, len(lm_names)))

    for row, key in enumerate(keys):
        if key not in proj_map:
            for ax in axes[row]:
                ax.axis('off')
            continue

        proj   = proj_map[key]
        pdata  = per_proj[key]
        R_epnp = proj.R_proj.copy()
        t_epnp = proj.t_proj.copy()

        best_delta = np.array(pdata.get('best_pose_delta', [0]*6))
        R_final, t_final = perturb_extrinsic(R_epnp, t_epnp, best_delta[:3], best_delta[3:])

        xray_disp = proj.image_raw   # float32 [0,1]

        for col, (R, t, ax) in enumerate(zip([R_epnp, R_final], [t_epnp, t_final], axes[row])):
            ax.imshow(xray_disp, cmap='gray', vmin=0, vmax=1,
                      extent=[0, proj.img_w, proj.img_h, 0], aspect='auto')

            # GT 2D landmarks (from annotation JSON)
            for lname, uv_gt in proj.gt_landmarks_2d.items():
                ax.plot(uv_gt[0], uv_gt[1], 'o', color='lime', markersize=9,
                        markeredgecolor='black', markeredgewidth=1.2, zorder=5)
                ax.text(uv_gt[0]+8, uv_gt[1], lname, color='lime',
                        fontsize=8, fontweight='bold', zorder=6)

            # Reprojected 3D landmarks
            uv_pred = project_landmarks(proj, pts3d, lm_names, R, t)
            for i, (name, uv) in enumerate(zip(lm_names, uv_pred)):
                ax.plot(uv[0], uv[1], 'x', color=colors_lm[i], markersize=10,
                        markeredgewidth=2, zorder=5)
                if name in proj.gt_landmarks_2d:
                    gt = proj.gt_landmarks_2d[name]
                    ax.plot([gt[0], uv[0]], [gt[1], uv[1]], '-',
                            color=colors_lm[i], linewidth=1, alpha=0.7, zorder=4)

            ax.set_xlim(0, proj.img_w); ax.set_ylim(proj.img_h, 0)
            ax.set_xticks([]); ax.set_yticks([])

            pde_tag = 'EPnP' if col == 0 else 'Final'
            pde_val = pdata.get('initial_pde_mm') if col == 0 else pdata.get('final_pde_mm')
            pde_str = f'{pde_val:.1f}mm' if pde_val and not (isinstance(pde_val, float) and np.isnan(pde_val)) else 'N/A'
            ax.set_xlabel(f'Frame {key.upper()} — {pde_tag} PDE={pde_str}', fontsize=10)

        # Legend in last column
        gt_patch    = plt.Line2D([0],[0], marker='o', color='lime', linestyle='None',
                                 markersize=8, markeredgecolor='black', label='GT 2D annotation')
        pred_patch  = plt.Line2D([0],[0], marker='x', color='gray', linestyle='None',
                                 markersize=8, markeredgewidth=2, label='3D→2D reprojection')
        axes[row][1].legend(handles=[gt_patch, pred_patch], fontsize=9, loc='upper right')

    fig.suptitle('Arjun — 2D Landmark Reprojection Check\n'
                 '○ GT annotation  ×  3D centroid reprojected',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading result JSON ...')
    results_json = json.load(open(RESULTS_JSON))

    print('Loading Arjun specimen ...')
    spec    = ArjunLoader().load(verbose=False)
    drr_gen = DeepFluoroDRR(spec, hu_threshold=150.0)

    print('\nGenerating overview figure ...')
    make_overview(spec, drr_gen, results_json)

    print('Generating summary chart ...')
    make_summary(results_json)

    print('Generating landmark reprojection figure ...')
    make_landmarks_figure(spec, results_json)

    print('\nAll figures saved to results/figures/')


if __name__ == '__main__':
    main()
