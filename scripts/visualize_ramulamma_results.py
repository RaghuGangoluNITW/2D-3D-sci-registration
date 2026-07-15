"""
Visualise Ramulamma 2D/3D registration results.

Generates figures for BOTH runs:
  • without_instruments (clean)  → ramulamma_overview_clean.png, ramulamma_summary_clean.png
  • with_instruments              → ramulamma_overview_instruments.png, ramulamma_summary_instruments.png
  • combined comparison           → ramulamma_comparison.png
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

from ramulamma_loader import (
    RamulamaLoader,
    RAMU_PIX_MM,
    RAMU_IMG_SIZE,
    DICOM_DIR_CLEAN,
    DICOM_DIR,
)
from deepfluoro_loader import perturb_extrinsic
from deepfluoro_drr import DeepFluoroDRR

# ── Constants ─────────────────────────────────────────────────────────────────
CLEAN_JSON = os.path.join(os.path.dirname(__file__), '..', 'results',
                          'ramulamma_results_clean.json')
INSTR_JSON = os.path.join(os.path.dirname(__file__), '..', 'results',
                          'ramulamma_results_instruments.json')
OUT_DIR    = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')

VIS_SIZE   = 180
VIS_PIX_MM = RAMU_PIX_MM * (RAMU_IMG_SIZE / VIS_SIZE)   # ~1.138 mm/px
VIS_STEPS  = 120

PERTURB_ROT_DEG  = 30.0
PERTURB_TRANS_MM = 60.0

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_drr(drr_gen, R, t, invert=True):
    drr = drr_gen.generate_from_extrinsic(R, t, VIS_SIZE, VIS_PIX_MM, VIS_STEPS)
    return 1.0 - drr if invert else drr


def resize_xray(img):
    return cv2.resize(img, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_AREA)


def checkerboard(a, b, tiles=8):
    """Checkerboard blend of two [0,1] images."""
    h, w = a.shape
    th, tw = h // tiles, max(1, w // tiles)
    mask = np.zeros((h, w), dtype=np.float32)
    for i in range(tiles):
        for j in range(tiles):
            if (i + j) % 2 == 0:
                mask[i*th:(i+1)*th, j*tw:(j+1)*tw] = 1.0
    return np.where(mask > 0.5, a, b)


# ── Overview figure ───────────────────────────────────────────────────────────

def make_overview(spec, drr_gen, results_json, label='', out_name='ramulamma_overview.png'):
    per_proj = results_json['ramulamma']['per_projection']
    frame_keys = sorted(per_proj.keys())
    proj_map   = {p.proj_key: p for p in spec.projections}

    n_rows = len(frame_keys)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = ['X-ray (real)', 'DRR @ init pose', 'DRR @ final pose', 'Overlay (final)']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

    for row, key in enumerate(frame_keys):
        if key not in proj_map:
            for ax in axes[row]:
                ax.axis('off')
            continue

        proj   = proj_map[key]
        pdata  = per_proj[key]
        success = pdata['success']
        init_go = pdata['initial_go']
        final_go = pdata['final_go']

        R_gt = proj.R_proj.copy()
        t_gt = proj.t_proj.copy()

        # Reproducible perturbation matching run_ramulamma.py
        rng = np.random.default_rng(42 + proj.proj_index)
        delta_rot   = rng.uniform(-PERTURB_ROT_DEG,   PERTURB_ROT_DEG,   3)
        delta_trans = rng.uniform(-PERTURB_TRANS_MM,  PERTURB_TRANS_MM,  3)
        R_init, t_init = perturb_extrinsic(R_gt, t_gt, delta_rot, delta_trans)

        # Use the saved optimized pose delta if available, otherwise fall back to init
        best_delta = np.array(pdata.get('best_pose_delta', [0]*6))
        R_final, t_final = perturb_extrinsic(R_gt, t_gt, best_delta[:3], best_delta[3:])

        # Images
        xray_small  = resize_xray(proj.image_raw)
        drr_init    = render_drr(drr_gen, R_init,  t_init,  invert=True)
        drr_final   = render_drr(drr_gen, R_final, t_final, invert=True)
        overlay     = checkerboard(xray_small, drr_final)

        images   = [xray_small, drr_init, drr_final, overlay]
        subtitles = [
            f'Frame {int(key)}',
            f'Init GO={init_go:.3f}',
            f'Final GO={final_go:.3f}',
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
    suffix = f'  —  {label}' if label else ''
    title = (f'Ramulamma Lumbar Spine Registration{suffix}\n'
             f'{n_succ}/{len(per_proj)} success  '
             f'(mean GO: {results_json["ramulamma"]["mean_initial_go"]:.3f} → '
             f'{results_json["ramulamma"]["mean_final_go"]:.3f})')
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Summary bar chart ─────────────────────────────────────────────────────────

def make_summary(results_json, label='', out_name='ramulamma_summary.png',
                 color_success='#2ecc71', color_fail='#e74c3c'):
    per_proj = results_json['ramulamma']['per_projection']
    keys     = sorted(per_proj.keys())
    init_go  = [per_proj[k]['initial_go']  for k in keys]
    final_go = [per_proj[k]['final_go']    for k in keys]
    success  = [per_proj[k]['success']     for k in keys]
    delta_go = [per_proj[k]['go_delta']    for k in keys]

    x = np.arange(len(keys))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: init vs final GO bars
    ax = axes[0]
    bars_i = ax.bar(x - w/2, init_go,  w, label='Initial GO',  color='#95a5a6', alpha=0.85)
    bars_f = ax.bar(x + w/2, final_go, w, label='Final GO',
                    color=[color_success if s else color_fail for s in success], alpha=0.85)
    ax.axhline(0.60, color='orange', linestyle='--', linewidth=1.2, label='Success threshold (GO<0.60)')
    ax.set_xticks(x); ax.set_xticklabels([f'Frame\n{int(k)}' for k in keys], fontsize=9)
    ax.set_ylabel('GO Score  (lower = better aligned)', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title('Initial vs Final GO Score per Frame', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Annotate bars with values
    for bar in bars_f:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.3f}',
                ha='center', va='bottom', fontsize=7.5)

    # Right: ΔGO improvement
    ax2 = axes[1]
    colors = [color_success if s else color_fail for s in success]
    bars_d = ax2.bar(x, delta_go, color=colors, alpha=0.85, edgecolor='white')
    ax2.axhline(0.05, color='orange', linestyle='--', linewidth=1.2, label='Min improvement (ΔGO>0.05)')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels([f'Frame\n{int(k)}' for k in keys], fontsize=9)
    ax2.set_ylabel('ΔGO  (initial − final,  higher = better)', fontsize=10)
    ax2.set_title('GO Score Improvement per Frame', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    for bar, d in zip(bars_d, delta_go):
        h = bar.get_height()
        ypos = h + 0.01 if h >= 0 else h - 0.03
        ax2.text(bar.get_x() + bar.get_width()/2, ypos, f'{d:+.3f}',
                 ha='center', va='bottom', fontsize=7.5)

    # Legend patches
    success_patch = mpatches.Patch(color=color_success, label='SUCCESS')
    fail_patch    = mpatches.Patch(color=color_fail,    label='FAIL')
    ax2.legend(handles=[success_patch, fail_patch], fontsize=9)

    n_success = sum(success)
    lbl = f' — {label}' if label else ''
    fig.suptitle(
        f'Ramulamma Registration Summary{lbl}\n'
        f'{n_success}/{len(keys)} success  |  Mean GO: '
        f'{results_json["ramulamma"]["mean_initial_go"]:.3f} → '
        f'{results_json["ramulamma"]["mean_final_go"]:.3f}  '
        f'(Δ = {results_json["ramulamma"]["mean_go_delta"]:+.3f})',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Side-by-side comparison ───────────────────────────────────────────────────

def make_comparison(clean_json, instr_json):
    cp = clean_json['ramulamma']['per_projection']
    ip = instr_json['ramulamma']['per_projection']
    ck = sorted(cp.keys()); ik = sorted(ip.keys())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1: final GO per frame (both runs side by side by position index)
    ax = axes[0]
    xc = np.arange(len(ck)); xi = np.arange(len(ik))
    ax.bar(xc - 0.2, [cp[k]['final_go'] for k in ck], 0.38,
           color=['#2ecc71' if cp[k]['success'] else '#e74c3c' for k in ck],
           alpha=0.9, label='Clean (no instruments)')
    ax.bar(xi + 0.2, [ip[k]['final_go'] for k in ik], 0.38,
           color=['#3498db' if ip[k]['success'] else '#e67e22' for k in ik],
           alpha=0.75, label='With instruments', hatch='//')
    ax.axhline(0.60, color='black', linestyle='--', linewidth=1, label='GO<0.60 threshold')
    n_frames = max(len(ck), len(ik))
    ax.set_xticks(range(n_frames))
    ax.set_xticklabels([f'#{i+1}' for i in range(n_frames)], fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Final GO Score', fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Final GO per Frame', fontsize=11, fontweight='bold')

    # Panel 2: mean GO grouped bars
    ax2 = axes[1]
    init_m  = [np.mean([cp[k]['initial_go'] for k in ck]),
               np.mean([ip[k]['initial_go'] for k in ik])]
    final_m = [clean_json['ramulamma']['mean_final_go'],
               instr_json['ramulamma']['mean_final_go']]
    x2 = np.arange(2)
    ax2.bar(x2 - 0.2, init_m,  0.35, color='#95a5a6', alpha=0.85, label='Mean Initial GO')
    ax2.bar(x2 + 0.2, final_m, 0.35, color=['#2ecc71', '#3498db'], alpha=0.9, label='Mean Final GO')
    for i, (ig, fg) in enumerate(zip(init_m, final_m)):
        ax2.text(i - 0.2, ig + 0.02, f'{ig:.3f}', ha='center', fontsize=10, fontweight='bold')
        ax2.text(i + 0.2, fg + 0.02, f'{fg:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax2.axhline(0.60, color='orange', linestyle='--', linewidth=1.2, label='GO<0.60 threshold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['Without\nInstruments', 'With\nInstruments'], fontsize=11)
    ax2.set_ylim(0, 1.1); ax2.set_ylabel('Mean GO Score', fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
    ax2.set_title('Mean GO Comparison', fontsize=11, fontweight='bold')

    # Panel 3: success rate bar chart
    ax3 = axes[2]
    succ_r = [sum(cp[k]['success'] for k in ck) / len(ck) * 100,
              sum(ip[k]['success'] for k in ik) / len(ik) * 100]
    bars = ax3.bar(['Without\nInstruments', 'With\nInstruments'],
                   succ_r, color=['#2ecc71', '#3498db'], alpha=0.9,
                   width=0.4, edgecolor='white')
    for bar, r in zip(bars, succ_r):
        n = int(round(r / 100 * 8))
        ax3.text(bar.get_x() + bar.get_width()/2, r + 1.5,
                 f'{r:.0f}%\n({n}/8)', ha='center', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 115); ax3.set_ylabel('Success Rate (%)', fontsize=10)
    ax3.set_title('Success Rate Comparison', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    fig.suptitle(
        'Ramulamma Registration: Instrument-free Frames  vs  Frames with Instruments',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'ramulamma_comparison.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading result JSONs ...')
    clean_json = json.load(open(CLEAN_JSON))
    instr_json = json.load(open(INSTR_JSON))

    clean_frames = [int(k) for k in sorted(clean_json['ramulamma']['per_projection'].keys())]
    instr_frames = [int(k) for k in sorted(instr_json['ramulamma']['per_projection'].keys())]

    print('Loading clean specimen (without instruments) ...')
    spec_clean = RamulamaLoader(dicom_dir=DICOM_DIR_CLEAN).load(frames=clean_frames, verbose=False)
    drr_clean  = DeepFluoroDRR(spec_clean, hu_threshold=150.0)

    print('Loading instrument specimen (all frames) ...')
    spec_instr = RamulamaLoader(dicom_dir=DICOM_DIR).load(frames=instr_frames, verbose=False)
    drr_instr  = DeepFluoroDRR(spec_instr, hu_threshold=150.0)

    print('\nGenerating overview — without instruments ...')
    make_overview(spec_clean, drr_clean, clean_json,
                  label='Without Instruments (23 clean frames)',
                  out_name='ramulamma_overview_clean.png')

    print('Generating overview — with instruments ...')
    make_overview(spec_instr, drr_instr, instr_json,
                  label='With Instruments (84 frames)',
                  out_name='ramulamma_overview_instruments.png')

    print('Generating summary chart — without instruments ...')
    make_summary(clean_json,
                 label='Without Instruments',
                 out_name='ramulamma_summary_clean.png',
                 color_success='#2ecc71', color_fail='#e74c3c')

    print('Generating summary chart — with instruments ...')
    make_summary(instr_json,
                 label='With Instruments',
                 out_name='ramulamma_summary_instruments.png',
                 color_success='#3498db', color_fail='#e67e22')

    print('Generating comparison figure ...')
    make_comparison(clean_json, instr_json)

    print('\nAll figures saved to results/figures/')


if __name__ == '__main__':
    main()
