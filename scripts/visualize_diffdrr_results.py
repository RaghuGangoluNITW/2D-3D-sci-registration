#!/usr/bin/env python3
"""
visualize_diffdrr_results.py
============================
Summary dashboard for swaroopa_diffdrr_results.json.

Produces results/figures/diffdrr_summary.png with four panels:
  1. Per-frame PDE: initial vs final (sorted by frame key), bars coloured
     green=success / red=fail.
  2. PDE improvement  (Δ = initial − final), descending order.
  3. Per-frame GO cost: initial vs final.
  4. Runtime per frame (seconds).

No DRR rendering required — purely from the JSON.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS = Path('results/swaroopa_diffdrr_results.json')
OUT     = Path('results/figures/diffdrr_summary.png')
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
with open(RESULTS) as f:
    data = json.load(f)

pp     = data['swaroopa']['per_projection']
frames = sorted(pp.keys())

init_pde  = np.array([pp[k]['initial_pde_mm']  for k in frames])
final_pde = np.array([pp[k]['final_pde_mm']     for k in frames])
delta_pde = init_pde - final_pde
init_go   = np.array([pp[k]['initial_go']       for k in frames])
final_go  = np.array([pp[k]['final_go']         for k in frames])
runtime   = np.array([pp[k]['runtime_s']        for k in frames])
success   = np.array([pp[k]['success'] == 'True' for k in frames])

n = len(frames)
x = np.arange(n)
ap_mask  = np.array([k.startswith('ap')  for k in frames])
lat_mask = np.array([k.startswith('lat') for k in frames])

C_SUCC = '#44dd88'
C_FAIL = '#ee4444'
C_INIT = '#aaaaaa'
C_AP   = '#66aaff'
C_LAT  = '#ffaa44'
BG     = '#111111'
FG     = '#dddddd'

matplotlib.rcParams.update({
    'text.color':        FG,
    'axes.labelcolor':   FG,
    'xtick.color':       FG,
    'ytick.color':       FG,
    'axes.edgecolor':    '#444444',
    'axes.facecolor':    '#1e1e1e',
    'figure.facecolor':  BG,
    'grid.color':        '#333333',
    'grid.linestyle':    '--',
    'grid.alpha':        0.6,
})

fig, axes = plt.subplots(4, 1, figsize=(22, 20))
fig.suptitle(
    f'DiffDRR Registration — Swaroopa  '
    f'({n} frames  |  success {success.sum()}/{n} = {success.mean()*100:.0f}%  |  '
    f'mean PDE {init_pde.mean():.1f} → {final_pde.mean():.1f} mm)',
    fontsize=15, fontweight='bold', y=0.995,
)

def _setup(ax, title, ylabel):
    ax.set_facecolor('#1e1e1e')
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(frames, rotation=55, ha='right', fontsize=7.5)
    ax.grid(axis='y')
    ax.tick_params(axis='x', length=3)

# ── Panel 1: initial vs final PDE ────────────────────────────────────────────
ax = axes[0]
bar_colors = [C_SUCC if s else C_FAIL for s in success]
ax.bar(x - 0.2, init_pde,  width=0.38, color=C_INIT, alpha=0.6, label='Initial PDE')
ax.bar(x + 0.2, final_pde, width=0.38, color=bar_colors, alpha=0.9, label='Final PDE')
# 30 mm threshold line
ax.axhline(30, color='yellow', lw=0.8, ls=':', alpha=0.7, label='30 mm threshold')
# AP / LAT zone shading
for i, k in enumerate(frames):
    c = '#002244' if k.startswith('ap') else '#442200'
    ax.axvspan(i - 0.5, i + 0.5, color=c, alpha=0.25, zorder=0)
_setup(ax, 'Per-frame PDE: initial (grey) vs final (green=success / red=fail)', 'PDE (mm)')
legend_patches = [
    mpatches.Patch(color=C_INIT,  alpha=0.6, label='Initial PDE'),
    mpatches.Patch(color=C_SUCC,  alpha=0.9, label='Final PDE — success'),
    mpatches.Patch(color=C_FAIL,  alpha=0.9, label='Final PDE — fail'),
    mpatches.Patch(color='#002244', alpha=0.5, label='AP frame'),
    mpatches.Patch(color='#442200', alpha=0.5, label='LAT frame'),
]
ax.legend(handles=legend_patches, fontsize=8, loc='upper right', ncol=3)

# ── Panel 2: PDE improvement ──────────────────────────────────────────────────
ax = axes[1]
sort_idx = np.argsort(delta_pde)[::-1]
sorted_frames = [frames[i] for i in sort_idx]
sorted_delta  = delta_pde[sort_idx]
sorted_succ   = success[sort_idx]
colors2 = [C_SUCC if s else C_FAIL for s in sorted_succ]
bars = ax.bar(x, sorted_delta, color=colors2, alpha=0.85)
ax.axhline(0, color=FG, lw=0.7, alpha=0.5)
ax.set_facecolor('#1e1e1e')
ax.set_title('PDE improvement  (Δ = initial − final, higher is better) — sorted descending',
             fontsize=11, pad=6)
ax.set_ylabel('ΔPDE (mm)', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(sorted_frames, rotation=55, ha='right', fontsize=7.5)
ax.grid(axis='y')
# Annotate numeric value on each bar
for xi, (val, suc) in enumerate(zip(sorted_delta, sorted_succ)):
    ax.text(xi, val + (2 if val >= 0 else -4),
            f'{val:+.0f}', ha='center', va='bottom' if val >= 0 else 'top',
            fontsize=6.5, color=FG)

# ── Panel 3: GO cost ──────────────────────────────────────────────────────────
ax = axes[2]
ax.bar(x - 0.2, init_go,  width=0.38, color=C_INIT, alpha=0.6, label='Initial GO')
ax.bar(x + 0.2, final_go, width=0.38, color=bar_colors, alpha=0.9, label='Final GO')
ax.axhline(0.5, color='yellow', lw=0.8, ls=':', alpha=0.7, label='GO = 0.5')
ax.axhline(1.0, color='orange',  lw=0.7, ls=':', alpha=0.5, label='GO = 1.0 (penalty)')
for i, k in enumerate(frames):
    c = '#002244' if k.startswith('ap') else '#442200'
    ax.axvspan(i - 0.5, i + 0.5, color=c, alpha=0.25, zorder=0)
_setup(ax, 'Per-frame GO cost: initial (grey) vs final', 'Gradient Orientation cost')
ax.legend(fontsize=8, loc='upper right', ncol=4)

# ── Panel 4: runtime ─────────────────────────────────────────────────────────
ax = axes[3]
rt_colors = [C_AP if k.startswith('ap') else C_LAT for k in frames]
ax.bar(x, runtime / 60, color=rt_colors, alpha=0.85)
ax.axhline(runtime.mean() / 60, color='white', lw=0.9, ls='--', alpha=0.7,
           label=f'Mean = {runtime.mean()/60:.1f} min')
_setup(ax, 'Runtime per frame', 'Time (minutes)')
legend_patches2 = [
    mpatches.Patch(color=C_AP,  alpha=0.85, label='AP frame'),
    mpatches.Patch(color=C_LAT, alpha=0.85, label='LAT frame'),
]
ax.legend(handles=legend_patches2 + [
    mpatches.Patch(color='white', alpha=0.7, label=f'Mean {runtime.mean()/60:.1f} min')],
    fontsize=8, loc='upper right')

# ── Print summary table ───────────────────────────────────────────────────────
print('\n══════════════════════════════════════════════════════════════')
print(f'  DiffDRR Results — Swaroopa ({n} frames)')
print('══════════════════════════════════════════════════════════════')
print(f'  Success rate     : {success.sum()}/{n} = {success.mean()*100:.1f}%')
print(f'  Mean initial PDE : {init_pde.mean():.1f} mm')
print(f'  Mean final PDE   : {final_pde.mean():.1f} mm')
print(f'  Mean ΔPDE        : {delta_pde.mean():+.1f} mm')
print(f'  Mean initial GO  : {init_go.mean():.4f}')
print(f'  Mean final GO    : {final_go.mean():.4f}')
print(f'  Total runtime    : {runtime.sum()/3600:.2f} h')
print()
print(f'  {"Frame":<12}  {"Init PDE":>9}  {"Final PDE":>9}  {"ΔPDE":>7}  {"Init GO":>8}  {"Final GO":>8}  {"OK":>4}')
print(f'  {"-"*12}  {"-"*9}  {"-"*9}  {"-"*7}  {"-"*8}  {"-"*8}  {"-"*4}')
for k in frames:
    r = pp[k]
    ok = '✓' if r['success'] == 'True' else '✗'
    print(f'  {k:<12}  {r["initial_pde_mm"]:>9.1f}  {r["final_pde_mm"]:>9.1f}  '
          f'{r["initial_pde_mm"]-r["final_pde_mm"]:>+7.1f}  '
          f'{r["initial_go"]:>8.4f}  {r["final_go"]:>8.4f}  {ok:>4}')
print('══════════════════════════════════════════════════════════════\n')

plt.tight_layout(rect=[0, 0, 1, 0.993])
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'Saved → {OUT}')
