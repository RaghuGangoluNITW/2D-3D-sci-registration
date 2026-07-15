#!/usr/bin/env python3
"""
analyse_loss_pde.py — Correlation Analysis of Loss Functions vs PDE
=====================================================================
Reads the CSV produced by sample_loss_pde.py and generates:

  1. Summary stats table — Spearman / Pearson r per loss × (pde_mean, pde_max)
  2. Per-specimen breakdown
  3. Scatter plots: NCC cost vs PDE, GO cost vs PDE (saved as PNG)
  4. Per-DOF marginal plots — how each delta (rot/trans) affects loss & PDE

Usage:
  python analyse_loss_pde.py
  python analyse_loss_pde.py --csv results/loss_pde_samples.csv --out results/loss_pde_figs
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def corr_table(df: pd.DataFrame, loss_cols: list, pde_cols: list) -> pd.DataFrame:
    """Return a DataFrame of Spearman + Pearson r for each loss × pde pair."""
    rows = []
    for lc in loss_cols:
        for pc in pde_cols:
            sub = df[[lc, pc]].dropna()
            if len(sub) < 5:
                continue
            sp_r, sp_p = spearmanr(sub[lc], sub[pc])
            pe_r, pe_p = pearsonr(sub[lc],  sub[pc])
            rows.append({
                'loss':          lc,
                'pde_metric':    pc,
                'n':             len(sub),
                'spearman_r':    round(sp_r, 4),
                'spearman_p':    round(sp_p, 6),
                'pearson_r':     round(pe_r, 4),
                'pearson_p':     round(pe_p, 6),
            })
    return pd.DataFrame(rows)


def scatter_per_xray(df: pd.DataFrame,
                     loss_col: str,
                     pde_col: str,
                     epnp_df: pd.DataFrame = None) -> plt.Figure:
    """
    One subplot per X-ray (proj_key).  Each subplot shows:
      - Scatter of random-pose samples (dots)
      - EPnP point as a cross (×)
      - Vertical dashed line at the EPnP loss value
    Grid size is chosen automatically to be roughly square.
    """
    proj_keys = sorted(df['proj_key'].unique())
    n = len(proj_keys)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.8),
                             squeeze=False)
    fig.suptitle(f'{loss_col}  vs  {pde_col}', fontsize=12, fontweight='bold')

    # Global axis limits for consistency across subplots
    x_all = df[loss_col].dropna()
    y_all = df[pde_col].dropna()
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    x_pad = (x_max - x_min) * 0.05 or 0.05
    y_pad = (y_max - y_min) * 0.05 or 1.0

    for idx, pk in enumerate(proj_keys):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = df[df['proj_key'] == pk]

        ax.scatter(sub[loss_col], sub[pde_col],
                   s=12, alpha=0.55, color='steelblue', zorder=2)

        # EPnP point + vertical line
        if epnp_df is not None:
            epnp_row = epnp_df[epnp_df['proj_key'] == pk]
            if not epnp_row.empty and loss_col in epnp_row.columns:
                ex = float(epnp_row[loss_col].iloc[0])
                ey = float(epnp_row[pde_col].iloc[0]) if pde_col in epnp_row.columns else None
                ax.axvline(ex, color='crimson', linewidth=1.2,
                           linestyle='--', zorder=3, alpha=0.8)
                if ey is not None:
                    ax.scatter([ex], [ey], marker='x', s=80, linewidths=2,
                               color='crimson', zorder=5)

        # Spearman r
        s = sub[[loss_col, pde_col]].dropna()
        if len(s) >= 4:
            sp_r, _ = spearmanr(s[loss_col], s[pde_col])
            ax.set_title(f"{pk}\nr={sp_r:.2f}", fontsize=7.5)
        else:
            ax.set_title(pk, fontsize=7.5)

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.tick_params(labelsize=6)
        ax.set_xlabel(loss_col, fontsize=6.5)
        ax.set_ylabel(pde_col + ' (mm)', fontsize=6.5)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def dof_marginal_plot(df: pd.DataFrame,
                      loss_col: str,
                      pde_col: str,
                      out_path: Path) -> None:
    """
    6-panel plot: for each DOF (rot_x/y/z, trans_x/y/z), show how the
    delta value relates to both loss and PDE (using hexbin density).
    """
    dof_cols = [
        'rot_delta_x', 'rot_delta_y', 'rot_delta_z',
        'trans_delta_x', 'trans_delta_y', 'trans_delta_z',
    ]
    fig, axes = plt.subplots(2, 6, figsize=(20, 6))
    fig.suptitle(f'Per-DOF marginals  |  loss={loss_col}  pde={pde_col}', fontsize=11)

    for col_idx, dof in enumerate(dof_cols):
        sub = df[[dof, loss_col, pde_col]].dropna()
        # Top row: dof vs loss
        axes[0, col_idx].hexbin(sub[dof], sub[loss_col], gridsize=20, cmap='Blues')
        axes[0, col_idx].set_xlabel(dof, fontsize=7)
        axes[0, col_idx].set_ylabel(loss_col, fontsize=7)
        # Bottom row: dof vs pde
        axes[1, col_idx].hexbin(sub[dof], sub[pde_col], gridsize=20, cmap='Reds')
        axes[1, col_idx].set_xlabel(dof, fontsize=7)
        axes[1, col_idx].set_ylabel(pde_col + ' (mm)', fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results/loss_pde_samples.csv',
                        help='Input CSV from sample_loss_pde.py')
    parser.add_argument('--epnp_csv', default='results/loss_pde_epnp.csv',
                        help='EPnP baseline CSV from sample_loss_pde.py')
    parser.add_argument('--out', default=None,
                        help='Output directory (default: <csv_stem>_analysis next to the CSV)')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    # Derive output dir from CSV name unless explicitly given
    if args.out is None:
        out_dir = csv_path.parent / (csv_path.stem + '_analysis')
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, {df['specimen'].nunique()} specimens, "
          f"{df['proj_key'].nunique()} unique proj_keys")
    print(f"  Columns: {list(df.columns)}")
    print()

    # Load EPnP baseline CSV if present
    epnp_df = None
    epnp_path = Path(args.epnp_csv)
    if epnp_path.exists():
        epnp_df = pd.read_csv(epnp_path)
        print(f"Loaded EPnP baseline: {len(epnp_df)} rows from {epnp_path}")
    else:
        print(f"EPnP CSV not found ({epnp_path}), crosses will be omitted")
    print()

    # Drop rows where DRR had no coverage (loss = 1.0 sentinel)
    df_valid = df[df['drr_coverage'] >= 0.05].copy()
    print(f"  After coverage filter (>=5%): {len(df_valid)} rows "
          f"({len(df)-len(df_valid)} dropped)")

    loss_cols = [
        'ncc_cost', 'go_cost', 'lncc_cost', 'ms_ncc_cost', 'grad_ncc_cost',
        'mi_cost', 'nmi_cost', 'eod_cost', 'cr_cost', 'src_cost',
        'gd_cost', 'pi_cost', 'grad_ms_ncc_cost', 'ngi_cost', 'lgncc_cost',
    ]
    # Keep only columns present in the CSV (backwards compatible)
    loss_cols = [c for c in loss_cols if c in df_valid.columns]
    pde_cols  = ['pde_mean_mm', 'pde_max_mm']

    # ── 1. Global correlation table ─────────────────────────────────────────
    print("\n── Global Correlation (all specimens) ──────────────────────────")
    ct_global = corr_table(df_valid, loss_cols, pde_cols)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(ct_global.to_string(index=False))

    ct_global.to_csv(out_dir / 'correlation_global.csv', index=False)
    print(f"\n  Saved: {out_dir / 'correlation_global.csv'}")

    # ── 2. Per-specimen breakdown ────────────────────────────────────────────
    print("\n── Per-Specimen Correlations ───────────────────────────────────")
    per_spec_rows = []
    for spec in sorted(df_valid['specimen'].unique()):
        sub = df_valid[df_valid['specimen'] == spec]
        for lc in loss_cols:
            for pc in pde_cols:
                s = sub[[lc, pc]].dropna()
                if len(s) < 5:
                    continue
                sp_r, _ = spearmanr(s[lc], s[pc])
                per_spec_rows.append({
                    'specimen': spec, 'loss': lc, 'pde_metric': pc,
                    'n': len(s), 'spearman_r': round(sp_r, 4)
                })
    df_per_spec = pd.DataFrame(per_spec_rows)
    print(df_per_spec.to_string(index=False))
    df_per_spec.to_csv(out_dir / 'correlation_per_specimen.csv', index=False)

    # ── 3. Scatter plots — one figure per loss, one subplot per X-ray ────────
    print("\n── Scatter plots ──────────────────────────────────────────────")
    pde_col = 'pde_mean_mm'
    for lc in loss_cols:
        fig = scatter_per_xray(df_valid, lc, pde_col, epnp_df=epnp_df)
        scatter_path = out_dir / f'scatter_{lc}.png'
        fig.savefig(scatter_path, dpi=130)
        plt.close(fig)
        print(f"  Saved: {scatter_path}")

    # ── 4. Per-DOF marginal plots ────────────────────────────────────────────
    print("\n── DOF marginal plots ─────────────────────────────────────────")
    for lc in loss_cols:
        for pc in ['pde_mean_mm']:
            dof_marginal_plot(df_valid, lc, pc,
                              out_dir / f'dof_marginals_{lc}_{pc}.png')

    # ── 5. PDE distribution ──────────────────────────────────────────────────
    print("\n── PDE distribution ───────────────────────────────────────────")
    print(df_valid['pde_mean_mm'].describe().round(2).to_string())
    print(f"\n  % samples with PDE < 10 mm : "
          f"{(df_valid['pde_mean_mm'] < 10).mean()*100:.1f}%")
    print(f"  % samples with PDE < 20 mm : "
          f"{(df_valid['pde_mean_mm'] < 20).mean()*100:.1f}%")
    print(f"  % samples with PDE < 50 mm : "
          f"{(df_valid['pde_mean_mm'] < 50).mean()*100:.1f}%")

    # ── 6. Loss distribution ─────────────────────────────────────────────────
    print("\n── Loss distribution ──────────────────────────────────────────")
    for lc in loss_cols:
        print(f"\n  {lc}:")
        print(df_valid[lc].describe().round(4).to_string())

    # ── 7. Rank correlation at different PDE bins ────────────────────────────
    print("\n── Spearman r within PDE distance bins ────────────────────────")
    bins = [(0, 10), (10, 30), (30, 60), (60, 120), (120, 9999)]
    bin_rows = []
    for lo, hi in bins:
        mask = (df_valid['pde_mean_mm'] >= lo) & (df_valid['pde_mean_mm'] < hi)
        sub  = df_valid[mask]
        if len(sub) < 10:
            continue
        row = {'pde_bin': f'{lo}-{hi}mm', 'n': len(sub)}
        for lc in loss_cols:
            s = sub[[lc, 'pde_mean_mm']].dropna()
            if len(s) >= 5:
                sp_r, _ = spearmanr(s[lc], s['pde_mean_mm'])
                row[f'{lc}_spearman_r'] = round(sp_r, 4)
        bin_rows.append(row)
    df_bins = pd.DataFrame(bin_rows)
    print(df_bins.to_string(index=False))
    df_bins.to_csv(out_dir / 'correlation_by_pde_bin.csv', index=False)

    # ── 8. Poses-better-than-best-PDE count ──────────────────────────────────
    print("\n── Poses with lower loss than at lowest-PDE pose ──────────────")

    # Determine sign of each loss from global Spearman r
    # negative_corr losses: lower loss value = WORSE alignment → invert logic
    neg_corr = set()
    for lc in loss_cols:
        s = df_valid[[lc, 'pde_mean_mm']].dropna()
        if len(s) >= 5:
            sp_r, _ = spearmanr(s[lc], s['pde_mean_mm'])
            if sp_r < 0:
                neg_corr.add(lc)
    print(f"  Negative-correlation losses (inverted count): "
          f"{sorted(neg_corr)}")

    # For each X-ray, find the pose with the lowest PDE, then count how many
    # other poses beat it on each loss function.
    count_rows = []
    for pk in sorted(df_valid['proj_key'].unique()):
        sub = df_valid[df_valid['proj_key'] == pk].copy()
        if len(sub) < 2:
            continue
        best_idx = sub['pde_mean_mm'].idxmin()
        row = {'proj_key': pk,
               'best_pde_mm': sub.loc[best_idx, 'pde_mean_mm'],
               'n_poses': len(sub)}
        for lc in loss_cols:
            best_loss = sub.loc[best_idx, lc]
            others    = sub.drop(index=best_idx)[lc].dropna()
            if lc in neg_corr:
                # negative corr: better alignment should have HIGHER loss value
                count = int((others > best_loss).sum())
            else:
                count = int((others < best_loss).sum())
            row[lc] = count
        count_rows.append(row)

    df_counts = pd.DataFrame(count_rows)
    df_counts.to_csv(out_dir / 'better_than_best_pde_counts.csv', index=False)
    print(df_counts[['proj_key', 'best_pde_mm', 'n_poses'] + loss_cols]
          .to_string(index=False))

    # Whisker plot — 3×1 grid sorted by min / max / mean
    n_poses_per_xray = int(df_counts['n_poses'].mode()[0])
    colors = plt.cm.tab20.colors

    sort_keys = [
        ('min',  lambda d: np.min(d)),
        ('max',  lambda d: np.max(d)),
        ('mean', lambda d: np.mean(d)),
    ]

    neg_label = ', '.join(lc.replace('_cost', '') for lc in sorted(neg_corr))
    suptitle  = (
        f'Poses beating best-PDE alignment on each loss  '
        f'(per X-ray, n_poses={n_poses_per_xray}, {len(df_counts)} X-rays)\n'
        f'★ negative-corr losses inverted: {neg_label}'
    )

    fig, axes = plt.subplots(3, 1,
                             figsize=(max(12, len(loss_cols) * 1.0), 14),
                             sharex=False)
    fig.suptitle(suptitle, fontsize=10, y=1.01)

    for ax, (sort_label, sort_fn) in zip(axes, sort_keys):
        # Sort loss_cols by the chosen statistic (ascending)
        data_all   = {lc: df_counts[lc].values for lc in loss_cols}
        order      = sorted(loss_cols, key=lambda lc: sort_fn(data_all[lc]))
        data_sorted = [data_all[lc] for lc in order]
        labels      = [lc.replace('_cost', '') for lc in order]

        bp = ax.boxplot(data_sorted, patch_artist=True, notch=False,
                        medianprops=dict(color='crimson', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
        ax.axhline(n_poses_per_xray / 2, color='grey', linestyle='--',
                   linewidth=0.9, label='chance (n/2)')
        ax.set_ylabel('# poses better\nthan lowest-PDE', fontsize=9)
        ax.set_title(f'Sorted by {sort_label}', fontsize=10)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    plt.tight_layout()
    whisker_path = out_dir / 'better_than_best_pde_whisker.png'
    fig.savefig(whisker_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {whisker_path}")

    # ── 9. Correct-side fraction per X-ray ──────────────────────────────────
    # A random pose is "on the correct side" if the sign of its loss deviation
    # from the EPnP loss matches the expected direction given its PDE deviation.
    #   pos-corr loss: correct when (loss - epnp_loss) and (PDE - epnp_PDE)
    #                  have the same sign.
    #   neg-corr loss: correct when they have opposite sign.
    print("\n── Correct-side fraction per X-ray ────────────────────────────────")

    thresholds = [0.95, 0.90, 0.80]

    # Build a lookup of EPnP values keyed by proj_key
    epnp_lookup = {}
    if epnp_df is not None and not epnp_df.empty:
        for _, row in epnp_df.iterrows():
            epnp_lookup[row['proj_key']] = row

    correct_counts = {thresh: {lc: 0 for lc in loss_cols}
                      for thresh in thresholds}
    n_xrays_total = 0

    for pk in sorted(df_valid['proj_key'].unique()):
        if pk not in epnp_lookup:
            continue
        epnp_row = epnp_lookup[pk]
        sub = df_valid[df_valid['proj_key'] == pk]
        if len(sub) < 5:
            continue
        n_xrays_total += 1

        epnp_pde = float(epnp_row['pde_mean_mm'])
        pde_delta = sub['pde_mean_mm'].values - epnp_pde   # + means worse

        for lc in loss_cols:
            if lc not in epnp_row.index:
                continue
            epnp_loss = float(epnp_row[lc])
            loss_delta = sub[lc].values - epnp_loss

            if lc in neg_corr:
                # neg-corr: lower loss = WORSE alignment → higher PDE
                # correct: loss_delta and pde_delta have OPPOSITE sign
                correct = (loss_delta * pde_delta) < 0
            else:
                # pos-corr: correct when same sign
                correct = (loss_delta * pde_delta) > 0

            frac_correct = correct.mean()  # fraction of poses on correct side

            for thresh in thresholds:
                if frac_correct >= thresh:
                    correct_counts[thresh][lc] += 1

    print(f"  X-rays analysed: {n_xrays_total}")
    for thresh in thresholds:
        row_str = "  | ".join(
            f"{lc.replace('_cost',''):>14}: {correct_counts[thresh][lc]:>3}"
            for lc in loss_cols
        )
        print(f"  ≥{int(thresh*100):>3}%  {row_str}")

    # Save to CSV
    rows_cs = []
    for lc in loss_cols:
        r = {'loss': lc}
        for thresh in thresholds:
            r[f'n_xrays_ge{int(thresh*100)}pct'] = correct_counts[thresh][lc]
        rows_cs.append(r)
    df_cs = pd.DataFrame(rows_cs)
    df_cs.to_csv(out_dir / 'correct_side_counts.csv', index=False)

    # Visualise — grouped bar chart, one group per loss, one bar per threshold
    short_names = [lc.replace('_cost', '') for lc in loss_cols]
    x = np.arange(len(loss_cols))
    bar_width = 0.25
    offsets    = [-bar_width, 0, bar_width]
    thresh_colors = ['#2196F3', '#4CAF50', '#FF9800']   # blue, green, orange
    thresh_labels = ['≥95%', '≥90%', '≥80%']

    fig_cs, ax_cs = plt.subplots(figsize=(max(14, len(loss_cols) * 1.1), 6))

    for i, (thresh, offset, color, label) in enumerate(
            zip(thresholds, offsets, thresh_colors, thresh_labels)):
        counts = [correct_counts[thresh][lc] for lc in loss_cols]
        bars = ax_cs.bar(x + offset, counts, bar_width,
                         label=label, color=color, alpha=0.82,
                         edgecolor='white', linewidth=0.5)
        # annotate value on top of each bar
        for bar, val in zip(bars, counts):
            if val > 0:
                ax_cs.text(bar.get_x() + bar.get_width() / 2,
                           bar.get_height() + 0.2,
                           str(val), ha='center', va='bottom', fontsize=7)

    ax_cs.axhline(n_xrays_total, color='black', linestyle='--',
                  linewidth=0.9, label=f'total X-rays ({n_xrays_total})')
    ax_cs.set_xticks(x)
    ax_cs.set_xticklabels(short_names, rotation=35, ha='right', fontsize=10)
    ax_cs.set_ylabel('Number of X-rays', fontsize=11)
    ax_cs.set_title(
        'X-rays where ≥95/90/80 % of random poses are on the correct side\n'
        '(loss deviation from EPnP has expected sign relative to PDE deviation)',
        fontsize=11
    )
    ax_cs.set_ylim(0, n_xrays_total + 4)
    ax_cs.legend(fontsize=10)
    plt.tight_layout()

    cs_path = out_dir / 'correct_side_counts.png'
    fig_cs.savefig(cs_path, dpi=150, bbox_inches='tight')
    plt.close(fig_cs)
    print(f"\n  Saved: {cs_path}")

    print(f"\n✓ All outputs in: {out_dir}")


if __name__ == '__main__':
    main()
