#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    specimens = []
    per_proj = []
    for sid, s in data.items():
        if not isinstance(s, dict) or "per_projection" not in s:
            continue
        specimens.append(
            {
                "specimen": sid,
                "n_projections": int(s.get("n_projections", 0)),
                "mean_final_pde": float(s.get("mean_final_pde", np.nan)),
                "max_final_pde": float(s.get("max_final_pde", np.nan)),
                "success_rate": float(s.get("success_rate", np.nan)),
            }
        )
        for proj_id, p in s.get("per_projection", {}).items():
            per_proj.append(
                {
                    "specimen": sid,
                    "projection": proj_id,
                    "initial_pde_mm": float(p.get("initial_pde_mm", np.nan)),
                    "final_pde_mm": float(p.get("final_pde_mm", np.nan)),
                    "success": bool(p.get("success", False)),
                }
            )
    return specimens, per_proj


def save_dashboard(specimens, per_proj, out_path: Path):
    specimen_ids = [s["specimen"] for s in specimens]
    success_pct = [100.0 * s["success_rate"] for s in specimens]
    mean_pde = [s["mean_final_pde"] for s in specimens]
    max_pde = [s["max_final_pde"] for s in specimens]

    all_final = np.array([p["final_pde_mm"] for p in per_proj], dtype=float)
    all_success = np.array([p["success"] for p in per_proj], dtype=bool)
    all_init = np.array([p["initial_pde_mm"] for p in per_proj], dtype=float)

    overall_success = 100.0 * np.mean(all_success) if len(all_success) else 0.0
    overall_mean_pde = float(np.mean(all_final)) if len(all_final) else np.nan

    fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=160)
    fig.suptitle(
        f"DeepFluoro Registration Summary  |  Success: {overall_success:.1f}% ({all_success.sum()}/{len(all_success)})  |  Mean PDE: {overall_mean_pde:.2f} mm",
        fontsize=13,
        fontweight="bold",
    )

    # 1) Success by specimen
    ax = axs[0, 0]
    colors = ["#2ca02c" if v >= 99.9 else "#ff7f0e" for v in success_pct]
    bars = ax.bar(specimen_ids, success_pct, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(90, color="red", linestyle="--", linewidth=1, label="90% target")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success Rate by Specimen")
    ax.legend(loc="lower right", fontsize=8)
    for b, v in zip(bars, success_pct):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", rotation=20)

    # 2) Mean/Max PDE by specimen
    ax = axs[0, 1]
    x = np.arange(len(specimen_ids))
    w = 0.38
    b1 = ax.bar(x - w / 2, mean_pde, width=w, label="Mean final PDE", color="#1f77b4")
    b2 = ax.bar(x + w / 2, max_pde, width=w, label="Max final PDE", color="#d62728")
    ax.axhline(10, color="red", linestyle="--", linewidth=1, label="10 mm threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(specimen_ids, rotation=20)
    ax.set_ylabel("PDE (mm)")
    ax.set_title("Per-Specimen PDE")
    ax.legend(fontsize=8)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    # 3) Per-projection final PDE (sorted)
    ax = axs[1, 0]
    order = np.argsort(all_final)
    sorted_final = all_final[order]
    sorted_success = all_success[order]
    c = np.where(sorted_success, "#2ca02c", "#d62728")
    ax.bar(np.arange(len(sorted_final)), sorted_final, color=c, edgecolor="black", linewidth=0.3)
    ax.axhline(10, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Projection index (sorted by final PDE)")
    ax.set_ylabel("Final PDE (mm)")
    ax.set_title("Per-Projection Final PDE")

    # 4) CDF improvement view
    ax = axs[1, 1]
    sorted_init = np.sort(all_init)
    sorted_fin = np.sort(all_final)
    y_init = np.arange(1, len(sorted_init) + 1) / len(sorted_init)
    y_fin = np.arange(1, len(sorted_fin) + 1) / len(sorted_fin)
    ax.plot(sorted_init, y_init, label="Initial PDE", color="#7f7f7f", linewidth=2)
    ax.plot(sorted_fin, y_fin, label="Final PDE", color="#1f77b4", linewidth=2)
    ax.axvline(10, color="red", linestyle="--", linewidth=1, label="10 mm")
    ax.set_xlabel("PDE (mm)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF: Initial vs Final PDE")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_summary_table_png(specimens, out_path: Path):
    headers = ["Specimen", "Success", "Mean PDE (mm)", "Max PDE (mm)"]
    rows = []
    for s in specimens:
        success = f"{100*s['success_rate']:.0f}% ({int(round(s['success_rate']*s['n_projections']))}/{s['n_projections']})"
        rows.append([
            s["specimen"],
            success,
            f"{s['mean_final_pde']:.2f}",
            f"{s['max_final_pde']:.2f}",
        ])

    fig, ax = plt.subplots(figsize=(9.5, 2.2 + 0.45 * len(rows)), dpi=180)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2f5597")
        elif c == 1:
            txt = cell.get_text().get_text()
            cell.set_facecolor("#d9ead3" if txt.startswith("100%") else "#fce5cd")

    fig.suptitle("DeepFluoro Per-Specimen Results", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create DeepFluoro results visualizations.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/deepfluoro_results.json"),
        help="Path to results JSON",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for figures",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    specimens, per_proj = load_results(args.results)
    if not specimens or not per_proj:
        raise RuntimeError(f"No valid data found in {args.results}")

    # sort for stable ordering
    specimens = sorted(specimens, key=lambda x: x["specimen"])

    dashboard_path = args.out_dir / "deepfluoro_dashboard.png"
    table_path = args.out_dir / "deepfluoro_summary_table.png"

    save_dashboard(specimens, per_proj, dashboard_path)
    save_summary_table_png(specimens, table_path)

    print("Saved:")
    print(f"  {dashboard_path}")
    print(f"  {table_path}")


if __name__ == "__main__":
    main()
