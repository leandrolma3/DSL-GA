#!/usr/bin/env python3
"""Generate violin plots of EGIS transition metrics (TCS, RIR, AMS)
by drift type and configuration for the IEEE paper supplementary material.

Reads from: paper_data/egis_transition_metrics.csv
Outputs to:  paper/figures/fig_violin_tcs.pdf
             paper/figures/fig_violin_rir.pdf
             paper/figures/fig_violin_ams.pdf
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# IEEE-style rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "paper_data" / "egis_transition_metrics.csv"
OUTPUT_DIR = SCRIPT_DIR / "paper" / "figures"

CONFIG_ORDER = [
    "EXP-500-NP",
    "EXP-500-P",
    "EXP-500-P03",
    "EXP-1000-NP",
    "EXP-1000-P",
    "EXP-2000-NP",
    "EXP-2000-P",
]

# Display labels for drift categories (title-cased)
DRIFT_ORDER = ["abrupt", "gradual", "noisy", "stationary", "real"]
DRIFT_LABELS = {
    "abrupt": "Abrupt",
    "gradual": "Gradual",
    "noisy": "Noisy",
    "stationary": "Stationary",
    "real": "Real",
}

# Color scheme: NP -> blues, P -> oranges, P03 -> green
CONFIG_COLORS = {
    "EXP-500-NP":  "#1f77b4",   # blue
    "EXP-500-P":   "#ff7f0e",   # orange
    "EXP-500-P03": "#2ca02c",   # green
    "EXP-1000-NP": "#4a90d9",   # lighter blue
    "EXP-1000-P":  "#ffaa44",   # lighter orange
    "EXP-2000-NP": "#7cb9e8",   # lightest blue
    "EXP-2000-P":  "#ffc680",   # lightest orange
}

METRICS = {
    "TCS": {
        "column": "TCS",
        "title": "Transition Change Score by Drift Type and Configuration",
        "ylabel": "TCS",
        "filename": "fig_violin_tcs.pdf",
    },
    "RIR": {
        "column": "RIR",
        "title": "Rule Instability Rate by Drift Type and Configuration",
        "ylabel": "RIR",
        "filename": "fig_violin_rir.pdf",
    },
    "AMS": {
        "column": "AMS",
        "title": "Antecedent Modification Score by Drift Type and Configuration",
        "ylabel": "AMS",
        "filename": "fig_violin_ams.pdf",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(csv_path: Path) -> list[dict]:
    """Load CSV rows as list of dicts."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def aggregate_metric(rows: list[dict], metric_col: str) -> dict:
    """Return {(config_label, drift_type): [float values]} for the given metric."""
    data = defaultdict(list)
    for row in rows:
        config = row["config_label"]
        drift = row["drift_type"]
        val_str = row[metric_col]
        if val_str == "" or val_str.lower() == "nan":
            continue
        try:
            val = float(val_str)
        except ValueError:
            continue
        if np.isfinite(val):
            data[(config, drift)].append(val)
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def generate_violin_figure(
    aggregated: dict,
    metric_info: dict,
    output_path: Path,
) -> None:
    """Create and save a single violin-plot figure for one metric."""

    n_configs = len(CONFIG_ORDER)
    n_drifts = len(DRIFT_ORDER)

    # Spacing parameters
    group_width = 0.8          # total width allocated per drift group (in x-units)
    violin_width_frac = 0.85   # fraction of slot width for each violin
    group_gap = 1.0            # gap between drift-group centres

    # Compute x positions for each config within each drift group
    # Group centres at 0, group_gap, 2*group_gap, ...
    group_centres = [i * group_gap for i in range(n_drifts)]
    slot_width = group_width / n_configs
    offsets = [
        -group_width / 2 + slot_width * (j + 0.5)
        for j in range(n_configs)
    ]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Draw violins for each (drift, config) cell
    for i, drift in enumerate(DRIFT_ORDER):
        cx = group_centres[i]
        for j, config in enumerate(CONFIG_ORDER):
            values = aggregated.get((config, drift), [])
            pos = cx + offsets[j]
            color = CONFIG_COLORS[config]

            if len(values) < 2:
                # Not enough data for a violin -- draw a small marker instead
                if values:
                    ax.plot(pos, values[0], "o", color=color, markersize=4, zorder=5)
                continue

            parts = ax.violinplot(
                values,
                positions=[pos],
                widths=slot_width * violin_width_frac,
                showmeans=True,
                showmedians=True,
                showextrema=False,
            )

            # Style the violin body
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.5)
                pc.set_alpha(0.75)

            # Style mean and median lines
            if "cmeans" in parts:
                parts["cmeans"].set_color("black")
                parts["cmeans"].set_linewidth(0.8)
            if "cmedians" in parts:
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(0.8)
                parts["cmedians"].set_linestyle("--")

    # X-axis: drift type labels at group centres
    ax.set_xticks(group_centres)
    ax.set_xticklabels([DRIFT_LABELS[d] for d in DRIFT_ORDER])

    # Extend x limits so edge violins are not clipped
    ax.set_xlim(
        group_centres[0] - group_gap * 0.55,
        group_centres[-1] + group_gap * 0.55,
    )

    # Y-axis
    ax.set_ylabel(metric_info["ylabel"])
    ax.set_title(metric_info["title"])

    # Grid
    ax.yaxis.grid(True, alpha=0.3, linestyle="-")
    ax.set_axisbelow(True)

    # Legend (colour patches for each config)
    legend_handles = [
        mpatches.Patch(facecolor=CONFIG_COLORS[c], edgecolor="black",
                       linewidth=0.5, label=c)
        for c in CONFIG_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=2,
        framealpha=0.9,
        edgecolor="gray",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {CSV_PATH} ...")
    rows = load_data(CSV_PATH)
    print(f"  Loaded {len(rows)} transition records.")

    for metric_key, metric_info in METRICS.items():
        print(f"\nGenerating violin plot for {metric_key} ...")
        aggregated = aggregate_metric(rows, metric_info["column"])

        # Report data counts per drift type
        for drift in DRIFT_ORDER:
            total = sum(
                len(aggregated.get((c, drift), []))
                for c in CONFIG_ORDER
            )
            print(f"  {DRIFT_LABELS[drift]:12s}: {total:5d} values across {len(CONFIG_ORDER)} configs")

        out_path = OUTPUT_DIR / metric_info["filename"]
        generate_violin_figure(aggregated, metric_info, out_path)
        print(f"  Saved: {out_path}")

    print("\nAll violin plots generated successfully.")


if __name__ == "__main__":
    main()
