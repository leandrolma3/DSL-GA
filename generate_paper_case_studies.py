#!/usr/bin/env python3
"""
Generate case study figures for the IEEE TKDE paper.

Produces two composite 2x2 figures:
  - fig_case_rbf_gradual.pdf   (RBF_Gradual_Moderate, EXP-500-NP)
  - fig_case_stagger_abrupt.pdf (STAGGER_Abrupt_Chain, EXP-500-NP)

Each figure contains:
  (a) Rule Evolution Counts       (top-left)
  (b) Rule Complexity Over Time   (top-right)
  (c) Transition Metrics Over Time (bottom-left)
  (d) Rule Turnover Proportions   (bottom-right)

Usage:
    python generate_paper_case_studies.py
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "paper_data")
FIG_DIR = os.path.join(BASE_DIR, "paper", "figures")

AST_CSV = os.path.join(DATA_DIR, "ast_chunk_quantitatives.csv")
EVOL_CSV = os.path.join(DATA_DIR, "evolution_analysis_summary.csv")
TRANS_CSV = os.path.join(DATA_DIR, "egis_transition_metrics.csv")

# ---------------------------------------------------------------------------
# Style configuration (IEEE-compatible)
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
    "legend.framealpha": 0.85,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# EGIS primary blue
EGIS_BLUE = "#2166ac"

# Colors for rule categories
COLOR_UNCHANGED = "#999999"   # gray
COLOR_MODIFIED  = "#2166ac"   # blue (EGIS blue)
COLOR_NEW       = "#2ca02c"   # green
COLOR_DELETED   = "#d62728"   # red

# Colors for transition metrics
COLOR_TCS = "#e6550d"  # orange
COLOR_RIR = "#3182bd"  # blue
COLOR_AMS = "#31a354"  # green

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASETS = {
    "RBF_Gradual_Moderate": {
        "title": "Case Study: RBF Gradual Moderate (EXP-500-NP)",
        "output": "fig_case_rbf_gradual.pdf",
        "drift_type": "gradual",
        # Gradual drift roughly between instances 4000-8000 -> chunks 8-16
        "drift_annotation": {
            "type": "shaded_region",
            "start": 8,
            "end": 16,
            "label": "Drift region",
        },
    },
    "STAGGER_Abrupt_Chain": {
        "title": "Case Study: STAGGER Abrupt Chain (EXP-500-NP)",
        "output": "fig_case_stagger_abrupt.pdf",
        "drift_type": "abrupt",
        # Abrupt drifts at ~4000 and ~8000 instances -> chunks 8 and 16
        "drift_annotation": {
            "type": "vertical_lines",
            "positions": [8, 16],
            "label": "Drift point",
        },
    },
}

CONFIG_LABEL = "EXP-500-NP"


# ---------------------------------------------------------------------------
# Helper: load CSV safely
# ---------------------------------------------------------------------------
def load_csv(path, name):
    """Load a CSV file; return None and warn if missing."""
    if not os.path.isfile(path):
        warnings.warn(f"WARNING: {name} not found at {path}")
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as exc:
        warnings.warn(f"WARNING: Failed to read {name}: {exc}")
        return None


def filter_dataset(df, dataset, config_label=CONFIG_LABEL):
    """Filter dataframe by config_label and dataset name."""
    if df is None:
        return None
    mask = (df["config_label"] == config_label) & (df["dataset"] == dataset)
    filtered = df[mask].copy()
    if filtered.empty:
        warnings.warn(
            f"WARNING: No data for dataset='{dataset}', "
            f"config_label='{config_label}'"
        )
        return None
    return filtered


# ---------------------------------------------------------------------------
# Drift annotation helpers
# ---------------------------------------------------------------------------
def add_drift_annotation(ax, drift_cfg, x_values):
    """Add drift markers (shaded region or vertical lines) to an axes."""
    if drift_cfg is None:
        return

    dtype = drift_cfg.get("type")
    if dtype == "shaded_region":
        start = drift_cfg["start"]
        end = drift_cfg["end"]
        ax.axvspan(start, end, alpha=0.10, color="red", zorder=0,
                   label=drift_cfg.get("label", "Drift region"))
    elif dtype == "shaded_regions":
        for i, (start, end) in enumerate(drift_cfg["regions"]):
            label = drift_cfg.get("label", "Drift region") if i == 0 else None
            ax.axvspan(start, end, alpha=0.10, color="red", zorder=0,
                       label=label)
    elif dtype == "vertical_lines":
        for i, pos in enumerate(drift_cfg["positions"]):
            label = drift_cfg.get("label", "Drift") if i == 0 else None
            ax.axvline(pos, color="red", linestyle="--", linewidth=1.0,
                       alpha=0.7, label=label, zorder=0)


def add_drift_annotation_transitions(ax, drift_cfg, transition_labels):
    """
    Add drift markers for transition-based x-axes.
    Maps chunk numbers to transition indices.
    """
    if drift_cfg is None or not transition_labels:
        return

    # Build mapping from chunk_to -> transition index
    chunk_to_idx = {}
    for idx, lbl in enumerate(transition_labels):
        parts = lbl.split("->")
        if len(parts) == 2:
            try:
                chunk_to_idx[int(parts[1].strip())] = idx
            except ValueError:
                pass
    # Also map chunk_from
    chunk_from_idx = {}
    for idx, lbl in enumerate(transition_labels):
        parts = lbl.split("->")
        if len(parts) == 2:
            try:
                chunk_from_idx[int(parts[0].strip())] = idx
            except ValueError:
                pass

    dtype = drift_cfg.get("type")
    if dtype == "shaded_region":
        start_chunk = drift_cfg["start"]
        end_chunk = drift_cfg["end"]
        # Find nearest transition indices
        start_idx = chunk_from_idx.get(start_chunk)
        end_idx = chunk_from_idx.get(end_chunk)
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(transition_labels) - 1
        ax.axvspan(start_idx - 0.5, end_idx + 0.5, alpha=0.10, color="red",
                   zorder=0, label=drift_cfg.get("label", "Drift region"))
    elif dtype == "shaded_regions":
        for i, (start_chunk, end_chunk) in enumerate(drift_cfg["regions"]):
            start_idx = chunk_from_idx.get(start_chunk)
            end_idx = chunk_from_idx.get(end_chunk)
            if start_idx is None:
                start_idx = 0
            if end_idx is None:
                end_idx = len(transition_labels) - 1
            label = drift_cfg.get("label", "Drift region") if i == 0 else None
            ax.axvspan(start_idx - 0.5, end_idx + 0.5, alpha=0.10,
                       color="red", zorder=0, label=label)
    elif dtype == "vertical_lines":
        for i, pos in enumerate(drift_cfg["positions"]):
            idx = chunk_from_idx.get(pos)
            if idx is None:
                # Try chunk_to
                idx = chunk_to_idx.get(pos)
            if idx is not None:
                label = drift_cfg.get("label", "Drift") if i == 0 else None
                ax.axvline(idx, color="red", linestyle="--", linewidth=1.0,
                           alpha=0.7, label=label, zorder=0)


# ---------------------------------------------------------------------------
# Subplot (a): Rule Evolution Counts
# ---------------------------------------------------------------------------
def plot_rule_evolution_counts(ax, evol_df, drift_cfg):
    """
    Stacked area chart: unchanged, modified, new, deleted counts over
    chunk transitions.
    """
    if evol_df is None:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title("(a) Rule Evolution Counts", fontweight="bold")
        return

    evol_df = evol_df.sort_values("chunk_from").reset_index(drop=True)
    transitions = [
        f"{int(r.chunk_from)}->{int(r.chunk_to)}"
        for _, r in evol_df.iterrows()
    ]
    x = np.arange(len(transitions))

    unchanged = evol_df["unchanged"].values.astype(float)
    modified = evol_df["modified"].values.astype(float)
    new_rules = evol_df["new"].values.astype(float)
    deleted = evol_df["deleted"].values.astype(float)

    ax.fill_between(x, 0, unchanged,
                    color=COLOR_UNCHANGED, alpha=0.7, label="Unchanged")
    ax.fill_between(x, unchanged, unchanged + modified,
                    color=COLOR_MODIFIED, alpha=0.7, label="Modified")
    ax.fill_between(x, unchanged + modified, unchanged + modified + new_rules,
                    color=COLOR_NEW, alpha=0.7, label="New")
    ax.fill_between(
        x, unchanged + modified + new_rules,
        unchanged + modified + new_rules + deleted,
        color=COLOR_DELETED, alpha=0.7, label="Deleted",
    )

    # Drift markers
    add_drift_annotation_transitions(ax, drift_cfg, transitions)

    ax.set_xlabel("Chunk Transition")
    ax.set_ylabel("Rule Count")
    ax.set_title("(a) Rule Evolution Counts", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(transitions, rotation=45, ha="right", fontsize=6)
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax.set_xlim(-0.5, len(transitions) - 0.5)


# ---------------------------------------------------------------------------
# Subplot (b): Rule Complexity Over Time
# ---------------------------------------------------------------------------
def plot_rule_complexity(ax, ast_df, drift_cfg):
    """
    Line chart of total_rules, avg_conditions_per_rule (dual y-axis),
    and total_and_ops/10 (scaled).
    """
    if ast_df is None:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title("(b) Rule Complexity Over Time", fontweight="bold")
        return

    ast_df = ast_df.sort_values("chunk").reset_index(drop=True)
    chunks = ast_df["chunk"].values.astype(int)

    total_rules = ast_df["total_rules"].values.astype(float)
    avg_conds = ast_df["avg_conditions_per_rule"].values.astype(float)
    and_ops_scaled = ast_df["total_and_ops"].values.astype(float) / 10.0

    # Left y-axis: total rules and AND ops / 10
    ln1 = ax.plot(chunks, total_rules, color=EGIS_BLUE, marker="o",
                  markersize=3, linewidth=1.5, label="Total Rules")
    ln2 = ax.plot(chunks, and_ops_scaled, color="#756bb1", marker="s",
                  markersize=2.5, linewidth=1.0, linestyle="--",
                  label="AND Ops / 10")
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Count", color=EGIS_BLUE)
    ax.tick_params(axis="y", labelcolor=EGIS_BLUE)

    # Right y-axis: avg conditions per rule
    ax2 = ax.twinx()
    ln3 = ax2.plot(chunks, avg_conds, color="#d95f02", marker="^",
                   markersize=2.5, linewidth=1.0, linestyle="-.",
                   label="Avg Conds/Rule")
    ax2.set_ylabel("Avg Conditions/Rule", color="#d95f02")
    ax2.tick_params(axis="y", labelcolor="#d95f02")

    # Drift markers
    add_drift_annotation(ax, drift_cfg, chunks)

    # Combined legend
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="upper left", fontsize=7)

    ax.set_title("(b) Rule Complexity Over Time", fontweight="bold")


# ---------------------------------------------------------------------------
# Subplot (c): Transition Metrics Over Time
# ---------------------------------------------------------------------------
def plot_transition_metrics(ax, trans_df, drift_cfg):
    """
    Line chart of TCS, RIR, AMS over chunk transitions.
    """
    if trans_df is None:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title("(c) Transition Metrics Over Time", fontweight="bold")
        return

    trans_df = trans_df.sort_values("chunk_from").reset_index(drop=True)

    # Use the transition column if available, otherwise build it
    if "transition" in trans_df.columns:
        transitions = trans_df["transition"].tolist()
    else:
        transitions = [
            f"{int(r.chunk_from)}->{int(r.chunk_to)}"
            for _, r in trans_df.iterrows()
        ]

    x = np.arange(len(transitions))
    tcs = trans_df["TCS"].values.astype(float)
    rir = trans_df["RIR"].values.astype(float)
    ams = trans_df["AMS"].values.astype(float)

    ax.plot(x, tcs, color=COLOR_TCS, marker="o", markersize=3,
            linewidth=1.5, label="TCS")
    ax.plot(x, rir, color=COLOR_RIR, marker="s", markersize=3,
            linewidth=1.5, label="RIR")
    ax.plot(x, ams, color=COLOR_AMS, marker="^", markersize=3,
            linewidth=1.5, label="AMS")

    # Drift markers
    add_drift_annotation_transitions(ax, drift_cfg, transitions)

    ax.set_xlabel("Chunk Transition")
    ax.set_ylabel("Metric Value")
    ax.set_title("(c) Transition Metrics Over Time", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(transitions, rotation=45, ha="right", fontsize=6)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(-0.5, len(transitions) - 0.5)


# ---------------------------------------------------------------------------
# Subplot (d): Rule Turnover Proportions (normalized stacked bar)
# ---------------------------------------------------------------------------
def plot_turnover_proportions(ax, evol_df, drift_cfg):
    """
    Stacked bar chart of unchanged, modified, new, deleted as fractions
    of total rules per transition.
    """
    if evol_df is None:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title("(d) Rule Turnover Proportions", fontweight="bold")
        return

    evol_df = evol_df.sort_values("chunk_from").reset_index(drop=True)
    transitions = [
        f"{int(r.chunk_from)}->{int(r.chunk_to)}"
        for _, r in evol_df.iterrows()
    ]
    x = np.arange(len(transitions))

    unchanged = evol_df["unchanged"].values.astype(float)
    modified = evol_df["modified"].values.astype(float)
    new_rules = evol_df["new"].values.astype(float)
    deleted = evol_df["deleted"].values.astype(float)

    # Total per transition (sum of all categories)
    totals = unchanged + modified + new_rules + deleted
    # Avoid division by zero
    totals = np.where(totals == 0, 1.0, totals)

    frac_unchanged = unchanged / totals
    frac_modified = modified / totals
    frac_new = new_rules / totals
    frac_deleted = deleted / totals

    bar_width = 0.7
    ax.bar(x, frac_unchanged, bar_width,
           color=COLOR_UNCHANGED, label="Unchanged")
    ax.bar(x, frac_modified, bar_width, bottom=frac_unchanged,
           color=COLOR_MODIFIED, label="Modified")
    ax.bar(x, frac_new, bar_width, bottom=frac_unchanged + frac_modified,
           color=COLOR_NEW, label="New")
    ax.bar(x, frac_deleted, bar_width,
           bottom=frac_unchanged + frac_modified + frac_new,
           color=COLOR_DELETED, label="Deleted")

    # Drift markers
    add_drift_annotation_transitions(ax, drift_cfg, transitions)

    ax.set_xlabel("Chunk Transition")
    ax.set_ylabel("Proportion")
    ax.set_title("(d) Rule Turnover Proportions", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(transitions, rotation=45, ha="right", fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax.set_xlim(-0.5 - bar_width / 2, len(transitions) - 0.5 + bar_width / 2)


# ---------------------------------------------------------------------------
# Main: generate one composite figure per dataset
# ---------------------------------------------------------------------------
def generate_case_study_figure(dataset_name, dataset_cfg, df_ast, df_evol,
                               df_trans):
    """Create a 2x2 composite figure for one dataset and save as PDF."""

    print(f"\n--- Generating figure for: {dataset_name} ---")

    # Filter data
    ast_data = filter_dataset(df_ast, dataset_name)
    evol_data = filter_dataset(df_evol, dataset_name)
    trans_data = filter_dataset(df_trans, dataset_name)

    drift_cfg = dataset_cfg.get("drift_annotation")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(dataset_cfg["title"], fontsize=11, fontweight="bold", y=0.98)

    # (a) Rule Evolution Counts - top left
    plot_rule_evolution_counts(axes[0, 0], evol_data, drift_cfg)

    # (b) Rule Complexity Over Time - top right
    plot_rule_complexity(axes[0, 1], ast_data, drift_cfg)

    # (c) Transition Metrics Over Time - bottom left
    plot_transition_metrics(axes[1, 0], trans_data, drift_cfg)

    # (d) Rule Turnover Proportions - bottom right
    plot_turnover_proportions(axes[1, 1], evol_data, drift_cfg)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    os.makedirs(FIG_DIR, exist_ok=True)
    output_path = os.path.join(FIG_DIR, dataset_cfg["output"])
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    print("=" * 60)
    print("EGIS Case Study Figure Generator")
    print("=" * 60)

    # Load CSVs
    df_ast = load_csv(AST_CSV, "ast_chunk_quantitatives.csv")
    df_evol = load_csv(EVOL_CSV, "evolution_analysis_summary.csv")
    df_trans = load_csv(TRANS_CSV, "egis_transition_metrics.csv")

    if df_ast is None and df_evol is None and df_trans is None:
        print("ERROR: No data files found. Cannot generate figures.")
        sys.exit(1)

    # Summary of loaded data
    for name, df in [("AST", df_ast), ("Evolution", df_evol),
                     ("Transition", df_trans)]:
        if df is not None:
            print(f"  Loaded {name}: {len(df)} rows")
        else:
            print(f"  WARNING: {name} data not available")

    # Generate figures
    for dataset_name, dataset_cfg in DATASETS.items():
        generate_case_study_figure(
            dataset_name, dataset_cfg, df_ast, df_evol, df_trans
        )

    print("\n" + "=" * 60)
    print("Done. All figures saved to:", FIG_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
