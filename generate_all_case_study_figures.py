#!/usr/bin/env python3
"""
Generate ALL case study figures for the EGIS project.

Produces 144 composite 2x2 figures (48 datasets x 3 NP configs):
  - EXP-500-NP   (48 PDFs)
  - EXP-1000-NP  (48 PDFs)
  - EXP-2000-NP  (48 PDFs)

Each figure contains four subplots:
  (a) Rule Evolution Counts       (top-left)
  (b) Rule Complexity Over Time   (top-right)
  (c) Transition Metrics Over Time (bottom-left)
  (d) Rule Turnover Proportions   (bottom-right)

Usage:
    python generate_all_case_study_figures.py
    python generate_all_case_study_figures.py --configs EXP-500-NP EXP-1000-NP
    python generate_all_case_study_figures.py --datasets SEA_Abrupt_Simple STAGGER_Abrupt_Chain
    python generate_all_case_study_figures.py --output-dir paper/figures/case_studies --dpi 200
    python generate_all_case_study_figures.py --dry-run
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Import subplot functions from generate_paper_case_studies.py
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from generate_paper_case_studies import (
    plot_rule_evolution_counts,
    plot_rule_complexity,
    plot_transition_metrics,
    plot_turnover_proportions,
    load_csv,
    filter_dataset,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "paper_data")
AST_CSV = os.path.join(DATA_DIR, "ast_chunk_quantitatives.csv")
EVOL_CSV = os.path.join(DATA_DIR, "evolution_analysis_summary.csv")
TRANS_CSV = os.path.join(DATA_DIR, "egis_transition_metrics.csv")
STREAM_DEFS_JSON = os.path.join(DATA_DIR, "stream_definitions.json")

# ---------------------------------------------------------------------------
# Config chunk sizes
# ---------------------------------------------------------------------------
CONFIG_CHUNK_SIZES = {
    "EXP-500-NP": 500,
    "EXP-1000-NP": 1000,
    "EXP-2000-NP": 2000,
}

TOTAL_INSTANCES = 12000

# ---------------------------------------------------------------------------
# Style configuration (IEEE-compatible) -- mirrors generate_paper_case_studies
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


# ---------------------------------------------------------------------------
# Drift annotation computation from stream_definitions.json
# ---------------------------------------------------------------------------
def load_stream_definitions():
    """Load stream_definitions.json and return the dict."""
    if not os.path.isfile(STREAM_DEFS_JSON):
        warnings.warn(
            f"WARNING: stream_definitions.json not found at {STREAM_DEFS_JSON}"
        )
        return {}
    with open(STREAM_DEFS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_drift_cfg(dataset_name, stream_defs, chunk_size):
    """
    Compute drift annotation config for a dataset and chunk size.

    Returns a dict with 'type' and positions/regions, or None for
    stationary/real datasets.
    """
    if dataset_name not in stream_defs:
        return None

    defn = stream_defs[dataset_name]

    # Must have concept_sequence to have drift points
    concept_sequence = defn.get("concept_sequence")
    if concept_sequence is None:
        return None

    gradual_width_chunks = defn.get("gradual_drift_width_chunks", 0)

    # Calculate total duration in abstract chunk units
    total_duration = sum(c["duration_chunks"] for c in concept_sequence)
    if total_duration == 0:
        return None

    # Number of actual chunks for this config
    n_chunks = TOTAL_INSTANCES // chunk_size

    # Calculate boundary positions (between concept i and concept i+1)
    cumulative = 0
    boundaries = []
    for i in range(len(concept_sequence) - 1):
        cumulative += concept_sequence[i]["duration_chunks"]
        boundary_chunk = cumulative / total_duration * n_chunks
        boundaries.append(boundary_chunk)

    if len(boundaries) == 0:
        return None

    drift_type = defn.get("drift_type", "abrupt")

    if gradual_width_chunks == 0:
        # Abrupt drift: vertical lines
        positions = [round(b) for b in boundaries]
        return {
            "type": "vertical_lines",
            "positions": positions,
            "label": "Drift point",
        }
    else:
        # Gradual drift: shaded regions
        width = gradual_width_chunks / total_duration * n_chunks
        regions = []
        for b in boundaries:
            start = max(0, b - width / 2)
            end = min(n_chunks, b + width / 2)
            regions.append((start, end))
        return {
            "type": "shaded_regions",
            "regions": regions,
            "label": "Drift region",
        }


def build_drift_type_map(df_ast):
    """Build a dataset -> drift_type mapping from the AST CSV data."""
    if df_ast is None:
        return {}
    mapping = {}
    for _, row in df_ast[["dataset", "drift_type"]].drop_duplicates().iterrows():
        mapping[row["dataset"]] = row["drift_type"]
    return mapping


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
def discover_datasets(df_ast):
    """Auto-discover all unique dataset names from the AST CSV."""
    if df_ast is None:
        return []
    datasets = sorted(df_ast["dataset"].unique().tolist())
    return datasets


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def generate_figure(dataset_name, config_label, drift_cfg, df_ast, df_evol,
                    df_trans, output_path, dpi):
    """
    Create a 2x2 composite figure for one (dataset, config) pair and save
    as PDF.
    """
    # Filter data for this dataset + config
    ast_data = filter_dataset(df_ast, dataset_name, config_label=config_label)
    evol_data = filter_dataset(df_evol, dataset_name, config_label=config_label)
    trans_data = filter_dataset(df_trans, dataset_name, config_label=config_label)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(
        f"{dataset_name} ({config_label})",
        fontsize=11, fontweight="bold", y=0.98,
    )

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate ALL case study figures (48 datasets x 3 NP configs)."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["EXP-500-NP", "EXP-1000-NP", "EXP-2000-NP"],
        help=(
            "Config labels to generate figures for. "
            "Default: EXP-500-NP EXP-1000-NP EXP-2000-NP"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Specific datasets to generate figures for. "
            "If not specified, auto-discovers all datasets from the AST CSV."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="paper/figures/case_studies",
        help="Output directory. Default: paper/figures/case_studies",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for saved figures. Default: 200",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without creating files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 60)
    print("EGIS - Batch Case Study Figure Generator")
    print("=" * 60)

    # Validate configs
    for cfg in args.configs:
        if cfg not in CONFIG_CHUNK_SIZES:
            print(
                f"ERROR: Unknown config '{cfg}'. "
                f"Valid configs: {list(CONFIG_CHUNK_SIZES.keys())}"
            )
            sys.exit(1)

    # Load data
    print("\nLoading data files...")
    df_ast = load_csv(AST_CSV, "ast_chunk_quantitatives.csv")
    df_evol = load_csv(EVOL_CSV, "evolution_analysis_summary.csv")
    df_trans = load_csv(TRANS_CSV, "egis_transition_metrics.csv")

    if df_ast is None and df_evol is None and df_trans is None:
        print("ERROR: No data files found. Cannot generate figures.")
        sys.exit(1)

    for name, df in [("AST", df_ast), ("Evolution", df_evol),
                     ("Transition", df_trans)]:
        if df is not None:
            print(f"  Loaded {name}: {len(df)} rows")
        else:
            print(f"  WARNING: {name} data not available")

    # Load stream definitions for drift annotations
    stream_defs = load_stream_definitions()
    if stream_defs:
        print(f"  Loaded stream definitions: {len(stream_defs)} entries")
    else:
        print("  WARNING: stream definitions not available")

    # Build drift type map from CSV data (authoritative source)
    drift_type_map = build_drift_type_map(df_ast)

    # Discover datasets
    if args.datasets:
        datasets = args.datasets
        print(f"\nUsing user-specified datasets: {len(datasets)}")
    else:
        datasets = discover_datasets(df_ast)
        if not datasets:
            # Fallback: try evolution or transition CSVs
            if df_evol is not None:
                datasets = sorted(df_evol["dataset"].unique().tolist())
            elif df_trans is not None:
                datasets = sorted(df_trans["dataset"].unique().tolist())
        print(f"\nAuto-discovered datasets: {len(datasets)}")

    if not datasets:
        print("ERROR: No datasets found. Check your data files.")
        sys.exit(1)

    configs = args.configs
    output_dir = os.path.join(BASE_DIR, args.output_dir)
    total_figures = len(datasets) * len(configs)

    print(f"\nGeneration plan:")
    print(f"  Configs:  {configs}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Total:    {total_figures} figures")
    print(f"  Output:   {output_dir}")
    print(f"  DPI:      {args.dpi}")

    if args.dry_run:
        print("\n--- DRY RUN: listing figures that would be generated ---\n")
        for idx, config_label in enumerate(configs):
            for ds_idx, dataset_name in enumerate(datasets):
                fig_num = idx * len(datasets) + ds_idx + 1
                drift_type = drift_type_map.get(dataset_name, "unknown")
                rel_path = os.path.join(config_label, f"{dataset_name}.pdf")
                print(
                    f"[{fig_num}/{total_figures}] "
                    f"{config_label} / {dataset_name} "
                    f"(drift={drift_type}) -> {rel_path}"
                )
        print(f"\nDry run complete. {total_figures} figures would be generated.")
        return

    # Generate all figures
    print(f"\nGenerating {total_figures} figures...\n")
    manifest_figures = []
    success_count = 0
    error_count = 0

    fig_counter = 0
    for config_label in configs:
        chunk_size = CONFIG_CHUNK_SIZES[config_label]
        config_output_dir = os.path.join(output_dir, config_label)
        os.makedirs(config_output_dir, exist_ok=True)

        for dataset_name in datasets:
            fig_counter += 1
            drift_type = drift_type_map.get(dataset_name, "unknown")
            drift_cfg = compute_drift_cfg(dataset_name, stream_defs, chunk_size)
            rel_path = os.path.join(config_label, f"{dataset_name}.pdf")
            output_path = os.path.join(output_dir, rel_path)

            try:
                generate_figure(
                    dataset_name=dataset_name,
                    config_label=config_label,
                    drift_cfg=drift_cfg,
                    df_ast=df_ast,
                    df_evol=df_evol,
                    df_trans=df_trans,
                    output_path=output_path,
                    dpi=args.dpi,
                )
                status = "ok"
                success_count += 1
                print(
                    f"[{fig_counter}/{total_figures}] "
                    f"{config_label} / {dataset_name} ... OK"
                )
            except Exception as exc:
                status = f"error: {exc}"
                error_count += 1
                print(
                    f"[{fig_counter}/{total_figures}] "
                    f"{config_label} / {dataset_name} ... ERROR: {exc}"
                )

            manifest_figures.append({
                "config": config_label,
                "dataset": dataset_name,
                "drift_type": drift_type,
                "file": rel_path.replace("\\", "/"),
                "status": status,
            })

    # Write manifest
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_figures": total_figures,
        "configs": configs,
        "datasets": datasets,
        "figures": manifest_figures,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 60)
    print("Generation complete.")
    print(f"  Success: {success_count}/{total_figures}")
    if error_count > 0:
        print(f"  Errors:  {error_count}/{total_figures}")
    print(f"  Output:  {output_dir}")
    print(f"  Manifest: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
