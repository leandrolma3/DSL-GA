#!/usr/bin/env python3
"""
Prepare Auxiliary Data Document for IEEE TKDE Paper.

Reads all CSVs from paper_data/ and generates a consolidated
AUXILIARY_DATA_DOCUMENT.md with all data needed for the paper's
16 tables and 13 figures.

Author: Automated Analysis
Date: 2026-02-23
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

PAPER_DATA_DIR = Path("paper_data")
OUTPUT_FILE = PAPER_DATA_DIR / "AUXILIARY_DATA_DOCUMENT.md"
EXPERIMENTS_BASE = Path("experiments_unified")

# Multiclass datasets (ACDWM/CDCMS have no results for these)
MULTICLASS_DATASETS = [
    'LED_Abrupt_Simple', 'LED_Gradual_Simple', 'LED_Stationary',
    'RBF_Stationary',
    'WAVEFORM_Abrupt_Simple', 'WAVEFORM_Gradual_Simple', 'WAVEFORM_Stationary'
]

# Binary-only models (cannot handle multiclass)
BINARY_ONLY_MODELS = ['ACDWM', 'CDCMS']

# All models (8 models: EGIS + 7 baselines)
ALL_MODELS = ['EGIS', 'ARF', 'SRP', 'HAT', 'ROSE', 'ACDWM', 'ERulesD2S', 'CDCMS']

# Baseline models (all except EGIS)
BASELINE_MODELS = ['ARF', 'SRP', 'HAT', 'ROSE', 'ACDWM', 'ERulesD2S', 'CDCMS']

# EGIS variants by chunk size (for comparative tables)
EGIS_VARIANTS_BY_CHUNK = {
    500:  ['EXP-500-NP', 'EXP-500-P', 'EXP-500-P03'],
    1000: ['EXP-1000-NP', 'EXP-1000-P'],
    2000: ['EXP-2000-NP', 'EXP-2000-P'],
}

# Short display names for EGIS variants
EGIS_VARIANT_DISPLAY = {
    'EXP-500-NP': 'EGIS (NP)',
    'EXP-500-P': 'EGIS (P)',
    'EXP-500-P03': 'EGIS (P03)',
    'EXP-1000-NP': 'EGIS (NP)',
    'EXP-1000-P': 'EGIS (P)',
    'EXP-2000-NP': 'EGIS (NP)',
    'EXP-2000-P': 'EGIS (P)',
}

# Map chunk_size -> NP config label (for baseline data)
CHUNK_TO_NP_CONFIG = {
    500: 'EXP-500-NP',
    1000: 'EXP-1000-NP',
    2000: 'EXP-2000-NP',
}

# Configs with baselines (no-penalty configs)
BASELINE_CONFIGS = ['EXP-500-NP', 'EXP-1000-NP', 'EXP-2000-NP']

# All EGIS configs
EGIS_CONFIGS = ['EXP-500-NP', 'EXP-500-P', 'EXP-500-P03',
                'EXP-1000-NP', 'EXP-1000-P',
                'EXP-2000-NP', 'EXP-2000-P']

# Config metadata
CONFIG_META = {
    'EXP-500-NP':  {'chunk_size': 500,  'penalty': 0.0, 'num_chunks': 24},
    'EXP-500-P':   {'chunk_size': 500,  'penalty': 0.1, 'num_chunks': 24},
    'EXP-500-P03': {'chunk_size': 500,  'penalty': 0.3, 'num_chunks': 24},
    'EXP-1000-NP': {'chunk_size': 1000, 'penalty': 0.0, 'num_chunks': 12},
    'EXP-1000-P':  {'chunk_size': 1000, 'penalty': 0.1, 'num_chunks': 12},
    'EXP-2000-NP': {'chunk_size': 2000, 'penalty': 0.0, 'num_chunks': 6},
    'EXP-2000-P':  {'chunk_size': 2000, 'penalty': 0.1, 'num_chunks': 6},
}


# Dataset -> drift type mapping (from batch_evolution_graphs_data.py)
DATASET_DRIFT_MAP = {
    'SEA_Abrupt_Simple': 'abrupt', 'SEA_Abrupt_Chain': 'abrupt', 'SEA_Abrupt_Recurring': 'abrupt',
    'AGRAWAL_Abrupt_Simple_Mild': 'abrupt', 'AGRAWAL_Abrupt_Simple_Severe': 'abrupt',
    'AGRAWAL_Abrupt_Chain_Long': 'abrupt', 'RBF_Abrupt_Severe': 'abrupt', 'RBF_Abrupt_Blip': 'abrupt',
    'STAGGER_Abrupt_Chain': 'abrupt', 'STAGGER_Abrupt_Recurring': 'abrupt',
    'HYPERPLANE_Abrupt_Simple': 'abrupt', 'RANDOMTREE_Abrupt_Simple': 'abrupt',
    'SINE_Abrupt_Simple': 'abrupt', 'LED_Abrupt_Simple': 'abrupt', 'WAVEFORM_Abrupt_Simple': 'abrupt',
    'RANDOMTREE_Abrupt_Recurring': 'abrupt',
    'SEA_Gradual_Simple_Fast': 'gradual', 'SEA_Gradual_Simple_Slow': 'gradual',
    'SEA_Gradual_Recurring': 'gradual', 'STAGGER_Gradual_Chain': 'gradual',
    'RBF_Gradual_Moderate': 'gradual', 'RBF_Gradual_Severe': 'gradual',
    'HYPERPLANE_Gradual_Simple': 'gradual', 'RANDOMTREE_Gradual_Simple': 'gradual',
    'LED_Gradual_Simple': 'gradual', 'SINE_Gradual_Recurring': 'gradual',
    'WAVEFORM_Gradual_Simple': 'gradual',
    'SEA_Abrupt_Chain_Noise': 'noisy', 'STAGGER_Abrupt_Chain_Noise': 'noisy',
    'AGRAWAL_Abrupt_Simple_Severe_Noise': 'noisy', 'SINE_Abrupt_Recurring_Noise': 'noisy',
    'RBF_Abrupt_Blip_Noise': 'noisy', 'RBF_Gradual_Severe_Noise': 'noisy',
    'HYPERPLANE_Gradual_Noise': 'noisy', 'RANDOMTREE_Gradual_Noise': 'noisy',
    'Electricity': 'real', 'AssetNegotiation_F2': 'real',
    'AssetNegotiation_F3': 'real', 'AssetNegotiation_F4': 'real',
    'SEA_Stationary': 'stationary', 'AGRAWAL_Stationary': 'stationary',
    'RBF_Stationary': 'stationary', 'LED_Stationary': 'stationary',
    'HYPERPLANE_Stationary': 'stationary', 'RANDOMTREE_Stationary': 'stationary',
    'STAGGER_Stationary': 'stationary', 'WAVEFORM_Stationary': 'stationary',
    'SINE_Stationary': 'stationary',
}

DRIFT_TYPE_ORDER = ['abrupt', 'gradual', 'noisy', 'stationary', 'real']


def fmt(val, decimals=4):
    """Format numeric value."""
    if pd.isna(val):
        return "N/A"
    return f"{val:.{decimals}f}"


def fmt_pm(mean, std, decimals=4):
    """Format mean +/- std."""
    if pd.isna(mean):
        return "N/A"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


# =============================================================================
# SECTION GENERATORS
# =============================================================================

def section_1_dimensions(df, out):
    """Section 1: Experiment Dimensions."""
    out.append("## Section 1 -- Experiment Dimensions\n")

    n_datasets = df['dataset'].nunique()
    all_datasets = sorted(df['dataset'].unique())
    binary_ds = [d for d in all_datasets if d not in MULTICLASS_DATASETS]
    multiclass_ds = [d for d in all_datasets if d in MULTICLASS_DATASETS]

    out.append(f"- **Total datasets**: {n_datasets}")
    out.append(f"- **Binary datasets**: {len(binary_ds)}")
    out.append(f"- **Multiclass datasets**: {len(multiclass_ds)}")
    out.append(f"- **Multiclass list**: {', '.join(multiclass_ds)}")

    n_configs = df['config_label'].nunique()
    n_models = df['model'].nunique()
    out.append(f"- **EGIS configs**: {len(EGIS_CONFIGS)}")
    out.append(f"- **Baseline configs (with comparative models)**: {len(BASELINE_CONFIGS)}")
    out.append(f"- **Total models**: {n_models}")
    out.append(f"- **Total config_labels in CSV**: {n_configs}")
    out.append(f"- **Total rows in consolidated_results.csv**: {len(df)}")

    # Runs count
    egis_rows = len(df[df['model'] == 'EGIS'])
    baseline_rows = len(df[df['model'] != 'EGIS'])
    out.append(f"- **EGIS result rows**: {egis_rows}")
    out.append(f"- **Baseline result rows**: {baseline_rows}")

    # Drift type counts
    drift_counts = df.drop_duplicates('dataset').groupby('drift_type')['dataset'].count()
    out.append("\n### Datasets per Drift Type\n")
    out.append("| Drift Type | Count |")
    out.append("|---|---|")
    for dt in ['abrupt', 'gradual', 'noisy', 'stationary', 'real']:
        cnt = drift_counts.get(dt, 0)
        out.append(f"| {dt} | {cnt} |")

    # Binary drift type counts
    binary_drift = df[~df['dataset'].isin(MULTICLASS_DATASETS)].drop_duplicates('dataset')
    binary_drift_counts = binary_drift.groupby('drift_type')['dataset'].count()
    out.append("\n### Binary Datasets per Drift Type\n")
    out.append("| Drift Type | Count |")
    out.append("|---|---|")
    for dt in ['abrupt', 'gradual', 'noisy', 'stationary', 'real']:
        cnt = binary_drift_counts.get(dt, 0)
        out.append(f"| {dt} | {cnt} |")

    # tab:dataset_sizes data
    out.append("\n### Data for tab:dataset_sizes (Dataset Dimensions and Chunk Structure)\n")
    out.append("| Config | Chunk Size | Penalty | Num Chunks | Evals per Run | Total Datasets |")
    out.append("|---|---|---|---|---|---|")
    for cfg_label in EGIS_CONFIGS:
        meta = CONFIG_META[cfg_label]
        evals = meta['num_chunks']  # each chunk is an evaluation
        out.append(f"| {cfg_label} | {meta['chunk_size']} | {meta['penalty']} | {meta['num_chunks']} | {evals} | 48 |")

    out.append("")


def section_2_configs(df, out):
    """Section 2: Experimental Configurations."""
    out.append("## Section 2 -- Experimental Configurations (tab:exp_config)\n")

    out.append("| Label | Chunk Size | Penalty (gamma) | Num Chunks | Instances/Chunk | Total Instances | Datasets |")
    out.append("|---|---|---|---|---|---|---|")
    for cfg_label in EGIS_CONFIGS:
        meta = CONFIG_META[cfg_label]
        instances_per_chunk = meta['chunk_size']
        total_instances = instances_per_chunk * meta['num_chunks']
        out.append(f"| {cfg_label} | {meta['chunk_size']} | {meta['penalty']} | "
                   f"{meta['num_chunks']} | {instances_per_chunk} | {total_instances} | 48 |")

    # EGIS hyperparameters (tab:egis_params)
    out.append("\n### EGIS Hyperparameters (tab:egis_params)\n")
    out.append("| Parameter | Value |")
    out.append("|---|---|")
    params = [
        ("Population size", "120"),
        ("Max generations", "200"),
        ("Elitism rate", "0.1"),
        ("Intelligent mutation rate", "0.8"),
        ("Tournament size (initial/final)", "2 / 5"),
        ("Max rules per class", "15"),
        ("Initial max depth", "10"),
        ("Stagnation threshold", "10"),
        ("Early stopping patience", "20"),
        ("DT seeding on init", "True (ratio=0.8)"),
        ("DT seeding depths", "4, 7, 10, 13"),
        ("Balanced crossover", "True"),
        ("Penalty gamma values", "0.0, 0.1, 0.3"),
        ("Memory max size", "20"),
        ("Active memory pruning", "True"),
    ]
    for name, val in params:
        out.append(f"| {name} | {val} |")

    out.append("")


def _emit_per_dataset_table(df, chunk_size, egis_variants, out):
    """Emit per-dataset comparison table grouped by drift type with W/L/D summary.

    Columns: Dataset | EGIS variants... | baselines...
    Rows grouped by drift type, with summary rows at the end.
    """
    np_config = CHUNK_TO_NP_CONFIG[chunk_size]

    # Build column definitions: (display_label, config_label, model_name)
    columns = []
    for v_cfg in egis_variants:
        columns.append((EGIS_VARIANT_DISPLAY[v_cfg], v_cfg, 'EGIS'))
    for model in BASELINE_MODELS:
        columns.append((model, np_config, model))

    col_labels = [c[0] for c in columns]

    # Gather all datasets present in df
    all_datasets = sorted(df['dataset'].unique())

    # Categorize datasets by drift type
    drift_groups = defaultdict(list)
    for ds in all_datasets:
        if ds in MULTICLASS_DATASETS:
            drift_groups['multiclass'].append(ds)
        else:
            dt = DATASET_DRIFT_MAP.get(ds, 'unknown')
            drift_groups[dt].append(ds)

    # Get values for each dataset x column
    values = {}
    for ds in all_datasets:
        row_vals = []
        for _, cfg, model in columns:
            sub = df[(df['config_label'] == cfg) & (df['model'] == model) & (df['dataset'] == ds)]
            if len(sub) > 0 and not pd.isna(sub['gmean_mean'].values[0]):
                row_vals.append(sub['gmean_mean'].values[0])
            else:
                row_vals.append(np.nan)
        values[ds] = row_vals

    # Find best value per row for bold marking
    def _fmt_row(ds, vals):
        """Format a row, marking best value with ** (markdown bold)."""
        valid_vals = [(i, v) for i, v in enumerate(vals) if not pd.isna(v)]
        best_idx = max(valid_vals, key=lambda x: x[1])[0] if valid_vals else -1
        formatted = []
        for i, v in enumerate(vals):
            _, cfg, model = columns[i]
            if model in BINARY_ONLY_MODELS and ds in MULTICLASS_DATASETS:
                formatted.append("--")
            elif pd.isna(v):
                formatted.append("N/A")
            elif i == best_idx:
                formatted.append(f"**{fmt(v)}**")
            else:
                formatted.append(fmt(v))
        return formatted

    # Emit table header
    out.append(f"| Dataset | {' | '.join(col_labels)} |")
    out.append(f"|---|{'---|' * len(col_labels)}")

    # Emit rows grouped by drift type
    group_order = DRIFT_TYPE_ORDER + ['multiclass']
    for group_name in group_order:
        datasets_in_group = sorted(drift_groups.get(group_name, []))
        if not datasets_in_group:
            continue

        label = group_name.capitalize()
        out.append(f"| **{label} ({len(datasets_in_group)})** |{'|' * len(col_labels)}")

        for ds in datasets_in_group:
            vals = values.get(ds, [np.nan] * len(columns))
            formatted = _fmt_row(ds, vals)
            out.append(f"| {ds} | {' | '.join(formatted)} |")

    # --- Summary: Binary only ---
    binary_datasets = [ds for ds in all_datasets if ds not in MULTICLASS_DATASETS]
    n_binary = len(binary_datasets)

    out.append(f"|---|{'---|' * len(col_labels)}")
    out.append(f"| **Binary Summary (n={n_binary})** |{'|' * len(col_labels)}")

    # Mean per column (binary)
    mean_row = []
    for col_idx in range(len(columns)):
        col_vals = [values[ds][col_idx] for ds in binary_datasets if not pd.isna(values[ds][col_idx])]
        mean_row.append(fmt(np.mean(col_vals)) if col_vals else "N/A")
    out.append(f"| Mean | {' | '.join(mean_row)} |")

    # Std per column (binary)
    std_row = []
    for col_idx in range(len(columns)):
        col_vals = [values[ds][col_idx] for ds in binary_datasets if not pd.isna(values[ds][col_idx])]
        std_row.append(fmt(np.std(col_vals)) if col_vals else "N/A")
    out.append(f"| Std | {' | '.join(std_row)} |")

    # W/L/D: EGIS(NP) vs each other column (binary)
    np_idx = 0  # First column is always EGIS(NP)
    wld_row = ["--"]  # NP vs itself
    for col_idx in range(1, len(columns)):
        w, l, d = 0, 0, 0
        for ds in binary_datasets:
            np_val = values[ds][np_idx]
            other_val = values[ds][col_idx]
            if pd.isna(np_val) or pd.isna(other_val):
                continue
            if np_val > other_val + 0.001:
                w += 1
            elif np_val < other_val - 0.001:
                l += 1
            else:
                d += 1
        wld_row.append(f"{w}/{l}/{d}")
    out.append(f"| EGIS(NP) W/L/D | {' | '.join(wld_row)} |")

    # Avg Rank per column (binary, scipy-style ranking)
    rank_sums = defaultdict(list)
    for ds in binary_datasets:
        ds_vals = {}
        for col_idx, label in enumerate(col_labels):
            v = values[ds][col_idx]
            if not pd.isna(v):
                ds_vals[col_idx] = v
        if len(ds_vals) >= 2:
            # Rank: highest value = rank 1 (ascending=False)
            sorted_items = sorted(ds_vals.items(), key=lambda x: -x[1])
            current_rank = 1
            i = 0
            while i < len(sorted_items):
                # Handle ties (average rank)
                j = i
                while j < len(sorted_items) and abs(sorted_items[j][1] - sorted_items[i][1]) < 1e-9:
                    j += 1
                avg_rank = (current_rank + current_rank + j - i - 1) / 2.0
                for k in range(i, j):
                    rank_sums[sorted_items[k][0]].append(avg_rank)
                current_rank += j - i
                i = j

    rank_row = []
    for col_idx in range(len(columns)):
        ranks = rank_sums.get(col_idx, [])
        rank_row.append(fmt(np.mean(ranks), 2) if ranks else "N/A")
    out.append(f"| Avg Rank | {' | '.join(rank_row)} |")

    # --- Summary: All datasets ---
    out.append(f"| **All Summary (n={len(all_datasets)})** |{'|' * len(col_labels)}")

    mean_all_row = []
    for col_idx in range(len(columns)):
        col_vals = [values[ds][col_idx] for ds in all_datasets if not pd.isna(values[ds][col_idx])]
        mean_all_row.append(fmt(np.mean(col_vals)) if col_vals else "N/A")
    out.append(f"| Mean | {' | '.join(mean_all_row)} |")

    std_all_row = []
    for col_idx in range(len(columns)):
        col_vals = [values[ds][col_idx] for ds in all_datasets if not pd.isna(values[ds][col_idx])]
        std_all_row.append(fmt(np.std(col_vals)) if col_vals else "N/A")
    out.append(f"| Std | {' | '.join(std_all_row)} |")


def _emit_comparison_table(df, chunk_size, egis_variants, baseline_models, out,
                           is_binary=True, drift_type=None, title_suffix=""):
    """Helper: emit a comparison table with EGIS variants + baselines.

    Args:
        df: DataFrame already filtered to the right config/datasets scope.
        chunk_size: int (500, 1000, 2000)
        egis_variants: list of config_labels for EGIS variants at this chunk_size
        baseline_models: list of baseline model names to include
        is_binary: if True, include ACDWM/CDCMS; if False, exclude them
        drift_type: optional drift type filter
        title_suffix: extra text for the header
    """
    np_config = CHUNK_TO_NP_CONFIG[chunk_size]

    # Optionally filter by drift type
    if drift_type:
        df = df[df['drift_type'] == drift_type]

    out.append("| Model | Mean G-Mean | Std | N |")
    out.append("|---|---|---|---|")

    # EGIS variants
    for variant_cfg in egis_variants:
        display_name = EGIS_VARIANT_DISPLAY.get(variant_cfg, variant_cfg)
        egis_df = df[(df['config_label'] == variant_cfg) & (df['model'] == 'EGIS')]
        if len(egis_df) > 0:
            out.append(f"| {display_name} | {fmt(egis_df['gmean_mean'].mean())} | "
                       f"{fmt(egis_df['gmean_mean'].std())} | {len(egis_df)} |")
        else:
            out.append(f"| {display_name} | N/A | N/A | 0 |")

    # Baselines (from NP config)
    cfg_df = df[df['config_label'] == np_config]
    for model in baseline_models:
        if not is_binary and model in BINARY_ONLY_MODELS:
            continue
        model_df = cfg_df[cfg_df['model'] == model]
        if len(model_df) > 0:
            out.append(f"| {model} | {fmt(model_df['gmean_mean'].mean())} | "
                       f"{fmt(model_df['gmean_mean'].std())} | {len(model_df)} |")
        else:
            out.append(f"| {model} | N/A | N/A | 0 |")


def section_3_performance(df, out):
    """Section 3: G-Mean Performance (tab:summary_all, tab:binary_comparison, tab:drift_performance)."""
    out.append("## Section 3 -- G-Mean Performance\n")

    binary_df = df[~df['dataset'].isin(MULTICLASS_DATASETS)]
    multiclass_df = df[df['dataset'].isin(MULTICLASS_DATASETS)]

    # ---- Block A: Binary (41 datasets) -- EGIS variants + baselines per chunk_size ----
    out.append("### Block A -- Binary Datasets (41) -- EGIS Variants + Baselines per Chunk Size\n")
    out.append("For tab:summary_all (binary block). Each chunk_size shows all EGIS variants + baselines.\n")

    for chunk_size in [500, 1000, 2000]:
        variants = EGIS_VARIANTS_BY_CHUNK[chunk_size]
        out.append(f"\n#### Chunk Size: {chunk_size}\n")
        _emit_comparison_table(binary_df, chunk_size, variants, BASELINE_MODELS, out, is_binary=True)

    # ---- Block A2: Multiclass (7 datasets) -- EGIS variants + eligible baselines ----
    out.append("\n### Block A2 -- Multiclass Datasets (7) -- EGIS Variants + Baselines per Chunk Size\n")
    out.append("ACDWM/CDCMS excluded (binary-only models).\n")

    for chunk_size in [500, 1000, 2000]:
        variants = EGIS_VARIANTS_BY_CHUNK[chunk_size]
        out.append(f"\n#### Chunk Size: {chunk_size}\n")
        _emit_comparison_table(multiclass_df, chunk_size, variants, BASELINE_MODELS, out, is_binary=False)

    # ---- Block B: All (48 datasets) ----
    out.append("\n### Block B -- All Datasets (48) -- EGIS Variants + Baselines\n")
    out.append("ACDWM/CDCMS only have 41 datasets (binary only).\n")

    for chunk_size in [500, 1000, 2000]:
        variants = EGIS_VARIANTS_BY_CHUNK[chunk_size]
        out.append(f"\n#### Chunk Size: {chunk_size}\n")
        _emit_comparison_table(df, chunk_size, variants, BASELINE_MODELS, out, is_binary=True)

    # ---- Block C: By Drift Type (binary only) ----
    out.append("\n### Block C -- Performance by Drift Type (Binary)\n")
    out.append("For tab:drift_performance. Shows EGIS variants + baselines per drift type.\n")

    for chunk_size in [500, 1000, 2000]:
        variants = EGIS_VARIANTS_BY_CHUNK[chunk_size]

        for dt in ['abrupt', 'gradual', 'noisy', 'stationary', 'real']:
            dt_df = binary_df[binary_df['drift_type'] == dt]
            n_datasets = dt_df['dataset'].nunique()
            out.append(f"\n#### Chunk {chunk_size} -- Drift Type: {dt} ({n_datasets} binary datasets)\n")
            _emit_comparison_table(binary_df, chunk_size, variants, BASELINE_MODELS, out,
                                   is_binary=True, drift_type=dt)

    # ---- Block D/E/F: Per-dataset tables with drift-type grouping ----
    for chunk_size, block_label in [(500, 'D'), (1000, 'E'), (2000, 'F')]:
        egis_variants = EGIS_VARIANTS_BY_CHUNK[chunk_size]
        np_config = CHUNK_TO_NP_CONFIG[chunk_size]

        out.append(f"\n### Block {block_label} -- Per-Dataset G-Mean (EXP-{chunk_size})\n")
        out.append(f"Columns: EGIS variants ({', '.join(EGIS_VARIANT_DISPLAY[v] for v in egis_variants)}) + baselines.")
        out.append(f"Grouped by drift type. Best value per row in **bold**.")
        out.append(f"ACDWM/CDCMS shown as '--' for multiclass datasets.\n")

        # Filter df to relevant configs: EGIS variants + NP baselines
        relevant_configs = list(set(egis_variants + [np_config]))
        block_df = df[df['config_label'].isin(relevant_configs)]

        _emit_per_dataset_table(block_df, chunk_size, egis_variants, out)

    out.append("")


def section_4_transitions(df_trans, out):
    """Section 4: Transition Metrics TCS, RIR, AMS (tab:transitions)."""
    out.append("## Section 4 -- Transition Metrics (TCS, RIR, AMS)\n")
    out.append("For tab:transitions. Values from egis_transition_metrics.csv.\n")
    out.append("**CRITICAL**: Paper tab:transitions currently shows TCS ~0.957-1.000.")
    out.append("Correct values from CSV are ~0.22-0.32.\n")

    # Overall summary per config
    out.append("### Overall Summary per Config\n")
    out.append("| Config | TCS mean | TCS std | RIR mean | RIR std | AMS mean | AMS std | N transitions |")
    out.append("|---|---|---|---|---|---|---|---|")

    for cfg in EGIS_CONFIGS:
        cfg_df = df_trans[df_trans['config_label'] == cfg]
        if len(cfg_df) > 0:
            out.append(f"| {cfg} | {fmt(cfg_df['TCS'].mean())} | {fmt(cfg_df['TCS'].std())} | "
                       f"{fmt(cfg_df['RIR'].mean())} | {fmt(cfg_df['RIR'].std())} | "
                       f"{fmt(cfg_df['AMS'].mean())} | {fmt(cfg_df['AMS'].std())} | {len(cfg_df)} |")

    # By drift type per config
    out.append("\n### By Drift Type per Config\n")

    for cfg in EGIS_CONFIGS:
        cfg_df = df_trans[df_trans['config_label'] == cfg]
        if len(cfg_df) == 0:
            continue
        out.append(f"\n#### Config: {cfg}\n")
        out.append("| Drift Type | TCS mean+std | RIR mean+std | AMS mean+std | N |")
        out.append("|---|---|---|---|---|")

        for dt in ['abrupt', 'gradual', 'noisy', 'stationary', 'real']:
            dt_df = cfg_df[cfg_df['drift_type'] == dt]
            if len(dt_df) > 0:
                out.append(f"| {dt} | {fmt_pm(dt_df['TCS'].mean(), dt_df['TCS'].std())} | "
                           f"{fmt_pm(dt_df['RIR'].mean(), dt_df['RIR'].std())} | "
                           f"{fmt_pm(dt_df['AMS'].mean(), dt_df['AMS'].std())} | {len(dt_df)} |")

    out.append("")


def section_5_model_params(out):
    """Section 5: Model Parameters (tab:baseline_params)."""
    out.append("## Section 5 -- Model Parameters (tab:baseline_params)\n")

    out.append("| Model | Framework | Key Parameters | Evaluation Protocol |")
    out.append("|---|---|---|---|")
    params = [
        ("EGIS", "Python (custom)", "pop=120, gen=200, elitism=0.1, mut=0.8, DT seeding=0.8, gamma={0.0,0.1,0.3}", "Train-then-test (chunk-based)"),
        ("ARF", "River", "n_models=10, defaults", "Prequential (per-instance)"),
        ("SRP", "River", "n_models=10, defaults", "Prequential (per-instance)"),
        ("HAT", "River", "defaults (grace_period=200, split_confidence=1e-7)", "Prequential (per-instance)"),
        ("ROSE", "MOA/ROSE", "WindowAUC evaluator, window=500", "Prequential (windowed)"),
        ("ACDWM", "Python/DWMIL", "theta=0.001, err_func=gm, r=1.0, binary only", "Prequential (per-instance)"),
        ("ERulesD2S", "MOA/JCLEC", "pop=25, gen=50, rules_per_class=5", "Prequential (windowed)"),
        ("CDCMS.CIL", "MOA/CIL", "f=chunk_size, holdout per-chunk eval", "Holdout (chunk-based)"),
    ]
    for row in params:
        out.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    out.append("")


def section_6_rankings(df, out):
    """Section 6: Rankings for Critical Difference Diagram (tab:stat_tests, table_ix_ranking)."""
    out.append("## Section 6 -- Statistical Rankings and Tests\n")
    out.append("For tab:stat_tests, table_ix_ranking, fig:cd_diagram\n")
    out.append("Rankings use NP configs for EGIS vs baselines comparison.\n")

    binary_df = df[~df['dataset'].isin(MULTICLASS_DATASETS)]

    for cfg in BASELINE_CONFIGS:
        out.append(f"\n### Config: {cfg}\n")

        # Binary rankings (EGIS + 7 baselines = 8 models)
        cfg_binary = binary_df[binary_df['config_label'] == cfg]
        pivot = cfg_binary.pivot_table(index='dataset', columns='model', values='gmean_mean', aggfunc='mean')

        # Only use datasets/models with data
        pivot = pivot.dropna(axis=1, how='all').dropna(axis=0, how='any')
        models_in_pivot = list(pivot.columns)

        if len(pivot) < 3 or len(models_in_pivot) < 2:
            out.append("Insufficient data for statistical tests.\n")
            continue

        # Rank (1 = best = highest gmean)
        ranks = pivot.rank(axis=1, ascending=False, method='average')
        mean_ranks = ranks.mean()

        out.append(f"#### Binary ({len(pivot)} datasets, {len(models_in_pivot)} models)\n")
        out.append("| Model | Mean Rank |")
        out.append("|---|---|")
        for model in mean_ranks.sort_values().index:
            out.append(f"| {model} | {fmt(mean_ranks[model], 2)} |")

        # Friedman test
        try:
            groups = [pivot[col].values for col in pivot.columns]
            friedman_stat, friedman_p = stats.friedmanchisquare(*groups)
            k = len(models_in_pivot)
            n = len(pivot)
            cd = 2.576 * np.sqrt((k * (k + 1)) / (6.0 * n))  # alpha=0.05 approx

            out.append(f"\n**Friedman test**: chi2={fmt(friedman_stat, 2)}, p={friedman_p:.2e}")
            out.append(f"**Nemenyi CD** (alpha=0.05, approx): {fmt(cd, 2)}")
            out.append(f"**k**={k}, **n**={n}")
        except Exception as e:
            out.append(f"\nFriedman test failed: {e}")

        # Pairwise Wilcoxon for EGIS vs all others
        if 'EGIS' in pivot.columns:
            out.append(f"\n#### Pairwise Wilcoxon (EGIS vs others, {cfg})\n")
            out.append("| Comparison | p-value | Bonferroni p | Cliff's delta | Effect Size |")
            out.append("|---|---|---|---|---|")

            egis_vals = pivot['EGIS'].values
            n_comparisons = len(models_in_pivot) - 1

            for model in sorted(models_in_pivot):
                if model == 'EGIS':
                    continue
                other_vals = pivot[model].values

                # Wilcoxon signed-rank
                try:
                    stat_w, p_w = stats.wilcoxon(egis_vals, other_vals, alternative='two-sided')
                    p_bonf = min(p_w * n_comparisons, 1.0)
                except Exception:
                    p_w = np.nan
                    p_bonf = np.nan

                # Cliff's delta
                n1, n2 = len(egis_vals), len(other_vals)
                more = np.sum(egis_vals[:, None] > other_vals[None, :])
                less = np.sum(egis_vals[:, None] < other_vals[None, :])
                cliff_d = (more - less) / (n1 * n2)

                if abs(cliff_d) < 0.147:
                    effect = "negligible"
                elif abs(cliff_d) < 0.33:
                    effect = "small"
                elif abs(cliff_d) < 0.474:
                    effect = "medium"
                else:
                    effect = "large"

                out.append(f"| EGIS vs {model} | {p_w:.2e} | {p_bonf:.2e} | {fmt(cliff_d, 3)} | {effect} |")

        # Multiclass rankings (exclude ACDWM/CDCMS)
        multiclass_df_all = df[df['dataset'].isin(MULTICLASS_DATASETS)]
        cfg_multi = multiclass_df_all[multiclass_df_all['config_label'] == cfg]

        if len(cfg_multi) > 0:
            pivot_m = cfg_multi.pivot_table(index='dataset', columns='model', values='gmean_mean', aggfunc='mean')
            pivot_m = pivot_m.drop(columns=[c for c in BINARY_ONLY_MODELS if c in pivot_m.columns], errors='ignore')
            pivot_m = pivot_m.dropna(axis=1, how='all').dropna(axis=0, how='any')

            if len(pivot_m) >= 3:
                ranks_m = pivot_m.rank(axis=1, ascending=False, method='average')
                mean_ranks_m = ranks_m.mean()

                out.append(f"\n#### Multiclass ({len(pivot_m)} datasets, {len(pivot_m.columns)} models)\n")
                out.append("| Model | Mean Rank |")
                out.append("|---|---|")
                for model in mean_ranks_m.sort_values().index:
                    out.append(f"| {model} | {fmt(mean_ranks_m[model], 2)} |")

    # ---- EGIS Variants Comparison ----
    out.append("\n### EGIS Variants Comparison (Binary)\n")
    out.append("Comparing EGIS penalty variants against each other.\n")

    for chunk_size in [500, 1000, 2000]:
        variants = EGIS_VARIANTS_BY_CHUNK[chunk_size]
        if len(variants) < 2:
            continue

        out.append(f"\n#### Chunk Size: {chunk_size} ({len(variants)} variants)\n")

        # Build pivot: dataset x variant
        variant_data = {}
        for v_cfg in variants:
            v_df = binary_df[(binary_df['config_label'] == v_cfg) & (binary_df['model'] == 'EGIS')]
            if len(v_df) > 0:
                variant_data[EGIS_VARIANT_DISPLAY[v_cfg]] = v_df.set_index('dataset')['gmean_mean']

        if len(variant_data) < 2:
            out.append("Insufficient variant data.\n")
            continue

        pivot_v = pd.DataFrame(variant_data).dropna()
        if len(pivot_v) < 3:
            out.append("Insufficient datasets for comparison.\n")
            continue

        ranks_v = pivot_v.rank(axis=1, ascending=False, method='average')
        mean_ranks_v = ranks_v.mean()

        out.append("| EGIS Variant | Mean G-Mean | Mean Rank |")
        out.append("|---|---|---|")
        for v_name in mean_ranks_v.sort_values().index:
            out.append(f"| {v_name} | {fmt(pivot_v[v_name].mean())} | {fmt(mean_ranks_v[v_name], 2)} |")

        # Wilcoxon: NP vs P
        if len(variants) >= 2:
            np_name = EGIS_VARIANT_DISPLAY[variants[0]]
            p_name = EGIS_VARIANT_DISPLAY[variants[1]]
            if np_name in pivot_v.columns and p_name in pivot_v.columns:
                try:
                    _, p_val = stats.wilcoxon(pivot_v[np_name].values, pivot_v[p_name].values)
                    out.append(f"\nWilcoxon {np_name} vs {p_name}: p={p_val:.4e}")
                except Exception as e:
                    out.append(f"\nWilcoxon test failed: {e}")

    out.append("")


def section_7_complexity(df_rules, out):
    """Section 7: Rule Complexity (tab:complexity_detailed)."""
    out.append("## Section 7 -- EGIS Rule Complexity (tab:complexity_detailed)\n")

    # Average per config (across all chunks and datasets)
    out.append("### Overall Complexity per Config\n")
    out.append("| Config | Avg Rules | Avg Cond/Rule | Avg AND/chunk | Avg OR/chunk | Avg Total Cond/chunk | N chunks |")
    out.append("|---|---|---|---|---|---|---|")

    for cfg in EGIS_CONFIGS:
        cfg_df = df_rules[df_rules['config_label'] == cfg]
        if len(cfg_df) > 0:
            out.append(f"| {cfg} | {fmt(cfg_df['n_rules'].mean(), 2)} | "
                       f"{fmt(cfg_df['avg_conditions_per_rule'].mean(), 2)} | "
                       f"{fmt(cfg_df['total_and_ops'].mean(), 2)} | "
                       f"{fmt(cfg_df['total_or_ops'].mean(), 2)} | "
                       f"{fmt(cfg_df['total_conditions'].mean(), 2)} | {len(cfg_df)} |")

    # By drift type (EXP-500-NP)
    out.append("\n### Complexity by Drift Type (EXP-500-NP)\n")
    out.append("| Drift Type | Avg Rules | Avg Cond/Rule | N chunks |")
    out.append("|---|---|---|---|")

    cfg_df = df_rules[df_rules['config_label'] == 'EXP-500-NP']
    for dt in ['abrupt', 'gradual', 'noisy', 'stationary', 'real']:
        dt_df = cfg_df[cfg_df['drift_type'] == dt]
        if len(dt_df) > 0:
            out.append(f"| {dt} | {fmt(dt_df['n_rules'].mean(), 2)} | "
                       f"{fmt(dt_df['avg_conditions_per_rule'].mean(), 2)} | {len(dt_df)} |")

    out.append("")


def section_8_dataset_analysis(df, out):
    """Section 8: Per-dataset analysis data for figures."""
    out.append("## Section 8 -- Per-Dataset Analysis Data\n")
    out.append("Data for generating figures (fig:chunk_size_effect, fig:config_comparison, etc.)\n")

    # Chunk size effect: EGIS performance by drift type and chunk size
    out.append("### Chunk Size Effect on EGIS (fig:chunk_size_effect)\n")
    out.append("| Drift Type | EXP-500-NP | EXP-1000-NP | EXP-2000-NP |")
    out.append("|---|---|---|---|")

    egis_df = df[df['model'] == 'EGIS']
    for dt in ['abrupt', 'gradual', 'noisy', 'stationary', 'real']:
        vals = []
        for cfg in ['EXP-500-NP', 'EXP-1000-NP', 'EXP-2000-NP']:
            sub = egis_df[(egis_df['config_label'] == cfg) & (egis_df['drift_type'] == dt)]
            if len(sub) > 0:
                vals.append(fmt_pm(sub['gmean_mean'].mean(), sub['gmean_mean'].std()))
            else:
                vals.append("N/A")
        out.append(f"| {dt} | {vals[0]} | {vals[1]} | {vals[2]} |")

    # Penalty effect (tab:penalty_effect)
    out.append("\n### Penalty Effect on EGIS (tab:penalty_effect)\n")
    out.append("| Chunk Size | No Penalty | Penalty 0.1 | Penalty 0.3 | Delta (P-NP) |")
    out.append("|---|---|---|---|---|")

    for cs_label, np_cfg, p_cfg, p03_cfg in [
        (500, 'EXP-500-NP', 'EXP-500-P', 'EXP-500-P03'),
        (1000, 'EXP-1000-NP', 'EXP-1000-P', None),
        (2000, 'EXP-2000-NP', 'EXP-2000-P', None),
    ]:
        np_df = egis_df[egis_df['config_label'] == np_cfg]
        p_df = egis_df[egis_df['config_label'] == p_cfg]

        np_mean = np_df['gmean_mean'].mean() if len(np_df) > 0 else np.nan
        p_mean = p_df['gmean_mean'].mean() if len(p_df) > 0 else np.nan
        delta = p_mean - np_mean if not (pd.isna(p_mean) or pd.isna(np_mean)) else np.nan

        p03_str = "N/A"
        if p03_cfg:
            p03_df = egis_df[egis_df['config_label'] == p03_cfg]
            if len(p03_df) > 0:
                p03_str = fmt_pm(p03_df['gmean_mean'].mean(), p03_df['gmean_mean'].std())

        out.append(f"| {cs_label} | {fmt_pm(np_mean, np_df['gmean_mean'].std() if len(np_df) > 0 else 0)} | "
                   f"{fmt_pm(p_mean, p_df['gmean_mean'].std() if len(p_df) > 0 else 0)} | "
                   f"{p03_str} | {fmt(delta)} |")

    # Config comparison data (fig:config_comparison)
    out.append("\n### All EGIS Configs by Drift Type (fig:config_comparison)\n")
    header = "| Drift Type | " + " | ".join(EGIS_CONFIGS) + " |"
    out.append(header)
    out.append("|---| " + "---|" * len(EGIS_CONFIGS))

    for dt in ['abrupt', 'gradual', 'noisy', 'stationary', 'real']:
        vals = []
        for cfg in EGIS_CONFIGS:
            sub = egis_df[(egis_df['config_label'] == cfg) & (egis_df['drift_type'] == dt)]
            if len(sub) > 0:
                vals.append(fmt(sub['gmean_mean'].mean()))
            else:
                vals.append("N/A")
        out.append(f"| {dt} | {' | '.join(vals)} |")

    # Boxplot data (fig:boxplots) - EXP-500-NP all models
    out.append("\n### Performance Distribution (fig:boxplots, EXP-500-NP, binary)\n")
    out.append("| Model | Min | Q1 | Median | Q3 | Max | Mean | Std |")
    out.append("|---|---|---|---|---|---|---|---|")

    binary_df = df[~df['dataset'].isin(MULTICLASS_DATASETS)]
    cfg_df = binary_df[binary_df['config_label'] == 'EXP-500-NP']

    for model in ALL_MODELS:
        model_df = cfg_df[cfg_df['model'] == model]
        if len(model_df) > 0:
            vals = model_df['gmean_mean'].values
            out.append(f"| {model} | {fmt(np.min(vals))} | {fmt(np.percentile(vals, 25))} | "
                       f"{fmt(np.median(vals))} | {fmt(np.percentile(vals, 75))} | "
                       f"{fmt(np.max(vals))} | {fmt(np.mean(vals))} | {fmt(np.std(vals))} |")

    # Drift detection summary (if available)
    drift_summary_path = PAPER_DATA_DIR / "drift_detection_summary.csv"
    if drift_summary_path.exists():
        out.append("\n### Drift Detection Summary\n")
        drift_df = pd.read_csv(drift_summary_path)
        out.append(f"Total records: {len(drift_df)}")

        if 'n_drifts_detected' in drift_df.columns:
            out.append("\n#### Average Drifts Detected per Config\n")
            out.append("| Config | Avg Drifts | Max Drifts |")
            out.append("|---|---|---|")
            for cfg in EGIS_CONFIGS:
                cfg_d = drift_df[drift_df['config_label'] == cfg]
                if len(cfg_d) > 0:
                    out.append(f"| {cfg} | {fmt(cfg_d['n_drifts_detected'].mean(), 2)} | "
                               f"{int(cfg_d['n_drifts_detected'].max())} |")

    out.append("")


def section_9_evolution(out):
    """Section 9: Rule Evolution Data (for figures)."""
    out.append("## Section 9 -- Rule Evolution Data\n")
    out.append("Data for fig:rule_evolution, fig:transition_matrix, fig:evolution_heatmaps\n")

    # Rule evolution summary (from batch_rule_diff_analysis.py)
    rule_evo_path = PAPER_DATA_DIR / "rule_evolution_summary.csv"
    if rule_evo_path.exists():
        evo_df = pd.read_csv(rule_evo_path)
        out.append(f"\n### Rule Evolution Summary ({len(evo_df)} records)\n")

        out.append("#### Rule Change Counts per Config\n")
        out.append("| Config | Avg Unchanged | Avg Modified | Avg New | Avg Deleted | Avg Similarity | Avg Rules/Chunk | N |")
        out.append("|---|---|---|---|---|---|---|---|")
        for cfg in EGIS_CONFIGS:
            cfg_d = evo_df[evo_df['config_label'] == cfg] if 'config_label' in evo_df.columns else pd.DataFrame()
            if len(cfg_d) > 0:
                out.append(f"| {cfg} | "
                           f"{fmt(cfg_d['avg_unchanged_per_transition'].mean(), 2)} | "
                           f"{fmt(cfg_d['avg_modified_per_transition'].mean(), 2)} | "
                           f"{fmt(cfg_d['avg_new_per_transition'].mean(), 2)} | "
                           f"{fmt(cfg_d['avg_deleted_per_transition'].mean(), 2)} | "
                           f"{fmt(cfg_d['avg_similarity'].mean())} | "
                           f"{fmt(cfg_d['avg_rules_per_chunk'].mean(), 1)} | {len(cfg_d)} |")

        # By drift type
        out.append("\n#### Rule Changes by Drift Type (EXP-500-NP)\n")
        out.append("| Drift Type | Avg Modified | Avg New | Avg Deleted | Avg Similarity | N |")
        out.append("|---|---|---|---|---|---|")
        cfg_d = evo_df[evo_df['config_label'] == 'EXP-500-NP'] if 'config_label' in evo_df.columns else pd.DataFrame()
        for dt in ['abrupt', 'gradual', 'noisy', 'stationary', 'real']:
            dt_d = cfg_d[cfg_d['drift_type'] == dt] if 'drift_type' in cfg_d.columns else pd.DataFrame()
            if len(dt_d) > 0:
                out.append(f"| {dt} | "
                           f"{fmt(dt_d['avg_modified_per_transition'].mean(), 2)} | "
                           f"{fmt(dt_d['avg_new_per_transition'].mean(), 2)} | "
                           f"{fmt(dt_d['avg_deleted_per_transition'].mean(), 2)} | "
                           f"{fmt(dt_d['avg_similarity'].mean())} | {len(dt_d)} |")
    else:
        out.append("\n*rule_evolution_summary.csv not yet generated. Run batch_rule_diff_analysis.py first.*\n")

    # AST-based evolution analysis summary (optional, from batch_evolution_graphs_data.py)
    evo_analysis_path = PAPER_DATA_DIR / "evolution_analysis_summary.csv"
    if evo_analysis_path.exists():
        evo_a = pd.read_csv(evo_analysis_path)
        if len(evo_a) > 0:
            out.append(f"\n### AST-based Detailed Evolution Analysis ({len(evo_a)} records)\n")

            if 'TCS' in evo_a.columns:
                out.append("#### AST-based Transition Metrics per Config\n")
                out.append("| Config | TCS mean | RIR mean | AMS mean | N |")
                out.append("|---|---|---|---|---|")
                for cfg in EGIS_CONFIGS:
                    cfg_d = evo_a[evo_a['config_label'] == cfg] if 'config_label' in evo_a.columns else pd.DataFrame()
                    if len(cfg_d) > 0:
                        out.append(f"| {cfg} | {fmt(cfg_d['TCS'].mean())} | "
                                   f"{fmt(cfg_d['RIR'].mean())} | {fmt(cfg_d['AMS'].mean())} | {len(cfg_d)} |")
    else:
        out.append("\n*Note: AST-based evolution_analysis_summary.csv not available (optional, computationally expensive).*")
        out.append("*Levenshtein-based metrics in Section 4 are sufficient for the paper.*\n")

    out.append("")


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 80)
    logger.info("PREPARING AUXILIARY DATA DOCUMENT FOR IEEE TKDE PAPER")
    logger.info("=" * 80)

    # Load data
    results_path = PAPER_DATA_DIR / "consolidated_results.csv"
    rules_path = PAPER_DATA_DIR / "egis_rules_per_chunk.csv"
    transitions_path = PAPER_DATA_DIR / "egis_transition_metrics.csv"

    if not results_path.exists():
        logger.error(f"consolidated_results.csv not found at {results_path}")
        logger.error("Run collect_results_for_paper.py first.")
        return 1

    df = pd.read_csv(results_path)
    logger.info(f"Loaded consolidated_results.csv: {len(df)} rows, "
                f"{df['dataset'].nunique()} datasets, {df['model'].nunique()} models, "
                f"{df['config_label'].nunique()} configs")

    df_rules = pd.read_csv(rules_path) if rules_path.exists() else pd.DataFrame()
    df_trans = pd.read_csv(transitions_path) if transitions_path.exists() else pd.DataFrame()

    logger.info(f"Loaded egis_rules_per_chunk.csv: {len(df_rules)} rows")
    logger.info(f"Loaded egis_transition_metrics.csv: {len(df_trans)} rows")

    # Build document
    out = []
    out.append(f"# Auxiliary Data Document for IEEE TKDE Paper")
    out.append(f"")
    out.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(f"")
    out.append(f"**Source data**:")
    out.append(f"- consolidated_results.csv: {len(df)} rows")
    out.append(f"- egis_rules_per_chunk.csv: {len(df_rules)} rows")
    out.append(f"- egis_transition_metrics.csv: {len(df_trans)} rows")
    out.append(f"")
    out.append(f"**Dimensions**: {df['dataset'].nunique()} datasets ({len([d for d in df['dataset'].unique() if d not in MULTICLASS_DATASETS])} binary + "
               f"{len([d for d in df['dataset'].unique() if d in MULTICLASS_DATASETS])} multiclass), "
               f"{df['model'].nunique()} models, {df['config_label'].nunique()} configs")
    out.append(f"")
    out.append(f"---")
    out.append(f"")

    # Generate sections
    section_1_dimensions(df, out)
    section_2_configs(df, out)
    section_3_performance(df, out)

    if not df_trans.empty:
        section_4_transitions(df_trans, out)
    else:
        out.append("## Section 4 -- Transition Metrics\n")
        out.append("*No transition data available.*\n")

    section_5_model_params(out)
    section_6_rankings(df, out)

    if not df_rules.empty:
        section_7_complexity(df_rules, out)
    else:
        out.append("## Section 7 -- Rule Complexity\n")
        out.append("*No rule data available.*\n")

    section_8_dataset_analysis(df, out)
    section_9_evolution(out)

    # Write document
    doc_content = "\n".join(out)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(doc_content)

    logger.info(f"\nSaved: {OUTPUT_FILE} ({len(doc_content)} chars, {len(out)} lines)")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
