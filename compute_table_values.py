#!/usr/bin/env python3
"""
compute_table_values.py

Reads data files and outputs all computed values needed for the IEEE TKDE paper tables.
Covers: tab:summary_all, tab:complexity_detailed, tab:penalty_effect, W/L/D stats, Rankings.

Usage:
    python compute_table_values.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONSOLIDATED_CSV = os.path.join(BASE_DIR, "paper_data", "consolidated_results.csv")
STATISTICAL_JSON = os.path.join(BASE_DIR, "paper_data", "statistical_results.json")
COMPLEXITY_CSV = os.path.join(BASE_DIR, "paper_data", "egis_complexity_summary.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MULTICLASS_DATASETS = [
    "LED_Abrupt_Simple",
    "LED_Gradual_Simple",
    "LED_Stationary",
    "RBF_Stationary",
    "WAVEFORM_Abrupt_Simple",
    "WAVEFORM_Gradual_Simple",
    "WAVEFORM_Stationary",
]

BINARY_ONLY_MODELS = ["EGIS", "ARF", "SRP", "HAT", "ROSE", "ACDWM", "ERulesD2S", "CDCMS"]
# Models that support multiclass (exclude ACDWM, CDCMS for all-48 table)
MULTICLASS_CAPABLE_MODELS = ["EGIS", "ARF", "SRP", "HAT", "ROSE", "ERulesD2S"]

NP_CONFIGS = ["EXP-500-NP", "EXP-1000-NP", "EXP-2000-NP"]

ALL_COMPLEXITY_CONFIGS = [
    "EXP-500-NP", "EXP-500-P", "EXP-500-P03",
    "EXP-1000-NP", "EXP-1000-P",
    "EXP-2000-NP", "EXP-2000-P",
]

PENALTY_PAIRS = [
    # (NP config, P config, chunk_size label)
    ("EXP-500-NP", "EXP-500-P", 500),
    ("EXP-1000-NP", "EXP-1000-P", 1000),
    ("EXP-2000-NP", "EXP-2000-P", 2000),
]


def separator(title):
    """Print a section separator."""
    width = 80
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def load_data():
    """Load all data files and return them."""
    df = pd.read_csv(CONSOLIDATED_CSV)
    with open(STATISTICAL_JSON, "r", encoding="utf-8") as f:
        stat = json.load(f)
    complexity = pd.read_csv(COMPLEXITY_CSV)
    return df, stat, complexity


# ===================================================================
# SECTION 1: tab:summary_all values
# ===================================================================
def section1_summary_all(df):
    separator("SECTION 1: tab:summary_all -- G-Mean per model per config")

    binary_datasets = df[~df["dataset"].isin(MULTICLASS_DATASETS)]["dataset"].unique()

    for config_label in NP_CONFIGS:
        cfg = df[df["config_label"] == config_label]
        print(f"\n--- {config_label} ---")

        # Binary only (n=41), all 8 models
        print(f"\n  Binary only (n={len(binary_datasets)}):")
        cfg_bin = cfg[~cfg["dataset"].isin(MULTICLASS_DATASETS)]
        for model in BINARY_ONLY_MODELS:
            model_data = cfg_bin[cfg_bin["model"] == model]
            if model_data.empty:
                print(f"    {model:<12s}: NO DATA")
                continue
            mean_gm = model_data["gmean_mean"].mean()
            std_gm = model_data["gmean_mean"].std()
            n = model_data["dataset"].nunique()
            latex = f"{mean_gm:.3f} & {std_gm:.3f}"
            print(f"    {model:<12s}: mean={mean_gm:.4f}  std={std_gm:.4f}  n={n}  LaTeX: {latex}")

        # All datasets (n=48), only 6 multiclass-capable models
        print(f"\n  All datasets (n={cfg['dataset'].nunique()}):")
        for model in MULTICLASS_CAPABLE_MODELS:
            model_data = cfg[cfg["model"] == model]
            if model_data.empty:
                print(f"    {model:<12s}: NO DATA")
                continue
            mean_gm = model_data["gmean_mean"].mean()
            std_gm = model_data["gmean_mean"].std()
            n = model_data["dataset"].nunique()
            latex = f"{mean_gm:.3f} & {std_gm:.3f}"
            print(f"    {model:<12s}: mean={mean_gm:.4f}  std={std_gm:.4f}  n={n}  LaTeX: {latex}")


# ===================================================================
# SECTION 2: tab:complexity_detailed values
# ===================================================================
def section2_complexity(complexity):
    separator("SECTION 2: tab:complexity_detailed -- EGIS rule complexity")

    for config_label in ALL_COMPLEXITY_CONFIGS:
        cfg = complexity[complexity["config_label"] == config_label]
        if cfg.empty:
            print(f"\n  {config_label}: NO DATA")
            continue
        n_datasets = cfg["dataset"].nunique()
        rules_mean = cfg["n_rules"].mean()
        rules_std = cfg["n_rules"].std()
        conds_mean = cfg["avg_conditions_per_rule"].mean()
        conds_std = cfg["avg_conditions_per_rule"].std()
        and_mean = cfg["total_and_ops"].mean()
        or_mean = cfg["total_or_ops"].mean()

        print(f"\n  {config_label} (n={n_datasets} datasets):")
        print(f"    n_rules:               {rules_mean:.2f} +/- {rules_std:.2f}")
        print(f"    avg_conds_per_rule:     {conds_mean:.2f} +/- {conds_std:.2f}")
        print(f"    total_and_ops (mean):   {and_mean:.2f}")
        print(f"    total_or_ops  (mean):   {or_mean:.2f}")
        latex_rules = f"{rules_mean:.1f} $\\pm$ {rules_std:.1f}"
        latex_conds = f"{conds_mean:.2f} $\\pm$ {conds_std:.2f}"
        print(f"    LaTeX rules: {latex_rules}")
        print(f"    LaTeX conds: {latex_conds}")
        print(f"    LaTeX AND:   {and_mean:.1f}")
        print(f"    LaTeX OR:    {or_mean:.1f}")


# ===================================================================
# SECTION 3: tab:penalty_effect values
# ===================================================================
def section3_penalty_effect(df):
    separator("SECTION 3: tab:penalty_effect -- NP vs P G-Mean (EGIS only, all 48 datasets)")

    egis = df[df["model"] == "EGIS"]

    def get_paired_gmeans(config_label):
        """Return a Series indexed by dataset with gmean_mean for EGIS."""
        cfg_data = egis[egis["config_label"] == config_label]
        return cfg_data.set_index("dataset")["gmean_mean"].sort_index()

    for np_config, p_config, chunk_size in PENALTY_PAIRS:
        np_gmeans = get_paired_gmeans(np_config)
        p_gmeans = get_paired_gmeans(p_config)

        # Align on common datasets
        common = np_gmeans.index.intersection(p_gmeans.index)
        np_vals = np_gmeans.loc[common].values
        p_vals = p_gmeans.loc[common].values

        np_mean = np_vals.mean()
        np_std = np_vals.std()
        p_mean = p_vals.mean()
        p_std = p_vals.std()
        delta = np_mean - p_mean

        # Wilcoxon signed-rank test
        try:
            w_stat, w_pval = stats.wilcoxon(np_vals, p_vals)
        except ValueError:
            w_stat, w_pval = float("nan"), float("nan")

        print(f"\n  Chunk {chunk_size}: {np_config} vs {p_config} (n={len(common)} datasets)")
        print(f"    NP: {np_mean:.4f} +/- {np_std:.4f}")
        print(f"    P:  {p_mean:.4f} +/- {p_std:.4f}")
        print(f"    Delta (NP - P): {delta:+.4f}")
        print(f"    Wilcoxon: stat={w_stat:.1f}, p={w_pval:.6f}")
        sig = "Yes" if w_pval < 0.05 else "No"
        print(f"    Significant (p<0.05)? {sig}")
        latex_np = f"{np_mean:.3f} $\\pm$ {np_std:.3f}"
        latex_p = f"{p_mean:.3f} $\\pm$ {p_std:.3f}"
        print(f"    LaTeX NP:    {latex_np}")
        print(f"    LaTeX P:     {latex_p}")
        print(f"    LaTeX Delta: {delta:+.3f}")
        print(f"    LaTeX p-val: {w_pval:.4f}" if not np.isnan(w_pval) else "    LaTeX p-val: N/A")

    # EXP-500-P03 vs EXP-500-NP separately
    print("\n  --- EXP-500-P03 vs EXP-500-NP ---")
    np_gmeans = get_paired_gmeans("EXP-500-NP")
    p03_gmeans = get_paired_gmeans("EXP-500-P03")
    common = np_gmeans.index.intersection(p03_gmeans.index)
    np_vals = np_gmeans.loc[common].values
    p03_vals = p03_gmeans.loc[common].values

    np_mean = np_vals.mean()
    np_std = np_vals.std()
    p03_mean = p03_vals.mean()
    p03_std = p03_vals.std()
    delta = np_mean - p03_mean

    try:
        w_stat, w_pval = stats.wilcoxon(np_vals, p03_vals)
    except ValueError:
        w_stat, w_pval = float("nan"), float("nan")

    print(f"    EXP-500-NP: {np_mean:.4f} +/- {np_std:.4f}")
    print(f"    EXP-500-P03: {p03_mean:.4f} +/- {p03_std:.4f}")
    print(f"    Delta (NP - P03): {delta:+.4f}")
    print(f"    Wilcoxon: stat={w_stat:.1f}, p={w_pval:.6f}")
    sig = "Yes" if w_pval < 0.05 else "No"
    print(f"    Significant (p<0.05)? {sig}")
    latex_np = f"{np_mean:.3f} $\\pm$ {np_std:.3f}"
    latex_p03 = f"{p03_mean:.3f} $\\pm$ {p03_std:.3f}"
    print(f"    LaTeX NP:    {latex_np}")
    print(f"    LaTeX P03:   {latex_p03}")
    print(f"    LaTeX Delta: {delta:+.3f}")
    print(f"    LaTeX p-val: {w_pval:.4f}" if not np.isnan(w_pval) else "    LaTeX p-val: N/A")


# ===================================================================
# SECTION 4: W/L/D stats
# ===================================================================
def section4_wld(stat):
    separator("SECTION 4: W/L/D -- EGIS vs each model (binary_only)")

    binary_configs = stat["binary_only"]["configurations"]
    for cfg in binary_configs:
        config_label = cfg["config_label"]
        if "egis_wld" not in cfg:
            continue
        print(f"\n  --- {config_label} (n={cfg['n_datasets']} binary datasets) ---")
        wld = cfg["egis_wld"]
        print(f"    {'Model':<12s}  {'W':>4s}  {'L':>4s}  {'D':>4s}  LaTeX")
        print(f"    {'-'*44}")
        for model in ["ARF", "SRP", "HAT", "ROSE", "ACDWM", "ERulesD2S", "CDCMS"]:
            if model not in wld:
                continue
            w = wld[model]["wins"]
            l = wld[model]["losses"]
            d = wld[model]["ties"]
            latex = f"{w}/{l}/{d}"
            print(f"    {model:<12s}  {w:4d}  {l:4d}  {d:4d}  {latex}")


# ===================================================================
# SECTION 5: Rankings
# ===================================================================
def section5_rankings(stat):
    separator("SECTION 5: Rankings -- Friedman test & average ranks (binary_only)")

    binary_configs = stat["binary_only"]["configurations"]
    for cfg in binary_configs:
        config_label = cfg["config_label"]
        friedman = cfg["friedman_test"]
        cd = cfg["critical_distance"]
        rankings = cfg["average_rankings"]

        print(f"\n  --- {config_label} ---")
        print(f"    Friedman chi2({friedman['df']}) = {friedman['statistic']:.2f}")
        print(f"    p-value = {friedman['p_value']:.2e}")
        print(f"    Significant: {friedman['significant']}")
        print(f"    Nemenyi CD = {cd:.4f}")
        print()
        print(f"    Average Rankings (sorted):")
        sorted_ranks = sorted(rankings.items(), key=lambda x: x[1])
        for model, rank in sorted_ranks:
            print(f"      {model:<12s}: {rank:.2f}")

        # Pairwise EGIS comparisons summary
        print()
        print(f"    EGIS pairwise comparisons:")
        print(f"    {'Comparison':<22s}  {'p_adj':>10s}  {'Sig?':>5s}  {'delta':>8s}  {'Effect':<12s}")
        print(f"    {'-'*65}")
        for pw in cfg["pairwise_tests"]:
            comp = pw["comparison"]
            p_adj = pw["adjusted_p_value"]
            sig = "Yes" if pw["significant_bonferroni"] else "No"
            delta = pw["cliffs_delta"]
            effect = pw["effect_interpretation"]
            print(f"    {comp:<22s}  {p_adj:10.4f}  {sig:>5s}  {delta:+8.3f}  {effect:<12s}")


# ===================================================================
# Main
# ===================================================================
def main():
    print("compute_table_values.py")
    print("Computes all values needed for IEEE TKDE paper tables.")
    print(f"Data directory: {os.path.join(BASE_DIR, 'paper_data')}")

    # Verify files exist
    for path, name in [
        (CONSOLIDATED_CSV, "consolidated_results.csv"),
        (STATISTICAL_JSON, "statistical_results.json"),
        (COMPLEXITY_CSV, "egis_complexity_summary.csv"),
    ]:
        if not os.path.isfile(path):
            print(f"ERROR: {name} not found at {path}", file=sys.stderr)
            sys.exit(1)

    df, stat, complexity = load_data()

    section1_summary_all(df)
    section2_complexity(complexity)
    section3_penalty_effect(df)
    section4_wld(stat)
    section5_rankings(stat)

    separator("DONE")
    print("  All table values computed successfully.")
    print()


if __name__ == "__main__":
    main()
