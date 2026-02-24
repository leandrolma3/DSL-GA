#!/usr/bin/env python3
"""
Statistical Analysis Script for IEEE TKDE Paper - Section VI

Performs comprehensive statistical analysis on consolidated results:
- Friedman test for overall ranking differences
- Nemenyi post-hoc test for Critical Difference diagram
- Pairwise Wilcoxon signed-rank tests with Bonferroni correction
- Cliff's Delta effect sizes
- Average ranking calculations

Output: paper_data/statistical_results.json and related files

Author: Automated Analysis
Date: 2026-01-27
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = Path("paper_data")
OUTPUT_DIR = Path("paper_data")

# Datasets excluded from all analyses
EXCLUDED_DATASETS = ['CovType', 'IntelLabSensors', 'PokerHand', 'Shuttle']

# Significance level
ALPHA = 0.05

# Models to include in analysis (in preferred order)
MODELS_ORDER = ['EGIS', 'ARF', 'SRP', 'HAT', 'ROSE', 'ACDWM', 'ERulesD2S', 'CDCMS']

# Drift types for stratified analysis
DRIFT_TYPES = ['abrupt', 'gradual', 'noisy', 'stationary', 'real']

# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def friedman_test(data: pd.DataFrame) -> Dict:
    """
    Perform Friedman test for comparing multiple models across datasets.

    Args:
        data: DataFrame with datasets as rows and models as columns

    Returns:
        Dictionary with test statistic, p-value, and rankings
    """
    # Get values as list of arrays (one per model)
    models = data.columns.tolist()
    n_datasets = len(data)
    n_models = len(models)

    # Calculate ranks for each dataset (row)
    ranks = data.rank(axis=1, ascending=False, method='average')
    avg_ranks = ranks.mean()

    # Friedman test statistic
    try:
        stat, p_value = stats.friedmanchisquare(*[data[col].values for col in models])
    except Exception as e:
        logger.warning(f"Friedman test failed: {e}")
        stat, p_value = np.nan, 1.0

    return {
        'statistic': float(stat) if not np.isnan(stat) else None,
        'p_value': float(p_value),
        'n_datasets': n_datasets,
        'n_models': n_models,
        'df': n_models - 1,
        'average_ranks': {model: float(rank) for model, rank in avg_ranks.items()},
        'significant': p_value < ALPHA
    }


def nemenyi_critical_distance(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    """
    Calculate Critical Distance for Nemenyi post-hoc test.

    CD = q_alpha * sqrt(k(k+1) / 6N)
    where k = number of models, N = number of datasets
    """
    # q_alpha values for alpha=0.05 (from studentized range table)
    # Index corresponds to number of groups - 2
    q_alpha_values = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }

    q_alpha = q_alpha_values.get(n_models, 3.164)  # Default to k=10 for larger

    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))
    return cd


def wilcoxon_test(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Perform Wilcoxon signed-rank test between two models.

    Args:
        x, y: Performance arrays for two models

    Returns:
        Dictionary with test results
    """
    # Remove pairs where both are equal or either is NaN
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if len(x_valid) < 5:
        return {
            'statistic': None,
            'p_value': 1.0,
            'n_samples': len(x_valid),
            'significant': False
        }

    try:
        stat, p_value = stats.wilcoxon(x_valid, y_valid, alternative='two-sided')
    except Exception as e:
        logger.warning(f"Wilcoxon test failed: {e}")
        return {
            'statistic': None,
            'p_value': 1.0,
            'n_samples': len(x_valid),
            'significant': False
        }

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'n_samples': int(len(x_valid)),
        'mean_diff': float(np.mean(x_valid - y_valid)),
        'significant': p_value < ALPHA
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Calculate Cliff's Delta effect size.

    Args:
        x, y: Performance arrays for two models

    Returns:
        Tuple of (delta value, interpretation)
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if len(x_valid) == 0:
        return 0.0, 'negligible'

    n1, n2 = len(x_valid), len(y_valid)

    # Count dominances
    dominates = 0
    for xi in x_valid:
        for yj in y_valid:
            if xi > yj:
                dominates += 1
            elif xi < yj:
                dominates -= 1

    delta = dominates / (n1 * n2)

    # Interpretation thresholds (Romano et al., 2006)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = 'negligible'
    elif abs_delta < 0.33:
        interpretation = 'small'
    elif abs_delta < 0.474:
        interpretation = 'medium'
    else:
        interpretation = 'large'

    return float(delta), interpretation


def pairwise_wilcoxon_with_bonferroni(data: pd.DataFrame, reference_model: str = 'EGIS') -> List[Dict]:
    """
    Perform pairwise Wilcoxon tests with Bonferroni correction.

    Args:
        data: DataFrame with datasets as rows and models as columns
        reference_model: Model to compare others against

    Returns:
        List of comparison results
    """
    models = [m for m in data.columns if m != reference_model]
    n_comparisons = len(models)
    alpha_corrected = ALPHA / n_comparisons

    results = []
    for other_model in models:
        x = data[reference_model].values
        y = data[other_model].values

        wilcoxon_result = wilcoxon_test(x, y)
        delta, interpretation = cliffs_delta(x, y)

        results.append({
            'comparison': f'{reference_model} vs {other_model}',
            'model_1': reference_model,
            'model_2': other_model,
            'raw_p_value': wilcoxon_result['p_value'],
            'adjusted_p_value': min(wilcoxon_result['p_value'] * n_comparisons, 1.0),
            'significant_raw': wilcoxon_result['p_value'] < ALPHA,
            'significant_bonferroni': wilcoxon_result['p_value'] < alpha_corrected,
            'mean_difference': wilcoxon_result.get('mean_diff', 0.0),
            'cliffs_delta': delta,
            'effect_interpretation': interpretation,
            'n_samples': wilcoxon_result['n_samples'],
            'alpha_corrected': alpha_corrected
        })

    return results


def calculate_model_rankings(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-dataset rankings for each model.

    Args:
        data: DataFrame with datasets as rows and models as columns

    Returns:
        DataFrame with rankings (1 = best)
    """
    return data.rank(axis=1, ascending=False, method='average')


def count_wins_ties_losses(data: pd.DataFrame, model1: str, model2: str) -> Dict:
    """Count wins, ties, and losses between two models."""
    diff = data[model1] - data[model2]
    wins = int((diff > 0.001).sum())  # Small threshold for numerical stability
    losses = int((diff < -0.001).sum())
    ties = int(len(diff) - wins - losses)

    return {
        'wins': wins,
        'ties': ties,
        'losses': losses
    }


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def perform_overall_analysis(df_results: pd.DataFrame) -> Dict:
    """Perform overall statistical analysis across all configurations."""
    logger.info("Performing overall statistical analysis...")

    results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'overall',
        'configurations': [],
        'combined': {}
    }

    # Analyze each configuration separately
    for config_label in df_results['config_label'].unique():
        config_df = df_results[df_results['config_label'] == config_label]

        # Pivot to get models as columns
        pivot = config_df.pivot_table(
            index='dataset',
            columns='model',
            values='gmean_mean',
            aggfunc='mean'
        )

        # Filter to models with sufficient data
        valid_models = [m for m in MODELS_ORDER if m in pivot.columns]
        pivot = pivot[valid_models].dropna(how='all')

        if len(valid_models) < 2 or len(pivot) < 5:
            logger.warning(f"Insufficient data for config {config_label}")
            continue

        # Fill missing values with 0 (for failed models like ERulesD2S)
        pivot = pivot.fillna(0)

        # Friedman test
        friedman_result = friedman_test(pivot)

        # Nemenyi critical distance
        cd = nemenyi_critical_distance(len(valid_models), len(pivot))

        # Pairwise Wilcoxon tests (if EGIS present)
        if 'EGIS' in valid_models:
            pairwise_results = pairwise_wilcoxon_with_bonferroni(pivot, 'EGIS')
        else:
            pairwise_results = []

        # Model performance summary
        model_summary = {}
        for model in valid_models:
            model_data = pivot[model]
            model_summary[model] = {
                'mean': float(model_data.mean()),
                'std': float(model_data.std()),
                'median': float(model_data.median()),
                'min': float(model_data.min()),
                'max': float(model_data.max()),
                'n_datasets': int((model_data > 0).sum())
            }

        # Rankings
        rankings = calculate_model_rankings(pivot)
        avg_rankings = rankings.mean().to_dict()

        # Count wins for each model
        wins_count = {}
        for model in valid_models:
            wins_count[model] = int((rankings[model] == 1).sum())

        config_results = {
            'config_label': config_label,
            'n_datasets': len(pivot),
            'n_models': len(valid_models),
            'models': valid_models,
            'friedman_test': friedman_result,
            'critical_distance': float(cd),
            'pairwise_tests': pairwise_results,
            'model_summary': model_summary,
            'average_rankings': avg_rankings,
            'wins_count': wins_count
        }

        results['configurations'].append(config_results)

    return results


def perform_stratified_analysis(df_results: pd.DataFrame) -> Dict:
    """Perform statistical analysis stratified by drift type."""
    logger.info("Performing stratified analysis by drift type...")

    results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'stratified_by_drift',
        'drift_types': {}
    }

    for drift_type in DRIFT_TYPES:
        drift_df = df_results[df_results['drift_type'] == drift_type]

        if drift_df.empty:
            continue

        drift_results = {
            'n_datasets': drift_df['dataset'].nunique(),
            'configurations': []
        }

        for config_label in drift_df['config_label'].unique():
            config_df = drift_df[drift_df['config_label'] == config_label]

            pivot = config_df.pivot_table(
                index='dataset',
                columns='model',
                values='gmean_mean',
                aggfunc='mean'
            )

            valid_models = [m for m in MODELS_ORDER if m in pivot.columns]
            pivot = pivot[valid_models].dropna(how='all').fillna(0)

            if len(valid_models) < 2 or len(pivot) < 3:
                continue

            # Rankings
            rankings = calculate_model_rankings(pivot)
            avg_rankings = {m: float(rankings[m].mean()) for m in valid_models}

            # Model performance
            model_perf = {m: {'mean': float(pivot[m].mean()),
                            'std': float(pivot[m].std())}
                        for m in valid_models}

            drift_results['configurations'].append({
                'config_label': config_label,
                'n_datasets': len(pivot),
                'average_rankings': avg_rankings,
                'model_performance': model_perf
            })

        results['drift_types'][drift_type] = drift_results

    return results


def perform_egis_penalty_analysis(df_results: pd.DataFrame) -> Dict:
    """Analyze EGIS performance with vs without complexity penalty."""
    logger.info("Performing EGIS penalty effect analysis...")

    egis_df = df_results[df_results['model'] == 'EGIS'].copy()

    results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'egis_penalty_comparison',
        'comparisons': []
    }

    # Compare chunk_500 vs chunk_500_penalty
    for chunk_size in [500, 1000, 2000]:
        no_penalty_label = f'EXP-{chunk_size}-NP'
        with_penalty_label = f'EXP-{chunk_size}-P'

        no_penalty_df = egis_df[egis_df['config_label'] == no_penalty_label]
        with_penalty_df = egis_df[egis_df['config_label'] == with_penalty_label]

        if no_penalty_df.empty or with_penalty_df.empty:
            continue

        # Merge on dataset
        merged = pd.merge(
            no_penalty_df[['dataset', 'gmean_mean']],
            with_penalty_df[['dataset', 'gmean_mean']],
            on='dataset',
            suffixes=('_no_penalty', '_with_penalty')
        )

        if len(merged) < 5:
            continue

        x = merged['gmean_mean_no_penalty'].values
        y = merged['gmean_mean_with_penalty'].values

        wilcoxon_result = wilcoxon_test(x, y)
        delta, interpretation = cliffs_delta(x, y)

        results['comparisons'].append({
            'chunk_size': chunk_size,
            'no_penalty_label': no_penalty_label,
            'with_penalty_label': with_penalty_label,
            'no_penalty_mean': float(np.mean(x)),
            'no_penalty_std': float(np.std(x)),
            'with_penalty_mean': float(np.mean(y)),
            'with_penalty_std': float(np.std(y)),
            'mean_difference': float(np.mean(x - y)),
            'wilcoxon_p_value': wilcoxon_result['p_value'],
            'significant': wilcoxon_result['significant'],
            'cliffs_delta': delta,
            'effect_interpretation': interpretation,
            'n_datasets': len(merged)
        })

    return results


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("STATISTICAL ANALYSIS FOR IEEE TKDE PAPER - SECTION VI")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load consolidated results
    results_file = INPUT_DIR / "consolidated_results.csv"
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.error("Please run collect_results_for_paper.py first")
        return 1

    df_results = pd.read_csv(results_file)
    df_results = df_results[~df_results['dataset'].isin(EXCLUDED_DATASETS)]
    logger.info(f"Loaded {len(df_results)} results from {results_file} (after excluding {EXCLUDED_DATASETS})")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Perform analyses
    overall_results = perform_overall_analysis(df_results)
    stratified_results = perform_stratified_analysis(df_results)
    penalty_results = perform_egis_penalty_analysis(df_results)

    # Combine all results
    all_results = {
        'overall': overall_results,
        'stratified': stratified_results,
        'penalty_comparison': penalty_results,
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'alpha': ALPHA,
            'n_total_records': len(df_results),
            'n_unique_datasets': df_results['dataset'].nunique(),
            'models_analyzed': list(df_results['model'].unique()),
            'configurations': list(df_results['config_label'].unique())
        }
    }

    # Save JSON results
    output_file = OUTPUT_DIR / "statistical_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Saved: {output_file}")

    # Generate summary report
    generate_summary_report(all_results, OUTPUT_DIR / "statistical_summary.txt")

    # Print key findings
    print_key_findings(all_results)

    logger.info(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    return 0


def generate_summary_report(results: Dict, output_file: Path):
    """Generate a human-readable summary report."""
    lines = []
    lines.append("=" * 80)
    lines.append("STATISTICAL ANALYSIS SUMMARY REPORT")
    lines.append("IEEE TKDE Paper - Section VI Results and Discussion")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Overall analysis
    lines.append("\n" + "-" * 80)
    lines.append("1. OVERALL PERFORMANCE ANALYSIS")
    lines.append("-" * 80)

    for config in results['overall'].get('configurations', []):
        lines.append(f"\n--- Configuration: {config['config_label']} ---")
        lines.append(f"Datasets: {config['n_datasets']}, Models: {config['n_models']}")

        # Friedman test
        friedman = config['friedman_test']
        lines.append(f"\nFriedman Test:")
        lines.append(f"  Chi-square: {friedman['statistic']:.3f}")
        lines.append(f"  p-value: {friedman['p_value']:.6f}")
        lines.append(f"  Significant: {friedman['significant']}")

        # Critical Distance
        lines.append(f"\nNemenyi Critical Distance (alpha=0.05): {config['critical_distance']:.3f}")

        # Average Rankings
        lines.append("\nAverage Rankings (lower is better):")
        sorted_ranks = sorted(config['average_rankings'].items(), key=lambda x: x[1])
        for rank, (model, avg_rank) in enumerate(sorted_ranks, 1):
            wins = config['wins_count'].get(model, 0)
            perf = config['model_summary'].get(model, {})
            gmean = perf.get('mean', 0)
            std = perf.get('std', 0)
            lines.append(f"  {rank}. {model:20s}: {avg_rank:.2f} (G-Mean: {gmean:.4f}±{std:.4f}, Wins: {wins})")

        # Pairwise tests
        if config['pairwise_tests']:
            lines.append("\nPairwise Wilcoxon Tests (EGIS vs others, Bonferroni corrected):")
            for test in config['pairwise_tests']:
                sig_marker = "*" if test['significant_bonferroni'] else ""
                lines.append(f"  {test['comparison']:30s}: p={test['raw_p_value']:.4f} "
                           f"(adj={test['adjusted_p_value']:.4f}){sig_marker}, "
                           f"Δ={test['mean_difference']:+.4f}, δ={test['cliffs_delta']:.3f} ({test['effect_interpretation']})")

    # Stratified analysis
    lines.append("\n" + "-" * 80)
    lines.append("2. PERFORMANCE BY DRIFT TYPE")
    lines.append("-" * 80)

    for drift_type, drift_data in results['stratified'].get('drift_types', {}).items():
        lines.append(f"\n--- {drift_type.upper()} ({drift_data['n_datasets']} datasets) ---")

        for config in drift_data.get('configurations', []):
            lines.append(f"\n  {config['config_label']}:")
            sorted_ranks = sorted(config['average_rankings'].items(), key=lambda x: x[1])
            for model, rank in sorted_ranks[:5]:  # Top 5
                perf = config['model_performance'].get(model, {})
                gmean = perf.get('mean', 0)
                lines.append(f"    {model:20s}: Rank={rank:.2f}, G-Mean={gmean:.4f}")

    # EGIS Penalty analysis
    lines.append("\n" + "-" * 80)
    lines.append("3. EGIS COMPLEXITY PENALTY EFFECT")
    lines.append("-" * 80)

    for comp in results['penalty_comparison'].get('comparisons', []):
        lines.append(f"\n--- Chunk Size: {comp['chunk_size']} ---")
        lines.append(f"  Without Penalty (γ=0.0): {comp['no_penalty_mean']:.4f} ± {comp['no_penalty_std']:.4f}")
        lines.append(f"  With Penalty (γ=0.1):    {comp['with_penalty_mean']:.4f} ± {comp['with_penalty_std']:.4f}")
        lines.append(f"  Difference:              {comp['mean_difference']:+.4f}")
        lines.append(f"  Wilcoxon p-value:        {comp['wilcoxon_p_value']:.4f}")
        lines.append(f"  Significant:             {comp['significant']}")
        lines.append(f"  Effect Size:             {comp['cliffs_delta']:.3f} ({comp['effect_interpretation']})")

    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info(f"Saved: {output_file}")


def print_key_findings(results: Dict):
    """Print key findings to console."""
    print("\n" + "=" * 60)
    print("KEY STATISTICAL FINDINGS")
    print("=" * 60)

    # Best configuration
    best_config = None
    best_rank = float('inf')
    for config in results['overall'].get('configurations', []):
        if 'EGIS' in config['average_rankings']:
            rank = config['average_rankings']['EGIS']
            if rank < best_rank:
                best_rank = rank
                best_config = config

    if best_config:
        print(f"\nBest EGIS Configuration: {best_config['config_label']}")
        print(f"  Average Rank: {best_rank:.2f}")
        print(f"  Friedman p-value: {best_config['friedman_test']['p_value']:.6f}")
        print(f"  Critical Distance: {best_config['critical_distance']:.3f}")

        # Significant differences
        sig_count = sum(1 for t in best_config['pairwise_tests'] if t['significant_bonferroni'])
        print(f"  Significant differences (Bonferroni): {sig_count}/{len(best_config['pairwise_tests'])}")

    # Penalty effect summary
    print("\nPenalty Effect Summary:")
    for comp in results['penalty_comparison'].get('comparisons', []):
        effect = "No significant difference" if not comp['significant'] else f"Significant (p={comp['wilcoxon_p_value']:.4f})"
        print(f"  Chunk {comp['chunk_size']}: Delta={comp['mean_difference']:+.4f}, {effect}")


if __name__ == "__main__":
    sys.exit(main())
