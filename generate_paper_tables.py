#!/usr/bin/env python3
"""
Table Generation Script for IEEE TKDE Paper - Section VI

Generates all LaTeX tables for the Results and Discussion section:
- Table VII: Summary Performance Across All Experiments
- Table VIII: Performance by Drift Type
- Table IX: Complete Model Ranking (Friedman)
- Table X: Model Rankings by Drift Type
- Table XI: Pairwise Wilcoxon Tests
- Table XII: EGIS Penalty Effect Analysis
- Table XIII: Rule Complexity Comparison
- Table XIV: Transition Metrics by Drift Type

Output: paper/tables/*.tex

Author: Automated Analysis
Date: 2026-01-27
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

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
OUTPUT_DIR = Path("paper/tables")

# Datasets excluded from all analyses
EXCLUDED_DATASETS = ['CovType', 'IntelLabSensors', 'PokerHand', 'Shuttle']

# Model display names
MODEL_DISPLAY_NAMES = {
    'EGIS': 'EGIS',
    'ARF': 'ARF',
    'SRP': 'SRP',
    'HAT': 'HAT',
    'ROSE_Original': 'ROSE',
    'ROSE_ChunkEval': 'ROSE\\_CE',
    'ACDWM': 'ACDWM',
    'ERulesD2S': 'ERulesD2S'
}

# Preferred model order
MODELS_ORDER = ['EGIS', 'ARF', 'SRP', 'ROSE_Original', 'ROSE_ChunkEval', 'HAT', 'ACDWM', 'ERulesD2S']

# Configuration display names
CONFIG_DISPLAY = {
    'EXP-500-NP': 'EXP-500',
    'EXP-500-P': 'EXP-500-P',
    'EXP-1000-NP': 'EXP-1000',
    'EXP-1000-P': 'EXP-1000-P',
    'EXP-2000-NP': 'EXP-2000',
    'EXP-2000-P': 'EXP-2000-P'
}

# =============================================================================
# LATEX GENERATION FUNCTIONS
# =============================================================================

def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    return text.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')


def format_value(value: float, fmt: str = '.3f') -> str:
    """Format a numeric value for LaTeX."""
    if pd.isna(value):
        return '--'
    if value == 0:
        return '0.000'
    return f'{value:{fmt}}'


def generate_table_header(caption: str, label: str, columns: List[str]) -> str:
    """Generate LaTeX table header."""
    col_spec = 'l' + 'c' * (len(columns) - 1)

    header = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\footnotesize
\\begin{{tabular}}{{{col_spec}}}
\\toprule
"""
    return header


def generate_table_footer() -> str:
    """Generate LaTeX table footer."""
    return """\\bottomrule
\\end{tabular}
\\end{table}
"""


# =============================================================================
# TABLE VII: SUMMARY PERFORMANCE
# =============================================================================

def _compute_summary_block(df_results: pd.DataFrame, models: List[str], configs: List[str]) -> List[dict]:
    """Compute summary data rows for a given set of models and filtered results."""
    summary_data = []
    for model in models:
        row = {'Model': MODEL_DISPLAY_NAMES.get(model, model), 'model_key': model}
        for config in configs:
            config_model = df_results[(df_results['config_label'] == config) &
                                     (df_results['model'] == model)]
            if len(config_model) > 0:
                mean_val = config_model['gmean_mean'].mean()
                std_val = config_model['gmean_mean'].std()
                row[f'{config}_mean'] = mean_val
                row[f'{config}_std'] = std_val
            else:
                row[f'{config}_mean'] = np.nan
                row[f'{config}_std'] = np.nan
        # average performance across configs
        vals = [row.get(f'{c}_mean', np.nan) for c in configs]
        valid_vals = [v for v in vals if not pd.isna(v)]
        row['avg_perf'] = np.mean(valid_vals) if valid_vals else np.nan
        summary_data.append(row)
    summary_data.sort(key=lambda x: -x.get('avg_perf', 0))
    return summary_data


def _render_summary_rows(summary_data: List[dict], configs: List[str]) -> str:
    """Render LaTeX rows for summary data."""
    latex = ""
    for row in summary_data:
        model_name = row['Model']
        if row.get('model_key') == 'EGIS':
            model_name = '\\textbf{EGIS}'

        line_parts = [model_name]
        for config in configs:
            mean_val = row.get(f'{config}_mean', np.nan)
            std_val = row.get(f'{config}_std', np.nan)
            if pd.isna(mean_val):
                line_parts.extend(['--', '--'])
            else:
                line_parts.extend([f'{mean_val:.3f}', f'{std_val:.3f}'])

        latex += ' & '.join(line_parts) + ' \\\\\n'
    return latex


def generate_table_summary_performance(df_results: pd.DataFrame) -> str:
    """Generate Table VII: Summary Performance Across All Experiments."""
    logger.info("Generating Table VII: Summary Performance")

    MULTICLASS_DATASETS = ['LED_Abrupt_Simple', 'LED_Gradual_Simple', 'LED_Stationary',
                           'WAVEFORM_Abrupt_Simple', 'WAVEFORM_Gradual_Simple', 'WAVEFORM_Stationary']
    MULTICLASS_MODELS = ['EGIS', 'ARF', 'SRP', 'HAT', 'ROSE_Original', 'ERulesD2S']

    configs = ['EXP-500-NP', 'EXP-1000-NP', 'EXP-2000-NP']

    # Binary block: all datasets except multiclass
    df_binary = df_results[~df_results['dataset'].isin(MULTICLASS_DATASETS)]
    binary_models = [m for m in MODELS_ORDER if m in df_binary['model'].unique()]
    binary_data = _compute_summary_block(df_binary, binary_models, configs)

    # All datasets block: only models that support multiclass (no ACDWM, no CDCMS)
    all_models = [m for m in MULTICLASS_MODELS if m in df_results['model'].unique()]
    all_data = _compute_summary_block(df_results, all_models, configs)

    # Generate LaTeX
    latex = """\\begin{table}[htbp]
\\centering
\\caption{Summary Performance Across All Experiments (G-Mean)}
\\label{tab:summary_all}
\\footnotesize
\\begin{tabular}{lcccccc}
\\toprule
& \\multicolumn{2}{c}{\\textbf{EXP-500}} & \\multicolumn{2}{c}{\\textbf{EXP-1000}} & \\multicolumn{2}{c}{\\textbf{EXP-2000}} \\\\
\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}
\\textbf{Model} & Mean & Std & Mean & Std & Mean & Std \\\\
\\midrule
\\textit{Binary only (n=42):} & & & & & & \\\\
"""

    latex += _render_summary_rows(binary_data, configs)

    latex += """\\midrule
\\textit{All datasets (n=48):} & & & & & & \\\\
"""

    latex += _render_summary_rows(all_data, configs)

    latex += generate_table_footer()
    return latex


# =============================================================================
# TABLE VIII: PERFORMANCE BY DRIFT TYPE
# =============================================================================

def generate_table_by_drift_type(df_results: pd.DataFrame) -> str:
    """Generate Table VIII: Performance by Drift Type."""
    logger.info("Generating Table VIII: Performance by Drift Type")

    drift_types = ['abrupt', 'gradual', 'noisy', 'stationary', 'real']
    models = [m for m in MODELS_ORDER if m in df_results['model'].unique()]

    # Use a single configuration for clarity (EXP-1000-NP)
    config = 'EXP-1000-NP'
    config_df = df_results[df_results['config_label'] == config]

    latex = """\\begin{table}[htbp]
\\centering
\\caption{Performance by Drift Type (G-Mean, EXP-1000 Configuration)}
\\label{tab:drift_performance}
\\footnotesize
\\begin{tabular}{l"""

    # Add columns for each model
    latex += 'c' * len(models) + '}\n\\toprule\n\\textbf{Drift Type}'

    for model in models:
        latex += f' & \\textbf{{{MODEL_DISPLAY_NAMES.get(model, model)}}}'
    latex += ' \\\\\n\\midrule\n'

    # Calculate mean G-Mean per drift type per model
    for drift_type in drift_types:
        drift_df = config_df[config_df['drift_type'] == drift_type]
        n_datasets = drift_df['dataset'].nunique()
        drift_label = drift_type.capitalize()
        if n_datasets > 0:
            drift_label += f' ({n_datasets})'

        line_parts = [drift_label]
        for model in models:
            model_df = drift_df[drift_df['model'] == model]
            if len(model_df) > 0:
                mean_val = model_df['gmean_mean'].mean()
                line_parts.append(f'{mean_val:.3f}')
            else:
                line_parts.append('--')

        latex += ' & '.join(line_parts) + ' \\\\\n'

    latex += generate_table_footer()
    return latex


# =============================================================================
# TABLE IX: COMPLETE MODEL RANKING
# =============================================================================

def generate_table_complete_ranking(df_results: pd.DataFrame, stats_results: Dict) -> str:
    """Generate Table IX: Complete Model Ranking Based on Friedman Test."""
    logger.info("Generating Table IX: Complete Model Ranking")

    # Find the best configuration for ranking
    best_config = None
    for config in stats_results.get('overall', {}).get('configurations', []):
        if config['config_label'] == 'EXP-1000-NP':
            best_config = config
            break

    if not best_config:
        # Use first available
        configs = stats_results.get('overall', {}).get('configurations', [])
        if configs:
            best_config = configs[0]

    if not best_config:
        logger.warning("No configuration data found for ranking table")
        return "% No data available for ranking table\n"

    # Get rankings and sort
    avg_ranks = best_config.get('average_rankings', {})
    model_summary = best_config.get('model_summary', {})
    wins_count = best_config.get('wins_count', {})
    friedman = best_config.get('friedman_test', {})

    sorted_models = sorted(avg_ranks.items(), key=lambda x: x[1])

    latex = """\\begin{table}[htbp]
\\centering
\\caption{Complete Model Ranking Based on Friedman Test}
\\label{tab:complete_ranking}
\\footnotesize
\\begin{tabular}{clcccc}
\\toprule
\\textbf{Rank} & \\textbf{Model} & \\textbf{Avg Rank} & \\textbf{Mean G-Mean} & \\textbf{Std} & \\textbf{Wins} \\\\
\\midrule
"""

    for idx, (model, rank) in enumerate(sorted_models, 1):
        model_name = MODEL_DISPLAY_NAMES.get(model, model)
        if model == 'EGIS':
            model_name = '\\textbf{EGIS}'

        perf = model_summary.get(model, {})
        mean_gmean = perf.get('mean', 0)
        std_gmean = perf.get('std', 0)
        wins = wins_count.get(model, 0)

        latex += f'{idx} & {model_name} & {rank:.2f} & {mean_gmean:.3f} & {std_gmean:.3f} & {wins} \\\\\n'

    # Add Friedman test info
    chi_sq = friedman.get('statistic', 0)
    p_val = friedman.get('p_value', 1)
    df = friedman.get('df', 0)
    cd = best_config.get('critical_distance', 0)

    latex += f"""\\midrule
\\multicolumn{{6}}{{l}}{{\\textit{{Friedman Test: $\\chi^2$({df}) = {chi_sq:.1f}, p < 0.001}}}} \\\\
\\multicolumn{{6}}{{l}}{{\\textit{{Critical Distance (Nemenyi, $\\alpha$=0.05): CD = {cd:.2f}}}}} \\\\
"""

    latex += generate_table_footer()
    return latex


# =============================================================================
# TABLE XI: PAIRWISE WILCOXON TESTS
# =============================================================================

def generate_table_wilcoxon(stats_results: Dict) -> str:
    """Generate Table XI: Pairwise Wilcoxon Tests with Bonferroni Correction."""
    logger.info("Generating Table XI: Pairwise Wilcoxon Tests")

    # Find configuration with pairwise tests
    best_config = None
    for config in stats_results.get('overall', {}).get('configurations', []):
        if config.get('pairwise_tests'):
            best_config = config
            break

    if not best_config or not best_config.get('pairwise_tests'):
        logger.warning("No pairwise test data available")
        return "% No pairwise test data available\n"

    tests = best_config['pairwise_tests']

    latex = """\\begin{table}[htbp]
\\centering
\\caption{Statistical Significance Tests (Pairwise Wilcoxon with Bonferroni)}
\\label{tab:stat_tests}
\\footnotesize
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Comparison} & \\textbf{p-value} & \\textbf{Adj. p} & \\textbf{Sig.} & \\textbf{Mean $\\Delta$} & \\textbf{Effect} \\\\
\\midrule
"""

    for test in tests:
        comparison = test['comparison'].replace('_', '\\_')
        p_val = test['raw_p_value']
        adj_p = test['adjusted_p_value']
        sig = 'Yes' if test['significant_bonferroni'] else 'No'
        mean_diff = test['mean_difference']
        effect = f"{test['cliffs_delta']:.2f} ({test['effect_interpretation'][:1]})"

        # Format p-value
        if p_val < 0.0001:
            p_str = '<0.0001'
        else:
            p_str = f'{p_val:.4f}'

        if adj_p < 0.0001:
            adj_str = '<0.0001'
        else:
            adj_str = f'{adj_p:.4f}'

        latex += f'{comparison} & {p_str} & {adj_str} & {sig} & {mean_diff:+.3f} & {effect} \\\\\n'

    latex += generate_table_footer()
    return latex


# =============================================================================
# TABLE XII: EGIS PENALTY EFFECT
# =============================================================================

def generate_table_penalty_effect(stats_results: Dict) -> str:
    """Generate Table XII: EGIS Penalty Effect Analysis."""
    logger.info("Generating Table XII: EGIS Penalty Effect")

    comparisons = stats_results.get('penalty_comparison', {}).get('comparisons', [])

    if not comparisons:
        logger.warning("No penalty comparison data available")
        return "% No penalty comparison data available\n"

    latex = """\\begin{table}[htbp]
\\centering
\\caption{EGIS Complexity Penalty Effect Analysis}
\\label{tab:penalty_effect}
\\footnotesize
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Chunk} & \\textbf{No Penalty} & \\textbf{With Penalty} & \\textbf{$\\Delta$} & \\textbf{p-value} & \\textbf{Sig.} & \\textbf{Effect} \\\\
\\textbf{Size} & \\textbf{($\\gamma$=0.0)} & \\textbf{($\\gamma$=0.1)} & \\textbf{G-Mean} & & & \\textbf{Size} \\\\
\\midrule
"""

    for comp in comparisons:
        chunk_size = comp['chunk_size']
        no_pen = f"{comp['no_penalty_mean']:.3f}±{comp['no_penalty_std']:.3f}"
        with_pen = f"{comp['with_penalty_mean']:.3f}±{comp['with_penalty_std']:.3f}"
        diff = comp['mean_difference']
        p_val = comp['wilcoxon_p_value']
        sig = 'Yes' if comp['significant'] else 'No'
        effect = f"{comp['cliffs_delta']:.2f}"

        latex += f'{chunk_size} & {no_pen} & {with_pen} & {diff:+.3f} & {p_val:.3f} & {sig} & {effect} \\\\\n'

    latex += generate_table_footer()
    return latex


# =============================================================================
# TABLE XIII: RULE COMPLEXITY
# =============================================================================

def generate_table_rule_complexity(df_rules: pd.DataFrame) -> str:
    """Generate Table XIII: Rule Complexity Comparison."""
    logger.info("Generating Table XIII: Rule Complexity")

    if df_rules.empty:
        logger.warning("No rule complexity data available")
        return "% No rule complexity data available\n"

    # Calculate statistics per configuration
    complexity_stats = df_rules.groupby('config_label').agg({
        'n_rules': ['mean', 'std'],
        'total_conditions': ['mean', 'std'],
        'avg_conditions_per_rule': ['mean', 'std'],
        'total_and_ops': 'mean',
        'total_or_ops': 'mean'
    }).round(2)

    latex = """\\begin{table}[htbp]
\\centering
\\caption{EGIS Rule Complexity by Configuration}
\\label{tab:complexity}
\\footnotesize
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Config} & \\textbf{Avg Rules} & \\textbf{Total Cond.} & \\textbf{Cond/Rule} & \\textbf{AND ops} & \\textbf{OR ops} \\\\
\\midrule
"""

    for config in ['EXP-500-NP', 'EXP-1000-NP', 'EXP-2000-NP', 'EXP-500-P', 'EXP-1000-P', 'EXP-2000-P']:
        if config not in complexity_stats.index:
            continue

        row = complexity_stats.loc[config]
        n_rules = f"{row[('n_rules', 'mean')]:.1f}±{row[('n_rules', 'std')]:.1f}"
        total_cond = f"{row[('total_conditions', 'mean')]:.1f}±{row[('total_conditions', 'std')]:.1f}"
        cond_rule = f"{row[('avg_conditions_per_rule', 'mean')]:.2f}±{row[('avg_conditions_per_rule', 'std')]:.2f}"
        and_ops = f"{row[('total_and_ops', 'mean')]:.1f}"
        or_ops = f"{row[('total_or_ops', 'mean')]:.1f}"

        config_display = CONFIG_DISPLAY.get(config, config)
        latex += f'{config_display} & {n_rules} & {total_cond} & {cond_rule} & {and_ops} & {or_ops} \\\\\n'

    latex += generate_table_footer()
    return latex


# =============================================================================
# TABLE XIV: TRANSITION METRICS
# =============================================================================

def generate_table_transition_metrics(df_transitions: pd.DataFrame) -> str:
    """Generate Table XIV: Transition Metrics by Drift Type."""
    logger.info("Generating Table XIV: Transition Metrics")

    if df_transitions.empty:
        logger.warning("No transition metrics data available")
        return "% No transition metrics data available\n"

    # Calculate statistics per drift type
    trans_stats = df_transitions.groupby('drift_type').agg({
        'TCS': ['mean', 'std'],
        'RIR': ['mean', 'std'],
        'AMS': ['mean', 'std']
    }).round(4)

    latex = """\\begin{table}[htbp]
\\centering
\\caption{Transition Metrics by Drift Type (EGIS)}
\\label{tab:transitions}
\\footnotesize
\\begin{tabular}{lccc}
\\toprule
\\textbf{Drift Type} & \\textbf{TCS} & \\textbf{RIR} & \\textbf{AMS} \\\\
\\midrule
"""

    for drift_type in ['abrupt', 'gradual', 'noisy', 'stationary', 'real']:
        if drift_type not in trans_stats.index:
            continue

        row = trans_stats.loc[drift_type]
        tcs = f"{row[('TCS', 'mean')]:.3f}±{row[('TCS', 'std')]:.3f}"
        rir = f"{row[('RIR', 'mean')]:.3f}±{row[('RIR', 'std')]:.3f}"
        ams = f"{row[('AMS', 'mean')]:.3f}±{row[('AMS', 'std')]:.3f}"

        latex += f'{drift_type.capitalize()} & {tcs} & {rir} & {ams} \\\\\n'

    latex += generate_table_footer()
    return latex


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("TABLE GENERATION FOR IEEE TKDE PAPER - SECTION VI")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data files
    results_file = INPUT_DIR / "consolidated_results.csv"
    rules_file = INPUT_DIR / "egis_rules_per_chunk.csv"
    transitions_file = INPUT_DIR / "egis_transition_metrics.csv"
    stats_file = INPUT_DIR / "statistical_results.json"

    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.error("Please run collect_results_for_paper.py first")
        return 1

    df_results = pd.read_csv(results_file)
    df_results = df_results[~df_results['dataset'].isin(EXCLUDED_DATASETS)]
    logger.info(f"Loaded results: {len(df_results)} records (after excluding {EXCLUDED_DATASETS})")

    df_rules = pd.DataFrame()
    if rules_file.exists():
        df_rules = pd.read_csv(rules_file)
        df_rules = df_rules[~df_rules['dataset'].isin(EXCLUDED_DATASETS)]
        logger.info(f"Loaded rules: {len(df_rules)} records")

    df_transitions = pd.DataFrame()
    if transitions_file.exists():
        df_transitions = pd.read_csv(transitions_file)
        df_transitions = df_transitions[~df_transitions['dataset'].isin(EXCLUDED_DATASETS)]
        logger.info(f"Loaded transitions: {len(df_transitions)} records")

    stats_results = {}
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats_results = json.load(f)
        logger.info("Loaded statistical results")

    # Generate tables
    tables = {
        'table_vii_summary': generate_table_summary_performance(df_results),
        'table_viii_drift': generate_table_by_drift_type(df_results),
        'table_ix_ranking': generate_table_complete_ranking(df_results, stats_results),
        'table_xi_wilcoxon': generate_table_wilcoxon(stats_results),
        'table_xii_penalty': generate_table_penalty_effect(stats_results),
        'table_xiii_complexity': generate_table_rule_complexity(df_rules),
        'table_xiv_transitions': generate_table_transition_metrics(df_transitions)
    }

    # Save tables
    for name, latex in tables.items():
        output_file = OUTPUT_DIR / f"{name}.tex"
        with open(output_file, 'w') as f:
            f.write(f"% Auto-generated table: {name}\n")
            f.write(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(latex)
        logger.info(f"Saved: {output_file}")

    # Generate combined tables file
    combined_file = OUTPUT_DIR / "all_tables.tex"
    with open(combined_file, 'w') as f:
        f.write("% Auto-generated tables for IEEE TKDE Paper Section VI\n")
        f.write(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for name, latex in tables.items():
            f.write(f"% === {name} ===\n")
            f.write(latex)
            f.write("\n\n")
    logger.info(f"Saved combined: {combined_file}")

    logger.info(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"Generated {len(tables)} tables")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
