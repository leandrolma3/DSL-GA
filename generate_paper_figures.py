#!/usr/bin/env python3
"""
Figure Generation Script for IEEE TKDE Paper - Section VI

Generates all figures for the Results and Discussion section:
- Figure 3: Critical Difference Diagram
- Figure 4: Transition Metrics Evolution (Example Dataset)
- Figure 5: Performance Boxplots by Model
- Figure 6: Rule Evolution Over Stream
- Figure 7: Heatmap - Performance by Drift Type

Output: paper/figures/*.pdf

Author: Automated Analysis
Date: 2026-01-27
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Figures will not be generated.")

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
OUTPUT_DIR = Path("paper/figures")

# Figure settings for IEEE papers
FIGURE_SETTINGS = {
    'figure.figsize': (7, 5),  # Single column width
    'figure.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
}

# Model display names and colors
MODEL_DISPLAY_NAMES = {
    'EGIS': 'EGIS',
    'ARF': 'ARF',
    'SRP': 'SRP',
    'HAT': 'HAT',
    'ROSE': 'ROSE',
    'ACDWM': 'ACDWM',
    'ERulesD2S': 'eRulesD2S',
    'CDCMS': 'CDCMS'
}

MODEL_COLORS = {
    'EGIS': '#2166ac',      # Blue (highlight)
    'ARF': '#1a9850',       # Green
    'SRP': '#91cf60',       # Light green
    'HAT': '#fee08b',       # Yellow
    'ROSE': '#d73027',      # Red
    'ACDWM': '#9970ab',     # Purple
    'ERulesD2S': '#636363', # Gray
    'CDCMS': '#fc8d59'      # Orange
}

MODELS_ORDER = ['EGIS', 'ARF', 'SRP', 'HAT', 'ROSE', 'ACDWM', 'ERulesD2S', 'CDCMS']

# =============================================================================
# FIGURE 3: CRITICAL DIFFERENCE DIAGRAM
# =============================================================================

def _find_best_egis_config(configurations: List[Dict]) -> Optional[Dict]:
    """Find the configuration with the lowest EGIS rank among those with ranking data.

    Args:
        configurations: List of configuration dicts from statistical_results.json.

    Returns:
        The config dict with the best (lowest) EGIS average rank, or None.
    """
    best_config = None
    best_egis_rank = float('inf')
    for config in configurations:
        avg_ranks = config.get('average_rankings')
        if not avg_ranks:
            continue
        egis_rank = avg_ranks.get('EGIS', float('inf'))
        if egis_rank < best_egis_rank:
            best_egis_rank = egis_rank
            best_config = config
    return best_config


def _render_cd_diagram(config: Dict, output_file: Path, title_suffix: str = ""):
    """Render and save a single Critical Difference diagram from a config dict.

    Args:
        config: A configuration dict containing average_rankings, critical_distance, n_datasets.
        output_file: Path to write the PDF figure.
        title_suffix: Extra text appended to the diagram title (e.g. " - Binary Only").
    """
    avg_ranks = config['average_rankings']
    cd = config.get('critical_distance', 1.0)
    n_datasets = config.get('n_datasets', 0)

    # Sort models by rank
    sorted_models = sorted(avg_ranks.items(), key=lambda x: x[1])
    models = [m for m, r in sorted_models]
    ranks = [r for m, r in sorted_models]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot parameters
    n_models = len(models)
    y_positions = np.arange(n_models)
    max_rank = max(ranks) + 1
    min_rank = min(ranks) - 0.5

    # Draw horizontal lines for each model
    for i, (model, rank) in enumerate(sorted_models):
        color = MODEL_COLORS.get(model, '#333333')
        display_name = MODEL_DISPLAY_NAMES.get(model, model)

        # Model name on left
        ax.text(min_rank - 0.3, i, display_name, ha='right', va='center',
               fontsize=10, fontweight='bold' if model == 'EGIS' else 'normal')

        # Rank point
        ax.scatter(rank, i, s=100, c=color, zorder=5, edgecolors='black', linewidths=1)

        # Rank value on right
        ax.text(rank + 0.2, i, f'{rank:.2f}', ha='left', va='center', fontsize=9)

    # Draw CD bar at top
    cd_y = n_models - 0.3
    ax.plot([1, 1 + cd], [cd_y, cd_y], 'k-', linewidth=2)
    ax.plot([1, 1], [cd_y - 0.1, cd_y + 0.1], 'k-', linewidth=2)
    ax.plot([1 + cd, 1 + cd], [cd_y - 0.1, cd_y + 0.1], 'k-', linewidth=2)
    ax.text((1 + 1 + cd) / 2, cd_y + 0.3, f'CD = {cd:.2f}', ha='center', fontsize=9)

    # Draw connections between non-significantly different models
    # Models within CD of each other are connected
    connections = []
    for i in range(len(sorted_models)):
        for j in range(i + 1, len(sorted_models)):
            model_i, rank_i = sorted_models[i]
            model_j, rank_j = sorted_models[j]
            if abs(rank_i - rank_j) < cd:
                connections.append((i, j, rank_i, rank_j))

    # Draw connection lines
    for i, j, rank_i, rank_j in connections:
        y_offset = -0.5 - (j - i) * 0.15
        ax.plot([rank_i, rank_j], [y_offset, y_offset], 'k-', linewidth=2, alpha=0.5)
        ax.plot([rank_i, rank_i], [i, y_offset], 'k--', linewidth=0.5, alpha=0.3)
        ax.plot([rank_j, rank_j], [j, y_offset], 'k--', linewidth=0.5, alpha=0.3)

    # Format axes
    ax.set_xlim(min_rank - 2, max_rank + 0.5)
    ax.set_ylim(-1.5, n_models + 0.5)
    ax.set_xlabel('Average Rank', fontsize=11)
    ax.set_yticks([])

    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    ax.axhline(y=-0.5, color='gray', linestyle='-', linewidth=0.5)

    # Title
    title = f'Critical Difference Diagram (n={n_datasets} datasets)'
    if title_suffix:
        title += f' {title_suffix}'
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_file}")


def generate_critical_difference_diagram(stats_results: Dict, output_file: Path):
    """Generate Critical Difference diagrams for Nemenyi post-hoc test.

    Produces up to three diagrams:
      - fig_critical_difference.pdf       : binary-only analysis (PRIMARY)
      - fig_critical_difference_all48.pdf  : all-48 datasets (complementary)
      - fig_critical_difference_multiclass.pdf : multiclass-only (if available)

    The primary diagram uses binary_only data. The best EGIS configuration
    is selected as the one with the lowest EGIS average rank.
    """
    logger.info("Generating Figure 3: Critical Difference Diagrams")

    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available")
        return

    output_dir = output_file.parent
    generated_any = False

    # ---- PRIMARY: binary-only CD diagram ----
    binary_configs = stats_results.get('binary_only', {}).get('configurations', [])
    if binary_configs:
        best_binary = _find_best_egis_config(binary_configs)
        if best_binary:
            binary_output = output_dir / 'fig_critical_difference.pdf'
            _render_cd_diagram(best_binary, binary_output, title_suffix="- Binary Only")
            generated_any = True
        else:
            logger.warning("No ranking data in binary_only configurations")
    else:
        logger.warning("No binary_only configurations found in statistical results")

    # ---- COMPLEMENTARY: all-48 datasets CD diagram ----
    overall_configs = stats_results.get('overall', {}).get('configurations', [])
    if overall_configs:
        best_overall = _find_best_egis_config(overall_configs)
        if best_overall:
            overall_output = output_dir / 'fig_critical_difference_all48.pdf'
            _render_cd_diagram(best_overall, overall_output, title_suffix="- All Datasets")
            generated_any = True
        else:
            logger.warning("No ranking data in overall configurations")
    else:
        logger.warning("No overall configurations found in statistical results")

    # ---- OPTIONAL: multiclass-only CD diagram ----
    multiclass_configs = stats_results.get('multiclass_only', {}).get('configurations', [])
    if multiclass_configs:
        best_multiclass = _find_best_egis_config(multiclass_configs)
        if best_multiclass:
            multiclass_output = output_dir / 'fig_critical_difference_multiclass.pdf'
            _render_cd_diagram(best_multiclass, multiclass_output, title_suffix="- Multiclass Only")
            generated_any = True
        else:
            logger.warning("No ranking data in multiclass_only configurations")
    else:
        logger.info("No multiclass_only section in statistical results (skipping)")

    if not generated_any:
        logger.warning("No CD diagrams were generated (no ranking data available)")


# =============================================================================
# FIGURE 4: TRANSITION METRICS EVOLUTION
# =============================================================================

def generate_transition_evolution_plot(df_transitions: pd.DataFrame, output_file: Path):
    """Generate transition metrics evolution for an example dataset."""
    logger.info("Generating Figure 4: Transition Metrics Evolution")

    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available")
        return

    if df_transitions.empty:
        logger.warning("No transition data available")
        return

    # Select an example dataset with abrupt drift
    example_datasets = df_transitions[df_transitions['drift_type'] == 'abrupt']['dataset'].unique()
    if len(example_datasets) == 0:
        example_datasets = df_transitions['dataset'].unique()

    if len(example_datasets) == 0:
        logger.warning("No datasets available for transition plot")
        return

    # Use STAGGER_Abrupt_Chain if available, else first available
    dataset_name = 'STAGGER_Abrupt_Chain'
    if dataset_name not in example_datasets:
        dataset_name = example_datasets[0]

    # Filter data for this dataset
    dataset_df = df_transitions[df_transitions['dataset'] == dataset_name].copy()
    dataset_df = dataset_df.sort_values('chunk_from')

    if len(dataset_df) < 2:
        logger.warning(f"Insufficient data for dataset {dataset_name}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    chunks = dataset_df['chunk_from'].values + 0.5  # Center between chunks
    tcs = dataset_df['TCS'].values
    rir = dataset_df['RIR'].values
    ams = dataset_df['AMS'].values

    # Plot metrics
    ax.plot(chunks, tcs, 'b-o', label='TCS', linewidth=2, markersize=6)
    ax.plot(chunks, rir, 'r-s', label='RIR', linewidth=2, markersize=5)
    ax.plot(chunks, ams, 'g-^', label='AMS', linewidth=2, markersize=5)

    # Mark potential drift points (high TCS)
    drift_threshold = 0.35
    drift_points = chunks[tcs > drift_threshold]
    for dp in drift_points:
        ax.axvline(x=dp, color='gray', linestyle='--', alpha=0.5)
        ax.annotate('Drift', xy=(dp, max(tcs)), xytext=(dp + 0.3, max(tcs) + 0.05),
                   fontsize=8, alpha=0.7)

    # Format
    ax.set_xlabel('Chunk Transition', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title(f'Transition Metrics Evolution ({dataset_name.replace("_", " ")})',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_file}")


# =============================================================================
# FIGURE 5: PERFORMANCE BOXPLOTS
# =============================================================================

def generate_performance_boxplots(df_results: pd.DataFrame, output_file: Path):
    """Generate boxplots showing G-Mean distribution per model."""
    logger.info("Generating Figure 5: Performance Boxplots")

    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available")
        return

    # Use EXP-1000-NP configuration
    config_df = df_results[df_results['config_label'] == 'EXP-1000-NP']

    if config_df.empty:
        # Fall back to any available
        config_df = df_results

    # Get models present in data
    models = [m for m in MODELS_ORDER if m in config_df['model'].unique()]

    # Prepare data for boxplot
    data_for_plot = []
    labels = []
    colors = []

    for model in models:
        model_df = config_df[config_df['model'] == model]
        values = model_df['gmean_mean'].dropna().values
        if len(values) > 0:
            data_for_plot.append(values)
            labels.append(MODEL_DISPLAY_NAMES.get(model, model))
            colors.append(MODEL_COLORS.get(model, '#333333'))

    if len(data_for_plot) == 0:
        logger.warning("No data available for boxplots")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create boxplot
    bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Style
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='black')
    plt.setp(bp['medians'], linewidth=2)

    # Add horizontal line for EGIS mean
    egis_idx = labels.index('EGIS') if 'EGIS' in labels else None
    if egis_idx is not None:
        egis_mean = np.mean(data_for_plot[egis_idx])
        ax.axhline(y=egis_mean, color=MODEL_COLORS['EGIS'], linestyle='--',
                  alpha=0.7, label=f'EGIS mean ({egis_mean:.3f})')

    # Format
    ax.set_ylabel('G-Mean', fontsize=11)
    ax.set_xlabel('Model', fontsize=11)
    ax.set_title('Performance Distribution by Model (EXP-1000 Configuration)',
                fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)

    if egis_idx is not None:
        ax.legend(loc='lower left')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_file}")


# =============================================================================
# FIGURE 6: RULE EVOLUTION
# =============================================================================

def generate_rule_evolution_plot(df_rules: pd.DataFrame, output_file: Path):
    """Generate rule evolution plot showing complexity over chunks."""
    logger.info("Generating Figure 6: Rule Evolution")

    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available")
        return

    if df_rules.empty:
        logger.warning("No rule data available")
        return

    # Select example dataset
    example_datasets = df_rules[df_rules['drift_type'] == 'gradual']['dataset'].unique()
    if len(example_datasets) == 0:
        example_datasets = df_rules['dataset'].unique()

    if len(example_datasets) == 0:
        logger.warning("No datasets available for rule evolution plot")
        return

    dataset_name = example_datasets[0]

    # Filter data
    dataset_df = df_rules[df_rules['dataset'] == dataset_name].copy()
    dataset_df = dataset_df.sort_values('chunk')

    if len(dataset_df) < 3:
        logger.warning(f"Insufficient data for dataset {dataset_name}")
        return

    # Create figure with dual y-axis
    fig, ax1 = plt.subplots(figsize=(8, 5))

    chunks = dataset_df['chunk'].values
    n_rules = dataset_df['n_rules'].values
    cond_per_rule = dataset_df['avg_conditions_per_rule'].values

    # Plot number of rules (left y-axis)
    color1 = '#2166ac'
    ax1.set_xlabel('Chunk', fontsize=11)
    ax1.set_ylabel('Number of Rules', color=color1, fontsize=11)
    line1 = ax1.plot(chunks, n_rules, color=color1, marker='o', linewidth=2,
                    label='Number of Rules')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, max(n_rules) * 1.2)

    # Plot conditions per rule (right y-axis)
    ax2 = ax1.twinx()
    color2 = '#d73027'
    ax2.set_ylabel('Conditions per Rule', color=color2, fontsize=11)
    line2 = ax2.plot(chunks, cond_per_rule, color=color2, marker='s', linewidth=2,
                    linestyle='--', label='Conditions/Rule')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(cond_per_rule) * 1.2)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    ax1.set_title(f'Rule Evolution Over Stream ({dataset_name.replace("_", " ")})',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_file}")


# =============================================================================
# FIGURE 7: HEATMAP BY DRIFT TYPE
# =============================================================================

def generate_drift_heatmap(df_results: pd.DataFrame, output_file: Path):
    """Generate heatmap showing performance by drift type and model."""
    logger.info("Generating Figure 7: Drift Type Heatmap")

    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available")
        return

    # Use EXP-1000-NP configuration
    config_df = df_results[df_results['config_label'] == 'EXP-1000-NP']

    if config_df.empty:
        config_df = df_results

    # Calculate mean G-Mean per drift type per model
    pivot = config_df.pivot_table(
        index='drift_type',
        columns='model',
        values='gmean_mean',
        aggfunc='mean'
    )

    # Reorder columns and rows
    models = [m for m in MODELS_ORDER if m in pivot.columns]
    drift_order = ['abrupt', 'gradual', 'noisy', 'stationary', 'real']
    drifts = [d for d in drift_order if d in pivot.index]

    pivot = pivot.loc[drifts, models]

    # Rename for display
    pivot.columns = [MODEL_DISPLAY_NAMES.get(m, m) for m in pivot.columns]
    pivot.index = [d.capitalize() for d in pivot.index]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.6,
               vmin=0, vmax=1, ax=ax, linewidths=0.5, cbar_kws={'label': 'G-Mean'})

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Drift Type', fontsize=11)
    ax.set_title('Performance by Drift Type and Model (G-Mean)',
                fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_file}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("FIGURE GENERATION FOR IEEE TKDE PAPER - SECTION VI")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not PLOTTING_AVAILABLE:
        logger.error("Matplotlib/Seaborn not available. Cannot generate figures.")
        return 1

    # Apply plotting settings
    plt.rcParams.update(FIGURE_SETTINGS)

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
    logger.info(f"Loaded results: {len(df_results)} records")

    df_rules = pd.DataFrame()
    if rules_file.exists():
        df_rules = pd.read_csv(rules_file)
        logger.info(f"Loaded rules: {len(df_rules)} records")

    df_transitions = pd.DataFrame()
    if transitions_file.exists():
        df_transitions = pd.read_csv(transitions_file)
        logger.info(f"Loaded transitions: {len(df_transitions)} records")

    stats_results = {}
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats_results = json.load(f)
        logger.info("Loaded statistical results")

    # Generate figures
    figures = [
        ('fig3_critical_difference_diagrams', lambda: generate_critical_difference_diagram(
            stats_results, OUTPUT_DIR / 'fig_critical_difference.pdf')),
        ('fig4_transition_evolution', lambda: generate_transition_evolution_plot(
            df_transitions, OUTPUT_DIR / 'fig4_transition_evolution.pdf')),
        ('fig5_performance_boxplots', lambda: generate_performance_boxplots(
            df_results, OUTPUT_DIR / 'fig5_performance_boxplots.pdf')),
        ('fig6_rule_evolution', lambda: generate_rule_evolution_plot(
            df_rules, OUTPUT_DIR / 'fig6_rule_evolution.pdf')),
        ('fig7_drift_heatmap', lambda: generate_drift_heatmap(
            df_results, OUTPUT_DIR / 'fig7_drift_heatmap.pdf'))
    ]

    for name, gen_func in figures:
        try:
            gen_func()
        except Exception as e:
            logger.error(f"Failed to generate {name}: {e}")

    logger.info(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
