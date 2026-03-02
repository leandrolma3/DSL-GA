# generate_paper_transition_figures.py
# Generates figures for the IEEE TKDE paper from consolidated transition metrics:
# - TCS Time-Series Plot by drift type
# - RIR vs AMS Scatter Plot
# - Evolution Matrix Heatmaps (examples)
# - Multi-config comparison figures (chunk_500 vs chunk_1000)

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from typing import Dict, List, Optional

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("PaperFigures")

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color palettes
DRIFT_TYPE_COLORS = {
    'abrupt': '#E74C3C',      # Red
    'gradual': '#3498DB',     # Blue
    'noisy': '#F39C12',       # Orange
    'stationary': '#2ECC71',  # Green
    'real': '#9B59B6',        # Purple
    'unknown': '#95A5A6'      # Gray
}

# Config colors for multi-config comparison
# Source 1 uses config_label (e.g. EXP-500-NP), Source 3 used config (e.g. chunk_500)
CONFIG_COLORS = {
    'EXP-500-NP': '#1f77b4',         # Blue
    'EXP-500-P': '#ff7f0e',          # Orange
    'EXP-1000-NP': '#2ca02c',        # Green
    'EXP-1000-P': '#d62728',         # Red
    'EXP-2000-NP': '#9467bd',        # Purple
    'EXP-2000-P': '#8c564b',         # Brown
    'EXP-BEST': '#e377c2',           # Pink
    # Legacy keys for backward compatibility
    'chunk_500': '#1f77b4',
    'chunk_500_penalty': '#ff7f0e',
    'chunk_1000': '#2ca02c',
    'chunk_1000_penalty': '#d62728',
}

# Config display names
CONFIG_DISPLAY_NAMES = {
    'EXP-500-NP': 'EXP-500',
    'EXP-500-P': 'EXP-500-P',
    'EXP-1000-NP': 'EXP-1000',
    'EXP-1000-P': 'EXP-1000-P',
    'EXP-2000-NP': 'EXP-2000',
    'EXP-2000-P': 'EXP-2000-P',
    'EXP-BEST': 'EXP-BEST',
    # Legacy
    'chunk_500': 'Chunk 500',
    'chunk_500_penalty': 'Chunk 500 (Penalty)',
    'chunk_1000': 'Chunk 1000',
    'chunk_1000_penalty': 'Chunk 1000 (Penalty)',
}

# Line styles for multi-config plots
CONFIG_LINE_STYLES = {
    'EXP-500-NP': '-',
    'EXP-500-P': '--',
    'EXP-1000-NP': '-',
    'EXP-1000-P': '--',
    'EXP-2000-NP': '-',
    'EXP-2000-P': '--',
    'EXP-BEST': '-.',
    'chunk_500': '-',
    'chunk_500_penalty': '--',
    'chunk_1000': '-',
    'chunk_1000_penalty': '--',
}

# Markers for multi-config plots
CONFIG_MARKERS = {
    'EXP-500-NP': 'o',
    'EXP-500-P': 's',
    'EXP-1000-NP': '^',
    'EXP-1000-P': 'D',
    'EXP-2000-NP': 'v',
    'EXP-2000-P': 'P',
    'EXP-BEST': '*',
    'chunk_500': 'o',
    'chunk_500_penalty': 's',
    'chunk_1000': '^',
    'chunk_1000_penalty': 'D',
}

# Column name for config filtering - Source 1 uses 'config_label'
CONFIG_COLUMN = 'config_label'


def generate_tcs_timeseries(df: pd.DataFrame, output_path: str,
                            config: str = "chunk_500", max_chunks: int = 25):
    """
    Generate TCS (Total Change Score) time-series plot by drift type.

    Shows how TCS evolves across chunk transitions, with separate lines
    for different drift types.
    """
    logger.info(f"Generating TCS time-series plot for config: {config}")

    # Filter by config
    df_config = df[df[CONFIG_COLUMN] == config].copy().copy()

    if df_config.empty:
        logger.warning(f"No data found for config: {config}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot by drift type
    drift_types = ['abrupt', 'gradual', 'noisy', 'stationary', 'real']

    for drift_type in drift_types:
        df_type = df_config[df_config['drift_type'] == drift_type]
        if df_type.empty:
            continue

        # Group by chunk transition and calculate mean TCS
        tcs_by_chunk = df_type.groupby('chunk_from')['TCS'].agg(['mean', 'std']).reset_index()
        tcs_by_chunk = tcs_by_chunk[tcs_by_chunk['chunk_from'] < max_chunks]

        if len(tcs_by_chunk) < 2:
            continue

        color = DRIFT_TYPE_COLORS.get(drift_type, '#95A5A6')

        # Plot mean line
        ax.plot(tcs_by_chunk['chunk_from'], tcs_by_chunk['mean'],
                marker='o', markersize=5, linewidth=2, label=f'{drift_type.capitalize()}',
                color=color)

        # Add confidence band
        ax.fill_between(tcs_by_chunk['chunk_from'],
                        tcs_by_chunk['mean'] - tcs_by_chunk['std'],
                        tcs_by_chunk['mean'] + tcs_by_chunk['std'],
                        alpha=0.2, color=color)

    # Formatting
    ax.set_xlabel('Chunk Transition (from Chunk i to Chunk i+1)')
    ax.set_ylabel('Total Change Score (TCS)')
    ax.set_title(f'Rule Evolution Over Time by Drift Type ({config})')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(-0.5, max_chunks - 0.5)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Save
    plt.savefig(output_path, format='pdf' if output_path.endswith('.pdf') else 'png')
    plt.close()
    logger.info(f"TCS time-series saved to: {output_path}")


def generate_rir_vs_ams_scatter(df: pd.DataFrame, output_path: str,
                                 config: str = "chunk_500"):
    """
    Generate RIR vs AMS scatter plot.

    Shows relationship between Rule Instability Rate and Average Modification Score,
    colored by drift type and sized by TCS.
    """
    logger.info(f"Generating RIR vs AMS scatter plot for config: {config}")

    # Filter by config
    df_config = df[df[CONFIG_COLUMN] == config].copy().copy()

    if df_config.empty:
        logger.warning(f"No data found for config: {config}")
        return

    # Aggregate by dataset (mean across transitions)
    df_agg = df_config.groupby(['dataset', 'drift_type']).agg({
        'rir': 'mean',
        'ams': 'mean',
        'tcs': 'mean'
    }).reset_index()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot by drift type
    drift_types = ['abrupt', 'gradual', 'noisy', 'stationary', 'real']

    for drift_type in drift_types:
        df_type = df_agg[df_agg['drift_type'] == drift_type]
        if df_type.empty:
            continue

        color = DRIFT_TYPE_COLORS.get(drift_type, '#95A5A6')

        # Size based on TCS (scaled for visibility)
        sizes = (df_type['TCS'] * 200) + 50

        ax.scatter(df_type['RIR'], df_type['AMS'],
                   c=color, s=sizes, alpha=0.7,
                   label=f'{drift_type.capitalize()} (n={len(df_type)})',
                   edgecolors='white', linewidth=0.5)

    # Add diagonal reference line (RIR = AMS)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='RIR = AMS')

    # Formatting
    ax.set_xlabel('Rule Instability Rate (RIR)')
    ax.set_ylabel('Average Modification Score (AMS)')
    ax.set_title(f'Rule Change Characteristics by Drift Type ({config})')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add annotation explaining point size
    ax.text(0.95, 0.05, 'Point size ~ TCS',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, style='italic', alpha=0.7)

    # Save
    plt.savefig(output_path, format='pdf' if output_path.endswith('.pdf') else 'png')
    plt.close()
    logger.info(f"RIR vs AMS scatter saved to: {output_path}")


def generate_metrics_boxplot(df: pd.DataFrame, output_path: str,
                              config: str = "chunk_500"):
    """
    Generate boxplot comparing TCS, RIR, AMS across drift types.
    """
    logger.info(f"Generating metrics boxplot for config: {config}")

    # Filter by config
    df_config = df[df[CONFIG_COLUMN] == config].copy().copy()

    if df_config.empty:
        logger.warning(f"No data found for config: {config}")
        return

    # Melt for plotting
    df_melt = df_config.melt(
        id_vars=['dataset', 'drift_type'],
        value_vars=['TCS', 'RIR', 'AMS'],
        var_name='metric',
        value_name='value'
    )

    # Rename metrics for display
    df_melt['metric'] = df_melt['metric'].map({
        'TCS': 'TCS (Total Change)',
        'RIR': 'RIR (Instability)',
        'AMS': 'AMS (Modification)'
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Custom palette
    palette = [DRIFT_TYPE_COLORS.get(dt, '#95A5A6')
               for dt in sorted(df_melt['drift_type'].unique())]

    # Create boxplot
    sns.boxplot(data=df_melt, x='metric', y='value', hue='drift_type',
                palette=DRIFT_TYPE_COLORS, ax=ax)

    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Transition Metrics Distribution by Drift Type ({config})')
    ax.legend(title='Drift Type', loc='upper right', framealpha=0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # Save
    plt.savefig(output_path, format='pdf' if output_path.endswith('.pdf') else 'png')
    plt.close()
    logger.info(f"Metrics boxplot saved to: {output_path}")


def generate_evolution_heatmap_examples(df: pd.DataFrame, output_dir: str,
                                         config: str = "chunk_500"):
    """
    Generate evolution heatmap examples for representative datasets.

    Creates one heatmap per drift type showing unchanged/modified/new/deleted
    rule counts across transitions.
    """
    logger.info(f"Generating evolution heatmap examples for config: {config}")

    # Filter by config
    df_config = df[df[CONFIG_COLUMN] == config].copy().copy()

    if df_config.empty:
        logger.warning(f"No data found for config: {config}")
        return

    # Select representative datasets (one per drift type)
    example_datasets = {
        'abrupt': ['SEA_Abrupt_Simple', 'AGRAWAL_Abrupt_Simple_Severe', 'RBF_Abrupt_Severe'],
        'gradual': ['SEA_Gradual_Simple_Slow', 'RBF_Gradual_Moderate', 'STAGGER_Gradual_Chain'],
        'noisy': ['RBF_Noise', 'SEA_Noise', 'AGRAWAL_Noise'],
        'stationary': ['SEA_Stationary', 'RBF_Stationary', 'AGRAWAL_Stationary'],
        'real': ['Electricity', 'CovType', 'PokerHand']
    }

    for drift_type, candidates in example_datasets.items():
        df_type = df_config[df_config['drift_type'] == drift_type]

        if df_type.empty:
            logger.warning(f"No data for drift type: {drift_type}")
            continue

        # Find first available candidate dataset
        selected = None
        for candidate in candidates:
            if candidate in df_type['dataset'].values:
                selected = candidate
                break

        if selected is None:
            # Use first available dataset
            selected = df_type['dataset'].iloc[0]

        df_dataset = df_type[df_type['dataset'] == selected].copy()
        df_dataset = df_dataset.sort_values('chunk_from')

        if len(df_dataset) < 2:
            logger.warning(f"Insufficient data for {selected}")
            continue

        # Create heatmap data
        categories = ['Unchanged', 'Modified', 'New', 'Deleted']
        data = df_dataset[['unchanged', 'modified', 'new', 'deleted']].values.T
        transitions = [f"{int(row['chunk_from'])}->{int(row['chunk_to'])}"
                       for _, row in df_dataset.iterrows()]

        # Limit to first 15 transitions for readability
        if len(transitions) > 15:
            data = data[:, :15]
            transitions = transitions[:15]

        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, len(transitions) * 0.6), 5))

        # Create heatmap
        im = ax.imshow(data, cmap='Blues', aspect='auto')

        # Formatting
        ax.set_xticks(np.arange(len(transitions)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels(transitions, rotation=45, ha='right')
        ax.set_yticklabels(categories)

        # Add values as text
        for i in range(len(categories)):
            for j in range(len(transitions)):
                text = ax.text(j, i, int(data[i, j]),
                               ha='center', va='center',
                               color='white' if data[i, j] > data.max() / 2 else 'black',
                               fontsize=9)

        ax.set_title(f'Rule Evolution: {selected} ({drift_type.capitalize()})')
        ax.set_xlabel('Chunk Transition')
        ax.set_ylabel('Change Category')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Number of Rules')

        # Save
        output_path = os.path.join(output_dir, f'evolution_heatmap_{drift_type}_{selected}.pdf')
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Evolution heatmap saved to: {output_path}")


def generate_summary_table(df: pd.DataFrame, output_path: str):
    """
    Generate LaTeX table for Table XIV in the paper.
    """
    logger.info("Generating summary table for paper")

    # Aggregate by drift type and config
    summary = df.groupby([CONFIG_COLUMN, 'drift_type']).agg({
        'tcs': ['mean', 'std'],
        'rir': ['mean', 'std'],
        'ams': ['mean', 'std'],
        'dataset': 'nunique'
    }).reset_index()

    # Flatten column names
    summary.columns = [CONFIG_COLUMN, 'drift_type', 'tcs_mean', 'tcs_std',
                       'rir_mean', 'rir_std', 'ams_mean', 'ams_std', 'n_datasets']

    # Format as LaTeX
    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Transition Metrics by Drift Type and Configuration}",
        r"\label{tab:transition_metrics}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Config & Drift Type & TCS & RIR & AMS & N \\",
        r"\midrule"
    ]

    for config in sorted(summary[CONFIG_COLUMN].unique()):
        config_df = summary[summary[CONFIG_COLUMN] == config]
        for _, row in config_df.iterrows():
            tcs = f"${row['tcs_mean']:.3f} \\pm {row['tcs_std']:.3f}$"
            rir = f"${row['rir_mean']:.3f} \\pm {row['rir_std']:.3f}$"
            ams = f"${row['ams_mean']:.3f} \\pm {row['ams_std']:.3f}$"
            latex_lines.append(
                f"{row[CONFIG_COLUMN]} & {row['drift_type']} & {tcs} & {rir} & {ams} & {int(row['n_datasets'])} \\\\"
            )
        latex_lines.append(r"\midrule")

    # Remove last midrule and add bottomrule
    latex_lines[-1] = r"\bottomrule"
    latex_lines.extend([
        r"\end{tabular}",
        r"\end{table}"
    ])

    # Save
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))

    logger.info(f"LaTeX table saved to: {output_path}")


# ============================================================================
# MULTI-CONFIG COMPARISON FUNCTIONS
# ============================================================================

def generate_tcs_comparison_multiconfig(df: pd.DataFrame, output_path: str,
                                         drift_type: str = "abrupt",
                                         max_chunks: int = 20):
    """
    Generate TCS time-series comparison across all configurations.

    Shows TCS evolution with separate lines for each config,
    allowing direct comparison of chunk_500 vs chunk_1000.
    """
    logger.info(f"Generating TCS multi-config comparison for drift type: {drift_type}")

    # Filter by drift type
    df_type = df[df['drift_type'] == drift_type].copy()

    if df_type.empty:
        logger.warning(f"No data found for drift type: {drift_type}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each config
    configs = sorted(df[CONFIG_COLUMN].unique().tolist())

    for config in configs:
        df_config = df_type[df_type[CONFIG_COLUMN] == config]
        if df_config.empty:
            continue

        # Group by chunk transition and calculate mean TCS
        tcs_by_chunk = df_config.groupby('chunk_from')['TCS'].agg(['mean', 'std']).reset_index()
        tcs_by_chunk = tcs_by_chunk[tcs_by_chunk['chunk_from'] < max_chunks]

        if len(tcs_by_chunk) < 2:
            continue

        color = CONFIG_COLORS.get(config, '#95A5A6')
        linestyle = CONFIG_LINE_STYLES.get(config, '-')
        marker = CONFIG_MARKERS.get(config, 'o')
        display_name = CONFIG_DISPLAY_NAMES.get(config, config)

        # Plot mean line
        ax.plot(tcs_by_chunk['chunk_from'], tcs_by_chunk['mean'],
                marker=marker, markersize=6, linewidth=2, linestyle=linestyle,
                label=display_name, color=color)

        # Add confidence band
        ax.fill_between(tcs_by_chunk['chunk_from'],
                        tcs_by_chunk['mean'] - tcs_by_chunk['std'],
                        tcs_by_chunk['mean'] + tcs_by_chunk['std'],
                        alpha=0.15, color=color)

    # Formatting
    ax.set_xlabel('Chunk Transition (from Chunk i to Chunk i+1)')
    ax.set_ylabel('Total Change Score (TCS)')
    ax.set_title(f'TCS Evolution Comparison - {drift_type.capitalize()} Drift')
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)
    ax.set_xlim(-0.5, max_chunks - 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, format='pdf' if output_path.endswith('.pdf') else 'png')
    plt.close()
    logger.info(f"TCS multi-config comparison saved to: {output_path}")


def generate_tcs_comparison_all_drifts(df: pd.DataFrame, output_path: str,
                                        max_chunks: int = 20):
    """
    Generate TCS time-series comparison with subplots for each drift type.

    Creates a 2x3 subplot figure comparing all configs across drift types.
    """
    logger.info("Generating TCS comparison across all drift types")

    drift_types = ['abrupt', 'gradual', 'noisy', 'stationary', 'real']
    configs = sorted(df[CONFIG_COLUMN].unique().tolist())

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, drift_type in enumerate(drift_types):
        ax = axes[idx]
        df_type = df[df['drift_type'] == drift_type]

        if df_type.empty:
            ax.set_title(f'{drift_type.capitalize()} (No data)')
            ax.set_visible(False)
            continue

        for config in configs:
            df_config = df_type[df_type[CONFIG_COLUMN] == config]
            if df_config.empty:
                continue

            tcs_by_chunk = df_config.groupby('chunk_from')['TCS'].agg(['mean', 'std']).reset_index()
            tcs_by_chunk = tcs_by_chunk[tcs_by_chunk['chunk_from'] < max_chunks]

            if len(tcs_by_chunk) < 2:
                continue

            color = CONFIG_COLORS.get(config, '#95A5A6')
            linestyle = CONFIG_LINE_STYLES.get(config, '-')
            marker = CONFIG_MARKERS.get(config, 'o')
            display_name = CONFIG_DISPLAY_NAMES.get(config, config)

            ax.plot(tcs_by_chunk['chunk_from'], tcs_by_chunk['mean'],
                    marker=marker, markersize=4, linewidth=1.5, linestyle=linestyle,
                    label=display_name, color=color)

            ax.fill_between(tcs_by_chunk['chunk_from'],
                            tcs_by_chunk['mean'] - tcs_by_chunk['std'],
                            tcs_by_chunk['mean'] + tcs_by_chunk['std'],
                            alpha=0.1, color=color)

        ax.set_title(f'{drift_type.capitalize()} Drift')
        ax.set_xlabel('Chunk Transition')
        ax.set_ylabel('TCS')
        ax.set_xlim(-0.5, max_chunks - 0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    # Hide empty subplot
    axes[-1].set_visible(False)

    # Add single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.95, 0.15),
               ncol=2, framealpha=0.9)

    plt.suptitle('TCS Evolution Comparison Across All Drift Types and Configurations',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, format='pdf' if output_path.endswith('.pdf') else 'png')
    plt.close()
    logger.info(f"TCS comparison (all drifts) saved to: {output_path}")


def generate_metrics_comparison_multiconfig(df: pd.DataFrame, output_path: str):
    """
    Generate grouped boxplot comparing TCS, RIR, AMS across all configurations.

    Shows distribution of each metric with separate boxes for each config,
    grouped by drift type.
    """
    logger.info("Generating metrics comparison boxplot (multi-config)")

    # Aggregate by dataset to avoid overplotting
    df_agg = df.groupby(['dataset', CONFIG_COLUMN, 'drift_type']).agg({
        'tcs': 'mean',
        'rir': 'mean',
        'ams': 'mean'
    }).reset_index()

    # Melt for plotting
    df_melt = df_agg.melt(
        id_vars=['dataset', CONFIG_COLUMN, 'drift_type'],
        value_vars=['TCS', 'RIR', 'AMS'],
        var_name='metric',
        value_name='value'
    )

    # Metrics already uppercase from Source 1

    # Add config display names
    df_melt['config_display'] = df_melt[CONFIG_COLUMN].map(CONFIG_DISPLAY_NAMES)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Create boxplot with hue for configs
    sns.boxplot(data=df_melt, x='metric', y='value', hue='config_display',
                palette=[CONFIG_COLORS[k] for k in CONFIG_COLORS.keys()],
                ax=ax, order=['TCS', 'RIR', 'AMS'])

    # Formatting
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Transition Metrics Distribution by Configuration')
    ax.legend(title='Configuration', loc='upper right', framealpha=0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # Save
    plt.tight_layout()
    plt.savefig(output_path, format='pdf' if output_path.endswith('.pdf') else 'png')
    plt.close()
    logger.info(f"Metrics comparison boxplot saved to: {output_path}")


def generate_metrics_comparison_by_drift(df: pd.DataFrame, output_path: str):
    """
    Generate faceted boxplot: metrics by drift type, comparing configurations.
    """
    logger.info("Generating metrics comparison faceted by drift type")

    # Aggregate by dataset
    df_agg = df.groupby(['dataset', CONFIG_COLUMN, 'drift_type']).agg({
        'tcs': 'mean',
        'rir': 'mean',
        'ams': 'mean'
    }).reset_index()

    # Melt for plotting
    df_melt = df_agg.melt(
        id_vars=['dataset', CONFIG_COLUMN, 'drift_type'],
        value_vars=['TCS', 'RIR', 'AMS'],
        var_name='metric',
        value_name='value'
    )

    # Metrics already uppercase from Source 1
    df_melt['config_display'] = df_melt[CONFIG_COLUMN].map(CONFIG_DISPLAY_NAMES)
    df_melt['drift_type'] = df_melt['drift_type'].str.capitalize()

    # Create faceted figure
    g = sns.catplot(
        data=df_melt, x='metric', y='value', hue='config_display',
        col='drift_type', kind='box', col_wrap=3,
        palette=[CONFIG_COLORS[k] for k in CONFIG_COLORS.keys()],
        height=4, aspect=1.2, order=['TCS', 'RIR', 'AMS'],
        sharey=True
    )

    g.set_axis_labels('Metric', 'Value')
    g.set_titles('{col_name} Drift')
    g.add_legend(title='Configuration')
    g.set(ylim=(-0.05, 1.05))

    for ax in g.axes.flat:
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Transition Metrics by Drift Type and Configuration', y=1.02,
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, format='pdf' if output_path.endswith('.pdf') else 'png')
    plt.close()
    logger.info(f"Metrics comparison (by drift) saved to: {output_path}")


def generate_rir_vs_ams_comparison(df: pd.DataFrame, output_path: str):
    """
    Generate RIR vs AMS scatter plot comparing all configurations.

    Uses different colors and markers for each config.
    """
    logger.info("Generating RIR vs AMS scatter comparison (multi-config)")

    # Aggregate by dataset
    df_agg = df.groupby(['dataset', CONFIG_COLUMN, 'drift_type']).agg({
        'rir': 'mean',
        'ams': 'mean',
        'tcs': 'mean'
    }).reset_index()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    configs = sorted(df[CONFIG_COLUMN].unique().tolist())

    for config in configs:
        df_config = df_agg[df_agg[CONFIG_COLUMN] == config]
        if df_config.empty:
            continue

        color = CONFIG_COLORS.get(config, '#95A5A6')
        marker = CONFIG_MARKERS.get(config, 'o')
        display_name = CONFIG_DISPLAY_NAMES.get(config, config)

        # Size based on TCS
        sizes = (df_config['TCS'] * 200) + 30

        ax.scatter(df_config['RIR'], df_config['AMS'],
                   c=color, s=sizes, alpha=0.6, marker=marker,
                   label=f'{display_name} (n={len(df_config)})',
                   edgecolors='white', linewidth=0.5)

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='RIR = AMS')

    # Formatting
    ax.set_xlabel('Rule Instability Rate (RIR)')
    ax.set_ylabel('Average Modification Score (AMS)')
    ax.set_title('RIR vs AMS Across All Configurations')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(0.95, 0.05, 'Point size ~ TCS',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, style='italic', alpha=0.7)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, format='pdf' if output_path.endswith('.pdf') else 'png')
    plt.close()
    logger.info(f"RIR vs AMS comparison saved to: {output_path}")


def generate_config_comparison_barplot(df: pd.DataFrame, output_path: str):
    """
    Generate grouped barplot comparing mean metrics across configurations.

    Shows mean TCS, RIR, AMS with error bars for each config,
    grouped by drift type.
    """
    logger.info("Generating config comparison barplot")

    # Aggregate by config and drift type
    summary = df.groupby([CONFIG_COLUMN, 'drift_type']).agg({
        'tcs': ['mean', 'std'],
        'rir': ['mean', 'std'],
        'ams': ['mean', 'std']
    }).reset_index()

    summary.columns = [CONFIG_COLUMN, 'drift_type', 'tcs_mean', 'tcs_std',
                       'rir_mean', 'rir_std', 'ams_mean', 'ams_std']

    # Melt for grouped barplot
    summary_melt = []
    for _, row in summary.iterrows():
        for metric in ['TCS', 'RIR', 'AMS']:
            metric_lower = metric.lower()
            summary_melt.append({
                CONFIG_COLUMN: row[CONFIG_COLUMN],
                'drift_type': row['drift_type'].capitalize(),
                'metric': metric,
                'mean': row[f'{metric_lower}_mean'],
                'std': row[f'{metric_lower}_std']
            })

    df_summary = pd.DataFrame(summary_melt)
    df_summary['config_display'] = df_summary[CONFIG_COLUMN].map(CONFIG_DISPLAY_NAMES)

    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = ['TCS', 'RIR', 'AMS']
    titles = ['Total Change Score (TCS)', 'Rule Instability Rate (RIR)',
              'Average Modification Score (AMS)']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        df_metric = df_summary[df_summary['metric'] == metric]

        # Pivot for grouped barplot
        df_pivot = df_metric.pivot(index='drift_type', columns='config_display', values='mean')
        df_pivot_std = df_metric.pivot(index='drift_type', columns='config_display', values='std')

        # Reorder columns
        cols_order = [CONFIG_DISPLAY_NAMES[k] for k in CONFIG_COLORS.keys()
                      if CONFIG_DISPLAY_NAMES[k] in df_pivot.columns]
        df_pivot = df_pivot[cols_order]
        df_pivot_std = df_pivot_std[cols_order]

        # Plot
        x = np.arange(len(df_pivot.index))
        width = 0.2

        for i, col in enumerate(df_pivot.columns):
            config_key = [k for k, v in CONFIG_DISPLAY_NAMES.items() if v == col][0]
            color = CONFIG_COLORS[config_key]

            ax.bar(x + i * width, df_pivot[col], width,
                   yerr=df_pivot_std[col], label=col, color=color,
                   capsize=3, alpha=0.8)

        ax.set_xlabel('Drift Type')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df_pivot.index, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        if idx == 2:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Mean Transition Metrics Comparison Across Configurations',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, format='pdf' if output_path.endswith('.pdf') else 'png')
    plt.close()
    logger.info(f"Config comparison barplot saved to: {output_path}")


def generate_chunk_size_effect_plot(df: pd.DataFrame, output_path: str):
    """
    Generate plot showing the effect of chunk size (500 vs 1000) on metrics.

    Compares chunk_500 vs chunk_1000 (without penalty) to isolate chunk size effect.
    """
    logger.info("Generating chunk size effect plot")

    # Filter to non-penalty configs
    # Filter to non-penalty configs for chunk size comparison
    non_penalty = [c for c in df[CONFIG_COLUMN].unique() if 'NP' in str(c) or (c in ['chunk_500', 'chunk_1000'])]
    df_filtered = df[df[CONFIG_COLUMN].isin(non_penalty)].copy()

    if df_filtered.empty:
        logger.warning("No data for chunk size comparison")
        return

    # Aggregate by drift type and config
    summary = df_filtered.groupby([CONFIG_COLUMN, 'drift_type']).agg({
        'tcs': ['mean', 'std'],
        'rir': ['mean', 'std'],
        'ams': ['mean', 'std'],
        'dataset': 'nunique'
    }).reset_index()

    summary.columns = [CONFIG_COLUMN, 'drift_type', 'tcs_mean', 'tcs_std',
                       'rir_mean', 'rir_std', 'ams_mean', 'ams_std', 'n_datasets']

    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    drift_types = sorted(summary['drift_type'].unique())
    x = np.arange(len(drift_types))
    width = 0.35

    metrics = [('tcs', 'TCS'), ('rir', 'RIR'), ('ams', 'AMS')]

    for idx, (metric, metric_label) in enumerate(metrics):
        ax = axes[idx]

        for i, config in enumerate(sorted(non_penalty)):
            df_config = summary[summary[CONFIG_COLUMN] == config]

            means = [df_config[df_config['drift_type'] == dt][f'{metric}_mean'].values[0]
                     if dt in df_config['drift_type'].values else 0
                     for dt in drift_types]
            stds = [df_config[df_config['drift_type'] == dt][f'{metric}_std'].values[0]
                    if dt in df_config['drift_type'].values else 0
                    for dt in drift_types]

            color = CONFIG_COLORS[config]
            label = CONFIG_DISPLAY_NAMES[config]

            ax.bar(x + i * width - width/2, means, width, yerr=stds,
                   label=label, color=color, capsize=3, alpha=0.8)

        ax.set_xlabel('Drift Type')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} by Chunk Size')
        ax.set_xticks(x)
        ax.set_xticklabels([dt.capitalize() for dt in drift_types], rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right')

    plt.suptitle('Effect of Chunk Size on Transition Metrics (Without Penalty)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, format='pdf' if output_path.endswith('.pdf') else 'png')
    plt.close()
    logger.info(f"Chunk size effect plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures from consolidated transition metrics."
    )
    parser.add_argument(
        "--input",
        default="paper_data/egis_transition_metrics.csv",
        help="Path to transition metrics CSV (Source 1: paper_data/egis_transition_metrics.csv)"
    )
    parser.add_argument(
        "--heatmap_input",
        default="paper_data/evolution_analysis_summary.csv",
        help="Path to evolution summary CSV with rule counts (Source 2: for heatmaps)"
    )
    parser.add_argument(
        "--output_dir",
        default="paper/figures",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--config",
        default="EXP-500-NP",
        help="Configuration to use for single-config figures (default: EXP-500-NP)"
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png"],
        help="Output format for figures (default: pdf)"
    )
    parser.add_argument(
        "--all_configs",
        action="store_true",
        help="Generate comparative figures for all configurations"
    )
    parser.add_argument(
        "--only_comparisons",
        action="store_true",
        help="Only generate multi-config comparison figures (skip single-config)"
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_dir = os.path.join(script_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Input file: {input_path}")
    logger.info(f"Output directory: {output_dir}")

    # Load metrics data (Source 1: TCS, RIR, AMS)
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        logger.info("Expected: paper_data/egis_transition_metrics.csv (Source 1)")
        return

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} transitions from {df['dataset'].nunique()} datasets")

    # Detect config column name (Source 1 uses config_label, Source 3 uses config)
    global CONFIG_COLUMN
    if 'config_label' in df.columns:
        CONFIG_COLUMN = 'config_label'
    elif 'config' in df.columns:
        CONFIG_COLUMN = 'config'
    logger.info(f"Config column: {CONFIG_COLUMN}")
    logger.info(f"Configurations available: {df[CONFIG_COLUMN].unique().tolist()}")

    # Load heatmap data (Source 2: rule counts) for evolution heatmaps
    heatmap_path = os.path.join(script_dir, args.heatmap_input) if not os.path.isabs(args.heatmap_input) else args.heatmap_input
    df_heatmap = None
    if os.path.exists(heatmap_path):
        df_heatmap = pd.read_csv(heatmap_path)
        logger.info(f"Loaded {len(df_heatmap)} records for heatmaps from {heatmap_path}")
    else:
        logger.warning(f"Heatmap data not found: {heatmap_path}. Evolution heatmaps will use main data.")

    # Generate figures
    ext = f".{args.format}"

    # Create directory structure
    if args.all_configs:
        # Create subdirectories
        por_config_dir = os.path.join(output_dir, "por_config")
        comparativas_dir = os.path.join(output_dir, "comparativas")
        tabelas_dir = os.path.join(output_dir, "tabelas")

        os.makedirs(por_config_dir, exist_ok=True)
        os.makedirs(comparativas_dir, exist_ok=True)
        os.makedirs(tabelas_dir, exist_ok=True)

        # Generate figures for each config
        if not args.only_comparisons:
            configs_in_data = df[CONFIG_COLUMN].unique().tolist()
            for config in configs_in_data:
                config_dir = os.path.join(por_config_dir, config)
                os.makedirs(config_dir, exist_ok=True)

                logger.info(f"\n{'='*60}")
                logger.info(f"Generating figures for config: {config}")
                logger.info(f"{'='*60}")

                # TCS Time-Series
                generate_tcs_timeseries(
                    df, os.path.join(config_dir, f"tcs_timeseries{ext}"),
                    config=config
                )

                # RIR vs AMS Scatter
                generate_rir_vs_ams_scatter(
                    df, os.path.join(config_dir, f"rir_vs_ams_scatter{ext}"),
                    config=config
                )

                # Metrics Boxplot
                generate_metrics_boxplot(
                    df, os.path.join(config_dir, f"metrics_boxplot{ext}"),
                    config=config
                )

                # Evolution Heatmap Examples
                generate_evolution_heatmap_examples(df_heatmap if df_heatmap is not None else df, config_dir, config=config)

        # Generate comparative figures
        logger.info(f"\n{'='*60}")
        logger.info("Generating comparative figures (all configs)")
        logger.info(f"{'='*60}")

        # TCS comparison for each drift type
        for drift_type in ['abrupt', 'gradual', 'noisy', 'stationary', 'real']:
            generate_tcs_comparison_multiconfig(
                df, os.path.join(comparativas_dir, f"tcs_comparison_{drift_type}{ext}"),
                drift_type=drift_type
            )

        # TCS comparison all drifts in one figure
        generate_tcs_comparison_all_drifts(
            df, os.path.join(comparativas_dir, f"tcs_comparison_all_drifts{ext}")
        )

        # Metrics comparison boxplot
        generate_metrics_comparison_multiconfig(
            df, os.path.join(comparativas_dir, f"metrics_comparison_boxplot{ext}")
        )

        # Metrics comparison by drift type (faceted)
        generate_metrics_comparison_by_drift(
            df, os.path.join(comparativas_dir, f"metrics_comparison_by_drift{ext}")
        )

        # RIR vs AMS comparison
        generate_rir_vs_ams_comparison(
            df, os.path.join(comparativas_dir, f"rir_vs_ams_comparison{ext}")
        )

        # Config comparison barplot
        generate_config_comparison_barplot(
            df, os.path.join(comparativas_dir, f"config_comparison_barplot{ext}")
        )

        # Chunk size effect plot
        generate_chunk_size_effect_plot(
            df, os.path.join(comparativas_dir, f"chunk_size_effect{ext}")
        )

        # LaTeX Table (in tabelas dir)
        generate_summary_table(
            df, os.path.join(tabelas_dir, "table_xiv_transition_metrics.tex")
        )

    else:
        # Single config mode (original behavior)
        if not args.only_comparisons:
            # 1. TCS Time-Series
            generate_tcs_timeseries(
                df,
                os.path.join(output_dir, f"tcs_timeseries_by_drift{ext}"),
                config=args.config
            )

            # 2. RIR vs AMS Scatter
            generate_rir_vs_ams_scatter(
                df,
                os.path.join(output_dir, f"rir_vs_ams_scatter{ext}"),
                config=args.config
            )

            # 3. Metrics Boxplot
            generate_metrics_boxplot(
                df,
                os.path.join(output_dir, f"metrics_boxplot{ext}"),
                config=args.config
            )

            # 4. Evolution Heatmap Examples
            generate_evolution_heatmap_examples(df_heatmap if df_heatmap is not None else df, output_dir, config=args.config)

        # 5. LaTeX Table
        generate_summary_table(df, os.path.join(output_dir, "table_xiv_transition_metrics.tex"))

    logger.info("\n" + "="*60)
    logger.info("Paper figures generation complete!")
    logger.info(f"All figures saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
