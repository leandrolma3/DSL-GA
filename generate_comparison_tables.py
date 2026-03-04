#!/usr/bin/env python3
"""
Generate LaTeX comparison tables for the IEEE TKDE paper.

Creates two tables:
- Table A: Binary datasets only (41 datasets, all 7 models)
- Table B: All datasets (52 datasets, 6 models excluding ACDWM/CDCMS)

Also generates Win/Lose/Draw statistics and mean +/- std.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def calculate_win_lose_draw(df: pd.DataFrame, model1: str, model2: str) -> tuple:
    """Calculate win/lose/draw between two models."""
    # Get common datasets
    mask = df[model1].notna() & df[model2].notna()
    m1 = df.loc[mask, model1].values
    m2 = df.loc[mask, model2].values

    wins = np.sum(m1 > m2)
    losses = np.sum(m1 < m2)
    draws = np.sum(m1 == m2)

    return wins, losses, draws


def calculate_rankings(df: pd.DataFrame, models: list) -> dict:
    """Calculate mean rankings for each model (lower is better)."""
    rankings = {m: [] for m in models}

    for idx, row in df.iterrows():
        values = {m: row[m] for m in models if pd.notna(row[m])}
        if len(values) < 2:
            continue

        # Sort by G-Mean (descending - higher is better)
        sorted_models = sorted(values.keys(), key=lambda x: values[x], reverse=True)

        # Assign ranks (1 = best)
        for rank, m in enumerate(sorted_models, 1):
            rankings[m].append(rank)

    # Calculate mean rank
    mean_ranks = {m: np.mean(rankings[m]) if rankings[m] else np.nan for m in models}
    return mean_ranks


def generate_table_binary(df: pd.DataFrame, config: str = 'chunk_500') -> str:
    """Generate Table A: Binary datasets comparison (all 7 models)."""
    models = ['EGIS', 'ARF', 'HAT', 'SRP', 'ROSE', 'eRulesD2S', 'CDCMS']

    # Filter for config and binary datasets
    df_filtered = df[(df['config'] == config) & (df['is_binary'] == True)].copy()

    # Sort by drift type and dataset name
    drift_order = {'abrupt': 0, 'gradual': 1, 'noisy': 2, 'stationary': 3, 'real': 4}
    df_filtered['drift_order'] = df_filtered['drift_type'].map(drift_order)
    df_filtered = df_filtered.sort_values(['drift_order', 'dataset'])

    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance Comparison on Binary Datasets (G-Mean) - Chunk Size " + config.split('_')[1] + "}")
    lines.append(r"\label{tab:binary_comparison}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{l|ccccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{EGIS} & \textbf{ARF} & \textbf{HAT} & \textbf{SRP} & \textbf{ROSE} & \textbf{eRulesD2S} & \textbf{CDCMS} \\")
    lines.append(r"\midrule")

    current_drift = None

    for idx, row in df_filtered.iterrows():
        drift = row['drift_type']

        # Add section header for drift type
        if drift != current_drift:
            drift_name = drift.capitalize()
            count = len(df_filtered[df_filtered['drift_type'] == drift])
            lines.append(r"\multicolumn{8}{l}{\textbf{" + drift_name + f" Drift ({count})" + r"}} \\")
            current_drift = drift

        # Dataset row
        dataset = row['dataset'].replace('_', r'\_')

        # Collect values
        values = []
        best_val = -np.inf
        for m in models:
            val = row[m]
            if pd.notna(val):
                values.append(val)
                if val > best_val:
                    best_val = val
            else:
                values.append(None)

        # Format cells (bold if best)
        cells = []
        for i, m in enumerate(models):
            val = values[i]
            if val is None:
                cells.append('--')
            elif abs(val - best_val) < 0.0001:  # Best value
                cells.append(r'\textbf{' + f'{val:.3f}' + '}')
            else:
                cells.append(f'{val:.3f}')

        line = dataset + ' & ' + ' & '.join(cells) + r' \\'
        lines.append(line)

    lines.append(r"\midrule")

    # Win/Lose/Draw vs EGIS
    wins_vs_egis = {}
    for m in models:
        if m == 'EGIS':
            wins_vs_egis[m] = '--'
        else:
            w, l, d = calculate_win_lose_draw(df_filtered, 'EGIS', m)
            wins_vs_egis[m] = f'{w}/{l}/{d}'

    lines.append(r"\textbf{EGIS W/L/D} & -- & " + ' & '.join([wins_vs_egis[m] for m in models[1:]]) + r' \\')

    # Mean and Std
    means = []
    stds = []
    for m in models:
        vals = df_filtered[m].dropna()
        if len(vals) > 0:
            means.append(f'{vals.mean():.3f}')
            stds.append(f'{vals.std():.3f}')
        else:
            means.append('--')
            stds.append('--')

    lines.append(r"\textbf{Mean} & " + ' & '.join(means) + r' \\')
    lines.append(r"\textbf{Std} & " + ' & '.join(stds) + r' \\')

    # Rankings
    mean_ranks = calculate_rankings(df_filtered, models)
    ranks = [f'{mean_ranks[m]:.2f}' if pd.notna(mean_ranks[m]) else '--' for m in models]
    lines.append(r"\textbf{Avg Rank} & " + ' & '.join(ranks) + r' \\')

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return '\n'.join(lines)


def generate_table_all(df: pd.DataFrame, config: str = 'chunk_500') -> str:
    """Generate Table B: All datasets comparison (6 models, excluding ACDWM/CDCMS)."""
    models = ['EGIS', 'ARF', 'HAT', 'SRP', 'ROSE', 'eRulesD2S']

    # Filter for config
    df_filtered = df[df['config'] == config].copy()

    # Sort by drift type and dataset name
    drift_order = {'abrupt': 0, 'gradual': 1, 'noisy': 2, 'stationary': 3, 'real': 4, 'base': 5}
    df_filtered['drift_order'] = df_filtered['drift_type'].map(drift_order)
    df_filtered = df_filtered.sort_values(['drift_order', 'dataset'])

    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance Comparison on All Datasets (G-Mean) - Chunk Size " + config.split('_')[1] + "}")
    lines.append(r"\label{tab:full_comparison}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{l|cccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{EGIS} & \textbf{ARF} & \textbf{HAT} & \textbf{SRP} & \textbf{ROSE} & \textbf{eRulesD2S} \\")
    lines.append(r"\midrule")

    current_drift = None

    for idx, row in df_filtered.iterrows():
        drift = row['drift_type']

        # Add section header for drift type
        if drift != current_drift:
            drift_name = drift.capitalize()
            count = len(df_filtered[df_filtered['drift_type'] == drift])
            lines.append(r"\multicolumn{7}{l}{\textbf{" + drift_name + f" ({count})" + r"}} \\")
            current_drift = drift

        # Dataset row
        dataset = row['dataset'].replace('_', r'\_')

        # Collect values
        values = []
        best_val = -np.inf
        for m in models:
            val = row[m]
            if pd.notna(val):
                values.append(val)
                if val > best_val:
                    best_val = val
            else:
                values.append(None)

        # Format cells (bold if best)
        cells = []
        for i, m in enumerate(models):
            val = values[i]
            if val is None:
                cells.append('--')
            elif abs(val - best_val) < 0.0001:  # Best value
                cells.append(r'\textbf{' + f'{val:.3f}' + '}')
            else:
                cells.append(f'{val:.3f}')

        line = dataset + ' & ' + ' & '.join(cells) + r' \\'
        lines.append(line)

    lines.append(r"\midrule")

    # Mean and Std
    means = []
    stds = []
    for m in models:
        vals = df_filtered[m].dropna()
        if len(vals) > 0:
            means.append(f'{vals.mean():.3f}')
            stds.append(f'{vals.std():.3f}')
        else:
            means.append('--')
            stds.append('--')

    lines.append(r"\textbf{Mean} & " + ' & '.join(means) + r' \\')
    lines.append(r"\textbf{Std} & " + ' & '.join(stds) + r' \\')

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return '\n'.join(lines)


def generate_table_viii_corrected(df: pd.DataFrame, config: str = 'chunk_500') -> str:
    """Generate corrected Table VIII: Performance by Drift Type."""
    models = ['EGIS', 'ARF', 'HAT', 'SRP', 'ROSE', 'eRulesD2S', 'CDCMS']

    # Filter for config and binary datasets
    df_filtered = df[(df['config'] == config) & (df['is_binary'] == True)].copy()

    # Group by drift type
    drift_types = ['abrupt', 'gradual', 'noisy', 'stationary', 'real']

    lines = []
    lines.append(r"% Auto-generated table: table_viii_drift_corrected")
    lines.append(r"% Generated with data from all models")
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance by Drift Type (G-Mean, EXP-" + config.split('_')[1] + r" Configuration)}")
    lines.append(r"\label{tab:drift_performance}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Drift Type} & \textbf{EGIS} & \textbf{ARF} & \textbf{HAT} & \textbf{SRP} & \textbf{ROSE} & \textbf{eRulesD2S} & \textbf{CDCMS} \\")
    lines.append(r"\midrule")

    for drift in drift_types:
        drift_df = df_filtered[df_filtered['drift_type'] == drift]
        count = len(drift_df)

        cells = [f"{drift.capitalize()} ({count})"]
        for m in models:
            vals = drift_df[m].dropna()
            if len(vals) > 0:
                cells.append(f'{vals.mean():.3f}')
            else:
                cells.append('--')

        lines.append(' & '.join(cells) + r' \\')

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return '\n'.join(lines)


def generate_summary_statistics(df: pd.DataFrame, config: str = 'chunk_500') -> str:
    """Generate summary statistics table."""
    models = ['EGIS', 'ARF', 'HAT', 'SRP', 'ROSE', 'eRulesD2S', 'CDCMS']

    # Filter for config and binary datasets
    df_filtered = df[(df['config'] == config) & (df['is_binary'] == True)].copy()

    lines = []
    lines.append("="*80)
    lines.append(f"SUMMARY STATISTICS - {config}")
    lines.append("="*80)
    lines.append(f"\nTotal binary datasets: {len(df_filtered)}")
    lines.append("\nMean G-Mean by Model:")
    lines.append("-"*40)

    for m in models:
        vals = df_filtered[m].dropna()
        if len(vals) > 0:
            lines.append(f"  {m:12s}: {vals.mean():.4f} +/- {vals.std():.4f} (n={len(vals)})")
        else:
            lines.append(f"  {m:12s}: N/A")

    # Win/Lose/Draw for EGIS vs each model
    lines.append("\nEGIS Win/Lose/Draw vs each model:")
    lines.append("-"*40)

    for m in models:
        if m == 'EGIS':
            continue
        w, l, d = calculate_win_lose_draw(df_filtered, 'EGIS', m)
        lines.append(f"  vs {m:12s}: {w:2d}W / {l:2d}L / {d:2d}D")

    # Mean rankings
    mean_ranks = calculate_rankings(df_filtered, models)
    lines.append("\nMean Rankings (lower is better):")
    lines.append("-"*40)

    for m in models:
        if pd.notna(mean_ranks[m]):
            lines.append(f"  {m:12s}: {mean_ranks[m]:.2f}")
        else:
            lines.append(f"  {m:12s}: N/A")

    return '\n'.join(lines)


def main():
    """Main function to generate all comparison tables."""
    # Load results
    results_path = 'all_models_results.csv'
    if not Path(results_path).exists():
        print(f"Results file not found: {results_path}")
        print("Please run collect_all_model_results.py first.")
        return

    df = pd.read_csv(results_path)

    # Create output directory
    output_dir = Path('paper/tables')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate tables for chunk_500 (main config)
    config = 'chunk_500'

    print(f"Generating tables for {config}...")

    # Table A: Binary datasets
    table_binary = generate_table_binary(df, config)
    with open(output_dir / 'table_binary_comparison.tex', 'w') as f:
        f.write(table_binary)
    print(f"  Saved: table_binary_comparison.tex")

    # Table B: All datasets
    table_all = generate_table_all(df, config)
    with open(output_dir / 'table_full_comparison.tex', 'w') as f:
        f.write(table_all)
    print(f"  Saved: table_full_comparison.tex")

    # Table VIII corrected
    table_viii = generate_table_viii_corrected(df, config)
    with open(output_dir / 'table_viii_drift_corrected.tex', 'w') as f:
        f.write(table_viii)
    print(f"  Saved: table_viii_drift_corrected.tex")

    # Summary statistics
    summary = generate_summary_statistics(df, config)
    print("\n" + summary)

    # Also generate for chunk_1000 for comparison
    config = 'chunk_1000'
    print(f"\nGenerating tables for {config}...")

    table_binary_1000 = generate_table_binary(df, config)
    with open(output_dir / 'table_binary_comparison_1000.tex', 'w') as f:
        f.write(table_binary_1000)
    print(f"  Saved: table_binary_comparison_1000.tex")

    print("\nDone!")


if __name__ == '__main__':
    main()
