#!/usr/bin/env python3
"""
Generate Critical Difference (CD) diagram for the IEEE TKDE paper.

Uses Friedman test for statistical significance and Nemenyi post-hoc test
to identify significantly different classifiers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')


def friedman_test(data: np.ndarray) -> tuple:
    """
    Perform Friedman test on the data.

    Args:
        data: 2D array with shape (n_datasets, n_models)

    Returns:
        chi2: Friedman chi-squared statistic
        p_value: p-value of the test
    """
    stat, p = stats.friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])
    return stat, p


def nemenyi_cd(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    """
    Calculate the critical difference for Nemenyi post-hoc test.

    Args:
        n_models: Number of models being compared
        n_datasets: Number of datasets
        alpha: Significance level

    Returns:
        Critical difference value
    """
    # q-values for alpha=0.05 (from Demsar 2006, Table 5)
    q_alpha = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }

    if n_models not in q_alpha:
        # Approximate for larger values
        q = 2.326 + 0.1 * (n_models - 2)
    else:
        q = q_alpha[n_models]

    cd = q * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))
    return cd


def compute_ranks(data: np.ndarray) -> np.ndarray:
    """
    Compute average ranks for each model across datasets.
    Lower rank is better (1 = best performance).

    Args:
        data: 2D array with shape (n_datasets, n_models)

    Returns:
        Array of average ranks for each model
    """
    n_datasets, n_models = data.shape
    ranks = np.zeros((n_datasets, n_models))

    for i in range(n_datasets):
        row = data[i, :]
        # Handle NaN values
        valid_mask = ~np.isnan(row)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            valid_values = row[valid_mask]
            # Rank from highest (best) to lowest
            sorted_indices = np.argsort(-valid_values)
            for rank, idx in enumerate(sorted_indices, 1):
                ranks[i, valid_indices[idx]] = rank

            # Assign worst rank to NaN values
            nan_indices = np.where(~valid_mask)[0]
            for idx in nan_indices:
                ranks[i, idx] = n_models

    return np.mean(ranks, axis=0)


def draw_cd_diagram(model_names: list, avg_ranks: np.ndarray, cd: float,
                    output_path: str = 'critical_difference_diagram.pdf',
                    title: str = 'Critical Difference Diagram'):
    """
    Draw Critical Difference diagram.

    Args:
        model_names: List of model names
        avg_ranks: Average ranks for each model
        cd: Critical difference value
        output_path: Path to save the figure
        title: Title for the diagram
    """
    n_models = len(model_names)

    # Sort models by rank
    sorted_indices = np.argsort(avg_ranks)
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_ranks = avg_ranks[sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # Drawing parameters
    lowv = 0.5
    highv = n_models + 0.5
    height = 0.7

    # Draw horizontal axis
    ax.hlines(height, lowv, highv, colors='black', linewidth=1)

    # Draw tick marks
    for i in range(1, n_models + 1):
        ax.vlines(i, height - 0.05, height + 0.05, colors='black', linewidth=1)
        ax.text(i, height + 0.08, str(i), ha='center', va='bottom', fontsize=10)

    # Draw model names and their ranks
    half = n_models // 2
    top_models = list(range(half))
    bottom_models = list(range(half, n_models))

    # Top models (better ranks)
    for idx, i in enumerate(top_models):
        rank = sorted_ranks[i]
        name = sorted_names[i]

        # Draw line to axis
        y_offset = 0.3 + idx * 0.15
        ax.vlines(rank, height, height + y_offset, colors='gray', linewidth=0.5)
        ax.text(rank, height + y_offset + 0.02, f'{name} ({rank:.2f})',
                ha='center', va='bottom', fontsize=9, fontweight='bold' if idx == 0 else 'normal')

    # Bottom models (worse ranks)
    for idx, i in enumerate(bottom_models):
        rank = sorted_ranks[i]
        name = sorted_names[i]

        # Draw line to axis
        y_offset = 0.3 + idx * 0.15
        ax.vlines(rank, height - y_offset, height, colors='gray', linewidth=0.5)
        ax.text(rank, height - y_offset - 0.02, f'{name} ({rank:.2f})',
                ha='center', va='top', fontsize=9)

    # Draw CD bar
    cd_x = lowv + 0.2
    cd_y = height + 0.6
    ax.hlines(cd_y, cd_x, cd_x + cd, colors='red', linewidth=2)
    ax.vlines(cd_x, cd_y - 0.03, cd_y + 0.03, colors='red', linewidth=2)
    ax.vlines(cd_x + cd, cd_y - 0.03, cd_y + 0.03, colors='red', linewidth=2)
    ax.text(cd_x + cd/2, cd_y + 0.05, f'CD = {cd:.2f}', ha='center', va='bottom',
            fontsize=10, color='red')

    # Draw cliques (groups of non-significantly different models)
    cliques = find_cliques(sorted_ranks, cd)

    clique_y = height - 0.5
    for clique in cliques:
        if len(clique) > 1:
            start_rank = sorted_ranks[min(clique)]
            end_rank = sorted_ranks[max(clique)]
            ax.hlines(clique_y, start_rank, end_rank, colors='black', linewidth=3)
            clique_y -= 0.08

    # Set axis limits
    ax.set_xlim(0, n_models + 1)
    ax.set_ylim(height - 1, height + 1)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved CD diagram to {output_path}")


def find_cliques(ranks: np.ndarray, cd: float) -> list:
    """
    Find cliques of models that are not significantly different.

    Args:
        ranks: Sorted array of average ranks
        cd: Critical difference value

    Returns:
        List of cliques (each clique is a list of indices)
    """
    n = len(ranks)
    cliques = []

    for i in range(n):
        clique = [i]
        for j in range(i + 1, n):
            if ranks[j] - ranks[i] < cd:
                clique.append(j)
            else:
                break

        # Only add if it's a new clique
        if len(clique) > 1:
            is_subset = False
            for existing in cliques:
                if set(clique).issubset(set(existing)):
                    is_subset = True
                    break
            if not is_subset:
                cliques.append(clique)

    return cliques


def main():
    """Main function to generate Critical Difference diagram."""
    # Load results
    results_path = 'all_models_results.csv'
    if not Path(results_path).exists():
        print(f"Results file not found: {results_path}")
        print("Please run collect_all_model_results.py first.")
        return

    df = pd.read_csv(results_path)

    # Exclude removed datasets
    EXCLUDED_DATASETS = ['CovType', 'IntelLabSensors', 'PokerHand', 'Shuttle']
    df = df[~df['dataset'].isin(EXCLUDED_DATASETS)]

    # Filter for chunk_500 config and binary datasets
    config = 'chunk_500'
    df_filtered = df[(df['config'] == config) & (df['is_binary'] == True)].copy()

    print(f"Processing {config} configuration...")
    print(f"Binary datasets: {len(df_filtered)}")

    # Models to compare
    models = ['EGIS', 'ARF', 'HAT', 'SRP', 'ROSE', 'eRulesD2S', 'CDCMS']

    # Extract G-Mean values
    data_matrix = df_filtered[models].values
    n_datasets, n_models = data_matrix.shape

    print(f"Data matrix shape: {n_datasets} datasets x {n_models} models")

    # Handle NaN values - replace with 0 (worst performance)
    data_filled = np.nan_to_num(data_matrix, nan=0)

    # Compute average ranks
    avg_ranks = compute_ranks(data_filled)

    print("\nAverage Ranks (lower is better):")
    for i, m in enumerate(models):
        print(f"  {m:12s}: {avg_ranks[i]:.3f}")

    # Perform Friedman test
    chi2, p_value = friedman_test(data_filled)
    print(f"\nFriedman test:")
    print(f"  Chi-squared: {chi2:.2f}")
    print(f"  p-value: {p_value:.2e}")

    # Calculate Critical Difference
    cd = nemenyi_cd(n_models, n_datasets, alpha=0.05)
    print(f"\nCritical Difference (alpha=0.05): {cd:.3f}")

    # Create output directory
    output_dir = Path('paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate CD diagram
    draw_cd_diagram(
        models, avg_ranks, cd,
        output_path=str(output_dir / 'fig_critical_difference.pdf'),
        title=f'Critical Difference Diagram (Chunk Size {config.split("_")[1]}, {n_datasets} datasets)'
    )

    # Also save as PNG for easier viewing
    draw_cd_diagram(
        models, avg_ranks, cd,
        output_path=str(output_dir / 'fig_critical_difference.png'),
        title=f'Critical Difference Diagram (Chunk Size {config.split("_")[1]}, {n_datasets} datasets)'
    )

    # Generate statistical summary
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)

    # Identify significantly different pairs
    print("\nSignificantly different pairs (rank difference > CD):")
    for i in range(n_models):
        for j in range(i+1, n_models):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            if diff > cd:
                better = models[i] if avg_ranks[i] < avg_ranks[j] else models[j]
                worse = models[j] if avg_ranks[i] < avg_ranks[j] else models[i]
                print(f"  {better} significantly better than {worse} (diff={diff:.3f})")

    # Save summary to file
    summary_path = output_dir / 'statistical_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Statistical Analysis Summary\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"Number of datasets: {n_datasets}\n")
        f.write(f"Number of models: {n_models}\n\n")
        f.write(f"Friedman test:\n")
        f.write(f"  Chi-squared: {chi2:.2f}\n")
        f.write(f"  p-value: {p_value:.2e}\n\n")
        f.write(f"Critical Difference (alpha=0.05): {cd:.3f}\n\n")
        f.write(f"Average Ranks:\n")
        for i, m in enumerate(models):
            f.write(f"  {m:12s}: {avg_ranks[i]:.3f}\n")

    print(f"\nSaved statistical summary to {summary_path}")

    # Generate multiclass CD diagram
    generate_multiclass_cd(df, config, output_dir)


def generate_multiclass_cd(df: pd.DataFrame, config: str, output_dir: Path):
    """Generate Critical Difference diagram for multiclass datasets."""
    print("\n" + "="*60)
    print("MULTICLASS CRITICAL DIFFERENCE DIAGRAM")
    print("="*60)

    # Filter for multiclass datasets
    df_multi = df[(df['config'] == config) & (df['is_binary'] == False)].copy()

    print(f"Multiclass datasets: {len(df_multi)}")
    if len(df_multi) > 0:
        print(f"Datasets: {df_multi['dataset'].tolist()}")

    if len(df_multi) < 3:
        print("Not enough multiclass datasets for Friedman test (need >= 3). Skipping.")
        return

    # Models for multiclass (no ACDWM, no CDCMS)
    models_multi = ['EGIS', 'ARF', 'SRP', 'HAT', 'ROSE', 'eRulesD2S']

    # Check which columns exist
    available = [m for m in models_multi if m in df_multi.columns]
    if len(available) < len(models_multi):
        missing = set(models_multi) - set(available)
        print(f"Warning: Missing columns: {missing}")
        models_multi = available

    if len(models_multi) < 2:
        print("Not enough models available. Skipping.")
        return

    # Extract data matrix
    data_matrix = df_multi[models_multi].values
    n_datasets, n_models = data_matrix.shape

    print(f"Data matrix shape: {n_datasets} datasets x {n_models} models")

    # Handle NaN values
    data_filled = np.nan_to_num(data_matrix, nan=0)

    # Compute average ranks
    avg_ranks = compute_ranks(data_filled)

    print("\nAverage Ranks (lower is better):")
    for i, m in enumerate(models_multi):
        print(f"  {m:12s}: {avg_ranks[i]:.3f}")

    # Perform Friedman test
    chi2, p_value = friedman_test(data_filled)
    print(f"\nFriedman test:")
    print(f"  Chi-squared: {chi2:.2f}")
    print(f"  p-value: {p_value:.2e}")

    # Calculate Critical Difference
    cd = nemenyi_cd(n_models, n_datasets, alpha=0.05)
    print(f"\nCritical Difference (alpha=0.05): {cd:.3f}")

    # Generate CD diagram
    draw_cd_diagram(
        models_multi, avg_ranks, cd,
        output_path=str(output_dir / 'fig_critical_difference_multiclass.pdf'),
        title=f'Critical Difference Diagram - Multiclass (Chunk Size {config.split("_")[1]}, {n_datasets} datasets)'
    )

    # Also save as PNG
    draw_cd_diagram(
        models_multi, avg_ranks, cd,
        output_path=str(output_dir / 'fig_critical_difference_multiclass.png'),
        title=f'Critical Difference Diagram - Multiclass (Chunk Size {config.split("_")[1]}, {n_datasets} datasets)'
    )

    # Print significantly different pairs
    print("\nSignificantly different pairs (rank difference > CD):")
    for i in range(n_models):
        for j in range(i+1, n_models):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            if diff > cd:
                better = models_multi[i] if avg_ranks[i] < avg_ranks[j] else models_multi[j]
                worse = models_multi[j] if avg_ranks[i] < avg_ranks[j] else models_multi[i]
                print(f"  {better} significantly better than {worse} (diff={diff:.3f})")


if __name__ == '__main__':
    main()
