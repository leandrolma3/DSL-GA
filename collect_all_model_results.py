#!/usr/bin/env python3
"""
Collect results from all models in experiments_unified.

This script collects G-Mean results from:
- EGIS (chunk_metrics.json)
- ARF, HAT, SRP (river_*_results.csv)
- ROSE (rose_original_results.csv)
- eRulesD2S (erulesd2s_results.csv)
- CDCMS (cdcms_results/chunk_metrics.json at dataset level)
- ACDWM (acdwm_results.csv)

Output: Consolidated CSV with mean G-Mean per dataset for each model.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# Define drift type classification based on dataset name
def classify_drift_type(dataset_name: str) -> str:
    """Classify drift type based on dataset name."""
    name_lower = dataset_name.lower()

    # Real-world datasets
    real_datasets = ['electricity', 'covtype', 'shuttle', 'intellabsensors',
                     'pokerhand', 'assetnegotiation']
    for real in real_datasets:
        if real in name_lower:
            return 'real'

    # Stationary
    if 'stationary' in name_lower:
        return 'stationary'

    # Noisy datasets
    if 'noise' in name_lower:
        return 'noisy'

    # Gradual drift
    if 'gradual' in name_lower:
        return 'gradual'

    # Abrupt drift
    if 'abrupt' in name_lower:
        return 'abrupt'

    # Base streams (no drift pattern specified)
    return 'base'


def is_binary_dataset(dataset_name: str) -> bool:
    """Check if dataset is binary (not multiclass)."""
    multiclass_datasets = [
        'covtype', 'shuttle', 'intellabsensors', 'pokerhand',
        'led', 'waveform'  # LED has 10 classes, WAVEFORM has 3
    ]
    name_lower = dataset_name.lower()
    for mc in multiclass_datasets:
        if mc in name_lower:
            return False
    return True


def read_egis_results(run_path: Path) -> Optional[float]:
    """Read EGIS results from chunk_metrics.json."""
    filepath = run_path / 'chunk_metrics.json'
    if not filepath.exists():
        return None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            gmean_values = [chunk.get('test_gmean', 0) for chunk in data
                           if isinstance(chunk, dict)]
        elif isinstance(data, dict):
            gmean_values = data.get('test_gmean', [])
            if not isinstance(gmean_values, list):
                gmean_values = [gmean_values]
        else:
            return None

        # Filter NaN/None values
        gmean_values = [g for g in gmean_values if g is not None and not np.isnan(g)]
        if gmean_values:
            return np.mean(gmean_values)
    except Exception as e:
        print(f"Error reading EGIS results from {filepath}: {e}")
    return None


def read_acdwm_results(run_path: Path) -> Optional[float]:
    """Read ACDWM baseline results from acdwm_results.csv."""
    filepath = run_path / 'acdwm_results.csv'
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(filepath)
        if 'test_gmean' in df.columns:
            gmean_values = df['test_gmean'].values
            if len(gmean_values) > 1:
                # Skip first value if it's 0 (initialization)
                if gmean_values[0] == 0:
                    gmean_values = gmean_values[1:]
            return np.mean(gmean_values)
    except Exception as e:
        print(f"Error reading ACDWM results from {filepath}: {e}")
    return None


def read_river_results(run_path: Path, model: str) -> Optional[float]:
    """Read River model results (ARF, HAT, SRP)."""
    filepath = run_path / f'river_{model}_results.csv'
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(filepath)
        if 'test_gmean' in df.columns:
            gmean_values = df['test_gmean'].values
            return np.mean(gmean_values)
    except Exception as e:
        print(f"Error reading {model} results from {filepath}: {e}")
    return None


def read_rose_results(run_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Read ROSE results (original and chunk_eval)."""
    rose_original = None
    rose_ce = None

    # ROSE Original
    filepath = run_path / 'rose_original_results.csv'
    if filepath.exists():
        try:
            df = pd.read_csv(filepath)
            if 'G-Mean' in df.columns:
                # Convert to numeric, coercing errors to NaN
                gmean_values = pd.to_numeric(df['G-Mean'], errors='coerce').dropna().values
                if len(gmean_values) > 0:
                    rose_original = float(np.mean(gmean_values))
        except Exception as e:
            print(f"Error reading ROSE original from {filepath}: {e}")

    # ROSE Chunk Eval
    filepath = run_path / 'rose_chunk_eval_results.csv'
    if filepath.exists():
        try:
            df = pd.read_csv(filepath)
            if 'G-Mean' in df.columns:
                gmean_values = pd.to_numeric(df['G-Mean'], errors='coerce').dropna().values
                if len(gmean_values) > 0:
                    rose_ce = float(np.mean(gmean_values))
        except Exception as e:
            print(f"Error reading ROSE chunk eval from {filepath}: {e}")

    return rose_original, rose_ce


def read_erulesd2s_results(run_path: Path) -> Optional[float]:
    """Read eRulesD2S results."""
    filepath = run_path / 'erulesd2s_results.csv'
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(filepath)
        if 'test_gmean' in df.columns:
            gmean_values = df['test_gmean'].values
            return np.mean(gmean_values)
    except Exception as e:
        print(f"Error reading eRulesD2S from {filepath}: {e}")
    return None


def read_cdcms_results(dataset_path: Path) -> Optional[float]:
    """Read CDCMS results from cdcms_results/chunk_metrics.json."""
    cdcms_path = dataset_path / 'cdcms_results' / 'chunk_metrics.json'
    if not cdcms_path.exists():
        return None
    try:
        with open(cdcms_path, 'r') as f:
            data = json.load(f)

        # Extract prequential_gmean or holdout_gmean
        gmean_values = []
        for chunk in data:
            if isinstance(chunk, dict):
                # Prefer holdout_gmean, fall back to prequential_gmean
                gmean = chunk.get('holdout_gmean')
                if gmean is None or (isinstance(gmean, float) and np.isnan(gmean)):
                    gmean = chunk.get('prequential_gmean')
                if gmean is not None and not (isinstance(gmean, float) and np.isnan(gmean)):
                    gmean_values.append(gmean)

        if gmean_values:
            return np.mean(gmean_values)
    except Exception as e:
        print(f"Error reading CDCMS from {cdcms_path}: {e}")
    return None


EXCLUDED_DATASETS = ['CovType', 'IntelLabSensors', 'PokerHand', 'Shuttle']


def collect_results_for_config(config_path: Path) -> pd.DataFrame:
    """Collect results for a specific configuration (chunk_500, chunk_1000, etc.)."""
    results = []

    # Iterate through batches
    for batch_dir in sorted(config_path.iterdir()):
        if not batch_dir.is_dir() or not batch_dir.name.startswith('batch_'):
            continue

        # Iterate through datasets
        for dataset_dir in sorted(batch_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name

            # Skip excluded datasets
            if any(excl in dataset_name for excl in EXCLUDED_DATASETS):
                continue

            # Skip _temp directories (leftover from repair)
            if dataset_name.endswith('_temp'):
                continue

            run_path = dataset_dir / 'run_1'

            # Skip if no run_1 directory (placeholder)
            if not run_path.exists():
                continue

            # Collect results from all models
            egis = read_egis_results(run_path)
            arf = read_river_results(run_path, 'ARF')
            hat = read_river_results(run_path, 'HAT')
            srp = read_river_results(run_path, 'SRP')
            rose, rose_ce = read_rose_results(run_path)
            erulesd2s = read_erulesd2s_results(run_path)
            cdcms = read_cdcms_results(dataset_dir)
            acdwm = read_acdwm_results(run_path)

            # Classify drift type
            drift_type = classify_drift_type(dataset_name)
            is_binary = is_binary_dataset(dataset_name)

            results.append({
                'dataset': dataset_name,
                'batch': batch_dir.name,
                'drift_type': drift_type,
                'is_binary': is_binary,
                'EGIS': egis,
                'ARF': arf,
                'HAT': hat,
                'SRP': srp,
                'ROSE': rose,
                'ROSE_CE': rose_ce,
                'eRulesD2S': erulesd2s,
                'CDCMS': cdcms,
                'ACDWM': acdwm
            })

    return pd.DataFrame(results)


def main():
    """Main function to collect all model results."""
    base_path = Path('experiments_unified')

    # Configurations to process
    configs = ['chunk_500', 'chunk_1000', 'chunk_500_penalty', 'chunk_1000_penalty']

    all_results = []

    for config in configs:
        config_path = base_path / config
        if not config_path.exists():
            print(f"Config path not found: {config_path}")
            continue

        print(f"Processing {config}...")
        df = collect_results_for_config(config_path)
        df['config'] = config
        all_results.append(df)

        # Print summary
        print(f"  Datasets found: {len(df)}")
        print(f"  Binary datasets: {df['is_binary'].sum()}")
        print(f"  Models with data:")
        for model in ['EGIS', 'ARF', 'HAT', 'SRP', 'ROSE', 'eRulesD2S', 'CDCMS', 'ACDWM']:
            count = df[model].notna().sum()
            print(f"    {model}: {count}/{len(df)}")
        print()

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Save full results
        output_path = 'all_models_results.csv'
        combined_df.to_csv(output_path, index=False)
        print(f"Saved combined results to {output_path}")

        # Create summary by config
        summary_path = 'all_models_results_summary.csv'
        summary = combined_df.groupby(['config', 'drift_type']).agg({
            'EGIS': ['mean', 'std', 'count'],
            'ARF': ['mean', 'std', 'count'],
            'HAT': ['mean', 'std', 'count'],
            'SRP': ['mean', 'std', 'count'],
            'ROSE': ['mean', 'std', 'count'],
            'eRulesD2S': ['mean', 'std', 'count'],
            'CDCMS': ['mean', 'std', 'count'],
            'ACDWM': ['mean', 'std', 'count']
        }).round(4)
        summary.to_csv(summary_path)
        print(f"Saved summary to {summary_path}")

        # Print overall statistics
        print("\n" + "="*80)
        print("OVERALL STATISTICS")
        print("="*80)

        for config in configs:
            if config not in combined_df['config'].values:
                continue
            config_df = combined_df[combined_df['config'] == config]
            print(f"\n{config}:")
            print("-"*40)

            # Binary datasets only
            binary_df = config_df[config_df['is_binary'] == True]
            print(f"  Binary datasets: {len(binary_df)}")

            # Calculate mean G-Mean per model
            for model in ['EGIS', 'ARF', 'HAT', 'SRP', 'ROSE', 'eRulesD2S', 'CDCMS', 'ACDWM']:
                values = binary_df[model].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    print(f"    {model}: {mean_val:.4f} +/- {std_val:.4f} (n={len(values)})")

        return combined_df

    return None


if __name__ == '__main__':
    main()
