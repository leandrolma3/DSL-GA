#!/usr/bin/env python3
"""
Batch Drift Detection Analysis.

For each EGIS config x dataset, loads chunk_metrics.json and detects
performance drifts. Saves paper_data/drift_detection_summary.csv.

Author: Automated Analysis
Date: 2026-02-23
"""

import os
import sys
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Import from existing script
from analyze_standard_drift import detect_performance_drifts

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENTS_BASE = Path("experiments_unified")
OUTPUT_DIR = Path("paper_data")
EXCLUDED_DATASETS = ['CovType', 'IntelLabSensors', 'PokerHand', 'Shuttle']

EXPERIMENT_CONFIGS = {
    'chunk_500':            {'chunk_size': 500,  'penalty': 0.0, 'label': 'EXP-500-NP'},
    'chunk_500_penalty':    {'chunk_size': 500,  'penalty': 0.1, 'label': 'EXP-500-P'},
    'chunk_500_penalty_03': {'chunk_size': 500,  'penalty': 0.3, 'label': 'EXP-500-P03'},
    'chunk_1000':           {'chunk_size': 1000, 'penalty': 0.0, 'label': 'EXP-1000-NP'},
    'chunk_1000_penalty':   {'chunk_size': 1000, 'penalty': 0.1, 'label': 'EXP-1000-P'},
    'chunk_2000':           {'chunk_size': 2000, 'penalty': 0.0, 'label': 'EXP-2000-NP'},
    'chunk_2000_penalty':   {'chunk_size': 2000, 'penalty': 0.1, 'label': 'EXP-2000-P'},
}

DATASET_METADATA = {
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


def main():
    print("=" * 60)
    print("BATCH DRIFT DETECTION ANALYSIS")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for config_name, config in EXPERIMENT_CONFIGS.items():
        config_dir = EXPERIMENTS_BASE / config_name
        if not config_dir.exists():
            print(f"  SKIP {config_name}: directory not found")
            continue

        for batch_dir in sorted(config_dir.iterdir()):
            if not batch_dir.is_dir() or not batch_dir.name.startswith('batch_'):
                continue

            for dataset_dir in sorted(batch_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue

                dataset_name = dataset_dir.name
                if dataset_name in EXCLUDED_DATASETS:
                    continue

                run_dir = dataset_dir / "run_1"
                metrics_file = run_dir / "chunk_metrics.json"

                if not metrics_file.exists():
                    continue

                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        chunk_metrics = data
                    else:
                        continue

                    # Detect drifts
                    drift_points = detect_performance_drifts(chunk_metrics, threshold=0.10)

                    # Extract per-chunk gmean
                    gmeans = []
                    for cm in chunk_metrics:
                        g = cm.get('test_gmean')
                        if g is not None and not np.isnan(g):
                            gmeans.append(float(g))

                    drift_type = DATASET_METADATA.get(dataset_name, 'unknown')

                    records.append({
                        'config': config_name,
                        'config_label': config['label'],
                        'chunk_size': config['chunk_size'],
                        'dataset': dataset_name,
                        'drift_type': drift_type,
                        'n_chunks': len(gmeans),
                        'gmean_mean': np.mean(gmeans) if gmeans else np.nan,
                        'gmean_std': np.std(gmeans) if gmeans else np.nan,
                        'gmean_min': np.min(gmeans) if gmeans else np.nan,
                        'gmean_max': np.max(gmeans) if gmeans else np.nan,
                        'n_drifts_detected': len(drift_points),
                        'drift_points': str(drift_points),
                    })

                except Exception as e:
                    logger.warning(f"Error processing {metrics_file}: {e}")

    df = pd.DataFrame(records)
    output_path = OUTPUT_DIR / "drift_detection_summary.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path} ({len(df)} records)")
    print(f"Configs: {df['config_label'].nunique()}")
    print(f"Datasets: {df['dataset'].nunique()}")

    # Summary
    print("\nDrifts detected per config:")
    for cfg in sorted(df['config_label'].unique()):
        cfg_df = df[df['config_label'] == cfg]
        print(f"  {cfg}: avg={cfg_df['n_drifts_detected'].mean():.1f}, "
              f"max={cfg_df['n_drifts_detected'].max()}, "
              f"datasets={len(cfg_df)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
