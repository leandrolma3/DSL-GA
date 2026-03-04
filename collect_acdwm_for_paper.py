#!/usr/bin/env python3
"""Collect ACDWM results from experiments_unified and add to all_models_results.csv."""

import os
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("experiments_unified")

def collect_acdwm():
    records = []
    for config_dir in sorted(BASE.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.startswith("chunk_"):
            continue
        config_name = config_dir.name  # e.g. chunk_500
        for batch_dir in sorted(config_dir.iterdir()):
            if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
                continue
            for dataset_dir in sorted(batch_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                acdwm_file = dataset_dir / "run_1" / "acdwm_results.csv"
                if not acdwm_file.exists():
                    continue
                try:
                    df = pd.read_csv(acdwm_file)
                    if 'test_gmean' in df.columns:
                        gmean_values = df['test_gmean'].dropna()
                        gmean_mean = gmean_values.mean() if len(gmean_values) > 0 else np.nan
                    else:
                        continue
                except Exception:
                    continue
                records.append({
                    'dataset': dataset_dir.name,
                    'config': config_name,
                    'batch': batch_dir.name,
                    'ACDWM': gmean_mean,
                })
    return pd.DataFrame(records)

def main():
    acdwm_df = collect_acdwm()
    print(f"Collected {len(acdwm_df)} ACDWM records")
    print(f"Configs: {acdwm_df['config'].unique()}")
    print(f"Datasets: {acdwm_df['dataset'].nunique()}")

    # Load existing results
    all_results = pd.read_csv("all_models_results.csv")
    print(f"\nExisting columns: {all_results.columns.tolist()}")
    print(f"Existing rows: {len(all_results)}")

    # If ACDWM column already exists, drop it
    if 'ACDWM' in all_results.columns:
        all_results = all_results.drop(columns=['ACDWM'])

    # Merge ACDWM into all_results by dataset, batch, config
    merged = all_results.merge(
        acdwm_df[['dataset', 'config', 'batch', 'ACDWM']],
        on=['dataset', 'batch', 'config'],
        how='left'
    )

    print(f"\nAfter merge: {len(merged)} rows")
    print(f"ACDWM non-null: {merged['ACDWM'].notna().sum()}")

    # Save
    merged.to_csv("all_models_results.csv", index=False)
    print("Saved all_models_results.csv with ACDWM column")

    # Print summary for tables
    for cfg in ['chunk_500', 'chunk_1000']:
        subset = merged[(merged['config'] == cfg) & (merged['is_binary'] == True)]
        acdwm_valid = subset[subset['ACDWM'].notna()]
        print(f"\n{cfg} binary datasets with ACDWM: {len(acdwm_valid)}")
        if len(acdwm_valid) > 0:
            print(f"  Mean ACDWM: {acdwm_valid['ACDWM'].mean():.3f}")
            print(f"  Std ACDWM: {acdwm_valid['ACDWM'].std():.3f}")

        # Print per-dataset for table generation
        for _, row in acdwm_valid.sort_values('dataset').iterrows():
            print(f"  {row['dataset']}: {row['ACDWM']:.3f}")

if __name__ == '__main__':
    main()
