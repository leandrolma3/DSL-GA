#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnóstico ERulesD2S - Investigar Performance Baixa
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path("experiments_6chunks_phase2_gbml/batch_1")

DATASETS = [
    "SEA_Abrupt_Simple",
    "SEA_Abrupt_Chain",
    "SEA_Abrupt_Recurring",
    "AGRAWAL_Abrupt_Simple_Mild",
    "AGRAWAL_Abrupt_Simple_Severe",
    "AGRAWAL_Abrupt_Chain_Long",
    "RBF_Abrupt_Severe",
    "RBF_Abrupt_Blip",
    "STAGGER_Abrupt_Chain",
    "STAGGER_Abrupt_Recurring",
    "HYPERPLANE_Abrupt_Simple",
    "RANDOMTREE_Abrupt_Simple"
]

print("="*80)
print("DIAGNÓSTICO ERULESD2S - BATCH 1")
print("="*80)

for dataset in DATASETS:
    dataset_dir = BASE_DIR / dataset / "run_1"
    erulesd2s_file = dataset_dir / "erulesd2s_results.csv"

    print(f"\n{dataset}:")

    if not erulesd2s_file.exists():
        print("  [MISSING] erulesd2s_results.csv NOT FOUND")
        continue

    try:
        df = pd.read_csv(erulesd2s_file)

        print(f"  [OK] Found: {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")

        # Verificar valores
        if 'gmean' in df.columns:
            gmean_values = df['gmean'].values
            print(f"  G-mean values: {gmean_values}")
            print(f"  G-mean mean: {df['gmean'].mean():.4f}")
            print(f"  G-mean std: {df['gmean'].std():.4f}")

        if 'accuracy' in df.columns:
            acc_values = df['accuracy'].values
            print(f"  Accuracy values: {acc_values}")
            print(f"  Accuracy mean: {df['accuracy'].mean():.4f}")

        # Verificar se há zeros ou valores muito baixos
        if 'gmean' in df.columns:
            zeros = (df['gmean'] == 0).sum()
            very_low = (df['gmean'] < 0.1).sum()
            if zeros > 0:
                print(f"  [WARNING] {zeros} chunks with G-mean = 0")
            if very_low > 0:
                print(f"  [WARNING] {very_low} chunks with G-mean < 0.1")

        # Mostrar primeiras linhas
        print("\n  First 3 rows:")
        print(df.head(3).to_string(index=False))

    except Exception as e:
        print(f"  [ERROR] Failed to read: {e}")

print("\n" + "="*80)
print("RESUMO")
print("="*80)

# Carregar consolidated
consolidated_file = BASE_DIR / "batch_1_all_models_with_gbml.csv"
if consolidated_file.exists():
    df_all = pd.read_csv(consolidated_file)

    erulesd2s_data = df_all[df_all['model'] == 'ERulesD2S']

    print(f"\nERulesD2S total rows: {len(erulesd2s_data)}")
    print(f"ERulesD2S datasets: {erulesd2s_data['dataset'].nunique()}")
    print(f"\nERulesD2S test_gmean statistics:")
    print(erulesd2s_data['test_gmean'].describe())

    print(f"\nERulesD2S per dataset:")
    for dataset in erulesd2s_data['dataset'].unique():
        dataset_gmean = erulesd2s_data[erulesd2s_data['dataset'] == dataset]['test_gmean'].mean()
        print(f"  {dataset:<35}: {dataset_gmean:.4f}")

    # Comparar com outros modelos
    print(f"\n\nComparação com outros modelos (test_gmean):")
    for model in ['GBML', 'ACDWM', 'ARF', 'SRP', 'HAT', 'ERulesD2S']:
        model_gmean = df_all[df_all['model'] == model]['test_gmean'].mean()
        print(f"  {model:<12}: {model_gmean:.4f}")

print("\n" + "="*80)
