#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CÉLULA 11 CORRIGIDA - Consolidar Resultados (com todas as correções aplicadas)

Correções aplicadas:
1. Ignorar rule_diff_analysis_*_matrix.csv
2. Filtrar ERulesD2S para chunks 0-4
3. Adicionar Bonferroni correction e comparações pairwise completas
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(WORK_DIR) / "experiments_6chunks_phase2_gbml" / "batch_1"

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
print("RESULTS CONSOLIDATION - BATCH 1 (WITH GBML)")
print("="*80)

all_results = []

# ============================================================================
# CARREGAR RESULTADOS COMPARATIVOS (River, ACDWM, ERulesD2S)
# ============================================================================

for dataset in DATASETS:
    dataset_dir = BASE_DIR / dataset / "run_1"

    print(f"\n{dataset}:")

    # CORRECAO 1: Listar CSVs ignorando rule_diff_analysis e matrices
    result_files = [
        f for f in dataset_dir.glob("*.csv")
        if "chunk_" not in f.name
        and "comparative" not in f.name
        and "rule_diff" not in f.name      # NOVO: Ignorar rule_diff_analysis
        and "_matrix" not in f.name         # NOVO: Ignorar matrizes
    ]

    for csv_file in result_files:
        try:
            df = pd.read_csv(csv_file)
            original_len = len(df)

            # CORRECAO 2: ERULESD2S - Converter formato e filtrar chunks
            if 'ERulesD2S' in csv_file.name or 'erulesd2s' in csv_file.name.lower():
                # ERulesD2S so tem test metrics (sem train)
                # Renomear colunas para padronizar
                if 'accuracy' in df.columns and 'test_accuracy' not in df.columns:
                    df.rename(columns={
                        'accuracy': 'test_accuracy',
                        'gmean': 'test_gmean',
                        'f1_weighted': 'test_f1'
                    }, inplace=True)

                    # Adicionar colunas train vazias (ERulesD2S nao tem)
                    df['train_gmean'] = np.nan
                    df['train_accuracy'] = np.nan
                    df['train_f1'] = np.nan

                    # Ajustar numeracao de chunks (ERulesD2S usa 1-6, outros usam 0-4)
                    if 'chunk' in df.columns:
                        df['chunk'] = df['chunk'] - 1  # 1-6 -> 0-5

                        # NOVO: Filtrar apenas chunks 0-4 (treino)
                        df = df[df['chunk'] <= 4].copy()

                print(f"  - {csv_file.name}: {len(df)} rows (ERulesD2S - converting format)", end="")
                if original_len > len(df):
                    print(f" [filtered {original_len} -> {len(df)} chunks]")
                else:
                    print()
            else:
                print(f"  - {csv_file.name}: {len(df)} rows")

            # Adicionar coluna de dataset
            if 'dataset' not in df.columns:
                df['dataset'] = dataset

            all_results.append(df)

        except Exception as e:
            print(f"  ! Error reading {csv_file.name}: {e}")

# ============================================================================
# CARREGAR RESULTADOS GBML (chunk_metrics.json)
# ============================================================================

print(f"\n{'='*60}")
print("Loading GBML results (chunk_metrics.json)...")
print('='*60)

for dataset in DATASETS:
    dataset_dir = BASE_DIR / dataset / "run_1"
    chunk_metrics_file = dataset_dir / "chunk_metrics.json"

    if chunk_metrics_file.exists():
        try:
            import json
            with open(chunk_metrics_file, 'r') as f:
                gbml_data = json.load(f)

            # Converter para DataFrame
            df_gbml = pd.DataFrame(gbml_data)

            # Renomear colunas para padronizar
            df_gbml['model'] = 'GBML'
            df_gbml['dataset'] = dataset
            df_gbml['train_size'] = 1000
            df_gbml['test_size'] = 1000

            # Adicionar colunas faltantes
            if 'train_accuracy' not in df_gbml.columns:
                df_gbml['train_accuracy'] = np.nan
            if 'test_accuracy' not in df_gbml.columns:
                df_gbml['test_accuracy'] = np.nan
            if 'train_f1' not in df_gbml.columns:
                df_gbml['train_f1'] = np.nan
            if 'test_f1' not in df_gbml.columns:
                if 'test_f1_orig' in df_gbml.columns:
                    df_gbml['test_f1'] = df_gbml['test_f1_orig']
                else:
                    df_gbml['test_f1'] = np.nan

            all_results.append(df_gbml)
            print(f"  - {dataset}: {len(df_gbml)} rows (GBML)")

        except Exception as e:
            print(f"  ! Error loading GBML from {dataset}: {e}")
    else:
        print(f"  ! {dataset}: chunk_metrics.json NOT FOUND")

# ============================================================================
# CONSOLIDAR E SALVAR
# ============================================================================

if all_results:
    consolidated = pd.concat(all_results, ignore_index=True)

    # Salvar consolidado
    consolidated_file = BASE_DIR / "batch_1_all_models_with_gbml.csv"
    consolidated.to_csv(consolidated_file, index=False)

    print(f"\n{'='*80}")
    print("CONSOLIDATED RESULTS (WITH GBML)")
    print(f"{'='*80}")
    print(f"Total rows: {len(consolidated)}")
    print(f"Models: {sorted(consolidated['model'].unique())}")
    print(f"Datasets: {sorted(consolidated['dataset'].unique())}")
    print(f"\nFile saved: {consolidated_file}")

    # ========================================================================
    # ESTATISTICAS RESUMIDAS
    # ========================================================================

    print(f"\n{'='*80}")
    print("STATISTICS BY MODEL (Mean +/- Std Dev)")
    print(f"{'='*80}")

    summary = consolidated.groupby('model').agg({
        'train_gmean': ['mean', 'std', 'count'],
        'test_gmean': ['mean', 'std'],
        'train_accuracy': ['mean', 'std'],
        'test_accuracy': ['mean', 'std']
    }).round(4)

    print(summary)

    # ========================================================================
    # RANKING POR TEST G-MEAN
    # ========================================================================

    print(f"\n{'='*80}")
    print("RANKING BY TEST G-MEAN (Best to Worst)")
    print(f"{'='*80}")

    ranking = consolidated.groupby('model')['test_gmean'].mean().sort_values(ascending=False)
    for rank, (model, gmean) in enumerate(ranking.items(), 1):
        print(f"{rank}. {model:15s} test_gmean = {gmean:.4f}")

    # ========================================================================
    # COMPARACAO POR DATASET
    # ========================================================================

    print(f"\n{'='*80}")
    print("COMPARISON BY DATASET (Test G-mean)")
    print(f"{'='*80}")

    pivot_table = consolidated.pivot_table(
        values='test_gmean',
        index='dataset',
        columns='model',
        aggfunc='mean'
    ).round(4)

    print(pivot_table)

    # ========================================================================
    # TESTES ESTATISTICOS (CORRECAO 3: COM BONFERRONI E PAIRWISE COMPLETO)
    # ========================================================================

    print(f"\n{'='*80}")
    print("STATISTICAL TESTS (on Test G-mean)")
    print(f"{'='*80}")
    print("Reference: Demsar (2006), Garcia & Herrera (2008)")
    print("Using TEST metrics to evaluate generalization performance")

    # Preparar dados para testes (matriz: datasets x modelos)
    models_to_compare = ['GBML', 'ACDWM', 'ARF', 'SRP', 'HAT']

    # Filtrar apenas modelos que temos dados e que tem test_gmean
    models_available = []
    for model in models_to_compare:
        model_data = consolidated[
            (consolidated['model'] == model) &
            (consolidated['test_gmean'].notna())
        ]
        if len(model_data) > 0:
            models_available.append(model)

    models_to_compare = models_available

    # Criar matriz de test_gmean por dataset e modelo
    datasets_for_test = consolidated['dataset'].unique()
    data_matrix = []

    for model in models_to_compare:
        model_scores = []
        for dataset in datasets_for_test:
            score = consolidated[
                (consolidated['model'] == model) &
                (consolidated['dataset'] == dataset)
            ]['test_gmean'].mean()

            if not np.isnan(score):
                model_scores.append(score)
            else:
                model_scores.append(0.0)  # Placeholder para datasets sem dados

        data_matrix.append(model_scores)

    data_matrix = np.array(data_matrix).T  # Transpor: datasets x modelos

    print(f"\nData matrix (test_gmean):")
    print(f"Datasets: {len(data_matrix)} | Models: {len(models_to_compare)}")
    print(f"Models compared: {models_to_compare}")

    # ========================================================================
    # FRIEDMAN TEST
    # ========================================================================

    print(f"\n--- Friedman Test ---")
    print("H0: All models have equivalent performance")
    print("H1: At least one model differs from the others")

    try:
        stat, p_value = friedmanchisquare(*data_matrix.T)
        print(f"\nFriedman statistic: {stat:.4f}")
        print(f"p-value: {p_value:.6f}")

        alpha = 0.05
        if p_value < alpha:
            print(f"Result: REJECT H0 (p < {alpha})")
            print("Conclusion: Significant difference exists between models")
        else:
            print(f"Result: FAIL TO REJECT H0 (p >= {alpha})")
            print("Conclusion: No significant difference between models")
    except Exception as e:
        print(f"Error in Friedman test: {e}")

    # ========================================================================
    # WILCOXON SIGNED-RANK TEST (ALL PAIRWISE COMPARISONS)
    # ========================================================================

    print(f"\n--- Wilcoxon Signed-Rank Tests (All Pairwise Comparisons) ---")

    # Número de comparações para Bonferroni
    num_comparisons = len(models_to_compare) * (len(models_to_compare) - 1) // 2
    alpha_bonferroni = 0.05 / num_comparisons

    print(f"Total pairwise comparisons: {num_comparisons}")
    print(f"Alpha (original): 0.05")
    print(f"Alpha (Bonferroni corrected): {alpha_bonferroni:.6f}")
    print()

    # Resultados das comparações
    comparison_results = []

    for i in range(len(models_to_compare)):
        for j in range(i+1, len(models_to_compare)):
            model1 = models_to_compare[i]
            model2 = models_to_compare[j]

            scores1 = data_matrix[:, i]
            scores2 = data_matrix[:, j]

            try:
                # Wilcoxon test (two-sided)
                stat, p_value = wilcoxon(scores1, scores2, alternative='two-sided')

                # Calcular diferença média
                mean_diff = np.mean(scores1 - scores2)

                # Determinar vencedor
                if p_value < alpha_bonferroni:
                    if mean_diff > 0:
                        winner = model1
                        significance = "SIGNIFICANT"
                    else:
                        winner = model2
                        significance = "SIGNIFICANT"
                else:
                    winner = "TIE"
                    significance = "not significant"

                comparison_results.append({
                    'model1': model1,
                    'model2': model2,
                    'p_value': p_value,
                    'mean_diff': mean_diff,
                    'winner': winner,
                    'significance': significance
                })

            except Exception as e:
                comparison_results.append({
                    'model1': model1,
                    'model2': model2,
                    'p_value': np.nan,
                    'mean_diff': np.nan,
                    'winner': 'ERROR',
                    'significance': str(e)
                })

    # Ordenar por p-value (menor primeiro)
    comparison_results.sort(key=lambda x: x['p_value'] if not np.isnan(x['p_value']) else 1.0)

    # Imprimir resultados
    print(f"{'Model 1':<15} vs {'Model 2':<15} | {'p-value':>10} | {'Mean Diff':>10} | {'Winner':<15} | Significance")
    print("-" * 95)

    for result in comparison_results:
        m1 = result['model1']
        m2 = result['model2']
        p = result['p_value']
        diff = result['mean_diff']
        winner = result['winner']
        sig = result['significance']

        if not np.isnan(p):
            sig_marker = " [SIGNIFICANT]" if sig == "SIGNIFICANT" else ""
            print(f"{m1:<15} vs {m2:<15} | {p:>10.6f} | {diff:>+10.4f} | {winner:<15} | {sig}{sig_marker}")
        else:
            print(f"{m1:<15} vs {m2:<15} | {'ERROR':>10} | {'N/A':>10} | {winner:<15} | {sig}")

    # ========================================================================
    # RANKING FINAL COM SIGNIFICÂNCIA
    # ========================================================================

    print(f"\n--- Final Ranking (with Statistical Significance) ---")
    print()

    # Calcular média e ranking
    model_means = {}
    for idx, model in enumerate(models_to_compare):
        model_means[model] = np.mean(data_matrix[:, idx])

    # Ordenar por média (maior = melhor)
    ranking = sorted(model_means.items(), key=lambda x: x[1], reverse=True)

    # Para cada modelo, contar vitórias e empates
    model_stats = {model: {'wins': 0, 'ties': 0, 'losses': 0} for model in models_to_compare}

    for result in comparison_results:
        if result['significance'] == 'SIGNIFICANT':
            winner = result['winner']
            loser = result['model1'] if winner == result['model2'] else result['model2']
            model_stats[winner]['wins'] += 1
            model_stats[loser]['losses'] += 1
        else:
            model_stats[result['model1']]['ties'] += 1
            model_stats[result['model2']]['ties'] += 1

    print(f"{'Rank':<6} {'Model':<15} {'Mean Test G-mean':<20} {'Wins':<6} {'Ties':<6} {'Losses':<8}")
    print("-" * 70)

    for rank, (model, mean_score) in enumerate(ranking, 1):
        stats = model_stats[model]
        print(f"{rank:<6} {model:<15} {mean_score:<20.4f} {stats['wins']:<6} {stats['ties']:<6} {stats['losses']:<8}")

    print()
    print("Legend:")
    print("  - Wins: Number of statistically significant victories (with Bonferroni correction)")
    print("  - Ties: Number of non-significant comparisons")
    print("  - Losses: Number of statistically significant defeats")

    print(f"\n{'='*80}")

else:
    print("\nNo results found!")
    print("Verify that models were executed successfully")
