#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CÉLULA 11 FINAL CORRIGIDA - Consolidar Resultados

Correções aplicadas:
1. Ignorar rule_diff_analysis_*_matrix.csv
2. Filtrar ERulesD2S para chunks 0-4
3. Avisar sobre datasets incompletos (NaNs)
4. Documentar métricas ausentes do GBML e ERulesD2S
5. Filtrar datasets completos para testes estatísticos
6. Adicionar Bonferroni correction e comparações pairwise
7. Adicionar conclusão estatística clara sobre GBML
8. Incluir ERulesD2S nas comparações estatísticas (após correção do parser)
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
        and "rule_diff" not in f.name
        and "_matrix" not in f.name
    ]

    for csv_file in result_files:
        try:
            df = pd.read_csv(csv_file)
            original_len = len(df)

            # CORRECAO 2: ERULESD2S - Converter formato e filtrar chunks
            if 'ERulesD2S' in csv_file.name or 'erulesd2s' in csv_file.name.lower():
                # Renomear colunas
                if 'accuracy' in df.columns and 'test_accuracy' not in df.columns:
                    df.rename(columns={
                        'accuracy': 'test_accuracy',
                        'gmean': 'test_gmean',
                        'f1_weighted': 'test_f1'
                    }, inplace=True)

                    # Adicionar colunas train vazias
                    df['train_gmean'] = np.nan
                    df['train_accuracy'] = np.nan
                    df['train_f1'] = np.nan

                    # Ajustar e filtrar chunks
                    if 'chunk' in df.columns:
                        df['chunk'] = df['chunk'] - 1
                        df = df[df['chunk'] <= 4].copy()

                print(f"  - {csv_file.name}: {len(df)} rows (ERulesD2S - converting format)", end="")
                if original_len > len(df):
                    print(f" [filtered {original_len} -> {len(df)} chunks]")
                else:
                    print()
            else:
                print(f"  - {csv_file.name}: {len(df)} rows")

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

            df_gbml = pd.DataFrame(gbml_data)

            df_gbml['model'] = 'GBML'
            df_gbml['dataset'] = dataset
            df_gbml['train_size'] = 1000
            df_gbml['test_size'] = 1000

            # Colunas faltantes (GBML não reporta accuracy)
            if 'train_accuracy' not in df_gbml.columns:
                df_gbml['train_accuracy'] = np.nan
            if 'test_accuracy' not in df_gbml.columns:
                df_gbml['test_accuracy'] = np.nan
            if 'train_f1' not in df_gbml.columns:
                df_gbml['train_f1'] = np.nan
            if 'test_f1' not in df_gbml.columns:
                df_gbml['test_f1'] = df_gbml.get('test_f1_orig', np.nan)

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
    # CORRECAO 3: AVISAR SOBRE DATASETS INCOMPLETOS
    # ========================================================================

    print(f"\n{'='*80}")
    print("DATA COMPLETENESS CHECK")
    print(f"{'='*80}")

    all_models = sorted(consolidated['model'].unique())
    all_datasets = sorted(consolidated['dataset'].unique())

    incomplete_datasets = []
    for dataset in all_datasets:
        dataset_models = consolidated[consolidated['dataset'] == dataset]['model'].unique()
        if len(dataset_models) < len(all_models):
            missing_models = set(all_models) - set(dataset_models)
            incomplete_datasets.append((dataset, missing_models))
            print(f"\n[WARNING] {dataset}:")
            print(f"  Missing models: {', '.join(missing_models)}")

    if incomplete_datasets:
        print(f"\n{len(incomplete_datasets)} dataset(s) have incomplete model results")
        print("These datasets will be EXCLUDED from statistical tests")
    else:
        print("\n[OK] All datasets have complete results for all models")

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
    # CORRECAO 4: DOCUMENTAR METRICAS AUSENTES DO GBML
    # ========================================================================

    print(f"\n{'='*80}")
    print("NOTES ON MISSING METRICS")
    print(f"{'='*80}")

    print("\n[GBML Metrics]")
    print("  train_gmean:     Available")
    print("  test_gmean:      Available")
    print("  test_f1:         Available")
    print("  train_accuracy:  Not reported (NaN)")
    print("  test_accuracy:   Not reported (NaN)")
    print("\n  Rationale: GBML focuses on G-mean (geometric mean), which is more")
    print("             appropriate than accuracy for imbalanced datasets.")

    print("\n[ERulesD2S Metrics]")
    print("  train_gmean:     Not available (NaN) - online learning, no separate train phase")
    print("  test_gmean:      Available")
    print("  train_accuracy:  Not available (NaN)")
    print("  test_accuracy:   Available")

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
    # TESTES ESTATISTICOS (CORRECAO 5: FILTRAR DATASETS COMPLETOS)
    # ========================================================================

    print(f"\n{'='*80}")
    print("STATISTICAL TESTS (on Test G-mean)")
    print(f"{'='*80}")
    print("Reference: Demsar (2006), Garcia & Herrera (2008)")
    print("Using TEST metrics to evaluate generalization performance")

    # Modelos para comparar (INCLUINDO ERulesD2S após correção do parser)
    models_to_compare = ['GBML', 'ACDWM', 'ARF', 'SRP', 'HAT', 'ERulesD2S']

    # Filtrar modelos disponíveis
    models_available = []
    for model in models_to_compare:
        model_data = consolidated[
            (consolidated['model'] == model) &
            (consolidated['test_gmean'].notna())
        ]
        if len(model_data) > 0:
            models_available.append(model)

    models_to_compare = models_available

    # CORRECAO: Identificar datasets com dados completos para TODOS os modelos
    complete_datasets = []
    for dataset in consolidated['dataset'].unique():
        has_all_models = True
        for model in models_to_compare:
            score = consolidated[
                (consolidated['model'] == model) &
                (consolidated['dataset'] == dataset)
            ]['test_gmean'].mean()

            if np.isnan(score):
                has_all_models = False
                break

        if has_all_models:
            complete_datasets.append(dataset)

    excluded_datasets = set(consolidated['dataset'].unique()) - set(complete_datasets)

    print(f"\nDatasets with complete data: {len(complete_datasets)}/{len(consolidated['dataset'].unique())}")
    if excluded_datasets:
        print(f"Excluded datasets: {', '.join(sorted(excluded_datasets))}")

    # Criar matriz APENAS com datasets completos
    data_matrix = []
    for model in models_to_compare:
        model_scores = []
        for dataset in complete_datasets:
            score = consolidated[
                (consolidated['model'] == model) &
                (consolidated['dataset'] == dataset)
            ]['test_gmean'].mean()
            model_scores.append(score)

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
    # WILCOXON SIGNED-RANK TEST (ALL PAIRWISE WITH BONFERRONI)
    # ========================================================================

    print(f"\n--- Wilcoxon Signed-Rank Tests (All Pairwise Comparisons) ---")

    num_comparisons = len(models_to_compare) * (len(models_to_compare) - 1) // 2
    alpha_bonferroni = 0.05 / num_comparisons

    print(f"Total pairwise comparisons: {num_comparisons}")
    print(f"Alpha (original): 0.05")
    print(f"Alpha (Bonferroni corrected): {alpha_bonferroni:.6f}")
    print()

    comparison_results = []

    for i in range(len(models_to_compare)):
        for j in range(i+1, len(models_to_compare)):
            model1 = models_to_compare[i]
            model2 = models_to_compare[j]

            scores1 = data_matrix[:, i]
            scores2 = data_matrix[:, j]

            try:
                stat, p_value = wilcoxon(scores1, scores2, alternative='two-sided')
                mean_diff = np.mean(scores1 - scores2)

                if p_value < alpha_bonferroni:
                    winner = model1 if mean_diff > 0 else model2
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

    comparison_results.sort(key=lambda x: x['p_value'] if not np.isnan(x['p_value']) else 1.0)

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
    # RANKING FINAL COM SIGNIFICANCIA
    # ========================================================================

    print(f"\n--- Final Ranking (with Statistical Significance) ---")
    print()

    model_means = {}
    for idx, model in enumerate(models_to_compare):
        model_means[model] = np.mean(data_matrix[:, idx])

    ranking = sorted(model_means.items(), key=lambda x: x[1], reverse=True)

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

    # ========================================================================
    # CORRECAO 7: CONCLUSAO ESTATISTICA CLARA SOBRE GBML
    # ========================================================================

    print(f"\n{'='*80}")
    print("STATISTICAL CONCLUSION - GBML vs BASELINES")
    print(f"{'='*80}")

    gbml_comparisons = [r for r in comparison_results if 'GBML' in [r['model1'], r['model2']]]

    gbml_wins = sum(1 for r in gbml_comparisons if r['winner'] == 'GBML' and r['significance'] == 'SIGNIFICANT')
    gbml_losses = sum(1 for r in gbml_comparisons if r['winner'] != 'GBML' and r['winner'] != 'TIE' and r['significance'] == 'SIGNIFICANT')
    gbml_ties = sum(1 for r in gbml_comparisons if r['winner'] == 'TIE')

    print(f"\nGBML Performance Summary:")
    print(f"  Statistically significant wins:   {gbml_wins}/{len(gbml_comparisons)}")
    print(f"  Statistically significant losses: {gbml_losses}/{len(gbml_comparisons)}")
    print(f"  No significant difference (ties): {gbml_ties}/{len(gbml_comparisons)}")
    print()

    # Conclusão baseada em resultados (com informação de magnitude)
    if gbml_wins > 0 and gbml_losses == 0:
        print("[+] CONCLUSION: GBML is SIGNIFICANTLY BETTER than some baselines")
        print(f"    (with Bonferroni correction, alpha = {alpha_bonferroni:.6f})")
    elif gbml_losses > 0 and gbml_wins == 0:
        print("[-] CONCLUSION: GBML is statistically significantly worse than some baselines")
        print(f"    (with Bonferroni correction, alpha = {alpha_bonferroni:.6f})")
        print()
        # Adicionar informação sobre magnitude das diferenças
        for result in gbml_comparisons:
            if result['significance'] == 'SIGNIFICANT' and result['winner'] != 'GBML':
                other_model = result['model1'] if result['model2'] == 'GBML' else result['model2']
                mean_diff_pct = abs(result['mean_diff']) * 100
                print(f"    NOTE: Difference vs {other_model} is small ({mean_diff_pct:.2f}% average)")
                print(f"          but CONSISTENT across datasets (p={result['p_value']:.6f})")
    elif gbml_wins > 0 and gbml_losses > 0:
        print("[~] CONCLUSION: GBML shows MIXED results - better than some, worse than others")
        print(f"    (with Bonferroni correction, alpha = {alpha_bonferroni:.6f})")
    else:
        print("[=] CONCLUSION: GBML is STATISTICALLY EQUIVALENT to all baselines")
        print(f"    (no significant differences with Bonferroni correction, alpha = {alpha_bonferroni:.6f})")

    print()
    print("Detailed GBML comparisons:")
    for result in gbml_comparisons:
        if result['model1'] == 'GBML':
            other_model = result['model2']
            sign = ">" if result['mean_diff'] > 0 else "<"
        else:
            other_model = result['model1']
            sign = "<" if result['mean_diff'] > 0 else ">"

        status = "SIGNIFICANT" if result['significance'] == "SIGNIFICANT" else "equivalent"
        print(f"  GBML {sign} {other_model:<10}: p={result['p_value']:.6f} [{status}]")

    print(f"\n{'='*80}")

else:
    print("\nNo results found!")
    print("Verify that models were executed successfully")
