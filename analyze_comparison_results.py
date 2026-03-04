#!/usr/bin/env python3
"""
Script de Analise: Experimento Comparativo GBML vs River

Analisa o log completo do experimento e gera relatorio detalhado
de performance de todos os modelos em todos os datasets.
"""

import re
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import sys

def parse_experiment_log(log_file_path):
    """
    Parse log file and extract performance metrics

    Returns:
        dict: Structured data with all results
    """

    results = defaultdict(lambda: defaultdict(list))
    current_dataset = None
    current_model = None
    current_chunk = None

    print(f"Analisando log: {log_file_path}")
    print("Processando...")

    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Total de linhas: {total_lines}")

    # Patterns for extraction
    dataset_pattern = re.compile(r'Stream: (RBF_\w+)')
    model_pattern_gbml = re.compile(r'GBMLEvaluator\[GBML\]: \[RESULTADO\].*Testou chunk (\d+).*Acc:\s+([\d.]+),\s+F1:\s+([\d.]+),\s+G-mean:\s+([\d.]+)')
    model_pattern_river = re.compile(r'RiverEvaluator\[(\w+)\]: \[RESULTADO\].*Testou chunk (\d+).*Acc:\s+([\d.]+),\s+F1:\s+([\d.]+),\s+G-mean:\s+([\d.]+)')

    # Summary pattern at the end
    summary_pattern = re.compile(r'Modelo:\s+(\w+)')
    summary_metrics = re.compile(r'(Accuracy|F1|G-mean)\s+média:\s+([\d.]+)\s+±\s+([\d.]+)')

    i = 0
    while i < total_lines:
        line = lines[i]

        # Detect dataset
        dataset_match = dataset_pattern.search(line)
        if dataset_match:
            current_dataset = dataset_match.group(1)
            print(f"  Dataset detectado: {current_dataset}")

        # Detect GBML test chunk with metrics
        gbml_test_match = model_pattern_gbml.search(line)
        if gbml_test_match and current_dataset:
            current_chunk = int(gbml_test_match.group(1))
            accuracy = float(gbml_test_match.group(2))
            f1 = float(gbml_test_match.group(3))
            gmean = float(gbml_test_match.group(4))

            results[current_dataset]['GBML_accuracy'].append(accuracy)
            results[current_dataset]['GBML_gmean'].append(gmean)
            results[current_dataset]['GBML_f1'].append(f1)

        # Detect River model test chunk with metrics
        river_test_match = model_pattern_river.search(line)
        if river_test_match and current_dataset:
            model_name = river_test_match.group(1)
            current_chunk = int(river_test_match.group(2))
            accuracy = float(river_test_match.group(3))
            f1 = float(river_test_match.group(4))
            gmean = float(river_test_match.group(5))

            results[current_dataset][f'{model_name}_accuracy'].append(accuracy)
            results[current_dataset][f'{model_name}_gmean'].append(gmean)
            results[current_dataset][f'{model_name}_f1'].append(f1)

        i += 1

    return results

def create_dataframe_from_results(results):
    """
    Convert parsed results to structured DataFrame

    Args:
        results: Dict with parsed metrics

    Returns:
        pd.DataFrame: Consolidated results
    """

    records = []

    for dataset, metrics in results.items():
        # Extract all models
        models = set()
        for key in metrics.keys():
            model = key.rsplit('_', 1)[0]
            models.add(model)

        for model in models:
            acc_vals = metrics.get(f'{model}_accuracy', [])
            gmean_vals = metrics.get(f'{model}_gmean', [])
            f1_vals = metrics.get(f'{model}_f1', [])

            # Create one record per chunk
            n_chunks = max(len(acc_vals), len(gmean_vals), len(f1_vals))

            for chunk_idx in range(n_chunks):
                record = {
                    'dataset': dataset,
                    'model': model,
                    'chunk': chunk_idx,
                    'accuracy': acc_vals[chunk_idx] if chunk_idx < len(acc_vals) else None,
                    'gmean': gmean_vals[chunk_idx] if chunk_idx < len(gmean_vals) else None,
                    'f1_weighted': f1_vals[chunk_idx] if chunk_idx < len(f1_vals) else None
                }
                records.append(record)

    df = pd.DataFrame(records)
    return df

def analyze_results(df):
    """
    Generate comprehensive analysis of results

    Args:
        df: DataFrame with all results

    Returns:
        dict: Analysis results
    """

    analysis = {}

    # Overall performance by model
    analysis['overall'] = df.groupby('model').agg({
        'accuracy': ['mean', 'std', 'min', 'max', 'count'],
        'gmean': ['mean', 'std', 'min', 'max'],
        'f1_weighted': ['mean', 'std', 'min', 'max']
    }).round(4)

    # Performance by dataset and model
    analysis['by_dataset'] = df.groupby(['dataset', 'model']).agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'gmean': ['mean', 'std', 'min', 'max'],
        'f1_weighted': ['mean', 'std', 'min', 'max']
    }).round(4)

    # Performance by chunk (to see drift impact)
    analysis['by_chunk'] = df.groupby(['dataset', 'chunk', 'model'])['gmean'].mean().round(4)

    # Ranking by G-mean
    gmean_ranking = df.groupby('model')['gmean'].mean().sort_values(ascending=False)
    analysis['ranking'] = gmean_ranking.round(4)

    # GBML vs Best River comparison
    river_models = [m for m in df['model'].unique() if m != 'GBML']
    if river_models:
        river_df = df[df['model'].isin(river_models)]
        gbml_df = df[df['model'] == 'GBML']

        best_river_gmean = river_df.groupby('model')['gmean'].mean().max()
        best_river_model = river_df.groupby('model')['gmean'].mean().idxmax()
        gbml_gmean = gbml_df['gmean'].mean() if not gbml_df.empty else 0.0

        analysis['gbml_vs_best_river'] = {
            'best_river_model': best_river_model,
            'best_river_gmean': round(best_river_gmean, 4),
            'gbml_gmean': round(gbml_gmean, 4),
            'difference': round(gbml_gmean - best_river_gmean, 4),
            'gbml_better': gbml_gmean > best_river_gmean
        }

    # Per-dataset comparisons
    analysis['dataset_comparisons'] = {}
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        dataset_summary = dataset_df.groupby('model')['gmean'].mean().sort_values(ascending=False)
        analysis['dataset_comparisons'][dataset] = dataset_summary.round(4).to_dict()

    return analysis

def print_analysis_report(analysis, df):
    """
    Print comprehensive analysis report

    Args:
        analysis: Analysis results dict
        df: Original DataFrame
    """

    print("\n" + "="*80)
    print("ANALISE COMPARATIVA: GBML VS RIVER MODELS")
    print("="*80)

    print(f"\nData da analise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total de avaliacoes: {len(df)}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print(f"Modelos: {df['model'].unique().tolist()}")
    print(f"Chunks por dataset: {df.groupby('dataset')['chunk'].max().to_dict()}")

    # Ranking geral
    print("\n" + "-"*80)
    print("1. RANKING GERAL POR G-MEAN MEDIO")
    print("-"*80)
    print("\nModelo               G-mean Medio")
    print("-" * 40)
    for i, (model, gmean) in enumerate(analysis['ranking'].items(), 1):
        marker = "  <--- GBML" if model == 'GBML' else ""
        print(f"{i}. {model:15s}    {gmean:.4f}{marker}")

    # Performance geral
    print("\n" + "-"*80)
    print("2. ESTATISTICAS GERAIS POR MODELO")
    print("-"*80)
    print("\n" + str(analysis['overall']))

    # GBML vs Best River
    if 'gbml_vs_best_river' in analysis:
        comp = analysis['gbml_vs_best_river']
        print("\n" + "-"*80)
        print("3. GBML VS MELHOR MODELO RIVER")
        print("-"*80)
        print(f"\nMelhor River: {comp['best_river_model']} (G-mean: {comp['best_river_gmean']})")
        print(f"GBML:         G-mean: {comp['gbml_gmean']}")
        print(f"Diferenca:    {comp['difference']:+.4f}")

        if comp['gbml_better']:
            print(f"\nCONCLUSAO: GBML SUPERA o melhor modelo River em {abs(comp['difference']):.4f} pontos")
        else:
            print(f"\nCONCLUSAO: GBML INFERIOR ao melhor modelo River em {abs(comp['difference']):.4f} pontos")

    # Performance por dataset
    print("\n" + "-"*80)
    print("4. PERFORMANCE POR DATASET")
    print("-"*80)
    print("\n" + str(analysis['by_dataset']))

    # Analise por dataset
    print("\n" + "-"*80)
    print("5. COMPARACAO DETALHADA POR DATASET")
    print("-"*80)

    for dataset, models_gmean in analysis['dataset_comparisons'].items():
        print(f"\n{dataset}:")
        print("-" * 40)
        for i, (model, gmean) in enumerate(models_gmean.items(), 1):
            marker = "  <--- GBML" if model == 'GBML' else ""
            print(f"  {i}. {model:10s}  G-mean: {gmean:.4f}{marker}")

        if 'GBML' in models_gmean:
            gbml_gmean = models_gmean['GBML']
            others = {k: v for k, v in models_gmean.items() if k != 'GBML'}
            if others:
                best_other = max(others.values())
                best_other_model = [k for k, v in others.items() if v == best_other][0]
                diff = gbml_gmean - best_other

                if diff > 0:
                    print(f"  Analise: GBML SUPERIOR ao {best_other_model} em {diff:+.4f}")
                else:
                    print(f"  Analise: GBML INFERIOR ao {best_other_model} em {diff:.4f}")

    # Analise por chunk (drift impact)
    print("\n" + "-"*80)
    print("6. ANALISE POR CHUNK (Impacto do Drift)")
    print("-"*80)

    for dataset in df['dataset'].unique():
        print(f"\n{dataset}:")
        dataset_chunk = df[df['dataset'] == dataset].groupby(['chunk', 'model'])['gmean'].mean().unstack(fill_value=0)
        print(dataset_chunk.round(4))

        # Identify worst chunk (drift transition)
        gbml_chunks = dataset_chunk['GBML'] if 'GBML' in dataset_chunk.columns else None
        if gbml_chunks is not None and len(gbml_chunks) > 0:
            worst_chunk = gbml_chunks.idxmin()
            worst_gmean = gbml_chunks.min()
            best_chunk = gbml_chunks.idxmax()
            best_gmean = gbml_chunks.max()

            print(f"\n  GBML: Pior chunk = {worst_chunk} (G-mean: {worst_gmean:.4f})")
            print(f"  GBML: Melhor chunk = {best_chunk} (G-mean: {best_gmean:.4f})")
            print(f"  Variacao GBML: {best_gmean - worst_gmean:.4f}")

    # Statistical significance analysis
    print("\n" + "-"*80)
    print("7. ANALISE DE VARIABILIDADE")
    print("-"*80)

    variability = df.groupby('model')['gmean'].agg(['mean', 'std', 'min', 'max'])
    variability['range'] = variability['max'] - variability['min']
    variability['cv'] = (variability['std'] / variability['mean'] * 100).round(2)  # Coefficient of variation

    print("\n" + str(variability.round(4)))
    print("\nCV = Coeficiente de Variacao (std/mean * 100%)")
    print("Menor CV = Modelo mais estavel")

    # Final conclusions
    print("\n" + "="*80)
    print("8. CONCLUSOES FINAIS")
    print("="*80)

    print("\nDesempenho Geral (G-mean medio):")
    for i, (model, gmean) in enumerate(analysis['ranking'].items(), 1):
        print(f"  {i}. {model}: {gmean:.4f}")

    if 'gbml_vs_best_river' in analysis:
        comp = analysis['gbml_vs_best_river']
        print(f"\nComparacao GBML vs Melhor River ({comp['best_river_model']}):")
        print(f"  Diferenca: {comp['difference']:+.4f}")
        print(f"  Percentual: {(comp['difference']/comp['best_river_gmean']*100):+.2f}%")

        if comp['gbml_better']:
            print(f"\n  Resultado: GBML e COMPETITIVO, superando o melhor baseline River")
        elif abs(comp['difference']) < 0.05:
            print(f"\n  Resultado: GBML e COMPETITIVO, com performance EQUIVALENTE ao melhor River")
        else:
            print(f"\n  Resultado: GBML e INFERIOR ao melhor baseline River")

    print("\nEstabilidade (Coeficiente de Variacao):")
    cv_ranking = variability['cv'].sort_values()
    for i, (model, cv) in enumerate(cv_ranking.items(), 1):
        stability = "MUITO ESTAVEL" if cv < 10 else "ESTAVEL" if cv < 15 else "VARIAVEL"
        print(f"  {i}. {model}: {cv:.2f}% ({stability})")

    print("\nAnalise por Tipo de Drift:")
    abrupt_severe = df[df['dataset'] == 'RBF_Abrupt_Severe'].groupby('model')['gmean'].mean().round(4)
    abrupt_moderate = df[df['dataset'] == 'RBF_Abrupt_Moderate'].groupby('model')['gmean'].mean().round(4)
    gradual_moderate = df[df['dataset'] == 'RBF_Gradual_Moderate'].groupby('model')['gmean'].mean().round(4)

    print("\n  Drift Abrupto Severo:")
    if 'GBML' in abrupt_severe.index:
        gbml_rank = (abrupt_severe >= abrupt_severe['GBML']).sum()
        print(f"    GBML: {abrupt_severe['GBML']:.4f} (Posicao: {gbml_rank}/{len(abrupt_severe)})")

    print("\n  Drift Abrupto Moderado:")
    if 'GBML' in abrupt_moderate.index:
        gbml_rank = (abrupt_moderate >= abrupt_moderate['GBML']).sum()
        print(f"    GBML: {abrupt_moderate['GBML']:.4f} (Posicao: {gbml_rank}/{len(abrupt_moderate)})")

    print("\n  Drift Gradual Moderado:")
    if 'GBML' in gradual_moderate.index:
        gbml_rank = (gradual_moderate >= gradual_moderate['GBML']).sum()
        print(f"    GBML: {gradual_moderate['GBML']:.4f} (Posicao: {gbml_rank}/{len(gradual_moderate)})")

def save_analysis_to_file(df, analysis, output_file):
    """
    Save analysis to text file

    Args:
        df: DataFrame with results
        analysis: Analysis dict
        output_file: Output file path
    """

    import sys
    original_stdout = sys.stdout

    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print_analysis_report(analysis, df)

        # Additional CSV export
        csv_file = output_file.replace('.txt', '_data.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nDados completos exportados para: {csv_file}")

    sys.stdout = original_stdout
    print(f"\nAnalise salva em: {output_file}")
    print(f"Dados CSV salvos em: {csv_file}")

def main():
    """Main function"""

    log_file = 'experiment_comparison_full.log'
    output_file = 'ANALISE_COMPARATIVA_GBML_VS_RIVER.txt'

    print("="*80)
    print("SCRIPT DE ANALISE - EXPERIMENTO COMPARATIVO")
    print("="*80)

    # Parse log
    print("\nEtapa 1: Parseando log file...")
    results = parse_experiment_log(log_file)

    # Create DataFrame
    print("\nEtapa 2: Estruturando dados...")
    df = create_dataframe_from_results(results)

    if df.empty:
        print("\nERRO: Nenhum dado encontrado no log!")
        print("Verifique se o arquivo experiment_comparison_full.log contem metricas de performance.")
        return 1

    print(f"  Total de registros extraidos: {len(df)}")
    print(f"  Datasets: {df['dataset'].nunique()}")
    print(f"  Modelos: {df['model'].nunique()}")

    # Analyze
    print("\nEtapa 3: Analisando resultados...")
    analysis = analyze_results(df)

    # Print report
    print_analysis_report(analysis, df)

    # Save to file
    print("\n" + "="*80)
    print("Salvando analise em arquivo...")
    save_analysis_to_file(df, analysis, output_file)

    print("\n" + "="*80)
    print("ANALISE CONCLUIDA COM SUCESSO")
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
