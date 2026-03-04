#!/usr/bin/env python3
"""
Analise Detalhada do Log de Experimento

Extrai metricas completas de todos os modelos:
- GBML (por chunk)
- ACDWM (por chunk)
- River: HAT, ARF, SRP (por chunk)

Gera tabela consolidada comparando todos os modelos.
"""

import re
from collections import defaultdict
from typing import Dict, List

def extract_all_metrics(log_path: str) -> Dict:
    """Extrai todas as metricas de todos os modelos"""

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = {
        'datasets': defaultdict(lambda: {
            'GBML': [],
            'ACDWM': [],
            'HAT': [],
            'ARF': [],
            'SRP': []
        })
    }

    current_dataset = None

    for line in lines:
        # Detecta dataset atual
        if 'Stream:' in line and 'RBF_' in line:
            match = re.search(r'Stream: (RBF_\w+)', line)
            if match:
                current_dataset = match.group(1)
                continue

        if not current_dataset:
            continue

        # GBML metrics
        if 'GBMLEvaluator[GBML]' in line and '[RESULTADO]' in line:
            match_chunks = re.search(r'Treinou chunk (\d+) → Testou chunk (\d+)', line)
            match_metrics = re.search(r'Acc: ([\d.]+), F1: ([\d.]+), G-mean: ([\d.]+)', line)

            if match_chunks and match_metrics:
                train_chunk = int(match_chunks.group(1))
                test_chunk = int(match_chunks.group(2))
                results['datasets'][current_dataset]['GBML'].append({
                    'train_chunk': train_chunk,
                    'test_chunk': test_chunk,
                    'accuracy': float(match_metrics.group(1)),
                    'f1': float(match_metrics.group(2)),
                    'gmean': float(match_metrics.group(3))
                })

        # ACDWM metrics
        if 'baseline_acdwm' in line and 'Chunk' in line and 'G-mean:' in line:
            match = re.search(r'Chunk (\d+) - G-mean: ([\d.]+)', line)
            if match:
                chunk_idx = int(match.group(1))
                gmean = float(match.group(2))
                results['datasets'][current_dataset]['ACDWM'].append({
                    'chunk': chunk_idx,
                    'gmean': gmean
                })

        # River models (HAT, ARF, SRP)
        if 'RiverEvaluator[' in line and '[RESULTADO]' in line:
            match_model = re.search(r'RiverEvaluator\[(\w+)\]', line)
            match_chunks = re.search(r'Treinou chunk (\d+) → Testou chunk (\d+)', line)
            match_metrics = re.search(r'Acc: ([\d.]+), F1: ([\d.]+), G-mean: ([\d.]+)', line)

            if match_model and match_chunks and match_metrics:
                model = match_model.group(1)
                train_chunk = int(match_chunks.group(1))
                test_chunk = int(match_chunks.group(2))

                if model in ['HAT', 'ARF', 'SRP']:
                    results['datasets'][current_dataset][model].append({
                        'train_chunk': train_chunk,
                        'test_chunk': test_chunk,
                        'accuracy': float(match_metrics.group(1)),
                        'f1': float(match_metrics.group(2)),
                        'gmean': float(match_metrics.group(3))
                    })

    return results

def generate_detailed_report(results: Dict, output_path: str):
    """Gera relatorio detalhado com tabelas consolidadas"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("RELATORIO DETALHADO - EXPERIMENTO COMPLETO\n")
        f.write("Comparacao: GBML vs ACDWM vs River (HAT, ARF, SRP)\n")
        f.write("=" * 100 + "\n\n")

        datasets = sorted(results['datasets'].keys())

        for dataset in datasets:
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"Dataset: {dataset}\n")
            f.write("=" * 100 + "\n\n")

            ds_data = results['datasets'][dataset]

            # Tabela consolidada por chunk
            f.write("Tabela Consolidada - G-mean por Chunk\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Avaliacao':<20} | {'GBML':>8} | {'ACDWM':>8} | {'HAT':>8} | {'ARF':>8} | {'SRP':>8} |\n")
            f.write("-" * 100 + "\n")

            # Identifica numero de avaliacoes
            num_evals = len(ds_data['GBML'])

            for i in range(num_evals):
                gbml_metric = ds_data['GBML'][i] if i < len(ds_data['GBML']) else None
                acdwm_metric = ds_data['ACDWM'][i] if i < len(ds_data['ACDWM']) else None
                hat_metric = ds_data['HAT'][i] if i < len(ds_data['HAT']) else None
                arf_metric = ds_data['ARF'][i] if i < len(ds_data['ARF']) else None
                srp_metric = ds_data['SRP'][i] if i < len(ds_data['SRP']) else None

                if gbml_metric:
                    eval_name = f"Chunk {gbml_metric['train_chunk']} -> {gbml_metric['test_chunk']}"
                else:
                    eval_name = f"Avaliacao {i+1}"

                gbml_str = f"{gbml_metric['gmean']:8.4f}" if gbml_metric else "    -   "
                acdwm_str = f"{acdwm_metric['gmean']:8.4f}" if acdwm_metric else "    -   "
                hat_str = f"{hat_metric['gmean']:8.4f}" if hat_metric else "    -   "
                arf_str = f"{arf_metric['gmean']:8.4f}" if arf_metric else "    -   "
                srp_str = f"{srp_metric['gmean']:8.4f}" if srp_metric else "    -   "

                f.write(f"{eval_name:<20} | {gbml_str} | {acdwm_str} | {hat_str} | {arf_str} | {srp_str} |\n")

            f.write("-" * 100 + "\n\n")

            # Estatisticas por modelo
            f.write("Estatisticas por Modelo (G-mean)\n")
            f.write("-" * 100 + "\n")

            models = ['GBML', 'ACDWM', 'HAT', 'ARF', 'SRP']
            stats = []

            for model in models:
                if model == 'ACDWM':
                    gmeans = [m['gmean'] for m in ds_data[model] if 'gmean' in m]
                else:
                    gmeans = [m['gmean'] for m in ds_data[model] if 'gmean' in m]

                if gmeans:
                    mean_val = sum(gmeans) / len(gmeans)
                    min_val = min(gmeans)
                    max_val = max(gmeans)
                    std_val = (sum((x - mean_val)**2 for x in gmeans) / len(gmeans))**0.5 if len(gmeans) > 1 else 0

                    stats.append({
                        'model': model,
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val,
                        'n': len(gmeans)
                    })

            # Ordena por media decrescente
            stats.sort(key=lambda x: x['mean'], reverse=True)

            f.write(f"{'Modelo':<10} | {'Media':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'N':>4} |\n")
            f.write("-" * 100 + "\n")

            for stat in stats:
                f.write(f"{stat['model']:<10} | {stat['mean']:8.4f} | {stat['std']:8.4f} | " +
                       f"{stat['min']:8.4f} | {stat['max']:8.4f} | {stat['n']:4d} |\n")

            f.write("-" * 100 + "\n\n")

            # Ranking
            f.write("Ranking (por G-mean media):\n")
            f.write("-" * 100 + "\n")
            for rank, stat in enumerate(stats, 1):
                diff = stat['mean'] - stats[0]['mean']
                f.write(f"  {rank}. {stat['model']:10s}: {stat['mean']:.4f} ({diff:+.4f} vs melhor)\n")
            f.write("\n")

        # Resumo geral
        f.write("\n" + "=" * 100 + "\n")
        f.write("RESUMO GERAL - TODOS OS DATASETS\n")
        f.write("=" * 100 + "\n\n")

        # Agrega metricas de todos os datasets
        all_models = ['GBML', 'ACDWM', 'HAT', 'ARF', 'SRP']
        overall_stats = []

        for model in all_models:
            all_gmeans = []
            for dataset in datasets:
                ds_data = results['datasets'][dataset]
                if model == 'ACDWM':
                    gmeans = [m['gmean'] for m in ds_data[model] if 'gmean' in m]
                else:
                    gmeans = [m['gmean'] for m in ds_data[model] if 'gmean' in m]
                all_gmeans.extend(gmeans)

            if all_gmeans:
                mean_val = sum(all_gmeans) / len(all_gmeans)
                min_val = min(all_gmeans)
                max_val = max(all_gmeans)
                std_val = (sum((x - mean_val)**2 for x in all_gmeans) / len(all_gmeans))**0.5 if len(all_gmeans) > 1 else 0

                overall_stats.append({
                    'model': model,
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'n': len(all_gmeans)
                })

        overall_stats.sort(key=lambda x: x['mean'], reverse=True)

        f.write(f"{'Modelo':<10} | {'Media':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'N':>4} |\n")
        f.write("-" * 100 + "\n")

        for stat in overall_stats:
            f.write(f"{stat['model']:<10} | {stat['mean']:8.4f} | {stat['std']:8.4f} | " +
                   f"{stat['min']:8.4f} | {stat['max']:8.4f} | {stat['n']:4d} |\n")

        f.write("-" * 100 + "\n\n")

        f.write("Ranking Geral (por G-mean media em todos os datasets):\n")
        f.write("-" * 100 + "\n")
        for rank, stat in enumerate(overall_stats, 1):
            diff = stat['mean'] - overall_stats[0]['mean']
            f.write(f"  {rank}. {stat['model']:10s}: {stat['mean']:.4f} ({diff:+.4f} vs melhor)\n")

def main():
    log_path = 'experiment_comparison_full2.log'
    output_path = 'experiment_detailed_analysis.txt'

    print("ANALISE DETALHADA - EXPERIMENTO COMPLETO")
    print("=" * 100)
    print()
    print(f"Lendo log: {log_path}")

    # Extrai metricas
    results = extract_all_metrics(log_path)

    print(f"Datasets encontrados: {len(results['datasets'])}")
    for dataset in sorted(results['datasets'].keys()):
        ds_data = results['datasets'][dataset]
        print(f"\n{dataset}:")
        for model in ['GBML', 'ACDWM', 'HAT', 'ARF', 'SRP']:
            count = len(ds_data[model])
            if count > 0:
                print(f"  {model:8s}: {count} avaliacoes")

    # Gera relatorio
    print(f"\nGerando relatorio detalhado em: {output_path}")
    generate_detailed_report(results, output_path)
    print("Relatorio gerado com sucesso!")
    print()

    # Exibe tabela rapida no console
    print("=" * 100)
    print("RESUMO RAPIDO - G-mean Media por Dataset e Modelo")
    print("=" * 100)
    print(f"{'Dataset':<30} | {'GBML':>8} | {'ACDWM':>8} | {'HAT':>8} | {'ARF':>8} | {'SRP':>8} |")
    print("-" * 100)

    for dataset in sorted(results['datasets'].keys()):
        ds_data = results['datasets'][dataset]

        gbml_gmeans = [m['gmean'] for m in ds_data['GBML'] if 'gmean' in m]
        acdwm_gmeans = [m['gmean'] for m in ds_data['ACDWM'] if 'gmean' in m]
        hat_gmeans = [m['gmean'] for m in ds_data['HAT'] if 'gmean' in m]
        arf_gmeans = [m['gmean'] for m in ds_data['ARF'] if 'gmean' in m]
        srp_gmeans = [m['gmean'] for m in ds_data['SRP'] if 'gmean' in m]

        gbml_str = f"{sum(gbml_gmeans)/len(gbml_gmeans):8.4f}" if gbml_gmeans else "    -   "
        acdwm_str = f"{sum(acdwm_gmeans)/len(acdwm_gmeans):8.4f}" if acdwm_gmeans else "    -   "
        hat_str = f"{sum(hat_gmeans)/len(hat_gmeans):8.4f}" if hat_gmeans else "    -   "
        arf_str = f"{sum(arf_gmeans)/len(arf_gmeans):8.4f}" if arf_gmeans else "    -   "
        srp_str = f"{sum(srp_gmeans)/len(srp_gmeans):8.4f}" if srp_gmeans else "    -   "

        print(f"{dataset:<30} | {gbml_str} | {acdwm_str} | {hat_str} | {arf_str} | {srp_str} |")

    print("-" * 100)

if __name__ == '__main__':
    main()
