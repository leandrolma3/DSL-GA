#!/usr/bin/env python3
"""
Script para Analisar Log de Experimento Completo

Analisa experiment_comparison_full2.log e gera relatorio com:
- Tempo de execucao por dataset e modelo
- Metricas finais (Accuracy, F1, G-mean)
- Analise de erros
- Resumo consolidado
"""

import re
import sys
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_timestamp(line: str) -> datetime:
    """Extrai timestamp do inicio da linha"""
    match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if match:
        return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
    return None

def extract_metrics(lines: List[str]) -> Dict:
    """Extrai metricas finais de um conjunto de linhas"""
    metrics = {}

    for line in lines:
        # GBML metrics
        if '[RESULTADO]' in line and 'Treinou chunk' in line:
            match = re.search(r'Acc: ([\d.]+), F1: ([\d.]+), G-mean: ([\d.]+)', line)
            if match:
                chunk_match = re.search(r'Treinou chunk (\d+) → Testou chunk (\d+)', line)
                if chunk_match:
                    key = f"chunk_{chunk_match.group(1)}_to_{chunk_match.group(2)}"
                    metrics[key] = {
                        'accuracy': float(match.group(1)),
                        'f1': float(match.group(2)),
                        'gmean': float(match.group(3))
                    }

        # ACDWM metrics
        if 'baseline_acdwm' in line and 'G-mean:' in line:
            match = re.search(r'Chunk (\d+) - G-mean: ([\d.]+)', line)
            if match:
                chunk_idx = int(match.group(1))
                gmean = float(match.group(2))
                if 'acdwm_chunks' not in metrics:
                    metrics['acdwm_chunks'] = {}
                metrics['acdwm_chunks'][chunk_idx] = gmean

        # River metrics summary
        if 'baseline_river' in line and 'média:' in line:
            if 'Accuracy média:' in line:
                match = re.search(r'Accuracy média: ([\d.]+)', line)
                if match:
                    metrics['river_accuracy_avg'] = float(match.group(1))
            elif 'G-mean média:' in line:
                match = re.search(r'G-mean média: ([\d.]+)', line)
                if match:
                    metrics['river_gmean_avg'] = float(match.group(1))

        # Summary statistics (final section)
        if 'Accuracy média:' in line and '±' in line:
            match = re.search(r'Accuracy média:\s+([\d.]+)\s+±\s+([\d.]+)', line)
            if match:
                if 'summary' not in metrics:
                    metrics['summary'] = []
                metrics['summary'].append({
                    'metric': 'accuracy',
                    'mean': float(match.group(1)),
                    'std': float(match.group(2))
                })

        if 'G-mean média:' in line and '±' in line:
            match = re.search(r'G-mean média:\s+([\d.]+)\s+±\s+([\d.]+)', line)
            if match:
                if 'summary' not in metrics:
                    metrics['summary'] = []
                metrics['summary'].append({
                    'metric': 'gmean',
                    'mean': float(match.group(1)),
                    'std': float(match.group(2))
                })

    return metrics

def analyze_log(log_path: str) -> Dict:
    """Analisa arquivo de log completo"""

    print(f"Analisando log: {log_path}")
    print("=" * 80)

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total de linhas: {len(lines)}")
    print()

    # Estrutura de dados
    datasets = []
    current_dataset = None
    dataset_start_time = None
    dataset_lines = []

    # Primeira passagem: identifica datasets
    for i, line in enumerate(lines):
        if 'Stream:' in line and 'RBF_' in line:
            # Salva dataset anterior se existir
            if current_dataset:
                datasets.append({
                    'name': current_dataset,
                    'start_time': dataset_start_time,
                    'lines': dataset_lines,
                    'line_start': i - len(dataset_lines),
                    'line_end': i
                })

            # Novo dataset
            match = re.search(r'Stream: (RBF_\w+)', line)
            if match:
                current_dataset = match.group(1)
                dataset_start_time = parse_timestamp(line)
                dataset_lines = [line]
        elif current_dataset:
            dataset_lines.append(line)

    # Salva ultimo dataset
    if current_dataset:
        datasets.append({
            'name': current_dataset,
            'start_time': dataset_start_time,
            'lines': dataset_lines,
            'line_start': len(lines) - len(dataset_lines),
            'line_end': len(lines)
        })

    print(f"Datasets encontrados: {len(datasets)}")
    for ds in datasets:
        print(f"  - {ds['name']}: {len(ds['lines'])} linhas (L{ds['line_start']}-{ds['line_end']})")
    print()

    # Segunda passagem: analisa cada dataset
    results = {
        'datasets': [],
        'total_lines': len(lines)
    }

    for ds in datasets:
        print(f"\nAnalisando dataset: {ds['name']}")
        print("-" * 80)

        dataset_result = {
            'name': ds['name'],
            'start_time': ds['start_time'],
            'models': {},
            'timing': {}
        }

        # Identifica execucao de modelos
        gbml_start = None
        acdwm_start = None
        river_models = []

        for i, line in enumerate(ds['lines']):
            ts = parse_timestamp(line)

            if '[3/5] Executando GBML' in line:
                gbml_start = ts
                print(f"  GBML inicio: {ts}")

            elif '[3.5/5] Executando ACDWM' in line:
                if gbml_start and ts:
                    gbml_duration = (ts - gbml_start).total_seconds() / 60
                    dataset_result['timing']['GBML'] = gbml_duration
                    print(f"  GBML duracao: {gbml_duration:.1f} min")
                acdwm_start = ts
                print(f"  ACDWM inicio: {ts}")

            elif 'RiverEvaluator[' in line and '[RESULTADO]' in line:
                match = re.search(r'RiverEvaluator\[(\w+)\]', line)
                if match:
                    model = match.group(1)
                    if model not in river_models:
                        river_models.append(model)

        # Extrai metricas
        metrics = extract_metrics(ds['lines'])
        dataset_result['metrics'] = metrics

        # Identifica fim do dataset
        end_time = parse_timestamp(ds['lines'][-1]) if ds['lines'] else None
        if end_time and ds['start_time']:
            total_duration = (end_time - ds['start_time']).total_seconds() / 60
            dataset_result['total_duration'] = total_duration
            print(f"  Duracao total: {total_duration:.1f} min ({total_duration/60:.2f} h)")

        # Resume metricas
        print(f"\n  Metricas extraidas:")
        for chunk_key, chunk_metrics in metrics.items():
            if chunk_key.startswith('chunk_'):
                print(f"    {chunk_key}: Acc={chunk_metrics['accuracy']:.4f}, " +
                      f"F1={chunk_metrics['f1']:.4f}, G-mean={chunk_metrics['gmean']:.4f}")

        if 'acdwm_chunks' in metrics:
            print(f"    ACDWM chunks: {len(metrics['acdwm_chunks'])}")
            for chunk_idx, gmean in sorted(metrics['acdwm_chunks'].items()):
                print(f"      Chunk {chunk_idx}: G-mean={gmean:.4f}")

        results['datasets'].append(dataset_result)

    return results

def generate_report(results: Dict, output_path: str):
    """Gera relatorio em arquivo texto"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RELATORIO DE ANALISE - EXPERIMENTO COMPLETO\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total de linhas no log: {results['total_lines']}\n")
        f.write(f"Datasets analisados: {len(results['datasets'])}\n\n")

        # Analise por dataset
        for ds in results['datasets']:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Dataset: {ds['name']}\n")
            f.write("=" * 80 + "\n\n")

            if ds.get('start_time'):
                f.write(f"Inicio: {ds['start_time']}\n")

            if ds.get('total_duration'):
                f.write(f"Duracao total: {ds['total_duration']:.1f} min " +
                       f"({ds['total_duration']/60:.2f} horas)\n\n")

            # Timing por modelo
            if ds.get('timing'):
                f.write("Tempo de execucao por modelo:\n")
                f.write("-" * 80 + "\n")
                for model, duration in ds['timing'].items():
                    f.write(f"  {model:15s}: {duration:8.1f} min ({duration/60:6.2f} h)\n")
                f.write("\n")

            # Metricas
            metrics = ds.get('metrics', {})

            if any(k.startswith('chunk_') for k in metrics.keys()):
                f.write("Resultados GBML por avaliacao:\n")
                f.write("-" * 80 + "\n")
                for chunk_key in sorted([k for k in metrics.keys() if k.startswith('chunk_')]):
                    m = metrics[chunk_key]
                    f.write(f"  {chunk_key:20s}: Acc={m['accuracy']:.4f}, " +
                           f"F1={m['f1']:.4f}, G-mean={m['gmean']:.4f}\n")
                f.write("\n")

            if 'acdwm_chunks' in metrics:
                f.write("Resultados ACDWM por chunk:\n")
                f.write("-" * 80 + "\n")
                for chunk_idx, gmean in sorted(metrics['acdwm_chunks'].items()):
                    f.write(f"  Chunk {chunk_idx}: G-mean={gmean:.4f}\n")
                f.write("\n")

            if 'summary' in metrics:
                f.write("Resumo estatistico:\n")
                f.write("-" * 80 + "\n")
                for stat in metrics['summary']:
                    f.write(f"  {stat['metric']:15s}: {stat['mean']:.4f} +/- {stat['std']:.4f}\n")
                f.write("\n")

        # Resumo geral
        f.write("\n" + "=" * 80 + "\n")
        f.write("RESUMO GERAL\n")
        f.write("=" * 80 + "\n\n")

        total_duration = sum(ds.get('total_duration', 0) for ds in results['datasets'])
        f.write(f"Tempo total de execucao: {total_duration:.1f} min ({total_duration/60:.2f} horas)\n")
        f.write(f"Tempo medio por dataset: {total_duration/len(results['datasets']):.1f} min\n\n")

        # Coleta todas as metricas GBML
        all_gbml_gmeans = []
        for ds in results['datasets']:
            metrics = ds.get('metrics', {})
            for chunk_key, chunk_metrics in metrics.items():
                if chunk_key.startswith('chunk_'):
                    all_gbml_gmeans.append(chunk_metrics['gmean'])

        if all_gbml_gmeans:
            avg_gmean = sum(all_gbml_gmeans) / len(all_gbml_gmeans)
            min_gmean = min(all_gbml_gmeans)
            max_gmean = max(all_gbml_gmeans)
            f.write(f"GBML G-mean:\n")
            f.write(f"  Media:  {avg_gmean:.4f}\n")
            f.write(f"  Min:    {min_gmean:.4f}\n")
            f.write(f"  Max:    {max_gmean:.4f}\n")
            f.write(f"  N eval: {len(all_gbml_gmeans)}\n\n")

def main():
    log_path = 'experiment_comparison_full2.log'
    output_path = 'experiment_analysis_report.txt'

    print("ANALISE DE LOG - EXPERIMENTO COMPLETO")
    print("=" * 80)
    print()

    # Analisa log
    results = analyze_log(log_path)

    # Gera relatorio
    print("\n" + "=" * 80)
    print(f"Gerando relatorio em: {output_path}")
    generate_report(results, output_path)
    print(f"Relatorio gerado com sucesso!")
    print()

    # Exibe resumo rapido
    print("RESUMO RAPIDO:")
    print("-" * 80)
    for ds in results['datasets']:
        duration = ds.get('total_duration', 0)
        print(f"{ds['name']:30s}: {duration:.1f} min ({duration/60:.2f} h)")

    total = sum(ds.get('total_duration', 0) for ds in results['datasets'])
    print("-" * 80)
    print(f"{'TOTAL':30s}: {total:.1f} min ({total/60:.2f} h)")

if __name__ == '__main__':
    main()
