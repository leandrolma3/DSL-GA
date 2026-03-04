#!/usr/bin/env python3
"""Script para verificar progresso dos experimentos."""

import os
import json
from pathlib import Path
from datetime import datetime

base_dir = Path('experiments_unified')

# Estrutura esperada
experiments_config = {
    'chunk_500': {'batches': ['batch_1', 'batch_2', 'batch_3'], 'expected_chunks': 23},
    'chunk_500_penalty': {'batches': ['batch_1', 'batch_2', 'batch_3'], 'expected_chunks': 23},
    'chunk_1000': {'batches': ['batch_1', 'batch_2', 'batch_3', 'batch_4'], 'expected_chunks': 11},
    'chunk_1000_penalty': {'batches': ['batch_1', 'batch_2', 'batch_3', 'batch_4'], 'expected_chunks': 11},
    'chunk_2000': {'batches': ['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5', 'batch_6', 'batch_7'], 'expected_chunks': 5},
    'chunk_2000_penalty': {'batches': ['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5', 'batch_6', 'batch_7'], 'expected_chunks': 5},
}

results = {}
total_complete = 0
total_partial = 0

for chunk_type, config in experiments_config.items():
    chunk_dir = base_dir / chunk_type
    if not chunk_dir.exists():
        continue

    results[chunk_type] = {'complete': [], 'partial': [], 'batches': {}}

    for batch in config['batches']:
        batch_dir = chunk_dir / batch
        if not batch_dir.exists():
            continue

        results[chunk_type]['batches'][batch] = {'complete': [], 'partial': []}

        for dataset_dir in batch_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name
            run_dir = dataset_dir / 'run_1'

            if not run_dir.exists():
                continue

            metrics_file = run_dir / 'chunk_metrics.json'
            partial_file = run_dir / 'chunk_metrics_partial.json'
            chunk_data_dir = run_dir / 'chunk_data'

            status = 'unknown'
            num_chunks = 0

            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    num_chunks = len(metrics)
                    if num_chunks >= config['expected_chunks']:
                        status = 'complete'
                        results[chunk_type]['complete'].append(dataset_name)
                        results[chunk_type]['batches'][batch]['complete'].append(dataset_name)
                        total_complete += 1
                    else:
                        status = 'partial'
                except:
                    pass

            if status != 'complete':
                if partial_file.exists():
                    try:
                        with open(partial_file, 'r') as f:
                            partial = json.load(f)
                        num_chunks = len(partial)
                        status = 'partial'
                    except:
                        pass
                elif chunk_data_dir.exists():
                    checkpoints = list(chunk_data_dir.glob('best_individual_trained_on_chunk_*.pkl'))
                    if checkpoints:
                        num_chunks = len(checkpoints)
                        status = 'partial'

                if status == 'partial':
                    results[chunk_type]['partial'].append(f'{dataset_name} ({num_chunks}/{config["expected_chunks"]})')
                    results[chunk_type]['batches'][batch]['partial'].append(f'{dataset_name} ({num_chunks}/{config["expected_chunks"]})')
                    total_partial += 1

print('=' * 80)
print('RELATORIO DE PROGRESSO DOS EXPERIMENTOS')
print('=' * 80)
print(f'Data da analise: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print()

for chunk_type in sorted(results.keys()):
    data = results[chunk_type]
    print(f'\n{"="*60}')
    print(f'{chunk_type.upper()}')
    print(f'{"="*60}')
    print(f'Completos: {len(data["complete"])} | Parciais: {len(data["partial"])}')

    for batch, batch_data in sorted(data['batches'].items()):
        if batch_data['complete'] or batch_data['partial']:
            print(f'\n  [{batch}]')
            if batch_data['complete']:
                print(f'    [OK] Completos ({len(batch_data["complete"])}):')
                for ds in sorted(batch_data['complete']):
                    print(f'         - {ds}')
            if batch_data['partial']:
                print(f'    [..] Parciais ({len(batch_data["partial"])}):')
                for ds in sorted(batch_data['partial']):
                    print(f'         - {ds}')

print(f'\n{"="*80}')
print(f'RESUMO GERAL')
print(f'{"="*80}')
print(f'Total Completos: {total_complete}')
print(f'Total Parciais:  {total_partial}')
print(f'Total Processado: {total_complete + total_partial}')
