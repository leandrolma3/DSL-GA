#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para Listar Experimentos de Drift Simulation
Extrai todos os experimentos do config.yaml e verifica quais ja tem resultados
"""

import yaml
import os
from pathlib import Path
from collections import defaultdict

# Caminhos
CONFIG_FILE = Path("config.yaml")
RESULTS_DIR = Path("experiments_6chunks_phase2_gbml/batch_1")

# Carregar config
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

experimental_streams = config.get('experimental_streams', {})

# Filtrar apenas experimentos de drift simulation
drift_keywords = ['Abrupt', 'Gradual', 'Incremental', 'Recurring', 'Blip', 'Mixed']

drift_experiments = {}
for exp_name, exp_config in experimental_streams.items():
    if any(keyword in exp_name for keyword in drift_keywords):
        drift_experiments[exp_name] = exp_config

print(f"Total de experimentos drift simulation encontrados: {len(drift_experiments)}")
print()

# Verificar quais ja tem resultados
experiments_with_results = set()
if RESULTS_DIR.exists():
    for item in RESULTS_DIR.iterdir():
        if item.is_dir() and any(keyword in item.name for keyword in drift_keywords):
            experiments_with_results.add(item.name)

print(f"Experimentos com resultados: {len(experiments_with_results)}")
print()

# Experimentos faltantes
missing_experiments = set(drift_experiments.keys()) - experiments_with_results
print(f"Experimentos faltantes: {len(missing_experiments)}")
print()

# Organizar por tipo de drift e dataset
def classify_experiment(exp_name):
    """Classifica experimento por tipo"""
    if 'Noise' in exp_name:
        return 'noise'
    elif 'Gradual' in exp_name:
        return 'gradual'
    elif 'Abrupt' in exp_name:
        return 'abrupt'
    elif 'Mixed' in exp_name:
        return 'mixed'
    elif 'Incremental' in exp_name:
        return 'incremental'
    return 'other'

def get_dataset_base(exp_name):
    """Extrai nome base do dataset"""
    for ds in ['SEA', 'AGRAWAL', 'RBF', 'STAGGER', 'HYPERPLANE', 'RANDOMTREE',
               'LED', 'WAVEFORM', 'SINE']:
        if exp_name.startswith(ds):
            return ds
    return 'OTHER'

# Organizar experimentos completos
print("="*80)
print("EXPERIMENTOS COM RESULTADOS (BATCH 1)")
print("="*80)
for exp in sorted(experiments_with_results):
    print(f"  {exp}")

print()
print("="*80)
print("EXPERIMENTOS FALTANTES - ORGANIZADOS POR TIPO")
print("="*80)

by_type = defaultdict(list)
for exp in missing_experiments:
    exp_type = classify_experiment(exp)
    by_type[exp_type].append(exp)

for exp_type, exps in sorted(by_type.items()):
    print(f"\n{exp_type.upper()} ({len(exps)} experimentos):")
    for exp in sorted(exps):
        dataset = get_dataset_base(exp)
        print(f"  [{dataset:<12}] {exp}")

# Organizar por dataset
print()
print("="*80)
print("EXPERIMENTOS FALTANTES - ORGANIZADOS POR DATASET")
print("="*80)

by_dataset = defaultdict(list)
for exp in missing_experiments:
    dataset = get_dataset_base(exp)
    by_dataset[dataset].append(exp)

for dataset, exps in sorted(by_dataset.items()):
    print(f"\n{dataset} ({len(exps)} experimentos):")
    for exp in sorted(exps):
        exp_type = classify_experiment(exp)
        print(f"  [{exp_type:<12}] {exp}")

# Estatisticas gerais
print()
print("="*80)
print("ESTATISTICAS")
print("="*80)
print(f"Total experimentos drift: {len(drift_experiments)}")
print(f"Com resultados (Batch 1): {len(experiments_with_results)}")
print(f"Faltantes: {len(missing_experiments)}")
print()
print("Por tipo:")
for exp_type, exps in sorted(by_type.items()):
    print(f"  {exp_type:<15}: {len(exps)} experimentos")
print()
print("Por dataset:")
for dataset, exps in sorted(by_dataset.items()):
    print(f"  {dataset:<15}: {len(exps)} experimentos")

# Salvar lista completa em arquivo
with open('experimentos_drift_completo.txt', 'w') as f:
    f.write("EXPERIMENTOS DE DRIFT SIMULATION\n")
    f.write("="*80 + "\n\n")

    f.write("COM RESULTADOS (Batch 1):\n")
    f.write("-"*80 + "\n")
    for exp in sorted(experiments_with_results):
        f.write(f"  {exp}\n")

    f.write("\n\nFALTANTES:\n")
    f.write("-"*80 + "\n")
    for exp in sorted(missing_experiments):
        f.write(f"  {exp}\n")

print()
print(f"Lista completa salva em: experimentos_drift_completo.txt")
