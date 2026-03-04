#!/usr/bin/env python3
"""
generate_unified_configs.py

Gera arquivos YAML de configuracao para os experimentos unificados.
Cria batches separados para cada chunk_size, respeitando o limite de tempo do Colab (~24h).

Estrutura de batches:
- chunk_size 2000: 7 batches (6-12 datasets cada)
- chunk_size 1000: 4 batches (~13 datasets cada)
- chunk_size 500: 3 batches (~17-18 datasets cada)

Cada tamanho tem versoes com e sem penalidade para o EGIS.
"""

import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Lista completa dos 52 datasets (mesma ordem dos batches originais)
ALL_DATASETS = [
    # Batch 1 original (12 datasets - Abrupt drift)
    "SEA_Abrupt_Simple", "SEA_Abrupt_Chain", "SEA_Abrupt_Recurring",
    "AGRAWAL_Abrupt_Simple_Mild", "AGRAWAL_Abrupt_Simple_Severe", "AGRAWAL_Abrupt_Chain_Long",
    "RBF_Abrupt_Severe", "RBF_Abrupt_Blip",
    "STAGGER_Abrupt_Chain", "STAGGER_Abrupt_Recurring",
    "HYPERPLANE_Abrupt_Simple", "RANDOMTREE_Abrupt_Simple",

    # Batch 2 original (9 datasets - Gradual drift)
    "SEA_Gradual_Simple_Fast", "SEA_Gradual_Simple_Slow", "SEA_Gradual_Recurring",
    "STAGGER_Gradual_Chain",
    "RBF_Gradual_Moderate", "RBF_Gradual_Severe",
    "HYPERPLANE_Gradual_Simple", "RANDOMTREE_Gradual_Simple",
    "LED_Gradual_Simple",

    # Batch 3 original (8 datasets - Noise)
    "SEA_Abrupt_Chain_Noise",
    "STAGGER_Abrupt_Chain_Noise",
    "AGRAWAL_Abrupt_Simple_Severe_Noise",
    "SINE_Abrupt_Recurring_Noise",
    "RBF_Abrupt_Blip_Noise", "RBF_Gradual_Severe_Noise",
    "HYPERPLANE_Gradual_Noise",
    "RANDOMTREE_Gradual_Noise",

    # Batch 4 original (6 datasets - SINE, LED, WAVEFORM, RANDOMTREE)
    "SINE_Abrupt_Simple", "SINE_Gradual_Recurring",
    "LED_Abrupt_Simple",
    "WAVEFORM_Abrupt_Simple", "WAVEFORM_Gradual_Simple",
    "RANDOMTREE_Abrupt_Recurring",

    # Batch 5 original (5 datasets - Reais)
    "Electricity", "Shuttle", "CovType", "PokerHand", "IntelLabSensors",

    # Batch 6 original (6 datasets - Stationary parte 1)
    "SEA_Stationary", "AGRAWAL_Stationary", "RBF_Stationary",
    "LED_Stationary", "HYPERPLANE_Stationary", "RANDOMTREE_Stationary",

    # Batch 7 original (6 datasets - Stationary parte 2 + AssetNegotiation)
    "STAGGER_Stationary", "WAVEFORM_Stationary", "SINE_Stationary",
    "AssetNegotiation_F2", "AssetNegotiation_F3", "AssetNegotiation_F4",
]

# Verificar que temos exatamente 52 datasets
assert len(ALL_DATASETS) == 52, f"Esperado 52 datasets, encontrado {len(ALL_DATASETS)}"

# Distribuicao de batches por chunk_size
# Baseado na experiencia anterior com chunk_size 2000
BATCH_CONFIG = {
    2000: {
        # 7 batches, mesmo que antes
        'batches': [
            ALL_DATASETS[0:12],   # Batch 1: 12 datasets (Abrupt)
            ALL_DATASETS[12:21],  # Batch 2: 9 datasets (Gradual)
            ALL_DATASETS[21:29],  # Batch 3: 8 datasets (Noise)
            ALL_DATASETS[29:35],  # Batch 4: 6 datasets (SINE, LED, WAVEFORM)
            ALL_DATASETS[35:40],  # Batch 5: 5 datasets (Reais)
            ALL_DATASETS[40:46],  # Batch 6: 6 datasets (Stationary 1)
            ALL_DATASETS[46:52],  # Batch 7: 6 datasets (Stationary 2 + Asset)
        ]
    },
    1000: {
        # 4 batches (tempo ~metade, podemos dobrar datasets)
        'batches': [
            ALL_DATASETS[0:13],   # Batch 1: 13 datasets
            ALL_DATASETS[13:26],  # Batch 2: 13 datasets
            ALL_DATASETS[26:39],  # Batch 3: 13 datasets
            ALL_DATASETS[39:52],  # Batch 4: 13 datasets
        ]
    },
    500: {
        # 3 batches (tempo ~1/4, podemos quadruplicar datasets)
        'batches': [
            ALL_DATASETS[0:18],   # Batch 1: 18 datasets
            ALL_DATASETS[18:35],  # Batch 2: 17 datasets
            ALL_DATASETS[35:52],  # Batch 3: 17 datasets
        ]
    }
}


def load_base_config(config_path: str) -> Dict[str, Any]:
    """Carrega o config base para usar como template."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_config(
    base_config: Dict[str, Any],
    chunk_size: int,
    batch_num: int,
    datasets: List[str],
    with_penalty: bool,
    chunks_base_dir: str,
    results_base_dir: str
) -> Dict[str, Any]:
    """
    Gera um config para um batch especifico.
    """
    config = {}

    # Calcular num_chunks baseado no tamanho
    num_chunks = 6 * (2000 // chunk_size)
    max_instances = chunk_size * num_chunks

    # Nome do subdiretorio
    penalty_suffix = "_penalty" if with_penalty else ""
    chunk_subdir = f"chunk_{chunk_size}"
    results_subdir = f"chunk_{chunk_size}{penalty_suffix}"

    # Configuracoes de experimento
    config['experiment_settings'] = {
        'run_mode': 'unified_pregenerated',
        'use_pregenerated_chunks': True,
        'pregenerated_chunks_base_dir': f"{chunks_base_dir}/{chunk_subdir}",
        'unified_experiments': datasets,
        'num_runs': 1,
        'base_results_dir': f"{results_base_dir}/{results_subdir}/batch_{batch_num}",
        'logging_level': 'INFO',
        'evaluation_period': chunk_size
    }

    # Parametros de dados
    config['data_params'] = {
        'chunk_size': chunk_size,
        'num_chunks': num_chunks,
        'max_instances': max_instances
    }

    # Parametros de GA (copiar do base ou usar defaults)
    config['ga_params'] = base_config.get('ga_params', {
        'population_size': 120,
        'max_generations': 200,
        'max_generations_recovery': 25,
        'recovery_generation_multiplier': 1.5,
        'enable_explicit_drift_adaptation': True,
        'recovery_mutation_override_rate': 0.5,
        'recovery_mutation_override_generations': 10,
        'recovery_initialization_strategy': 'full_random',
        'recovery_random_individual_ratio': 0.6,
        'recovery_max_depth_multiplier': 1.5,
        'recovery_max_rules_multiplier': 1.5,
        'elitism_rate': 0.1,
        'intelligent_mutation_rate': 0.8,
        'initial_tournament_size': 2,
        'final_tournament_size': 5,
        'max_rules_per_class': 15,
        'initial_max_depth': 10,
        'stagnation_threshold': 10,
        'early_stopping_patience': 20,
        'hc_enable_adaptive': False,
        'hc_gmean_threshold': 0.9,
        'hc_hierarchical_enabled': True,
        'enable_dt_seeding_on_init': True,
        'dt_seeding_ratio_on_init': 0.8,
        'dt_seeding_depths_on_init': [4, 7, 10, 13],
        'dt_seeding_sample_size_on_init': 2000,
        'dt_seeding_rules_to_replace_per_class': 4,
        'dt_rule_injection_ratio': 0.5,
        'enable_adaptive_seeding': True,
        'adaptive_seeding_strategy': 'dt_probe',
        'adaptive_complexity_simple_threshold': 0.9,
        'adaptive_complexity_medium_threshold': 0.75,
        'use_balanced_crossover': True
    })

    # Parametros de memoria
    config['memory_params'] = base_config.get('memory_params', {
        'max_memory_size': 20,
        'enable_active_memory_pruning': True,
        'memory_max_age_chunks': 10,
        'memory_fitness_threshold_percentile_for_old_removal': 0.25,
        'memory_min_retain_count_during_pruning': 5,
        'abandon_memory_on_severe_performance_drop': True,
        'performance_drop_threshold_for_memory_abandon': 0.55,
        'consecutive_bad_chunks_for_memory_abandon': 1,
        'historical_reference_size': 500
    })

    # Parametros de fitness (penalidade)
    fitness_params = base_config.get('fitness_params', {}).copy()

    if with_penalty:
        fitness_params['feature_penalty_coefficient'] = 0.1
        fitness_params['operator_penalty_coefficient'] = 0.0001
        fitness_params['threshold_penalty_coefficient'] = 0.0001
    else:
        fitness_params['feature_penalty_coefficient'] = 0.0
        fitness_params['operator_penalty_coefficient'] = 0.0
        fitness_params['threshold_penalty_coefficient'] = 0.0

    # Defaults para fitness
    defaults = {
        'class_coverage_coefficient': 0.2,
        'gmean_bonus_coefficient': 0.1,
        'initial_regularization_coefficient': 0.001,
        'operator_change_coefficient': 0.05,
        'gamma': 0.1,
        'drift_penalty_reduction_threshold': 0.1,
        'absolute_bad_threshold_for_label': 0.6,
        'min_regularization_coeff': 0.01,
        'max_regularization_coeff': 0.3
    }
    for key, value in defaults.items():
        if key not in fitness_params:
            fitness_params[key] = value

    config['fitness_params'] = fitness_params

    # Parametros de paralelismo
    config['parallelism'] = base_config.get('parallelism', {
        'enabled': True,
        'num_workers': None
    })

    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Salva o config como arquivo YAML."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(description='Gera configs YAML unificados com batches')
    parser.add_argument('--base-config', type=str, default='config.yaml',
                        help='Arquivo de config base (template)')
    parser.add_argument('--output-dir', type=str, default='configs',
                        help='Diretorio de saida para os YAMLs')
    parser.add_argument('--chunks-base-dir', type=str,
                        default='/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/unified_chunks',
                        help='Diretorio base dos chunks pre-gerados (Google Drive)')
    parser.add_argument('--results-base-dir', type=str,
                        default='/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/experiments_unified',
                        help='Diretorio base para resultados')

    args = parser.parse_args()

    # Criar diretorio de saida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carregar config base
    print(f"Carregando config base: {args.base_config}")
    base_config = load_base_config(args.base_config)

    print(f"\nDiretorio de saida: {output_dir}")
    print(f"Chunks base dir: {args.chunks_base_dir}")
    print(f"Results base dir: {args.results_base_dir}")
    print(f"Total de datasets: {len(ALL_DATASETS)}")

    # Contador de arquivos gerados
    files_generated = []

    # Gerar configs para cada chunk_size
    for chunk_size, batch_info in BATCH_CONFIG.items():
        batches = batch_info['batches']
        num_batches = len(batches)

        print(f"\n{'='*60}")
        print(f"CHUNK SIZE: {chunk_size}")
        print(f"{'='*60}")
        print(f"Numero de batches: {num_batches}")

        # Verificar que todos os datasets estao cobertos
        all_in_batches = []
        for b in batches:
            all_in_batches.extend(b)

        if len(all_in_batches) != 52:
            print(f"AVISO: Total de datasets nos batches = {len(all_in_batches)}, esperado 52")
        if set(all_in_batches) != set(ALL_DATASETS):
            missing = set(ALL_DATASETS) - set(all_in_batches)
            extra = set(all_in_batches) - set(ALL_DATASETS)
            if missing:
                print(f"ERRO: Datasets faltando: {missing}")
            if extra:
                print(f"ERRO: Datasets extras: {extra}")

        # Gerar versao SEM penalty
        print(f"\n  Gerando configs SEM penalty...")
        for batch_num, datasets in enumerate(batches, 1):
            filename = f"config_unified_chunk{chunk_size}_batch_{batch_num}.yaml"
            config = generate_config(
                base_config=base_config,
                chunk_size=chunk_size,
                batch_num=batch_num,
                datasets=datasets,
                with_penalty=False,
                chunks_base_dir=args.chunks_base_dir,
                results_base_dir=args.results_base_dir
            )
            output_path = output_dir / filename
            save_config(config, str(output_path))
            files_generated.append(filename)
            print(f"    Batch {batch_num}: {len(datasets)} datasets -> {filename}")

        # Gerar versao COM penalty
        print(f"\n  Gerando configs COM penalty...")
        for batch_num, datasets in enumerate(batches, 1):
            filename = f"config_unified_chunk{chunk_size}_penalty_batch_{batch_num}.yaml"
            config = generate_config(
                base_config=base_config,
                chunk_size=chunk_size,
                batch_num=batch_num,
                datasets=datasets,
                with_penalty=True,
                chunks_base_dir=args.chunks_base_dir,
                results_base_dir=args.results_base_dir
            )
            output_path = output_dir / filename
            save_config(config, str(output_path))
            files_generated.append(filename)
            print(f"    Batch {batch_num}: {len(datasets)} datasets -> {filename}")

    # Resumo final
    print(f"\n{'='*60}")
    print("RESUMO DA GERACAO")
    print(f"{'='*60}")

    print(f"\nTotal de arquivos gerados: {len(files_generated)}")

    print("\nDistribuicao por chunk_size:")
    for chunk_size, batch_info in BATCH_CONFIG.items():
        num_batches = len(batch_info['batches'])
        datasets_per_batch = [len(b) for b in batch_info['batches']]
        print(f"  chunk_{chunk_size}:")
        print(f"    - Batches sem penalty: {num_batches}")
        print(f"    - Batches com penalty: {num_batches}")
        print(f"    - Datasets por batch: {datasets_per_batch}")
        print(f"    - Total datasets: {sum(datasets_per_batch)}")

    print(f"\nArquivos gerados:")
    for f in sorted(files_generated):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
