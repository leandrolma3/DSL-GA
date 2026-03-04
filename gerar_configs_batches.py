#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para Gerar Configs Corrigidos dos Batches 2, 3 e 4
Aplica correcoes automaticas de duration_chunks e paths
"""

import yaml
from pathlib import Path
from collections import OrderedDict

# Configuracoes
CONFIG_ORIGINAL = Path("config.yaml")
CONFIGS_DIR = Path("configs")
CHUNK_SIZE = 1000
NUM_CHUNKS = 6
TRAIN_END_INSTANCE = 5000

# Representadores YAML para manter ordem
def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())

yaml.add_representer(OrderedDict, represent_ordereddict)

def calculate_corrected_durations(num_concepts, gradual_width):
    """Calcula duration_chunks corrigidos"""
    # Total de chunks disponiveis (6 chunks menos as transicoes)
    available_chunks = NUM_CHUNKS - (num_concepts - 1) * gradual_width

    if available_chunks < num_concepts:
        return None  # Impossivel

    # Distribuir chunks
    base_duration = available_chunks // num_concepts
    remainder = available_chunks % num_concepts

    durations = [base_duration] * num_concepts

    # Distribuir o resto (dar aos ultimos conceitos)
    for i in range(remainder):
        durations[-(i+1)] += 1

    return durations

def validate_drift_positions(concept_sequence, gradual_width):
    """Valida que drifts estao dentro do range"""
    current_inst = 0
    drifts = []

    for i, concept in enumerate(concept_sequence[:-1]):
        duration = concept['duration_chunks']
        current_inst += duration * CHUNK_SIZE

        drift_start = current_inst
        drift_end = current_inst + gradual_width * CHUNK_SIZE

        if drift_start >= TRAIN_END_INSTANCE or drift_end > TRAIN_END_INSTANCE:
            return False, f"Drift {i+1} em {drift_start}-{drift_end} (FORA!)"

        drifts.append((drift_start, drift_end))
        current_inst = drift_end

    return True, drifts

# ============================================================================
# BATCH 2: GRADUAL (9 experimentos viaveis)
# ============================================================================

BATCH_2_EXPERIMENTS = {
    'SEA_Gradual_Simple_Fast': {
        'original_durations': [5, 5],
        'corrected_durations': [2, 3],
        'gradual_width': 1,
        'justificativa': 'Drift gradual rapido em 2000-3000'
    },
    'SEA_Gradual_Simple_Slow': {
        'original_durations': [5, 5],
        'corrected_durations': [2, 2],
        'gradual_width': 2,
        'justificativa': 'Drift gradual lento em 2000-4000'
    },
    'SEA_Gradual_Recurring': {
        'original_durations': [4, 5, 4],
        'corrected_durations': [2, 1, 2],
        'gradual_width': 1,
        'justificativa': 'Drifts em 2000-3000, 4000-5000 (recorrente)'
    },
    'STAGGER_Gradual_Chain': {
        'original_durations': [4, 4, 4],
        'corrected_durations': [2, 1, 2],
        'gradual_width': 1,
        'justificativa': 'Drifts em 2000-3000, 4000-5000'
    },
    'RBF_Gradual_Moderate': {
        'original_durations': [5, 5],
        'corrected_durations': [2, 2],
        'gradual_width': 2,
        'justificativa': 'Drift moderado em 2000-4000'
    },
    'RBF_Gradual_Severe': {
        'original_durations': [5, 5],
        'corrected_durations': [2, 2],
        'gradual_width': 2,
        'justificativa': 'Drift severo em 2000-4000'
    },
    'HYPERPLANE_Gradual_Simple': {
        'original_durations': [6, 6],
        'corrected_durations': [1, 2],
        'gradual_width': 3,
        'justificativa': 'Drift em 1000-4000 (width 3)'
    },
    'RANDOMTREE_Gradual_Simple': {
        'original_durations': [5, 5],
        'corrected_durations': [2, 2],
        'gradual_width': 2,
        'justificativa': 'Drift em 2000-4000'
    },
    'LED_Gradual_Simple': {
        'original_durations': [5, 5],
        'corrected_durations': [2, 2],
        'gradual_width': 2,
        'justificativa': 'Drift em 2000-4000'
    },
}

# ============================================================================
# BATCH 3: NOISE & MIXED (9 experimentos viaveis)
# ============================================================================

BATCH_3_EXPERIMENTS = {
    'SEA_Abrupt_Chain_Noise': {
        'original_durations': [3, 3, 3],
        'corrected_durations': [2, 2, 2],
        'gradual_width': 0,
        'justificativa': 'Drifts abrupt em 2000, 4000 com ruido'
    },
    'STAGGER_Abrupt_Chain_Noise': {
        'original_durations': [4, 4, 4],
        'corrected_durations': [2, 2, 2],
        'gradual_width': 0,
        'justificativa': 'Drifts abrupt em 2000, 4000 com ruido'
    },
    # EXCLUIDO: STAGGER_Mixed_Recurring (4 conceitos, width 2, impossivel)
    'AGRAWAL_Abrupt_Simple_Severe_Noise': {
        'original_durations': [5, 5],
        'corrected_durations': [3, 3],
        'gradual_width': 0,
        'justificativa': 'Drift severo em 3000 com ruido'
    },
    'SINE_Abrupt_Recurring_Noise': {
        'original_durations': [4, 5, 4],
        'corrected_durations': [2, 2, 2],
        'gradual_width': 0,
        'justificativa': 'Drifts em 2000, 4000 (recorrente) com ruido'
    },
    'RBF_Abrupt_Blip_Noise': {
        'original_durations': [6, 1, 6],
        'corrected_durations': [2, 2, 2],
        'gradual_width': 0,
        'justificativa': 'Blip em 2000-4000 com ruido'
    },
    'RBF_Gradual_Severe_Noise': {
        'original_durations': [5, 5],
        'corrected_durations': [2, 2],
        'gradual_width': 2,
        'justificativa': 'Drift gradual severo em 2000-4000 com ruido'
    },
    'HYPERPLANE_Gradual_Noise': {
        'original_durations': [6, 6],
        'corrected_durations': [1, 2],
        'gradual_width': 3,
        'justificativa': 'Drift gradual em 1000-4000 com ruido'
    },
    'RANDOMTREE_Gradual_Noise': {
        'original_durations': [5, 5],
        'corrected_durations': [2, 2],
        'gradual_width': 2,
        'justificativa': 'Drift gradual em 2000-4000 com ruido'
    },
}

# ============================================================================
# BATCH 4: COMPLEMENTARES (8 experimentos viaveis)
# ============================================================================

BATCH_4_EXPERIMENTS = {
    'SINE_Abrupt_Simple': {
        'original_durations': [5, 5],
        'corrected_durations': [3, 3],
        'gradual_width': 0,
        'justificativa': 'Drift abrupt em 3000'
    },
    'SINE_Gradual_Recurring': {
        'original_durations': [4, 5, 4],
        'corrected_durations': [2, 1, 2],
        'gradual_width': 1,
        'justificativa': 'Drifts graduais em 2000-3000, 4000-5000 (recorrente)'
    },
    'LED_Abrupt_Simple': {
        'original_durations': [5, 5],
        'corrected_durations': [3, 3],
        'gradual_width': 0,
        'justificativa': 'Drift abrupt em 3000'
    },
    'WAVEFORM_Abrupt_Simple': {
        'original_durations': [5, 5],
        'corrected_durations': [3, 3],
        'gradual_width': 0,
        'justificativa': 'Drift abrupt em 3000'
    },
    'WAVEFORM_Gradual_Simple': {
        'original_durations': [5, 5],
        'corrected_durations': [2, 2],
        'gradual_width': 2,
        'justificativa': 'Drift gradual em 2000-4000'
    },
    'RANDOMTREE_Abrupt_Recurring': {
        'original_durations': [4, 5, 4],
        'corrected_durations': [2, 2, 2],
        'gradual_width': 0,
        'justificativa': 'Drifts abrupt em 2000, 4000 (recorrente)'
    },
}

# Excluidos por terem duracao/width problematica:
# - RBF_Severe_Gradual_Recurrent (3 conceitos, width 2)

def create_batch_config(batch_number, experiments_dict, base_config):
    """Cria config de um batch com correcoes"""

    # Copiar estrutura base
    batch_config = OrderedDict()

    # Experiment settings
    batch_config['experiment_settings'] = OrderedDict([
        ('run_mode', 'drift_simulation'),
        ('drift_simulation_experiments', list(experiments_dict.keys())),
        ('num_runs', 1)
    ])

    # Evaluation settings
    batch_config['evaluation_settings'] = OrderedDict([
        ('chunk_size', CHUNK_SIZE),
        ('chunks_to_process', NUM_CHUNKS)
    ])

    # Paths
    batch_config['base_results_dir'] = f'experiments_6chunks_phase2_gbml/batch_{batch_number}'

    # GA params (copiar do original)
    batch_config['ga_params'] = base_config.get('ga_params', {})
    batch_config['fitness_params'] = base_config.get('fitness_params', {})
    batch_config['memory_params'] = base_config.get('memory_params', {})
    batch_config['parallelization'] = base_config.get('parallelization', {})

    # Drift analysis
    batch_config['drift_analysis'] = OrderedDict([
        ('severity_samples', 20000),
        ('heatmap_save_directory', 'experiments_6chunks_phase2_gbml/test_real_results_heatmaps/concept_heatmaps'),
        ('datasets', base_config.get('drift_analysis', {}).get('datasets', {}))
    ])

    # Experimental streams (aplicar correcoes)
    batch_config['experimental_streams'] = OrderedDict()

    original_streams = base_config.get('experimental_streams', {})

    for exp_name, corrections in experiments_dict.items():
        if exp_name not in original_streams:
            print(f"[WARNING] {exp_name} nao encontrado no config original")
            continue

        # Copiar stream original
        stream = OrderedDict(original_streams[exp_name])

        # Aplicar correcoes
        corrected_durations = corrections['corrected_durations']

        for i, concept in enumerate(stream.get('concept_sequence', [])):
            if i < len(corrected_durations):
                concept['duration_chunks'] = corrected_durations[i]

        # Ajustar gradual_drift_width se necessario
        if 'gradual_width' in corrections:
            stream['gradual_drift_width_chunks'] = corrections['gradual_width']

        batch_config['experimental_streams'][exp_name] = stream

        # Validar
        valid, info = validate_drift_positions(
            stream.get('concept_sequence', []),
            stream.get('gradual_drift_width_chunks', 0)
        )

        if valid:
            print(f"[OK] {exp_name}: Drifts em {info}")
        else:
            print(f"[ERRO] {exp_name}: {info}")

    return batch_config

def main():
    """Funcao principal"""
    print("="*100)
    print("GERACAO DE CONFIGS CORRIGIDOS - BATCHES 2, 3, 4")
    print("="*100)
    print()

    # Carregar config original
    with open(CONFIG_ORIGINAL, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    print(f"Config original carregado: {CONFIG_ORIGINAL}")
    print()

    # Criar diretorio configs se nao existir
    CONFIGS_DIR.mkdir(exist_ok=True)

    # Gerar configs
    batches = [
        (2, BATCH_2_EXPERIMENTS, "Gradual Fundamentais"),
        (3, BATCH_3_EXPERIMENTS, "Noise & Mixed"),
        (4, BATCH_4_EXPERIMENTS, "Complementares")
    ]

    for batch_num, experiments, description in batches:
        print(f"\n{'='*100}")
        print(f"BATCH {batch_num}: {description} ({len(experiments)} experimentos)")
        print(f"{'='*100}\n")

        batch_config = create_batch_config(batch_num, experiments, base_config)

        # Salvar
        output_file = CONFIGS_DIR / f"config_batch_{batch_num}.yaml"

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(batch_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"\n[SALVO] {output_file}")
        print(f"  - Experimentos: {len(experiments)}")
        print(f"  - Path: experiments_6chunks_phase2_gbml/batch_{batch_num}")

    print()
    print("="*100)
    print("RESUMO")
    print("="*100)
    print("Configs criados:")
    print(f"  - config_batch_2.yaml: 9 experimentos gradual")
    print(f"  - config_batch_3.yaml: 8 experimentos noise + mixed")
    print(f"  - config_batch_4.yaml: 6 experimentos complementares")
    print(f"\nTotal: 23 experimentos novos (+ 12 do Batch 1 = 35 total)")
    print()
    print("Experimentos EXCLUIDOS (impossíveis - 7 experimentos):")
    print("  - AGRAWAL_Gradual_Chain (4 conceitos, width 2)")
    print("  - AGRAWAL_Gradual_Mild_to_Severe (3 conceitos, width 2)")
    print("  - AGRAWAL_Gradual_Blip (3 conceitos, width 1)")
    print("  - AGRAWAL_Gradual_Recurring (3 conceitos, width 2)")
    print("  - AGRAWAL_Gradual_Recurring_Noise (3 conceitos, width 2)")
    print("  - RBF_Severe_Gradual_Recurrent (3 conceitos, width 2)")
    print("  - STAGGER_Mixed_Recurring (4 conceitos, width 2)")
    print()
    print("[SUCESSO] Configs prontos para uso!")

if __name__ == '__main__':
    main()
