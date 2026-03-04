# =============================================================================
# CELULA 2.1: Configuracao dos Experimentos (UNIFIED + ORIGINAIS)
# =============================================================================
# SUBSTITUA a CELULA 2.1 original por este codigo
# Adiciona experimentos unified (chunk_500, chunk_1000) com suporte a penalty
# Mantem experimentos originais (exp_a, exp_b, exp_c) para compatibilidade
# =============================================================================

import numpy as np
import pandas as pd
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS IMPORTANTES
# =============================================================================
UNIFIED_CHUNKS_DIR = Path(WORK_DIR) / "unified_chunks"
EXPERIMENTS_UNIFIED_DIR = Path(WORK_DIR) / "experiments_unified"

# =============================================================================
# DATASETS MULTICLASSE (CDCMS N/A, ACDWM N/A)
# =============================================================================
MULTICLASS_DATASETS = {
    'LED_Abrupt_Simple': 10,
    'LED_Gradual_Simple': 10,
    'LED_Stationary': 10,
    'WAVEFORM_Abrupt_Simple': 3,
    'WAVEFORM_Gradual_Simple': 3,
    'WAVEFORM_Stationary': 3,
    'CovType': 7,
    'Shuttle': 7,
    'RBF_Stationary': 4,
}

# =============================================================================
# CONFIGURACAO DOS EXPERIMENTOS UNIFIED (NOVOS)
# =============================================================================

UNIFIED_EXPERIMENT_CONFIGS = {
    # =========================================================================
    # CHUNK 500 - SEM PENALTY (foco em performance)
    # =========================================================================
    'exp_unified_500': {
        'chunk_size': 500,
        'penalty_weight': 0.0,
        'data_source': 'unified',
        'data_dir': 'chunk_500',
        'results_dir': 'chunk_500',
        'egis_model_name': 'EGIS',
        'description': 'Unified chunks 500 (sem penalty)',
        'batches': {
            'batch_1': {
                'base_dir': 'experiments_unified/chunk_500/batch_1',
                'datasets': [
                    'SEA_Abrupt_Simple', 'SEA_Abrupt_Chain', 'SEA_Abrupt_Recurring',
                    'SEA_Gradual_Simple_Fast', 'SEA_Gradual_Simple_Slow', 'SEA_Gradual_Recurring',
                    'AGRAWAL_Abrupt_Simple_Mild', 'AGRAWAL_Abrupt_Simple_Severe', 'AGRAWAL_Abrupt_Chain_Long',
                    'HYPERPLANE_Abrupt_Simple',
                    'RANDOMTREE_Abrupt_Simple',
                    'RBF_Abrupt_Severe', 'RBF_Abrupt_Blip', 'RBF_Gradual_Moderate', 'RBF_Gradual_Severe',
                    'STAGGER_Abrupt_Chain', 'STAGGER_Abrupt_Recurring', 'STAGGER_Gradual_Chain'
                ]
            },
            'batch_2': {
                'base_dir': 'experiments_unified/chunk_500/batch_2',
                'datasets': [
                    'SEA_Abrupt_Chain_Noise',
                    'AGRAWAL_Abrupt_Simple_Severe_Noise',
                    'HYPERPLANE_Gradual_Simple', 'HYPERPLANE_Gradual_Noise',
                    'RANDOMTREE_Gradual_Simple', 'RANDOMTREE_Abrupt_Recurring', 'RANDOMTREE_Gradual_Noise',
                    'RBF_Abrupt_Blip_Noise', 'RBF_Gradual_Severe_Noise',
                    'STAGGER_Abrupt_Chain_Noise',
                    'LED_Abrupt_Simple', 'LED_Gradual_Simple',
                    'WAVEFORM_Abrupt_Simple', 'WAVEFORM_Gradual_Simple',
                    'SINE_Abrupt_Simple', 'SINE_Gradual_Recurring', 'SINE_Abrupt_Recurring_Noise'
                ]
            },
            'batch_3': {
                'base_dir': 'experiments_unified/chunk_500/batch_3',
                'datasets': [
                    'SEA_Stationary', 'AGRAWAL_Stationary', 'HYPERPLANE_Stationary',
                    'RANDOMTREE_Stationary', 'RBF_Stationary', 'STAGGER_Stationary',
                    'LED_Stationary', 'WAVEFORM_Stationary', 'SINE_Stationary',
                    'Electricity', 'Shuttle', 'CovType', 'PokerHand', 'IntelLabSensors',
                    'AssetNegotiation_F2', 'AssetNegotiation_F3', 'AssetNegotiation_F4'
                ]
            }
        }
    },

    # =========================================================================
    # CHUNK 500 - COM PENALTY (foco em interpretabilidade)
    # =========================================================================
    'exp_unified_500_penalty': {
        'chunk_size': 500,
        'penalty_weight': 0.1,
        'data_source': 'unified',
        'data_dir': 'chunk_500',  # Dados sao os mesmos
        'results_dir': 'chunk_500_penalty',  # Resultados EGIS diferentes
        'egis_model_name': 'EGIS_Penalty',
        'description': 'Unified chunks 500 (com penalty)',
        'reuse_comparative_from': 'exp_unified_500',  # Reutiliza modelos comparativos
        'batches': {
            'batch_1': {
                'base_dir': 'experiments_unified/chunk_500_penalty/batch_1',
                'datasets': [
                    'SEA_Abrupt_Simple', 'SEA_Abrupt_Chain', 'SEA_Abrupt_Recurring',
                    'SEA_Gradual_Simple_Fast', 'SEA_Gradual_Simple_Slow', 'SEA_Gradual_Recurring',
                    'AGRAWAL_Abrupt_Simple_Mild', 'AGRAWAL_Abrupt_Simple_Severe', 'AGRAWAL_Abrupt_Chain_Long',
                    'HYPERPLANE_Abrupt_Simple',
                    'RANDOMTREE_Abrupt_Simple',
                    'RBF_Abrupt_Severe', 'RBF_Abrupt_Blip', 'RBF_Gradual_Moderate', 'RBF_Gradual_Severe',
                    'STAGGER_Abrupt_Chain', 'STAGGER_Abrupt_Recurring', 'STAGGER_Gradual_Chain'
                ]
            },
            'batch_2': {
                'base_dir': 'experiments_unified/chunk_500_penalty/batch_2',
                'datasets': [
                    'SEA_Abrupt_Chain_Noise',
                    'AGRAWAL_Abrupt_Simple_Severe_Noise',
                    'HYPERPLANE_Gradual_Simple', 'HYPERPLANE_Gradual_Noise',
                    'RANDOMTREE_Gradual_Simple', 'RANDOMTREE_Abrupt_Recurring', 'RANDOMTREE_Gradual_Noise',
                    'RBF_Abrupt_Blip_Noise', 'RBF_Gradual_Severe_Noise',
                    'STAGGER_Abrupt_Chain_Noise',
                    'LED_Abrupt_Simple', 'LED_Gradual_Simple',
                    'WAVEFORM_Abrupt_Simple', 'WAVEFORM_Gradual_Simple',
                    'SINE_Abrupt_Simple', 'SINE_Gradual_Recurring', 'SINE_Abrupt_Recurring_Noise'
                ]
            },
            'batch_3': {
                'base_dir': 'experiments_unified/chunk_500_penalty/batch_3',
                'datasets': [
                    'SEA_Stationary', 'AGRAWAL_Stationary', 'HYPERPLANE_Stationary',
                    'RANDOMTREE_Stationary', 'RBF_Stationary', 'STAGGER_Stationary',
                    'LED_Stationary', 'WAVEFORM_Stationary', 'SINE_Stationary',
                    'Electricity', 'Shuttle', 'CovType', 'PokerHand', 'IntelLabSensors',
                    'AssetNegotiation_F2', 'AssetNegotiation_F3', 'AssetNegotiation_F4'
                ]
            }
        }
    },

    # =========================================================================
    # CHUNK 1000 - SEM PENALTY (foco em performance)
    # =========================================================================
    'exp_unified_1000': {
        'chunk_size': 1000,
        'penalty_weight': 0.0,
        'data_source': 'unified',
        'data_dir': 'chunk_1000',
        'results_dir': 'chunk_1000',
        'egis_model_name': 'EGIS',
        'description': 'Unified chunks 1000 (sem penalty)',
        'batches': {
            'batch_1': {
                'base_dir': 'experiments_unified/chunk_1000/batch_1',
                'datasets': [
                    'SEA_Abrupt_Simple', 'SEA_Abrupt_Chain', 'SEA_Abrupt_Recurring', 'SEA_Gradual_Simple_Fast',
                    'AGRAWAL_Abrupt_Simple_Mild', 'AGRAWAL_Abrupt_Simple_Severe', 'AGRAWAL_Abrupt_Chain_Long',
                    'HYPERPLANE_Abrupt_Simple',
                    'RANDOMTREE_Abrupt_Simple',
                    'RBF_Abrupt_Blip', 'RBF_Abrupt_Severe',
                    'STAGGER_Abrupt_Chain', 'STAGGER_Abrupt_Recurring'
                ]
            },
            'batch_2': {
                'base_dir': 'experiments_unified/chunk_1000/batch_2',
                'datasets': [
                    'SEA_Abrupt_Chain_Noise', 'SEA_Gradual_Recurring', 'SEA_Gradual_Simple_Slow',
                    'AGRAWAL_Abrupt_Simple_Severe_Noise',
                    'HYPERPLANE_Gradual_Simple',
                    'RANDOMTREE_Gradual_Simple',
                    'RBF_Abrupt_Blip_Noise', 'RBF_Gradual_Moderate', 'RBF_Gradual_Severe',
                    'STAGGER_Abrupt_Chain_Noise', 'STAGGER_Gradual_Chain',
                    'LED_Gradual_Simple',
                    'SINE_Abrupt_Recurring_Noise'
                ]
            },
            'batch_3': {
                'base_dir': 'experiments_unified/chunk_1000/batch_3',
                'datasets': [
                    'HYPERPLANE_Gradual_Noise',
                    'RANDOMTREE_Abrupt_Recurring', 'RANDOMTREE_Gradual_Noise',
                    'RBF_Gradual_Severe_Noise',
                    'LED_Abrupt_Simple',
                    'WAVEFORM_Abrupt_Simple', 'WAVEFORM_Gradual_Simple',
                    'SINE_Abrupt_Simple', 'SINE_Gradual_Recurring',
                    'Electricity', 'Shuttle', 'CovType', 'PokerHand'
                ]
            },
            'batch_4': {
                'base_dir': 'experiments_unified/chunk_1000/batch_4',
                'datasets': [
                    'SEA_Stationary', 'AGRAWAL_Stationary', 'HYPERPLANE_Stationary',
                    'RANDOMTREE_Stationary', 'RBF_Stationary', 'STAGGER_Stationary',
                    'LED_Stationary', 'WAVEFORM_Stationary', 'SINE_Stationary',
                    'IntelLabSensors',
                    'AssetNegotiation_F2', 'AssetNegotiation_F3', 'AssetNegotiation_F4'
                ]
            }
        }
    },

    # =========================================================================
    # CHUNK 1000 - COM PENALTY (foco em interpretabilidade)
    # =========================================================================
    'exp_unified_1000_penalty': {
        'chunk_size': 1000,
        'penalty_weight': 0.1,
        'data_source': 'unified',
        'data_dir': 'chunk_1000',  # Dados sao os mesmos
        'results_dir': 'chunk_1000_penalty',  # Resultados EGIS diferentes
        'egis_model_name': 'EGIS_Penalty',
        'description': 'Unified chunks 1000 (com penalty)',
        'reuse_comparative_from': 'exp_unified_1000',  # Reutiliza modelos comparativos
        'batches': {
            'batch_1': {
                'base_dir': 'experiments_unified/chunk_1000_penalty/batch_1',
                'datasets': [
                    'SEA_Abrupt_Simple', 'SEA_Abrupt_Chain', 'SEA_Abrupt_Recurring', 'SEA_Gradual_Simple_Fast',
                    'AGRAWAL_Abrupt_Simple_Mild', 'AGRAWAL_Abrupt_Simple_Severe', 'AGRAWAL_Abrupt_Chain_Long',
                    'HYPERPLANE_Abrupt_Simple',
                    'RANDOMTREE_Abrupt_Simple',
                    'RBF_Abrupt_Blip', 'RBF_Abrupt_Severe',
                    'STAGGER_Abrupt_Chain', 'STAGGER_Abrupt_Recurring'
                ]
            },
            'batch_2': {
                'base_dir': 'experiments_unified/chunk_1000_penalty/batch_2',
                'datasets': [
                    'SEA_Abrupt_Chain_Noise', 'SEA_Gradual_Recurring', 'SEA_Gradual_Simple_Slow',
                    'AGRAWAL_Abrupt_Simple_Severe_Noise',
                    'HYPERPLANE_Gradual_Simple',
                    'RANDOMTREE_Gradual_Simple',
                    'RBF_Abrupt_Blip_Noise', 'RBF_Gradual_Moderate', 'RBF_Gradual_Severe',
                    'STAGGER_Abrupt_Chain_Noise', 'STAGGER_Gradual_Chain',
                    'LED_Gradual_Simple',
                    'SINE_Abrupt_Recurring_Noise'
                ]
            },
            'batch_3': {
                'base_dir': 'experiments_unified/chunk_1000_penalty/batch_3',
                'datasets': [
                    'HYPERPLANE_Gradual_Noise',
                    'RANDOMTREE_Abrupt_Recurring', 'RANDOMTREE_Gradual_Noise',
                    'RBF_Gradual_Severe_Noise',
                    'LED_Abrupt_Simple',
                    'WAVEFORM_Abrupt_Simple', 'WAVEFORM_Gradual_Simple',
                    'SINE_Abrupt_Simple', 'SINE_Gradual_Recurring',
                    'Electricity', 'Shuttle', 'CovType', 'PokerHand'
                ]
            },
            'batch_4': {
                'base_dir': 'experiments_unified/chunk_1000_penalty/batch_4',
                'datasets': [
                    'SEA_Stationary', 'AGRAWAL_Stationary', 'HYPERPLANE_Stationary',
                    'RANDOMTREE_Stationary', 'RBF_Stationary', 'STAGGER_Stationary',
                    'LED_Stationary', 'WAVEFORM_Stationary', 'SINE_Stationary',
                    'IntelLabSensors',
                    'AssetNegotiation_F2', 'AssetNegotiation_F3', 'AssetNegotiation_F4'
                ]
            }
        }
    }
}

# =============================================================================
# COMBINAR CONFIGS (ORIGINAIS + UNIFIED)
# =============================================================================
# Manter experimentos originais para compatibilidade
EXPERIMENT_CONFIGS = {
    'exp_a_chunk1000': {
        'chunk_size': 1000,
        'penalty_weight': 0.0,
        'data_source': 'legacy',  # Fonte antiga
        'description': 'Baseline configuration (chunk_size=1000)',
        'batches': {
            'batch_1': {
                'base_dir': 'experiments_6chunks_phase2_gbml/batch_1',
                'datasets': [
                    'SEA_Abrupt_Simple', 'SEA_Abrupt_Chain', 'SEA_Abrupt_Recurring',
                    'AGRAWAL_Abrupt_Simple_Mild', 'AGRAWAL_Abrupt_Simple_Severe', 'AGRAWAL_Abrupt_Chain_Long',
                    'RBF_Abrupt_Severe', 'RBF_Abrupt_Blip',
                    'STAGGER_Abrupt_Chain', 'STAGGER_Abrupt_Recurring',
                    'HYPERPLANE_Abrupt_Simple', 'RANDOMTREE_Abrupt_Simple'
                ]
            }
            # ... outros batches mantidos igual ao original
        }
    }
    # ... exp_b_chunk2000 e exp_c_balanced mantidos igual ao original
}

# Adicionar experimentos unified
EXPERIMENT_CONFIGS.update(UNIFIED_EXPERIMENT_CONFIGS)

# =============================================================================
# SELECIONAR EXPERIMENTOS A EXECUTAR
# =============================================================================
# Descomente os experimentos que deseja executar:

EXPERIMENTS_TO_RUN = [
    # Novos experimentos unified
    'exp_unified_500',           # Chunk 500 sem penalty
    'exp_unified_500_penalty',   # Chunk 500 com penalty
    'exp_unified_1000',          # Chunk 1000 sem penalty
    'exp_unified_1000_penalty',  # Chunk 1000 com penalty

    # Experimentos legados (descomente se necessario)
    # 'exp_a_chunk1000',
    # 'exp_b_chunk2000',
    # 'exp_c_balanced',
]

# =============================================================================
# MODELOS A EXECUTAR
# =============================================================================
# EGIS e CDCMS serao carregados de cache (ja executados)
# Demais modelos serao executados em tempo real

MODELS_TO_RUN = [
    'ROSE_Original',    # ROSE como no paper (metrica global)
    'ROSE_ChunkEval',   # ROSE com avaliacao por chunk
    'HAT',              # Hoeffding Adaptive Tree (River)
    'ARF',              # Adaptive Random Forest (River)
    'SRP',              # Streaming Random Patches (River)
    'ACDWM',            # Adaptive Chunk-based DWM (so binario!)
    # 'ERulesD2S'       # Desabilitado por ser lento
]

# Timeout por modelo (segundos)
MODEL_TIMEOUT = {
    'ROSE_Original': 600,
    'ROSE_ChunkEval': 600,
    'HAT': 300,
    'ARF': 600,
    'SRP': 600,
    'ACDWM': 600,
    'ERulesD2S': 1800
}

# Controle de cache
USE_CACHE = True  # Se True, usa resultados existentes
FORCE_RERUN = []  # Lista de modelos para forcar re-execucao

# =============================================================================
# RESUMO DA CONFIGURACAO
# =============================================================================
print("=" * 70)
print("CONFIGURACAO DOS EXPERIMENTOS")
print("=" * 70)
print(f"\nUNIFIED_CHUNKS_DIR: {UNIFIED_CHUNKS_DIR}")
print(f"EXPERIMENTS_UNIFIED_DIR: {EXPERIMENTS_UNIFIED_DIR}")
print(f"\nExperimentos a executar: {EXPERIMENTS_TO_RUN}")
print(f"Modelos a executar: {MODELS_TO_RUN}")
print(f"Usar cache: {USE_CACHE}")
print(f"\nDatasets multiclasse (CDCMS/ACDWM N/A): {len(MULTICLASS_DATASETS)}")
print("=" * 70)
