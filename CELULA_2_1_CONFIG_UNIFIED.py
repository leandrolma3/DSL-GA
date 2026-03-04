# =============================================================================
# CELULA 2.1: Configuracoes para Unified Chunks (COM PENALTY)
# =============================================================================
# SUBSTITUA a CELULA 2.1 original por este codigo
# Suporta chunk_500, chunk_1000, chunk_2000 COM e SEM penalidade
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CELULA 2.1: CONFIGURACOES UNIFIED CHUNKS (COM PENALTY)")
print("="*70)

# =============================================================================
# 1. Paths Base
# =============================================================================
# AJUSTE ESTES PATHS CONFORME SEU AMBIENTE NO COLAB
DRIVE_BASE = "/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid"
WORK_DIR = "/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid"

# Paths para dados e resultados
UNIFIED_CHUNKS_DIR = Path(WORK_DIR) / "unified_chunks"
EXPERIMENTS_DIR = Path(WORK_DIR) / "experiments_unified"

print(f"WORK_DIR: {WORK_DIR}")
print(f"UNIFIED_CHUNKS_DIR: {UNIFIED_CHUNKS_DIR}")
print(f"EXPERIMENTS_DIR: {EXPERIMENTS_DIR}")

# =============================================================================
# 2. Configuracoes de Chunk Sizes e Penalty
# =============================================================================
# IMPORTANTE: Os dados brutos estao em unified_chunks (sem distincao de penalty)
# Os resultados EGIS estao separados: chunk_XXX (sem) e chunk_XXX_penalty (com)

CHUNK_CONFIGS = {
    # Sem penalidade (foco em performance)
    'chunk_500': {
        'size': 500,
        'num_chunks': 24,
        'penalty': False,
        'data_dir': 'chunk_500',           # unified_chunks/chunk_500
        'results_dir': 'chunk_500',        # experiments_unified/chunk_500
        'egis_model_name': 'GBML',
    },
    'chunk_1000': {
        'size': 1000,
        'num_chunks': 12,
        'penalty': False,
        'data_dir': 'chunk_1000',
        'results_dir': 'chunk_1000',
        'egis_model_name': 'GBML',
    },

    # Com penalidade (foco em interpretabilidade)
    'chunk_500_penalty': {
        'size': 500,
        'num_chunks': 24,
        'penalty': True,
        'data_dir': 'chunk_500',           # Mesmos dados brutos!
        'results_dir': 'chunk_500_penalty',# Resultados EGIS com penalty
        'egis_model_name': 'GBML_Penalty',
    },
    'chunk_1000_penalty': {
        'size': 1000,
        'num_chunks': 12,
        'penalty': True,
        'data_dir': 'chunk_1000',
        'results_dir': 'chunk_1000_penalty',
        'egis_model_name': 'GBML_Penalty',
    },

    # Futuro
    # 'chunk_2000': {'size': 2000, 'num_chunks': 6, 'penalty': False, ...},
    # 'chunk_2000_penalty': {'size': 2000, 'num_chunks': 6, 'penalty': True, ...},
}

# Batches disponiveis por chunk_size base
BATCHES = {
    'chunk_500': ['batch_1', 'batch_2', 'batch_3'],
    'chunk_500_penalty': ['batch_1', 'batch_2', 'batch_3'],
    'chunk_1000': ['batch_1', 'batch_2', 'batch_3', 'batch_4'],
    'chunk_1000_penalty': ['batch_1', 'batch_2', 'batch_3', 'batch_4'],
    'chunk_2000': ['batch_1', 'batch_2', 'batch_3'],
    'chunk_2000_penalty': ['batch_1', 'batch_2', 'batch_3'],
}

# =============================================================================
# 3. Selecionar o que executar
# =============================================================================
# AJUSTE ESTAS LISTAS CONFORME NECESSARIO

# Opcao 1: Executar tudo (com e sem penalidade)
CHUNK_SIZES_TO_RUN = ['chunk_500', 'chunk_500_penalty', 'chunk_1000', 'chunk_1000_penalty']

# Opcao 2: Apenas sem penalidade
# CHUNK_SIZES_TO_RUN = ['chunk_500', 'chunk_1000']

# Opcao 3: Apenas chunk_500 (para teste)
# CHUNK_SIZES_TO_RUN = ['chunk_500', 'chunk_500_penalty']

# Opcao 4: Teste minimo
# CHUNK_SIZES_TO_RUN = ['chunk_500']

# =============================================================================
# 4. Datasets Multiclasse (nao suportados pelo CDCMS, mas OK para outros)
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

# Datasets a excluir (problematicos - NaN ou muito lentos)
EXCLUDE_DATASETS = ['IntelLabSensors', 'PokerHand']

# =============================================================================
# 5. Modelos a Executar
# =============================================================================
# NOTA: GBML e GBML_Penalty sao carregados automaticamente baseado no chunk_config

MODELS_TO_RUN = [
    'GBML',           # EGIS sem penalidade (automatico)
    'GBML_Penalty',   # EGIS com penalidade (automatico)
    'CDCMS',          # CDCMS (binario apenas)
    'ROSE_Original',  # ROSE com avaliacao completa
    'ROSE_ChunkEval', # ROSE com avaliacao por chunk
    'HAT',            # Hoeffding Adaptive Tree
    'ARF',            # Adaptive Random Forest
    'SRP',            # Streaming Random Patches
    'ACDWM',          # Adaptive Chunk-based DWM
    'ERulesD2S',      # Evolutionary Rules (apenas cache)
]

# Modelos que precisam ser executados (nao sao apenas leitura de cache)
MODELS_TO_EXECUTE = ['ROSE_Original', 'ROSE_ChunkEval', 'HAT', 'ARF', 'SRP', 'ACDWM']

# Modelos que sao apenas leitura de resultados existentes
MODELS_CACHED_ONLY = ['GBML', 'GBML_Penalty', 'CDCMS', 'ERulesD2S']

# Timeouts por modelo (segundos)
MODEL_TIMEOUT = {
    'ROSE_Original': 600,
    'ROSE_ChunkEval': 600,
    'HAT': 300,
    'ARF': 300,
    'SRP': 300,
    'ACDWM': 600,
    'ERulesD2S': 1800,
}

# =============================================================================
# 6. Funcoes auxiliares
# =============================================================================
def get_datasets_for_batch(chunk_size_name: str, batch_name: str) -> list:
    """
    Retorna lista de datasets disponiveis para um batch.
    Baseado nos experimentos EGIS existentes em experiments_unified.
    """
    config = CHUNK_CONFIGS.get(chunk_size_name, {})
    results_dir = config.get('results_dir', chunk_size_name)

    batch_path = EXPERIMENTS_DIR / results_dir / batch_name

    if not batch_path.exists():
        print(f"[AVISO] Batch nao encontrado: {batch_path}")
        return []

    datasets = []
    for d in batch_path.iterdir():
        if d.is_dir() and not d.name.startswith('.') and d.name not in ['desktop.ini']:
            # Verificar se tem resultados EGIS (run_1)
            run_dir = d / 'run_1'
            if run_dir.exists():
                datasets.append(d.name)

    # Filtrar datasets problematicos
    datasets = [d for d in datasets if d not in EXCLUDE_DATASETS]

    return sorted(datasets)


def is_multiclass_dataset(dataset_name: str) -> bool:
    """Verifica se dataset e multiclasse."""
    return dataset_name in MULTICLASS_DATASETS


def get_base_chunk_size(chunk_size_name: str) -> str:
    """Retorna o chunk_size base (sem _penalty)."""
    return chunk_size_name.replace('_penalty', '')


def is_penalty_config(chunk_size_name: str) -> bool:
    """Verifica se e configuracao com penalidade."""
    config = CHUNK_CONFIGS.get(chunk_size_name, {})
    return config.get('penalty', False)

# =============================================================================
# 7. Verificar estrutura
# =============================================================================
print("\n" + "-"*50)
print("VERIFICANDO ESTRUTURA:")
print("-"*50)

for chunk_size_name in CHUNK_SIZES_TO_RUN:
    config = CHUNK_CONFIGS.get(chunk_size_name, {})
    penalty_str = "(COM penalty)" if config.get('penalty', False) else "(SEM penalty)"

    print(f"\n{chunk_size_name} {penalty_str}:")

    # Verificar dados
    data_dir = config.get('data_dir', chunk_size_name)
    data_path = UNIFIED_CHUNKS_DIR / data_dir
    if data_path.exists():
        n_datasets_data = len([d for d in data_path.iterdir() if d.is_dir()])
        print(f"  Dados (unified_chunks/{data_dir}): {n_datasets_data} datasets")
    else:
        print(f"  [ERRO] Dados nao encontrados: {data_path}")

    # Verificar resultados EGIS
    results_dir = config.get('results_dir', chunk_size_name)
    results_path = EXPERIMENTS_DIR / results_dir
    if results_path.exists():
        batches = BATCHES.get(chunk_size_name, [])
        total_datasets = 0

        for batch in batches:
            datasets = get_datasets_for_batch(chunk_size_name, batch)
            total_datasets += len(datasets)
            print(f"  {batch}: {len(datasets)} datasets")

        print(f"  Total: {total_datasets} datasets")
    else:
        print(f"  [ERRO] Resultados EGIS nao encontrados: {results_path}")

# =============================================================================
# 8. Resumo
# =============================================================================
print("\n" + "="*70)
print("CONFIGURACAO CARREGADA:")
print("="*70)
print(f"Chunk sizes a executar: {CHUNK_SIZES_TO_RUN}")

penalty_configs = [c for c in CHUNK_SIZES_TO_RUN if is_penalty_config(c)]
no_penalty_configs = [c for c in CHUNK_SIZES_TO_RUN if not is_penalty_config(c)]
print(f"  - Sem penalidade: {no_penalty_configs}")
print(f"  - Com penalidade: {penalty_configs}")

print(f"\nModelos: {len(MODELS_TO_RUN)}")
print(f"  - EGIS: GBML (sem penalty), GBML_Penalty (com penalty)")
print(f"  - Comparativos: {[m for m in MODELS_TO_RUN if m not in ['GBML', 'GBML_Penalty']]}")
print(f"\nDatasets multiclasse: {len(MULTICLASS_DATASETS)}")
print(f"Datasets excluidos: {EXCLUDE_DATASETS}")
print("="*70)
