# =============================================================================
# CELULA 3.1: Funcoes para Carregar Dados (UNIFIED + LEGACY)
# =============================================================================
# SUBSTITUA a CELULA 3.1 original por este codigo
# Adiciona funcoes para carregar dados de unified_chunks
# Adiciona funcoes para carregar resultados EGIS e CDCMS
# Mantem funcoes legadas para compatibilidade
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import json

# =============================================================================
# 1. FUNCOES PARA UNIFIED_CHUNKS (NOVAS)
# =============================================================================

def load_chunks_from_unified(data_dir, dataset_name):
    """
    Carrega chunks de dados do diretorio unified_chunks.
    Formato: unified_chunks/chunk_XXX/dataset/chunk_N.csv

    Args:
        data_dir: Nome do diretorio de dados (ex: 'chunk_500', 'chunk_1000')
        dataset_name: Nome do dataset (ex: 'SEA_Abrupt_Simple')

    Returns:
        X_chunks: Lista de arrays numpy (features)
        y_chunks: Lista de arrays numpy (labels)
    """
    data_path = UNIFIED_CHUNKS_DIR / data_dir / dataset_name

    if not data_path.exists():
        print(f"  [ERRO] Diretorio nao encontrado: {data_path}")
        return None, None

    X_chunks = []
    y_chunks = []

    # Ordenar chunks por numero
    chunk_files = sorted(
        data_path.glob("chunk_*.csv"),
        key=lambda x: int(x.stem.split('_')[1])
    )

    if not chunk_files:
        print(f"  [ERRO] Nenhum chunk encontrado em: {data_path}")
        return None, None

    for chunk_file in chunk_files:
        try:
            df = pd.read_csv(chunk_file)

            # Assumir ultima coluna como classe
            X = df.iloc[:, :-1].values.astype(float)
            y = df.iloc[:, -1].values

            # Converter classe para int
            if y.dtype == bool:
                y = y.astype(int)
            elif y.dtype == object:
                y = np.array([1 if str(v).lower() in ['true', '1', '1.0'] else 0 for v in y])
            elif 'float' in str(y.dtype):
                y = y.astype(int)

            X_chunks.append(X)
            y_chunks.append(y)

        except Exception as e:
            print(f"  [ERRO] Lendo {chunk_file.name}: {e}")
            continue

    if len(X_chunks) == 0:
        return None, None

    return X_chunks, y_chunks


def load_egis_results(results_dir, batch_name, dataset_name, egis_model_name='EGIS'):
    """
    Carrega resultados EGIS do chunk_metrics.json.

    Path: experiments_unified/{results_dir}/batch_Y/dataset/run_1/chunk_metrics.json

    Args:
        results_dir: Diretorio de resultados (ex: 'chunk_500', 'chunk_500_penalty')
        batch_name: Nome do batch (ex: 'batch_1')
        dataset_name: Nome do dataset
        egis_model_name: Nome do modelo EGIS ('EGIS' ou 'EGIS_Penalty')

    Returns:
        dict com 'gmean', 'f1', 'model_name' ou None se nao encontrado
    """
    run_dir = EXPERIMENTS_UNIFIED_DIR / results_dir / batch_name / dataset_name / "run_1"
    metrics_file = run_dir / "chunk_metrics.json"

    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file) as f:
            chunk_metrics = json.load(f)

        # Calcular media dos test_gmean
        test_gmeans = [m.get('test_gmean', 0) for m in chunk_metrics if m.get('test_gmean') is not None]
        test_f1s = [m.get('test_f1', 0) for m in chunk_metrics if m.get('test_f1') is not None]
        test_accs = [m.get('test_accuracy', 0) for m in chunk_metrics if m.get('test_accuracy') is not None]

        if test_gmeans:
            return {
                'gmean': np.mean(test_gmeans),
                'f1': np.mean(test_f1s) if test_f1s else 0.0,
                'accuracy': np.mean(test_accs) if test_accs else 0.0,
                'num_chunks': len(chunk_metrics),
                'model_name': egis_model_name,
                'chunk_results': [
                    {
                        'chunk': i + 1,
                        'test_gmean': m.get('test_gmean', 0),
                        'test_accuracy': m.get('test_accuracy', 0),
                        'test_f1': m.get('test_f1', 0)
                    }
                    for i, m in enumerate(chunk_metrics)
                ]
            }
    except Exception as e:
        print(f"  [ERRO] Lendo EGIS results: {e}")

    return None


def load_cdcms_results(results_dir, batch_name, dataset_name):
    """
    Carrega resultados CDCMS do chunk_metrics.json.

    NOTA: CDCMS so existe nas pastas SEM penalty.
    Path: experiments_unified/chunk_XXX/batch_Y/dataset/cdcms_results/chunk_metrics.json

    Args:
        results_dir: Diretorio de resultados (automaticamente remove _penalty)
        batch_name: Nome do batch
        dataset_name: Nome do dataset

    Returns:
        dict com 'gmean', 'prequential_gmean', 'holdout_gmean' ou None
    """
    # CDCMS sempre na pasta sem penalty
    base_results_dir = results_dir.replace('_penalty', '')

    cdcms_dir = EXPERIMENTS_UNIFIED_DIR / base_results_dir / batch_name / dataset_name / "cdcms_results"
    metrics_file = cdcms_dir / "chunk_metrics.json"

    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file) as f:
            chunk_metrics = json.load(f)

        # Calcular medias
        prequential_gmeans = [m.get('prequential_gmean', 0) for m in chunk_metrics if m.get('prequential_gmean') is not None]
        holdout_gmeans = [m.get('holdout_gmean') for m in chunk_metrics if m.get('holdout_gmean') is not None]

        # Preferir holdout_gmean se disponivel
        if holdout_gmeans:
            gmean = np.mean(holdout_gmeans)
        elif prequential_gmeans:
            gmean = np.mean(prequential_gmeans)
        else:
            gmean = 0.0

        return {
            'gmean': gmean,
            'prequential_gmean': np.mean(prequential_gmeans) if prequential_gmeans else 0.0,
            'holdout_gmean': np.mean(holdout_gmeans) if holdout_gmeans else None,
            'num_chunks': len(chunk_metrics),
            'chunk_results': [
                {
                    'chunk': i + 1,
                    'prequential_gmean': m.get('prequential_gmean', 0),
                    'holdout_gmean': m.get('holdout_gmean', 0)
                }
                for i, m in enumerate(chunk_metrics)
            ]
        }
    except Exception as e:
        print(f"  [ERRO] Lendo CDCMS results: {e}")

    return None


def load_existing_comparative_results(results_dir, batch_name, dataset_name, model_name):
    """
    Carrega resultados existentes de um modelo comparativo em experimentos unified.

    Os modelos comparativos sao salvos nas pastas SEM penalty.

    Args:
        results_dir: Diretorio de resultados
        batch_name: Nome do batch
        dataset_name: Nome do dataset
        model_name: Nome do modelo (HAT, ARF, SRP, ROSE_Original, etc.)

    Returns:
        dict com 'gmean', 'status' ou None se nao encontrado
    """
    # Modelos comparativos sempre na pasta sem penalty
    base_results_dir = results_dir.replace('_penalty', '')
    dataset_dir = EXPERIMENTS_UNIFIED_DIR / base_results_dir / batch_name / dataset_name
    run_dir = dataset_dir / "run_1"

    # Mapear nome do modelo para arquivo
    file_map = {
        'HAT': 'river_HAT_results.csv',
        'ARF': 'river_ARF_results.csv',
        'SRP': 'river_SRP_results.csv',
        'ACDWM': 'acdwm_results.csv',
        'ERulesD2S': 'erulesd2s_results.csv',
        'ROSE_Original': 'rose_original_results.csv',
        'ROSE_ChunkEval': 'rose_chunk_eval_results.csv'
    }

    if model_name not in file_map:
        return None

    result_file = run_dir / file_map[model_name]

    if not result_file.exists():
        return None

    try:
        df = pd.read_csv(result_file)

        # Procurar coluna de gmean
        for col in ['test_gmean', 'gmean', 'G-Mean', 'g_mean']:
            if col in df.columns:
                return {'gmean': df[col].mean(), 'status': 'CACHED'}

        # Fallback: accuracy
        if 'accuracy' in df.columns or 'test_accuracy' in df.columns:
            col = 'test_accuracy' if 'test_accuracy' in df.columns else 'accuracy'
            return {'gmean': df[col].mean(), 'status': 'CACHED_ACC'}

    except Exception as e:
        pass

    return None


# =============================================================================
# 2. FUNCOES LEGADAS (MANTIDAS PARA COMPATIBILIDADE)
# =============================================================================

def load_chunks_from_gbml(dataset_dir):
    """
    [LEGADO] Carrega chunks de dados do diretorio GBML antigo.
    Formato: chunk_data/chunk_X_test.csv (X = 1, 2, 3, ...)
    """
    dataset_dir = Path(dataset_dir)
    run_dir = dataset_dir / "run_1"
    chunk_data_dir = run_dir / "chunk_data"

    X_chunks = []
    y_chunks = []

    if not chunk_data_dir.exists():
        return None, None

    # Procurar chunks de teste
    test_files = sorted(chunk_data_dir.glob("chunk_*_test.csv"))

    for test_file in test_files:
        try:
            df = pd.read_csv(test_file)

            # Identificar coluna de classe
            if 'class' in df.columns:
                y = df['class'].values
                X = df.drop(columns=['class']).values
            elif 'label' in df.columns:
                y = df['label'].values
                X = df.drop(columns=['label']).values
            else:
                y = df.iloc[:, -1].values
                X = df.iloc[:, :-1].values

            X_chunks.append(X.astype(float))
            y_chunks.append(y.astype(int))
        except Exception as e:
            print(f"  Erro ao carregar {test_file}: {e}")
            continue

    if len(X_chunks) > 0:
        return X_chunks, y_chunks
    return None, None


def load_existing_model_results(dataset_dir, model_name):
    """
    [LEGADO] Carrega resultados existentes de um modelo comparativo.
    Retorna dict com gmean ou None.
    """
    dataset_dir = Path(dataset_dir)
    run_dir = dataset_dir / "run_1"

    # Mapear nome do modelo para arquivo
    file_map = {
        'HAT': 'river_HAT_results.csv',
        'ARF': 'river_ARF_results.csv',
        'SRP': 'river_SRP_results.csv',
        'ACDWM': 'acdwm_results.csv',
        'ERulesD2S': 'erulesd2s_results.csv',
        'ROSE_Original': 'rose_original_results.csv',
        'ROSE_ChunkEval': 'rose_chunk_eval_results.csv'
    }

    if model_name not in file_map:
        return None

    result_file = run_dir / file_map[model_name]

    if result_file.exists():
        try:
            df = pd.read_csv(result_file)

            # Procurar coluna de gmean
            for col in ['test_gmean', 'gmean', 'G-Mean', 'g_mean']:
                if col in df.columns:
                    return {'gmean': df[col].mean(), 'source': 'cached'}

            # Fallback: accuracy
            if 'accuracy' in df.columns or 'test_accuracy' in df.columns:
                col = 'test_accuracy' if 'test_accuracy' in df.columns else 'accuracy'
                return {'gmean': df[col].mean(), 'source': 'cached_accuracy'}

        except Exception as e:
            pass

    return None


# =============================================================================
# 3. FUNCOES AUXILIARES
# =============================================================================

def get_dataset_output_dir(exp_config, batch_name, dataset_name):
    """
    Retorna o diretorio onde salvar resultados de modelos comparativos.
    Para experimentos com penalty, salva na pasta sem penalty.
    """
    data_source = exp_config.get('data_source', 'legacy')

    if data_source == 'unified':
        # Modelos comparativos sempre na pasta sem penalty
        results_dir = exp_config.get('results_dir', 'chunk_500').replace('_penalty', '')
        return EXPERIMENTS_UNIFIED_DIR / results_dir / batch_name / dataset_name
    else:
        # Legacy: usar base_dir do batch
        batch_info = exp_config['batches'].get(batch_name, {})
        return Path(WORK_DIR) / batch_info.get('base_dir', '') / dataset_name


# =============================================================================
# TESTE RAPIDO DAS FUNCOES
# =============================================================================
print("=" * 70)
print("CELULA 3.1: FUNCOES DE CARREGAMENTO")
print("=" * 70)

# Testar se diretorios existem
print(f"\nVerificando diretorios:")
print(f"  UNIFIED_CHUNKS_DIR: {UNIFIED_CHUNKS_DIR.exists()}")
print(f"  EXPERIMENTS_UNIFIED_DIR: {EXPERIMENTS_UNIFIED_DIR.exists()}")

# Testar carregamento de um dataset unified
test_data_dir = 'chunk_500'
test_dataset = 'SEA_Abrupt_Simple'

if UNIFIED_CHUNKS_DIR.exists():
    X_test, y_test = load_chunks_from_unified(test_data_dir, test_dataset)
    if X_test is not None:
        print(f"\n  Teste load_chunks_from_unified:")
        print(f"    Dataset: {test_data_dir}/{test_dataset}")
        print(f"    Chunks: {len(X_test)}, Samples/chunk: {len(X_test[0])}, Features: {X_test[0].shape[1]}")
        print(f"    Classes: {np.unique(y_test[0])}")

# Testar carregamento de EGIS
if EXPERIMENTS_UNIFIED_DIR.exists():
    egis = load_egis_results('chunk_500', 'batch_1', test_dataset, 'EGIS')
    if egis:
        print(f"\n  Teste load_egis_results:")
        print(f"    Model: {egis['model_name']}")
        print(f"    G-Mean: {egis['gmean']:.4f}")
        print(f"    Chunks: {egis['num_chunks']}")

    cdcms = load_cdcms_results('chunk_500', 'batch_1', test_dataset)
    if cdcms:
        print(f"\n  Teste load_cdcms_results:")
        print(f"    G-Mean: {cdcms['gmean']:.4f}")
        print(f"    Chunks: {cdcms['num_chunks']}")

print("\n" + "=" * 70)
print("FUNCOES CARREGADAS:")
print("=" * 70)
print("  UNIFIED:")
print("    - load_chunks_from_unified(data_dir, dataset_name)")
print("    - load_egis_results(results_dir, batch_name, dataset_name)")
print("    - load_cdcms_results(results_dir, batch_name, dataset_name)")
print("    - load_existing_comparative_results(results_dir, batch_name, dataset_name, model_name)")
print("    - get_dataset_output_dir(exp_config, batch_name, dataset_name)")
print("  LEGACY:")
print("    - load_chunks_from_gbml(dataset_dir)")
print("    - load_existing_model_results(dataset_dir, model_name)")
print("=" * 70)
