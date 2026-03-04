# =============================================================================
# CELULA 3.4: Funcoes para Carregar Chunks (Unified COM PENALTY)
# =============================================================================
# SUBSTITUA a CELULA 3.4 original por este codigo
# Suporta carregamento de dados e resultados com/sem penalidade
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import json

print("="*70)
print("CELULA 3.4: FUNCOES DE CARREGAMENTO (UNIFIED COM PENALTY)")
print("="*70)

# =============================================================================
# 1. Carregar dados brutos de unified_chunks
# =============================================================================

def load_chunks_from_unified(chunk_size_name: str, dataset_name: str):
    """
    Carrega chunks de dados do diretorio unified_chunks.

    IMPORTANTE: Os dados brutos sao os mesmos para configs com e sem penalty.
    Usa CHUNK_CONFIGS[chunk_size_name]['data_dir'] para determinar o path.

    Args:
        chunk_size_name: 'chunk_500', 'chunk_500_penalty', 'chunk_1000', etc.
        dataset_name: Nome do dataset (ex: 'SEA_Abrupt_Simple')

    Returns:
        X_chunks: Lista de arrays numpy (features)
        y_chunks: Lista de arrays numpy (classes)
    """
    # Obter diretorio de dados (mesmo para versoes com/sem penalty)
    config = CHUNK_CONFIGS.get(chunk_size_name, {})
    data_dir = config.get('data_dir', chunk_size_name.replace('_penalty', ''))

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
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            # Converter classe para int se necessario
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


# =============================================================================
# 2. Carregar resultados EGIS (GBML / GBML_Penalty)
# =============================================================================

def load_egis_results(chunk_size_name: str, batch_name: str, dataset_name: str):
    """
    Carrega resultados EGIS do chunk_metrics.json.

    Path: experiments_unified/{results_dir}/batch_Y/dataset/run_1/chunk_metrics.json

    O results_dir e determinado pelo CHUNK_CONFIGS (chunk_500 ou chunk_500_penalty).
    """
    config = CHUNK_CONFIGS.get(chunk_size_name, {})
    results_dir = config.get('results_dir', chunk_size_name)
    egis_model_name = config.get('egis_model_name', 'GBML')

    run_dir = EXPERIMENTS_DIR / results_dir / batch_name / dataset_name / "run_1"
    metrics_file = run_dir / "chunk_metrics.json"

    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file) as f:
            chunk_metrics = json.load(f)

        # Calcular media dos test_gmean
        test_gmeans = [m.get('test_gmean', 0) for m in chunk_metrics if m.get('test_gmean') is not None]
        test_f1s = [m.get('test_f1', 0) for m in chunk_metrics if m.get('test_f1') is not None]

        if test_gmeans:
            return {
                'gmean': np.mean(test_gmeans),
                'f1': np.mean(test_f1s) if test_f1s else 0.0,
                'num_chunks': len(chunk_metrics),
                'model_name': egis_model_name,
            }
    except Exception as e:
        print(f"  [ERRO] Lendo EGIS results: {e}")

    return None


# =============================================================================
# 3. Carregar resultados CDCMS
# =============================================================================

def load_cdcms_results(chunk_size_name: str, batch_name: str, dataset_name: str):
    """
    Carrega resultados CDCMS do chunk_metrics.json.

    NOTA: CDCMS so existe nas pastas SEM penalty (chunk_500, chunk_1000).
    Para configs com penalty, busca na pasta correspondente sem penalty.

    Path: experiments_unified/chunk_XXX/batch_Y/dataset/cdcms_results/chunk_metrics.json
    """
    # CDCMS sempre na pasta sem penalty
    base_chunk_size = chunk_size_name.replace('_penalty', '')
    config = CHUNK_CONFIGS.get(base_chunk_size, CHUNK_CONFIGS.get(chunk_size_name, {}))
    results_dir = config.get('results_dir', base_chunk_size)

    cdcms_dir = EXPERIMENTS_DIR / results_dir / batch_name / dataset_name / "cdcms_results"
    metrics_file = cdcms_dir / "chunk_metrics.json"

    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file) as f:
            chunk_metrics = json.load(f)

        # Calcular medias
        prequential_gmeans = [m.get('prequential_gmean', 0) for m in chunk_metrics]
        holdout_gmeans = [m.get('holdout_gmean') for m in chunk_metrics if m.get('holdout_gmean') is not None]

        return {
            'gmean': np.mean(holdout_gmeans) if holdout_gmeans else np.mean(prequential_gmeans),
            'prequential_gmean': np.mean(prequential_gmeans),
            'holdout_gmean': np.mean(holdout_gmeans) if holdout_gmeans else None,
            'num_chunks': len(chunk_metrics),
        }
    except Exception as e:
        print(f"  [ERRO] Lendo CDCMS results: {e}")

    return None


# =============================================================================
# 4. Carregar resultados de modelos comparativos
# =============================================================================

def load_existing_model_results(chunk_size_name: str, batch_name: str, dataset_name: str, model_name: str):
    """
    Carrega resultados existentes de um modelo comparativo.

    NOTA: Modelos comparativos (ROSE, HAT, etc.) sao salvos nas pastas SEM penalty.
    """
    # Modelos comparativos sempre na pasta sem penalty
    base_chunk_size = chunk_size_name.replace('_penalty', '')
    config = CHUNK_CONFIGS.get(base_chunk_size, CHUNK_CONFIGS.get(chunk_size_name, {}))
    results_dir = config.get('results_dir', base_chunk_size)

    dataset_dir = EXPERIMENTS_DIR / results_dir / batch_name / dataset_name

    # Mapear nome do modelo para arquivo/pasta
    file_map = {
        'HAT': ('river_results', 'HAT_chunk_metrics.json'),
        'ARF': ('river_results', 'ARF_chunk_metrics.json'),
        'SRP': ('river_results', 'SRP_chunk_metrics.json'),
        'ROSE_Original': ('rose_results', 'rose_original_metrics.json'),
        'ROSE_ChunkEval': ('rose_results', 'rose_chunk_eval_metrics.json'),
        'ACDWM': ('acdwm_results', 'chunk_metrics.json'),
        'ERulesD2S': ('erulesd2s_results', 'chunk_metrics.json'),
    }

    if model_name not in file_map:
        return None

    folder, filename = file_map[model_name]
    results_file = dataset_dir / folder / filename

    if not results_file.exists():
        # Tentar path alternativo (dentro de run_1)
        results_file_alt = dataset_dir / "run_1" / folder / filename
        if results_file_alt.exists():
            results_file = results_file_alt
        else:
            # Tentar nome alternativo para ROSE
            if model_name == 'ROSE_Original':
                results_file = dataset_dir / "rose_original_output" / "metrics.json"
            elif model_name == 'ROSE_ChunkEval':
                results_file = dataset_dir / "rose_chunk_eval_output" / "metrics.json"

            if not results_file.exists():
                return None

    try:
        with open(results_file) as f:
            data = json.load(f)

        # Se for lista de chunks, calcular media
        if isinstance(data, list):
            gmeans = [m.get('gmean', m.get('test_gmean', 0)) for m in data
                     if m.get('gmean') is not None or m.get('test_gmean') is not None]
            if gmeans:
                return {'gmean': np.mean(gmeans), 'status': 'CACHED'}

        # Se for dict direto
        elif isinstance(data, dict):
            if 'gmean' in data:
                return {'gmean': data['gmean'], 'status': 'CACHED'}
            elif 'avg_gmean' in data:
                return {'gmean': data['avg_gmean'], 'status': 'CACHED'}

    except Exception as e:
        print(f"  [ERRO] Lendo {model_name} results: {e}")

    return None


# =============================================================================
# 5. Funcao para obter path de resultados para salvar
# =============================================================================

def get_results_path(chunk_size_name: str, batch_name: str, dataset_name: str, model_name: str):
    """
    Retorna o path onde salvar resultados de um modelo.

    NOTA: Modelos comparativos (ROSE, HAT, CDCMS, etc.) sao salvos nas pastas SEM penalty.
    EGIS e salvo na pasta correspondente (com ou sem penalty).
    """
    if model_name in ['GBML', 'GBML_Penalty']:
        # EGIS vai na pasta correspondente ao chunk_size_name
        config = CHUNK_CONFIGS.get(chunk_size_name, {})
        results_dir = config.get('results_dir', chunk_size_name)
    else:
        # Outros modelos vao na pasta sem penalty
        base_chunk_size = chunk_size_name.replace('_penalty', '')
        config = CHUNK_CONFIGS.get(base_chunk_size, {})
        results_dir = config.get('results_dir', base_chunk_size)

    return EXPERIMENTS_DIR / results_dir / batch_name / dataset_name


# =============================================================================
# 6. Funcao LEGADA (compatibilidade com paths antigos)
# =============================================================================

def load_chunks_from_gbml(dataset_dir):
    """
    [LEGADO] Carrega chunks de dados do diretorio GBML antigo.
    Mantida para compatibilidade.
    """
    dataset_dir = Path(dataset_dir)
    run_dir = dataset_dir / "run_1"

    X_chunks = []
    y_chunks = []

    chunk_data_dir = run_dir / "chunk_data"

    if chunk_data_dir.exists():
        test_files = sorted(chunk_data_dir.glob("chunk_*_test.csv"))

        if test_files:
            for test_file in test_files:
                try:
                    df = pd.read_csv(test_file)

                    if 'class' in df.columns:
                        y = df['class'].values
                        X = df.drop('class', axis=1).values
                    else:
                        y = df.iloc[:, -1].values
                        X = df.iloc[:, :-1].values

                    X_chunks.append(X)
                    y_chunks.append(y)

                except Exception as e:
                    print(f"  Erro lendo {test_file.name}: {e}")

            if len(X_chunks) > 0:
                return X_chunks, y_chunks

    return None, None


# =============================================================================
# 7. Funcao para criar ARFF (para ROSE)
# =============================================================================

def create_arff_file(X, y, output_path, relation_name="dataset"):
    """
    Cria arquivo ARFF a partir de arrays numpy.
    Necessario para executar ROSE (MOA Java).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_features = X.shape[1]
    unique_classes = sorted(np.unique(y))

    with open(output_path, 'w') as f:
        f.write(f"@relation {relation_name}\n\n")

        # Atributos
        for i in range(n_features):
            f.write(f"@attribute attr{i} numeric\n")

        # Classe
        class_str = ",".join(str(int(c)) for c in unique_classes)
        f.write(f"@attribute class {{{class_str}}}\n\n")

        # Dados
        f.write("@data\n")
        for i in range(len(X)):
            values = [str(v) for v in X[i]]
            values.append(str(int(y[i])))
            f.write(",".join(values) + "\n")

    return output_path


# =============================================================================
# 8. Teste rapido
# =============================================================================

print("\n" + "-"*50)
print("TESTE RAPIDO:")
print("-"*50)

# Testar carregamento de dados
test_configs = ['chunk_500', 'chunk_500_penalty']
test_dataset = 'SEA_Abrupt_Simple'

for test_config in test_configs:
    if test_config in CHUNK_SIZES_TO_RUN or test_config.replace('_penalty', '') in CHUNK_SIZES_TO_RUN:
        X_test, y_test = load_chunks_from_unified(test_config, test_dataset)

        if X_test is not None:
            print(f"\n[OK] {test_config} - {test_dataset}:")
            print(f"     Chunks: {len(X_test)}, Samples/chunk: {len(X_test[0])}, Features: {X_test[0].shape[1]}")

            # Testar carregamento de resultados EGIS
            egis = load_egis_results(test_config, 'batch_1', test_dataset)
            if egis:
                print(f"     EGIS ({egis['model_name']}): G-Mean = {egis['gmean']:.4f}")
            else:
                print(f"     EGIS: Nao encontrado")

print("\n" + "="*70)
print("FUNCOES CARREGADAS:")
print("="*70)
print("  - load_chunks_from_unified(chunk_size_name, dataset_name)")
print("  - load_egis_results(chunk_size_name, batch_name, dataset_name)")
print("  - load_cdcms_results(chunk_size_name, batch_name, dataset_name)")
print("  - load_existing_model_results(chunk_size_name, batch_name, dataset_name, model_name)")
print("  - get_results_path(chunk_size_name, batch_name, dataset_name, model_name)")
print("  - create_arff_file(X, y, output_path, relation_name)")
print("="*70)
