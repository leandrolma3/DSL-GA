# =============================================================================
# CELULA 7.3: Executar CDCMS em Lote
# =============================================================================
# Funcoes para executar CDCMS em multiplos datasets
# =============================================================================

import subprocess
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

# Variaveis globais (definidas nas celulas anteriores)
# CDCMS_JAR, MOA_DEPS_JAR, TEST_DIR, RESULTS_DIR

def run_cdcms_on_arff(arff_path, output_path, classifier="CDCMS_CIL_GMean", timeout=600):
    """
    Executa CDCMS em um arquivo ARFF.

    Args:
        arff_path: Caminho do arquivo ARFF
        output_path: Caminho do arquivo de saida CSV
        classifier: 'CDCMS_CIL_GMean' ou 'CDCMS_CIL'
        timeout: Timeout em segundos

    Returns:
        Dict com resultados ou None se erro
    """
    # Classpath de execucao (definido nas celulas 5/6)
    classpath = f"{CDCMS_JAR}:{MOA_DEPS_JAR}"

    cmd = [
        "java", "-Xmx4g",
        "-cp", f"{classpath}:{TEST_DIR}",
        "CDCMSEvaluator",
        str(arff_path),
        str(output_path),
        classifier
    ]

    start = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        duration = time.time() - start

        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            result_df = pd.read_csv(output_path)
            final_accuracy = result_df['accuracy'].iloc[-1]
            num_instances = len(result_df)

            return {
                'success': True,
                'accuracy': final_accuracy,
                'instances': num_instances,
                'time_seconds': duration,
                'output_file': str(output_path)
            }
        else:
            return {
                'success': False,
                'error': result.stderr[:500] if result.stderr else 'Unknown error',
                'time_seconds': duration
            }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': f'Timeout after {timeout}s',
            'time_seconds': timeout
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'time_seconds': time.time() - start
        }


def run_cdcms_batch(datasets, chunks_dir, arff_dir, results_dir,
                     chunk_size='chunk_2000', classifier="CDCMS_CIL_GMean"):
    """
    Executa CDCMS em um lote de datasets.

    Args:
        datasets: Lista de nomes de datasets
        chunks_dir: Diretorio base (unified_chunks/)
        arff_dir: Diretorio para ARFFs temporarios
        results_dir: Diretorio para resultados
        chunk_size: Tamanho do chunk
        classifier: Classificador a usar

    Returns:
        DataFrame com resultados
    """
    arff_dir = Path(arff_dir)
    results_dir = Path(results_dir)
    arff_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    results = []
    total = len(datasets)

    print("="*70)
    print(f"EXECUTAR CDCMS - {classifier}")
    print(f"Datasets: {total} | Chunk size: {chunk_size}")
    print("="*70)

    for i, dataset in enumerate(datasets, 1):
        print(f"\n[{i}/{total}] {dataset}")

        # Converter para ARFF
        arff_path = load_dataset_as_arff(dataset, chunks_dir, arff_dir, chunk_size)

        if arff_path is None:
            results.append({
                'dataset': dataset,
                'classifier': classifier,
                'chunk_size': chunk_size,
                'success': False,
                'error': 'Failed to create ARFF'
            })
            continue

        # Executar CDCMS
        output_path = results_dir / f"{dataset}_{classifier}.csv"
        result = run_cdcms_on_arff(arff_path, output_path, classifier)

        result['dataset'] = dataset
        result['classifier'] = classifier
        result['chunk_size'] = chunk_size

        if result['success']:
            print(f"  [OK] Accuracy: {result['accuracy']:.4f} | Tempo: {result['time_seconds']:.1f}s")
        else:
            print(f"  [ERRO] {result.get('error', 'Unknown')[:50]}")

        results.append(result)

    # Criar DataFrame com resultados
    results_df = pd.DataFrame(results)

    # Salvar resumo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = results_dir / f"cdcms_batch_summary_{timestamp}.csv"
    results_df.to_csv(summary_file, index=False)

    # Mostrar resumo
    print("\n" + "="*70)
    print("RESUMO")
    print("="*70)

    successful = results_df[results_df['success'] == True]
    failed = results_df[results_df['success'] == False]

    print(f"Sucesso: {len(successful)}/{total}")
    print(f"Falhas: {len(failed)}/{total}")

    if len(successful) > 0:
        print(f"\nAccuracy media: {successful['accuracy'].mean():.4f}")
        print(f"Tempo medio: {successful['time_seconds'].mean():.1f}s")

    print(f"\nResultados salvos em: {summary_file}")

    return results_df


def run_all_datasets(chunks_dir, arff_dir, results_dir, chunk_size='chunk_2000',
                      classifier="CDCMS_CIL_GMean", exclude_patterns=None):
    """
    Executa CDCMS em TODOS os datasets disponiveis.

    Args:
        chunks_dir: Diretorio unified_chunks/
        arff_dir: Diretorio para ARFFs
        results_dir: Diretorio para resultados
        chunk_size: Tamanho do chunk
        classifier: Classificador
        exclude_patterns: Lista de padroes para excluir (ex: ['_backup', 'IntelLab'])

    Returns:
        DataFrame com resultados
    """
    chunk_path = Path(chunks_dir) / chunk_size

    if not chunk_path.exists():
        print(f"[ERRO] Diretorio nao encontrado: {chunk_path}")
        return None

    # Listar datasets
    datasets = sorted([d.name for d in chunk_path.iterdir()
                       if d.is_dir() and not d.name.startswith('.')])

    # Aplicar filtros de exclusao
    if exclude_patterns:
        for pattern in exclude_patterns:
            datasets = [d for d in datasets if pattern not in d]

    print(f"Encontrados {len(datasets)} datasets em {chunk_size}")

    return run_cdcms_batch(datasets, chunks_dir, arff_dir, results_dir,
                           chunk_size, classifier)


# Exemplo de uso
print("[OK] Funcoes de execucao em lote carregadas:")
print("  - run_cdcms_on_arff(arff_path, output_path, classifier)")
print("  - run_cdcms_batch(datasets, chunks_dir, arff_dir, results_dir)")
print("  - run_all_datasets(chunks_dir, arff_dir, results_dir)")
print()
print("Exemplo de uso:")
print("  # Executar em datasets especificos")
print("  datasets = ['SINE_Abrupt_Simple', 'SEA_Gradual_Simple_Slow']")
print("  results = run_cdcms_batch(datasets, UNIFIED_CHUNKS_DIR, TEST_DIR, RESULTS_DIR)")
print()
print("  # Executar em TODOS os datasets")
print("  results = run_all_datasets(UNIFIED_CHUNKS_DIR, TEST_DIR, RESULTS_DIR)")
