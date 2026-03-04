#!/usr/bin/env python3
"""
Executa ERulesD2S nos datasets ja processados.

Usa os mesmos datasets e configuracoes da FASE 1.
"""

import sys
import time
import yaml
from pathlib import Path
import logging
import pickle
import numpy as np

sys.path.insert(0, '.')

from arff_converter import ARFFConverter
from erulesd2s_wrapper import ERulesD2SWrapper, ERulesD2SEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def dicts_to_matrix(dict_list):
    """
    Converte lista de dicionários (formato River/incremental) para matriz numpy.

    Args:
        dict_list: Lista de dicionários, cada um com features como chaves

    Returns:
        np.ndarray: Matriz (n_samples, n_features)
    """
    if not dict_list:
        return np.array([])

    # Obter features ordenadas do primeiro dicionário
    features = sorted(dict_list[0].keys())

    # Extrair valores na mesma ordem para cada instância
    matrix = np.array([
        [d[f] for f in features]
        for d in dict_list
    ])

    return matrix


def main():
    print("="*80)
    print("EXECUTANDO ERULESD2S - FASE 2")
    print("="*80)
    print()

    # Carregar configuracao
    config_file = 'config_experiment_expanded.yaml'
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Datasets executados (9 de 11)
    datasets = [
        'RBF_Abrupt_Severe',
        'RBF_Gradual_Moderate',
        'SEA_Abrupt_Simple',
        'SEA_Gradual_Simple_Fast',
        'SEA_Abrupt_Recurring',
        'AGRAWAL_Abrupt_Simple_Severe',
        'AGRAWAL_Gradual_Chain',
        'HYPERPLANE_Abrupt_Simple',
        'STAGGER_Abrupt_Chain'
    ]

    print(f"Datasets a processar: {len(datasets)}")
    print(f"Tempo estimado: ~{len(datasets) * 16} minutos (~{len(datasets) * 16 / 60:.1f} horas)")
    print()

    # Verificar se JAR existe
    jar_path = Path('erulesd2s.jar')
    if not jar_path.exists():
        logger.error("ERulesD2S JAR nao encontrado!")
        logger.error("Execute: python setup_erulesd2s.py")
        return 1

    logger.info(f"JAR encontrado: {jar_path} ({jar_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print()

    # Criar wrapper ERulesD2S
    wrapper = ERulesD2SWrapper(
        moa_jar_path=str(jar_path),
        java_memory="4g",
        gpu_enabled=False
    )

    # Diretorio de saida
    output_base = Path('experiment_expanded_results_erulesd2s')
    output_base.mkdir(parents=True, exist_ok=True)

    # Processar cada dataset
    all_results = []

    for idx, dataset in enumerate(datasets, 1):
        print()
        print("-"*80)
        print(f"[{idx}/{len(datasets)}] Processando: {dataset}")
        print("-"*80)

        start_time = time.time()

        try:
            # Carregar chunks do cache
            chunk_size = config['data_params']['chunk_size']
            num_chunks = config['data_params']['num_chunks']
            cache_file = Path('chunks_cache') / f"{dataset}_cs{chunk_size}_nc{num_chunks + 1}_seed42.pkl"

            if not cache_file.exists():
                logger.error(f"Cache de chunks nao encontrado: {cache_file}")
                logger.error("Execute primeiro compare_gbml_vs_river.py para gerar o cache")
                continue

            logger.info(f"Carregando chunks do cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                chunks_raw = pickle.load(f)

            logger.info(f"Chunks carregados: {len(chunks_raw)}")

            # Verificar formato dos chunks e converter para numpy arrays
            chunks = []
            for i, chunk in enumerate(chunks_raw):
                if isinstance(chunk, (list, tuple)) and len(chunk) == 2:
                    X_raw = chunk[0]
                    y_raw = chunk[1]

                    # Verificar se X é lista de dicionários (formato River/incremental)
                    if isinstance(X_raw, list) and len(X_raw) > 0 and isinstance(X_raw[0], dict):
                        logger.info(f"  Chunk {i}: Convertendo lista de dicionários para matriz...")
                        X = dicts_to_matrix(X_raw)
                    elif isinstance(X_raw, np.ndarray):
                        X = X_raw
                    else:
                        # Tentar converter diretamente
                        X = np.array(X_raw)

                    # Converter y para array se necessário
                    if not isinstance(y_raw, np.ndarray):
                        y = np.array(y_raw)
                    else:
                        y = y_raw

                    chunks.append((X, y))
                    logger.info(f"  Chunk {i}: X.shape={X.shape}, y.shape={y.shape}, y.dtype={y.dtype}")
                else:
                    logger.error(f"Formato de chunk invalido: {type(chunk)}")
                    raise ValueError(f"Chunk deve ser tupla/lista (X, y), recebido: {type(chunk)}")

            logger.info(f"Chunks processados: {len(chunks)}")

            # Converter para ARFF
            logger.info("Convertendo para ARFF...")
            converter = ARFFConverter(relation_name=dataset)
            arff_dir = output_base / dataset / 'arff'
            arff_files = converter.convert_stream(
                chunks,
                output_dir=arff_dir,
                base_name=f"{dataset}_chunk"
            )
            logger.info(f"Arquivos ARFF criados: {len(arff_files)}")

            # Avaliar com ERulesD2S
            logger.info("Executando ERulesD2S...")
            evaluator = ERulesD2SEvaluator(
                wrapper=wrapper,
                output_dir=output_base / dataset,
                population_size=25,
                num_generations=50,
                rules_per_class=5
            )

            results_df = evaluator.evaluate_chunks(chunks, arff_files)

            # Adicionar metadados
            results_df['dataset'] = dataset
            results_df['drift_type'] = 'Abrupt' if 'Abrupt' in dataset else 'Gradual'
            results_df['generator'] = dataset.split('_')[0]

            # Salvar resultados do dataset
            output_file = output_base / f"{dataset}_erulesd2s_results.csv"
            results_df.to_csv(output_file, index=False)

            all_results.append(results_df)

            duration = time.time() - start_time
            logger.info(f"Dataset {dataset} concluido em {duration/60:.1f} min")

        except Exception as e:
            logger.error(f"Erro ao processar {dataset}: {e}")
            import traceback
            traceback.print_exc()

            response = input("\nContinuar com proximo dataset? (y/n): ")
            if response.lower() != 'y':
                logger.warning("Execucao interrompida pelo usuario")
                break

    # Consolidar todos os resultados
    if all_results:
        print()
        print("="*80)
        print("CONSOLIDANDO RESULTADOS")
        print("="*80)

        import pandas as pd
        df_consolidated = pd.concat(all_results, ignore_index=True)

        # Salvar consolidado
        consolidated_file = 'experiment_expanded_results_erulesd2s.csv'
        df_consolidated.to_csv(consolidated_file, index=False)

        logger.info(f"Resultados consolidados: {consolidated_file}")
        logger.info(f"Total de avaliacoes: {len(df_consolidated)}")
        logger.info(f"Datasets processados: {df_consolidated['dataset'].nunique()}")

        # Estatisticas resumidas
        print()
        print("-"*80)
        print("ESTATISTICAS ERULESD2S")
        print("-"*80)

        if 'gmean' in df_consolidated.columns:
            gmean_stats = df_consolidated.groupby('dataset')['gmean'].agg(['mean', 'std', 'count'])
            print("\nG-mean por dataset:")
            print(gmean_stats.round(4))

            print(f"\nG-mean geral: {df_consolidated['gmean'].mean():.4f} +/- {df_consolidated['gmean'].std():.4f}")

    print()
    print("="*80)
    print("ERULESD2S CONCLUIDO")
    print("="*80)
    print()
    print("Proximo passo:")
    print("  Execute novamente: python analyze_complete_results.py")
    print("  Isso ira integrar os resultados ERulesD2S com os 5 modelos Python")
    print()

    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nExecucao interrompida pelo usuario")
        sys.exit(1)
