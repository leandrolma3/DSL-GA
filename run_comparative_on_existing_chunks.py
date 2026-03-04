#!/usr/bin/env python3
"""
Script para executar modelos comparativos em chunks JA GERADOS pelo GBML.

Carrega chunks salvos em CSV e executa:
- River models (HAT, ARF, SRP)
- ACDWM (opcional)
- ERulesD2S (opcional)

Os resultados sao salvos na MESMA estrutura de diretorios do GBML.

Uso:
    python run_comparative_on_existing_chunks.py \
        --dataset SEA_Abrupt_Simple \
        --base-dir experiments_6chunks_phase1_gbml/batch_1 \
        --models HAT ARF SRP \
        --acdwm \
        --acdwm-path /content/ACDWM \
        --erulesd2s
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime

# Imports dos modulos do projeto
from baseline_river import create_river_model
from baseline_acdwm import ACDWMEvaluator
from shared_evaluation import calculate_shared_metrics
from arff_converter import ARFFConverter
from erulesd2s_wrapper import ERulesD2SWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FUNCOES PARA CARREGAR CHUNKS EXISTENTES
# ============================================================================

def load_chunks_from_csv(chunks_dir: Path) -> List[Tuple[List[Dict], List]]:
    """
    Carrega chunks de arquivos CSV salvos pelo GBML.

    Estrutura esperada:
        chunk_0_train.csv
        chunk_1_test.csv
        chunk_2_test.csv
        ...

    Args:
        chunks_dir: Diretorio contendo os CSVs dos chunks

    Returns:
        Lista de chunks no formato River: [(X, y), (X, y), ...]
        Onde X = List[Dict] e y = List
    """
    logger.info(f"Carregando chunks de: {chunks_dir}")

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Diretorio de chunks nao encontrado: {chunks_dir}")

    chunks = []

    # Carregar chunk de treino inicial
    train_file = chunks_dir / "chunk_0_train.csv"
    if train_file.exists():
        df_train = pd.read_csv(train_file)
        chunk_train = dataframe_to_river_format(df_train)
        chunks.append(chunk_train)
        logger.info(f"Chunk 0 (train): {len(chunk_train[0])} instancias")
    else:
        logger.warning(f"Chunk de treino inicial nao encontrado: {train_file}")

    # Carregar chunks de teste
    test_files = sorted(chunks_dir.glob("chunk_*_test.csv"))

    for test_file in test_files:
        chunk_idx = int(test_file.stem.split('_')[1])
        df_test = pd.read_csv(test_file)
        chunk_test = dataframe_to_river_format(df_test)
        chunks.append(chunk_test)
        logger.info(f"Chunk {chunk_idx} (test): {len(chunk_test[0])} instancias")

    logger.info(f"Total de chunks carregados: {len(chunks)}")

    return chunks


def dataframe_to_river_format(df: pd.DataFrame) -> Tuple[List[Dict], List]:
    """
    Converte DataFrame para formato River (lista de dicts).

    Args:
        df: DataFrame com features + classe na ultima coluna

    Returns:
        (X, y) onde X = List[Dict] e y = List
    """
    # Separar features e labels
    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]

    # Converter para lista de dicts (formato River)
    X = df[feature_cols].to_dict('records')
    y = df[label_col].tolist()

    return X, y


# ============================================================================
# FUNCOES PARA EXECUTAR MODELOS
# ============================================================================

def run_river_model_on_chunks(
    model_name: str,
    chunks: List[Tuple[List[Dict], List]],
    classes: List
) -> pd.DataFrame:
    """
    Executa um modelo River em chunks carregados usando metodologia train-then-test.

    IMPORTANTE:
    1. CRIA NOVO MODELO a cada chunk (nao incremental!) para match com GBML
    2. Avalia modelo TANTO no chunk de treino quanto no chunk de teste
    3. Garante comparacao justa com GBML (mesmo modelo treinado apenas no chunk i)

    Args:
        model_name: Nome do modelo (HAT, ARF, SRP)
        chunks: Lista de chunks no formato River
        classes: Lista de classes

    Returns:
        DataFrame com resultados por chunk (train E test metrics)
    """
    logger.info(f"Executando modelo River: {model_name}")

    results = []

    # Avaliar no modo train-then-test
    for i in range(len(chunks) - 1):
        X_train, y_train = chunks[i]
        X_test, y_test = chunks[i + 1]

        logger.info(f"Chunk {i+1}/{len(chunks)-1}: treino={len(X_train)}, teste={len(X_test)}")

        # CRIAR NOVO MODELO para este chunk (NAO incremental!)
        # Garante que modelo ve APENAS chunk i (como GBML)
        evaluator = create_river_model(model_name, classes)

        # Treinar no chunk atual usando RiverEvaluator
        evaluator.train_on_chunk(X_train, y_train)

        # AVALIAR NO CHUNK DE TREINO (train metrics)
        train_metrics = evaluator.test_on_chunk(X_train, y_train)

        # AVALIAR NO CHUNK DE TESTE (test metrics)
        test_metrics = evaluator.test_on_chunk(X_test, y_test)

        result = {
            'chunk': i,
            'train_chunk': i,
            'test_chunk': i + 1,
            'model': model_name,
            'train_size': len(X_train),
            'test_size': len(X_test),
            # Metricas de TREINO (prefixo train_)
            'train_gmean': train_metrics['gmean'],
            'train_accuracy': train_metrics['accuracy'],
            'train_f1': train_metrics['f1_weighted'],
            # Metricas de TESTE (prefixo test_)
            'test_gmean': test_metrics['gmean'],
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1_weighted'],
            'test_f1_macro': test_metrics['f1_macro']
        }

        results.append(result)
        logger.info(f"  Train G-mean: {train_metrics['gmean']:.4f}, Test G-mean: {test_metrics['gmean']:.4f}")

    return pd.DataFrame(results)


def run_acdwm_on_chunks(
    chunks: List[Tuple[List[Dict], List]],
    classes: List,
    acdwm_path: str
) -> pd.DataFrame:
    """
    Executa ACDWM em chunks carregados usando metodologia train-then-test.

    IMPORTANTE:
    1. CRIA NOVO MODELO a cada chunk (nao incremental!) para match com GBML
    2. Avalia modelo TANTO no chunk de treino quanto no chunk de teste
    3. Garante comparacao justa com GBML (mesmo modelo treinado apenas no chunk i)

    Args:
        chunks: Lista de chunks no formato River
        classes: Lista de classes
        acdwm_path: Caminho para o repositorio ACDWM

    Returns:
        DataFrame com resultados por chunk (train E test metrics)
    """
    logger.info("Executando ACDWM...")

    results = []

    # Avaliar no modo train-then-test
    for i in range(len(chunks) - 1):
        X_train, y_train = chunks[i]
        X_test, y_test = chunks[i + 1]

        logger.info(f"Chunk {i+1}/{len(chunks)-1}: treino={len(X_train)}, teste={len(X_test)}")

        # CRIAR NOVO MODELO ACDWM para este chunk (NAO incremental!)
        # Garante que modelo ve APENAS chunk i (como GBML)
        acdwm_evaluator = ACDWMEvaluator(
            acdwm_path=acdwm_path,
            classes=classes,
            evaluation_mode='train-then-test',
            theta=0.001
        )

        # Treinar no chunk atual
        train_info = acdwm_evaluator.train_on_chunk(X_train, y_train)

        # AVALIAR NO CHUNK DE TREINO (train metrics)
        train_metrics = acdwm_evaluator.test_on_chunk(X_train, y_train)

        # AVALIAR NO CHUNK DE TESTE (test metrics)
        test_metrics = acdwm_evaluator.test_on_chunk(X_test, y_test)

        row = {
            'chunk': i,
            'train_chunk': i,
            'test_chunk': i + 1,
            'model': 'ACDWM',
            'train_size': len(X_train),
            'test_size': len(X_test),
            # Metricas de TREINO (prefixo train_)
            'train_gmean': train_metrics['gmean'],
            'train_accuracy': train_metrics['accuracy'],
            'train_f1': train_metrics['f1_weighted'],
            # Metricas de TESTE (prefixo test_)
            'test_gmean': test_metrics['gmean'],
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1_weighted'],
            'test_f1_macro': test_metrics.get('f1_macro', test_metrics['f1_weighted']),
            # Informacoes do ensemble
            'ensemble_size': train_info.get('ensemble_size', None)
        }

        results.append(row)
        logger.info(f"  Train G-mean: {train_metrics['gmean']:.4f}, Test G-mean: {test_metrics['gmean']:.4f}, ensemble_size={row['ensemble_size']}")

    return pd.DataFrame(results)


def run_erulesd2s_on_chunks(
    chunks: List[Tuple[List[Dict], List]],
    dataset_name: str,
    output_dir: Path,
    population_size: int = 25,
    num_generations: int = 50,
    rules_per_class: int = 5
) -> pd.DataFrame:
    """
    Executa ERulesD2S em chunks carregados.

    Args:
        chunks: Lista de chunks no formato River
        dataset_name: Nome do dataset
        output_dir: Diretorio de saida
        population_size: Tamanho da populacao GP
        num_generations: Numero de geracoes
        rules_per_class: Regras por classe

    Returns:
        DataFrame com resultados por chunk
    """
    logger.info("Executando ERulesD2S...")

    # Verificar se JAR existe
    jar_path = Path("erulesd2s.jar")
    if not jar_path.exists():
        logger.error("erulesd2s.jar nao encontrado!")
        logger.error("Execute setup_erulesd2s.py primeiro")
        return pd.DataFrame()

    # Criar diretorio para ARFF
    arff_dir = output_dir / "arff_chunks"
    arff_dir.mkdir(parents=True, exist_ok=True)

    # Converter chunks para ARFF
    converter = ARFFConverter(relation_name=dataset_name)
    arff_files = []
    chunks_numpy = []

    for i, (X_river, y_river) in enumerate(chunks):
        # Converter River format para numpy
        if not X_river:
            continue

        # Pegar nomes de features do primeiro exemplo
        feature_names = list(X_river[0].keys())

        # Converter para numpy array
        X_array = np.array([[x[feat] for feat in feature_names] for x in X_river])
        y_array = np.array(y_river)

        chunks_numpy.append((X_array, y_array))

        # Converter para ARFF
        arff_file = arff_dir / f"chunk_{i}.arff"
        converter.convert_chunk(
            X=X_array,
            y=y_array,
            feature_names=feature_names,
            output_file=arff_file
        )
        arff_files.append(arff_file)

        logger.info(f"Chunk {i} convertido para ARFF: {arff_file.name}")

    # Criar wrapper ERulesD2S
    wrapper = ERulesD2SWrapper(
        moa_jar_path=str(jar_path),
        java_memory="4g",
        gpu_enabled=False
    )

    # Executar ERulesD2S em cada chunk
    results = []

    for chunk_idx, (arff_file, (X, y)) in enumerate(zip(arff_files, chunks_numpy)):
        logger.info(f"Processando chunk {chunk_idx+1}/{len(arff_files)} com ERulesD2S...")

        chunk_output_dir = output_dir / f"erulesd2s_chunk_{chunk_idx}"

        success, metrics = wrapper.run(
            arff_file=arff_file,
            output_dir=chunk_output_dir,
            population_size=population_size,
            num_generations=num_generations,
            rules_per_class=rules_per_class,
            chunk_size=len(X),
            max_instances=len(X),
            timeout=600  # 10 minutos timeout
        )

        if not success:
            logger.warning(f"ERulesD2S falhou no chunk {chunk_idx}")
            metrics = {'accuracy': 0.0, 'gmean': 0.0}

        # Calcular gmean se nao retornado
        if 'gmean' not in metrics and 'accuracy' in metrics:
            metrics['gmean'] = metrics['accuracy']

        result = {
            'chunk': chunk_idx + 1,
            'model': 'ERulesD2S',
            'train_size': len(X),
            'test_size': len(X),
            'accuracy': metrics.get('accuracy', 0.0),
            'gmean': metrics.get('gmean', 0.0),
            'f1_weighted': metrics.get('f1', metrics.get('accuracy', 0.0)),
            'execution_time': metrics.get('execution_time', 0.0)
        }

        results.append(result)
        logger.info(f"Chunk {chunk_idx+1}: accuracy={result['accuracy']:.4f}, gmean={result['gmean']:.4f}")

    return pd.DataFrame(results)


# ============================================================================
# FUNCAO PRINCIPAL
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Executar modelos comparativos em chunks GBML existentes'
    )

    parser.add_argument('--dataset', type=str, required=True,
                       help='Nome do dataset (ex: SEA_Abrupt_Simple)')
    parser.add_argument('--base-dir', type=str,
                       default='experiments_6chunks_phase1_gbml/batch_1',
                       help='Diretorio base dos experimentos')
    parser.add_argument('--models', nargs='+', default=['HAT', 'ARF', 'SRP'],
                       help='Modelos River a executar')
    parser.add_argument('--acdwm', action='store_true',
                       help='Incluir ACDWM')
    parser.add_argument('--acdwm-path', type=str, default='ACDWM',
                       help='Caminho para o repositorio ACDWM')
    parser.add_argument('--erulesd2s', action='store_true',
                       help='Incluir ERulesD2S')
    parser.add_argument('--erulesd2s-pop', type=int, default=25,
                       help='Tamanho da populacao ERulesD2S')
    parser.add_argument('--erulesd2s-gen', type=int, default=50,
                       help='Numero de geracoes ERulesD2S')
    parser.add_argument('--run', type=int, default=1,
                       help='Numero da run (padrao: 1)')

    args = parser.parse_args()

    # Construir paths
    base_dir = Path(args.base_dir)
    dataset_dir = base_dir / args.dataset / f"run_{args.run}"
    chunks_dir = dataset_dir / "chunk_data"

    logger.info("="*80)
    logger.info(f"EXECUTANDO MODELOS COMPARATIVOS - {args.dataset}")
    logger.info("="*80)
    logger.info(f"Base dir: {base_dir}")
    logger.info(f"Dataset dir: {dataset_dir}")
    logger.info(f"Chunks dir: {chunks_dir}")
    logger.info(f"Modelos River: {args.models}")
    logger.info(f"ACDWM: {args.acdwm}")
    logger.info(f"ERulesD2S: {args.erulesd2s}")
    logger.info("="*80)

    # Verificar se diretorio existe
    if not dataset_dir.exists():
        logger.error(f"Diretorio do dataset nao encontrado: {dataset_dir}")
        return 1

    start_time = time.time()

    # ========================================================================
    # PASSO 1: CARREGAR CHUNKS
    # ========================================================================
    logger.info("\n[1/3] Carregando chunks existentes...")

    try:
        chunks = load_chunks_from_csv(chunks_dir)
    except Exception as e:
        logger.error(f"Erro ao carregar chunks: {e}")
        return 1

    # Detectar classes
    all_labels = []
    for _, y in chunks:
        all_labels.extend(y)
    classes = sorted(list(set(all_labels)))
    logger.info(f"Classes detectadas: {classes}")

    # ========================================================================
    # PASSO 2: EXECUTAR MODELOS RIVER
    # ========================================================================
    logger.info(f"\n[2/3] Executando modelos River ({len(args.models)} modelos)...")

    river_results = []

    for model_name in args.models:
        logger.info(f"\n--- Modelo: {model_name} ---")

        try:
            df_result = run_river_model_on_chunks(model_name, chunks, classes)
            river_results.append(df_result)

            # Salvar resultado individual
            output_file = dataset_dir / f"river_{model_name}_results.csv"
            df_result.to_csv(output_file, index=False)
            logger.info(f"Resultados salvos: {output_file}")

        except Exception as e:
            logger.error(f"Erro ao executar {model_name}: {e}", exc_info=True)

    # ========================================================================
    # PASSO 3: EXECUTAR ACDWM (SE SOLICITADO)
    # ========================================================================
    if args.acdwm:
        logger.info("\n[3/4] Executando ACDWM...")

        try:
            df_acdwm = run_acdwm_on_chunks(chunks, classes, args.acdwm_path)
            river_results.append(df_acdwm)

            # Salvar resultado
            output_file = dataset_dir / "acdwm_results.csv"
            df_acdwm.to_csv(output_file, index=False)
            logger.info(f"Resultados salvos: {output_file}")

        except Exception as e:
            logger.error(f"Erro ao executar ACDWM: {e}", exc_info=True)
    else:
        logger.info("\n[3/4] ACDWM desabilitado")

    # ========================================================================
    # PASSO 4: EXECUTAR ERULESD2S (SE SOLICITADO)
    # ========================================================================
    if args.erulesd2s:
        logger.info("\n[4/4] Executando ERulesD2S...")

        try:
            df_erulesd2s = run_erulesd2s_on_chunks(
                chunks=chunks,
                dataset_name=args.dataset,
                output_dir=dataset_dir,
                population_size=args.erulesd2s_pop,
                num_generations=args.erulesd2s_gen,
                rules_per_class=5
            )

            if not df_erulesd2s.empty:
                river_results.append(df_erulesd2s)

                # Salvar resultado
                output_file = dataset_dir / "erulesd2s_results.csv"
                df_erulesd2s.to_csv(output_file, index=False)
                logger.info(f"Resultados salvos: {output_file}")

        except Exception as e:
            logger.error(f"Erro ao executar ERulesD2S: {e}", exc_info=True)
    else:
        logger.info("\n[4/4] ERulesD2S desabilitado")

    # ========================================================================
    # CONSOLIDAR RESULTADOS
    # ========================================================================
    if river_results:
        logger.info("\nConsolidando resultados...")

        consolidated = pd.concat(river_results, ignore_index=True)

        # Salvar consolidado
        consolidated_file = dataset_dir / "comparative_models_results.csv"
        consolidated.to_csv(consolidated_file, index=False)
        logger.info(f"Resultados consolidados salvos: {consolidated_file}")

        # Estatisticas resumidas
        logger.info("\n" + "="*80)
        logger.info("RESUMO DOS RESULTADOS")
        logger.info("="*80)

        summary = consolidated.groupby('model').agg({
            'train_gmean': ['mean', 'std'],
            'test_gmean': ['mean', 'std'],
            'train_accuracy': ['mean', 'std'],
            'test_accuracy': ['mean', 'std'],
            'train_f1': ['mean', 'std'],
            'test_f1': ['mean', 'std']
        }).round(4)

        print(summary)

    duration = time.time() - start_time

    logger.info("\n" + "="*80)
    logger.info("EXECUCAO CONCLUIDA")
    logger.info("="*80)
    logger.info(f"Duracao: {duration/60:.2f} minutos")
    logger.info(f"Resultados salvos em: {dataset_dir}")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\nExecucao interrompida pelo usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nErro nao tratado: {e}", exc_info=True)
        sys.exit(1)
