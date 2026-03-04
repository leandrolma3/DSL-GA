#!/usr/bin/env python3
"""
generate_unified_chunks.py

Gera chunks unificados para todos os 52 datasets com a mesma semente (RANDOM_SEED=42).
Estrategia:
1. Gerar chunks de 2000 (base)
2. Dividir deterministicamente para 1000
3. Dividir deterministicamente para 500

Isso garante comparabilidade justa entre experimentos com diferentes chunk sizes.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime

# Configuracao de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("generate_unified_chunks")

# Semente global para reproducibilidade
RANDOM_SEED = 42

# Lista completa dos 52 datasets (extraida do EXP-B)
ALL_DATASETS = [
    # SEA (8 datasets)
    "SEA_Abrupt_Simple", "SEA_Abrupt_Chain", "SEA_Abrupt_Chain_Noise", "SEA_Abrupt_Recurring",
    "SEA_Gradual_Simple_Fast", "SEA_Gradual_Simple_Slow", "SEA_Gradual_Recurring", "SEA_Stationary",

    # AGRAWAL (5 datasets)
    "AGRAWAL_Abrupt_Simple_Mild", "AGRAWAL_Abrupt_Simple_Severe", "AGRAWAL_Abrupt_Simple_Severe_Noise",
    "AGRAWAL_Abrupt_Chain_Long", "AGRAWAL_Stationary",

    # RBF (8 datasets)
    "RBF_Abrupt_Severe", "RBF_Abrupt_Blip", "RBF_Abrupt_Blip_Noise",
    "RBF_Gradual_Moderate", "RBF_Gradual_Severe", "RBF_Gradual_Severe_Noise", "RBF_Stationary",

    # STAGGER (5 datasets)
    "STAGGER_Abrupt_Chain", "STAGGER_Abrupt_Chain_Noise", "STAGGER_Abrupt_Recurring",
    "STAGGER_Gradual_Chain", "STAGGER_Stationary",

    # HYPERPLANE (4 datasets)
    "HYPERPLANE_Abrupt_Simple", "HYPERPLANE_Gradual_Simple", "HYPERPLANE_Gradual_Noise", "HYPERPLANE_Stationary",

    # RANDOMTREE (5 datasets)
    "RANDOMTREE_Abrupt_Simple", "RANDOMTREE_Abrupt_Recurring",
    "RANDOMTREE_Gradual_Simple", "RANDOMTREE_Gradual_Noise", "RANDOMTREE_Stationary",

    # SINE (4 datasets)
    "SINE_Abrupt_Simple", "SINE_Abrupt_Recurring_Noise", "SINE_Gradual_Recurring", "SINE_Stationary",

    # LED (3 datasets)
    "LED_Abrupt_Simple", "LED_Gradual_Simple", "LED_Stationary",

    # WAVEFORM (3 datasets)
    "WAVEFORM_Abrupt_Simple", "WAVEFORM_Gradual_Simple", "WAVEFORM_Stationary",

    # Datasets Reais (5 datasets)
    "Electricity", "Shuttle", "CovType", "PokerHand", "IntelLabSensors",

    # AssetNegotiation (3 datasets)
    "AssetNegotiation_F2", "AssetNegotiation_F3", "AssetNegotiation_F4",
]

# Nota: A lista acima tem 51 datasets. Verificando a lista do EXP-B, faltou RBF_Stationary que estava duplicado
# Ajustando para 52:
ALL_DATASETS = list(set(ALL_DATASETS))  # Remove duplicatas se houver


def river_format_to_dataframe(X_chunk: List[Dict], y_chunk: List) -> pd.DataFrame:
    """Converte dados no formato River (list of dicts) para DataFrame."""
    df = pd.DataFrame(X_chunk)
    df['target'] = y_chunk
    return df


def dataframe_to_river_format(df: pd.DataFrame) -> Tuple[List[Dict], List]:
    """Converte DataFrame para formato River (list of dicts, list of labels)."""
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col]).to_dict('records')
    y = df[target_col].tolist()
    return X, y


def split_chunks_2000_to_smaller(chunks_2000: List[pd.DataFrame], target_size: int) -> List[pd.DataFrame]:
    """
    Divide chunks de 2000 instancias para tamanhos menores deterministicamente.

    Args:
        chunks_2000: Lista de DataFrames, cada um com 2000 instancias
        target_size: Tamanho alvo (1000 ou 500)

    Returns:
        Lista de DataFrames com o tamanho alvo
    """
    result = []
    num_splits = 2000 // target_size  # 2 para 1000, 4 para 500

    for chunk_df in chunks_2000:
        for i in range(num_splits):
            start = i * target_size
            end = start + target_size
            split_chunk = chunk_df.iloc[start:end].reset_index(drop=True)
            result.append(split_chunk)

    return result


def generate_chunks_for_dataset(
    dataset_name: str,
    chunk_size: int,
    num_chunks: int,
    config_path: str
) -> List[Tuple[List[Dict], List]]:
    """
    Gera chunks para um dataset usando data_handling.

    Args:
        dataset_name: Nome do dataset/stream
        chunk_size: Tamanho de cada chunk
        num_chunks: Numero de chunks a gerar
        config_path: Caminho para o arquivo de configuracao

    Returns:
        Lista de tuples (X_chunk, y_chunk) no formato River
    """
    # Importar data_handling
    import data_handling

    # Setar semente antes de gerar
    np.random.seed(RANDOM_SEED)

    # Reset do config global para forcar recarga
    data_handling.CONFIG = None

    max_instances = chunk_size * num_chunks

    chunks = data_handling.generate_dataset_chunks(
        stream_or_dataset_name=dataset_name,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        max_instances=max_instances,
        config_path=config_path
    )

    return chunks


def save_chunks_to_csv(
    chunks: List[Tuple[List[Dict], List]],
    output_dir: Path,
    dataset_name: str
) -> None:
    """
    Salva chunks como arquivos CSV.

    Args:
        chunks: Lista de tuples (X_chunk, y_chunk)
        output_dir: Diretorio base de saida
        dataset_name: Nome do dataset
    """
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for i, (X_chunk, y_chunk) in enumerate(chunks):
        df = river_format_to_dataframe(X_chunk, y_chunk)
        csv_path = dataset_dir / f"chunk_{i}.csv"
        df.to_csv(csv_path, index=False)

    logger.info(f"  Salvos {len(chunks)} chunks em {dataset_dir}")


def load_chunks_from_csv(chunks_dir: Path) -> List[pd.DataFrame]:
    """
    Carrega chunks de arquivos CSV como DataFrames.

    Args:
        chunks_dir: Diretorio contendo os arquivos chunk_*.csv

    Returns:
        Lista de DataFrames
    """
    csv_files = sorted(chunks_dir.glob("chunk_*.csv"), key=lambda x: int(x.stem.split('_')[1]))
    chunks = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        chunks.append(df)
    return chunks


def derive_smaller_chunks(
    base_dir: Path,
    target_size: int,
    output_dir: Path,
    datasets: List[str]
) -> None:
    """
    Deriva chunks menores a partir de chunks de 2000.

    Args:
        base_dir: Diretorio com chunks de 2000
        target_size: Tamanho alvo (1000 ou 500)
        output_dir: Diretorio de saida
        datasets: Lista de datasets a processar
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Derivando chunks de {target_size} a partir de chunks de 2000")
    logger.info(f"{'='*60}")

    for dataset_name in datasets:
        source_dir = base_dir / dataset_name

        if not source_dir.exists():
            logger.warning(f"  {dataset_name}: Diretorio fonte nao encontrado, pulando...")
            continue

        logger.info(f"  Processando {dataset_name}...")

        # Carregar chunks de 2000
        chunks_2000 = load_chunks_from_csv(source_dir)

        if not chunks_2000:
            logger.warning(f"  {dataset_name}: Nenhum chunk encontrado, pulando...")
            continue

        # Dividir para o tamanho alvo
        smaller_chunks = split_chunks_2000_to_smaller(chunks_2000, target_size)

        # Salvar
        target_dataset_dir = output_dir / dataset_name
        target_dataset_dir.mkdir(parents=True, exist_ok=True)

        for i, df in enumerate(smaller_chunks):
            csv_path = target_dataset_dir / f"chunk_{i}.csv"
            df.to_csv(csv_path, index=False)

        logger.info(f"    {len(chunks_2000)} chunks de 2000 -> {len(smaller_chunks)} chunks de {target_size}")


def main():
    parser = argparse.ArgumentParser(description='Gera chunks unificados para experimentos')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Caminho para o arquivo de configuracao')
    parser.add_argument('--output-base', type=str, default='unified_chunks',
                        help='Diretorio base de saida')
    parser.add_argument('--num-chunks', type=int, default=6,
                        help='Numero de chunks base (para tamanho 2000)')
    parser.add_argument('--datasets', type=str, nargs='*', default=None,
                        help='Lista de datasets especificos (default: todos os 52)')
    parser.add_argument('--skip-generation', action='store_true',
                        help='Pular geracao de chunks 2000, apenas derivar menores')
    parser.add_argument('--only-size', type=int, choices=[2000, 1000, 500], default=None,
                        help='Gerar apenas um tamanho especifico')

    args = parser.parse_args()

    # Definir datasets a processar
    datasets = args.datasets if args.datasets else ALL_DATASETS

    # Diretorio base de saida
    output_base = Path(args.output_base)
    output_2000 = output_base / "chunk_2000"
    output_1000 = output_base / "chunk_1000"
    output_500 = output_base / "chunk_500"

    # Criar diretorios
    output_2000.mkdir(parents=True, exist_ok=True)
    output_1000.mkdir(parents=True, exist_ok=True)
    output_500.mkdir(parents=True, exist_ok=True)

    # Log de inicio
    logger.info(f"\n{'='*60}")
    logger.info(f"GERACAO UNIFICADA DE CHUNKS")
    logger.info(f"{'='*60}")
    logger.info(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Semente: {RANDOM_SEED}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_base}")
    logger.info(f"Datasets: {len(datasets)}")
    logger.info(f"Num chunks (2000): {args.num_chunks}")

    # Salvar metadados
    metadata = {
        "random_seed": RANDOM_SEED,
        "config_path": args.config,
        "num_chunks_2000": args.num_chunks,
        "num_chunks_1000": args.num_chunks * 2,
        "num_chunks_500": args.num_chunks * 4,
        "datasets": datasets,
        "generated_at": datetime.now().isoformat()
    }

    with open(output_base / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # FASE 1: Gerar chunks de 2000
    if not args.skip_generation and args.only_size in [None, 2000]:
        logger.info(f"\n{'='*60}")
        logger.info("FASE 1: Gerando chunks de 2000 instancias")
        logger.info(f"{'='*60}")

        np.random.seed(RANDOM_SEED)

        progress_file = output_2000 / "progress.json"
        completed = []
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                completed = json.load(f)
            logger.info(f"Retomando: {len(completed)} datasets ja processados")

        for i, dataset_name in enumerate(datasets):
            if dataset_name in completed:
                logger.info(f"[{i+1}/{len(datasets)}] {dataset_name}: ja processado, pulando...")
                continue

            logger.info(f"[{i+1}/{len(datasets)}] Gerando chunks para {dataset_name}...")

            try:
                # Setar semente antes de cada dataset para reproducibilidade
                np.random.seed(RANDOM_SEED + hash(dataset_name) % 1000)

                chunks = generate_chunks_for_dataset(
                    dataset_name=dataset_name,
                    chunk_size=2000,
                    num_chunks=args.num_chunks,
                    config_path=args.config
                )

                save_chunks_to_csv(chunks, output_2000, dataset_name)

                # Atualizar progresso
                completed.append(dataset_name)
                with open(progress_file, 'w') as f:
                    json.dump(completed, f, indent=2)

            except Exception as e:
                logger.error(f"  ERRO ao processar {dataset_name}: {e}")
                continue

        logger.info(f"\nFASE 1 concluida: {len(completed)}/{len(datasets)} datasets processados")

    # FASE 2: Derivar chunks de 1000
    if args.only_size in [None, 1000]:
        derive_smaller_chunks(output_2000, 1000, output_1000, datasets)

    # FASE 3: Derivar chunks de 500
    if args.only_size in [None, 500]:
        derive_smaller_chunks(output_2000, 500, output_500, datasets)

    # Log final
    logger.info(f"\n{'='*60}")
    logger.info("GERACAO CONCLUIDA!")
    logger.info(f"{'='*60}")
    logger.info(f"Chunks de 2000: {output_2000}")
    logger.info(f"Chunks de 1000: {output_1000}")
    logger.info(f"Chunks de 500: {output_500}")


if __name__ == "__main__":
    main()
