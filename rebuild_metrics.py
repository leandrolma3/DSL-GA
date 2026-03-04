#!/usr/bin/env python3
"""
Script para reconstruir chunk_metrics.json a partir dos checkpoints .pkl
quando as metricas foram perdidas durante o resume.
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np

# Adicionar o diretorio raiz ao path para imports
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.metrics import f1_score
from metrics import calculate_gmean_contextual

def load_chunks_from_csv(chunks_dir):
    """Carrega chunks de arquivos CSV pre-gerados."""
    import pandas as pd

    chunks = []
    csv_files = sorted(Path(chunks_dir).glob("chunk_*.csv"),
                       key=lambda x: int(x.stem.split('_')[1]))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        target_col = df.columns[-1]
        X = df.drop(columns=[target_col]).to_dict('records')
        y = df[target_col].tolist()
        chunks.append((X, y))

    return chunks

def rebuild_metrics_for_experiment(experiment_dir, chunks_dir):
    """Reconstroi metricas para um experimento a partir dos checkpoints."""

    experiment_dir = Path(experiment_dir)
    chunks_dir = Path(chunks_dir)

    run_dir = experiment_dir / 'run_1'
    chunk_data_dir = run_dir / 'chunk_data'

    if not chunk_data_dir.exists():
        print(f"  [ERRO] Diretorio chunk_data nao existe: {chunk_data_dir}")
        return None

    # Encontrar todos os checkpoints
    checkpoints = sorted(
        chunk_data_dir.glob('best_individual_trained_on_chunk_*.pkl'),
        key=lambda x: int(x.stem.split('_')[-1])
    )

    if not checkpoints:
        print(f"  [ERRO] Nenhum checkpoint encontrado em {chunk_data_dir}")
        return None

    print(f"  Encontrados {len(checkpoints)} checkpoints")

    # Carregar chunks pre-gerados
    if not chunks_dir.exists():
        print(f"  [ERRO] Diretorio de chunks nao existe: {chunks_dir}")
        return None

    chunks = load_chunks_from_csv(chunks_dir)
    print(f"  Carregados {len(chunks)} chunks de dados")

    # Obter classes unicas
    all_classes = set()
    for _, y in chunks:
        all_classes.update(y)
    classes = sorted(list(all_classes))
    print(f"  Classes encontradas: {len(classes)}")

    # Reconstruir metricas
    all_metrics = []

    for ckpt_path in checkpoints:
        chunk_idx = int(ckpt_path.stem.split('_')[-1])

        # Carregar individuo
        try:
            with open(ckpt_path, 'rb') as f:
                individual = pickle.load(f)
        except Exception as e:
            print(f"  [ERRO] Falha ao carregar {ckpt_path}: {e}")
            continue

        # Chunk de treino e teste
        train_X, train_y = chunks[chunk_idx]
        test_X, test_y = chunks[chunk_idx + 1] if chunk_idx + 1 < len(chunks) else ([], [])

        if not test_X:
            print(f"  [AVISO] Chunk {chunk_idx}: sem dados de teste")
            continue

        # Calcular metricas de treino
        try:
            train_preds = [individual._predict(inst) for inst in train_X]
            train_gmean = calculate_gmean_contextual(train_y, train_preds, classes)
        except Exception as e:
            print(f"  [AVISO] Chunk {chunk_idx}: erro no calculo de train_gmean: {e}")
            train_gmean = 0.0

        # Calcular metricas de teste
        try:
            test_preds = [individual._predict(inst) for inst in test_X]
            test_gmean = calculate_gmean_contextual(test_y, test_preds, classes)
            test_f1 = f1_score(test_y, test_preds, average='weighted', zero_division=0)
        except Exception as e:
            print(f"  [AVISO] Chunk {chunk_idx}: erro no calculo de test metrics: {e}")
            test_gmean = 0.0
            test_f1 = 0.0

        metrics = {
            'chunk': chunk_idx,
            'train_gmean': float(train_gmean),
            'test_gmean': float(test_gmean),
            'test_f1': float(test_f1)
        }
        all_metrics.append(metrics)
        print(f"    Chunk {chunk_idx}: train_gmean={train_gmean:.4f}, test_gmean={test_gmean:.4f}, test_f1={test_f1:.4f}")

    return all_metrics


def main():
    base_results = Path('experiments_unified')
    base_chunks = Path('unified_chunks')

    # Experimentos a reconstruir - metricas parciais para resume funcionar
    experiments_to_rebuild = [
        # Datasets parciais com checkpoints mas sem metricas
        ('chunk_500/batch_3/RBF_Stationary', 'chunk_500/RBF_Stationary'),
        ('chunk_500_penalty/batch_1/RBF_Gradual_Severe', 'chunk_500/RBF_Gradual_Severe'),
        ('chunk_500_penalty/batch_2/RANDOMTREE_Gradual_Noise', 'chunk_500/RANDOMTREE_Gradual_Noise'),
        ('chunk_500_penalty/batch_3/RBF_Stationary', 'chunk_500/RBF_Stationary'),
    ]

    for exp_rel, chunks_rel in experiments_to_rebuild:
        exp_dir = base_results / exp_rel
        chunks_dir = base_chunks / chunks_rel

        print(f"\n{'='*60}")
        print(f"Reconstruindo: {exp_rel}")
        print(f"{'='*60}")

        if not exp_dir.exists():
            print(f"  [SKIP] Diretorio nao existe: {exp_dir}")
            continue

        metrics = rebuild_metrics_for_experiment(exp_dir, chunks_dir)

        if metrics:
            # Determinar se e completo ou parcial (23 chunks para chunk_500)
            expected_chunks = 23
            is_complete = len(metrics) >= expected_chunks

            # Salvar metricas reconstruidas
            if is_complete:
                output_path = exp_dir / 'run_1' / 'chunk_metrics.json'
            else:
                output_path = exp_dir / 'run_1' / 'chunk_metrics_partial.json'

            backup_path = exp_dir / 'run_1' / 'chunk_metrics_backup.json'

            # Fazer backup do arquivo existente
            if output_path.exists():
                import shutil
                shutil.copy(output_path, backup_path)
                print(f"  Backup salvo em: {backup_path}")

            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            status = "COMPLETO" if is_complete else "PARCIAL"
            print(f"  [{status}] Metricas salvas em: {output_path}")
            print(f"  Total de chunks reconstruidos: {len(metrics)}/{expected_chunks}")


if __name__ == '__main__':
    main()
