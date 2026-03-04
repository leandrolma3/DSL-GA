#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar o status dos experimentos chunk_1000 (com e sem penalidade).
Verifica quais datasets estao completos, parciais ou nao iniciados.
"""

import os
import json
import sys
from pathlib import Path
from collections import defaultdict

# Forcar UTF-8 no Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuracao
BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR / "experiments_unified"
UNIFIED_CHUNKS_DIR = BASE_DIR / "unified_chunks" / "chunk_1000"

# Numero esperado de chunks processados (num_chunks - 1, pois primeiro chunk e treino)
EXPECTED_CHUNKS = 11  # 12 chunks total, 11 de teste

# Definicao dos batches baseada nos configs
BATCHES = {
    "batch_1": [
        "SEA_Abrupt_Simple",
        "SEA_Abrupt_Chain",
        "SEA_Abrupt_Recurring",
        "AGRAWAL_Abrupt_Simple_Mild",
        "AGRAWAL_Abrupt_Simple_Severe",
        "AGRAWAL_Abrupt_Chain_Long",
        "RBF_Abrupt_Severe",
        "RBF_Abrupt_Blip",
        "STAGGER_Abrupt_Chain",
        "STAGGER_Abrupt_Recurring",
        "HYPERPLANE_Abrupt_Simple",
        "RANDOMTREE_Abrupt_Simple",
        "SEA_Gradual_Simple_Fast",
    ],
    "batch_2": [
        "SEA_Gradual_Simple_Slow",
        "SEA_Gradual_Recurring",
        "STAGGER_Gradual_Chain",
        "RBF_Gradual_Moderate",
        "RBF_Gradual_Severe",
        "HYPERPLANE_Gradual_Simple",
        "RANDOMTREE_Gradual_Simple",
        "LED_Gradual_Simple",
        "SEA_Abrupt_Chain_Noise",
        "STAGGER_Abrupt_Chain_Noise",
        "AGRAWAL_Abrupt_Simple_Severe_Noise",
        "SINE_Abrupt_Recurring_Noise",
        "RBF_Abrupt_Blip_Noise",
    ],
    "batch_3": [
        "RBF_Gradual_Severe_Noise",
        "HYPERPLANE_Gradual_Noise",
        "RANDOMTREE_Gradual_Noise",
        "SINE_Abrupt_Simple",
        "SINE_Gradual_Recurring",
        "LED_Abrupt_Simple",
        "WAVEFORM_Abrupt_Simple",
        "WAVEFORM_Gradual_Simple",
        "RANDOMTREE_Abrupt_Recurring",
        "Electricity",
        "Shuttle",
        "CovType",
        "PokerHand",
    ],
    "batch_4": [
        "IntelLabSensors",
        "SEA_Stationary",
        "AGRAWAL_Stationary",
        "RBF_Stationary",
        "LED_Stationary",
        "HYPERPLANE_Stationary",
        "RANDOMTREE_Stationary",
        "STAGGER_Stationary",
        "WAVEFORM_Stationary",
        "SINE_Stationary",
        "AssetNegotiation_F2",
        "AssetNegotiation_F3",
        "AssetNegotiation_F4",
    ],
}


def check_experiment_status(exp_dir: Path, dataset: str) -> dict:
    """
    Verifica o status de um experimento especifico.
    Retorna: {'status': 'complete'|'partial'|'not_started', 'chunks': int, 'details': str}
    """
    dataset_dir = exp_dir / dataset / "run_1"

    if not dataset_dir.exists():
        return {"status": "not_started", "chunks": 0, "details": "Diretorio nao existe"}

    # Verificar chunk_metrics.json (metricas completas)
    metrics_file = dataset_dir / "chunk_metrics.json"
    partial_metrics_file = dataset_dir / "chunk_metrics_partial.json"

    chunks_count = 0
    metrics_source = None

    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                chunks_count = len(metrics)
                metrics_source = "chunk_metrics.json"
        except:
            pass

    if chunks_count == 0 and partial_metrics_file.exists():
        try:
            with open(partial_metrics_file, 'r') as f:
                metrics = json.load(f)
                chunks_count = len(metrics)
                metrics_source = "chunk_metrics_partial.json"
        except:
            pass

    # Contar checkpoints como backup
    checkpoints = list(dataset_dir.glob("chunk_data/best_individual_trained_on_chunk_*.pkl"))
    checkpoint_count = len(checkpoints)

    # Determinar status
    if chunks_count >= EXPECTED_CHUNKS:
        return {
            "status": "complete",
            "chunks": chunks_count,
            "details": f"{chunks_count}/{EXPECTED_CHUNKS} chunks ({metrics_source})"
        }
    elif chunks_count > 0:
        return {
            "status": "partial",
            "chunks": chunks_count,
            "details": f"{chunks_count}/{EXPECTED_CHUNKS} chunks ({metrics_source}), {checkpoint_count} checkpoints"
        }
    elif checkpoint_count > 0:
        return {
            "status": "partial",
            "chunks": checkpoint_count,
            "details": f"0 metricas, {checkpoint_count} checkpoints (precisa rebuild_metrics)"
        }
    else:
        return {
            "status": "not_started",
            "chunks": 0,
            "details": "Diretorio existe mas sem dados"
        }


def check_chunks_available(dataset: str) -> bool:
    """Verifica se os chunks pre-gerados existem para o dataset."""
    dataset_chunks_dir = UNIFIED_CHUNKS_DIR / dataset
    if not dataset_chunks_dir.exists():
        return False
    chunks = list(dataset_chunks_dir.glob("chunk_*.csv"))
    return len(chunks) >= 12


def print_separator(char="=", length=100):
    print(char * length)


def main():
    print_separator()
    print("VERIFICACAO DE STATUS - EXPERIMENTOS CHUNK_1000")
    print_separator()
    print(f"\nDiretorio base: {BASE_DIR}")
    print(f"Chunks esperados por dataset: {EXPECTED_CHUNKS}")
    print(f"Total de datasets: {sum(len(datasets) for datasets in BATCHES.values())}")
    print()

    # Verificar chunks pre-gerados
    print_separator("-")
    print("1. VERIFICACAO DE CHUNKS PRE-GERADOS")
    print_separator("-")

    missing_chunks = []
    available_chunks = []
    for batch_name, datasets in BATCHES.items():
        for dataset in datasets:
            if not check_chunks_available(dataset):
                missing_chunks.append(dataset)
            else:
                available_chunks.append(dataset)

    if missing_chunks:
        print(f"\n[!] {len(missing_chunks)} datasets SEM chunks pre-gerados:")
        for ds in missing_chunks:
            print(f"    - {ds}")
    else:
        print(f"\n[OK] Todos os {len(available_chunks)} datasets tem chunks pre-gerados em:")
        print(f"     {UNIFIED_CHUNKS_DIR}")

    # Experimentos sem penalidade
    experiments_to_check = [
        ("chunk_1000", "SEM penalidade (feature_penalty=0.0)"),
        ("chunk_1000_penalty", "COM penalidade (feature_penalty=0.1)"),
    ]

    overall_stats = {}

    for exp_name, exp_desc in experiments_to_check:
        print()
        print_separator("=")
        print(f"2. EXPERIMENTOS {exp_name.upper()}")
        print(f"   {exp_desc}")
        print_separator("=")

        exp_base_dir = EXPERIMENTS_DIR / exp_name

        if not exp_base_dir.exists():
            print(f"\n[X] Diretorio {exp_base_dir} NAO EXISTE")
            print("    Os experimentos ainda nao foram iniciados.")
            overall_stats[exp_name] = {
                "complete": 0,
                "partial": 0,
                "not_started": sum(len(datasets) for datasets in BATCHES.values())
            }
            continue

        stats = {"complete": 0, "partial": 0, "not_started": 0}
        details_by_status = defaultdict(list)

        for batch_name, datasets in BATCHES.items():
            print(f"\n--- {batch_name.upper()} ({len(datasets)} datasets) ---")
            batch_dir = exp_base_dir / batch_name

            batch_complete = 0
            batch_partial = 0
            batch_not_started = 0

            for dataset in datasets:
                result = check_experiment_status(batch_dir, dataset)
                status = result["status"]

                if status == "complete":
                    batch_complete += 1
                    stats["complete"] += 1
                    icon = "[OK]"
                elif status == "partial":
                    batch_partial += 1
                    stats["partial"] += 1
                    icon = "[~~]"
                    details_by_status["partial"].append(f"{batch_name}/{dataset}: {result['details']}")
                else:
                    batch_not_started += 1
                    stats["not_started"] += 1
                    icon = "[  ]"
                    details_by_status["not_started"].append(f"{batch_name}/{dataset}")

                print(f"  {icon} {dataset}: {result['details']}")

            print(f"\n  Resumo {batch_name}: {batch_complete} completos, {batch_partial} parciais, {batch_not_started} nao iniciados")

        # Resumo do experimento
        total = stats["complete"] + stats["partial"] + stats["not_started"]
        pct_complete = (stats["complete"] / total * 100) if total > 0 else 0

        print()
        print_separator("-")
        print(f"RESUMO {exp_name.upper()}:")
        print_separator("-")
        print(f"  [OK] Completos:     {stats['complete']:3d} / {total} ({pct_complete:.1f}%)")
        print(f"  [~~] Parciais:      {stats['partial']:3d} / {total}")
        print(f"  [  ] Nao iniciados: {stats['not_started']:3d} / {total}")

        overall_stats[exp_name] = stats

    # Resumo geral
    print()
    print_separator("=")
    print("RESUMO GERAL - CHUNK_1000")
    print_separator("=")

    total_experiments = sum(len(datasets) for datasets in BATCHES.values()) * 2  # com e sem penalty
    total_complete = sum(s["complete"] for s in overall_stats.values())
    total_partial = sum(s["partial"] for s in overall_stats.values())
    total_not_started = sum(s["not_started"] for s in overall_stats.values())

    print(f"\nTotal de experimentos: {total_experiments}")
    print(f"  - chunk_1000 (sem penalty): 52 datasets")
    print(f"  - chunk_1000_penalty:       52 datasets")
    print()
    print(f"[OK] Completos:     {total_complete:3d} / {total_experiments} ({total_complete/total_experiments*100:.1f}%)")
    print(f"[~~] Parciais:      {total_partial:3d} / {total_experiments}")
    print(f"[  ] Nao iniciados: {total_not_started:3d} / {total_experiments}")

    # Proximos passos
    print()
    print_separator("-")
    print("PROXIMOS PASSOS:")
    print_separator("-")

    if total_complete == total_experiments:
        print("[***] Todos os experimentos chunk_1000 estao COMPLETOS!")
    else:
        if total_not_started > 0:
            print(f"1. Executar {total_not_started} experimentos nao iniciados")
            print("   Comandos sugeridos:")
            for exp_name, _ in experiments_to_check:
                if overall_stats.get(exp_name, {}).get("not_started", 0) > 0:
                    for batch_num in range(1, 5):
                        config_file = f"configs/config_unified_chunk1000{'_penalty' if 'penalty' in exp_name else ''}_batch_{batch_num}.yaml"
                        print(f"   python main.py --config {config_file}")

        if total_partial > 0:
            print(f"\n2. Retomar {total_partial} experimentos parciais")
            print("   (O sistema de resume deve continuar automaticamente)")

    print()
    print_separator("=")


if __name__ == "__main__":
    main()
