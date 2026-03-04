#!/usr/bin/env python3
"""
Script de Execucao: Experimento Comparativo GBML vs River vs ACDWM

PROPOSITO: Executar comparacao completa em 3 datasets no Google Colab
TEMPO ESTIMADO: 15.8h
DATASETS: RBF_Abrupt_Severe, RBF_Abrupt_Moderate, RBF_Gradual_Moderate
MODELOS: GBML + River (HAT, ARF, SRP) + ACDWM
"""

import subprocess
import os
import sys
from datetime import datetime
import json

# Configuracao
DATASETS = [
    'RBF_Abrupt_Severe',
    'RBF_Abrupt_Moderate',
    'RBF_Gradual_Moderate'
]

RIVER_MODELS = ['HAT', 'ARF', 'SRP']
NUM_CHUNKS = 3
CHUNK_SIZE = 6000
SEED = 42
CONFIG_FILE = 'config_comparison.yaml'
OUTPUT_BASE = 'comparison_results'

def print_header(message):
    """Imprime cabecalho formatado"""
    print("\n" + "="*70)
    print(message)
    print("="*70 + "\n")

def print_progress(current, total, dataset_name):
    """Imprime progresso"""
    percentage = (current / total) * 100
    print(f"\n[{current}/{total}] ({percentage:.1f}%) Executando {dataset_name}...")

def run_dataset_comparison(dataset_name, output_dir):
    """
    Executa comparacao para um dataset especifico

    Args:
        dataset_name: Nome do dataset
        output_dir: Diretorio de saida

    Returns:
        True se sucesso, False se erro
    """
    print(f"\nIniciando comparacao para: {dataset_name}")
    print(f"Modelos: GBML + River ({', '.join(RIVER_MODELS)}) + ACDWM")
    print(f"Chunks: {NUM_CHUNKS}")
    print(f"Chunk size: {CHUNK_SIZE}")

    cmd = [
        'python', 'compare_gbml_vs_river.py',
        '--stream', dataset_name,
        '--config', CONFIG_FILE,
        '--models'] + RIVER_MODELS + [
        '--chunks', str(NUM_CHUNKS),
        '--chunk-size', str(CHUNK_SIZE),
        '--acdwm',
        '--seed', str(SEED),
        '--output', output_dir
    ]

    print(f"\nComando: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Mostra output em tempo real
            text=True,
            check=True
        )

        print(f"\n✓ {dataset_name} concluido com sucesso!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERRO ao executar {dataset_name}:")
        print(f"  Return code: {e.returncode}")
        return False

    except Exception as e:
        print(f"\n✗ ERRO inesperado ao executar {dataset_name}:")
        print(f"  {type(e).__name__}: {e}")
        return False

def consolidate_results(output_dir):
    """Consolida resultados de todos os datasets"""

    print_header("CONSOLIDANDO RESULTADOS")

    try:
        import pandas as pd
        import glob

        # Busca todos os arquivos de comparacao
        pattern = os.path.join(output_dir, '**/comparison_table.csv')
        comparison_files = glob.glob(pattern, recursive=True)

        if not comparison_files:
            print("⚠ Nenhum arquivo de comparacao encontrado!")
            return False

        print(f"Encontrados {len(comparison_files)} arquivos de comparacao")

        # Consolida todos em um DataFrame
        all_results = []
        for file_path in comparison_files:
            try:
                df = pd.read_csv(file_path)

                # Extrai nome do dataset do caminho
                parts = file_path.split(os.sep)
                dataset_name = None
                for part in parts:
                    if part.startswith('RBF_'):
                        dataset_name = part.split('_seed')[0]  # Remove timestamp
                        break

                if dataset_name:
                    df['dataset'] = dataset_name
                    all_results.append(df)
                    print(f"  ✓ {dataset_name}: {len(df)} linhas")

            except Exception as e:
                print(f"  ✗ Erro ao ler {file_path}: {e}")

        if not all_results:
            print("⚠ Nenhum resultado valido encontrado!")
            return False

        # Combina tudo
        consolidated = pd.concat(all_results, ignore_index=True)

        # Salva consolidado
        output_path = os.path.join(output_dir, 'consolidated_results.csv')
        consolidated.to_csv(output_path, index=False)
        print(f"\n✓ Resultados consolidados salvos em: {output_path}")

        # Gera estatisticas resumidas
        summary = consolidated.groupby(['dataset', 'model']).agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'gmean': ['mean', 'std', 'min', 'max'],
            'f1_weighted': ['mean', 'std', 'min', 'max']
        }).round(4)

        # Salva summary
        summary_path = os.path.join(output_dir, 'summary_statistics.txt')
        with open(summary_path, 'w') as f:
            f.write('RESUMO ESTATISTICO - EXPERIMENTO COMPARATIVO\n')
            f.write('='*70 + '\n\n')
            f.write('Datasets: ' + ', '.join(DATASETS) + '\n')
            f.write('Modelos: GBML + River (' + ', '.join(RIVER_MODELS) + ') + ACDWM\n')
            f.write('Chunks por dataset: ' + str(NUM_CHUNKS) + '\n')
            f.write('Total de avaliacoes: ' + str(len(consolidated)) + '\n\n')
            f.write('ESTATISTICAS POR DATASET E MODELO:\n')
            f.write('-'*70 + '\n\n')
            f.write(str(summary))

        print(f"✓ Resumo estatistico salvo em: {summary_path}")

        # Imprime resumo na tela
        print("\n" + "="*70)
        print("RESUMO ESTATISTICO")
        print("="*70 + "\n")
        print(summary)
        print("\n" + "="*70)

        return True

    except ImportError as e:
        print(f"✗ Erro de importacao: {e}")
        print("  Certifique-se que pandas esta instalado: pip install pandas")
        return False

    except Exception as e:
        print(f"✗ Erro na consolidacao: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Funcao principal"""

    start_time = datetime.now()

    print_header("EXPERIMENTO COMPARATIVO: GBML VS RIVER VS ACDWM")

    print(f"Data/hora inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Chunks por dataset: {NUM_CHUNKS}")
    print(f"Modelos: GBML + {len(RIVER_MODELS)} River + ACDWM")
    print(f"Tempo estimado: 15.8 horas")
    print(f"Seed: {SEED}")

    # Cria diretorio de output com timestamp
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(OUTPUT_BASE, f'experiment_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output dir: {output_dir}")

    # Verifica arquivos necessarios
    required_files = [
        'compare_gbml_vs_river.py',
        'baseline_river.py',
        'shared_evaluation.py',
        'gbml_evaluator.py',
        CONFIG_FILE
    ]

    print("\nVerificando arquivos necessarios...")
    missing_files = []
    for filename in required_files:
        if os.path.exists(filename):
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} NAO ENCONTRADO")
            missing_files.append(filename)

    if missing_files:
        print(f"\n✗ Arquivos faltando: {', '.join(missing_files)}")
        print("Abortando experimento.")
        return 1

    # Salva configuracao do experimento
    config = {
        'start_time': start_time.isoformat(),
        'datasets': DATASETS,
        'river_models': RIVER_MODELS,
        'num_chunks': NUM_CHUNKS,
        'chunk_size': CHUNK_SIZE,
        'seed': SEED,
        'output_dir': output_dir
    }

    config_path = os.path.join(output_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Configuracao salva em: {config_path}")

    # Executa cada dataset
    results = {}

    for i, dataset in enumerate(DATASETS, 1):
        print_progress(i, len(DATASETS), dataset)

        dataset_start = datetime.now()
        success = run_dataset_comparison(dataset, output_dir)
        dataset_end = datetime.now()

        dataset_duration = (dataset_end - dataset_start).total_seconds() / 60

        results[dataset] = {
            'success': success,
            'duration_minutes': dataset_duration
        }

        if success:
            print(f"✓ {dataset} concluido em {dataset_duration:.1f} minutos")
        else:
            print(f"✗ {dataset} FALHOU apos {dataset_duration:.1f} minutos")
            print("Continuando com proximo dataset...")

    # Consolida resultados
    if any(r['success'] for r in results.values()):
        consolidate_results(output_dir)
    else:
        print("\n✗ Nenhum dataset foi executado com sucesso. Pulando consolidacao.")

    # Resumo final
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 3600

    print_header("EXPERIMENTO CONCLUIDO")

    print(f"Data/hora inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data/hora fim:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duracao total:    {total_duration:.2f} horas")
    print()

    print("RESULTADOS POR DATASET:")
    for dataset, result in results.items():
        status = "✓ Sucesso" if result['success'] else "✗ Falhou"
        duration = result['duration_minutes']
        print(f"  {dataset:30s} {status:12s} ({duration:.1f} min)")

    print(f"\nResultados salvos em: {output_dir}")

    # Salva resumo final
    final_summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'total_duration_hours': total_duration,
        'results': results
    }

    summary_path = os.path.join(output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2)

    print(f"Resumo final salvo em: {summary_path}")

    # Return code
    all_success = all(r['success'] for r in results.values())
    return 0 if all_success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
