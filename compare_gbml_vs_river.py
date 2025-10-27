# compare_gbml_vs_river.py
"""
Script de Comparação Unificada: GBML vs River Baselines

PROPÓSITO: Executar GBML e modelos River nos MESMOS dados com a MESMA
           metodologia de avaliação para comparação científica válida.

GARANTE:
- Mesmos chunks de treino/teste
- Mesma ordem de instâncias
- Mesmas seeds de aleatoriedade
- Mesmas métricas de avaliação

AUTOR: Claude Code
DATA: 2025-01-06
"""

import os
import logging
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Optional

# Módulos unificados
from shared_evaluation import (
    load_or_generate_chunks,
    validate_chunks_consistency,
    compare_results
)
from baseline_river import create_river_model, run_river_baselines
from gbml_evaluator import create_gbml_from_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("comparison")


# ============================================================================
# FUNÇÃO PRINCIPAL DE COMPARAÇÃO
# ============================================================================

def run_full_comparison(
    stream_name: str,
    config_path: str = 'config.yaml',
    river_models: List[str] = None,
    chunk_size: int = 6000,
    num_chunks: int = 5,
    seed: int = 42,
    output_dir: str = 'comparison_results',
    run_gbml: bool = True,
    run_river: bool = True,
    force_regenerate: bool = False
) -> Dict:
    """
    Executa comparação completa entre GBML e modelos River.

    Args:
        stream_name: Nome do stream (e.g., 'SEA_Abrupt_Simple')
        config_path: Caminho para config.yaml
        river_models: Lista de modelos River a testar
        chunk_size: Tamanho dos chunks
        num_chunks: Número de chunks
        seed: Seed para reprodutibilidade
        output_dir: Diretório para salvar resultados
        run_gbml: Se True, executa GBML
        run_river: Se True, executa River
        force_regenerate: Se True, força regeneração de chunks

    Returns:
        Dicionário com todos os resultados
    """
    logger.info("="*70)
    logger.info(f"COMPARAÇÃO: GBML vs River")
    logger.info(f"Stream: {stream_name}")
    logger.info(f"Chunks: {num_chunks} x {chunk_size} instâncias")
    logger.info(f"Seed: {seed}")
    logger.info("="*70)

    # Cria diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    experiment_dir = os.path.join(
        output_dir,
        f"{stream_name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    logger.info(f"Resultados serão salvos em: {experiment_dir}")

    # ========================================================================
    # PASSO 1: CARREGAR CONFIGURAÇÃO
    # ========================================================================
    logger.info("\n[1/5] Carregando configuração...")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Configuração carregada de: {config_path}")
    except Exception as e:
        logger.error(f"✗ Erro ao carregar config: {e}")
        raise

    # ========================================================================
    # PASSO 2: GERAR/CARREGAR CHUNKS (COMPARTILHADOS)
    # ========================================================================
    logger.info("\n[2/5] Gerando/carregando chunks (compartilhados)...")

    data_params = config.get('data_params', {})
    max_instances = data_params.get('max_instances', chunk_size * (num_chunks + 1))

    chunks = load_or_generate_chunks(
        stream_name=stream_name,
        chunk_size=chunk_size,
        num_chunks=num_chunks + 1,  # +1 para ter chunk de teste final
        max_instances=max_instances,
        config_path=config_path,
        seed=seed,
        force_regenerate=force_regenerate
    )

    # Valida chunks
    if not validate_chunks_consistency(chunks):
        raise ValueError("Chunks inválidos! Verifique a geração de dados.")

    logger.info(f"✓ {len(chunks)} chunks validados e prontos")

    # Extrai metadados dos chunks
    X_sample, y_sample = chunks[0]
    attributes = sorted(list(X_sample[0].keys()))
    classes = sorted(list(set([y for _, y_chunk in chunks for y in y_chunk])))

    logger.info(f"  Atributos: {len(attributes)}")
    logger.info(f"  Classes: {classes}")

    # ========================================================================
    # PASSO 3: EXECUTAR GBML (SE SOLICITADO)
    # ========================================================================
    gbml_results = None

    if run_gbml:
        logger.info("\n[3/5] Executando GBML...")

        try:
            # Extrai informações necessárias dos chunks
            from data_handling import load_full_config
            full_config = load_full_config(config_path)

            # Detecta tipos de features
            categorical_features = set()
            numeric_features = set()
            value_ranges = {}
            category_values = {}

            for attr in attributes:
                sample_values = [x[attr] for x, _ in chunks[:3] for x in chunks[0][0][:100]]
                if all(isinstance(v, (int, float)) for v in sample_values):
                    numeric_features.add(attr)
                    value_ranges[attr] = (min(sample_values), max(sample_values))
                else:
                    categorical_features.add(attr)
                    category_values[attr] = set(sample_values)

            # Cria avaliador GBML (com output_dir para salvar indivíduos)
            gbml_evaluator = create_gbml_from_config(
                config=config,
                attributes=attributes,
                value_ranges=value_ranges,
                category_values=category_values,
                categorical_features=categorical_features,
                classes=classes,
                output_dir=experiment_dir  # Salva no diretório do experimento
            )

            # Executa avaliação prequential
            gbml_results = gbml_evaluator.evaluate_prequential(chunks)

            # Salva resultados
            gbml_path = os.path.join(experiment_dir, 'gbml_results.csv')
            gbml_results.to_csv(gbml_path, index=False)
            logger.info(f"✓ Resultados GBML salvos em: {gbml_path}")

        except Exception as e:
            logger.error(f"✗ Erro ao executar GBML: {e}", exc_info=True)
            gbml_results = None

    else:
        logger.info("\n[3/5] GBML desabilitado (run_gbml=False)")

    # ========================================================================
    # PASSO 4: EXECUTAR RIVER (SE SOLICITADO)
    # ========================================================================
    river_results = {}

    if run_river:
        logger.info("\n[4/5] Executando modelos River...")

        if river_models is None:
            river_models = ['HAT', 'ARF', 'SRP']

        river_results = run_river_baselines(
            chunks=chunks,
            classes=classes,
            model_names=river_models
        )

        # Salva resultados de cada modelo
        for model_name, df_result in river_results.items():
            if df_result is not None:
                river_path = os.path.join(experiment_dir, f'river_{model_name}_results.csv')
                df_result.to_csv(river_path, index=False)
                logger.info(f"✓ Resultados {model_name} salvos em: {river_path}")

    else:
        logger.info("\n[4/5] River desabilitado (run_river=False)")

    # ========================================================================
    # PASSO 5: ANÁLISE COMPARATIVA E VISUALIZAÇÕES
    # ========================================================================
    logger.info("\n[5/5] Gerando análise comparativa...")

    try:
        # Combina todos os resultados
        all_results = []

        if gbml_results is not None:
            all_results.append(gbml_results)

        for model_name, df in river_results.items():
            if df is not None:
                all_results.append(df)

        if len(all_results) > 0:
            # Tabela comparativa
            combined_df = pd.concat(all_results, ignore_index=True)
            comparison_path = os.path.join(experiment_dir, 'comparison_table.csv')
            combined_df.to_csv(comparison_path, index=False)
            logger.info(f"✓ Tabela comparativa salva em: {comparison_path}")

            # Gera gráficos
            generate_comparison_plots(combined_df, experiment_dir)

            # Estatísticas resumidas
            summary = generate_summary_statistics(combined_df)
            summary_path = os.path.join(experiment_dir, 'summary.txt')
            with open(summary_path, 'w') as f:
                f.write(summary)
            logger.info(f"✓ Resumo salvo em: {summary_path}")

            print("\n" + "="*70)
            print("RESUMO DOS RESULTADOS")
            print("="*70)
            print(summary)

        else:
            logger.warning("Nenhum resultado disponível para análise")

    except Exception as e:
        logger.error(f"✗ Erro na análise comparativa: {e}", exc_info=True)

    logger.info("\n" + "="*70)
    logger.info("COMPARAÇÃO CONCLUÍDA!")
    logger.info(f"Resultados em: {experiment_dir}")
    logger.info("="*70)

    return {
        'gbml_results': gbml_results,
        'river_results': river_results,
        'experiment_dir': experiment_dir,
        'stream_name': stream_name,
        'seed': seed
    }


# ============================================================================
# FUNÇÕES AUXILIARES PARA ANÁLISE E VISUALIZAÇÃO
# ============================================================================

def generate_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Gera gráficos comparativos"""

    # Plot 1: Accuracy ao longo dos chunks
    plt.figure(figsize=(12, 6))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.plot(model_data['chunk'], model_data['accuracy'],
                marker='o', label=model, linewidth=2)

    plt.xlabel('Chunk', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Comparison Across Chunks', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
    plt.close()

    # Plot 2: G-mean comparison
    plt.figure(figsize=(12, 6))
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.plot(model_data['chunk'], model_data['gmean'],
                marker='s', label=model, linewidth=2)

    plt.xlabel('Chunk', fontsize=12)
    plt.ylabel('G-mean', fontsize=12)
    plt.title('G-mean Comparison Across Chunks', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gmean_comparison.png'), dpi=300)
    plt.close()

    # Plot 3: Heatmap de performance
    pivot_acc = df.pivot(index='chunk', columns='model', values='accuracy')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': 'Accuracy'})
    plt.title('Accuracy Heatmap: Models vs Chunks', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_heatmap.png'), dpi=300)
    plt.close()

    logger.info(f"✓ Gráficos salvos em: {output_dir}")


def generate_summary_statistics(df: pd.DataFrame) -> str:
    """Gera estatísticas resumidas"""

    summary_lines = []

    summary_lines.append("ESTATÍSTICAS RESUMIDAS")
    summary_lines.append("-" * 70)

    for model in df['model'].unique():
        model_data = df[df['model'] == model]

        summary_lines.append(f"\nModelo: {model}")
        summary_lines.append(f"  Accuracy média:  {model_data['accuracy'].mean():.4f} ± {model_data['accuracy'].std():.4f}")
        summary_lines.append(f"  F1 média:        {model_data['f1_weighted'].mean():.4f} ± {model_data['f1_weighted'].std():.4f}")
        summary_lines.append(f"  G-mean média:    {model_data['gmean'].mean():.4f} ± {model_data['gmean'].std():.4f}")

    return "\n".join(summary_lines)


# ============================================================================
# INTERFACE DE LINHA DE COMANDO
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Comparação GBML vs River Baselines'
    )

    parser.add_argument('--stream', type=str, required=True,
                       help='Nome do stream (e.g., SEA_Abrupt_Simple)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Caminho para config.yaml')
    parser.add_argument('--models', nargs='+', default=['HAT', 'ARF', 'SRP'],
                       help='Modelos River a testar')
    parser.add_argument('--chunks', type=int, default=5,
                       help='Número de chunks')
    parser.add_argument('--chunk-size', type=int, default=6000,
                       help='Tamanho dos chunks')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed para reprodutibilidade')
    parser.add_argument('--output', type=str, default='comparison_results',
                       help='Diretório de saída')
    parser.add_argument('--no-gbml', action='store_true',
                       help='Não executar GBML')
    parser.add_argument('--no-river', action='store_true',
                       help='Não executar River')
    parser.add_argument('--force', action='store_true',
                       help='Forçar regeneração de chunks')

    args = parser.parse_args()

    # Processa lista de modelos (suporta tanto --models HAT ARF quanto --models HAT,ARF,SRP)
    river_models = []
    for model_arg in args.models:
        if ',' in model_arg:
            # Split por vírgula se fornecido como string única
            river_models.extend(model_arg.split(','))
        else:
            river_models.append(model_arg)

    # Executa comparação
    results = run_full_comparison(
        stream_name=args.stream,
        config_path=args.config,
        river_models=river_models,
        chunk_size=args.chunk_size,
        num_chunks=args.chunks,
        seed=args.seed,
        output_dir=args.output,
        run_gbml=not args.no_gbml,
        run_river=not args.no_river,
        force_regenerate=args.force
    )

    print("\n✓ Comparação concluída com sucesso!")
