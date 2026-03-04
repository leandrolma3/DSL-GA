#!/usr/bin/env python3
"""
Analise Completa dos Resultados do Experimento

Consolida resultados de multiplos datasets e modelos.
Gera relatorio estatistico detalhado.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_all_results(base_dir='experiment_expanded_results'):
    """Carrega todos os arquivos comparison_table.csv e resultados ERulesD2S"""
    csv_files = glob.glob(f'{base_dir}/**/comparison_table.csv', recursive=True)

    logger.info("="*80)
    logger.info("CARREGANDO RESULTADOS")
    logger.info("="*80)
    logger.info(f"Arquivos CSV (modelos Python) encontrados: {len(csv_files)}")

    dfs = []
    datasets_loaded = []

    for csv_file in sorted(csv_files):
        # Extrair nome do dataset do caminho
        parts = csv_file.split('\\')
        if len(parts) < 2:
            parts = csv_file.split('/')
        dataset_dir = parts[-2]
        dataset_name = dataset_dir.split('_seed42')[0]

        try:
            df = pd.read_csv(csv_file)
            df['dataset'] = dataset_name
            dfs.append(df)
            datasets_loaded.append(dataset_name)
            logger.info(f"  Carregado: {dataset_name} ({len(df)} linhas)")
        except Exception as e:
            logger.error(f"  ERRO ao carregar {csv_file}: {e}")

    if not dfs:
        logger.error("Nenhum resultado carregado!")
        return None, []

    df_all = pd.concat(dfs, ignore_index=True)

    logger.info(f"\nTotal de linhas (modelos Python): {len(df_all)}")
    logger.info(f"Datasets carregados: {len(datasets_loaded)}")

    # Carregar resultados ERulesD2S se existirem
    erulesd2s_file = 'experiment_expanded_results_erulesd2s.csv'
    if Path(erulesd2s_file).exists():
        logger.info(f"\nCarregando resultados ERulesD2S: {erulesd2s_file}")
        try:
            df_erulesd2s = pd.read_csv(erulesd2s_file)

            # Renomear colunas se necessario
            if 'chunk_idx' in df_erulesd2s.columns:
                df_erulesd2s.rename(columns={'chunk_idx': 'chunk'}, inplace=True)

            # Adicionar modelo ERulesD2S
            df_erulesd2s['model'] = 'ERulesD2S'

            # Merge com resultados Python
            df_all = pd.concat([df_all, df_erulesd2s], ignore_index=True)

            logger.info(f"  ERulesD2S: {len(df_erulesd2s)} linhas adicionadas")
            logger.info(f"  Total apos merge: {len(df_all)} linhas")
            logger.info(f"  Modelos: {sorted(df_all['model'].unique())}")

        except Exception as e:
            logger.warning(f"Erro ao carregar ERulesD2S: {e}")
    else:
        logger.info(f"\nERulesD2S nao encontrado: {erulesd2s_file}")
        logger.info("  Execute: python run_erulesd2s_only.py")

    return df_all, datasets_loaded


def calculate_statistics(df):
    """Calcula estatisticas por modelo"""
    logger.info("\n" + "="*80)
    logger.info("ESTATISTICAS POR MODELO")
    logger.info("="*80)

    # Verificar colunas disponiveis
    metric_cols = ['accuracy', 'f1_weighted', 'gmean']
    available_metrics = [col for col in metric_cols if col in df.columns]

    stats_dict = {}

    for model in sorted(df['model'].unique()):
        df_model = df[df['model'] == model]

        stats = {
            'model': model,
            'n_evaluations': len(df_model),
            'n_datasets': df_model['dataset'].nunique()
        }

        for metric in available_metrics:
            stats[f'{metric}_mean'] = df_model[metric].mean()
            stats[f'{metric}_std'] = df_model[metric].std()
            stats[f'{metric}_min'] = df_model[metric].min()
            stats[f'{metric}_max'] = df_model[metric].max()

        stats_dict[model] = stats

    df_stats = pd.DataFrame(stats_dict).T

    return df_stats


def print_summary_table(df_stats):
    """Imprime tabela resumo"""
    logger.info("\nRESUMO GERAL")
    logger.info("-"*80)

    # Ordenar por G-mean medio (metrica principal)
    if 'gmean_mean' in df_stats.columns:
        df_stats = df_stats.sort_values('gmean_mean', ascending=False)

        logger.info(f"\n{'Modelo':<15} {'N':<8} {'G-mean':<20} {'Accuracy':<20} {'F1':<20}")
        logger.info("-"*80)

        for idx, row in df_stats.iterrows():
            gmean_str = f"{row['gmean_mean']:.4f} ± {row['gmean_std']:.4f}"
            acc_str = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
            f1_str = f"{row['f1_weighted_mean']:.4f} ± {row['f1_weighted_std']:.4f}"

            logger.info(f"{idx:<15} {int(row['n_evaluations']):<8} {gmean_str:<20} {acc_str:<20} {f1_str:<20}")


def calculate_per_dataset_performance(df):
    """Calcula desempenho por dataset"""
    logger.info("\n" + "="*80)
    logger.info("DESEMPENHO POR DATASET")
    logger.info("="*80)

    datasets = sorted(df['dataset'].unique())

    for dataset in datasets:
        df_ds = df[df['dataset'] == dataset]

        logger.info(f"\n{dataset}")
        logger.info("-"*60)

        # G-mean medio por modelo neste dataset
        gmean_by_model = df_ds.groupby('model')['gmean'].agg(['mean', 'std', 'count'])
        gmean_by_model = gmean_by_model.sort_values('mean', ascending=False)

        logger.info(f"{'Modelo':<15} {'G-mean medio':<20} {'N':<5}")
        for model, row in gmean_by_model.iterrows():
            logger.info(f"{model:<15} {row['mean']:.4f} ± {row['std']:.4f}     {int(row['count']):<5}")


def generate_ranking(df):
    """Gera ranking final dos modelos"""
    logger.info("\n" + "="*80)
    logger.info("RANKING FINAL (G-MEAN MEDIO)")
    logger.info("="*80)

    # Calcular G-mean medio por modelo
    ranking = df.groupby('model')['gmean'].mean().sort_values(ascending=False)

    logger.info(f"\n{'Posicao':<10} {'Modelo':<15} {'G-mean Medio':<15}")
    logger.info("-"*40)

    for pos, (model, gmean) in enumerate(ranking.items(), 1):
        logger.info(f"{pos:<10} {model:<15} {gmean:.4f}")

    return ranking


def analyze_drift_types(df):
    """Analisa desempenho por tipo de drift"""
    logger.info("\n" + "="*80)
    logger.info("ANALISE POR TIPO DE DRIFT")
    logger.info("="*80)

    # Classificar datasets por tipo de drift
    df['drift_type'] = df['dataset'].apply(lambda x: 'Abrupt' if 'Abrupt' in x else 'Gradual')

    logger.info("\nDESEMPENHO POR TIPO DE DRIFT")
    logger.info("-"*60)

    for drift_type in ['Abrupt', 'Gradual']:
        df_drift = df[df['drift_type'] == drift_type]

        logger.info(f"\n{drift_type} Drift:")
        logger.info(f"  Datasets: {df_drift['dataset'].nunique()}")
        logger.info(f"  Avaliacoes: {len(df_drift)}")

        gmean_by_model = df_drift.groupby('model')['gmean'].mean().sort_values(ascending=False)

        logger.info(f"\n  {'Modelo':<15} {'G-mean Medio':<15}")
        for model, gmean in gmean_by_model.items():
            logger.info(f"  {model:<15} {gmean:.4f}")


def analyze_by_generator(df):
    """Analisa desempenho por gerador de dados"""
    logger.info("\n" + "="*80)
    logger.info("ANALISE POR GERADOR DE DADOS")
    logger.info("="*80)

    # Extrair gerador do nome do dataset
    df['generator'] = df['dataset'].apply(lambda x: x.split('_')[0])

    generators = sorted(df['generator'].unique())

    for generator in generators:
        df_gen = df[df['generator'] == generator]

        logger.info(f"\n{generator}")
        logger.info("-"*60)
        logger.info(f"Datasets: {df_gen['dataset'].nunique()}")
        logger.info(f"Avaliacoes: {len(df_gen)}")

        gmean_by_model = df_gen.groupby('model')['gmean'].agg(['mean', 'std']).sort_values('mean', ascending=False)

        logger.info(f"\n{'Modelo':<15} {'G-mean Medio':<20}")
        for model, row in gmean_by_model.iterrows():
            logger.info(f"{model:<15} {row['mean']:.4f} ± {row['std']:.4f}")


def save_consolidated_results(df, output_file='experiment_results_consolidated.csv'):
    """Salva resultados consolidados"""
    df.to_csv(output_file, index=False)
    logger.info(f"\nResultados consolidados salvos em: {output_file}")
    logger.info(f"Total de linhas: {len(df)}")
    logger.info(f"Colunas: {', '.join(df.columns)}")


def generate_latex_table(df_stats, output_file='results_table.tex'):
    """Gera tabela LaTeX para paper"""
    logger.info("\n" + "="*80)
    logger.info("GERANDO TABELA LATEX")
    logger.info("="*80)

    # Ordenar por G-mean
    df_stats = df_stats.sort_values('gmean_mean', ascending=False)

    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Desempenho dos Modelos - Experimento Completo}")
    latex_lines.append("\\label{tab:results}")
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("\\hline")
    latex_lines.append("Modelo & N & G-mean & Accuracy & F1-score \\\\")
    latex_lines.append("\\hline")

    for idx, row in df_stats.iterrows():
        gmean = f"{row['gmean_mean']:.4f} ± {row['gmean_std']:.4f}"
        acc = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
        f1 = f"{row['f1_weighted_mean']:.4f} ± {row['f1_weighted_std']:.4f}"
        n = int(row['n_evaluations'])

        latex_lines.append(f"{idx} & {n} & {gmean} & {acc} & {f1} \\\\")

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_content = "\n".join(latex_lines)

    with open(output_file, 'w') as f:
        f.write(latex_content)

    logger.info(f"Tabela LaTeX salva em: {output_file}")

    return latex_content


def main():
    logger.info("="*80)
    logger.info("ANALISE COMPLETA - EXPERIMENTO EXPANDIDO")
    logger.info("="*80)
    logger.info("")

    # Carregar resultados
    df_all, datasets = load_all_results()

    if df_all is None:
        logger.error("Falha ao carregar resultados!")
        return 1

    # Calcular estatisticas
    df_stats = calculate_statistics(df_all)

    # Imprimir resumo
    print_summary_table(df_stats)

    # Gerar ranking
    ranking = generate_ranking(df_all)

    # Analise por dataset
    calculate_per_dataset_performance(df_all)

    # Analise por tipo de drift
    analyze_drift_types(df_all)

    # Analise por gerador
    analyze_by_generator(df_all)

    # Salvar resultados consolidados
    save_consolidated_results(df_all)

    # Gerar tabela LaTeX
    generate_latex_table(df_stats)

    logger.info("\n" + "="*80)
    logger.info("ANALISE CONCLUIDA")
    logger.info("="*80)
    logger.info("\nArquivos gerados:")
    logger.info("  - experiment_results_consolidated.csv")
    logger.info("  - results_table.tex")
    logger.info("")

    return 0


if __name__ == '__main__':
    exit(main())
