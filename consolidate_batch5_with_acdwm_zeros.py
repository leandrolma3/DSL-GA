"""
Consolida resultados do Batch 5 adicionando ACDWM=0.0 para datasets multiclasse.

OBJETIVO:
- Adicionar linhas ACDWM para Shuttle, CovType, IntelLabSensors com G-mean=0.0
- PokerHand já tem ACDWM com G-mean=0.0 (manter como está)
- Electricity funciona corretamente (binário, manter como está)

AUTOR: Claude Code
DATA: 2025-11-23
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def add_acdwm_zeros_for_multiclass(df):
    """
    Adiciona linhas ACDWM com G-mean=0.0 para datasets multiclasse onde
    ACDWM falhou (Shuttle, CovType, IntelLabSensors).

    Args:
        df: DataFrame com resultados atuais

    Returns:
        DataFrame consolidado com linhas ACDWM adicionadas
    """
    # Datasets multiclasse que precisam de ACDWM=0.0
    multiclass_datasets = ['Shuttle', 'CovType', 'IntelLabSensors']

    # Template para linhas ACDWM
    acdwm_template = {
        'model': 'ACDWM',
        'train_gmean': 0.0,
        'train_accuracy': np.nan,
        'train_f1': np.nan,
        'test_gmean': 0.0,
        'test_accuracy': np.nan,
        'test_f1': np.nan,
        'test_f1_macro': np.nan,
        'ensemble_size': 0.0,
        'execution_time': np.nan,
        'train_chunk': np.nan,
        'test_chunk': np.nan
    }

    new_rows = []

    for dataset in multiclass_datasets:
        # Verificar se dataset existe nos resultados
        dataset_exists = dataset in df['dataset'].values

        if not dataset_exists:
            print(f"⚠️ Dataset {dataset} não encontrado nos resultados")
            continue

        # Verificar se ACDWM já existe para este dataset
        acdwm_exists = ((df['dataset'] == dataset) & (df['model'] == 'ACDWM')).any()

        if acdwm_exists:
            print(f"ℹ️ ACDWM já existe para {dataset}, pulando...")
            continue

        # Obter chunks para este dataset (baseado em outros modelos)
        dataset_data = df[df['dataset'] == dataset]
        chunks = sorted(dataset_data['chunk'].unique())

        print(f"✓ Adicionando ACDWM=0.0 para {dataset} ({len(chunks)} chunks)")

        # Criar linhas ACDWM para cada chunk
        for chunk in chunks:
            row = acdwm_template.copy()
            row['chunk'] = chunk
            row['dataset'] = dataset
            row['train_size'] = 1000
            row['test_size'] = 1000
            new_rows.append(row)

    if new_rows:
        # Adicionar novas linhas ao DataFrame
        new_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"\n✓ Adicionadas {len(new_rows)} linhas ACDWM")
    else:
        new_df = df
        print("\nℹ️ Nenhuma linha ACDWM foi adicionada")

    # Reordenar colunas para consistência
    column_order = [
        'chunk', 'train_chunk', 'test_chunk', 'model', 'train_size', 'test_size',
        'train_gmean', 'train_accuracy', 'train_f1',
        'test_gmean', 'test_accuracy', 'test_f1', 'test_f1_macro',
        'dataset', 'ensemble_size', 'execution_time'
    ]

    # Garantir que todas as colunas existam
    for col in column_order:
        if col not in new_df.columns:
            new_df[col] = np.nan

    new_df = new_df[column_order]

    # Ordenar por dataset, chunk, model
    new_df = new_df.sort_values(['dataset', 'chunk', 'model']).reset_index(drop=True)

    return new_df


def calculate_average_gmean_per_model(df):
    """
    Calcula média de G-mean por modelo por dataset.

    Args:
        df: DataFrame consolidado

    Returns:
        DataFrame com médias
    """
    # Agrupar por dataset e modelo
    summary = df.groupby(['dataset', 'model']).agg({
        'test_gmean': ['mean', 'std', 'count'],
        'test_accuracy': ['mean', 'std'],
        'test_f1': ['mean', 'std']
    }).round(4)

    # Achatar colunas multi-nível
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Renomear para clareza
    summary.rename(columns={
        'test_gmean_mean': 'avg_gmean',
        'test_gmean_std': 'std_gmean',
        'test_gmean_count': 'num_chunks',
        'test_accuracy_mean': 'avg_accuracy',
        'test_accuracy_std': 'std_accuracy',
        'test_f1_mean': 'avg_f1',
        'test_f1_std': 'std_f1'
    }, inplace=True)

    return summary


def create_ranking_table(summary_df):
    """
    Cria tabela de ranking baseada em G-mean médio.

    Args:
        summary_df: DataFrame com médias por modelo

    Returns:
        DataFrame com rankings
    """
    # Pivotar para ter modelos como colunas
    pivot = summary_df.pivot(index='dataset', columns='model', values='avg_gmean')

    # Calcular ranking (quanto maior G-mean, melhor)
    rankings = pivot.rank(axis=1, ascending=False)

    # Calcular média de ranking por modelo
    avg_rank = rankings.mean(axis=0).sort_values()

    print("\n" + "="*60)
    print("RANKING MÉDIO DOS MODELOS (Phase 3 - Batch 5)")
    print("="*60)
    print("\nPor G-mean (menor ranking = melhor):\n")
    for model, rank in avg_rank.items():
        gmean_avg = pivot[model].mean()
        print(f"{rank:5.2f}  {model:15s}  (G-mean médio: {gmean_avg:.4f})")

    return rankings, avg_rank


def main():
    """Executa consolidação completa."""
    print("="*60)
    print("CONSOLIDAÇÃO BATCH 5 - ACDWM ZEROS PARA MULTICLASSE")
    print("="*60)

    # Carregar resultados atuais
    input_file = Path("experiments_6chunks_phase3_real/batch_5/batch_1_all_models_with_gbml.csv")

    if not input_file.exists():
        print(f"❌ Arquivo não encontrado: {input_file}")
        return

    print(f"\n✓ Carregando: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Shape original: {df.shape}")
    print(f"  Modelos: {sorted(df['model'].unique())}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")

    # Verificar status atual ACDWM
    print("\n" + "-"*60)
    print("STATUS ATUAL ACDWM:")
    print("-"*60)
    for dataset in sorted(df['dataset'].unique()):
        acdwm_data = df[(df['dataset'] == dataset) & (df['model'] == 'ACDWM')]
        if len(acdwm_data) > 0:
            avg_gmean = acdwm_data['test_gmean'].mean()
            print(f"{dataset:20s}: {len(acdwm_data):2d} chunks, G-mean médio = {avg_gmean:.4f}")
        else:
            print(f"{dataset:20s}: AUSENTE ❌")

    # Adicionar ACDWM zeros
    print("\n" + "-"*60)
    print("ADICIONANDO ACDWM=0.0:")
    print("-"*60)
    df_consolidated = add_acdwm_zeros_for_multiclass(df)
    print(f"\n✓ Shape consolidado: {df_consolidated.shape}")

    # Salvar resultados consolidados
    output_file = Path("experiments_6chunks_phase3_real/batch_5/batch_5_consolidated_with_acdwm_zeros.csv")
    df_consolidated.to_csv(output_file, index=False)
    print(f"\n✓ Salvo: {output_file}")

    # Calcular médias por modelo
    print("\n" + "-"*60)
    print("CALCULANDO MÉDIAS POR MODELO:")
    print("-"*60)
    summary = calculate_average_gmean_per_model(df_consolidated)

    # Salvar resumo
    summary_file = Path("experiments_6chunks_phase3_real/batch_5/batch_5_model_averages.csv")
    summary.to_csv(summary_file, index=False)
    print(f"✓ Salvo: {summary_file}")

    # Mostrar resumo
    print("\n" + "="*60)
    print("RESUMO POR DATASET E MODELO:")
    print("="*60)
    for dataset in sorted(summary['dataset'].unique()):
        print(f"\n{dataset}:")
        dataset_summary = summary[summary['dataset'] == dataset].sort_values('avg_gmean', ascending=False)
        for _, row in dataset_summary.iterrows():
            print(f"  {row['model']:15s}: G-mean={row['avg_gmean']:.4f} ± {row['std_gmean']:.4f} "
                  f"(Acc={row['avg_accuracy']:.4f})")

    # Criar ranking
    rankings, avg_rank = create_ranking_table(summary)

    # Salvar ranking
    ranking_file = Path("experiments_6chunks_phase3_real/batch_5/batch_5_rankings.csv")
    rankings.to_csv(ranking_file)
    print(f"\n✓ Rankings salvos: {ranking_file}")

    print("\n" + "="*60)
    print("CONSOLIDAÇÃO COMPLETA!")
    print("="*60)
    print("\nArquivos gerados:")
    print(f"  1. {output_file.name}")
    print(f"  2. {summary_file.name}")
    print(f"  3. {ranking_file.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
