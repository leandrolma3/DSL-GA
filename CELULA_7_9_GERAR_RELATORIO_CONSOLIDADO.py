# =============================================================================
# CELULA 7.9: Gerar Relatorio Consolidado EGIS vs CDCMS
# =============================================================================
# Gera tabela comparativa com tratamento adequado para datasets multiclasse
# =============================================================================

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

print("="*70)
print("CELULA 7.9: GERAR RELATORIO CONSOLIDADO EGIS vs CDCMS")
print("="*70)

# =============================================================================
# 1. Funcao para carregar resultados EGIS
# =============================================================================

def load_egis_results_detailed(chunk_size_name: str, batch_name: str, dataset_name: str) -> dict:
    """Carrega resultados detalhados do EGIS."""
    metrics_file = EXPERIMENTS_DIR / chunk_size_name / batch_name / dataset_name / 'run_1' / 'chunk_metrics.json'

    if not metrics_file.exists():
        return None

    with open(metrics_file) as f:
        chunk_metrics = json.load(f)

    # Calcular medias
    test_gmeans = [m.get('test_gmean') for m in chunk_metrics if m.get('test_gmean') is not None]
    test_f1s = [m.get('test_f1') for m in chunk_metrics if m.get('test_f1') is not None]

    if not test_gmeans:
        return None

    return {
        'avg_test_gmean': sum(test_gmeans) / len(test_gmeans),
        'avg_test_f1': sum(test_f1s) / len(test_f1s) if test_f1s else None,
        'num_chunks': len(chunk_metrics)
    }


# =============================================================================
# 2. Funcao para carregar resultados CDCMS
# =============================================================================

def load_cdcms_results_detailed(chunk_size_name: str, batch_name: str, dataset_name: str) -> dict:
    """Carrega resultados detalhados do CDCMS."""

    # Verificar se e multiclasse primeiro
    if dataset_name in MULTICLASS_DATASETS:
        return {
            'status': 'N/A',
            'reason': f'Multiclass ({MULTICLASS_DATASETS[dataset_name]} classes)',
            'avg_prequential_gmean': None,
            'avg_holdout_gmean': None,
            'avg_prequential_f1': None,
            'avg_holdout_f1': None
        }

    metrics_file = EXPERIMENTS_DIR / chunk_size_name / batch_name / dataset_name / 'cdcms_results' / 'chunk_metrics.json'

    if not metrics_file.exists():
        return {
            'status': 'NOT_RUN',
            'reason': 'Results not found',
            'avg_prequential_gmean': None,
            'avg_holdout_gmean': None
        }

    with open(metrics_file) as f:
        chunk_metrics = json.load(f)

    # Calcular medias
    prequential_gmeans = [m['prequential_gmean'] for m in chunk_metrics]
    holdout_gmeans = [m['holdout_gmean'] for m in chunk_metrics if m.get('holdout_gmean') is not None]

    return {
        'status': 'OK',
        'avg_prequential_gmean': sum(prequential_gmeans) / len(prequential_gmeans),
        'avg_holdout_gmean': sum(holdout_gmeans) / len(holdout_gmeans) if holdout_gmeans else None
    }


# =============================================================================
# 3. Gerar tabela comparativa completa
# =============================================================================

def generate_comparison_table(chunk_size_name: str) -> pd.DataFrame:
    """
    Gera tabela comparativa EGIS vs CDCMS para um chunk_size.
    Inclui marcacao N/A para datasets multiclasse.
    """
    print(f"\nGerando tabela comparativa para {chunk_size_name}...")

    results = []
    batches = BATCHES.get(chunk_size_name, [])

    for batch in batches:
        datasets = get_datasets_for_batch(chunk_size_name, batch)

        for dataset in datasets:
            egis = load_egis_results_detailed(chunk_size_name, batch, dataset)
            cdcms = load_cdcms_results_detailed(chunk_size_name, batch, dataset)

            row = {
                'dataset': dataset,
                'batch': batch,
                'chunk_size': chunk_size_name,
            }

            # EGIS results
            if egis:
                row['egis_gmean'] = egis['avg_test_gmean']
                row['egis_f1'] = egis.get('avg_test_f1')
            else:
                row['egis_gmean'] = None
                row['egis_f1'] = None

            # CDCMS results
            if cdcms:
                row['cdcms_status'] = cdcms.get('status', 'OK')
                row['cdcms_prequential_gmean'] = cdcms.get('avg_prequential_gmean')
                row['cdcms_holdout_gmean'] = cdcms.get('avg_holdout_gmean')
                row['cdcms_note'] = cdcms.get('reason', '')
            else:
                row['cdcms_status'] = 'NOT_RUN'
                row['cdcms_prequential_gmean'] = None
                row['cdcms_holdout_gmean'] = None
                row['cdcms_note'] = ''

            # Calcular diferenca (apenas para datasets binarios com resultados)
            if (row['cdcms_status'] == 'OK' and
                row['egis_gmean'] is not None and
                row['cdcms_holdout_gmean'] is not None):
                row['diff_holdout'] = row['cdcms_holdout_gmean'] - row['egis_gmean']
            else:
                row['diff_holdout'] = None

            results.append(row)

    df = pd.DataFrame(results)
    return df


# =============================================================================
# 4. Gerar relatorio formatado
# =============================================================================

def generate_full_report(chunk_size_name: str):
    """Gera relatorio completo com estatisticas."""

    df = generate_comparison_table(chunk_size_name)

    if df.empty:
        print("Sem dados para gerar relatorio.")
        return

    # Separar por status
    binary_ok = df[df['cdcms_status'] == 'OK']
    multiclass = df[df['cdcms_status'] == 'N/A']
    not_run = df[df['cdcms_status'] == 'NOT_RUN']

    print("\n" + "="*70)
    print(f"RELATORIO CONSOLIDADO: {chunk_size_name}")
    print("="*70)

    print(f"\nTotal de datasets: {len(df)}")
    print(f"  - Binarios (comparaveis): {len(binary_ok)}")
    print(f"  - Multiclasse (N/A):      {len(multiclass)}")
    print(f"  - Nao executados:         {len(not_run)}")

    # Estatisticas apenas para datasets binarios
    if not binary_ok.empty:
        print("\n" + "-"*50)
        print("ESTATISTICAS (apenas datasets BINARIOS)")
        print("-"*50)

        print(f"\nEGIS:")
        print(f"  Media G-Mean: {binary_ok['egis_gmean'].mean():.4f}")

        print(f"\nCDCMS:")
        print(f"  Media Prequential G-Mean: {binary_ok['cdcms_prequential_gmean'].mean():.4f}")
        if binary_ok['cdcms_holdout_gmean'].notna().any():
            print(f"  Media Holdout G-Mean:     {binary_ok['cdcms_holdout_gmean'].mean():.4f}")

        # Comparacao
        valid_diff = binary_ok['diff_holdout'].dropna()
        if not valid_diff.empty:
            print(f"\nComparacao (CDCMS Holdout - EGIS):")
            print(f"  Media:  {valid_diff.mean():+.4f}")
            print(f"  Mediana: {valid_diff.median():+.4f}")

            # Contagem de vitorias
            cdcms_wins = (valid_diff > 0).sum()
            egis_wins = (valid_diff < 0).sum()
            ties = (valid_diff == 0).sum()
            print(f"\n  CDCMS melhor: {cdcms_wins}")
            print(f"  EGIS melhor:  {egis_wins}")
            print(f"  Empates:      {ties}")

    # Listar datasets multiclasse
    if not multiclass.empty:
        print("\n" + "-"*50)
        print("DATASETS MULTICLASSE (excluidos da comparacao)")
        print("-"*50)
        for _, row in multiclass.iterrows():
            print(f"  - {row['dataset']}: {row['cdcms_note']}")

    # Salvar CSV
    output_dir = EXPERIMENTS_DIR / chunk_size_name
    output_dir.mkdir(exist_ok=True)

    # CSV completo
    csv_file = output_dir / f'comparison_egis_cdcms_{chunk_size_name}_full.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n[SALVO] {csv_file}")

    # CSV apenas binarios
    if not binary_ok.empty:
        csv_binary = output_dir / f'comparison_egis_cdcms_{chunk_size_name}_binary_only.csv'
        binary_ok.to_csv(csv_binary, index=False)
        print(f"[SALVO] {csv_binary}")

    return df


# =============================================================================
# 5. Executar
# =============================================================================

print("\nFuncoes disponiveis:")
print("  - generate_comparison_table(chunk_size_name)")
print("  - generate_full_report(chunk_size_name)")
print()
print("Para gerar relatorio:")
print("  df_500 = generate_full_report('chunk_500')")
print("  df_1000 = generate_full_report('chunk_1000')")
