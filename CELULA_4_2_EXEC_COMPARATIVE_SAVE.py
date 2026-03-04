# =============================================================================
# CELULA 4.2: Salvar Resultados Consolidados (UNIFIED)
# =============================================================================
# SUBSTITUA a CELULA 4.2 original por este codigo
# Salva resultados separados por chunk_size e penalty
# Gera pivot tables e rankings para analise estatistica
# =============================================================================

import pandas as pd
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("CELULA 4.2: SALVAR RESULTADOS CONSOLIDADOS")
print("=" * 70)

# =============================================================================
# 1. CRIAR DATAFRAME COM TODOS OS RESULTADOS
# =============================================================================
df_results = pd.DataFrame(ALL_RESULTS)

print(f"\nTotal de registros: {len(df_results)}")
print(f"Colunas: {list(df_results.columns)}")

# =============================================================================
# 2. CRIAR DIRETORIO DE SAIDA
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_dir = Path(WORK_DIR) / "comparison_results"
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 3. SALVAR ARQUIVO CONSOLIDADO (TODOS OS RESULTADOS)
# =============================================================================
output_file = output_dir / f"all_models_unified_results_{timestamp}.csv"
df_results.to_csv(output_file, index=False)
print(f"\n[SALVO] {output_file}")

# CSV mais recente (para facilitar uso)
output_latest = output_dir / "all_models_unified_results_latest.csv"
df_results.to_csv(output_latest, index=False)
print(f"[SALVO] {output_latest}")

# =============================================================================
# 4. SALVAR ARQUIVOS POR EXPERIMENTO
# =============================================================================
print("\n" + "-" * 50)
print("ARQUIVOS POR EXPERIMENTO:")
print("-" * 50)

for exp_name in df_results['experiment'].unique():
    exp_df = df_results[df_results['experiment'] == exp_name]
    exp_file = output_dir / f"comparative_results_{exp_name}.csv"
    exp_df.to_csv(exp_file, index=False)
    print(f"[SALVO] {exp_file.name} ({len(exp_df)} registros)")

# =============================================================================
# 5. ESTATISTICAS POR EXPERIMENTO
# =============================================================================
print("\n" + "=" * 70)
print("ESTATISTICAS POR EXPERIMENTO")
print("=" * 70)

for exp_name in df_results['experiment'].unique():
    exp_df = df_results[df_results['experiment'] == exp_name]
    config = EXPERIMENT_CONFIGS.get(exp_name, {})
    is_penalty = config.get('penalty_weight', 0.0) > 0
    penalty_str = "(COM penalty)" if is_penalty else "(SEM penalty)"

    print(f"\n{exp_name} {penalty_str}:")
    print(f"  Datasets: {exp_df['dataset'].nunique()}")
    print(f"  Registros: {len(exp_df)}")

    # Contagem por status
    status_counts = exp_df['status'].value_counts()
    print(f"  Status: {dict(status_counts)}")

    # Media por modelo
    print(f"\n  Media G-Mean por modelo:")
    all_models = ['EGIS', 'EGIS_Penalty', 'CDCMS', 'ROSE_Original', 'ROSE_ChunkEval', 'HAT', 'ARF', 'SRP', 'ACDWM', 'ERulesD2S']
    for model in all_models:
        model_df = exp_df[exp_df['model'] == model]
        if len(model_df) > 0:
            ok_df = model_df[model_df['status'].isin(['OK', 'CACHED', 'REUSED'])]
            if len(ok_df) > 0:
                avg_gmean = ok_df['gmean'].mean()
                print(f"    {model:15s}: {avg_gmean:.4f} (n={len(ok_df)})")
            else:
                na_count = len(model_df[model_df['status'].str.contains('N/A', na=False)])
                if na_count > 0:
                    print(f"    {model:15s}: N/A ({na_count} datasets)")

# =============================================================================
# 6. TABELA COMPARATIVA: EGIS vs EGIS_Penalty vs OUTROS
# =============================================================================
print("\n" + "=" * 70)
print("COMPARACAO: EGIS vs EGIS_Penalty vs OUTROS MODELOS")
print("=" * 70)

# Agrupar por chunk_size base (sem _penalty)
chunk_sizes = ['chunk_500', 'chunk_1000']

for base_cs in chunk_sizes:
    exp_sem_penalty = f'exp_unified_{base_cs.split("_")[1]}'
    exp_com_penalty = f'{exp_sem_penalty}_penalty'

    if exp_sem_penalty not in df_results['experiment'].values:
        continue

    print(f"\n--- {base_cs} ---")

    # EGIS (sem penalty)
    egis_df = df_results[
        (df_results['experiment'] == exp_sem_penalty) &
        (df_results['model'] == 'EGIS')
    ]
    egis_ok = egis_df[egis_df['status'].isin(['OK', 'CACHED'])]

    # EGIS_Penalty (com penalty)
    egis_penalty_df = df_results[
        (df_results['experiment'] == exp_com_penalty) &
        (df_results['model'] == 'EGIS_Penalty')
    ]
    egis_penalty_ok = egis_penalty_df[egis_penalty_df['status'].isin(['OK', 'CACHED'])]

    # Outros modelos (da versao sem penalty)
    other_models = ['CDCMS', 'ROSE_Original', 'ROSE_ChunkEval', 'HAT', 'ARF', 'SRP', 'ACDWM']

    print(f"\n  {'Modelo':<15} | {'Media G-Mean':>12} | {'N':>4}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*4}")

    if len(egis_ok) > 0:
        print(f"  {'EGIS':<15} | {egis_ok['gmean'].mean():>12.4f} | {len(egis_ok):>4}")

    if len(egis_penalty_ok) > 0:
        print(f"  {'EGIS_Penalty':<15} | {egis_penalty_ok['gmean'].mean():>12.4f} | {len(egis_penalty_ok):>4}")

    for model in other_models:
        model_df = df_results[
            (df_results['experiment'] == exp_sem_penalty) &
            (df_results['model'] == model)
        ]
        ok_df = model_df[model_df['status'].isin(['OK', 'CACHED', 'REUSED'])]
        if len(ok_df) > 0:
            print(f"  {model:<15} | {ok_df['gmean'].mean():>12.4f} | {len(ok_df):>4}")

# =============================================================================
# 7. PIVOT TABLE PARA ANALISE ESTATISTICA
# =============================================================================
print("\n" + "=" * 70)
print("PIVOT TABLE (para analise estatistica)")
print("=" * 70)

# Criar pivot: cada linha = dataset, cada coluna = modelo
df_for_pivot = df_results[df_results['status'].isin(['OK', 'CACHED', 'REUSED', 'N/A', 'N/A (multiclass)'])]

for base_cs in ['500', '1000']:
    exp_sem_penalty = f'exp_unified_{base_cs}'
    exp_com_penalty = f'{exp_sem_penalty}_penalty'

    # Combinar EGIS (sem penalty) e EGIS_Penalty (com penalty)
    df_base = df_for_pivot[df_for_pivot['experiment'] == exp_sem_penalty].copy()
    df_penalty = df_for_pivot[df_for_pivot['experiment'] == exp_com_penalty].copy()

    if len(df_base) == 0:
        continue

    # Juntar
    df_combined = pd.concat([df_base, df_penalty], ignore_index=True)

    # Substituir N/A por NaN
    df_combined.loc[df_combined['status'].str.contains('N/A', na=False), 'gmean'] = np.nan

    pivot = df_combined.pivot_table(
        values='gmean',
        index=['batch', 'dataset'],
        columns='model',
        aggfunc='mean'
    )

    print(f"\nchunk_{base_cs}:")
    print(f"  Shape: {pivot.shape}")
    print(f"  Colunas: {list(pivot.columns)}")

    # Salvar pivot
    pivot_file = output_dir / f"pivot_gmean_chunk_{base_cs}.csv"
    pivot.to_csv(pivot_file)
    print(f"  [SALVO] {pivot_file.name}")

# =============================================================================
# 8. RANKINGS POR CHUNK SIZE
# =============================================================================
print("\n" + "=" * 70)
print("RANKING MEDIO DOS MODELOS")
print("=" * 70)

for base_cs in ['500', '1000']:
    exp_sem_penalty = f'exp_unified_{base_cs}'
    exp_com_penalty = f'{exp_sem_penalty}_penalty'

    df_base = df_for_pivot[df_for_pivot['experiment'] == exp_sem_penalty].copy()
    df_penalty = df_for_pivot[df_for_pivot['experiment'] == exp_com_penalty].copy()

    if len(df_base) == 0:
        continue

    df_combined = pd.concat([df_base, df_penalty], ignore_index=True)
    df_combined.loc[df_combined['status'].str.contains('N/A', na=False), 'gmean'] = np.nan

    pivot = df_combined.pivot_table(
        values='gmean',
        index=['batch', 'dataset'],
        columns='model',
        aggfunc='mean'
    )

    # Calcular ranking (maior = melhor, rank 1 = melhor)
    rankings = pivot.rank(axis=1, ascending=False, na_option='bottom')
    avg_rankings = rankings.mean().sort_values()

    print(f"\nchunk_{base_cs} - Ranking medio (menor = melhor):")
    for model, rank in avg_rankings.items():
        print(f"  {model:15s}: {rank:.2f}")

    # Salvar
    rankings_file = output_dir / f"rankings_chunk_{base_cs}.csv"
    avg_rankings.to_frame('avg_rank').to_csv(rankings_file)

# =============================================================================
# 9. RESUMO FINAL
# =============================================================================
print("\n" + "=" * 70)
print("ARQUIVOS GERADOS:")
print("=" * 70)

for f in sorted(output_dir.glob("*.csv")):
    print(f"  {f.name}")

print("\n" + "=" * 70)
print("PROXIMO PASSO:")
print("  - Executar celulas 5.x para sincronizar com Drive")
print("  - Executar unified_analysis.py para gerar tabelas e figuras")
print("=" * 70)
