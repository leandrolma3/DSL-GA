# =============================================================================
# CELULA 4.2: Salvar Resultados Consolidados (Unified COM PENALTY)
# =============================================================================
# SUBSTITUA a CELULA 4.2 original por este codigo
# Salva resultados separados por chunk_size e penalty
# =============================================================================

import pandas as pd
from pathlib import Path
from datetime import datetime

print("="*70)
print("CELULA 4.2: SALVAR RESULTADOS CONSOLIDADOS (COM PENALTY)")
print("="*70)

# =============================================================================
# 1. Criar DataFrame com todos os resultados
# =============================================================================
df_results = pd.DataFrame(ALL_RESULTS)

print(f"\nTotal de registros: {len(df_results)}")
print(f"Colunas: {list(df_results.columns)}")

# =============================================================================
# 2. Criar diretorio de saida
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_dir = Path(WORK_DIR) / "comparison_results"
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 3. Salvar arquivo consolidado (todos os resultados)
# =============================================================================
output_file = output_dir / f"all_models_unified_results_{timestamp}.csv"
df_results.to_csv(output_file, index=False)
print(f"\n[SALVO] {output_file}")

# CSV mais recente (para facilitar uso)
output_latest = output_dir / "all_models_unified_results_latest.csv"
df_results.to_csv(output_latest, index=False)
print(f"[SALVO] {output_latest}")

# =============================================================================
# 4. Salvar arquivos por chunk_size (incluindo penalty)
# =============================================================================
print("\n" + "-"*50)
print("ARQUIVOS POR CHUNK_SIZE:")
print("-"*50)

for chunk_size_name in df_results['chunk_size'].unique():
    df_chunk = df_results[df_results['chunk_size'] == chunk_size_name]
    chunk_file = output_dir / f"all_models_{chunk_size_name}_results.csv"
    df_chunk.to_csv(chunk_file, index=False)
    print(f"[SALVO] {chunk_file.name} ({len(df_chunk)} registros)")

# =============================================================================
# 5. Estatisticas por chunk_size e penalty
# =============================================================================
print("\n" + "="*70)
print("ESTATISTICAS POR CHUNK SIZE E PENALTY")
print("="*70)

for chunk_size_name in df_results['chunk_size'].unique():
    config = CHUNK_CONFIGS.get(chunk_size_name, {})
    is_penalty = config.get('penalty', False)
    penalty_str = "(COM penalty)" if is_penalty else "(SEM penalty)"

    df_chunk = df_results[df_results['chunk_size'] == chunk_size_name]

    print(f"\n{chunk_size_name} {penalty_str}:")
    print(f"  Datasets: {df_chunk['dataset'].nunique()}")
    print(f"  Registros: {len(df_chunk)}")

    # Media por modelo
    for model in ['GBML', 'GBML_Penalty', 'CDCMS', 'ROSE_ChunkEval', 'HAT', 'ARF', 'SRP', 'ACDWM']:
        model_df = df_chunk[df_chunk['model'] == model]
        ok_df = model_df[model_df['status'].isin(['OK', 'CACHED'])]
        if len(ok_df) > 0:
            avg_gmean = ok_df['gmean'].mean()
            print(f"    {model:15s}: {avg_gmean:.4f} (n={len(ok_df)})")

# =============================================================================
# 6. Tabela comparativa: GBML vs GBML_Penalty vs Outros
# =============================================================================
print("\n" + "="*70)
print("COMPARACAO: GBML vs GBML_Penalty vs OUTROS MODELOS")
print("="*70)

# Separar por base chunk_size (sem _penalty)
base_chunk_sizes = ['chunk_500', 'chunk_1000']

for base_cs in base_chunk_sizes:
    if base_cs not in df_results['chunk_size'].values:
        continue

    print(f"\n--- {base_cs} ---")

    # GBML (sem penalty)
    gbml_df = df_results[(df_results['chunk_size'] == base_cs) & (df_results['model'] == 'GBML')]
    gbml_ok = gbml_df[gbml_df['status'].isin(['OK', 'CACHED'])]

    # GBML_Penalty (com penalty)
    penalty_cs = f"{base_cs}_penalty"
    gbml_penalty_df = df_results[(df_results['chunk_size'] == penalty_cs) & (df_results['model'] == 'GBML_Penalty')]
    gbml_penalty_ok = gbml_penalty_df[gbml_penalty_df['status'].isin(['OK', 'CACHED'])]

    # Outros modelos (usar da versao sem penalty)
    other_models = ['CDCMS', 'ROSE_ChunkEval', 'HAT', 'ARF', 'SRP', 'ACDWM']

    print(f"\n  {'Modelo':<15} | {'Media G-Mean':>12} | {'N':>4}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*4}")

    if len(gbml_ok) > 0:
        print(f"  {'GBML':<15} | {gbml_ok['gmean'].mean():>12.4f} | {len(gbml_ok):>4}")

    if len(gbml_penalty_ok) > 0:
        print(f"  {'GBML_Penalty':<15} | {gbml_penalty_ok['gmean'].mean():>12.4f} | {len(gbml_penalty_ok):>4}")

    for model in other_models:
        model_df = df_results[(df_results['chunk_size'] == base_cs) & (df_results['model'] == model)]
        ok_df = model_df[model_df['status'].isin(['OK', 'CACHED'])]
        if len(ok_df) > 0:
            print(f"  {model:<15} | {ok_df['gmean'].mean():>12.4f} | {len(ok_df):>4}")

# =============================================================================
# 7. Pivot Table para analise estatistica
# =============================================================================
print("\n" + "="*70)
print("PIVOT TABLE (para analise estatistica)")
print("="*70)

# Criar pivot: cada linha = dataset, cada coluna = modelo
# Usar dados de chunk_500 (sem penalty) para comparacao principal

df_for_pivot = df_results[df_results['status'].isin(['OK', 'CACHED', 'N/A'])]

# Pivot por chunk_size base
for base_cs in ['chunk_500', 'chunk_1000']:
    # Combinar GBML (sem penalty) e GBML_Penalty (com penalty) no mesmo pivot
    df_base = df_for_pivot[df_for_pivot['chunk_size'] == base_cs].copy()
    df_penalty = df_for_pivot[df_for_pivot['chunk_size'] == f"{base_cs}_penalty"].copy()

    if len(df_base) == 0:
        continue

    # Juntar
    df_combined = pd.concat([df_base, df_penalty], ignore_index=True)

    pivot = df_combined.pivot_table(
        values='gmean',
        index=['batch', 'dataset'],
        columns='model',
        aggfunc='mean'
    )

    print(f"\n{base_cs}:")
    print(f"  Shape: {pivot.shape}")
    print(f"  Colunas: {list(pivot.columns)}")

    # Salvar pivot
    pivot_file = output_dir / f"pivot_gmean_{base_cs}.csv"
    pivot.to_csv(pivot_file)
    print(f"  [SALVO] {pivot_file.name}")

# =============================================================================
# 8. Rankings por chunk_size
# =============================================================================
print("\n" + "="*70)
print("RANKING MEDIO DOS MODELOS")
print("="*70)

for base_cs in ['chunk_500', 'chunk_1000']:
    df_base = df_for_pivot[df_for_pivot['chunk_size'] == base_cs].copy()
    df_penalty = df_for_pivot[df_for_pivot['chunk_size'] == f"{base_cs}_penalty"].copy()

    if len(df_base) == 0:
        continue

    df_combined = pd.concat([df_base, df_penalty], ignore_index=True)

    pivot = df_combined.pivot_table(
        values='gmean',
        index=['batch', 'dataset'],
        columns='model',
        aggfunc='mean'
    )

    # Calcular ranking (maior = melhor, rank 1 = melhor)
    rankings = pivot.rank(axis=1, ascending=False)
    avg_rankings = rankings.mean().sort_values()

    print(f"\n{base_cs} - Ranking medio (menor = melhor):")
    for model, rank in avg_rankings.items():
        print(f"  {model:15s}: {rank:.2f}")

    # Salvar
    rankings_file = output_dir / f"rankings_{base_cs}.csv"
    avg_rankings.to_frame('avg_rank').to_csv(rankings_file)

# =============================================================================
# 9. Resumo final
# =============================================================================
print("\n" + "="*70)
print("ARQUIVOS GERADOS:")
print("="*70)

for f in sorted(output_dir.glob("*.csv")):
    print(f"  {f.name}")

print("\n" + "="*70)
print("PROXIMO PASSO: Executar celulas 5.x para analise estatistica")
print("="*70)
