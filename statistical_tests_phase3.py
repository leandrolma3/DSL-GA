"""
Testes estatísticos para Phase 3 - Batch 5

Executa:
- Friedman Test (diferenças significativas entre modelos)
- Wilcoxon Signed-Rank (comparações pareadas)
- Nemenyi Post-hoc (grupos homogêneos)
- Cliff's Delta (effect size)

AUTOR: Claude Code
DATA: 2025-11-23
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
import sys
import itertools

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def load_data():
    """Carrega dados consolidados."""
    rankings_file = Path("experiments_6chunks_phase3_real/batch_5/batch_5_rankings.csv")
    averages_file = Path("experiments_6chunks_phase3_real/batch_5/batch_5_model_averages.csv")

    if not rankings_file.exists():
        raise FileNotFoundError(f"Rankings file not found: {rankings_file}")
    if not averages_file.exists():
        raise FileNotFoundError(f"Averages file not found: {averages_file}")

    rankings = pd.read_csv(rankings_file, index_col=0)
    averages = pd.read_csv(averages_file)

    return rankings, averages


def friedman_test(rankings):
    """
    Friedman Test: Verifica se há diferenças significativas entre modelos.

    H0: Todos os modelos têm performance similar
    Ha: Pelo menos um modelo difere significativamente

    Args:
        rankings: DataFrame com rankings (datasets x modelos)

    Returns:
        dict com estatística, p-value, conclusão
    """
    print("\n" + "="*70)
    print("FRIEDMAN TEST")
    print("="*70)

    # Preparar dados: cada coluna é um modelo, cada linha é um dataset
    data_for_test = [rankings[col].values for col in rankings.columns]

    # Executar Friedman test
    statistic, p_value = friedmanchisquare(*data_for_test)

    # Calcular graus de liberdade
    k = len(rankings.columns)  # número de modelos
    n = len(rankings)  # número de datasets
    df = k - 1

    print(f"\nNúmero de modelos (k): {k}")
    print(f"Número de datasets (n): {n}")
    print(f"Graus de liberdade (df): {df}")
    print(f"\nEstatística do teste: {statistic:.4f}")
    print(f"P-value: {p_value:.6f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"\n✓ RESULTADO: Rejeitamos H0 (p={p_value:.6f} < {alpha})")
        print("  → Há diferenças significativas entre os modelos")
        conclusion = "significant"
    else:
        print(f"\n✗ RESULTADO: Não rejeitamos H0 (p={p_value:.6f} >= {alpha})")
        print("  → Não há evidência de diferenças significativas")
        conclusion = "not_significant"

    # Calcular ranking médio por modelo
    avg_ranks = rankings.mean(axis=0).sort_values()
    print("\nRanking médio por modelo:")
    for model, rank in avg_ranks.items():
        print(f"  {model:15s}: {rank:.2f}")

    return {
        'statistic': statistic,
        'p_value': p_value,
        'df': df,
        'conclusion': conclusion,
        'avg_ranks': avg_ranks
    }


def wilcoxon_pairwise(averages):
    """
    Wilcoxon Signed-Rank Test: Comparações pareadas entre modelos.

    Para cada par de modelos, testa se há diferença significativa
    em G-mean médio.

    Args:
        averages: DataFrame com médias de G-mean por modelo

    Returns:
        DataFrame com resultados das comparações
    """
    print("\n" + "="*70)
    print("WILCOXON SIGNED-RANK PAIRWISE TESTS")
    print("="*70)

    # Pivotar para ter modelos como colunas
    pivot = averages.pivot(index='dataset', columns='model', values='avg_gmean')

    models = pivot.columns.tolist()
    n_models = len(models)

    print(f"\nComparando {n_models} modelos:")
    print(f"  {', '.join(models)}")
    print(f"\nTotal de comparações: {n_models * (n_models - 1) // 2}")

    results = []

    # Comparações pareadas
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            # Obter valores para os dois modelos
            values1 = pivot[model1].values
            values2 = pivot[model2].values

            # Remover NaN se houver
            mask = ~(np.isnan(values1) | np.isnan(values2))
            v1 = values1[mask]
            v2 = values2[mask]

            if len(v1) < 2:
                print(f"\n⚠️ Insuficientes dados para {model1} vs {model2}")
                continue

            # Wilcoxon signed-rank test
            try:
                statistic, p_value = wilcoxon(v1, v2, alternative='two-sided')
            except ValueError as e:
                # Pode ocorrer se todos os pares forem idênticos
                print(f"\n⚠️ Erro em {model1} vs {model2}: {e}")
                statistic, p_value = np.nan, 1.0

            # Calcular diferença média
            mean_diff = (v1 - v2).mean()
            winner = model1 if mean_diff < 0 else model2  # Menor ranking é melhor

            # Cliff's Delta (effect size)
            cliff_delta = cliffs_delta(v1, v2)

            results.append({
                'model1': model1,
                'model2': model2,
                'mean_diff': mean_diff,
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'winner': winner,
                'cliff_delta': cliff_delta,
                'effect_size': interpret_cliff_delta(cliff_delta)
            })

    results_df = pd.DataFrame(results)

    # Mostrar resultados significativos
    print("\n" + "-"*70)
    print("COMPARAÇÕES SIGNIFICATIVAS (p < 0.05):")
    print("-"*70)

    significant = results_df[results_df['significant']].sort_values('p_value')

    if len(significant) == 0:
        print("\n  Nenhuma comparação significativa encontrada")
    else:
        for _, row in significant.iterrows():
            print(f"\n{row['model1']:12s} vs {row['model2']:12s}")
            print(f"  P-value: {row['p_value']:.6f}")
            print(f"  Vencedor: {row['winner']} (diff={row['mean_diff']:+.4f})")
            print(f"  Effect size: {row['effect_size']} (Cliff's δ={row['cliff_delta']:+.3f})")

    # Mostrar todas as comparações
    print("\n" + "-"*70)
    print("TODAS AS COMPARAÇÕES:")
    print("-"*70)
    print(results_df.to_string(index=False))

    return results_df


def cliffs_delta(x, y):
    """
    Calcula Cliff's Delta (effect size não-paramétrico).

    δ = (# de pares onde x < y - # de pares onde x > y) / (n_x * n_y)

    Interpretação:
    - |δ| < 0.147: negligible
    - |δ| < 0.330: small
    - |δ| < 0.474: medium
    - |δ| >= 0.474: large

    Args:
        x, y: Arrays de valores

    Returns:
        float: Cliff's delta (-1 a +1)
    """
    nx = len(x)
    ny = len(y)

    # Contar pares onde x < y e x > y
    less = sum(1 for xi in x for yi in y if xi < yi)
    greater = sum(1 for xi in x for yi in y if xi > yi)

    delta = (less - greater) / (nx * ny)

    return delta


def interpret_cliff_delta(delta):
    """
    Interpreta magnitude do Cliff's Delta.

    Args:
        delta: Cliff's delta value

    Returns:
        str: Interpretação
    """
    abs_delta = abs(delta)

    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.330:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


def nemenyi_critical_distance(k, n, alpha=0.05):
    """
    Calcula distância crítica para Nemenyi post-hoc test.

    CD = q_α * sqrt(k(k+1) / (6n))

    onde q_α é o valor crítico da distribuição studentized range.

    Args:
        k: número de modelos
        n: número de datasets
        alpha: nível de significância

    Returns:
        float: critical distance
    """
    # Valores críticos aproximados para q_α (alpha=0.05)
    q_values = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }

    q_alpha = q_values.get(k, 3.164)  # Default para k>=10

    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    return cd


def nemenyi_posthoc(rankings):
    """
    Nemenyi Post-hoc Test: Identifica grupos homogêneos de modelos.

    Modelos são significativamente diferentes se:
    |R_i - R_j| > CD

    onde R_i é o ranking médio do modelo i.

    Args:
        rankings: DataFrame com rankings

    Returns:
        dict com critical distance e grupos homogêneos
    """
    print("\n" + "="*70)
    print("NEMENYI POST-HOC TEST")
    print("="*70)

    k = len(rankings.columns)  # número de modelos
    n = len(rankings)  # número de datasets

    # Calcular ranking médio
    avg_ranks = rankings.mean(axis=0).sort_values()

    print(f"\nNúmero de modelos (k): {k}")
    print(f"Número de datasets (n): {n}")

    # Calcular critical distance
    cd = nemenyi_critical_distance(k, n, alpha=0.05)

    print(f"\nCritical Distance (CD, α=0.05): {cd:.3f}")
    print("\nRanking médio (ordenado):")
    for model, rank in avg_ranks.items():
        print(f"  {model:15s}: {rank:.3f}")

    # Identificar grupos homogêneos
    models = avg_ranks.index.tolist()
    groups = []

    print("\n" + "-"*70)
    print("COMPARAÇÕES PAREADAS:")
    print("-"*70)

    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            diff = abs(avg_ranks[model1] - avg_ranks[model2])
            significant = diff > cd

            symbol = "***" if significant else "n.s."
            print(f"{model1:12s} vs {model2:12s}: "
                  f"diff={diff:5.3f}  {'>' if significant else '<='} CD  {symbol}")

    # Criar grupos homogêneos (modelos não significativamente diferentes)
    print("\n" + "-"*70)
    print("GRUPOS HOMOGÊNEOS:")
    print("-"*70)

    # Algoritmo simples: agrupar modelos consecutivos se diff < CD
    current_group = [models[0]]
    groups = []

    for i in range(1, len(models)):
        diff = abs(avg_ranks[models[i]] - avg_ranks[models[i-1]])

        if diff <= cd:
            current_group.append(models[i])
        else:
            groups.append(current_group)
            current_group = [models[i]]

    groups.append(current_group)

    for i, group in enumerate(groups, 1):
        print(f"\nGrupo {i}: {', '.join(group)}")
        ranks_in_group = [f"{avg_ranks[m]:.3f}" for m in group]
        print(f"  Rankings: {', '.join(ranks_in_group)}")

    return {
        'critical_distance': cd,
        'avg_ranks': avg_ranks,
        'groups': groups
    }


def create_comparison_matrix(pairwise_results):
    """
    Cria matriz de comparações para visualização.

    Args:
        pairwise_results: DataFrame com resultados Wilcoxon

    Returns:
        DataFrame: matriz de p-values
    """
    print("\n" + "="*70)
    print("MATRIZ DE P-VALUES (WILCOXON)")
    print("="*70)

    # Obter lista de modelos
    models = sorted(set(pairwise_results['model1'].tolist() +
                       pairwise_results['model2'].tolist()))

    # Criar matriz
    matrix = pd.DataFrame(index=models, columns=models, dtype=float)

    # Preencher diagonal com 1.0
    for model in models:
        matrix.loc[model, model] = 1.0

    # Preencher com p-values
    for _, row in pairwise_results.iterrows():
        matrix.loc[row['model1'], row['model2']] = row['p_value']
        matrix.loc[row['model2'], row['model1']] = row['p_value']

    print("\n", matrix.round(4))

    return matrix


def main():
    """Executa todos os testes estatísticos."""
    print("="*70)
    print("TESTES ESTATÍSTICOS - PHASE 3 BATCH 5")
    print("="*70)

    # Carregar dados
    print("\nCarregando dados...")
    rankings, averages = load_data()

    print(f"\nRankings shape: {rankings.shape}")
    print(f"Datasets: {rankings.index.tolist()}")
    print(f"Modelos: {rankings.columns.tolist()}")

    # 1. Friedman Test
    friedman_results = friedman_test(rankings)

    # 2. Wilcoxon Pairwise
    pairwise_results = wilcoxon_pairwise(averages)

    # 3. Nemenyi Post-hoc
    nemenyi_results = nemenyi_posthoc(rankings)

    # 4. Matriz de comparações
    comparison_matrix = create_comparison_matrix(pairwise_results)

    # Salvar resultados
    print("\n" + "="*70)
    print("SALVANDO RESULTADOS")
    print("="*70)

    output_dir = Path("experiments_6chunks_phase3_real/batch_5")

    # Friedman
    friedman_file = output_dir / "statistical_friedman.csv"
    pd.DataFrame([friedman_results]).to_csv(friedman_file, index=False)
    print(f"\n✓ Friedman results: {friedman_file.name}")

    # Wilcoxon
    wilcoxon_file = output_dir / "statistical_wilcoxon_pairwise.csv"
    pairwise_results.to_csv(wilcoxon_file, index=False)
    print(f"✓ Wilcoxon pairwise: {wilcoxon_file.name}")

    # Nemenyi
    nemenyi_file = output_dir / "statistical_nemenyi.csv"
    nemenyi_df = pd.DataFrame({
        'model': nemenyi_results['avg_ranks'].index,
        'avg_rank': nemenyi_results['avg_ranks'].values,
        'critical_distance': nemenyi_results['critical_distance']
    })
    nemenyi_df.to_csv(nemenyi_file, index=False)
    print(f"✓ Nemenyi results: {nemenyi_file.name}")

    # Comparison matrix
    matrix_file = output_dir / "statistical_pvalue_matrix.csv"
    comparison_matrix.to_csv(matrix_file)
    print(f"✓ P-value matrix: {matrix_file.name}")

    # Sumário final
    print("\n" + "="*70)
    print("SUMÁRIO FINAL")
    print("="*70)

    print(f"\n1. Friedman Test: {friedman_results['conclusion']}")
    print(f"   P-value: {friedman_results['p_value']:.6f}")

    if friedman_results['conclusion'] == 'significant':
        print("\n2. Modelos significativamente diferentes (Wilcoxon, p<0.05):")
        significant = pairwise_results[pairwise_results['significant']]
        print(f"   {len(significant)} de {len(pairwise_results)} comparações")

        for _, row in significant.iterrows():
            print(f"   - {row['winner']} > {row['model1'] if row['winner']==row['model2'] else row['model2']}")

        print("\n3. Grupos homogêneos (Nemenyi):")
        for i, group in enumerate(nemenyi_results['groups'], 1):
            print(f"   Grupo {i}: {', '.join(group)}")

    print("\n" + "="*70)
    print("TESTES COMPLETOS!")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()
