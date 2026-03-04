# Correções - Batch_1_Comparative_Models.ipynb

**Data**: 2025-11-19
**Problemas Identificados**: 3

---

## PROBLEMA 1: Erro ao Ler rule_diff_analysis_*_matrix.csv

### Descrição
```
! Error reading rule_diff_analysis_SEA_Abrupt_Simple_matrix.csv: 'model'
```

**Causa**: O código na CÉLULA 11 está tentando ler TODOS os arquivos CSV, incluindo `rule_diff_analysis_*_matrix.csv` que é gerado pelo pos-processamento e contém matriz de diferenças entre regras (não resultados de modelos).

### Correção - CÉLULA 11 (linha ~37)

**ANTES**:
```python
# Listar todos os CSVs de resultados (exceto chunks)
result_files = [
    f for f in dataset_dir.glob("*.csv")
    if "chunk_" not in f.name and "comparative" not in f.name
]
```

**DEPOIS**:
```python
# Listar todos os CSVs de resultados (exceto chunks e rule_diff_analysis)
result_files = [
    f for f in dataset_dir.glob("*.csv")
    if "chunk_" not in f.name
    and "comparative" not in f.name
    and "rule_diff" not in f.name  # Ignorar rule_diff_analysis
    and "_matrix" not in f.name     # Ignorar matrizes
]
```

---

## PROBLEMA 2: ERulesD2S com 6 Chunks (deveria ter 5)

### Descrição
```
WARNING: Filtered ERulesD2S to chunks 0-4 (removed extra chunk)
```

**Causa**: ERulesD2S usa numeração 1-6, que após ajuste (-1) vira 0-5 (6 chunks). Deveria ter apenas 0-4 (5 chunks para treino).

### Correção - CÉLULA 11 (linha ~50)

**ANTES**:
```python
# CORRECAO PARA ERULESD2S: Converter formato
if 'ERulesD2S' in csv_file.name or (df['model'] == 'ERulesD2S').any():
    print(f"  - {csv_file.name}: {len(df)} linhas (ERulesD2S - convertendo formato)")

    # ERulesD2S so tem test metrics (sem train)
    # Renomear colunas para padronizar
    if 'accuracy' in df.columns and 'test_accuracy' not in df.columns:
        df.rename(columns={
            'accuracy': 'test_accuracy',
            'gmean': 'test_gmean',
            'f1_weighted': 'test_f1'
        }, inplace=True)

        # Adicionar colunas train vazias (ERulesD2S nao tem)
        df['train_gmean'] = np.nan
        df['train_accuracy'] = np.nan
        df['train_f1'] = np.nan

        # Ajustar numeracao de chunks (ERulesD2S usa 1-6, outros usam 0-4)
        if 'chunk' in df.columns:
            df['chunk'] = df['chunk'] - 1  # 1-6 -> 0-5
```

**DEPOIS**:
```python
# CORRECAO PARA ERULESD2S: Converter formato
if 'ERulesD2S' in csv_file.name or 'erulesd2s' in csv_file.name.lower():
    original_len = len(df)

    # ERulesD2S so tem test metrics (sem train)
    # Renomear colunas para padronizar
    if 'accuracy' in df.columns and 'test_accuracy' not in df.columns:
        df.rename(columns={
            'accuracy': 'test_accuracy',
            'gmean': 'test_gmean',
            'f1_weighted': 'test_f1'
        }, inplace=True)

        # Adicionar colunas train vazias (ERulesD2S nao tem)
        df['train_gmean'] = np.nan
        df['train_accuracy'] = np.nan
        df['train_f1'] = np.nan

        # Ajustar numeracao de chunks (ERulesD2S usa 1-6, outros usam 0-4)
        if 'chunk' in df.columns:
            df['chunk'] = df['chunk'] - 1  # 1-6 -> 0-5

            # NOVO: Filtrar apenas chunks 0-4 (treino)
            df = df[df['chunk'] <= 4].copy()

    print(f"  - {csv_file.name}: {len(df)} rows (ERulesD2S - converted format)", end="")
    if original_len > len(df):
        print(f" [filtered {original_len} -> {len(df)} chunks]")
    else:
        print()
```

---

## PROBLEMA 3: Testes Estatísticos - Falta Bonferroni e Clareza

### Descrição

**Problemas atuais**:
1. Apenas compara GBML vs cada modelo (não todas comparações pairwise)
2. Não aplica correção de Bonferroni para múltiplas comparações
3. Não indica claramente quem é estatisticamente melhor/pior
4. Não mostra ranking com significância

### Correção - CÉLULA 11 (Substituir seção de testes estatísticos)

**Localização**: Após o pivot_table (linha ~170), SUBSTITUIR toda seção de testes estatísticos.

**CÓDIGO COMPLETO CORRIGIDO**:

```python
# ========================================================================
# TESTES ESTATISTICOS (COMPLETOS COM BONFERRONI)
# ========================================================================

print(f"\n{'='*80}")
print("TESTES ESTATISTICOS (on Test G-mean)")
print('='*80)
print("Reference: Demsar (2006), Garcia & Herrera (2008)")
print("Using TEST metrics to evaluate generalization performance")

# Preparar dados para testes (matriz: datasets x modelos)
models_to_compare = ['GBML', 'ACDWM', 'ARF', 'SRP', 'HAT']

# Filtrar apenas modelos que temos dados e que tem test_gmean
models_available = []
for model in models_to_compare:
    model_data = consolidated[
        (consolidated['model'] == model) &
        (consolidated['test_gmean'].notna())
    ]
    if len(model_data) > 0:
        models_available.append(model)

models_to_compare = models_available

# Criar matriz de test_gmean por dataset e modelo
datasets_for_test = consolidated['dataset'].unique()
data_matrix = []

for model in models_to_compare:
    model_scores = []
    for dataset in datasets_for_test:
        score = consolidated[
            (consolidated['model'] == model) &
            (consolidated['dataset'] == dataset)
        ]['test_gmean'].mean()

        if not np.isnan(score):
            model_scores.append(score)
        else:
            model_scores.append(0.0)  # Placeholder para datasets sem dados

    data_matrix.append(model_scores)

data_matrix = np.array(data_matrix).T  # Transpor: datasets x modelos

print(f"\nData matrix (test_gmean):")
print(f"Datasets: {len(data_matrix)} | Models: {len(models_to_compare)}")
print(f"Models compared: {models_to_compare}")

# ============================================================================
# FRIEDMAN TEST
# ============================================================================

print(f"\n--- Friedman Test ---")
print("H0: All models have equivalent performance")
print("H1: At least one model differs from the others")

try:
    stat, p_value = friedmanchisquare(*data_matrix.T)
    print(f"\nFriedman statistic: {stat:.4f}")
    print(f"p-value: {p_value:.6f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"Result: REJECT H0 (p < {alpha})")
        print("Conclusion: Significant difference exists between models")
    else:
        print(f"Result: FAIL TO REJECT H0 (p >= {alpha})")
        print("Conclusion: No significant difference between models")
except Exception as e:
    print(f"Error in Friedman test: {e}")

# ============================================================================
# WILCOXON SIGNED-RANK TEST (ALL PAIRWISE COMPARISONS)
# ============================================================================

print(f"\n--- Wilcoxon Signed-Rank Tests (All Pairwise Comparisons) ---")

# Número de comparações para Bonferroni
num_comparisons = len(models_to_compare) * (len(models_to_compare) - 1) // 2
alpha_bonferroni = 0.05 / num_comparisons

print(f"Total pairwise comparisons: {num_comparisons}")
print(f"Alpha (original): 0.05")
print(f"Alpha (Bonferroni corrected): {alpha_bonferroni:.6f}")
print()

# Resultados das comparações
comparison_results = []

for i in range(len(models_to_compare)):
    for j in range(i+1, len(models_to_compare)):
        model1 = models_to_compare[i]
        model2 = models_to_compare[j]

        scores1 = data_matrix[:, i]
        scores2 = data_matrix[:, j]

        try:
            # Wilcoxon test (two-sided)
            stat, p_value = wilcoxon(scores1, scores2, alternative='two-sided')

            # Calcular diferença média
            mean_diff = np.mean(scores1 - scores2)

            # Determinar vencedor
            if p_value < alpha_bonferroni:
                if mean_diff > 0:
                    winner = model1
                    significance = "SIGNIFICANT"
                else:
                    winner = model2
                    significance = "SIGNIFICANT"
            else:
                winner = "TIE"
                significance = "not significant"

            comparison_results.append({
                'model1': model1,
                'model2': model2,
                'p_value': p_value,
                'mean_diff': mean_diff,
                'winner': winner,
                'significance': significance
            })

        except Exception as e:
            comparison_results.append({
                'model1': model1,
                'model2': model2,
                'p_value': np.nan,
                'mean_diff': np.nan,
                'winner': 'ERROR',
                'significance': str(e)
            })

# Ordenar por p-value (menor primeiro)
comparison_results.sort(key=lambda x: x['p_value'] if not np.isnan(x['p_value']) else 1.0)

# Imprimir resultados
print(f"{'Model 1':<15} vs {'Model 2':<15} | {'p-value':>10} | {'Mean Diff':>10} | {'Winner':<15} | Significance")
print("-" * 95)

for result in comparison_results:
    m1 = result['model1']
    m2 = result['model2']
    p = result['p_value']
    diff = result['mean_diff']
    winner = result['winner']
    sig = result['significance']

    if not np.isnan(p):
        sig_marker = " ***" if sig == "SIGNIFICANT" else ""
        print(f"{m1:<15} vs {m2:<15} | {p:>10.6f} | {diff:>+10.4f} | {winner:<15} | {sig}{sig_marker}")
    else:
        print(f"{m1:<15} vs {m2:<15} | {'ERROR':>10} | {'N/A':>10} | {winner:<15} | {sig}")

# ============================================================================
# RANKING FINAL COM SIGNIFICÂNCIA
# ============================================================================

print(f"\n--- Final Ranking (with Statistical Significance) ---")
print()

# Calcular média e ranking
model_means = {}
for idx, model in enumerate(models_to_compare):
    model_means[model] = np.mean(data_matrix[:, idx])

# Ordenar por média (maior = melhor)
ranking = sorted(model_means.items(), key=lambda x: x[1], reverse=True)

# Para cada modelo, contar vitórias e empates
model_stats = {model: {'wins': 0, 'ties': 0, 'losses': 0} for model in models_to_compare}

for result in comparison_results:
    if result['significance'] == 'SIGNIFICANT':
        winner = result['winner']
        loser = result['model1'] if winner == result['model2'] else result['model2']
        model_stats[winner]['wins'] += 1
        model_stats[loser]['losses'] += 1
    else:
        model_stats[result['model1']]['ties'] += 1
        model_stats[result['model2']]['ties'] += 1

print(f"{'Rank':<6} {'Model':<15} {'Mean Test G-mean':<20} {'Wins':<6} {'Ties':<6} {'Losses':<8}")
print("-" * 70)

for rank, (model, mean_score) in enumerate(ranking, 1):
    stats = model_stats[model]
    print(f"{rank:<6} {model:<15} {mean_score:<20.4f} {stats['wins']:<6} {stats['ties']:<6} {stats['losses']:<8}")

print()
print("Legend:")
print("  - Wins: Number of statistically significant victories (with Bonferroni correction)")
print("  - Ties: Number of non-significant comparisons")
print("  - Losses: Number of statistically significant defeats")

print(f"\n{'='*80}")
```

---

## RESUMO DAS CORREÇÕES

### CÉLULA 11: Consolidar Resultados

**3 correções necessárias**:

1. **Linha ~37**: Adicionar filtros para ignorar `rule_diff` e `_matrix`
2. **Linha ~50**: Filtrar ERulesD2S para apenas chunks 0-4
3. **Linha ~170**: Substituir seção de testes estatísticos completa

### Arquivos Afetados

- `Batch_1_Comparative_Models.ipynb` - CÉLULA 11

### Impacto

- ✅ Remove erro "'model'"
- ✅ Corrige WARNING de ERulesD2S (mantém apenas chunks de treino)
- ✅ Adiciona Bonferroni correction
- ✅ Mostra todas comparações pairwise
- ✅ Indica claramente vencedores/perdedores
- ✅ Ranking final com estatísticas completas

---

## PRÓXIMOS PASSOS

1. Aplicar correções na CÉLULA 11 do notebook
2. Re-executar a CÉLULA 11 no Colab
3. Verificar que não há mais erros
4. Validar testes estatísticos com Bonferroni
5. Gerar relatório final com resultados corrigidos

---

**Status**: CORREÇÕES PRONTAS PARA APLICAR
