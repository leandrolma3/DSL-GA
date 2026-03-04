# Análise de NaNs - Batch 1 Comparative Models

**Data**: 2025-11-19

---

## PROBLEMAS IDENTIFICADOS

### 1. **RANDOMTREE_Abrupt_Simple - SEM RESULTADOS COMPARATIVOS** 🔴

#### Sintoma
```
RANDOMTREE_Abrupt_Simple         NaN     NaN        NaN  0.6328     NaN     NaN
```

#### Causa Raiz
Na **CÉLULA 7**, a lista de datasets está INCOMPLETA:

```python
DATASETS = [
    "SEA_Abrupt_Simple",
    "AGRAWAL_Abrupt_Simple_Severe",
    "RBF_Abrupt_Severe",
    "HYPERPLANE_Abrupt_Simple",
    "STAGGER_Abrupt_Chain"
]
```

**Faltam 7 datasets do Batch 1**:
- SEA_Abrupt_Chain ❌
- SEA_Abrupt_Recurring ❌
- AGRAWAL_Abrupt_Simple_Mild ❌
- AGRAWAL_Abrupt_Chain_Long ❌
- RBF_Abrupt_Blip ❌
- STAGGER_Abrupt_Recurring ❌
- **RANDOMTREE_Abrupt_Simple** ❌

Os modelos comparativos (River, ACDWM, ERulesD2S) NÃO foram executados nesses 7 datasets!

#### Verificação
```bash
# RANDOMTREE_Abrupt_Simple só tem GBML:
ls batch_1/RANDOMTREE_Abrupt_Simple/run_1/*.csv
# Output: rule_diff_analysis_RANDOMTREE_Abrupt_Simple_matrix.csv (apenas)

# SEA_Abrupt_Simple tem todos modelos:
ls batch_1/SEA_Abrupt_Simple/run_1/*.csv
# Output: river_HAT_results.csv, river_ARF_results.csv, river_SRP_results.csv,
#         acdwm_results.csv, erulesd2s_results.csv
```

#### Solução

**Opção A - EXECUTAR MODELOS FALTANTES (IDEAL)**:

Atualizar CÉLULA 7 com lista completa:

```python
DATASETS = [
    # EXECUTADOS (5 datasets)
    "SEA_Abrupt_Simple",
    "AGRAWAL_Abrupt_Simple_Severe",
    "RBF_Abrupt_Severe",
    "HYPERPLANE_Abrupt_Simple",
    "STAGGER_Abrupt_Chain",

    # FALTANTES (7 datasets) - EXECUTAR!
    "SEA_Abrupt_Chain",
    "SEA_Abrupt_Recurring",
    "AGRAWAL_Abrupt_Simple_Mild",
    "AGRAWAL_Abrupt_Chain_Long",
    "RBF_Abrupt_Blip",
    "STAGGER_Abrupt_Recurring",
    "RANDOMTREE_Abrupt_Simple"
]
```

**Tempo estimado**: ~30 minutos (7 datasets × 2-5 min cada)

**Opção B - ACEITAR E DOCUMENTAR**:

Se não for executar os faltantes, adicionar nota na CÉLULA 11:

```python
# Após carregar resultados
print("\n[WARNING] Datasets with incomplete comparative results:")
missing_datasets = []
for dataset in DATASETS:
    dataset_dir = BASE_DIR / dataset / "run_1"
    if not (dataset_dir / "river_HAT_results.csv").exists():
        missing_datasets.append(dataset)
        print(f"  - {dataset}: No comparative models (only GBML)")

if missing_datasets:
    print(f"\n{len(missing_datasets)} datasets excluded from statistical tests")
```

---

### 2. **GBML - train_accuracy e test_accuracy como NaN** ⚠️

#### Sintoma
```
          train_accuracy          test_accuracy
                   mean     std          mean     std
model
GBML                NaN     NaN           NaN     NaN
```

#### Causa Raiz

O `chunk_metrics.json` do GBML contém apenas:
- `train_gmean` ✅
- `test_gmean` ✅
- `test_f1` ✅

**NÃO contém**:
- `train_accuracy` ❌
- `test_accuracy` ❌

Isso é esperado porque o GBML reporta G-mean (métrica mais importante para classes desbalanceadas), não accuracy.

#### Verificação
```json
// chunk_metrics.json (GBML)
[
  {
    "chunk": 0,
    "train_gmean": 0.9948,
    "test_gmean": 0.9772,
    "test_f1": 0.9790
    // NÃO TEM train_accuracy nem test_accuracy
  }
]
```

Enquanto River models têm:
```csv
chunk,model,train_gmean,train_accuracy,test_gmean,test_accuracy,...
0,HAT,0.9206,0.943,0.9129,0.941,...
```

#### Solução

**Opção A - CALCULAR accuracy do GBML** (se houver confusion matrix):

Se o GBML salva confusion matrix ou predictions, podemos calcular accuracy:
```python
# accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Opção B - DEIXAR NaN E DOCUMENTAR** (MAIS SIMPLES):

Adicionar nota explicativa na CÉLULA 11:

```python
print("\n[NOTE] GBML metrics:")
print("  - train_gmean: Available ✓")
print("  - test_gmean: Available ✓")
print("  - test_f1: Available ✓")
print("  - train_accuracy: Not reported by GBML (using G-mean instead)")
print("  - test_accuracy: Not reported by GBML (using G-mean instead)")
print("\nRationale: G-mean is more appropriate for imbalanced datasets")
```

**Opção C - CALCULAR accuracy a partir de confusion matrix** (SE DISPONÍVEL):

Verificar se há arquivo `confusion_matrices.json` ou similar.

---

### 3. **Ranking Inconsistente** 📊

#### Sintoma

**RANKING BY TEST G-MEAN** (usa todos datasets):
```
1. ACDWM           test_gmean = 0.8124  ← PRIMEIRO
2. GBML            test_gmean = 0.7872  ← SEGUNDO
```

**Final Ranking (with Statistical Significance)** (exclui datasets incompletos):
```
1. GBML            0.7872  ← PRIMEIRO
2. ACDWM           0.7447  ← SEGUNDO (média diferente!)
```

#### Causa Raiz

O teste estatístico remove datasets que não têm dados para TODOS os modelos:

```python
# Criar matriz apenas com datasets completos
for model in models_to_compare:
    for dataset in datasets_for_test:
        score = consolidated[
            (consolidated['model'] == model) &
            (consolidated['dataset'] == dataset)
        ]['test_gmean'].mean()

        if not np.isnan(score):
            model_scores.append(score)
        else:
            model_scores.append(0.0)  # ← Problema: 0 afeta média!
```

**Datasets excluídos do teste**:
- RANDOMTREE_Abrupt_Simple (só tem GBML)

Isso cria **duas médias diferentes**:
- Média geral (12 datasets): ACDWM = 0.8124
- Média para teste (11 datasets válidos): ACDWM = 0.7447

#### Solução

**CORREÇÃO**: Filtrar apenas datasets com dados completos ANTES de calcular matriz:

```python
# Identificar datasets com dados para TODOS os modelos
complete_datasets = []
for dataset in consolidated['dataset'].unique():
    has_all_models = True
    for model in models_to_compare:
        score = consolidated[
            (consolidated['model'] == model) &
            (consolidated['dataset'] == dataset)
        ]['test_gmean'].mean()

        if np.isnan(score):
            has_all_models = False
            break

    if has_all_models:
        complete_datasets.append(dataset)

print(f"\nDatasets with complete data for all models: {len(complete_datasets)}/{len(consolidated['dataset'].unique())}")
print(f"Excluded datasets: {set(consolidated['dataset'].unique()) - set(complete_datasets)}")

# Usar apenas datasets completos
datasets_for_test = complete_datasets
```

---

### 4. **Falta Conclusão Estatística Clara** 📝

#### Sintoma

Após os testes, não há conclusão clara sobre:
- GBML é estatisticamente melhor/pior/equivalente aos baselines?
- Quais modelos são estatisticamente equivalentes?

#### Solução

Adicionar ao final da CÉLULA 11:

```python
# ========================================================================
# CONCLUSÃO ESTATÍSTICA - GBML vs BASELINES
# ========================================================================

print(f"\n{'='*80}")
print("STATISTICAL CONCLUSION - GBML vs BASELINES")
print(f"{'='*80}")

# Analisar resultados de GBML
gbml_comparisons = [r for r in comparison_results if 'GBML' in [r['model1'], r['model2']]]

gbml_wins = sum(1 for r in gbml_comparisons if r['winner'] == 'GBML' and r['significance'] == 'SIGNIFICANT')
gbml_losses = sum(1 for r in gbml_comparisons if r['winner'] != 'GBML' and r['winner'] != 'TIE' and r['significance'] == 'SIGNIFICANT')
gbml_ties = sum(1 for r in gbml_comparisons if r['winner'] == 'TIE')

print(f"\nGBML Performance Summary:")
print(f"  Statistically significant wins:   {gbml_wins}/{len(gbml_comparisons)}")
print(f"  Statistically significant losses: {gbml_losses}/{len(gbml_comparisons)}")
print(f"  No significant difference (ties): {gbml_ties}/{len(gbml_comparisons)}")
print()

# Conclusão baseada em Bonferroni
if gbml_wins > 0 and gbml_losses == 0:
    print("✓ CONCLUSION: GBML is SIGNIFICANTLY BETTER than some baselines")
    print(f"  (with Bonferroni correction, α = {alpha_bonferroni:.6f})")
elif gbml_losses > 0 and gbml_wins == 0:
    print("✗ CONCLUSION: GBML is SIGNIFICANTLY WORSE than some baselines")
    print(f"  (with Bonferroni correction, α = {alpha_bonferroni:.6f})")
elif gbml_wins > 0 and gbml_losses > 0:
    print("~ CONCLUSION: GBML is MIXED - better than some, worse than others")
    print(f"  (with Bonferroni correction, α = {alpha_bonferroni:.6f})")
else:
    print("≈ CONCLUSION: GBML is STATISTICALLY EQUIVALENT to all baselines")
    print(f"  (no significant differences with Bonferroni correction, α = {alpha_bonferroni:.6f})")

print()

# Detalhar comparações de GBML
print("Detailed GBML comparisons:")
for result in gbml_comparisons:
    if result['model1'] == 'GBML':
        other_model = result['model2']
        sign = ">" if result['mean_diff'] > 0 else "<"
    else:
        other_model = result['model1']
        sign = "<" if result['mean_diff'] > 0 else ">"

    status = "SIGNIFICANT" if result['significance'] == "SIGNIFICANT" else "equivalent"
    print(f"  GBML {sign} {other_model:<10}: p={result['p_value']:.6f} [{status}]")

print(f"\n{'='*80}")
```

---

## RESUMO DAS CORREÇÕES NECESSÁRIAS

### CÉLULA 7 - Executar Modelos Faltantes

**CRÍTICO**: Adicionar 7 datasets faltantes à lista:

```python
DATASETS = [
    "SEA_Abrupt_Simple",
    "SEA_Abrupt_Chain",              # FALTANTE
    "SEA_Abrupt_Recurring",          # FALTANTE
    "AGRAWAL_Abrupt_Simple_Mild",    # FALTANTE
    "AGRAWAL_Abrupt_Simple_Severe",
    "AGRAWAL_Abrupt_Chain_Long",     # FALTANTE
    "RBF_Abrupt_Severe",
    "RBF_Abrupt_Blip",               # FALTANTE
    "STAGGER_Abrupt_Chain",
    "STAGGER_Abrupt_Recurring",      # FALTANTE
    "HYPERPLANE_Abrupt_Simple",
    "RANDOMTREE_Abrupt_Simple"       # FALTANTE
]
```

### CÉLULA 11 - Consolidação

**4 correções**:

1. **Avisar sobre datasets incompletos** (após carregar resultados)
2. **Documentar métricas ausentes do GBML** (após estatísticas)
3. **Filtrar datasets completos para testes** (antes da matriz de dados)
4. **Adicionar conclusão estatística clara** (após ranking final)

---

## IMPACTO DAS CORREÇÕES

### Antes
- ❌ 7 datasets sem resultados comparativos
- ❌ NaNs confusos nas tabelas
- ❌ Rankings inconsistentes (0.8124 vs 0.7447)
- ❌ Sem conclusão clara sobre significância

### Depois
- ✅ Todos 12 datasets com resultados completos (se executar)
- ✅ NaNs documentados e explicados
- ✅ Rankings consistentes e válidos
- ✅ Conclusão estatística clara sobre GBML

---

## AÇÃO RECOMENDADA

**OPÇÃO 1 - COMPLETA** (IDEAL para paper):
1. Executar CÉLULA 7 com 12 datasets (adicionar 7 faltantes) - ~30 min
2. Aplicar correções na CÉLULA 11
3. Re-executar CÉLULA 11 com dados completos

**OPÇÃO 2 - RÁPIDA** (se tempo limitado):
1. Aplicar apenas correções na CÉLULA 11
2. Documentar que 7 datasets não foram incluídos
3. Resultados válidos mas menos robustos

---

**Status**: ANÁLISE COMPLETA - AGUARDANDO DECISÃO
