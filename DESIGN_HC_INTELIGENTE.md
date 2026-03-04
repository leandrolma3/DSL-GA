# DESIGN: HILL CLIMBING INTELIGENTE (FASE 2)

## Data: 2025-10-13

---

## 1. DIAGNÓSTICO DO PROBLEMA ATUAL

### Bug Identificado no `dt_error_correction()`

**Localização**: `hill_climbing_v2.py` (linhas ~226-336)

**O que faz atualmente:**
```python
def dt_error_correction(elite, X_train, y_train, ...):
    # 1. Identifica erros do elite
    y_pred_elite = elite.predict(X_train)
    error_mask = (y_pred_elite != y_train)
    X_errors = X_train[error_mask]
    y_errors = y_train[error_mask]

    # 2. Treina DT nos erros ✅ CORRETO
    dt_error = DecisionTreeClassifier(max_depth=...).fit(X_errors, y_errors)

    # 3. ❌ BUG: Ignora a DT e injeta regras ALEATÓRIAS da população
    variants = [inject_random_population_rules(elite) for _ in range(n_variants)]
    return variants
```

**Resultado**: 8.3% taxa de aprovação (variantes aleatórias raramente são melhores que elite)

---

## 2. SOLUÇÃO PROPOSTA: HC INTELIGENTE MULTI-ESTRATÉGIA

### Visão Geral

Ao invés de **uma única estratégia** (DT error correction), vamos criar **múltiplas estratégias inteligentes** que operam em camadas:

```
┌─────────────────────────────────────────────────────────────┐
│         HILL CLIMBING INTELIGENTE (3 ESTRATÉGIAS)          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 1. ERROR-FOCUSED DT RULES (40% variantes)          │  │
│  │    - Extrai regras DT via root→leaf paths          │  │
│  │    - Substitui 20% piores regras do elite          │  │
│  │    - Foco: Corrigir erros específicos              │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 2. ENSEMBLE BOOSTING (30% variantes)               │  │
│  │    - Treina DT com pesos nos erros (boosting)      │  │
│  │    - Combina regras DT + regras elite (híbrido)    │  │
│  │    - Foco: Aumentar cobertura em regiões difíceis  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 3. GUIDED MUTATION (30% variantes)                 │  │
│  │    - Analisa features importantes da DT            │  │
│  │    - Muta regras do elite focando nessas features  │  │
│  │    - Foco: Refinar regras existentes com insight DT│  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. ESTRATÉGIA 1: ERROR-FOCUSED DT RULES

### Objetivo
Extrair regras da DT que foi treinada nos erros e substituir as piores regras do elite

### Algoritmo Detalhado

```python
def error_focused_dt_rules(elite, X_train, y_train, n_variants=5):
    """
    Extrai regras DT dos erros e substitui piores regras do elite

    Processo:
    1. Identifica erros do elite (regiões onde ele falha)
    2. Treina DT profunda nos erros (max_depth=8-12)
    3. Extrai TODAS as regras DT via root→leaf paths
    4. Ranqueia regras DT por cobertura de erros
    5. Substitui 20-30% piores regras do elite por regras DT top
    """

    # 1. Identificar erros do elite
    y_pred = elite.predict(X_train)
    error_mask = (y_pred != y_train)
    X_errors = X_train[error_mask]
    y_errors = y_train[error_mask]

    if len(X_errors) < 10:  # Elite quase perfeito
        return []

    # 2. Treinar DT profunda nos erros
    dt_error = DecisionTreeClassifier(
        max_depth=10,  # Profunda para capturar padrões complexos
        min_samples_leaf=5,  # Evitar overfitting
        class_weight='balanced'  # Balancear classes nos erros
    ).fit(X_errors, y_errors)

    # 3. Extrair regras DT via root→leaf paths
    dt_rules = extract_rules_from_tree(dt_error, X_train.columns, y_errors.unique())

    # 4. Ranquear regras DT por cobertura de erros
    dt_rules_ranked = rank_rules_by_error_coverage(dt_rules, X_errors, y_errors)

    # 5. Para cada variante, substituir piores regras do elite
    variants = []
    for i in range(n_variants):
        variant = elite.clone()

        # Identificar 20-30% piores regras do elite
        elite_rules_ranked = rank_rules_by_fitness(variant, X_train, y_train)
        n_replace = int(len(elite_rules_ranked) * 0.25)  # 25%
        worst_rules_idx = elite_rules_ranked[-n_replace:]

        # Substituir por top DT rules
        for idx in worst_rules_idx:
            if dt_rules_ranked:
                new_rule = dt_rules_ranked.pop(0)  # Pega top DT rule
                variant.replace_rule(idx, new_rule)

        variants.append(variant)

    return variants
```

### Extração de Regras DT (Root→Leaf Paths)

```python
def extract_rules_from_tree(dt_model, feature_names, classes):
    """
    Extrai regras de uma Decision Tree como paths root→leaf

    Exemplo de regra extraída:
    IF (feature_3 <= 0.45) AND (feature_7 > 0.82) THEN class=1
    """
    from sklearn.tree import _tree

    tree = dt_model.tree_
    rules = []

    def recurse(node, conditions):
        # Folha: cria regra completa
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            class_id = np.argmax(tree.value[node])
            confidence = tree.value[node][0][class_id] / tree.value[node][0].sum()

            rule = Rule(
                conditions=conditions.copy(),
                predicted_class=classes[class_id],
                confidence=confidence
            )
            rules.append(rule)
            return

        # Nó interno: bifurca
        feature_id = tree.feature[node]
        threshold = tree.threshold[node]
        feature_name = feature_names[feature_id]

        # Left: feature <= threshold
        left_conditions = conditions.copy()
        left_conditions.append((feature_name, '<=', threshold))
        recurse(tree.children_left[node], left_conditions)

        # Right: feature > threshold
        right_conditions = conditions.copy()
        right_conditions.append((feature_name, '>', threshold))
        recurse(tree.children_right[node], right_conditions)

    recurse(0, [])
    return rules
```

### Ranqueamento de Regras por Cobertura de Erros

```python
def rank_rules_by_error_coverage(rules, X_errors, y_errors):
    """
    Ranqueia regras DT por quantos erros elas cobrem corretamente

    Métrica: coverage × accuracy
    - coverage: % de erros que a regra ativa
    - accuracy: % de acertos quando regra ativa
    """
    ranked = []

    for rule in rules:
        # Aplicar regra nos erros
        mask = rule.match(X_errors)

        if mask.sum() == 0:
            continue

        # Calcular coverage e accuracy
        coverage = mask.sum() / len(X_errors)
        predictions = rule.predict(X_errors[mask])
        accuracy = (predictions == y_errors[mask]).mean()

        # Score combinado
        score = coverage * accuracy

        ranked.append((score, rule))

    # Ordenar decrescente por score
    ranked.sort(key=lambda x: x[0], reverse=True)

    return [rule for score, rule in ranked]
```

---

## 4. ESTRATÉGIA 2: ENSEMBLE BOOSTING

### Objetivo
Criar híbridos elite+DT onde DT foca em corrigir erros do elite com pesos boosting

### Algoritmo

```python
def ensemble_boosting(elite, X_train, y_train, n_variants=4):
    """
    Cria híbridos elite+DT com boosting nos erros

    Conceito: DT treina com PESOS nos erros do elite
    - Erros do elite: peso 3.0
    - Acertos do elite: peso 1.0

    DT aprende a "focar" nas regiões difíceis
    """

    # 1. Calcular pesos baseados em erros do elite
    y_pred = elite.predict(X_train)
    sample_weights = np.ones(len(X_train))
    sample_weights[y_pred != y_train] = 3.0  # Erros têm peso 3×

    variants = []

    for i in range(n_variants):
        # 2. Treinar DT com pesos (boosting manual)
        dt_boosted = DecisionTreeClassifier(
            max_depth=8 + i,  # Varia profundidade
            min_samples_leaf=10,
            class_weight='balanced'
        ).fit(X_train, y_train, sample_weight=sample_weights)

        # 3. Extrair regras DT boosted
        dt_rules = extract_rules_from_tree(dt_boosted, X_train.columns, y_train.unique())

        # 4. Criar híbrido: 70% elite + 30% DT boosted
        variant = elite.clone()

        # Substituir 30% regras aleatórias do elite por DT boosted
        n_replace = int(len(variant.rules) * 0.3)
        replace_idx = np.random.choice(len(variant.rules), n_replace, replace=False)

        for idx in replace_idx:
            if dt_rules:
                variant.replace_rule(idx, dt_rules.pop(0))

        variants.append(variant)

    return variants
```

---

## 5. ESTRATÉGIA 3: GUIDED MUTATION

### Objetivo
Mutar regras do elite focando nas features importantes identificadas pela DT

### Algoritmo

```python
def guided_mutation(elite, X_train, y_train, n_variants=4):
    """
    Muta regras do elite guiado por importância de features da DT

    Conceito: DT revela quais features são importantes para corrigir erros
    Mutação foca nessas features ao invés de mutar aleatoriamente
    """

    # 1. Treinar DT nos erros
    y_pred = elite.predict(X_train)
    error_mask = (y_pred != y_train)
    X_errors = X_train[error_mask]
    y_errors = y_train[error_mask]

    if len(X_errors) < 10:
        return []

    dt_error = DecisionTreeClassifier(max_depth=8).fit(X_errors, y_errors)

    # 2. Extrair importância de features
    feature_importance = dt_error.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-5:]  # Top 5 features
    top_features = X_train.columns[top_features_idx].tolist()

    variants = []

    for i in range(n_variants):
        variant = elite.clone()

        # 3. Mutar 40% das regras do variant
        n_mutate = int(len(variant.rules) * 0.4)
        mutate_idx = np.random.choice(len(variant.rules), n_mutate, replace=False)

        for idx in mutate_idx:
            rule = variant.rules[idx]

            # 4. Mutação guiada: foca nas top features
            if np.random.random() < 0.7:  # 70% chance de mutação guiada
                # Mutar condição que usa top feature
                mutate_condition_guided(rule, top_features, X_train)
            else:  # 30% mutação aleatória (diversidade)
                mutate_condition_random(rule, X_train)

        variants.append(variant)

    return variants


def mutate_condition_guided(rule, top_features, X_train):
    """
    Muta uma condição da regra focando em top features
    """
    # Escolher top feature aleatória
    feature = np.random.choice(top_features)

    # Escolher threshold baseado em percentis dos dados
    percentile = np.random.choice([25, 50, 75])
    threshold = np.percentile(X_train[feature], percentile)

    # Escolher operador
    operator = np.random.choice(['<=', '>'])

    # Substituir ou adicionar condição
    if len(rule.conditions) > 0:
        # Substituir condição aleatória
        idx = np.random.randint(len(rule.conditions))
        rule.conditions[idx] = (feature, operator, threshold)
    else:
        # Adicionar nova condição
        rule.conditions.append((feature, operator, threshold))
```

---

## 6. INTEGRAÇÃO NO SISTEMA HIERÁRQUICO

### Modificar `hill_climbing_v2.py`

```python
def generate_hierarchical_variants(elite, level, X_train, y_train, best_memory, ...):
    """
    Sistema hierárquico MODIFICADO com HC Inteligente
    """

    if level == 'AGGRESSIVE':
        # 40% Error-Focused DT
        # 30% Ensemble Boosting
        # 30% Guided Mutation

        variants_dt = error_focused_dt_rules(elite, X_train, y_train, n_variants=6)
        variants_boost = ensemble_boosting(elite, X_train, y_train, n_variants=4)
        variants_guided = guided_mutation(elite, X_train, y_train, n_variants=4)

        all_variants = variants_dt + variants_boost + variants_guided

    elif level == 'MODERATE':
        # 50% Error-Focused DT
        # 30% Ensemble Boosting
        # 20% Operações legadas (memory injection)

        variants_dt = error_focused_dt_rules(elite, X_train, y_train, n_variants=6)
        variants_boost = ensemble_boosting(elite, X_train, y_train, n_variants=4)
        variants_legacy = [inject_memory_rules(elite, best_memory) for _ in range(2)]

        all_variants = variants_dt + variants_boost + variants_legacy

    elif level == 'FINE_TUNING':
        # 60% Guided Mutation (refinamento fino)
        # 40% Error-Focused DT (pequenas correções)

        variants_guided = guided_mutation(elite, X_train, y_train, n_variants=7)
        variants_dt = error_focused_dt_rules(elite, X_train, y_train, n_variants=5)

        all_variants = variants_guided + variants_dt

    return all_variants
```

---

## 7. EXPECTATIVAS DE MELHORIA

### Métricas Alvo

| Métrica | Atual (HC v2) | Meta (HC Inteligente) | Melhoria |
|---------|---------------|----------------------|----------|
| **Taxa de Aprovação** | 8.3% | 30-50% | 3.6-6× |
| **G-mean por Ativação** | +0.01-0.02% | +0.3-0.8% | 15-40× |
| **G-mean Elite Final** | 82.87% | 85-88% | +2.1-5.1% |
| **Ativações HC Necessárias** | 2 (pouco impacto) | 1-2 (alto impacto) | - |

### Projeção de Impacto

**Cenário Conservador** (Taxa 30%, +0.3% por ativação):
- 2 ativações HC × 0.3% = +0.6% G-mean
- Elite: 82.87% → 83.47%

**Cenário Otimista** (Taxa 50%, +0.8% por ativação):
- 2 ativações HC × 0.8% = +1.6% G-mean
- Elite: 82.87% → 84.47%

**Cenário Ideal** (Taxa 50%, +1.0% por ativação, 3 ativações):
- 3 ativações HC × 1.0% = +3.0% G-mean
- Elite: 82.87% → 85.87%

---

## 8. VALIDAÇÃO E TESTE

### Plano de Teste

1. **Teste Unitário**: Validar extração de regras DT
2. **Teste Integração**: Validar 3 estratégias isoladamente
3. **Teste E2E**: Executar 1 chunk RBF_Abrupt_Severe
4. **Teste Comparativo**: Comparar HC v2 vs HC Inteligente

### Critérios de Sucesso

✅ **APROVADO** se:
- Taxa de aprovação ≥ 25%
- Impacto por ativação ≥ 0.2% G-mean
- G-mean final ≥ 84%

❌ **REPROVADO** se:
- Taxa de aprovação < 15%
- Impacto por ativação < 0.1% G-mean
- Erros de execução ou crashes

---

## 9. IMPLEMENTAÇÃO FASEADA

### Fase 2A: Extração de Regras DT (1h)
- Implementar `extract_rules_from_tree()`
- Implementar `rank_rules_by_error_coverage()`
- Testar extração com DT simples

### Fase 2B: Estratégia 1 (1.5h)
- Implementar `error_focused_dt_rules()`
- Integrar no sistema hierárquico
- Testar isoladamente

### Fase 2C: Estratégias 2 e 3 (1.5h)
- Implementar `ensemble_boosting()`
- Implementar `guided_mutation()`
- Integrar todas as 3 estratégias

### Fase 2D: Teste E2E (1h)
- Executar 1 chunk RBF_Abrupt_Severe
- Analisar logs e taxa de aprovação
- Validar critérios de sucesso

**Tempo Total Estimado**: 5h implementação + teste

---

## 10. PRÓXIMOS PASSOS

1. ✅ Revisar design com usuário
2. ⏳ Implementar Fase 2A (extração DT)
3. ⏳ Implementar Fase 2B (estratégia 1)
4. ⏳ Implementar Fase 2C (estratégias 2 e 3)
5. ⏳ Testar com 1 chunk
6. ⏳ Analisar resultados e ajustar

---

## Assinatura

**Projetado por**: Claude Code
**Data**: 2025-10-13
**Status**: ✅ Design aprovado - Aguardando implementação
**Impacto Esperado**: +2-5% G-mean, Taxa HC 30-50%
