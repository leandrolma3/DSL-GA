# ANALISE DO SISTEMA DE FITNESS DO GBML

**Data**: 2025-12-15
**Objetivo**: Entender a logica de fitness e planejar experimento balanceado performance vs complexidade

---

## 1. RESUMO EXECUTIVO

### Descoberta Principal

O sistema de fitness atual tem um **multiplicador 0** que **ANULA TODAS as penalidades de complexidade**:

```python
# fitness.py, linha 382
fitness_score = performance_score - (0 * total_penalty)  # <-- MULTIPLICADOR 0!
```

Isso explica porque o GBML gera mais regras que o ERulesD2S - **nao ha penalizacao por complexidade**.

---

## 2. FLUXO DOS PARAMETROS

### 2.1 Caminho: YAML -> main.py -> ga.py -> fitness.py

```
config.yaml (fitness_params)
    |
    v
main.py: fit_p.get('initial_regularization_coefficient', 0.001)
         fit_p.get('feature_penalty_coefficient', 0.0)
    |
    v
ga.py: run_genetic_algorithm(..., regularization_coefficient, ...)
    |
    v
ga.py: constant_args = {'regularization_coefficient': ..., ...}
    |
    v
fitness.py: calculate_fitness(..., regularization_coefficient, ...)
    |
    v
fitness.py: complexity_penalty = regularization_coefficient * (total_nodes + 5 * total_rules)
    |
    v
fitness.py: fitness_score = performance_score - (0 * total_penalty)  # <-- ANULADO!
```

### 2.2 Parametros do YAML (fitness_params)

| Parametro | Valor Atual | Usado Para |
|-----------|-------------|------------|
| `initial_regularization_coefficient` | 0.001 | Penalidade por nodes + regras |
| `feature_penalty_coefficient` | 0.1 | Penalidade por numero de features |
| `operator_penalty_coefficient` | 0.0001 | Penalidade por operadores logicos |
| `threshold_penalty_coefficient` | 0.0001 | Penalidade por variacao de thresholds |
| `operator_change_coefficient` | 0.05 | Penalidade por mudanca de operadores |
| `gamma` | 0.1 | Penalidade por mudanca de features |
| `class_coverage_coefficient` | 0.2 | Bonus por cobertura de classes |
| `gmean_bonus_coefficient` | 0.1 | Peso do F1 no calculo de performance |

---

## 3. ESTRUTURA DA FUNCAO DE FITNESS

### 3.1 Calculo de Performance (Linha 373)

```python
performance_score = g_mean + (weighted_f1 * gmean_bonus_coefficient) + coverage_bonus
```

Onde:
- `g_mean`: Media geometrica dos recalls por classe (0-1)
- `weighted_f1`: F1-Score ponderado (0-1)
- `coverage_bonus`: Bonus por cobrir mais classes

### 3.2 Penalidades Calculadas (Linhas 347-366)

```python
# a) Penalidade de Complexidade
total_nodes = individual.count_total_nodes()
total_rules = individual.count_total_rules()
complexity_penalty = regularization_coefficient * (total_nodes + 5 * total_rules)

# b) Penalidade por Numero de Features
feature_penalty = feature_penalty_coefficient * len(used_features)

# c) Penalidade de Instabilidade vs. Memoria (Jaccard Distance)
distance_penalty = beta * compute_features_distance(used_features, reference_features)

# d) Penalidade de Instabilidade vs. Chunk Anterior
change_penalty = gamma * delta_features
```

### 3.3 Calculo Final (PROBLEMA - Linha 382)

```python
total_penalty = complexity_penalty + feature_penalty + distance_penalty + change_penalty

# AQUI ESTA O PROBLEMA - MULTIPLICADOR 0!
fitness_score = performance_score - (0 * total_penalty)
```

O multiplicador `0` **anula completamente** o `total_penalty`, fazendo com que:
- `fitness_score = performance_score` (sem penalidades)

---

## 4. ANALISE DAS PENALIDADES

### 4.1 Complexity Penalty

```python
complexity_penalty = regularization_coefficient * (total_nodes + 5 * total_rules)
```

**Componentes**:
- `total_nodes`: Soma de todos os nos nas arvores de regras
- `total_rules`: Numero total de regras (multiplicado por 5 para maior peso)

**Exemplo com valores atuais**:
- Se `total_nodes = 100`, `total_rules = 20`
- `complexity_penalty = 0.001 * (100 + 5*20) = 0.001 * 200 = 0.2`

### 4.2 Feature Penalty

```python
feature_penalty = feature_penalty_coefficient * len(used_features)
```

**Exemplo**:
- Se `used_features = 8`, `feature_penalty_coefficient = 0.1`
- `feature_penalty = 0.1 * 8 = 0.8`

### 4.3 Impacto Potencial das Penalidades

| Cenario | G-Mean | Total Penalty | Fitness (com 0*) | Fitness (com 1*) |
|---------|--------|---------------|------------------|------------------|
| Alta performance, alta complexidade | 0.95 | 1.5 | 0.95 | -0.55 |
| Alta performance, baixa complexidade | 0.95 | 0.3 | 0.95 | 0.65 |
| Media performance, baixa complexidade | 0.75 | 0.2 | 0.75 | 0.55 |
| Baixa performance, baixa complexidade | 0.50 | 0.1 | 0.50 | 0.40 |

---

## 5. PROPOSTA DE BALANCEAMENTO

### 5.1 Principios

1. **Nao penalizar demais**: Penalidades muito altas podem eliminar solucoes complexas necessarias
2. **Escala apropriada**: Performance (0-1), penalidades devem estar na mesma escala
3. **Gradual**: Comecar com penalidades baixas e ajustar

### 5.2 Estrategias Propostas

#### Estrategia A: Penalidade Suave (RECOMENDADA PARA INICIO)

```python
# Modificar linha 382 de fitness.py
fitness_score = performance_score - (0.1 * total_penalty)
```

**Efeito**: Penalidade representa ~10% do impacto total

#### Estrategia B: Penalidade Moderada

```python
fitness_score = performance_score - (0.3 * total_penalty)
```

**Efeito**: Penalidade representa ~30% do impacto total

#### Estrategia C: Penalidade Forte

```python
fitness_score = performance_score - (0.5 * total_penalty)
```

**Efeito**: Penalidade representa ~50% do impacto total

### 5.3 Ajustes nos Coeficientes do YAML

Para a Estrategia A (suave), podemos ajustar:

```yaml
fitness_params:
  # Coeficientes atuais (focados em performance)
  initial_regularization_coefficient: 0.001  # Penalidade por complexidade
  feature_penalty_coefficient: 0.1           # Penalidade por features

  # Novos valores propostos (balanceados)
  initial_regularization_coefficient: 0.01   # 10x maior
  feature_penalty_coefficient: 0.05          # Reduzido pela metade
```

### 5.4 Formula Proposta para Balanceamento

```python
# Opcao 1: Multiplicador global
fitness_score = performance_score - (PENALTY_WEIGHT * total_penalty)

# Opcao 2: Normalizacao por escala
max_expected_penalty = 2.0  # Estimativa
normalized_penalty = min(total_penalty / max_expected_penalty, 1.0)
fitness_score = performance_score - (PENALTY_WEIGHT * normalized_penalty)
```

---

## 6. LOCAIS DE MODIFICACAO

### 6.1 Para Ativar Penalidades (fitness.py)

**Linha 382** - Alterar de:
```python
fitness_score = performance_score - (0 * total_penalty)
```

Para:
```python
PENALTY_WEIGHT = 0.1  # Ajustar conforme estrategia
fitness_score = performance_score - (PENALTY_WEIGHT * total_penalty)
```

### 6.2 Para Parametrizar via YAML

1. Adicionar novo parametro no YAML:
```yaml
fitness_params:
  penalty_weight: 0.1  # Novo parametro
```

2. Passar para funcao de fitness no main.py e ga.py

3. Usar na formula em fitness.py

---

## 7. EXPERIMENTOS PROPOSTOS

### 7.1 Experimento Baseline (Atual)

- **Descricao**: Sem penalidade de complexidade (multiplicador 0)
- **Status**: JA EXECUTADO (chunk_size=1000 e chunk_size=2000)
- **Resultado**: GBML gera ~25 regras/chunk vs ERulesD2S ~14 regras/chunk

### 7.2 Experimento Balanceado (Proposto)

- **Descricao**: Com penalidade de complexidade moderada
- **Multiplicador**: 0.1 (10% de peso para penalidades)
- **Hipotese**: GBML gerara menos regras, possivelmente com G-Mean similar ou levemente menor

### 7.3 Metricas de Comparacao

| Metrica | Experimento Baseline | Experimento Balanceado |
|---------|---------------------|------------------------|
| G-Mean medio | A medir | A medir |
| Numero medio de regras | ~25 | Esperado: ~15-20 |
| Numero medio de condicoes | ~5.6 | Esperado: ~4-5 |
| TCS (estabilidade) | A medir | Esperado: menor |

---

## 8. PROXIMOS PASSOS

1. **DECIDIR**: Qual estrategia de balanceamento usar (A, B ou C)
2. **MODIFICAR**: Linha 382 do fitness.py
3. **CRIAR**: Novos YAMLs com parametros ajustados
4. **EXECUTAR**: Experimentos com chunk_size=2000
5. **COMPARAR**: Resultados com experimento baseline

---

## 9. RISCOS E MITIGACOES

| Risco | Impacto | Mitigacao |
|-------|---------|-----------|
| Penalidade muito alta | Modelos muito simples, baixo G-Mean | Comecar com peso 0.1 |
| Penalidade muito baixa | Nenhuma diferenca significativa | Aumentar gradualmente |
| Problemas de escala | Penalidades dominam ou sao irrelevantes | Normalizar valores |

---

**Autor**: Claude Code
**Versao**: 1.0
**Status**: ANALISE COMPLETA - AGUARDANDO DECISAO
