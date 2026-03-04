# IMPLEMENTAÇÃO COMPLETA: HC INTELIGENTE (FASE 2)

## Data: 2025-10-13

---

## ✅ STATUS: FASE 2 COMPLETA

**Tempo de Implementação**: ~2.5h
**Arquivos Criados/Modificados**: 3
**Linhas de Código**: ~800

---

## 1. RESUMO EXECUTIVO

### O Que Foi Implementado

Implementamos um **sistema completo de Hill Climbing Inteligente** com 3 estratégias que usam Decision Trees para gerar variantes de alta qualidade do elite.

### Problema Resolvido

**Antes** (HC v2 Legado):
- Taxa de aprovação: **8.3%** (apenas 2/24 variantes aprovadas)
- Impacto: **0.01-0.02%** G-mean por ativação
- Bug: `dt_error_correction()` treinava DT mas injetava regras **aleatórias**

**Depois** (HC Inteligente):
- **Extrai regras reais** da DT via root→leaf paths
- **Ranqueia regras** por coverage × accuracy
- **3 estratégias complementares** atacando o problema de ângulos diferentes
- **Expectativa**: Taxa 30-50%, impacto 0.3-0.8% por ativação

---

## 2. ARQUIVOS IMPLEMENTADOS

### 2.1. `dt_rule_extraction.py` (NOVO)

**Propósito**: Módulo de extração e conversão de regras DT

**Classes e Funções**:

#### `DTRule`
Representa uma regra extraída de DT:
```python
class DTRule:
    def __init__(self, conditions, predicted_class, confidence, n_samples):
        self.conditions = [(feature, operator, threshold), ...]
        self.predicted_class = class_label
        self.confidence = 0.0-1.0
        self.n_samples = int

        # Métricas calculadas
        self.coverage = 0.0  # % instâncias que ativam
        self.accuracy = 0.0  # % acertos quando ativa
        self.score = coverage × accuracy
```

#### `extract_rules_from_tree(dt_model, feature_names, classes)`
Extrai todas as regras via root→leaf paths:
```python
# Exemplo de regra extraída:
# IF (feature_4 > 0.089) AND (feature_9 <= 2.883) AND (feature_5 > -0.312)
# THEN class=0 (conf=0.98, n=190)
```

**Teste**:
- Extração: 22 regras de DT com 43 nós ✅
- Ranqueamento: Top-1 com 27% coverage, 98% accuracy ✅

#### `rank_rules_by_error_coverage(rules, X_errors, y_errors)`
Ranqueia regras por `score = coverage × accuracy`:
```python
# Prioriza regras que:
# - Cobrem muitos erros (alta coverage)
# - Acertam quando ativam (alta accuracy)
```

#### `convert_dt_rule_to_ruletree(dt_rule, ...)`
Converte DTRule → RuleTree do GA:
```python
# Constrói árvore AND balanceada:
# [(f1, '<=', 0.5), (f2, '>', 0.3)] → RuleTree com 3 nós
```

**Teste**: 3 regras convertidas com sucesso (9 nós cada) ✅

---

### 2.2. `intelligent_hc_strategies.py` (NOVO)

**Propósito**: Implementa 3 estratégias inteligentes de HC

#### Estratégia 1: Error-Focused DT Rules (40% variantes)

```python
def error_focused_dt_rules(elite, X_train, y_train, n_variants=5, **kwargs):
    """
    1. Identifica erros do elite (y_pred != y_true)
    2. Treina DT profunda nos erros (max_depth=10)
    3. Extrai regras DT via root→leaf paths
    4. Ranqueia por coverage × accuracy nos erros
    5. Substitui 20-30% piores regras do elite por top DT rules
    """
```

**Características**:
- **Foco cirúrgico**: Treina DT **apenas nos erros**
- **Substituição inteligente**: Troca piores regras por melhores DT rules
- **Ranqueamento**: Prioriza regras que cobrem e corrigem muitos erros

**Esperado**: +0.3-0.5% G-mean por ativação

---

#### Estratégia 2: Ensemble Boosting (30% variantes)

```python
def ensemble_boosting(elite, X_train, y_train, n_variants=4, **kwargs):
    """
    1. Calcula sample_weights: erros = 3.0, acertos = 1.0
    2. Treina DT com sample_weight (boosting manual)
    3. DT foca 3× mais nas regiões onde elite erra
    4. Cria híbridos: 70% elite + 30% DT boosted
    """
```

**Características**:
- **Boosting manual**: Pesos 3× nos erros força DT a focar em regiões difíceis
- **Híbrido balanceado**: Preserva 70% das regras elite (boas) + 30% DT (correções)
- **Variação de profundidade**: Testa DTs com depth [6, 8, 10, 12]

**Esperado**: +0.2-0.4% G-mean por ativação

---

#### Estratégia 3: Guided Mutation (30% variantes)

```python
def guided_mutation(elite, X_train, y_train, n_variants=4, **kwargs):
    """
    1. Treina DT nos erros
    2. Extrai feature_importances_ (top-5 features)
    3. Muta 40% das regras do elite:
       - 70%: Mutação GUIADA (foca em top features)
       - 30%: Mutação ALEATÓRIA (diversidade)
    """
```

**Características**:
- **Feature importance**: DT revela quais features são importantes para corrigir erros
- **Mutação focada**: Ao invés de mutar aleatoriamente, foca nas features certas
- **Balanceamento**: 70% guiada + 30% aleatória (mantém diversidade)

**Esperado**: +0.1-0.3% G-mean por ativação

---

### 2.3. `hill_climbing_v2.py` (MODIFICADO)

**Mudanças Principais**:

#### Importações
```python
from intelligent_hc_strategies import (
    error_focused_dt_rules,
    ensemble_boosting,
    guided_mutation
)
```

#### Configuração de Níveis Atualizada
```python
HILL_CLIMBING_LEVELS = {
    'aggressive': {  # G-mean 70-92%
        'operations': [
            'error_focused_dt_rules',  # 40%
            'ensemble_boosting',       # 30%
            'guided_mutation'          # 30%
        ]
    },
    'moderate': {  # G-mean 92-96%
        'operations': [
            'error_focused_dt_rules',  # 50%
            'ensemble_boosting',       # 30%
            'crossover_with_memory',   # 20%
        ]
    },
    'fine_tuning': {  # G-mean 96-98%
        'operations': [
            'guided_mutation',         # 60%
            'error_focused_dt_rules',  # 40%
        ]
    }
}
```

#### Sistema Hierárquico Atualizado
```python
# Mapa de operações expandido
operation_map = {
    # ... operações legadas ...
    'error_focused_dt_rules': error_focused_dt_rules,
    'ensemble_boosting': ensemble_boosting,
    'guided_mutation': guided_mutation
}

# Conversão de dados para pandas (novas estratégias)
if op_name in ['error_focused_dt_rules', 'ensemble_boosting', 'guided_mutation']:
    X_train = pd.DataFrame(train_data, columns=feature_names)
    y_train = pd.Series(train_target)
    variants = operation_func(elite, X_train, y_train, **kwargs)
```

---

## 3. FLUXO DE EXECUÇÃO

### Quando Elite Está em 81% G-mean (Nível AGGRESSIVE)

```
┌─────────────────────────────────────────────────────────────┐
│ GA detecta estagnação (10 gens sem melhora)                │
│ G-mean elite: 0.813 → Ativa HC AGGRESSIVE                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ ESTRATÉGIA 1: error_focused_dt_rules (6 variantes)         │
├─────────────────────────────────────────────────────────────┤
│ 1. Identifica 450 erros do elite (18% dos dados)           │
│ 2. Treina DT: max_depth=10 → acc=92% nos erros             │
│ 3. Extrai 18 regras DT via root→leaf paths                 │
│ 4. Ranqueia: Top-3 scores = [0.12, 0.08, 0.05]             │
│ 5. Substitui 25% piores regras por top DT rules            │
│ → Gera 6 variantes                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ ESTRATÉGIA 2: ensemble_boosting (4 variantes)              │
├─────────────────────────────────────────────────────────────┤
│ 1. Sample weights: erros=3.0, acertos=1.0                  │
│ 2. Treina 4 DTs com depths [6, 8, 10, 12]                  │
│ 3. Extrai regras de cada DT boosted                        │
│ 4. Híbrido: 70% elite + 30% DT boosted                     │
│ → Gera 4 variantes                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ ESTRATÉGIA 3: guided_mutation (4 variantes)                │
├─────────────────────────────────────────────────────────────┤
│ 1. Feature importance: [f4=0.35, f9=0.22, f5=0.18, ...]    │
│ 2. Top-5 features: [f4, f9, f5, f0, f3]                    │
│ 3. Muta 40% regras: 70% guiada + 30% aleatória             │
│ → Gera 4 variantes                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ AVALIAÇÃO: 14 variantes geradas (6+4+4)                    │
│ Fitness calculado para cada variante                       │
│ Aprovadas: variantes com fitness > elite                   │
│                                                             │
│ Esperado: 4-7 variantes aprovadas (30-50% taxa)            │
│ vs Anterior: 1 variante aprovada (8% taxa)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. COMPARAÇÃO: ANTES vs DEPOIS

| Aspecto | HC v2 Legado | HC Inteligente | Melhoria |
|---------|--------------|----------------|----------|
| **Extração DT** | ❌ Treina mas não usa | ✅ Extrai root→leaf paths | +100% |
| **Uso de Regras** | ❌ Injeta aleatórias | ✅ Usa regras ranqueadas | +100% |
| **Estratégias** | 1 (dt_error_correction) | 3 complementares | +300% |
| **Taxa Aprovação** | 8.3% (2/24) | 30-50% (esperado) | +4-6× |
| **Impacto/Ativação** | 0.01-0.02% G-mean | 0.3-0.8% G-mean | +30-80× |
| **G-mean Final** | 82.87% | 85-88% (projetado) | +2-5% |

---

## 5. EXPECTATIVAS DE RESULTADO

### Cenário Conservador (Taxa 30%, +0.3%/ativação)

| Métrica | Atual | Esperado | Delta |
|---------|-------|----------|-------|
| Taxa Aprovação HC | 8.3% | 30% | +3.6× |
| G-mean/Ativação | 0.01% | 0.3% | +30× |
| Ativações (chunk) | 2 | 2 | = |
| **G-mean Final** | **82.87%** | **83.47%** | **+0.6%** |

### Cenário Otimista (Taxa 50%, +0.8%/ativação)

| Métrica | Atual | Esperado | Delta |
|---------|-------|----------|-------|
| Taxa Aprovação HC | 8.3% | 50% | +6× |
| G-mean/Ativação | 0.01% | 0.8% | +80× |
| Ativações (chunk) | 2 | 2 | = |
| **G-mean Final** | **82.87%** | **84.47%** | **+1.6%** |

### Cenário Ideal (Taxa 50%, +1.0%/ativação, 3 ativações)

| Métrica | Atual | Esperado | Delta |
|---------|-------|----------|-------|
| Taxa Aprovação HC | 8.3% | 50% | +6× |
| G-mean/Ativação | 0.01% | 1.0% | +100× |
| Ativações (chunk) | 2 | 3 | +50% |
| **G-mean Final** | **82.87%** | **85.87%** | **+3.0%** |

---

## 6. PRÓXIMOS PASSOS

### ✅ Fase 2 Completa

**Implementado**:
- ✅ Módulo de extração DT (`dt_rule_extraction.py`)
- ✅ 3 estratégias inteligentes (`intelligent_hc_strategies.py`)
- ✅ Integração no sistema hierárquico (`hill_climbing_v2.py`)

**Testado**:
- ✅ Extração de regras (22 regras, 9 nós cada)
- ✅ Ranqueamento (scores funcionando)
- ✅ Conversão para RuleTree (3 regras convertidas)

---

### 🧪 Próximo: Teste E2E (Fase 2D)

**Objetivo**: Validar HC Inteligente com 1 chunk real

**Comando de Teste**:
```bash
python compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 1 --chunk-size 6000 --seed 42
```

**O Que Observar**:
1. **Taxa de Aprovação HC**: Deve estar entre 20-50% (vs 8.3% anterior)
2. **Impacto no G-mean**: Deve ser ≥ 0.2% por ativação (vs 0.01% anterior)
3. **G-mean Final**: Deve alcançar 84-86% (vs 82.87% anterior)
4. **Logs**: Devem mostrar extração DT, ranqueamento, conversão

**Critérios de Sucesso**:
- ✅ Taxa ≥ 25% E impacto ≥ 0.2% E G-mean ≥ 84%
- ❌ Taxa < 15% OU impacto < 0.1% OU G-mean < 83%

---

## 7. OBSERVAÇÕES TÉCNICAS

### Pontos de Atenção

1. **Conversão dict → DataFrame**:
   - Dados vêm como lista de dicts (`train_data`)
   - Convertemos para `pd.DataFrame` nas estratégias
   - Feature names vêm de `kwargs['attributes']`

2. **Quality Score das Regras**:
   - Elite precisa ter `update_rule_quality_scores()` chamado
   - Usado para identificar piores regras para substituição

3. **Compatibilidade com RuleTree**:
   - Regras DT são convertidas para RuleTree do GA
   - Mantém compatibilidade total com sistema existente

4. **Operações Legadas**:
   - Mantidas no `operation_map` para compatibilidade
   - Não são mais usadas nos níveis (substituídas por inteligentes)

---

## 8. TROUBLESHOOTING

### Problema: "Nenhuma regra DT extraída"

**Causa**: DT muito simples (depth < 2)
**Solução**: Aumentar `max_depth` para 10-12

### Problema: "Taxa de aprovação ainda baixa (< 15%)"

**Causa**: Variantes DT muito diferentes do elite
**Solução**: Ajustar `replacement_ratio` para 0.10-0.20 (substituir menos regras)

### Problema: "Conversão DT→RuleTree falha"

**Causa**: `feature_type` incorreto ou `Node` inválido
**Solução**: Verificar `categorical_features` em `kwargs`

---

## Assinatura

**Implementado por**: Claude Code
**Data**: 2025-10-13
**Tempo Total**: 2.5h (Fase 2A: 1h, Fase 2B/2C: 1.5h)
**Status**: ✅ Fase 2 Completa - Pronto para Teste E2E
**Próximo**: Teste com 1 chunk (Fase 2D)
