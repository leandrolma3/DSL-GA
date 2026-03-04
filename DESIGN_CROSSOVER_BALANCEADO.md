# DESIGN: CROSSOVER BALANCEADO INTELIGENTE

## Data: 2025-10-15

---

## 1. CONTEXTO E MOTIVAÇÃO

### Problema Identificado

**Experimento atual (HC Inteligente)**:
- G-mean final: 90.38% (treino), 88.77% (teste)
- Gap vs ARF: -3.85% (92.62% - 88.77%)
- Diversidade populacional: oscila 0.414-0.623 (média ~0.48)
- **Hipótese**: Crossover não está balanceando qualidade e diversidade adequadamente

### Crossover Atual (ga_operators.py:147-203)

#### Modo REFINAMENTO (pais completos)
```python
# Estratégia: Copia Pai 1, substitui 1 regra pior pela 1 regra melhor do Pai 2
child_rules = deepcopy(parent1.rules)
worst_child = min(rules_child, key=lambda r: r.quality_score)
best_parent2 = max(rules_parent2, key=lambda r: r.quality_score)
if best_parent2.quality_score > worst_child.quality_score:
    child_rules[class_label][idx] = deepcopy(best_parent2)
```

**Problemas**:
1. ❌ **Muito conservador**: Apenas 1 regra muda por classe
2. ❌ **100% elitista**: Sempre pega a melhor do Pai 2
3. ❌ **Sem diversidade**: Não há componente aleatório/exploratório
4. ❌ **Não adaptativo**: Ratio fixo durante toda evolução

#### Modo EXPANSÃO (pais incompletos)
```python
# Estratégia: Crossover de nó + preenchimento aleatório
rule1, rule2 = random.choice(rules1), random.choice(rules2)
swap_subtrees(node1, node2)  # Crossover de nó
# Preenche restante com regras aleatórias
```

**Problemas**:
1. ❌ **100% aleatório**: Não considera qualidade das regras
2. ❌ **Não prioriza boas regras**: Seleção uniforme
3. ❌ **Crossover de nó arriscado**: Pode quebrar regras boas

---

## 2. SOLUÇÃO PROPOSTA: CROSSOVER BALANCEADO ADAPTATIVO

### Filosofia de Design

```
┌─────────────────────────────────────────────────────────────────┐
│  OBJETIVO: Acelerar convergência SEM sacrificar diversidade    │
├─────────────────────────────────────────────────────────────────┤
│  ✓ EXPLOITAÇÃO (70%): Prioriza regras de alta qualidade        │
│  ✓ EXPLORAÇÃO (30%): Mantém diversidade com regras variadas    │
│  ✓ ADAPTATIVO: Ajusta ratio conforme geração (↑ exploitação)   │
│  ✓ BALANCEADO: Evita elitismo excessivo E convergência lenta   │
└─────────────────────────────────────────────────────────────────┘
```

### Princípios Fundamentais

1. **Avaliação Individual de Regras**: Cada regra é avaliada independentemente (não apenas quality_score herdado)
2. **Seleção Ponderada**: 70% melhores + 30% diversas (não 100% melhores)
3. **Ratio Adaptativo**: Mais exploração no início, mais exploitação no final
4. **Preenchimento Inteligente**: Se faltar regras, cria novas ao invés de copiar ruins

---

## 3. COMPONENTES DO SISTEMA

### 3.1. Função de Avaliação de Regras

```python
def evaluate_rule_quality(rule: RuleTree, data: List[Dict], target: List,
                         class_label: Any) -> float:
    """
    Avalia a qualidade de UMA regra individual com métricas balanceadas.

    Retorna: Score no intervalo [0.0, 1.0]

    Métricas:
    - Precision (50%): Das instâncias que a regra ativa, quantas são da classe correta?
    - Coverage (30%): Quantas instâncias da classe a regra cobre?
    - Balance (20%): Penaliza regras muito específicas ou muito gerais
    """
    activated_correct = 0
    activated_total = 0
    target_instances_covered = 0
    target_instances_total = sum(1 for y in target if y == class_label)

    # Aplica a regra em cada instância
    for x, y in zip(data, target):
        activates = rule.evaluate(x)  # Retorna True/False

        if activates:
            activated_total += 1
            if y == class_label:
                activated_correct += 1
                target_instances_covered += 1

    # 1. PRECISION: Evita regras que ativam errado
    precision = activated_correct / activated_total if activated_total > 0 else 0.0

    # 2. COVERAGE: Recompensa regras que cobrem instâncias da classe
    coverage = target_instances_covered / target_instances_total if target_instances_total > 0 else 0.0

    # 3. BALANCE PENALTY: Penaliza extremos
    activation_rate = activated_total / len(data) if len(data) > 0 else 0.0

    if activation_rate < 0.01:  # Muito específica (< 1% dos dados)
        balance_penalty = 0.5
    elif activation_rate > 0.90:  # Muito geral (> 90% dos dados)
        balance_penalty = 0.7
    else:
        balance_penalty = 1.0  # OK

    # SCORE FINAL: Média ponderada
    score = (0.50 * precision +
             0.30 * coverage +
             0.20 * balance_penalty)

    return score
```

**Justificativa dos pesos**:
- **Precision (50%)**: Mais importante - regras que erram são inúteis
- **Coverage (30%)**: Importante - regras que não cobrem nada também são inúteis
- **Balance (20%)**: Menos importante - evita apenas casos extremos

---

### 3.2. Função de Ratio Adaptativo

```python
def get_quality_ratio_adaptive(current_generation: int, max_generations: int) -> float:
    """
    Retorna o ratio de qualidade/diversidade baseado na geração atual.

    Estratégia em 3 fases:
    - FASE 1 (Gen 1-20):    50% qualidade + 50% diversidade (EXPLORAÇÃO)
    - FASE 2 (Gen 21-60):   70% qualidade + 30% diversidade (BALANCEADO)
    - FASE 3 (Gen 61-max):  85% qualidade + 15% diversidade (EXPLOITAÇÃO)

    Retorna: quality_ratio ∈ [0.5, 0.85]
    """
    if current_generation <= 20:
        # Fase inicial: Exploração agressiva
        return 0.50
    elif current_generation <= 60:
        # Fase intermediária: Balanceado
        return 0.70
    else:
        # Fase final: Exploitação dominante
        return 0.85
```

**Justificativa das fases**:
- **Fase 1**: População ainda diversa, precisa explorar espaço de busca
- **Fase 2**: População já convergindo, equilibra qualidade e diversidade
- **Fase 3**: População madura, foca em refinamento (HC também ativo)

---

### 3.3. Função Principal: Crossover Balanceado

```python
def balanced_crossover(
    parent1: Individual,
    parent2: Individual,
    data: List[Dict],
    target: List,
    max_rules_per_class: int,
    current_generation: int,
    max_generations: int,
    classes: List[Any],
    base_attributes_info: Dict,
    **kwargs
) -> Individual:
    """
    Crossover Balanceado Adaptativo: 70% qualidade + 30% diversidade (adaptável).

    Processo:
    1. Avalia TODAS as regras dos pais individualmente
    2. Ranqueia regras por score (melhor → pior)
    3. Seleciona top N% por qualidade (exploitação)
    4. Seleciona (100-N)% aleatórias do restante (exploração)
    5. Preenche slots vazios com regras NOVAS (exploração pura)

    Args:
        parent1, parent2: Pais selecionados
        data, target: Dados de treino do chunk atual (para avaliação)
        max_rules_per_class: Número máximo de regras por classe
        current_generation: Geração atual (para ratio adaptativo)
        max_generations: Total de gerações (para ratio adaptativo)
        classes: Lista de classes do problema
        base_attributes_info: Info de atributos para criar regras novas

    Returns:
        Filho (Individual) com regras balanceadas
    """
    # 1. Determina ratio adaptativo
    quality_ratio = get_quality_ratio_adaptive(current_generation, max_generations)
    diversity_ratio = 1.0 - quality_ratio

    logging.debug(f"  Crossover Balanceado (Gen {current_generation}): {quality_ratio:.0%} qualidade + {diversity_ratio:.0%} diversidade")

    # 2. Cria filho vazio
    child = Individual(
        max_rules_per_class=max_rules_per_class,
        classes=classes,
        **base_attributes_info
    )
    child.rules = {c: [] for c in classes}

    # 3. Para cada classe, cria regras balanceadas
    for class_label in classes:
        # 3a. Coleta regras dos pais
        rules_p1 = parent1.rules.get(class_label, [])
        rules_p2 = parent2.rules.get(class_label, [])

        if not rules_p1 and not rules_p2:
            # Nenhum pai tem regras para esta classe - cria regra nova
            new_rule = _create_specialist_rule(class_label, data, target, base_attributes_info)
            if new_rule:
                child.rules[class_label] = [new_rule]
            continue

        # 3b. Avalia TODAS as regras e ranqueia
        rules_with_scores = []

        for rule in rules_p1:
            score = evaluate_rule_quality(rule, data, target, class_label)
            rules_with_scores.append((score, rule, 'p1'))

        for rule in rules_p2:
            score = evaluate_rule_quality(rule, data, target, class_label)
            rules_with_scores.append((score, rule, 'p2'))

        # Ordena por score (melhor → pior)
        rules_with_scores.sort(key=lambda x: x[0], reverse=True)

        # 3c. Calcula quantas regras por categoria
        n_quality = int(max_rules_per_class * quality_ratio)  # Ex: 70% de 15 = 10
        n_diverse = max_rules_per_class - n_quality           # Ex: 30% de 15 = 5

        # 3d. SELEÇÃO POR QUALIDADE (Top N)
        selected_rules = []
        quality_rules = rules_with_scores[:n_quality]

        for score, rule, parent_id in quality_rules:
            selected_rules.append(copy.deepcopy(rule))

        # 3e. SELEÇÃO POR DIVERSIDADE (Amostra aleatória do restante)
        remaining_rules = rules_with_scores[n_quality:]

        if remaining_rules:
            # Amostra sem reposição do restante
            n_to_sample = min(n_diverse, len(remaining_rules))
            diverse_sample = random.sample(remaining_rules, n_to_sample)

            for score, rule, parent_id in diverse_sample:
                selected_rules.append(copy.deepcopy(rule))

        # 3f. PREENCHIMENTO INTELIGENTE (se ainda faltar regras)
        while len(selected_rules) < max_rules_per_class:
            # Cria regra nova especializada para esta classe
            new_rule = _create_specialist_rule(class_label, data, target, base_attributes_info)

            if new_rule:
                selected_rules.append(new_rule)
            else:
                # Se falhar em criar regra especializada, para de tentar
                break

        # 3g. Atribui regras ao filho
        child.rules[class_label] = selected_rules

        # Log para debug
        n_from_quality = n_quality
        n_from_diversity = len(selected_rules) - n_quality if len(selected_rules) > n_quality else 0
        n_created = max_rules_per_class - len(selected_rules) if len(selected_rules) < max_rules_per_class else 0

        logging.debug(f"    Classe {class_label}: {len(selected_rules)} regras "
                     f"({n_from_quality} qualidade, {n_from_diversity} diversidade, {n_created} novas)")

    # 4. Finaliza filho
    child.default_class = parent1.default_class if random.random() < 0.5 else parent2.default_class
    child.remove_duplicate_rules()

    return child
```

---

## 4. INTEGRAÇÃO NO GA

### 4.1. Modificação em ga.py

**Localização**: Loop principal do GA (~linha 800-1000)

**ANTES**:
```python
# Crossover
offspring = ga_operators.crossover(
    parent1, parent2,
    max_depth=max_depth,
    attributes=attributes,
    value_ranges=value_ranges,
    category_values=category_values,
    categorical_features=categorical_features,
    classes=classes,
    max_rules_per_class=max_rules_per_class
)
```

**DEPOIS**:
```python
# Crossover (escolhe operador baseado em flag)
if config.get('use_balanced_crossover', False):
    # NOVO: Crossover Balanceado Inteligente
    offspring = ga_operators.balanced_crossover(
        parent1, parent2,
        data=X_chunk_dicts,  # Dados de treino do chunk atual
        target=y_chunk,
        max_rules_per_class=max_rules_per_class,
        current_generation=generation + 1,
        max_generations=max_generations,
        classes=classes,
        base_attributes_info={
            'max_depth': max_depth,
            'attributes': attributes,
            'value_ranges': value_ranges,
            'category_values': category_values,
            'categorical_features': categorical_features
        }
    )
else:
    # LEGADO: Crossover Adaptativo (2 modos)
    offspring = ga_operators.crossover(
        parent1, parent2,
        max_depth=max_depth,
        attributes=attributes,
        value_ranges=value_ranges,
        category_values=category_values,
        categorical_features=categorical_features,
        classes=classes,
        max_rules_per_class=max_rules_per_class
    )
```

---

### 4.2. Configuração em config.yaml

```yaml
ga_params:
  # ... parâmetros existentes ...

  # Crossover Balanceado Inteligente (NOVO)
  use_balanced_crossover: true

  # Ratios adaptativos por fase (Gen 1-20, 21-60, 61+)
  balanced_crossover_quality_ratio_phase1: 0.50  # 50% qualidade, 50% diversidade
  balanced_crossover_quality_ratio_phase2: 0.70  # 70% qualidade, 30% diversidade
  balanced_crossover_quality_ratio_phase3: 0.85  # 85% qualidade, 15% diversidade

  # Thresholds de fase
  balanced_crossover_phase1_end: 20    # Gen 1-20: Exploração
  balanced_crossover_phase2_end: 60    # Gen 21-60: Balanceado
  # Gen 61+: Exploitação (implícito)
```

---

## 5. EXPECTATIVAS DE RESULTADO

### 5.1. Métricas de Validação

| Métrica | Baseline (HC Inteligente) | Meta (+ Crossover Balanceado) | Melhoria Esperada |
|---------|--------------------------|------------------------------|-------------------|
| **G-mean final (teste)** | 88.77% | 90-92% | +1.2-3.2% |
| **Diversidade média** | 0.48 (Gen 30-76) | 0.55-0.65 | +14-35% |
| **Diversidade Gen 30** | 0.45-0.50 | 0.55-0.60 | +11-22% |
| **Diversidade Gen 60** | 0.43-0.48 | 0.50-0.55 | +14-16% |
| **Gap vs ARF** | -3.85% | -1.5 a -2.5% | -35-60% gap |
| **Taxa convergência** | Gen ~40-50 | Gen ~50-70 | +25-40% gerações úteis |

### 5.2. Impacto Esperado por Fase

**Fase 1 (Gen 1-20): Exploração Agressiva**
- Ratio 50-50 mantém alta diversidade (0.60-0.65)
- Elite cresce mais devagar (~0.5%/gen vs 1%/gen atual)
- **Trade-off**: -5-10 gens para atingir 85% G-mean, mas +10-15% diversidade

**Fase 2 (Gen 21-60): Balanceado**
- Ratio 70-30 acelera convergência sem perder diversidade
- Elite cresce ~0.3-0.5%/gen (balanceado)
- Diversidade mantém-se em 0.55-0.60 (vs 0.45-0.50 atual)

**Fase 3 (Gen 61+): Exploitação com HC**
- Ratio 85-15 + HC Inteligente = combinação poderosa
- Elite refina últimos 1-2% G-mean
- Diversidade controlada em 0.50-0.55 (suficiente)

---

## 6. CENÁRIOS DE TESTE

### Teste 1: Validação Unitária
**Objetivo**: Verificar se avaliação de regras funciona
```python
# Criar regras sintéticas e avaliar
rule_good = RuleTree(...)  # Alta precision, alta coverage
rule_bad = RuleTree(...)   # Baixa precision ou baixa coverage

score_good = evaluate_rule_quality(rule_good, data, target, class_label)
score_bad = evaluate_rule_quality(rule_bad, data, target, class_label)

assert score_good > score_bad  # Deve ranquear corretamente
```

**Critério de sucesso**: ✅ Regras boas têm score > regras ruins

---

### Teste 2: Validação de Crossover (1 chunk)
**Comando**:
```bash
python compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 1 --chunk-size 6000 --seed 42
```

**Configuração**:
```yaml
use_balanced_crossover: true
max_generations: 80
population_size: 120
```

**Métricas a observar**:
1. **Diversidade por geração**:
   - Gen 1-20: Deve estar em 0.55-0.65 (vs 0.45-0.55 atual)
   - Gen 21-60: Deve estar em 0.50-0.60 (vs 0.40-0.50 atual)
   - Gen 61-80: Deve estar em 0.45-0.55 (vs 0.35-0.45 atual)

2. **G-mean Elite**:
   - Gen 20: 85-87% (pode ser ~2% menor que atual, OK)
   - Gen 40: 88-90% (similar ou +1% vs atual)
   - Gen 60: 90-92% (deve ser +1-2% vs atual)
   - Gen 80: 91-93% (deve ser +2-3% vs atual)

3. **Taxa de aprovação HC**:
   - Deve manter-se em 5-10% (similar ao atual)
   - Impacto/ativação: 0.5-1.0% (pode ser maior se diversidade ajudar)

**Critérios de sucesso**:
- ✅ G-mean final (teste) ≥ 90.0% (vs 88.77% baseline)
- ✅ Diversidade média (Gen 30-60) ≥ 0.52 (vs 0.48 baseline)
- ✅ Não há regressão de performance nos primeiros 20 gens (OK se -1-2%)

---

### Teste 3: Experimento Completo (3 chunks)
**Comando**:
```bash
python compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 3 --chunk-size 6000 --seed 42
```

**Configuração**:
```yaml
use_balanced_crossover: true
max_generations: 200
population_size: 120
```

**Objetivo**: Validar consistência em múltiplos chunks

**Critérios de sucesso**:
- ✅ G-mean médio (3 chunks) ≥ 89.5%
- ✅ Desvio padrão G-mean < 2.0% (consistência)
- ✅ Gap vs ARF ≤ 2.5% (vs -3.85% baseline)

---

## 7. PLANO DE TROUBLESHOOTING

### Problema 1: Diversidade não aumenta
**Sintomas**: Diversidade mantém-se em 0.40-0.45 mesmo com ratio 50-50

**Causas possíveis**:
1. Regras "diversas" selecionadas são muito similares às de qualidade
2. `remove_duplicate_rules()` está removendo muitas regras
3. Preenchimento inteligente não está criando regras novas

**Soluções**:
1. Aumentar peso de coverage (0.30 → 0.40) para penalizar regras similares
2. Criar regras REALMENTE novas (random) ao invés de DT specialist
3. Reduzir threshold de balance_penalty (aceitar mais regras específicas)

---

### Problema 2: Performance piora nos primeiros 20 gens
**Sintomas**: G-mean em Gen 20 é 80-82% (vs 85-87% baseline)

**Causas possíveis**:
1. Ratio 50-50 muito exploratório (muitas regras ruins selecionadas)
2. Avaliação de regras está incorreta (priorizando regras ruins)

**Soluções**:
1. Ajustar Fase 1 para ratio 60-40 ao invés de 50-50
2. Aumentar peso de precision (0.50 → 0.60)
3. Reduzir Fase 1 para Gen 1-10 (não 1-20)

---

### Problema 3: G-mean final não melhora (+0-1% vs baseline)
**Sintomas**: G-mean final 89.0-89.5% (vs 88.77% baseline) - melhoria insuficiente

**Causas possíveis**:
1. Diversidade não está ajudando HC a encontrar melhores variantes
2. Crossover balanceado está "diluindo" o efeito do HC
3. Gap vs ARF é arquitetural (single-model vs ensemble)

**Soluções**:
1. Ajustar Fase 3 para ratio 90-10 (mais exploitação)
2. Combinar com ensemble voting (Top-3 indivíduos)
3. Aceitar que gap <2% vs ARF é o melhor possível

---

## 8. CRONOGRAMA DE IMPLEMENTAÇÃO

### Fase A: Implementação (2-3h)
- [ ] Implementar `evaluate_rule_quality()` em ga_operators.py (30min)
- [ ] Implementar `get_quality_ratio_adaptive()` em ga_operators.py (15min)
- [ ] Implementar `balanced_crossover()` em ga_operators.py (1h)
- [ ] Integrar em ga.py com flag condicional (30min)
- [ ] Adicionar configurações em config.yaml (15min)

### Fase B: Teste Unitário (30min)
- [ ] Criar regras sintéticas (boa vs ruim)
- [ ] Validar avaliação de qualidade (scores corretos)
- [ ] Validar seleção balanceada (70-30 respeitado)

### Fase C: Teste E2E (1h execução + 30min análise)
- [ ] Executar Teste 2 (1 chunk)
- [ ] Analisar logs de diversidade
- [ ] Analisar curva de G-mean
- [ ] Validar critérios de sucesso

### Fase D: Ajustes (1-2h se necessário)
- [ ] Ajustar pesos de avaliação se performance ruim
- [ ] Ajustar ratios de fase se diversidade não aumentar
- [ ] Re-testar com ajustes

**Tempo Total Estimado**: 4-6h (implementação + teste + ajustes)

---

## 9. ASSINATURA

**Projetado por**: Claude Code
**Data**: 2025-10-15
**Status**: ✅ Design completo - Pronto para implementação
**Impacto Esperado**: +1.2-3.2% G-mean, +14-35% diversidade, gap vs ARF reduzido para -1.5 a -2.5%

**Próximo passo**: Implementar Fase A (evaluate_rule_quality + balanced_crossover)
