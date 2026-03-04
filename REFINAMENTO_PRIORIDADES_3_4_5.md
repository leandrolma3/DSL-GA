# 🎯 REFINAMENTO: Prioridades 3, 4 e 5 - Balanceando Exploração vs Exploitação

**Data:** 13/10/2025
**Objetivo:** Refinar operadores genéticos e Hill Climbing para **evitar elitismo excessivo** e **manter diversidade**.

---

## 🧬 PRIORIDADE 3: OPERADORES GENÉTICOS INTELIGENTES (BALANCEADOS)

### **⚠️ PROBLEMA IDENTIFICADO: Risco de Elitismo Excessivo**

**Proposta inicial (ingênua):**
```python
# PROBLEMA: Seleciona APENAS as melhores regras
best_rules_p1 = select_top_rules(parent1, ratio=0.5)  # Top 50%
best_rules_p2 = select_top_rules(parent2, ratio=0.5)  # Top 50%
offspring = combine_rules(best_rules_p1, best_rules_p2)  # 100% elite
```

**Por que é ruim:**
- Offspring herda APENAS regras boas → Convergência prematura
- Perde diversidade genética → Fica preso em ótimo local
- Não explora regiões novas do espaço de busca
- Replica o problema do seeding (elite perfeito intransponível)

---

### **✅ SOLUÇÃO: Crossover Híbrido (Exploração + Exploitação)**

#### **Estratégia: 70% Qualidade + 30% Diversidade**

```python
def balanced_crossover(parent1: Individual, parent2: Individual,
                      X_train, y_train,
                      quality_ratio: float = 0.7,
                      **kwargs) -> Individual:
    """
    Crossover balanceado: Combina regras de qualidade com diversidade.

    Args:
        quality_ratio: Fração de regras selecionadas por qualidade (0.7 = 70%)
                      O restante (0.3 = 30%) é selecionado aleatoriamente

    Estratégia:
    - 70% regras por QUALIDADE (melhores de cada pai)
    - 30% regras ALEATÓRIAS (diversidade + exploração)
    """
    offspring = Individual(classes=parent1.classes)

    # Para cada classe, combina regras dos pais
    for cls in offspring.classes:
        # 1. Avalia performance de CADA regra dos pais
        p1_rules_with_scores = []
        if cls in parent1.rules:
            for rule in parent1.rules[cls]:
                score = evaluate_rule_quality(rule, X_train, y_train, cls)
                p1_rules_with_scores.append((score, rule))

        p2_rules_with_scores = []
        if cls in parent2.rules:
            for rule in parent2.rules[cls]:
                score = evaluate_rule_quality(rule, X_train, y_train, cls)
                p2_rules_with_scores.append((score, rule))

        # 2. Ordena regras por score (melhores primeiro)
        all_rules_sorted = sorted(
            p1_rules_with_scores + p2_rules_with_scores,
            key=lambda x: x[0],
            reverse=True  # Melhores primeiro
        )

        # 3. Calcula quantas regras usar
        max_rules = kwargs.get('max_rules_per_class', 15)
        num_quality_rules = int(max_rules * quality_ratio)     # 70% por qualidade
        num_diverse_rules = max_rules - num_quality_rules      # 30% por diversidade

        # 4. SELEÇÃO BALANCEADA
        selected_rules = []

        # 4a. TOP regras por qualidade (70%)
        quality_rules = [rule for score, rule in all_rules_sorted[:num_quality_rules]]
        selected_rules.extend(quality_rules)

        # 4b. Regras ALEATÓRIAS para diversidade (30%)
        # Seleciona do restante (não do top)
        remaining_rules = [rule for score, rule in all_rules_sorted[num_quality_rules:]]
        if remaining_rules:
            diverse_rules = random.sample(
                remaining_rules,
                min(num_diverse_rules, len(remaining_rules))
            )
            selected_rules.extend(diverse_rules)

        # 4c. Se ainda falta regras, cria NOVAS aleatórias (EXPLORAÇÃO PURA)
        while len(selected_rules) < max_rules:
            new_rule = create_random_rule(
                classes=offspring.classes,
                attributes=kwargs['attributes'],
                value_ranges=kwargs['value_ranges'],
                category_values=kwargs['category_values'],
                categorical_features=kwargs['categorical_features'],
                max_depth=kwargs['max_depth']
            )
            selected_rules.append(new_rule)

        # 5. Adiciona regras ao offspring (deep copy)
        offspring.rules[cls] = [copy.deepcopy(rule) for rule in selected_rules]

    return offspring


def evaluate_rule_quality(rule, X_train, y_train, target_class) -> float:
    """
    Avalia qualidade de UMA regra específica.

    Métricas:
    - Precision: Das instâncias que a regra ativa, quantas são da classe correta?
    - Coverage: Quantas instâncias da classe a regra cobre?
    - Balance: Penaliza regras muito específicas (< 1% ativação) ou muito gerais (> 90%)
    """
    activated_correct = 0
    activated_total = 0
    target_instances_covered = 0
    target_instances_total = sum(1 for y in y_train if y == target_class)

    for x, y in zip(X_train, y_train):
        activates = rule.evaluate(x)

        if activates:
            activated_total += 1
            if y == target_class:
                activated_correct += 1
                target_instances_covered += 1

    # Precision: Evita regras que ativam errado
    precision = activated_correct / activated_total if activated_total > 0 else 0

    # Coverage: Recompensa regras que cobrem instâncias da classe
    coverage = target_instances_covered / target_instances_total if target_instances_total > 0 else 0

    # Balance penalty: Penaliza extremos
    activation_rate = activated_total / len(X_train)
    if activation_rate < 0.01:  # Muito específica (< 1%)
        balance_penalty = 0.5
    elif activation_rate > 0.90:  # Muito geral (> 90%)
        balance_penalty = 0.7
    else:
        balance_penalty = 1.0  # Ok

    # Score final: Média ponderada
    score = (0.6 * precision + 0.3 * coverage + 0.1 * balance_penalty)

    return score
```

---

### **📊 COMPARAÇÃO: Crossover Cego vs Balanceado**

| Método | Qualidade | Diversidade | Exploração | Risco Elitismo |
|--------|-----------|-------------|------------|----------------|
| **Cego (atual)** | Baixa | Alta | Alta | Baixo |
| **100% Elite (ingênuo)** | Alta | **Muito Baixa** | **Muito Baixa** | **ALTO** |
| **70-30 Balanceado** | **Alta** | **Média** | **Média** | **Baixo** |

**Configuração recomendada:**
```python
# Início da evolução (Gen 1-30): Mais exploração
if current_generation < 30:
    quality_ratio = 0.5  # 50% qualidade, 50% diversidade

# Meio (Gen 30-100): Balanceado
elif current_generation < 100:
    quality_ratio = 0.7  # 70% qualidade, 30% diversidade

# Final (Gen 100+): Mais exploitação
else:
    quality_ratio = 0.85  # 85% qualidade, 15% diversidade
```

---

### **🔧 IMPLEMENTAÇÃO: Onde Modificar**

**Arquivo:** `ga.py` ou criar `ga_operators_enhanced.py`

**Função a adicionar:**
```python
def balanced_crossover(...):
    # Implementação acima
    pass

def evaluate_rule_quality(...):
    # Implementação acima
    pass
```

**Integração no GA (ga.py linha ~800-900):**
```python
# ANTES (crossover cego):
offspring = ga_operators.crossover(parent1, parent2, ...)

# DEPOIS (crossover balanceado):
if USE_BALANCED_CROSSOVER:
    offspring = balanced_crossover(
        parent1, parent2,
        X_train=X_chunk, y_train=y_chunk,
        quality_ratio=get_quality_ratio(current_generation),
        max_rules_per_class=max_rules_per_class,
        ...
    )
else:
    offspring = ga_operators.crossover(parent1, parent2, ...)  # Fallback
```

**Config.yaml:**
```yaml
ga_params:
  use_balanced_crossover: true             # ← HABILITAR
  balanced_crossover_quality_ratio_early: 0.5   # Gen 1-30
  balanced_crossover_quality_ratio_mid: 0.7     # Gen 30-100
  balanced_crossover_quality_ratio_late: 0.85   # Gen 100+
```

---

## 🎯 PRIORIDADE 4: HILL CLIMBING FOCADO EM ERROS (REFINADO)

### **⚠️ PROBLEMA ATUAL: HC Gera Variantes Cegas**

**Análise do código atual (`hill_climbing_v2.py:226-336`):**

```python
def dt_error_correction(elite, population, best_ever_memory, **kwargs):
    """
    ATUAL: Treina DT nos erros do elite, mas INJETA REGRAS ALEATÓRIAS!
    """
    # 1. Identifica erros ✅
    error_data = [x for x, y_true in ... if elite.predict(x) != y_true]

    # 2. Treina DT nos erros ✅
    dt = DecisionTreeClassifier(max_depth=random.choice([5,8,10,12]))
    dt.fit(X_error, y_error)

    # 3. PROBLEMA: Não extrai regras do DT! ❌
    # Apenas seleciona doadores aleatórios da população:
    for donor in random.sample(population, 2):
        donor_rules.extend(donor.rules[cls])

    # 4. Injeta regras ALEATÓRIAS dos doadores ❌
    variant.rules[cls].append(random.choice(donor_rules))
```

**Por que não funciona:**
- Treina DT nos erros, mas **NÃO USA** as regras do DT!
- Injeta regras aleatórias da população (que também pode errar lá)
- DT é desperdiçado (só serve como "proxy" para complexidade)

---

### **✅ SOLUÇÃO: HC que REALMENTE Extrai e Usa Regras do DT de Erros**

```python
def intelligent_error_correction(elite: Individual, population: List[Individual],
                                best_ever_memory: List[Individual], **kwargs) -> List[Individual]:
    """
    Hill Climbing inteligente: Corrige erros específicos com regras DT.

    Estratégia:
    1. Identifica onde elite erra
    2. Treina DT PEQUENA focada APENAS nos erros
    3. EXTRAI regras do DT (caminho raiz→folha)
    4. INJETA essas regras no elite (substitui regras fracas)
    5. BALANCEIA: Mantém 80% das regras originais, substitui 20% fracas
    """
    train_data = kwargs.get('train_data', [])
    train_target = kwargs.get('train_target', [])

    if not train_data or len(train_data) < 10:
        return []

    try:
        # 1. Identifica instâncias onde elite ERRA
        error_indices = []
        error_data = []
        error_target = []

        for idx, (x, y_true) in enumerate(zip(train_data, train_target)):
            y_pred = elite._predict(x)
            if y_pred != y_true:
                error_indices.append(idx)
                error_data.append(x)
                error_target.append(y_true)

        # Se elite tem poucos erros (< 5% ou < 10 instâncias), não faz nada
        error_rate = len(error_data) / len(train_data)
        if len(error_data) < 10 or error_rate < 0.05:
            logger.info(f"       intelligent_error_correction: Elite tem poucos erros ({error_rate:.1%}) - não aplicável")
            return []

        logger.info(f"       intelligent_error_correction: Elite erra em {len(error_data)} instâncias ({error_rate:.1%})")

        # 2. Treina DT RASA focada APENAS nos erros
        from sklearn.tree import DecisionTreeClassifier
        X_error = pd.DataFrame(error_data)
        y_error = np.array(error_target)

        # DT RASA para regras simples e interpretáveis
        dt = DecisionTreeClassifier(
            max_depth=random.choice([3, 4, 5]),  # RASA (vs [5,8,10,12] atual)
            min_samples_split=max(2, len(error_data) // 10),
            min_samples_leaf=max(1, len(error_data) // 30),
            random_state=random.randint(0, 10000)
        )
        dt.fit(X_error, y_error)

        # 3. EXTRAI regras do DT (caminho raiz → folha)
        dt_rules_by_class = extract_rules_from_dt(
            dt,
            feature_names=list(X_error.columns),
            class_names=elite.classes
        )

        logger.info(f"       intelligent_error_correction: DT gerou {sum(len(rules) for rules in dt_rules_by_class.values())} regras")

        # 4. Cria variante substituindo regras FRACAS do elite
        variant = copy.deepcopy(elite)

        for cls in variant.classes:
            if cls not in dt_rules_by_class or not dt_rules_by_class[cls]:
                continue

            # 4a. Avalia qualidade de CADA regra do elite para essa classe
            rule_scores = []
            for rule in variant.rules[cls]:
                score = evaluate_rule_quality(rule, train_data, train_target, cls)
                rule_scores.append((score, rule))

            # 4b. Ordena (piores primeiro)
            rule_scores.sort(key=lambda x: x[0])  # Pior → Melhor

            # 4c. SUBSTITUI 20% das PIORES regras por regras DT de erros
            num_to_replace = max(1, int(len(rule_scores) * 0.2))  # 20%
            num_to_replace = min(num_to_replace, len(dt_rules_by_class[cls]))  # Limita às regras DT disponíveis

            # Remove piores regras
            rules_to_keep = [rule for score, rule in rule_scores[num_to_replace:]]

            # Adiciona regras DT de correção
            rules_dt_correction = dt_rules_by_class[cls][:num_to_replace]

            # Atualiza variante
            variant.rules[cls] = rules_to_keep + rules_dt_correction

            logger.info(f"       intelligent_error_correction: Classe {cls} - Substituiu {num_to_replace} regras fracas por DT")

        return [variant]

    except Exception as e:
        logger.warning(f"intelligent_error_correction falhou: {e}")
        return []


def extract_rules_from_dt(dt, feature_names, class_names) -> Dict[Any, List]:
    """
    Extrai regras de um Decision Tree treinado.

    Retorna: Dict[class_label, List[Rule]]
        Ex: {0: [Rule1, Rule2], 1: [Rule3, Rule4]}
    """
    from sklearn.tree import _tree
    tree_ = dt.tree_

    rules_by_class = {cls: [] for cls in class_names}

    def recurse(node, path_conditions):
        """Percorre árvore DFS extraindo caminhos raiz→folha"""
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # Nó interno: Tem split
            feature_idx = tree_.feature[node]
            threshold = tree_.threshold[node]
            feature_name = feature_names[feature_idx]

            # Left: feature <= threshold
            left_conditions = path_conditions + [(feature_name, '<=', threshold)]
            recurse(tree_.children_left[node], left_conditions)

            # Right: feature > threshold
            right_conditions = path_conditions + [(feature_name, '>', threshold)]
            recurse(tree_.children_right[node], right_conditions)

        else:
            # Folha: Retorna regra
            class_id = np.argmax(tree_.value[node][0])
            class_label = class_names[class_id]
            n_samples = tree_.n_node_samples[node]

            # Cria Rule a partir das condições
            if path_conditions and n_samples >= 5:  # Mínimo 5 instâncias
                rule = build_rule_from_conditions(path_conditions, class_label)
                rules_by_class[class_label].append(rule)

    # Inicia recursão na raiz
    recurse(0, [])

    return rules_by_class


def build_rule_from_conditions(conditions, class_label):
    """
    Converte lista de condições em objeto Rule.

    Args:
        conditions: [(feature, op, value), ...]
        class_label: Classe predita

    Returns:
        Rule object compatível com Individual
    """
    # Implementação depende da estrutura do Rule/RuleNode
    # Exemplo simplificado:
    from rule_tree import Rule, RuleNode

    # Cria nó raiz
    root = None
    current_node = None

    for feature, operator, value in conditions:
        node = RuleNode(
            attribute=feature,
            operator=operator,
            value=value,
            node_type='comparison'
        )

        if root is None:
            root = node
            current_node = node
        else:
            # Conecta com AND
            current_node.left = node
            current_node = node

    rule = Rule(root=root, predicted_class=class_label)
    return rule
```

---

### **📊 COMPARAÇÃO: HC Atual vs Inteligente**

| Aspecto | HC Atual | HC Inteligente |
|---------|----------|----------------|
| **Identifica erros** | ✅ Sim | ✅ Sim |
| **Treina DT nos erros** | ✅ Sim | ✅ Sim (mais rasa: 3-5 vs 5-12) |
| **Extrai regras do DT** | ❌ NÃO! | ✅ **SIM** (raiz→folha) |
| **Injeta regras** | Aleatórias da pop | **DT de correção** |
| **Substitui regras** | Adiciona (cresce) | **Substitui 20% piores** |
| **Taxa de sucesso estimada** | 0% (confirmado) | **30-50%?** (a testar) |

---

### **🔧 IMPLEMENTAÇÃO: Onde Modificar**

**Arquivo:** `hill_climbing_v2.py`

**Linha 226-336:** Substituir `dt_error_correction()` por `intelligent_error_correction()`

**Linha 554:** Atualizar mapa de operações:
```python
operation_map = {
    ...
    'dt_error_correction': intelligent_error_correction,  # ← NOVA versão
    ...
}
```

**Linha 54:** Configurar no nível MODERATE:
```python
'moderate': {
    'operations': [
        'dt_error_correction',     # ← Agora usa versão inteligente
        'crossover_with_memory',
        'optimize_thresholds',
        'targeted_mutation'
    ]
}
```

---

## 🛑 PRIORIDADE 5: EARLY STOPPING AGRESSIVO (REFINADO)

### **⚠️ PROBLEMA ATUAL: Para Tarde Demais**

**Análise do log (experimento Pop=120, Gens=200):**

```
Chunk 0: Elite 88.2% em Gen 3 → 90.4% em Gen 126 (+2.2% em 123 gens)
         Early stop em Gen 126 (20 gens sem melhora)
         Tempo desperdiçado: ~2.3h das 3.2h (72%)

Chunk 1: Elite 88.1% em Gen 3 → 90.0% em Gen 130 (+1.9% em 127 gens)
         Early stop em Gen 130 (20 gens sem melhora)
         Tempo desperdiçado: ~2.4h das 3.3h (73%)
```

**Padrão:**
- **70% da melhoria acontece em 3 gerações** (Gen 1-3)
- **30% da melhoria acontece em 138 gerações** (Gen 3-141)
- **Early stopping só ativa após 120+ gerações** (já estagnado há 100+ gens!)

---

### **✅ SOLUÇÃO: Early Stopping Adaptativo em 3 Camadas**

#### **Camada 1: Stop Rápido se Elite Satisfatório**

```python
def adaptive_early_stopping_layer1(best_fitness, gmean, stagnation_count, current_generation):
    """
    Camada 1: Para rápido se elite JÁ É BOM e estagnado.

    Critério: Elite >= 88% G-mean + 15 gens estagnado
    """
    SATISFACTORY_GMEAN = 0.88  # 88% já é bom para maioria dos casos
    EARLY_PATIENCE = 15        # 15 gens (vs 20 atual)

    if gmean >= SATISFACTORY_GMEAN and stagnation_count >= EARLY_PATIENCE:
        logger.info(f"")
        logger.info(f"╔══════════════════════════════════════════════════════════════╗")
        logger.info(f"║ EARLY STOPPING LAYER 1: Elite Satisfatório + Estagnado     ║")
        logger.info(f"╠══════════════════════════════════════════════════════════════╣")
        logger.info(f"║ Elite G-mean:     {gmean:.1%} (≥ {SATISFACTORY_GMEAN:.0%})          ║")
        logger.info(f"║ Estagnação:       {stagnation_count} gerações (≥ {EARLY_PATIENCE})                ║")
        logger.info(f"║ Geração atual:    {current_generation}                                     ║")
        logger.info(f"║ Decisão:          PARAR (performance satisfatória)         ║")
        logger.info(f"╚══════════════════════════════════════════════════════════════╝")
        logger.info(f"")
        return True

    return False
```

#### **Camada 2: Stop se Melhoria é Marginal**

```python
def adaptive_early_stopping_layer2(fitness_history, stagnation_count, current_generation):
    """
    Camada 2: Para se últimas 30 gerações tiveram melhoria < 0.5%.

    Critério: Últimas 30 gens melhoraram < 0.005 G-mean (0.5%)
    """
    MIN_IMPROVEMENT_WINDOW = 30
    MIN_IMPROVEMENT_THRESHOLD = 0.005  # 0.5% G-mean

    if len(fitness_history) >= MIN_IMPROVEMENT_WINDOW and current_generation >= MIN_IMPROVEMENT_WINDOW:
        # Compara fitness atual vs 30 gerações atrás
        fitness_current = fitness_history[-1]
        fitness_30_gens_ago = fitness_history[-MIN_IMPROVEMENT_WINDOW]
        improvement = fitness_current - fitness_30_gens_ago

        if improvement < MIN_IMPROVEMENT_THRESHOLD:
            logger.info(f"")
            logger.info(f"╔══════════════════════════════════════════════════════════════╗")
            logger.info(f"║ EARLY STOPPING LAYER 2: Melhoria Marginal                   ║")
            logger.info(f"╠══════════════════════════════════════════════════════════════╣")
            logger.info(f"║ Fitness 30 gens atrás: {fitness_30_gens_ago:.4f}                          ║")
            logger.info(f"║ Fitness atual:         {fitness_current:.4f}                          ║")
            logger.info(f"║ Melhoria:              {improvement:.4f} (< {MIN_IMPROVEMENT_THRESHOLD:.4f})          ║")
            logger.info(f"║ Geração atual:         {current_generation}                              ║")
            logger.info(f"║ Decisão:               PARAR (retorno decrescente severo)  ║")
            logger.info(f"╚══════════════════════════════════════════════════════════════╝")
            logger.info(f"")
            return True

    return False
```

#### **Camada 3: Stop Tradicional (Fallback)**

```python
def adaptive_early_stopping_layer3(stagnation_count, max_patience):
    """
    Camada 3: Stop tradicional (fallback se camadas 1-2 não ativaram).

    Critério: Estagnação > max_patience gerações
    """
    if stagnation_count >= max_patience:
        logger.info(f"")
        logger.info(f"╔══════════════════════════════════════════════════════════════╗")
        logger.info(f"║ EARLY STOPPING LAYER 3: Estagnação Longa                    ║")
        logger.info(f"╠══════════════════════════════════════════════════════════════╣")
        logger.info(f"║ Estagnação:    {stagnation_count} gerações (≥ {max_patience})                    ║")
        logger.info(f"║ Decisão:       PARAR (estagnação prolongada)                ║")
        logger.info(f"╚══════════════════════════════════════════════════════════════╝")
        logger.info(f"")
        return True

    return False
```

#### **Sistema Integrado**

```python
def should_stop_early(best_fitness, gmean, stagnation_count, current_generation,
                     fitness_history, early_stopping_patience):
    """
    Sistema de Early Stopping em 3 camadas (executadas em ordem).

    Returns: (should_stop: bool, reason: str)
    """
    # Layer 1: Elite satisfatório + estagnado
    if adaptive_early_stopping_layer1(best_fitness, gmean, stagnation_count, current_generation):
        return True, "Layer 1: Elite satisfatório + estagnado"

    # Layer 2: Melhoria marginal nas últimas 30 gens
    if adaptive_early_stopping_layer2(fitness_history, stagnation_count, current_generation):
        return True, "Layer 2: Melhoria marginal < 0.5% em 30 gens"

    # Layer 3: Estagnação longa (fallback)
    if adaptive_early_stopping_layer3(stagnation_count, early_stopping_patience):
        return True, "Layer 3: Estagnação prolongada"

    return False, ""
```

---

### **📊 COMPARAÇÃO: Early Stop Atual vs Adaptativo**

**Cenário real (Chunk 0, Pop=120, Gens=200):**

| Camada | Critério | Ativaria em Gen | Tempo Economizado |
|--------|----------|-----------------|-------------------|
| **Atual** | 20 gens estagnado | 126 | 0h (baseline) |
| **Layer 1** | 88% + 15 gens | **~20** | **~2.5h (78%)** |
| **Layer 2** | <0.5% melhoria/30 gens | **~35** | **~2.2h (69%)** |
| **Layer 3** | 20 gens estagnado | 126 | 0h (igual atual) |

**Economia estimada:**
- **Layer 1 ativa:** Para em ~Gen 20 → **Economiza 78% do tempo** (106 gens × 91s = 2.5h)
- **Layer 2 ativa:** Para em ~Gen 35 → **Economiza 69% do tempo** (91 gens × 91s = 2.2h)
- **Layer 3 (fallback):** Igual atual (Gen 126)

---

### **🔧 IMPLEMENTAÇÃO: Onde Modificar**

**Arquivo:** `ga.py`

**Localizar:** Loop principal do GA (~linha 600-1000)

**ANTES:**
```python
# Verificação de early stopping simples
if stagnation_count >= early_stopping_patience:
    logger.info(f"Parada antecipada na Geração {generation+1} devido à estagnação ({stagnation_count} gerações sem melhora).")
    break
```

**DEPOIS:**
```python
# Early stopping adaptativo em 3 camadas
should_stop, reason = should_stop_early(
    best_fitness=current_best_fitness,
    gmean=current_gmean,
    stagnation_count=stagnation_count,
    current_generation=generation + 1,
    fitness_history=fitness_history,  # Lista [gen1_fitness, gen2_fitness, ...]
    early_stopping_patience=early_stopping_patience
)

if should_stop:
    logger.info(f"Parada antecipada na Geração {generation+1}.")
    logger.info(f"Motivo: {reason}")
    break
```

**Config.yaml:**
```yaml
ga_params:
  early_stopping_patience: 20              # Layer 3 fallback
  early_stopping_layer1_gmean: 0.88        # Layer 1: Elite satisfatório
  early_stopping_layer1_patience: 15       # Layer 1: Estagnação mínima
  early_stopping_layer2_window: 30         # Layer 2: Janela de análise
  early_stopping_layer2_min_improvement: 0.005  # Layer 2: 0.5% mínimo
```

---

## 📊 RESUMO: IMPACTO ESPERADO DAS 3 PRIORIDADES

| Prioridade | Problema | Solução | Impacto Esperado |
|------------|----------|---------|------------------|
| **3: Crossover Balanceado** | Crossover cego perde info útil | 70% qualidade + 30% diversidade | +3-5% accuracy, mantém diversidade |
| **4: HC Inteligente** | HC gera variantes cegas (0% sucesso) | Extrai e injeta regras DT de erros | 30-50% taxa de sucesso HC |
| **5: Early Stop Adaptativo** | Para tarde (Gen 126 vs ideal ~20) | 3 camadas (satisfatório + marginal + estagnação) | **-70-80% tempo** (2.5h → 0.5h por chunk) |

**Ganho total estimado:**
- **Accuracy:** +3-5% (crossover + HC)
- **Tempo:** -70-80% (early stop)
- **Eficiência:** +400-500% (mesmo resultado em 1/5 do tempo)

---

## 🎯 PLANO DE IMPLEMENTAÇÃO (3 FASES)

### **Fase 1: Early Stopping Adaptativo (1-2h implementação)**
**Prioridade:** MÁXIMA (maior impacto/esforço)
**Arquivos:** `ga.py` (~50 linhas novas)
**Teste:** Executar 1 chunk RBF_Abrupt_Severe, verificar se para em ~Gen 20-35

### **Fase 2: HC Inteligente (2-3h implementação)**
**Prioridade:** ALTA
**Arquivos:** `hill_climbing_v2.py` (substituir `dt_error_correction`, ~150 linhas)
**Teste:** Verificar taxa de aprovação HC > 0% (atual 0%)

### **Fase 3: Crossover Balanceado (3-4h implementação)**
**Prioridade:** MÉDIA
**Arquivos:** `ga_operators_enhanced.py` (novo), `ga.py` (integração, ~200 linhas)
**Teste:** Comparar acc com/sem crossover balanceado (espera +2-3%)

**Tempo total:** ~6-9h de implementação + ~10h de testes = **2-3 dias de trabalho**

---

## ✅ CHECKLIST DE VALIDAÇÃO

**Após implementar cada prioridade, validar:**

### **Prioridade 3 (Crossover Balanceado):**
- [ ] Offspring tem 70% regras de qualidade + 30% diversas
- [ ] Diversidade populacional mantém-se > 40% até Gen 50
- [ ] Accuracy melhora +2-3% vs crossover cego
- [ ] Não converge prematuramente (fitness continua evoluindo até Gen 30+)

### **Prioridade 4 (HC Inteligente):**
- [ ] HC extrai regras do DT (verificar logs: "DT gerou X regras")
- [ ] HC substitui 20% regras piores (verificar logs: "Substituiu X regras fracas")
- [ ] Taxa de aprovação HC > 20% (vs 0% atual)
- [ ] Elite melhora após HC em pelo menos 30% das ativações

### **Prioridade 5 (Early Stop Adaptativo):**
- [ ] Layer 1 ativa em ~Gen 15-25 (elite 88%+)
- [ ] Layer 2 ativa em ~Gen 30-40 (melhoria <0.5%)
- [ ] Layer 3 (fallback) NÃO ativa (layers 1-2 pegam antes)
- [ ] Tempo por chunk reduz de ~3h para ~0.5-1h (70-80% economia)

---

## 🎉 CONCLUSÃO

**Foco nas 3 prioridades resolve os problemas identificados:**

1. **Crossover Balanceado:** Evita elitismo excessivo, mantém diversidade
2. **HC Inteligente:** Corrige erros de forma direcionada (vs cegas)
3. **Early Stop Adaptativo:** Economiza 70-80% do tempo desperdiçado

**Próximo passo:** Implementar Fase 1 (Early Stopping) → Testar → Fase 2 (HC) → Testar → Fase 3 (Crossover)

---

**📅 Data:** 13/10/2025
**👤 Autor:** Claude Code (Refinamento Colaborativo)
**📂 Arquivo:** REFINAMENTO_PRIORIDADES_3_4_5.md
