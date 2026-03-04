# Plano de Implementação - Melhorias GBML
## Baseado na Análise do Log RBF_Abrupt_Severe (6 chunks)

**Data**: 2025-10-19
**Contexto**: GBML G-mean 0.8156 vs ARF 0.8526 (gap de 3.7pp)
**Modo de operação mantido**: Batch learner (train-and-test), NÃO prequential

---

## RESUMO EXECUTIVO

Baseado na análise do experimento de 9 horas (6 chunks × 6000 instâncias), identificamos 3 melhorias prioritárias:

| Prioridade | Melhoria | Impacto Esperado | Tempo Est. |
|------------|----------|------------------|------------|
| 🔴 ALTA | Drift Severity Detection + Adaptive Reset | +2-4pp no G-mean | 3-4h |
| 🔴 ALTA | Intelligent Mutation Multi-Strategy | +1-2pp no G-mean | 4-5h |
| 🟡 MÉDIA | Ensemble Voting (top-3) | +0.5-1pp no G-mean | 2-3h |

**Objetivo**: Reduzir gap de 3.7pp para ~1-2pp, especialmente em drifts severos (chunk 4→5 teve gap de 4.4pp).

---

## MELHORIA 1: DRIFT SEVERITY DETECTION + ADAPTIVE RESET
**Prioridade**: 🔴 ALTA
**Impacto esperado**: +2-4pp no G-mean
**Tempo de implementação**: 3-4 horas

### Problema Identificado

Do log (chunk 4→5):
```
Chunk 3→4: GBML 0.9065 vs ARF 0.9280 (-2.15pp)
Chunk 4→5: GBML 0.4645 vs ARF 0.5087 (-4.42pp) ← SEVERE DRIFT
Chunk 5→6: GBML 0.8336 vs ARF 0.8911 (-5.75pp) ← Recuperação lenta
```

**Problema**: GBML herdou 55% da população anterior (performance GOOD no chunk 3→4), mas o drift severo tornou essa herança prejudicial. O modelo demorou 2 chunks para se recuperar.

**Solução**: Classificar severidade do drift e ajustar herança dinamicamente.

---

### Implementação

#### PASSO 1.1: Adicionar classificação de severidade do drift
**Arquivo**: `main.py`
**Localização**: Após linha 704 (onde já existe detecção de performance drop)

**Código a adicionar**:
```python
# main.py, linha ~705 (após cálculo de performance_drop)

# Classificação de severidade do drift
drift_severity = 'NONE'
if performance_drop > 0.20:
    drift_severity = 'SEVERE'
    logger.warning(f"🔴 SEVERE DRIFT detected: {previous_gmean:.3f} → {current_gmean:.3f} (drop: {performance_drop:.1%})")
    best_ever_memory.clear()
    logger.info("→ Memory cleared due to severe drift")
elif performance_drop > 0.10:
    drift_severity = 'MODERATE'
    logger.warning(f"🟡 MODERATE DRIFT detected: {previous_gmean:.3f} → {current_gmean:.3f} (drop: {performance_drop:.1%})")
    # Reduz best_ever_memory pela metade (mantém apenas os melhores)
    if best_ever_memory:
        best_ever_memory = sorted(best_ever_memory, key=lambda ind: ind.fitness, reverse=True)[:len(best_ever_memory)//2]
        logger.info(f"→ Memory reduced to top {len(best_ever_memory)} individuals")
elif performance_drop > 0.05:
    drift_severity = 'MILD'
    logger.info(f"🟢 MILD DRIFT detected: {previous_gmean:.3f} → {current_gmean:.3f} (drop: {performance_drop:.1%})")
else:
    logger.info(f"✓ STABLE: {previous_gmean:.3f} → {current_gmean:.3f}")
```

#### PASSO 1.2: Ajustar population_carry_over_rate dinamicamente
**Arquivo**: `main.py`
**Localização**: Linha 989 (onde calcula `num_to_keep`)

**Código ANTES (linha 989)**:
```python
num_to_keep = int(len(final_population) * population_carry_over_rate)
```

**Código DEPOIS**:
```python
# Ajuste dinâmico da taxa de herança baseado em severidade do drift
base_carry_over = population_carry_over_rate  # 0.5 padrão
if 'drift_severity' in locals():
    if drift_severity == 'SEVERE':
        # Drift severo: ZERA herança (reset completo)
        adjusted_carry_over = 0.0
        logger.warning(f"→ Inheritance DISABLED due to SEVERE drift (was {base_carry_over:.0%})")
    elif drift_severity == 'MODERATE':
        # Drift moderado: REDUZ herança para 25%
        adjusted_carry_over = 0.25
        logger.info(f"→ Inheritance REDUCED to {adjusted_carry_over:.0%} due to MODERATE drift (was {base_carry_over:.0%})")
    elif drift_severity == 'MILD':
        # Drift leve: MANTÉM herança padrão
        adjusted_carry_over = base_carry_over
    else:  # STABLE
        # Estável: AUMENTA herança para 65%
        adjusted_carry_over = 0.65
        logger.info(f"→ Inheritance INCREASED to {adjusted_carry_over:.0%} (stable performance)")
else:
    adjusted_carry_over = base_carry_over

num_to_keep = int(len(final_population) * adjusted_carry_over)
```

#### PASSO 1.3: Passar drift_severity para GA
**Arquivo**: `main.py`
**Localização**: Linha ~573 (chamada `run_ga_chunk`)

**Adicionar parâmetro**:
```python
# Linha ~573 (antes da chamada run_ga_chunk)
drift_severity_to_pass = drift_severity if 'drift_severity' in locals() else 'NONE'

# Na chamada run_ga_chunk (linha ~620):
result = run_ga_chunk(
    ...,
    drift_severity=drift_severity_to_pass,  # ← NOVO
    ...
)
```

#### PASSO 1.4: Ajustar recipes no ga.py baseado em severidade
**Arquivo**: `ga.py`
**Localização**: Função `run_ga_chunk`, linha ~573 (adicionar parâmetro)

**Adicionar parâmetro na assinatura**:
```python
def run_ga_chunk(
    ...,
    drift_severity='NONE',  # ← NOVO parâmetro
    ...
):
```

**Localização**: Linha 591-598 (onde define recipes)

**Código ANTES**:
```python
if performance_label == 'good':
    recipe = {'seeded': 0.1, 'mutants': 0.15, 'memory': 0.3, 'prev_pop': 0.25, 'random': 0.2}
elif performance_label == 'medium':
    recipe = {'seeded': 0.2, 'mutants': prev_best_mutant_ratio, 'memory': 0.2, 'prev_pop': 0.2}
    recipe['random'] = 1.0 - sum(recipe.values())
else:  # 'bad'
    recipe = {'seeded': 0.5, 'mutants': 0.1, 'memory': 0.05, 'prev_pop': 0.05, 'random': 0.3}
```

**Código DEPOIS**:
```python
# Base recipes (mantém as originais)
if performance_label == 'good':
    recipe = {'seeded': 0.1, 'mutants': 0.15, 'memory': 0.3, 'prev_pop': 0.25, 'random': 0.2}
elif performance_label == 'medium':
    recipe = {'seeded': 0.2, 'mutants': prev_best_mutant_ratio, 'memory': 0.2, 'prev_pop': 0.2}
    recipe['random'] = 1.0 - sum(recipe.values())
else:  # 'bad'
    recipe = {'seeded': 0.5, 'mutants': 0.1, 'memory': 0.05, 'prev_pop': 0.05, 'random': 0.3}

# Ajuste baseado em drift severity (SOBRESCREVE se necessário)
if drift_severity == 'SEVERE':
    # RESET TOTAL: Muito seeding, zero herança
    recipe = {'seeded': 0.6, 'mutants': 0.1, 'memory': 0.0, 'prev_pop': 0.0, 'random': 0.3}
    logger.warning(f"🔴 Recipe adjusted for SEVERE drift: {recipe}")
elif drift_severity == 'MODERATE':
    # RESET PARCIAL: Aumenta seeding, reduz herança
    recipe = {'seeded': 0.3, 'mutants': 0.15, 'memory': 0.1, 'prev_pop': 0.1, 'random': 0.35}
    logger.info(f"🟡 Recipe adjusted for MODERATE drift: {recipe}")
elif drift_severity == 'MILD':
    # MANTÉM recipe original
    logger.info(f"🟢 Recipe kept (MILD drift): {recipe}")
# Se NONE ou STABLE, usa recipe original baseado em performance_label
```

---

### Testes para Melhoria 1

**Teste 1: Drift severo detectado e reset aplicado**
```bash
python3 compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 3
```

**Saída esperada no log**:
```
Chunk 2→3: 🔴 SEVERE DRIFT detected: 0.9065 → 0.4645 (drop: 48.7%)
→ Memory cleared due to severe drift
→ Inheritance DISABLED due to SEVERE drift (was 50%)
🔴 Recipe adjusted for SEVERE drift: {'seeded': 0.6, 'mutants': 0.1, 'memory': 0.0, 'prev_pop': 0.0, 'random': 0.3}
```

**Teste 2: Drift moderado detectado**
```bash
python3 compare_gbml_vs_river.py --stream SEA_Gradual --chunks 3
```

**Saída esperada**:
```
Chunk 1→2: 🟡 MODERATE DRIFT detected: 0.8800 → 0.7500 (drop: 14.8%)
→ Memory reduced to top 5 individuals
→ Inheritance REDUCED to 25% due to MODERATE drift (was 50%)
```

**Validação de sucesso**:
- ✅ Gap no chunk 4→5 reduzido de 4.4pp para ~2-3pp
- ✅ Recuperação no chunk 5→6 mais rápida (gap < 4pp)

---

## MELHORIA 2: INTELLIGENT MUTATION MULTI-STRATEGY
**Prioridade**: 🔴 ALTA
**Impacto esperado**: +1-2pp no G-mean
**Tempo de implementação**: 4-5 horas

### Problema Identificado

Do log (chunk 0→1, análise HC v2):
```
Mutação: 8 variantes (61.5% das 13 propostas)
HC v2: 4 variantes aprovadas (30.8% das 13 propostas)
```

**Problema**: A mutação atual é básica/legacy (ga_operators.py) e gera muitas variantes de baixa qualidade. O HC v2 corrige algumas delas (30% de aprovação), mas seria melhor ter mutação mais inteligente desde o início.

**Análise adicional do código**:
- `ga_operators.py` linha ~120-180: Mutação atual é puramente aleatória (muda limites/operadores sem considerar erros)
- Não há foco em features que causam erros
- Não há análise de quais regras estão falhando

**Solução**: Criar mutação inteligente com 3 estratégias baseadas em análise de erros.

---

### Implementação

#### PASSO 2.1: Criar função de análise de erros
**Arquivo**: `ga_operators.py`
**Localização**: Antes da função `mutate` (linha ~120)

**Código a adicionar**:
```python
def analyze_rule_errors(individual, X_train, y_train):
    """
    Analisa quais regras estão falhando e em quais features.

    Returns:
        dict: {
            'error_features': [list of feature indices with most errors],
            'weak_rules': [list of rule indices with low accuracy],
            'coverage_gaps': [list of feature indices with low coverage]
        }
    """
    if not individual.rules or len(X_train) == 0:
        return {'error_features': [], 'weak_rules': [], 'coverage_gaps': []}

    import numpy as np

    # Predições por regra
    errors_by_rule = []
    for i, rule in enumerate(individual.rules):
        matches = np.array([rule.match(x) for x in X_train])
        if matches.sum() == 0:
            errors_by_rule.append((i, 1.0))  # Regra não cobre nada = 100% erro
            continue

        y_pred = np.array([rule.output for x in X_train if rule.match(x)])
        y_true = y_train[matches]
        accuracy = (y_pred == y_true).mean()
        errors_by_rule.append((i, 1.0 - accuracy))

    # Identifica regras fracas (accuracy < 60%)
    weak_rules = [i for i, err in errors_by_rule if err > 0.4]

    # Identifica features com mais erros
    feature_errors = {}
    for rule_idx, error_rate in errors_by_rule:
        if error_rate > 0.3:  # Apenas regras com >30% erro
            rule = individual.rules[rule_idx]
            for condition in rule.conditions:
                feat_idx = condition.feature
                feature_errors[feat_idx] = feature_errors.get(feat_idx, 0) + error_rate

    # Top 3 features com mais erros
    error_features = sorted(feature_errors.keys(), key=lambda f: feature_errors[f], reverse=True)[:3]

    # Identifica features com baixa cobertura (aparecem em <30% das regras)
    feature_coverage = {f: 0 for f in range(len(X_train[0]))}
    for rule in individual.rules:
        for condition in rule.conditions:
            feature_coverage[condition.feature] += 1

    min_coverage = len(individual.rules) * 0.3
    coverage_gaps = [f for f, count in feature_coverage.items() if count < min_coverage]

    return {
        'error_features': error_features,
        'weak_rules': weak_rules,
        'coverage_gaps': coverage_gaps
    }
```

#### PASSO 2.2: Criar mutação inteligente com multi-estratégia
**Arquivo**: `ga_operators.py`
**Localização**: Substituir função `mutate` atual (linha ~120-180)

**Código ANTES** (mutate atual - básico):
```python
def mutate(individual, mutation_rate, num_features, feature_ranges, ...):
    # Mutação puramente aleatória
    for rule in individual.rules:
        for condition in rule.conditions:
            if random.random() < mutation_rate:
                # Muda limite aleatoriamente
                condition.threshold += random.gauss(0, sigma)
```

**Código DEPOIS** (mutate inteligente):
```python
def mutate(individual, mutation_rate, num_features, feature_ranges,
           X_train=None, y_train=None, strategy='intelligent', **kwargs):
    """
    Mutação inteligente com 3 estratégias:
    - ERROR_FOCUSED (40%): Foca em features que causam erros
    - FEATURE_GUIDED (40%): Preenche gaps de cobertura
    - RANDOM (20%): Exploração aleatória (legacy)

    Args:
        strategy: 'intelligent' ou 'random' (legacy)
        X_train, y_train: Dados para análise de erros (necessário para 'intelligent')
    """
    if not individual.rules:
        return individual

    # Se strategy='random' ou sem dados, usa mutação legacy
    if strategy == 'random' or X_train is None or y_train is None:
        return _mutate_random_legacy(individual, mutation_rate, num_features, feature_ranges, **kwargs)

    # Mutação inteligente
    error_analysis = analyze_rule_errors(individual, X_train, y_train)

    # Define estratégia para cada regra (40% ERROR, 40% FEATURE, 20% RANDOM)
    for rule_idx, rule in enumerate(individual.rules):
        if random.random() > mutation_rate:
            continue  # Não muta essa regra

        # Escolhe estratégia
        rand = random.random()
        if rand < 0.4:
            # Estratégia 1: ERROR_FOCUSED
            if rule_idx in error_analysis['weak_rules'] and error_analysis['error_features']:
                # Regra fraca: muda condições nas features problemáticas
                _mutate_error_focused(rule, error_analysis['error_features'], feature_ranges, **kwargs)
            else:
                # Regra OK: mutação leve
                _mutate_random_legacy_single_rule(rule, mutation_rate * 0.5, feature_ranges, **kwargs)

        elif rand < 0.8:
            # Estratégia 2: FEATURE_GUIDED
            if error_analysis['coverage_gaps']:
                # Adiciona condições em features com baixa cobertura
                _mutate_feature_guided(rule, error_analysis['coverage_gaps'], num_features, feature_ranges, **kwargs)
            else:
                _mutate_random_legacy_single_rule(rule, mutation_rate, feature_ranges, **kwargs)

        else:
            # Estratégia 3: RANDOM (exploração)
            _mutate_random_legacy_single_rule(rule, mutation_rate, feature_ranges, **kwargs)

    return individual


def _mutate_error_focused(rule, error_features, feature_ranges, **kwargs):
    """
    Muta condições nas features que causam mais erros.
    Estratégia: Ajusta thresholds de forma mais agressiva.
    """
    sigma = kwargs.get('mutation_sigma', 0.2)

    for condition in rule.conditions:
        if condition.feature in error_features:
            # Mutação agressiva: 2x sigma
            feat_min, feat_max = feature_ranges[condition.feature]
            delta = random.gauss(0, sigma * 2.0) * (feat_max - feat_min)
            condition.threshold += delta
            condition.threshold = max(feat_min, min(feat_max, condition.threshold))


def _mutate_feature_guided(rule, coverage_gaps, num_features, feature_ranges, **kwargs):
    """
    Adiciona ou modifica condições para cobrir features negligenciadas.
    Estratégia: Aumenta cobertura de features.
    """
    from ga_operators import Condition  # Assumindo que Condition está definido

    # Escolhe uma feature com gap de cobertura
    target_feature = random.choice(coverage_gaps)

    # Verifica se já existe condição nessa feature
    existing = [c for c in rule.conditions if c.feature == target_feature]

    if existing:
        # Já existe: ajusta threshold
        condition = random.choice(existing)
        feat_min, feat_max = feature_ranges[target_feature]
        sigma = kwargs.get('mutation_sigma', 0.2)
        delta = random.gauss(0, sigma) * (feat_max - feat_min)
        condition.threshold += delta
        condition.threshold = max(feat_min, min(feat_max, condition.threshold))
    else:
        # Não existe: adiciona nova condição (se rule tem espaço)
        max_conditions = kwargs.get('max_conditions_per_rule', 5)
        if len(rule.conditions) < max_conditions:
            feat_min, feat_max = feature_ranges[target_feature]
            new_threshold = random.uniform(feat_min, feat_max)
            new_operator = random.choice(['<=', '>'])
            new_condition = Condition(target_feature, new_operator, new_threshold)
            rule.conditions.append(new_condition)


def _mutate_random_legacy_single_rule(rule, mutation_rate, feature_ranges, **kwargs):
    """
    Mutação legacy para uma regra (mantém compatibilidade).
    """
    sigma = kwargs.get('mutation_sigma', 0.2)

    for condition in rule.conditions:
        if random.random() < mutation_rate:
            feat_min, feat_max = feature_ranges[condition.feature]
            delta = random.gauss(0, sigma) * (feat_max - feat_min)
            condition.threshold += delta
            condition.threshold = max(feat_min, min(feat_max, condition.threshold))

    # 10% chance de mudar output da regra
    if random.random() < 0.1:
        rule.output = 1 - rule.output


def _mutate_random_legacy(individual, mutation_rate, num_features, feature_ranges, **kwargs):
    """
    Mutação legacy completa (mantém compatibilidade com código antigo).
    """
    for rule in individual.rules:
        _mutate_random_legacy_single_rule(rule, mutation_rate, feature_ranges, **kwargs)
    return individual
```

#### PASSO 2.3: Integrar mutação inteligente no GA
**Arquivo**: `ga.py`
**Localização**: Linha ~450-470 (onde chama `mutate`)

**Código ANTES**:
```python
offspring = mutate(offspring, mutation_rate, num_features, feature_ranges, ...)
```

**Código DEPOIS**:
```python
# Passa dados de treino para mutação inteligente
offspring = mutate(
    offspring,
    mutation_rate,
    num_features,
    feature_ranges,
    X_train=X_train,  # ← NOVO
    y_train=y_train,  # ← NOVO
    strategy='intelligent',  # ← NOVO
    ...
)
```

#### PASSO 2.4: Adicionar parâmetro no config.yaml
**Arquivo**: `config.yaml`
**Localização**: Seção `genetic_algorithm_params`

**Adicionar**:
```yaml
genetic_algorithm_params:
  ...
  mutation_strategy: 'intelligent'  # 'intelligent' ou 'random'
  mutation_error_focus_ratio: 0.4   # 40% ERROR_FOCUSED
  mutation_feature_guided_ratio: 0.4  # 40% FEATURE_GUIDED
  mutation_random_ratio: 0.2         # 20% RANDOM
```

---

### Testes para Melhoria 2

**Teste 1: Mutação inteligente vs legacy**
```bash
# Com mutação inteligente
python3 compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 2

# Trocar config.yaml para mutation_strategy: 'random'
python3 compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 2
```

**Saída esperada no log (intelligent)**:
```
Gen 5: Mutation applied to 12 individuals
  - ERROR_FOCUSED: 5 individuals (weak rules improved)
  - FEATURE_GUIDED: 5 individuals (coverage increased)
  - RANDOM: 2 individuals (exploration)
```

**Validação de sucesso**:
- ✅ Taxa de aprovação HC v2 cai de 30% para ~15-20% (menos variantes ruins)
- ✅ Diversidade de features aumenta (coverage_gaps reduzidos)
- ✅ G-mean melhora +1-2pp

---

## MELHORIA 3: ENSEMBLE VOTING (TOP-3 INDIVIDUALS)
**Prioridade**: 🟡 MÉDIA
**Impacto esperado**: +0.5-1pp no G-mean
**Tempo de implementação**: 2-3 horas

### Problema Identificado

Do log (chunk 0):
```
Best individual: 8 rules, Gmean Train=0.931, Precision=0.933, Recall=0.929
```

**Problema**: GBML usa apenas o melhor indivíduo (elite) para predições. Modelos como Random Forest/ARF usam ensemble (múltiplas árvores votando), o que aumenta robustez.

**Oportunidade**: GBML já tem uma população de 200 indivíduos. Os top-3 podem ter regras complementares que melhoram predições via votação.

---

### Implementação

#### PASSO 3.1: Modificar retorno do run_ga_chunk
**Arquivo**: `ga.py`
**Localização**: Final da função `run_ga_chunk` (linha ~730)

**Código ANTES**:
```python
return {
    'best_individual': sorted_final_pop[0],
    'final_population': final_population,
    ...
}
```

**Código DEPOIS**:
```python
# Retorna top-3 para ensemble
top_3_individuals = sorted_final_pop[:3] if len(sorted_final_pop) >= 3 else sorted_final_pop

return {
    'best_individual': sorted_final_pop[0],
    'top_3_individuals': top_3_individuals,  # ← NOVO
    'final_population': final_population,
    ...
}
```

#### PASSO 3.2: Criar função de ensemble voting
**Arquivo**: `utils.py`
**Localização**: Adicionar no final do arquivo

**Código a adicionar**:
```python
def ensemble_predict(top_individuals, instance):
    """
    Predição por ensemble voting (maioria simples).

    Args:
        top_individuals: Lista de indivíduos (top-3)
        instance: Instância a predizer

    Returns:
        int: Classe predita (0 ou 1)
    """
    if not top_individuals:
        return 0

    votes = []
    for individual in top_individuals:
        # Predição do indivíduo (usa lógica de match de regras)
        prediction = individual.predict(instance)  # Assumindo que Individual tem método predict
        votes.append(prediction)

    # Maioria simples
    return 1 if sum(votes) > len(votes) / 2 else 0


def ensemble_predict_proba(top_individuals, instance):
    """
    Predição probabilística por ensemble.

    Returns:
        float: Probabilidade da classe 1 (média dos votos)
    """
    if not top_individuals:
        return 0.5

    votes = [ind.predict(instance) for ind in top_individuals]
    return sum(votes) / len(votes)
```

#### PASSO 3.3: Integrar ensemble no loop de teste
**Arquivo**: `main.py`
**Localização**: Linha ~396 (loop de teste prequential)

**Código ANTES**:
```python
# Teste no chunk atual
for i, instance in enumerate(X_test):
    prediction = best_individual.predict(instance)  # Usa apenas elite
    y_pred_test.append(prediction)
```

**Código DEPOIS**:
```python
# Importar função ensemble
from utils import ensemble_predict

# Teste no chunk atual com ensemble
use_ensemble = config.get('use_ensemble_voting', True)  # Novo parâmetro
top_3 = result.get('top_3_individuals', [best_individual])  # Fallback para elite

for i, instance in enumerate(X_test):
    if use_ensemble and len(top_3) >= 2:
        prediction = ensemble_predict(top_3, instance)
    else:
        prediction = best_individual.predict(instance)
    y_pred_test.append(prediction)
```

#### PASSO 3.4: Adicionar método predict no Individual
**Arquivo**: Procurar onde a classe `Individual` está definida (provavelmente `ga.py` ou `ga_operators.py`)

**Adicionar método**:
```python
class Individual:
    def __init__(self, ...):
        ...

    def predict(self, instance):
        """
        Prediz classe para uma instância.
        Retorna output da primeira regra que faz match.
        """
        for rule in self.rules:
            if rule.match(instance):
                return rule.output

        # Se nenhuma regra faz match, retorna classe padrão (maioria no treino)
        return self.default_class if hasattr(self, 'default_class') else 0
```

#### PASSO 3.5: Adicionar parâmetro no config.yaml
**Arquivo**: `config.yaml`

**Adicionar**:
```yaml
# Ensemble voting
use_ensemble_voting: true  # true = usa top-3, false = usa apenas elite
ensemble_size: 3  # Número de indivíduos no ensemble (padrão 3)
```

---

### Testes para Melhoria 3

**Teste 1: Ensemble vs Elite**
```bash
# Com ensemble (config: use_ensemble_voting: true)
python3 compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 2

# Sem ensemble (config: use_ensemble_voting: false)
python3 compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 2
```

**Saída esperada no log (ensemble)**:
```
[PREQUENTIAL] Testando no chunk 1 com ENSEMBLE VOTING (top-3)
  Individual 1: 8 rules, fitness=0.931
  Individual 2: 7 rules, fitness=0.928
  Individual 3: 9 rules, fitness=0.925
Chunk 0→1: GBML 0.9130 vs ARF 0.9227 (gap -0.97pp)
```

**Validação de sucesso**:
- ✅ G-mean aumenta +0.5-1pp com ensemble
- ✅ Robustez melhora (menos variabilidade entre runs)

---

## ROADMAP DE IMPLEMENTAÇÃO

### Semana 1: Drift Severity Detection (Melhoria 1)
- **Dia 1-2**: Implementar PASSO 1.1 a 1.3 (classificação + ajuste herança)
- **Dia 3**: Implementar PASSO 1.4 (recipes adaptados)
- **Dia 4**: Testes + validação

**Entregável**: Sistema de detecção de drift com reset adaptativo funcionando

### Semana 2: Intelligent Mutation (Melhoria 2)
- **Dia 1-2**: Implementar PASSO 2.1 (análise de erros)
- **Dia 3-4**: Implementar PASSO 2.2 (mutação multi-estratégia)
- **Dia 5**: Integração (PASSO 2.3-2.4) + testes

**Entregável**: Mutação inteligente com 3 estratégias funcionando

### Semana 3: Ensemble Voting (Melhoria 3)
- **Dia 1**: Implementar PASSO 3.1-3.2 (retorno top-3 + função ensemble)
- **Dia 2**: Implementar PASSO 3.3-3.4 (integração + método predict)
- **Dia 3**: Testes comparativos

**Entregável**: Sistema de ensemble voting funcionando

### Semana 4: Testes Finais
- **Dia 1-2**: Experimento completo com todas melhorias (10 chunks)
- **Dia 3**: Análise de resultados + comparação com baseline
- **Dia 4-5**: Ajustes finos + documentação

**Entregável**: Relatório de resultados + código final

---

## MÉTRICAS DE SUCESSO

### Objetivo Principal
**Reduzir gap GBML vs ARF de 3.7pp para ~1-2pp**

### Métricas por Melhoria

| Melhoria | Métrica | Baseline | Meta | Como medir |
|----------|---------|----------|------|------------|
| Drift Detection | Gap chunk 4→5 | 4.4pp | 2.5pp | G-mean chunk severo |
| Intelligent Mutation | Taxa aprovação HC | 30% | <20% | % variantes aprovadas |
| Ensemble Voting | G-mean médio | 0.8156 | 0.8300 | Média 6 chunks |

### Teste Final (Todas melhorias combinadas)
```bash
python3 compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 10
```

**Resultados esperados**:
- ✅ G-mean médio: 0.8300-0.8400 (vs baseline 0.8156)
- ✅ Gap vs ARF: 1.5-2.0pp (vs baseline 3.7pp)
- ✅ Gap chunk severo: <3.0pp (vs baseline 4.4pp)
- ✅ Tempo de recuperação: 1 chunk (vs baseline 2 chunks)

---

## CONSIDERAÇÕES TÉCNICAS

### 1. Compatibilidade com código existente
- Todas as melhorias são **retrocompatíveis**
- Parâmetros `strategy='random'` e `use_ensemble_voting: false` permitem desabilitar
- Mutação legacy mantida em `_mutate_random_legacy`

### 2. Impacto no tempo de execução
- **Drift Detection**: +0% (apenas lógica condicional)
- **Intelligent Mutation**: +10-15% (análise de erros adicional)
- **Ensemble Voting**: +200% no teste (3x predições), mas teste é <5% do tempo total
- **Total**: ~+5-8% no tempo total de experimento

### 3. Memória
- **Drift Detection**: +0 MB (apenas variáveis escalares)
- **Intelligent Mutation**: +2-5 MB (análise de erros temporária)
- **Ensemble Voting**: +10 MB (2 indivíduos adicionais no top-3)
- **Total**: Negligível (<20 MB em experimentos de 36k instâncias)

### 4. Logs e debugging
- Todos os PASSOS incluem logs detalhados (`logger.info`, `logger.warning`)
- Formato: 🔴/🟡/🟢 para facilitar análise visual
- Métricas rastreáveis para cada estratégia de mutação

---

## PRÓXIMOS PASSOS

1. **Revisar este plano** e esclarecer dúvidas
2. **Decidir ordem de implementação** (recomendo: 1 → 2 → 3)
3. **Implementar Melhoria 1** (drift detection - mais impacto)
4. **Testar isoladamente** antes de combinar
5. **Experimento final** com todas melhorias

---

**Observação importante**: Este plano mantém o GBML como **batch learner** (train-and-test). As melhorias focam em:
- Melhor adaptação a drifts via reset inteligente
- Melhor qualidade das variantes geradas (mutação)
- Maior robustez via ensemble

Não convertemos para prequential verdadeiro (test-then-train incremental) como ERulesD2S/HAT/ARF.
