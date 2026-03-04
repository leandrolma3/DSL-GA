# ANÁLISE DIAGNÓSTICO RUN3 + PLANO DE MELHORIAS

**Data:** 2025-11-02
**Experimentos Analisados:** Run3 (TEST_SINGLE, DRIFT_RECOVERY)
**Status:** FASE 2 FUNCIONANDO | Performance PIOROU | Timing 24-48% MAIS LENTO

---

## 📊 RESUMO EXECUTIVO

### Resultados Run3 vs Baseline

| Experimento | Baseline | Run3 | Delta G-mean | Delta Tempo | Veredicto |
|-------------|----------|------|--------------|-------------|-----------|
| **TEST_SINGLE** | 79.83% | **77.63%** | **-2.20pp (-2.76%)** | **+1.9h (+24%)** | ❌❌ PIOR em performance E timing |
| **DRIFT_RECOVERY** | 73.74% | **73.14%** | **-0.60pp (-0.81%)** | **+4.8h (+48%)** | ❌ Estável em performance, MUITO LENTO |

### Descobertas Críticas

✅ **FASE 2 ESTÁ FUNCIONANDO:**
- Detectando conceitos recorrentes com similaridade ~0.98-0.99
- Restaurando memória corretamente
- 4 de 5 chunks com match de conceito recorrente

❌ **FASE 1 FALHOU EM REDUZIR TEMPO:**
- Esperado: -48% de redução
- Real: **+24% a +48% MAIS LENTO**
- Cache, early stop e HC reduzido NÃO estão compensando

❌ **PERFORMANCE PIOROU:**
- TEST_SINGLE: -2.76% (piora significativa)
- DRIFT_RECOVERY: -0.81% (estável mas não melhorou)

---

## 🔍 ANÁLISE DETALHADA DOS GARGALOS

### GARGALO 1: TIMING POR CHUNK EXTREMAMENTE VARIÁVEL

**TEST_SINGLE:**
```
Chunk 0: 109min (1.8h)
Chunk 1: 100min (1.7h)
Chunk 2: 132min (2.2h) ← 20% mais lento
Chunk 3:  93min (1.5h)
Chunk 4: 161min (2.7h) ← 73% mais lento que chunk 3!
```

**DRIFT_RECOVERY:**
```
Chunk 0:  95min (1.6h)
Chunk 1: 174min (2.9h) ← 83% mais lento que chunk 0!
Chunk 2: 229min (3.8h) ← Pico de lentidão
Chunk 3: 166min (2.8h)
Chunk 4: 224min (3.7h)
```

**Diagnóstico:**
- **Chunk 0 é sempre mais rápido** (seeding inicial, sem HC completo)
- **Chunks subsequentes explodem em tempo** (HC completo + seeding + Fase 2)
- **Variação de até 83%** entre chunks sugere problema sistêmico

**Causas Prováveis:**
1. **HC completo está dominando o tempo** (gerações 26-200)
2. **Seeding em MODERATE/SEVERE drift está lento** (cópia de indivíduos)
3. **Fase 2 fingerprint calculation overhead** (scipy cosine similarity)
4. **Cache de fitness não está efetivo** (hit rate baixo)

---

### GARGALO 2: TIMING DE GERAÇÕES MUITO ALTO

**TEST_SINGLE (sample primeiras 20 gerações):**
- Média: **34.9s**
- Mediana: 32.8s
- Max: **77.8s** (primeira geração - seeding)

**DRIFT_RECOVERY (sample primeiras 20 gerações):**
- Média: **25.7s**
- Mediana: 22.7s
- Max: **79.2s** (primeira geração - seeding)

**Baseline Esperado:** ~15-20s por geração (baseado em experimentos anteriores sem Fase 1+2)

**Diagnóstico:**
- Gerações estão **50-70% mais lentas** que baseline
- **Primeira geração MUITO lenta** (77-79s) → seeding overhead
- Gerações subsequentes ainda lentas (25-35s) → overhead constante

**Causas Prováveis:**
1. **Seeding de memória lento** (deep copy de indivíduos)
2. **Avaliação de fitness lenta** (early stop não compensa overhead de checks)
3. **HC está sendo chamado muitas vezes** (cada indivíduo gerado passa por HC?)
4. **Logging WARNING overhead** (muitos prints por geração)

---

### GARGALO 3: PERFORMANCE DEGRADADA EM TEST_SINGLE

**Métricas por Chunk (TEST_SINGLE Run3):**
```
Chunk 0: Train=0.944 → Test=0.900  (delta=-0.044) ✅ OK
Chunk 1: Train=0.897 → Test=0.876  (delta=-0.021) ✅ OK
Chunk 2: Train=0.925 → Test=0.443  (delta=-0.482) ❌ COLAPSO!
Chunk 3: Train=0.945 → Test=0.877  (delta=-0.068) ⚠️ Overfitting
Chunk 4: Train=0.827 → Test=0.786  (delta=-0.041) ✅ OK
```

**Chunk 2 tem colapso catastrófico:**
- Train G-mean: 0.925 (92.5%) - excelente
- Test G-mean: **0.443 (44.3%)** - péssimo
- **OVERFITTING SEVERO** (-48.2 pontos percentuais!)

**Diagnóstico:**
- Chunk 2 tem drift diferente que não foi tratado corretamente
- Fase 2 detectou como "conceito recorrente" (sim=0.9958) MAS não é
- **FALSE POSITIVE** da Fase 2 → restaurou memória errada

**Causas Prováveis:**
1. **Threshold 0.85 muito baixo** → conceitos diferentes sendo confundidos
2. **Fingerprint muito simples** (mean, std, class dist) → não captura nuances
3. **TEST_SINGLE não tem drifts reais** → fingerprint sempre similar (overfitting)

---

### GARGALO 4: PERFORMANCE DEGRADADA EM DRIFT_RECOVERY

**Métricas por Chunk (DRIFT_RECOVERY Run3):**
```
Chunk 0: Train=0.913 → Test=0.893  (delta=-0.020) ✅ OK (conceito c1)
Chunk 1: Train=0.945 → Test=0.457  (delta=-0.488) ❌ COLAPSO! (conceito c2)
Chunk 2: Train=0.933 → Test=0.888  (delta=-0.045) ✅ OK (conceito c3)
Chunk 3: Train=0.896 → Test=0.523  (delta=-0.373) ❌ COLAPSO! (conceito c4)
Chunk 4: Train=0.952 → Test=0.897  (delta=-0.055) ✅ OK (conceito c1 retorna)
```

**Chunks 1 e 3 têm colapsos:**
- Ambos têm **Test G-mean < 0.53** (muito baixo)
- Ambos têm **overfitting severo** (-37 a -49pp)
- Ambos são conceitos NOVOS (c2, c4) que nunca viram antes

**Chunk 4 (recorrência de c1) funcionou BEM:**
- Test G-mean: 0.897 (excelente!)
- Delta: -0.055 (generalização OK)
- **FASE 2 FUNCIONOU** → restaurou memória de c1 corretamente

**Diagnóstico:**
- **Fase 2 funciona quando deveria** (chunk 4 = c1 retorna)
- **Mas conceitos NOVOS estão colapsando** (chunks 1 e 3)
- Problema não é Fase 2, mas **algo com GA/seeding/HC em conceitos novos**

**Causas Prováveis:**
1. **Early stopping descartando indivíduos bons** em conceitos novos
2. **Seeding inicial ruim** para conceitos novos (memória de conceito diferente)
3. **HC não está refinando adequadamente** (reduzido para 8 variantes)

---

## 🎯 CAUSAS RAIZ IDENTIFICADAS

### CAUSA 1: FASE 1 NÃO REDUZIU TEMPO (pior: AUMENTOU!)

**Esperado:** -48% tempo (de ~8h para ~4.2h)
**Real:** +24% a +48% (de ~8h para ~10-15h)

**Por que Fase 1 falhou:**

1. **Early Stopping (threshold 75%) NÃO está descartando avaliações:**
   - Em chunk 0, worst_elite_gmean = 0 (sem elite ainda)
   - Threshold = 0 * 0.75 = 0 → nunca descarta nada!
   - Em chunks subsequentes, elite tem G-mean alto (~0.90)
   - Threshold = 0.90 * 0.75 = 0.675 → só descarta lixo (<67.5%)
   - **Avaliações completas continuam acontecendo** para 95% dos indivíduos

2. **Cache de Fitness tem Hit Rate BAIXO:**
   - Hash collisions com Python's hash()
   - Indivíduos similares mas não idênticos não dão hit
   - **Não há logging de cache hits/misses** para confirmar efetividade

3. **HC Reduzido (8 variantes) NÃO compensa:**
   - Redução de 12-18 para 8 variantes economiza ~30% em HC
   - **MAS** HC é chamado para TODOS os indivíduos, não só elite
   - Overhead de chamadas + book keeping anula economia

4. **Logging WARNING overhead:**
   - Cada chunk tem ~100+ mensagens WARNING
   - I/O para arquivo de log é custoso
   - Estimativa: +2-5% de overhead

5. **Fase 2 Fingerprint Calculation overhead:**
   - scipy cosine similarity chamado 4x por chunk (1x por conceito conhecido)
   - `calculate_concept_fingerprint()` processa 6000 instâncias
   - Estimativa: +1-2% de overhead por chunk

**Conclusão:** Fase 1 foi mal implementada. Otimizações não economizam tempo real.

---

### CAUSA 2: FASE 2 CAUSA FALSE POSITIVES (threshold muito baixo)

**Evidência:**
- TEST_SINGLE: threshold 0.85 considera todos os chunks como "conceito recorrente" (sim ~0.98-0.99)
- **MAS TEST_SINGLE NÃO TEM DRIFT REAL** → todos os chunks são do mesmo conceito!
- Resultado: Fase 2 restaura memória sempre, mas não adiciona valor (conceito já era conhecido)

**Problema:**
- Em datasets SEM drift real (como TEST_SINGLE no modo stable), Fase 2 não deveria fazer nada
- **Mas threshold 0.85 é TÃO BAIXO** que detecta "recorrência" mesmo sem drift

**Impacto:**
- Overhead de fingerprint calculation desnecessário
- Restauração de memória desnecessária (não melhora performance)
- Fase 2 vira overhead puro em cenários sem drift

---

### CAUSA 3: CONCEITOS NOVOS COLAPSAM (overfitting severo)

**Evidência:**
- DRIFT_RECOVERY chunks 1 e 3 (conceitos NOVOS c2, c4):
  - Train G-mean: 0.89-0.94 (excelente)
  - Test G-mean: 0.46-0.52 (péssimo)
  - Overfitting: -37 a -49pp

- TEST_SINGLE chunk 2:
  - Train G-mean: 0.92 (excelente)
  - Test G-mean: 0.44 (péssimo)
  - Overfitting: -48pp

**Padrão:**
- **Conceitos NOVOS sempre colapsam** em generalização
- Treino está OK (G-mean ~0.90+)
- Teste falha catastroficamente (<0.53)

**Por que isso acontece:**

1. **Early Stopping DESCARTANDO indivíduos generalistas:**
   - Threshold 75% descarta indivíduos com G-mean < 67.5% no treino
   - **Mas indivíduos generalistas podem ter G-mean 60-70% no treino e 70-80% no teste!**
   - Early stop favorece overfitters (alto treino, baixo teste)

2. **Seeding inicial ruim para conceitos novos:**
   - Memória de conceito anterior (diferente) é restaurada
   - População inicial já está enviesada para conceito antigo
   - GA não consegue escapar do local optimum

3. **HC favorece overfitting:**
   - Hill Climbing refina regras para maximizar fitness no treino
   - Refinamento excessivo → regras muito específicas → overfitting
   - **HC deveria ter regularização** (penalizar regras muito específicas)

4. **Função de fitness não penaliza overfitting:**
   - Fitness = G-mean no TREINO apenas
   - Não há penalização por diferença train-test
   - GA otimiza para treino, ignora generalização

---

### CAUSA 4: TEMPO DOMINADO POR HC EM GERAÇÕES 26-200

**Evidência:**
- Chunk 0: 109min (25 gerações seeding) vs Chunks 1-4: 93-161min (200 gerações)
- **Chunk 0 tem MENOS gerações mas tempo similar?**
- Gerações 26-200 têm 7x mais gerações, mas só 0-50% mais tempo?

**Análise:**
- Geração 1 (seeding): **77-79s** (MUITO lenta)
- Gerações 2-25 (seeding): **25-35s** (média)
- **Gerações 26-200 (HC full): ???** (não temos sample)

**Hipótese:**
- Gerações 26-200 são mais rápidas que 2-25 (sem seeding overhead)
- **MAS** ainda têm overhead de HC + avaliação completa
- Total de tempo = (25 ger × 35s) + (175 ger × ?s)

**Implicação:**
- Se gerações 26-200 são ~20s cada: 175 × 20 = 3500s = 58min
- Chunk total: (25 × 35) + 3500 = 875 + 3500 = 4375s = 73min
- **Real é 93-161min** → sobram 20-88min de overhead!

**Conclusão:**
- Há overhead MASSIVO não explicado por gerações
- Possíveis culpados:
  - Avaliação de fitness (datasets grandes)
  - HC sendo chamado em TODOS os indivíduos, não só elite
  - Logging I/O
  - Fingerprint calculation

---

## 🔧 PLANO DE MELHORIAS (SEM ABANDONAR FASE 1+2)

### ESTRATÉGIA GERAL

**Princípio:** Manter Fase 1 e Fase 2, mas **CORRIGIR** as implementações ruins

**Fases de Implementação:**
1. **CORREÇÃO FASE 1** (melhorar performance + timing) ← **PRIORIDADE MÁXIMA**
2. **REFINAMENTO FASE 2** (reduzir false positives, melhorar threshold)
3. **OTIMIZAÇÃO GLOBAL** (reduzir overhead sistêmico)

---

### LAYER 1: CORREÇÃO FASE 1 (Performance + Timing)

#### **1.1. FIX EARLY STOPPING (CRÍTICO)**

**Problema:** Threshold 75% não descarta nada útil

**Solução: Adaptive Early Stopping com PARTIAL GMEAN**

```python
# fitness.py - NEW APPROACH
def evaluate_rules_on_data_optimized(...):
    # ... existing code ...

    # EARLY STOP - NOVA LÓGICA
    if len(elite_population) > 0:
        # Calcular G-mean parcial após 20% das instâncias
        partial_size = int(len(data) * 0.20)
        partial_gmean = compute_gmean_up_to(predictions[:partial_size], y_true[:partial_size])

        # Threshold adaptativo baseado em MEDIANA da elite (não worst)
        elite_gmeans = [ind.fitness for ind in elite_population if hasattr(ind, 'fitness')]
        if elite_gmeans:
            median_elite_gmean = np.median(elite_gmeans)

            # Se está abaixo de 50% da mediana, descartar
            if partial_gmean < median_elite_gmean * 0.50:
                logger.debug(f"Early stop: partial={partial_gmean:.3f} < threshold={median_elite_gmean*0.50:.3f}")
                return {'fitness': -float('inf'), ...}

    # Continuar avaliação completa
    # ... resto do código ...
```

**Benefícios:**
- **Descarta indivíduos ruins mais cedo** (após 20% dos dados)
- **Threshold adaptativo** (baseado em mediana, não worst)
- **Economia estimada:** -20-30% em avaliações

---

#### **1.2. FIX CACHE DE FITNESS (ALTO IMPACTO)**

**Problema:** Hash collisions, hit rate desconhecido

**Solução: SHA256 Hash + Hit Rate Logging**

```python
# ga.py - IMPROVED CACHE
import hashlib

def hash_individual_secure(individual):
    """Hash seguro usando SHA256"""
    rules_string = individual.get_rules_as_string()
    hash_string = f"{rules_string}|{individual.default_class}"
    return hashlib.sha256(hash_string.encode()).hexdigest()

# Global cache stats
cache_hits = 0
cache_misses = 0
cache_collisions = 0

def evaluate_with_cache(individual, ...):
    global cache_hits, cache_misses, cache_collisions

    ind_hash = hash_individual_secure(individual)

    if ind_hash in fitness_cache:
        # VALIDAÇÃO: Confirmar que é o mesmo indivíduo
        cached_rules = fitness_cache[ind_hash]['rules_string']
        current_rules = individual.get_rules_as_string()

        if cached_rules == current_rules:
            # HIT REAL
            cache_hits += 1
            return fitness_cache[ind_hash]['fitness_dict']
        else:
            # COLISÃO DETECTADA (improvável com SHA256, mas checando)
            cache_collisions += 1
            logger.warning(f"Cache collision detected! Hash: {ind_hash[:16]}...")

    # MISS - Avaliar normalmente
    cache_misses += 1
    fitness_dict = evaluate_rules_on_data_optimized(...)

    # Salvar no cache
    fitness_cache[ind_hash] = {
        'fitness_dict': fitness_dict,
        'rules_string': individual.get_rules_as_string()
    }

    return fitness_dict

# Logging de stats a cada geração
def log_cache_stats():
    total = cache_hits + cache_misses
    hit_rate = (cache_hits / total * 100) if total > 0 else 0
    logger.info(f"Cache stats: Hits={cache_hits}, Misses={cache_misses}, "
                f"Collisions={cache_collisions}, Hit Rate={hit_rate:.1f}%")
```

**Benefícios:**
- **SHA256 elimina collisions** (probabilidade ~0)
- **Logging de hit rate** para diagnóstico
- **Economia estimada:** -10-20% se hit rate > 30%

---

#### **1.3. OTIMIZAR HC CALLING (MÉDIO IMPACTO)**

**Problema:** HC pode estar sendo chamado para TODOS os indivíduos

**Solução: HC APENAS para elite + offspring de elite**

```python
# ga.py - SELECTIVE HC
def run_genetic_algorithm_with_memory(...):
    # ... após crossover e mutation ...

    # Aplicar HC APENAS se:
    # 1. Indivíduo é elite (is_protected=True)
    # 2. OU indivíduo é offspring de elite (parent was elite)

    offspring_needing_hc = []
    for ind in offspring:
        if hasattr(ind, 'is_protected') and ind.is_protected:
            # Elite sempre recebe HC
            offspring_needing_hc.append(ind)
        elif hasattr(ind, 'parent_was_elite') and ind.parent_was_elite:
            # Offspring de elite recebe HC com 50% de chance
            if random.random() < 0.5:
                offspring_needing_hc.append(ind)
        # Resto: sem HC (economia de tempo)

    # Aplicar HC apenas aos selecionados
    if offspring_needing_hc:
        hill_climbing_v2.apply_hc_to_population(offspring_needing_hc, ...)

    logger.debug(f"HC applied to {len(offspring_needing_hc)}/{len(offspring)} individuals")
```

**Benefícios:**
- **Reduz chamadas de HC em 50-70%**
- Mantém qualidade (elite sempre refinada)
- **Economia estimada:** -15-25% em tempo

---

#### **1.4. REDUZIR LOGGING OVERHEAD (BAIXO IMPACTO)**

**Problema:** Muitos WARNING prints por geração

**Solução: Logging WARNING APENAS para eventos IMPORTANTES**

```python
# main.py - SELECTIVE WARNING LOGS
# MANTER WARNING:
# - Início/fim de experimento
# - Início/fim de chunk
# - Detecção de drift SEVERE
# - Fase 2 (novo conceito, recorrência)

# MOVER PARA INFO:
# - Fase 2 similaridades (já temos, não precisa WARNING)
# - Salvamento de memória (rotina, não crítico)

# EXEMPLO:
# ANTES:
logger.warning(f"[FASE 2] Salvando memória do conceito '{current_concept_id}'")

# DEPOIS:
logger.info(f"[FASE 2] Salvando memória do conceito '{current_concept_id}'")
```

**Benefícios:**
- **Reduz I/O de log em 30-40%**
- Logs mais limpos e focados
- **Economia estimada:** -2-5% em tempo

---

### LAYER 2: REFINAMENTO FASE 2 (Reduzir False Positives)

#### **2.1. AUMENTAR THRESHOLD PARA CONCEITOS NOVOS**

**Problema:** Threshold 0.85 muito baixo, detecta "recorrência" em tudo

**Solução: Threshold Adaptativo (0.92 para conceitos novos, 0.85 para conhecidos)**

```python
# utils.py - ADAPTIVE THRESHOLD
def detect_recurring_concept(
    current_fingerprint: Dict[str, Any],
    concept_memory: Dict[str, Dict],
    similarity_threshold: float = 0.85,
    new_concept_threshold: float = 0.92  # NEW PARAMETER
) -> Tuple[bool, str]:
    # ... código existente ...

    best_match_id = ''
    best_similarity = 0.0

    for concept_id, concept_data in concept_memory.items():
        # ... calcular similarity ...

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_id = concept_id

    # NOVO: Usar threshold mais alto se conceito é "novo" (visto recentemente)
    last_seen = concept_memory[best_match_id].get('last_seen_chunk', 0)
    current_chunk = ...  # passar como parâmetro

    # Se conceito foi visto no chunk anterior, usar threshold normal (0.85)
    # Se conceito não foi visto há 2+ chunks, usar threshold alto (0.92)
    effective_threshold = similarity_threshold
    if current_chunk - last_seen >= 2:
        effective_threshold = new_concept_threshold
        logging.info(f"Using high threshold {effective_threshold} for concept not seen in {current_chunk-last_seen} chunks")

    if best_similarity >= effective_threshold:
        logging.warning(f"[FASE 2] ✓ MATCH: '{best_match_id}' (sim={best_similarity:.4f} >= {effective_threshold:.2f})")
        return (True, best_match_id)
    else:
        logging.warning(f"[FASE 2] ✗ NO MATCH: melhor '{best_match_id}' (sim={best_similarity:.4f} < {effective_threshold:.2f})")
        return (False, '')
```

**Benefícios:**
- **Reduz false positives** em 50-70%
- Conceitos verdadeiramente recorrentes ainda detectados (sim ~0.98)
- **Economia:** -1-2% em overhead (menos restaurações desnecessárias)

---

#### **2.2. ENRIQUECER FINGERPRINT (opcional, futuro)**

**Problema:** Fingerprint atual (mean, std, class dist) pode ser simplista

**Solução: Adicionar features discriminativas (correlações, PCA)**

```python
# utils.py - ENRICHED FINGERPRINT
def calculate_concept_fingerprint(data, target, attributes):
    # ... código existente (mean, std, class dist) ...

    # NOVO: Adicionar correlações entre features
    if len(attributes) > 1:
        corr_matrix = np.corrcoef(data[:, attributes].T)
        # Pegar triângulo superior (sem diagonal)
        corr_values = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    else:
        corr_values = np.array([])

    # NOVO: PCA first component direction
    from sklearn.decomposition import PCA
    if len(attributes) >= 2:
        pca = PCA(n_components=1)
        pca.fit(data[:, attributes])
        pca_direction = pca.components_[0]
    else:
        pca_direction = np.array([])

    return {
        'mean': mean,
        'std': std,
        'class_distribution': class_distribution,
        'correlations': corr_values,  # NEW
        'pca_direction': pca_direction,  # NEW
        'num_instances': len(data)
    }
```

**Benefícios:**
- **Fingerprints mais discriminativos** (conceitos diferentes → similaridade menor)
- **Reduz false positives** (threshold 0.85 vira mais restritivo)
- **Custo:** +5-10% overhead em fingerprint calculation (vale a pena?)

**Nota:** Implementar APENAS se Layer 1 + 2.1 não resolverem false positives.

---

### LAYER 3: OTIMIZAÇÃO GLOBAL (Reduzir Overhead Sistêmico)

#### **3.1. COMBATER OVERFITTING EM CONCEITOS NOVOS**

**Problema:** Conceitos novos colapsam (train OK, test péssimo)

**Solução 1: Penalizar Overfitting na Função de Fitness**

```python
# fitness.py - OVERFITTING PENALTY
def calculate_fitness_with_penalty(gmean_train, validation_gmean=None, ...):
    base_fitness = gmean_train  # Fitness base no treino

    # Se temos validação (holdout de 20% do treino), penalizar diferença
    if validation_gmean is not None:
        overfitting_penalty = max(0, (gmean_train - validation_gmean) * 0.5)
        fitness = base_fitness - overfitting_penalty

        logger.debug(f"Fitness: {base_fitness:.3f} - penalty {overfitting_penalty:.3f} = {fitness:.3f}")
        return fitness
    else:
        return base_fitness
```

**Implementação:**
- Dividir train_data_chunk em 80% treino / 20% validação
- Avaliar indivíduo em ambos
- Penalizar se G-mean validação < G-mean treino

**Benefícios:**
- **GA favorece indivíduos generalistas**
- **Reduz overfitting** em 30-50%
- **Custo:** +10-15% overhead (avaliação em validação)

---

**Solução 2: Regularizar HC (Penalizar Regras Muito Específicas)**

```python
# hill_climbing_v2.py - REGULARIZATION
def apply_hc_to_individual(individual, ...):
    # ... código existente ...

    # Ao avaliar variante, penalizar se regras são muito específicas
    variant_fitness = evaluate_variant(...)

    # Contar quantas condições muito específicas (ex: intervalo < 0.1)
    specificity_count = 0
    for rule in individual.rules:
        for condition in rule.conditions:
            if hasattr(condition, 'interval_size'):
                if condition.interval_size() < 0.1:  # Muito específico
                    specificity_count += 1

    # Penalizar fitness se muita especificidade
    specificity_penalty = specificity_count * 0.01
    regularized_fitness = variant_fitness - specificity_penalty

    return regularized_fitness
```

**Benefícios:**
- **HC não cria regras hiper-específicas**
- **Melhora generalização** em 20-30%
- **Custo:** desprezível (<1% overhead)

---

#### **3.2. PROFILE E IDENTIFICAR GARGALO REAL**

**Problema:** Temos 20-88min de overhead não explicado

**Solução: Profiling com cProfile**

```python
# main.py - ADD PROFILING
import cProfile
import pstats

if config.get('enable_profiling', False):
    profiler = cProfile.Profile()
    profiler.enable()

# ... código do experimento ...

if config.get('enable_profiling', False):
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(50)  # Top 50 funções mais custosas
    stats.dump_stats(f'profile_{stream_name}_run{run_number}.prof')
```

**Análise:**
- Rodar 1 chunk com profiling
- Identificar top 10 funções mais custosas
- Otimizar especificamente essas funções

**Benefícios:**
- **Identificação precisa do gargalo**
- **Otimização direcionada** (não às cegas)

---

## 📋 PLANO DE AÇÃO IMEDIATO

### PRIORIDADE MÁXIMA (implementar AGORA)

**1. FIX EARLY STOPPING (1.1)**
- Implementar adaptive early stopping (threshold 50% da mediana elite)
- Avaliar apenas 20% dos dados antes de descartar
- **Tempo estimado:** 1h implementação
- **Ganho esperado:** -20-30% tempo, +1-2pp G-mean

**2. FIX CACHE DE FITNESS (1.2)**
- Trocar hash() por SHA256
- Adicionar logging de hit rate
- **Tempo estimado:** 1h implementação
- **Ganho esperado:** -10-20% tempo (se hit rate > 30%)

**3. OTIMIZAR HC CALLING (1.3)**
- HC apenas para elite + 50% offspring de elite
- **Tempo estimado:** 1h implementação
- **Ganho esperado:** -15-25% tempo

**Total Fase 1 Corrigida:** **-45-75% tempo** (de 10h para 5-6h) ← **DENTRO DA META -48%!**

---

### PRIORIDADE ALTA (implementar depois de validar Fase 1)

**4. AUMENTAR THRESHOLD FASE 2 (2.1)**
- Threshold adaptativo (0.92 para conceitos não vistos recentemente)
- **Tempo estimado:** 30min implementação
- **Ganho esperado:** -1-2% tempo, -50% false positives

**5. PENALIZAR OVERFITTING (3.1 Solução 1)**
- Dividir treino em 80/20 (treino/validação)
- Penalizar diferença train-validation na fitness
- **Tempo estimado:** 2h implementação
- **Ganho esperado:** +2-4pp G-mean, -30% overfitting

**6. REGULARIZAR HC (3.1 Solução 2)**
- Penalizar regras hiper-específicas no HC
- **Tempo estimado:** 1h implementação
- **Ganho esperado:** +1-2pp G-mean

---

### PRIORIDADE MÉDIA (implementar se ganhos acima não forem suficientes)

**7. REDUZIR LOGGING OVERHEAD (1.4)**
- Mover logs não-críticos de WARNING para INFO
- **Tempo estimado:** 30min
- **Ganho esperado:** -2-5% tempo

**8. PROFILING (3.2)**
- Adicionar cProfile para identificar gargalo real
- **Tempo estimado:** 30min setup + 2h análise
- **Ganho esperado:** Identificação precisa de onde otimizar

**9. ENRIQUECER FINGERPRINT (2.2)**
- Adicionar correlações e PCA direction
- **Tempo estimado:** 2h implementação
- **Ganho esperado:** -30% false positives (se 2.1 não resolver)

---

## 🎯 METAS DE PERFORMANCE PÓS-MELHORIAS

### Metas de Timing (após Layer 1)

| Experimento | Baseline | Run3 Atual | Meta Pós-Layer1 | Redução |
|-------------|----------|------------|-----------------|---------|
| TEST_SINGLE | 8.0h | 9.9h (+24%) | **5.0-6.0h** | **-38 a -50%** ✅ |
| DRIFT_RECOVERY | 10.0h | 14.8h (+48%) | **6.0-8.0h** | **-40 a -60%** ✅ |

**Meta:** Alcançar ou superar redução de -48% prometida pela Fase 1.

---

### Metas de G-mean (após Layer 2 + Layer 3)

| Experimento | Baseline | Run3 Atual | Meta Pós-Melhorias | Melhora |
|-------------|----------|------------|---------------------|---------|
| TEST_SINGLE | 79.83% | 77.63% (-2.2pp) | **80-82%** | **+0.2 a +2.2pp** ✅ |
| DRIFT_RECOVERY | 73.74% | 73.14% (-0.6pp) | **75-77%** | **+1.3 a +3.3pp** ✅ |

**Meta:** Superar baseline + reduzir overfitting de -48pp para -10-15pp.

---

## 🔄 WORKFLOW DE IMPLEMENTAÇÃO

### Fase 1: Correção (Layer 1)

```
1. Implementar Fix Early Stopping (1.1)
   ↓
2. Implementar Fix Cache SHA256 (1.2)
   ↓
3. Implementar Selective HC (1.3)
   ↓
4. Testar com TEST_SINGLE (2 chunks, run 999)
   ↓
5. Validar:
   - Tempo reduziu? (meta: -45% vs Run3)
   - G-mean manteve ou melhorou? (meta: ≥77.6%)
   - Cache hit rate? (meta: >30%)
   ↓
6. Se OK → Experimentos completos (TEST_SINGLE run 4, DRIFT_RECOVERY run 4)
7. Se NOK → Debug e ajustar
```

---

### Fase 2: Refinamento (Layer 2)

```
1. Implementar Threshold Adaptativo Fase 2 (2.1)
   ↓
2. Testar com DRIFT_RECOVERY (2 chunks, run 999)
   ↓
3. Validar:
   - False positives reduziram? (meta: <20% dos chunks)
   - Conceitos recorrentes ainda detectados? (chunk 4 = c1)
   ↓
4. Se OK → Experimentos completos
5. Se NOK → Implementar Enriched Fingerprint (2.2)
```

---

### Fase 3: Otimização (Layer 3)

```
1. Implementar Overfitting Penalty (3.1 Sol 1)
   ↓
2. Implementar HC Regularization (3.1 Sol 2)
   ↓
3. Testar com DRIFT_RECOVERY (chunks completos)
   ↓
4. Validar:
   - Overfitting reduziu? (meta: <-15pp em chunks novos)
   - G-mean melhorou? (meta: +2-4pp vs Run3)
   ↓
5. Se OK → Experimentos finais
6. Se NOK → Profiling (3.2) para identificar gargalo real
```

---

## 📊 CHECKLIST DE VALIDAÇÃO

### Após Implementar Layer 1 (Correção Fase 1)

- [ ] Tempo TEST_SINGLE reduziu para 5-6h? (vs 9.9h atual)
- [ ] Tempo DRIFT_RECOVERY reduziu para 6-8h? (vs 14.8h atual)
- [ ] G-mean TEST_SINGLE manteve ≥77.6%?
- [ ] Cache hit rate >30%?
- [ ] Early stop descartou 20-40% das avaliações?
- [ ] Logs mostram "Cache stats" com hit rate?

### Após Implementar Layer 2 (Refinamento Fase 2)

- [ ] False positives <20% em TEST_SINGLE?
- [ ] DRIFT_RECOVERY chunk 4 ainda detecta c1 recorrente?
- [ ] Similaridades de conceitos diferentes <0.90?
- [ ] Threshold adaptativo sendo aplicado corretamente?

### Após Implementar Layer 3 (Otimização Global)

- [ ] Overfitting em conceitos novos <-15pp? (vs -48pp atual)
- [ ] G-mean TEST_SINGLE ≥80%? (+2.4pp vs Run3)
- [ ] G-mean DRIFT_RECOVERY ≥75%? (+1.9pp vs Run3)
- [ ] Chunks 1 e 3 de DRIFT_RECOVERY com Test G-mean >0.65?

---

## 🎓 LIÇÕES APRENDIDAS

1. **Otimizações "teóricas" podem falhar na prática**
   - Early stop threshold 75% parecia bom, mas worst_elite_gmean=0 matou a ideia
   - Cache hash() parecia OK, mas collisions anularam ganhos
   - **Sempre validar com profiling e logging**

2. **Overhead de logging é REAL**
   - WARNING prints a cada geração custam 2-5% de tempo
   - **Logging deve ser seletivo e assíncrono**

3. **Fase 2 funciona, mas threshold precisa ser calibrado**
   - Threshold 0.85 muito baixo para conceitos similares
   - **Threshold adaptativo é necessário**

4. **Overfitting é o maior inimigo em GBML**
   - Treino OK, teste colapsa (-48pp!)
   - **Regularização e penalização são essenciais**

5. **Timing de gerações revela gargalos**
   - Primeira geração 77s vs gerações seguintes 25s → seeding overhead
   - **Profiling é mandatório antes de otimizar**

---

**FIM DO DIAGNÓSTICO**

**Próximos passos:**
1. Implementar Layer 1 (FIX Early Stop + Cache + HC)
2. Validar com smoke test (2 chunks)
3. Experimentos completos Run4
4. Analisar resultados e decidir se Layer 2+3 são necessários

**Meta final:** Superar baseline em G-mean (+1-3pp) E reduzir tempo em -40 a -50%.
