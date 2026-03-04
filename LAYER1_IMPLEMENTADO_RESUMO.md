# LAYER 1 IMPLEMENTADO - CORREÇÃO FASE 1

**Data:** 2025-11-02
**Status:** ✅ COMPLETO e TESTADO (sintaxe OK)
**Arquivos Modificados:** `ga.py`, `fitness.py`

---

## 🎯 OBJETIVO

Corrigir as implementações ruins da Fase 1 que causaram:
- **+24% a +48% MAIS LENTO** (vs meta de -48%)
- **-2.76% em G-mean** (TEST_SINGLE)

---

## ✅ FIX 1: EARLY STOPPING ADAPTATIVO

### Problema Original
- Threshold baseado em **worst elite** (12º melhor)
- Threshold 75% muito conservador (nunca descartava nada útil)
- Avaliava 33% dos dados antes de verificar

### Solução Implementada

**ga.py (linhas 820-834):**
```python
# Usa MEDIANA do top-12 (elite) como baseline, não o pior
early_stop_threshold = 0.0
if generation > 0 and len(population) >= 12:
    sorted_pop = sorted(population, key=lambda x: getattr(x, 'gmean', 0.0), reverse=True)
    elite_gmeans = [ind.gmean for ind in sorted_pop[:12] if hasattr(ind, 'gmean') and ind.gmean > 0]

    if elite_gmeans:
        import numpy as np
        median_elite_gmean = np.median(elite_gmeans)
        early_stop_threshold = median_elite_gmean
```

**fitness.py (linhas 204-242):**
```python
# Avalia apenas 20% dos dados antes de decidir
partial_size = max(100, int(len(data) * 0.20))

partial_data = data[:partial_size]
partial_target = target[:partial_size]
partial_predictions = [individual._predict(inst) for inst in partial_data]

partial_gmean = calculate_gmean_contextual(...)

# Threshold 50% (mais agressivo que 75% anterior)
if partial_gmean < early_stop_threshold * 0.50:
    logging.debug(f"Early stop: partial_gmean={partial_gmean:.3f} < threshold={early_stop_threshold*0.50:.3f}")
    return {'fitness': -float('inf'), 'g_mean': partial_gmean, 'early_stopped': True}

# Se passou, avalia os 80% restantes
remaining_data = data[partial_size:]
remaining_predictions = [individual._predict(inst) for inst in remaining_data]
predictions = partial_predictions + remaining_predictions
```

### Ganhos Esperados
- **-20-30% de tempo**: Descarta indivíduos ruins após avaliar apenas 20% dos dados
- **Logging de quantos foram descartados** para diagnóstico

**ga.py (linhas 914-916, 935-937):**
```python
# Verificar se foi early stopped
if fitness_score == -float('inf') and gmean > 0:
    early_stopped_count += 1

# Log ao final da geração
if early_stopped_count > 0:
    early_stop_pct = (early_stopped_count / len(population)) * 100
    logging.info(f"Gen {generation+1}: Early stopped: {early_stopped_count}/{len(population)} ({early_stop_pct:.1f}%) individuals")
```

---

## ✅ FIX 2: CACHE SHA256 COM VALIDAÇÃO

### Problema Original
- Python's `hash()` tem collisions
- Sem validação de identidade (confiava 100% no hash)
- Sem logging de hit rate

### Solução Implementada

**ga.py (linhas 809-818):**
```python
def hash_individual(individual):
    """
    Gera hash único baseado na estrutura completa de regras.
    USA SHA256 para evitar colisões (vs Python's hash()).
    """
    import hashlib
    rules_string = individual.get_rules_as_string()
    hash_string = f"{rules_string}|{individual.default_class}"
    # SHA256 retorna hash seguro de 64 caracteres hexadecimais
    return hashlib.sha256(hash_string.encode('utf-8')).hexdigest()
```

**ga.py (linhas 899-943):**
```python
# Tentar usar cache
cache_hit = False
if ind_hash in fitness_cache:
    # VALIDAR: Verificar se é realmente o mesmo indivíduo
    cached = fitness_cache[ind_hash]
    current_rules = individual.get_rules_as_string()

    if 'rules_string' in cached and cached['rules_string'] == current_rules:
        # CACHE HIT REAL: Mesmo hash E mesmas regras
        individual.fitness = cached['fitness']
        individual.gmean = cached['gmean']
        cache_hits += 1
        cache_hit = True
    else:
        # COLISÃO DETECTADA (improvável com SHA256!)
        logging.warning(f"Gen {generation+1}: Cache collision detected! Hash: {ind_hash[:16]}...")
        cache_collisions_total += 1

if not cache_hit:
    # CACHE MISS (ou colisão): Avalia e armazena
    fitness_score, gmean, activation_rate = evaluate_individual_fitness_parallel(...)

    # Armazena no cache (com rules_string para validação)
    fitness_cache[ind_hash] = {
        'fitness': fitness_score,
        'gmean': gmean,
        'activation_rate': activation_rate,
        'rules_string': individual.get_rules_as_string()  # Para validação
    }
```

### Logging de Hit Rate

**ga.py (linhas 930-933):**
```python
# Log por geração
if cache_hits > 0 or cache_misses > 0:
    hit_rate = (cache_hits / (cache_hits + cache_misses)) * 100 if (cache_hits + cache_misses) > 0 else 0
    logging.info(f"Gen {generation+1}: Cache hits: {cache_hits}/{cache_hits + cache_misses} ({hit_rate:.1f}%)")
```

**ga.py (linhas 1357-1363):**
```python
# Log final do experimento
total_cache_ops = cache_hits_total + cache_misses_total
if total_cache_ops > 0:
    cache_hit_rate = (cache_hits_total / total_cache_ops) * 100
    logging.info(f"Cache stats: Hits={cache_hits_total}, Misses={cache_misses_total}, Hit Rate={cache_hit_rate:.1f}%")
    if cache_collisions_total > 0:
        logging.warning(f"Cache collisions detected: {cache_collisions_total} (SHA256 collisions!)")
```

### Ganhos Esperados
- **-10-20% de tempo** se hit rate > 30%
- **Zero collisions** (SHA256 é criptograficamente seguro)
- **Diagnóstico completo** via logging

---

## ✅ FIX 3: SELECTIVE HC (A CADA 3 GERAÇÕES DE ESTAGNAÇÃO)

### Problema Original
- HC aplicado a CADA geração de estagnação
- 8 variantes HC por geração de estagnação
- Overhead massivo em períodos longos de estagnação

### Solução Implementada

**ga.py (linhas 1208-1217):**
```python
# OTIMIZAÇÃO FASE 1.3: HC apenas a cada 3 gerações de estagnação
# Exemplo: estagnação gens 11, 12, 13, 14 → HC apenas em 11, 14, 17...
should_apply_hc = ((no_improvement_count - STAGNATION_THRESHOLD) % 3 == 0)

if should_apply_hc:
    # --- Hill Climbing Hierárquico v2.0 ---
    elite_gmean = best_individual_overall.gmean if best_individual_overall else 0.0
else:
    logging.info(f"   Pulando HC nesta geração (será aplicado a cada 3 ger de estagnação, economia de tempo)")
    elite_gmean = None  # Flag para pular HC
```

**ga.py (linhas 1219-1246):**
```python
# Aplicar HC apenas se should_apply_hc é True
hc_variants = []
if should_apply_hc and elite_gmean is not None:
    # Prepara kwargs para HC hierárquico
    hc_kwargs = {...}

    # Chama HC hierárquico
    hc_variants = hill_climbing_v2.hierarchical_hill_climbing(...)
```

### Ganhos Esperados
- **-66% de chamadas HC** (apenas 1 em cada 3 gerações de estagnação)
- **-15-25% de tempo total** (HC é custoso)
- **Qualidade mantida** (estagnação longa ainda recebe HC regularmente)

---

## 📊 GANHOS TOTAIS ESPERADOS

| Fix | Economia de Tempo | Impacto em G-mean |
|-----|-------------------|-------------------|
| **Fix 1: Early Stopping** | -20-30% | Neutro (+0pp) |
| **Fix 2: Cache SHA256** | -10-20% (se hit rate > 30%) | Neutro (+0pp) |
| **Fix 3: Selective HC** | -15-25% | Neutro (+0pp) |
| **TOTAL (composto)** | **-40-55%** | **Neutro** |

### Meta vs Real

| Métrica | Baseline | Run3 Atual | Meta Pós-Layer1 | Status |
|---------|----------|------------|-----------------|--------|
| **Tempo TEST_SINGLE** | 8.0h | 9.9h | **4.5-6.0h** | 🎯 Alcançável |
| **Tempo DRIFT_RECOVERY** | 10.0h | 14.8h | **6.5-8.0h** | 🎯 Alcançável |
| **G-mean TEST_SINGLE** | 79.83% | 77.63% | **77-79%** | 🎯 Manter ou leve melhora |
| **G-mean DRIFT_RECOVERY** | 73.74% | 73.14% | **73-75%** | 🎯 Manter ou leve melhora |

**Redução esperada:** -40-55% tempo (de 9.9h → 4.5-6h, de 14.8h → 6.5-8h)

---

## 🔍 VALIDAÇÕES NECESSÁRIAS

### 1. Smoke Test (2 chunks)

```bash
# Criar config de teste com apenas 2 chunks
python main.py config_test_single.yaml --num_chunks 2 --run_number 999
```

**Verificações:**
- [ ] Logs mostram `Early stop threshold = X.XXX (median of top-12)`
- [ ] Logs mostram `Early stopped: X/120 (X.X%) individuals`
- [ ] Logs mostram `Cache hits: X/Y (X.X%)`
- [ ] Hit rate > 20% (elite sendo reutilizada)
- [ ] Logs mostram `Pulando HC nesta geração` (1-2x durante estagnação)
- [ ] Tempo de 2 chunks < 3h (vs ~3.5-4h esperado sem otimizações)

### 2. Experimento Completo (5 chunks)

```bash
python main.py config_test_single.yaml --run_number 4
```

**Verificações:**
- [ ] Tempo total 4.5-6.0h (vs 9.9h Run3)
- [ ] G-mean ≥ 77.0% (não piorar muito vs 77.63%)
- [ ] Cache hit rate final > 30%
- [ ] Early stop descartou 20-40% dos indivíduos
- [ ] HC pulado em ~66% das gerações de estagnação

---

## 📝 COMANDOS PARA TESTAR

### Smoke Test (rápido, ~2-3h)
```bash
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"

# Windows PowerShell
python main.py config_test_single.yaml --num_chunks 2 --run_number 999 2>&1 | Tee-Object -FilePath "smoke_test_layer1.log"
```

### Experimentos Completos (após smoke test OK)
```bash
# TEST_SINGLE Run 4 (esperado: 4.5-6h vs 9.9h Run3)
python main.py config_test_single.yaml --run_number 4 2>&1 | Tee-Object -FilePath "experimento_test_single4.log"

# DRIFT_RECOVERY Run 4 (esperado: 6.5-8h vs 14.8h Run3)
python main.py config_test_drift_recovery.yaml --run_number 4 2>&1 | Tee-Object -FilePath "experimento_test_recovery4.log"
```

### Análise Pós-Experimento
```bash
# Analisar Run4 vs Run3
python analyze_run3_experiments.py  # Adaptar para Run4

# Verificar cache stats
grep "Cache stats:" experimento_test_single4.log
grep "Early stopped:" experimento_test_single4.log | head -20
grep "Pulando HC" experimento_test_single4.log | wc -l
```

---

## 🎓 MUDANÇAS DETALHADAS POR ARQUIVO

### `ga.py`

| Linhas | Mudança | Descrição |
|--------|---------|-----------|
| 803-808 | Cache structure | Adiciona `cache_collisions_total`, atualiza comentários |
| 809-818 | hash_individual() | SHA256 em vez de hash() |
| 820-834 | Early stop threshold | Mediana elite (não worst elite) |
| 899-943 | Cache lookup | Validação de colisão + logging |
| 914-916, 935-937 | Early stop logging | Conta quantos indivíduos descartados |
| 930-933 | Cache logging (gen) | Hit rate por geração |
| 1208-1217 | Selective HC | HC a cada 3 ger de estagnação |
| 1219-1246 | HC conditional | Pula HC se não deve aplicar |
| 1357-1363 | Cache logging (final) | Hit rate total + collisions |

### `fitness.py`

| Linhas | Mudança | Descrição |
|--------|---------|-----------|
| 204-242 | Early stopping | Avalia 20% dos dados, threshold 50% mediana |
| 231 | Early stop flag | Adiciona `'early_stopped': True` para diagnóstico |

---

## 🚀 PRÓXIMOS PASSOS

**AGORA:**
1. ✅ **Smoke test** (2 chunks, run 999)
2. ⏸️ Validar logs (early stop, cache, HC)
3. ⏸️ Se OK → Experimentos completos (Run4)

**DEPOIS (se Run4 OK):**
4. ⏸️ Implementar Layer 2 (Threshold Adaptativo Fase 2)
5. ⏸️ Implementar Layer 3 (Penalizar Overfitting)

**SE Run4 FALHAR:**
- Debug com profiling (cProfile)
- Ajustar thresholds (early stop 50% → 40%?)
- Verificar cache hit rate (se < 20%, problema)

---

## ✅ CHECKLIST DE VALIDAÇÃO

- [x] Sintaxe Python OK (ga.py, fitness.py)
- [x] **BUGFIX: numpy import duplicado corrigido** (ver BUGFIX_LAYER1_NUMPY.md)
- [ ] Smoke test executado (2 chunks)
- [ ] Logs contêm early stop stats
- [ ] Logs contêm cache hit rate
- [ ] Logs contêm "Pulando HC"
- [ ] Tempo smoke test < 3h
- [ ] Experimento completo Run4 executado
- [ ] Tempo Run4 < 6h (TEST_SINGLE)
- [ ] G-mean Run4 ≥ 77% (TEST_SINGLE)
- [ ] Cache hit rate > 30%

---

## 🐛 BUGFIX APLICADO

**Erro:** `UnboundLocalError: cannot access local variable 'np'`
**Causa:** Import duplicado de numpy dentro de bloco if (ga.py linha 835)
**Correção:** Removido import duplicado, usa numpy global (já importado linha 6)
**Detalhes:** Ver `BUGFIX_LAYER1_NUMPY.md`

---

**FIM DO RESUMO LAYER 1**

**Status:** ✅ IMPLEMENTADO E CORRIGIDO, pronto para smoke test
**Arquivos modificados:** `ga.py` (13 blocos + 1 bugfix), `fitness.py` (1 bloco)
**Ganho esperado:** -40-55% tempo, G-mean neutro ou leve melhora
