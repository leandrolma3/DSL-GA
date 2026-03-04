# RESUMO FINAL - ANALISE EXPERIMENTO RUN4 (LAYER 1)

**Data:** 2025-11-03
**Experimento:** 5 chunks TEST_SINGLE
**Objetivo:** Validar Layer 1 (Early Stop + Cache SHA256 + HC Seletivo)

---

## RESULTADO GERAL

### Tempo
- **Run3 (baseline):** 9.9h
- **Run4 (Layer 1):** 13.42h
- **Delta:** +35.6% PIOR

### G-mean
- **Run3:** 0.7763
- **Run4:** 0.7928
- **Delta:** +2.1% MELHOR

### Status
**FRACASSO PARCIAL**
- Meta de redução de tempo: -40-55% (obteve-se +35.6% PIOR)
- Meta de G-mean: neutro ou melhora (obteve-se +2.1% OK)

---

## LAYER 1 - STATUS DAS OTIMIZACOES

### Fix 1: Early Stopping Adaptativo
**STATUS:** NAO FUNCIONANDO

**Evidências:**
- Threshold CALCULADO: 27 logs encontrados
- Descarte: 0 logs (ZERO indivíduos descartados)
- Diagnóstico: early_stopped_count sempre 0

**Problema identificado:**
```python
# ga.py linha 958-960
if early_stopped_count > 0:  # NUNCA EXECUTOU
    logging.warning(f"[EARLY STOP] Gen {generation+1}: Descartados=...")
```

**Causa raiz:**
- Early stop threshold está sendo calculado
- MAS nenhum indivíduo está sendo descartado
- Threshold muito alto OU lógica de descarte não funciona

**Impacto no tempo:**
- Esperado: -20-30% tempo
- Real: 0% economia
- Perda: ~2-4h no experimento

---

### Fix 2: Cache SHA256
**STATUS:** NAO FUNCIONANDO

**Evidências:**
- Cache logs por geração: 0
- Cache logs finais: 0
- Diagnóstico: cache_hits e cache_misses sempre 0

**Problema identificado:**
```python
# ga.py linha 955-956
if cache_hits > 0 or cache_misses > 0:  # NUNCA EXECUTOU
    logging.warning(f"[CACHE] Gen {generation+1}: Hits=...")
```

**Causa raiz:**
- Cache não está sendo usado
- Possível: hash_individual() não sendo chamado
- Possível: fitness_cache não sendo populado
- Possível: lógica de cache lookup quebrada

**Impacto no tempo:**
- Esperado: -10-20% tempo (com hit rate > 30%)
- Real: 0% economia
- Perda: ~1-3h no experimento

---

### Fix 3: Hill Climbing Seletivo
**STATUS:** FUNCIONANDO

**Evidências:**
- HC aplicado: 19x
- HC pulado: 9x
- Economia: 32.1%

**Por chunk:**
- Chunk 0: 0% economia (2 aplicado, 0 pulado)
- Chunk 1: 33% economia (4 aplicado, 2 pulado)
- Chunk 2: 0% economia (4 aplicado, 0 pulado)
- Chunk 3: 56% economia (4 aplicado, 5 pulado)
- Chunk 4: 29% economia (5 aplicado, 2 pulado)

**Taxa de aprovação:**
- 32-37.5% consistente (HC gera variantes úteis)

**Impacto no tempo:**
- Esperado: -15-25% tempo
- Real: ~5-10% economia estimada
- Economia parcial (não atingiu meta)

---

## DESEMPENHO POR CHUNK

| Chunk | Tempo | Train | Test | Delta | Status |
|-------|-------|-------|------|-------|--------|
| 0 | 93min | 0.913 | 0.883 | -0.030 | OK |
| 1 | 144min | 0.937 | 0.894 | -0.044 | OK |
| 2 | 182min | 0.941 | 0.439 | -0.501 | OVERFITTING EXTREMO |
| 3 | 178min | 0.926 | 0.873 | -0.053 | Recuperou |
| 4 | 209min | 0.932 | 0.876 | -0.056 | OK |

### Chunk 2: Problema Crítico
- Test G-mean: 0.439 (MUITO BAIXO)
- Delta: -0.501 (OVERFITTING SEVERO)
- Consequência: SEVERE DRIFT detectado no chunk 3
- Causa: Modelo especializou demais no treino, falhou no teste

---

## DIAGNOSTICO: POR QUE TEMPO PIOROU?

### 1. Early Stop NÃO funcionou
- Impacto: 0% economia (esperava-se -20-30%)
- Perda: ~2-4h

### 2. Cache NÃO funcionou
- Impacto: 0% economia (esperava-se -10-20%)
- Perda: ~1-3h

### 3. HC Seletivo funcionou parcialmente
- Impacto: ~5-10% economia (esperava-se -15-25%)
- Economia abaixo do esperado

### Total esperado vs real
- **Esperado:** -40-55% tempo (de 9.9h para 4.5-6h)
- **Real:** +35.6% tempo (de 9.9h para 13.4h)
- **Diferença:** ~8-10h de perda

---

## PROBLEMAS IDENTIFICADOS

### CRITICO (Bloqueia otimizações)
1. Cache completamente não funcional (0 hits, 0 misses)
2. Early stop descarte não funcional (0 indivíduos descartados)
3. Tempo 35% PIOR que baseline

### ALTO (Impacta qualidade)
4. Chunk 2: Overfitting extremo (delta=-0.50)
5. Chunk 2: Test G-mean=0.439 (DROP de 45% vs chunk 1)

### MEDIO
6. HC economia variável (0-56%, média 32%)
7. Gerações completadas altas (40-57 em alguns chunks)

---

## INVESTIGACAO NECESSARIA

### 1. Cache (URGENTE)

**Verificar ga.py linhas 899-943:**
```python
# Tentar usar cache
cache_hit = False
if ind_hash in fitness_cache:  # <- Este bloco está executando?
    ...
```

**Debug necessário:**
- Adicionar print antes do if: `print(f"Checking cache for hash {ind_hash[:16]}...")`
- Verificar se ind_hash está sendo gerado
- Verificar se fitness_cache está sendo populado

**Linha 809-818 (hash_individual):**
- Verificar se esta função está sendo chamada
- Adicionar print: `print(f"Hashing individual: {hash[:16]}...")`

---

### 2. Early Stop (URGENTE)

**Verificar fitness.py linhas 204-242:**
```python
if partial_gmean < early_stop_threshold * 0.50:
    return {
        'fitness': -float('inf'),
        'g_mean': partial_gmean,
        'early_stopped': True  # <- Este flag está sendo setado?
    }
```

**Debug necessário:**
- Adicionar print: `print(f"partial_gmean={partial_gmean:.3f}, threshold={early_stop_threshold*0.50:.3f}")`
- Verificar quantas vezes este if é True

**Verificar ga.py linhas 914-916:**
```python
if fitness_score == -float('inf') and gmean > 0:
    early_stopped_count += 1  # <- Este contador está incrementando?
```

**Debug necessário:**
- Adicionar print: `print(f"early_stopped_count={early_stopped_count}")`

---

### 3. Chunk 2 Overfitting

**Análise necessária:**
- Comparar distribuição de classes chunk 2 vs outros
- Verificar se chunk 2 tem anomalias nos dados
- Analisar por que modelo convergiu no treino mas falhou no teste

---

## RECOMENDACOES IMEDIATAS

### URGENTE (Antes de próximo experimento)

1. **Adicionar debug logging detalhado**
   - Cache: hash generation, cache lookup, hit/miss
   - Early stop: partial_gmean vs threshold, descarte count

2. **Smoke test com debug**
   - Executar 2 chunks com prints de debug
   - Validar se cache e early stop funcionam

3. **Code review de ga.py linhas 899-943 e 955-960**
   - Verificar indentação
   - Verificar lógica de if/else
   - Verificar variáveis sendo usadas

---

### CURTO PRAZO (Após debug)

4. **Corrigir cache e early stop**
   - Se não funcionando: fix código
   - Se funcionando mas sem efeito: ajustar thresholds

5. **Re-executar Run5 com correções**
   - Validar redução de tempo
   - Meta: -30-40% vs Run3

6. **Implementar Layer 2**
   - Penalizar overfitting durante evolução
   - Evitar chunk 2 problema

---

## CODIGO DEBUG SUGERIDO

### ga.py (adicionar no início da função run_genetic_algorithm)

```python
# DEBUG LAYER 1
DEBUG_LAYER1 = True

def debug_print(msg):
    if DEBUG_LAYER1:
        print(f"[DEBUG] {msg}")
```

### ga.py linha 810 (dentro de hash_individual)

```python
def hash_individual(individual):
    import hashlib
    rules_string = individual.get_rules_as_string()
    hash_string = f"{rules_string}|{individual.default_class}"
    hash_val = hashlib.sha256(hash_string.encode('utf-8')).hexdigest()
    debug_print(f"Hash gerado: {hash_val[:16]}... para {len(rules_string)} chars")  # ADD
    return hash_val
```

### ga.py linha 900 (cache lookup)

```python
ind_hash = hash_individual(individual)
debug_print(f"Gen {generation+1}: Procurando hash {ind_hash[:16]} no cache (tamanho={len(fitness_cache)})")  # ADD

cache_hit = False
if ind_hash in fitness_cache:
    debug_print(f"Cache HIT para {ind_hash[:16]}")  # ADD
    ...
else:
    debug_print(f"Cache MISS para {ind_hash[:16]}")  # ADD
```

### ga.py linha 916 (early stop count)

```python
if fitness_score == -float('inf') and gmean > 0:
    early_stopped_count += 1
    debug_print(f"Early stopped individual. Count now: {early_stopped_count}")  # ADD
```

### fitness.py linha 220 (early stop decision)

```python
if partial_gmean < early_stop_threshold * 0.50:
    debug_print(f"EARLY STOP: {partial_gmean:.3f} < {early_stop_threshold*0.50:.3f}")  # ADD
    return {
        'fitness': -float('inf'),
        'g_mean': partial_gmean,
        'early_stopped': True
    }
```

---

## CONCLUSAO

### Sucessos
- HC Seletivo funcionando (32% economia)
- Fase 2 Concept Fingerprinting funcionando perfeitamente
- G-mean levemente melhor (+2.1%)
- Logs informativos e visíveis

### Fracassos
- Cache completamente não funcional (0 uso)
- Early stop descarte não funcional (0 descartados)
- Tempo 35% PIOR em vez de -40% melhor
- Chunk 2 overfitting extremo

### Próximos passos (ordem de prioridade)
1. DEBUG imediato com prints
2. SMOKE TEST (2 chunks) validar correções
3. FIX cache e early stop
4. RE-RUN Run5 com fixes
5. IMPLEMENTAR Layer 2 (penalizar overfitting)

---

**STATUS FINAL:** LAYER 1 PARCIALMENTE IMPLEMENTADO
- 1/3 otimizações funcionando (HC Seletivo)
- 2/3 otimizações não funcionais (Cache, Early Stop)
- Tempo piorou em vez de melhorar
- Requer debug urgente antes de próximo experimento

---

**PRIORIDADE MAXIMA:** Adicionar debug logging e executar smoke test
