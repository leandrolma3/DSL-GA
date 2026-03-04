# DEBUG LAYER 1 - LOGS ADICIONADOS

**Data:** 2025-11-03
**Objetivo:** Diagnosticar por que Cache e Early Stop não funcionaram no Run4
**Arquivos modificados:** ga.py, fitness.py

---

## RESUMO

Adicionado logging detalhado para diagnosticar:
1. **Cache SHA256** - Por que cache_hits e cache_misses sempre 0?
2. **Early Stop** - Por que early_stopped_count sempre 0?

---

## MUDANCAS EM ga.py

### 1. Flag de Debug Global (linhas 28-36)

```python
# DEBUG LAYER 1 - Diagnóstico de Cache e Early Stop
DEBUG_LAYER1 = True  # Ativar para diagnóstico detalhado

def debug_print(msg):
    """Helper para debug logging condicional"""
    if DEBUG_LAYER1:
        logging.warning(f"[DEBUG L1] {msg}")
```

**Uso:** Permite ativar/desativar debug facilmente mudando DEBUG_LAYER1 = False

---

### 2. Debug em hash_individual() (linhas 831-833)

```python
# DEBUG: Log hash gerado
if generation == 1 and len(fitness_cache) < 3:
    debug_print(f"Hash gerado: {hash_val[:16]}... (rules_len={len(rules_string)})")
```

**O que mostra:**
- Hash SHA256 gerado (primeiros 16 chars)
- Tamanho da string de regras
- Apenas primeiras 3 chamadas da geração 1 (evita poluir log)

**Exemplo esperado:**
```
[DEBUG L1] Hash gerado: 3f2a8b9c1d4e5f6a... (rules_len=234)
```

---

### 3. Debug no início da avaliação (linhas 916-917)

```python
# DEBUG: Log inicial da geração
if generation <= 2:
    debug_print(f"Gen {generation+1}: Avaliando {len(population)} indivíduos (cache size={len(fitness_cache)})")
```

**O que mostra:**
- Quantos indivíduos serão avaliados
- Tamanho atual do cache
- Apenas gerações 0, 1, 2

**Exemplo esperado:**
```
[DEBUG L1] Gen 1: Avaliando 120 indivíduos (cache size=0)
[DEBUG L1] Gen 2: Avaliando 120 indivíduos (cache size=120)
```

---

### 4. Debug no cache lookup (linhas 923-925)

```python
# DEBUG: Log cache lookup (apenas primeiras gerações)
if generation <= 2 and cache_hits + cache_misses < 5:
    debug_print(f"Gen {generation+1}: Procurando hash {ind_hash[:16]}... no cache")
```

**O que mostra:**
- Hash sendo procurado no cache
- Apenas primeiros 5 lookups das gerações 0-2

**Exemplo esperado:**
```
[DEBUG L1] Gen 2: Procurando hash 3f2a8b9c1d4e5f6a... no cache
```

---

### 5. Debug cache HIT (linhas 945-947)

```python
# DEBUG: Log cache hit
if generation <= 2 and cache_hits <= 3:
    debug_print(f"Gen {generation+1}: CACHE HIT #{cache_hits} - hash {ind_hash[:16]}...")
```

**O que mostra:**
- Quando cache encontra indivíduo idêntico
- Número do hit
- Hash encontrado

**Exemplo esperado:**
```
[DEBUG L1] Gen 2: CACHE HIT #1 - hash 3f2a8b9c1d4e5f6a...
[DEBUG L1] Gen 2: CACHE HIT #2 - hash 7a9b2c3d4e5f6g8h...
```

---

### 6. Debug cache MISS (linhas 964-966)

```python
# DEBUG: Log cache miss
if generation <= 2 and cache_misses <= 3:
    debug_print(f"Gen {generation+1}: CACHE MISS #{cache_misses} - avaliando e armazenando hash {ind_hash[:16]}...")
```

**O que mostra:**
- Quando cache NÃO encontra indivíduo (precisa avaliar)
- Número do miss
- Hash que será armazenado

**Exemplo esperado:**
```
[DEBUG L1] Gen 1: CACHE MISS #1 - avaliando e armazenando hash 3f2a8b9c1d4e5f6a...
[DEBUG L1] Gen 1: CACHE MISS #2 - avaliando e armazenando hash 7a9b2c3d4e5f6g8h...
```

---

### 7. Debug early stopped individual (linhas 972-974)

```python
# DEBUG: Log early stopped
if generation <= 5 and early_stopped_count <= 5:
    debug_print(f"Gen {generation+1}: EARLY STOPPED individual #{early_stopped_count} (gmean={gmean:.3f})")
```

**O que mostra:**
- Quando indivíduo é descartado por early stop
- G-mean parcial do indivíduo descartado
- Primeiros 5 descartes das gerações 0-5

**Exemplo esperado:**
```
[DEBUG L1] Gen 2: EARLY STOPPED individual #1 (gmean=0.234)
[DEBUG L1] Gen 2: EARLY STOPPED individual #2 (gmean=0.189)
```

---

### 8. Debug resumo da geração (linhas 989-991)

```python
# DEBUG: Resumo da geração
if generation <= 3:
    debug_print(f"Gen {generation+1}: cache_hits={cache_hits}, cache_misses={cache_misses}, early_stopped={early_stopped_count}")
```

**O que mostra:**
- Totais de cache e early stop por geração
- Gerações 0-3

**Exemplo esperado:**
```
[DEBUG L1] Gen 1: cache_hits=0, cache_misses=120, early_stopped=0
[DEBUG L1] Gen 2: cache_hits=12, cache_misses=108, early_stopped=8
```

---

### 9. Debug problema cache (linhas 997-1000)

```python
# DEBUG: Por que não teve cache?
if generation <= 3:
    debug_print(f"Gen {generation+1}: PROBLEMA - Nenhuma operação de cache (hits=0, misses=0)")
```

**O que mostra:**
- ALERTA quando cache não está sendo usado
- Indica que código de cache não executou

**Exemplo esperado (SE PROBLEMA):**
```
[DEBUG L1] Gen 1: PROBLEMA - Nenhuma operação de cache (hits=0, misses=0)
```

---

### 10. Debug problema early stop (linhas 1006-1008)

```python
# DEBUG: Por que não teve early stop?
if generation >= 2 and generation <= 5:
    debug_print(f"Gen {generation+1}: Nenhum early stop (threshold pode estar muito alto)")
```

**O que mostra:**
- ALERTA quando early stop não descarta ninguém
- Gerações 2-5 (após threshold estar calculado)

**Exemplo esperado (SE PROBLEMA):**
```
[DEBUG L1] Gen 2: Nenhum early stop (threshold pode estar muito alto)
[DEBUG L1] Gen 3: Nenhum early stop (threshold pode estar muito alto)
```

---

## MUDANCAS EM fitness.py

### 1. Flag de Debug Global (linhas 9-15)

```python
# DEBUG LAYER 1 - Diagnóstico de Early Stop
DEBUG_LAYER1_FITNESS = True

def debug_print_fitness(msg):
    """Helper para debug logging condicional em fitness"""
    if DEBUG_LAYER1_FITNESS:
        logging.warning(f"[DEBUG L1 FITNESS] {msg}")
```

---

### 2. Contadores estáticos (linhas 216-218)

```python
# DEBUG: Contadores estáticos para logging controlado
if not hasattr(calculate_fitness_hybrid, 'early_stop_checks'):
    calculate_fitness_hybrid.early_stop_checks = 0
    calculate_fitness_hybrid.early_stop_discards = 0
```

**O que faz:**
- Cria contadores globais para early stop
- Persiste entre chamadas da função

---

### 3. Debug early stop check (linhas 224-226)

```python
# DEBUG: Log primeiras verificações
if calculate_fitness_hybrid.early_stop_checks < 5:
    debug_print_fitness(f"Early stop check #{calculate_fitness_hybrid.early_stop_checks + 1}: threshold={early_stop_threshold:.3f}, partial_size={partial_size}/{len(data)}")
```

**O que mostra:**
- Threshold sendo usado
- Tamanho da amostra parcial (20%)
- Primeiras 5 verificações

**Exemplo esperado:**
```
[DEBUG L1 FITNESS] Early stop check #1: threshold=0.763, partial_size=1200/6000
```

---

### 4. Debug early stop comparação (linhas 247-249)

```python
# DEBUG: Log comparação
if calculate_fitness_hybrid.early_stop_checks <= 10:
    debug_print_fitness(f"Early stop eval: partial_gmean={partial_gmean:.3f} vs threshold_50%={threshold_50pct:.3f} (should_discard={partial_gmean < threshold_50pct})")
```

**O que mostra:**
- G-mean parcial calculado
- Threshold 50% (0.50 * mediana elite)
- Se deveria descartar (True/False)
- Primeiras 10 avaliações

**Exemplo esperado:**
```
[DEBUG L1 FITNESS] Early stop eval: partial_gmean=0.234 vs threshold_50%=0.382 (should_discard=True)
[DEBUG L1 FITNESS] Early stop eval: partial_gmean=0.512 vs threshold_50%=0.382 (should_discard=False)
```

---

### 5. Debug early stop descarte (linhas 254-256)

```python
# DEBUG: Log descarte
if calculate_fitness_hybrid.early_stop_discards <= 5:
    debug_print_fitness(f"EARLY STOP DESCARTE #{calculate_fitness_hybrid.early_stop_discards}: {partial_gmean:.3f} < {threshold_50pct:.3f}")
```

**O que mostra:**
- Quando indivíduo é efetivamente descartado
- G-mean parcial e threshold
- Primeiros 5 descartes

**Exemplo esperado:**
```
[DEBUG L1 FITNESS] EARLY STOP DESCARTE #1: 0.234 < 0.382
[DEBUG L1 FITNESS] EARLY STOP DESCARTE #2: 0.189 < 0.382
```

---

### 6. Debug early stop desativado (linhas 278-280)

```python
# DEBUG: Log quando early stop não está ativo
if not hasattr(calculate_fitness_hybrid, 'no_early_stop_logged'):
    calculate_fitness_hybrid.no_early_stop_logged = True
    debug_print_fitness(f"Early stop DESATIVADO: threshold={early_stop_threshold}")
```

**O que mostra:**
- Quando early stop não está ativo (threshold None ou 0)
- Apenas uma vez

**Exemplo esperado (SE PROBLEMA):**
```
[DEBUG L1 FITNESS] Early stop DESATIVADO: threshold=None
```

---

## EXEMPLO DE LOG COMPLETO ESPERADO

### Geração 1 (sem cache ainda)

```
[DEBUG L1] Gen 1: Avaliando 120 indivíduos (cache size=0)
[DEBUG L1] Hash gerado: 3f2a8b9c1d4e5f6a... (rules_len=234)
[DEBUG L1] Hash gerado: 7a9b2c3d4e5f6g8h... (rules_len=189)
[DEBUG L1 FITNESS] Early stop DESATIVADO: threshold=0.0
[DEBUG L1] Gen 1: Procurando hash 3f2a8b9c1d4e5f6a... no cache
[DEBUG L1] Gen 1: CACHE MISS #1 - avaliando e armazenando hash 3f2a8b9c1d4e5f6a...
[DEBUG L1] Gen 1: Procurando hash 7a9b2c3d4e5f6g8h... no cache
[DEBUG L1] Gen 1: CACHE MISS #2 - avaliando e armazenando hash 7a9b2c3d4e5f6g8h...
[DEBUG L1] Gen 1: cache_hits=0, cache_misses=120, early_stopped=0
[CACHE] Gen 1: Hits=0/120 (0.0%)
```

### Geração 2 (com threshold e cache)

```
[DEBUG L1] Gen 2: Avaliando 120 indivíduos (cache size=120)
[DEBUG L1 FITNESS] Early stop check #1: threshold=0.763, partial_size=1200/6000
[DEBUG L1] Gen 2: Procurando hash 3f2a8b9c1d4e5f6a... no cache
[DEBUG L1] Gen 2: CACHE HIT #1 - hash 3f2a8b9c1d4e5f6a...
[DEBUG L1] Gen 2: Procurando hash 9c8d7e6f5a4b3c2d... no cache
[DEBUG L1] Gen 2: CACHE MISS #1 - avaliando e armazenando hash 9c8d7e6f5a4b3c2d...
[DEBUG L1 FITNESS] Early stop eval: partial_gmean=0.234 vs threshold_50%=0.382 (should_discard=True)
[DEBUG L1 FITNESS] EARLY STOP DESCARTE #1: 0.234 < 0.382
[DEBUG L1] Gen 2: EARLY STOPPED individual #1 (gmean=0.234)
[DEBUG L1 FITNESS] Early stop eval: partial_gmean=0.512 vs threshold_50%=0.382 (should_discard=False)
[DEBUG L1] Gen 2: cache_hits=12, cache_misses=100, early_stopped=8
[CACHE] Gen 2: Hits=12/112 (10.7%)
[EARLY STOP] Gen 2: Descartados=8/120 (6.7%)
```

---

## CENARIOS DE DIAGNOSTICO

### Cenário 1: Cache NÃO funciona

**Sintomas esperados no log:**
```
[DEBUG L1] Gen 1: PROBLEMA - Nenhuma operação de cache (hits=0, misses=0)
[DEBUG L1] Gen 2: PROBLEMA - Nenhuma operação de cache (hits=0, misses=0)
```

**Causa provável:**
- Código de cache não está executando
- hash_individual() não sendo chamado
- Erro de indentação

---

### Cenário 2: Early Stop NÃO funciona

**Sintomas esperados no log:**
```
[DEBUG L1 FITNESS] Early stop check #1: threshold=0.763, partial_size=1200/6000
[DEBUG L1 FITNESS] Early stop eval: partial_gmean=0.512 vs threshold_50%=0.382 (should_discard=False)
[DEBUG L1 FITNESS] Early stop eval: partial_gmean=0.634 vs threshold_50%=0.382 (should_discard=False)
[DEBUG L1] Gen 2: Nenhum early stop (threshold pode estar muito alto)
```

**Causa provável:**
- Threshold muito alto (todos indivíduos passam)
- Threshold 50% precisa ser ajustado para 60-70%

---

### Cenário 3: Early Stop DESATIVADO

**Sintomas esperados no log:**
```
[DEBUG L1 FITNESS] Early stop DESATIVADO: threshold=0.0
[DEBUG L1 FITNESS] Early stop DESATIVADO: threshold=None
```

**Causa provável:**
- Threshold não sendo calculado (generation == 0 ou population < 12)
- Early stop só ativa após geração 1

---

## PROXIMO PASSO: SMOKE TEST

### Comando

```powershell
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"

python main.py config_test_single.yaml --num_chunks 2 --run_number 998 2>&1 | Tee-Object -FilePath "smoke_test_debug_layer1.log"
```

### O que verificar no log

1. **Cache funcionando?**
   - [ ] Logs `[DEBUG L1] Hash gerado` aparecem?
   - [ ] Logs `[DEBUG L1] CACHE HIT` aparecem (gen 2+)?
   - [ ] Logs `[DEBUG L1] CACHE MISS` aparecem?
   - [ ] Logs `[CACHE] Gen X: Hits=...` aparecem?

2. **Early Stop funcionando?**
   - [ ] Logs `[DEBUG L1 FITNESS] Early stop check` aparecem?
   - [ ] Logs `[DEBUG L1 FITNESS] Early stop eval` aparecem?
   - [ ] Logs `[DEBUG L1 FITNESS] EARLY STOP DESCARTE` aparecem?
   - [ ] Logs `[EARLY STOP] Gen X: Descartados=...` aparecem?

3. **Problemas?**
   - [ ] Logs `PROBLEMA - Nenhuma operação de cache` aparecem?
   - [ ] Logs `Nenhum early stop (threshold pode estar muito alto)` aparecem?
   - [ ] Logs `Early stop DESATIVADO` aparecem?

### Análise rápida

```powershell
# Ver todos debug logs
Select-String -Path "smoke_test_debug_layer1.log" -Pattern "\[DEBUG L1"

# Ver cache
Select-String -Path "smoke_test_debug_layer1.log" -Pattern "CACHE"

# Ver early stop
Select-String -Path "smoke_test_debug_layer1.log" -Pattern "EARLY STOP"

# Ver problemas
Select-String -Path "smoke_test_debug_layer1.log" -Pattern "PROBLEMA|DESATIVADO"
```

---

## DESATIVAR DEBUG (APOS DIAGNOSTICO)

Quando problema for identificado e corrigido:

**ga.py linha 31:**
```python
DEBUG_LAYER1 = False  # Desativar debug
```

**fitness.py linha 10:**
```python
DEBUG_LAYER1_FITNESS = False  # Desativar debug
```

---

**Status:** DEBUG ADICIONADO, SINTAXE VALIDADA

**Próximo comando:** Executar smoke test com debug ativo
