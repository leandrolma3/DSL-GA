# DESIGN - IMPLEMENTACAO LAYER 1 NO MODO PARALELO

**Data:** 2025-11-04
**Objetivo:** Implementar Cache SHA256 e Early Stop no modo paralelo
**Complexidade:** MEDIA-ALTA
**Tempo estimado:** 4-6h

---

## ANALISE DO CODIGO ATUAL

### Modo Paralelo (ga.py linhas 858-906)

**Fluxo atual:**
1. Cria `constant_args_for_worker` com `early_stop_threshold` (linha 884)
2. Cria pool de workers (linha 894)
3. Chama `evaluate_individual_fitness_parallel` para cada individuo
4. Coleta resultados

**Problemas identificados:**
1. `evaluate_individual_fitness_parallel` NAO extrai `early_stop_threshold` do dict
2. `evaluate_individual_fitness_parallel` NAO passa threshold para `calculate_fitness`
3. Nao ha cache compartilhado entre workers
4. Nao ha contadores de cache hits/misses

---

### Modo Serial (ga.py linhas 907-1007)

**Fluxo atual:**
1. Cria `fitness_cache` local (dict comum)
2. Para cada individuo:
   - Gera hash SHA256 (hash_individual)
   - Verifica cache (com validacao de rules_string)
   - Se cache miss: chama `evaluate_individual_fitness_parallel`
   - Armazena resultado no cache
3. Contadores: cache_hits, cache_misses, early_stopped_count
4. Logs detalhados com debug_print

**Vantagens:**
- Cache funciona perfeitamente
- Early stop funciona (threshold passado via constant_args)
- Debug logging completo

---

## ARQUITETURA PROPOSTA

### Componente 1: Cache Compartilhado

**Solucao:** Usar `multiprocessing.Manager().dict()`

**Por que Manager.dict():**
- Compartilhado entre processos (pool workers)
- Thread-safe (lock automatico)
- Aceita hashable keys (str hash)
- Performance aceitavel para nosso caso (5-15% lookups por geracao)

**Alternativas descartadas:**
- dict comum: NAO funciona entre processos
- Redis/memcached: overhead desnecessario
- shared memory (multiprocessing.shared_memory): complexidade muito alta

**Localizacao:**
- Criar manager e shared_cache ANTES do loop de geracoes (linha ~750)
- Passar shared_cache via constant_args_for_worker
- Acessar em evaluate_individual_fitness_parallel

---

### Componente 2: Early Stop Threshold

**Problema atual:**
```python
# ga.py linha 161-173
metrics = fitness_module.calculate_fitness(
    individual, train_data, train_target,
    ...,
    reduce_change_penalties
    # FALTA: early_stop_threshold=...
)
```

**Solucao:**
```python
# Extrair do constant_args
early_stop_threshold = constant_args.get('early_stop_threshold', None)

# Passar para calculate_fitness
metrics = fitness_module.calculate_fitness(
    individual, train_data, train_target,
    ...,
    reduce_change_penalties,
    early_stop_threshold=early_stop_threshold  # ADICIONAR
)
```

---

### Componente 3: Logica de Cache no Paralelo

**Desafio:** Cache deve ser verificado ANTES de enviar para pool

**Abordagem 1 (ESCOLHIDA): Pre-filtrar antes do pool**
```python
# Antes do pool.map
individuals_to_evaluate = []
cache_hits = 0

for individual in population:
    ind_hash = hash_individual(individual)
    if ind_hash in shared_cache:
        # Cache hit
        cached = shared_cache[ind_hash]
        if cached['rules_string'] == individual.get_rules_as_string():
            individual.fitness = cached['fitness']
            individual.gmean = cached['gmean']
            # ... (atualizar listas)
            cache_hits += 1
            continue
    # Cache miss: adiciona para avaliar
    individuals_to_evaluate.append(individual)

# Avaliar apenas os que nao estao em cache
if individuals_to_evaluate:
    map_iterable = [(ind, constant_args_for_worker) for ind in individuals_to_evaluate]
    results = pool.map(evaluate_individual_fitness_parallel, map_iterable)
    # Processar resultados e atualizar cache
```

**Vantagens:**
- Cache verificado no processo principal (sem overhead de IPC)
- Pool avalia apenas individuos novos
- Economia real de CPU

**Abordagem 2 (DESCARTADA): Cache dentro do worker**
- Overhead de IPC (cada worker acessa Manager.dict)
- Menos eficiente
- Mais complexo

---

## MODIFICACOES NECESSARIAS

### Modificacao 1: Inicializacao do Manager (ANTES do loop de geracoes)

**Localizacao:** ga.py linha ~750 (antes de `for generation in range(num_generations)`)

```python
# LAYER 1: Inicializar cache compartilhado para modo paralelo
from multiprocessing import Manager
manager = Manager()
shared_cache = manager.dict()
cache_hits_total = 0
cache_misses_total = 0
cache_collisions_total = 0
```

---

### Modificacao 2: Passar shared_cache no constant_args

**Localizacao:** ga.py linha 863-885

```python
constant_args_for_worker = {
    'train_data': train_data,
    ...,
    'early_stop_threshold': early_stop_threshold,  # JA EXISTE
    'shared_cache': shared_cache  # ADICIONAR
}
```

---

### Modificacao 3: Logica de cache ANTES do pool.map

**Localizacao:** ga.py linha 888-906 (substituir bloco)

```python
if parallel_enabled and workers_to_use > 1 and len(population) > 1:
    logging.debug(f"Gen {generation+1}: Starting parallel fitness evaluation with {workers_to_use} workers.")

    # LAYER 1: Pre-filtrar cache
    individuals_to_evaluate = []
    cache_hits = 0
    cache_misses = 0

    # DEBUG: Log inicial da geracao
    if generation <= 2:
        debug_print(f"Gen {generation+1}: Avaliando {len(population)} individuos (cache size={len(shared_cache)})")

    for individual in population:
        # Verificar cache
        ind_hash = hash_individual(individual)

        # DEBUG: Log cache lookup (apenas primeiras geracoes)
        if generation <= 2 and cache_hits + cache_misses < 5:
            debug_print(f"Gen {generation+1}: Procurando hash {ind_hash[:16]}... no cache")

        cache_hit = False
        if ind_hash in shared_cache:
            cached = shared_cache[ind_hash]
            current_rules = individual.get_rules_as_string()

            if 'rules_string' in cached and cached['rules_string'] == current_rules:
                # CACHE HIT
                individual.fitness = cached['fitness']
                individual.gmean = cached['gmean']
                fitness_values.append(cached['fitness'])
                gmean_values.append(cached['gmean'])
                activation_rates.append(cached['activation_rate'])
                cache_hits += 1
                cache_hit = True

                # DEBUG: Log cache hit
                if generation <= 2 and cache_hits <= 3:
                    debug_print(f"Gen {generation+1}: CACHE HIT #{cache_hits} - hash {ind_hash[:16]}...")
            else:
                # Colisao detectada
                logging.warning(f"Gen {generation+1}: Cache collision detected! Hash: {ind_hash[:16]}...")
                cache_collisions_total += 1

        if not cache_hit:
            # CACHE MISS: adiciona para avaliar
            individuals_to_evaluate.append(individual)
            cache_misses += 1

            # DEBUG: Log cache miss
            if generation <= 2 and cache_misses <= 3:
                debug_print(f"Gen {generation+1}: CACHE MISS #{cache_misses} - avaliando hash {ind_hash[:16]}...")

    # Avaliar individuos com cache miss
    early_stopped_count = 0
    if individuals_to_evaluate:
        map_iterable = [(ind, constant_args_for_worker) for ind in individuals_to_evaluate]
        try:
            with multiprocessing.Pool(processes=workers_to_use) as pool:
                results = pool.map(evaluate_individual_fitness_parallel, map_iterable)

            # Processar resultados
            valid_results = 0
            for i, individual in enumerate(individuals_to_evaluate):
                fitness_score, gmean, activation_rate = results[i]
                individual.fitness = fitness_score
                individual.gmean = gmean
                fitness_values.append(fitness_score)
                gmean_values.append(gmean)
                activation_rates.append(activation_rate)

                # Verificar se foi early stopped
                if fitness_score == -float('inf') and gmean > 0:
                    early_stopped_count += 1

                    # DEBUG: Log early stopped
                    if generation <= 5 and early_stopped_count <= 5:
                        debug_print(f"Gen {generation+1}: EARLY STOPPED individual #{early_stopped_count} (gmean={gmean:.3f})")

                # Armazenar no cache
                ind_hash = hash_individual(individual)
                shared_cache[ind_hash] = {
                    'fitness': fitness_score,
                    'gmean': gmean,
                    'activation_rate': activation_rate,
                    'rules_string': individual.get_rules_as_string()
                }

                if fitness_score > -float('inf'):
                    valid_results += 1

            if valid_results < len(individuals_to_evaluate):
                logging.warning(f"Gen {generation+1}: Only {valid_results}/{len(individuals_to_evaluate)} individuals evaluated successfully in parallel.")
            if valid_results == 0:
                logging.error(f"Gen {generation+1}: Parallel eval failed for all. Stopping.")
                break
        except Exception as pool_e:
            logging.error(f"Error during parallel fitness eval: {pool_e}", exc_info=True)
            logging.error("Stopping GA run.")
            break

    # Atualizar contadores globais
    cache_hits_total += cache_hits
    cache_misses_total += cache_misses

    # DEBUG: Resumo da geracao
    if generation <= 3:
        debug_print(f"Gen {generation+1}: cache_hits={cache_hits}, cache_misses={cache_misses}, early_stopped={early_stopped_count}")

    # Log do cache e early stopping
    if cache_hits > 0 or cache_misses > 0:
        hit_rate = (cache_hits / (cache_hits + cache_misses)) * 100 if (cache_hits + cache_misses) > 0 else 0
        logging.warning(f"   [CACHE] Gen {generation+1}: Hits={cache_hits}/{cache_hits + cache_misses} ({hit_rate:.1f}%)")

    if early_stopped_count > 0:
        early_stop_pct = (early_stopped_count / len(individuals_to_evaluate)) * 100 if individuals_to_evaluate else 0
        logging.warning(f"   [EARLY STOP] Gen {generation+1}: Descartados={early_stopped_count}/{len(individuals_to_evaluate)} ({early_stop_pct:.1f}%)")

else: # Execucao Serial
    # ... (codigo serial permanece inalterado)
```

---

### Modificacao 4: Extrair e passar early_stop_threshold no worker

**Localizacao:** ga.py linha 135-173 (evaluate_individual_fitness_parallel)

```python
def evaluate_individual_fitness_parallel(worker_args):
    individual, constant_args = worker_args
    individual_repr = f"Ind (Rules: {individual.count_total_rules()})"
    fitness_score, gmean, activation_rate = 0.0, 0.0, 0.0
    try:
        train_data = constant_args['train_data']
        train_target = constant_args['train_target']
        class_weights = constant_args['class_weights']
        regularization_coefficient = constant_args['regularization_coefficient']
        feature_penalty_coefficient = constant_args['feature_penalty_coefficient']
        reference_features = constant_args['reference_features']
        beta = constant_args['beta']
        previous_used_features = constant_args['previous_used_features']
        gamma = constant_args['gamma']
        operator_penalty_coefficient = constant_args['operator_penalty_coefficient']
        threshold_penalty_coefficient = constant_args['threshold_penalty_coefficient']
        previous_operator_info = constant_args['previous_operator_info']
        operator_change_coefficient = constant_args['operator_change_coefficient']
        attributes = constant_args['attributes']
        categorical_features = constant_args['categorical_features']
        reduce_change_penalties = constant_args.get('reduce_change_penalties', False)
        gmean_bonus_coefficient_ga = constant_args['gmean_bonus_coefficient']
        class_coverage_coefficient = constant_args['class_coverage_coefficient']

        # LAYER 1: Extrair early_stop_threshold
        early_stop_threshold = constant_args.get('early_stop_threshold', None)

        metrics = fitness_module.calculate_fitness(
            individual, train_data, train_target,
            class_weights,
            regularization_coefficient, feature_penalty_coefficient,
            class_coverage_coefficient, gmean_bonus_coefficient_ga,
            operator_change_coefficient,
            gamma,
            beta,
            reference_features,
            previous_used_features,
            previous_operator_info,
            reduce_change_penalties,
            early_stop_threshold=early_stop_threshold  # LAYER 1: PASSAR THRESHOLD
        )

        fitness_score = metrics.get('fitness', -float('inf'))
        gmean = metrics.get('g_mean', 0.0)

        individual.update_rule_quality_scores(train_data, train_target)

        activation_rate = 0.0
        if train_data:
            activations = sum(1 for inst in train_data if individual._predict_with_activation_check(inst)[1])
            activation_rate = activations / len(train_data) if len(train_data) > 0 else 0.0

        return (fitness_score, gmean, activation_rate)

    except Exception as e:
        logging.error(f"Worker {os.getpid()}: Erro critico na avaliacao do individuo: {e}", exc_info=True)
        return (-float('inf'), 0.0, 0.0)
```

---

## VALIDACAO

### Teste 1: Sintaxe

```powershell
python -m py_compile ga.py
```

Deve executar sem erros.

---

### Teste 2: Smoke Test (2 chunks)

```powershell
python main.py config_test_single.yaml --num_chunks 2 --run_number 997 2>&1 | Tee-Object -FilePath "smoke_test_layer1_paralelo.log"
```

**Verificar no log:**
1. [ ] `[DEBUG L1] Gen X: Avaliando Y individuos (cache size=Z)`
2. [ ] `[DEBUG L1] Gen X: Procurando hash ...`
3. [ ] `[DEBUG L1] Gen X: CACHE HIT #...`
4. [ ] `[DEBUG L1] Gen X: CACHE MISS #...`
5. [ ] `[DEBUG L1] Gen X: EARLY STOPPED individual #...`
6. [ ] `[CACHE] Gen X: Hits=.../... (...%)`
7. [ ] `[EARLY STOP] Gen X: Descartados=.../...`
8. [ ] Tempo por chunk < Run5 (esperado: -30-40%)

---

### Teste 3: Analise Quantitativa

```powershell
# Contar logs de cache
Select-String -Path "smoke_test_layer1_paralelo.log" -Pattern "CACHE HIT" | Measure-Object

# Contar logs de early stop
Select-String -Path "smoke_test_layer1_paralelo.log" -Pattern "EARLY STOPPED" | Measure-Object

# Ver resumos de geracao
Select-String -Path "smoke_test_layer1_paralelo.log" -Pattern "cache_hits=.*cache_misses=.*early_stopped="
```

**Metricas de sucesso:**
- Cache hits >= 5% (geracao 2+)
- Early stop descartes >= 3% (geracao 2+)
- Tempo < 90min por chunk (vs 154min no Run5)

---

## RISCOS E MITIGACOES

### Risco 1: Manager.dict() muito lento

**Sintoma:** Tempo por geracao aumenta em vez de diminuir

**Mitigacao:**
- Manager.dict() tem overhead de IPC (~10-20 microsegundos por operacao)
- Nossa taxa de acesso: ~120 lookups/geracao = 2.4ms overhead
- Aceitavel comparado ao ganho de evitar avaliacoes (~100ms cada)

**Plano B:** Se overhead for inaceitavel (>5%), usar cache local por worker

---

### Risco 2: Race conditions no cache

**Sintoma:** Colisoes de hash espurias, dados corrompidos

**Mitigacao:**
- Manager.dict() e thread-safe por design
- Operacoes atomicas: `dict[key] = value` e `key in dict`
- Validacao de rules_string previne falsos positivos

**Plano B:** Adicionar locks explicitos se necessario (improvavel)

---

### Risco 3: Early stop threshold None

**Sintoma:** Logs "Early stop DESATIVADO: threshold=None" persistem

**Mitigacao:**
- Extrair com `.get('early_stop_threshold', None)` (seguro)
- Passar explicitamente como kwarg
- Debug logging mostra se chegou ao worker

**Plano B:** Forcar threshold minimo (0.1) se None

---

## CRONOGRAMA DE IMPLEMENTACAO

### Fase 1: Modificacoes basicas (1-2h)
1. [ ] Adicionar inicializacao do Manager (Mod 1)
2. [ ] Adicionar shared_cache ao constant_args (Mod 2)
3. [ ] Extrair e passar early_stop_threshold no worker (Mod 4)
4. [ ] Validar sintaxe

### Fase 2: Logica de cache paralelo (2-3h)
1. [ ] Implementar pre-filtragem de cache (Mod 3)
2. [ ] Adicionar debug logging
3. [ ] Adicionar contadores de cache e early stop
4. [ ] Validar sintaxe

### Fase 3: Testes (1-2h)
1. [ ] Smoke test 2 chunks
2. [ ] Analisar logs
3. [ ] Ajustes finos se necessario
4. [ ] Documentar resultados

---

## METRICAS DE SUCESSO

Implementacao bem-sucedida se:

1. **Sintaxe valida:** `py_compile` sem erros
2. **Logs aparecem:**
   - Cache hits >= 5 ocorrencias
   - Early stop descartes >= 3 ocorrencias
3. **Performance melhora:**
   - Tempo por chunk < 90min (vs 154min Run5)
   - Reducao >= 35-40%
4. **Qualidade mantida:**
   - G-mean >= 0.75 (similar ou melhor)

---

**Status:** DESIGN COMPLETO, PRONTO PARA IMPLEMENTACAO
**Proximo passo:** Implementar Modificacao 1
