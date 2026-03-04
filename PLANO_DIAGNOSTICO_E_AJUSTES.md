# PLANO DE DIAGNÓSTICO E AJUSTES - FASE 1 E FASE 2

**Data:** 2025-11-01
**Status:** Fase 2 sincronizada mas invisível (logging WARNING)

---

## 🔍 DIAGNÓSTICO: POR QUE FASE 2 ESTÁ INVISÍVEL?

### Problema Identificado

**CAUSA RAIZ CONFIRMADA:**
```yaml
# config_test_drift_recovery.yaml linha 10
logging_level: WARNING  # OTIMIZAÇÃO FASE 1.5: Reduzir overhead de logging
```

**Código de Fase 2 em main.py linha 573 e 584:**
```python
logging.info(f"📌 Chunk {i}: RECUPERANDO memória...")  # INFO < WARNING → NÃO APARECE
logging.info(f"🆕 Chunk {i}: NOVO conceito...")        # INFO < WARNING → NÃO APARECE
```

**Resultado:**
- ✅ Fase 2 **ESTÁ RODANDO** (código sincronizado)
- ❌ Fase 2 **ESTÁ INVISÍVEL** (logs não aparecem)
- ❓ Fase 2 **PODE ESTAR COM BUGS** (sem visibilidade para diagnosticar)

---

## 🎯 ESTRATÉGIA DE DIAGNÓSTICO E CORREÇÃO

### Abordagem em 3 Camadas

**CAMADA 1: LOGGING SUPER INFORMATIVO** ⭐⭐⭐
- Adicionar logging WARNING/INFO em pontos críticos
- Métricas de performance por fase
- Timing detalhado por chunk e geração
- Status de cache, early stop, HC

**CAMADA 2: CALIBRAÇÃO DE FASE 1** ⭐⭐
- Ajustar early stopping threshold (75% → 50% ou adaptativo)
- Validar cache de fitness (detectar colisões)
- Revisar HC (8 variantes pode ser insuficiente)

**CAMADA 3: VALIDAÇÃO DE FASE 2** ⭐⭐
- Elevar logs críticos para WARNING
- Adicionar métricas de detecção de recorrência
- Validar fingerprint similarity

---

## 📊 CAMADA 1: LOGGING SUPER INFORMATIVO

### 1.1. Estrutura de Logging Proposta

**Níveis de Logging:**
```python
CRITICAL: Erros que impedem execução
ERROR: Problemas que afetam resultados
WARNING: Informações importantes de diagnóstico (FASE 1 E FASE 2)
INFO: Detalhes de execução normal (progresso de gerações)
DEBUG: Informações técnicas detalhadas
```

**Configuração Recomendada:**
```yaml
# Durante diagnóstico: INFO (ver tudo)
logging_level: INFO

# Produção (após validação): WARNING (performance)
logging_level: WARNING
```

---

### 1.2. Logs Críticos para Fase 1 (nível WARNING)

**Em ga.py - Início de cada chunk:**
```python
logger.warning(f"")
logger.warning(f"{'='*70}")
logger.warning(f"CHUNK {chunk_index} - INÍCIO GA")
logger.warning(f"{'='*70}")
logger.warning(f"População: {pop_size}")
logger.warning(f"Gerações máx: {max_gen}")
logger.warning(f"Early stop threshold: {early_stop_threshold:.4f}")
logger.warning(f"Cache fitness: {'ATIVO' if cache_enabled else 'INATIVO'}")
logger.warning(f"HC max variantes: {hc_max_variants}")
```

**Em ga.py - A cada 10 gerações:**
```python
if generation % 10 == 0:
    logger.warning(f"")
    logger.warning(f"--- Gen {generation}/{max_gen} - Status ---")
    logger.warning(f"  Best G-mean: {best_gmean:.4f}")
    logger.warning(f"  Cache hits: {cache_hits}/{cache_hits+cache_misses} ({cache_hits/(cache_hits+cache_misses)*100:.1f}%)")
    logger.warning(f"  Early stops: {early_stop_count} ({early_stop_count/pop_size*100:.1f}%)")
    logger.warning(f"  HC invocações: {hc_invocations}")
    logger.warning(f"  Tempo acumulado: {time_elapsed:.1f}s")
```

**Em ga.py - Final do chunk:**
```python
logger.warning(f"")
logger.warning(f"{'='*70}")
logger.warning(f"CHUNK {chunk_index} - FINAL GA")
logger.warning(f"{'='*70}")
logger.warning(f"Best final G-mean: {best_gmean:.4f}")
logger.warning(f"Gerações executadas: {final_gen}/{max_gen}")
logger.warning(f"Cache - Hits: {cache_hits}, Misses: {cache_misses}, Taxa: {cache_hit_rate:.1f}%")
logger.warning(f"Early stop - Ativações: {early_stop_count}/{total_evaluations} ({early_stop_rate:.1f}%)")
logger.warning(f"Tempo total chunk: {chunk_time:.1f}s")
logger.warning(f"Tempo médio/geração: {avg_gen_time:.1f}s")
```

---

### 1.3. Logs Críticos para Fase 2 (nível WARNING)

**Em main.py - Cálculo de fingerprint (linha 553):**
```python
logger.warning(f"")
logger.warning(f"[FASE 2] Chunk {i} - Calculando concept fingerprint...")
try:
    chunk_fingerprint = utils.calculate_concept_fingerprint(
        data=train_data_chunk,
        target=train_target_chunk,
        attributes=numeric_features
    )
    logger.warning(f"[FASE 2]   ✓ Fingerprint calculada: {chunk_fingerprint['num_instances']} instâncias")
    logger.warning(f"[FASE 2]   - Mean shape: {chunk_fingerprint['mean'].shape}")
    logger.warning(f"[FASE 2]   - Classes: {chunk_fingerprint['class_distribution'].shape}")
except Exception as e:
    logger.error(f"[FASE 2]   ✗ ERRO ao calcular fingerprint: {e}", exc_info=True)
    # Fallback: não usar Fase 2 neste chunk
    chunk_fingerprint = {'num_instances': 0}
```

**Em main.py - Detecção de recorrência (linha 560):**
```python
logger.warning(f"[FASE 2] Chunk {i} - Detectando conceito recorrente...")
logger.warning(f"[FASE 2]   Conceitos conhecidos: {list(concept_memory.keys())}")
logger.warning(f"[FASE 2]   Threshold similaridade: {fingerprint_threshold:.2f}")

try:
    is_recurring, matched_concept_id = utils.detect_recurring_concept(
        current_fingerprint=chunk_fingerprint,
        concept_memory=concept_memory,
        similarity_threshold=fingerprint_threshold
    )

    if is_recurring:
        logger.warning(f"[FASE 2]   ✓ CONCEITO RECORRENTE: '{matched_concept_id}'")
    else:
        logger.warning(f"[FASE 2]   ✓ NOVO CONCEITO: 'concept_{i}'")
except Exception as e:
    logger.error(f"[FASE 2]   ✗ ERRO ao detectar recorrência: {e}", exc_info=True)
    is_recurring = False
    matched_concept_id = None
```

**Em main.py - Restauração/salvamento de memória:**
```python
if is_recurring:
    logger.warning(f"[FASE 2] Chunk {i} - Restaurando memória de '{current_concept_id}'")
    stored_memory = concept_memory[current_concept_id].get('memory', [])
    logger.warning(f"[FASE 2]   → {len(stored_memory)} indivíduos disponíveis")
    if stored_memory:
        best_ever_memory = [copy.deepcopy(ind) for ind in stored_memory]
        logger.warning(f"[FASE 2]   ✓ {len(best_ever_memory)} indivíduos restaurados")
else:
    logger.warning(f"[FASE 2] Chunk {i} - Criando novo conceito 'concept_{i}'")

# Após GA executar (linha 959)
logger.warning(f"[FASE 2] Chunk {i} - Salvando memória do conceito '{current_concept_id}'")
logger.warning(f"[FASE 2]   → {len(best_ever_memory)} indivíduos salvos")
concept_memory[current_concept_id]['memory'] = [copy.deepcopy(ind) for ind in best_ever_memory]
concept_memory[current_concept_id]['last_seen_chunk'] = i
logger.warning(f"[FASE 2]   ✓ Memória atualizada (last_seen: {i})")
```

**Em main.py - Drift detection com Fase 2 (linha 1043):**
```python
if performance_drop > 0.20:  # SEVERE
    drift_severity = 'SEVERE'
    logger.warning(f"🔴 SEVERE DRIFT detected: {previous_gmean:.3f} → {current_gmean:.3f}")

    if not is_recurring:
        logger.warning(f"[FASE 2]   → Memory REDUCED (novo conceito)")
        # ... código de redução ...
    else:
        logger.warning(f"[FASE 2]   → Memory PRESERVADA (conceito recorrente '{current_concept_id}')")
```

---

### 1.4. Logs de Timing (Crítico para Performance)

**Em main.py - Início de cada chunk:**
```python
import time

chunk_start_time = time.time()
logger.warning(f"")
logger.warning(f"{'='*70}")
logger.warning(f"CHUNK {i} - INÍCIO")
logger.warning(f"{'='*70}")
logger.warning(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
```

**Em main.py - Final de cada chunk:**
```python
chunk_end_time = time.time()
chunk_duration = chunk_end_time - chunk_start_time

logger.warning(f"")
logger.warning(f"{'='*70}")
logger.warning(f"CHUNK {i} - FINAL")
logger.warning(f"{'='*70}")
logger.warning(f"Tempo total: {chunk_duration:.1f}s ({chunk_duration/60:.1f}min)")
logger.warning(f"Train G-mean: {train_gmean:.4f}")
logger.warning(f"Test G-mean: {test_gmean:.4f}")
logger.warning(f"Delta: {test_gmean - train_gmean:+.4f}")
logger.warning(f"Drift severity: {drift_severity}")
```

**Em main.py - Final do experimento:**
```python
experiment_end_time = time.time()
total_duration = experiment_end_time - experiment_start_time

logger.warning(f"")
logger.warning(f"{'#'*70}")
logger.warning(f"EXPERIMENTO FINALIZADO")
logger.warning(f"{'#'*70}")
logger.warning(f"Tempo total: {total_duration:.1f}s ({total_duration/3600:.2f}h)")
logger.warning(f"Média por chunk: {total_duration/num_chunks_to_process:.1f}s")
logger.warning(f"Avg Test G-mean: {np.mean(historical_gmean):.4f}")
logger.warning(f"Chunks processados: {num_chunks_to_process}")
```

---

## 🔧 CAMADA 2: CALIBRAÇÃO DE FASE 1

### 2.1. Early Stopping - Threshold Adaptativo

**Problema Atual:**
```python
# fitness.py linha 192-247
if partial_gmean < early_stop_threshold * 0.75:  # Muito agressivo
    return {'fitness': -float('inf'), ...}
```

**Análise:**
- Threshold fixo 75% é muito agressivo
- Em chunks com drift severo, worst_elite pode ter G-mean ~40%
- 40% * 0.75 = 30% → descarta tudo abaixo de 30%
- Muitos indivíduos com potencial (35-45%) são descartados

**Solução Proposta - Threshold Adaptativo:**
```python
# fitness.py - SUBSTITUIR linha 192-247

# FASE 1 OTIMIZADO: Early stopping com threshold adaptativo
if early_stop_threshold is not None and early_stop_threshold > 0:
    batch_size = 1000
    num_batches = (len(data) + batch_size - 1) // batch_size

    all_predictions = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))

        batch_data = data[start_idx:end_idx]
        batch_predictions = [individual._predict(inst) for inst in batch_data]
        all_predictions.extend(batch_predictions)

        # Após primeiro batch (16% dos dados), verifica se vale continuar
        if batch_idx >= 1:  # Avaliou pelo menos 2000 exemplos (33%)
            try:
                partial_gmean = calculate_gmean_contextual(
                    target[:end_idx],
                    all_predictions,
                    individual.classes
                )

                # NOVO: Threshold adaptativo baseado no contexto
                # Se worst_elite é alto (>70%), usa threshold conservador (50%)
                # Se worst_elite é baixo (<40%), usa threshold liberal (30%)
                if early_stop_threshold > 0.70:
                    adaptive_threshold = early_stop_threshold * 0.50  # Conservador
                elif early_stop_threshold < 0.40:
                    adaptive_threshold = max(0.25, early_stop_threshold * 0.65)  # Liberal
                else:
                    adaptive_threshold = early_stop_threshold * 0.55  # Balanceado

                # Early stop: Se G-mean parcial é muito pior que threshold adaptativo
                if partial_gmean < adaptive_threshold:
                    # Log de early stop (nível INFO para não poluir)
                    logging.debug(f"Early stop: partial_gmean={partial_gmean:.3f} < {adaptive_threshold:.3f}")
                    return {
                        'fitness': -float('inf'),
                        'g_mean': partial_gmean,
                        'weighted_f1': 0.0,
                        'early_stopped': True  # Flag para logging
                    }
            except:
                pass
```

**Ganho Esperado:**
- Menos false negatives (descarte de indivíduos bons)
- Melhor balanceamento tempo vs qualidade
- +1 a +2pp em G-mean

---

### 2.2. Cache de Fitness - Validação de Colisões

**Problema Atual:**
```python
# ga.py linha 803-918
def hash_individual(individual):
    rules_string = individual.get_rules_as_string()
    hash_string = f"{rules_string}|{individual.default_class}"
    return hash(hash_string)  # Python hash() pode ter colisões
```

**Solução Proposta - Hash SHA256 + Validação:**
```python
# ga.py - SUBSTITUIR função hash_individual

import hashlib

def hash_individual(individual):
    """
    Gera hash único baseado na estrutura completa de regras.
    Usa SHA256 para evitar colisões.
    """
    rules_string = individual.get_rules_as_string()
    hash_string = f"{rules_string}|{individual.default_class}"

    # Usa SHA256 (sem colisões práticas) ao invés de hash() nativo
    return hashlib.sha256(hash_string.encode('utf-8')).hexdigest()

# No código de cache, adicionar validação:
for individual in population:
    ind_hash = hash_individual(individual)

    if ind_hash in fitness_cache:
        # VALIDAÇÃO: Verifica se é realmente o mesmo indivíduo
        cached_rules = fitness_cache[ind_hash].get('rules_string', '')
        current_rules = individual.get_rules_as_string()

        if cached_rules == current_rules:
            # CACHE HIT VÁLIDO
            cached = fitness_cache[ind_hash]
            individual.fitness = cached['fitness']
            individual.gmean = cached['gmean']
            individual.activation_rate = cached['activation_rate']
            cache_hits += 1
        else:
            # COLISÃO DETECTADA (muito raro com SHA256, mas checamos)
            logger.warning(f"CACHE COLLISION detected! Hash: {ind_hash[:16]}...")
            # Avalia normalmente
            cache_misses += 1
            # ... código de avaliação normal ...
            # Salva com regra completa para validação futura
            fitness_cache[ind_hash] = {
                'fitness': fitness_score,
                'gmean': gmean,
                'activation_rate': activation_rate,
                'rules_string': current_rules  # NOVO
            }
    else:
        # CACHE MISS: Avalia e armazena
        # ... código normal ...
        fitness_cache[ind_hash] = {
            'fitness': fitness_score,
            'gmean': gmean,
            'activation_rate': activation_rate,
            'rules_string': individual.get_rules_as_string()  # NOVO
        }
        cache_misses += 1
```

**Ganho Esperado:**
- Zero colisões (SHA256 é criptograficamente seguro)
- Cache 100% confiável
- +0.5 a +1pp em G-mean (se havia colisões)

---

### 2.3. Hill Climbing - Revisar Limite de Variantes

**Problema Atual:**
```python
# hill_climbing_v2.py linha 603
max_variants_to_return = min(8, level_config['num_variants_base'], len(all_variants))
```

**Análise:**
- Redução de 12-18 para 8 pode ter reduzido exploração local
- HC é crítico para refinamento fino
- Trade-off: 8 variantes economiza ~25% tempo, mas pode perder ~5-10% qualidade

**Solução Proposta - Limite Adaptativo por Drift:**
```python
# hill_climbing_v2.py - SUBSTITUIR linha 603

# NOVO: Limite adaptativo baseado em severidade de drift
# Se drift severo: HC mais agressivo (12 variantes) para recovery rápido
# Se estável: HC econômico (8 variantes) para manter performance

# Pega severidade do drift do contexto (passado via config)
drift_severity = level_config.get('drift_severity', 'NONE')

if drift_severity in ['SEVERE', 'MODERATE']:
    max_variants = 12  # HC agressivo para recovery
    logger.info(f"     HC AGRESSIVO: {drift_severity} drift detectado, usando 12 variantes")
elif drift_severity == 'MILD':
    max_variants = 10  # HC balanceado
    logger.info(f"     HC BALANCEADO: MILD drift, usando 10 variantes")
else:
    max_variants = 8   # HC econômico (estável)
    logger.debug(f"     HC ECONÔMICO: STABLE, usando 8 variantes")

max_variants_to_return = min(max_variants, level_config['num_variants_base'], len(all_variants))

logger.info(f"     Total gerado: {len(all_variants)} variantes, retornando {max_variants_to_return} (limite: {max_variants})")
```

**Ganho Esperado:**
- Melhor balanceamento tempo vs qualidade
- HC mais agressivo quando necessário (drift)
- HC econômico quando possível (estável)
- +0.3 a +0.8pp em G-mean

---

## 🧪 CAMADA 3: VALIDAÇÃO DE FASE 2

### 3.1. Diagnóstico de Fingerprint Similarity

**Em utils.py - Adicionar logging em detect_recurring_concept:**
```python
def detect_recurring_concept(
    current_fingerprint: Dict[str, Any],
    concept_memory: Dict[str, Dict],
    similarity_threshold: float = 0.85
) -> Tuple[bool, str]:
    """Detecta se o conceito atual corresponde a algum conceito já conhecido."""

    if not concept_memory or current_fingerprint['num_instances'] == 0:
        logging.debug(f"[FASE 2] detect_recurring: memory vazia ou fingerprint inválida")
        return (False, '')

    best_match_id = ''
    best_similarity = 0.0

    # NOVO: Lista todas as similaridades para diagnóstico
    all_similarities = []

    try:
        for concept_id, concept_data in concept_memory.items():
            known_fingerprint = concept_data.get('fingerprint', {})

            if known_fingerprint.get('num_instances', 0) == 0:
                continue

            similarity = fingerprint_similarity(current_fingerprint, known_fingerprint)
            all_similarities.append((concept_id, similarity))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = concept_id

        # NOVO: Log todas as similaridades (nível WARNING para diagnóstico)
        if all_similarities:
            logging.warning(f"[FASE 2] Similaridades calculadas:")
            for cid, sim in sorted(all_similarities, key=lambda x: x[1], reverse=True):
                logging.warning(f"[FASE 2]   {cid}: {sim:.4f}")

        # Verifica threshold
        if best_similarity >= similarity_threshold:
            logging.warning(f"[FASE 2] ✓ MATCH: '{best_match_id}' (sim={best_similarity:.4f} >= {similarity_threshold:.2f})")
            return (True, best_match_id)
        else:
            logging.warning(f"[FASE 2] ✗ NO MATCH: melhor '{best_match_id}' (sim={best_similarity:.4f} < {similarity_threshold:.2f})")
            return (False, '')

    except Exception as e:
        logging.error(f"[FASE 2] ERRO em detect_recurring_concept: {e}", exc_info=True)
        return (False, '')
```

---

### 3.2. Threshold de Similaridade - Pode Estar Muito Alto

**Análise:**
- Threshold atual: 0.85 (85%)
- Conceitos similares mas com pequenas variações podem ter sim < 0.85
- Exemplo: c1 com diferentes seeds pode ter sim ~0.75-0.80

**Teste Proposto:**
```yaml
# config_test_drift_recovery.yaml
memory_params:
  concept_fingerprint_similarity_threshold: 0.75  # Reduzir de 0.85 para 0.75 (teste)
```

**Validação:**
- Rodar com 0.75 e ver se detecta recorrências
- Se detectar muitas false positives, aumentar para 0.80
- Se não detectar, problema é outro (fingerprint errado)

---

## 📋 PLANO DE IMPLEMENTAÇÃO

### Fase A: Logging Super Informativo (PRIORIDADE 1)

**Tempo:** 2-3h implementação

**Arquivos a modificar:**
1. `main.py` - Adicionar logs WARNING em Fase 2 (linhas 553-592, 959-964, 1043-1063)
2. `ga.py` - Adicionar logs de status a cada 10 gerações + summary no final
3. `utils.py` - Adicionar logs em detect_recurring_concept()

**Resultado esperado:**
- Logs completos mostrando:
  - Timing por chunk
  - Status de Fase 1 (cache, early stop, HC)
  - Status de Fase 2 (fingerprint, detecção, restauração)
  - Métricas finais

---

### Fase B: Calibração de Fase 1 (PRIORIDADE 2)

**Tempo:** 1-2h implementação

**Arquivos a modificar:**
1. `fitness.py` - Early stop adaptativo (linha 192-247)
2. `ga.py` - Cache com SHA256 + validação (linha 803-918)
3. `hill_climbing_v2.py` - HC adaptativo por drift (linha 603)

**Resultado esperado:**
- Early stop menos agressivo: +1-2pp
- Cache sem colisões: +0.5-1pp
- HC adaptativo: +0.3-0.8pp
- **Total: +2-4pp em G-mean**

---

### Fase C: Ajuste de Fase 2 (PRIORIDADE 3)

**Tempo:** 30min implementação

**Arquivos a modificar:**
1. `config_test_drift_recovery.yaml` - Threshold 0.85 → 0.75
2. `config_test_multi_drift.yaml` - Threshold 0.85 → 0.75

**Resultado esperado:**
- Mais detecções de recorrência
- Se funcionar: +10-15pp nos chunks recorrentes

---

## 🎯 CRONOGRAMA SUGERIDO

### DIA 1 (hoje):
1. **Implementar Fase A** (logging super informativo) - 2-3h
2. **Testar localmente** com 2 chunks pequenos - 30min
3. **Upload para Colab** - 10min
4. **Executar TEST_SINGLE primeiro** (validação rápida) - 6-8h

### DIA 2:
5. **Analisar logs de TEST_SINGLE** - 1h
6. **Se OK: Implementar Fase B** (calibração Fase 1) - 1-2h
7. **Upload e executar TEST_SINGLE novamente** - 6-8h

### DIA 3:
8. **Analisar resultados** - 1h
9. **Se OK: Implementar Fase C** (ajuste Fase 2) - 30min
10. **Executar DRIFT_RECOVERY e MULTI_DRIFT** - 12-16h

### DIA 4:
11. **Análise final** - 2h
12. **Documentar resultados** - 1h

---

## ✅ CHECKLIST DE VALIDAÇÃO

### Antes de cada execução:
- [ ] Logging configurado (INFO para diagnóstico, WARNING para produção)
- [ ] Arquivos sincronizados (utils.py, main.py, ga.py, fitness.py, configs)
- [ ] Teste local com 2 chunks (verificar logs aparecem)
- [ ] Backup dos logs anteriores

### Após cada execução:
- [ ] Verificar logs aparecem (Fase 2 deve ter mensagens)
- [ ] Extrair métricas (avg test G-mean)
- [ ] Comparar com baseline
- [ ] Identificar problemas nos logs

---

**FIM DO PLANO**

**Próximo passo:** Implementar Fase A (logging) ou discutir ajustes?
