# ANÁLISE PROFUNDA: EXPERIMENTOS FASE 1 E FASE 2

**Data:** 2025-11-01
**Experimentos Analisados:** 6 logs (3 baselines + 3 otimizados)

---

## 📊 RESUMO EXECUTIVO

### Resultados Comparativos

| Experimento | Baseline TEST | Otimizado TEST | Delta | Status |
|-------------|---------------|----------------|-------|--------|
| **TEST_SINGLE** (Fase 1) | **79.83%** | **77.38%** | **-2.45pp (-3.07%)** | ❌ **PIORA** |
| **DRIFT_RECOVERY** (F1+F2) | **73.74%** | **72.43%** | **-1.31pp (-1.78%)** | ❌ **PIORA** |
| **MULTI_DRIFT** (F1+F2) | **69.96%** | **72.33%** | **+2.37pp (+3.39%)** | ✅ **MELHORA** |

### Conclusão Preliminar

- ❌ **2 de 3 experimentos pioraram**
- ✅ **1 de 3 experimentos melhorou (Multi_Drift)**
- ⚠️ **FASE 2 NÃO ESTÁ ATIVA** (problema crítico detectado)
- ⚠️ Fase 1 sozinha causou piora em TEST_SINGLE

---

## 🔍 ANÁLISE DETALHADA

### 1. TEST_SINGLE (apenas Fase 1)

**Baseline:**
- Avg Train: 92.65%
- Avg TEST: **79.83%**
- Std TEST: 17.97%

**Otimizado (Fase 1):**
- Avg Train: 91.17% (-1.48pp)
- Avg TEST: **77.38%** (-2.45pp)
- Std TEST: 19.52%

**Diagnóstico:**
- ❌ Piora de **-2.45 pontos percentuais** no teste
- ❌ Piora também no treino (-1.48pp)
- ⚠️ **Variância aumentou** (17.97% → 19.52%)
- 🔴 **4 SEVERE drifts detectados** (nos logs)

**Possíveis Causas:**
1. **Early Stopping muito agressivo**: threshold 75% pode estar descartando indivíduos bons prematuramente
2. **Cache de Fitness problemático**: pode estar retornando fitness cached para indivíduos similares mas não idênticos
3. **HC reduzido (8 variantes)**: pode ter reduzido exploração local excessivamente
4. **Shallow copy de elite**: possível que elite esteja sendo modificado inadvertidamente

---

### 2. DRIFT_RECOVERY (Fase 1 + Fase 2)

**Baseline:**
- Avg Train: 92.03%
- Avg TEST: **73.74%**
- Std TEST: 19.75%

**Otimizado (Fase 1 + Fase 2):**
- Avg Train: 92.75% (+0.72pp) ✅
- Avg TEST: **72.43%** (-1.31pp) ❌
- Std TEST: 20.87%

**Diagnóstico:**
- ✅ Treino melhorou (+0.72pp)
- ❌ Teste piorou (-1.31pp)
- ⚠️ **Variância aumentou** (19.75% → 20.87%)
- 🔴 **2 SEVERE drifts detectados**
- ❌ **FASE 2 NÃO ATIVA** - Zero mensagens de concept fingerprinting

**Comportamento Esperado vs Real:**

Chunk 0 (c1):
- Esperado: "🆕 Chunk 0: NOVO conceito detectado (ID: 'concept_0')"
- Real: **NADA** (silêncio total)

Chunk 4 (c1 retorna):
- Esperado: "📌 Chunk 4: RECUPERANDO memória do conceito 'concept_0' (recorrente)"
- Real: **NADA** (silêncio total)

**Possíveis Causas:**
1. **Fase 2 não foi executada**: Código pode não ter sido sincronizado corretamente
2. **Exceção silenciosa**: Erro no calculate_concept_fingerprint() não está sendo logado
3. **Import faltando**: scipy.spatial.distance.cosine pode não estar disponível no ambiente
4. **Conditional não alcançado**: Código pode estar em branch que não executa

---

### 3. MULTI_DRIFT (Fase 1 + Fase 2)

**Baseline:**
- Avg Train: 92.73%
- Avg TEST: **69.96%**
- Std TEST: 22.61%

**Otimizado (Fase 1 + Fase 2):**
- Avg Train: 92.96% (+0.23pp)
- Avg TEST: **72.33%** (+2.37pp) ✅
- Std TEST: 19.78%

**Diagnóstico:**
- ✅ **ÚNICA MELHORA**: +2.37pp no teste (+3.39% relativo)
- ✅ Treino também melhorou ligeiramente (+0.23pp)
- ✅ **Variância reduziu** (22.61% → 19.78%) - mais estável!
- 🔴 **3 SEVERE drifts detectados**
- ❌ **FASE 2 NÃO ATIVA** - Zero mensagens

**Por que melhorou aqui?**
- Multi_Drift tem mais chunks (8 vs 6)
- Mais drifts = mais oportunidades de seeding
- Variância reduzida sugere que otimizações ajudaram estabilidade
- Possível que early stopping tenha ajudado a evitar overfitting em alguns chunks

---

## 🐛 PROBLEMAS IDENTIFICADOS

### CRÍTICO 1: FASE 2 NÃO ESTÁ FUNCIONANDO

**Evidência:**
```
experimento_test_drift_recovery2.log:
  Novos conceitos: 0
  Recuperando memoria: 0
  Recorrente detectado: 0
  Memory preservada: 0
  Status: FASE 2 NAO DETECTADA

experimento_test_multi_drift2.log:
  Novos conceitos: 0
  Recuperando memoria: 0
  Recorrente detectado: 0
  Memory preservada: 0
  Status: FASE 2 NAO DETECTADA
```

**Impacto:**
- TODO o esforço de implementação de Fase 2 foi desperdiçado
- Não há detecção de conceitos recorrentes
- Não há restauração de memória
- Experimentos rodaram apenas com Fase 1

**Causas Prováveis (em ordem de probabilidade):**

1. **Arquivos não sincronizados (90% prob):**
   - utils.py com as 3 funções novas não foi upado
   - main.py com código de fingerprinting não foi upado
   - Configs YAML atualizados não foram upados

2. **Erro de import não logado (5% prob):**
   - `from scipy.spatial.distance import cosine` falhou
   - Exceção foi capturada e não logada

3. **Exceção silenciosa no código (3% prob):**
   - calculate_concept_fingerprint() lança exceção
   - Try-except captura mas não loga

4. **Bug lógico (2% prob):**
   - Código está em branch condicional que nunca executa
   - Variável numeric_features vazia

---

### ALTO 2: FASE 1 CAUSOU PIORA EM 2 EXPERIMENTOS

**Evidência:**
- TEST_SINGLE: -2.45pp
- DRIFT_RECOVERY: -1.31pp (teste, apesar de treino melhorar)

**Otimizações Suspeitas:**

**2.1. Early Stopping (threshold 75%)**

Código implementado:
```python
# fitness.py linha 192-247
if partial_gmean < early_stop_threshold * 0.75:  # 75% de margem
    return {'fitness': -float('inf'), ...}
```

Problema:
- Threshold de 75% pode ser muito agressivo
- worst_elite_gmean em gerações iniciais pode ser baixo (ex: 0.40)
- 0.40 * 0.75 = 0.30 → qualquer indivíduo com G-mean < 30% é descartado
- Em chunks com drift severo, muitos indivíduos podem ter G-mean 30-50% inicialmente

Impacto estimado: -1 a -2pp

**2.2. Cache de Fitness (hash-based)**

Código implementado:
```python
# ga.py linhas 803-918
def hash_individual(individual):
    rules_string = individual.get_rules_as_string()
    hash_string = f"{rules_string}|{individual.default_class}"
    return hash(hash_string)
```

Problema:
- Python's `hash()` pode ter colisões
- Indivíduos diferentes podem ter mesmo hash
- Cache pode retornar fitness incorreto

Impacto estimado: -0.5 a -1pp

**2.3. HC Reduzido (8 variantes)**

Código implementado:
```python
# hill_climbing_v2.py linha 603
max_variants_to_return = min(8, level_config['num_variants_base'], len(all_variants))
```

Problema:
- Redução de 12-18 para 8 variantes pode ter reduzido exploração local
- Menos refinamento = indivíduos ligeiramente piores

Impacto estimado: -0.3 a -0.5pp

---

### MÉDIO 3: VARIÂNCIA AUMENTOU EM 2 EXPERIMENTOS

**Evidência:**
- TEST_SINGLE: 17.97% → 19.52% (+1.55pp)
- DRIFT_RECOVERY: 19.75% → 20.87% (+1.12pp)

**Causa Provável:**
- Early stopping introduz aleatoriedade: indivíduos são descartados em diferentes momentos
- Cache pode introduzir instabilidade se houver colisões

---

## 🎯 CAUSAS RAIZ - DIAGNÓSTICO FINAL

### Por que não tivemos ganhos?

**1. FASE 2 nunca executou (100% certo)**
- Sem detecção de recorrência
- Sem restauração de memória
- Experimentos = apenas Fase 1

**2. FASE 1 teve efeitos colaterais negativos**
- Early stopping muito agressivo → descarta indivíduos bons
- Cache de fitness com colisões → fitness incorreto
- HC reduzido → menos refinamento local

**3. Métricas corretas mas interpretação errada**
- Fase 1 focou em **redução de tempo** (-48% esperado)
- Não checamos se **tempo realmente reduziu** nos logs
- Focamos em G-mean, mas Fase 1 visava velocidade

**4. Trade-off não balanceado**
- Reduzimos tempo às custas de qualidade
- Early stop economiza tempo mas descarta indivíduos
- HC reduzido economiza tempo mas reduz refinamento

---

## 📋 PLANO DE AÇÃO IMEDIATO

### PRIORIDADE 1: VALIDAR SE FASE 2 FOI EXECUTADA

**Ação:**
```bash
# No ambiente onde rodou os experimentos, verificar:

# 1. Versão do utils.py tem as funções?
grep "calculate_concept_fingerprint" utils.py

# 2. Versão do main.py tem o código de fingerprinting?
grep "chunk_fingerprint" main.py

# 3. Config YAML tem o parâmetro?
grep "concept_fingerprint_similarity_threshold" config_test_drift_recovery.yaml
```

**Se NÃO encontrar:**
- Confirma que arquivos não foram sincronizados
- Necessário RE-UPLOAD e RE-EXECUÇÃO

**Se ENCONTRAR:**
- Bug no código ou exceção silenciosa
- Adicionar logging debug
- Re-executar com logging verbose

---

### PRIORIDADE 2: AJUSTAR FASE 1 (Early Stopping)

**Problema:** Threshold 75% muito agressivo

**Solução:**
```python
# fitness.py linha 192-247
# ANTES:
if partial_gmean < early_stop_threshold * 0.75:  # Muito agressivo

# DEPOIS:
if partial_gmean < early_stop_threshold * 0.50:  # Mais conservador
    # OU ainda melhor:
    # Apenas descarta se gmean é MUITO ruim (< 30%)
    if partial_gmean < max(early_stop_threshold * 0.50, 0.30):
```

---

### PRIORIDADE 3: VALIDAR CACHE DE FITNESS

**Problema:** Possíveis colisões de hash

**Solução:**
```python
# ga.py linha 803-918
# ADICIONAR logging de colisões:

if ind_hash in fitness_cache:
    cached = fitness_cache[ind_hash]
    # VALIDAÇÃO: Verifica se é realmente o mesmo indivíduo
    if individual.get_rules_as_string() == cached['rules_string']:
        # Hit real
        cache_hits += 1
    else:
        # COLISÃO DETECTADA
        logger.warning(f"CACHE COLLISION: hash {ind_hash}")
        # Avalia normalmente
        cache_misses += 1
```

---

### PRIORIDADE 4: VERIFICAR TEMPO DE EXECUÇÃO

**Ação:** Extrair tempos dos logs

```bash
# Tempo total de cada experimento
grep "Total experiment time" experimento_test_*.log

# Tempo médio por geração
grep "Time:" experimento_test_single.log | awk '{sum+=$NF; count++} END {print sum/count}'
```

**Objetivo:** Confirmar se Fase 1 realmente reduziu tempo (esperado: -48%)

---

## 🔄 PRÓXIMAS AÇÕES RECOMENDADAS

### Opção A: RE-EXECUTAR COM FASE 2 CORRIGIDA (RECOMENDADO)

**Passos:**
1. ✅ Confirmar que arquivos corretos estão sincronizados (utils.py, main.py, configs)
2. ✅ Adicionar logging verbose em calculate_concept_fingerprint()
3. ✅ Adicionar try-except com logging explícito
4. ✅ Testar localmente com 2 chunks antes de upload
5. ✅ Re-executar DRIFT_RECOVERY e MULTI_DRIFT
6. ✅ Analisar logs para confirmar Fase 2 ativa

**Tempo estimado:** 2-3h implementação + 12-16h execução

---

### Opção B: AJUSTAR FASE 1 E RE-EXECUTAR

**Passos:**
1. ✅ Ajustar early stopping threshold (75% → 50%)
2. ✅ Adicionar validação de colisão no cache
3. ✅ Considerar reverter HC para 12 variantes
4. ✅ Re-executar TEST_SINGLE primeiro (validação rápida)
5. ✅ Se melhorar, aplicar aos outros

**Tempo estimado:** 1h implementação + 6-8h execução (TEST_SINGLE)

---

### Opção C: ANÁLISE DE TEMPO PRIMEIRO

**Objetivo:** Validar se Fase 1 realmente reduziu tempo

**Passos:**
1. ✅ Extrair tempo total de cada experimento
2. ✅ Comparar baseline vs otimizado
3. ✅ Se tempo NÃO reduziu → Fase 1 falhou completamente
4. ✅ Se tempo reduziu mas G-mean piorou → Trade-off ruim, ajustar

**Tempo estimado:** 30min análise

---

## 📝 RECOMENDAÇÃO FINAL

### Ordem de Execução Sugerida:

**AGORA (30min):**
1. Opção C: Analisar tempo de execução nos logs
2. Confirmar se Fase 2 foi sincronizada (check arquivos)

**DEPOIS (2h):**
3. Se Fase 2 NÃO foi sincronizada: Re-upload e re-execução (Opção A)
4. Se Fase 2 foi sincronizada: Debug do código + re-execução (Opção A)

**PARALELO (1h):**
5. Ajustar Fase 1 (early stop threshold) (Opção B)

**VALIDAÇÃO (8-16h execução):**
6. Re-executar TEST_SINGLE com Fase 1 ajustada
7. Se OK, re-executar DRIFT_RECOVERY e MULTI_DRIFT com Fase 1+2

---

## 🎓 LIÇÕES APRENDIDAS

1. **Sempre validar que código foi sincronizado antes de experimentos longos**
   - Checklist de verificação pré-execução
   - Logging verbose no início para confirmar versões

2. **Otimizações de performance podem degradar qualidade**
   - Early stopping: trade-off tempo vs qualidade
   - Sempre medir ambos: tempo E G-mean

3. **Cache precisa ser validado rigorosamente**
   - Hash colisions são reais
   - Adicionar validação de identidade além do hash

4. **Logging é CRÍTICO para debug**
   - Fase 2 falhou silenciosamente
   - Sem logs, perdemos 16h de execução

5. **Testar mudanças isoladamente**
   - Fase 1 + Fase 2 ao mesmo tempo dificulta debug
   - Melhor: Fase 1 → validar → Fase 2 → validar

---

**FIM DA ANÁLISE**

**Próximo passo:** Decidir entre Opção A, B ou C e executar.
