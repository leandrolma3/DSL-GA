# ANALISE COMPLETA - EXPERIMENTO RUN4 (LAYER 1)

**Data:** 2025-11-03
**Experimento:** 5 chunks, config_test_single.yaml
**Objetivo:** Validar Layer 1 (Early Stop + Cache + HC Seletivo)

---

## 1. RESUMO EXECUTIVO

### Tempo Total
- **13.42h** (48313.6s = 805.2min)
- Média por chunk: **2.68h** (161min)

### G-mean Médio
- **Avg Test G-mean: 0.7928** (+/- 0.1771)
- **Avg Train G-mean: 0.9296**

### Comparação com Run3 (Baseline)
- **Run3**: 9.9h, G-mean=0.7763
- **Run4**: 13.42h, G-mean=0.7928
- **Delta tempo**: +35.6% (PIOR)
- **Delta G-mean**: +0.0165 (+2.1%)

### Status
- TEMPO: PIOR QUE BASELINE (esperava-se -40-55%, obteve-se +35.6%)
- G-MEAN: LEVEMENTE MELHOR (+2.1%)
- LAYER 1: LOGS VISIVEIS (HC funcionando, Early Stop/Cache NÃO VISÍVEIS)

---

## 2. DESEMPENHO POR CHUNK

| Chunk | Tempo (min) | Train G-mean | Test G-mean | Test F1 | Delta | Status |
|-------|-------------|--------------|-------------|---------|-------|--------|
| 0 | 93.0 | 0.9127 | 0.8830 | 0.8845 | -0.0297 | OK |
| 1 | 143.6 | 0.9373 | 0.8937 | 0.8937 | -0.0436 | OK |
| 2 | 181.8 | 0.9405 | 0.4389 | 0.4557 | -0.5015 | SEVERE DRIFT |
| 3 | 178.3 | 0.9255 | 0.8725 | 0.8759 | -0.0531 | Recuperou |
| 4 | 208.5 | 0.9318 | 0.8760 | 0.8791 | -0.0558 | OK |

### Observações

**Chunk 0 (Primeiro):**
- 25 gerações (config mais curto)
- Train: 0.913, Test: 0.883
- Tempo: 93min (1.55h) - RÁPIDO
- Overfitting moderado (delta=-0.03)

**Chunk 1:**
- 40 gerações completadas (de 200 max)
- Train: 0.937, Test: 0.894
- Tempo: 143.6min (2.4h)
- Conceito recorrente (sim=0.9949)
- 4 aplicações HC, 2 puladas (33% economia)

**Chunk 2 (PROBLEMÁTICO):**
- 57 gerações completadas (de 200 max)
- Train: 0.941, Test: 0.439 (MUITO BAIXO)
- Tempo: 181.8min (3.0h) - MAIS LONGO
- Overfitting SEVERO (delta=-0.50)
- Conceito recorrente (sim=0.9953)
- 4 aplicações HC, 0 puladas
- **SEVERE DRIFT** detectado ao final

**Chunk 3 (Recuperação):**
- 29 gerações completadas (de 200 max)
- Train: 0.926, Test: 0.873
- Tempo: 178.3min (3.0h)
- Contramedidas de SEVERE DRIFT ativadas
- Conceito recorrente (sim=0.9814)
- 4 aplicações HC, 5 puladas (56% economia) - MELHOR

**Chunk 4:**
- 54 gerações completadas (de 200 max)
- Train: 0.932, Test: 0.876
- Tempo: 208.5min (3.5h) - MAIS LONGO
- Conceito recorrente (sim=0.9834)
- 5 aplicações HC, 2 puladas (29% economia)

---

## 3. LAYER 1 - OTIMIZACOES

### Fix 1: Early Stopping (PROBLEMA)

**Status:** LOGS INVISÍVEIS

**Evidências:**
- Nenhum log "[EARLY STOP] ... Descartados=X/Y" encontrado no experimento
- Apenas logs de threshold apareceram:
  - Chunk 0, Gen 2: threshold=0.763
  - Chunk 0, Gen 3: threshold=0.864
  - Chunk 1, Gen 2: threshold=0.893
  - etc.

**Diagnóstico:**
- Early stop threshold ESTÁ sendo calculado (logs aparecem)
- MAS early stop count NÃO ESTÁ aparecendo
- Possível causa: Nenhum indivíduo foi descartado (threshold muito alto ou early stop não ativado)

**Verificação no código:**
```python
# fitness.py linha 958-960
if early_stopped_count > 0:
    logging.warning(f"[EARLY STOP] Gen {generation+1}: Descartados=...")
```

**Hipótese:**
- `early_stopped_count` sempre 0 (nenhum descarte)
- Early stop não está funcionando como esperado

---

### Fix 2: Cache SHA256 (PROBLEMA CRÍTICO)

**Status:** LOGS COMPLETAMENTE INVISÍVEIS

**Evidências:**
- NENHUM log "[CACHE]" encontrado no experimento inteiro
- NENHUM log "[CACHE FINAL]" ao final dos chunks

**Diagnóstico:**
- Código de cache não está sendo executado OU
- Logs estão sendo suprimidos

**Verificação esperada:**
```
[CACHE] Gen X: Hits=Y/Z (W%)
[CACHE FINAL] Hits=..., Misses=..., Hit Rate=...
```

**Hipótese:**
- `cache_hits` e `cache_misses` sempre 0 OU
- Código de logging foi comentado/removido OU
- Erro de indentação (logging fora do bloco correto)

---

### Fix 3: Hill Climbing Seletivo (FUNCIONANDO)

**Status:** FUNCIONANDO PARCIALMENTE

**Evidências por chunk:**

**Chunk 0:**
- HC aplicado: 2x (Gen 12, Gen 24)
- HC pulado: 0x
- Economia: 0% (não houve estagnação suficiente para pular)
- Variantes: 16 geradas, 6 aprovadas (37.5%)

**Chunk 1:**
- HC aplicado: 4x
- HC pulado: 2x (Gen 34, Gen 35)
- Economia: 33%
- Variantes: 32 geradas, 12 aprovadas (37.5%)

**Chunk 2:**
- HC aplicado: 4x
- HC pulado: 0x
- Economia: 0%
- Variantes: 32 geradas, 12 aprovadas (37.5%)

**Chunk 3 (MELHOR):**
- HC aplicado: 4x
- HC pulado: 5x (Gen 12, Gen 13, Gen 26, Gen 27, Gen 29)
- Economia: 56% (EXCELENTE)
- Variantes: 32 geradas, 12 aprovadas (37.5%)

**Chunk 4:**
- HC aplicado: 5x
- HC pulado: 2x (Gen 23, Gen 24)
- Economia: 29%
- Variantes: 40 geradas, 13 aprovadas (32.5%)

**Observações:**
- HC Seletivo ESTÁ funcionando (logs visíveis)
- Economia varia: 0-56% (média ~24%)
- Taxa de aprovação consistente: 32-37.5%
- Chunk 3 teve melhor economia (56%) devido a estagnação prolongada

---

## 4. FASE 2 - CONCEPT FINGERPRINTING

### Detecção de Conceitos

**Chunk 0:**
- NOVO CONCEITO: 'concept_0'
- Fingerprint calculada: 6000 instâncias

**Chunk 1:**
- CONCEITO RECORRENTE: 'concept_0'
- Similaridade: 0.9949 (MUITO ALTA)
- Indivíduos restaurados: 1

**Chunk 2:**
- CONCEITO RECORRENTE: 'concept_0'
- Similaridade: 0.9953 (MUITO ALTA)
- Indivíduos restaurados: 2

**Chunk 3:**
- CONCEITO RECORRENTE: 'concept_0'
- Similaridade: 0.9814 (ALTA)
- Indivíduos restaurados: 3
- **SEVERE DRIFT detectado** (do chunk 2)
- Contramedidas ativadas: inheritance reduzido para 20%

**Chunk 4:**
- CONCEITO RECORRENTE: 'concept_0'
- Similaridade: 0.9834 (ALTA)
- Indivíduos restaurados: 1

### Observação
- Dataset RBF_Abrupt_Severe tem conceito ÚNICO (concept_0)
- Similaridade sempre > 0.98 (excelente detecção)
- Severe drift detectado apenas após chunk 2 (test G-mean=0.439)

---

## 5. ANALISE DO CHUNK PROBLEMÁTICO (CHUNK 2)

### Métricas
- Train G-mean: 0.9405 (EXCELENTE)
- Test G-mean: 0.4389 (PÉSSIMO)
- Delta: -0.5015 (OVERFITTING EXTREMO)
- Tempo: 181.8min (3.0h) - 30% MAIS LENTO que chunk 1

### Evolução das Gerações
- Gen 1: Best Gmean = 0.863
- Gen 21: Best Gmean = 0.923
- Gen 31: Best Gmean = 0.937
- Gen 51: Best Gmean = 0.940 (convergiu)

Melhoria total: +0.077 (de 0.863 → 0.940)

### Diagnóstico
1. **Modelo convergiu no treino** (Train=0.941) mas **falhou no teste** (Test=0.439)
2. **Não foi drift** - similaridade conceitual alta (0.9953)
3. **Overfitting clássico** - modelo especializou demais no chunk 2
4. **Consequência**: Chunk 3 detectou SEVERE DRIFT e ativou contramedidas

### Hipóteses
1. Chunk 2 tinha distribuição anômala (não representativa)
2. Early stopping não descartou indivíduos overfitados
3. Falta de penalização de overfitting durante evolução
4. HC gerou variantes que melhoraram treino mas pioraram teste

---

## 6. EVOLUÇÃO DO G-MEAN POR CHUNK

| Chunk | Gen 1 | Gen Final | Melhoria | Gerações |
|-------|-------|-----------|----------|----------|
| 0 | 0.763 | 0.913 | +0.150 | 25 |
| 1 | 0.893 | 0.937 | +0.044 | 40 |
| 2 | 0.863 | 0.940 | +0.077 | 57 |
| 3 | 0.924 | 0.926 | +0.002 | 29 |
| 4 | 0.859 | 0.932 | +0.073 | 54 |

### Observações
- Chunk 0: Maior melhoria (+15%) - iniciando do zero
- Chunk 1-2-4: Melhoria moderada (4-8%) - já tem seeding
- Chunk 3: Quase sem melhoria (+0.2%) - contramedidas ativas, GA teve dificuldade

---

## 7. COMPARAÇÃO RUN3 vs RUN4

| Métrica | Run3 (Baseline) | Run4 (Layer 1) | Delta | Status |
|---------|-----------------|----------------|-------|--------|
| Tempo total | 9.9h | 13.42h | +35.6% | PIOR |
| Avg Test G-mean | 0.7763 | 0.7928 | +2.1% | MELHOR |
| Tempo/chunk | ~2.0h | ~2.7h | +35% | PIOR |

### Meta Layer 1
- Redução tempo: -40-55% (FALHOU - obteve +35.6%)
- G-mean: neutro ou leve melhora (OK - obteve +2.1%)

---

## 8. DIAGNOSTICO: POR QUE TEMPO PIOROU?

### Hipóteses

**1. Early Stopping NÃO está funcionando**
- Logs de descarte não aparecem
- Se early stop não descarta ninguém, não há economia
- Esperado: -20-30% tempo
- Real: 0% economia (nenhum descarte)

**2. Cache NÃO está funcionando**
- NENHUM log de cache
- Se cache não funciona, avalia todos indivíduos sempre
- Esperado: -10-20% tempo (com hit rate > 30%)
- Real: 0% economia (sem cache)

**3. HC Seletivo funcionando, mas economia baixa**
- Economia média: ~24% (não 66%)
- Chunks 0 e 2 tiveram 0% economia
- Esperado: -15-25% tempo
- Real: ~5-10% economia estimada

**4. Gerações completadas AUMENTARAM**
- Run3: Provavelmente early stopping layer 1-2-3 ativo
- Run4: Chunk 2 rodou 57 gerações (muito)
- Chunk 4 rodou 54 gerações (muito)

**5. Tempo por geração AUMENTOU**
- Possível causa: Cache não funciona → avalia tudo sempre
- Possível causa: Early stop não funciona → avalia 100% dos dados sempre

---

## 9. PROBLEMAS IDENTIFICADOS

### CRÍTICO
1. **Cache completamente invisível** - ZERO logs
2. **Early stop descarte invisível** - ZERO logs de descarte
3. **Tempo 35% PIOR** que baseline (esperava-se -40% melhor)

### ALTO
4. **Chunk 2: Overfitting extremo** (delta=-0.50)
5. **Chunk 2: Test G-mean=0.439** (muito baixo)

### MÉDIO
6. HC economia variável (0-56%, média 24%)
7. Gerações completadas muito altas (40-57 em alguns chunks)

---

## 10. INVESTIGAÇÃO NECESSÁRIA

### 1. Verificar código de Cache
- [ ] ga.py linha 955-956: Por que não logou?
- [ ] ga.py linha 1372-1381: Por que CACHE FINAL não apareceu?
- [ ] Verificar se `cache_hits` e `cache_misses` estão sendo incrementados
- [ ] Verificar se código está dentro do bloco correto

### 2. Verificar código de Early Stop
- [ ] ga.py linha 958-960: Por que não logou descarte?
- [ ] fitness.py linha 204-242: Early stop está retornando `early_stopped=True`?
- [ ] ga.py linha 914-916: `early_stopped_count` está sendo incrementado?
- [ ] Verificar se threshold está muito alto (nunca descarta)

### 3. Análise de Chunk 2
- [ ] Verificar distribuição de classes no chunk 2
- [ ] Comparar features do chunk 2 vs outros chunks
- [ ] Verificar se houve drift real ou apenas distribuição anômala

---

## 11. RECOMENDAÇÕES IMEDIATAS

### URGENTE
1. **Adicionar debug logging** para cache e early stop
2. **Verificar contadores** de cache_hits, cache_misses, early_stopped_count
3. **Testar com smoke test** (2 chunks) para validar logs

### CURTO PRAZO
4. **Implementar Layer 2** - Penalizar overfitting (delta > 0.10)
5. **Ajustar threshold** de early stop (50% → 60% mediana?)
6. **Revisar early stopping layers** 1-2-3 (podem estar desativados)

### MÉDIO PRAZO
7. **Profiling** - Identificar onde está o gargalo de tempo
8. **Análise de chunk 2** - Por que overfittou tanto?

---

## 12. CONCLUSÃO

### O que funcionou
- HC Seletivo (logs visíveis, economia 0-56%)
- Fase 2 Concept Fingerprinting (detecção perfeita)
- G-mean levemente melhor (+2.1%)

### O que NÃO funcionou
- Cache SHA256 (ZERO logs, provavelmente não funcionando)
- Early Stopping descarte (ZERO logs, provavelmente não funcionando)
- Tempo total (35.6% PIOR em vez de -40-55% melhor)

### Próximos passos
1. DEBUG imediato: Adicionar logging de contadores
2. SMOKE TEST: Validar cache e early stop com 2 chunks
3. FIX: Corrigir cache e early stop se não funcionando
4. RE-RUN: Executar Run5 com correções

---

**Status Final:** LAYER 1 PARCIALMENTE FUNCIONAL (apenas HC Seletivo validado)

**Prioridade:** DEBUG URGENTE de Cache e Early Stop
