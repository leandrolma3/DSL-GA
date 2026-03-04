# 🧪 EXPERIMENTO: 8 Chunks - Validação Seeding 85%

**Data**: 2025-10-24
**Objetivo**: Validar correção de seeding 85% após drift SEVERE
**Configuração**: RBF_Abrupt_Severe com 8 chunks
**Tempo estimado**: 18-20 horas

---

## 🎯 OBJETIVO PRINCIPAL

Testar se **seeding intensivo 85%** melhora recovery após drift SEVERE, comparando:
- **Chunk 4→5**: Drift SEVERE detectado (esperado G-mean ~39-40%, como experimentos anteriores)
- **Chunks 5, 6, 7**: Usar seeding 85% para recovery após SEVERE
- **Meta**: G-mean chunks 5-7 ≥ 55-60% (melhor que 39% do chunk 4)

---

## 📊 CONFIGURAÇÃO DO EXPERIMENTO

### Mudanças Aplicadas

| Arquivo | Mudança | Linha |
|---------|---------|-------|
| **config.yaml** | `num_chunks: 6 → 8` | 36 |
| **config.yaml** | `max_instances: 42000 → 54000` | 38 |
| **main.py** | Heurística preditiva drift | 779-782 |
| **main.py** | Log drift passado para GA | 773-777 |
| **ga.py** | Sobrescrita seeding 85% em SEVERE | 526-530 |

### Estrutura do Dataset

```
RBF_Abrupt_Severe (10 chunks disponíveis):
┌─────────────────────────────────────────┐
│ Conceito c1         │ Conceito c2_severe│
│ Chunks 0-4 (5 ch.)  │ Chunks 5-9 (5 ch.)│
├─────────────────────┴─────────────────┬─┘
│ Experimento usa: 8 chunks (0-7)       │
│ Chunks 0-4: c1 (normal, ~85-90%)      │
│ Chunks 5-7: c2_severe (drift, ~40%?)  │
└───────────────────────────────────────┘
```

---

## 📋 PREVISÃO DE RESULTADOS POR CHUNK

### Chunk 0→1 (Baseline)

| Métrica | Experimento Anterior | Esperado |
|---------|---------------------|----------|
| **Train G-mean** | 91.49% | **90-92%** |
| **Test G-mean** | 89.86% | **88-90%** |
| **Drift** | N/A (primeiro chunk) | N/A |
| **Seeding** | 60% (72 ind.) | **60%** (72 ind.) |
| **Tempo** | 1h 57min | **~2h** |

---

### Chunks 1→2, 2→3, 3→4 (Estáveis)

| Chunk | Test G-mean Anterior | Esperado | Drift Esperado |
|-------|---------------------|----------|----------------|
| **1→2** | 89.03% | **88-90%** | STABLE ou IMPROVED |
| **2→3** | 90.66% | **88-91%** | STABLE ou IMPROVED |
| **3→4** | 87.40% | **87-90%** | STABLE |

**Observação**: Chunks em conceito c1 (estável), performance esperada ~88-90%.

---

### 🔴 Chunk 4→5 (DRIFT SEVERE) ← **PONTO CRÍTICO**

| Métrica | Exp. Anterior (5 ch.) | Esperado (8 ch.) | Observação |
|---------|----------------------|------------------|------------|
| **Train G-mean** | 88.68% | **87-90%** | Treina em c1 (ainda bom) |
| **Test G-mean** | **39.00%** | **38-42%** | ❌ Testa em c2_severe (COLAPSO) |
| **Drift Detectado** | SEVERE (-48.4%) | **SEVERE (-45-50%)** | ✅ Detectado corretamente |
| **Memory** | Top 1 (de 1) | **Top 10%** | ✅ Aplicado |
| **Herança** | 20% | **20%** | ✅ Aplicado |
| **Seeding** | ❌ 60% (bug timing) | ❌ **60%** (mesmo problema) | ⚠️ Chunk 4 ainda usa drift do chunk 3 |
| **drift_severity_to_pass** | 'STABLE' | **'STABLE'** | Chunk 3 foi estável |
| **Tempo** | 1h 54min | **~2h** |

**IMPORTANTE**: Chunk 4 **AINDA usará seeding 60%** porque:
- Drift SEVERE só é detectado **APÓS** chunk 4 executar
- Chunk 4 usa `drift_severity` do chunk 3 (STABLE)
- **Seeding 85% será usado nos chunks 5, 6, 7** (não no chunk 4)

---

### ⭐ Chunk 5→6 (RECOVERY com Seeding 85%) ← **FOCO PRINCIPAL**

| Métrica | Experimento Anterior | Esperado (8 ch.) | Observação |
|---------|---------------------|------------------|------------|
| **Train G-mean** | N/A (não existia) | **50-65%** | Treina em c2_severe (difícil) |
| **Test G-mean** | N/A | **🎯 55-65%** | ⭐ **META: Melhor que 39% do chunk 4** |
| **Drift Detectado** | N/A | **STABLE ou MILD** | Drift do chunk anterior (4→5 foi SEVERE) |
| **drift_severity_to_pass** | N/A | **'SEVERE'** ✅ | Usa drift do chunk 4 |
| **Heurística Preditiva** | N/A | ✅ **Ativa** | historical_gmean[-1] = 39% < 50% |
| **Seeding** | N/A | **✅ 85% (102 ind.)** | **CORREÇÃO APLICADA!** |
| **Tempo** | N/A | **~2h 30min** | Recovery mode (25 gens) |

**Mensagens Esperadas no Log**:
```
Chunk 5: Previous chunk had very low G-mean (0.390) - assuming SEVERE drift preventively
Chunk 5: Using drift_severity='SEVERE' from previous chunk for GA adaptation
  -> SEEDING ADAPTATIVO ATIVADO: Estimando complexidade do chunk...
  -> Complexidade estimada: HARD (DT probe acc: 0.XXX)
  -> SEVERE DRIFT DETECTED: Seeding INTENSIVO ativado (85% seeding, 90% injection)
     Parâmetros adaptativos: seeding_ratio=0.85, injection_ratio=0.90, depths=[5, 8, 10]
  -> Seeding Probabilístico ATIVADO: Injetando 90% das regras DT
População de reset criada: 120 indivíduos (102 semeados, 18 aleatórios).
```

---

### Chunks 6→7, 7→8 (Estabilização)

| Chunk | Test G-mean Esperado | Drift Esperado | Seeding | Observação |
|-------|---------------------|----------------|---------|------------|
| **6→7** | **60-70%** | STABLE ou IMPROVED | **85%** (se chunk 5 < 50%) ou **60%** | Continuando recovery |
| **7→8** | **65-75%** | STABLE ou IMPROVED | **60%** (esperado) | Estabilizando em c2_severe |

**Observação**: Performance deve **melhorar gradualmente** à medida que o sistema aprende c2_severe.

---

## 📊 MÉTRICAS DE SUCESSO

### Critério 1: Seeding 85% Foi Aplicado? ✅

**Validação no log do Chunk 5**:
- [ ] Mensagem: `"Chunk 5: Using drift_severity='SEVERE'"`
- [ ] Mensagem: `"SEVERE DRIFT DETECTED: Seeding INTENSIVO ativado"`
- [ ] Mensagem: `"seeding_ratio=0.85, injection_ratio=0.90"`
- [ ] Mensagem: `"População de reset criada: 120 indivíduos (102 semeados, 18 aleatórios)"`

**Se 102 semeados** → ✅ **Seeding 85% aplicado corretamente**
**Se 72 semeados** → ❌ **Bug ainda presente**

---

### Critério 2: Recovery Melhorou? 🎯

| Métrica | Meta | Como Medir |
|---------|------|------------|
| **Chunk 5 G-mean** | **≥ 55%** | Melhor que chunk 4 (39%) em **+16pp** |
| **Chunk 6 G-mean** | **≥ 60%** | Melhoria contínua (+5pp vs chunk 5) |
| **Chunk 7 G-mean** | **≥ 65%** | Aproximando da meta de 70% |
| **Média chunks 5-7** | **≥ 60%** | Média dos 3 chunks após drift |

**Comparação com experimentos anteriores**:
- **Baseline** (chunk 4→5, sem drift detection): 52.58%
- **P1+P2** (chunk 4→5, com drift detection, reset 100%): 39.02%
- **Fase 1-Novo** (chunk 4→5, memory 10%, herança 20%, sem seeding 85%): 39.00%
- **ESPERADO** (chunks 5-7, COM seeding 85%): **60%+ média** ✅

---

### Critério 3: HC Melhorou? ⚠️

**Ainda usando 13 variantes** (bug de sincronização não resolvido):
- Taxa de aprovação esperada: **17-20%**
- **Meta**: ≥ 25% (não atingível sem corrigir 18 variantes)

**Ação futura**: Após validar seeding 85%, sincronizar `hill_climbing_v2.py` corretamente.

---

## ⏱️ TEMPO ESTIMADO

### Por Chunk

| Chunk | Tipo | Generations | Tempo Estimado |
|-------|------|-------------|----------------|
| **0** | Recovery (c1) | 25 | **~2h** |
| **1** | Normal (c1) | ~60 (ES Layer 1) | **~3h 30min** |
| **2** | Normal (c1) | ~60 (ES Layer 1) | **~3h 30min** |
| **3** | Normal (c1) | ~60 (ES Layer 1) | **~2h 30min** |
| **4** | Normal (c1) | 25 (recovery) | **~2h** |
| **5** | Recovery (c2_severe) + Seeding 85% | 25 | **~2h 30min** ⭐ |
| **6** | Normal (c2_severe) | ~60 (ES Layer 1) | **~3h** |
| **7** | Normal (c2_severe) | ~60 (ES Layer 1) | **~3h** |

**Total estimado**: **22-24 horas**

**Fatores que podem aumentar tempo**:
- Chunks com performance ruim (c2_severe) podem demorar mais
- HC ativações aumentam tempo (~5-10min por ativação)

---

## 🧪 PROTOCOLO DE TESTE

### 1. Sincronizar Arquivos

```bash
# Arquivos modificados
scp config.yaml <ssh-host>:/root/DSL-AG-hybrid/
scp main.py <ssh-host>:/root/DSL-AG-hybrid/
scp ga.py <ssh-host>:/root/DSL-AG-hybrid/

# Opcional: Verificar se hill_climbing_v2.py tem 18 variantes
scp hill_climbing_v2.py <ssh-host>:/root/DSL-AG-hybrid/
```

---

### 2. Executar Experimento

```bash
ssh <ssh-host>
cd /root/DSL-AG-hybrid

# Backup do experimento anterior (opcional)
mv drift_test_6chunks_*.log arquivos_antigos/

# Executar novo experimento (8 chunks)
nohup python main.py > experimento_8chunks_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Pegar PID
echo $!
```

---

### 3. Monitorar Execução

```bash
# Progresso geral
tail -f experimento_8chunks_*.log | grep -E "Chunk.*Results|DRIFT"

# Validar seeding 85% no Chunk 5
grep -A5 "Chunk 5.*drift_severity" experimento_8chunks_*.log
grep "SEVERE DRIFT DETECTED: Seeding" experimento_8chunks_*.log
grep "População de reset criada" experimento_8chunks_*.log

# Ver G-means de todos os chunks
grep "Chunk.*Results:" experimento_8chunks_*.log
```

---

### 4. Comandos Úteis Durante Execução

```bash
# Ver último chunk processado
tail -20 experimento_8chunks_*.log | grep "Chunk.*Results"

# Ver tempo decorrido
head -1 experimento_8chunks_*.log  # Hora inicial
date  # Hora atual

# Estimar tempo restante (baseado em média 2h 45min/chunk)
# 8 chunks × 2.75h = 22h total
```

---

## 📋 CHECKLIST DE VALIDAÇÃO (Após Execução)

### Validação Técnica

- [ ] **Log completo salvo** (arquivo .log existe e não está vazio)
- [ ] **8 chunks executados** (chunks 0-7)
- [ ] **Chunk 4→5**: Drift SEVERE detectado
- [ ] **Chunk 5**: drift_severity='SEVERE' usado
- [ ] **Chunk 5**: Seeding 85% aplicado (102 semeados)
- [ ] **Chunks 5-7**: G-mean ≥ 55%, 60%, 65% respectivamente

---

### Validação de Performance

| Métrica | Meta | Resultado | Status |
|---------|------|-----------|--------|
| **Chunk 5 G-mean** | ≥ 55% | ___% | ☐ |
| **Chunk 6 G-mean** | ≥ 60% | ___% | ☐ |
| **Chunk 7 G-mean** | ≥ 65% | ___% | ☐ |
| **Média chunks 5-7** | ≥ 60% | ___% | ☐ |
| **Avg Test G-mean (8 chunks)** | ≥ 75% | ___% | ☐ |
| **HC Taxa** | ≥ 17% | ___% | ☐ |

---

### Decisão GO/NO-GO

**✅ SUCESSO** (GO para melhorias HC):
- Chunk 5-7 média ≥ 60%
- Seeding 85% aplicado corretamente
- Recovery visível (melhoria gradual)

**➡️ Próximos passos**:
1. Sincronizar `hill_climbing_v2.py` (18 variantes)
2. Aumentar tolerância HC para 1.5-2%
3. Aumentar variantes para 25
4. Executar experimento final

---

**⚠️ PARCIAL** (Investigar):
- Chunk 5-7 média 50-60%
- Seeding 85% aplicado, mas impacto menor que esperado

**➡️ Próximos passos**:
1. Analisar por que seeding 85% não foi suficiente
2. Considerar seeding 90-95%
3. Ou testar outras estratégias (detecção preditiva)

---

**❌ FALHA** (Revisar abordagem):
- Chunk 5-7 média < 50%
- Seeding 85% não aplicado (bug ainda presente)
- Sem melhoria vs experimentos anteriores

**➡️ Próximos passos**:
1. Se bug: Corrigir e re-testar
2. Se não houve bug: Aceitar limitação, focar em HC
3. Considerar abordagens alternativas (ensemble, transfer learning)

---

## 📊 ANÁLISE ESPERADA

Após 22-24h, executar análise comparativa:

```bash
# Script de análise (pseudo-código)
python analyze_experiment.py experimento_8chunks_*.log

# Métricas a extrair:
# - G-mean por chunk (0-7)
# - Drift detectado (tipo e severidade)
# - Seeding usado em cada chunk
# - HC taxa de aprovação
# - Tempo por chunk
# - Comparação com experimentos anteriores
```

---

**Criado por**: Claude Code
**Data**: 2025-10-24
**Status**: ⏳ **PRONTO PARA EXECUÇÃO**
**Tempo Estimado**: **22-24 horas**
**Próximo Passo**: **SINCRONIZAR ARQUIVOS E EXECUTAR EXPERIMENTO**
