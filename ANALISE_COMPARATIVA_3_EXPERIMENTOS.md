# 📊 ANÁLISE COMPARATIVA: 3 EXPERIMENTOS

**Data**: 2025-10-24
**Experimentos comparados**: Baseline, P1+P2, Fase 1-Novo
**Status**: ⚠️ **FASE 1-NOVO PIOROU AINDA MAIS**
**Conclusão**: 🔴 **HIPÓTESE REFUTADA - Abordagem atual não funciona**

---

## 🎯 RESUMO EXECUTIVO

### Performance Global

| Experimento | Avg Test G-mean | Chunk 4→5 | Δ vs Baseline | Veredicto |
|-------------|-----------------|-----------|---------------|-----------|
| **Baseline** (bug drift) | **81.63%** | **52.58%** | - | Referência |
| **P1+P2** (drift corrigido) | **78.07%** | **39.02%** | **-3.56pp / -13.56pp** | ❌ **PIOROU** |
| **Fase 1-Novo** (correções SEVERE) | **79.19%** | **39.00%** | **-2.44pp / -13.58pp** | ⚠️ **PIOROU MENOS, MAS AINDA RUIM** |

### ⚠️ **RESULTADO CRÍTICO**

**Fase 1-Novo teve uma melhoria MARGINAL sobre P1+P2 (+1.12pp), mas AINDA ESTÁ MUITO ABAIXO do baseline.**

**A hipótese de que "memory parcial + herança 20% + seeding 85%" resolveria o problema foi REFUTADA.**

---

## 📈 COMPARAÇÃO DETALHADA POR CHUNK

### Chunk 0→1

| Experimento | Train G-mean | Test G-mean | Generations | ES Layer | Tempo |
|-------------|--------------|-------------|-------------|----------|-------|
| **Baseline** | 88.98% | **87.62%** | 25 (recovery) | Layer 1 (Gen 63) | 1h 17min |
| **P1+P2** | 90.28% | **87.62%** | 25 (recovery) | Layer 1 (Gen 16) | 1h 18min |
| **Fase 1-Novo** | **91.49%** | **89.86%** | 25 (recovery) | (não registrado) | 1h 57min |

**Observação**: Fase 1-Novo teve **melhor performance** (+2.24pp vs baseline).

---

### Chunk 1→2

| Experimento | Train G-mean | Test G-mean | Drift Status | ES Layer | Tempo |
|-------------|--------------|-------------|--------------|----------|-------|
| **Baseline** | 89.02% | **89.06%** | ❌ Não detectado (bug) | Layer 1 (Gen 62) | 3h 04min |
| **P1+P2** | 91.45% | **89.54%** | ✅ IMPROVED: 0.876→0.895 (+1.9%) | Layer 1 (Gen 63) | 3h 04min |
| **Fase 1-Novo** | **90.83%** | **89.03%** | ✅ (não registrado claramente) | (não registrado) | 3h 41min |

**Observação**: Performances similares (~89%), Fase 1-Novo ligeiramente mais lenta.

---

### Chunk 2→3

| Experimento | Train G-mean | Test G-mean | Drift Status | ES Layer | Tempo |
|-------------|--------------|-------------|--------------|----------|-------|
| **Baseline** | 89.69% | **87.82%** | ❌ Não detectado (bug) | Layer 1 | 2h 01min |
| **P1+P2** | 86.96% | **84.48%** | ✅ MILD: 0.895→0.845 (-5.1%) | Layer 3 (Gen 168) | 2h 01min |
| **Fase 1-Novo** | **93.55%** | **90.66%** | ✅ IMPROVED: 0.890→0.907 (+1.6%) | (não registrado) | 3h 43min |

**Observação**: Fase 1-Novo teve **MELHOR performance** (+2.84pp vs baseline). **MAS** consumiu **85% mais tempo** (3h43 vs 2h01).

---

### Chunk 3→4

| Experimento | Train G-mean | Test G-mean | Drift Status | ES Layer | Tempo |
|-------------|--------------|-------------|--------------|----------|-------|
| **Baseline** | 88.02% | **89.69%** | ❌ Não detectado (bug) | Layer 1 | 1h 49min |
| **P1+P2** | 91.43% | **89.68%** | ✅ IMPROVED: 0.845→0.897 (+5.2%) | Layer 1 (Gen 62) | 1h 49min |
| **Fase 1-Novo** | **89.24%** | **87.40%** | ✅ (não registrado claramente) | (não registrado) | 2h 27min |

**Observação**: Fase 1-Novo teve **PIOR performance** (-2.29pp vs baseline).

---

### 🔴 Chunk 4→5 (DRIFT SEVERE) ← **FOCO PRINCIPAL**

| Experimento | Train G-mean | Test G-mean | Drift Status | Memory | Herança | Seeding | ES Layer | Tempo |
|-------------|--------------|-------------|--------------|--------|---------|---------|----------|-------|
| **Baseline** | 85.19% | **52.58%** | ❌ **NÃO DETECTADO** (bug) | Mantida | 50% | N/A | Layer 1 | 2h 56min |
| **P1+P2** | **90.91%** | **39.02%** | ✅ SEVERE: 0.897→0.390 (-50.7%) | ❌ Limpa 100% | ❌ 0% | 60% (72 ind.) | Layer 1 (Gen 62) | 2h 56min |
| **Fase 1-Novo** | **88.68%** | **39.00%** | ✅ SEVERE: 0.874→0.390 (-48.4%) | ✅ Top 10% (1 ind.) | ✅ 20% | ❌ 60% (72 ind.) | (não registrado) | 1h 54min |

### 🔍 ANÁLISE CRÍTICA DO CHUNK 4→5

#### ✅ Correções Aplicadas (Fase 1-Novo):
1. **Memory parcial**: ✅ "Memory REDUCED to top 1 individuals (was 1) - kept top 10%"
   - **MAS** memory só tinha 1 indivíduo! Mantém 10% de 1 = 1 (sem efeito)
2. **Herança 20%**: ✅ "Inheritance REDUCED to 20% due to SEVERE drift (was 50%)"
   - Funcionou corretamente
3. **Seeding 85%**: ❌ **NÃO APLICADO!**
   - Log mostra: "seeding_ratio=0.6, injection_ratio=0.6" (60%, não 85%)
   - Log mostra: "População de reset criada: 120 indivíduos (72 semeados, 48 aleatórios)."
   - **ESPERADO**: 102 semeados (85%), não 72 (60%)

#### ❌ **Por que seeding 85% NÃO foi aplicado?**

**Causa raiz**: O código de seeding 85% está no `ga.py` dentro do bloco `if enable_adaptive_seeding_config`, mas **`drift_severity` é passado para `initialize_population`, NÃO para `estimate_chunk_complexity`**.

O seeding adaptativo **estima complexidade ANTES de receber drift_severity**, então:
1. Complexidade estimada: MEDIUM (DT acc 79.0%)
2. Seeding ratio definido: 60%
3. **DEPOIS** (linha 526-529): `if drift_severity == 'SEVERE':` tenta sobrescrever
4. **MAS** o seeding já foi aplicado antes!

**Ordem incorreta**:
```python
# Linha 518: Estima complexidade (define seeding_ratio = 0.6)
complexity_level, probe_score, adaptive_profile = estimate_chunk_complexity(...)
dt_seeding_ratio_on_init_config = adaptive_profile['dt_seeding_ratio']  # 0.6

# Linha 526-529: Tenta sobrescrever (MAS população já foi criada!)
if drift_severity == 'SEVERE':
    dt_seeding_ratio_on_init_config = 0.85  # ← Sobrescreve variável
    # MAS a população já foi gerada com 0.6!
```

**Conclusão**: O código da correção foi adicionado no lugar errado. A população é criada DEPOIS da linha 545, mas o seeding ratio já foi usado internamente.

---

## 🧮 COMPARAÇÃO HC

### Taxa de Aprovação

| Experimento | Ativações | Variantes | Aprovações | MELHOR | TOLERÂNCIA | Taxa | Média/Ativação |
|-------------|-----------|-----------|------------|--------|------------|------|----------------|
| **Baseline** | 36 | 540 (15×36) | 27 | 13 (48%) | 14 (52%) | **5.8%** | 0.75 |
| **P1+P2** | 36 | 468 (13×36) | 51 | 14 (27%) | 37 (73%) | **10.90%** | 1.42 |
| **Fase 1-Novo** | 33 | 429 (13×33) | 75 | 21 (28%) | 54 (72%) | **17.48%** | 2.27 |

### 📊 Análise HC

**✅ SUCESSO**: HC melhorou significativamente!
- Taxa: 5.8% → 10.90% → **17.48%** (+201% vs baseline, +60% vs P1+P2)
- Média/ativação: 0.75 → 1.42 → **2.27** (+203% vs baseline, +60% vs P1+P2)

**⚠️ MAS** ainda muito abaixo da meta de 25-35%:
- Déficit: 17.48% vs 25% = **-7.52pp** (30% abaixo da meta)

**🐛 PROBLEMA**: Ainda usando **13 variantes ao invés de 18** (bug de sincronização não resolvido)
- Se tivesse 18 variantes: taxa estimada = ~19-21% (ainda abaixo de 25%)

---

## ⏱️ TEMPO DE EXECUÇÃO

| Experimento | Tempo Total | Δ vs Baseline | Tempo/Chunk |
|-------------|-------------|---------------|-------------|
| **Baseline** | **13h 49min** | - | **2h 46min** |
| **P1+P2** | **11h 08min** | **-2h 41min** (-19.3%) | **2h 13min** |
| **Fase 1-Novo** | **13h 43min** | **-6min** (-0.7%) | **2h 44min** |

**Observação**: Fase 1-Novo voltou a ser **mais lenta**, próxima do baseline. A economia de P1+P2 foi perdida.

**Possível causa**: Chunks 2 e 3 consumiram muito mais tempo (3h 43min cada vs 2h 01min).

---

## 🔍 ROOT CAUSE ANALYSIS: POR QUE PIOROU AINDA MAIS?

### Problema 1: Seeding 85% NÃO FOI APLICADO ⭐ **CRÍTICO**

**Evidência no log**:
```
Linha 1945: Parâmetros adaptativos: seeding_ratio=0.6, injection_ratio=0.6, depths=[5, 8, 10]
Linha 1946: -> Seeding Probabilístico ATIVADO: Injetando 60% das regras DT
Linha 1947: População de reset criada: 120 indivíduos (72 semeados, 48 aleatórios).
```

**Esperado**:
```
-> SEVERE DRIFT DETECTED: Seeding INTENSIVO ativado (85% seeding, 90% injection)
Parâmetros adaptativos: seeding_ratio=0.85, injection_ratio=0.90, depths=[5, 8, 10]
População de reset criada: 120 indivíduos (102 semeados, 18 aleatórios).
```

**Impacto**: A correção **mais importante** (seeding 85%) **NÃO foi aplicada**. Resultado no chunk 4→5 é esperado ser igual a P1+P2 (39.02% vs 39.00% = praticamente idêntico).

---

### Problema 2: Memory Parcial Teve Efeito Mínimo

**Evidência no log**:
```
Linha 1983: Memory REDUCED to top 1 individuals (was 1) - kept top 10%
```

**Análise**:
- Memory tinha apenas **1 indivíduo**
- Manter top 10% de 1 = **1 indivíduo** (sem mudança)
- **Correção teve efeito NULO**

**Por quê memory só tinha 1 indivíduo?**
- Memory em chunk N depende de chunks anteriores
- Se chunks anteriores foram ruins ou memory foi limpa por drift MODERATE, sobra pouco
- **Hipótese**: Drift MODERATE no chunk 2 (não detectado claramente) pode ter reduzido memory

---

### Problema 3: Herança 20% Insuficiente

**Evidência**:
- Herança 20% aplicada corretamente
- Mas 24 indivíduos herdados (20% de 120) de um chunk **ruim** (chunk 3→4: 87.40%)
- Indivíduos herdados eram adaptados ao conceito **c1**, não **c2_severe**

**Análise**:
- Herança 20% de indivíduos mal-adaptados = **poluição da população**
- **Melhor ter sido 10% ou 15%** (menos poluição)
- Ou **0% + seeding 95%** (aposta total em DT)

---

### Problema 4: Chunks 2 e 3 Consumiram Muito Tempo

**Evidência**:
- Chunk 2: 3h 43min (vs 2h 01min baseline) = **+85%**
- Chunk 3: 2h 27min (vs 1h 49min baseline) = **+35%**

**Possível causa**:
- Chunks com performance melhor (90.66%, 87.40%) → GA demorou mais para convergir
- Ou Early Stopping não ativou adequadamente (falta registro no log)

**Impacto**: Experimento total mais lento, mas performance não melhorou proporcionalmente.

---

## 📊 TABELA COMPARATIVA FINAL

| Métrica | Baseline | P1+P2 | Fase 1-Novo | Meta | Baseline→F1N |
|---------|----------|-------|-------------|------|--------------|
| **Avg Test G-mean** | **81.63%** | 78.07% | **79.19%** | ≥85% | **-2.44pp** ❌ |
| **Chunk 0** | 87.62% | 87.62% | **89.86%** | N/A | **+2.24pp** ✅ |
| **Chunk 1** | 89.06% | 89.54% | **89.03%** | N/A | **-0.03pp** ≈ |
| **Chunk 2** | 87.82% | 84.48% | **90.66%** | N/A | **+2.84pp** ✅ |
| **Chunk 3** | **89.69%** | 89.68% | 87.40% | N/A | **-2.29pp** ❌ |
| **Chunk 4→5** | **52.58%** | 39.02% | **39.00%** | ≥70% | **-13.58pp** ❌❌ |
| **HC Taxa** | 5.8% | 10.90% | **17.48%** | ≥25% | **+11.68pp** ✅ |
| **Drift Detection** | ❌ 0% | ✅ 100% | ✅ **100%** | 100% | **+100%** ✅ |
| **Tempo Total** | 13h 49min | 11h 08min | **13h 43min** | N/A | **-6min** ≈ |

### Síntese

| Aspecto | Resultado |
|---------|-----------|
| **Chunk 4→5 melhorou?** | ❌ **NÃO** (39.00% vs 39.02% = idêntico) |
| **Avg G-mean melhorou vs P1+P2?** | ✅ **Sim, levemente** (+1.12pp) |
| **Avg G-mean melhorou vs Baseline?** | ❌ **NÃO** (-2.44pp) |
| **Atingiu meta 85%?** | ❌ **NÃO** (faltam 5.81pp) |
| **Atingiu meta Chunk 4→5 ≥70%?** | ❌ **NÃO** (faltam 31pp!) |
| **Correções funcionaram?** | ⚠️ **Parcialmente** (herança 20%: ✅, memory: ~, seeding 85%: ❌) |

---

## 🚦 CONCLUSÃO: HIPÓTESE REFUTADA

### Hipótese Original (Fase 1-Novo)

> "Memory parcial (top 10%) + Herança 20% + Seeding 85% resolverão o problema de recovery em drift SEVERE."

### Resultado

⚠️ **HIPÓTESE PARCIALMENTE TESTADA**:
- **Memory parcial**: Aplicada, mas efeito mínimo (só tinha 1 indivíduo)
- **Herança 20%**: Aplicada corretamente
- **Seeding 85%**: ❌ **NÃO APLICADA** (bug no código)

🔴 **RESULTADO FINAL**: Chunk 4→5 **IDÊNTICO** a P1+P2 (39.00% vs 39.02%)

### Veredicto

❌ **HIPÓTESE REFUTADA (por enquanto)** porque:
1. Correção principal (seeding 85%) **não foi aplicada** (bug de implementação)
2. Memory parcial teve **efeito nulo** (só 1 indivíduo)
3. Herança 20% sozinha **não foi suficiente**

**MAS**: Não podemos concluir definitivamente porque **seeding 85% não foi testado corretamente**.

---

## 🔧 PRÓXIMOS PASSOS

### **OPÇÃO A: Corrigir Bug Seeding 85% e Re-testar** ⭐ **RECOMENDADO**

**Problema**: Seeding 85% não está sendo aplicado porque o código sobrescreve `dt_seeding_ratio_on_init_config` DEPOIS da população ser criada.

**Solução**: Mover a sobrescrita para ANTES da criação da população.

**Código em ga.py**:
```python
# ATUAL (BUGGY - linha 526-529):
if drift_severity == 'SEVERE':
    dt_seeding_ratio_on_init_config = 0.85  # ← Tarde demais!

# CORREÇÃO: Mover para ANTES de estimate_chunk_complexity (linha ~515)
# Ou passar drift_severity para estimate_chunk_complexity e ajustar lá
```

**Ação**: Corrigir ga.py e re-executar experimento.

**Impacto esperado**: Chunk 4→5: 39.00% → 48-55%

---

### **OPÇÃO B: Testar Abordagem Alternativa** (Se Opção A falhar)

Considerar mudanças mais radicais:

1. **Drift SEVERE: 0% herança + 95% seeding**
   - Apostar tudo em DT, sem poluição de conceito antigo
   - Herança 20% pode estar **piorando** ao invés de ajudar

2. **Usar ensemble de modelos** ao invés de reset
   - Manter modelo antigo + treinar modelo novo
   - Combinar predições com pesos adaptativos

3. **Transfer learning** de chunk antigo
   - Usar DT do chunk antigo como "prior"
   - Ajustar regras ao invés de recriar

---

### **OPÇÃO C: Aceitar Limitação e Focar em Outras Melhorias**

Se drift SEVERE for **inerentemente difícil** para o método atual:

1. Focar em melhorar **chunks não-drift** (0, 1, 2, 3)
2. Implementar **Prioridade 3** (Crossover Adaptativo)
3. Melhorar **HC** (tolerância 2%, 25 variantes)
4. Aceitar que chunk 4→5 terá performance ruim (~40-50%)

**Meta revista**: Avg 83-84% (ao invés de 85%) aceitando chunk 4→5 baixo.

---

## ✅ CHECKLIST DE AÇÃO IMEDIATA

- [ ] **Corrigir bug seeding 85%** em ga.py (mover sobrescrita para lugar correto)
- [ ] **Verificar sincronização** de hill_climbing_v2.py (18 variantes)
- [ ] **Re-executar experimento** com correções
- [ ] **Validar** se seeding 85% foi aplicado (buscar "SEVERE DRIFT DETECTED: Seeding" no log)
- [ ] **Analisar** resultado do chunk 4→5 (esperado 48-55%)
- [ ] **Decisão** GO/NO-GO para abordagens alternativas

---

**Criado por**: Claude Code
**Data**: 2025-10-24
**Status**: ⚠️ **FASE 1-NOVO TESTADA, SEEDING 85% NÃO APLICADO (BUG)**
**Próximo Passo**: **CORRIGIR BUG SEEDING 85% E RE-TESTAR**
