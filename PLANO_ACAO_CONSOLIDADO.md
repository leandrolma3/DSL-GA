# 🎯 PLANO DE AÇÃO CONSOLIDADO - GBML DSL-AG-HYBRID

**Data**: 2025-10-22
**Baseado em**: Experimento 6 chunks (novo_experimento.log) + Documentação completa

---

## 📊 CONTEXTO: RESULTADO DO EXPERIMENTO 6 CHUNKS

### Performance Atual

| Métrica | Resultado | Meta | Gap |
|---------|-----------|------|-----|
| **Avg Test G-mean** | 81.63% | 85-87% | **-3.37 a -5.37pp** |
| **Chunks 0-3 G-mean** | 87.82-89.69% | ✅ Excelente | Atingiu meta |
| **Chunk 4→5 (drift)** | **52.58%** 💥 | 70%+ | **-17.42pp** (CRÍTICO) |
| **HC Taxa Aprovação** | 5.8% | 30-50% | **-24.2 a -44.2pp** |
| **Early Stop Economia** | 44.7% | 70-80% | **-25.3 a -35.3pp** |

### ✅ O Que Funciona

1. **Early Stopping Layer 1**: Ativou corretamente (Gen 63, elite 89%, 15 gens estagnado)
2. **HC Inteligente - Implementação**: Extração DT (53-153 regras), ranqueamento funcional
3. **Crossover Balanceado**: Diversidade mantida em 0.55-0.60 (vs 0.41-0.43 anterior)
4. **Seeding Adaptativo**: Classificação MEDIUM consistente (78-79% probe accuracy)
5. **Chunks 0-3**: Performance excelente, estável

### ❌ Problemas Críticos

1. **Drift Detection NÃO FUNCIONA** - Bug crítico: usa train G-mean ao invés de test G-mean
2. **HC Taxa 5.8%** - Muito baixa (meta 30-50%), 18/36 ativações com 0% aprovação
3. **Chunk 4→5 Colapso** - 52.58% G-mean (sem adaptação ao severe drift)

---

## 🔥 PRIORIDADE 1: CORRIGIR DRIFT DETECTION (CRÍTICO)

### Status: ❌ **BUG CONFIRMADO**

**Problema**: Sistema classifica drift baseado em **train G-mean**, mas drift só aparece em **test G-mean**

**Evidência**:
```
Chunk 3: Train=91.61%, Test=89.69%  ← historical_gmean.append(0.9161)
Chunk 4: Train=89.02%, Test=52.58%  ← historical_gmean.append(0.8902)
performance_drop = (0.9161 - 0.8902) / 0.9161 = 2.83% < 20% ← NÃO DETECTA SEVERE!

Mas o drift real é:
Test: 89.69% → 52.58% = -37.11pp = -41.4% ← SEVERE DRIFT! (mas não detectado)
```

### Correção

**Arquivo**: `main.py` linha ~695

```python
# ANTES (ERRADO):
historical_gmean.append(train_gmean_final)  # ← Usa treino!

# DEPOIS (CORRETO):
historical_gmean.append(test_gmean_current)  # ← Usar teste!
```

### Validação Rápida

```bash
# Teste chunk 4→5 isolado
python main.py --chunks 2 --start-chunk 4 --seed 43

# Esperado no log:
# 🔴 SEVERE DRIFT detected: 0.8969 → 0.5258 (drop: 41.4%)
# → Memory cleared due to SEVERE drift
# → Inheritance DISABLED due to SEVERE drift (was 50%)
# 🔴 Recipe adjusted for SEVERE drift: {'seeded': 0.6, 'random': 0.3, 'memory': 0.0}
```

### Impacto Esperado

| Métrica | Atual | Após Correção | Melhoria |
|---------|-------|---------------|----------|
| Chunk 4→5 G-mean | 52.58% | **70-75%** (estimado) | **+17.42-22.42pp** |
| Drift detection ativa? | ❌ Nunca | ✅ Sim | +100% |
| Memory cleared? | ❌ Não | ✅ Sim (drift severo) | +100% |
| Recipe ajustada? | ❌ Não | ✅ Sim (seeded 60%, random 30%) | +100% |

**Tempo**: 30 minutos implementação + 3h teste isolado = **3.5h**

---

## 🔥 PRIORIDADE 2: MELHORAR TAXA HC INTELIGENTE (MÉDIO-ALTO)

### Status: ⚠️ **IMPLEMENTAÇÃO OK, TAXA BAIXA**

**Problema**: Taxa global 5.8% (vs meta 30-50%)

**Análise Detalhada**:
- ✅ Implementação funcionando (extração DT, ranqueamento OK)
- ✅ Top rules de alta qualidade (39.66% coverage, 100% accuracy)
- ❌ Inconsistência brutal: 0% a 30.8% por ativação
- ❌ 18/36 ativações com 0% aprovação (50% das ativações inúteis)

**Distribuição de Taxas**:
- 30.8% taxa: 2 ativações (melhor caso)
- 15.4% taxa: 5 ativações
- 7.7% taxa: 3 ativações
- **0.0% taxa: 18 ativações** ← PROBLEMA!

### Hipótese da Causa Raiz

**Quando HC ativa**: Estagnação de 10 gerações
**Situação típica**: Elite em 86-90% G-mean (muito forte)
**Variantes DT**: Fitness ~85-88% < elite 89% → rejeitadas

**Por quê 0% aprovação?**
1. DT treina apenas em **erros do elite** (12.7% = 764/6000 instâncias)
2. DT aprende padrões dos erros, mas **não tem dados suficientes** (amostra pequena)
3. Regras DT ficam muito **específicas** (high precision, low coverage)
4. Quando injetadas no elite, **não melhoram fitness global** (G-mean todas as classes)

### Soluções Propostas (3 Opções)

#### **Opção 2A: Tolerância de Aprovação** ⭐ RECOMENDADO

**Conceito**: Aprovar variantes "quase tão boas quanto elite" (exploração)

```python
# ga.py linha ~1100
def should_approve_variant(variant_fitness, elite_fitness, tolerance=0.005):
    """
    Aprova variante se fitness >= elite - tolerance

    tolerance = 0.005 → aceita até 0.5% pior que elite
    Exemplo: elite=0.890, variante=0.887 → APROVADO (diff=0.3% < 0.5%)
    """
    return variant_fitness >= (elite_fitness - tolerance)

# ANTES:
if variant_fitness > elite_fitness:
    approve_variant()

# DEPOIS:
if should_approve_variant(variant_fitness, elite_fitness, tolerance=0.005):
    approve_variant()
```

**Justificativa**:
- Variantes "quase tão boas" podem ter **regras diferentes** (diversidade)
- Foco em erros pode sacrificar 0.5% fitness mas **corrigir 20% dos erros**
- Trade-off: -0.5% fitness, +diversidade, +especialização

**Impacto Esperado**:
- Taxa 5.8% → **20-30%** (4-5× melhoria)
- Ativações 0%: 18 → **5-8** (redução de 55%)

**Tempo**: 1h implementação + 14h teste = **15h**

---

#### **Opção 2B: Mais Variantes DT** (Complementar)

**Conceito**: Gerar 8-10 variantes DT (vs 5 atual)

```python
# hill_climbing_v2.py linha ~75
# ANTES:
error_focused_variants = 5
ensemble_boosting_variants = 5
guided_mutation_variants = 3
Total = 13 variantes

# DEPOIS:
error_focused_variants = 8  # +60%
ensemble_boosting_variants = 6  # +20%
guided_mutation_variants = 4  # +33%
Total = 18 variantes
```

**Justificativa**:
- Mais variantes → mais chances de encontrar combinação vencedora
- Custo baixo (apenas +5 variantes por ativação = +38% tempo HC)

**Impacto Esperado**:
- Taxa 5.8% → **8-12%** (1.4-2× melhoria)
- Custo: +10min por chunk (de 2h 46min → 2h 56min por chunk)

**Tempo**: 30min implementação + 14h teste = **14.5h**

---

#### **Opção 2C: Substituir 40% Regras** (Mais Agressivo)

**Conceito**: Substituir 40% piores regras do elite (vs 20-25% atual)

```python
# intelligent_hc_strategies.py linha ~104
# ANTES:
replacement_ratio = 0.25  # 25% das regras

# DEPOIS (ADAPTATIVO):
if elite_gmean < 0.85:
    replacement_ratio = 0.50  # 50% se elite fraco
elif elite_gmean < 0.88:
    replacement_ratio = 0.40  # 40% se elite médio
else:
    replacement_ratio = 0.30  # 30% se elite forte (vs 25% antes)
```

**Justificativa**:
- Elite forte (88-90%) → substituir mais regras para forçar mudança
- Risco: pode piorar elite temporariamente (mas tolerância 2A compensa)

**Impacto Esperado**:
- Taxa 5.8% → **12-18%** (2-3× melhoria)
- Risco: Pode desestabilizar elite temporariamente

**Tempo**: 1h implementação + 14h teste = **15h**

---

### 🎯 Estratégia Recomendada

**COMBINAR 2A + 2B** (tolerância + mais variantes):

```python
# 1. Tolerância 0.5% (Opção 2A)
tolerance = 0.005

# 2. Mais variantes (Opção 2B)
error_focused = 8  # (vs 5)
ensemble_boosting = 6  # (vs 5)
guided_mutation = 4  # (vs 3)
```

**Impacto Combinado**:
- Taxa 5.8% → **25-35%** (4-6× melhoria)
- Custo: +10min por chunk
- **Meta 30-50% atingida!** ✅

**Tempo Total**: 1.5h implementação + 14h teste = **15.5h**

---

## 🟢 PRIORIDADE 3: CROSSOVER BALANCEADO REFINAMENTO (BAIXO)

### Status: ✅ **JÁ IMPLEMENTADO E FUNCIONANDO**

**Evidência no log**:
```
[INFO] Crossover Balanceado Inteligente ATIVADO (70% qualidade + 30% diversidade)
```

**Diversidade observada**:
- Chunks 1-2: Manteve 0.55-0.60 (vs 0.41-0.43 experimento anterior) ✅
- Chunk 4: Recuperou para 0.59 no final (Gen 63)

### Possível Melhoria (Opcional)

**Crossover Adaptativo por Fase da Evolução**:

```python
# ga.py linha ~850 (crossover)
def get_adaptive_quality_ratio(generation, max_generations):
    """
    Adapta ratio exploração/exploitação conforme geração

    Gen 1-20%: 50-50 (exploração agressiva)
    Gen 20-60%: 70-30 (balanceado)
    Gen 60-100%: 85-15 (exploitação/refinamento)
    """
    progress = generation / max_generations

    if progress < 0.20:  # Primeiras 20% gerações
        return 0.50  # 50% qualidade, 50% diversidade
    elif progress < 0.60:  # Gerações 20-60%
        return 0.70  # 70% qualidade, 30% diversidade
    else:  # Últimas 40% gerações
        return 0.85  # 85% qualidade, 15% diversidade
```

**Impacto Esperado**:
- G-mean +0.5-1.0pp (refinamento final)
- Diversidade inicial maior (exploração)

**Tempo**: 2h implementação + 14h teste = **16h**

**Prioridade**: **BAIXA** (já funciona bem, melhoria marginal)

---

## 🟢 PRIORIDADE 4: EARLY STOPPING REFINAMENTO (OPCIONAL)

### Status: ✅ **LAYER 1 FUNCIONANDO PERFEITAMENTE**

**Layer 1 ativou no chunk 4**:
- Gen 63: Elite 89%, estagnação 15 gens → PAROU
- Economia: 68.5% das gerações (137 de 200)

**Problema**: Layer 1 só ativou **1 vez em 5 chunks** (20%)

**Por quê Layer 1 não ativa mais?**
- Threshold: Elite ≥ 88% + 15 gens estagnado
- Chunks 0,2,3: Recovery mode (max_gen=25) → termina antes de estagnar 15 gens
- Chunk 1: Elite não atingiu 88% até Gen 200

### Possível Melhoria (Opcional)

**Layer 1 Adaptativo**:

```python
# ga.py linha ~900
# ANTES:
SATISFACTORY_GMEAN = 0.88  # Fixo
EARLY_PATIENCE_LAYER1 = 15  # Fixo

# DEPOIS (ADAPTATIVO):
if max_generations <= 50:  # Recovery mode
    SATISFACTORY_GMEAN = 0.85  # Mais permissivo
    EARLY_PATIENCE_LAYER1 = 8   # Menos paciência
else:  # Modo normal
    SATISFACTORY_GMEAN = 0.88
    EARLY_PATIENCE_LAYER1 = 15
```

**Impacto Esperado**:
- Chunks 0,2,3: Layer 1 ativaria em Gen 15-20 (vs Gen 25 atual)
- Economia: 44.7% → **55-60%** das gerações

**Tempo**: 1h implementação + 14h teste = **15h**

**Prioridade**: **BAIXA** (Layer 1 já funciona, economia marginal)

---

## 📅 CRONOGRAMA DE IMPLEMENTAÇÃO

### **Semana 1: Correções Críticas**

**Dia 1-2** (3.5h):
- ✅ **PRIORIDADE 1**: Corrigir drift detection
- ✅ Testar chunk 4→5 isolado
- ✅ Validar mensagens de drift (🔴 SEVERE)

**Dia 3-4** (15.5h):
- ✅ **PRIORIDADE 2**: Implementar 2A (tolerância) + 2B (mais variantes)
- ✅ Teste completo 6 chunks
- ✅ Validar taxa HC ≥ 25%

**Dia 5** (Análise):
- ✅ Analisar resultados
- ✅ Decisão GO/NO-GO para Prioridade 3

---

### **Semana 2: Refinamentos (Se Necessário)**

**Dia 6-7** (16h):
- 🟢 **PRIORIDADE 3**: Crossover adaptativo (se G-mean < 85%)
- 🟢 Teste completo 6 chunks

**Dia 8** (Análise):
- 🟢 Validar G-mean ≥ 85%
- 🟢 Comparação com ARF/SRP (gap ≤ 3%)

---

## 🎯 METAS FINAIS

### Após Prioridade 1 + 2

| Métrica | Atual | Meta | Esperado |
|---------|-------|------|----------|
| **Avg Test G-mean** | 81.63% | 85-87% | **85.5%** ✅ |
| **Chunk 4→5** | 52.58% | 70%+ | **72-75%** ✅ |
| **HC Taxa** | 5.8% | 30-50% | **28-35%** ✅ |
| **Drift Detection** | ❌ 0% | ✅ 100% | **100%** ✅ |
| **Gap vs ARF** | -3.0pp | -2.0pp | **-2.2pp** 🟡 |

### Após Prioridade 3 (Opcional)

| Métrica | Meta Após P1+P2 | Meta Final | Esperado |
|---------|-----------------|------------|----------|
| **Avg Test G-mean** | 85.5% | **87%** | **86.8%** ✅ |
| **Gap vs ARF** | -2.2pp | **-1.5pp** | **-1.8pp** 🟡 |

---

## 🚀 DECISÃO CRÍTICA: GO/NO-GO

### Após Prioridade 1 + 2:

**GO** se:
- ✅ Test G-mean ≥ 85%
- ✅ Chunk 4→5 ≥ 70%
- ✅ HC Taxa ≥ 25%
- ✅ Drift detection funcionando

**NO-GO** se:
- ❌ Test G-mean < 83%
- ❌ Chunk 4→5 < 65%
- ❌ HC Taxa < 20%

**Se GO**: Prosseguir para Prioridade 3 (crossover adaptativo)
**Se NO-GO**: Investigar causas raiz adicionais

---

## 📝 RESUMO EXECUTIVO

### Implementação Imediata (Semana 1)

1. **Corrigir drift detection** (linha 695, `main.py`): `historical_gmean.append(test_gmean)`
2. **Tolerância HC 0.5%** (linha 1100, `ga.py`): `if fitness >= elite - 0.005`
3. **Mais variantes DT** (linha 75, `hill_climbing_v2.py`): `8+6+4 = 18 variantes`

### Validação (Semana 1-2)

4. **Teste completo 6 chunks** (~14-16h execução)
5. **Analisar**: G-mean ≥ 85%, HC ≥ 25%, Chunk 4→5 ≥ 70%

### Refinamento Opcional (Semana 2)

6. **Crossover adaptativo** (se necessário para atingir 87% G-mean)

### Tempo Total Estimado

- **Implementação**: 3-5h
- **Testes**: 28-32h (2 experimentos completos)
- **Análise**: 4-6h
- **TOTAL**: **35-43h** (~1 semana de trabalho)

---

## 🎬 CONCLUSÃO

**Caminho claro para atingir metas**:
1. Drift detection é **bug simples** (1 linha)
2. HC taxa é **ajuste de threshold** (tolerância 0.5%)
3. Ambos juntos: **impacto estimado +3.87pp G-mean**

**Meta 85% G-mean é alcançável** em ~1 semana de trabalho.

**Se atingir 85%**, considerar publicação/documentação acadêmica.
**Se não atingir 85%**, investigar gap vs ARF (arquitetura ensemble vs single-model).

---

**Criado por**: Claude Code
**Data**: 2025-10-22
**Baseado em**: novo_experimento.log + 15 documentos de design/análise
