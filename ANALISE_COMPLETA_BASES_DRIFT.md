# Análise Completa - Bases de Drift Simulation Disponíveis

**Data**: 2025-01-18
**Total de bases identificadas**: 48 bases de drift simulation (excluindo stationary e real datasets)

---

## CATEGORIZAÇÃO COMPLETA

### Por Tipo de Dataset:

**SEA** (10 bases):
1. SEA_Abrupt_Simple [JÁ CORRIGIDO]
2. SEA_Abrupt_Chain
3. SEA_Abrupt_Recurring
4. SEA_Abrupt_Chain_Noise
5. SEA_Gradual_Simple_Fast
6. SEA_Gradual_Simple_Slow
7. SEA_Gradual_Recurring
8. Bartosz_SEA_drift_noise

**AGRAWAL** (11 bases):
1. AGRAWAL_Abrupt_Simple_Mild
2. AGRAWAL_Abrupt_Simple_Severe [JÁ CORRIGIDO]
3. AGRAWAL_Abrupt_Simple_Severe_Noise
4. AGRAWAL_Abrupt_Chain_Long
5. AGRAWAL_Gradual_Chain
6. AGRAWAL_Gradual_Recurring
7. AGRAWAL_Gradual_Recurring_Noise
8. AGRAWAL_Gradual_Mild_to_Severe
9. AGRAWAL_Gradual_Blip
10. Bartosz_Agrawal_recurring_drift

**RBF** (8 bases):
1. RBF_Abrupt_Severe [JÁ CORRIGIDO]
2. RBF_Abrupt_Blip
3. RBF_Abrupt_Blip_Noise
4. RBF_Gradual_Moderate
5. RBF_Gradual_Severe
6. RBF_Gradual_Severe_Noise
7. RBF_Severe_Gradual_Recurrent

**STAGGER** (5 bases):
1. STAGGER_Abrupt_Chain [JÁ CORRIGIDO]
2. STAGGER_Abrupt_Chain_Noise
3. STAGGER_Abrupt_Recurring
4. STAGGER_Mixed_Recurring
5. STAGGER_Gradual_Chain

**HYPERPLANE** (3 bases):
1. HYPERPLANE_Abrupt_Simple [JÁ CORRIGIDO]
2. HYPERPLANE_Gradual_Simple
3. HYPERPLANE_Gradual_Noise

**SINE** (3 bases):
1. SINE_Abrupt_Simple
2. SINE_Gradual_Recurring
3. SINE_Abrupt_Recurring_Noise

**RANDOMTREE** (3 bases):
1. RANDOMTREE_Abrupt_Simple
2. RANDOMTREE_Abrupt_Recurring
3. RANDOMTREE_Gradual_Simple
4. RANDOMTREE_Gradual_Noise
5. Bartosz_RandomTree_drift

**LED** (2 bases):
1. LED_Abrupt_Simple
2. LED_Gradual_Simple

**WAVEFORM** (2 bases):
1. WAVEFORM_Abrupt_Simple
2. WAVEFORM_Gradual_Simple

---

### Por Tipo de Drift:

**ABRUPT** (20 bases):
- SEA_Abrupt_Simple, SEA_Abrupt_Chain, SEA_Abrupt_Recurring, SEA_Abrupt_Chain_Noise
- AGRAWAL_Abrupt_Simple_Mild, AGRAWAL_Abrupt_Simple_Severe, AGRAWAL_Abrupt_Simple_Severe_Noise, AGRAWAL_Abrupt_Chain_Long
- RBF_Abrupt_Severe, RBF_Abrupt_Blip, RBF_Abrupt_Blip_Noise
- STAGGER_Abrupt_Chain, STAGGER_Abrupt_Chain_Noise, STAGGER_Abrupt_Recurring
- HYPERPLANE_Abrupt_Simple
- SINE_Abrupt_Simple, SINE_Abrupt_Recurring_Noise
- RANDOMTREE_Abrupt_Simple, RANDOMTREE_Abrupt_Recurring
- LED_Abrupt_Simple, WAVEFORM_Abrupt_Simple

**GRADUAL** (21 bases):
- SEA_Gradual_Simple_Fast, SEA_Gradual_Simple_Slow, SEA_Gradual_Recurring
- AGRAWAL_Gradual_Chain, AGRAWAL_Gradual_Recurring, AGRAWAL_Gradual_Recurring_Noise, AGRAWAL_Gradual_Mild_to_Severe, AGRAWAL_Gradual_Blip
- RBF_Gradual_Moderate, RBF_Gradual_Severe, RBF_Gradual_Severe_Noise, RBF_Severe_Gradual_Recurrent
- STAGGER_Gradual_Chain
- HYPERPLANE_Gradual_Simple, HYPERPLANE_Gradual_Noise
- SINE_Gradual_Recurring
- RANDOMTREE_Gradual_Simple, RANDOMTREE_Gradual_Noise
- LED_Gradual_Simple, WAVEFORM_Gradual_Simple

**MIXED** (1 base):
- STAGGER_Mixed_Recurring

**Bartosz (legacy)** (3 bases):
- Bartosz_RandomTree_drift, Bartosz_Agrawal_recurring_drift, Bartosz_SEA_drift_noise

---

### Por Padrão de Drift:

**SIMPLE** (drift único, 2 conceitos) (18 bases):
- SEA_Abrupt_Simple, SEA_Gradual_Simple_Fast, SEA_Gradual_Simple_Slow
- AGRAWAL_Abrupt_Simple_Mild, AGRAWAL_Abrupt_Simple_Severe, AGRAWAL_Abrupt_Simple_Severe_Noise
- RBF_Abrupt_Severe, RBF_Gradual_Moderate, RBF_Gradual_Severe, RBF_Gradual_Severe_Noise
- HYPERPLANE_Abrupt_Simple, HYPERPLANE_Gradual_Simple, HYPERPLANE_Gradual_Noise
- SINE_Abrupt_Simple
- RANDOMTREE_Abrupt_Simple, RANDOMTREE_Gradual_Simple, RANDOMTREE_Gradual_Noise
- LED_Abrupt_Simple, LED_Gradual_Simple, WAVEFORM_Abrupt_Simple, WAVEFORM_Gradual_Simple

**CHAIN** (3+ conceitos sequenciais) (7 bases):
- SEA_Abrupt_Chain, SEA_Abrupt_Chain_Noise
- AGRAWAL_Gradual_Chain, AGRAWAL_Abrupt_Chain_Long, AGRAWAL_Gradual_Mild_to_Severe
- STAGGER_Abrupt_Chain, STAGGER_Abrupt_Chain_Noise, STAGGER_Gradual_Chain

**RECURRING** (conceito volta) (9 bases):
- SEA_Abrupt_Recurring, SEA_Gradual_Recurring
- AGRAWAL_Gradual_Recurring, AGRAWAL_Gradual_Recurring_Noise, Bartosz_Agrawal_recurring_drift
- RBF_Severe_Gradual_Recurrent
- STAGGER_Abrupt_Recurring, STAGGER_Mixed_Recurring
- SINE_Gradual_Recurring, SINE_Abrupt_Recurring_Noise
- RANDOMTREE_Abrupt_Recurring

**BLIP** (conceito temporário) (3 bases):
- RBF_Abrupt_Blip, RBF_Abrupt_Blip_Noise
- AGRAWAL_Gradual_Blip

---

### Com/Sem Ruído:

**SEM RUÍDO** (38 bases): Maioria

**COM RUÍDO** (10 bases):
- SEA_Abrupt_Chain_Noise, Bartosz_SEA_drift_noise
- AGRAWAL_Abrupt_Simple_Severe_Noise, AGRAWAL_Gradual_Recurring_Noise
- RBF_Abrupt_Blip_Noise, RBF_Gradual_Severe_Noise
- STAGGER_Abrupt_Chain_Noise
- HYPERPLANE_Gradual_Noise
- SINE_Abrupt_Recurring_Noise
- RANDOMTREE_Gradual_Noise

---

## ESTRATÉGIA DE ORGANIZAÇÃO EM BATCHES

### Proposta: Dividir em 4 Batches Temáticos

**BATCH 1: Abrupt Drifts - Fundamentais** (12 bases, ~12-14 horas)
Foco: Drifts abruptos, cenários fundamentais

1. SEA_Abrupt_Simple [CORRIGIDO]
2. SEA_Abrupt_Chain
3. SEA_Abrupt_Recurring
4. AGRAWAL_Abrupt_Simple_Mild
5. AGRAWAL_Abrupt_Simple_Severe [CORRIGIDO]
6. AGRAWAL_Abrupt_Chain_Long
7. RBF_Abrupt_Severe [CORRIGIDO]
8. RBF_Abrupt_Blip
9. STAGGER_Abrupt_Chain [CORRIGIDO]
10. STAGGER_Abrupt_Recurring
11. HYPERPLANE_Abrupt_Simple [CORRIGIDO]
12. RANDOMTREE_Abrupt_Simple

**Características**:
- Todos drift abrupt
- Mix de simple (6), chain (3), recurring (2), blip (1)
- Datasets: SEA (3), AGRAWAL (3), RBF (2), STAGGER (2), HYPERPLANE (1), RANDOMTREE (1)
- 5 bases já corrigidas

---

**BATCH 2: Gradual Drifts - Transições Suaves** (12 bases, ~12-14 horas)
Foco: Drifts graduais, diferentes velocidades

1. SEA_Gradual_Simple_Fast
2. SEA_Gradual_Simple_Slow
3. SEA_Gradual_Recurring
4. AGRAWAL_Gradual_Chain
5. AGRAWAL_Gradual_Recurring
6. AGRAWAL_Gradual_Mild_to_Severe
7. RBF_Gradual_Moderate
8. RBF_Gradual_Severe
9. RBF_Severe_Gradual_Recurrent
10. STAGGER_Gradual_Chain
11. HYPERPLANE_Gradual_Simple
12. RANDOMTREE_Gradual_Simple

**Características**:
- Todos drift gradual
- Mix de simple (5), chain (3), recurring (3), severity progression (1)
- Velocidades: Fast, Slow, Moderate
- Datasets: SEA (3), AGRAWAL (3), RBF (3), STAGGER (1), HYPERPLANE (1), RANDOMTREE (1)

---

**BATCH 3: Noise & Mixed - Cenários Complexos** (10 bases, ~10-12 horas)
Foco: Ruído, mixed drifts, cenários desafiadores

1. SEA_Abrupt_Chain_Noise
2. AGRAWAL_Abrupt_Simple_Severe_Noise
3. AGRAWAL_Gradual_Recurring_Noise
4. AGRAWAL_Gradual_Blip
5. RBF_Abrupt_Blip_Noise
6. RBF_Gradual_Severe_Noise
7. STAGGER_Abrupt_Chain_Noise
8. STAGGER_Mixed_Recurring
9. HYPERPLANE_Gradual_Noise
10. RANDOMTREE_Gradual_Noise

**Características**:
- 7 com ruído, 3 padrões complexos (blip, mixed)
- Mix de abrupt e gradual
- Datasets diversos

---

**BATCH 4: Complementares & Especiais** (11 bases, ~11-13 horas)
Foco: Novos datasets, recurring, legacy, especiais

1. SINE_Abrupt_Simple
2. SINE_Gradual_Recurring
3. SINE_Abrupt_Recurring_Noise
4. LED_Abrupt_Simple
5. LED_Gradual_Simple
6. WAVEFORM_Abrupt_Simple
7. WAVEFORM_Gradual_Simple
8. RANDOMTREE_Abrupt_Recurring
9. Bartosz_RandomTree_drift
10. Bartosz_Agrawal_recurring_drift
11. Bartosz_SEA_drift_noise

**Características**:
- Datasets novos: SINE (3), LED (2), WAVEFORM (2)
- Legacy Bartosz (3)
- Mix de abrupt/gradual

---

## ESTIMATIVAS DE TEMPO POR BATCH

### Batch 1 (12 bases, Abrupt Fundamentais):

Assumindo tempos similares às 5 bases já executadas:
- Rápidas (SEA, STAGGER): ~30 min cada (6 bases) = 180 min
- Médias (AGRAWAL): ~60-70 min cada (3 bases) = 200 min
- Lentas (RBF, HYPERPLANE, RANDOMTREE): ~120-180 min cada (3 bases) = 420 min

**Total estimado**: ~800 min (~13.3 horas)

**Agrupamento para Colab**:
- Opção A: 2 instâncias de ~6.5h cada (Colab Free)
- Opção B: 1 instância de ~13h (Colab Pro)

---

### Batch 2 (12 bases, Gradual):

Gradual drifts tendem a ser ligeiramente mais lentos (mais gerações do GA):
- Estimativa: +10-15% sobre abrupt
- **Total estimado**: ~900 min (~15 horas)

**Agrupamento para Colab**:
- Requer 2 instâncias (7.5h cada) ou 1 Colab Pro

---

### Batch 3 (10 bases, Noise & Mixed):

Noise adiciona complexidade (+15-20%):
- **Total estimado**: ~750 min (~12.5 horas)

**Agrupamento para Colab**:
- 2 instâncias de ~6h cada (confortável)

---

### Batch 4 (11 bases, Complementares):

Novos datasets (SINE, LED, WAVEFORM) - tempos desconhecidos:
- Estimativa conservadora
- **Total estimado**: ~800 min (~13.3 horas)

**Agrupamento para Colab**:
- 2 instâncias de ~6.5h cada

---

## CRONOGRAMA DE EXECUÇÃO TOTAL

### Sequencial (não recomendado):
- Batch 1: 13 horas
- Batch 2: 15 horas
- Batch 3: 12 horas
- Batch 4: 13 horas
- **Total**: ~53 horas

### Paralelo (2 instâncias por batch):
- Cada batch: ~7-8 horas wall-clock
- 4 batches sequenciais: ~30-32 horas wall-clock
- **Total**: ~4 dias (executando 1 batch/dia)

### Paralelo Agressivo (executar múltiplos batches simultaneamente):
- Batch 1 + Batch 2 em paralelo: ~15 horas (dia 1)
- Batch 3 + Batch 4 em paralelo: ~13 horas (dia 2)
- **Total**: ~2 dias

---

## PRIORIZAÇÃO RECOMENDADA

### Fase 1 (IMEDIATO): Batch 1 Completo
**12 bases abrupt fundamentais**

Motivos:
1. Inclui as 5 bases já corrigidas (aproveitamento)
2. Cenários fundamentais (mais citados na literatura)
3. Baseline para comparar com gradual
4. Menor risco (abrupt é mais comum)

**Ação**: Completar correções das 7 bases restantes do Batch 1

---

### Fase 2 (CURTO PRAZO): Batch 2
**12 bases gradual**

Motivos:
1. Complementa Batch 1 (abrupt vs gradual)
2. Permite análise comparativa de tipos de drift
3. Importante para publicação (cobrir diferentes cenários)

---

### Fase 3 (MÉDIO PRAZO): Batch 3
**10 bases noise & mixed**

Motivos:
1. Cenários mais realistas (ruído)
2. Testa robustez dos modelos
3. Diferencial para publicação

---

### Fase 4 (LONGO PRAZO): Batch 4
**11 bases complementares**

Motivos:
1. Novos datasets (diversidade)
2. Legacy (comparação com trabalhos anteriores)
3. Menor prioridade científica

---

## RECOMENDAÇÃO FINAL

### Proposta: Focar no Batch 1 Expandido (12 bases)

**Bases a adicionar às 5 já corrigidas**:
1. SEA_Abrupt_Chain
2. SEA_Abrupt_Recurring
3. AGRAWAL_Abrupt_Simple_Mild
4. AGRAWAL_Abrupt_Chain_Long
5. RBF_Abrupt_Blip
6. STAGGER_Abrupt_Recurring
7. RANDOMTREE_Abrupt_Simple

**Total**: 12 bases abrupt (fundamentais)
**Tempo**: ~13 horas (2 instâncias de 6.5h cada)

**Vantagens**:
1. Aproveita 5 bases já corrigidas
2. Cobre cenários fundamentais abrupt
3. 4 datasets diferentes (SEA, AGRAWAL, RBF, STAGGER, HYPERPLANE, RANDOMTREE)
4. 4 padrões (simple, chain, recurring, blip)
5. Excelente baseline para comparações

**Próximos passos**:
1. Corrigir 7 bases novas
2. Validar todas as 12
3. Executar em 2 instâncias Colab
4. Gerar resultados completos

---

## CORREÇÕES NECESSÁRIAS (7 bases novas)

### Resumo:

| # | Base | Mudanças Necessárias |
|---|------|---------------------|
| 1 | SEA_Abrupt_Chain | 3 linhas: duration 3→2 |
| 2 | SEA_Abrupt_Recurring | 3 linhas: duration 4,5,4→2,2,2 |
| 3 | AGRAWAL_Abrupt_Simple_Mild | 2 linhas: duration 5→3 |
| 4 | AGRAWAL_Abrupt_Chain_Long | 5 linhas: duration 2→? (verificar) |
| 5 | RBF_Abrupt_Blip | 3 linhas: duration 6,1,6→? (verificar) |
| 6 | STAGGER_Abrupt_Recurring | 3 linhas: duration 4,5,4→2,2,2 |
| 7 | RANDOMTREE_Abrupt_Simple | 2 linhas: duration 5→3 (estimar) |

**Total**: ~21 linhas a modificar

---

**Status**: PLANO COMPLETO - AGUARDANDO DECISÃO
**Recomendação**: Executar Batch 1 (12 bases abrupt) como prioridade
**Tempo estimado**: 2 dias (preparação + execução)
**Impacto**: Cobertura abrangente de cenários abrupt fundamentais
