# Plano de Execucao - Batches 2, 3 e 4

**Data**: 2025-11-19
**Experimentos Totais**: 42 drift simulation
**Batch 1 (COMPLETO)**: 12 experimentos abrupt
**Restantes**: 30 experimentos

---

## VISAO GERAL

| Batch | Tema | Experimentos | Tempo Estimado | Sessoes Colab Free |
|-------|------|--------------|----------------|-------------------|
| **Batch 1** | Abrupt Fundamentais | 12 | 13h40min | COMPLETO ✅ |
| **Batch 2** | Gradual Fundamentais | 12 | ~13-15h | 2 sessoes |
| **Batch 3** | Noise & Mixed | 10 | ~11-13h | 1-2 sessoes |
| **Batch 4** | Complementares | 8 | ~10-12h | 1-2 sessoes |
| **TOTAL** | 4 batches | **42** | **~48-53h** | **6-8 sessoes** |

---

## BATCH 2: GRADUAL DRIFTS (Fundamentais)

**Objetivo**: Comparar abrupt vs gradual, fundamental para publicacao
**Total**: 12 experimentos
**Tempo Estimado**: ~13-15 horas

### Experimentos Selecionados:

#### Rapidos (~30-40 min cada):
1. **SEA_Gradual_Simple_Fast** - Gradual rapido, 1 drift
2. **SEA_Gradual_Simple_Slow** - Gradual lento, 1 drift
3. **SEA_Gradual_Recurring** - Gradual com recorrencia
4. **STAGGER_Gradual_Chain** - Gradual com cadeia

#### Medios (~50-70 min cada):
5. **AGRAWAL_Gradual_Chain** - Gradual com cadeia
6. **AGRAWAL_Gradual_Mild_to_Severe** - Gradual progressivo
7. **AGRAWAL_Gradual_Blip** - Gradual com blip

#### Lentos (~90-150 min cada):
8. **RBF_Gradual_Moderate** - Moderado
9. **RBF_Gradual_Severe** - Severo
10. **HYPERPLANE_Gradual_Simple** - Simples
11. **RANDOMTREE_Gradual_Simple** - Simples
12. **LED_Gradual_Simple** - Simples (novo dataset)

### Justificativa:
- **Gradual vs Abrupt**: Comparacao direta com Batch 1
- **Mesmos datasets**: SEA, AGRAWAL, RBF, STAGGER, HYPERPLANE, RANDOMTREE
- **Novos datasets**: LED (adiciona diversidade)
- **Padroes variados**: Simple, Chain, Recurring, Blip, Progressive

### Agrupamento para Colab:

**Grupo 2A** (~6.5h):
- SEA_Gradual_Simple_Fast (35 min)
- SEA_Gradual_Simple_Slow (35 min)
- SEA_Gradual_Recurring (35 min)
- STAGGER_Gradual_Chain (28 min)
- AGRAWAL_Gradual_Chain (55 min)
- AGRAWAL_Gradual_Blip (55 min)
- LED_Gradual_Simple (90 min)

**Grupo 2B** (~7.5h):
- AGRAWAL_Gradual_Mild_to_Severe (60 min)
- RBF_Gradual_Moderate (110 min)
- RBF_Gradual_Severe (140 min)
- HYPERPLANE_Gradual_Simple (140 min)
- RANDOMTREE_Gradual_Simple (140 min)

---

## BATCH 3: NOISE & MIXED (Complexidade)

**Objetivo**: Testar robustez a ruido
**Total**: 10 experimentos
**Tempo Estimado**: ~11-13 horas

### Experimentos Selecionados:

#### Com Noise (~30-150 min cada):
1. **SEA_Abrupt_Chain_Noise** - Abrupt com ruido
2. **STAGGER_Abrupt_Chain_Noise** - Abrupt com ruido
3. **AGRAWAL_Abrupt_Simple_Severe_Noise** - Abrupt severo com ruido
4. **AGRAWAL_Gradual_Recurring_Noise** - Gradual recorrente com ruido
5. **RBF_Abrupt_Blip_Noise** - Blip com ruido
6. **RBF_Gradual_Severe_Noise** - Gradual severo com ruido
7. **HYPERPLANE_Gradual_Noise** - Gradual com ruido
8. **RANDOMTREE_Gradual_Noise** - Gradual com ruido
9. **SINE_Abrupt_Recurring_Noise** - Abrupt recorrente com ruido

#### Mixed:
10. **STAGGER_Mixed_Recurring** - Abrupt + Gradual misturado

### Justificativa:
- **Robustez**: Testar desempenho com ruido
- **Variedade**: Abrupt + Gradual com noise
- **Mixed**: Combinacao de tipos de drift
- **Novos datasets**: SINE (adiciona diversidade)

### Agrupamento para Colab:

**Grupo 3A** (~5.5h):
- SEA_Abrupt_Chain_Noise (40 min)
- STAGGER_Abrupt_Chain_Noise (28 min)
- STAGGER_Mixed_Recurring (30 min)
- AGRAWAL_Abrupt_Simple_Severe_Noise (60 min)
- AGRAWAL_Gradual_Recurring_Noise (60 min)
- SINE_Abrupt_Recurring_Noise (35 min)
- RBF_Abrupt_Blip_Noise (110 min)

**Grupo 3B** (~6.5h):
- RBF_Gradual_Severe_Noise (150 min)
- HYPERPLANE_Gradual_Noise (150 min)
- RANDOMTREE_Gradual_Noise (150 min)

---

## BATCH 4: COMPLEMENTARES (Finalizacao)

**Objetivo**: Completar cobertura de datasets e padroes
**Total**: 8 experimentos
**Tempo Estimado**: ~10-12 horas

### Experimentos Selecionados:

#### Abrupt Complementares:
1. **SINE_Abrupt_Simple** - Novo dataset, simples
2. **LED_Abrupt_Simple** - Novo dataset, simples
3. **WAVEFORM_Abrupt_Simple** - Novo dataset, simples
4. **RANDOMTREE_Abrupt_Recurring** - Recorrencia

#### Gradual Complementares:
5. **SINE_Gradual_Recurring** - Recorrencia gradual
6. **WAVEFORM_Gradual_Simple** - Simples gradual
7. **AGRAWAL_Gradual_Recurring** - Recorrencia gradual
8. **RBF_Severe_Gradual_Recurrent** - Severo recorrente

### Justificativa:
- **Novos datasets**: SINE, LED, WAVEFORM (cobertura completa)
- **Padroes especiais**: Recurring em diferentes datasets
- **Completude**: Todas combinacoes importantes

### Agrupamento para Colab:

**Grupo 4A** (~5.5h):
- SINE_Abrupt_Simple (35 min)
- SINE_Gradual_Recurring (35 min)
- LED_Abrupt_Simple (90 min)
- WAVEFORM_Abrupt_Simple (90 min)
- WAVEFORM_Gradual_Simple (90 min)

**Grupo 4B** (~6h):
- RANDOMTREE_Abrupt_Recurring (140 min)
- AGRAWAL_Gradual_Recurring (60 min)
- RBF_Severe_Gradual_Recurrent (160 min)

---

## CRONOGRAMA SUGERIDO

### Usando Colab Free (~12h por sessao):

**Semana 1:**
- Dia 1: Batch 2A (6.5h)
- Dia 2: Batch 2B (7.5h)

**Semana 2:**
- Dia 3: Batch 3A (5.5h)
- Dia 4: Batch 3B (6.5h)

**Semana 3:**
- Dia 5: Batch 4A (5.5h)
- Dia 6: Batch 4B (6h)

**Total**: 6 sessoes ao longo de 3 semanas

### Usando Colab Pro (~24h por sessao):

**Semana 1:**
- Dia 1: Batch 2 completo (13-15h)
- Dia 2: Batch 3 completo (11-13h)
- Dia 3: Batch 4 completo (10-12h)

**Total**: 3 dias consecutivos

---

## ESTIMATIVAS DE TEMPO POR DATASET

Baseado no Batch 1 (tempo medio por base):

| Dataset Base | Tempo Medio | Categoria |
|-------------|-------------|-----------|
| **SEA** | ~33 min | Rapido |
| **STAGGER** | ~25 min | Rapido |
| **SINE** | ~35 min | Rapido (estimado) |
| **AGRAWAL** | ~55 min | Medio |
| **WAVEFORM** | ~90 min | Medio (estimado) |
| **LED** | ~90 min | Medio (estimado) |
| **RBF** | ~120 min | Lento |
| **HYPERPLANE** | ~135 min | Lento |
| **RANDOMTREE** | ~135 min | Lento |

**Notas**:
- Gradual pode ser 10-20% mais lento que Abrupt (mais geracoes do GA)
- Noise pode ser 5-10% mais lento (dados mais complexos)
- Estimativas para LED, WAVEFORM e SINE baseadas em complexidade similar

---

## PRIORIDADES

### Prioridade ALTA (essencial para publicacao):
1. **Batch 2** - Gradual Fundamentais
   - Comparacao abrupt vs gradual e CRITICA
   - Mesmos datasets do Batch 1 para comparacao direta

### Prioridade MEDIA (robus

tez):
2. **Batch 3** - Noise & Mixed
   - Demonstra robustez do metodo
   - Importante mas nao essencial para primeira submissao

### Prioridade BAIXA (completude):
3. **Batch 4** - Complementares
   - Adiciona novos datasets
   - Pode ser executado depois se tempo escasso

---

## ESTRUTURA DE DIRETORIOS

Todos os batches usam a mesma estrutura do Batch 1:

```
experiments_6chunks_phase2_gbml/
├── batch_1/  (COMPLETO - 12 abrupt)
│   ├── SEA_Abrupt_Simple/
│   ├── SEA_Abrupt_Chain/
│   └── ...
├── batch_2/  (NOVO - 12 gradual)
│   ├── SEA_Gradual_Simple_Fast/
│   ├── AGRAWAL_Gradual_Chain/
│   └── ...
├── batch_3/  (NOVO - 10 noise + mixed)
│   ├── SEA_Abrupt_Chain_Noise/
│   ├── STAGGER_Mixed_Recurring/
│   └── ...
└── batch_4/  (NOVO - 8 complementares)
    ├── SINE_Abrupt_Simple/
    ├── LED_Abrupt_Simple/
    └── ...
```

---

## ARQUIVOS DE CONFIGURACAO

Serao criados:
- `configs/config_batch_2.yaml` - 12 experimentos gradual
- `configs/config_batch_3.yaml` - 10 experimentos noise + mixed
- `configs/config_batch_4.yaml` - 8 experimentos complementares

Cada config tera:
```yaml
experiment_settings:
  run_mode: drift_simulation
  drift_simulation_experiments:
    - EXPERIMENTO_1
    - EXPERIMENTO_2
    - ...
  num_runs: 1

# Paths especificos do batch
base_results_dir: experiments_6chunks_phase2_gbml/batch_X

drift_analysis:
  heatmap_save_directory: experiments_6chunks_phase2_gbml/test_real_results_heatmaps/concept_heatmaps
```

---

## POS-PROCESSAMENTO

Para cada batch, apos execucao:

1. **Executar analyze_concept_difference.py** (1x por batch)
2. **Executar generate_plots.py** (1x por experimento)
3. **Executar rule_diff_analyzer.py** (1x por experimento)

Scripts serao criados:
- `post_process_batch_2.py`
- `post_process_batch_3.py`
- `post_process_batch_4.py`

Tempo de pos-processamento: ~1h por batch

---

## CONSOLIDACAO FINAL

Apos todos os batches:

1. **Consolidar metricas**: Combinar resultados dos 4 batches
2. **Analise comparativa**: Abrupt vs Gradual vs Noise
3. **Analise por dataset**: Desempenho em cada tipo de dado
4. **Statistical tests**: Wilcoxon, Friedman, etc.
5. **Plots finais**: Comparacoes para paper

Script: `consolidate_all_batches.py` (a criar)

---

## RESUMO EXECUTIVO

**Batch 1**: ✅ COMPLETO (12 abrupt, 13h40min)

**Proximos Passos**:
1. Criar configs para Batches 2, 3, 4
2. Executar Batch 2 (gradual) - **PRIORIDADE ALTA**
3. Executar Batch 3 (noise) - **PRIORIDADE MEDIA**
4. Executar Batch 4 (complementares) - **PRIORIDADE BAIXA**
5. Consolidar todos os resultados

**Tempo Total Estimado**: ~50 horas de experimentos + ~4 horas de pos-processamento

**Cronograma Otimo**: 3 semanas (2 sessoes Colab/semana) ou 3 dias (Colab Pro)

---

**Status**: PLANO PRONTO PARA EXECUCAO
**Proximo Passo**: Criar arquivos de configuracao para os batches
