# Plano de Integracao ROSE e Modelos Faltantes

**Data:** 2025-11-25
**Status:** Pronto para execucao

---

## 1. Resumo da Situacao Atual

### 1.1 Estrutura de Batches

| Batch | Fase | Tipo | Datasets |
|-------|------|------|----------|
| batch_1 | Phase 2 | Sinteticos - Abrupt Drifts | 12 datasets |
| batch_2 | Phase 2 | Sinteticos - Gradual Drifts | 9 datasets |
| batch_3 | Phase 2 | Sinteticos - Com Ruido | 8 datasets |
| batch_4 | Phase 2 | Sinteticos - Outros | 6 datasets |
| batch_5 | Phase 3 | Reais | 5 datasets |
| batch_6 | Phase 3 | Sinteticos Estacionarios 1 | 6 datasets |
| batch_7 | Phase 3 | Sinteticos Estacionarios 2 | 6 datasets |

**Total: 52 datasets**

### 1.2 Cobertura de Modelos Atual

| Modelo | batch_1-4 | batch_5 | batch_6-7 | Total |
|--------|-----------|---------|-----------|-------|
| GBML | 100% | 100% | 100% | 100% |
| HAT | 67% | 100% | 0% | ~50% |
| ARF | 67% | 100% | 0% | ~50% |
| SRP | 67% | 100% | 0% | ~50% |
| ACDWM | 60% | 40% | 0% | ~40% |
| ERulesD2S | 67% | 100% | 0% | ~50% |
| **ROSE** | **0%** | **0%** | **0%** | **0%** |

---

## 2. Datasets por Batch

### Batch 1 (12 datasets - Abrupt Drifts)
```
1.  SEA_Abrupt_Simple
2.  SEA_Abrupt_Chain
3.  SEA_Abrupt_Recurring
4.  AGRAWAL_Abrupt_Simple_Mild
5.  AGRAWAL_Abrupt_Simple_Severe
6.  AGRAWAL_Abrupt_Chain_Long
7.  RBF_Abrupt_Severe
8.  RBF_Abrupt_Blip
9.  STAGGER_Abrupt_Chain
10. STAGGER_Abrupt_Recurring
11. HYPERPLANE_Abrupt_Simple
12. RANDOMTREE_Abrupt_Simple
```

### Batch 2 (9 datasets - Gradual Drifts)
```
1. SEA_Gradual_Simple_Fast
2. SEA_Gradual_Simple_Slow
3. SEA_Gradual_Recurring
4. STAGGER_Gradual_Chain
5. RBF_Gradual_Moderate
6. RBF_Gradual_Severe
7. HYPERPLANE_Gradual_Simple
8. RANDOMTREE_Gradual_Simple
9. LED_Gradual_Simple
```

### Batch 3 (8 datasets - Com Ruido)
```
1. SEA_Abrupt_Chain_Noise
2. STAGGER_Abrupt_Chain_Noise
3. AGRAWAL_Abrupt_Simple_Severe_Noise
4. SINE_Abrupt_Recurring_Noise
5. RBF_Abrupt_Blip_Noise
6. RBF_Gradual_Severe_Noise
7. HYPERPLANE_Gradual_Noise
8. RANDOMTREE_Gradual_Noise
```

### Batch 4 (6 datasets - Outros)
```
1. SINE_Abrupt_Simple
2. SINE_Gradual_Recurring
3. LED_Abrupt_Simple
4. WAVEFORM_Abrupt_Simple
5. WAVEFORM_Gradual_Simple
6. RANDOMTREE_Abrupt_Recurring
```

### Batch 5 (5 datasets - Reais)
```
1. Electricity
2. Shuttle
3. CovType
4. PokerHand
5. IntelLabSensors
```

### Batch 6 (6 datasets - Estacionarios 1)
```
1. SEA_Stationary
2. AGRAWAL_Stationary
3. RBF_Stationary
4. LED_Stationary
5. HYPERPLANE_Stationary
6. RANDOMTREE_Stationary
```

### Batch 7 (6 datasets - Estacionarios 2)
```
1. STAGGER_Stationary
2. WAVEFORM_Stationary
3. SINE_Stationary
4. AssetNegotiation_F2
5. AssetNegotiation_F3
6. AssetNegotiation_F4
```

---

## 3. Plano de Execucao

### Fase A: Executar ROSE em TODOS os datasets (Prioridade Alta)

ROSE ainda nao foi executado em nenhum dataset. Precisa ser executado em todos os 52 datasets.

**Tempo estimado:** ~5-10 minutos por dataset = ~4-8 horas total

### Fase B: Completar modelos faltantes no batch_6 e batch_7 (Prioridade Media)

Batches 6 e 7 so tem resultados do GBML. Precisam executar:
- HAT, ARF, SRP (River)
- ACDWM
- ERulesD2S

**Datasets:** 12 datasets
**Tempo estimado:** ~30-60 min por dataset = ~6-12 horas total

### Fase C: Completar datasets faltantes em batches 3-4 (Prioridade Baixa)

Alguns datasets especificos faltam resultados:
- SINE_Abrupt_Recurring_Noise (batch_3) - TODOS modelos
- SINE_Abrupt_Simple (batch_4) - TODOS modelos
- SINE_Gradual_Recurring (batch_4) - TODOS modelos

**Datasets:** 3 datasets
**Tempo estimado:** ~2-3 horas

---

## 4. Estrategia de Execucao no Colab

### 4.1 Notebook Unificado

Criar um unico notebook que:
1. Instala todas as dependencias (Java, River, ACDWM, ROSE JARs)
2. Permite selecionar quais batches/datasets executar
3. Permite selecionar quais modelos executar
4. Salva resultados incrementalmente no Drive
5. Pode ser retomado se interrompido

### 4.2 Estrutura do Notebook

```
1. Setup Ambiente
   - Montar Drive
   - Instalar Java
   - Instalar dependencias Python
   - Baixar JARs (ROSE, ERulesD2S)
   - Clonar ACDWM

2. Configuracao
   - Selecionar batches
   - Selecionar modelos
   - Definir timeout

3. Execucao por Batch
   - Loop por datasets
   - Para cada modelo:
     - Verificar se resultado ja existe
     - Se nao existe, executar
     - Salvar resultado

4. Consolidacao
   - Juntar todos os CSVs
   - Calcular estatisticas
   - Gerar tabela comparativa
```

### 4.3 Ordem de Execucao Recomendada

**Sessao 1 (4-6h):** ROSE em batch_1 a batch_4 (35 datasets)
**Sessao 2 (2-4h):** ROSE em batch_5 a batch_7 (17 datasets)
**Sessao 3 (6-8h):** Modelos faltantes batch_6 e batch_7 (12 datasets x 5 modelos)
**Sessao 4 (2-3h):** Datasets SINE faltantes (3 datasets x 6 modelos)

---

## 5. Arquivos a Criar

### 5.1 Notebook Principal
- `Execute_All_Comparative_Models.ipynb`
  - Notebook unificado para execucao de todos os modelos
  - Inclui ROSE, River (HAT/ARF/SRP), ACDWM, ERulesD2S

### 5.2 Scripts de Suporte
- `rose/rose_wrapper.py` - Wrapper para ROSE (ja criado)
- `rose/Test_ROSE_Colab.ipynb` - Notebook de validacao (ja criado)

### 5.3 Resultados Esperados
- `rose_results.csv` por dataset
- Consolidacao em `batch_X_all_models_with_rose.csv`

---

## 6. Metricas de Avaliacao

### Metrica Principal
- **G-mean** (Media geometrica de sensibilidade e especificidade)

### Metricas Secundarias
- Accuracy
- Kappa
- F1-weighted
- AUC (quando disponivel)

### Testes Estatisticos
- Friedman test (comparacao global)
- Wilcoxon signed-rank test (comparacoes pareadas com GBML)

---

## 7. Proximos Passos Imediatos

1. [x] Validar ROSE no Colab (CONCLUIDO - G-mean 90.54%)
2. [x] Corrigir parsing de resultados ROSE
3. [ ] Criar notebook `Execute_All_Comparative_Models.ipynb`
4. [ ] Executar ROSE nos batch_1-4 (Sessao 1)
5. [ ] Executar ROSE nos batch_5-7 (Sessao 2)
6. [ ] Executar modelos faltantes batch_6-7 (Sessao 3)
7. [ ] Consolidar todos os resultados
8. [ ] Gerar tabela final comparativa

---

## 8. Estrutura de Pastas dos Resultados

```
experiments_6chunks_phase2_gbml/
├── batch_1/
│   └── {dataset}/
│       └── run_1/
│           ├── chunk_metrics.json          (GBML)
│           ├── river_HAT_results.csv       (HAT)
│           ├── river_ARF_results.csv       (ARF)
│           ├── river_SRP_results.csv       (SRP)
│           ├── acdwm_results.csv           (ACDWM)
│           ├── erulesd2s_results.csv       (ERulesD2S)
│           └── rose_results.csv            (ROSE - A CRIAR)
├── batch_2/
│   └── ...
├── batch_3/
│   └── ...
└── batch_4/
    └── ...

experiments_6chunks_phase3_real/
├── batch_5/
│   └── ...
├── batch_6/
│   └── ...
└── batch_7/
    └── ...
```

---

**Criado por:** Claude Code
**Data:** 2025-11-25
**Status:** Plano aprovado - Aguardando execucao
