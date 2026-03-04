# Plano de Ação: Fase 3 - Datasets Estacionários e Reais

**Data:** 2025-11-21
**Objetivo:** Avaliar GBML e baselines em datasets estacionários e reais

---

## Contexto das Fases Anteriores

### Fase 1: experiments_6chunks_phase1_gbml
- **Objetivo:** Validação inicial
- **Datasets:** 2 datasets com drift simulation
- **Modelos:** GBML + modelos River
- **Configuração:** 6 chunks, 2 drifts
- **Status:** ✅ Concluída

### Fase 2: experiments_6chunks_phase2_gbml
- **Objetivo:** Avaliação completa em drift simulation
- **Datasets:** 32 datasets de drift simulation (4 batches)
- **Modelos:** GBML + ACDWM + ARF + SRP + HAT + ERulesD2S
- **Configuração:** 6 chunks, drifts variados
- **Status:** ✅ Concluída e paper gerado

### Fase 3: experiments_6chunks_phase3_real (NOVA)
- **Objetivo:** Avaliação em datasets estacionários e reais
- **Datasets:** Datasets reais (ELECTRICITY, SHUTTLE, etc.) + sintéticos estacionários
- **Modelos:** GBML + todos os baselines
- **Configuração:** 6 chunks, 1000 pontos cada, avaliar em 5
- **Status:** 🔄 Planejamento

---

## Mudanças Necessárias para Fase 3

### 1. Modo de Execução
**ANTES (Fase 2):**
```yaml
experiment_settings:
  run_mode: drift_simulation  # <-- Modo drift simulation
  drift_simulation_experiments:
    - SEA_Abrupt_Simple
    - AGRAWAL_Abrupt_Simple_Severe
    # ... outros datasets de drift
```

**DEPOIS (Fase 3):**
```yaml
experiment_settings:
  run_mode: standard  # <-- Modo standard
  standard_experiments:
    - Electricity
    - Shuttle
    - CovType
    - PokerHand
    - IntelLabSensors
    - SEA_Stationary
    - AGRAWAL_Stationary
    # ... outros datasets estacionários
```

### 2. Diretório de Resultados
**ANTES:** `experiments_6chunks_phase2_gbml/batch_X`
**DEPOIS:** `experiments_6chunks_phase3_real/batch_X`

### 3. Parâmetros de Dados (mantém igual)
```yaml
data_params:
  chunk_size: 1000      # ✅ Já correto
  num_chunks: 6         # ✅ Já correto
  max_instances: 24000  # ✅ 6 chunks x 1000 = 6000, mas esse é o máximo
```

**Nota:** No modo standard, serão usados 6 chunks (train-then-test), avaliando em 5.

---

## Datasets Disponíveis para Fase 3

### Datasets Reais (5 datasets)
1. **Electricity** - Predição de demanda elétrica (river.datasets.Elec2)
2. **Shuttle** - NASA Shuttle (CSV local)
3. **CovType** - Forest Cover Type (CSV local)
4. **PokerHand** - Poker Hand (CSV local)
5. **IntelLabSensors** - Intel Lab Sensors (CSV local)

### Datasets Sintéticos Estacionários (12 datasets)
1. **SEA_Stationary** - Sem drift
2. **AGRAWAL_Stationary** - Sem drift
3. **RBF_Stationary** - Sem drift (50 features, 4 classes)
4. **LED_Stationary** - Sem drift
5. **HYPERPLANE_Stationary** - Sem drift
6. **RANDOMTREE_Stationary** - Sem drift
7. **STAGGER_Stationary** - Sem drift
8. **WAVEFORM_Stationary** - Sem drift
9. **SINE_Stationary** - Sem drift
10. **AssetNegotiation_F2** - Função 1
11. **AssetNegotiation_F3** - Função 2
12. **AssetNegotiation_F4** - Função 3

**Total: 17 datasets estacionários/reais**

---

## Organização em Batches

### Opção 1: Por Tipo de Dataset (RECOMENDADO)

**Batch 5: Datasets Reais (5 datasets)**
- Electricity
- Shuttle
- CovType
- PokerHand
- IntelLabSensors

**Batch 6: Datasets Sintéticos Estacionários - Parte 1 (6 datasets)**
- SEA_Stationary
- AGRAWAL_Stationary
- RBF_Stationary
- LED_Stationary
- HYPERPLANE_Stationary
- RANDOMTREE_Stationary

**Batch 7: Datasets Sintéticos Estacionários - Parte 2 (6 datasets)**
- STAGGER_Stationary
- WAVEFORM_Stationary
- SINE_Stationary
- AssetNegotiation_F2
- AssetNegotiation_F3
- AssetNegotiation_F4

### Opção 2: Misturado por Complexidade

**Batch 5: Simples (5 datasets)**
- SEA_Stationary
- STAGGER_Stationary
- Electricity
- AGRAWAL_Stationary
- Shuttle

**Batch 6: Moderados (6 datasets)**
- HYPERPLANE_Stationary
- RANDOMTREE_Stationary
- RBF_Stationary
- WAVEFORM_Stationary
- CovType
- AssetNegotiation_F2

**Batch 7: Complexos (6 datasets)**
- LED_Stationary
- SINE_Stationary
- PokerHand
- IntelLabSensors
- AssetNegotiation_F3
- AssetNegotiation_F4

---

## Arquivos YAML a Serem Criados

### Estrutura dos Arquivos

```
C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid\configs\
├── config_batch_5.yaml  # Batch 5 - Datasets Reais
├── config_batch_6.yaml  # Batch 6 - Estacionários Parte 1
└── config_batch_7.yaml  # Batch 7 - Estacionários Parte 2
```

### Template Base (config_batch_5.yaml)

```yaml
experiment_settings:
  run_mode: standard  # <-- MUDANÇA PRINCIPAL
  standard_experiments:  # <-- LISTA DE DATASETS ESTACIONÁRIOS
    # BATCH 5: Datasets Reais (5 datasets)
    - Electricity
    - Shuttle
    - CovType
    - PokerHand
    - IntelLabSensors
  num_runs: 1
  base_results_dir: /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/experiments_6chunks_phase3_real/batch_5
  logging_level: INFO
  evaluation_period: 1000

data_params:
  chunk_size: 1000
  num_chunks: 6
  max_instances: 24000

# Manter TODOS os parâmetros GA, memory, fitness, parallelism iguais à Fase 2
ga_params:
  population_size: 120
  max_generations: 200
  # ... (copiar todos os parâmetros)

memory_params:
  # ... (copiar todos)

fitness_params:
  # ... (copiar todos)

parallelism:
  enabled: true
  num_workers: null

# Manter seção drift_analysis completa com TODOS os datasets
drift_analysis:
  severity_samples: 20000
  heatmap_save_directory: experiments_6chunks_phase3_real/test_real_results_heatmaps/concept_heatmaps
  datasets:
    ELECTRICITY:
      class: river.datasets.Elec2
    SHUTTLE:
      loader: local_csv
      source_path: datasets/processed/shuttle_processed.csv
      target_column: class
    # ... (copiar todos)

# Manter seção experimental_streams completa
experimental_streams:
  Electricity:
    dataset_type: ELECTRICITY
  Shuttle:
    dataset_type: SHUTTLE
  CovType:
    dataset_type: COVERTYPE
  # ... (copiar todos)
```

---

## Passo a Passo de Implementação

### Passo 1: Criar config_batch_5.yaml (Datasets Reais)
```bash
# Copiar config_batch_1.yaml como base
cp configs/config_batch_1.yaml configs/config_batch_5.yaml
```

**Modificações necessárias:**
1. Mudar `run_mode: drift_simulation` para `run_mode: standard`
2. Substituir `drift_simulation_experiments:` por `standard_experiments:`
3. Listar apenas os 5 datasets reais:
   - Electricity
   - Shuttle
   - CovType
   - PokerHand
   - IntelLabSensors
4. Atualizar `base_results_dir` para `experiments_6chunks_phase3_real/batch_5`
5. Atualizar `heatmap_save_directory` para `experiments_6chunks_phase3_real/...`

### Passo 2: Criar config_batch_6.yaml (Estacionários Parte 1)
```bash
cp configs/config_batch_5.yaml configs/config_batch_6.yaml
```

**Modificações necessárias:**
1. Listar 6 datasets sintéticos estacionários:
   - SEA_Stationary
   - AGRAWAL_Stationary
   - RBF_Stationary
   - LED_Stationary
   - HYPERPLANE_Stationary
   - RANDOMTREE_Stationary
2. Atualizar `base_results_dir` para `batch_6`

### Passo 3: Criar config_batch_7.yaml (Estacionários Parte 2)
```bash
cp configs/config_batch_6.yaml configs/config_batch_7.yaml
```

**Modificações necessárias:**
1. Listar 6 datasets sintéticos estacionários:
   - STAGGER_Stationary
   - WAVEFORM_Stationary
   - SINE_Stationary
   - AssetNegotiation_F2
   - AssetNegotiation_F3
   - AssetNegotiation_F4
2. Atualizar `base_results_dir` para `batch_7`

### Passo 4: Verificar Disponibilidade dos Datasets

**Datasets River (disponíveis automaticamente):**
- ✅ Electricity (river.datasets.Elec2)

**Datasets CSV Locais (verificar se existem):**
- ❓ shuttle_processed.csv
- ❓ covertype_processed.csv
- ❓ poker_processed.csv
- ❓ intellabsensors_processed.csv

**Comando de verificação:**
```bash
cd C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid
ls datasets/processed/
```

### Passo 5: Executar Batch 5 (Teste)
```bash
cd C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid

# Executar com GBML apenas (teste inicial)
python main.py --config configs/config_batch_5.yaml
```

### Passo 6: Executar Todos os Modelos (Batch 5)
```bash
# Executar notebook ou script comparativo
# Similar ao que foi feito na Fase 2
python run_comparative_on_existing_chunks.py --config configs/config_batch_5.yaml
```

### Passo 7: Executar Batches 6 e 7
```bash
# Batch 6
python run_comparative_on_existing_chunks.py --config configs/config_batch_6.yaml

# Batch 7
python run_comparative_on_existing_chunks.py --config configs/config_batch_7.yaml
```

---

## Parâmetros Importantes

### Modo Standard vs Drift Simulation

**Standard Mode:**
- Não injeta drifts artificiais
- Usa os dados como estão (estacionários ou com drift natural)
- Train-then-test: treina no chunk N, testa no chunk N+1
- 6 chunks = 5 avaliações (chunk 0 treina, avalia em 1-5)

**Drift Simulation Mode:**
- Injeta drifts artificiais conforme configurado
- Controla posição e tipo de drift
- Usado na Fase 2

### Train-Then-Test com 6 Chunks

```
Chunk 0: [0-999]     -> TREINO inicial
Chunk 1: [1000-1999] -> TESTE 1
Chunk 2: [2000-2999] -> TESTE 2
Chunk 3: [3000-3999] -> TESTE 3
Chunk 4: [4000-4999] -> TESTE 4
Chunk 5: [5000-5999] -> TESTE 5
```

**Total:** 5 avaliações por dataset

---

## Expectativas de Resultados

### Datasets Reais
- **Electricity:** Pode ter drift natural (variação temporal)
- **Shuttle, CovType, PokerHand:** Geralmente estacionários
- **IntelLabSensors:** Pode ter variação temporal

### Datasets Sintéticos Estacionários
- **Performance esperada:** Mais alta que em datasets com drift
- **Razão:** Sem drift para se adaptar, apenas aprender conceito fixo
- **GBML vs Baselines:** Comparação mais justa sem drift

### Comparação Fase 2 vs Fase 3

**Fase 2 (Drift):**
- GBML: 0.7775 (com adaptação a drift)
- Baselines: Variados

**Fase 3 (Estacionário):**
- Expectativa: Performance geral mais alta
- Questão chave: GBML mantém vantagem sem drift?

---

## Checklist de Preparação

### Antes de Começar
- [ ] Verificar se datasets CSV existem em `datasets/processed/`
- [ ] Se não existirem, executar scripts de pré-processamento
- [ ] Confirmar estrutura de diretórios
- [ ] Testar com 1 dataset primeiro (Electricity)

### Criação dos Configs
- [ ] Criar config_batch_5.yaml (Reais)
- [ ] Criar config_batch_6.yaml (Estacionários Parte 1)
- [ ] Criar config_batch_7.yaml (Estacionários Parte 2)
- [ ] Validar sintaxe YAML dos 3 arquivos

### Execução
- [ ] Executar Batch 5 com GBML (teste)
- [ ] Executar Batch 5 com todos os modelos
- [ ] Executar Batch 6 com todos os modelos
- [ ] Executar Batch 7 com todos os modelos

### Pós-Processamento
- [ ] Consolidar resultados dos 3 batches
- [ ] Gerar tabelas comparativas
- [ ] Gerar plots
- [ ] Análise estatística
- [ ] Documentar resultados

---

## Estrutura de Diretórios Esperada

```
DSL-AG-hybrid/
├── configs/
│   ├── config_batch_1.yaml  # Fase 2 - Batch 1 (Abrupt)
│   ├── config_batch_2.yaml  # Fase 2 - Batch 2 (Gradual)
│   ├── config_batch_3.yaml  # Fase 2 - Batch 3 (Noise)
│   ├── config_batch_4.yaml  # Fase 2 - Batch 4 (Complementary)
│   ├── config_batch_5.yaml  # Fase 3 - Batch 5 (Reais) ⭐ NOVO
│   ├── config_batch_6.yaml  # Fase 3 - Batch 6 (Estacionários 1) ⭐ NOVO
│   └── config_batch_7.yaml  # Fase 3 - Batch 7 (Estacionários 2) ⭐ NOVO
│
├── experiments_6chunks_phase2_gbml/  # Fase 2 - Drift Simulation
│   ├── batch_1/
│   ├── batch_2/
│   ├── batch_3/
│   └── batch_4/
│
└── experiments_6chunks_phase3_real/  # Fase 3 - Estacionários/Reais ⭐ NOVO
    ├── batch_5/  # Reais
    ├── batch_6/  # Estacionários Parte 1
    └── batch_7/  # Estacionários Parte 2
```

---

## Tempo Estimado

**Por Batch (estimativa):**
- GBML: ~2-4 horas (depende da complexidade dos datasets)
- Cada baseline: ~1-2 horas
- Total por batch: ~8-12 horas com todos os modelos

**Total Fase 3:**
- 3 batches x 10 horas = **~30 horas de execução**
- Execução paralela no Colab pode reduzir para ~10-15 horas

---

## Próximos Passos Imediatos

1. ✅ **Entender e planejar** - FEITO (este documento)
2. 🔄 **Verificar datasets CSV** - Próximo
3. 🔄 **Criar config_batch_5.yaml** - Próximo
4. 🔄 **Testar com 1 dataset** - Depois
5. 🔄 **Executar Batch 5 completo** - Depois

---

## Observações Importantes

### Diferença Fundamental: Standard vs Drift Simulation

**No modo standard:**
- O GBML não ativa mecanismos específicos de adaptação a drift
- Funciona como um classificador evolutivo "normal"
- Ainda usa memória e GA, mas sem recuperação ativa

**Parâmetros que podem não ser usados em standard mode:**
- `enable_explicit_drift_adaptation` - pode ser ignorado
- `max_generations_recovery` - não aplicável
- Drift detection - não aplicável

**Vantagem:**
- Comparação mais justa com baselines que não têm adaptação específica a drift
- Avalia capacidade de aprendizado "puro" sem adaptação

---

## Questões para Decidir

1. **Organização dos batches:** Opção 1 (por tipo) ou Opção 2 (por complexidade)?
   - **Recomendação:** Opção 1 (por tipo) - mais claro e organizado

2. **Incluir SINE_Stationary?**
   - Na Fase 2, SINE teve problemas (falta de variação)
   - **Recomendação:** Incluir, mas monitorar

3. **Ordem de execução:**
   - Começar pelos reais (Batch 5) ou pelos sintéticos (Batch 6)?
   - **Recomendação:** Começar por Batch 5 (reais) - são mais importantes

4. **Executar no Colab ou localmente?**
   - **Recomendação:** Colab (mais rápido, GPUs disponíveis)

---

**Status:** PLANO COMPLETO - PRONTO PARA IMPLEMENTAÇÃO
**Próxima Ação:** Verificar datasets CSV e criar config_batch_5.yaml
