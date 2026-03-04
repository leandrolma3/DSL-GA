# Documentação Completa - Fase 3: Datasets Reais e Estacionários

**Data de Criação:** 2025-11-21
**Status:** CONFIGS CRIADOS - PRONTO PARA EXECUÇÃO
**Objetivo:** Avaliar GBML e baselines em cenários sem drift artificial (modo standard)

---

## Índice

1. [Visão Geral da Fase 3](#visão-geral-da-fase-3)
2. [Diferenças: Fase 2 vs Fase 3](#diferenças-fase-2-vs-fase-3)
3. [Estrutura dos Batches](#estrutura-dos-batches)
4. [Detalhes dos Configs](#detalhes-dos-configs)
5. [Como Executar](#como-executar)
6. [Estrutura de Resultados](#estrutura-de-resultados)
7. [Expectativas e Hipóteses](#expectativas-e-hipóteses)
8. [Checklist de Execução](#checklist-de-execução)
9. [Troubleshooting](#troubleshooting)
10. [Pós-Processamento](#pós-processamento)

---

## Visão Geral da Fase 3

### Contexto
A **Fase 2** focou em avaliar a capacidade de adaptação dos modelos a **concept drifts artificialmente injetados**. Foram testados 32 datasets com drifts simulados (abrupt, gradual, noise, multiclass).

A **Fase 3** muda o foco para avaliar o **desempenho puro** dos modelos em cenários **sem drift artificial**, usando:
- **5 datasets reais** (que podem ter drift natural ou serem estacionários)
- **12 datasets sintéticos estacionários** (sem drift)

### Objetivos da Fase 3
1. **Avaliar desempenho em cenários estacionários:** Como os modelos se comportam quando não há drift?
2. **Comparar com Fase 2:** A vantagem de GBML é mantida sem drift?
3. **Validar robustez:** GBML continua robusto em cenários diferentes?
4. **Identificar trade-offs:** Modelos adaptativos têm overhead desnecessário sem drift?

### Modo de Execução
- **run_mode: standard** (não `drift_simulation`)
- **Train-then-test:** Treino no chunk 0, avaliação nos chunks 1-5
- **Sem injeção de drift:** Dados usados como estão
- **Sem recuperação ativa:** Mecanismos de recovery não são acionados artificialmente

---

## Diferenças: Fase 2 vs Fase 3

| Aspecto | Fase 2 (Drift Simulation) | Fase 3 (Standard/Real) |
|---------|---------------------------|------------------------|
| **run_mode** | `drift_simulation` | `standard` |
| **Experimentos** | `drift_simulation_experiments` | `standard_experiments` |
| **Datasets** | 32 sintéticos com drift injetado | 17 reais/estacionários |
| **Drift** | Artificial (posições controladas) | Natural ou ausente |
| **Recovery** | Acionada em posições específicas | Não há acionamento artificial |
| **Objetivo** | Avaliar adaptação a drift | Avaliar aprendizado puro |
| **Diretório** | `experiments_6chunks_phase2_gbml` | `experiments_6chunks_phase3_real` |
| **Batches** | 4 batches (1-4) | 3 batches (5-7) |
| **Total Datasets** | 32 | 17 |

### Parâmetros Mantidos (Iguais na Fase 2 e 3)
- **Chunks:** 6 chunks de 1000 pontos
- **GA params:** Population=120, Generations=200, elitism=0.1, etc.
- **Memory params:** max_memory_size=20, pruning ativo, etc.
- **Fitness params:** Todos os coeficientes mantidos
- **Paralelismo:** Habilitado

---

## Estrutura dos Batches

### Batch 5: Datasets Reais (5 datasets)
**Arquivo:** `configs/config_batch_5.yaml` (3.4 KB)

| Dataset | Tipo | Origem | Características |
|---------|------|--------|----------------|
| **Electricity** | Real | River (Elec2) | 45,312 instâncias, 8 features, 2 classes |
| **Shuttle** | Real | CSV local | ~58,000 instâncias, 9 features, 7 classes |
| **CovType** | Real | CSV local | 581,012 instâncias, 54 features, 7 classes |
| **PokerHand** | Real | CSV local | 1,025,010 instâncias, 10 features, 10 classes |
| **IntelLabSensors** | Real | CSV local | ~2.3M instâncias, 4 features, binárias |

**Diretório de resultados:** `experiments_6chunks_phase3_real/batch_5/`

---

### Batch 6: Datasets Sintéticos Estacionários - Parte 1 (6 datasets)
**Arquivo:** `configs/config_batch_6.yaml` (3.5 KB)

| Dataset | Generator | Classes | Características |
|---------|-----------|---------|----------------|
| **SEA_Stationary** | SEAGenerator | 2 | 3 features, threshold classification |
| **AGRAWAL_Stationary** | AGRAWALGenerator | 2 | 9 features, função empréstimo |
| **RBF_Stationary** | RandomRBFGenerator | 5 | Features contínuas, centróides |
| **LED_Stationary** | LEDGenerator | 10 | 24 features binárias, 7-segment display |
| **HYPERPLANE_Stationary** | HyperplaneGenerator | 2 | Hiperplano n-dimensional |
| **RANDOMTREE_Stationary** | RandomTreeGenerator | 5 | Árvore aleatória fixa |

**Diretório de resultados:** `experiments_6chunks_phase3_real/batch_6/`

---

### Batch 7: Datasets Sintéticos Estacionários - Parte 2 (6 datasets)
**Arquivo:** `configs/config_batch_7.yaml` (3.7 KB)

| Dataset | Generator | Classes | Características |
|---------|-----------|---------|----------------|
| **STAGGER_Stationary** | STAGGERGenerator | 2 | 3 features categóricas |
| **WAVEFORM_Stationary** | WaveformGenerator | 3 | 21 features, waveforms |
| **SINE_Stationary** | SineGenerator | 2 | 2 features, função seno |
| **AssetNegotiation_F2** | AssetNegotiation (f=2) | 2 | 4 features, negociação |
| **AssetNegotiation_F3** | AssetNegotiation (f=3) | 2 | 4 features, negociação |
| **AssetNegotiation_F4** | AssetNegotiation (f=4) | 2 | 4 features, negociação |

**Diretório de resultados:** `experiments_6chunks_phase3_real/batch_7/`

---

## Detalhes dos Configs

### Estrutura Geral dos Configs (Batches 5, 6, 7)

```yaml
experiment_settings:
  run_mode: standard                    # MUDANÇA PRINCIPAL
  standard_experiments:                  # LISTA DE DATASETS
    - Dataset1
    - Dataset2
    # ...
  num_runs: 1
  base_results_dir: .../experiments_6chunks_phase3_real/batch_X
  logging_level: INFO
  evaluation_period: 1000

data_params:
  chunk_size: 1000                       # 1000 pontos por chunk
  num_chunks: 6                          # 6 chunks total
  max_instances: 24000                   # Máximo

ga_params:
  population_size: 120
  max_generations: 200
  # ... (todos os outros parâmetros mantidos da Fase 2)

memory_params:
  max_memory_size: 20
  # ... (todos os parâmetros mantidos)

fitness_params:
  class_coverage_coefficient: 0.2
  # ... (todos os parâmetros mantidos)

parallelism:
  enabled: true
  num_workers: null

drift_analysis:
  severity_samples: 20000
  datasets:
    DATASET_NAME:
      # Configuração específica do dataset

experimental_streams:
  Dataset_Name:
    dataset_type: DATASET_TYPE
```

### Diferença para Configs da Fase 2

**Fase 2 (drift_simulation):**
```yaml
experiment_settings:
  run_mode: drift_simulation
  drift_simulation_experiments:
    - Dataset_Abrupt_Simple:
        drift_positions: [2000]
        drift_types: [abrupt]
    - Dataset_Gradual_Simple:
        drift_positions: [2000]
        drift_types: [gradual]
        drift_widths: [1000]
```

**Fase 3 (standard):**
```yaml
experiment_settings:
  run_mode: standard
  standard_experiments:
    - Dataset1
    - Dataset2
```

---

## Como Executar

### Pré-requisitos
1. **Ambiente Python configurado**
2. **Dependências instaladas:** river, scikit-learn, pandas, numpy, etc.
3. **Datasets CSV disponíveis:** `datasets/processed/*.csv`
4. **Espaço em disco:** ~10 GB para resultados

### Passo 1: Teste com Dataset Único (Electricity)

**Objetivo:** Validar que o modo standard funciona corretamente.

#### 1.1. Modificar config_batch_5.yaml temporariamente
```yaml
standard_experiments:
  - Electricity  # Apenas este para teste
```

#### 1.2. Executar
```bash
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"
python main.py --config configs/config_batch_5.yaml
```

#### 1.3. O que observar
- ✅ Carregamento do dataset Electricity
- ✅ Criação de 6 chunks
- ✅ Treino no chunk 0
- ✅ 5 avaliações (chunks 1-5)
- ✅ Métricas salvas em `experiments_6chunks_phase3_real/batch_5/Electricity/run_1/`
- ⚠️ Verificar se não há erros de "drift detection" (pode ter se drift natural existir)

**Tempo estimado:** 30-60 minutos

---

### Passo 2: Executar Batch 5 Completo (GBML)

Restaurar a lista completa no config:
```yaml
standard_experiments:
  - Electricity
  - Shuttle
  - CovType
  - PokerHand
  - IntelLabSensors
```

Executar:
```bash
python main.py --config configs/config_batch_5.yaml
```

**Tempo estimado:** 2-4 horas

---

### Passo 3: Executar Batch 6 (GBML)

```bash
python main.py --config configs/config_batch_6.yaml
```

**Tempo estimado:** 2-3 horas

---

### Passo 4: Executar Batch 7 (GBML)

```bash
python main.py --config configs/config_batch_7.yaml
```

**Tempo estimado:** 2-3 horas

---

### Passo 5: Executar Baselines em Todos os Batches

**Opção A: Script Comparativo (Recomendado)**
```bash
python run_comparative_on_existing_chunks.py --config configs/config_batch_5.yaml
python run_comparative_on_existing_chunks.py --config configs/config_batch_6.yaml
python run_comparative_on_existing_chunks.py --config configs/config_batch_7.yaml
```

**Opção B: Modelos Individuais**
```bash
# Para cada batch (5, 6, 7):
python baseline_acdwm.py --config configs/config_batch_X.yaml
python baseline_river.py --config configs/config_batch_X.yaml --model ARF
python baseline_river.py --config configs/config_batch_X.yaml --model SRP
python baseline_river.py --config configs/config_batch_X.yaml --model HAT
python run_erulesd2s_only.py --config configs/config_batch_X.yaml
```

**Tempo estimado total (todos modelos, 3 batches):** 24-36 horas

---

## Estrutura de Resultados

### Diretórios Esperados

```
experiments_6chunks_phase3_real/
├── batch_5/
│   ├── Electricity/
│   │   └── run_1/
│   │       ├── chunk_metrics.json
│   │       ├── final_metrics.json
│   │       ├── best_population.json
│   │       └── execution_log.txt
│   ├── Shuttle/
│   ├── CovType/
│   ├── PokerHand/
│   └── IntelLabSensors/
│
├── batch_6/
│   ├── SEA_Stationary/
│   ├── AGRAWAL_Stationary/
│   ├── RBF_Stationary/
│   ├── LED_Stationary/
│   ├── HYPERPLANE_Stationary/
│   └── RANDOMTREE_Stationary/
│
└── batch_7/
    ├── STAGGER_Stationary/
    ├── WAVEFORM_Stationary/
    ├── SINE_Stationary/
    ├── AssetNegotiation_F2/
    ├── AssetNegotiation_F3/
    └── AssetNegotiation_F4/
```

### Arquivos Gerados por Dataset

**chunk_metrics.json**
```json
[
  {
    "chunk": 1,
    "test_gmean": 0.85,
    "test_accuracy": 0.87,
    "train_gmean": 0.90,
    "per_class_recall": [0.83, 0.87]
  },
  // ... chunks 2-5
]
```

**final_metrics.json**
```json
{
  "mean_test_gmean": 0.8234,
  "mean_test_accuracy": 0.8456,
  "std_test_gmean": 0.0342,
  "total_chunks_evaluated": 5,
  "dataset": "Electricity",
  "run": 1
}
```

---

## Expectativas e Hipóteses

### Performance Esperada

**Fase 2 (Com Drift Artificial):**
- GBML: **0.7775** (1º lugar)
- ARF: 0.7240
- SRP: 0.7114
- ACDWM: 0.6998 (com zeros de falhas)
- HAT: 0.6262
- ERulesD2S: 0.5511

**Fase 3 (Sem Drift - Expectativa):**
- **Hipótese:** Todos os modelos devem ter performance **maior** que na Fase 2
- **Razão:** Sem drift para se adaptar, cenários mais fáceis
- **GBML estimado:** 0.85-0.90
- **Baselines:** Também devem melhorar

### Questões de Pesquisa

#### 1. GBML mantém vantagem sem drift?
- **Fase 2:** GBML melhor com drift artificial
- **Fase 3:** GBML melhor sem drift?
- **Possibilidade:** Modelos ensemble podem se destacar em cenários estacionários

#### 2. Qual modelo é melhor em cenários estacionários?
- ARF e SRP (ensemble) podem ter vantagem
- GBML pode ter overhead de GA desnecessário
- HAT pode sofrer menos sem drift

#### 3. Datasets reais têm drift natural?
- **Electricity:** Provavelmente sim (variação temporal de preços)
- **Shuttle, CovType, PokerHand:** Provavelmente não (estacionários)
- **IntelLabSensors:** Possível drift sazonal
- Performance vai revelar

#### 4. ACDWM vai falhar novamente?
- **Fase 2:** Falhou em 4 multiclass (LED, WAVEFORM)
- **Fase 3:** Tem LED_Stationary, WAVEFORM_Stationary, e datasets multiclass reais
- **Expectativa:** Pode falhar novamente em multiclass

---

## Checklist de Execução

### Preparação
- [X] Plano criado (PLANO_FASE3_DATASETS_REAIS.md)
- [X] Datasets CSV verificados
- [X] config_batch_5.yaml criado
- [X] config_batch_6.yaml criado
- [X] config_batch_7.yaml criado
- [X] Documentação consolidada criada
- [ ] Testar com Electricity apenas

### Execução GBML - Batch 5
- [ ] Executar teste Electricity
- [ ] Verificar logs
- [ ] Executar Batch 5 completo
- [ ] Confirmar 5 datasets processados
- [ ] Verificar chunk_metrics.json salvos

### Execução GBML - Batch 6
- [ ] Executar Batch 6 completo
- [ ] Confirmar 6 datasets processados
- [ ] Verificar resultados

### Execução GBML - Batch 7
- [ ] Executar Batch 7 completo
- [ ] Confirmar 6 datasets processados
- [ ] Verificar resultados

### Execução Baselines - Todos os Batches
- [ ] ACDWM executado (batches 5, 6, 7)
- [ ] ARF executado (batches 5, 6, 7)
- [ ] SRP executado (batches 5, 6, 7)
- [ ] HAT executado (batches 5, 6, 7)
- [ ] ERulesD2S executado (batches 5, 6, 7)

### Verificação de Resultados
- [ ] Total de 17 diretórios de datasets criados
- [ ] Cada dataset tem chunk_metrics.json
- [ ] Cada dataset tem final_metrics.json
- [ ] Logs completos salvos

### Pós-Processamento
- [ ] Consolidar resultados de todos os modelos
- [ ] Gerar CSV consolidado (similar a Fase 2)
- [ ] Calcular médias por modelo
- [ ] Executar testes estatísticos
- [ ] Gerar plots comparativos
- [ ] Comparar Fase 2 vs Fase 3

---

## Troubleshooting

### Problema 1: Datasets CSV Não Encontrados
**Sintoma:** `FileNotFoundError: datasets/processed/shuttle_processed.csv`

**Solução:**
```bash
# Verificar datasets
ls -lh datasets/processed/*.csv

# Se não existirem, executar pré-processamento
python preprocess_datasets.py
```

---

### Problema 2: Modo Standard Não Reconhecido
**Sintoma:** `ValueError: Invalid run_mode: standard`

**Solução:**
1. Verificar se o código suporta modo standard
2. Procurar em `main.py` por `if run_mode == "standard"`
3. Se não existir, pode ser necessário adicionar suporte

**Verificação:**
```bash
grep -n "run_mode.*standard" main.py
```

---

### Problema 3: Memory Error em Datasets Grandes
**Sintoma:** `MemoryError` ao carregar CovType ou PokerHand

**Solução:**
1. Reduzir `max_instances` no config:
```yaml
data_params:
  max_instances: 6000  # Reduzido de 24000
```

2. Ou processar dataset em partes separadas

---

### Problema 4: ACDWM Falha em Multiclass
**Sintoma:** ACDWM gera erro em LED_Stationary, WAVEFORM_Stationary

**Solução:**
1. **Esperado:** ACDWM falha em multiclass
2. **Ação:** Documentar falha
3. **Análise:** Incluir G-mean=0.0 na consolidação (como na Fase 2)

---

### Problema 5: Execução Muito Lenta
**Sintoma:** Batch 5 levando >8 horas

**Solução:**
1. Verificar paralelismo:
```yaml
parallelism:
  enabled: true
  num_workers: null  # Usa todos os cores
```

2. Reduzir population_size temporariamente:
```yaml
ga_params:
  population_size: 60  # Reduzido de 120
```

3. Executar datasets individualmente:
```yaml
standard_experiments:
  - Electricity  # Um por vez
```

---

## Pós-Processamento

### Script de Consolidação

Criar script similar ao usado na Fase 2:

**consolidate_phase3_results.py**
```python
import os
import json
import pandas as pd

def consolidate_phase3():
    base_dir = "experiments_6chunks_phase3_real"
    batches = ["batch_5", "batch_6", "batch_7"]

    all_results = []

    for batch in batches:
        batch_dir = os.path.join(base_dir, batch)

        for dataset in os.listdir(batch_dir):
            dataset_dir = os.path.join(batch_dir, dataset, "run_1")

            # Ler chunk_metrics.json
            chunk_file = os.path.join(dataset_dir, "chunk_metrics.json")
            if os.path.exists(chunk_file):
                with open(chunk_file) as f:
                    chunks = json.load(f)

                for chunk in chunks:
                    all_results.append({
                        'batch': batch,
                        'dataset': dataset,
                        'model': 'GBML',
                        'chunk': chunk['chunk'],
                        'test_gmean': chunk['test_gmean'],
                        'test_accuracy': chunk['test_accuracy']
                    })

    df = pd.DataFrame(all_results)
    df.to_csv(f"{base_dir}/phase3_consolidated.csv", index=False)
    print(f"Consolidado: {len(df)} registros")

if __name__ == "__main__":
    consolidate_phase3()
```

---

### Análise Estatística

Executar testes similares à Fase 2:

```python
from scipy import stats

# Ler dados consolidados
df = pd.read_csv("experiments_6chunks_phase3_real/phase3_consolidated.csv")

# Calcular médias por modelo
means = df.groupby(['model', 'dataset'])['test_gmean'].mean()

# Friedman test
models = df['model'].unique()
datasets = df['dataset'].unique()

data_by_model = []
for model in models:
    model_means = []
    for dataset in datasets:
        mean = df[(df['model']==model) & (df['dataset']==dataset)]['test_gmean'].mean()
        model_means.append(mean)
    data_by_model.append(model_means)

statistic, p_value = stats.friedmanchisquare(*data_by_model)
print(f"Friedman test: statistic={statistic}, p-value={p_value}")
```

---

### Comparação Fase 2 vs Fase 3

**Criar tabela comparativa:**

| Modelo | Fase 2 (32 datasets com drift) | Fase 3 (17 datasets sem drift) | Diferença |
|--------|--------------------------------|-------------------------------|-----------|
| GBML | 0.7775 | ? | ? |
| ARF | 0.7240 | ? | ? |
| SRP | 0.7114 | ? | ? |
| ACDWM | 0.6998 | ? | ? |
| HAT | 0.6262 | ? | ? |
| ERulesD2S | 0.5511 | ? | ? |

**Expectativa:** Todos devem aumentar na Fase 3 (sem drift = mais fácil)

---

## Resumo Executivo

### O Que Temos
✅ **3 configs criados:** batch_5.yaml, batch_6.yaml, batch_7.yaml
✅ **17 datasets planejados:** 5 reais + 12 sintéticos estacionários
✅ **Documentação completa:** Este documento
✅ **Plano de execução:** Passos detalhados
✅ **Pronto para rodar:** Apenas executar comandos

### Próxima Ação Imediata
**Testar com Electricity:**
1. Modificar config_batch_5.yaml (apenas Electricity)
2. Executar `python main.py --config configs/config_batch_5.yaml`
3. Verificar resultados
4. Se OK, executar batches completos

### Timeline Estimado
- **Teste Electricity:** 30-60 min
- **Batch 5 GBML completo:** 2-4 horas
- **Batch 6 GBML completo:** 2-3 horas
- **Batch 7 GBML completo:** 2-3 horas
- **Todos modelos (3 batches):** 24-36 horas
- **Pós-processamento:** 2-4 horas
- **Total Fase 3:** 2-3 dias de trabalho

---

**Status Atual:** CONFIGS CRIADOS - PRONTO PARA INICIAR TESTES
**Próximo Passo:** Executar teste com Electricity
**Documentação:** COMPLETA

---

## Referências

- **Plano Fase 3:** `PLANO_FASE3_DATASETS_REAIS.md`
- **Batch 5 Pronto:** `FASE3_BATCH5_PRONTO.md`
- **Configs:** `configs/config_batch_5.yaml`, `config_batch_6.yaml`, `config_batch_7.yaml`
- **Conexão paperLu:** `C:\Users\Leandro Almeida\Downloads\paperLu\CONEXAO_COM_DSL_AG_HYBRID.md`

---

**Criado por:** Claude Code
**Data:** 2025-11-21
**Versão:** 1.0
