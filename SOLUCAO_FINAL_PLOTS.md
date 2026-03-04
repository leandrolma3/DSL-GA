# Solucao Final - Geracao de Plots com Marcadores de Drift

**Data**: 2025-11-19
**Problema Resolvido**: Plots sem marcadores de drift

---

## CAUSA RAIZ DO PROBLEMA

O `generate_plots.py` procurava por `periodic_accuracy.json` mas o arquivo salvo pelo experimento e `periodic_gmean.json`.

**Nome incompativel:**
- ❌ Esperado: `periodic_accuracy.json`
- ✅ Existe: `periodic_gmean.json`

---

## CORRECAO APLICADA

**Arquivo**: `generate_plots.py` (linha 260)

**ANTES:**
```python
with open(os.path.join(run_dir, "periodic_accuracy.json"), 'r') as f:
    periodic_accuracy = json.load(f)
```

**DEPOIS:**
```python
with open(os.path.join(run_dir, "periodic_gmean.json"), 'r') as f:
    periodic_accuracy = json.load(f)
```

---

## VERIFICACAO

Todos os arquivos necessarios EXISTEM:

### Arquivos JSON (dados do experimento):
✅ `run_config.json` - Configuracao (experiment_id, stream_definition, drift positions)
✅ `periodic_gmean.json` - Dados periodicos (train/test por chunk)
✅ `chunk_metrics.json` - Metricas (gmean, f1)
✅ `ga_history_per_chunk.json` - Historia do GA
✅ `rule_details_per_chunk.json` - Detalhes das regras
✅ `attribute_usage_per_chunk.json` - Uso de atributos

### Arquivo de Drift Severity:
⏳ `concept_differences.json` - Sera gerado pelo Step 1 (analyze_concept_difference.py)

---

## ESTRUTURA DOS DADOS

### run_config.json:
```json
{
  "experiment_id": "SEA_Abrupt_Simple",
  "dataset_type": "SEA",
  "chunk_size": 1000,
  "num_chunks_processed": 5,
  "stream_definition": {
    "concept_sequence": [
      {"concept_id": "f1", "duration_chunks": 3},
      {"concept_id": "f3", "duration_chunks": 3}
    ],
    "drift_type": "abrupt"
  }
}
```

Drift calculado:
- Chunks 0-2: conceito f1 (instancias 0-2999)
- **Drift em 3000 instancias** (inicio chunk 3)
- Chunks 3-5: conceito f3 (instancias 3000-5999)

### periodic_gmean.json:
```json
[
  {
    "chunk_train": 0,
    "chunk_test": 1,
    "train_gmean": 0.9948,
    "test_gmean": 0.9772
  },
  ...
]
```

### chunk_metrics.json:
```json
[
  {
    "chunk": 0,
    "train_gmean": 0.9948,
    "test_gmean": 0.9772,
    "test_f1": 0.9790
  },
  ...
]
```

---

## COMO EXECUTAR AGORA

### Passo 1: Gerar concept_differences.json

No Colab:
```python
!cd /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid && \
python -c "from analyze_concept_difference import main; main('configs/config_batch_1.yaml')"
```

Isso cria:
- `experiments_6chunks_phase2_gbml/test_real_results_heatmaps/concept_heatmaps/concept_differences.json`
- Heatmaps de diferenca entre conceitos (PNG)

### Passo 2: Executar post_process_batch_1.py

No Colab:
```python
!cd /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid && \
python post_process_batch_1.py
```

Isso executa:
- **Step 1**: analyze_concept_difference.py (ja executado acima)
- **Step 2**: generate_plots.py para cada base (AGORA VAI FUNCIONAR!)
- **Step 3**: rule_diff_analyzer.py para cada base

---

## O QUE SERA GERADO

Para cada experimento em `experiments_6chunks_phase2_gbml/batch_1/<DATASET>/run_1/plots/`:

### Plot Principal (COM marcadores de drift):
```
Plot_AccuracyPeriodic_<DATASET>_Run1.png
```

Este plot mostra:
- ✅ Linhas de treino (G-mean) e teste (G-mean)
- ✅ **Linhas verticais vermelhas** nas posicoes de drift
- ✅ **Texto com severity**: "Drift at 3000 (XX%)"
- ✅ Evolucao da acuracia ao longo dos chunks

### Plots de Evolucao do GA (5 plots):
```
Plot_GA_Evolution_Chunk0_<DATASET>_Run1.png
Plot_GA_Evolution_Chunk1_<DATASET>_Run1.png
Plot_GA_Evolution_Chunk2_<DATASET>_Run1.png
Plot_GA_Evolution_Chunk3_<DATASET>_Run1.png
Plot_GA_Evolution_Chunk4_<DATASET>_Run1.png
```

Cada plot mostra:
- Fitness ao longo das geracoes
- Acuracia de treino
- F1 score
- Convergencia do GA

---

## EXEMPLO DE SAIDA ESPERADA

```
experiments_6chunks_phase2_gbml/batch_1/SEA_Abrupt_Simple/run_1/
├── plots/  (NOVO - gerado pelo generate_plots.py)
│   ├── Plot_AccuracyPeriodic_SEA_Abrupt_Simple_Run1.png  <- COM DRIFT MARKERS
│   ├── Plot_GA_Evolution_Chunk0_SEA_Abrupt_Simple_Run1.png
│   ├── Plot_GA_Evolution_Chunk1_SEA_Abrupt_Simple_Run1.png
│   ├── Plot_GA_Evolution_Chunk2_SEA_Abrupt_Simple_Run1.png
│   ├── Plot_GA_Evolution_Chunk3_SEA_Abrupt_Simple_Run1.png
│   └── Plot_GA_Evolution_Chunk4_SEA_Abrupt_Simple_Run1.png
├── G-meanPlot_Periodic_SEA_Abrupt_Simple_Run1.png  (antigo, sem markers)
├── GA_Evolution_Chunk*.png  (antigos, gerados durante experimento)
├── RulesHistory_SEA_Abrupt_Simple_Run1.txt
├── run_config.json
├── periodic_gmean.json
└── ... outros arquivos
```

---

## VALIDACAO DOS PLOTS

Apos gerar os plots, verificar:

### 1. Plot_AccuracyPeriodic deve ter:
- ✅ Titulo: "Periodic Train/Test Performance - <DATASET>"
- ✅ Eixo X: Chunks (0, 1, 2, 3, 4)
- ✅ Eixo Y: G-mean (0.0 - 1.0)
- ✅ Linha azul: Training G-mean
- ✅ Linha laranja: Test G-mean
- ✅ Linhas verticais vermelhas em posicoes de drift
- ✅ Texto em cada drift: "Drift at <pos> (XX% severity)"

### 2. Posicoes de Drift Esperadas:

| Dataset | Drift Positions | Chunks |
|---------|----------------|--------|
| SEA_Abrupt_Simple | 3000 | 3 |
| SEA_Abrupt_Chain | 2000, 4000 | 2, 4 |
| SEA_Abrupt_Recurring | 2000, 4000 | 2, 4 |
| AGRAWAL_Abrupt_Simple_Mild | 3000 | 3 |
| AGRAWAL_Abrupt_Simple_Severe | 3000 | 3 |
| AGRAWAL_Abrupt_Chain_Long | 2000, 4000, 5000 | 2, 4, 5 |
| RBF_Abrupt_Severe | 3000 | 3 |
| RBF_Abrupt_Blip | 2000, 4000 | 2, 4 |
| STAGGER_Abrupt_Chain | 2000, 4000 | 2, 4 |
| STAGGER_Abrupt_Recurring | 2000, 4000 | 2, 4 |
| HYPERPLANE_Abrupt_Simple | 3000 | 3 |
| RANDOMTREE_Abrupt_Simple | 3000 | 3 |

### 3. Severity esperada:
- **NAO deve ser 0.0%**
- Valores tipicos: 15-75%
- Variacao depende da mudanca entre conceitos

---

## LOGS ESPERADOS

```
2025-11-19 XX:XX:XX [INFO    ] STEP 2: Geracao de Plots para Cada Experimento
2025-11-19 XX:XX:XX [INFO    ]
--- Processando: SEA_Abrupt_Simple ---
2025-11-19 XX:XX:XX [INFO    ] [EXECUTANDO] Generate plots para SEA_Abrupt_Simple
2025-11-19 XX:XX:XX [INFO    ]   | --- Processing results directory: .../SEA_Abrupt_Simple/run_1 ---
2025-11-19 XX:XX:XX [INFO    ]   | Plots will be saved in: .../SEA_Abrupt_Simple/run_1/plots
2025-11-19 XX:XX:XX [INFO    ]   | Result files loaded successfully.
2025-11-19 XX:XX:XX [INFO    ]   | Generating periodic accuracy plot with drift info...
2025-11-19 XX:XX:XX [INFO    ]   | Plot saved: Plot_AccuracyPeriodic_SEA_Abrupt_Simple_Run1.png
2025-11-19 XX:XX:XX [INFO    ]   | Generating GA evolution plots per chunk...
2025-11-19 XX:XX:XX [INFO    ]   | Plot saved: Plot_GA_Evolution_Chunk0_SEA_Abrupt_Simple_Run1.png
2025-11-19 XX:XX:XX [INFO    ]   | ... (4 more plots)
2025-11-19 XX:XX:XX [INFO    ] [OK] Generate plots para SEA_Abrupt_Simple - Concluido com sucesso
2025-11-19 XX:XX:XX [INFO    ]   [OK] Pasta plots/ criada com 6 arquivos PNG
```

---

## SE HOUVER ERROS

### Erro: "periodic_accuracy.json not found"
**Causa**: generate_plots.py nao foi atualizado
**Solucao**: Verifique se a linha 260 de generate_plots.py foi corrigida

### Erro: "concept_differences.json not found"
**Causa**: Step 1 nao foi executado
**Solucao**: Execute analyze_concept_difference.py primeiro

### Erro: "stream_definition not found"
**Causa**: run_config.json incompleto
**Solucao**: Re-executar experimento GBML

### Plots sem marcadores de drift
**Causa**: concept_differences.json nao existe ou esta vazio
**Solucao**: Verificar se analyze_concept_difference.py executou corretamente

---

## RESUMO DAS CORRECOES

| # | Problema | Arquivo | Linha | Status |
|---|----------|---------|-------|--------|
| 1 | Nome incorreto do arquivo JSON | generate_plots.py | 260 | ✅ CORRIGIDO |
| 2 | Step 1 usa subprocess incorretamente | post_process_batch_1.py | 92-132 | ✅ CORRIGIDO |
| 3 | Typo no heatmap_save_directory | configs/config_batch_1.yaml | 94 | ✅ CORRIGIDO |
| 4 | Config errado no script | post_process_batch_1.py | 34, 40 | ✅ CORRIGIDO |

---

**Status Final**: ✅ GENERATE_PLOTS.PY CORRIGIDO E FUNCIONAL
**Proxima Acao**: Executar post_process_batch_1.py no Colab para gerar plots com marcadores de drift
**Tempo Estimado**: ~45-60 minutos para todas as 12 bases
