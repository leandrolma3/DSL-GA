# PLANO - EXPERIMENTO COMPARATIVO GBML VS RIVER

**Data:** 2025-11-05
**Objetivo:** Comparar GBML vs modelos River em 3 bases de drift simulation
**Restricao:** < 18h tempo total (Google Colab)
**Codigo:** compare_gbml_vs_river.py (JA EXISTE)

---

## PROBLEMA: 3 DATASETS X 5 CHUNKS EXCEDE 18H

### Calculo de Tempo

**Tempo GBML:** 90min/chunk (com Layer 1 validado)
**Tempo River:** ~5min/chunk/modelo (3 modelos: HAT, ARF, SRP)

| Cenario | Datasets | Chunks/dataset | Total chunks | GBML | River | TOTAL | Status |
|---------|----------|----------------|--------------|------|-------|-------|--------|
| 1 | 3 | 5 | 15 | 22.5h | 3.8h | **26.2h** | **EXCEDE** |
| 2 | 3 | 3 | 9 | 13.5h | 2.2h | **15.8h** | OK |
| 3 | 2 | 5 | 10 | 15.0h | 2.5h | **17.5h** | OK |
| 4 | 3 | 4 | 12 | 18.0h | 3.0h | **21.0h** | EXCEDE |

---

## OPCOES VIAVEIS

### OPCAO A: 3 Datasets x 3 Chunks (RECOMENDADA)

**Tempo:** 15.8h (< 18h com margem de 2.2h)

**Datasets propostos:**
1. **RBF_Abrupt_Severe** (c1 → c2_severe)
2. **RBF_Abrupt_Moderate** (c1 → c3_moderate)
3. **RBF_Gradual_Severe** (c1 → c2_severe gradual)

**Chunks:** 3 por dataset

**Vantagem:**
- 3 datasets diferentes (drift types variados)
- Margem de segurança (2.2h)
- Ainda permite comparacao solida

**Desvantagem:**
- Menos chunks que o ideal (3 vs 5)

---

### OPCAO B: 2 Datasets x 5 Chunks

**Tempo:** 17.5h (< 18h com margem de 0.5h)

**Datasets propostos:**
1. **RBF_Abrupt_Severe** (c1 → c2_severe)
2. **RBF_Gradual_Severe** (c1 → c2_severe gradual)

**Chunks:** 5 por dataset

**Vantagem:**
- 5 chunks (mais dados)
- Contraste abrupt vs gradual

**Desvantagem:**
- Apenas 2 datasets
- Margem apertada (30min)

---

### OPCAO C: 3 Datasets x 5 Chunks com Config Reduzida

**Reducao:** population: 100 → 70, generations: 200 → 150

**Tempo estimado GBML:** ~65min/chunk (estimativa -28%)
**Total:** 3x5x65 + 3x5x3x5 = 975 + 225 = 1200min = **20.0h** (AINDA EXCEDE)

**Desvantagem:**
- Ainda excede 18h
- Performance pode piorar com menos populacao/geracoes

---

## RECOMENDACAO FINAL: OPCAO A (3 datasets x 3 chunks)

**Justificativa:**
- Cumpre requisito de 3 datasets
- Tempo seguro (15.8h com 2.2h margem)
- Variedade de drift types
- Chunks suficientes para analise

---

## DATASETS DEFINIDOS (OPCAO A)

### Dataset 1: RBF_Abrupt_Severe

**Config:**
```yaml
RBF_Abrupt_Severe:
  dataset_type: RBF
  drift_type: abrupt
  concept_sequence:
  - concept_id: c1
    duration_chunks: 2  # Chunks 0, 1
  - concept_id: c2_severe
    duration_chunks: 2  # Chunks 2, 3
```

**Caracteristica:** Drift abrupto severo (65% severity)

---

### Dataset 2: RBF_Abrupt_Moderate

**Config:**
```yaml
RBF_Abrupt_Moderate:
  dataset_type: RBF
  drift_type: abrupt
  concept_sequence:
  - concept_id: c1
    duration_chunks: 2
  - concept_id: c3_moderate
    duration_chunks: 2
```

**Caracteristica:** Drift abrupto moderado (45% severity)

---

### Dataset 3: RBF_Gradual_Moderate

**Config:**
```yaml
RBF_Gradual_Moderate:
  dataset_type: RBF
  drift_type: gradual
  gradual_drift_width_chunks: 1  # 1 chunk de transicao
  concept_sequence:
  - concept_id: c1
    duration_chunks: 1
  - concept_id: c3_moderate  # Transicao gradual no chunk 1
    duration_chunks: 2
```

**Caracteristica:** Drift gradual moderado

---

## MODELOS RIVER

**Modelos a comparar:**
1. **HAT** (Hoeffding Adaptive Tree)
2. **ARF** (Adaptive Random Forest)
3. **SRP** (Streaming Random Patches)

**Total:** 3 modelos x 3 datasets x 3 chunks = 27 avaliacoes River

---

## CONFIGURACAO DO EXPERIMENTO

### Config YAML Base

Criar `config_comparison.yaml`:

```yaml
experiment_settings:
  run_mode: drift_simulation
  drift_simulation_experiments:
  - RBF_Abrupt_Severe
  - RBF_Abrupt_Moderate
  - RBF_Gradual_Moderate
  num_runs: 1
  base_results_dir: /content/drive/MyDrive/DSL-AG-hybrid/comparison_results
  logging_level: WARNING
  evaluation_period: 6000

data_params:
  chunk_size: 6000
  num_chunks: 4  # 3 chunks treino/teste + 1 extra
  max_instances: 24000

# Configuracoes GA (identicas a config_test_single.yaml)
ga_params:
  population_size: 100
  max_generations: 200
  # ... (resto igual)

# Drift analysis datasets
drift_analysis:
  datasets:
    RBF:
      class: river.datasets.synth.RandomRBF
      n_features: 10
      feature_bounds: [[0, 1]]
      concepts:
        c1:
          seed_model: 42
        c2_severe:
          seed_model: 84
        c3_moderate:
          seed_model: 60

# Experimental streams
experimental_streams:
  RBF_Abrupt_Severe:
    dataset_type: RBF
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: c1
      duration_chunks: 2
    - concept_id: c2_severe
      duration_chunks: 2

  RBF_Abrupt_Moderate:
    dataset_type: RBF
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: c1
      duration_chunks: 2
    - concept_id: c3_moderate
      duration_chunks: 2

  RBF_Gradual_Moderate:
    dataset_type: RBF
    drift_type: gradual
    gradual_drift_width_chunks: 1
    concept_sequence:
    - concept_id: c1
      duration_chunks: 1
    - concept_id: c3_moderate
      duration_chunks: 2
```

---

## SCRIPT DE EXECUCAO

### Script Bash: `run_comparison.sh`

```bash
#!/bin/bash

# EXPERIMENTO COMPARATIVO GBML VS RIVER
# 3 datasets x 3 chunks cada = 15.8h estimado

# Diretorio de trabalho
cd /content/drive/MyDrive/DSL-AG-hybrid

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="comparison_results/experiment_${TIMESTAMP}"

echo "==============================================="
echo "EXPERIMENTO COMPARATIVO: GBML VS RIVER"
echo "==============================================="
echo "Datasets: 3 (RBF_Abrupt_Severe, RBF_Abrupt_Moderate, RBF_Gradual_Moderate)"
echo "Chunks por dataset: 3"
echo "Modelos River: HAT, ARF, SRP"
echo "Tempo estimado: 15.8h"
echo "Output dir: ${OUTPUT_DIR}"
echo "==============================================="
echo ""

# Dataset 1: RBF_Abrupt_Severe
echo "[1/3] Executando RBF_Abrupt_Severe..."
python compare_gbml_vs_river.py \
  --stream RBF_Abrupt_Severe \
  --config config_comparison.yaml \
  --models HAT ARF SRP \
  --chunks 3 \
  --chunk-size 6000 \
  --seed 42 \
  --output ${OUTPUT_DIR} \
  2>&1 | tee ${OUTPUT_DIR}/RBF_Abrupt_Severe.log

echo ""
echo "[1/3] RBF_Abrupt_Severe concluido!"
echo ""

# Dataset 2: RBF_Abrupt_Moderate
echo "[2/3] Executando RBF_Abrupt_Moderate..."
python compare_gbml_vs_river.py \
  --stream RBF_Abrupt_Moderate \
  --config config_comparison.yaml \
  --models HAT ARF SRP \
  --chunks 3 \
  --chunk-size 6000 \
  --seed 42 \
  --output ${OUTPUT_DIR} \
  2>&1 | tee ${OUTPUT_DIR}/RBF_Abrupt_Moderate.log

echo ""
echo "[2/3] RBF_Abrupt_Moderate concluido!"
echo ""

# Dataset 3: RBF_Gradual_Moderate
echo "[3/3] Executando RBF_Gradual_Moderate..."
python compare_gbml_vs_river.py \
  --stream RBF_Gradual_Moderate \
  --config config_comparison.yaml \
  --models HAT ARF SRP \
  --chunks 3 \
  --chunk-size 6000 \
  --seed 42 \
  --output ${OUTPUT_DIR} \
  2>&1 | tee ${OUTPUT_DIR}/RBF_Gradual_Moderate.log

echo ""
echo "==============================================="
echo "EXPERIMENTO CONCLUIDO!"
echo "==============================================="
echo "Resultados em: ${OUTPUT_DIR}"
echo ""

# Gera resumo consolidado
python -c "
import os
import pandas as pd
import glob

output_dir = '${OUTPUT_DIR}'

# Coleta todos os CSVs de comparacao
comparison_files = glob.glob(os.path.join(output_dir, '**/comparison_table.csv'), recursive=True)

print(f'Encontrados {len(comparison_files)} arquivos de comparacao')

# Consolida tudo em um unico DataFrame
all_results = []
for file in comparison_files:
    df = pd.read_csv(file)
    dataset_name = os.path.basename(os.path.dirname(file))
    df['dataset'] = dataset_name
    all_results.append(df)

if all_results:
    consolidated = pd.concat(all_results, ignore_index=True)
    output_path = os.path.join(output_dir, 'consolidated_results.csv')
    consolidated.to_csv(output_path, index=False)
    print(f'Resultados consolidados salvos em: {output_path}')

    # Estatisticas resumidas
    summary = consolidated.groupby(['dataset', 'model']).agg({
        'accuracy': ['mean', 'std'],
        'gmean': ['mean', 'std'],
        'f1_weighted': ['mean', 'std']
    }).round(4)

    summary_path = os.path.join(output_dir, 'summary_statistics.txt')
    with open(summary_path, 'w') as f:
        f.write('RESUMO ESTATISTICO - EXPERIMENTO COMPARATIVO\\n')
        f.write('='*70 + '\\n\\n')
        f.write(str(summary))

    print(f'Resumo estatistico salvo em: {summary_path}')
    print('\\n' + str(summary))
"

echo ""
echo "Analise consolidada concluida!"
```

---

## COMANDO SIMPLIFICADO PARA COLAB

### Comando Unico (1 linha)

```bash
cd /content/drive/MyDrive/DSL-AG-hybrid && bash run_comparison.sh 2>&1 | tee experiment_comparison_full.log
```

### OU Execucao Python Direta (sem bash):

```python
# No notebook Colab
import subprocess
import os

os.chdir('/content/drive/MyDrive/DSL-AG-hybrid')

datasets = ['RBF_Abrupt_Severe', 'RBF_Abrupt_Moderate', 'RBF_Gradual_Moderate']

for i, dataset in enumerate(datasets, 1):
    print(f"\\n[{i}/3] Executando {dataset}...")

    cmd = [
        'python', 'compare_gbml_vs_river.py',
        '--stream', dataset,
        '--config', 'config_comparison.yaml',
        '--models', 'HAT', 'ARF', 'SRP',
        '--chunks', '3',
        '--chunk-size', '6000',
        '--seed', '42',
        '--output', 'comparison_results'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    print(f"[{i}/3] {dataset} concluido!\\n")

print("\\nEXPERIMENTO COMPLETO!")
```

---

## SALVAMENTO DE DADOS

**Estrutura de diretorios criada automaticamente:**

```
comparison_results/
└── experiment_20251105_123456/
    ├── RBF_Abrupt_Severe_seed42_20251105_123456/
    │   ├── gbml_results.csv
    │   ├── river_HAT_results.csv
    │   ├── river_ARF_results.csv
    │   ├── river_SRP_results.csv
    │   ├── comparison_table.csv
    │   ├── summary.txt
    │   ├── accuracy_comparison.png
    │   ├── gmean_comparison.png
    │   └── accuracy_heatmap.png
    ├── RBF_Abrupt_Moderate_seed42_.../ (mesma estrutura)
    ├── RBF_Gradual_Moderate_seed42_.../ (mesma estrutura)
    ├── consolidated_results.csv
    └── summary_statistics.txt
```

**Arquivos importantes:**
- `gbml_results.csv`: Resultados GBML por chunk
- `river_*_results.csv`: Resultados de cada modelo River
- `comparison_table.csv`: Tabela consolidada dataset especifico
- `consolidated_results.csv`: TODOS datasets juntos
- `summary_statistics.txt`: Estatisticas resumidas
- `*.png`: Graficos comparativos

---

## VERIFICACOES PRE-EXECUCAO

### Checklist:

- [ ] `compare_gbml_vs_river.py` existe
- [ ] `baseline_river.py` existe
- [ ] `shared_evaluation.py` existe
- [ ] `gbml_evaluator.py` existe
- [ ] `config_comparison.yaml` criado
- [ ] River instalado: `pip install river`
- [ ] Matplotlib/seaborn instalados
- [ ] Google Drive montado no Colab
- [ ] Espaco em disco suficiente (estimativa: 500MB)

---

## METRICAS A ANALISAR

### Por Dataset:
1. Accuracy media e std
2. G-mean media e std
3. F1 weighted media e std
4. Tempo de execucao

### Comparacao GBML vs River:
1. Performance em drift abrupto
2. Performance em drift gradual
3. Trade-off tempo vs accuracy
4. Robustez (std menor = melhor)

### Graficos Gerados:
1. Accuracy ao longo dos chunks
2. G-mean ao longo dos chunks
3. Heatmap performance (modelo x chunk)

---

## TIMELINE ESTIMADA

| Hora | Atividade | Status |
|------|-----------|--------|
| 0:00 | Inicio - RBF_Abrupt_Severe | |
| 5:30 | RBF_Abrupt_Severe concluido (3 chunks GBML + 9 River) | |
| 5:30 | Inicio - RBF_Abrupt_Moderate | |
| 11:00 | RBF_Abrupt_Moderate concluido | |
| 11:00 | Inicio - RBF_Gradual_Moderate | |
| 16:30 | RBF_Gradual_Moderate concluido | |
| 16:30 | Analise consolidada | |
| 17:00 | EXPERIMENTO COMPLETO | |

**Margem de seguranca:** 1h (ate 18:00h max)

---

## TROUBLESHOOTING

### Se River falhar:
- Verificar instalacao: `pip install river --upgrade`
- Verificar versao: `python -c "import river; print(river.__version__)"`
- Logs em `comparison_results/.../river_*.log`

### Se GBML falhar:
- Verificar Layer 1 funcionando
- Logs detalhados em `gbml_results.csv`
- Verificar memoria/GPU disponivel

### Se tempo exceder:
- Parar experimento
- Usar resultados parciais
- Considerar Opcao B (2 datasets)

---

## PROXIMOS PASSOS APOS EXPERIMENTO

1. Analisar `consolidated_results.csv`
2. Revisar graficos comparativos
3. Identificar:
   - Qual modelo teve melhor G-mean medio?
   - GBML competitivo vs River?
   - Trade-offs tempo vs accuracy?
4. Documentar conclusoes
5. Preparar para publicacao

---

**Status:** PLANO COMPLETO
**Recomendacao:** OPCAO A (3 datasets x 3 chunks)
**Tempo estimado:** 15.8h (< 18h)
**Proxima acao:** Criar config_comparison.yaml e executar
