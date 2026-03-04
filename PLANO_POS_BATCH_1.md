# PLANO COMPLETO: POS-EXECUCAO BATCH 1

**Data**: 2025-11-17
**Status**: Batch 1 GBML concluido com sucesso
**Proximos passos**: Analise, pos-processamento e modelos comparativos

---

## FASE 1: ANALISE DOS RESULTADOS BATCH 1

### 1.1 Analise dos Logs

**Arquivos**:
- `batch_1_exec_1.log` - SEA_Abrupt_Simple + AGRAWAL_Abrupt_Simple_Severe
- `batch_1_exec_2.log` - RBF_Abrupt_Severe + HYPERPLANE_Abrupt_Simple
- `batch_1_exec_3.log` - STAGGER_Abrupt_Chain

**Metricas a extrair**:
- Tempo total de cada execucao
- Tempo medio por chunk
- G-mean final de cada dataset
- Numero de regras geradas
- Taxa de hit do cache
- Deteccao de drift severo

**Scripts para analise**:
```python
# Criar script: analyze_batch_1_logs.py
# - Parse dos 3 logs
# - Extracao de metricas
# - Geracao de tabela consolidada
```

### 1.2 Validacao da Estrutura de Resultados

**Verificar para cada dataset**:
```
experiments_6chunks_phase1_gbml/batch_1/
  <DATASET_NAME>/
    run_1/
      chunk_data/          # 12 arquivos (6 train + 6 test)
      plots/               # Minimo 3 plots
      experiment_summary.json
      rule_history.txt
      fitness_gmean_history.pkl
      periodic_test_gmean.pkl
      best_individuals/    # 6 arquivos
      used_attributes.pkl
      concept_differences.json  # Se analyze_concept_difference rodou
```

**Comando de validacao**:
```bash
# Contar arquivos por dataset
for dataset in SEA_Abrupt_Simple AGRAWAL_Abrupt_Simple_Severe RBF_Abrupt_Severe HYPERPLANE_Abrupt_Simple STAGGER_Abrupt_Chain; do
  echo "=== $dataset ==="
  echo "Chunks: $(ls experiments_6chunks_phase1_gbml/batch_1/$dataset/run_1/chunk_data/*.csv 2>/dev/null | wc -l)"
  echo "Plots: $(ls experiments_6chunks_phase1_gbml/batch_1/$dataset/run_1/plots/*.png 2>/dev/null | wc -l)"
  echo "Summary: $([ -f experiments_6chunks_phase1_gbml/batch_1/$dataset/run_1/experiment_summary.json ] && echo 'OK' || echo 'MISSING')"
done
```

### 1.3 Analise de Qualidade dos Resultados

**Metricas por dataset**:
```python
import json
import pandas as pd

datasets = [
    "SEA_Abrupt_Simple",
    "AGRAWAL_Abrupt_Simple_Severe",
    "RBF_Abrupt_Severe",
    "HYPERPLANE_Abrupt_Simple",
    "STAGGER_Abrupt_Chain"
]

results = []
for dataset in datasets:
    summary_file = f"experiments_6chunks_phase1_gbml/batch_1/{dataset}/run_1/experiment_summary.json"
    with open(summary_file, 'r') as f:
        data = json.load(f)
        results.append({
            'Dataset': dataset,
            'Test_Gmean': data.get('test_gmean'),
            'Train_Gmean': data.get('train_gmean'),
            'Num_Rules': data.get('total_rules'),
            'Chunks_Processed': data.get('chunks_processed')
        })

df = pd.DataFrame(results)
print(df.to_string(index=False))
```

---

## FASE 2: POS-PROCESSAMENTO GBML

### 2.1 Re-executar analyze_concept_difference.py

**Problema**: Foi executado com config errado antes

**Solucao**: Re-executar para cada dataset

```python
# Para cada dataset, ajustar config e executar
for exec_num in [1, 2, 3]:
    # Configurar
    !cp configs/config_batch_1_exec_{exec_num}.yaml config_test_drift_recovery.yaml

    # Executar analyze_concept_difference.py
    !python analyze_concept_difference.py

    print(f"✓ Concept analysis exec_{exec_num} concluido")
```

**Ou criar script especifico**:
```python
# Script: analyze_batch_1_concepts.py
# - Le cada dataset
# - Calcula concept differences
# - Gera heatmaps
# - Salva concept_differences.json em cada pasta
```

### 2.2 Executar generate_plots.py

```python
# Gera plots para todos os datasets do batch_1
!python generate_plots.py --batch 1

# Ou manualmente para cada dataset
datasets = ["SEA_Abrupt_Simple", "AGRAWAL_Abrupt_Simple_Severe",
            "RBF_Abrupt_Severe", "HYPERPLANE_Abrupt_Simple", "STAGGER_Abrupt_Chain"]

for dataset in datasets:
    print(f"\n=== Gerando plots para {dataset} ===")
    !python generate_plots.py --dataset {dataset} --batch 1
```

**Plots esperados**:
- Periodic G-mean plot
- Fitness evolution
- Drift analysis
- Rule activation

### 2.3 Executar rule_diff_analyzer.py

```python
# Analisa diferencas nas regras entre chunks
!python rule_diff_analyzer.py --batch 1

# Ou por dataset
for dataset in datasets:
    print(f"\n=== Analisando regras de {dataset} ===")
    !python rule_diff_analyzer.py --dataset {dataset} --batch 1
```

**Outputs esperados**:
- Rule similarity matrix
- Rule change analysis
- Adaptation to drift analysis

### 2.4 Consolidar Resultados GBML

```python
# Script: consolidate_batch_1_results.py
# - Le todos os experiment_summary.json
# - Consolida em um unico CSV
# - Gera estatisticas agregadas
# - Cria tabela LaTeX para paper

import json
import pandas as pd

results = []
for dataset in datasets:
    summary_file = f"experiments_6chunks_phase1_gbml/batch_1/{dataset}/run_1/experiment_summary.json"
    with open(summary_file, 'r') as f:
        data = json.load(f)
        data['dataset'] = dataset
        results.append(data)

df = pd.DataFrame(results)
df.to_csv('experiments_6chunks_phase1_gbml/batch_1/consolidated_results_gbml.csv', index=False)

# Estatisticas
print("=== ESTATISTICAS BATCH 1 - GBML ===")
print(f"Total datasets: {len(df)}")
print(f"Avg Test G-mean: {df['test_gmean'].mean():.4f} ± {df['test_gmean'].std():.4f}")
print(f"Avg Train G-mean: {df['train_gmean'].mean():.4f} ± {df['train_gmean'].std():.4f}")
print(f"Avg Num Rules: {df['total_rules'].mean():.2f}")
```

---

## FASE 3: EXECUTAR MODELOS RIVER

### 3.1 Preparar Chunks Salvos

Os chunks ja foram salvos pelo GBML. Verificar:

```python
# Verificar se chunks estao disponiveis
for dataset in datasets:
    chunk_dir = f"experiments_6chunks_phase1_gbml/batch_1/{dataset}/run_1/chunk_data"
    chunk_count = len([f for f in os.listdir(chunk_dir) if f.endswith('_test.csv')])
    print(f"{dataset}: {chunk_count} chunks")
```

### 3.2 Executar ACDWM

```python
# Script: run_acdwm_batch_1.py
# Usar chunks salvos do GBML

from baseline_acdwm import ACDWMClassifier
import pandas as pd

datasets = ["SEA_Abrupt_Simple", "AGRAWAL_Abrupt_Simple_Severe",
            "RBF_Abrupt_Severe", "HYPERPLANE_Abrupt_Simple", "STAGGER_Abrupt_Chain"]

results_acdwm = []

for dataset in datasets:
    print(f"\n=== Executando ACDWM em {dataset} ===")

    # Carregar chunks
    chunk_dir = f"experiments_6chunks_phase1_gbml/batch_1/{dataset}/run_1/chunk_data"

    # Inicializar modelo
    model = ACDWMClassifier()

    # Processar chunks sequencialmente
    gmeans = []
    for i in range(6):
        # Carregar chunk
        train_df = pd.read_csv(f"{chunk_dir}/chunk_{i}_train.csv")
        test_df = pd.read_csv(f"{chunk_dir}/chunk_{i}_test.csv")

        # Treinar
        X_train = train_df.drop('class', axis=1)
        y_train = train_df['class']
        model.partial_fit(X_train, y_train)

        # Testar
        X_test = test_df.drop('class', axis=1)
        y_test = test_df['class']
        predictions = model.predict(X_test)

        # Calcular G-mean
        from sklearn.metrics import balanced_accuracy_score
        gmean = balanced_accuracy_score(y_test, predictions)
        gmeans.append(gmean)

        print(f"  Chunk {i}: G-mean = {gmean:.4f}")

    # Salvar resultados
    results_acdwm.append({
        'dataset': dataset,
        'avg_gmean': np.mean(gmeans),
        'std_gmean': np.std(gmeans),
        'gmeans_per_chunk': gmeans
    })

# Salvar resultados ACDWM
pd.DataFrame(results_acdwm).to_csv('experiments_6chunks_phase1_gbml/batch_1/results_acdwm.csv', index=False)
```

### 3.3 Executar HAT, ARF, SRP

```python
# Similar ao ACDWM, mas usando modelos River
from river import tree, forest, ensemble

models = {
    'HAT': tree.HoeffdingAdaptiveTreeClassifier(),
    'ARF': forest.ARFClassifier(),
    'SRP': ensemble.SRPClassifier()
}

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Executando {model_name}")
    print('='*60)

    results_model = []

    for dataset in datasets:
        print(f"\n  Dataset: {dataset}")
        chunk_dir = f"experiments_6chunks_phase1_gbml/batch_1/{dataset}/run_1/chunk_data"

        gmeans = []
        for i in range(6):
            train_df = pd.read_csv(f"{chunk_dir}/chunk_{i}_train.csv")
            test_df = pd.read_csv(f"{chunk_dir}/chunk_{i}_test.csv")

            # Treinar incrementalmente
            for _, row in train_df.iterrows():
                x = row.drop('class').to_dict()
                y = row['class']
                model.learn_one(x, y)

            # Testar
            predictions = []
            for _, row in test_df.iterrows():
                x = row.drop('class').to_dict()
                pred = model.predict_one(x)
                predictions.append(pred)

            gmean = balanced_accuracy_score(test_df['class'], predictions)
            gmeans.append(gmean)
            print(f"    Chunk {i}: G-mean = {gmean:.4f}")

        results_model.append({
            'dataset': dataset,
            'avg_gmean': np.mean(gmeans),
            'std_gmean': np.std(gmeans),
            'gmeans_per_chunk': gmeans
        })

    # Salvar resultados
    pd.DataFrame(results_model).to_csv(
        f'experiments_6chunks_phase1_gbml/batch_1/results_{model_name.lower()}.csv',
        index=False
    )
```

---

## FASE 4: EXECUTAR ERULESD2S

### 4.1 Preparar Dados para ERulesD2S

ERulesD2S requer formato ARFF:

```python
# Converter chunks para ARFF
from arff_converter import convert_csv_to_arff

for dataset in datasets:
    chunk_dir = f"experiments_6chunks_phase1_gbml/batch_1/{dataset}/run_1/chunk_data"
    arff_dir = f"experiments_6chunks_phase1_gbml/batch_1/{dataset}/run_1/chunk_data_arff"

    os.makedirs(arff_dir, exist_ok=True)

    for i in range(6):
        train_csv = f"{chunk_dir}/chunk_{i}_train.csv"
        test_csv = f"{chunk_dir}/chunk_{i}_test.csv"

        convert_csv_to_arff(train_csv, f"{arff_dir}/chunk_{i}_train.arff")
        convert_csv_to_arff(test_csv, f"{arff_dir}/chunk_{i}_test.arff")
```

### 4.2 Executar ERulesD2S

```python
# Script: run_erulesd2s_batch_1.py
from erulesd2s_wrapper import ERulesD2SClassifier

results_erulesd2s = []

for dataset in datasets:
    print(f"\n=== Executando ERulesD2S em {dataset} ===")

    arff_dir = f"experiments_6chunks_phase1_gbml/batch_1/{dataset}/run_1/chunk_data_arff"

    model = ERulesD2SClassifier()
    gmeans = []

    for i in range(6):
        train_arff = f"{arff_dir}/chunk_{i}_train.arff"
        test_arff = f"{arff_dir}/chunk_{i}_test.arff"

        # Treinar
        model.train(train_arff)

        # Testar
        predictions = model.predict(test_arff)

        # Calcular G-mean
        test_df = pd.read_csv(f"experiments_6chunks_phase1_gbml/batch_1/{dataset}/run_1/chunk_data/chunk_{i}_test.csv")
        gmean = balanced_accuracy_score(test_df['class'], predictions)
        gmeans.append(gmean)

        print(f"  Chunk {i}: G-mean = {gmean:.4f}")

    results_erulesd2s.append({
        'dataset': dataset,
        'avg_gmean': np.mean(gmeans),
        'std_gmean': np.std(gmeans),
        'gmeans_per_chunk': gmeans
    })

# Salvar resultados
pd.DataFrame(results_erulesd2s).to_csv('experiments_6chunks_phase1_gbml/batch_1/results_erulesd2s.csv', index=False)
```

---

## FASE 5: CONSOLIDACAO E COMPARACAO FINAL

### 5.1 Consolidar Todos os Resultados

```python
# Script: consolidate_all_models_batch_1.py

import pandas as pd

# Carregar resultados de todos os modelos
results_gbml = pd.read_csv('experiments_6chunks_phase1_gbml/batch_1/consolidated_results_gbml.csv')
results_acdwm = pd.read_csv('experiments_6chunks_phase1_gbml/batch_1/results_acdwm.csv')
results_hat = pd.read_csv('experiments_6chunks_phase1_gbml/batch_1/results_hat.csv')
results_arf = pd.read_csv('experiments_6chunks_phase1_gbml/batch_1/results_arf.csv')
results_srp = pd.read_csv('experiments_6chunks_phase1_gbml/batch_1/results_srp.csv')
results_erulesd2s = pd.read_csv('experiments_6chunks_phase1_gbml/batch_1/results_erulesd2s.csv')

# Criar tabela comparativa
comparison = pd.DataFrame({
    'Dataset': results_gbml['dataset'],
    'GBML': results_gbml['test_gmean'],
    'ACDWM': results_acdwm['avg_gmean'],
    'HAT': results_hat['avg_gmean'],
    'ARF': results_arf['avg_gmean'],
    'SRP': results_srp['avg_gmean'],
    'ERulesD2S': results_erulesd2s['avg_gmean']
})

# Calcular ranking
for col in ['GBML', 'ACDWM', 'HAT', 'ARF', 'SRP', 'ERulesD2S']:
    comparison[f'{col}_rank'] = comparison[col].rank(ascending=False)

# Salvar
comparison.to_csv('experiments_6chunks_phase1_gbml/batch_1/comparison_all_models.csv', index=False)

# Imprimir tabela
print("\n=== COMPARACAO BATCH 1 - TODOS OS MODELOS ===\n")
print(comparison.to_string(index=False))

# Estatisticas agregadas
print("\n=== ESTATISTICAS AGREGADAS ===\n")
for col in ['GBML', 'ACDWM', 'HAT', 'ARF', 'SRP', 'ERulesD2S']:
    print(f"{col:12s}: {comparison[col].mean():.4f} ± {comparison[col].std():.4f}")
```

### 5.2 Analise Estatistica

```python
# Teste de Friedman e Nemenyi
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# Preparar dados
models = ['GBML', 'ACDWM', 'HAT', 'ARF', 'SRP', 'ERulesD2S']
data_matrix = comparison[models].values

# Teste de Friedman
stat, p_value = friedmanchisquare(*data_matrix.T)
print(f"\nFriedman test: statistic={stat:.4f}, p-value={p_value:.6f}")

if p_value < 0.05:
    print("Diferenca significativa entre os modelos!")

    # Post-hoc Nemenyi
    nemenyi_results = sp.posthoc_nemenyi_friedman(data_matrix)
    print("\nNemenyi post-hoc test:")
    print(nemenyi_results)
```

### 5.3 Gerar Relatorio Final

```python
# Script: generate_batch_1_report.py
# - Tabelas consolidadas
# - Graficos comparativos
# - Analise estatistica
# - Critical difference diagram
# - Relatorio LaTeX

# Salvar em:
# experiments_6chunks_phase1_gbml/batch_1/REPORT_BATCH_1.md
# experiments_6chunks_phase1_gbml/batch_1/REPORT_BATCH_1.tex
```

---

## CRONOGRAMA DE EXECUCAO

### Hoje (Analise e Pos-processamento):
- **1-2h**: Analise dos logs
- **30min**: Validacao da estrutura
- **1h**: Re-executar analyze_concept_difference
- **1h**: Executar generate_plots
- **30min**: Executar rule_diff_analyzer
- **30min**: Consolidar resultados GBML

### Amanha (Modelos Comparativos):
- **2-3h**: Executar ACDWM (rapido)
- **2-3h**: Executar HAT, ARF, SRP (rapido)
- **3-4h**: Executar ERulesD2S (mais lento)
- **1h**: Consolidar e comparar
- **1h**: Analise estatistica
- **1h**: Gerar relatorio final

**TOTAL**: 2 dias

---

## CHECKLIST DE EXECUCAO

### Fase 1: Analise
- [ ] Analisar 3 logs (exec_1, exec_2, exec_3)
- [ ] Validar estrutura de arquivos
- [ ] Extrair metricas de qualidade
- [ ] Gerar tabela consolidada GBML

### Fase 2: Pos-processamento
- [ ] Re-executar analyze_concept_difference (correto)
- [ ] Executar generate_plots
- [ ] Executar rule_diff_analyzer
- [ ] Consolidar resultados GBML

### Fase 3: Modelos River
- [ ] Executar ACDWM
- [ ] Executar HAT
- [ ] Executar ARF
- [ ] Executar SRP
- [ ] Salvar resultados

### Fase 4: ERulesD2S
- [ ] Converter chunks para ARFF
- [ ] Executar ERulesD2S
- [ ] Salvar resultados

### Fase 5: Consolidacao
- [ ] Consolidar todos os modelos
- [ ] Analise estatistica
- [ ] Critical difference diagram
- [ ] Gerar relatorio final

---

**Criado em**: 2025-11-17
**Status**: Pronto para iniciar Fase 1
