# PLANO DE AJUSTE: Execute_Comparative_All_Experiments.ipynb

## Data: 2026-01-26

## STATUS: IMPLEMENTADO

Arquivos criados:
- CELULA_2_1_EXEC_COMPARATIVE_CONFIG.py
- CELULA_2_2_EXEC_COMPARATIVE_DETECT.py
- CELULA_3_1_EXEC_COMPARATIVE_LOAD.py
- CELULA_4_1_EXEC_COMPARATIVE_MAIN.py
- CELULA_4_2_EXEC_COMPARATIVE_SAVE.py
- INSTRUCOES_EXEC_COMPARATIVE.md

---

## 1. ANALISE DO NOTEBOOK ATUAL

### 1.1 Estrutura de Experimentos Existentes
```python
EXPERIMENT_CONFIGS = {
    'exp_a_chunk1000': {chunk_size: 1000, penalty: 0.0, ...},  # experiments_6chunks_*
    'exp_b_chunk2000': {chunk_size: 2000, penalty: 0.0, ...},  # experiments_chunk2000_*
    'exp_c_balanced':  {chunk_size: 2000, penalty: 0.1, ...},  # experiments_balanced_*
}
```

### 1.2 Celulas do Notebook
| Celula | Funcao | Acao |
|--------|--------|------|
| 1.x | Setup (Drive, Java, Python) | MANTER |
| 2.1 | EXPERIMENT_CONFIGS | **ADICIONAR novos experimentos** |
| 2.2 | detect_available_datasets() | **ADAPTAR para unified_chunks** |
| 3.1 | load_chunks_from_gbml() | **ADICIONAR load_chunks_from_unified()** |
| 3.2 | Funcoes ROSE | MANTER |
| 3.3 | Funcoes River | MANTER |
| 3.4 | Funcoes ACDWM | MANTER |
| 3.5 | save_model_results() | MANTER |
| 3.6 | Funcoes ERulesD2S | MANTER |
| 4.1 | Loop principal | **ADAPTAR para unified** |
| 4.2 | Salvar resultados | MANTER |

---

## 2. NOVOS EXPERIMENTOS A ADICIONAR

### 2.1 Estrutura Proposta
```python
# NOVOS EXPERIMENTOS (unified_chunks)
'exp_unified_500': {
    'chunk_size': 500,
    'penalty_weight': 0.0,
    'data_source': 'unified',  # NOVO: indica fonte de dados
    'data_dir': 'chunk_500',   # NOVO: pasta em unified_chunks
    'results_dir': 'chunk_500', # NOVO: pasta em experiments_unified
    'description': 'Unified chunks 500 (sem penalty)',
    'batches': {...}
},
'exp_unified_500_penalty': {
    'chunk_size': 500,
    'penalty_weight': 0.1,
    'data_source': 'unified',
    'data_dir': 'chunk_500',
    'results_dir': 'chunk_500_penalty',
    ...
},
'exp_unified_1000': {...},
'exp_unified_1000_penalty': {...}
```

### 2.2 Diferenca entre Antigo e Novo

| Aspecto | ANTIGO (exp_a, b, c) | NOVO (exp_unified_*) |
|---------|---------------------|----------------------|
| Dados | run_1/chunk_data/chunk_X_test.csv | unified_chunks/chunk_XXX/dataset/chunk_N.csv |
| EGIS | run_1/chunk_metrics.json | experiments_unified/.../run_1/chunk_metrics.json |
| CDCMS | N/A | experiments_unified/.../cdcms_results/chunk_metrics.json |
| Outros modelos | run_1/*.csv | run_1/*.csv (mesmo) |

---

## 3. FUNCOES A ADICIONAR/MODIFICAR

### 3.1 Nova Funcao: load_chunks_from_unified()
```python
def load_chunks_from_unified(chunk_size_name: str, dataset_name: str):
    """
    Carrega chunks de unified_chunks/chunk_XXX/dataset/chunk_N.csv
    """
    data_path = UNIFIED_CHUNKS_DIR / chunk_size_name / dataset_name
    # ... carregar chunk_0.csv, chunk_1.csv, ...
    return X_chunks, y_chunks
```

### 3.2 Nova Funcao: load_egis_results()
```python
def load_egis_results(results_dir: str, batch_name: str, dataset_name: str):
    """
    Carrega resultados EGIS de experiments_unified/
    """
    metrics_file = EXPERIMENTS_DIR / results_dir / batch_name / dataset_name / 'run_1' / 'chunk_metrics.json'
    # ... calcular media de test_gmean
    return {'gmean': avg_gmean, ...}
```

### 3.3 Nova Funcao: load_cdcms_results()
```python
def load_cdcms_results(results_dir: str, batch_name: str, dataset_name: str):
    """
    Carrega resultados CDCMS de experiments_unified/
    """
    metrics_file = EXPERIMENTS_DIR / results_dir / batch_name / dataset_name / 'cdcms_results' / 'chunk_metrics.json'
    # ... calcular media de holdout_gmean
    return {'gmean': avg_gmean, ...}
```

### 3.4 Modificar: detect_available_datasets()
```python
def detect_available_datasets(exp_name, config):
    # Se data_source == 'unified':
    #   Verificar em unified_chunks e experiments_unified
    # Senao:
    #   Manter comportamento antigo (run_1/chunk_data)
```

---

## 4. MODIFICACOES NO LOOP PRINCIPAL (Celula 4.1)

### 4.1 Deteccao do Tipo de Experimento
```python
for exp_name in EXPERIMENTS_TO_RUN:
    config = EXPERIMENT_CONFIGS[exp_name]
    is_unified = config.get('data_source') == 'unified'

    if is_unified:
        # Carregar de unified_chunks
        X_chunks, y_chunks = load_chunks_from_unified(
            config['data_dir'], dataset_name
        )
        # Carregar EGIS e CDCMS existentes
        egis_results = load_egis_results(config['results_dir'], batch_name, dataset_name)
        cdcms_results = load_cdcms_results(config['results_dir'], batch_name, dataset_name)
    else:
        # Comportamento antigo
        X_chunks, y_chunks = load_chunks_from_gbml(dataset_dir)
```

### 4.2 Resultados EGIS (renomeado de GBML)
```python
# Adicionar EGIS aos resultados (nao executar, apenas carregar)
if is_unified:
    egis = load_egis_results(...)
    ALL_RESULTS.append({
        'experiment': exp_name,
        'model': 'EGIS',  # Renomeado de GBML
        'gmean': egis['gmean'],
        ...
    })
```

---

## 5. CELULAS A CRIAR/MODIFICAR

### Celula 2.1 - Configuracoes
- ADICIONAR novos experimentos em EXPERIMENT_CONFIGS
- ADICIONAR variaveis UNIFIED_CHUNKS_DIR, EXPERIMENTS_UNIFIED_DIR
- ADICIONAR MULTICLASS_DATASETS

### Celula 2.2 - Deteccao de Datasets
- MODIFICAR detect_available_datasets() para suportar unified

### Celula 3.1 - Carregamento de Dados
- ADICIONAR load_chunks_from_unified()
- ADICIONAR load_egis_results()
- ADICIONAR load_cdcms_results()
- MANTER load_chunks_from_gbml() para compatibilidade

### Celula 4.1 - Loop Principal
- ADAPTAR para detectar tipo de experimento
- ADICIONAR carregamento de EGIS e CDCMS para unified
- RENOMEAR GBML para EGIS

---

## 6. ORDEM DE IMPLEMENTACAO

1. **BACKUP** do notebook original
2. **Celula 2.1**: Adicionar configs dos novos experimentos
3. **Celula 2.2**: Adaptar detect_available_datasets()
4. **Celula 3.1**: Adicionar funcoes de carregamento unified
5. **Celula 4.1**: Adaptar loop principal
6. **TESTE** com 1 dataset de unified_500

---

## 7. MAPEAMENTO DE BATCHES (unified_chunks)

### chunk_500 e chunk_500_penalty (3 batches)
| Batch | Datasets |
|-------|----------|
| batch_1 | ~18 datasets (SEA, AGRAWAL, etc.) |
| batch_2 | ~18 datasets |
| batch_3 | ~14 datasets |

### chunk_1000 e chunk_1000_penalty (4 batches)
| Batch | Datasets |
|-------|----------|
| batch_1 | ~14 datasets |
| batch_2 | ~12 datasets |
| batch_3 | ~12 datasets |
| batch_4 | ~12 datasets |

---

## 8. DATASETS MULTICLASSE (CDCMS N/A)

```python
MULTICLASS_DATASETS = {
    'LED_Abrupt_Simple': 10,
    'LED_Gradual_Simple': 10,
    'LED_Stationary': 10,
    'WAVEFORM_Abrupt_Simple': 3,
    'WAVEFORM_Gradual_Simple': 3,
    'WAVEFORM_Stationary': 3,
    'CovType': 7,
    'Shuttle': 7,
    'RBF_Stationary': 4,
}
```

---

## 9. COLUNAS DO CSV DE RESULTADOS

| Coluna | Descricao |
|--------|-----------|
| experiment | exp_unified_500, exp_unified_500_penalty, etc. |
| batch | batch_1, batch_2, etc. |
| dataset | nome do dataset |
| model | EGIS, CDCMS, ROSE_ChunkEval, HAT, ARF, SRP, ACDWM, ERulesD2S |
| gmean | G-Mean do modelo |
| status | OK, CACHED, N/A, NOT_FOUND, ERROR |
| chunk_size | 500, 1000 |
| penalty | True/False |

---

## 10. PROXIMOS PASSOS

1. Criar arquivos .py para cada celula modificada
2. Usuario copia celula por celula no Colab
3. Testar com EXPERIMENTS_TO_RUN = ['exp_unified_500']
4. Se funcionar, executar todos

