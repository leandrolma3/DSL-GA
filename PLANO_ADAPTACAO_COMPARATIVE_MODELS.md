# PLANO DE ADAPTACAO: Execute_All_Comparative_Models.ipynb

## Data: 2026-01-26

---

## 1. ANALISE DA ESTRUTURA ATUAL

### 1.1 Estrutura de Paths (ANTIGO)
```
experiments_6chunks_phase2_gbml/
  batch_1/
    SEA_Abrupt_Simple/
      run_1/
        chunk_data/
          chunk_0_train.csv    <- treino inicial
          chunk_1_test.csv     <- teste chunk 1
          chunk_2_test.csv     <- teste chunk 2
          ...
        chunk_metrics.json     <- resultados EGIS
```

### 1.2 Estrutura de Paths (NOVO - Unified)
```
unified_chunks/                          <- DADOS BRUTOS
  chunk_500/
    SEA_Abrupt_Simple/
      chunk_0.csv, chunk_1.csv, ... chunk_23.csv  (24 chunks)
  chunk_1000/
    SEA_Abrupt_Simple/
      chunk_0.csv, chunk_1.csv, ... chunk_11.csv  (12 chunks)
  chunk_2000/
    SEA_Abrupt_Simple/
      chunk_0.csv, chunk_1.csv, ... chunk_5.csv   (6 chunks)

experiments_unified/                     <- RESULTADOS
  chunk_500/
    batch_1/
      SEA_Abrupt_Simple/
        run_1/                           <- resultados EGIS
          chunk_metrics.json
        cdcms_results/                   <- resultados CDCMS
          chunk_metrics.json
          cdcms_raw_output.csv
  chunk_1000/
    batch_1/
      ...
```

---

## 2. ANALISE DAS CELULAS DO NOTEBOOK

### 2.1 CELULA 2.1 - Configuracoes (REQUER ADAPTACAO)

**Atual:**
```python
BATCH_CONFIG = {
    'batch_1': {
        'phase': 'phase2',
        'base_dir': 'experiments_6chunks_phase2_gbml/batch_1',  # PATH ANTIGO
        'datasets': ['SEA_Abrupt_Simple', ...]
    },
    ...
}
CHUNK_SIZE = 1000  # FIXO
```

**Proposta:**
```python
# Configuracoes por chunk_size
CHUNK_SIZES = {
    'chunk_500': {'size': 500, 'num_chunks': 24},
    'chunk_1000': {'size': 1000, 'num_chunks': 12},
    'chunk_2000': {'size': 2000, 'num_chunks': 6},
}

# Batches por chunk_size (baseado no CDCMS_PARTE7_COMPLETA.py)
BATCHES = {
    'chunk_500': ['batch_1', 'batch_2', 'batch_3'],
    'chunk_1000': ['batch_1', 'batch_2', 'batch_3', 'batch_4'],
    'chunk_2000': ['batch_1', 'batch_2', 'batch_3'],  # futuro
}

# Paths base
UNIFIED_CHUNKS_DIR = Path(WORK_DIR) / "unified_chunks"
EXPERIMENTS_DIR = Path(WORK_DIR) / "experiments_unified"
```

### 2.2 CELULA 3.4 - Carregamento de Dados (REQUER NOVA FUNCAO)

**Funcao atual:**
```python
def load_chunks_from_gbml(dataset_dir):
    # Le de: run_1/chunk_data/chunk_X_test.csv
```

**Nova funcao proposta:**
```python
def load_chunks_from_unified(chunk_size_name: str, dataset_name: str):
    """
    Carrega chunks de dados do diretorio unified_chunks.
    Formato: unified_chunks/chunk_XXX/dataset/chunk_N.csv

    Returns:
        X_chunks: Lista de arrays numpy (features)
        y_chunks: Lista de arrays numpy (classes)
    """
    data_dir = UNIFIED_CHUNKS_DIR / chunk_size_name / dataset_name

    X_chunks = []
    y_chunks = []

    # Ordenar chunks por numero
    chunk_files = sorted(data_dir.glob("chunk_*.csv"),
                        key=lambda x: int(x.stem.split('_')[1]))

    for chunk_file in chunk_files:
        df = pd.read_csv(chunk_file)

        # Assumir ultima coluna como classe
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Converter classe para int (importante para CDCMS)
        if y.dtype == bool or str(y.dtype) == 'object':
            y = y.astype(int)

        X_chunks.append(X)
        y_chunks.append(y)

    return X_chunks, y_chunks
```

### 2.3 CELULA 4.1 - Loop Principal (REQUER ADAPTACAO)

**Atual:** Itera sobre batches fixos
**Proposta:** Itera sobre chunk_sizes -> batches -> datasets

```python
for chunk_size_name in CHUNK_SIZES_TO_RUN:  # ['chunk_500', 'chunk_1000']
    chunk_config = CHUNK_SIZES[chunk_size_name]
    chunk_size = chunk_config['size']

    for batch_name in BATCHES[chunk_size_name]:
        # Obter datasets do batch
        datasets = get_datasets_for_batch(chunk_size_name, batch_name)

        for dataset_name in datasets:
            # Carregar dados
            X_chunks, y_chunks = load_chunks_from_unified(chunk_size_name, dataset_name)

            # Executar modelos...

            # Salvar resultados em:
            # experiments_unified/chunk_XXX/batch_Y/dataset/model_results/
```

### 2.4 Funcoes de Modelos (MANTER)

As funcoes de execucao dos modelos **NAO precisam de alteracao**:
- `run_rose_original()` - OK
- `run_rose_chunk_eval()` - OK (parametro chunk_size ja existe)
- `run_river_model()` - OK
- `run_acdwm()` - OK

---

## 3. MAPEAMENTO BATCH -> DATASETS

### 3.1 chunk_500 (3 batches, 50 datasets)

| Batch | Datasets |
|-------|----------|
| batch_1 | 18 datasets (SEA, AGRAWAL, HYPERPLANE, STAGGER, SINE, RANDOMTREE, Electricity) |
| batch_2 | 18 datasets (variantes Gradual, LED, WAVEFORM) |
| batch_3 | 14 datasets (Noise, Stationary, Real) |

### 3.2 chunk_1000 (4 batches, 50 datasets)

| Batch | Datasets |
|-------|----------|
| batch_1 | 14 datasets |
| batch_2 | 12 datasets |
| batch_3 | 12 datasets |
| batch_4 | 12 datasets |

---

## 4. TRATAMENTO DE DATASETS MULTICLASSE

### 4.1 Datasets Multiclasse (mesma lista do CDCMS)
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

### 4.2 Tratamento por Modelo

| Modelo | Suporta Multiclasse? | Tratamento |
|--------|---------------------|------------|
| EGIS | SIM | Executar normalmente |
| CDCMS | NAO | Marcar N/A |
| ROSE | SIM | Executar normalmente |
| HAT | SIM | Executar normalmente |
| ARF | SIM | Executar normalmente |
| SRP | SIM | Executar normalmente |
| ACDWM | PARCIAL | Pode falhar, assumir 0.0 |
| ERulesD2S | SIM | Executar normalmente |

---

## 5. FORMATO DE SAIDA DOS RESULTADOS

### 5.1 Estrutura de Arquivos por Modelo
```
experiments_unified/chunk_500/batch_1/SEA_Abrupt_Simple/
  run_1/                          <- EGIS (ja existe)
    chunk_metrics.json
  cdcms_results/                  <- CDCMS (ja existe)
    chunk_metrics.json
  rose_results/                   <- ROSE (NOVO)
    chunk_metrics.json
    rose_original_output.csv
    rose_chunk_eval_output.csv
  river_results/                  <- HAT, ARF, SRP (NOVO)
    HAT_chunk_metrics.json
    ARF_chunk_metrics.json
    SRP_chunk_metrics.json
  acdwm_results/                  <- ACDWM (NOVO)
    chunk_metrics.json
  erulesd2s_results/              <- ERulesD2S (NOVO)
    chunk_metrics.json
```

### 5.2 Formato do chunk_metrics.json (padronizado)
```json
[
  {
    "chunk": 0,
    "gmean": 0.85,
    "f1": 0.82,
    "accuracy": 0.88,
    "note": "optional"
  },
  ...
]
```

---

## 6. PLANO DE ACAO DETALHADO

### FASE 1: Preparacao (sem modificar codigo funcional)

**Passo 1.1:** Criar arquivo de configuracao unificado
- Criar `configs/unified_comparative_config.py`
- Definir CHUNK_SIZES, BATCHES, MULTICLASS_DATASETS

**Passo 1.2:** Criar funcao de carregamento
- Criar `load_chunks_from_unified()` em novo arquivo
- Testar com 1 dataset

### FASE 2: Adaptacao do Notebook

**Passo 2.1:** Criar copia de seguranca
- `Execute_All_Comparative_Models_BACKUP.ipynb`

**Passo 2.2:** Adaptar CELULA 2.1
- Substituir BATCH_CONFIG por nova configuracao
- Adicionar variaveis de controle (CHUNK_SIZES_TO_RUN)

**Passo 2.3:** Adaptar CELULA 3.4
- Adicionar `load_chunks_from_unified()`
- Manter `load_chunks_from_gbml()` para compatibilidade

**Passo 2.4:** Adaptar CELULA 4.1
- Adicionar loop externo por chunk_size
- Ajustar paths de saida

**Passo 2.5:** Adaptar CELULA 4.2
- Salvar resultados por chunk_size
- Arquivo: `all_models_results_{chunk_size}.csv`

### FASE 3: Validacao

**Passo 3.1:** Teste com 1 dataset
- chunk_500/batch_1/SEA_Abrupt_Simple
- Verificar todos os modelos

**Passo 3.2:** Teste com 1 batch completo
- chunk_500/batch_1 (todos datasets)

**Passo 3.3:** Execucao completa
- chunk_500 (todos batches)
- chunk_1000 (todos batches)

---

## 7. DECISOES PENDENTES

1. **Onde salvar resultados dos modelos comparativos?**
   - Opcao A: Dentro de experiments_unified (junto com EGIS/CDCMS)
   - Opcao B: Nova pasta comparison_results/

2. **Executar chunk_500 e chunk_1000 em paralelo ou sequencial?**
   - Sequencial recomendado (menor uso de memoria)

3. **Reusar resultados existentes do notebook antigo?**
   - Nao recomendado (estrutura diferente)

---

## 8. ESTIMATIVA DE DATASETS

| Chunk Size | Batches | Datasets | Binarios | Multiclasse |
|------------|---------|----------|----------|-------------|
| chunk_500  | 3       | ~50      | ~41      | 9           |
| chunk_1000 | 4       | ~50      | ~41      | 9           |
| chunk_2000 | 3       | ~50      | ~41      | 9 (futuro)  |

---

## 9. PROXIMOS PASSOS IMEDIATOS

1. **Revisar este plano** com o usuario
2. **Criar backup** do notebook original
3. **Implementar Fase 1** (configuracao e funcao de carregamento)
4. **Testar com 1 dataset** antes de executar em massa

---

---

## 10. DETALHES DOS MODELOS

### 10.1 ROSE (Robust Online Self-Adjusting Ensemble)
- **Implementacao:** MOA Java (rose_jars/)
- **Funcoes:** `run_rose_original()`, `run_rose_chunk_eval()`
- **Entrada:** Arquivo ARFF
- **Tempo:** ~1-5 minutos por dataset

### 10.2 HAT, ARF, SRP (River Models)
- **Implementacao:** River Python
- **Funcao:** `run_river_model()`
- **Entrada:** X_chunks, y_chunks (numpy arrays)
- **Tempo:** ~30-60 segundos por dataset

### 10.3 ACDWM (Adaptive Chunk-based DWM)
- **Implementacao:** Python (ACDWM/)
- **Funcao:** `run_acdwm()`
- **Entrada:** X_chunks, y_chunks
- **Nota:** Pode falhar em datasets multiclasse (assume gmean=0)

### 10.4 ERulesD2S (Evolutionary Rules for Data Streams)
- **Implementacao:** MOA Java + JCLEC4
- **Wrapper:** `erulesd2s_wrapper.py`
- **Entrada:** Arquivo ARFF
- **Nota:** MUITO LENTO - usa apenas resultados em cache no notebook atual
- **Tempo:** ~30-60 minutos por dataset

### 10.5 CDCMS (ja implementado separadamente)
- **Implementacao:** MOA Java
- **Script:** `CDCMS_PARTE7_COMPLETA.py`
- **Restricao:** Somente classificacao BINARIA

---

## 11. ANALISE ESTATISTICA (Celulas 5.x)

### 11.1 Friedman Test (CELULA 5.2)
- Testa se existe diferenca significativa entre modelos
- H0: Todos os modelos tem desempenho equivalente
- Output: chi-squared statistic, p-value

### 11.2 Rankings (CELULA 5.3)
- Calcula ranking medio de cada modelo
- Menor ranking = melhor desempenho

### 11.3 Nemenyi Post-hoc (CELULA 5.4)
- Compara todos os pares de modelos
- Correcao de Bonferroni para multiplas comparacoes

### 11.4 Wilcoxon Signed-Rank (CELULA 5.5)
- Testes pareados com GBML (modelo de referencia)

### 11.5 Critical Difference Diagram (CELULA 5.6)
- Visualizacao grafica dos rankings
- Modelos conectados por linha = sem diferenca significativa

---

## 12. RESUMO DAS ALTERACOES NECESSARIAS

| Celula | Descricao | Alteracao |
|--------|-----------|-----------|
| 2.1 | Configuracoes | SUBSTITUIR BATCH_CONFIG por UNIFIED_CONFIG |
| 3.4 | Load chunks | ADICIONAR load_chunks_from_unified() |
| 4.1 | Loop principal | ADAPTAR para iterar por chunk_size |
| 4.2 | Salvar resultados | ADAPTAR paths de saida |
| 5.x | Analise estatistica | MANTER (funciona com qualquer formato) |
| 6.x | Relatorios | ADAPTAR nomes de arquivos |

---

## ARQUIVOS RELACIONADOS

- `Execute_All_Comparative_Models.ipynb` - Notebook original
- `CDCMS_PARTE7_COMPLETA.py` - Referencia para configuracoes
- `unified_chunks/` - Dados brutos
- `experiments_unified/` - Resultados
- `erulesd2s_wrapper.py` - Wrapper para ERulesD2S
- `rose_jars/` - JARs do ROSE
- `ACDWM/` - Codigo do ACDWM

