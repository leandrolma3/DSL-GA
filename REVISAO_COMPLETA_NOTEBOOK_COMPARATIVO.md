# REVISAO COMPLETA - Execute_All_Comparative_Models.ipynb

**Data**: 2025-12-16
**Objetivo**: Documentar estrutura completa do notebook para adaptacao a novos experimentos

---

## 1. VISAO GERAL DO NOTEBOOK

### 1.1 Proposito
Executa **8 modelos** (incluindo 2 versoes do ROSE) em **52 datasets** distribuidos em **7 batches**.

### 1.2 Modelos Comparativos (CORRIGIDO)

| # | Modelo | Descricao | Tipo | Limitacoes |
|---|--------|-----------|------|------------|
| 1 | **GBML** | Modelo proposto (resultados existentes) | Regras | - |
| 2 | **ROSE_Original** | Prequential instance-by-instance | Ensemble | - |
| 3 | **ROSE_ChunkEval** | Avaliacao por chunk (comparavel GBML) | Ensemble | - |
| 4 | **HAT** | Hoeffding Adaptive Tree | Arvore | - |
| 5 | **ARF** | Adaptive Random Forest | Ensemble | - |
| 6 | **SRP** | Streaming Random Patches | Ensemble | - |
| 7 | **ACDWM** | Adaptive Chunk-based DWM | Ensemble | **So binario** |
| 8 | **ERulesD2S** | Evolutionary Rules for Data Streams | Regras | - |

**NOTA IMPORTANTE**: ROSE aparece em **duas versoes**:
- **ROSE_Original**: Avaliacao prequential (metrica global final)
- **ROSE_ChunkEval**: Avaliacao por chunk (media das metricas, comparavel com GBML)

---

## 2. ESTRUTURA DAS CELULAS

### PARTE 1: SETUP DO AMBIENTE
| Celula | Descricao | Funcao |
|--------|-----------|--------|
| 1.1 | Montar Google Drive | `drive.mount()` |
| 1.3 | Instalar Java 11 + Maven | Necessario para ROSE e ERulesD2S |
| 1.4 | Instalar Dependencias Python | river, imbalanced-learn, scipy, etc. |
| 1.5 | Baixar JARs do ROSE | ROSE-1.0.jar, MOA-dependencies.jar |
| 1.6 | Clonar ACDWM Repository | github.com/jasonyanglu/ACDWM |
| 1.7 | Setup ERulesD2S | JCLEC4 JAR |
| 1.8 | Verificacao do Ambiente | Diagnostico de dependencias |

### PARTE 2: CONFIGURACAO DOS EXPERIMENTOS
| Celula | Descricao | Variaveis Importantes |
|--------|-----------|----------------------|
| 2.1 | Definir Configuracoes | BATCH_CONFIG, MODELS_TO_RUN, MODEL_TIMEOUT, CHUNK_SIZE |

### PARTE 3: FUNCOES AUXILIARES
| Celula | Descricao | Funcoes |
|--------|-----------|---------|
| 3.1 | Funcoes ROSE + ARFF | `create_arff()`, `run_rose_original()`, `run_rose_chunk_eval()` |
| 3.2 | Funcoes River | `run_river_model()` - HAT, ARF, SRP |
| 3.3 | Funcoes ACDWM | `run_acdwm()` |
| 3.4 | Carregar Chunks | `load_chunks_from_gbml()`, `load_arff_files()` |

### PARTE 4: EXECUCAO DOS EXPERIMENTOS
| Celula | Descricao |
|--------|-----------|
| 4.1 | Executar TODOS os Modelos em TODOS os Datasets |
| 4.2 | Salvar Resultados Consolidados |

### PARTE 5: ANALISE ESTATISTICA
| Celula | Descricao |
|--------|-----------|
| 5.1 | Preparar Dados para Analise |
| 5.2 | Friedman Test |
| 5.3 | Calculo de Rankings |
| 5.4 | Nemenyi Post-hoc Test |
| 5.5 | Wilcoxon Signed-Rank Tests |
| 5.6 | Critical Difference Diagram |

### PARTE 6: RELATORIO FINAL
| Celula | Descricao |
|--------|-----------|
| 6.1 | Gerar Relatorio Final |
| 6.2 | Tabela Final de Resultados |
| 6.3 | Salvar Resultados no Drive |

---

## 3. CONFIGURACOES ATUAIS

### 3.1 Variaveis Globais

```python
# Diretorios
DRIVE_BASE = "/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid"
WORK_DIR = "/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid"

# Chunk size (CRITICO - precisa ser alterado para experimentos chunk2000)
CHUNK_SIZE = 1000  # <<< ALTERAR PARA 2000 nos novos experimentos

# Modelos
MODELS_TO_RUN = [
    'ROSE_Original',   # ROSE prequential
    'ROSE_ChunkEval',  # ROSE por chunk
    'HAT',             # Hoeffding Adaptive Tree
    'ARF',             # Adaptive Random Forest
    'SRP',             # Streaming Random Patches
    'ACDWM',           # So binario!
    'ERulesD2S'        # Evolutionary Rules
]

# Timeouts (segundos)
MODEL_TIMEOUT = {
    'ROSE_Original': 600,
    'ROSE_ChunkEval': 600,
    'HAT': 300,
    'ARF': 600,
    'SRP': 600,
    'ACDWM': 600,
    'ERulesD2S': 1800
}

# Batches
BATCHES_TO_RUN = ['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5', 'batch_6', 'batch_7']
```

### 3.2 Configuracao dos Batches (BATCH_CONFIG)

```python
BATCH_CONFIG = {
    'batch_1': {
        'phase': 'phase2',
        'base_dir': 'experiments_6chunks_phase2_gbml/batch_1',  # <<< ALTERAR
        'datasets': [
            'SEA_Abrupt_Simple', 'SEA_Abrupt_Chain', 'SEA_Abrupt_Recurring',
            'AGRAWAL_Abrupt_Simple_Mild', 'AGRAWAL_Abrupt_Simple_Severe', 'AGRAWAL_Abrupt_Chain_Long',
            'RBF_Abrupt_Severe', 'RBF_Abrupt_Blip',
            'STAGGER_Abrupt_Chain', 'STAGGER_Abrupt_Recurring',
            'HYPERPLANE_Abrupt_Simple', 'RANDOMTREE_Abrupt_Simple'
        ]  # 12 datasets
    },
    'batch_2': {
        'phase': 'phase2',
        'base_dir': 'experiments_6chunks_phase2_gbml/batch_2',
        'datasets': [
            'SEA_Gradual_Simple_Fast', 'SEA_Gradual_Simple_Slow', 'SEA_Gradual_Recurring',
            'STAGGER_Gradual_Chain',
            'RBF_Gradual_Moderate', 'RBF_Gradual_Severe',
            'HYPERPLANE_Gradual_Simple', 'RANDOMTREE_Gradual_Simple', 'LED_Gradual_Simple'
        ]  # 9 datasets
    },
    'batch_3': {
        'phase': 'phase2',
        'base_dir': 'experiments_6chunks_phase2_gbml/batch_3',
        'datasets': [
            'SEA_Abrupt_Chain_Noise', 'STAGGER_Abrupt_Chain_Noise',
            'AGRAWAL_Abrupt_Simple_Severe_Noise', 'SINE_Abrupt_Recurring_Noise',
            'RBF_Abrupt_Blip_Noise', 'RBF_Gradual_Severe_Noise',
            'HYPERPLANE_Gradual_Noise', 'RANDOMTREE_Gradual_Noise'
        ]  # 8 datasets
    },
    'batch_4': {
        'phase': 'phase2',
        'base_dir': 'experiments_6chunks_phase2_gbml/batch_4',
        'datasets': [
            'SINE_Abrupt_Simple', 'SINE_Gradual_Recurring',
            'LED_Abrupt_Simple',
            'WAVEFORM_Abrupt_Simple', 'WAVEFORM_Gradual_Simple',
            'RANDOMTREE_Abrupt_Recurring'
        ]  # 6 datasets
    },
    'batch_5': {
        'phase': 'phase3',
        'base_dir': 'experiments_6chunks_phase3_real/batch_5',
        'datasets': [
            'Electricity', 'Shuttle', 'CovType', 'PokerHand', 'IntelLabSensors'
        ]  # 5 datasets
    },
    'batch_6': {
        'phase': 'phase3',
        'base_dir': 'experiments_6chunks_phase3_real/batch_6',
        'datasets': [
            'SEA_Stationary', 'AGRAWAL_Stationary', 'RBF_Stationary',
            'LED_Stationary', 'HYPERPLANE_Stationary', 'RANDOMTREE_Stationary'
        ]  # 6 datasets
    },
    'batch_7': {
        'phase': 'phase3',
        'base_dir': 'experiments_6chunks_phase3_real/batch_7',
        'datasets': [
            'STAGGER_Stationary', 'WAVEFORM_Stationary', 'SINE_Stationary',
            'AssetNegotiation_F2', 'AssetNegotiation_F3', 'AssetNegotiation_F4'
        ]  # 6 datasets
    }
}
# TOTAL: 52 datasets
```

---

## 4. FUNCOES PRINCIPAIS

### 4.1 ROSE - Duas Versoes

```python
def run_rose_original(arff_file, output_dir, n_classes=2, max_instances=None, timeout=600):
    """
    ROSE_Original: Executa ROSE como no paper original.
    - Avaliacao prequential (instance-by-instance)
    - Retorna metrica GLOBAL (ultima linha do output)
    """

def run_rose_chunk_eval(arff_file, output_dir, n_classes=2, chunk_size=1000, n_chunks=6, timeout=600):
    """
    ROSE_ChunkEval: Executa ROSE com avaliacao por chunk.
    - Avaliacao a cada chunk_size instancias
    - Retorna MEDIA das metricas por chunk (comparavel com GBML)
    """
```

### 4.2 River Models (HAT, ARF, SRP)

```python
def run_river_model(model_name, X_chunks, y_chunks, timeout=300):
    """
    Executa modelo River (HAT, ARF, SRP) nos chunks.
    - HAT: HoeffdingAdaptiveTreeClassifier
    - ARF: AdaptiveRandomForestClassifier (n_models=10)
    - SRP: SRPClassifier (n_models=10)
    """
```

### 4.3 ACDWM

```python
def run_acdwm(X_chunks, y_chunks, acdwm_path="/content/ACDWM", timeout=600):
    """
    Executa ACDWM nos chunks.
    LIMITACAO: ACDWM so funciona com problemas BINARIOS (2 classes).
    Para multiclasse, retorna gmean=0.0 com erro explicativo.
    """
```

### 4.4 Funcoes de Carregamento

```python
def load_chunks_from_gbml(dataset_dir):
    """Carrega chunks de dados do diretorio GBML (chunk_data/)."""

def load_arff_files(dataset_dir):
    """Carrega arquivos ARFF existentes para usar com ROSE."""

def load_gbml_results(dataset_dir):
    """Carrega resultados GBML do chunk_metrics.json."""
```

---

## 5. OUTPUTS GERADOS

### 5.1 Por Dataset
- `rose_original_results.csv` - Resultados ROSE prequential
- `rose_original_log.txt` - Log ROSE prequential
- `rose_chunk_eval_results.csv` - Resultados ROSE por chunk
- `rose_chunk_eval_log.txt` - Log ROSE por chunk
- `river_HAT_results.csv` - Resultados HAT
- `river_ARF_results.csv` - Resultados ARF
- `river_SRP_results.csv` - Resultados SRP
- `acdwm_results.csv` - Resultados ACDWM
- `erulesd2s_results.csv` - Resultados ERulesD2S

### 5.2 Consolidados
- `all_models_consolidated_results.csv` - Todos os resultados
- `pivot_gmean_all_models.csv` - Tabela pivoteada por modelo
- `critical_difference_diagram.png` - Diagrama CD

---

## 6. ADAPTACOES NECESSARIAS PARA NOVOS EXPERIMENTOS

### 6.1 Para experiments_chunk2000_*

**Alterar em Celula 2.1:**
```python
# ANTES
CHUNK_SIZE = 1000

# DEPOIS
CHUNK_SIZE = 2000
```

**Alterar BATCH_CONFIG:**
```python
# ANTES
'base_dir': 'experiments_6chunks_phase2_gbml/batch_1'

# DEPOIS
'base_dir': 'experiments_chunk2000_phase1/batch_1'
```

### 6.2 Para experiments_balanced_*

**Alterar BATCH_CONFIG:**
```python
# ANTES
'base_dir': 'experiments_6chunks_phase2_gbml/batch_1'

# DEPOIS
'base_dir': 'experiments_balanced_phase1/batch_1'
```

### 6.3 Mapeamento Completo de Diretorios

| Experimento | Batches 1-4 | Batches 5-7 |
|-------------|-------------|-------------|
| Original (chunk=1000) | experiments_6chunks_phase2_gbml | experiments_6chunks_phase3_real |
| Chunk 2000 | experiments_chunk2000_phase1 | experiments_chunk2000_phase2 |
| Balanced | experiments_balanced_phase1 | experiments_balanced_phase2 |

---

## 7. TESTES ESTATISTICOS INCLUIDOS

### 7.1 Friedman Test
- Compara todos os modelos simultaneamente
- p < 0.05 indica diferenca significativa entre pelo menos dois modelos

### 7.2 Nemenyi Post-hoc Test
- Compara pares de modelos apos Friedman
- Com correcao de Bonferroni

### 7.3 Wilcoxon Signed-Rank Test
- Comparacao pareada GBML vs cada modelo
- Indica se GBML e significativamente melhor/pior

### 7.4 Critical Difference Diagram
- Visualiza rankings e diferencias significativas
- Gera `critical_difference_diagram.png`

---

## 8. PROBLEMAS CONHECIDOS E SOLUCOES

### 8.1 ACDWM em Multiclasse
**Problema**: ACDWM so suporta problemas binarios
**Solucao**: Retorna gmean=0.0 automaticamente para n_classes > 2

### 8.2 ROSE em Multiclasse
**Solucao**: Usa evaluator diferente:
- Binario: WindowAUCImbalancedPerformanceEvaluator
- Multiclasse: WindowAUCMultiClassImbalancedPerformanceEvaluator

### 8.3 Timeouts
**Problema**: Alguns modelos podem demorar muito
**Solucao**: MODEL_TIMEOUT configurado por modelo (ate 1800s para ERulesD2S)

---

## 9. CHECKLIST PARA NOVOS EXPERIMENTOS

- [ ] Alterar CHUNK_SIZE (se diferente de 1000)
- [ ] Alterar base_dir em BATCH_CONFIG para todos os 7 batches
- [ ] Verificar se GBML ja foi executado nos novos diretorios
- [ ] Verificar existencia de chunk_data/ em cada dataset
- [ ] Executar com USE_CACHE = False para garantir re-execucao
- [ ] Salvar outputs em diretorio separado para cada experimento

---

**Autor**: Claude Code
**Status**: REVISAO COMPLETA
