# INSTRUCOES PARA ATUALIZAR Execute_Comparative_All_Experiments.ipynb

## Data: 2026-01-26

---

## RESUMO DAS MUDANCAS

Este ajuste permite que o notebook Execute_Comparative_All_Experiments.ipynb trabalhe com:

1. **Experimentos Unified** (novos):
   - `exp_unified_500` - chunk 500 sem penalty
   - `exp_unified_500_penalty` - chunk 500 com penalty
   - `exp_unified_1000` - chunk 1000 sem penalty
   - `exp_unified_1000_penalty` - chunk 1000 com penalty

2. **Experimentos Legados** (mantidos para compatibilidade):
   - `exp_a_chunk1000`, `exp_b_chunk2000`, `exp_c_balanced`

---

## ESTRUTURA DOS DADOS

```
unified_chunks/               <- Dados brutos (usados por todos os experimentos unified)
  chunk_500/
    SEA_Abrupt_Simple/
      chunk_0.csv, chunk_1.csv, ...
  chunk_1000/
  chunk_2000/

experiments_unified/          <- Resultados EGIS e CDCMS
  chunk_500/                  <- EGIS SEM penalty + CDCMS
    batch_1/
      SEA_Abrupt_Simple/
        run_1/
          chunk_metrics.json  <- Resultados EGIS
        cdcms_results/
          chunk_metrics.json  <- Resultados CDCMS
  chunk_500_penalty/          <- EGIS COM penalty (CDCMS em chunk_500/)
  chunk_1000/
  chunk_1000_penalty/
```

---

## ARQUIVOS CRIADOS

| Arquivo | Celula no Colab | Descricao |
|---------|-----------------|-----------|
| CELULA_2_1_EXEC_COMPARATIVE_CONFIG.py | CELULA 2.1 | Configuracao dos experimentos |
| CELULA_2_2_EXEC_COMPARATIVE_DETECT.py | CELULA 2.2 | Deteccao de datasets |
| CELULA_3_1_EXEC_COMPARATIVE_LOAD.py | CELULA 3.1 | Funcoes de carregamento |
| **CELULA_3_METRICAS_COMPLETAS.py** | CELULAs 3.3-3.6 | **METRICAS COMPLETAS (f1, f1_weighted)** |
| CELULA_4_1_EXEC_COMPARATIVE_MAIN.py | CELULA 4.1 | Loop principal de execucao |
| CELULA_4_2_EXEC_COMPARATIVE_SAVE.py | CELULA 4.2 | Salvamento de resultados |

---

## ORDEM DE ATUALIZACAO

### PASSO 1: Backup
1. No Colab: File > Save a copy in Drive
2. Renomeie para: `Execute_Comparative_All_Experiments_BACKUP.ipynb`

### PASSO 2: Atualizar CELULAs
Copie o conteudo de cada arquivo .py para a celula correspondente:

| Arquivo | Como encontrar a celula |
|---------|------------------------|
| CELULA_2_1_EXEC_COMPARATIVE_CONFIG.py | Procure "EXPERIMENT_CONFIGS" |
| CELULA_2_2_EXEC_COMPARATIVE_DETECT.py | Procure "detect_available_datasets" |
| CELULA_3_1_EXEC_COMPARATIVE_LOAD.py | Procure "load_chunks_from_gbml" |
| CELULA_4_1_EXEC_COMPARATIVE_MAIN.py | Procure "ALL_RESULTS = []" |
| CELULA_4_2_EXEC_COMPARATIVE_SAVE.py | Procure "df_results = pd.DataFrame" |

### PASSO 3: Salvar
Ctrl+S ou File > Save

---

## CELULAS QUE NAO PRECISAM ALTERACAO

- CELULA 1.1 - Montar Drive
- CELULA 1.2 - Instalar Java/Maven
- CELULA 1.3 - Instalar dependencias Python
- CELULA 1.4 - Baixar JARs do ROSE
- CELULA 1.5 - Clonar ACDWM
- CELULA 1.6 - Setup ERulesD2S
- CELULA 1.7 - Verificacao do Ambiente
- CELULA 3.2 - Funcoes ARFF e ROSE (run_rose_original, run_rose_chunk_eval)
- CELULA 5.x - Sincronizar com Drive

**IMPORTANTE**: As CELULAs 3.3, 3.4, 3.5 e 3.6 devem ser substituidas pelo conteudo de `CELULA_3_METRICAS_COMPLETAS.py` para ter metricas consistentes com CDCMS.

---

## METRICAS SALVAS (Consistente com CDCMS)

| Modelo | test_gmean | test_accuracy | test_f1 | test_f1_weighted |
|--------|:----------:|:-------------:|:-------:|:----------------:|
| CDCMS | OK | OK | OK | OK |
| EGIS | OK | -- | OK | -- |
| HAT/ARF/SRP | OK | OK | OK | OK |
| ACDWM | OK | OK | OK | OK |
| ERulesD2S | proxy* | OK | proxy* | proxy* |
| ROSE | OK | OK | -- | -- |

\* ERulesD2S usa accuracy como proxy (limitacao do output do MOA)

---

## CONFIGURACOES IMPORTANTES (CELULA 2.1)

```python
# Opcao 1: Executar TODOS os experimentos unified
EXPERIMENTS_TO_RUN = [
    'exp_unified_500',
    'exp_unified_500_penalty',
    'exp_unified_1000',
    'exp_unified_1000_penalty',
]

# Opcao 2: Apenas chunk_500 (teste rapido)
EXPERIMENTS_TO_RUN = [
    'exp_unified_500',
    'exp_unified_500_penalty',
]

# Opcao 3: Apenas sem penalty
EXPERIMENTS_TO_RUN = [
    'exp_unified_500',
    'exp_unified_1000',
]
```

---

## MODELOS NA COMPARACAO

| Modelo | Fonte | Executado? |
|--------|-------|------------|
| EGIS | experiments_unified/chunk_XXX/run_1/chunk_metrics.json | Carregado de cache |
| EGIS_Penalty | experiments_unified/chunk_XXX_penalty/run_1/chunk_metrics.json | Carregado de cache |
| CDCMS | experiments_unified/chunk_XXX/cdcms_results/chunk_metrics.json | Carregado de cache |
| ROSE_Original | Java/MOA | Executado se nao houver cache |
| ROSE_ChunkEval | Java/MOA | Executado se nao houver cache |
| HAT | River | Executado se nao houver cache |
| ARF | River | Executado se nao houver cache |
| SRP | River | Executado se nao houver cache |
| ACDWM | Python | Executado se nao houver cache (so binario!) |
| ERulesD2S | Java/JCLEC4 | Desabilitado por padrao (lento) |

---

## LOGICA DE REUTILIZACAO

Para experimentos COM penalty (`exp_unified_500_penalty`, `exp_unified_1000_penalty`):
- EGIS_Penalty: carregado da pasta com penalty
- Modelos comparativos: reutilizados da versao SEM penalty

Isso evita executar os mesmos modelos duas vezes (os dados de entrada sao identicos).

---

## DATASETS MULTICLASSE (CDCMS e ACDWM = N/A)

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

## ORDEM DE EXECUCAO NO COLAB

```
1. CELULA 1.1 - Montar Drive
2. CELULA 1.2 - Instalar Java/Maven
3. CELULA 1.3 - Instalar dependencias Python
4. CELULA 1.4 - Baixar JARs do ROSE
5. CELULA 1.5 - Clonar ACDWM
6. CELULA 1.6 - Setup ERulesD2S
7. CELULA 1.7 - Verificacao do Ambiente

8. CELULA 2.1 - Configuracoes (ATUALIZADA)
9. CELULA 2.2 - Deteccao de datasets (ATUALIZADA)

10. CELULA 3.1 - Funcoes carregamento (ATUALIZADA)
11. CELULA 3.2 - Funcoes ROSE
12. CELULA 3.3 - Funcoes River
13. CELULA 3.4 - Funcoes ACDWM
14. CELULA 3.5 - Funcoes salvar
15. CELULA 3.6 - Funcoes ERulesD2S

16. CELULA 4.1 - Execucao principal (ATUALIZADA)
17. CELULA 4.2 - Salvar resultados (ATUALIZADA)

18. CELULA 5.x - Sincronizar com Drive
```

---

## RESULTADOS ESPERADOS

### Arquivos gerados em `comparison_results/`:
```
all_models_unified_results_latest.csv     <- Todos os resultados
all_models_unified_results_20260126_*.csv <- Com timestamp
comparative_results_exp_unified_500.csv
comparative_results_exp_unified_500_penalty.csv
comparative_results_exp_unified_1000.csv
comparative_results_exp_unified_1000_penalty.csv
pivot_gmean_chunk_500.csv                 <- Para analise estatistica
pivot_gmean_chunk_1000.csv
rankings_chunk_500.csv
rankings_chunk_1000.csv
```

### Colunas do CSV:
- experiment: exp_unified_500, exp_unified_500_penalty, etc.
- batch: batch_1, batch_2, etc.
- dataset: nome do dataset
- model: EGIS, EGIS_Penalty, CDCMS, HAT, etc.
- gmean: G-Mean do modelo
- status: OK, CACHED, REUSED, N/A, NOT_FOUND, ERROR
- chunk_size: 500, 1000
- penalty: True/False

---

## DEBUG

Se ocorrer erro, adicione no inicio de CELULA 4.1:
```python
print(f"UNIFIED_CHUNKS_DIR existe: {UNIFIED_CHUNKS_DIR.exists()}")
print(f"EXPERIMENTS_UNIFIED_DIR existe: {EXPERIMENTS_UNIFIED_DIR.exists()}")

# Verificar estrutura
for exp_name in EXPERIMENTS_TO_RUN:
    config = EXPERIMENT_CONFIGS[exp_name]
    data_dir = config.get('data_dir', 'chunk_500')
    results_dir = config.get('results_dir', data_dir)

    data_path = UNIFIED_CHUNKS_DIR / data_dir
    results_path = EXPERIMENTS_UNIFIED_DIR / results_dir

    print(f"{exp_name}:")
    print(f"  data: {data_path.exists()}")
    print(f"  results: {results_path.exists()}")
```

---

## CONTATO

Em caso de problemas, verifique:
1. Se os diretorios unified_chunks e experiments_unified existem
2. Se os arquivos chunk_metrics.json existem nos paths esperados
3. Se WORK_DIR esta apontando para o diretorio correto no Drive
