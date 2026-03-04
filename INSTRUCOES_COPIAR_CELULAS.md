# INSTRUCOES PARA ATUALIZAR O NOTEBOOK NO COLAB

## Data: 2026-01-26 (Atualizado com suporte a PENALTY)

---

## ESTRUTURA IDENTIFICADA

```
experiments_unified/
  chunk_500/           -> EGIS SEM penalidade (foco em performance)
  chunk_500_penalty/   -> EGIS COM penalidade (foco em interpretabilidade)
  chunk_1000/          -> EGIS SEM penalidade
  chunk_1000_penalty/  -> EGIS COM penalidade

unified_chunks/        -> Dados brutos (mesmos para com/sem penalty)
  chunk_500/
  chunk_1000/
  chunk_2000/
```

---

## ARQUIVOS CRIADOS

```
CELULA_2_1_CONFIG_UNIFIED.py      -> Substituir CELULA 2.1
CELULA_3_4_LOAD_UNIFIED.py        -> Substituir CELULA 3.4
CELULA_4_1_MAIN_LOOP_UNIFIED.py   -> Substituir CELULA 4.1
CELULA_4_2_SAVE_RESULTS_UNIFIED.py -> Substituir CELULA 4.2
```

---

## LOGICA DE EXECUCAO

1. **Para configs SEM penalty** (chunk_500, chunk_1000):
   - Carrega EGIS (modelo: GBML)
   - Executa ROSE, HAT, ARF, SRP, ACDWM
   - Carrega CDCMS de cache
   - Carrega ERulesD2S de cache

2. **Para configs COM penalty** (chunk_500_penalty, chunk_1000_penalty):
   - Carrega EGIS com penalty (modelo: GBML_Penalty)
   - Reutiliza resultados dos modelos comparativos (da versao sem penalty)

**Resultado**: Os modelos comparativos sao executados apenas UMA vez por dataset (na versao sem penalty), mas os resultados aparecem em ambas as comparacoes.

---

## ORDEM DE ATUALIZACAO

### PASSO 1: Backup
1. No Colab: File > Save a copy in Drive
2. Renomeie para: `Execute_All_Comparative_Models_BACKUP.ipynb`

### PASSO 2: Atualizar CELULAs
Copie o conteudo de cada arquivo .py para a celula correspondente:

| Arquivo | Celula no Colab | Como encontrar |
|---------|-----------------|----------------|
| CELULA_2_1_CONFIG_UNIFIED.py | CELULA 2.1 | Procure "BATCH_CONFIG" |
| CELULA_3_4_LOAD_UNIFIED.py | CELULA 3.4 | Procure "load_chunks_from_gbml" |
| CELULA_4_1_MAIN_LOOP_UNIFIED.py | CELULA 4.1 | Procure "EXECUTAR TODOS OS MODELOS" |
| CELULA_4_2_SAVE_RESULTS_UNIFIED.py | CELULA 4.2 | Procure "Salvar Resultados" |

### PASSO 3: Salvar
Ctrl+S ou File > Save

---

## CONFIGURACOES IMPORTANTES (CELULA 2.1)

```python
# Opcao 1: Executar TUDO (recomendado para experimento final)
CHUNK_SIZES_TO_RUN = ['chunk_500', 'chunk_500_penalty', 'chunk_1000', 'chunk_1000_penalty']

# Opcao 2: Apenas sem penalidade
CHUNK_SIZES_TO_RUN = ['chunk_500', 'chunk_1000']

# Opcao 3: Apenas chunk_500 (para teste rapido)
CHUNK_SIZES_TO_RUN = ['chunk_500', 'chunk_500_penalty']

# Opcao 4: Teste minimo
CHUNK_SIZES_TO_RUN = ['chunk_500']
```

---

## MODELOS NA COMPARACAO

| Modelo | Tipo | Origem |
|--------|------|--------|
| GBML | EGIS sem penalty | experiments_unified/chunk_XXX/ |
| GBML_Penalty | EGIS com penalty | experiments_unified/chunk_XXX_penalty/ |
| CDCMS | Cache | experiments_unified/chunk_XXX/cdcms_results/ |
| ROSE_Original | Executa | Java/MOA |
| ROSE_ChunkEval | Executa | Java/MOA |
| HAT | Executa | River |
| ARF | Executa | River |
| SRP | Executa | River |
| ACDWM | Executa | Python |
| ERulesD2S | Cache | Java/JCLEC4 |

---

## ORDEM DE EXECUCAO NO COLAB

```
1. CELULA 1.1 - Montar Drive
2. CELULA 1.3 - Instalar Java/Maven
3. CELULA 1.4 - Instalar dependencias Python
4. CELULA 1.5 - Baixar JARs do ROSE
5. CELULA 1.6 - Clonar ACDWM
6. CELULA 1.7 - Setup ERulesD2S
7. CELULA 2.1 - Configuracoes (ATUALIZADA)
8. CELULA 3.1 - Funcoes ROSE
9. CELULA 3.2 - Funcoes River
10. CELULA 3.3 - Funcoes ACDWM
11. CELULA 3.4 - Funcoes carregamento (ATUALIZADA)
12. CELULA 4.1 - Execucao principal (ATUALIZADA)
13. CELULA 4.2 - Salvar resultados (ATUALIZADA)
14. CELULA 5.x - Analise estatistica (sem alteracao)
```

---

## RESULTADOS ESPERADOS

### Arquivos gerados em `comparison_results/`:
```
all_models_unified_results_latest.csv     <- Todos os resultados
all_models_chunk_500_results.csv          <- chunk_500 sem penalty
all_models_chunk_500_penalty_results.csv  <- chunk_500 com penalty
all_models_chunk_1000_results.csv         <- chunk_1000 sem penalty
all_models_chunk_1000_penalty_results.csv <- chunk_1000 com penalty
pivot_gmean_chunk_500.csv                 <- Para analise estatistica
pivot_gmean_chunk_1000.csv
rankings_chunk_500.csv
rankings_chunk_1000.csv
```

### Colunas do CSV:
- chunk_size: chunk_500, chunk_500_penalty, etc.
- batch: batch_1, batch_2, etc.
- dataset: nome do dataset
- model: GBML, GBML_Penalty, CDCMS, HAT, etc.
- gmean: G-Mean do modelo
- status: OK, CACHED, N/A, NOT_FOUND, etc.
- penalty: True/False

---

## CELULAS QUE NAO PRECISAM ALTERACAO

- CELULA 1.x (Setup)
- CELULA 3.1 (ROSE)
- CELULA 3.2 (River)
- CELULA 3.3 (ACDWM)
- CELULA 5.x (Estatistica)
- CELULA 6.x (Relatorios)

---

## DEBUG

Se ocorrer erro, adicione no inicio de CELULA 4.1:
```python
print(f"UNIFIED_CHUNKS_DIR existe: {UNIFIED_CHUNKS_DIR.exists()}")
print(f"EXPERIMENTS_DIR existe: {EXPERIMENTS_DIR.exists()}")

# Verificar estrutura
for cs in CHUNK_SIZES_TO_RUN:
    config = CHUNK_CONFIGS[cs]
    data_path = UNIFIED_CHUNKS_DIR / config['data_dir']
    results_path = EXPERIMENTS_DIR / config['results_dir']
    print(f"{cs}: data={data_path.exists()}, results={results_path.exists()}")
```
