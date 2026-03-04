# Instrucoes para Execucao do CDCMS nos Experimentos

**Data:** 2026-01-26
**Versao:** 2.0

---

## Visao Geral

Este documento descreve como executar o CDCMS nos experimentos chunk_500 e chunk_1000, salvando resultados compativeis com o formato EGIS para comparacao posterior.

---

## Pre-requisitos

Antes de executar a Parte 7, certifique-se de que:

1. **Partes 1-6 do Setup_CDCMS_CIL.ipynb foram executadas**
   - Java e Maven instalados
   - CDCMS.CIL clonado e compilado
   - MOA-dependencies.jar baixado
   - CDCMSEvaluator.java compilado

2. **JARs necessarios existem:**
   - `/content/cdcms_jars/cdcms_cil_final.jar`
   - `/content/rose_jars/MOA-dependencies.jar`

3. **Google Drive montado com os dados:**
   - Path: `/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid`

---

## Estrutura de Arquivos

```
CDCMS_PARTE7_COMPLETA.py     # Arquivo completo com todas as celulas
CDCMS_INSTRUCOES_EXECUCAO.md # Este arquivo
```

---

## Como Usar no Google Colab

### Opcao 1: Copiar arquivo completo

1. Abra o notebook `Setup_CDCMS_CIL.ipynb` no Colab
2. Apos executar as Partes 1-6, crie uma nova celula
3. Copie todo o conteudo de `CDCMS_PARTE7_COMPLETA.py`
4. Execute a celula (vai carregar todas as funcoes)
5. Siga os passos descritos no output

### Opcao 2: Copiar celula por celula

Copie cada secao `# CELULA 7.X` para uma celula separada no Colab.

---

## Fluxo de Execucao Recomendado

### Passo 1: Setup (CELULA 7.1)
```python
# Executa automaticamente ao carregar o arquivo
# Verifica paths, JARs e estrutura de dados
```

### Passo 2: Teste Rapido (CELULA 7.5)
```python
# Testa em 3 datasets para validar que tudo funciona
# SEA_Abrupt_Simple, HYPERPLANE_Abrupt_Simple, AGRAWAL_Abrupt_Simple_Mild
```
**Tempo estimado:** ~5 minutos

### Passo 3: Executar chunk_500 (CELULA 7.6)
```python
# Opcao A: Todos os batches de uma vez
results_chunk500 = run_cdcms_all_batches('chunk_500', skip_existing=True)

# Opcao B: Batch por batch (recomendado para sessoes longas)
results_b1 = run_cdcms_batch('chunk_500', 'batch_1')
results_b2 = run_cdcms_batch('chunk_500', 'batch_2')
results_b3 = run_cdcms_batch('chunk_500', 'batch_3')
```
**Tempo estimado:** ~2-3 horas (85 datasets)

### Passo 4: Executar chunk_1000 (CELULA 7.7)
```python
# Opcao A: Todos os batches de uma vez
results_chunk1000 = run_cdcms_all_batches('chunk_1000', skip_existing=True)

# Opcao B: Batch por batch
results_b1 = run_cdcms_batch('chunk_1000', 'batch_1')
results_b2 = run_cdcms_batch('chunk_1000', 'batch_2')
results_b3 = run_cdcms_batch('chunk_1000', 'batch_3')
results_b4 = run_cdcms_batch('chunk_1000', 'batch_4')
```
**Tempo estimado:** ~2-3 horas (93 datasets)

### Passo 5: Comparar com EGIS (CELULA 7.8)
```python
# Gerar tabela comparativa
generate_comparison_table('chunk_500')
generate_comparison_table('chunk_1000')
```

---

## Metricas Calculadas

Para cada dataset, o CDCMS calcula duas perspectivas:

### 1. Prequential (padrao MOA)
- **prequential_gmean**: G-Mean do chunk completo
- **prequential_f1**: F1-Score macro
- **prequential_f1_weighted**: F1-Score weighted
- **prequential_accuracy**: Accuracy

### 2. Holdout (comparavel com EGIS)
- **holdout_gmean**: G-Mean das primeiras 100 instancias do chunk
- **holdout_f1**: F1-Score das primeiras 100 instancias
- **holdout_f1_weighted**: F1-weighted das primeiras 100 instancias
- **holdout_accuracy**: Accuracy das primeiras 100 instancias

**Nota:** Holdout so e calculado para chunks >= 1 (chunk 0 e treinamento inicial).

---

## Estrutura de Saida

Para cada dataset, os resultados sao salvos em:
```
experiments_unified/chunk_500/batch_X/DATASET/cdcms_results/
├── chunk_metrics.json      # Metricas por chunk (prequential + holdout)
├── run_config.json         # Configuracao da execucao
└── cdcms_raw_output.csv    # Saida bruta do CDCMSEvaluator
```

### Formato do chunk_metrics.json
```json
[
  {
    "chunk": 0,
    "instances_in_chunk": 500,
    "prequential_gmean": 0.72,
    "prequential_f1": 0.71,
    "prequential_f1_weighted": 0.70,
    "prequential_accuracy": 0.75,
    "holdout_gmean": null,
    "holdout_f1": null,
    "note": "Chunk 0 - modelo iniciando do zero"
  },
  {
    "chunk": 1,
    "instances_in_chunk": 500,
    "prequential_gmean": 0.88,
    "prequential_f1": 0.87,
    "prequential_f1_weighted": 0.86,
    "prequential_accuracy": 0.89,
    "holdout_gmean": 0.90,
    "holdout_f1": 0.89,
    "holdout_f1_weighted": 0.88,
    "holdout_accuracy": 0.91,
    "holdout_window_size": 100
  }
]
```

---

## Comparacao EGIS vs CDCMS

### Alinhamento de Metricas

| EGIS | CDCMS | Comparacao |
|------|-------|------------|
| test_gmean (chunk i) | holdout_gmean (chunk i) | Justa |
| Modelo treinado em chunk i-1 | Primeiras 100 inst. do chunk i | Similar |

### Tabela de Saida
```
comparison_egis_cdcms_chunk_500.csv
comparison_egis_cdcms_chunk_1000.csv
```

Colunas:
- dataset, batch
- egis_gmean
- cdcms_prequential
- cdcms_holdout
- diff_prequential (CDCMS - EGIS)
- diff_holdout (CDCMS - EGIS)

---

## Troubleshooting

### Erro: "CDCMS_JAR nao encontrado"
- Execute as celulas 4.1 e 4.2 do Setup_CDCMS_CIL.ipynb

### Erro: "MOA_DEPS_JAR nao encontrado"
- Execute a celula 3.3 do Setup_CDCMS_CIL.ipynb

### Erro: "CDCMSEvaluator not found"
- Execute a celula 5.1 do Setup_CDCMS_CIL.ipynb

### Timeout em datasets grandes
- Aumente CDCMS_TIMEOUT (padrao: 600s)
- Ou execute o dataset individualmente com timeout maior

### Sessao do Colab expirando
- Use `skip_existing=True` para continuar de onde parou
- Execute batch por batch ao inves de todos de uma vez

---

## Datasets Excluidos

Os seguintes datasets sao excluidos automaticamente:
- **IntelLabSensors**: Muitas classes, pode ter NaN
- **PokerHand**: Muito lento

Para incluir, remova de `EXCLUDE_DATASETS` na CELULA 7.1.

---

## Configuracoes Ajustaveis

```python
# Na CELULA 7.1:
HOLDOUT_WINDOW_SIZE = 100  # Primeiras N instancias para holdout
CDCMS_TIMEOUT = 600        # Timeout em segundos
EXCLUDE_DATASETS = [...]   # Datasets a excluir
```

---

## Resumo dos Comandos Principais

```python
# Teste rapido
result = run_cdcms_on_dataset("SEA_Abrupt_Simple", "chunk_500", "batch_1")

# Executar um batch
results = run_cdcms_batch('chunk_500', 'batch_1')

# Executar todos os batches
results = run_cdcms_all_batches('chunk_500')

# Comparar com EGIS
generate_comparison_table('chunk_500')
```

---

**Criado por:** Claude Code
**Baseado em:** Plano de Acao CDCMS 2026-01-26
