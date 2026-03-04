# GUIA DE EXECUCAO - BATCH 1 TESTE

## OBJETIVO

Testar a estrategia de 2 datasets por execucao, validando que:
1. 2 datasets cabem no limite de 24h do Colab Pro
2. Qualidade dos resultados e mantida
3. Estrutura de salvamento funciona corretamente

---

## PREPARACAO COMPLETA

### Arquivos Criados

**Configs (3 execucoes)**:
- `configs/config_batch_1_exec_1.yaml` - SEA + AGRAWAL (22h est.)
- `configs/config_batch_1_exec_2.yaml` - RBF + HYPERPLANE (22h est.)
- `configs/config_batch_1_exec_3.yaml` - STAGGER (11h est.)

**Scripts**:
- `run_batch_1_exec_1_colab.py` - Script automatizado para Colab
- `fix_config_paths.py` - Correcao de caminhos (JA EXECUTADO)

**Documentacao**:
- `ESTRATEGIA_2_DATASETS.md` - Estrategia completa
- `ANALISE_BATCH_1_LOG.md` - Analise do log anterior
- `ANALISE_CONFIG_BATCH_1.md` - Validacao do config
- `PROBLEMAS_E_SOLUCOES_BATCH.md` - Problemas identificados

---

## TESTE INICIAL: EXECUCAO 1

### Datasets
1. SEA_Abrupt_Simple (mais rapido, validacao basica)
2. AGRAWAL_Abrupt_Simple_Severe (drift severo, validacao robusta)

### Tempo Estimado
- SEA: 9-10 horas
- AGRAWAL: 10-12 horas
- **Total**: 19-22 horas

### Validacoes Esperadas
- [ ] Tempo total < 24 horas
- [ ] SEA executado com sucesso
- [ ] AGRAWAL executado com sucesso
- [ ] 6 chunks gerados por dataset
- [ ] Plots criados
- [ ] G-mean similar ao teste anterior (~0.70-0.90)

---

## PASSO A PASSO NO GOOGLE COLAB

### Opcao 1: Usando o Script Python (RECOMENDADO)

```python
# Nova celula no Colab

# 1. Fazer upload do script
from google.colab import files
uploaded = files.upload()  # Selecione run_batch_1_exec_1_colab.py

# 2. Executar
!python run_batch_1_exec_1_colab.py
```

### Opcao 2: Passo a Passo Manual

#### Celula 1: Montar Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### Celula 2: Copiar Repositorio
```python
import os

# Copiar do Drive para /content (mais rapido)
!cp -r /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid /content/

# Mudar para diretorio
os.chdir('/content/DSL-AG-hybrid')

# Verificar
!pwd
!ls -la
```

#### Celula 3: Instalar Dependencias
```python
!pip install -r requirements.txt
```

#### Celula 4: Configurar Execucao
```python
# Copiar config correto
!cp configs/config_batch_1_exec_1.yaml config.yaml

# Verificar datasets
!grep -A 5 "drift_simulation_experiments" config.yaml
```

#### Celula 5: Executar
```python
import time
from datetime import datetime

# Definir log
log_file = "/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/batch_1_exec_1.log"

# Timestamp inicio
print(f"Inicio: {datetime.now()}")

# Executar
!python main.py > {log_file} 2>&1

# Timestamp fim
print(f"Fim: {datetime.now()}")
```

#### Celula 6: Monitorar (Em outra celula, enquanto executa)
```python
# Executar esta celula periodicamente para ver progresso

# Ultimas 50 linhas do log
!tail -n 50 /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/batch_1_exec_1.log

# Tempo estimado
!grep -i "chunk.*final" /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/batch_1_exec_1.log | tail -5
```

---

## MONITORAMENTO DURANTE EXECUCAO

### Comandos Uteis (executar em celulas separadas)

#### Ver progresso geral
```python
!grep -E "(CHUNK|Dataset|G-mean)" /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/batch_1_exec_1.log | tail -20
```

#### Ver tempo por chunk
```python
!grep "Tempo total:" /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/batch_1_exec_1.log
```

#### Ver erros
```python
!grep -i "error" /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/batch_1_exec_1.log | tail -10
```

#### Verificar chunks gerados
```python
!ls -lh /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/experiments_6chunks_phase1_gbml/batch_1/*/run_1/chunk_data/
```

---

## VALIDACAO POS-EXECUCAO

### 1. Verificar Estrutura de Arquivos

```python
base_dir = "/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/experiments_6chunks_phase1_gbml/batch_1"

datasets = ["SEA_Abrupt_Simple", "AGRAWAL_Abrupt_Simple_Severe"]

for dataset in datasets:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print('='*60)

    # Verificar diretorio
    dataset_dir = f"{base_dir}/{dataset}/run_1"

    # Chunks
    !ls -lh {dataset_dir}/chunk_data/*.csv | wc -l

    # Plots
    !ls -lh {dataset_dir}/plots/*.png | wc -l

    # Arquivos importantes
    !ls -lh {dataset_dir}/*.json {dataset_dir}/*.pkl {dataset_dir}/*.txt
```

### 2. Analisar Resultados

```python
import json
import pandas as pd

for dataset in datasets:
    summary_file = f"{base_dir}/{dataset}/run_1/experiment_summary.json"

    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print('='*60)

        # Metricas principais
        print(f"Test G-mean: {summary.get('test_gmean', 'N/A')}")
        print(f"Test F1: {summary.get('test_f1', 'N/A')}")
        print(f"Train G-mean: {summary.get('train_gmean', 'N/A')}")
        print(f"Chunks processados: {summary.get('chunks_processed', 'N/A')}")
```

### 3. Comparar com Teste Anterior

**Teste anterior** (RBF_Drift_Recovery, 1 dataset, 11h):
- Avg Test G-mean: 0.7267
- Std Test G-mean: 0.2046

**Esperado** (SEA + AGRAWAL, 2 datasets, 22h):
- SEA: G-mean ~0.80-0.90 (normalmente mais alto)
- AGRAWAL: G-mean ~0.70-0.85 (drift severo)

---

## CRITERIOS DE SUCESSO

### TESTE PASSA SE:
- [x] Tempo total < 24 horas
- [x] Ambos datasets executaram sem erros
- [x] 6 chunks gerados por dataset (12 train + 12 test = 24 arquivos)
- [x] Plots gerados (minimo 3 por dataset)
- [x] G-mean dentro do esperado (>0.65)
- [x] Estrutura de pastas correta
- [x] Log completo salvo

### TESTE FALHA SE:
- Tempo > 24 horas (sessao Colab expirou)
- Erro durante execucao
- Chunks incompletos
- G-mean muito baixo (<0.5)
- Resultados nao salvos

---

## PROXIMOS PASSOS APOS TESTE

### Se TESTE PASSAR:

1. **Executar Execucao 2**
   - Datasets: RBF_Abrupt_Severe + HYPERPLANE_Abrupt_Simple
   - Usar: `run_batch_1_exec_2_colab.py` (criar similar)
   - Tempo: 22h estimadas

2. **Executar Execucao 3**
   - Dataset: STAGGER_Abrupt_Chain
   - Usar: `run_batch_1_exec_3_colab.py` (criar similar)
   - Tempo: 11h estimadas

3. **Consolidar Batch 1**
   - 5 datasets completos
   - Validar integridade
   - Gerar relatorio consolidado

4. **Escalar para Todos os Batches**
   - Gerar configs para batches 2-12
   - Executar em paralelo com multiplas contas
   - Seguir mesma estrategia (3 execucoes por batch)

### Se TESTE FALHAR:

1. **Analisar causa**
   - Tempo excedido?
   - Erro de execucao?
   - Problema de memoria?

2. **Ajustar estrategia**
   - Reduzir para 1 dataset por execucao?
   - Ajustar parametros (nao recomendado)?
   - Usar servico pago diferente?

3. **Re-testar**
   - Executar novamente com ajustes
   - Validar nova estrategia

---

## CHECKLIST PRE-EXECUCAO

Antes de executar, verificar:
- [ ] Google Colab Pro ativo
- [ ] Google Drive montado
- [ ] Repositorio no Drive atualizado
- [ ] Config config_batch_1_exec_1.yaml presente em configs/
- [ ] Script run_batch_1_exec_1_colab.py disponivel
- [ ] Espaco no Drive suficiente (~20GB)
- [ ] Tempo disponivel (deixar rodando 24h ininterruptas)

---

## ESTIMATIVA DE TEMPO COMPLETO

### Batch 1 Completo:
- Execucao 1: 22h (SEA + AGRAWAL)
- Execucao 2: 22h (RBF + HYPERPLANE)
- Execucao 3: 11h (STAGGER)
- **Total**: 55 horas em 3 execucoes

### Com Multiplas Contas (Paralelo):
- Executar 3 execucoes simultaneamente
- **Total**: 22 horas (batch completo)

---

## CONTATO E SUPORTE

Se houver problemas:
1. Verificar log completo
2. Revisar documentacao: PROBLEMAS_E_SOLUCOES_BATCH.md
3. Comparar com log anterior (batch_1.log)
4. Ajustar estrategia conforme necessario

---

**Data**: 2025-11-17
**Status**: PRONTO PARA TESTE
**Proximo passo**: Executar run_batch_1_exec_1_colab.py no Google Colab
