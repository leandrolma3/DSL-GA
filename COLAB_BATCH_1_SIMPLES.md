# EXECUCAO SIMPLES - BATCH 1 (Direto do Drive)

Execute estas celulas no Google Colab. SEM copia de arquivos!

---

## CELULA 1: Setup Inicial

```python
# Montar Drive
from google.colab import drive
drive.mount('/content/drive')

# Navegar para o diretorio (trabalhar direto no Drive)
%cd /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid

# Instalar dependencias
!pip install -q -r requirements.txt

print("\n✓ Setup completo!")
```

---

## BATCH 1 - EXECUCAO 1: SEA + AGRAWAL (~22h)

```python
# Configurar
!cp configs/config_batch_1_exec_1.yaml config.yaml

# Verificar datasets
print("Datasets configurados:")
!grep "drift_simulation_experiments:" config.yaml -A 3

# EXECUTAR (demora ~22 horas)
!python main.py 2>&1 | tee batch_1_exec_1.log

print("\n✓ Execucao 1 concluida!")
```

---

## BATCH 1 - EXECUCAO 2: RBF + HYPERPLANE (~22h)

```python
# Configurar
!cp configs/config_batch_1_exec_2.yaml config.yaml

# Verificar datasets
print("Datasets configurados:")
!grep "drift_simulation_experiments:" config.yaml -A 3

# EXECUTAR (demora ~22 horas)
!python main.py 2>&1 | tee batch_1_exec_2.log

print("\n✓ Execucao 2 concluida!")
```

---

## BATCH 1 - EXECUCAO 3: STAGGER (~11h)

```python
# Configurar
!cp configs/config_batch_1_exec_3.yaml config.yaml

# Verificar datasets
print("Datasets configurados:")
!grep "drift_simulation_experiments:" config.yaml -A 3

# EXECUTAR (demora ~11 horas)
!python main.py 2>&1 | tee batch_1_exec_3.log

print("\n✓ Execucao 3 concluida!")
```

---

## ANALISES (Apos cada execucao)

```python
# Executar apos cada batch
!python analyze_concept_difference.py
!python generate_plots.py
!python rule_diff_analyzer.py

print("\n✓ Analises concluidas!")
```

---

## MONITORAR PROGRESSO (Executar em outra aba)

```python
# Ver progresso da execucao atual
!tail -n 30 batch_1_exec_1.log  # ou exec_2, exec_3

# Ver chunks finalizados
!grep -i "chunk.*final" batch_1_exec_1.log | tail -5

# Ver tempo por chunk
!grep "Tempo total:" batch_1_exec_1.log
```

---

## VERIFICAR RESULTADOS

```python
# Listar resultados gerados
!ls -lh experiments_6chunks_phase1_gbml/batch_1/

# Ver detalhes de um dataset
!ls -lh experiments_6chunks_phase1_gbml/batch_1/SEA_Abrupt_Simple/run_1/

# Ver chunks gerados
!ls -lh experiments_6chunks_phase1_gbml/batch_1/SEA_Abrupt_Simple/run_1/chunk_data/
```

---

## RESUMO DE USO

### Ordem de Execucao:

1. **CELULA 1**: Setup (5 minutos)
2. **Executar uma das execucoes** (22h ou 11h):
   - Execucao 1: SEA + AGRAWAL
   - Execucao 2: RBF + HYPERPLANE
   - Execucao 3: STAGGER
3. **Analises**: Apos cada execucao
4. **Repetir** para as 3 execucoes

### Vantagens deste metodo:

- **SEM copia de arquivos** (economiza tempo e espaco)
- **Trabalha direto no Drive** (resultados ja salvos)
- **Simples e rapido** (3 celulas principais)
- **Logs automaticos** (batch_1_exec_X.log)

### Tempo total:

- Execucao 1: ~22h
- Execucao 2: ~22h
- Execucao 3: ~11h
- **Total**: 55h em 3 sessoes do Colab

### Paralelizacao (multiplas contas):

Execute as 3 execucoes em paralelo:
- **Conta 1**: Execucao 1 (22h)
- **Conta 2**: Execucao 2 (22h)
- **Conta 3**: Execucao 3 (11h)
- **Total**: 22h (tudo em paralelo!)

---

**Criado em**: 2025-11-17
**Metodo**: Execucao direta no Drive (SEM copia)
**Tempo estimado**: 22h por execucao
