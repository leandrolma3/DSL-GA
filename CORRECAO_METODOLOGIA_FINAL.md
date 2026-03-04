# CORRECAO METODOLOGIA TRAIN-THEN-TEST - VERSAO FINAL

**Data**: 2025-11-18
**Problema**: Aprendizado incremental vs. nao-incremental (comparacao injusta)
**Status**: CORRIGIDO

---

## PROBLEMA CRITICO IDENTIFICADO

### GBML (Comportamento Correto)

**main.py:515-525**:
```python
for i in range(num_chunks):
    train_chunk = chunks[i]
    test_chunk = chunks[i+1]

    # CRIA NOVA POPULACAO GA do zero (com seeding)
    best_individual = ga.run_genetic_algorithm(
        train_data=train_chunk,  # Ve APENAS chunk i
        ...
    )

    # Avalia no chunk de teste
    test_metrics = evaluate(best_individual, test_chunk)
```

**Comportamento**:
- Chunk 0: NOVO modelo treinado APENAS no chunk 0 → testa no chunk 1
- Chunk 1: NOVO modelo treinado APENAS no chunk 1 → testa no chunk 2
- Chunk 2: NOVO modelo treinado APENAS no chunk 2 → testa no chunk 3
- etc.

**Dados vistos por chunk**:
- Chunk 0: ve APENAS 1000 instancias (chunk 0)
- Chunk 1: ve APENAS 1000 instancias (chunk 1)
- Chunk 2: ve APENAS 1000 instancias (chunk 2)

### River/ACDWM (Versao INCORRETA - ANTES)

```python
# CRIADO UMA VEZ (FORA DO LOOP!)
evaluator = create_river_model(model_name, classes)

for i in range(num_chunks):
    train_chunk = chunks[i]
    test_chunk = chunks[i+1]

    # ADICIONA ao modelo existente!
    evaluator.train_on_chunk(train_chunk)

    # Avalia no teste
    test_metrics = evaluator.test_on_chunk(test_chunk)
```

**Comportamento**:
- Chunk 0: Modelo VE chunk 0 → testa no chunk 1
- Chunk 1: Modelo VE chunk 0+1 (ACUMULADO!) → testa no chunk 2
- Chunk 2: Modelo VE chunk 0+1+2 (ACUMULADO!) → testa no chunk 3
- etc.

**Dados vistos por chunk**:
- Chunk 0: ve 1000 instancias (chunk 0)
- Chunk 1: ve 2000 instancias (chunks 0+1)
- Chunk 2: ve 3000 instancias (chunks 0+1+2)
- Chunk 3: ve 4000 instancias (chunks 0+1+2+3)
- Chunk 4: ve 5000 instancias (chunks 0+1+2+3+4)

**Consequencia**:
- Modelos River/ACDWM viam 5X MAIS DADOS que GBML!
- Nao havia quedas de desempenho nos drifts porque o modelo tinha "memoria" de todos os conceitos anteriores
- Comparacao COMPLETAMENTE INVALIDA

---

## CORRECAO IMPLEMENTADA

### River/ACDWM (Versao CORRETA - DEPOIS)

**run_comparative_on_existing_chunks.py:151-191**:
```python
for i in range(num_chunks):
    train_chunk = chunks[i]
    test_chunk = chunks[i+1]

    # CRIAR NOVO MODELO a cada chunk! (DENTRO DO LOOP!)
    evaluator = create_river_model(model_name, classes)

    # Treina no chunk atual
    evaluator.train_on_chunk(train_chunk)

    # Avalia no treino (train metrics)
    train_metrics = evaluator.test_on_chunk(train_chunk)

    # Avalia no teste (test metrics)
    test_metrics = evaluator.test_on_chunk(test_chunk)
```

**Comportamento**:
- Chunk 0: NOVO modelo treinado APENAS no chunk 0 → avalia em 0 e 1
- Chunk 1: NOVO modelo treinado APENAS no chunk 1 → avalia em 1 e 2
- Chunk 2: NOVO modelo treinado APENAS no chunk 2 → avalia em 2 e 3
- etc.

**Dados vistos por chunk**:
- Chunk 0: ve APENAS 1000 instancias (chunk 0) - IGUAL AO GBML
- Chunk 1: ve APENAS 1000 instancias (chunk 1) - IGUAL AO GBML
- Chunk 2: ve APENAS 1000 instancias (chunk 2) - IGUAL AO GBML

**Resultado**:
- COMPARACAO JUSTA entre GBML e River/ACDWM
- Agora DEVE haver quedas de desempenho nos chunks com drift!

---

## MODIFICACOES NO CODIGO

### 1. run_river_model_on_chunks()

**ANTES (linha 144-147)**:
```python
# FORA do loop - modelo criado uma vez
evaluator = create_river_model(model_name, classes)

for i in range(len(chunks) - 1):
    ...
    evaluator.train_on_chunk(X_train, y_train)  # Incremental!
```

**DEPOIS (linha 157-162)**:
```python
for i in range(len(chunks) - 1):
    ...
    # DENTRO do loop - modelo criado a cada chunk
    evaluator = create_river_model(model_name, classes)
    evaluator.train_on_chunk(X_train, y_train)  # Nao-incremental!
```

### 2. run_acdwm_on_chunks()

**ANTES (linha 213-230)**:
```python
# FORA do loop - evaluator criado uma vez
acdwm_evaluator = ACDWMEvaluator(...)

for i in range(len(chunks) - 1):
    ...
    acdwm_evaluator.train_on_chunk(X_train, y_train)  # Incremental!
```

**DEPOIS (linha 226-236)**:
```python
for i in range(len(chunks) - 1):
    ...
    # DENTRO do loop - evaluator criado a cada chunk
    acdwm_evaluator = ACDWMEvaluator(...)
    acdwm_evaluator.train_on_chunk(X_train, y_train)  # Nao-incremental!
```

### 3. Correcao do summary aggregation

**ANTES (linha 548-552)**:
```python
summary = consolidated.groupby('model').agg({
    'accuracy': ['mean', 'std'],  # Colunas nao existem!
    'gmean': ['mean', 'std'],
    'f1_weighted': ['mean', 'std']
})
```

**DEPOIS (linha 548-555)**:
```python
summary = consolidated.groupby('model').agg({
    'train_gmean': ['mean', 'std'],
    'test_gmean': ['mean', 'std'],
    'train_accuracy': ['mean', 'std'],
    'test_accuracy': ['mean', 'std'],
    'train_f1': ['mean', 'std'],
    'test_f1': ['mean', 'std']
})
```

---

## IMPACTO ESPERADO

### Antes (INCORRETO)

**Log do Colab**:
```
Chunk 1: Train G-mean: 0.9009, Test G-mean: 0.8858
Chunk 2: Train G-mean: 0.9009, Test G-mean: 0.9131  <- Melhora (incremental!)
Chunk 3: Train G-mean: 0.9192, Test G-mean: 0.9222  <- Melhora (incremental!)
Chunk 4: Train G-mean: 0.9565, Test G-mean: 0.9422  <- Melhora (incremental!)
Chunk 5: Train G-mean: 0.9658, Test G-mean: 0.9547  <- Melhora (incremental!)
```

**Problema**: Desempenho so MELHORA porque modelo ve mais dados!

### Depois (CORRETO)

**Esperado**:
```
Chunk 1: Train G-mean: 0.95, Test G-mean: 0.88  <- Drift (queda!)
Chunk 2: Train G-mean: 0.93, Test G-mean: 0.91
Chunk 3: Train G-mean: 0.94, Test G-mean: 0.92
Chunk 4: Train G-mean: 0.96, Test G-mean: 0.85  <- Drift (queda!)
Chunk 5: Train G-mean: 0.97, Test G-mean: 0.94
```

**Esperado**: Quedas de desempenho nos chunks COM drift simulation!

---

## COMPARACAO COM GBML

### Dados do chunk_metrics.json (GBML - SEA_Abrupt_Simple)

```json
[
  {"chunk": 0, "train_gmean": 0.9886, "test_gmean": 0.9529},
  {"chunk": 1, "train_gmean": 0.9850, "test_gmean": 0.9702},
  {"chunk": 2, "train_gmean": 0.9931, "test_gmean": 0.9706},
  {"chunk": 3, "train_gmean": 0.9851, "test_gmean": 0.9603},
  {"chunk": 4, "train_gmean": 0.9855, "test_gmean": 0.9459}
]
```

**Observacoes**:
- train_gmean: 0.98-0.99 (modelo performa bem no chunk de treino)
- test_gmean: 0.94-0.97 (ligeira queda no teste - generalizacao)
- Chunk 4: test_gmean CAI para 0.9459 (possivelmente drift)

### Agora River/ACDWM devem apresentar comportamento SIMILAR!

---

## VALIDACAO DA CORRECAO

Para validar que a correcao funcionou:

```python
import pandas as pd

# Ler resultados River
df_river = pd.read_csv("river_HAT_results.csv")

# Verificar se test_gmean VARIA (nao so cresce)
print("Test G-mean por chunk:")
print(df_river[['chunk', 'test_gmean']])

# Se tiver drift, deve haver QUEDAS em algum chunk
# Exemplo esperado:
# chunk  test_gmean
#   0      0.88
#   1      0.91
#   2      0.85  <- QUEDA (drift!)
#   3      0.93
#   4      0.89  <- QUEDA (drift!)

# Comparar com GBML
df_gbml = pd.read_json("chunk_metrics.json")
print("\nComparacao GBML vs River:")
print(f"GBML test_gmean medio: {df_gbml['test_gmean'].mean():.4f}")
print(f"River test_gmean medio: {df_river['test_gmean'].mean():.4f}")
# Devem ser COMPARAVEIS (diferenca < 0.10)
```

---

## PROXIMOS PASSOS

1. Re-executar CELULA 6 do Colab com codigo corrigido
2. Validar que:
   - test_gmean apresenta VARIACOES (nao so crescimento)
   - Existem QUEDAS nos chunks com drift
   - Metricas sao COMPARAVEIS com GBML
3. Proceder com analises estatisticas

---

**Autor**: Claude Code
**Versao**: 3.0 (FINAL)
**Status**: PRONTO PARA RE-EXECUCAO
**Arquivos modificados**:
- run_comparative_on_existing_chunks.py (linhas 157-159, 226-233, 548-555)
