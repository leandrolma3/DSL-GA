# Plano de Ação: Justiça na Comparação Entre Modelos

**Data:** 2025-11-22
**Objetivo:** Garantir comparação justa entre GBML e baselines
**Status:** PLANEJAMENTO

---

## Índice
1. [Contexto e Motivação](#contexto-e-motivação)
2. [Tarefa 1: Investigar NaN do ACDWM](#tarefa-1-investigar-nan-do-acdwm)
3. [Tarefa 2: Sliding Window dos Modelos River](#tarefa-2-sliding-window-dos-modelos-river)
4. [Tarefa 3: Prequential vs Train-Then-Test](#tarefa-3-prequential-vs-train-then-test)
5. [Plano de Execução Consolidado](#plano-de-execução-consolidado)
6. [Cronograma Estimado](#cronograma-estimado)

---

## Contexto e Motivação

### Problema Identificado
A comparação atual entre **GBML** e **baselines** pode não ser **metodologicamente justa** devido a diferenças fundamentais nas abordagens de aprendizado:

**GBML (Current):**
- ✅ Train-then-test (treina no chunk 0, testa em chunks 1-5)
- ✅ Pode herdar conhecimento (seeding de gerações passadas via memória)
- ✅ Não aprende incrementalmente "puro"
- ⚠️ Pode ter vantagem: usa informação acumulada de chunks passados

**Baselines River (ARF, HAT, SRP):**
- ❓ Podem usar sliding window (aprende continuamente)
- ❓ Aprendizado prequential puro (incremental)
- ❓ Podem fazer múltiplas iterações por chunk
- ⚠️ Podem ter vantagem: aprendizado contínuo vs retreino

**ACDWM:**
- ❓ Sliding window + ensemble dinâmico
- ❌ Retornando NaN em alguns datasets
- ❓ Configuração de janela não está clara

**ERulesD2S:**
- ❓ Aprendizado incremental de regras
- ❓ Configuração de janela não está clara

### Questão Central
**Como garantir que todos os modelos sejam avaliados sob as mesmas condições?**

Opções:
1. **Opção A:** Fazer todos usarem train-then-test sem informação do passado
2. **Opção B:** Permitir que todos usem informação do passado de forma equivalente
3. **Opção C:** Documentar as diferenças e analisar separadamente

---

## Tarefa 1: Investigar NaN do ACDWM

### 1.1. Problema Observado

**Datasets afetados (Batch 5):**
- CovType
- IntelLabSensors
- Shuttle

**Sintoma:**
- Modelo executa sem erros
- Métricas retornam `NaN`
- Similar ao problema da Fase 2 com datasets multiclasse

### 1.2. Hipóteses

**Hipótese 1: Problema de Classes (Mais Provável)**
- CovType: 7 classes
- Shuttle: 7 classes
- IntelLabSensors: binárias, mas pode ter problema de balanceamento
- ACDWM pode falhar em multiclasse (já visto na Fase 2)

**Hipótese 2: Tamanho de Janela Inadequado**
- Janela muito pequena para datasets complexos
- Não consegue construir classificadores eficazes

**Hipótese 3: Formato dos Dados**
- CSV local vs River dataset
- Normalização/pré-processamento
- Tipos de features

### 1.3. Plano de Investigação

#### Passo 1.1: Verificar Logs Detalhados
```bash
# Procurar logs do ACDWM para esses 3 datasets
grep -A 50 "CovType\|IntelLabSensors\|Shuttle" batch_5_acdwm.log
```

**O que procurar:**
- Mensagens de erro suprimidas
- Warnings sobre classes
- Informações sobre janela/ensemble

#### Passo 1.2: Verificar Características dos Dados
```python
# Script: investigate_acdwm_nan.py
import pandas as pd

datasets = {
    'CovType': 'datasets/processed/covertype_processed.csv',
    'Shuttle': 'datasets/processed/shuttle_processed.csv',
    'IntelLabSensors': 'datasets/processed/intellabsensors_processed.csv'
}

for name, path in datasets.items():
    df = pd.read_csv(path)
    print(f"\n{name}:")
    print(f"  Shape: {df.shape}")
    print(f"  Classes: {df['class'].unique()}")
    print(f"  Class distribution:\n{df['class'].value_counts()}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Feature types: {df.dtypes.value_counts()}")
```

#### Passo 1.3: Verificar Código ACDWM
```bash
# Localizar implementação ACDWM
find . -name "*acdwm*" -o -name "*ACDWM*"
```

**O que verificar:**
- Método `predict()` - pode estar retornando None/NaN
- Tratamento de multiclasse
- Cálculo de G-mean
- Condições que levam a NaN

#### Passo 1.4: Testar ACDWM Isoladamente
```python
# Script: test_acdwm_isolated.py
from ACDWM.acdwm import ACDWM  # ou caminho correto
import pandas as pd

# Carregar CovType (menor)
df = pd.read_csv('datasets/processed/covertype_processed.csv')
X = df.drop('class', axis=1).values[:1000]  # Primeiros 1000
y = df['class'].values[:1000]

# Testar ACDWM
model = ACDWM(window_size=100)  # Testar diferentes tamanhos
for i, (xi, yi) in enumerate(zip(X, y)):
    pred = model.predict_one(xi)
    print(f"Sample {i}: pred={pred}, true={yi}")
    model.learn_one(xi, yi)
```

#### Passo 1.5: Comparar com Fase 2
- ACDWM falhou em LED e WAVEFORM (multiclasse) na Fase 2
- CovType e Shuttle também são multiclasse (7 classes)
- Pode ser a mesma limitação

### 1.4. Soluções Possíveis

**Solução 1: Confirmar Limitação de Multiclasse**
- Se ACDWM não suporta >2 classes, documentar
- Atribuir G-mean=0.0 (como na Fase 2)
- Incluir nota no paper sobre limitação

**Solução 2: Ajustar Parâmetros**
- Aumentar window_size
- Ajustar número de classificadores no ensemble
- Testar diferentes configurações

**Solução 3: Corrigir Implementação**
- Se for bug, corrigir código
- Re-executar datasets afetados

### 1.6. Cronograma Tarefa 1
- **Investigação:** 2-4 horas
- **Teste isolado:** 2-3 horas
- **Correção (se aplicável):** 2-8 horas
- **Re-execução:** 4-6 horas
- **Total:** 1-2 dias

---

## Tarefa 2: Sliding Window dos Modelos River

### 2.1. Questão Fundamental

**GBML usa train-then-test:**
```
Chunk 0: TREINO (1000 samples)
Chunk 1: TESTE (1000 samples) - sem retreino
Chunk 2: TESTE (1000 samples) - sem retreino
...
Chunk 5: TESTE (1000 samples) - sem retreino
```

**Modelos River podem usar prequential:**
```
Chunk 0: TREINO (1000 samples)
Chunk 1: Para cada sample:
  - TESTE (predict)
  - TREINO (learn_one)  ← 1000 iterações de aprendizado!
Chunk 2: Para cada sample:
  - TESTE (predict)
  - TREINO (learn_one)  ← Mais 1000 iterações!
```

**Isso é justo?**
- GBML: 1 treino (chunk 0) = 1000 samples
- River: 1 treino inicial + 5000 learn_one = 6000 "treinos"

### 2.2. Modelos a Investigar

#### ARF (Adaptive Random Forest)
**Arquivo de referência:** `baseline_river.py`

**O que verificar:**
- Como ARF é chamado no código?
- Usa `progressive_val_score` ou `iter_progressive_val_score`?
- Quantas vezes `learn_one` é chamado por chunk?

**Documentação River:**
```python
from river import forest, metrics

# Modo 1: Prequential (incremental)
model = forest.ARFClassifier()
for x, y in stream:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)  # Aprende após cada predição

# Modo 2: Holdout (nosso caso?)
# Treina em chunk 0, testa em chunks 1-5 SEM learn_one
```

#### SRP (Streaming Random Patches)
Similar ao ARF, verificar:
- Método de avaliação usado
- Frequência de `learn_one`

#### HAT (Hoeffding Adaptive Tree)
Similar ao ARF, verificar:
- Método de avaliação usado
- Frequência de `learn_one`

### 2.3. Plano de Investigação

#### Passo 2.1: Ler Código de Baseline River
```bash
# Abrir baseline_river.py
cat baseline_river.py
```

**O que procurar:**
```python
# Procurar por:
# 1. Modo de avaliação
progressive_val_score()  # ← Prequential (incremental)
# vs
# Train em chunk 0, predict em chunks 1-5  # ← Train-then-test

# 2. Chamadas a learn_one
model.learn_one(x, y)  # Quantas vezes é chamado?

# 3. Configuração de janela
# window_size, max_samples, etc.
```

#### Passo 2.2: Verificar Documentação River
**River - Evaluation Methods:**
- `progressive_val_score`: Prequential (test-then-train em cada sample)
- `iter_progressive_val_score`: Iterator version do acima

**Se estamos usando prequential:**
- River aprende continuamente
- GBML não aprende após chunk 0
- **Comparação injusta!**

#### Passo 2.3: Contar Iterações de Treino
```python
# Script: count_training_iterations.py

# Simular GBML
gbml_train_samples = 1000  # Chunk 0 apenas

# Simular River com prequential
river_train_samples = 1000  # Chunk 0
river_train_samples += 5 * 1000  # Chunks 1-5 (learn_one em cada)
# Total: 6000 samples de treino

print(f"GBML treina em: {gbml_train_samples} samples")
print(f"River treina em: {river_train_samples} samples")
print(f"River treina {river_train_samples/gbml_train_samples}x mais!")
```

### 2.4. Soluções Possíveis

**Solução A: Forçar Train-Then-Test em River (Recomendado)**
```python
# Modo 1: Treino em chunk 0, teste em chunks 1-5 SEM learn_one
# Chunk 0
for x, y in chunk_0:
    model.learn_one(x, y)

# Chunks 1-5 (teste apenas)
for chunk_i in [chunk_1, ..., chunk_5]:
    predictions = []
    for x, y in chunk_i:
        pred = model.predict_one(x)  # Sem learn_one!
        predictions.append(pred)
    # Calcular métricas
```

**Vantagem:** Comparação justa com GBML
**Desvantagem:** Não usa vantagem de aprendizado incremental do River

**Solução B: Fazer GBML Aprender Incrementalmente**
```python
# GBML re-treina ou atualiza após cada chunk
# Chunk 0: Treino inicial
# Chunk 1: Teste + Retreino ou seeding para próximo chunk
# Chunk 2: Teste + Retreino ou seeding
```

**Vantagem:** Comparação justa se River também aprende continuamente
**Desvantagem:** Muda abordagem do GBML, pode ser muito custoso

**Solução C: Sliding Window Controlado**
```python
# River usa sliding window de tamanho fixo (e.g., 1000 samples)
# Aprende continuamente, mas esquece samples antigos
# Equivalente a: memória limitada como GBML

from river import utils

# Wrapper com sliding window
model = utils.Rolling(
    model=forest.ARFClassifier(),
    window_size=1000
)
```

**Vantagem:** Controla quantidade de informação retida
**Desvantagem:** Ainda aprende continuamente (mais iterações)

### 2.5. Recomendação Inicial

**Opção mais justa:**
1. **River: Train-then-test** (Solução A)
   - Treina em chunk 0
   - Testa em chunks 1-5 **sem learn_one**
   - Mesma abordagem do GBML

2. **Documentar claramente** no paper:
   - "Para comparação justa, modelos River foram avaliados em modo train-then-test, sem aprendizado incremental durante teste"
   - "GBML pode usar memória de chunks passados via seeding, mas não re-treina"

### 2.6. Cronograma Tarefa 2
- **Investigação código:** 2-3 horas
- **Verificar documentação River:** 2-3 horas
- **Modificar baseline_river.py:** 3-4 horas
- **Re-executar batches 5, 6, 7:** 24-36 horas
- **Total:** 2-3 dias

---

## Tarefa 3: Prequential vs Train-Then-Test

### 3.1. Comparação Metodológica

#### GBML (Current Implementation)
**Modo:** Train-then-test com memória opcional

```
Chunk 0 [0-999]:
  - Treina população inicial (random + DT seeding)
  - Evolui 200 gerações
  - Salva melhores indivíduos em memória

Chunk 1 [1000-1999]:
  - USA MEMÓRIA: Seed população com indivíduos salvos (opcional)
  - Evolui 200 gerações
  - TESTA e calcula G-mean
  - Salva melhores indivíduos em memória

Chunk 2 [2000-2999]:
  - USA MEMÓRIA: Seed população com indivíduos salvos
  - Evolui 200 gerações
  - TESTA e calcula G-mean
  - ...
```

**Características:**
- ✅ Não é puramente "train-then-test" (usa memória)
- ✅ Não é puramente "incremental" (re-treina a cada chunk)
- ✅ Híbrido: re-treina com seeding de conhecimento passado

**Custo computacional:**
- 6 chunks × 200 gerações × 120 indivíduos = 144,000 avaliações de fitness

#### River Models (Current - Assumindo Prequential)
**Modo:** Prequential puro (test-then-train em cada sample)

```
Chunk 0 [0-999]:
  - Para cada sample (x, y):
    - model.learn_one(x, y)

Chunk 1 [1000-1999]:
  - Para cada sample (x, y):
    - y_pred = model.predict_one(x)
    - model.learn_one(x, y)  ← Aprende continuamente

Chunk 2 [2000-2999]:
  - Para cada sample (x, y):
    - y_pred = model.predict_one(x)
    - model.learn_one(x, y)
  ...
```

**Características:**
- ✅ Aprendizado incremental puro
- ✅ Aprende em TODOS os samples (6000 total)
- ⚠️ Pode ter vantagem sobre GBML em adaptação contínua

**Custo computacional:**
- 6000 × learn_one (depende do modelo)
- ARF: atualiza múltiplas árvores
- HAT: atualiza árvore Hoeffding

#### ACDWM (Assumindo Windowed Ensemble)
**Modo:** Ensemble com sliding window

```
Chunk 0 [0-999]:
  - Treina ensemble inicial
  - Window = últimos W samples

Chunk 1 [1000-1999]:
  - Para cada sample:
    - Testa com ensemble
    - Adiciona sample à window
    - Remove sample mais antigo (se window cheia)
    - Re-treina ou atualiza ensemble
```

**Características:**
- ✅ Mantém window de tamanho fixo
- ✅ Adapta continuamente
- ⚠️ Complexidade de configuração de window

#### ERulesD2S (Rule-based Incremental)
**Modo:** Aprendizado incremental de regras

```
Para cada sample (x, y):
  - Testa com regras existentes
  - Atualiza estatísticas das regras
  - Adiciona/remove regras conforme necessário
```

**Características:**
- ✅ Incremental puro
- ✅ Adapta regras continuamente
- ⚠️ Pode ter vantagem em adaptação

### 3.2. Problema de Justiça

**Cenário Atual (se River usa prequential):**

| Modelo | Treinos | Testes | Adaptação Contínua |
|--------|---------|--------|-------------------|
| GBML | 6 × GA (200 gen) | 5 chunks | Sim (via memória/seeding) |
| ARF | 6000 × learn_one | 5000 samples | Sim (incremental) |
| SRP | 6000 × learn_one | 5000 samples | Sim (incremental) |
| HAT | 6000 × learn_one | 5000 samples | Sim (incremental) |
| ACDWM | Window-based | Contínuo | Sim (ensemble) |
| ERulesD2S | Contínuo | Contínuo | Sim (regras) |

**Problema:**
- River: 6000 oportunidades de aprendizado
- GBML: 6 re-treinos (mas com seeding)
- **Não é comparação justa!**

### 3.3. Opções de Solução

#### Opção 1: Forçar Train-Then-Test em Todos (RECOMENDADO)

**Implementação:**
```python
# Todos os modelos:
# - Treinam em chunk 0
# - Testam em chunks 1-5 SEM re-treino/learn_one
# - GBML: Sem memória/seeding (retreino do zero em cada chunk)
```

**Vantagens:**
- ✅ Comparação mais justa
- ✅ Isola capacidade de generalização
- ✅ Todos partem do mesmo ponto

**Desvantagens:**
- ❌ Não usa vantagem de aprendizado incremental
- ❌ GBML sem seeding pode ter performance pior
- ❌ Não reflete uso real de modelos incrementais

#### Opção 2: Permitir Adaptação Controlada em Todos

**Implementação:**
```python
# Todos os modelos podem adaptar, mas de forma equivalente:
# - GBML: Re-treina após cada chunk com seeding
# - River: Aprende em chunk atual antes de testar próximo
# - ACDWM: Window size = 1 chunk (1000 samples)
```

**Vantagens:**
- ✅ Usa capacidade de adaptação de cada modelo
- ✅ Mais próximo de cenário real
- ✅ GBML pode usar memória

**Desvantagens:**
- ❌ Difícil garantir "equivalência" de adaptação
- ❌ Custo computacional maior (re-treino GBML)

#### Opção 3: Duas Avaliações Separadas (RECOMENDADO+)

**Implementação:**
```python
# Experimento 1: Train-Then-Test Puro (Sem Adaptação)
# - Todos treinam em chunk 0
# - Todos testam em chunks 1-5 sem retreino
# - GBML sem memória/seeding

# Experimento 2: Com Adaptação
# - GBML: Com memória/seeding
# - River: Incremental com learn_one
# - ACDWM: Window-based
# - Comparar capacidade de adaptação
```

**Vantagens:**
- ✅✅ Comparação justa no Exp. 1
- ✅✅ Avalia capacidade de adaptação no Exp. 2
- ✅✅ Dois resultados: generalização + adaptação
- ✅ Documentação clara de diferenças

**Desvantagens:**
- ❌ Dobro de execuções
- ❌ Mais complexo de apresentar

### 3.4. Investigação de Artigos

#### ERulesD2S (Bartosz et al.)
**Arquivo:** `C:\Users\Leandro Almeida\Downloads\Bartoz\paper-Bartosz.pdf`

**O que procurar:**
1. **Seção Methodology:**
   - Como é o protocolo de avaliação?
   - Prequential ou train-then-test?
   - Tamanho de window (se houver)?

2. **Seção Experiments:**
   - Quantos samples de treino?
   - Quantos samples de teste?
   - Re-treina ou aprende incrementalmente?

3. **Comparação com baselines:**
   - Como outros modelos foram avaliados?
   - Mesmo protocolo para todos?

#### ACDWM (Lu et al.)
**Arquivo:** `C:\Users\Leandro Almeida\Downloads\paperLu\lu2020.pdf`

**O que procurar:**
1. **Window Management:**
   - Tamanho de window usado?
   - Como window é atualizada?
   - Quantos classificadores no ensemble?

2. **Learning Protocol:**
   - Aprende em cada sample?
   - Re-treina em batches?
   - Como lida com concept drift?

3. **Evaluation Protocol:**
   - Prequential?
   - Holdout?
   - Cross-validation?

### 3.5. Plano de Investigação

#### Passo 3.1: Ler Artigos
```bash
# Abrir PDFs
# - paper-Bartosz.pdf (ERulesD2S)
# - lu2020.pdf (ACDWM)
```

**Criar tabela resumo:**
| Modelo | Protocolo | Window Size | Incremental | Re-treino |
|--------|-----------|-------------|-------------|-----------|
| ERulesD2S | ? | ? | ? | ? |
| ACDWM | ? | ? | ? | ? |

#### Passo 3.2: Verificar Implementações
```python
# ERulesD2S
# Arquivo: run_erulesd2s_only.py (ou similar)
# Verificar:
# - Como modelo é treinado
# - Como é testado
# - Chamadas a update/learn

# ACDWM
# Arquivo: baseline_acdwm.py
# Verificar:
# - window_size configurado
# - Como ensemble é atualizado
# - Protocolo de teste
```

#### Passo 3.3: Experimento Controlado
```python
# Script: compare_protocols.py
# Testar todos os modelos em 3 protocolos:

# Protocolo 1: Train-Then-Test Puro
# - Treina em chunk 0
# - Testa em chunks 1-5
# - SEM retreino/learn_one

# Protocolo 2: Prequential
# - Treina em chunk 0
# - Para chunks 1-5: test-then-train em cada sample

# Protocolo 3: Chunk-wise Adaptation
# - Treina em chunk 0
# - Testa em chunk 1, depois treina em chunk 1
# - Testa em chunk 2, depois treina em chunk 2
# ...

# Comparar resultados
```

### 3.6. Recomendação Final

**Abordagem Recomendada:**

1. **Experimento Principal: Train-Then-Test Puro**
   - Todos os modelos:
     - Treinam em chunk 0 (1000 samples)
     - Testam em chunks 1-5 (5000 samples)
     - **SEM** retreino/adaptação durante teste
   - GBML: **SEM** memória/seeding (cada chunk é independente)
   - River: **SEM** learn_one após chunk 0
   - ACDWM: Treina ensemble em chunk 0, testa sem atualizar
   - ERulesD2S: Treina regras em chunk 0, testa sem atualizar

   **Resultado:** Mede **capacidade de generalização** pura

2. **Experimento Secundário (Opcional): Com Adaptação**
   - GBML: **COM** memória/seeding
   - River: **COM** learn_one (prequential)
   - ACDWM: **COM** window-based update
   - ERulesD2S: **COM** update de regras

   **Resultado:** Mede **capacidade de adaptação**

3. **Documentação no Paper**
   ```
   "Para garantir comparação justa, avaliamos todos os modelos
   em dois protocolos:

   1) Train-then-test: Treino em chunk inicial, teste em chunks
      subsequentes sem adaptação. Mede capacidade de generalização.

   2) Adaptive: Cada modelo usa sua estratégia de adaptação nativa.
      Mede capacidade de adaptação a concept drift."
   ```

### 3.7. Cronograma Tarefa 3
- **Leitura de artigos:** 4-6 horas
- **Verificação de implementações:** 3-4 horas
- **Modificação de código:** 4-6 horas
- **Experimento train-then-test:** 24-36 horas
- **Experimento com adaptação (opcional):** 24-36 horas
- **Análise e documentação:** 4-6 horas
- **Total:** 3-5 dias (ou 5-8 dias com exp. adaptação)

---

## Plano de Execução Consolidado

### Fase 1: Investigação (Paralela)
**Duração:** 1-2 dias

**Tarefas paralelas:**
1. **Investigar NaN do ACDWM** (1 pessoa)
   - Ler logs
   - Verificar código
   - Testar isoladamente

2. **Investigar sliding window River** (1 pessoa)
   - Ler baseline_river.py
   - Verificar documentação River
   - Contar iterações de treino

3. **Ler artigos ERulesD2S e ACDWM** (1 pessoa)
   - Extrair protocolos de avaliação
   - Documentar window sizes
   - Criar tabela resumo

**Entregável:** Relatório de investigação com recomendações

---

### Fase 2: Implementação de Correções
**Duração:** 2-3 dias

**Tarefas sequenciais:**
1. **Corrigir NaN do ACDWM**
   - Aplicar solução identificada
   - Testar em 1 dataset primeiro

2. **Modificar baseline_river.py**
   - Implementar train-then-test puro
   - Remover learn_one após chunk 0
   - Testar em 1 dataset primeiro

3. **Modificar baseline_acdwm.py**
   - Ajustar window size (se necessário)
   - Garantir train-then-test

4. **Modificar run_erulesd2s_only.py**
   - Garantir train-then-test

5. **Modificar main.py (GBML)**
   - Criar flag: `use_memory_seeding` (default=False)
   - Modo train-then-test: desabilita memória
   - Testar em 1 dataset primeiro

**Entregável:** Código modificado e testado

---

### Fase 3: Re-execução de Experimentos
**Duração:** 2-3 dias (execução)

**Experimento 1: Train-Then-Test Puro**
```bash
# Batch 5 (5 datasets reais)
python main.py --config configs/config_batch_5.yaml --no-memory-seeding
python run_comparative_models.py --config configs/config_batch_5.yaml --protocol train-then-test

# Batch 6 (6 sintéticos)
python main.py --config configs/config_batch_6.yaml --no-memory-seeding
python run_comparative_models.py --config configs/config_batch_6.yaml --protocol train-then-test

# Batch 7 (6 sintéticos)
python main.py --config configs/config_batch_7.yaml --no-memory-seeding
python run_comparative_models.py --config configs/config_batch_7.yaml --protocol train-then-test
```

**Tempo estimado:** 24-36 horas

**Experimento 2 (Opcional): Com Adaptação**
```bash
# Mesmos comandos, mas com --with-adaptation
```

**Tempo adicional:** 24-36 horas

**Entregável:** Resultados consolidados de todos os modelos

---

### Fase 4: Análise e Documentação
**Duração:** 1-2 dias

**Tarefas:**
1. **Consolidar resultados**
   - Criar CSV com métricas de ambos os protocolos
   - Calcular médias, std, rankings

2. **Testes estatísticos**
   - Friedman test
   - Wilcoxon pairwise
   - Cliff's Delta

3. **Criar visualizações**
   - Gráficos comparando protocolos
   - Tabelas de ranking

4. **Documentar no paper**
   - Seção Methodology: Descrever protocolos
   - Seção Results: Apresentar ambos os experimentos
   - Seção Discussion: Analisar diferenças

**Entregável:** Paper atualizado com análise completa

---

## Cronograma Estimado

| Fase | Duração | Dias Corridos | Dependências |
|------|---------|---------------|--------------|
| **Fase 1: Investigação** | 8-16 horas | 1-2 dias | Nenhuma |
| **Fase 2: Implementação** | 12-20 horas | 2-3 dias | Fase 1 |
| **Fase 3: Re-execução (Exp. 1)** | 24-36 horas | 2-3 dias | Fase 2 |
| **Fase 3: Re-execução (Exp. 2)** | 24-36 horas | 2-3 dias | Fase 2 (opcional) |
| **Fase 4: Análise** | 8-16 horas | 1-2 dias | Fase 3 |
| **TOTAL (sem Exp. 2)** | 52-88 horas | **6-10 dias** | - |
| **TOTAL (com Exp. 2)** | 76-124 horas | **8-13 dias** | - |

---

## Priorização Recomendada

### Prioridade ALTA (Fazer Primeiro)
1. ✅ **Tarefa 1:** Investigar NaN do ACDWM
   - Bloqueia consolidação de resultados Fase 3
   - Pode revelar problemas graves

2. ✅ **Tarefa 2:** Investigar sliding window River
   - Afeta justiça de toda a comparação
   - Pode exigir re-execução completa

3. ✅ **Tarefa 3 (Leitura):** Ler artigos ERulesD2S e ACDWM
   - Informa decisões sobre protocolos
   - Necessário para Methodology do paper

### Prioridade MÉDIA
4. ⏳ **Implementar train-then-test puro**
   - Garante comparação justa
   - Baseline para Experimento 1

5. ⏳ **Re-executar Experimento 1**
   - Gera resultados justos
   - Prioridade sobre Experimento 2

### Prioridade BAIXA (Opcional)
6. 🔜 **Experimento 2: Com adaptação**
   - Interessante, mas não essencial
   - Pode ser trabalho futuro

7. 🔜 **Análise avançada**
   - Análise de drift detection
   - Visualizações adicionais

---

## Perguntas para Decisão

Antes de começar, precisamos decidir:

### Q1: Qual experimento priorizar?
- **Opção A:** Apenas Experimento 1 (train-then-test puro) - RECOMENDADO
- **Opção B:** Ambos os experimentos (com e sem adaptação)

**Recomendação:** Opção A (economiza 2-3 dias)

### Q2: Como lidar com ACDWM NaN?
- **Opção A:** Atribuir G-mean=0.0 (como Fase 2)
- **Opção B:** Tentar corrigir/ajustar parâmetros
- **Opção C:** Excluir ACDWM da análise

**Recomendação:** Investigar primeiro, depois decidir

### Q3: GBML com ou sem memória no Experimento 1?
- **Opção A:** Sem memória (comparação mais justa) - RECOMENDADO
- **Opção B:** Com memória (uso real do GBML)

**Recomendação:** Opção A para Exp. 1, Opção B para Exp. 2 (se fizer)

### Q4: Re-executar Fase 2 também?
- **Opção A:** Não, Fase 2 já está OK
- **Opção B:** Sim, aplicar mesmas correções

**Recomendação:** Opção A (Fase 2 foi drift simulation, protocolo diferente)

---

## Próximos Passos Imediatos

**AGORA:**
1. Ler este documento completo
2. Decidir sobre Q1-Q4 acima
3. Criar branch git para mudanças (se usar git)

**HOJE/AMANHÃ:**
1. Iniciar Fase 1 (Investigação)
2. Ler logs do ACDWM (Tarefa 1.1)
3. Ler baseline_river.py (Tarefa 2.1)
4. Começar leitura dos artigos (Tarefa 3.1)

**ESTA SEMANA:**
1. Completar Fase 1
2. Implementar correções (Fase 2)
3. Testar em 1 dataset de cada batch

**PRÓXIMA SEMANA:**
1. Re-executar Experimento 1 (Fase 3)
2. Consolidar resultados (Fase 4)
3. Atualizar paper

---

**Criado por:** Claude Code
**Data:** 2025-11-22
**Status:** PLANEJAMENTO - AGUARDANDO DECISÕES
**Próxima Atualização:** Após Fase 1 (Investigação)
