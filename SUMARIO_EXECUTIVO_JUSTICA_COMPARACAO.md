# Sumário Executivo: Justiça na Comparação de Modelos

**Data:** 2025-11-22
**Documento Completo:** `PLANO_JUSTICA_COMPARACAO_MODELOS.md`

---

## TL;DR - Problema e Solução

### 🔴 PROBLEMA IDENTIFICADO
A comparação atual entre GBML e baselines pode ser **metodologicamente injusta**:

1. **ACDWM retorna NaN** em 3 datasets (CovType, Shuttle, IntelLabSensors)
2. **Modelos River (ARF, HAT, SRP)** podem estar usando **prequential** (6000 treinos) vs GBML **train-then-test** (1 treino)
3. **GBML usa herança** (memória/seeding) enquanto outros modelos podem não usar

### ✅ SOLUÇÃO RECOMENDADA
**Forçar train-then-test puro em TODOS os modelos:**
- Treinar em chunk 0 (1000 samples)
- Testar em chunks 1-5 (5000 samples)
- **SEM** retreino, learn_one, ou adaptação durante teste
- **GBML SEM memória/seeding** (cada chunk independente)

**Resultado:** Comparação justa de **capacidade de generalização pura**

---

## Três Tarefas Críticas

### Tarefa 1: ACDWM Retornando NaN ❌

**Status:** 3/5 datasets do Batch 5 com NaN

**Hipótese mais provável:**
- ACDWM falha em multiclasse (>2 classes)
- CovType: 7 classes
- Shuttle: 7 classes
- IntelLabSensors: Pode ter problema de balanceamento

**Já visto antes:** ACDWM falhou em LED e WAVEFORM (multiclasse) na Fase 2

**Próximos passos:**
1. Verificar logs detalhados
2. Testar ACDWM isoladamente em CovType
3. Confirmar limitação de multiclasse
4. **Solução:** Atribuir G-mean=0.0 (como Fase 2) e documentar limitação

**Tempo:** 1 dia

---

### Tarefa 2: Sliding Window dos Modelos River 🔍

**Questão crítica:**
```
GBML (train-then-test):
  Chunk 0: TREINO (1000 samples)
  Chunks 1-5: TESTE (sem retreino)
  Total: 1000 treinos

River (prequential - hipótese):
  Chunk 0: TREINO (1000 samples)
  Chunk 1: 1000× (predict + learn_one)
  Chunk 2: 1000× (predict + learn_one)
  ...
  Total: 6000 treinos ← 6× MAIS!
```

**Se for verdade:** Comparação é injusta (River treina 6x mais)

**Próximos passos:**
1. Ler `baseline_river.py` para verificar protocolo
2. Verificar documentação River sobre evaluation modes
3. **Modificar para train-then-test puro:**
   ```python
   # Treina em chunk 0
   for x, y in chunk_0:
       model.learn_one(x, y)

   # Testa em chunks 1-5 SEM learn_one
   for x, y in chunk_i:
       pred = model.predict_one(x)  # Sem learn_one!
   ```
4. Re-executar batches 5, 6, 7

**Tempo:** 2-3 dias (incluindo re-execução)

---

### Tarefa 3: Prequential vs Train-Then-Test 🤔

**Comparação atual (hipotética):**

| Modelo | Protocolo | Adaptação | Justo? |
|--------|-----------|-----------|---------|
| GBML | Train-then-test + memória | Via seeding | ⚠️ |
| ARF/SRP/HAT | Prequential (?) | Contínua | ⚠️ |
| ACDWM | Window-based (?) | Ensemble | ⚠️ |
| ERulesD2S | Incremental (?) | Regras | ⚠️ |

**Problema:** Cada modelo usa protocolo diferente!

**Solução recomendada:** Dois experimentos

**Experimento 1: Train-Then-Test Puro (PRIORIDADE)**
- ✅ Todos treinam em chunk 0
- ✅ Todos testam em chunks 1-5 **SEM adaptação**
- ✅ GBML **SEM memória/seeding**
- ✅ River **SEM learn_one após chunk 0**
- ✅ Mede: **Capacidade de generalização pura**

**Experimento 2: Com Adaptação (OPCIONAL)**
- GBML COM memória/seeding
- River COM learn_one (prequential)
- ACDWM COM window update
- Mede: **Capacidade de adaptação**

**Próximos passos:**
1. Ler artigos ERulesD2S e ACDWM (verificar protocolos usados)
2. Modificar todos os códigos para train-then-test
3. Re-executar Experimento 1
4. (Opcional) Executar Experimento 2

**Tempo:** 3-5 dias (Exp. 1) ou 5-8 dias (ambos)

---

## Cronograma Consolidado

### Opção A: Apenas Experimento 1 (RECOMENDADO)

| Fase | Tarefas | Duração |
|------|---------|---------|
| **Investigação** | Ler logs, código, artigos | 1-2 dias |
| **Implementação** | Corrigir código, modificar protocolos | 2-3 dias |
| **Re-execução** | Rodar batches 5, 6, 7 (train-then-test) | 2-3 dias |
| **Análise** | Consolidar, estatísticas, paper | 1-2 dias |
| **TOTAL** | | **6-10 dias** |

### Opção B: Experimento 1 + 2

| Fase | Tarefas | Duração |
|------|---------|---------|
| **Investigação** | Ler logs, código, artigos | 1-2 dias |
| **Implementação** | Corrigir código, modificar protocolos | 2-3 dias |
| **Re-execução (Exp. 1)** | Rodar batches 5, 6, 7 (train-then-test) | 2-3 dias |
| **Re-execução (Exp. 2)** | Rodar batches 5, 6, 7 (com adaptação) | 2-3 dias |
| **Análise** | Consolidar ambos, estatísticas, paper | 1-2 dias |
| **TOTAL** | | **8-13 dias** |

---

## Decisões Necessárias AGORA

### Q1: Qual experimento fazer?
- ☐ **Opção A:** Apenas Experimento 1 (train-then-test puro) ← RECOMENDADO
  - Mais rápido (6-10 dias)
  - Comparação mais justa
  - Suficiente para paper

- ☐ **Opção B:** Ambos os experimentos
  - Mais completo (8-13 dias)
  - Dois resultados diferentes
  - Mais trabalho de análise

**Recomendação:** **Opção A** (economiza 2-3 dias)

---

### Q2: Como lidar com ACDWM NaN?
- ☐ **Opção A:** Atribuir G-mean=0.0 ← RECOMENDADO
  - Mesma abordagem da Fase 2
  - Documentar limitação no paper
  - Mais rápido

- ☐ **Opção B:** Tentar corrigir/ajustar
  - Pode não ser possível (limitação do modelo)
  - Mais tempo de investigação

**Recomendação:** **Opção A** após investigação rápida

---

### Q3: GBML com ou sem memória?
- ☐ **Experimento 1:** **SEM memória** ← OBRIGATÓRIO
  - Comparação justa
  - Isola capacidade de generalização
  - Cada chunk treina do zero

- ☐ **Experimento 2 (opcional):** **COM memória**
  - Uso real do GBML
  - Avalia adaptação

**Recomendação:** Sem memória para Exp. 1

---

### Q4: Re-executar Fase 2?
- ☐ **Opção A:** NÃO re-executar ← RECOMENDADO
  - Fase 2 foi drift simulation (protocolo diferente)
  - Já está OK
  - Economiza tempo

- ☐ **Opção B:** Re-executar tudo
  - Consistência total
  - Muito trabalho

**Recomendação:** **Opção A** (não re-executar Fase 2)

---

## Ações Imediatas (HOJE)

### Hora 1-2: Investigação Inicial
```bash
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"

# 1. Verificar logs ACDWM
grep -A 20 "CovType\|Shuttle\|IntelLabSensors" batch_5_acdwm_log.txt

# 2. Ler baseline_river.py
cat baseline_river.py | grep -A 10 "learn_one\|progressive_val"

# 3. Verificar implementação atual do GBML
grep -A 10 "memory\|seeding" main.py
```

### Hora 3-4: Leitura de Artigos
```bash
# Abrir PDFs
start "C:\Users\Leandro Almeida\Downloads\Bartoz\paper-Bartosz.pdf"
start "C:\Users\Leandro Almeida\Downloads\paperLu\lu2020.pdf"
```

**Procurar em cada artigo:**
- Seção "Experimental Setup" ou "Methodology"
- Como dados foram divididos?
- Protocolo de avaliação (prequential, holdout, etc.)
- Tamanho de window (se houver)
- Número de iterações de treino

### Hora 5-8: Análise e Planejamento
- Consolidar descobertas
- Decidir sobre Q1-Q4
- Criar branch git (se usar)
- Planejar modificações de código

---

## Arquivos para Investigar

### Código Próprio
1. `baseline_river.py` - Como ARF/SRP/HAT são executados?
2. `baseline_acdwm.py` - Como ACDWM é configurado?
3. `run_erulesd2s_only.py` - Como ERulesD2S é executado?
4. `main.py` - Implementação GBML, uso de memória

### Logs
1. `batch_5_acdwm_log.txt` (ou similar) - Erros do ACDWM
2. `batch_5_river_log.txt` (ou similar) - Protocolo River

### Artigos
1. `C:\Users\Leandro Almeida\Downloads\Bartoz\paper-Bartosz.pdf`
2. `C:\Users\Leandro Almeida\Downloads\paperLu\lu2020.pdf`

---

## Métricas de Sucesso

### Após Investigação (1-2 dias)
- ✅ Entendido protocolo de cada modelo
- ✅ Identificada causa do NaN do ACDWM
- ✅ Decisões sobre Q1-Q4 tomadas
- ✅ Plano de modificação de código criado

### Após Implementação (2-3 dias)
- ✅ Código modificado para train-then-test puro
- ✅ Testado em 1 dataset de cada batch
- ✅ Sem erros de execução
- ✅ Métricas salvas corretamente

### Após Re-execução (2-3 dias)
- ✅ 17 datasets executados com sucesso
- ✅ 6 modelos × 17 datasets = 102 resultados
- ✅ Nenhum NaN (exceto ACDWM multiclasse se confirmado)
- ✅ Resultados consolidados em CSV

### Após Análise (1-2 dias)
- ✅ Testes estatísticos executados
- ✅ Rankings calculados
- ✅ Paper atualizado com Methodology clara
- ✅ Discussão sobre diferenças metodológicas

---

## Riscos e Mitigações

### Risco 1: ACDWM não tem correção
**Mitigação:** Atribuir G-mean=0.0 e documentar limitação (já feito na Fase 2)

### Risco 2: River realmente usa prequential
**Mitigação:** Modificar para train-then-test (possível, documentado no River)

### Risco 3: Re-execução demora muito
**Mitigação:** Executar em paralelo (Google Colab múltiplas instâncias)

### Risco 4: GBML sem memória tem performance muito pior
**Mitigação:** Documentar que é comparação justa, não uso ideal do GBML

### Risco 5: Resultados mudam muito
**Mitigação:** Esperado! Documentar que protocolo anterior era injusto

---

## Referências Rápidas

**Documentação River:**
- Evaluation: https://riverml.xyz/latest/api/evaluate/
- Progressive validation: https://riverml.xyz/latest/recipes/on-hoeffding-trees/#progressive-validation

**Train-then-test em River:**
```python
# CORRETO (train-then-test)
for x, y in train_data:
    model.learn_one(x, y)

for x, y in test_data:
    y_pred = model.predict_one(x)  # Sem learn_one!
    # Calcular métricas
```

**Prequential em River:**
```python
# INCORRETO para nosso caso (prequential)
for x, y in stream:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)  # Aprende após cada predição
```

---

## Contatos e Recursos

**Documentação completa:** `PLANO_JUSTICA_COMPARACAO_MODELOS.md` (42 KB)

**Seções importantes:**
- Tarefa 1: Linha 44-130 (ACDWM NaN)
- Tarefa 2: Linha 134-251 (Sliding Window)
- Tarefa 3: Linha 255-501 (Prequential vs Train-Then-Test)
- Plano Consolidado: Linha 505-580
- Cronograma: Linha 584-597

---

## Status Atual

- ✅ Batch 5 executado (5/5 GBML, baselines com possível problema)
- ✅ Batch 6 executado (6/6 GBML, baselines com possível problema)
- ✅ Batch 7 executado (6/6 GBML, baselines com possível problema)
- ❌ ACDWM com NaN em 3 datasets
- ❓ Protocolo de River não confirmado
- ❓ Comparação pode ser injusta

**Próximo passo:** INVESTIGAÇÃO (começar hoje)

---

**Criado por:** Claude Code
**Data:** 2025-11-22
**Para começar:** Executar "Ações Imediatas" acima
**Decisões necessárias:** Q1-Q4 (marcar com [X])
