# Investigação Inicial: Protocolos de Avaliação

**Data:** 2025-11-22
**Status:** INVESTIGAÇÃO EM ANDAMENTO

---

## Descobertas Críticas

### 1. Protocolo Atual do River (ARF, SRP, HAT)

**Arquivo:** `baseline_river.py`
**Método:** `evaluate_prequential()` (de `shared_evaluation.py`)

**Protocolo identificado:**
```python
for i in range(len(chunks) - 1):
    # Treina no chunk i
    train_on_chunk(chunks[i])
    # Testa no chunk i+1
    test_on_chunk(chunks[i+1])
```

**Sequência de execução (6 chunks):**
```
Iteração 0: Treina chunk 0 (1000 samples) → Testa chunk 1
Iteração 1: Treina chunk 1 (1000 samples) → Testa chunk 2
Iteração 2: Treina chunk 2 (1000 samples) → Testa chunk 3
Iteração 3: Treina chunk 3 (1000 samples) → Testa chunk 4
Iteração 4: Treina chunk 4 (1000 samples) → Testa chunk 5
```

**Total de treinos:** 5 chunks × 1000 samples = **5000 samples**

**Modo:** **Chunk-wise sequential train-then-test**
- NÃO é prequential puro (test-then-train em cada sample)
- É train-then-test sequencial por chunks
- Modelo é **retreinado** em cada chunk antes de testar o próximo

### 2. Protocolo do ACDWM

**Arquivo:** `baseline_acdwm.py`
**Classe:** `ACDWMEvaluator`

**Descoberta:** ACDWM já tem suporte para dois modos!

```python
evaluation_mode:
    - 'train-then-test' (padrão?)
    - 'test-then-train' (prequential)
```

**Parâmetros adicionais:**
- `adaptive_chunk_size: bool` - Seleção adaptativa de chunk
- `theta: float = 0.001` - Threshold para remover classificadores
- `min_ensemble_size: int = 10` - Tamanho mínimo do ensemble

**Status:** Precisa verificar qual modo está configurado atualmente

### 3. Protocolo do GBML

**Status:** AINDA NÃO VERIFICADO

**Perguntas pendentes:**
1. GBML usa `evaluate_prequential()` também?
2. GBML treina em cada chunk ou só no chunk 0?
3. Como memória/seeding funciona entre chunks?
4. Qual método de avaliação é chamado no `main.py`?

---

## Comparação Preliminar

| Modelo | Método | Treinos | Status |
|--------|--------|---------|--------|
| **River (ARF/SRP/HAT)** | `evaluate_prequential()` | 5 chunks (5000 samples) | ✅ Confirmado |
| **ACDWM** | `ACDWMEvaluator` | Modo configurável | ⏳ Verificar config |
| **GBML** | ? | ? | ❓ Investigar |
| **ERulesD2S** | ? | ? | ❓ Investigar |

---

## Problema do NaN do ACDWM

**Datasets afetados:**
- CovType (7 classes)
- Shuttle (7 classes)
- IntelLabSensors (binárias)

**Hipóteses:**
1. **Limitação de multiclasse** (mais provável)
   - ACDWM falhou em LED e WAVEFORM (multiclasse) na Fase 2
   - CovType e Shuttle também são multiclasse

2. **Formato de labels**
   - ACDWM usa formato -1/+1 (não 0/1)
   - Conversão pode estar falhando em multiclasse

3. **Modo de avaliação**
   - Se `evaluation_mode='test-then-train'`, pode falhar em alguns datasets

**Ações necessárias:**
- [ ] Verificar logs detalhados do ACDWM
- [ ] Confirmar modo de avaliação usado
- [ ] Testar ACDWM isoladamente em CovType
- [ ] Verificar se suporta >2 classes

---

## Diferenças Train-Then-Test vs Test-Then-Train

### Train-Then-Test (Chunk-wise)
```python
# Usado atualmente por River
for i in range(n_chunks - 1):
    model.train(chunk[i])      # Treina chunk i
    metrics = model.test(chunk[i+1])  # Testa chunk i+1
```

**Características:**
- Treina completamente antes de testar
- Re-treina em cada chunk
- Cada chunk: 1000× `learn_one()`
- Total: 5 chunks × 1000 = 5000 treinos

### Test-Then-Train (Prequential Puro)
```python
# Possível modo alternativo
for chunk in chunks:
    for x, y in chunk:
        pred = model.predict_one(x)  # Testa primeiro
        model.learn_one(x, y)        # Treina depois
```

**Características:**
- Testa antes de aprender
- Aprendizado contínuo sample-por-sample
- Total: 6 chunks × 1000 = 6000 learn_one

### Train-Then-Test (Único - GBML?)
```python
# Possível modo do GBML?
model.train(chunk[0])  # Treina apenas chunk 0
for i in range(1, n_chunks):
    metrics = model.test(chunk[i])  # Testa chunks 1-5
```

**Características:**
- Treina apenas no início
- Testa em todos os chunks seguintes SEM retreino
- Total: 1 chunk × 1000 = 1000 treinos
- Pode usar memória/seeding (GBML)

---

## Impacto na Comparação

### Se River usa Chunk-wise Sequential (5000 treinos)
E GBML usa Train-Once (1000 treinos):

**Comparação é INJUSTA:**
- River treina 5× mais
- River se adapta continuamente
- GBML não se adapta (exceto via memória)

**Solução:**
- Forçar River a treinar apenas no chunk 0
- OU modificar GBML para retreinar em cada chunk
- OU documentar diferença claramente

### Se Todos Usam Chunk-wise Sequential
**Comparação pode ser JUSTA se:**
- GBML também treina em cada chunk
- Memória do GBML é equivalente ao modelo persistente do River

**Mas:**
- GBML treina via GA (200 gerações) em cada chunk = muito custoso
- River treina via `learn_one` incremental = rápido
- Custo computacional diferente

---

## Próximos Passos Imediatos

### Passo 1: Verificar Protocolo do GBML (PRIORIDADE)
```bash
# Procurar como GBML é executado
grep -A 20 "evaluate_prequential\|train_on_chunk\|test_on_chunk" main.py

# Verificar se GBML treina em cada chunk
grep -A 10 "for.*chunk\|range.*num_chunks" main.py
```

**O que procurar:**
- GBML usa `evaluate_prequential()`?
- GBML tem método próprio de avaliação?
- Quantas vezes GA é executado?

### Passo 2: Verificar Config do ACDWM
```bash
# Procurar como ACDWM é chamado
grep -A 10 "ACDWMEvaluator\|evaluation_mode" *.py

# Verificar logs para confirmar modo
grep "evaluation_mode\|train-then-test\|test-then-train" batch_5*.log
```

### Passo 3: Testar ACDWM Isoladamente
```python
# Script: test_acdwm_multiclass.py
# Testar ACDWM em dataset multiclasse pequeno
# Verificar se retorna NaN
```

### Passo 4: Ler Artigos (ERulesD2S e ACDWM)
- `C:\Users\Leandro Almeida\Downloads\Bartoz\paper-Bartosz.pdf`
- `C:\Users\Leandro Almeida\Downloads\paperLu\lu2020.pdf`
- Verificar protocolo de avaliação descrito

---

## Perguntas Críticas Pendentes

### Q1: GBML treina em cada chunk ou só no chunk 0?
- [ ] Verificar `main.py`
- [ ] Verificar método de avaliação usado
- [ ] Contar quantas vezes GA é executado

### Q2: Qual modo do ACDWM está sendo usado?
- [ ] Verificar chamada de `ACDWMEvaluator`
- [ ] Verificar logs
- [ ] Confirmar `evaluation_mode`

### Q3: ACDWM suporta multiclasse?
- [ ] Testar isoladamente
- [ ] Verificar código de conversão de labels
- [ ] Confirmar limitação

### Q4: ERulesD2S usa qual protocolo?
- [ ] Encontrar arquivo `run_erulesd2s_only.py`
- [ ] Verificar método de avaliação
- [ ] Ler artigo

---

## Cenários Possíveis

### Cenário A: Todos Usam Chunk-wise Sequential
```
GBML:        Treina em chunks 0-4, testa em chunks 1-5 (5000 treinos)
River:       Treina em chunks 0-4, testa em chunks 1-5 (5000 treinos)
ACDWM:       Treina em chunks 0-4, testa em chunks 1-5 (modo train-then-test)
ERulesD2S:   Treina em chunks 0-4, testa em chunks 1-5
```

**Resultado:** Comparação JUSTA (mesmo protocolo)
**Ação:** Nenhuma (apenas documentar)

### Cenário B: GBML Treina Apenas No Chunk 0
```
GBML:        Treina em chunk 0, testa em chunks 1-5 (1000 treinos)
River:       Treina em chunks 0-4, testa em chunks 1-5 (5000 treinos)
```

**Resultado:** Comparação INJUSTA (River treina 5× mais)
**Ação:** Modificar River para treinar apenas no chunk 0

### Cenário C: Mix de Protocolos
```
GBML:        Train-once no chunk 0 (1000 treinos)
River:       Chunk-wise sequential (5000 treinos)
ACDWM:       Test-then-train prequential (6000 learn_one)
ERulesD2S:   Incremental contínuo
```

**Resultado:** Comparação MUITO INJUSTA (cada um usa protocolo diferente)
**Ação:** Padronizar TODOS para train-once ou chunk-wise

---

## Recomendação Preliminar

**Baseado nas descobertas até agora:**

1. **Confirmar protocolo do GBML** (próximo passo)

2. **Padronizar para train-once** (mais justo):
   ```python
   # Treina apenas no chunk 0
   model.train(chunks[0])

   # Testa em chunks 1-5 sem retreino
   for chunk in chunks[1:]:
       metrics = model.test(chunk)
   ```

3. **GBML sem memória** neste experimento:
   - Cada chunk treina do zero
   - Sem seeding de gerações passadas
   - Comparação justa de capacidade de generalização

4. **Experimento separado** (opcional):
   - GBML com memória vs River incremental
   - Avaliar capacidade de adaptação

---

## Status Atual da Investigação

- [X] Verificado protocolo do River (chunk-wise sequential)
- [X] Verificado suporte de modos do ACDWM
- [ ] Verificar protocolo do GBML (PRÓXIMO)
- [ ] Verificar modo atual do ACDWM
- [ ] Testar ACDWM em multiclasse
- [ ] Verificar protocolo do ERulesD2S
- [ ] Ler artigos

**Tempo decorrido:** ~30 minutos
**Próximo:** Verificar main.py e protocolo GBML (15-20 min)

---

**Criado por:** Claude Code
**Data:** 2025-11-22 09:45
**Atualizado:** Em andamento
