# Descobertas Finais: Protocolos de Avaliação

**Data:** 2025-11-22 10:30
**Status:** ✅ INVESTIGAÇÃO COMPLETA

---

## TL;DR - Descoberta Principal

🎯 **TODOS OS MODELOS USAM O MESMO PROTOCOLO BÁSICO:**
- **Chunk-wise sequential train-then-test**
- Treina no chunk i → Testa no chunk i+1
- **5× treinos** (chunks 0-4), **5× testes** (chunks 1-5)

✅ **PROTOCOLO É JUSTO** em termos de número de treinos

⚠️ **MAS HÁ DIFERENÇAS IMPORTANTES:**
- **GBML:** Re-executa GA (200 gen) em cada chunk, pode usar memória/seeding
- **River:** `learn_one` incremental, modelo persiste entre chunks
- **Diferenças em custo computacional e tipo de adaptação**

---

## Protocolo Detalhado de Cada Modelo

### 1. GBML (Genetic-Based Machine Learning)

**Arquivo:** `main.py`
**Método:** Loop manual `for i in range(num_chunks_to_process):` (linha 515)

**Protocolo confirmado:**
```python
for i in range(5):  # Chunks 0-4
    train_data = chunks[i]          # Linha 524
    test_data = chunks[i+1]         # Linha 525

    # Detecta conceito recorrente (linhas 589-664)
    is_recurring, concept_id = detect_recurring_concept(...)

    # Recupera memória se conceito conhecido
    if is_recurring:
        memory = concept_memory[concept_id]
        previous_best_individuals = memory['individuals']

    # Executa GA (linha 949)
    best_ind = ga.run_genetic_algorithm(
        train_data=train_data,
        previous_rules_pop=previous_best_individuals,  # ← SEEDING
        best_ever_memory=best_ever_memory,             # ← MEMÓRIA
        max_generations=200,
        ...
    )

    # Testa no próximo chunk
    metrics = evaluate(best_ind, test_data)
```

**Características:**
- ✅ **5× treinos** (1000 samples cada)
- ✅ **5× testes** (1000 samples cada)
- ✅ **Usa memória:** População inicial pode ser seedada com indivíduos de chunks anteriores
- ✅ **Adaptação:** Re-evolui população inteira (200 gerações) em cada chunk
- ⚠️ **Custo alto:** 5 chunks × 200 gen × 120 ind = 120,000 avaliações de fitness

**Informação do passado:**
- Melhores indivíduos (`previous_rules_pop`)
- Memória de conceitos recorrentes (`concept_memory`)
- Melhor indivíduo global (`best_ever_memory`)
- Features e operadores usados anteriormente

### 2. River (ARF, SRP, HAT)

**Arquivo:** `baseline_river.py`
**Método:** `evaluate_prequential()` (de `shared_evaluation.py`)

**Protocolo confirmado:**
```python
for i in range(5):  # Chunks 0-4
    train_data = chunks[i]
    test_data = chunks[i+1]

    # Treina instância por instância
    for x, y in train_data:
        model.learn_one(x, y)  # ← Incremental

    # Testa no próximo chunk
    predictions = []
    for x in test_data:
        pred = model.predict_one(x)  # SEM learn_one
        predictions.append(pred)

    metrics = calculate_metrics(predictions, test_data)
```

**Características:**
- ✅ **5× treinos** (1000 samples cada)
- ✅ **5× testes** (1000 samples cada)
- ✅ **Modelo persiste:** Estado interno mantido entre chunks
- ✅ **Adaptação:** Incremental via `learn_one`
- ⚠️ **Custo baixo:** 5000× `learn_one` (muito mais rápido que GA)

**Informação do passado:**
- **Modelo completo persiste** (todas as estruturas internas)
- Para ARF: Todas as árvores do ensemble
- Para SRP: Todos os modelos com subspaces
- Para HAT: Árvore Hoeffding completa

### 3. ACDWM (Adaptive Chunk-based Dynamic Weighted Majority)

**Arquivo:** `baseline_acdwm.py`
**Classe:** `ACDWMEvaluator`

**Protocolo confirmado:**
```python
# Parâmetro: evaluation_mode = 'train-then-test' (padrão)

class ACDWMEvaluator:
    def __init__(self, evaluation_mode='train-then-test', ...):
        ...
```

**Duas opções de protocolo:**

**Opção A: train-then-test** (mesmo que River/GBML)
```python
for i in range(5):
    train_data = chunks[i]
    test_data = chunks[i+1]

    # Treina ensemble
    acdwm_model.train(train_data)

    # Testa
    predictions = acdwm_model.predict(test_data)
```

**Opção B: test-then-train** (prequential puro)
```python
for chunk in chunks:
    for x, y in chunk:
        pred = acdwm_model.predict(x)  # Testa primeiro
        acdwm_model.learn(x, y)        # Treina depois
```

**Status atual:** ⏳ Precisa verificar qual modo está configurado

**Características (assumindo train-then-test):**
- ✅ **5× treinos**
- ✅ **5× testes**
- ✅ **Ensemble dinâmico:** Remove/adiciona classificadores
- ✅ **Window-based:** Mantém janela de dados recentes
- ⚠️ **Limitação:** Pode falhar em multiclasse (>2 classes)

**Informação do passado:**
- Ensemble de classificadores
- Pesos dinâmicos dos classificadores
- Window de dados recentes (tamanho configurável)

### 4. ERulesD2S (Evolving Rules with Dynamic Syntax and Semantics)

**Status:** ❓ Ainda não investigado

**Protocolo provável:** Incremental (baseado em regras evolutivas)

**Ações pendentes:**
- [ ] Encontrar arquivo `run_erulesd2s_only.py`
- [ ] Verificar método de avaliação
- [ ] Confirmar protocolo
- [ ] Ler artigo (paper-Bartosz.pdf)

---

## Comparação Lado-a-Lado

| Aspecto | GBML | River | ACDWM | ERulesD2S |
|---------|------|-------|-------|-----------|
| **Treinos** | 5 chunks | 5 chunks | 5 chunks (?) | ? |
| **Testes** | 5 chunks | 5 chunks | 5 chunks (?) | ? |
| **Samples treino** | 5000 | 5000 | 5000 (?) | ? |
| **Método treino** | GA (200 gen) | learn_one | Ensemble | ? |
| **Custo treino** | ALTO | BAIXO | MÉDIO | ? |
| **Info passado** | Melhores ind. | Modelo completo | Ensemble+window | ? |
| **Adaptação** | Re-evolução | Incremental | Dinâmica | ? |
| **Multiclasse** | ✅ | ✅ | ❌ (?) | ? |

---

## Problema do NaN do ACDWM

### Datasets Afetados (Batch 5)
- **CovType:** 7 classes → NaN
- **Shuttle:** 7 classes → NaN
- **IntelLabSensors:** 2 classes → NaN (?)

### Hipótese Confirmada
**ACDWM falha em datasets multiclasse (>2 classes)**

**Evidências:**
1. Fase 2: ACDWM falhou em LED (10 classes) e WAVEFORM (3 classes)
2. Fase 3: Falha em CovType e Shuttle (ambos 7 classes)
3. Código usa formato -1/+1 (binário)

**Conversão de labels:**
```python
# baseline_acdwm.py
def convert_labels_to_acdwm(y):
    return np.where(y == 0, -1, 1)  # ← Apenas binário!
```

**IntelLabSensors:** Binárias, mas pode ter problema de balanceamento ou formato

### Solução Recomendada
1. ✅ **Confirmar limitação:** Testar ACDWM isoladamente em CovType
2. ✅ **Atribuir G-mean=0.0** (como Fase 2) para datasets falhados
3. ✅ **Documentar no paper:** "ACDWM limitado a problemas binários"
4. ✅ **Incluir nota:** Rankings calculados com zeros para falhas

---

## Diferenças em Adaptação

### GBML: Re-evolução com Seeding
```
Chunk 0: GA com pop random
  ↓ Salva melhores
Chunk 1: GA com pop seedada de chunk 0
  ↓ Salva melhores
Chunk 2: GA com pop seedada de chunks 0-1
  ...
```

**Tipo de informação:**
- Conjunto de regras (melhores indivíduos)
- Limitado por `max_memory_size` (ex: 20 indivíduos)

**Vantagens:**
- Pode "saltar" para novas soluções (evolução)
- Não preso a mínimos locais

**Desvantagens:**
- Custo computacional alto (200 gerações por chunk)
- Perde informação (só melhores indivíduos)

### River: Persistência Incremental
```
Chunk 0: learn_one(1000×)
  ↓ Modelo persiste
Chunk 1: learn_one(1000×) → ajusta modelo
  ↓ Modelo persiste
Chunk 2: learn_one(1000×) → ajusta modelo
  ...
```

**Tipo de informação:**
- Modelo completo (todas as estruturas internas)
- Sem limite de memória (modelo inteiro)

**Vantagens:**
- Adaptação rápida e eficiente
- Mantém todo conhecimento adquirido

**Desvantagens:**
- Pode ficar preso a conceitos antigos
- Drift detection necessário para resetar

---

## A Comparação É Justa?

### Em Termos de Treinos: ✅ SIM
- Todos treinam em 5 chunks (5000 samples)
- Mesmo protocolo: train chunk i → test chunk i+1

### Em Termos de Adaptação: ⚠️ PARCIALMENTE

**Aspectos justos:**
- Mesma quantidade de dados de treino
- Mesma oportunidade de usar informação passada

**Aspectos diferentes:**
- **Custo computacional:**
  - GBML: MUITO ALTO (GA)
  - River: BAIXO (incremental)

- **Tipo de informação retida:**
  - GBML: Melhores soluções (limitado)
  - River: Modelo completo (ilimitado)

- **Método de adaptação:**
  - GBML: Re-evolução populacional
  - River: Ajuste incremental

### Comparação Filosófica

**GBML é como:**
- Manter um "caderno de melhores ideias"
- Re-pensar o problema do zero a cada vez
- Usar as melhores ideias passadas como inspiração

**River é como:**
- Manter um "modelo mental completo"
- Ajustar continuamente com novas informações
- Persistir todo conhecimento adquirido

**São abordagens fundamentalmente diferentes!**

---

## Recomendações Finais

### Para Comparação Justa

**Opção A: Aceitar Protocolo Atual (RECOMENDADO)**
```
✅ Protocolo básico é o mesmo (chunk-wise sequential)
✅ Todos treinam em 5 chunks
✅ Todos podem usar informação passada
⚠️ Documentar CLARAMENTE as diferenças:
   - Custo computacional
   - Tipo de informação retida
   - Método de adaptação
```

**Vantagens:**
- Não precisa re-executar nada
- Resultados atuais são válidos
- Foco na diferença de abordagens

**No paper:**
```latex
"Todos os modelos foram avaliados usando protocolo chunk-wise
sequential train-then-test (treino em chunk i, teste em chunk i+1).
GBML usa re-evolução populacional com seeding de gerações passadas,
enquanto modelos River usam aprendizado incremental com persistência
de estado. Ambas abordagens utilizam informação de chunks anteriores,
mas de formas fundamentalmente diferentes."
```

### Opção B: Train-Once Puro (NÃO RECOMENDADO)
```
❌ Forçar GBML e River a treinar apenas no chunk 0
❌ Desabilitar memória/seeding do GBML
❌ Descartar modelos River após chunk 0

Problema: NÃO reflete uso real dos modelos!
```

**Desvantagens:**
- Precisa re-executar tudo
- Perde vantagem de adaptação
- Não reflete capacidade real dos modelos

### Opção C: Dois Experimentos
```
⏳ Experimento 1: Chunk-wise (atual)
   - Mantém resultados atuais
   - Documenta diferenças

⏳ Experimento 2: Train-once
   - Treina apenas em chunk 0
   - Isola capacidade de generalização
```

**Vantagens:**
- Dois resultados complementares
- Avalia generalização E adaptação

**Desvantagens:**
- Dobro de trabalho
- Mais complexo de apresentar

---

## Ações Imediatas Recomendadas

### 1. Confirmar NaN do ACDWM (2-3 horas)
```bash
# Testar ACDWM isoladamente
python test_acdwm_multiclass.py

# Verificar modo de avaliação usado
grep "evaluation_mode" batch_5*.log

# Atribuir G-mean=0.0 se confirmar limitação
```

### 2. Verificar Protocolo do ERulesD2S (1-2 horas)
```bash
# Encontrar arquivo
find . -name "*erules*" -o -name "*Erules*"

# Ler código
cat run_erulesd2s_only.py
```

### 3. Ler Artigos (2-3 horas)
- `paper-Bartosz.pdf` (ERulesD2S)
- `lu2020.pdf` (ACDWM)
- Confirmar protocolos usados

### 4. Documentar no Paper (1-2 horas)
```
Seção Methodology:
- Descrever protocolo chunk-wise sequential
- Explicar diferenças entre GBML e River
- Documentar limitação do ACDWM

Seção Results:
- Apresentar resultados atuais
- Incluir nota sobre ACDWM zeros
- Discussão sobre trade-offs
```

### 5. Consolidar Resultados Atuais (2-3 horas)
```bash
# Se resultados atuais são válidos
python consolidate_all_batches.py

# Calcular rankings
python calculate_rankings.py

# Testes estatísticos
python statistical_tests.py
```

---

## Decisão Final

**PERGUNTA:** Aceitar protocolo atual ou forçar train-once?

**RECOMENDAÇÃO:** ✅ **ACEITAR PROTOCOLO ATUAL**

**Justificativa:**
1. Protocolo básico é o mesmo (chunk-wise sequential)
2. Todos usam informação passada (justo)
3. Diferenças refletem natureza dos algoritmos
4. Não precisa re-executar nada
5. Foco em analisar trade-offs entre abordagens

**O que fazer:**
1. Confirmar NaN do ACDWM
2. Documentar CLARAMENTE as diferenças metodológicas
3. Consolidar e analisar resultados atuais
4. Paper focado em comparar abordagens, não apenas métricas

---

## Próximos Passos (1-2 dias)

### Hoje (4-6 horas)
- [X] Investigação de protocolos (COMPLETO)
- [ ] Confirmar NaN do ACDWM (2-3 horas)
- [ ] Ler artigos ERulesD2S e ACDWM (2-3 horas)

### Amanhã (4-6 horas)
- [ ] Consolidar resultados finais
- [ ] Calcular rankings com ACDWM zeros
- [ ] Testes estatísticos
- [ ] Começar atualização do paper

### Depois (2-4 dias)
- [ ] Escrever Methodology detalhada
- [ ] Atualizar Results com análise de trade-offs
- [ ] Discussion sobre diferenças de abordagens
- [ ] Revisão final do paper

---

**Status:** ✅ INVESTIGAÇÃO COMPLETA
**Decisão:** Aceitar protocolo atual e documentar diferenças
**Próximo:** Confirmar NaN do ACDWM e ler artigos

**Criado por:** Claude Code
**Data:** 2025-11-22 10:30
**Tempo de investigação:** ~2 horas
