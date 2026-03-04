# OTIMIZAÇÕES DE MEMORY RECORRENTE - FASE 2

**Data de Criação:** 2025-10-31
**Status:** 📝 PRÉ-IMPLEMENTAÇÃO
**Objetivo:** Aproveitar conceitos recorrentes para recovery imediata (+30-40% G-mean)
**Autor:** Análise conjunta via Claude Code
**Dependência:** Fase 1 completa e validada

---

## 📊 PROBLEMA IDENTIFICADO

### Evidência dos Experimentos Atuais

**RBF_Drift_Recovery** (c1 → c3_moderate → c1):
```
Chunk 0: c1 → 88.94% G-mean ✓
Chunk 3: c1 (retorna) → 44.56% G-mean ✗ (-44.38pp)
Chunk 4: c1 (recovery) → 90.52% G-mean ✓
```

**RBF_Multi_Drift** (c1 → c3_moderate → c1 → c2_severe):
```
Chunk 0: c1 → 89.51% G-mean ✓
Chunk 3: c1 (retorna) → 43.76% G-mean ✗ (-45.75pp)
Chunk 4: c1 (recovery) → 90.73% G-mean ✓
```

### Análise do Comportamento Atual

**PROBLEMA CRÍTICO:** Sistema "esquece" conceitos conhecidos

1. **Chunk 0:** c1 é aprendido com sucesso (~89% G-mean)
2. **Chunks 1-2:** Conceito muda, memory mantém elite de c1
3. **Chunk 3:** c1 RETORNA mas:
   - ❌ Memory de c1 foi abandonada (drift detection)
   - ❌ Sistema não reconhece que c1 já foi visto
   - ❌ Reinicializa população do zero (full_random)
   - **Resultado:** Colapso para ~44% G-mean
4. **Chunk 4:** Recovery lenta via seeding 90%

**POR QUE ISSO ACONTECE:**

Código atual em `main.py:987-991`:
```python
if best_ever_memory:
    original_size = len(best_ever_memory)
    keep_size = max(1, original_size // 10)  # Mantém apenas 10%
    best_ever_memory = sorted(best_ever_memory, key=lambda ind: ind.fitness, reverse=True)[:keep_size]
```

E em `main.py:723-727`:
```python
if should_abandon_memory_now and best_ever_memory:
    logger.warning(f"PERFORMANCE CRITICAL. Abandoning best_ever_memory.")
    best_ever_memory.clear()  # LIMPA TUDO!
```

**CONSEQUÊNCIA:**
- Memory é reduzida/limpa agressivamente
- Indivíduos de c1 são perdidos
- Quando c1 retorna, não há como recuperá-los

---

## 🎯 SOLUÇÃO PROPOSTA: CONCEPT FINGERPRINTING

### Ideia Central

**Em vez de apenas armazenar "melhores indivíduos", armazenar "melhores indivíduos POR CONCEITO":**

```python
# Estrutura atual (RUIM):
best_ever_memory = [ind1, ind2, ind3, ...]  # Lista linear

# Estrutura nova (BOA):
concept_memory = {
    "fingerprint_c1": [ind1, ind2, ind3],     # Indivíduos de c1
    "fingerprint_c2": [ind4, ind5, ind6],     # Indivíduos de c2_severe
    "fingerprint_c3": [ind7, ind8, ind9]      # Indivíduos de c3_moderate
}
```

### Como Funciona

**1. Calcular Fingerprint do Chunk:**
```python
def calculate_concept_fingerprint(data, target):
    """
    Gera um fingerprint único baseado em estatísticas do chunk.
    Conceitos similares terão fingerprints similares.
    """
    fingerprint = {
        'mean': np.mean(data, axis=0),         # Média de cada feature
        'std': np.std(data, axis=0),           # Desvio padrão
        'class_dist': np.bincount(target) / len(target),  # Distribuição de classes
        'correlations': calculate_feature_correlations(data)  # Correlações
    }
    return fingerprint
```

**2. Comparar Fingerprints:**
```python
def fingerprint_similarity(fp1, fp2):
    """
    Calcula similaridade entre dois fingerprints.
    Retorna valor entre 0 (completamente diferente) e 1 (idêntico).
    """
    # Similaridade de médias (cosine similarity)
    mean_sim = cosine_similarity(fp1['mean'], fp2['mean'])

    # Similaridade de distribuição de classes
    class_sim = 1 - np.sum(np.abs(fp1['class_dist'] - fp2['class_dist'])) / 2

    # Peso balanceado
    similarity = 0.6 * mean_sim + 0.4 * class_sim

    return similarity
```

**3. Detectar Conceito Recorrente:**
```python
def detect_recurring_concept(current_fp, concept_memory, threshold=0.85):
    """
    Verifica se chunk atual corresponde a um conceito já visto.
    """
    for concept_id, stored_fp in concept_memory.items():
        similarity = fingerprint_similarity(current_fp, stored_fp['fingerprint'])

        if similarity > threshold:  # 85% de similaridade
            return concept_id, similarity

    return None, 0.0
```

**4. Restaurar Memory de Conceito Recorrente:**
```python
if is_recurring:
    # Recupera indivíduos do conceito conhecido
    best_ever_memory = copy.deepcopy(concept_memory[recurring_concept_id]['individuals'])
    logger.info(f"🔄 CONCEITO RECORRENTE detectado! Restaurando {len(best_ever_memory)} indivíduos.")

    # Usa memory para inicializar população (em vez de full_random)
    initialization_strategy = "memory_based"
```

---

## 🛠️ IMPLEMENTAÇÃO DETALHADA

### MUDANÇA #1: Nova Estrutura de Memória

**Arquivo:** `main.py`
**Localização:** Linha ~490 (inicialização)

**ANTES:**
```python
best_ever_memory = []
```

**DEPOIS:**
```python
# FASE 2.1: Memory organizada por conceito (concept fingerprinting)
best_ever_memory = []  # Mantém para compatibilidade
concept_memory = {}     # NOVO: {concept_id: {'fingerprint': fp, 'individuals': [...]}}
concept_fingerprints = {}  # NOVO: {chunk_idx: fingerprint}
```

---

### MUDANÇA #2: Calcular Fingerprint ao Processar Chunk

**Arquivo:** `main.py`
**Localização:** Após linha ~516 (antes de treinar)

**Código Novo:**
```python
# FASE 2.1: Calcular fingerprint do chunk atual
current_fingerprint = utils.calculate_concept_fingerprint(
    train_data_chunk,
    train_target_chunk,
    attributes
)
concept_fingerprints[i] = current_fingerprint

# Detectar se é conceito recorrente
recurring_concept_id, similarity = utils.detect_recurring_concept(
    current_fingerprint,
    concept_memory,
    threshold=0.85
)

if recurring_concept_id is not None:
    logger.warning(f"🔄 CONCEITO RECORRENTE detectado! "
                  f"Chunk {i} é similar ({similarity:.1%}) ao conceito '{recurring_concept_id}'")

    # Restaura memory do conceito conhecido
    best_ever_memory = copy.deepcopy(concept_memory[recurring_concept_id]['individuals'])
    logger.info(f"   → Restaurando {len(best_ever_memory)} indivíduos do conceito '{recurring_concept_id}'")

    # Override initialization strategy para usar memory
    if enable_explicit_drift_adaptation:
        effective_initialization_strategy = "memory_based"
        logger.info(f"   → Initialization strategy overridden para 'memory_based'")
else:
    logger.info(f"Chunk {i}: Novo conceito detectado (não recorrente)")
    # Continua com lógica normal
```

---

### MUDANÇA #3: Salvar Memory por Conceito

**Arquivo:** `main.py`
**Localização:** Após linha ~900 (após adicionar à memory)

**Código Novo:**
```python
# FASE 2.1: Salvar indivíduos no concept_memory
if best_individual:
    # Adiciona à memory normal (compatibilidade)
    best_ever_memory.append(copy.deepcopy(best_individual))
    best_ever_memory.sort(key=lambda ind: (ind.fitness, ind.creation_chunk_index), reverse=True)
    best_ever_memory = best_ever_memory[:max_memory_size]

    # NOVO: Salva também no concept_memory
    current_fp = concept_fingerprints.get(i)
    if current_fp is not None:
        # Gera concept_id único para este fingerprint
        concept_id = f"concept_chunk{i}"

        # Verifica se já existe conceito similar
        existing_concept, sim = utils.detect_recurring_concept(
            current_fp,
            concept_memory,
            threshold=0.85
        )

        if existing_concept:
            # Adiciona ao conceito existente
            concept_id = existing_concept
            concept_memory[concept_id]['individuals'].append(copy.deepcopy(best_individual))
            # Mantém apenas top-K
            concept_memory[concept_id]['individuals'].sort(key=lambda x: x.fitness, reverse=True)
            concept_memory[concept_id]['individuals'] = concept_memory[concept_id]['individuals'][:max_memory_size]
            logger.debug(f"Adicionado indivíduo ao conceito existente '{concept_id}'")
        else:
            # Cria novo conceito
            concept_memory[concept_id] = {
                'fingerprint': current_fp,
                'individuals': [copy.deepcopy(best_individual)],
                'first_seen_chunk': i,
                'last_seen_chunk': i
            }
            logger.info(f"Novo conceito '{concept_id}' criado na memory")
```

---

### MUDANÇA #4: Não Abandonar Memory de Conceitos Recorrentes

**Arquivo:** `main.py`
**Localização:** Linha ~723-727 (abandon memory)

**ANTES:**
```python
if should_abandon_memory_now and best_ever_memory:
    logger.warning(f"PERFORMANCE CRITICAL. Abandoning best_ever_memory.")
    best_ever_memory.clear()
```

**DEPOIS:**
```python
if should_abandon_memory_now and best_ever_memory:
    logger.warning(f"PERFORMANCE CRITICAL. Checking if should abandon memory...")

    # FASE 2.1: NÃO abandona memory se for conceito recorrente
    if recurring_concept_id is not None:
        logger.warning(f"   → Memory NÃO abandonada (conceito recorrente '{recurring_concept_id}')")
    else:
        logger.warning(f"   → Abandoning best_ever_memory (novo conceito)")
        best_ever_memory.clear()
        # NÃO limpa concept_memory (mantém histórico de todos os conceitos)
```

---

### MUDANÇA #5: Ajustar Parâmetros de Abandon

**Arquivo:** `config_test_drift_recovery.yaml`, `config_test_multi_drift.yaml`
**Localização:** memory_params

**ANTES:**
```yaml
memory_params:
  abandon_memory_on_severe_performance_drop: true
  performance_drop_threshold_for_memory_abandon: 0.55
  consecutive_bad_chunks_for_memory_abandon: 1  # MUITO AGRESSIVO
```

**DEPOIS:**
```yaml
memory_params:
  abandon_memory_on_severe_performance_drop: true
  performance_drop_threshold_for_memory_abandon: 0.40  # Mais conservador (55% → 40%)
  consecutive_bad_chunks_for_memory_abandon: 2  # Requer 2 chunks ruins (era 1)

  # NOVO: Parâmetros de concept fingerprinting
  enable_concept_fingerprinting: true
  concept_similarity_threshold: 0.85  # 85% de similaridade para considerar recorrente
  never_abandon_memory_for_recurring: true  # Nunca abandona conceito recorrente
```

---

### MUDANÇA #6: Implementar Funções em utils.py

**Arquivo:** `utils.py` (criar funções novas)

**Funções a Implementar:**

```python
def calculate_concept_fingerprint(data, target, attributes):
    """
    Calcula fingerprint estatístico de um chunk.

    Args:
        data: List of dicts (instâncias)
        target: List of labels
        attributes: List of feature names

    Returns:
        dict: Fingerprint com estatísticas do conceito
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    # Converte data para numpy array
    data_array = np.array([[inst[attr] for attr in attributes] for inst in data])
    target_array = np.array(target)

    fingerprint = {
        # Estatísticas das features
        'mean': np.mean(data_array, axis=0).tolist(),
        'std': np.std(data_array, axis=0).tolist(),
        'median': np.median(data_array, axis=0).tolist(),

        # Distribuição de classes
        'class_distribution': np.bincount(target_array) / len(target_array),

        # Correlações entre features (simplificado)
        'feature_correlations_sum': np.sum(np.corrcoef(data_array.T)),

        # Range das features
        'min': np.min(data_array, axis=0).tolist(),
        'max': np.max(data_array, axis=0).tolist()
    }

    return fingerprint


def fingerprint_similarity(fp1, fp2):
    """
    Calcula similaridade entre dois fingerprints.

    Returns:
        float: Valor entre 0 (diferente) e 1 (idêntico)
    """
    from scipy.spatial.distance import cosine

    # Similaridade de médias (cosine similarity)
    mean1 = np.array(fp1['mean'])
    mean2 = np.array(fp2['mean'])
    mean_sim = 1 - cosine(mean1, mean2) if len(mean1) > 0 else 0.0

    # Similaridade de distribuição de classes
    class_dist1 = fp1['class_distribution']
    class_dist2 = fp2['class_distribution']

    # Garante mesmo tamanho (padding com zeros)
    max_len = max(len(class_dist1), len(class_dist2))
    cd1 = np.pad(class_dist1, (0, max_len - len(class_dist1)), mode='constant')
    cd2 = np.pad(class_dist2, (0, max_len - len(class_dist2)), mode='constant')

    class_sim = 1 - np.sum(np.abs(cd1 - cd2)) / 2

    # Similaridade de desvio padrão
    std1 = np.array(fp1['std'])
    std2 = np.array(fp2['std'])
    std_sim = 1 - cosine(std1, std2) if len(std1) > 0 else 0.0

    # Peso balanceado
    similarity = 0.5 * mean_sim + 0.3 * class_sim + 0.2 * std_sim

    return similarity


def detect_recurring_concept(current_fp, concept_memory, threshold=0.85):
    """
    Detecta se chunk atual corresponde a conceito já visto.

    Args:
        current_fp: Fingerprint do chunk atual
        concept_memory: Dict com conceitos conhecidos
        threshold: Threshold de similaridade (0-1)

    Returns:
        tuple: (concept_id or None, similarity)
    """
    best_match_id = None
    best_similarity = 0.0

    for concept_id, concept_data in concept_memory.items():
        stored_fp = concept_data['fingerprint']
        similarity = fingerprint_similarity(current_fp, stored_fp)

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_id = concept_id

    if best_similarity >= threshold:
        return best_match_id, best_similarity
    else:
        return None, 0.0
```

---

## 📈 GANHO ESPERADO

### Estimativa de Impacto

**RBF_Drift_Recovery:**
```
ANTES (Fase 1):
Chunk 0: c1 → 88.94%
Chunk 1: c3 → 55.27% (colapso)
Chunk 2: c3 → 89.40% (recovery)
Chunk 3: c1 → 44.56% (colapso ao retornar)  ← PROBLEMA
Chunk 4: c1 → 90.52% (recovery)
Chunk 5: c3 → 55.00% (estimado, colapso)

Média: ~72.1%

DEPOIS (Fase 2):
Chunk 0: c1 → 88.94%
Chunk 1: c3 → 55.27% (colapso - novo conceito)
Chunk 2: c3 → 89.40% (recovery)
Chunk 3: c1 → 88.94% (RECOVERY IMEDIATA!)  ← FIX
Chunk 4: c1 → 90.00% (mantém)
Chunk 5: c3 → 89.40% (RECOVERY IMEDIATA!)  ← FIX

Média: ~83.7%

GANHO: +11.6 pontos percentuais (+16% relativo)
```

**RBF_Multi_Drift:**
```
ANTES:
Avg G-mean: 69.96%

DEPOIS (estimado):
Avg G-mean: 82-85%

GANHO: +12-15 pontos percentuais (+17-21% relativo)
```

---

## ✅ CRITÉRIOS DE SUCESSO

| Métrica | Baseline (Fase 1) | Meta (Fase 2) | Como Medir |
|---------|------------------|---------------|------------|
| **Recovery G-mean Chunk 3** | 44.56% | > 85% | Chunk de retorno a c1 |
| **Recovery Multi Chunk 3** | 43.76% | > 85% | Chunk de retorno a c1 |
| **Avg G-mean Recovery** | 73.74% | > 82% | Média de 6 chunks |
| **Avg G-mean Multi** | 69.96% | > 82% | Média de 8 chunks |
| **Detecção Recorrência** | 0% | > 90% | Taxa de detecção correta |
| **False Positive** | N/A | < 5% | Falsos positivos de recorrência |

---

## 🧪 VALIDAÇÃO PLANEJADA

### Experimento 1: RBF_Drift_Recovery
**Config:** `config_test_drift_recovery.yaml`
**Stream:** c1 (2 chunks) → c3_moderate (2 chunks) → c1 (2 chunks)

**Validações:**
- [ ] Chunk 0 (c1): Cria conceito "concept_chunk0" na memory
- [ ] Chunk 2 (c3): Cria conceito "concept_chunk2" na memory
- [ ] Chunk 4 (c1 retorna): Detecta similaridade > 85% com "concept_chunk0"
- [ ] Chunk 4: Restaura memory de c1 (não inicializa do zero)
- [ ] Chunk 4: G-mean > 85% (vs 44.56% baseline)

### Experimento 2: RBF_Multi_Drift
**Config:** `config_test_multi_drift.yaml`
**Stream:** c1 → c3 → c1 → c2

**Validações:**
- [ ] Chunk 0 (c1): Cria conceito na memory
- [ ] Chunk 4 (c1 retorna): Detecta recorrência, G-mean > 85%
- [ ] Chunk 6 (c2): Novo conceito (não recorrente), comportamento normal

---

## 🔧 PLANO DE IMPLEMENTAÇÃO

### Passo 1: Criar Funções em utils.py (30min)
- `calculate_concept_fingerprint()`
- `fingerprint_similarity()`
- `detect_recurring_concept()`

### Passo 2: Modificar main.py - Estrutura (15min)
- Adicionar `concept_memory` e `concept_fingerprints`

### Passo 3: Modificar main.py - Calcular Fingerprint (20min)
- Calcular fingerprint antes de treinar
- Detectar conceito recorrente
- Restaurar memory se recorrente

### Passo 4: Modificar main.py - Salvar por Conceito (30min)
- Salvar indivíduos no `concept_memory`
- Associar ao conceito correto

### Passo 5: Modificar main.py - Não Abandonar Recorrente (10min)
- Verificar se conceito é recorrente antes de abandonar

### Passo 6: Atualizar Configs (5min)
- Ajustar parâmetros de abandon
- Adicionar parâmetros de fingerprinting

### Passo 7: Testar e Validar (2-3 horas)
- Executar RBF_Drift_Recovery
- Executar RBF_Multi_Drift
- Comparar com baseline

**Tempo Total Estimado:** 2-3 horas implementação + 2-3 horas validação = **4-6 horas**

---

## 📋 CHECKLIST DE IMPLEMENTAÇÃO

### Pré-Implementação
- [x] Documentar baseline Fase 1
- [x] Definir estrutura de concept_memory
- [x] Calcular ganhos esperados
- [ ] Criar funções em utils.py
- [ ] Backup de main.py

### Implementação
- [ ] utils.py: calculate_concept_fingerprint()
- [ ] utils.py: fingerprint_similarity()
- [ ] utils.py: detect_recurring_concept()
- [ ] main.py: Adicionar concept_memory
- [ ] main.py: Calcular fingerprint por chunk
- [ ] main.py: Detectar recorrência
- [ ] main.py: Restaurar memory
- [ ] main.py: Salvar por conceito
- [ ] main.py: Não abandonar recorrente
- [ ] configs: Ajustar parâmetros

### Validação
- [ ] Executar RBF_Drift_Recovery
- [ ] Verificar detecção de recorrência
- [ ] Comparar G-mean chunk 3
- [ ] Executar RBF_Multi_Drift
- [ ] Comparar média geral
- [ ] Atualizar documentação com resultados

---

**Documento criado por:** Claude Code
**Data:** 2025-10-31
**Status:** 📝 PRÉ-IMPLEMENTAÇÃO
**Próxima Atualização:** Após implementação e validação
