# 🔄 Análise: Configuração run_mode (drift_simulation vs standard)

**Data:** 07/10/2025
**Status:** ⚠️ **PROBLEMA CRÍTICO IDENTIFICADO**

---

## 🎯 RESUMO EXECUTIVO

O sistema GBML possui **2 modos de execução** diferentes, mas há um **erro de configuração crítico**:

| Aspecto | Status |
|---------|--------|
| **Implementação dos modos** | ✅ Correta |
| **Detecção automática** | ✅ Funciona |
| **Configuração atual** | ❌ **INVÁLIDA** |
| **Impacto** | ⚠️ **main.py não pode executar** |

---

## 📋 OS DOIS MODOS DE EXECUÇÃO

### **Modo 1: `standard`** (Datasets Reais/Estacionários)

**Propósito:** Experimentos com datasets reais ou geradores estacionários (sem drift explícito).

**Configuração:**
```yaml
experiment_settings:
  run_mode: 'standard'
  standard_experiments:
    - Electricity          # Dataset real
    - CovType              # Dataset real
    - PokerHand            # Dataset real
    - SEA_Stationary       # Gerador estacionário (sem drift)
    - AGRAWAL_Stationary   # Gerador estacionário (sem drift)
```

**Características:**
- Streams **não têm** `concept_sequence` definido
- Sistema **não tenta** detectar drift entre chunks
- Métricas de severidade de drift **não são calculadas**
- Penalidades inter-chunk **desabilitadas**

**Exemplos válidos:**
- `Electricity`, `Shuttle`, `CovType`, `PokerHand`, `IntelLabSensors` (reais)
- `SEA_Stationary`, `AGRAWAL_Stationary`, `RBF_Stationary`, etc. (sintéticos estacionários)

---

### **Modo 2: `drift_simulation`** (Simulação de Drift)

**Propósito:** Experimentos controlados com drift sintético (abrupt, gradual, recurring, etc.).

**Configuração:**
```yaml
experiment_settings:
  run_mode: 'drift_simulation'
  drift_simulation_experiments:
    - SEA_Abrupt_Simple
    - AGRAWAL_Gradual_Chain
    - RBF_Abrupt_Severe
```

**Características:**
- Streams **TÊM** `concept_sequence` definido
- Sistema **detecta** mudanças de conceito entre chunks
- Métricas de severidade de drift **são calculadas**
- Penalidades inter-chunk **ativadas** (operator_change, feature_change)

**Exemplos válidos:**
- `SEA_Abrupt_Simple`, `SEA_Gradual_Recurring`
- `AGRAWAL_Abrupt_Simple_Mild`, `AGRAWAL_Gradual_Chain`
- `RBF_Abrupt_Severe`, `HYPERPLANE_Gradual_Simple`
- Qualquer stream com `concept_sequence` em `experimental_streams`

---

## ⚠️ PROBLEMA CRÍTICO IDENTIFICADO

### **config.yaml Atual (INVÁLIDO):**

```yaml
experiment_settings:
  run_mode: 'standard'  # ← Modo STANDARD selecionado
  standard_experiments:
    - CovType
    - PokerHand
    # ❌ FALTA: drift_simulation_experiments
```

**ERRO:** A seção `drift_simulation_experiments` **NÃO EXISTE** no config.yaml!

### **Consequências:**

1. ✅ **Modo `standard` funciona** (CovType, PokerHand)
2. ❌ **Modo `drift_simulation` NÃO FUNCIONA**
   - Se mudar `run_mode: 'drift_simulation'`, main.py **FALHA IMEDIATAMENTE**:
   ```
   ERROR: Run mode is 'drift_simulation' but the 'drift_simulation_experiments' list is missing or empty in config.yaml. Exiting.
   ```

3. ❌ **compare_gbml_vs_river.py NÃO É AFETADO**
   - Wrapper usa `data_handling.generate_stream()` diretamente
   - Detecta automaticamente se é drift ou estacionário pela presença de `concept_sequence`
   - Não depende de `run_mode` do config.yaml

---

## 🔍 ANÁLISE TÉCNICA DETALHADA

### **1. Implementação em main.py (linhas 1413-1437)**

```python
run_mode = exp_settings.get('run_mode', 'standard').lower()
logger.info(f"Execution mode set to: '{run_mode}'")

stream_names_to_run = []
is_drift_mode = False

if run_mode == 'standard':
    stream_names_to_run = exp_settings.get('standard_experiments', [])
    is_drift_mode = False
    if not stream_names_to_run:
        logger.error("Run mode is 'standard' but 'standard_experiments' is missing/empty. Exiting.")
        exit(1)
    logger.info(f"Running in STANDARD mode. Found {len(stream_names_to_run)} experiments.")

elif run_mode == 'drift_simulation':
    stream_names_to_run = exp_settings.get('drift_simulation_experiments', [])  # ← BUSCA ESTA SEÇÃO
    is_drift_mode = True
    if not stream_names_to_run:
        logger.error("Run mode is 'drift_simulation' but 'drift_simulation_experiments' is missing/empty. Exiting.")
        exit(1)  # ← FALHA AQUI SE SEÇÃO NÃO EXISTIR
    logger.info(f"Running in DRIFT SIMULATION mode. Found {len(stream_names_to_run)} experiments.")

else:
    logger.error(f"Invalid 'run_mode': '{run_mode}'. Must be 'standard' or 'drift_simulation'. Exiting.")
    exit(1)
```

**Flag propagado:**
```python
run_result = run_experiment(
    experiment_id=stream_name,
    run_number=i + 1,
    config=config,
    base_results_dir=base_results_dir_main,
    is_drift_simulation=is_drift_mode  # ← Passa flag para run_experiment()
)
```

---

### **2. Uso do Flag `is_drift_simulation` em main.py**

**Detecção de drift (linha 353):**
```python
if is_drift_simulation:
    stream_definition = data_handling.get_stream_definition(experiment_id, config)
    # Carrega definição completa do stream (concept_sequence, drift_type, etc.)
```

**Tracking de conceitos (linha 574):**
```python
if is_drift_simulation and stream_definition:
    current_full_concept_info_tuple = data_handling.get_concept_for_chunk(stream_definition, i)
    # Retorna (id_a, id_b, p_mix_b) para o chunk atual
```

**Detecção de mudança de conceito (linha 592):**
```python
if is_drift_simulation and previous_full_concept_info_tuple is not None:
    if current_id_a != prev_id_a:  # Conceito mudou!
        # Ativa penalidades inter-chunk, memória ativa, etc.
```

**Pruning de memória (linha 803):**
```python
if mem_p_config.get('enable_active_memory_pruning', False) and \
   (is_drift_simulation and reduce_change_penalties_flag):
    # Remove indivíduos velhos da memória após drift
```

**Salvamento de metadados (linha 1172):**
```python
'run_mode': 'Drift Simulation' if is_drift_simulation else 'Standard',
'stream_definition': stream_definition if is_drift_simulation else None,
```

---

### **3. Detecção Automática em data_handling.py (linha 141)**

```python
is_drift_simulation = 'concept_sequence' in stream_config
```

**Lógica:**
- Se `stream_config` tem `concept_sequence` → **drift_simulation = True**
- Se `stream_config` NÃO tem `concept_sequence` → **drift_simulation = False**

**Exemplos:**
```yaml
# DRIFT SIMULATION (concept_sequence presente)
AGRAWAL_Abrupt_Simple_Mild:
  dataset_type: 'AGRAWAL'
  drift_type: 'abrupt'
  concept_sequence:
    - { concept_id: 'f1', duration_chunks: 5 }
    - { concept_id: 'f2', duration_chunks: 5 }

# STANDARD (sem concept_sequence)
AGRAWAL_Stationary:
  dataset_type: 'AGRAWAL'
```

---

## ✅ CORREÇÃO NECESSÁRIA

### **Adicionar Seção Faltante no config.yaml**

```yaml
experiment_settings:
  run_mode: 'drift_simulation'  # ← Pode alternar entre 'standard' e 'drift_simulation'

  # Lista para modo STANDARD
  standard_experiments:
    - Electricity
    - Shuttle
    - CovType
    - PokerHand
    - IntelLabSensors
    - SEA_Stationary
    - AGRAWAL_Stationary
    - RBF_Stationary
    - LED_Stationary
    - HYPERPLANE_Stationary
    - RANDOMTREE_Stationary
    - STAGGER_Stationary
    - WAVEFORM_Stationary
    - SINE_Stationary
    - AssetNegotiation_F2
    - AssetNegotiation_F3
    - AssetNegotiation_F4

  # ✅ ADICIONAR: Lista para modo DRIFT_SIMULATION
  drift_simulation_experiments:
    # Categoria 1: Abrupt Simple (2 conceitos)
    - SEA_Abrupt_Simple
    - AGRAWAL_Abrupt_Simple_Mild
    - AGRAWAL_Abrupt_Simple_Severe
    - RBF_Abrupt_Simple
    - HYPERPLANE_Abrupt_Simple

    # Categoria 2: Gradual Simple (2 conceitos)
    - SEA_Gradual_Simple
    - AGRAWAL_Gradual_Simple
    - RBF_Gradual_Simple

    # Categoria 3: Recurring (3 conceitos, volta ao inicial)
    - SEA_Abrupt_Recurring
    - AGRAWAL_Gradual_Recurring
    - RBF_Gradual_Recurring

    # Categoria 4: Chain (3-4 conceitos sequenciais)
    - SEA_Abrupt_Chain
    - AGRAWAL_Gradual_Chain
    - AGRAWAL_Abrupt_Chain_Long

    # Categoria 5: Severe Drift
    - RBF_Abrupt_Severe
    - AGRAWAL_Abrupt_Simple_Severe

    # Categoria 6: Com Ruído
    - SEA_Gradual_Simple_Noise
    - AGRAWAL_Gradual_Recurring_Noise
    - RBF_Abrupt_Blip_Noise

    # Categoria 7: Outros geradores
    - SINE_Abrupt_Simple
    - SINE_Gradual_Recurring
    - LED_Abrupt_Simple
    - WAVEFORM_Abrupt_Simple
    - RANDOMTREE_Abrupt_Simple
    - STAGGER_Gradual_Chain
    - HYPERPLANE_Gradual_Noise

  num_runs: 1
  base_results_dir: "drift_experiment_results"  # ← Nome diferente para cada modo
  logging_level: "INFO"
  evaluation_period: 6000
```

---

## 🧪 TESTES RECOMENDADOS

### **Teste 1: Validar Modo STANDARD (já funciona)**

```bash
# config.yaml:
#   run_mode: 'standard'
#   standard_experiments: [CovType, PokerHand]

python main.py
```

**Resultado esperado:**
```
[INFO] Execution mode set to: 'standard'
[INFO] Running in STANDARD mode. Found 2 experiments to run.
[INFO] --- Starting Experiment: ID='CovType', Run=1, Mode='Standard' ---
```

---

### **Teste 2: Validar Modo DRIFT_SIMULATION (após correção)**

```bash
# config.yaml:
#   run_mode: 'drift_simulation'
#   drift_simulation_experiments: [SEA_Abrupt_Simple, AGRAWAL_Abrupt_Simple_Mild]

python main.py
```

**Resultado esperado:**
```
[INFO] Execution mode set to: 'drift_simulation'
[INFO] Running in DRIFT SIMULATION mode. Found 2 experiments to run.
[INFO] --- Starting Experiment: ID='SEA_Abrupt_Simple', Run=1, Mode='Drift Simulation' ---
[INFO] Mode: Drift Simulation
[INFO] Generating 10 chunks with drift sequence...
```

---

### **Teste 3: Validar Alternância de Modo**

**Etapa 1 - Standard:**
```yaml
run_mode: 'standard'
base_results_dir: "standard_results"
```
```bash
python main.py
# Resultados em: standard_results/CovType_run001/
```

**Etapa 2 - Drift:**
```yaml
run_mode: 'drift_simulation'
base_results_dir: "drift_results"
```
```bash
python main.py
# Resultados em: drift_results/SEA_Abrupt_Simple_run001/
```

---

### **Teste 4: Validar compare_gbml_vs_river.py (NÃO É AFETADO)**

```bash
# Funciona independente de run_mode
python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --chunks 2
python compare_gbml_vs_river.py --stream CovType --chunks 2
```

**Motivo:** Wrapper usa `data_handling.generate_stream()` que detecta automaticamente o modo pela presença de `concept_sequence`.

---

## 📊 COMPARAÇÃO: main.py vs compare_gbml_vs_river.py

| Aspecto | main.py | compare_gbml_vs_river.py |
|---------|---------|--------------------------|
| **Depende de run_mode** | ✅ Sim | ❌ Não |
| **Requer drift_simulation_experiments** | ✅ Sim (se modo drift) | ❌ Não |
| **Detecção automática** | ❌ Não | ✅ Sim (via concept_sequence) |
| **Penalidades inter-chunk** | ✅ Ativadas (se drift) | ⚠️ Não implementadas no wrapper |
| **Métricas de severidade** | ✅ Calculadas (se drift) | ⚠️ Não implementadas no wrapper |
| **Uso típico** | Experimentos batch (10-50 streams) | Testes rápidos (1-3 streams) |

---

## 🎯 RECOMENDAÇÕES

### **PRIORIDADE 1: Adicionar `drift_simulation_experiments` no config.yaml** 🔥

Editar `config.yaml` (após linha 24):

```yaml
experiment_settings:
  run_mode: 'standard'  # ou 'drift_simulation'

  standard_experiments:
    - CovType
    - PokerHand

  # ✅ ADICIONAR ESTA SEÇÃO:
  drift_simulation_experiments:
    - SEA_Abrupt_Simple
    - AGRAWAL_Abrupt_Simple_Mild
    - RBF_Abrupt_Severe
    # (adicionar outros conforme necessário)
```

---

### **PRIORIDADE 2: Testar Ambos os Modos**

```bash
# 1. Testar standard (datasets reais)
python main.py  # run_mode: 'standard'

# 2. Testar drift_simulation (datasets sintéticos com drift)
# (editar config.yaml: run_mode: 'drift_simulation')
python main.py
```

---

### **PRIORIDADE 3: Documentar Uso Adequado de Cada Modo**

**Use `standard` para:**
- Experimentos com datasets reais (Electricity, CovType, PokerHand, etc.)
- Geradores estacionários sem drift (SEA_Stationary, AGRAWAL_Stationary)
- Quando NÃO precisa de métricas de severidade de drift

**Use `drift_simulation` para:**
- Simulações controladas de drift (abrupt, gradual, recurring)
- Experimentos científicos comparando comportamento com drift
- Análise de severidade de drift (mild, severe, etc.)
- Validação de mecanismos de adaptação a drift

---

### **PRIORIDADE 4: Adicionar Penalidades Inter-chunk no Wrapper (Opcional)**

**Limitação atual:** `compare_gbml_vs_river.py` não implementa:
- Detecção de mudança de conceito
- Penalidades inter-chunk (operator_change, feature_change)
- Métricas de severidade de drift

**Se necessário para comparações científicas:**
1. Adicionar lógica de drift tracking em `gbml_evaluator.py`
2. Implementar cálculo de severidade entre chunks
3. Ativar penalidades condicionalmente (se drift detectado)

---

## 📚 ESTRUTURA DE experimental_streams

**Contagem atual:**
- **Datasets Reais:** 5 (Electricity, Shuttle, CovType, PokerHand, IntelLabSensors)
- **Estacionários:** 14 (SEA_Stationary, AGRAWAL_Stationary, etc.)
- **Com Drift:** ~40 streams variados (abrupt, gradual, recurring, noise)

**Total:** ~59 streams definidos em `experimental_streams`

**Organização:**
```yaml
experimental_streams:
  # Reais (standard)
  Electricity: { dataset_type: 'ELECTRICITY' }
  CovType: { dataset_type: 'COVERTYPE' }

  # Estacionários (standard)
  SEA_Stationary: { dataset_type: 'SEA' }
  AGRAWAL_Stationary: { dataset_type: 'AGRAWAL' }

  # Com drift (drift_simulation)
  SEA_Abrupt_Simple:
    dataset_type: 'SEA'
    drift_type: 'abrupt'
    concept_sequence: [...]
```

---

## ✅ CHECKLIST DE VALIDAÇÃO

- [x] Analisado config.yaml atual
- [x] Identificado problema crítico (drift_simulation_experiments faltando)
- [x] Documentado implementação de run_mode em main.py
- [x] Documentado detecção automática em data_handling.py
- [x] Comparado main.py vs compare_gbml_vs_river.py
- [ ] Adicionado drift_simulation_experiments no config.yaml
- [ ] Testado modo standard (datasets reais)
- [ ] Testado modo drift_simulation (datasets sintéticos)
- [ ] Validado alternância entre modos
- [ ] Documentado uso adequado de cada modo

---

## 🎉 CONCLUSÃO

**Status:** 🟡 **PROBLEMA IDENTIFICADO - CORREÇÃO NECESSÁRIA**

**Descobertas:**
1. ✅ Sistema tem 2 modos bem implementados (standard e drift_simulation)
2. ❌ Config.yaml está incompleto (falta `drift_simulation_experiments`)
3. ✅ compare_gbml_vs_river.py funciona independente (detecção automática)
4. ⚠️ main.py FALHA se tentar usar modo drift_simulation

**Impacto:**
- **Baixo para usuário atual:** Compare wrapper funciona normalmente
- **Alto para main.py:** Não pode executar experimentos com drift controlado
- **Fácil de corrigir:** Adicionar seção faltante no config.yaml

**Próxima ação:**
1. Adicionar `drift_simulation_experiments` no config.yaml
2. Testar com `python main.py` em ambos os modos
3. Escolher qual modo usar para experimentos científicos:
   - **Standard:** Para comparação com River em datasets reais
   - **Drift:** Para análise de mecanismos de adaptação

---

**🔧 Correção urgente necessária antes de usar main.py com drift!**
