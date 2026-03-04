# 🔬 INFRAESTRUTURA DE DRIFT SIMULATION - ANÁLISE COMPLETA

**Data**: 2025-10-28
**Objetivo**: Documentar a infraestrutura existente de geração e análise de drift simulations antes de ajustar configurações para experimentos massivos com 6 chunks.

---

## 📋 RESUMO EXECUTIVO

A infraestrutura de drift simulations é **robusta e bem integrada**. Os principais arquivos analisados demonstram:

1. ✅ **Label functions sincronizadas** entre `data_handling_v8.py` e `analyze_concept_difference.py`
2. ✅ **Visualizações flexíveis** que suportam número variável de chunks
3. ✅ **Post-processamento estruturado** para análise de regras e transições
4. ✅ **Detecção de drift** baseada em performance e features

**Conclusão**: A infraestrutura **suporta ajuste para 6 chunks** sem quebras, desde que:
- Proporções de `concept_sequence` sejam mantidas
- `duration_chunks` somem exatamente 6
- Validação seja feita em subset antes da execução massiva

---

## 📁 ARQUIVOS ANALISADOS

### 1. `data_handling_v8.py` (398 linhas)

**Função**: Geração de chunks com drift simulation

#### Componentes Chave:

**Label Functions** (linhas 23-102):
```python
CONCEPT_FUNCTIONS = {
    "SEA": get_sea_label,           # Threshold-based (variants: 7.0, 8.0, 9.0, 9.5)
    "RBF": get_rbf_label,           # Centroid-based (seed-deterministic)
    "AGRAWAL": get_agrawal_label,   # 10 complex functions
    "STAGGER": get_stagger_label,   # 3 classification functions
    "SINE": get_sine_label,         # Type: sum/prod, threshold
    "HYPERPLANE": get_hyperplane_label,
    "RANDOMTREE": get_randomtree_label,
    "LED": get_led_label,
    "WAVEFORM": get_waveform_label,
    "FRIEDMAN": get_friedman_label
}
```

**Geração de Chunks com Drift** (linhas 194-242):
```python
def _generate_chunks_with_drift_simulation(
    base_stream_generator, chunk_size, max_chunks,
    stream_name, config, feature_names
):
    # Para cada chunk:
    # 1. Determina concept_id_a, concept_id_b, p_mix_b
    # 2. Obtém params_a e params_b do config
    # 3. Recalcula labels usando concept_func(instance_vec, params)
    # 4. Suporta drift gradual com mixture_prob_b
```

**get_concept_for_chunk** (linhas 144-168):
```python
def get_concept_for_chunk(stream_definition, chunk_index):
    concept_sequence = stream_definition.get('concept_sequence', [])
    drift_type = stream_definition.get('drift_type', 'abrupt')
    gradual_width = stream_definition.get('gradual_drift_width_chunks', 0)

    # Itera sobre concept_sequence e retorna:
    # - concept_id_a, concept_id_b, mixture_prob_b
    # - Para abrupt: concept_id_b=None, mixture_prob_b=0.0
    # - Para gradual: mistura proporcional durante janela
```

**Feature Drift Detection** (linhas 246-295):
```python
def detect_concept_drift(chunks, name_for_plot, run_number, results_dir):
    # Gera heatmap de drift magnitude entre chunks consecutivos
    # Usa mean drift + variance drift
    # Salva em: results_dir/FeatureDrift_{name}_Run{run}.png
```

#### ✅ Compatibilidade com 6 Chunks:
- **SIM**: A função `get_concept_for_chunk` funciona com qualquer `duration_chunks`
- **SIM**: Label recalculation não depende do número total de chunks
- **SIM**: Feature drift detection funciona com `num_chunks >= 2`

---

### 2. `analyze_concept_difference.py` (349 linhas)

**Função**: Análise de severidade de drift entre pares de conceitos

#### Componentes Chave:

**Label Functions Sincronizadas** (linhas 14-212):
```python
# IMPORTANTE: Funções IDÊNTICAS a data_handling_v8.py
def get_sea_label(instance, concept_params):
    variant_map = {0: 8.0, 1: 9.0, 2: 7.0, 3: 9.5}
    variant = concept_params.get('variant', 0)
    threshold = variant_map.get(variant, 8.0)
    return 0 if instance[0] + instance[1] <= threshold else 1

def get_rbf_label(instance, concept_params):
    seed = concept_params.get('seed_model', 42)
    rng = np.random.RandomState(seed)
    centroids = rng.rand(4, len(instance))
    # ... lógica determinística baseada em seed
```

**Cálculo de Diferença** (linhas 230-243):
```python
def calculate_difference(concept_func, params_a, params_b, n_samples, feature_bounds, n_features):
    # Gera n_samples no espaço de features
    # Compara label_a vs label_b
    # Retorna: (total_diff / n_samples) * 100.0
```

**Heatmap de Severidade** (linhas 269-287):
```python
def plot_difference_heatmap(dataset_name, dataset_results, save_dir):
    # Cria matriz de diferenças entre todos os pares
    # Salva em: save_dir/ConceptDifference_Heatmap_{dataset_name}.png
```

#### ✅ Compatibilidade com 6 Chunks:
- **SIM**: Análise de severidade é **independente de chunks**
- **SIM**: Compara conceitos individualmente (não depende de sequência temporal)
- **IMPORTANTE**: Sincronização perfeita com `data_handling_v8.py` garante consistência

---

### 3. `plotting.py` (299+ linhas)

**Função**: Geração de visualizações de performance, regras, atributos

#### Componentes Chave:

**Plot de Performance** (linhas 105-129):
```python
def plot_performance_metrics(performance_metrics, dataset_name, run_number=1, ax=None):
    chunks = [m.get('chunk', idx) for idx, m in enumerate(performance_metrics)]
    train_accuracies = [m.get('train_accuracy', np.nan) for m in performance_metrics]
    test_accuracies = [m.get('test_accuracy', np.nan) for m in performance_metrics]

    ax.plot(chunks, train_accuracies, marker='o', label='Train Accuracy')
    ax.plot(chunks, test_accuracies, marker='s', label='Test Accuracy')
    ax.set_xticks(chunks)  # ✅ Funciona com qualquer número de chunks
```

**Plot de Complexidade** (linhas 132-164):
```python
def plot_rules_conditionals(rules_conditionals, dataset_name, run_number=1, ax=None):
    chunks = [rc.get('chunk', idx) for idx, rc in enumerate(rules_conditionals)]
    total_rules = [rc.get('total_rules', np.nan) for rc in rules_conditionals]
    # ... plots com ax.set_xticks(chunks)
```

**Plot de Evolução GA** (linhas 168-251):
```python
def plot_ga_evolution(history, chunk_index, dataset_name, run_number=1, ax=None):
    # Plota fitness e accuracy por geração para UM chunk específico
    # Cria 2 subplots: fitness evolution + accuracy evolution
```

**Mosaic Plots** (linhas 254-299):
```python
def create_mosaic_plots(dataset_name, runs_results, attributes, save_path=None):
    num_runs = len(runs_results)
    fig, axes = plt.subplots(nrows=3, ncols=num_runs, figsize=(num_runs * 5.5, 15))

    # Linha 1: Attribute usage heatmap
    # Linha 2: Performance metrics
    # Linha 3: Rule complexity
```

#### ✅ Compatibilidade com 6 Chunks:
- **SIM**: Todas as funções usam `enumerate()` ou `chunks` dinâmicos
- **SIM**: `ax.set_xticks(chunks)` adapta-se automaticamente
- **SIM**: Mosaics funcionam para qualquer `num_runs` e chunks por run

---

### 4. `chunk_transition_analyzer.py` (220+ linhas)

**Função**: Análise de mudanças de regras entre chunks consecutivos

#### Componentes Chave:

**Parsing de Rules History** (linhas 30-220):
```python
def parse_rules_history_to_asts(file_path):
    # Parseia arquivo RulesHistory_*.txt
    # Extrai para cada chunk:
    #   - fitness, default_class
    #   - rules_asts: {class_label: [AST1, AST2, ...]}
    #   - rules_raw_strings: {class_label: [str1, str2, ...]}
    # Retorna: {chunk_idx: {metadata + rules}}
```

**Análise de Transição** (linhas 222-299):
```python
def analyze_chunk_transition(chunk_data_i, chunk_data_j,
                             levenshtein_similarity_threshold=0.7,
                             sm_threshold_for_modified=0.9):
    # Compara regras entre chunk i e i+1
    # Classifica como: unchanged, modified, new, deleted
    # Calcula:
    #   - MI (Modification Index)
    #   - SMM (Structural Modification Magnitude)
    #   - STT (Structural Transition Total)
```

#### ✅ Compatibilidade com 6 Chunks:
- **SIM**: Parsing trabalha chunk por chunk (independente de total)
- **SIM**: Análise de transição é **pairwise** (i → i+1)
- **SIM**: Funciona para qualquer sequência de chunks

---

### 5. `rule_diff_analyzer.py` (299+ linhas)

**Função**: Análise detalhada de diferenças de regras e geração de relatórios

#### Componentes Chave:

**Parsing de Rules History** (linhas 53-161):
```python
def parse_rules_history(file_path):
    # Similar a chunk_transition_analyzer.py
    # Extrai: fitness, default_class, train_acc, test_acc_next, test_f1_next
    # Parseia regras por classe
    # Retorna: {chunk_idx: {metadata + rules}}
```

**Comparação de Regras** (linhas 165-202):
```python
def compare_chunk_rules(chunk_data_i, chunk_data_i_plus_1, similarity_threshold=0.35):
    # Usa Levenshtein distance para detectar modificações
    # Classifica: unchanged, modified, new, deleted
    # Retorna: diff_results = {category: {class_label: [rules]}}
```

**Evolution Matrix** (linhas 246-257):
```python
def generate_evolution_matrix_table(all_counts_per_transition):
    # Cria tabela com colunas: Transition (0→1, 1→2, ...)
    # Linhas: Unchanged, Modified, New, Deleted, Remain Rules
    # Retorna: DataFrame pandas
```

**Heatmap de Evolução** (linhas 259-274):
```python
def plot_evolution_matrix(df_matrix, source_file_name, save_path=None):
    # Gera heatmap de evolução de regras
    # Eixo X: Transições (0→1, 1→2, ...)
    # Eixo Y: Categorias de mudança
```

#### ✅ Compatibilidade com 6 Chunks:
- **SIM**: Evolution matrix adapta-se a qualquer número de transições
- **SIM**: Heatmap redimensiona-se dinamicamente: `figsize=(max(8, n_transitions * 0.8), ...)`
- **SIM**: Relatórios funcionam chunk por chunk

---

### 6. `analyze_standard_drift.py` (299+ linhas)

**Função**: Detecção de drift baseada em queda de performance

#### Componentes Chave:

**Detecção de Performance Drift** (linhas 82-142):
```python
def detect_performance_drifts(chunk_metrics, threshold=0.10):
    # Ordena métricas por chunk index
    # Compara test_accuracy entre chunks consecutivos
    # Identifica quedas > threshold (ex: 10%)
    # Retorna: lista de chunk indices com drift
```

**Plot com Anotações** (linhas 146-240):
```python
def plot_periodic_accuracy_with_detected_drifts(
    periodic_accuracies, chunk_metrics, detected_drift_chunks,
    chunk_size, experiment_id, run_number, save_path
):
    # Plota periodic test accuracy
    # Marca chunk boundaries (linhas verticais cinza)
    # Marca detected drifts (linhas verticais vermelhas)
    # Adiciona texto anotando cada drift
```

#### ✅ Compatibilidade com 6 Chunks:
- **SIM**: Detecção funciona para `len(chunk_metrics) >= 2`
- **SIM**: Plot adapta-se ao número de chunks: `num_chunk_transitions = len(chunk_metrics)`
- **SIM**: Anotações posicionadas dinamicamente

---

## 🎯 ANÁLISE DE DEPENDÊNCIAS

### Dependências Críticas:

1. **`concept_sequence` em config.yaml**:
   - Deve somar exatamente `num_chunks` (6 no experimento massivo)
   - Exemplo válido para 6 chunks:
     ```yaml
     concept_sequence:
       - concept_id: 'c1'
         duration_chunks: 3
       - concept_id: 'c2'
         duration_chunks: 3
     ```

2. **Label Functions Sincronizadas**:
   - `data_handling_v8.py` (geração) == `analyze_concept_difference.py` (análise)
   - **IMPORTANTE**: Alterações em uma devem ser replicadas na outra

3. **Parsing de RulesHistory**:
   - `chunk_transition_analyzer.py` e `rule_diff_analyzer.py` esperam formato:
     ```
     === Chunk X ===
     ---
     Fitness: 0.XXXXX
     Default Class: X
     ...
     Class X
       IF <condition> THEN Class X
     ```
   - Formato gerado por `main.py` (não será alterado)

4. **JSON Outputs**:
   - `chunk_metrics.json`: Lista de dicts com `chunk`, `train_accuracy`, `test_accuracy`
   - `periodic_accuracy.json`: Lista de tuplas `(instance_count, accuracy)`
   - `ga_history_per_chunk.json`: Dict `{chunk_idx: {best_fitness: [...], ...}}`

---

## 🔧 AJUSTES NECESSÁRIOS PARA 6 CHUNKS

### ✅ O QUE FUNCIONA SEM MODIFICAÇÕES:

1. ✅ Todos os scripts de visualização (`plotting.py`)
2. ✅ Análise de transição (`chunk_transition_analyzer.py`, `rule_diff_analyzer.py`)
3. ✅ Detecção de drift (`analyze_standard_drift.py`)
4. ✅ Geração de chunks (`data_handling_v8.py`)
5. ✅ Análise de severidade (`analyze_concept_difference.py`)

### ⚠️ O QUE REQUER AJUSTE:

1. **config.yaml** - 41 drift simulations:
   - Ajustar `concept_sequence` para somar 6 chunks
   - Manter proporções quando possível
   - Validar que `gradual_drift_width_chunks <= 6`

2. **Parâmetros globais** (linhas 36-38 do config.yaml):
   ```yaml
   num_chunks: 6              # Era: 8-13 (varia por stream)
   max_instances: 36000       # 6 chunks × 6000 instances
   population_size: 80        # Era: 120
   ```

---

## 📊 ESTRATÉGIA DE RESCALING

### Para 2 Conceitos (Abrupt):
```yaml
# Original (8 chunks):
concept_sequence:
  - {concept_id: 'c1', duration_chunks: 4}
  - {concept_id: 'c2', duration_chunks: 4}

# Ajustado (6 chunks):
concept_sequence:
  - {concept_id: 'c1', duration_chunks: 3}
  - {concept_id: 'c2', duration_chunks: 3}
```

### Para 3 Conceitos (Recurring):
```yaml
# Original (9 chunks):
concept_sequence:
  - {concept_id: 'c1', duration_chunks: 3}
  - {concept_id: 'c2', duration_chunks: 3}
  - {concept_id: 'c1', duration_chunks: 3}

# Ajustado (6 chunks):
concept_sequence:
  - {concept_id: 'c1', duration_chunks: 2}
  - {concept_id: 'c2', duration_chunks: 2}
  - {concept_id: 'c1', duration_chunks: 2}
```

### Para Drift Gradual:
```yaml
# Original (10 chunks, gradual_width=2):
concept_sequence:
  - {concept_id: 'c1', duration_chunks: 5}
  - {concept_id: 'c2', duration_chunks: 5}
gradual_drift_width_chunks: 2

# Ajustado (6 chunks):
concept_sequence:
  - {concept_id: 'c1', duration_chunks: 3}
  - {concept_id: 'c2', duration_chunks: 3}
gradual_drift_width_chunks: 1  # Reduzido proporcionalmente
```

### Para Blips (Conceito Temporário):
```yaml
# Original (8 chunks):
concept_sequence:
  - {concept_id: 'c1', duration_chunks: 3}
  - {concept_id: 'c2_blip', duration_chunks: 2}
  - {concept_id: 'c1', duration_chunks: 3}

# Ajustado (6 chunks):
concept_sequence:
  - {concept_id: 'c1', duration_chunks: 2}
  - {concept_id: 'c2_blip', duration_chunks: 2}
  - {concept_id: 'c1', duration_chunks: 2}
```

---

## 🧪 VALIDAÇÕES PRÉ-DEPLOYMENT

### 1. Validar Label Functions:
```bash
python analyze_concept_difference.py
# Verificar: Heatmaps gerados corretamente
# Verificar: Severidades compatíveis com histórico
```

### 2. Testar Geração de Chunks (1 stream):
```python
from data_handling_v8 import generate_dataset_chunks

chunks = generate_dataset_chunks(
    stream_or_dataset_name='SEA_Abrupt_Mild',
    chunk_size=6000,
    num_chunks=6,
    config_path='config_6chunks.yaml'
)

print(f"Gerados {len(chunks)} chunks")
for i, (X, y) in enumerate(chunks):
    print(f"Chunk {i}: {len(X)} instances, {np.bincount(y).tolist()} labels")
```

### 3. Validar Visualizações (1 run completo):
```bash
python main.py
# Verificar arquivos gerados:
# - RulesHistory_*.txt (6 chunks)
# - chunk_metrics.json (6 entradas)
# - periodic_accuracy.json
# - plots/*.png (heatmaps, performance, complexity)
```

### 4. Validar Post-processamento:
```bash
python rule_diff_analyzer.py RulesHistory_*.txt
# Verificar: Evolution matrix 5 transições (0→1 até 5→6)
# Verificar: Relatórios gerados corretamente

python chunk_transition_analyzer.py RulesHistory_*.txt
# Verificar: Análise de severidade de modificações
```

---

## 🚨 PONTOS DE ATENÇÃO

### 1. **Gradual Drift Width**:
- **Problema**: Alguns streams têm `gradual_drift_width_chunks: 3` com 13 chunks
- **Solução**: Ajustar para 1 chunk com 6 chunks totais (proporção: 3/13 ≈ 1/6)
- **Validação**: `gradual_drift_width_chunks < duration_chunks` do primeiro conceito após drift

### 2. **Noise Injection**:
- **Problema**: Streams de noise têm parâmetros específicos (`noise_features`, `noise_percentage`)
- **Solução**: Manter parâmetros, ajustar apenas `duration_chunks`
- **Validação**: Labels ainda refletem noise esperado

### 3. **Blips Curtos**:
- **Problema**: Blips de 1 chunk podem ser imperceptíveis com 6000 instances
- **Solução**: Manter blips de 2 chunks mínimo (33% da stream)
- **Validação**: Drift detection captura o blip

### 4. **Seeding Adaptativo**:
- **Problema**: Seeding 85% está configurado para chunk 5 em main.py:526-530
- **Solução**: Ajustar para chunk 3 ou 4 com 6 chunks totais
- **Código**:
  ```python
  # main.py linha 526
  if drift_severity == 'SEVERE' and chunk_index >= 3:  # Era: chunk_index >= 5
      seeding_percentage = 0.85
  ```

---

## 📈 IMPACTO ESPERADO

### Performance:
- ⏱️ **Tempo por stream**: 13h → 8h (~38% redução)
- 🧬 **População**: 120 → 80 (~33% redução de indivíduos)
- 🎯 **Qualidade**: Espera-se manter ≥ 85% G-mean (seeding validado na Fase 2)

### Cobertura de Drift:
- ✅ **Abrupt**: Mantém características (transição clara)
- ✅ **Gradual**: Ajuste proporcional de janela
- ✅ **Recurring**: Ciclos preservados (2-2-2)
- ✅ **Blips**: Eventos curtos mantidos (2 chunks mínimo)
- ✅ **Noise**: Injeção preservada

---

## 🎯 PRÓXIMOS PASSOS

1. ✅ **Infraestrutura analisada**
2. ⏭️ **Criar script de ajuste**: `adjust_config_for_mass_experiments.py`
3. ⏭️ **Validar em 3 streams**: Um de cada categoria (Abrupt, Gradual, Recurring)
4. ⏭️ **Criar pipeline de execução**: `run_mass_experiments.py`
5. ⏭️ **Criar pipeline River**: `run_river_baselines.py`
6. ⏭️ **Executar experimentos**: 41 streams × (GBML + River)

---

## 📚 REFERÊNCIAS

| Arquivo | Linhas | Função Principal |
|---------|--------|------------------|
| `data_handling_v8.py` | 398 | Geração de chunks com drift simulation |
| `analyze_concept_difference.py` | 349 | Análise de severidade entre conceitos |
| `plotting.py` | 299+ | Visualizações (performance, regras, atributos) |
| `chunk_transition_analyzer.py` | 220+ | Análise de mudanças de regras (AST-based) |
| `rule_diff_analyzer.py` | 299+ | Relatórios e evolution matrix |
| `analyze_standard_drift.py` | 299+ | Detecção de drift por performance |

---

**Status**: ✅ **INFRAESTRUTURA COMPREENDIDA**
**Conclusão**: Seguro prosseguir com criação de scripts de ajuste para 6 chunks.

---

**Documentado por**: Claude Code
**Data**: 2025-10-28
**Próximo**: Criar `adjust_config_for_mass_experiments.py`
