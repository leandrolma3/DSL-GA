# 🚀 TESTE NO GOOGLE COLAB - SINGLE STREAM (6 CHUNKS)

**Data**: 2025-10-28
**Stream de teste**: `RBF_Abrupt_Severe`
**Objetivo**: Validar configuração de 6 chunks antes dos experimentos massivos

---

## 📋 CHECKLIST PRÉ-TESTE

- [ ] Arquivos sincronizados no Google Drive
- [ ] config_test_single.yaml criado
- [ ] Google Colab conectado
- [ ] Dependências instaladas
- [ ] Teste executado
- [ ] Logs validados

---

## 📁 ARQUIVOS PARA UPLOAD NO GOOGLE DRIVE

### Diretório: `/content/drive/MyDrive/DSL-AG-hybrid/`

**Arquivos principais** (obrigatórios):
```
main.py
ga.py
data_handling_v8.py
analyze_concept_difference.py
plotting.py
hill_climbing_v2.py
config_test_single.yaml         # ← NOVO (criar abaixo)
drift_analysis/
  └── concept_differences.json  # Se existir
```

**Arquivos de suporte**:
```
chunk_transition_analyzer.py
rule_diff_analyzer.py
analyze_standard_drift.py
compare_gbml_vs_river.py
```

---

## 🔧 PASSO 1: CRIAR CONFIG DE TESTE

Criar arquivo `config_test_single.yaml` (baseado em `config_6chunks.yaml`):

```yaml
---
# Configuration for Single Stream Test - 6 Chunks

# --- Experiment Setup ---
experiment_settings:
  run_mode: 'drift_simulation'
  drift_simulation_experiments:
    - RBF_Abrupt_Severe  # ← Apenas este stream para teste
  num_runs: 1   # Apenas 1 run
  base_results_dir: "/content/drive/MyDrive/DSL-AG-hybrid/experiments_test"
  logging_level: "INFO"
  evaluation_period: 6000

# --- Data Handling ---
data_params:
  chunk_size: 6000
  num_chunks: 6  # ← 6 chunks (ajustado)
  max_instances: 36000  # 6000 * 6

# --- Genetic Algorithm Parameters ---
ga_params:
  population_size: 80  # ← Reduzido para 80 (era 120)
  max_generations: 200
  max_generations_recovery: 25
  recovery_generation_multiplier: 1.5
  enable_explicit_drift_adaptation: true
  recovery_mutation_override_rate: 0.5
  recovery_mutation_override_generations: 10
  recovery_initialization_strategy: "full_random"
  recovery_random_individual_ratio: 0.6
  recovery_max_depth_multiplier: 1.5
  recovery_max_rules_multiplier: 1.5
  elitism_rate: 0.1
  intelligent_mutation_rate: 0.8
  initial_tournament_size: 2
  final_tournament_size: 5
  max_rules_per_class: 15
  initial_max_depth: 10
  stagnation_threshold: 10
  early_stopping_patience: 20

  # Hill Climbing
  hc_enable_adaptive: false
  hc_gmean_threshold: 0.90
  hc_hierarchical_enabled: true

  # Seeding Configuration
  enable_dt_seeding_on_init: true
  dt_seeding_ratio_on_init: 0.8
  dt_seeding_depths_on_init: [4, 7, 10, 13]
  dt_seeding_sample_size_on_init: 2000
  dt_seeding_rules_to_replace_per_class: 4
  dt_rule_injection_ratio: 0.5

  # Seeding Adaptativo
  enable_adaptive_seeding: true
  adaptive_seeding_strategy: 'dt_probe'
  adaptive_complexity_simple_threshold: 0.90
  adaptive_complexity_medium_threshold: 0.75

# --- Fitness Parameters ---
fitness_params:
  # ... (copiar do config_6chunks.yaml)
  alpha: 0.55
  beta: 0.45
  performance_weight: 0.85
  complexity_weight: 0.15
  penalty_per_rule_above_limit: 0.003
  penalty_per_depth_unit_above_limit: 0.002
  penalty_per_node_above_limit: 0.001
  change_penalty_factor: 0.01
  drift_penalty_reduction_factor: 0.5
  drift_penalty_reduction_threshold: 0.10

# --- Drift Analysis Configuration ---
drift_analysis:
  severity_samples: 10000
  heatmap_save_directory: "results"
  datasets:
    RBF:  # ← Apenas RBF para este teste
      n_features: 10
      n_classes: 2
      class: 'binary'
      feature_bounds: [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
      concepts:
        c1: {seed_model: 42}
        c2_severe: {seed_model: 84}
        c3_moderate: {seed_model: 126}
      pairs_to_compare:
        - ['c1', 'c2_severe']
        - ['c1', 'c3_moderate']
        - ['c2_severe', 'c3_moderate']

# --- Experimental Streams ---
experimental_streams:
  RBF_Abrupt_Severe:  # ← Apenas este stream
    dataset_type: 'RBF'
    drift_type: 'abrupt'
    gradual_drift_width_chunks: 0
    concept_sequence:
      - { concept_id: 'c1', duration_chunks: 3 }
      - { concept_id: 'c2_severe', duration_chunks: 3 }
```

---

## 🌐 PASSO 2: NOTEBOOK DO GOOGLE COLAB

### Célula 1: Montar Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/DSL-AG-hybrid')
print(f"Diretório atual: {os.getcwd()}")
```

### Célula 2: Instalar Dependências

```python
!pip install river
!pip install scikit-learn
!pip install pandas numpy matplotlib seaborn
!pip install PyYAML
!pip install python-Levenshtein
!pip install tqdm

print("✅ Dependências instaladas!")
```

### Célula 3: Validar Arquivos

```python
import os
import yaml

# Verificar arquivos principais
required_files = [
    'main.py',
    'ga.py',
    'data_handling_v8.py',
    'config_test_single.yaml',
    'hill_climbing_v2.py',
    'plotting.py'
]

print("🔍 Verificando arquivos...")
for file in required_files:
    exists = "✅" if os.path.exists(file) else "❌"
    print(f"{exists} {file}")

# Validar config
print("\n📋 Validando config_test_single.yaml...")
with open('config_test_single.yaml') as f:
    config = yaml.safe_load(f)

print(f"  - Stream: {config['experiment_settings']['drift_simulation_experiments']}")
print(f"  - num_chunks: {config['data_params']['num_chunks']}")
print(f"  - population_size: {config['ga_params']['population_size']}")
print(f"  - num_runs: {config['experiment_settings']['num_runs']}")

# Validar concept_sequence do stream
stream = config['experimental_streams']['RBF_Abrupt_Severe']
seq = stream['concept_sequence']
total = sum(s['duration_chunks'] for s in seq)
print(f"  - concept_sequence: {' → '.join([f\"{s['concept_id']}({s['duration_chunks']})\" for s in seq])}")
print(f"  - Total chunks: {total}")
assert total == 6, f"❌ ERRO: Total de chunks é {total}, esperado 6!"
print("✅ Config validado!")
```

### Célula 4: Executar Experimento

```python
import subprocess
import time
from datetime import datetime

# Nome do log
log_filename = f"experimento_teste_6chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
print(f"📝 Log será salvo em: {log_filename}")
print("🚀 Iniciando experimento...")
print("⏱️  Tempo estimado: ~8 horas")
print("")

start_time = time.time()

# Executar main.py em background
process = subprocess.Popen(
    ['python', 'main.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

# Salvar output em arquivo e exibir em tempo real
with open(log_filename, 'w') as log_file:
    for line in process.stdout:
        print(line, end='')  # Exibe no Colab
        log_file.write(line)  # Salva no arquivo
        log_file.flush()  # Força escrita imediata

process.wait()
elapsed = time.time() - start_time

print("")
print("="*70)
print(f"✅ Experimento concluído!")
print(f"⏱️  Tempo total: {elapsed/3600:.2f} horas ({elapsed/60:.1f} minutos)")
print(f"📝 Log salvo em: {log_filename}")
print("="*70)
```

### Célula 5: Validação Rápida dos Resultados

```python
import re

print("🔍 VALIDAÇÃO RÁPIDA DO LOG")
print("="*70)

# Ler log
with open(log_filename, 'r') as f:
    log_content = f.read()

# 1. Contar chunks processados
chunks_started = re.findall(r'Starting GA for chunk (\d+)', log_content)
print(f"\n✅ Chunks processados: {len(chunks_started)}")
if len(chunks_started) == 6:
    print(f"   → {', '.join(chunks_started)} ✅")
else:
    print(f"   ⚠️  ESPERADO: 6 chunks, ENCONTRADO: {len(chunks_started)}")

# 2. Verificar seeding 85%
seeding_85 = re.findall(r'SEVERE DRIFT DETECTED: Seeding INTENSIVO ativado', log_content)
print(f"\n✅ Seeding 85% ativado: {len(seeding_85)} vez(es)")
if seeding_85:
    # Encontrar em qual chunk
    severe_chunks = re.findall(r'SEVERE DRIFT detected.*for chunk (\d+)', log_content)
    if severe_chunks:
        print(f"   → Chunk(s) com drift SEVERE: {', '.join(severe_chunks)}")
else:
    print("   ⚠️  Seeding 85% NÃO foi ativado (drift pode não ter sido SEVERE)")

# 3. Verificar severidade do drift
drift_severities = re.findall(r'severity: (\d+\.\d+)%', log_content)
if drift_severities:
    severities_float = [float(s) for s in drift_severities]
    print(f"\n✅ Severidades detectadas:")
    print(f"   → Média: {sum(severities_float)/len(severities_float):.2f}%")
    print(f"   → Máxima: {max(severities_float):.2f}%")
    print(f"   → Mínima: {min(severities_float):.2f}%")

# 4. Performance final
final_metrics = re.findall(r'Avg G-mean.*: (\d+\.\d+)', log_content)
if final_metrics:
    gmean = float(final_metrics[-1])
    print(f"\n✅ G-mean final: {gmean:.2f}%")
    if gmean >= 85.0:
        print(f"   ✅ META ATINGIDA (≥ 85%)!")
    else:
        print(f"   ⚠️  Abaixo da meta (esperado: ≥ 85%)")

# 5. Taxa de aprovação HC
hc_approvals = re.findall(r'taxa de aprovação.*: (\d+\.\d+)%', log_content)
if hc_approvals:
    hc_rates = [float(r) for r in hc_approvals]
    avg_hc = sum(hc_rates) / len(hc_rates)
    print(f"\n✅ Hill Climbing - Taxa média de aprovação: {avg_hc:.2f}%")
    if avg_hc >= 25.0:
        print(f"   ✅ META ATINGIDA (≥ 25%)!")
    else:
        print(f"   ⚠️  Abaixo da meta (esperado: ≥ 25%)")

# 6. Verificar erros
errors = re.findall(r'ERROR.*', log_content)
if errors:
    print(f"\n⚠️  ERROS encontrados: {len(errors)}")
    print("   Primeiros 3 erros:")
    for err in errors[:3]:
        print(f"   - {err[:100]}")
else:
    print(f"\n✅ Nenhum erro encontrado!")

print("\n" + "="*70)
print("📁 Arquivos gerados (verificar diretório de resultados):")
print("   - RulesHistory_*.txt")
print("   - chunk_metrics.json")
print("   - periodic_accuracy.json")
print("   - ga_history_per_chunk.json")
print("   - plots/*.png")
print("="*70)
```

### Célula 6: Análise Detalhada (Opcional)

```python
import json
import os

results_dir = config['experiment_settings']['base_results_dir']
print(f"📂 Diretório de resultados: {results_dir}")

# Listar subdiretórios
if os.path.exists(results_dir):
    runs = os.listdir(results_dir)
    print(f"\n✅ Runs encontrados: {len(runs)}")
    for run in runs:
        run_path = os.path.join(results_dir, run)
        if os.path.isdir(run_path):
            print(f"\n📊 Run: {run}")
            files = os.listdir(run_path)
            for f in files:
                fpath = os.path.join(run_path, f)
                if f.endswith('.json'):
                    size = os.path.getsize(fpath)
                    print(f"   - {f} ({size} bytes)")
                elif f.endswith('.txt'):
                    lines = len(open(fpath).readlines())
                    print(f"   - {f} ({lines} linhas)")

    # Ler chunk_metrics.json se existir
    run1_path = os.path.join(results_dir, runs[0], 'run_1')
    metrics_path = os.path.join(run1_path, 'chunk_metrics.json')

    if os.path.exists(metrics_path):
        print(f"\n📈 Métricas por chunk:")
        with open(metrics_path) as f:
            metrics = json.load(f)

        print(f"\n{'Chunk':<8} {'Train Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
        print("-" * 50)
        for m in metrics:
            chunk = m.get('chunk', '?')
            train_acc = m.get('train_accuracy', 0) * 100
            test_acc = m.get('test_accuracy', 0) * 100
            test_f1 = m.get('test_f1', 0) * 100
            print(f"{chunk:<8} {train_acc:<12.2f} {test_acc:<12.2f} {test_f1:<12.2f}")
else:
    print(f"⚠️  Diretório de resultados não encontrado: {results_dir}")
```

---

## ✅ VALIDAÇÕES ESPERADAS

### No Log:

1. **6 chunks processados**:
   ```
   Starting GA for chunk 0
   Starting GA for chunk 1
   Starting GA for chunk 2
   Starting GA for chunk 3
   Starting GA for chunk 4
   Starting GA for chunk 5
   ```

2. **Seeding 85% ativado no chunk 3**:
   ```
   SEVERE DRIFT detected (severity: 45.00%) for chunk 3
   -> SEVERE DRIFT DETECTED: Seeding INTENSIVO ativado (85% seeding, 90% injection)
   ```

3. **G-mean final ≥ 85%**:
   ```
   Avg G-mean: 87.35%  ✅
   ```

4. **HC taxa ≥ 25%**:
   ```
   Hill Climbing - taxa de aprovação média: 28.5%  ✅
   ```

### Nos Arquivos Gerados:

1. **RulesHistory_RBF_Abrupt_Severe_Run1.txt**:
   - 6 seções `=== Chunk X ===` (X = 0 a 5)

2. **chunk_metrics.json**:
   - 6 entradas (uma por chunk)
   - Cada entrada com: `chunk`, `train_accuracy`, `test_accuracy`, `test_f1`

3. **Diretório `plots/`**:
   - Heatmaps de feature drift
   - Performance plots
   - GA evolution plots

---

## ⏱️ TEMPO ESTIMADO

- **População 80** indivíduos
- **6 chunks** × ~200 gerações (com early stopping ~30-50 gens)
- **Estimativa**: 6-8 horas

---

## 🚨 TROUBLESHOOTING

### Problema 1: "ModuleNotFoundError: No module named 'river'"
**Solução**: Reinstalar dependências (Célula 2)

### Problema 2: "FileNotFoundError: config_test_single.yaml"
**Solução**: Verificar se arquivo foi criado e está no diretório correto

### Problema 3: Colab desconecta durante execução
**Solução**:
- Manter aba aberta
- Usar Colab Pro (timeouts maiores)
- Adicionar código para manter vivo:
  ```python
  from google.colab import output
  output.enable_custom_widget_manager()
  ```

### Problema 4: Seeding 85% não ativou
**Verificar**:
- Severidade do drift RBF_Abrupt_Severe no log
- Se < 25%, drift não é SEVERE (ajustar conceitos ou escolher outro stream)

---

## 📞 PRÓXIMOS PASSOS APÓS TESTE

### Se teste for bem-sucedido (✅):
1. ✅ Validar G-mean ≥ 85%
2. ✅ Validar HC taxa ≥ 25%
3. ✅ Validar 6 chunks processados
4. ⏭️ Executar batch de 3-5 streams diferentes
5. ⏭️ Executar 41 streams completos

### Se teste falhar (❌):
1. Analisar logs de erro
2. Verificar concept_sequence
3. Verificar severidade do drift
4. Ajustar e re-executar

---

**Criado por**: Claude Code
**Data**: 2025-10-28
**Status**: ✅ PRONTO PARA EXECUÇÃO NO COLAB

**BOA SORTE COM O TESTE!** 🚀
