# 🔧 AJUSTES PARA RE-TESTE 6 CHUNKS

**Data**: 2025-10-29
**Objetivo**: Corrigir problema de deteccao proativa de drift
**Stream**: RBF_Abrupt_Severe (6 chunks)

---

## 📋 RESUMO DOS AJUSTES

### Ajustes Realizados:

1. ✅ **Fix em main.py linha 1474**: Usar `config_test_single.yaml` ao inves de `config.yaml`
2. ✅ **Fix em main.py linhas 344-346**: Resolver caminho absoluto de `concept_differences.json`

### Total de Mudancas: **2 ajustes em main.py**

---

## 🔍 DETALHAMENTO DOS AJUSTES

### Ajuste 1: Caminho do Config File

**Arquivo**: `main.py`
**Linha**: 1474

**ANTES**:
```python
config_file_path = os.path.join(script_dir, "config.yaml")
```

**DEPOIS**:
```python
config_file_path = os.path.join(script_dir, "config_test_single.yaml")
```

**Razao**:
- Garantir que o teste use a configuracao correta (6 chunks, 1 stream)
- Evitar necessidade de renomear arquivos manualmente

---

### Ajuste 2: Caminho Absoluto de concept_differences.json

**Arquivo**: `main.py`
**Linhas**: 344-346 (inseridas 2 novas linhas)

**ANTES**:
```python
concept_diff_data = None
diff_file_path = config.get('drift_analysis', {}).get('heatmap_save_directory', 'results/concept_heatmaps')
diff_file_path = os.path.join(diff_file_path, "concept_differences.json")
if os.path.exists(diff_file_path):
```

**DEPOIS**:
```python
concept_diff_data = None
diff_file_path = config.get('drift_analysis', {}).get('heatmap_save_directory', 'results/concept_heatmaps')
diff_file_path = os.path.join(diff_file_path, "concept_differences.json")
# Make path absolute relative to script directory to handle execution from different locations
if not os.path.isabs(diff_file_path):
    diff_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), diff_file_path)
if os.path.exists(diff_file_path):
```

**Razao**:
- No Google Colab, caminhos relativos nao sao resolvidos corretamente
- `config_test_single.yaml` tem: `heatmap_save_directory: test_real_results_heatmaps/concept_heatmapsS`
- O arquivo existe em: `test_real_results_heatmaps/concept_heatmapsS/concept_differences.json`
- Com ajuste, o caminho sera resolvido como: `/content/drive/MyDrive/DSL-AG-hybrid/test_real_results_heatmaps/concept_heatmapsS/concept_differences.json`

**Impacto**:
- ✅ `concept_differences.json` sera encontrado
- ✅ Severidade do drift sera lida: c1_vs_c2_severe = 60.5% (SEVERE)
- ✅ Seeding 85% sera ativado no chunk 3 (quando transitar de c1 para c2_severe)

---

## ✅ VALIDACAO DOS AJUSTES

### Checklist Pre-Execucao:

- [x] `main.py` ajustado (2 mudancas)
- [x] `config_test_single.yaml` existe e esta correto
- [x] `concept_differences.json` existe em `test_real_results_heatmaps/concept_heatmapsS/`
- [x] Conteudo de `concept_differences.json` validado:
  ```json
  {
    "RBF": {
      "c1_vs_c2_severe": 60.45,
      "c1_vs_c3_moderate": 65.295,
      "c2_severe_vs_c3_moderate": 42.995
    }
  }
  ```

### Teste Local (Opcional):

Para validar que o caminho esta correto localmente:

```bash
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"
python -c "
import os
import yaml
import json

script_dir = os.path.dirname(os.path.abspath('main.py'))
config_path = os.path.join(script_dir, 'config_test_single.yaml')

with open(config_path) as f:
    config = yaml.safe_load(f)

diff_file_path = config.get('drift_analysis', {}).get('heatmap_save_directory', 'results/concept_heatmaps')
diff_file_path = os.path.join(diff_file_path, 'concept_differences.json')

if not os.path.isabs(diff_file_path):
    diff_file_path = os.path.join(script_dir, diff_file_path)

print(f'Caminho resolvido: {diff_file_path}')
print(f'Arquivo existe: {os.path.exists(diff_file_path)}')

if os.path.exists(diff_file_path):
    with open(diff_file_path) as f:
        data = json.load(f)
    print(f'Severidade c1_vs_c2_severe: {data[\"RBF\"][\"c1_vs_c2_severe\"]:.2f}%')
"
```

**Resultado Esperado**:
```
Caminho resolvido: C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid\test_real_results_heatmaps\concept_heatmapsS\concept_differences.json
Arquivo existe: True
Severidade c1_vs_c2_severe: 60.45%
```

---

## 🎯 EXPECTATIVAS DO RE-TESTE

### O que DEVE mudar (comparado ao teste anterior):

#### Log de Execucao:

**ANTES** (teste anterior):
```
[WARNING] Concept difference data file not found at test_real_results_heatmaps/concept_heatmapsS/concept_differences.json
Chunk 3: Using drift_severity='STABLE' from previous chunk
```

**DEPOIS** (esperado neste re-teste):
```
[INFO] Concept difference data loaded successfully from /path/to/test_real_results_heatmaps/concept_heatmapsS/concept_differences.json
Chunk 3: drift_severity='SEVERE' detected (c1 -> c2_severe, 60.5% difference)
[INFO] -> SEVERE DRIFT DETECTED: Seeding INTENSIVO ativado (85% seeding, 90% injection)
```

#### Performance no Chunk 4:

**ANTES**:
```
Chunk 4: TrainGmean=93.58%, TestGmean=45.53%, TestF1=45.91%
```

**DEPOIS (expectativa)**:
```
Chunk 4: TrainGmean=~90-95%, TestGmean=~70-80%, TestF1=~70-80%
```

**Justificativa**:
- Seeding 85% ira injetar regras de ALTA QUALIDADE desde o inicio
- Populacao inicial do chunk 4 sera MELHOR adaptada ao novo conceito
- Recovery mode (26 geracoes) tera uma BASE melhor para evoluir

#### Metricas Finais:

| Metrica | Teste Anterior | Expectativa Re-teste | Melhoria |
|---------|----------------|----------------------|----------|
| **Avg Test G-mean** | 81.77% | **~85%** | +3-4pp |
| **Chunk 4 G-mean** | 45.53% | **~70-80%** | +25-35pp |
| **Seeding 85% ativado** | NAO | **SIM** | ✅ |
| **Drift detectado proativo** | NAO (chunk 4) | **SIM (chunk 3)** | ✅ |

---

## 📊 VALIDACAO POS-EXECUCAO

Apos o re-teste, verificar:

### 1. Log de Execucao

**Buscar por**:
```bash
grep "Concept difference data loaded" experimento_reteste_*.log
grep "SEVERE DRIFT DETECTED" experimento_reteste_*.log
grep "Seeding INTENSIVO ativado" experimento_reteste_*.log
```

**Resultado esperado**: 3 linhas encontradas (confirmando deteccao proativa)

### 2. Metricas de Performance

**Ler**:
```bash
tail -50 experimento_reteste_*.log | grep "Chunk 4 Results"
```

**Resultado esperado**:
```
Chunk 4 Results: TrainGmean=XX.XX%, TestGmean=~70-80%, TestF1=~70-80%
```

### 3. Plots Visuais

**Gerar plots**:
```bash
python generate_plots.py experiments_test/RBF_Abrupt_Severe/run_1 \
    --diff_file test_real_results_heatmaps/concept_heatmapsS/concept_differences.json
```

**Comparar**:
- Plot de Acuracia Periodica: Curva azul deve mostrar queda MENOR (70-80% vs 46%)
- Plot GA Chunk 4: Pode ter mais geracoes (se seeding for muito bom, early stopping pode parar antes)
- Heatmap: Chunk 4 pode ter complexidade similar, mas com MELHOR performance

---

## 🚀 COMANDOS DE EXECUCAO

### Local (Windows):

```powershell
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"
python main.py > experimento_reteste_6chunks_$(Get-Date -Format "yyyyMMdd_HHmmss").log 2>&1
```

### Google Colab:

**Celula 1 - Setup**:
```python
from google.colab import drive
import os
import time
from datetime import datetime

drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/DSL-AG-hybrid')

# Validar arquivos
required_files = [
    'main.py',
    'config_test_single.yaml',
    'test_real_results_heatmaps/concept_heatmapsS/concept_differences.json'
]

for f in required_files:
    if os.path.exists(f):
        print(f"✅ {f}")
    else:
        print(f"❌ FALTANDO: {f}")
```

**Celula 2 - Executar**:
```python
import subprocess

log_filename = f"experimento_reteste_6chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
print(f"Log: {log_filename}")
print(f"Tempo estimado: 8-10 horas")
print("")

start_time = time.time()

process = subprocess.Popen(
    ['python', 'main.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

with open(log_filename, 'w') as log_file:
    for line in process.stdout:
        print(line, end='')
        log_file.write(line)
        log_file.flush()

process.wait()
elapsed = time.time() - start_time

print("")
print("="*70)
print(f"Experimento concluido!")
print(f"Tempo total: {elapsed/3600:.2f} horas")
print(f"Log: {log_filename}")
print("="*70)
```

---

## 📝 NOTAS IMPORTANTES

### 1. Compatibilidade com Batch de Streams

Os ajustes realizados sao **COMPATIVEIS** com o batch de streams:

- Ajuste 1 (config file): Trocar de volta para `config.yaml` ou `config_6chunks.yaml` para batch
- Ajuste 2 (caminho absoluto): **Funciona para qualquer configuracao** (nao precisa mudar)

### 2. Reversao para Config Original

Para voltar ao config original depois do teste:

```python
# Em main.py linha 1474, trocar de volta:
config_file_path = os.path.join(script_dir, "config.yaml")
```

### 3. Arquivos que NAO Mudaram

- `config_test_single.yaml` - Nao precisa mudar
- `ga.py` - Nao precisa mudar (seeding 85% ja esta implementado)
- `data_handling_v8.py` - Nao precisa mudar
- `analyze_concept_difference.py` - Nao precisa mudar (ja foi executado)

---

## 🎯 PROXIMOS PASSOS

### Apos Re-Teste:

1. ✅ **Analisar log**: Buscar confirmacao de seeding 85%
2. ✅ **Gerar plots**: Comparar com teste anterior
3. ✅ **Validar metricas**: Chunk 4 deve ter G-mean ~70-80%
4. ✅ **Decidir proximo passo**:
   - Se G-mean chunk 4 ≥ 70%: **SUCESSO!** Prosseguir para batch de streams
   - Se G-mean chunk 4 < 70%: Ajustar parametros (mais geracoes, validacao, etc)

---

## 📊 COMPARACAO: Teste Anterior vs Re-Teste

| Aspecto | Teste Anterior | Re-Teste |
|---------|----------------|----------|
| **Config usado** | config_test_single.yaml (manual) | config_test_single.yaml (automatico) |
| **concept_differences.json** | ❌ NAO encontrado | ✅ Sera encontrado |
| **Drift detectado** | Chunk 4 (reativo) | Chunk 3 (proativo) |
| **Seeding 85%** | ❌ NAO ativado | ✅ Sera ativado |
| **Chunk 4 G-mean** | 45.53% | **~70-80% (esperado)** |
| **Avg G-mean** | 81.77% | **~85% (esperado)** |
| **Tempo execucao** | 10h 11min | ~8-10h (similar) |

---

**Documento criado por**: Claude Code
**Data**: 2025-10-29
**Status**: ✅ **AJUSTES PRONTOS - AGUARDANDO EXECUCAO DO RE-TESTE**
