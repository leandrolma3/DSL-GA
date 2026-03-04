# 🐛 BUGFIX: Config Path não estava sendo passado corretamente

**Data**: 2025-10-30
**Problema**: Stream RBF_Drift_Recovery não foi encontrado porque data_handling.py carregava `config.yaml` ao invés de `config_test_drift_recovery.yaml`
**Status**: ✅ CORRIGIDO

---

## 🔍 ANÁLISE DO PROBLEMA

### Erro Reportado pelo Usuário:

```
2025-10-30 16:01:21 [INFO] data_handling_final: Full configuration loaded from config.yaml
2025-10-30 16:01:21 [ERROR] main: Failed data generation for 'RBF_Drift_Recovery': Stream 'RBF_Drift_Recovery' not found in 'experimental_streams'.
ValueError: Stream 'RBF_Drift_Recovery' not found in 'experimental_streams'.
```

### Sintomas:

1. ✅ main.py carregou `config_test_drift_recovery.yaml` corretamente (linha 1486)
2. ✅ RBF_Drift_Recovery existe no `config_test_drift_recovery.yaml`
3. ❌ data_handling.py carregou `config.yaml` ao invés de `config_test_drift_recovery.yaml`
4. ❌ Stream não foi encontrado porque estava procurando no arquivo errado

---

## 🕵️ ROOT CAUSE ANALYSIS

### Fluxo do Código (ANTES):

```
1. main.py linha 1486:
   config_file_path = "config_test_drift_recovery.yaml"

2. main.py linha 1508:
   config = load_config(config_file_path)  # ✅ Carrega arquivo correto

3. main.py linha 1589-1595:
   run_experiment(
       experiment_id=stream_name,
       config=config,  # ✅ Passa dicionário de config
       # ❌ NÃO passa config_file_path!
   )

4. main.py linha 377-385 (dentro de run_experiment):
   chunks = data_handling.generate_dataset_chunks(
       config_path=config.get('config_path', 'config.yaml')  # ❌ PROBLEMA!
   )
   # config é um dicionário (conteúdo do YAML)
   # config NÃO tem chave 'config_path'
   # Fallback para 'config.yaml' (ERRADO!)

5. data_handling.py linha 130:
   config = load_full_config(config_path)  # ❌ Carrega 'config.yaml'
```

### Root Cause:

**main.py linha 384** tentava acessar `config.get('config_path', 'config.yaml')`, mas:

- `config` é o **conteúdo** do YAML (dicionário com experiment_settings, ga_params, etc.)
- `config_path` **NÃO é uma chave do YAML**, é uma variável local (`config_file_path`)
- Como a chave não existe, usava fallback `'config.yaml'`
- `data_handling.py` então carregava o arquivo errado

---

## ✅ CORREÇÃO IMPLEMENTADA

### Mudança 1: Adicionar parâmetro `config_file_path` em `run_experiment`

**Arquivo**: `main.py` (linha 266-272)

**ANTES**:
```python
def run_experiment(
    experiment_id: str,
    run_number: int,
    config: Dict,
    base_results_dir: str,
    is_drift_simulation: bool
    ) -> Union[Dict, None]:
```

**DEPOIS**:
```python
def run_experiment(
    experiment_id: str,
    run_number: int,
    config: Dict,
    base_results_dir: str,
    is_drift_simulation: bool,
    config_file_path: str = 'config.yaml'  # <<< ADICIONADO >>>
    ) -> Union[Dict, None]:
```

---

### Mudança 2: Passar `config_file_path` para `generate_dataset_chunks`

**Arquivo**: `main.py` (linha 377-385)

**ANTES**:
```python
chunks = data_handling.generate_dataset_chunks(
    stream_or_dataset_name=experiment_id,
    chunk_size=chunk_size,
    num_chunks=num_chunks,
    max_instances=max_instances,
    run_number=run_number,
    results_dir=run_results_dir,
    config_path=config.get('config_path', 'config.yaml')  # ❌ ERRADO
)
```

**DEPOIS**:
```python
chunks = data_handling.generate_dataset_chunks(
    stream_or_dataset_name=experiment_id,
    chunk_size=chunk_size,
    num_chunks=num_chunks,
    max_instances=max_instances,
    run_number=run_number,
    results_dir=run_results_dir,
    config_path=config_file_path  # ✅ CORRETO: Usa variável real
)
```

---

### Mudança 3: Passar `config_file_path` na chamada de `run_experiment`

**Arquivo**: `main.py` (linha 1589-1596)

**ANTES**:
```python
run_result = run_experiment(
    experiment_id=stream_name,
    run_number=i + 1,
    config=config,
    base_results_dir=base_results_dir_main,
    is_drift_simulation=is_drift_mode
    # ❌ Faltava passar config_file_path
)
```

**DEPOIS**:
```python
run_result = run_experiment(
    experiment_id=stream_name,
    run_number=i + 1,
    config=config,
    base_results_dir=base_results_dir_main,
    is_drift_simulation=is_drift_mode,
    config_file_path=config_file_path  # ✅ ADICIONADO
)
```

---

### Mudança 4 (BONUS): Ajustar para usar Fase 1

**Arquivo**: `main.py` (linha 1486)

**ANTES**:
```python
config_file_path = os.path.join(script_dir, "config_test_single.yaml")
```

**DEPOIS**:
```python
config_file_path = os.path.join(script_dir, "config_test_drift_recovery.yaml")  # Fase 1
```

---

## 🔄 FLUXO CORRIGIDO

```
1. main.py linha 1486:
   config_file_path = "config_test_drift_recovery.yaml"  ✅

2. main.py linha 1508:
   config = load_config(config_file_path)  ✅

3. main.py linha 1589-1596:
   run_experiment(
       experiment_id=stream_name,
       config=config,
       config_file_path=config_file_path  # ✅ Passa caminho do arquivo
   )

4. main.py linha 377-385 (dentro de run_experiment):
   chunks = data_handling.generate_dataset_chunks(
       config_path=config_file_path  # ✅ Usa caminho correto
   )

5. data_handling.py linha 130:
   config = load_full_config(config_path)  # ✅ Carrega arquivo correto!
```

---

## 📊 IMPACTO DA CORREÇÃO

### Antes:
- ❌ Sempre carregava `config.yaml` em data_handling.py
- ❌ Não conseguia encontrar streams definidos em outros configs
- ❌ Impossível testar com configs alternativos

### Depois:
- ✅ Carrega o arquivo de config correto especificado no main.py
- ✅ Encontra streams definidos em `config_test_drift_recovery.yaml`
- ✅ Permite usar múltiplos configs (single, recovery, multi_drift)

---

## ✅ VALIDAÇÃO

### Sintaxe Python:
```bash
python -m py_compile main.py  # ✅ SEM ERROS
```

### Arquivos Modificados:
- ✅ `main.py` (3 locais modificados)

### Arquivos que NÃO precisam ser modificados:
- ✅ `data_handling.py` (já estava correto, recebia config_path como parâmetro)
- ✅ `ga.py` (não afetado)
- ✅ Configs YAML (não afetados)

---

## 🚀 PRÓXIMOS PASSOS

### 1. Sincronizar com Google Drive

Apenas **1 arquivo** precisa ser sincronizado:
- ✅ `main.py` (com 3 correções + ajuste para Fase 1)

### 2. Executar Fase 1 no Colab

Agora o comando deve funcionar corretamente:

```python
%cd /content/drive/MyDrive/DSL-AG-hybrid
!python main.py
```

### 3. Validar nos Logs

Procurar no log:
```bash
# Deve mostrar arquivo correto:
grep "Full configuration loaded from" experimento_*.log
# Esperado: "Full configuration loaded from config_test_drift_recovery.yaml"

# Deve encontrar o stream:
grep "Generating stream: 'RBF_Drift_Recovery'" experimento_*.log
# Esperado: "--- Generating stream: 'RBF_Drift_Recovery' ---"
```

---

## 📚 LIÇÕES APRENDIDAS

### 1. Diferença entre Config Path vs Config Content

- **config_file_path**: String com caminho do arquivo (`"config_test_drift_recovery.yaml"`)
- **config**: Dicionário com conteúdo do YAML (`{'experiment_settings': {...}, 'ga_params': {...}}`)

Não confundir os dois!

### 2. Fallback values podem mascarar bugs

```python
config.get('config_path', 'config.yaml')  # Fallback silencioso
```

Se a chave não existe, retorna fallback sem erro → dificulta debug.

### 3. Propagar paths explicitamente

Melhor passar `config_file_path` como parâmetro explícito do que tentar extrair de `config`.

---

## 🔍 DEBUGGING TIPS

Se encontrar erro similar no futuro:

1. **Verificar qual config está sendo carregado**:
   ```bash
   grep "Full configuration loaded from" *.log
   ```

2. **Verificar se stream existe no config**:
   ```bash
   grep -A 5 "experimental_streams:" config_test_*.yaml
   ```

3. **Verificar se config_path está sendo passado**:
   ```bash
   grep "config_path=" main.py
   ```

---

**Documento criado por**: Claude Code
**Data**: 2025-10-30
**Status**: ✅ BUG CORRIGIDO - PRONTO PARA TESTAR
