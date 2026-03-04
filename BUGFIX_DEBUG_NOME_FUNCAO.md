# BUGFIX - Nome de Função Incorreto no Debug

**Data:** 2025-11-03
**Severidade:** CRITICO (bloqueava toda execução)
**Status:** CORRIGIDO

---

## ERRO ORIGINAL

```
NameError: name 'calculate_fitness_hybrid' is not defined. Did you mean: 'calculate_fitness'?
File "/content/drive/MyDrive/DSL-AG-hybrid/fitness.py", line 216, in calculate_fitness
    if not hasattr(calculate_fitness_hybrid, 'early_stop_checks'):
                   ^^^^^^^^^^^^^^^^^^^^^^^^
```

**Sintoma:** Experimento falhou imediatamente ao tentar avaliar primeiro indivíduo, com centenas de erros idênticos.

---

## CAUSA RAIZ

No debug logging adicionado em fitness.py, usei o nome **`calculate_fitness_hybrid`** para os contadores estáticos, mas a função real se chama apenas **`calculate_fitness`**.

**Código incorreto (linhas 216-280):**
```python
# ERRADO
if not hasattr(calculate_fitness_hybrid, 'early_stop_checks'):
    calculate_fitness_hybrid.early_stop_checks = 0
    calculate_fitness_hybrid.early_stop_discards = 0

if calculate_fitness_hybrid.early_stop_checks < 5:
    ...

calculate_fitness_hybrid.early_stop_checks += 1
calculate_fitness_hybrid.early_stop_discards += 1

if not hasattr(calculate_fitness_hybrid, 'no_early_stop_logged'):
    calculate_fitness_hybrid.no_early_stop_logged = True
```

**Nome correto da função:**
```python
def calculate_fitness(  # <- Nome real
    individual, data, target, ...
```

---

## CORRECAO APLICADA

Substituído **todas** as 6 ocorrências de `calculate_fitness_hybrid` por `calculate_fitness`:

**fitness.py linha 216-218:**
```python
# CORRETO
if not hasattr(calculate_fitness, 'early_stop_checks'):
    calculate_fitness.early_stop_checks = 0
    calculate_fitness.early_stop_discards = 0
```

**fitness.py linha 225-226:**
```python
# CORRETO
if calculate_fitness.early_stop_checks < 5:
    debug_print_fitness(f"Early stop check #{calculate_fitness.early_stop_checks + 1}...")
```

**fitness.py linha 241:**
```python
# CORRETO
calculate_fitness.early_stop_checks += 1
```

**fitness.py linha 248:**
```python
# CORRETO
if calculate_fitness.early_stop_checks <= 10:
    debug_print_fitness(...)
```

**fitness.py linha 252:**
```python
# CORRETO
calculate_fitness.early_stop_discards += 1
```

**fitness.py linha 255:**
```python
# CORRETO
if calculate_fitness.early_stop_discards <= 5:
    debug_print_fitness(...)
```

**fitness.py linha 278-280:**
```python
# CORRETO
if not hasattr(calculate_fitness, 'no_early_stop_logged'):
    calculate_fitness.no_early_stop_logged = True
    debug_print_fitness(f"Early stop DESATIVADO: threshold={early_stop_threshold}")
```

---

## VALIDACAO

```bash
python -m py_compile fitness.py
# SEM ERROS
```

---

## LICAO APRENDIDA

Ao adicionar debug logging com contadores estáticos em funções:
1. Verificar nome EXATO da função com `grep "^def nome_funcao"`
2. Usar nome correto para `hasattr()` e atributos estáticos
3. Testar sintaxe E execução antes de commit

---

## PROXIMO PASSO

Re-executar smoke test com correção:

```powershell
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"

python main.py config_test_single.yaml --num_chunks 2 --run_number 998 2>&1 | Tee-Object -FilePath "smoke_test_debug_corrigido.log"
```

---

**STATUS:** CORRIGIDO, sintaxe validada, pronto para smoke test
