# BUGFIX: UnboundLocalError - numpy import duplicado

**Data:** 2025-11-02
**Status:** ✅ CORRIGIDO
**Severidade:** CRÍTICO (bloqueava execução)

---

## 🐛 ERRO ORIGINAL

```
2025-11-02 11:10:53 [ERROR] main: GA execution error on chunk 0:
cannot access local variable 'np' where it is not associated with a value

Traceback (most recent call last):
  File "/content/drive/MyDrive/DSL-AG-hybrid/ga.py", line 967, in run_genetic_algorithm
    avg_fitness = np.nanmean(fitness_values); std_fitness = np.nanstd(fitness_values); ...
                  ^^
UnboundLocalError: cannot access local variable 'np' where it is not associated with a value
```

---

## 🔍 CAUSA RAIZ

### Problema: Import Duplicado Dentro de Bloco Condicional

**ga.py linha 6 (topo do arquivo):**
```python
import numpy as np  # Import GLOBAL
```

**ga.py linha 835 (DENTRO do if - ERRO):**
```python
if elite_gmeans:
    import numpy as np  # Import LOCAL - cria variável local 'np'
    median_elite_gmean = np.median(elite_gmeans)
```

**ga.py linha 967 (FORA do if - ERRO):**
```python
avg_fitness = np.nanmean(fitness_values)  # Tenta usar 'np' global, mas existe 'np' local não inicializada
```

### Explicação Técnica

Em Python, quando você faz `import numpy as np` DENTRO de uma função/bloco:
1. Cria uma variável LOCAL `np` naquele escopo
2. Essa variável local "shadowing" (sobrescreve) a variável global `np`
3. Fora do bloco if, Python vê que existe uma variável local `np` (mesmo que não inicializada naquele ponto)
4. Python não acessa a global `np`, mas a local `np` não existe ainda → **UnboundLocalError**

---

## ✅ CORREÇÃO APLICADA

**ga.py linha 835 (CORRIGIDO):**
```python
if elite_gmeans:
    # REMOVIDO: import numpy as np
    median_elite_gmean = np.median(elite_gmeans)  # Usa np global
    early_stop_threshold = median_elite_gmean
```

### Mudança
- **ANTES:** `import numpy as np` dentro do if (linha 835)
- **DEPOIS:** Removido, usa `np` global já importado na linha 6

---

## 🧪 VALIDAÇÃO

```bash
# Testar sintaxe
python -m py_compile ga.py
# ✅ SEM ERROS
```

### Teste de Execução
O erro acontecia na **primeira geração do GA** (generation 0), quando:
1. `generation > 0` é False → bloco if não executa
2. `np` local não é criado
3. Linha 967 tenta usar `np` → erro

**Após correção:**
- `np` é sempre a variável global (linha 6)
- Funciona em qualquer geração (0, 1, 2, ...)

---

## 📝 LIÇÕES APRENDIDAS

1. **NUNCA fazer import dentro de blocos condicionais** se a variável for usada fora
2. **Imports devem estar no topo do arquivo** (PEP 8)
3. **Variável local shadowing é perigoso** - Python prioriza local mesmo se não inicializada

### Boa Prática
```python
# ✅ BOM: Import no topo
import numpy as np

def funcao():
    if condicao:
        resultado = np.median(dados)  # Usa global
```

### Má Prática (causou o bug)
```python
# ❌ RUIM: Import condicional
import numpy as np  # Global

def funcao():
    if condicao:
        import numpy as np  # Local - shadowing!
        resultado = np.median(dados)

    # ERRO aqui: np local existe mas não foi inicializada (if não executou)
    media = np.mean(outros_dados)
```

---

## ✅ CHECKLIST PÓS-CORREÇÃO

- [x] Erro corrigido (linha 835 removida)
- [x] Sintaxe Python validada
- [x] Nenhum outro import condicional no código
- [ ] Smoke test executado com sucesso
- [ ] Experimento completo validado

---

**FIM DO BUGFIX**

**Próximo passo:** Re-executar smoke test
```bash
python main.py config_test_single.yaml --num_chunks 2 --run_number 999
```
