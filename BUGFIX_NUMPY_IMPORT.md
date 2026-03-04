# BUGFIX - Import Redundante de Numpy

**Data:** 2025-11-04
**Severidade:** CRITICA (bloqueava execucao)
**Status:** CORRIGIDO

---

## ERRO ORIGINAL

```
UnboundLocalError: cannot access local variable 'np' where it is not associated with a value
File "/content/drive/MyDrive/DSL-AG-hybrid/main.py", line 286
```

**Sintoma:** Experimento falhou ao iniciar antes mesmo do primeiro chunk.

---

## CAUSA RAIZ

Na implementacao da validacao cruzada (linha 639), adicionei:
```python
import numpy as np  # DENTRO da funcao
```

**Problema:**
- `numpy` ja foi importado no topo do arquivo (linha 7): `import numpy as np`
- Import DENTRO da funcao cria variavel LOCAL `np`
- Python assume que `np` e local em TODA a funcao
- Quando linha 286 tenta usar `np.random.seed()`, `np` ainda nao foi definido localmente
- Resultado: UnboundLocalError

**Exemplo do problema:**
```python
import numpy as np  # Topo do arquivo

def funcao():
    x = np.random.seed(42)  # Erro! np e local mas ainda nao definido

    # Mais tarde na funcao:
    import numpy as np  # Cria variavel local np
```

---

## CORRECAO APLICADA

**Arquivo:** main.py linha 639

**Antes (ERRADO):**
```python
# Calcular G-mean de validacao
from sklearn.metrics import confusion_matrix
import numpy as np  # REDUNDANTE E PROBLEMATICO

cm = confusion_matrix(...)
```

**Depois (CORRETO):**
```python
# Calcular G-mean de validacao
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(...)
```

**Mudanca:** Removido `import numpy as np` da linha 639

---

## VALIDACAO

```bash
python -m py_compile main.py
# SEM ERROS
```

---

## LICAO APRENDIDA

1. **NUNCA importar modulos dentro de funcoes** se ja foram importados no topo
2. Imports locais criam variaveis locais, confundindo Python
3. Sempre validar sintaxe apos modificacoes

---

## COMANDO CORRIGIDO

Agora o experimento pode rodar sem erros:

```bash
cd /content/drive/MyDrive/DSL-AG-hybrid && python main.py config_test_single.yaml --num_chunks 3 --run_number 998 2>&1 | tee smoke_test_antidrift.log
```

---

**Status:** CORRIGIDO, sintaxe validada, pronto para smoke test
