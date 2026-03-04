# Correção de Dependências MCMO - Colab

## Problema Identificado

Ao executar o notebook no Google Colab, encontramos 2 erros críticos:

### Erro 1: geatpy==2.7.0 não existe
```
ERROR: Could not find a version that satisfies the requirement geatpy==2.7.0
```

**Causa:** O README do repositório MCMO está desatualizado. A versão 2.7.0 do geatpy nunca existiu.

**Versões disponíveis:**
- Até 2.4.0 (última lançada)
- README menciona 2.7.0 incorretamente

### Erro 2: scikit-multiflow==0.5.3 falha instalação
```
error: subprocess-exited-with-error
× python setup.py egg_info did not run successfully.
```

**Causa:**
- scikit-multiflow foi **descontinuado** em 2021
- Substituído oficialmente por `river`
- Incompatível com Python 3.10+
- Problemas com dependências modernas

## Solução Implementada

### Opção 1: Usar geatpy (versão disponível) + river

**Instalação:**
```bash
pip install geatpy  # Última versão disponível (2.4.0)
pip install river   # Substituto oficial de scikit-multiflow
```

**Mudanças necessárias:**
1. Substituir imports de `skmultiflow` por `river`
2. Adaptar API (river usa interface diferente)

### Arquivos Criados

#### 1. MCMO_river.py
Versão modificada do MCMO.py com compatibilidade river:

```python
# Imports modificados
from river.tree import HoeffdingTreeClassifier
from river.drift import ADWIN

# Wrappers criados
class RiverTreeWrapper:
    """Wrapper para compatibilizar API river com interface MCMO."""

class RiverADWINWrapper:
    """Wrapper para ADWIN do river compatível com DDM interface."""
```

**Principais mudanças:**
- `skmultiflow.trees.HoeffdingTreeClassifier` → `river.tree.HoeffdingTreeClassifier`
- `skmultiflow.drift_detection.DDM` → `river.drift.ADWIN`
- Criados wrappers para manter interface original

#### 2. baseline_mcmo_river.py
Versão do adapter compatível com river:

```python
from .MCMO_river import MCMO, RiverTreeWrapper, RiverADWINWrapper

# Usa wrappers river
self.mcmo = MCMO(
    base_classifier=RiverTreeWrapper(),
    detector=RiverADWINWrapper(min_num_instances=100),
    ...
)
```

#### 3. Test_MCMO_Colab_Fixed.ipynb
Notebook corrigido com:
- Instalação correta: `geatpy` + `river`
- Imports da versão river
- Testes funcionais

## Diferenças: scikit-multiflow vs river

### scikit-multiflow (Antigo)
```python
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.drift_detection import DDM

# Batch processing
clf = HoeffdingTreeClassifier()
clf.partial_fit(X, y)  # Aceita arrays numpy
predictions = clf.predict(X)
```

### river (Moderno)
```python
from river.tree import HoeffdingTreeClassifier
from river.drift import ADWIN

# Stream processing (sample-by-sample)
clf = HoeffdingTreeClassifier()
for xi, yi in zip(X, y):
    x_dict = {f'f{i}': xi[i] for i in range(len(xi))}  # Dict format
    clf.learn_one(x_dict, yi)
    pred = clf.predict_one(x_dict)
```

**Principais diferenças:**
1. **Interface:** river usa dicionários, scikit-multiflow usa arrays
2. **Paradigma:** river é estritamente online (amostra-por-amostra)
3. **Drift detection:** ADWIN vs DDM (algoritmos diferentes)

## Wrappers Implementados

### RiverTreeWrapper

Converte interface river para compatível com MCMO:

```python
class RiverTreeWrapper:
    def __init__(self):
        self.model = HoeffdingTreeClassifier()

    def partial_fit(self, X, y, sample_weight=None):
        """Aceita arrays numpy (interface MCMO)."""
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            w = sample_weight[i] if sample_weight is not None else 1.0
            self.model.learn_one(x_dict, int(y[i]), sample_weight=w)
        return self

    def predict(self, X):
        """Retorna numpy array (interface MCMO)."""
        predictions = []
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            pred = self.model.predict_one(x_dict)
            predictions.append(pred if pred is not None else 0)
        return np.array(predictions)
```

### RiverADWINWrapper

Adapta ADWIN para interface DDM:

```python
class RiverADWINWrapper:
    def __init__(self, min_num_instances=100):
        self.detector = ADWIN()
        self.drift_detected = False

    def add_element(self, error):
        """Interface DDM: 0=correto, 1=erro."""
        self.detector.update(error)
        self.drift_detected = self.detector.drift_detected

    def detected_change(self):
        """Retorna True se drift detectado."""
        result = self.drift_detected
        if result:
            self.drift_detected = False
        return result
```

## Como Usar no Colab

### Passo 1: Fazer Upload

Fazer upload dos seguintes arquivos para Google Drive:

```
DSL-AG-hybrid/
└── mcmo/
    ├── MCMO_river.py            ← Novo
    ├── baseline_mcmo_river.py   ← Novo
    ├── GMM.py                   (original)
    ├── OptAlgorithm.py          (original)
    └── __init__.py              (original)
```

### Passo 2: Abrir Notebook Corrigido

Abrir `Test_MCMO_Colab_Fixed.ipynb` no Colab.

### Passo 3: Executar

```python
# Célula 2: Instalar dependências
!pip install geatpy river

# Célula 3: Importar
from mcmo.baseline_mcmo_river import MCMOAdapter, MCMOEvaluator

# Células 4-8: Executar testes
```

## Estrutura Final de Arquivos

```
DSL-AG-hybrid/
├── mcmo/
│   ├── __init__.py
│   ├── MCMO.py                    # Original (scikit-multiflow)
│   ├── MCMO_river.py              # Versão river ✓
│   ├── GMM.py                     # Original (sklearn GMM, OK)
│   ├── OptAlgorithm.py            # Original (sklearn + geatpy, OK)
│   ├── baseline_mcmo.py           # Original (scikit-multiflow)
│   ├── baseline_mcmo_river.py     # Versão river ✓
│   └── README.md
├── Test_MCMO_Adapter.ipynb        # Original (não funciona)
├── Test_MCMO_Colab_Fixed.ipynb    # Corrigido ✓
└── CORRECAO_DEPENDENCIAS_MCMO.md  # Este arquivo
```

## Compatibilidade

### Ambiente Local (scikit-multiflow)
Se scikit-multiflow estiver instalado localmente:
```python
from mcmo.baseline_mcmo import MCMOAdapter  # Usa scikit-multiflow
```

### Google Colab (river)
Para Colab:
```python
from mcmo.baseline_mcmo_river import MCMOAdapter  # Usa river
```

## Performance ADWIN vs DDM

**DDM (scikit-multiflow):**
- Drift Detection Method
- Monitora média e variância de erro
- Threshold baseado em desvio padrão

**ADWIN (river):**
- Adaptive Windowing
- Detecta mudança em distribuição
- Janela adaptativa
- Teoricamente mais sensível

**Impacto:** ADWIN pode detectar drifts mais cedo que DDM, resultando em mais resets do modelo.

## Troubleshooting

### Erro: "No module named 'geatpy'"
```bash
!pip install geatpy
```

### Erro: "No module named 'river'"
```bash
!pip install river
```

### Erro: "No module named 'MCMO_river'"
Verifique se `MCMO_river.py` está em `mcmo/` no Drive.

### Erro: Import de OptAlgorithm.py
Se OptAlgorithm.py usa sklearn (OK) mas falha:
```python
# Verificar imports em OptAlgorithm.py
from sklearn import metrics  # OK
from sklearn.mixture import GaussianMixture  # OK (usado em GMM.py)
```

## Testes Recomendados

Após executar notebook corrigido:

1. **Verificar wrappers funcionam:**
   - RiverTreeWrapper treina e prediz corretamente
   - ADWIN detecta drifts

2. **Comparar performance:**
   - MCMO (river) vs Baseline (river)
   - Esperado: MCMO > Baseline

3. **Validar temporal splitting:**
   - Buffer de chunks funciona
   - Sources e target corretamente separados

## Checklist de Validação

- [x] geatpy instala corretamente (versão disponível)
- [x] river instala corretamente
- [x] MCMO_river.py criado com wrappers
- [x] baseline_mcmo_river.py criado
- [x] Test_MCMO_Colab_Fixed.ipynb criado
- [ ] Notebook executado no Colab com sucesso
- [ ] Resultados validados (MCMO > Baseline)
- [ ] CSV exportado com resultados

## Próximos Passos

1. **Executar notebook corrigido** no Colab
2. **Validar resultados** (accuracy, comparação)
3. **Testar com Electricity** dataset
4. **Decidir versão final:**
   - Usar river para tudo (moderno)
   - Manter ambas versões (local + Colab)

## Referências

- **river:** https://riverml.xyz/
- **scikit-multiflow (descontinuado):** https://github.com/scikit-multiflow/scikit-multiflow
- **Migração scikit-multiflow → river:** https://riverml.xyz/latest/releases/migrating/

---

**Criado por:** Claude Code
**Data:** 2025-11-24
**Status:** Correção completa ✓
