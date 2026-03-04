# MCMO Module

Multistream Classification based on Multi-objective Optimization

## Estrutura

```
mcmo/
├── __init__.py           # Inicialização do módulo
├── MCMO.py              # Código original (do GitHub)
├── GMM.py               # Dynamic GMM (do GitHub)
├── OptAlgorithm.py      # NSGA-II feature selection (do GitHub)
├── baseline_mcmo.py     # Adapter para single-stream
└── README.md            # Esta documentação
```

## Código Original

**Fonte:** https://github.com/Jesen-BT/MCMO

**Paper:** "Reduced-space Multistream Classification based on Multi-objective Evolutionary Optimization" (IEEE TEVC 2023)

**Arquivos:**
- `MCMO.py`: Classe principal MCMO com métodos `source_fit()`, `predict()`, `partial_fit()`
- `GMM.py`: DGMM (Dynamic Gaussian Mixture Model) para sample weighting
- `OptAlgorithm.py`: NSGA-II para feature selection com objetivos MMD + Fisher criterion

## Adapter (baseline_mcmo.py)

Adaptação do MCMO para funcionar com **single-stream** data usando **temporal splitting**.

### Classes Principais

#### 1. MCMOAdapter

Converte single stream → multistream tratando chunks consecutivos como source streams.

```python
from mcmo.baseline_mcmo import MCMOAdapter

# Inicializar
adapter = MCMOAdapter(
    n_sources=3,          # Número de chunks passados como sources
    initial_beach=200,    # Warmup period
    max_pool=5,          # Tamanho da pool de classificadores
    verbose=True
)

# Processar chunks
for X_chunk, y_chunk in stream_chunks:
    predictions = adapter.partial_fit_predict(X_chunk, y_chunk)
    accuracy = (predictions == y_chunk).mean()
    print(f"Chunk accuracy: {accuracy:.4f}")
```

**Estratégia:**
- Buffer mantém últimos N chunks
- Chunks t-3, t-2, t-1 → Source streams (com labels)
- Chunk t → Target stream (sem labels durante predição)

#### 2. MCMOEvaluator

Interface simplificada para avaliação (compatível com pipeline Phase 3).

```python
from mcmo.baseline_mcmo import MCMOEvaluator

evaluator = MCMOEvaluator(n_sources=3, verbose=True)

for X_chunk, y_chunk in stream_chunks:
    predictions, metrics = evaluator.evaluate_chunk(X_chunk, y_chunk)
    print(f"Accuracy: {metrics['accuracy']:.4f}")

# Métricas globais
global_metrics = evaluator.get_global_metrics()
print(f"Global accuracy: {global_metrics['global_accuracy']:.4f}")
```

#### 3. test_mcmo_adapter()

Função de conveniência para testes rápidos.

```python
from mcmo.baseline_mcmo import test_mcmo_adapter

# Preparar chunks
X_chunks = [X1, X2, X3, ...]
y_chunks = [y1, y2, y3, ...]

# Testar
results = test_mcmo_adapter(X_chunks, y_chunks, n_sources=3)
print(f"Mean accuracy: {results['mean_accuracy']:.4f}")
```

## Dependências

```bash
pip install numpy geatpy==2.7.0 scikit-multiflow==0.5.3
```

**Nota:** scikit-multiflow está descontinuado, mas é requerido pelo código original do MCMO.

## Uso no Pipeline

### Integração com main.py

```python
# Em main.py
from mcmo.baseline_mcmo import MCMOEvaluator

baseline_models = {
    'GBML': gbml_evaluator,
    'ARF': arf_evaluator,
    'HAT': hat_evaluator,
    'MCMO': MCMOEvaluator(n_sources=3, verbose=False),
    # ... outros modelos
}
```

### Teste Isolado

```python
# test_mcmo_electricity.py
import sys
sys.path.append('C:\\Users\\Leandro Almeida\\Downloads\\DSL-AG-hybrid')

from mcmo.baseline_mcmo import MCMOEvaluator
import pandas as pd
import numpy as np

# Carregar Electricity
data = pd.read_csv('datasets/Electricity.csv')
X = data.drop('class', axis=1).values
y = data['class'].values

# Dividir em chunks
chunk_size = 1000
n_chunks = len(X) // chunk_size

evaluator = MCMOEvaluator(n_sources=3, verbose=True)

for i in range(n_chunks):
    start = i * chunk_size
    end = start + chunk_size

    X_chunk = X[start:end]
    y_chunk = y[start:end]

    predictions, metrics = evaluator.evaluate_chunk(X_chunk, y_chunk)
    print(f"Chunk {i+1}: Accuracy = {metrics['accuracy']:.4f}")

# Resultados globais
global_metrics = evaluator.get_global_metrics()
print(f"\nGlobal Accuracy: {global_metrics['global_accuracy']:.4f}")
```

## Funcionamento Interno

### Fluxo do MCMOAdapter

1. **Buffer de Chunks:**
   - Mantém deque com últimos `chunk_buffer_size` chunks
   - Cada chunk: tupla `(X, y)`

2. **Fase Inicial (≤ n_sources chunks):**
   - Usa baseline simples (HoeffdingTreeClassifier)
   - Acumula chunks no buffer

3. **Fase MCMO (> n_sources chunks):**
   - Extrai source streams: últimos n_sources chunks (excluindo atual)
   - Target stream: chunk atual
   - Para cada amostra no target:
     - Treina source classifiers com amostras correspondentes dos sources
     - Prediz com MCMO ensemble
     - Atualiza MCMO com true label

### Temporal Splitting

```
Buffer: [chunk_0, chunk_1, chunk_2, chunk_3, chunk_4]
                   ↓        ↓        ↓        ↓
                Source1  Source2  Source3   Target
                (t-3)    (t-2)    (t-1)      (t)
```

**Vantagens:**
- Mantém ordem temporal natural
- Simula covariate shift de drift temporal
- Não requer modificação do código MCMO original

**Desvantagens:**
- Latência inicial (precisa de n_sources+1 chunks)
- Requer memória para buffer
- Pode ter overfitting se chunks muito similares

## Parâmetros Recomendados

### Datasets Pequenos (< 50k samples)
```python
MCMOAdapter(
    n_sources=3,
    initial_beach=100,
    max_pool=3,
    gaussian_number=5
)
```

### Datasets Médios (50k - 500k samples)
```python
MCMOAdapter(
    n_sources=3,
    initial_beach=200,
    max_pool=5,
    gaussian_number=5
)
```

### Datasets Grandes (> 500k samples)
```python
MCMOAdapter(
    n_sources=4,
    initial_beach=500,
    max_pool=7,
    gaussian_number=7
)
```

## Troubleshooting

### ImportError: No module named 'geatpy'

```bash
pip install geatpy==2.7.0
```

### ImportError: No module named 'skmultiflow'

```bash
pip install scikit-multiflow==0.5.3
```

### Conflito com river

Se houver conflito entre scikit-multiflow e river, considerar:

1. **Ambiente virtual separado:**
```bash
conda create -n mcmo_env python=3.8
conda activate mcmo_env
pip install geatpy==2.7.0 scikit-multiflow==0.5.3
```

2. **Isolar imports no adapter** (já implementado com try/except)

3. **Substituir HoeffdingTreeClassifier por river** (requer modificar MCMO.py)

## Performance

### Tempo de Execução

**NSGA-II feature selection:**
- 50 gerações × 50 indivíduos = 2500 evaluations
- Cada evaluation: MMD O(n²) + Fisher O(d³)
- Executa a cada `initial_beach` amostras

**Estimativa:**
- Electricity (45k samples, 8 features): ~15-30s por inicialização
- CovType (581k samples, 54 features): ~60-120s por inicialização

### Otimizações Possíveis

1. Reduzir gerações/população NSGA-II (modificar MCMO.py:119-124)
2. Aumentar `initial_beach` para executar NSGA-II menos vezes
3. Usar warm start (reutilizar população NSGA-II anterior)

## Referências

- **Paper:** Jesen-BT et al. "Reduced-space Multistream Classification based on Multi-objective Evolutionary Optimization", IEEE TEVC 2023
- **GitHub:** https://github.com/Jesen-BT/MCMO
- **Documentação completa:** `MCMO_API_DOCUMENTATION.md`
- **Análise do paper:** `ANALISE_MCMO_PAPER.md`
- **Plano de integração:** `PLANO_INTEGRACAO_MCMO.md`

## Contato

Desenvolvido por: Claude Code
Data: 2025-11-24
Projeto: DSL-AG-hybrid Phase 3
