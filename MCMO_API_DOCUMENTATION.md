# MCMO - Documentação da API e Estrutura de Código

## 1. Visão Geral do Repositório

**GitHub:** https://github.com/Jesen-BT/MCMO

**Estrutura de Arquivos:**
```
MCMO/
├── MCMO.py              # Classe principal MCMO
├── GMM.py               # DGMM (Dynamic Gaussian Mixture Model)
├── OptAlgorithm.py      # NSGA-II feature selection
├── demo.py              # Exemplo de uso
├── data_generator.py    # Geração de dados sintéticos
├── README.md            # Dependências
└── DATA/                # Datasets (Agr, Hyp, Rbf, Tree, Weather)
    └── Weather/         # Dataset multistream (Target, Source1, Source2, Source3)
```

## 2. Dependências

```
numpy==1.21.2
geatpy==2.7.0          # Biblioteca de algoritmos genéticos (NSGA-II)
scikit-multiflow==0.5.3
```

## 3. API do MCMO

### 3.1. Inicialização

```python
from MCMO import MCMO

model = MCMO(
    source_number=3,              # Número de source streams (N)
    initial_beach=200,            # Buffer size para inicialização
    max_pool=5,                   # Tamanho máximo do classifier pool
    gaussian_number=5,            # Número de componentes Gaussianas no GMM (k)
    base_classifier=HoeffdingTreeClassifier(),  # Classificador base (default)
    detector=DDM(min_num_instances=100)         # Detector de drift (default)
)
```

**Parâmetros:**
- `source_number`: Número de source streams com labels
- `initial_beach`: Quantidade de amostras do target para buffer inicial (warmup)
- `max_pool`: Tamanho máximo da pool de classificadores antigos (para drift)
- `gaussian_number`: Número k de Gaussianas no GMM
- `base_classifier`: Classificador base para cada source stream
- `detector`: Detector de drift (DDM por padrão)

### 3.2. Loop de Treinamento

```python
from skmultiflow.data.file_stream import FileStream

# Carregar streams
Tstream = FileStream("data/Weather/Target.csv")
S1stream = FileStream("data/Weather/Source1.csv")
S2stream = FileStream("data/Weather/Source2.csv")
S3stream = FileStream("data/Weather/Source3.csv")

sources = [S1stream, S2stream, S3stream]

# Loop prequential
while Tstream.has_more_samples():
    # 1. Treinar com source streams (labeled)
    for i in range(3):
        X_source, y_source = sources[i].next_sample()
        model.source_fit(X=X_source, y=y_source, order=i)

    # 2. Predizer no target stream (unlabeled)
    X_target, y_target_true = Tstream.next_sample()
    prediction = model.predict(X=X_target)

    # 3. Avaliar (opcional)
    # evaluate(y_target_true, prediction)

    # 4. Atualizar modelo com true label (depois da predição)
    model.partial_fit(X=X_target, y=y_target_true)
```

**Fluxo:**
1. **`source_fit(X, y, order)`**: Treina o classificador da source stream `order` com amostra `(X, y)`
2. **`predict(X)`**: Prediz label para amostra `X` do target stream
3. **`partial_fit(X, y)`**: Atualiza buffer do target com true label (para drift detection)

### 3.3. Métodos Principais

#### `model.source_fit(X, y, order)`
Treina o classificador da source stream especificada.

**Parâmetros:**
- `X`: Features (1D ou 2D array)
- `y`: Labels (1D array)
- `order`: Índice da source stream (0 a source_number-1)

**Comportamento:**
- **Fase 0 (op=0, inicialização)**: Acumula dados em `S_list` e `LS_list`
- **Fase 1 (op=1, operação)**:
  - Aplica feature selection
  - Calcula pesos GMM
  - Treina classificador com `partial_fit` e sample weights
  - Detecta drift com DDM
  - Se drift detectado: move classificador para pool e cria novo

#### `model.predict(X)`
Prediz label para amostra do target stream usando ensemble.

**Parâmetros:**
- `X`: Features (1D ou 2D array)

**Retorno:**
- Predição binária (0 ou 1)

**Comportamento:**
- Aplica feature selection
- Ensemble voting com pesos iguais:
  - Source classifiers (N classificadores atuais)
  - Classifier pool (até max_pool classificadores antigos)
- Threshold 0.5 para binarização

#### `model.partial_fit(X, y, classes=None, sample_weight=None)`
Atualiza buffer do target stream (para drift detection).

**Parâmetros:**
- `X`: Features do target (1D ou 2D array)
- `y`: True labels do target (1D array)

**Comportamento:**
- **Fase 0 (i < initial_beach)**: Acumula amostras em buffer `T`
- **Fase 1 (i == initial_beach)**: Chama `Model_initialization()`
  - Executa NSGA-II para feature selection
  - Treina GMM no target (features reduzidas)
  - Inicializa source classifiers com pesos GMM
  - Define drift threshold
- **Fase 2 (i > initial_beach)**: Drift detection
  - Calcula probability do target via GMM
  - Se probability média < threshold: reset (volta para Fase 0)

## 4. Implementação Interna

### 4.1. Feature Selection (NSGA-II)

**Arquivo:** `OptAlgorithm.py`

**Classe:** `MyProblem(ea.Problem)`

**Objetivos:**
1. **f1 (minimizar)**: MMD (Maximum Mean Discrepancy) - distância entre distribuições
   - Soma de `mmd_rbf(Source_i, Target)` para todas as sources
   - Kernel RBF: `gamma=1.0`

2. **f2 (minimizar)**: Fisher criterion (inverso da discriminative power)
   - `1 / trace(Sw^-1 * Sb)`
   - Maximiza separação entre classes

**Decisão fuzzy:** `fuzz_decision(Vars, ObjV)` seleciona solução do Pareto front com maior fuzzy score.

**Encoding:** Binário (BG) - cada bit indica se feature é selecionada.

**Parâmetros NSGA-II:**
- População: 50 indivíduos
- Gerações: 50
- Mutação: Pm = 0.2
- Crossover: XOVR = 0.9

### 4.2. GMM (Gaussian Mixture Model)

**Arquivo:** `GMM.py`

**Classe:** `DGMM(GaussianMixture)` - Dynamic GMM

**Extensões sobre sklearn:**

1. **`partial_fit(x, alpha)`**: Atualização incremental
   - Atualiza weights, means, covariances via exponential smoothing
   - `alpha`: taxa de aprendizado

2. **`evaluation_weight(x)`**: Calcula sample weights
   - `weight = -1 / log_prob(x) * 1000`
   - Maior peso para amostras com menor probabilidade no GMM (correção de covariate shift)

### 4.3. Ensemble e Drift Detection

**Ensemble:**
- Voting com pesos iguais entre:
  - N source classifiers (atuais)
  - Classifier pool (antigos, até max_pool)
- Threshold: 0.5 para classificação binária

**Drift Detection:**
- **Por source stream:** DDM individual
  - Monitora erro de predição
  - Se drift detectado: move classificador para pool, cria novo
- **No target stream:** Monitoramento de GMM probability
  - Threshold: `mean(prob) - 3*var(prob)` no initial_beach
  - Se `mean(prob) < threshold`: reset completo (volta para inicialização)

## 5. Datasets Disponíveis

**Diretório:** `DATA/`

- **Agr.csv/arff**: Synthetic dataset (paper)
- **Hyp.csv/arff**: Hyperplane synthetic dataset
- **Rbf.csv/arff**: RBF synthetic dataset
- **Tree.csv/arff**: Random tree synthetic dataset
- **Weather/**: Real-world multistream dataset
  - `Target.csv`: Target stream (unlabeled in use)
  - `Source1.csv`, `Source2.csv`, `Source3.csv`: Source streams (labeled)

## 6. Exemplo Completo (demo.py)

```python
from MCMO import MCMO
from skmultiflow.data.file_stream import FileStream
from evaluator import ClassificationMeasurements
import numpy as np

# 1. Inicializar modelo
model = MCMO(source_number=3, initial_beach=200, max_pool=5)

# 2. Carregar streams
Tstream = FileStream("data/Weather/Target.csv")
S1stream = FileStream("data/Weather/Source1.csv")
S2stream = FileStream("data/Weather/Source2.csv")
S3stream = FileStream("data/Weather/Source3.csv")
sources = [S1stream, S2stream, S3stream]

# 3. Loop prequential
data_size = 0
result_list = []
matrix = ClassificationMeasurements(dtype=np.float64)

while Tstream.has_more_samples():
    # Treinar sources
    for i in range(3):
        X, y = sources[i].next_sample()
        model.source_fit(X=X, y=y, order=i)

    # Predizer e avaliar target
    X, y = Tstream.next_sample()
    prediction = model.predict(X=X)
    matrix.add_result(y, int(prediction))

    # Atualizar target
    model.partial_fit(X=X, y=y)
    data_size += 1

    # Log a cada 100 amostras
    if data_size % 100 == 0:
        result_list.append(matrix.get_accuracy())
        matrix = ClassificationMeasurements(dtype=np.float64)
        print(f"Sample {data_size}: Accuracy = {result_list[-1]:.4f}")

print(f"Mean Accuracy: {np.mean(result_list):.4f}")
```

## 7. Adaptação para Single-Stream

**Problema:** MCMO espera N source streams + 1 target stream, mas nosso pipeline tem 1 stream único.

**Solução:** **Temporal Splitting** (do PLANO_INTEGRACAO_MCMO.md)

```python
class MCMOAdapter:
    def __init__(self, n_sources=3, window_size=1000):
        self.n_sources = n_sources
        self.window_size = window_size
        self.mcmo = MCMO(source_number=n_sources, initial_beach=200, max_pool=5)
        self.chunk_buffer = []  # Buffer de chunks passados

    def partial_fit_predict(self, X, y):
        """
        Converte single stream em multistream via temporal splitting.

        Args:
            X: Features do chunk atual (n_samples, n_features)
            y: Labels do chunk atual (n_samples,)

        Returns:
            predictions: Predições para o chunk atual
        """
        # 1. Adicionar chunk ao buffer
        self.chunk_buffer.append((X, y))

        # 2. Se não temos chunks suficientes, usar baseline simples
        if len(self.chunk_buffer) <= self.n_sources:
            # Predição dummy (retornar classe majoritária)
            return np.full(len(y), np.bincount(y).argmax())

        # 3. Criar multistream via temporal splitting
        # Sources: últimos n_sources chunks
        # Target: chunk atual
        sources = self.chunk_buffer[-(self.n_sources+1):-1]
        target_X, target_y_true = self.chunk_buffer[-1]

        # 4. Treinar MCMO com sources (chunks passados com labels)
        for i, (source_X, source_y) in enumerate(sources):
            for j in range(len(source_X)):
                self.mcmo.source_fit(
                    X=source_X[j:j+1],
                    y=source_y[j:j+1],
                    order=i
                )

        # 5. Predizer no target (chunk atual, simular unlabeled)
        predictions = []
        for j in range(len(target_X)):
            pred = self.mcmo.predict(X=target_X[j:j+1])
            predictions.append(pred[0])

            # 6. Atualizar com true label
            self.mcmo.partial_fit(X=target_X[j:j+1], y=target_y_true[j:j+1])

        # 7. Limpar buffer antigo (manter últimos 2*n_sources chunks)
        if len(self.chunk_buffer) > 2 * self.n_sources:
            self.chunk_buffer.pop(0)

        return np.array(predictions)
```

**Vantagens:**
- Mantém ordem temporal (chunks consecutivos)
- Simula covariate shift natural (chunks passados → chunk atual)
- Permite usar MCMO sem modificar código original

**Desvantagens:**
- Requer buffer de chunks (memória)
- Latência inicial (precisa esperar n_sources chunks)
- Possível overfitting se chunks muito similares

## 8. Próximos Passos

1. **Criar baseline_mcmo.py** com MCMOAdapter
2. **Testar isoladamente** em Electricity dataset
3. **Integrar no pipeline** principal (main.py)
4. **Avaliar** em Phase 3 datasets (5 datasets)
5. **Comparar** com outros baselines via Friedman + Wilcoxon

## 9. Observações Importantes

### 9.1. Limitações do Código Original

- **Classificação binária apenas:** `predict()` usa threshold 0.5
  - Para multiclasse: modificar `predict_proba()` e `predict()`
- **scikit-multiflow 0.5.3:** Versão antiga (descontinuada)
  - Substituir por `river` se necessário
- **geatpy 2.7.0:** Biblioteca específica de GA
  - Alternativa: `pymoo` para NSGA-II

### 9.2. Dependências de Versão

**Potenciais conflitos:**
- `scikit-multiflow==0.5.3` vs `river` (nosso pipeline)
- `numpy==1.21.2` pode ser incompatível com versões mais novas de sklearn

**Solução recomendada:**
- Testar MCMO em ambiente isolado (Colab) primeiro
- Se funcionar: criar ambiente conda específico localmente
- Se não funcionar: reimplementar componentes com bibliotecas atuais

### 9.3. Performance

**NSGA-II é lento:**
- 50 gerações × 50 indivíduos = 2500 evaluations
- Cada evaluation calcula MMD (O(n²)) e Fisher (O(d³))
- Executado a cada `initial_beach` amostras

**Otimizações possíveis:**
- Reduzir gerações/população
- Usar warm start (reutilizar população anterior)
- Cache de MMD para mesmas features

## 10. Checklist de Integração

- [x] Clonar repositório
- [x] Entender estrutura de código
- [x] Documentar API
- [ ] Criar adapter (baseline_mcmo.py)
- [ ] Testar isoladamente em Electricity
- [ ] Integrar em main.py
- [ ] Resolver dependências (scikit-multiflow vs river)
- [ ] Executar Phase 3 experiments
- [ ] Análise estatística (Friedman, Wilcoxon)
- [ ] Atualizar paper com resultados
