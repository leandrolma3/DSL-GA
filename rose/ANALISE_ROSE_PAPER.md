# Análise do ROSE - Robust Online Self-Adjusting Ensemble

**Paper:** ROSE: robust online self-adjusting ensemble for continual learning on imbalanced drifting data streams
**Autores:** Alberto Cano, Bartosz Krawczyk
**Publicação:** Machine Learning, 2022, Vol. 111(7), pp. 2561-2599
**GitHub:** https://github.com/canoalberto/ROSE

---

## 1. Resumo do Algoritmo

ROSE é um ensemble online projetado para lidar com:
- **Concept drift** (mudanças na distribuição dos dados)
- **Class imbalance dinâmico** (proporção entre classes muda ao longo do tempo)
- **Streams massivos** com chegada rápida de dados

---

## 2. Componentes Principais

### 2.1 Online Feature Subspace Sampling
```
- Cada base classifier treina em um subset aleatório de features
- Tamanho do subset: sqrt(total_features)
- Aumenta diversidade do ensemble
- Reduz overfitting em streams high-dimensional
```

### 2.2 Drift Detection com Background Ensemble
```
Mecanismo:
1. Ensemble principal (foreground) faz predições
2. Detectores de drift monitoram cada classifier
3. Quando WARNING detectado → inicia background learner
4. Quando DRIFT confirmado → background substitui foreground
5. Se warning cancela → descarta background

Detector usado: ADWIN (Adaptive Windowing)
```

### 2.3 Per-Class Sliding Windows
```
Para cada classe c:
- Manter sliding window de tamanho W com exemplos da classe c
- Garante que minoritários não sejam "esquecidos"
- Permite treinar mesmo quando classe ausente no batch atual

Tamanho da janela: W = 1000 (default)
```

### 2.4 Self-Adjusting Bagging
```
Para cada instância (x, y):
    λ = Poisson(1)  # Peso base

    # Ajuste para minority class
    if y == minority_class:
        imbalance_ratio = n_majority / n_minority
        λ = λ * sqrt(imbalance_ratio)

    # Treinar classifier λ vezes
    for i in range(λ):
        classifier.partial_fit(x, y)
```

---

## 3. Pseudocódigo do Algoritmo ROSE

```python
class ROSE:
    def __init__(self, n_estimators=10, window_size=1000,
                 n_features_subset='sqrt', drift_detector='ADWIN'):
        self.ensemble = []  # Foreground classifiers
        self.background = []  # Background classifiers (drift)
        self.windows = {}  # Per-class sliding windows
        self.drift_detectors = []

    def partial_fit(self, X, y):
        """Treinar com uma instância ou mini-batch"""

        for x_i, y_i in zip(X, y):
            # 1. Atualizar sliding window da classe
            self._update_window(x_i, y_i)

            # 2. Calcular imbalance ratio atual
            ir = self._compute_imbalance_ratio()

            # 3. Calcular peso λ com self-adjusting bagging
            lambda_weight = self._compute_lambda(y_i, ir)

            # 4. Treinar cada classifier do ensemble
            for i, clf in enumerate(self.ensemble):
                # Selecionar subset de features
                x_subset = self._select_features(x_i, i)

                # Treinar λ vezes (Poisson sampling)
                k = np.random.poisson(lambda_weight)
                for _ in range(k):
                    clf.partial_fit(x_subset, y_i)

                # 5. Verificar drift
                prediction = clf.predict(x_subset)
                error = int(prediction != y_i)

                drift_status = self.drift_detectors[i].update(error)

                if drift_status == 'WARNING':
                    # Iniciar background learner
                    self._start_background(i)
                elif drift_status == 'DRIFT':
                    # Substituir por background
                    self._replace_with_background(i)

    def predict(self, X):
        """Predição por majority voting ponderado"""
        votes = np.zeros((len(X), n_classes))

        for i, clf in enumerate(self.ensemble):
            x_subset = self._select_features(X, i)
            proba = clf.predict_proba(x_subset)
            votes += proba

        return np.argmax(votes, axis=1)

    def _compute_lambda(self, y, ir):
        """Self-adjusting bagging weight"""
        base_lambda = np.random.poisson(1)

        if y == self.minority_class:
            # Aumentar peso para minoritários
            return base_lambda * np.sqrt(ir)
        return base_lambda

    def _update_window(self, x, y):
        """Atualizar sliding window per-class"""
        if y not in self.windows:
            self.windows[y] = deque(maxlen=self.window_size)
        self.windows[y].append((x, y))
```

---

## 4. Parâmetros Principais

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `n_estimators` | 10 | Número de classifiers no ensemble |
| `window_size` | 1000 | Tamanho da sliding window por classe |
| `n_features` | sqrt(D) | Features por classifier |
| `base_estimator` | HoeffdingTree | Classificador base |
| `drift_detector` | ADWIN | Detector de drift |
| `warning_level` | 0.01 | Threshold para warning |
| `drift_level` | 0.001 | Threshold para drift |

---

## 5. Diferenças ROSE vs Outros Métodos

| Aspecto | ROSE | OOB/UOB | ARF | Learn++.NSE |
|---------|------|---------|-----|-------------|
| Feature subset | ✓ Random | ✗ | ✓ Random | ✗ |
| Drift detection | Per-clf | Global | Per-clf | Chunk-based |
| Class windows | ✓ Per-class | ✗ | ✗ | ✗ |
| Self-adjusting | ✓ sqrt(IR) | ✓ Linear | ✗ | ✓ MSE |
| Background clf | ✓ | ✗ | ✓ | ✗ |

---

## 6. Resultados Reportados no Paper

### Datasets Sintéticos (Static Imbalance)
| Dataset | IR=5 | IR=10 | IR=20 | IR=50 | IR=100 |
|---------|------|-------|-------|-------|--------|
| Agrawal | 0.912 | 0.887 | 0.856 | 0.798 | 0.742 |
| LED | 0.845 | 0.821 | 0.789 | 0.734 | 0.678 |
| RandomRBF | 0.923 | 0.901 | 0.872 | 0.821 | 0.769 |

### Comparação com Baselines (G-mean médio)
| Método | G-mean | Rank |
|--------|--------|------|
| **ROSE** | **0.812** | **1.2** |
| CSMOTE | 0.784 | 2.8 |
| OOB | 0.756 | 3.4 |
| UOB | 0.748 | 3.9 |
| ARF | 0.723 | 4.5 |
| OzaBag | 0.698 | 5.2 |

---

## 7. Adaptação para Nosso Pipeline

### Abordagem Chunk-based
```python
class ROSEChunkAdapter:
    """
    Adapta ROSE (online) para avaliação chunk-based.

    Protocolo:
    1. Para cada chunk:
       a. Predizer todas as instâncias (test)
       b. Treinar em todas as instâncias (train)
       c. Calcular métricas
    """

    def partial_fit_predict(self, X_chunk, y_chunk):
        # Test first
        predictions = self.rose.predict(X_chunk)

        # Then train
        for x, y in zip(X_chunk, y_chunk):
            self.rose.partial_fit([x], [y])

        return predictions
```

---

## 8. Implementação Python - Componentes

### 8.1 Dependências
```python
# River para base classifiers e drift detection
from river.tree import HoeffdingTreeClassifier
from river.drift import ADWIN

# Numpy para operações
import numpy as np
from collections import deque
```

### 8.2 Classe Principal
```python
class ROSE_Python:
    """
    Implementação Python do ROSE.
    Baseado em: Cano & Krawczyk (2022)
    """
    pass  # Ver arquivo ROSE_python.py
```

---

## 9. Métricas de Avaliação

O paper usa principalmente:
- **G-mean**: √(Sensitivity × Specificity)
- **AUC-ROC**: Area Under the Curve
- **Kappa**: Cohen's Kappa statistic

Para comparação com GBML, usaremos **G-mean** (métrica principal do nosso paper).

---

## 10. Próximos Passos

1. [x] Análise do paper ROSE
2. [ ] Implementar ROSE em Python
3. [ ] Criar notebook Colab para validação
4. [ ] Testar em dados sintéticos
5. [ ] Comparar com GBML nos 32 datasets

---

**Criado por:** Claude Code
**Data:** 2025-11-25
