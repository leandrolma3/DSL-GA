"""
ROSE: Robust Online Self-Adjusting Ensemble
============================================
Implementacao Python baseada em:
Cano & Krawczyk (2022). "ROSE: robust online self-adjusting ensemble
for continual learning on imbalanced drifting data streams."
Machine Learning, 111(7), 2561-2599.

Autor da implementacao: Claude Code
Data: 2025-11-25
"""

import numpy as np
from collections import deque, defaultdict
from typing import List, Optional, Tuple, Dict, Any

# Flag para verificar disponibilidade
ROSE_AVAILABLE = False
IMPORT_ERROR = None

try:
    from river.tree import HoeffdingTreeClassifier
    from river.drift import ADWIN
    ROSE_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR = str(e)


class RiverTreeWrapper:
    """
    Wrapper para HoeffdingTree do River com interface compativel com ROSE.
    """

    def __init__(self, feature_indices: Optional[List[int]] = None):
        """
        Args:
            feature_indices: Indices das features a usar (para random subspace)
        """
        self.model = HoeffdingTreeClassifier(
            grace_period=100,
            split_confidence=1e-5,
            leaf_prediction='mc'
        )
        self.feature_indices = feature_indices
        self.classes_seen = set()

    def _to_dict(self, x: np.ndarray) -> dict:
        """Converte array numpy para dict (formato river)."""
        if self.feature_indices is not None:
            x = x[self.feature_indices]
        return {f'f{i}': float(v) for i, v in enumerate(x)}

    def partial_fit(self, x: np.ndarray, y: int):
        """Treina com uma instancia."""
        x_dict = self._to_dict(x)
        self.classes_seen.add(y)
        self.model.learn_one(x_dict, y)

    def predict(self, x: np.ndarray) -> int:
        """Prediz classe para uma instancia."""
        x_dict = self._to_dict(x)
        pred = self.model.predict_one(x_dict)
        return pred if pred is not None else 0

    def predict_proba(self, x: np.ndarray) -> Dict[int, float]:
        """Retorna probabilidades por classe."""
        x_dict = self._to_dict(x)
        proba = self.model.predict_proba_one(x_dict)
        return proba if proba else {0: 0.5, 1: 0.5}

    def clone(self) -> 'RiverTreeWrapper':
        """Cria uma copia do wrapper."""
        new_wrapper = RiverTreeWrapper(feature_indices=self.feature_indices)
        return new_wrapper


class ADWINDriftDetector:
    """
    Wrapper para ADWIN drift detector do River.
    """

    def __init__(self, delta: float = 0.002):
        """
        Args:
            delta: Confidence parameter (menor = menos sensivel)
        """
        self.detector = ADWIN(delta=delta)
        self.in_warning = False
        self.in_drift = False
        self._warning_delta = delta * 10  # Warning mais sensivel
        self._warning_detector = ADWIN(delta=self._warning_delta)

    def update(self, error: int) -> str:
        """
        Atualiza detector com erro (0=correto, 1=erro).

        Returns:
            'NORMAL', 'WARNING', ou 'DRIFT'
        """
        # Atualizar detector principal
        self.detector.update(error)
        self._warning_detector.update(error)

        # Verificar drift
        if self.detector.drift_detected:
            self.in_drift = True
            self.in_warning = False
            return 'DRIFT'

        # Verificar warning
        if self._warning_detector.drift_detected:
            self.in_warning = True
            return 'WARNING'

        return 'NORMAL'

    def reset(self):
        """Reseta o detector."""
        self.detector = ADWIN(delta=self.detector.delta)
        self._warning_detector = ADWIN(delta=self._warning_delta)
        self.in_warning = False
        self.in_drift = False


class ROSE:
    """
    ROSE: Robust Online Self-Adjusting Ensemble

    Caracteristicas principais:
    1. Online feature subspace sampling
    2. Drift detection com background ensemble
    3. Per-class sliding windows
    4. Self-adjusting bagging para class imbalance
    """

    def __init__(
        self,
        n_estimators: int = 10,
        window_size: int = 1000,
        feature_subset_size: str = 'sqrt',
        drift_delta: float = 0.002,
        random_state: Optional[int] = None
    ):
        """
        Args:
            n_estimators: Numero de classificadores no ensemble
            window_size: Tamanho da sliding window por classe
            feature_subset_size: 'sqrt', 'log2', ou int
            drift_delta: Parametro delta para ADWIN
            random_state: Seed para reproducibilidade
        """
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.feature_subset_size = feature_subset_size
        self.drift_delta = drift_delta
        self.random_state = random_state

        # Inicializar estado
        self.ensemble: List[RiverTreeWrapper] = []
        self.background_ensemble: List[Optional[RiverTreeWrapper]] = []
        self.drift_detectors: List[ADWINDriftDetector] = []
        self.feature_indices: List[List[int]] = []

        # Sliding windows por classe
        self.class_windows: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

        # Estatisticas
        self.class_counts: Dict[int, int] = defaultdict(int)
        self.n_features: Optional[int] = None
        self.classes_: List[int] = []
        self.n_samples_seen: int = 0

        # Random state
        if random_state is not None:
            np.random.seed(random_state)

    def _initialize_ensemble(self, n_features: int):
        """Inicializa ensemble na primeira instancia."""
        self.n_features = n_features

        # Calcular tamanho do subset de features
        if self.feature_subset_size == 'sqrt':
            subset_size = max(1, int(np.sqrt(n_features)))
        elif self.feature_subset_size == 'log2':
            subset_size = max(1, int(np.log2(n_features)))
        elif isinstance(self.feature_subset_size, int):
            subset_size = min(self.feature_subset_size, n_features)
        else:
            subset_size = n_features

        # Criar classifiers com subsets aleatorios de features
        for i in range(self.n_estimators):
            # Selecionar features aleatorias
            indices = np.random.choice(
                n_features, size=subset_size, replace=False
            ).tolist()
            self.feature_indices.append(indices)

            # Criar classifier
            clf = RiverTreeWrapper(feature_indices=indices)
            self.ensemble.append(clf)

            # Criar drift detector
            detector = ADWINDriftDetector(delta=self.drift_delta)
            self.drift_detectors.append(detector)

            # Background inicialmente vazio
            self.background_ensemble.append(None)

    def _compute_imbalance_ratio(self) -> float:
        """Calcula razao de imbalance atual."""
        if len(self.class_counts) < 2:
            return 1.0

        counts = list(self.class_counts.values())
        max_count = max(counts)
        min_count = min(counts) if min(counts) > 0 else 1

        return max_count / min_count

    def _get_minority_class(self) -> Optional[int]:
        """Retorna a classe minoritaria atual."""
        if len(self.class_counts) < 2:
            return None

        return min(self.class_counts.keys(), key=lambda c: self.class_counts[c])

    def _compute_lambda(self, y: int, ir: float) -> int:
        """
        Calcula peso lambda para self-adjusting bagging.

        Args:
            y: Classe da instancia
            ir: Imbalance ratio atual

        Returns:
            Peso (numero de vezes para treinar)
        """
        # Base lambda (Poisson com media 1)
        base_lambda = np.random.poisson(1)

        # Ajuste para classe minoritaria
        minority_class = self._get_minority_class()
        if minority_class is not None and y == minority_class:
            # Aumentar peso proporcional a sqrt(IR)
            adjusted_lambda = int(base_lambda * np.sqrt(ir))
            return max(1, adjusted_lambda)

        return max(1, base_lambda)

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina o ensemble com um batch de instancias.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        # Inicializar na primeira chamada
        if self.n_features is None:
            self._initialize_ensemble(X.shape[1])

        # Processar cada instancia
        for x_i, y_i in zip(X, y):
            self._partial_fit_single(x_i, int(y_i))

    def _partial_fit_single(self, x: np.ndarray, y: int):
        """Treina com uma unica instancia."""
        # Atualizar estatisticas
        self.n_samples_seen += 1
        self.class_counts[y] += 1
        if y not in self.classes_:
            self.classes_.append(y)

        # Atualizar sliding window da classe
        self.class_windows[y].append(x.copy())

        # Calcular imbalance ratio
        ir = self._compute_imbalance_ratio()

        # Calcular peso lambda
        lambda_weight = self._compute_lambda(y, ir)

        # Treinar cada classifier
        for i in range(self.n_estimators):
            clf = self.ensemble[i]
            detector = self.drift_detectors[i]

            # Predizer antes de treinar (para drift detection)
            pred = clf.predict(x)
            error = int(pred != y)

            # Treinar lambda vezes
            for _ in range(lambda_weight):
                clf.partial_fit(x, y)

            # Atualizar drift detector
            drift_status = detector.update(error)

            # Gerenciar background ensemble
            if drift_status == 'WARNING':
                # Iniciar background learner se ainda nao existe
                if self.background_ensemble[i] is None:
                    self.background_ensemble[i] = clf.clone()
                # Treinar background
                for _ in range(lambda_weight):
                    self.background_ensemble[i].partial_fit(x, y)

            elif drift_status == 'DRIFT':
                # Substituir por background
                if self.background_ensemble[i] is not None:
                    self.ensemble[i] = self.background_ensemble[i]
                    self.background_ensemble[i] = None
                else:
                    # Criar novo classifier
                    self.ensemble[i] = RiverTreeWrapper(
                        feature_indices=self.feature_indices[i]
                    )
                # Reset detector
                detector.reset()

            elif drift_status == 'NORMAL' and detector.in_warning:
                # Warning cancelado, descartar background
                self.background_ensemble[i] = None
                detector.in_warning = False

            # Treinar background se existir
            if self.background_ensemble[i] is not None:
                for _ in range(lambda_weight):
                    self.background_ensemble[i].partial_fit(x, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz classes para um batch de instancias.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicoes (n_samples,)
        """
        X = np.atleast_2d(X)
        predictions = []

        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)

        return np.array(predictions)

    def _predict_single(self, x: np.ndarray) -> int:
        """Prediz para uma unica instancia usando majority voting."""
        if len(self.ensemble) == 0:
            return 0

        # Coletar votos
        votes = defaultdict(float)

        for clf in self.ensemble:
            # Obter probabilidades
            proba = clf.predict_proba(x)

            # Adicionar votos ponderados
            for cls, prob in proba.items():
                votes[cls] += prob

        # Retornar classe com mais votos
        if len(votes) == 0:
            return 0

        return max(votes.keys(), key=lambda c: votes[c])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz probabilidades para um batch.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probabilidades (n_samples, n_classes)
        """
        X = np.atleast_2d(X)
        n_classes = max(2, len(self.classes_))
        probas = np.zeros((len(X), n_classes))

        for idx, x in enumerate(X):
            votes = defaultdict(float)

            for clf in self.ensemble:
                proba = clf.predict_proba(x)
                for cls, prob in proba.items():
                    if cls < n_classes:
                        votes[cls] += prob

            # Normalizar
            total = sum(votes.values())
            if total > 0:
                for cls, prob in votes.items():
                    if cls < n_classes:
                        probas[idx, cls] = prob / total
            else:
                probas[idx, :] = 1.0 / n_classes

        return probas

    def get_info(self) -> dict:
        """Retorna informacoes sobre o estado do ensemble."""
        return {
            'n_estimators': self.n_estimators,
            'n_samples_seen': self.n_samples_seen,
            'n_features': self.n_features,
            'classes': self.classes_,
            'class_counts': dict(self.class_counts),
            'imbalance_ratio': self._compute_imbalance_ratio(),
            'minority_class': self._get_minority_class(),
            'active_backgrounds': sum(
                1 for bg in self.background_ensemble if bg is not None
            )
        }


class ROSEAdapter:
    """
    Adapter para usar ROSE com avaliacao chunk-based.
    Compativel com o protocolo train-then-test do GBML.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        window_size: int = 1000,
        feature_subset_size: str = 'sqrt',
        drift_delta: float = 0.002,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Args:
            n_estimators: Numero de classificadores
            window_size: Tamanho da sliding window
            feature_subset_size: Tamanho do subset de features
            drift_delta: Parametro para drift detection
            random_state: Seed para reproducibilidade
            verbose: Imprimir informacoes de progresso
        """
        self.rose = ROSE(
            n_estimators=n_estimators,
            window_size=window_size,
            feature_subset_size=feature_subset_size,
            drift_delta=drift_delta,
            random_state=random_state
        )
        self.verbose = verbose
        self.chunk_metrics: List[dict] = []

    def partial_fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Protocolo test-then-train para um chunk.

        Args:
            X: Features do chunk
            y: Labels do chunk

        Returns:
            Predicoes para o chunk
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        # Test first
        if self.rose.n_samples_seen > 0:
            predictions = self.rose.predict(X)
        else:
            # Primeira iteracao: sem predicao
            predictions = np.zeros(len(y), dtype=int)

        # Then train
        self.rose.partial_fit(X, y)

        # Calcular metricas do chunk
        if self.verbose:
            accuracy = np.mean(predictions == y)
            info = self.rose.get_info()
            print(f"Chunk processed: acc={accuracy:.4f}, "
                  f"samples={info['n_samples_seen']}, "
                  f"IR={info['imbalance_ratio']:.2f}")

        return predictions

    def evaluate_stream(
        self,
        X_chunks: List[np.ndarray],
        y_chunks: List[np.ndarray]
    ) -> dict:
        """
        Avalia ROSE em uma sequencia de chunks.

        Args:
            X_chunks: Lista de chunks de features
            y_chunks: Lista de chunks de labels

        Returns:
            Dicionario com metricas
        """
        all_predictions = []
        all_labels = []
        chunk_accuracies = []

        for i, (X, y) in enumerate(zip(X_chunks, y_chunks)):
            preds = self.partial_fit_predict(X, y)

            all_predictions.extend(preds)
            all_labels.extend(y)

            # Metricas do chunk
            acc = np.mean(preds == y)
            chunk_accuracies.append(acc)

            if self.verbose:
                print(f"Chunk {i+1}/{len(X_chunks)}: Accuracy = {acc:.4f}")

        # Metricas globais
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        global_accuracy = np.mean(all_predictions == all_labels)

        # G-mean
        from sklearn.metrics import confusion_matrix
        classes = np.unique(all_labels)
        if len(classes) == 2:
            cm = confusion_matrix(all_labels, all_predictions)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            gmean = np.sqrt(sensitivity * specificity)
        else:
            # Multi-class G-mean
            recalls = []
            for c in classes:
                mask = all_labels == c
                if mask.sum() > 0:
                    recall = np.mean(all_predictions[mask] == c)
                    recalls.append(recall)
            gmean = np.prod(recalls) ** (1.0 / len(recalls)) if recalls else 0

        return {
            'global_accuracy': global_accuracy,
            'global_gmean': gmean,
            'chunk_accuracies': chunk_accuracies,
            'mean_chunk_accuracy': np.mean(chunk_accuracies),
            'std_chunk_accuracy': np.std(chunk_accuracies),
            'n_chunks': len(X_chunks),
            'total_samples': len(all_labels),
            'rose_info': self.rose.get_info()
        }


def test_rose_adapter(
    X_chunks: List[np.ndarray],
    y_chunks: List[np.ndarray],
    n_estimators: int = 10,
    verbose: bool = True
) -> dict:
    """
    Funcao de teste para ROSEAdapter.

    Args:
        X_chunks: Lista de chunks de features
        y_chunks: Lista de chunks de labels
        n_estimators: Numero de estimadores
        verbose: Imprimir progresso

    Returns:
        Dicionario com resultados
    """
    adapter = ROSEAdapter(
        n_estimators=n_estimators,
        verbose=verbose
    )

    results = adapter.evaluate_stream(X_chunks, y_chunks)

    if verbose:
        print("\n" + "="*60)
        print("RESULTADOS FINAIS - ROSE")
        print("="*60)
        print(f"Accuracy Global: {results['global_accuracy']:.4f}")
        print(f"G-mean Global:   {results['global_gmean']:.4f}")
        print(f"Accuracy Media:  {results['mean_chunk_accuracy']:.4f}")
        print(f"Total Chunks:    {results['n_chunks']}")
        print(f"Total Samples:   {results['total_samples']}")

    return results


# Verificar disponibilidade ao importar
if not ROSE_AVAILABLE:
    print(f"[AVISO] ROSE nao disponivel: {IMPORT_ERROR}")
    print("Instale as dependencias: pip install river scikit-learn")
