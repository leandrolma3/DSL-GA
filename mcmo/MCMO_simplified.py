"""
MCMO Simplified - Colab Compatible

Version without geatpy dependency.
Uses correlation-based feature selection instead of NSGA-II.

Author: Claude Code
Date: 2025-11-24
"""

import copy as cp
import numpy as np
from scipy.stats import pearsonr, spearmanr
from river.tree import HoeffdingTreeClassifier
from river.drift import ADWIN
from sklearn.mixture import GaussianMixture


class RiverTreeWrapper:
    """Wrapper for river HoeffdingTreeClassifier."""

    def __init__(self):
        self.model = HoeffdingTreeClassifier()
        self.n_samples_seen = 0

    def partial_fit(self, X, y, sample_weight=None):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = np.array([y])

        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            # Note: river HoeffdingTree doesn't support sample_weight in learn_one
            # We accept it for API compatibility but don't use it
            self.model.learn_one(x_dict, int(y[i]))
            self.n_samples_seen += 1

        return self

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            pred = self.model.predict_one(x_dict)
            predictions.append(pred if pred is not None else 0)

        return np.array(predictions)


class RiverADWINWrapper:
    """Wrapper for river ADWIN drift detector."""

    def __init__(self, min_num_instances=100):
        self.detector = ADWIN()
        self.n_samples = 0
        self.min_num_instances = min_num_instances
        self.drift_detected = False

    def add_element(self, error):
        self.n_samples += 1
        self.detector.update(error)

        if self.n_samples >= self.min_num_instances:
            self.drift_detected = self.detector.drift_detected

    def detected_change(self):
        result = self.drift_detected
        if result:
            self.drift_detected = False
        return result


class DGMM(GaussianMixture):
    """Dynamic GMM with evaluation weights."""

    def __init__(self, n_components=5, random_state=None, **kwargs):
        super().__init__(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state,
            **kwargs
        )

    def evaluation_weight(self, x):
        """Compute sample weights based on GMM probability."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        pro = -1. / self._estimate_log_prob(x)
        weight = np.zeros(len(pro))
        for i in range(len(pro)):
            weight[i] = max(pro[i])
        return weight * 1000


def correlation_feature_selection(sources, target, LS_list, n_features_select=None, method='pearson'):
    """
    Simplified feature selection using correlation.

    Replaces NSGA-II optimization with correlation-based filter.
    Selects features with highest correlation to target variable.
    """
    n_features = sources[0].shape[1] if len(sources) > 0 else target.shape[1]

    if n_features_select is None:
        n_features_select = max(1, int(n_features * 0.7))

    # Combine all source data
    if len(sources) > 0:
        X_all = np.vstack(sources)
        y_all = np.hstack(LS_list)
    else:
        # Fallback if no sources yet
        return np.ones(n_features, dtype=int)

    # Compute correlation for each feature
    correlations = []
    for f in range(n_features):
        try:
            if method == 'pearson':
                corr, _ = pearsonr(X_all[:, f], y_all)
            else:
                corr, _ = spearmanr(X_all[:, f], y_all)
            correlations.append(abs(corr))
        except:
            correlations.append(0.0)

    # Select top features
    selected_idx = np.argsort(correlations)[-n_features_select:]

    # Create binary mask
    solution = np.zeros(n_features, dtype=int)
    solution[selected_idx] = 1

    # Ensure at least one feature is selected
    if solution.sum() == 0:
        solution[0] = 1

    return solution


def Feature_Reduce(X, individual):
    """Reduce features based on binary mask."""
    if X.ndim == 1:
        X = X.reshape(1, -1)

    D = X.shape[1]
    f = np.sum(individual)

    if f == 0:
        return X[:, :1]  # Return first feature if none selected

    Trans = np.zeros((D, f))
    c = 0
    for i in range(len(individual)):
        if individual[i] == 1:
            Trans[i][c] = 1
            c = c + 1

    Tx = np.dot(X, Trans)
    return Tx


class MCMO_Simplified:
    """
    Simplified MCMO without geatpy dependency.

    Uses correlation-based feature selection instead of NSGA-II.
    Compatible with Google Colab.
    """

    def __init__(self, base_classifier=None, detector=None, source_number=1,
                 initial_beach=100, max_pool=5, gaussian_number=5,
                 feature_selection_ratio=0.7):
        self.base_classifier = base_classifier if base_classifier else RiverTreeWrapper()
        self.detector = detector if detector else RiverADWINWrapper(min_num_instances=100)

        self.source_number = source_number
        self.initial_beach = initial_beach
        self.max_pool = max_pool
        self.gaussian_number = gaussian_number
        self.feature_selection_ratio = feature_selection_ratio

        self.source_classifiers = []
        self.classifier_pool = []
        self.drift_detectors = []
        self.probability_set = []

        self.S_list = []
        self.LS_list = []

        for i in range(self.source_number):
            self.S_list.append([])
            self.LS_list.append([])

        self.i = -1
        self.op = 0
        self.solution = None

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if self.i == -1:
            N, D = X.shape
            self.D = D
            self.T = np.zeros((self.initial_beach, self.D))
            self.i = 0

        if self.i < self.initial_beach:
            for i in range(len(X)):
                self.T[self.i] = X[i]
                self.i = self.i + 1
        else:
            if self.solution is None:
                return

            Tx = Feature_Reduce(X, self.solution)
            probability = self.gmm.evaluation_weight(Tx)
            self.i = self.i + len(probability)

            for i in range(len(probability)):
                self.probability_set.pop(0)
                self.probability_set.append(probability[i])
                mean = np.mean(self.probability_set)

                if mean < self.drift_threshold:
                    self.op = 0
                    self.i = 0
                    self.S_list = []
                    self.LS_list = []
                    for j in range(self.source_number):
                        self.S_list.append([])
                        self.LS_list.append([])

        if self.i == self.initial_beach:
            self.Model_initialization()
            self.op = 1

    def source_fit(self, X, y, order):
        if self.op == 0:
            for i in range(len(X)):
                self.S_list[order].append(X[i])
                self.LS_list[order].append(y[i])
        else:
            if self.solution is None:
                return

            Tx = Feature_Reduce(X, self.solution)
            weight = self.gmm.evaluation_weight(Tx)
            pre = self.source_classifiers[order].predict(Tx)

            for i in range(len(pre)):
                if pre[i] == y[i]:
                    self.drift_detectors[order].add_element(0)
                else:
                    self.drift_detectors[order].add_element(1)

                if self.drift_detectors[order].detected_change():
                    self.classifier_pool.append(self.source_classifiers[order])
                    self.drift_detectors[order] = cp.deepcopy(self.detector)
                    self.source_classifiers[order] = cp.deepcopy(self.base_classifier)
                    if len(self.classifier_pool) > self.max_pool:
                        self.classifier_pool.pop(0)

            self.source_classifiers[order].partial_fit(X=Tx, y=y, sample_weight=weight)

    def Model_initialization(self):
        """Initialize model with correlation-based feature selection."""
        # Convert lists to arrays
        S_arrays = [np.array(s) for s in self.S_list if len(s) > 0]
        LS_arrays = [np.array(ls) for ls in self.LS_list if len(ls) > 0]

        # Feature selection using correlation
        n_select = max(1, int(self.D * self.feature_selection_ratio))
        self.solution = correlation_feature_selection(
            S_arrays, self.T, LS_arrays, n_features_select=n_select
        )

        # Reduce target features
        Tx = Feature_Reduce(self.T, self.solution)

        # Train GMM on reduced target
        self.gmm = DGMM(n_components=self.gaussian_number, random_state=0).fit(Tx)

        # Compute drift threshold
        initial_confidence = self.gmm.evaluation_weight(Tx)
        self.drift_threshold = np.mean(initial_confidence) - 3 * np.var(initial_confidence)
        self.probability_set = list(initial_confidence)

        # Initialize source classifiers
        self.source_classifiers = []
        self.classifier_pool = []
        self.drift_detectors = []

        for i in range(self.source_number):
            if len(self.S_list[i]) > 0:
                Sx = Feature_Reduce(np.array(self.S_list[i]), self.solution)
                weight = self.gmm.evaluation_weight(Sx)
                base_classifier = cp.deepcopy(self.base_classifier)
                base_classifier.partial_fit(X=Sx, y=np.array(self.LS_list[i]), sample_weight=weight)
                self.source_classifiers.append(base_classifier)
                self.drift_detectors.append(cp.deepcopy(self.detector))

    def predict_proba(self, X):
        N, D = X.shape
        votes = np.zeros(N)

        if len(self.source_classifiers) <= 0 or self.solution is None:
            return votes

        Tx = Feature_Reduce(X, self.solution)

        if len(self.classifier_pool) == 0:
            for h_i in self.source_classifiers:
                votes = votes + 1. / len(self.source_classifiers) * h_i.predict(Tx)
        else:
            total_classifiers = len(self.source_classifiers) + len(self.classifier_pool)
            for h_i in self.source_classifiers:
                votes = votes + 1. / total_classifiers * h_i.predict(Tx)
            for h_i in self.classifier_pool:
                votes = votes + 1. / total_classifiers * h_i.predict(Tx)

        return votes

    def predict(self, X):
        votes = self.predict_proba(X)
        return (votes >= 0.5) * 1.


# Alias for compatibility
MCMO = MCMO_Simplified
