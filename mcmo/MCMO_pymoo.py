"""
MCMO with pymoo - Google Colab Compatible

Uses pymoo for NSGA-II instead of geatpy.
Full NSGA-II implementation (not simplified correlation).

Author: Claude Code
Date: 2025-11-24
"""

import copy as cp
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from river.tree import HoeffdingTreeClassifier
from river.drift import ADWIN
from sklearn.mixture import GaussianMixture
from sklearn import metrics


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
            # River doesn't support sample_weight, but we accept it for compatibility
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


def Feature_Reduce(X, individual):
    """Reduce features based on binary mask."""
    if X.ndim == 1:
        X = X.reshape(1, -1)

    D = X.shape[1]
    f = int(np.sum(individual))

    if f == 0:
        return X[:, :1]

    Trans = np.zeros((D, f))
    c = 0
    for i in range(len(individual)):
        if individual[i] == 1:
            Trans[i][c] = 1
            c = c + 1

    Tx = np.dot(X, Trans)
    return Tx


def mmd_rbf(X, Y, gamma=1.0):
    """Maximum Mean Discrepancy with RBF kernel."""
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def F1_score(X, y):
    """Fisher criterion for discriminative power."""
    labels = list(set(y))
    if len(labels) < 2:
        return 10000.0  # Penalize if only one class

    xClasses = {}
    for label in labels:
        xClasses[label] = np.array([X[i] for i in range(len(X)) if y[i] == label])

    meanAll = np.mean(X, axis=0)
    meanClasses = {}
    for label in labels:
        if len(xClasses[label]) > 0:
            meanClasses[label] = np.mean(xClasses[label], axis=0)
        else:
            meanClasses[label] = meanAll

    St = np.dot((X - meanAll).T, X - meanAll)
    Sw = np.zeros((len(meanAll), len(meanAll)))
    for i in labels:
        if len(xClasses[i]) > 0:
            Sw += np.dot((xClasses[i] - meanClasses[i]).T, (xClasses[i] - meanClasses[i]))

    Sb = St - Sw
    try:
        d = np.dot(np.linalg.inv(Sw), Sb)
    except:
        try:
            d = np.dot(np.linalg.pinv(Sw), Sb)
        except:
            return 10000.0

    trace_val = np.trace(d)
    if trace_val < 1e-10:
        return 10000.0

    return 1.0 / (trace_val + 1e-10)


class FeatureSelectionProblem(ElementwiseProblem):
    """
    Multi-objective feature selection problem for pymoo.

    Objectives:
    1. Minimize MMD (distribution discrepancy between sources and target)
    2. Minimize Fisher criterion (maximize discriminative power)
    """

    def __init__(self, n_features, T, S_list, LS_list):
        self.T = T
        self.S_list = [np.array(s) for s in S_list if len(s) > 0]
        self.LS_list = [np.array(ls) for ls in LS_list if len(ls) > 0]
        self.n_features = n_features

        super().__init__(
            n_var=n_features,
            n_obj=2,
            n_constr=0,
            xl=0,
            xu=1,
            vtype=int
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate objectives for a feature subset.

        x: binary vector (1 = feature selected, 0 = not selected)
        """
        # Check if at least one feature is selected
        if np.sum(x) == 0:
            out["F"] = [10000.0, 10000.0]
            return

        # Objective 1: MMD (minimize distribution discrepancy)
        mmd_total = 0.0
        for i in range(len(self.S_list)):
            try:
                SX = Feature_Reduce(self.S_list[i], x)
                TX = Feature_Reduce(self.T, x)
                mmd_total += mmd_rbf(SX, TX)
            except:
                mmd_total += 10000.0

        # Objective 2: Fisher criterion (minimize for maximizing discriminative power)
        f1_total = 0.0
        for i in range(len(self.S_list)):
            try:
                SX = Feature_Reduce(self.S_list[i], x)
                f1_total += F1_score(SX, self.LS_list[i])
            except:
                f1_total += 10000.0

        out["F"] = [mmd_total, f1_total]


def fuzz_decision(pop_X, pop_F):
    """
    Select best solution from Pareto front using fuzzy decision.
    Similar to geatpy's approach.
    """
    if len(pop_X) == 0:
        return None

    if len(pop_X) == 1:
        return pop_X[0]

    # Normalize objectives
    max_objv = np.max(pop_F, axis=0)
    min_objv = np.min(pop_F, axis=0)

    # Avoid division by zero
    x = max_objv - min_objv
    x = np.where(x < 1e-10, 1.0, x)

    y = -1 * (pop_F - max_objv)
    i = y / x
    j = np.sum(i, axis=1) / np.sum(i)

    index = np.argmax(j)
    return pop_X[index]


class MCMO_Pymoo:
    """
    MCMO using pymoo for NSGA-II.

    Full implementation with multi-objective feature selection.
    """

    def __init__(self, base_classifier=None, detector=None, source_number=1,
                 initial_beach=100, max_pool=5, gaussian_number=5,
                 nsga_pop_size=50, nsga_n_gen=50):
        self.base_classifier = base_classifier if base_classifier else RiverTreeWrapper()
        self.detector = detector if detector else RiverADWINWrapper(min_num_instances=100)

        self.source_number = source_number
        self.initial_beach = initial_beach
        self.max_pool = max_pool
        self.gaussian_number = gaussian_number
        self.nsga_pop_size = nsga_pop_size
        self.nsga_n_gen = nsga_n_gen

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
        """Initialize model with NSGA-II feature selection using pymoo."""

        # Create problem
        problem = FeatureSelectionProblem(
            n_features=self.D,
            T=self.T,
            S_list=self.S_list,
            LS_list=self.LS_list
        )

        # Configure NSGA-II
        algorithm = NSGA2(
            pop_size=self.nsga_pop_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True
        )

        # Run optimization
        res = minimize(
            problem,
            algorithm,
            ('n_gen', self.nsga_n_gen),
            verbose=False,
            seed=42
        )

        # Select best solution from Pareto front using fuzzy decision
        if res.X is not None:
            if res.X.ndim == 1:
                # Single solution
                self.solution = res.X
            else:
                # Multiple solutions (Pareto front)
                self.solution = fuzz_decision(res.X, res.F)
        else:
            # Fallback: select all features
            self.solution = np.ones(self.D, dtype=int)

        # Ensure at least one feature is selected
        if np.sum(self.solution) == 0:
            self.solution[0] = 1

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
MCMO = MCMO_Pymoo
