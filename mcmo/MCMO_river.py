"""
MCMO with river compatibility

Modified version of MCMO.py to use river instead of scikit-multiflow.
"""

import copy as cp
import geatpy as ea
import numpy as np
from river.tree import HoeffdingTreeClassifier
from river.drift import ADWIN
from OptAlgorithm import MyProblem, Feature_Reduce
from GMM import DGMM


def fuzz_decision(Vars, ObjV):
    max_objv = np.array([np.max(ObjV[:, 0]), np.max(ObjV[:, 1])])
    min_objv = np.array([np.min(ObjV[:, 0]), np.min(ObjV[:, 1])])
    x = max_objv - min_objv
    y = -1*(ObjV - max_objv)
    i = y/x
    j = np.sum(i, axis=1)/np.sum(i)
    index = np.argmax(j)
    return Vars[index]


class RiverTreeWrapper:
    """Wrapper para HoeffdingTreeClassifier do river para interface compatível."""

    def __init__(self):
        self.model = HoeffdingTreeClassifier()
        self.n_samples_seen = 0

    def partial_fit(self, X, y, sample_weight=None):
        """Interface compatível com scikit-multiflow."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = np.array([y])

        # River usa dicionários
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            w = sample_weight[i] if sample_weight is not None else 1.0
            self.model.learn_one(x_dict, int(y[i]), sample_weight=w)
            self.n_samples_seen += 1

        return self

    def predict(self, X):
        """Interface compatível com scikit-multiflow."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            pred = self.model.predict_one(x_dict)
            # Se ainda não treinado, retorna 0
            predictions.append(pred if pred is not None else 0)

        return np.array(predictions)


class RiverADWINWrapper:
    """Wrapper para ADWIN do river para interface compatível com DDM."""

    def __init__(self, min_num_instances=100):
        self.detector = ADWIN()
        self.n_samples = 0
        self.min_num_instances = min_num_instances
        self.drift_detected = False

    def add_element(self, error):
        """Adiciona elemento (0=correto, 1=erro)."""
        self.n_samples += 1
        # ADWIN espera valor (0 ou 1)
        self.detector.update(error)

        # Detectar drift
        if self.n_samples >= self.min_num_instances:
            self.drift_detected = self.detector.drift_detected

    def detected_change(self):
        """Retorna True se drift foi detectado."""
        result = self.drift_detected
        if result:
            self.drift_detected = False  # Reset flag
        return result


class MCMO:
    """
    MCMO with river compatibility.

    Modified to use river.tree.HoeffdingTreeClassifier and river.drift.ADWIN
    instead of scikit-multiflow.
    """

    def __init__(self, base_classifier=None, detector=None, source_number=1,
                 initial_beach=100, max_pool=5, gaussian_number=5):
        # Use river wrappers se não especificado
        self.base_classifier = base_classifier if base_classifier else RiverTreeWrapper()
        self.detector = detector if detector else RiverADWINWrapper(min_num_instances=100)

        self.source_number = source_number
        self.initial_beach = initial_beach
        self.max_pool = max_pool
        self.gaussian_number = gaussian_number

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

    def Feature_Reduce(self, X, individual):
        f = np.sum(individual)
        Trans = np.zeros((self.D, f))

        c = 0
        for i in range(len(individual)):
            if individual[i] == 1:
                Trans[i][c] = 1
                c = c + 1

        Tx = np.dot(X, Trans)
        return Tx

    def Model_initialization(self):
        problem = MyProblem(n=self.D, T=self.T, S_list=self.S_list, LS_list=self.LS_list, d=self.D)
        algorithm = ea.moea_NSGA2_templet(problem,
                                          ea.Population(Encoding='BG', NIND=50),
                                          MAXGEN=50,
                                          logTras=0)
        algorithm.mutOper.Pm = 0.2
        algorithm.recOper.XOVR = 0.9
        res = ea.optimize(algorithm, verbose=True, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
        self.solution = fuzz_decision(res['Vars'], res['ObjV'])

        Tx = Feature_Reduce(self.T, self.solution)
        self.gmm = DGMM(n_components=self.gaussian_number, random_state=0).fit(Tx)

        initial_confidence = self.gmm.evaluation_weight(Tx)
        self.drift_threshold = np.mean(initial_confidence) - 3*(np.var(initial_confidence))
        self.probability_set = list(initial_confidence)

        self.source_classifiers = []
        self.classifier_pool = []
        self.drift_detectors = []
        for i in range(self.source_number):
            Sx = Feature_Reduce(np.array(self.S_list[i]), self.solution)
            weight = self.gmm.evaluation_weight(Sx)
            base_classifier = cp.deepcopy(self.base_classifier)
            base_classifier.partial_fit(X=Sx, y=np.array(self.LS_list[i]), sample_weight=weight)
            self.source_classifiers.append(base_classifier)
            self.drift_detectors.append(cp.deepcopy(self.detector))

    def predict_proba(self, X):
        N, D = X.shape
        votes = np.zeros(N)

        if len(self.source_classifiers) <= 0:
            return votes

        Tx = Feature_Reduce(X, self.solution)

        if len(self.classifier_pool) == 0:
            for h_i in self.source_classifiers:
                votes = votes + 1. / len(self.source_classifiers) * h_i.predict(Tx)
        else:
            for h_i in self.source_classifiers:
                votes = votes + 1. / (len(self.source_classifiers) + len(self.classifier_pool)) * h_i.predict(Tx)
            for h_i in self.classifier_pool:
                votes = votes + 1. / (len(self.source_classifiers) + len(self.classifier_pool)) * h_i.predict(Tx)

        return votes

    def predict(self, X):
        votes = self.predict_proba(X)
        return (votes >= 0.5) * 1.
