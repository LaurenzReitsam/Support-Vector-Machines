from abc import ABC, abstractmethod
from collections import namedtuple

from ._kernels import Kernel

import cvxpy as cp
import numpy as np

_SupportVectors = namedtuple("SupportVectors", "x y")
_ClassifierParams = namedtuple("ClassifierParams", "intercept lagranges weights")
_RegressorParams = namedtuple("RegressorParams", "intercept lagranges")


class _SupportVectorBasisModel(ABC):
    """Base for the following classifiaction and regression models."""

    _params: tuple
    _support_vectors: _SupportVectors

    def __init__(self, kernel: Kernel, c: float):
        self._kernel = kernel
        self._std = None
        self._c = c

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass

    def _normalize(self, X, set_value=False):
        if set_value: self._std = X.std(axis=0)
        return X / self._std

    @property
    def support_vectors(self):
        return self._support_vectors.x * self._std


class SupporVectorClassifier(_SupportVectorBasisModel):
    """Support-Vector classification model"""

    _params: _ClassifierParams

    def __init__(self, kernel: Kernel, c=1):
        super().__init__(kernel, c)

    def _solve_problem(self, X, y):
        n_samples, n_features = X.shape

        K = self._kernel(X, X)
        P = K * (y @ y.T)
        P = cp.Parameter(shape=P.shape, value=P, PSD=True)
        q = np.ones(n_samples)
        A = y.copy().reshape([-1])
        G = np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples)))
        h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self._c))

        a = cp.Variable(n_samples)

        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(a, P) - q.T @ a),
                          [G @ a <= h, A @ a == 0.0])

        prob.solve()
        return np.ravel(a.value)

    @staticmethod
    def _check_y(y):
        classes = np.unique(y)
        assert len(classes) == 2, "y must consist of 2 classes"
        assert abs(classes[0]) == 1, "Class labels must be 1 and -1"
        assert abs(classes[1]) == 1, "Class labels must be 1 and -1"

    def train(self, X: np.ndarray, y: np.ndarray):
        self._check_y(y)

        X = X.copy()
        X = self._normalize(X, set_value=True)
        y = np.copy(y)
        y = y.reshape([-1, 1])

        lagranges = self._solve_problem(X, y)

        mask = np.abs(lagranges) > 1e-5
        lagranges[np.invert(mask)] = 0
        lagranges = lagranges.reshape([-1, 1])

        sv_X = X[mask]
        sv_y = y[mask]
        sv_K = self._kernel(sv_X, sv_X)

        intercept = np.mean(sv_y - sv_K.T @ (lagranges[mask] * sv_y))
        weights = (sv_X.T @ (lagranges[mask] * sv_y.reshape([-1, 1]))).reshape([-1])

        self._params = _ClassifierParams(intercept, lagranges[mask], weights)
        self._support_vectors = _SupportVectors(sv_X, sv_y)

    def predict(self, X: np.ndarray):
        X = X.copy()
        X = self._normalize(X)

        lagranges = self._params.lagranges
        intercept = self._params.intercept
        sv_X = self._support_vectors.x
        sv_y = self._support_vectors.y.reshape([-1, 1])

        K = self._kernel(sv_X, X)
        y = K.T @ (lagranges * sv_y) + intercept

        return np.sign(y.reshape([-1]))


class SupportVectorRegressor(_SupportVectorBasisModel):
    """Support-Vector regression model"""

    _params = _RegressorParams

    def __init__(self, kernel: Kernel, c: float = 1, eta: float = 0.1):
        super().__init__(kernel, c)
        self.eta = eta

    def _solve_problem(self, X, y):
        n_samples, n_features = X.shape

        K = self._kernel(X, X)
        P = K
        P = cp.Parameter(shape=P.shape, value=P, PSD=True)
        q = np.ones(n_samples)
        # A = y.copy().reshape([-1])
        G = np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples)))
        h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self._c))

        a1 = cp.Variable(n_samples)
        a2 = cp.Variable(n_samples)
        delta = a1 - a2

        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(delta, P) - y.T @ delta + self.eta * q.T @ (a1 + a2)),
                          [G @ a1 <= h, G @ a2 <= h, q.T @ delta == 0.0])

        prob.solve()
        return np.ravel(delta.value)

    def train(self, X: np.ndarray, y: np.ndarray):

        X = X.copy()
        X = self._normalize(X, set_value=True)
        y = np.copy(y)
        y = y.reshape([-1])

        lagranges = self._solve_problem(X, y)

        mask = np.abs(lagranges) > 1e-5
        lagranges[np.invert(mask)] = 0

        sv_X = X[mask]
        sv_y = y[mask]
        sv_K = self._kernel(sv_X, sv_X)

        lagranges = lagranges.reshape([-1, 1])

        intercept = np.mean(sv_y - sv_K.T @ (lagranges[mask]))

        self._params = _RegressorParams(intercept, lagranges[mask])
        self._support_vectors = _SupportVectors(sv_X, sv_y)

    def predict(self, X: np.ndarray):
        X = X.copy()
        X = self._normalize(X)

        lagranges = self._params.lagranges
        intercept = self._params.intercept
        sv_X = self._support_vectors.x

        K = self._kernel(sv_X, X)
        y = K.T @ lagranges + intercept

        return y.reshape([-1])
