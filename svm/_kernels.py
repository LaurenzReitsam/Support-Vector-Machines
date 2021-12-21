from abc import ABC, abstractmethod

import numpy as np


class Kernel(ABC):

    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray):
        ...


class LinearKernel(Kernel):

    def __call__(self, X1: np.ndarray, X2: np.ndarray):
        return X1 @ X2.T


class RbfKernel(Kernel):

    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, X1: np.ndarray, X2: np.ndarray):
        if self.sigma is None:
            self.sigma = 1 / X2.shape[1]
        return (X1 @ X2.T + 1.0) ** self.sigma


class PolyKernel(Kernel):

    def __init__(self, poly: int):
        self.poly = poly

    def __call__(self, X1: np.ndarray, X2: np.ndarray):
        return (X1 @ X2.T + 1.0) ** self.poly
