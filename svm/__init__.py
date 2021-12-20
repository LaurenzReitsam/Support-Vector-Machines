from typing import Optional

from ._kernels import LinearKernel, RbfKernel, PolyKernel
from ._svm import SupportVectorRegressor, SupporVectorClassifier


class LinearSupportVectorRegressor(SupportVectorRegressor):

    def __init__(self, c: float = 1, eta: float = 0.1):
        super().__init__(kernel=LinearKernel(), c=c, eta=eta)


class LinearSupportVectorClassifier(SupporVectorClassifier):

    def __init__(self, c: float = 1):
        super().__init__(kernel=LinearKernel(), c=c)


class RbfSupportVectorRegressor(SupportVectorRegressor):

    def __init__(self, c: float = 1, eta: float = 0.1, sigma: Optional[float] = None):
        super().__init__(kernel=RbfKernel(sigma=sigma), c=c, eta=eta)


class RbfSupportVectorClassifier(SupporVectorClassifier):

    def __init__(self, c: float = 1, sigma: Optional[float] = None):
        super().__init__(kernel=RbfKernel(sigma=sigma), c=c)


class PolySupportVectorRegressor(SupportVectorRegressor):

    def __init__(self, c: float = 1, eta: float = 0.1, poly: int = 2):
        super().__init__(kernel=PolyKernel(poly=poly), c=c, eta=eta)


class PolySupportVectorClassifier(SupporVectorClassifier):

    def __init__(self, c: float = 1, poly: int = 2):
        super().__init__(kernel=PolyKernel(poly=poly), c=c)

