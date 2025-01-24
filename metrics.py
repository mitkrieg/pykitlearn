from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Metric(ABC):

    @abstractmethod
    @staticmethod
    def calculate(*args: Any, **kwds: Any):
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.calculate(*args, **kwargs)
    
    @abstractmethod
    def __repr__(self) -> str:
        pass

class LossFunction(Metric):
    
    @abstractmethod
    @staticmethod
    def gradient(*args: Any, **kwds: Any) -> Any:
        pass

class R2(Metric):

    def __init__(self, adjusted=False):
        self.adjusted = adjusted

    @staticmethod
    def calculate(*args: Any, **kwds: Any):
        pass

class Correlation(Metric):

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def calculate(*args: Any, **kwds: Any):
        pass

class MSE(LossFunction):

    @staticmethod
    def calculate(y_true, y_pred):
        assert len(y_true) == len(y_pred)
        return (1/len(y_pred)) * np.sum((y_true - y_pred) ** 2)
    
    @staticmethod
    def gradient(X: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        assert len(X) == len(y)
        X = np.c_[np.ones(len(X)), X]
        residuals = y - X.dot(params)
        return -2 / len(X) * X.T.dot(residuals)
    
class RMSE(LossFunction):

    @staticmethod
    def calculate(y_true, y_pred):
        return np.sqrt((1/len(y_pred)) * np.sum((y_true - y_pred) ** 2))
    
    @staticmethod
    def gradient(X: np.ndarray, y: np.ndarray, params: np.ndarray):
        assert len(X) == len(y)
        X = np.c_[np.ones(len(X)), X]
        residuals = y - X.dot(params)
        rmse = RMSE.calculate(y, X.dot(params))
        return -1 / (len(X) * rmse) * X.T.dot(residuals)