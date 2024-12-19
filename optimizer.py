from abc import ABC, abstractmethod
from typing import Any, Callable, Union
import numpy as np

class LossFunction(ABC):

    @abstractmethod
    @staticmethod
    def calculate_loss(*args: Any, **kwds: Any):
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.calculate_loss(*args, **kwargs)
    
    @abstractmethod
    @staticmethod
    def gradient(*args: Any, **kwds: Any) -> Any:
        pass

class MSE(LossFunction):

    @staticmethod
    def calculate_loss(y_true, y_pred):
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
    def calculate_loss(y_true, y_pred):
        return np.sqrt((1/len(y_pred)) * np.sum((y_true - y_pred) ** 2))
    
    @staticmethod
    def gradient(X: np.ndarray, y: np.ndarray, params: np.ndarray):
        assert len(X) == len(y)
        X = np.c_[np.ones(len(X)), X]
        residuals = y - X.dot(params)
        rmse = RMSE.calculate_loss(y, X.dot(params))
        return -1 / (len(X) * rmse) * X.T.dot(residuals)
    
class Regularizer(ABC):

    @abstractmethod
    def penalty(self, X) -> np.ndarray:
        pass


    def __call__(self, X) -> Any:
        return self.penalty(X)

class RidgeL2(Regularizer):

    def __init__(self, lmda: float):
        self.lmda = lmda 

    def penalty(self, X):
        I = np.eye(X.shape[1])
        I[0,0] = 0 #don't apply to intercept
        return self.lmda * I

class Optimizer(ABC):

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> np.ndarray:
        pass


class OLS(Optimizer):
    
    def step(self, X: np.ndarray, y: np.ndarray, regularizer: Union[str, Regularizer, Callable, None]= None, reg_strength:float=0) -> np.ndarray:
        """Use the normal equation of MSE to get params
        $\theata = (X^TX)^{-1}X^Ty $

        Optionally add in regularization
        """
        if regularizer in ['l2', 'ridge']:
            penalty = RidgeL2(reg_strength).penalty(X)
        elif isinstance(regularizer, Regularizer) or isinstance(regularizer, Callable):
            penalty = regularizer(X)
        else:
            penalty = None
            
        if penalty:
            return np.linalg.inv(X.T.dot(X) + penalty).dot(X.T).dot(y)
        else:
            return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)