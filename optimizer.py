from abc import ABC, abstractmethod
from typing import Any, Callable, Union
import numpy as np
    
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