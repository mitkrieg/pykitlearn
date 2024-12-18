from abc import ABC, abstractmethod
# import numpy as np
from typing import Callable, Sequence, Union

SequenceType = Sequence[float|int]

class AlreadyFitError(Exception):
    pass

class Model(ABC):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._is_fit: bool = False
        self._hyper_params: dict = kwargs
        self._params: dict = {}

    @abstractmethod
    def train(self, X: SequenceType, y: Union[SequenceType, None]=None) -> None:
        """Abstract method for training a model.

        Args:
            X: Squence-like object of data to train model on
            y: [optional] Sequence like object of data to use as targets for training
        """
        if self._is_fit:
            raise AlreadyFitError("Model is already fit. Reinstantiate or clear model to train.")
        
        self._is_fit = True
    
    @abstractmethod
    def predict(self, X: SequenceType) -> SequenceType:
        """Abstract method for getting predictions from model.

        Args:
            X: Squence-like object of data to have the model predict on"""
        pass
    
    @abstractmethod
    def eval(self, metric: Union[str, Callable], X: SequenceType, y: Union[SequenceType, None]= None) -> SequenceType:
        """Abstract method for evaluating predictions from model.

        Args:
            metric: string of metric to call or function that computes metric
            X: Squence-like object of data to evaluate the model on
            y: [optional] Sequence-like object of targets to evaluate the model against 
        """
        pass
    
    @property
    def is_fit(self):
        return self._is_fit
    
    @property
    def params(self):
        return self._params
    
    @abstractmethod
    def update_params(self):
        pass

    def clear(self):
        self._is_fit = False
        self._params.clear()

    @property
    @abstractmethod
    def optimizer(self):
        pass
