from abc import ABC, abstractmethod
from typing import Any

class LossFunction(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    @staticmethod
    def calculate_loss(*args: Any, **kwds: Any):
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.calculate_loss(*args, **kwargs)

class Optimizer(ABC):

    @abstractmethod
    def __init__(self):
        pass