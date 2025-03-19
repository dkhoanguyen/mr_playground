from abc import ABC, abstractmethod
import numpy as np


class ControllerBase(ABC):
    @abstractmethod
    def step(self,
             state: np.ndarray,
             velocity: np.ndarray,
             max_v: np.ndarray,
             min_v: np.ndarray,
             dt: float) -> np.ndarray:
        pass
