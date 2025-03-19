from abc import ABC, abstractmethod
import numpy as np
from scipy.stats._multivariate import multivariate_normal_frozen


class SensorBase(ABC):
    """
    Sensor is an abstract class that defines the interface for all sensors
    """

    @abstractmethod
    def read(self,
             pos: np.ndarray,
             ground_truth: np.ndarray,
             noise: np.ndarray) -> multivariate_normal_frozen:
        """
        Read the environment and return the sensor reading as a belief gaussian distribution
        """
        pass
