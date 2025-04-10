#!/usr/bin/python3

from abc import ABC, abstractmethod
import numpy as np


class SensingModel(ABC):

    def __init__(self):
        self._all_states = np.empty((0, 6))

    def update(self, all_states: dict):
        self._all_states = all_states

    def get_raw_all_states(self, target: str) -> np.ndarray:
        return self._all_states[target]

    @abstractmethod
    def sense(self, state: np.ndarray,
              target: str,
              add_noise: bool) -> np.ndarray:
        """
        """
