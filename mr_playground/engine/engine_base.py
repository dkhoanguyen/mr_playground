from abc import ABC, abstractmethod


class EngineBase(ABC):
    @abstractmethod
    def init(self, 
             general_config_path: str,
             robot_config_path: str):
        """
        """

    @abstractmethod
    def ok(self):
        """
        """

    @abstractmethod
    def step(self):
        """
        """

    @abstractmethod
    def render(self):
        """
        """
