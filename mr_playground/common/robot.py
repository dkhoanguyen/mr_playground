# !/usr/bin/python3
from mr_playground.controller import ControllerBase
from mr_playground.sensor import SensorBase

class Robot(object):
    """
    A wrapper to hold
    - Controller
    - Sensor
    Each robot maintains its own probability map and can be shared among other robots
    """

    def __init__(self,
                 id: int,
                 num_agents: int,
                 controller: ControllerBase,
                 sensor: SensorBase):
        self._controller = controller
        self._sensor = sensor

        self._id = id
        self._num_agents = num_agents


    def step(self, dt: float):
        # Handle inter robot comms
        pass

