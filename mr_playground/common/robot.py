# !/usr/bin/python3

class Robot(object):
    """
    A wrapper to hold
    - Controller
    - Dynamic
    - Sensor
    Each robot maintains its own probability map and can be shared among other robots
    """

    def __init__(self,
                 controller,
                 dynamic,
                 sensor):
        self._controller = controller
        self._dynamic = dynamic
        self._sensor = sensor

    def step(self, dt: float):
        pass
