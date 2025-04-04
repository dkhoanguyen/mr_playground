# !/usr/bin/python3
import irsim
import matplotlib.pyplot as plt
from mr_playground.engine import EngineBase

# Distribution


class IRSim(EngineBase):
    def __init__(self,
                 world_file: str = "world.yaml",
                 fig_size: tuple = (7, 7),
                 steps: int = 300):
        self._env = irsim.make(world_file)
        plt.gcf().set_size_inches(fig_size[0], fig_size[1])
        self._steps = steps
        self._current_step = 0

        self._robots = []
        self._obstacles = []
        self._targets = []

    def init(self,
             general_config_path: str,
             robot_config_path: str):
        pass

    def ok(self):
        self._current_step += 1
        return self._current_step <= self._steps

    def step(self):
        # self._env.step()
        self._env._world.step()

    def render(self, dt: float = 0.01):
        self._env.render(interval=dt)
