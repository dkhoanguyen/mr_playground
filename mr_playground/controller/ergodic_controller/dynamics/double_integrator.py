# !/usr/bin/python3

import numpy as np
from gymnasium.spaces import Box



class DoubleIntegrator(DynamicsBase):
    def __init__(self,
                 observation_space: Box,
                 action_space: Box,
                 exploration_space: Box,
                 max_velocity: float = 1.0,
                 max_acceleration: float = 2.0):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            exploration_space=exploration_space,
            state_idx=[0, 1],
        )

        self._max_v = max_velocity
        self._max_u = max_acceleration

        self._A = np.array([
            [0., 0., 1.0, 0.],
            [0., 0., 0., 1.0],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]
        ])

        self._B = np.array([
            [0., 0.],
            [0., 0.],
            [1.0, 0.],
            [0., 1.0]
        ])

    def f(self, x: np.ndarray, u: np.ndarray):
        '''
        Continuous time dynamics with acceleration clipping.

        Args:
            x (np.ndarray): The current state vector.
            u (np.ndarray): The control input vector.

        Returns:
            np.ndarray: The derivative of the state vector.
        '''
        # u_clipped = np.clip(u, -self._max_u, self._max_u)
        norm_u = np.linalg.norm(u)
        if norm_u > self._max_u:
            u = self._max_u * (u/np.linalg.norm(u))
        return np.dot(self._A, x) + np.dot(self._B, u)

    def step(self, x: np.ndarray, u: np.ndarray, dt: float = 0.1):
        '''
        Basic euler step with velocity and acceleration clipping.

        Args:
            x (np.ndarray): The current state vector.
            u (np.ndarray): The control input vector.
            dt (float): The time step for the Euler integration (default is 0.1).

        Returns:
            np.ndarray: The updated state vector after applying the control input.
        '''
        new_x = x + self.f(x, u) * dt

        # Enforce maximum velocity constraint
        # Indices for velocity components in the state
        velocity_indices = [2, 3]
        new_x[velocity_indices] = np.clip(
            new_x[velocity_indices],
            -self._max_v,
            self._max_u
        )

        return new_x
