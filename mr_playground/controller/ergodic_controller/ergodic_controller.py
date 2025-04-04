# !/usr/bin/python3

import numpy as np
from mr_playground.controller import ControllerBase
from mr_playground.controller.ergodic_controller.dynamics import DynamicsBase

# Ergodic Controller specifics
from mr_playground.controller.ergodic_controller import Basis
from mr_playground.controller.ergodic_controller import Barrier
from mr_playground.controller.ergodic_controller import ReplayBuffer
from mr_playground.controller.ergodic_controller import ErgodicMetric
from mr_playground.controller.ergodic_controller import Distribution


class ErgodicController(ControllerBase):
    def __init__(self,
                 dynamics: DynamicsBase,
                 weights=None,
                 horizon=100,
                 num_basis=5,
                 capacity=100000,
                 batch_size=20):
        '''
        Initialize the ErgodicController.

        Args:
            dynamics: The dynamics of the system.
            weights: The weights for the control input.
            horizon: The time horizon for the control sequence.
            num_basis: The number of basis functions.
            capacity: The capacity of the replay buffer.
            batch_size: The batch size for sampling from the replay buffer.
        '''
        self._dynamics = dynamics
        self._horizon = horizon
        self._replay_buffer = ReplayBuffer(capacity)
        self._batch_size = batch_size

        self._basis = Basis(self._dynamics.explr_space, num_basis=num_basis)
        self._lamk = np.exp(-0.8 * np.linalg.norm(self._basis.k, axis=1))
        self._barr = Barrier(self._dynamics.explr_space)

        self._u_seq = [0.0 * self._dynamics.action_space.sample()
                       for _ in range(horizon)]

        if weights is None:
            weights = {'R': np.eye(self._dynamics.action_space.shape[0])}

        self._Rinv = np.linalg.inv(weights['R'])

        self._phik = None
        self._ck = None

    def reset(self):
        '''
        Reset the controller.

        This method resets the control sequence and the replay buffer.
        '''
        self._u_seq = [0.0 * self._dynamics.action_space.sample()
                       for _ in range(self._horizon)]
        self._replay_buffer.reset()

    @property
    def phik(self):
        '''
        Get the phik value.

        Returns:
            The phik value.
        '''
        return self._phik

    @phik.setter
    def phik(self, phik):
        '''
        Set the phik value.

        Args:
            phik: The new phik value.
        '''
        assert len(
            phik) == self._basis.tot_num_basis, 'phik does not have the same number as ck'
        self._phik = phik

    @property
    def ck(self):
        '''
        Get the ck value.

        Returns:
            The ck value.
        '''
        return self._ck

    @property
    def lamk(self):
        '''
        Get the lamk value.

        Returns:
            The lamk value.
        '''
        return self._lamk

    @ck.setter
    def ck(self, ck):
        '''
        Set the ck value.

        Args:
            ck: The new ck value.
        '''
        self._ck = ck

    def set_t_dist(self, t_dist: Distribution):
        '''
        Set the target distribution t_dist.

        Args:
            t_dist: The target distribution.
        '''
        # Apply a fourier transform to the target distribution
        self._phik = ErgodicMetric.convert_phi2phik(
            self._basis, t_dist._grid_vals, t_dist._grid)
        
    def step(self,
             state: np.ndarray,
             control: np.ndarray,
             max_v: np.ndarray,
             min_v: np.ndarray,
             dt: float) -> np.ndarray:
        pass

    def _calculate_control_u(self, state: np.ndarray, dt: float, ck_list=None, agent_num=None):
        '''
        Compute the control input to drive the system towards the target distribution.

        Args:
            state: The current state of the system.
            dt: The time step.
            ck_list: The list of ck values for multiple agents.
            agent_num: The agent number.

        Returns:
            The control input to be applied to the system.
        '''
        assert self._phik is not None, 'Forgot to set phik, use set_target_phik method'

        self._u_seq[:-1] = self._u_seq[1:]
        self._u_seq[-1] = np.zeros(self._dynamics.action_space.shape)

        x = state
        pred_traj = []
        dfk = []
        fdx = []
        fdu = []
        dbar = []
        for t in range(self._horizon):
            pred_traj.append(x[self._dynamics.explr_idx])
            dfk.append(self._basis.dfk(x[self._dynamics.explr_idx]))
            fdx.append(self._dynamics.fdx(x, self._u_seq[t]))
            fdu.append(self._dynamics.fdu(x))
            dbar.append(self._barr.dx(x[self._dynamics.explr_idx]))
            x = self._dynamics.step(x, self._u_seq[t] * 0.)

        if len(self._replay_buffer) > self._batch_size:
            past_states = self._replay_buffer.sample(self._batch_size)
            pred_traj = pred_traj + past_states
        else:
            past_states = self._replay_buffer.sample(len(self._replay_buffer))
            pred_traj = pred_traj + past_states

        N = len(pred_traj)
        ck: np.ndarray = np.sum([self._basis.fk(xt)
                                for xt in pred_traj], axis=0) / N

        self.ck = ck.copy()
        if ck_list is not None:
            ck_list[agent_num] = ck
            ck = np.mean(ck_list, axis=0)

        fourier_diff: np.ndarray = self._lamk * (ck - self._phik)
        fourier_diff = fourier_diff.reshape(-1, 1)

        rho = np.zeros(self._dynamics.observation_space.shape)
        for t in reversed(range(self._horizon)):
            edx = np.zeros(self._dynamics.observation_space.shape)
            edx[self._dynamics.explr_idx] = np.sum(dfk[t] * fourier_diff, 0)

            bdx = np.zeros(self._dynamics.observation_space.shape)
            bdx[self._dynamics.explr_idx] = dbar[t]
            rho = rho - dt * (- edx - bdx - np.dot(fdx[t].T, rho))

            self._u_seq[t] = -np.dot(np.dot(self._Rinv, fdu[t].T), rho)
            if (np.abs(self._u_seq[t]) > 1.0).any():
                self._u_seq[t] /= np.linalg.norm(self._u_seq[t])

        self._replay_buffer.push(state[self._dynamics.explr_idx])
        return self._u_seq[0].copy()
