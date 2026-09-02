"""Nominal 3-D Hopfield synchronization environment with configurable inputs.

This environment is intentionally limited to the nominal ablation setting.  It
keeps the dynamics, integration, observation, scaling, and episode length of
``ContinuousHopfieldEnv`` while allowing one, two, or three controlled nodes.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ContinuousHopfieldMultiInputEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, controlled_nodes: tuple[int, ...] | list[int] = (2,)):
        super().__init__()
        nodes = tuple(int(node) for node in controlled_nodes)
        if not nodes:
            raise ValueError("At least one controlled node is required.")
        if len(set(nodes)) != len(nodes):
            raise ValueError("controlled_nodes must not contain duplicates.")
        if any(node not in (1, 2, 3) for node in nodes):
            raise ValueError("controlled_nodes must use one-based node numbers 1, 2, or 3.")

        self.controlled_nodes = nodes
        self.control_indices = np.asarray([node - 1 for node in nodes], dtype=np.int64)
        self.num_control_inputs = len(nodes)

        self.dt = 0.01
        self.rk4_steps = 5
        self.max_steps = 400
        self.current_step = 0
        self.scale = 4.0

        self.W = np.array(
            [
                [-1.4, 1.2, -7.0],
                [1.1, 0.0, 2.8],
                [0.8, -2.0, 4.0],
            ],
            dtype=np.float32,
        )
        self.B = np.zeros((3, self.num_control_inputs), dtype=np.float32)
        for action_index, state_index in enumerate(self.control_indices):
            self.B[state_index, action_index] = 1.0

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_control_inputs,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.statex = np.zeros(3, dtype=np.float32)
        self.statey = np.zeros(3, dtype=np.float32)

    def _derivatives(self, state: np.ndarray, control_force: np.ndarray) -> np.ndarray:
        derivative = -state + self.W @ np.tanh(state)
        derivative = derivative + self.B @ control_force
        return derivative.astype(np.float32)

    def _rk4_step(self, state: np.ndarray, control_force: np.ndarray) -> np.ndarray:
        k1 = self._derivatives(state, control_force)
        k2 = self._derivatives(state + 0.5 * self.dt * k1, control_force)
        k3 = self._derivatives(state + 0.5 * self.dt * k2, control_force)
        k4 = self._derivatives(state + self.dt * k3, control_force)
        next_state = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return np.clip(next_state, -10.0, 10.0).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        error = self.statey - self.statex
        return np.concatenate([self.statex, error]).astype(np.float32) / 5.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.statex = self.np_random.uniform(-1.0, 1.0, size=3).astype(np.float32)
        self.statey = self.np_random.uniform(-1.0, 1.0, size=3).astype(np.float32)
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        normalized_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if normalized_action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {normalized_action.shape}."
            )
        control_force = np.clip(normalized_action, -1.0, 1.0) * self.scale
        zero_control = np.zeros(self.num_control_inputs, dtype=np.float32)

        for _ in range(self.rk4_steps):
            self.statex = self._rk4_step(self.statex, zero_control)
            self.statey = self._rk4_step(self.statey, control_force)

        error = self.statey - self.statex
        error_norm = float(np.linalg.norm(error, ord=1))
        reward = -0.1 * error_norm
        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {
            "error_norm": error_norm,
            "controlled_nodes": self.controlled_nodes,
            "control_force": control_force.copy(),
        }
        return self._get_obs(), reward, terminated, truncated, info
