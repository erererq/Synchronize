from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ContinuousHopfield4DHHNNEnv(gym.Env):
    """4D hyperchaotic Hopfield neural network synchronization environment.

    Identifier intent: 4D_HHNN_2020.
    Model form:
        x_dot = -c * x + W @ tanh(x) + B * u
    with c = [1, 1, 1, 100].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        w11: float = 1.0,
        w22: float = 2.0,
        w33: float = 1.0,
        w44: float = 170.0,
        mismatch_scale: float = 0.0,
        pulse_step: int | None = None,
        pulse_vector: tuple[float, ...] | None = None,
        pulse_width: int = 1,
        pinning_node: int = 3,
        pinning_nodes: tuple[int, ...] | list[int] | None = None,
        process_noise_std: float = 0.0,
        control_scale: float = 20.0,
        obs_scale: float = 5.0,
    ):
        super().__init__()

        self.system_tag = "4D_HHNN_2020"
        self.dim = 4
        self.dt = 0.001
        self.rk4_steps = 10
        self.max_steps = 600
        self.current_step = 0

        self.w11 = float(w11)
        self.w22 = float(w22)
        self.w33 = float(w33)
        self.w44 = float(w44)
        self.c = np.array([1.0, 1.0, 1.0, 100.0], dtype=np.float32)
        self.W = self._build_weight_matrix()
        self.W_r = self.W.copy()

        self.mismatch_scale = float(mismatch_scale)
        self.process_noise_std = float(process_noise_std)
        self.scale = float(control_scale)
        self.obs_scale = float(obs_scale)

        if pinning_nodes is None:
            pinning_nodes = [int(pinning_node)]
        self.control_indices = np.asarray([int(node) for node in pinning_nodes], dtype=np.int64)
        if self.control_indices.size == 0:
            raise ValueError("At least one pinning node is required.")
        if len(set(self.control_indices.tolist())) != int(self.control_indices.size):
            raise ValueError("pinning_nodes must not contain duplicates.")
        if np.any(self.control_indices < 0) or np.any(self.control_indices >= self.dim):
            raise ValueError(f"pinning_nodes must be in [0, {self.dim - 1}].")

        self.B = np.zeros(self.dim, dtype=np.float32)
        self.B[self.control_indices] = 1.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.control_indices),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * self.dim,), dtype=np.float32)

        self.pulse_step = pulse_step
        self.pulse_width = max(int(pulse_width), 1)
        if pulse_vector is None:
            self.pulse_vector = np.zeros(self.dim, dtype=np.float32)
        else:
            self.pulse_vector = np.asarray(pulse_vector, dtype=np.float32)
            if self.pulse_vector.shape != (self.dim,):
                raise ValueError(f"pulse_vector must have shape ({self.dim},).")

        self.statex = np.zeros(self.dim, dtype=np.float32)
        self.statey = np.zeros(self.dim, dtype=np.float32)
        self._reset_scale_stats()

    def _build_weight_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.w11, 0.5, -3.0, -1.0],
                [0.0, self.w22, 3.0, 0.0],
                [3.0, -3.0, self.w33, 0.0],
                [100.0, 0.0, 0.0, self.w44],
            ],
            dtype=np.float32,
        )

    def _reset_scale_stats(self) -> None:
        self.natural_abs_sum = 0.0
        self.control_abs_sum = 0.0
        self.control_ratio_samples: list[float] = []
        self.error_sum = 0.0
        self.stats_steps = 0

    def _get_derivatives(
        self,
        state: np.ndarray,
        W_matrix: np.ndarray,
        B: np.ndarray,
        control_force: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        dx = -self.c * state + W_matrix @ np.tanh(state)
        control = np.asarray(control_force, dtype=np.float32)
        if control.ndim == 0:
            dx += B * float(control)
        else:
            dx[self.control_indices] += control
        return dx.astype(np.float32)

    def _rk4_step(
        self,
        state: np.ndarray,
        W_matrix: np.ndarray,
        B: np.ndarray,
        control_force: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        k1 = self._get_derivatives(state, W_matrix, B, control_force)
        k2 = self._get_derivatives(state + 0.5 * self.dt * k1, W_matrix, B, control_force)
        k3 = self._get_derivatives(state + 0.5 * self.dt * k2, W_matrix, B, control_force)
        k4 = self._get_derivatives(state + self.dt * k3, W_matrix, B, control_force)
        next_state = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return np.clip(next_state, -20.0, 20.0).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        error = self.statey - self.statex
        return np.concatenate([self.statex, error]).astype(np.float32) / self.obs_scale

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self._reset_scale_stats()

        if self.mismatch_scale > 0.0:
            random_perturbation = self.np_random.uniform(low=-0.03, high=0.03, size=self.W.shape).astype(np.float32)
            self.W_r = self.W + self.mismatch_scale * (self.W * random_perturbation)
        else:
            self.W_r = self.W.copy()

        self.statex = self.np_random.uniform(low=-1.0, high=1.0, size=(self.dim,)).astype(np.float32)
        self.statey = self.np_random.uniform(low=-1.0, high=1.0, size=(self.dim,)).astype(np.float32)
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        action = np.asarray(action, dtype=np.float32)
        control_force = np.clip(action, -1.0, 1.0) * self.scale

        natural_terms = self._get_derivatives(self.statey, self.W_r, self.B, control_force=0.0)
        natural_abs = float(np.abs(natural_terms).sum())
        control_abs = float(np.abs(control_force).sum())
        self.natural_abs_sum += natural_abs
        self.control_abs_sum += control_abs
        self.control_ratio_samples.append(control_abs / (natural_abs + 1e-6))
        self.stats_steps += 1

        for _ in range(self.rk4_steps):
            self.statex = self._rk4_step(self.statex, self.W, self.B, control_force=0.0)
            self.statey = self._rk4_step(self.statey, self.W_r, self.B, control_force=control_force)

        if self.process_noise_std > 0.0:
            noise = self.np_random.normal(loc=0.0, scale=self.process_noise_std, size=self.statey.shape).astype(np.float32)
            self.statey = np.clip(self.statey + noise, -20.0, 20.0)

        pulse_applied = False
        if self.pulse_step is not None and self.pulse_step <= self.current_step < self.pulse_step + self.pulse_width:
            self.statex = np.clip(self.statex + self.pulse_vector, -20.0, 20.0)
            pulse_applied = True

        error = self.statey - self.statex
        error_norm = float(np.linalg.norm(error, ord=1))
        self.error_sum += error_norm
        reward = -0.1 * error_norm

        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {
            "system_tag": self.system_tag,
            "error_norm": error_norm,
            "pulse_applied": pulse_applied,
            "control_ratio": self.control_ratio_samples[-1],
        }

        if truncated and self.stats_steps > 0:
            info["scale_diagnostics"] = {
                "scale": float(self.scale),
                "avg_control_abs": self.control_abs_sum / self.stats_steps,
                "avg_natural_abs": self.natural_abs_sum / self.stats_steps,
                "global_control_ratio": self.control_abs_sum / (self.natural_abs_sum + 1e-6),
                "max_control_ratio": float(np.max(self.control_ratio_samples)),
                "mean_error_norm": self.error_sum / self.stats_steps,
            }

        return self._get_obs(), float(reward), terminated, truncated, info
