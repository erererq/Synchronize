import gymnasium
import numpy as np


class FHNEnv3D(gymnasium.Env):
    """Drive-response synchronization task for a forced 3D FHN neuron with slow modulation."""

    def __init__(self, mismatch_scale: float = 0.0):
        super().__init__()

        self.dt = 0.001
        self.max_steps = 4000
        self.scale = 2.0
        self.obs_scale = 3.0
        self.state_clip = 20.0
        self.divergence_threshold = 15.0
        self.sync_threshold = 0.10
        self.success_bonus = 1.0
        self.mismatch_scale = mismatch_scale

        # Drive-system parameters.
        self.a = 0.7
        self.b = 0.8
        self.eps = 0.08
        self.bias = 0.40
        self.alpha = 0.20
        self.force_amp = 0.80
        self.omega = 1.0

        # Response-system parameters.
        self.a_r = self.a + 0.02 * mismatch_scale
        self.b_r = self.b - 0.02 * mismatch_scale
        self.eps_r = self.eps + 0.005 * mismatch_scale
        self.bias_r = self.bias + 0.02 * mismatch_scale
        self.alpha_r = self.alpha + 0.02 * mismatch_scale
        self.force_amp_r = self.force_amp
        self.omega_r = self.omega

        self.statex = np.zeros(3, dtype=np.float32)
        self.statey = np.zeros(3, dtype=np.float32)
        self.time = 0.0
        self.current_step = 0

        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        error = self.statey - self.statex
        return np.concatenate([self.statex, self.statey, error]).astype(np.float32) / self.obs_scale

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.time = 0.0
        self.statex = self.np_random.uniform(low=-0.8, high=0.8, size=3).astype(np.float32)
        self.statey = self.np_random.uniform(low=-0.8, high=0.8, size=3).astype(np.float32)
        return self._get_obs(), {}

    def get_derivatives(
        self,
        state: np.ndarray,
        a: float,
        b: float,
        eps: float,
        bias: float,
        alpha: float,
        force_amp: float,
        omega: float,
        time_value: float,
        control: float = 0.0,
    ) -> np.ndarray:
        x, y, z = state
        forcing = force_amp * np.sin(omega * time_value)
        dx = x - (x**3) / 3.0 - y + z + bias + control
        dy = eps * (x + a - b * y)
        dz = -alpha * z + forcing
        return np.array([dx, dy, dz], dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        scalar_action = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        real_action = scalar_action * self.scale

        self.statex += self.dt * self.get_derivatives(
            self.statex,
            self.a,
            self.b,
            self.eps,
            self.bias,
            self.alpha,
            self.force_amp,
            self.omega,
            self.time,
            control=0.0,
        )
        self.statey += self.dt * self.get_derivatives(
            self.statey,
            self.a_r,
            self.b_r,
            self.eps_r,
            self.bias_r,
            self.alpha_r,
            self.force_amp_r,
            self.omega_r,
            self.time,
            control=real_action,
        )
        self.time += self.dt

        self.statex = np.clip(self.statex, -self.state_clip, self.state_clip)
        self.statey = np.clip(self.statey, -self.state_clip, self.state_clip)

        error = self.statey - self.statex
        error_l2 = float(np.linalg.norm(error, ord=2))
        error_l1 = float(np.linalg.norm(error, ord=1))
        reward = -error_l2 - 0.01 * (real_action**2)

        if error_l2 < self.sync_threshold:
            reward += self.success_bonus

        state_peak = float(max(np.max(np.abs(self.statex)), np.max(np.abs(self.statey))))
        terminated = state_peak >= self.divergence_threshold
        truncated = self.current_step >= self.max_steps
        obs = self._get_obs()
        info = {
            "error_l1": error_l1,
            "error_l2": error_l2,
            "control": real_action,
            "mismatch_scale": self.mismatch_scale,
            "sync_success": error_l2 < self.sync_threshold,
            "time": self.time,
        }
        return obs, float(reward), terminated, truncated, info
