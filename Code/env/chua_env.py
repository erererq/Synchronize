import numpy as np
import gymnasium


class ChuaEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()
        # Normalized Chua circuit parameters.
        self.dt = 0.01
        self.alpha = 15.6
        self.beta = 28.0
        self.m0 = -8.0 / 7.0
        self.m1 = -5.0 / 7.0

        # The physical control is action * scale.
        self.scale = 8.0

        # Control the response system through y and z channels.
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observation = drive state + synchronization error.
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.statex = np.zeros(3, dtype=np.float32)
        self.statey = np.zeros(3, dtype=np.float32)

        self.max_steps = 2000
        self.current_step = 0
        self.enable_scale_diagnostics = True

        self._reset_scale_stats()

    def _reset_scale_stats(self):
        self.natural_abs_sum = 0.0
        self.control_abs_sum = 0.0
        self.control_ratio_samples = []
        self.obs_abs_max_samples = []
        self.error_penalty_sum = 0.0
        self.stats_steps = 0

    def _nonlinearity(self, x: float) -> float:
        return self.m1 * x + 0.5 * (self.m0 - self.m1) * (abs(x + 1.0) - abs(x - 1.0))

    def _get_obs(self):
        error = self.statey - self.statex
        obs = np.concatenate([self.statex, error]).astype(np.float32) / 5.0
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.statex = self.np_random.uniform(low=-0.5, high=0.5, size=(3,)).astype(np.float32)
        self.statey = self.np_random.uniform(low=-0.5, high=0.5, size=(3,)).astype(np.float32)
        self.current_step = 0
        self._reset_scale_stats()
        return self._get_obs(), {}

    def get_derivatives(self, state, action=None):
        if action is None:
            action = np.zeros(2, dtype=np.float32)

        x, y, z = state
        h_x = self._nonlinearity(float(x))

        dx = self.alpha * (y - x - h_x)
        dy = x - y + z + action[0]
        dz = -self.beta * y + action[1]
        return np.array([dx, dy, dz], dtype=np.float32)

    def rk4_step(self, state, action=None):
        if action is None:
            action = np.zeros(2, dtype=np.float32)

        k1 = self.get_derivatives(state, action=action)
        k2 = self.get_derivatives(state + 0.5 * self.dt * k1, action=action)
        k3 = self.get_derivatives(state + 0.5 * self.dt * k2, action=action)
        k4 = self.get_derivatives(state + self.dt * k3, action=action)
        return state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, action):
        self.current_step += 1
        real_action = np.asarray(action, dtype=np.float32) * self.scale

        x, y, z = self.statey
        natural_terms = np.array(
            [
                self.alpha * (y - x - self._nonlinearity(float(x))),
                x - y + z,
                -self.beta * y,
            ],
            dtype=np.float32,
        )

        control_abs = float(np.abs(real_action).sum())
        natural_abs = float(np.abs(natural_terms).sum())
        control_ratio = control_abs / (natural_abs + 1e-6)
        self.natural_abs_sum += natural_abs
        self.control_abs_sum += control_abs
        self.control_ratio_samples.append(control_ratio)
        self.stats_steps += 1

        self.statex = self.rk4_step(self.statex)
        self.statey = self.rk4_step(self.statey, action=real_action)

        error = self.statey - self.statex
        error_norm = np.linalg.norm(error, ord=1)
        obs = self._get_obs()
        self.obs_abs_max_samples.append(float(np.max(np.abs(obs))))

        error_penalty = error_norm
        reward = -np.log1p(error_norm)
        self.error_penalty_sum += error_penalty

        truncated = self.current_step >= self.max_steps
        terminated = False
        info = {
            "natural_abs": natural_abs,
            "control_abs": control_abs,
            "control_ratio": control_ratio,
            "error_norm": error_norm,
            "error_penalty": error_penalty,
        }

        if truncated and self.stats_steps > 0:
            ratio_array = np.asarray(self.control_ratio_samples, dtype=np.float32)
            obs_abs_max_array = np.asarray(self.obs_abs_max_samples, dtype=np.float32)
            info["scale_diagnostics"] = {
                "avg_natural_abs": self.natural_abs_sum / self.stats_steps,
                "avg_control_abs": self.control_abs_sum / self.stats_steps,
                "global_control_ratio": self.control_abs_sum / (self.natural_abs_sum + 1e-6),
                "p50_control_ratio": float(np.percentile(ratio_array, 50)),
                "p90_control_ratio": float(np.percentile(ratio_array, 90)),
                "p95_control_ratio": float(np.percentile(ratio_array, 95)),
                "max_control_ratio": float(np.max(ratio_array)),
                "avg_obs_abs_max": float(np.mean(obs_abs_max_array)),
                "p50_obs_abs_max": float(np.percentile(obs_abs_max_array, 50)),
                "p90_obs_abs_max": float(np.percentile(obs_abs_max_array, 90)),
                "max_obs_abs_max": float(np.max(obs_abs_max_array)),
                "avg_error_penalty": self.error_penalty_sum / self.stats_steps,
            }
            if self.enable_scale_diagnostics:
                stats = info["scale_diagnostics"]
                print(
                    "[scale-check] "
                    f"scale={self.scale}, "
                    f"avg|natural|={stats['avg_natural_abs']:.3f}, "
                    f"avg|u|={stats['avg_control_abs']:.3f}, "
                    f"global ratio={stats['global_control_ratio']:.3f}, "
                    f"p50={stats['p50_control_ratio']:.3f}, "
                    f"p90={stats['p90_control_ratio']:.3f}, "
                    f"p95={stats['p95_control_ratio']:.3f}, "
                    f"max ratio={stats['max_control_ratio']:.3f}, "
                    f"obs|max| avg={stats['avg_obs_abs_max']:.3f}, "
                    f"p50={stats['p50_obs_abs_max']:.3f}, "
                    f"p90={stats['p90_obs_abs_max']:.3f}, "
                    f"max={stats['max_obs_abs_max']:.3f}, "
                    f"reward parts: err={stats['avg_error_penalty']:.3f}"
                )

        return obs, reward, terminated, truncated, info
