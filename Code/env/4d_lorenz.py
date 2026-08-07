import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium

class LorenzEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()
        # 1. 系统参数 (超混沌 Chen 系统的典型参数)
        self.dt = 0.001
        self.a = 35.0
        self.b = 3.0
        self.c = 12.0
        self.d = 7.0
        self.r = 0.5
        # 2. 动作空间的“黑魔法”缩放因子
        # 真实物理控制力将是 action * self.scale
        self.scale = 120  # 这个值需要根据系统的自然演化尺度来调整，过大或过小都会让训练变得困难
        # 3. 动作空间定义 (Action Space)
        # 单变量控制 (shape=(1,))，且范围永远限定在 [-1.0, 1.0]
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # 4. 观测空间定义 (Observation Space)
        # 智能体需要看到驱动系统(x)和响应系统(y)之间的【误差】，
        # 误差 e = y - x，因为是 4 维系统，所以观测值是 4 维
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        # 初始化内部状态变量
        self.statex = np.zeros(4) # 发送端 (不受控)
        self.statey = np.zeros(4) # 接收端 (受控)
        
        self.max_steps = 2000
        self.current_step = 0
        self.enable_scale_diagnostics = True

        self._reset_scale_stats()

    def _reset_scale_stats(self):
        self.natural_dw_abs_sum = 0.0
        self.control_abs_sum = 0.0
        self.control_ratio_samples = []
        self.obs_abs_max_samples = []
        self.error_penalty_sum = 0.0
        self.stats_steps = 0

    def _get_obs(self):
        # 1. 计算真实的物理误差
        error = self.statey - self.statex
        
        # 4. 拼装最终观测值
        obs = np.concatenate([self.statex, error]).astype(np.float32)/30
        return obs
    def reset(self, seed=None, options=None):
        super().reset(seed=seed,options=options)
        # 2. 随机生成两套初始物理状态 (演员就位)
        # 超混沌 Chen 系统的变量一般在几十的量级。我们让它们在 [-5.0, 5.0] 附近随机出生。
        # 注意：一定要用 self.np_random 而不是 np.random，这样才能受 seed 控制！
        self.statex = self.np_random.uniform(low=-5.0, high=5.0, size=(4,))
        self.statey = self.np_random.uniform(low=-5.0, high=5.0, size=(4,))
        obs = self._get_obs()
        
        # 6. 重置时间步计数器
        self.current_step = 0
        self._reset_scale_stats()
        
        # 7. 返回 obs 和一个空的 info 字典 (Gymnasium 官方铁律)
        info = {}
        return obs, info
    def get_derivatives(self, state, action=None):
        if action is None:
            action = np.zeros(3, dtype=np.float32)
        x, y, z, w = state
        
        dx = self.a * (y - x) + w
        dy = self.d * x - x * z + self.c * y +action[0]
        dz = x * y - self.b * z + action[1]
        dw = y * z + self.r * w + action[2]
        return np.array([dx, dy, dz, dw], dtype=np.float32)

    def rk4_step(self, state, action=0.0):
        # Use RK4 for better numerical stability on chaotic dynamics.
        k1 = self.get_derivatives(state, action=action)
        k2 = self.get_derivatives(state + 0.5 * self.dt * k1, action=action)
        k3 = self.get_derivatives(state + 0.5 * self.dt * k2, action=action)
        k4 = self.get_derivatives(state + self.dt * k3, action=action)
        return state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, action):
        self.current_step += 1
        real_action = np.asarray(action, dtype=np.float32) * self.scale  # 将动作缩放到物理控制力的范围
        x, y, z, w = self.statey
        natural_terms = np.array(
            [
                self.d * x - x * z + self.c * y,
                x * y - self.b * z,
                y * z + self.r * w,
            ],
            dtype=np.float32,
        )
        control_abs = float(np.abs(real_action).sum())
        natural_abs = float(np.abs(natural_terms).sum())
        control_ratio = control_abs / (natural_abs + 1e-6)
        self.natural_dw_abs_sum += natural_abs
        self.control_abs_sum += control_abs
        self.control_ratio_samples.append(control_ratio)
        self.stats_steps += 1
        # 2. 状态更新默认使用欧拉法，便于和前面的实验保持一致。
        x_derivatives = self.get_derivatives(self.statex)
        y_derivatives = self.get_derivatives(self.statey, action=real_action)
        self.statex = self.statex + x_derivatives * self.dt
        self.statey = self.statey + y_derivatives * self.dt

        # RK4 fallback (kept for ablation / reproducibility):
        # self.statex = self.rk4_step(self.statex)
        # self.statey = self.rk4_step(self.statey, action=real_action)
        # 3. 计算新的误
        error = self.statey - self.statex
        error_norm = np.linalg.norm(error, ord=1)
        # 4. 拼装新的观测值 (绝对状态 + 误差)
        obs = self._get_obs()
        self.obs_abs_max_samples.append(float(np.max(np.abs(obs))))
        # 5. 计算奖励：仅保留绝对值误差惩罚
        error_penalty = error_norm
        reward = -np.log1p(error_norm)
        # reward = -(error_norm + 1e-6)**0.5
        self.error_penalty_sum += error_penalty
        # 6. 判断是否结束 (达到最大步数)
        truncated = self.current_step >= self.max_steps
        terminated = False  # 这个环境没有其他的终止条件
        info = {
            "natural_dw_abs": natural_abs,
            "control_abs": control_abs,
            "control_ratio": control_ratio,
            "error_norm": error_norm,
            "error_penalty": error_penalty,
        }
        if truncated and self.stats_steps > 0:
            ratio_array = np.asarray(self.control_ratio_samples, dtype=np.float32)
            obs_abs_max_array = np.asarray(self.obs_abs_max_samples, dtype=np.float32)
            info["scale_diagnostics"] = {
                "avg_natural_dw_abs": self.natural_dw_abs_sum / self.stats_steps,
                "avg_control_abs": self.control_abs_sum / self.stats_steps,
                "global_control_ratio": self.control_abs_sum / (self.natural_dw_abs_sum + 1e-6),
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
            # if self.enable_scale_diagnostics:
            #     stats = info["scale_diagnostics"]
            #     print(
            #         "[scale-check] "
            #         f"scale={self.scale}, "
            #         f"avg|natural_dw|={stats['avg_natural_dw_abs']:.3f}, "
            #         f"avg|u|={stats['avg_control_abs']:.3f}, "
            #         f"global ratio={stats['global_control_ratio']:.3f}, "
            #         f"p50={stats['p50_control_ratio']:.3f}, "
            #         f"p90={stats['p90_control_ratio']:.3f}, "
            #         f"p95={stats['p95_control_ratio']:.3f}, "
            #         f"max ratio={stats['max_control_ratio']:.3f}, "
            #         f"obs|max| avg={stats['avg_obs_abs_max']:.3f}, "
            #         f"p50={stats['p50_obs_abs_max']:.3f}, "
            #         f"p90={stats['p90_obs_abs_max']:.3f}, "
            #         f"max={stats['max_obs_abs_max']:.3f}, "
            #         f"reward parts: err={stats['avg_error_penalty']:.3f}"
            #     )
        return obs, reward, terminated, truncated, info
