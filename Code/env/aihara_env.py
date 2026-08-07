import gymnasium
import numpy as np


# Previous version kept here for quick rollback during reward/actuation ablations.
# class AiharaEnv(gymnasium.Env):
#     """Drive-response synchronization task built from two Aihara-type neurons."""
#
#     def __init__(
#         self,
#         mismatch_scale: float = 1.0,
#         pulse_step: int | None = None,
#         pulse_vector: tuple[float, float] = (0.0, 0.0),
#         pulse_width: int = 1,
#     ):
#         super().__init__()
#
#         self.k = 0.92
#         self.alpha = 0.78
#         self.beta = 0.12
#         self.bias = 0.18
#         self.epsilon = 0.08
#
#         self.mismatch_scale = mismatch_scale
#         self.k_r = self.k - 0.02 * mismatch_scale
#         self.alpha_r = self.alpha + 0.02 * mismatch_scale
#         self.beta_r = self.beta - 0.01 * mismatch_scale
#         self.bias_r = self.bias - 0.03 * mismatch_scale
#
#         self.coupling = 0.08
#         self.scale = 0.25
#         self.max_steps = 400
#         self.current_step = 0
#
#         self.pulse_step = pulse_step
#         self.pulse_width = max(int(pulse_width), 1)
#         self.pulse_vector = np.asarray(pulse_vector, dtype=np.float32)
#         self._pulse_active = False
#
#         self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
#         self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
#
#         self.statex = np.zeros(2, dtype=np.float32)
#         self.statey = np.zeros(2, dtype=np.float32)
#
#     def _activation(self, value: float) -> float:
#         value = np.clip(value / self.epsilon, -60.0, 60.0)
#         return float(1.0 / (1.0 + np.exp(-value)))
#
#     def _drive_step(self, state: np.ndarray) -> np.ndarray:
#         x, y = state
#         fx = self._activation(float(x))
#         x_next = self.k * x - self.alpha * y + self.bias
#         y_next = y + self.beta * (fx - y)
#         return np.array([x_next, y_next], dtype=np.float32)
#
#     def _response_step(self, drive_state: np.ndarray, response_state: np.ndarray, control: np.ndarray) -> np.ndarray:
#         x_d, y_d = drive_state
#         x_r, y_r = response_state
#         fx_r = self._activation(float(x_r))
#
#         x_next = self.k_r * x_r - self.alpha_r * y_r + self.bias_r
#         x_next += self.coupling * (x_d - x_r) + control[0]
#
#         y_next = y_r + self.beta_r * (fx_r - y_r)
#         y_next += 0.35 * self.coupling * (y_d - y_r) + control[1]
#
#         return np.array([x_next, y_next], dtype=np.float32)
#
#     def _get_obs(self) -> np.ndarray:
#         error = self.statey - self.statex
#         return np.concatenate([self.statex, error]).astype(np.float32) / 3.0
#
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed, options=options)
#         self.current_step = 0
#         self._pulse_active = False
#         self.statex = self.np_random.uniform(low=-0.6, high=0.6, size=(2,)).astype(np.float32)
#         self.statey = self.np_random.uniform(low=-0.6, high=0.6, size=(2,)).astype(np.float32)
#         return self._get_obs(), {}
#
#     def step(self, action):
#         self.current_step += 1
#         control = np.asarray(action, dtype=np.float32) * self.scale
#
#         next_drive = self._drive_step(self.statex)
#         next_response = self._response_step(self.statex, self.statey, control)
#
#         pulse_now = False
#         if self.pulse_step is not None:
#             pulse_start = self.pulse_step
#             pulse_end = self.pulse_step + self.pulse_width
#             if pulse_start <= self.current_step < pulse_end:
#                 next_drive = next_drive + self.pulse_vector
#                 pulse_now = True
#
#         self.statex = np.clip(next_drive, -5.0, 5.0)
#         self.statey = np.clip(next_response, -5.0, 5.0)
#
#         error = self.statey - self.statex
#         error_norm = float(np.linalg.norm(error, ord=1))
#         control_penalty = 0.02 * float(np.linalg.norm(control, ord=1))
#         reward = -np.log1p(error_norm) - control_penalty
#
#         terminated = False
#         truncated = self.current_step >= self.max_steps
#         info = {
#             "error_norm": error_norm,
#             "control_l1": float(np.linalg.norm(control, ord=1)),
#             "mismatch_scale": self.mismatch_scale,
#             "pulse_applied": pulse_now,
#         }
#         return self._get_obs(), reward, terminated, truncated, info


class AiharaEnv(gymnasium.Env):
    """
    基于离散 Aihara 混沌神经元的驱动-响应同步环境。
    支持参数失配测试 (Parameter Mismatch) 与 脉冲干扰测试 (Pulse Interference)。
    """

    def __init__(
        self,
        mismatch_scale: float = 0.0,  # 失配程度乘子，设为0即为完美匹配
        pulse_step: int | None = None, # 在第几步注入脉冲干扰
        pulse_vector: tuple[float, float] = (0.0, 0.0), # 脉冲的强度向量
        pulse_width: int = 1, # 脉冲持续的步数
    ):
        super().__init__()

        # ==========================================
        # 1. 驱动系统 (Drive Neuron) 的标称参数
        # ==========================================
        self.k = 0.92        # 内部状态阻尼衰减系数
        self.alpha = 0.78    # 不应期标度参数 (产生混沌的核心参数)
        self.beta = 0.12     # 耦合强度
        self.bias = 0.18     # 外部恒定刺激
        self.epsilon = 0.08  # Sigmoid 函数的陡峭度

        # ==========================================
        # 2. 响应系统 (Response Neuron) 的失配参数
        # ==========================================
        # 通过 mismatch_scale 制造出与驱动系统不同的硬件参数，用于测试算法的鲁棒性
        self.mismatch_scale = mismatch_scale
        self.k_r = self.k - 0.02 * mismatch_scale
        self.alpha_r = self.alpha + 0.02 * mismatch_scale
        self.beta_r = self.beta - 0.01 * mismatch_scale
        self.bias_r = self.bias - 0.03 * mismatch_scale

        # ==========================================
        # 3. 强化学习与物理执行器配置
        # ==========================================
        # 物理推力放缩：网络输出限幅在 ±1，实际作用于神经元的力放大 25 倍
        # RL actuator settings.
        self.scale = 50.0
        self.max_steps = 400
        self.current_step = 0
        action_limit: float = 1.0  # RL 输出动作的绝对值上限


        # Pulse disturbance settings.
        # 脉冲干扰配置
        self.pulse_step = pulse_step
        self.pulse_width = max(int(pulse_width), 1)
        self.pulse_vector = np.asarray(pulse_vector, dtype=np.float32)
        self._pulse_active = False

        self.action_dim = 1
        self.action_space = gymnasium.spaces.Box(low=-action_limit, high=action_limit, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.statex = np.zeros(2, dtype=np.float32)
        self.statey = np.zeros(2, dtype=np.float32)

    def _activation(self, value: float) -> float:
        """Aihara 神经元的 Sigmoid 激活函数，模拟轴突输出"""
        value = np.clip(value / self.epsilon, -60.0, 60.0)
        return float(1.0 / (1.0 + np.exp(-value)))

    def _drive_step(self, state: np.ndarray) -> np.ndarray:
        """驱动系统演化 (只有物理自身规律，绝不允许有控制输入)"""
        x, y = state
        fx = self._activation(float(x))
        x_next = self.k * x - self.alpha * y + self.bias
        y_next = y + self.beta * (fx - y)
        return np.array([x_next, y_next], dtype=np.float32)

    def _response_step(self, response_state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """响应系统演化 (带有失配参数，并且受到强化学习输出的强推力控制)"""
        x_r, y_r = response_state
        fx_r = self._activation(float(x_r))
        # 彻底去除了作弊的线性物理耦合，系统的演化 100% 依赖自身参数和外部 RL 控制

        x_next = self.k_r * x_r - self.alpha_r * y_r + self.bias_r + control[0]
        y_next = y_r + self.beta_r * (fx_r - y_r) 

        return np.array([x_next, y_next], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """构建观测向量"""
        error = self.statey - self.statex
        # 观测包含驱动状态和当前的误差，/3.0 是为了做一个极其简单的粗糙归一化
        return np.concatenate([self.statex, error]).astype(np.float32) / 3.0

    def reset(self, seed=None, options=None):
        """回合重置"""
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self._pulse_active = False
        # 初始状态在相空间内随机生成
        self.statex = self.np_random.uniform(low=-0.6, high=0.6, size=(2,)).astype(np.float32)
        self.statey = self.np_random.uniform(low=-0.6, high=0.6, size=(2,)).astype(np.float32)
        return self._get_obs(), {}

    def step(self, action):
        """核心交互逻辑"""
        self.current_step += 1
        
        # 1. 执行器放缩：将算法输出的 ±2 映射为实际的物理强推力 ±50
        control = np.asarray(action, dtype=np.float32) * self.scale

        # 2. 状态演化
        next_drive = self._drive_step(self.statex)
        next_response = self._response_step(self.statey, control)

        # 3. 脉冲干扰注入 (如果到达设定的步数，硬生生把驱动系统砸偏)
        pulse_now = False
        if self.pulse_step is not None:
            pulse_start = self.pulse_step
            pulse_end = self.pulse_step + self.pulse_width
            if pulse_start <= self.current_step < pulse_end:
                next_drive = next_drive + self.pulse_vector
                pulse_now = True

        # 4. 状态更新与物理防御性截断
        self.statex = np.clip(next_drive, -5.0, 5.0)
        self.statey = np.clip(next_response, -5.0, 5.0)

        error = self.statey - self.statex
        error_norm = float(np.linalg.norm(error, ord=1))
        reward = -np.log1p(error_norm)

        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {
            "error_norm": error_norm,
            "mismatch_scale": self.mismatch_scale,
            "pulse_applied": pulse_now,
        }
        return self._get_obs(), float(reward), terminated, truncated, info
