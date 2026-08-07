import numpy as np
import gymnasium
from gymnasium import spaces

class HopfieldEnv(gymnasium.Env):
    """
    连续时间 3节点 Hopfield 混沌神经网络环境。
    采用单输入牵制控制 (Pinning Control) 和 RK4 数值积分。
    """
    def __init__(self, mismatch_scale=0.0):
        super().__init__()

        # 1. 纯血 Hopfield 永久混沌矩阵 (已被李雅普诺夫指数严格证明)
        self.W = np.array([
            [  3.4,  -1.6,  0.7 ],
            [  2.5,   0.0,  0.95],
            [-12.5,   0.5,  0.0 ]
        ], dtype=np.float32)

        # 响应系统失配矩阵 (用于测试鲁棒性，训练时 mismatch_scale=0.0)
        self.W_r = self.W + mismatch_scale * np.array([
            [ 0.1, -0.05,  0.02],
            [ 0.05, 0.1,  -0.08],
            [-0.2,  0.05,  0.05]
        ], dtype=np.float32)

        # 2. 积分器与执行器参数
        self.dt = 0.01           # RK4 微小物理步长
        self.rk4_steps = 5       # RL 每输出一次动作，物理世界走 5 步
        self.scale = 25.0        # 动作放大系数 (师兄传下来的物理限幅规则)
        self.max_episode_steps = 400
        self.current_step = 0

        # 3. 核心维度：单输入欠驱动！仅控制节点 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 观测空间：6维 (3个目标状态，3个同步误差)。除以 5.0 防止神经网络饱和
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.statex = np.zeros(3, dtype=np.float32)
        self.statey = np.zeros(3, dtype=np.float32)

    def _get_derivatives(self, state, W_matrix, action=0.0):
        """计算 Hopfield 动力学导数 dx/dt"""
        x1, x2, x3 = state
        v1, v2, v3 = np.tanh(x1), np.tanh(x2), np.tanh(x3)

        # 控制力 action 仅仅砸在 dx1 上！隔山打牛的核心！
        dx1 = -x1 + W_matrix[0, 0]*v1 + W_matrix[0, 1]*v2 + W_matrix[0, 2]*v3 + action
        dx2 = -x2 + W_matrix[1, 0]*v1 + W_matrix[1, 1]*v2 + W_matrix[1, 2]*v3
        dx3 = -x3 + W_matrix[2, 0]*v1 + W_matrix[2, 1]*v2 + W_matrix[2, 2]*v3

        return np.array([dx1, dx2, dx3], dtype=np.float32)

    def _rk4_step(self, state, W_matrix, action=0.0):
        """四阶龙格-库塔数值积分"""
        k1 = self._get_derivatives(state, W_matrix, action)
        k2 = self._get_derivatives(state + 0.5 * self.dt * k1, W_matrix, action)
        k3 = self._get_derivatives(state + 0.5 * self.dt * k2, W_matrix, action)
        k4 = self._get_derivatives(state + self.dt * k3, W_matrix, action)
        return state + (self.dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    def _get_obs(self):
        error = self.statey - self.statex
        return np.concatenate([self.statex, error]).astype(np.float32) / 5.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0
        # 初始状态稍微拉开一点，让系统迅速进入混沌拉伸状态
        self.statex = self.np_random.uniform(low=-1.5, high=1.5, size=(3,)).astype(np.float32)
        self.statey = self.np_random.uniform(low=-1.5, high=1.5, size=(3,)).astype(np.float32)
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # 将神经网络输出的 [-1, 1] 放大为实际物理推力 [-25, 25]
        control_force = float(action[0]) * self.scale
        
        for _ in range(self.rk4_steps):
            self.statex = self._rk4_step(self.statex, self.W, action=0.0)
            self.statey = self._rk4_step(self.statey, self.W_r, action=control_force)

        # 绝对误差 + log1p 惩罚，坚决不加动作惩罚
        error = self.statey - self.statex
        error_norm = float(np.linalg.norm(error, ord=1))
        reward = -float(np.log1p(error_norm))

        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        info = {"error_norm": error_norm}
        return self._get_obs(), float(reward), terminated, truncated, info

# # 注册环境以便 gym.make 调用
# gymnasium.envs.registration.register(
#     id="HopfieldEnv-v0",
#     entry_point=__name__ + ":HopfieldEnv",
#     max_episode_steps=400,
# )