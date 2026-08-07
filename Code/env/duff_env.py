import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium

class DuffEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()
        # 1. 系统参数 (Duff 系统的典型参数)
        self.alpha = -1.0 # 线性弹簧系数线性刚度 负的线性刚度配合正的三次非线性刚度，在物理上硬生生造出了两个“势阱”（你可以想象成两个相邻的坑）。
        self.beta = 1.0 # 非线性刚度 控制系统在大位移时的恢复力，防止状态变量跑向无穷大。
        self.delta = 0.3 # 阻尼系数 阻尼会消耗系统的能量，防止系统无限制地振荡。
        self.gamma = 0.5 # 驱动力幅值 这个参数控制外部周期性驱动力的强度，过大或过小都会影响系统的混沌特性。
        self.omiga = 1.2 # 驱动力频率 这个参数控制外部周期性驱动力的频率，和系统的自然频率之间的关系会影响系统的动力学行为。
        self.dt = 0.001 # 时间步长 这个参数决定了数值积分的精度和稳定性，过大可能会导致数值不稳定，过小则会增加计算成本。
        self.max_steps = 2000
        self.current_step = 0
        # 2. 动作空间的“黑魔法”缩放因子
        # 真实物理控制力将是 action * self.scale
        self.scale = 50.0 # 这个值需要根据系统的自然演化尺度来调整，过大或过小都会让训练变得困难
        # 3. 动作空间定义 (Action Space)
        # 哲学：单变量控制 (shape=(1,))，且范围永远限定在 [-1.0, 1.0]
        self.action = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # 4. 观测空间定义 (Observation Space)
        # 哲学：智能体需要看到驱动系统(x)和响应系统(y)之间的【误差】，这样它才知道该往哪用力
        # 误差 e = y - x，因为是二阶系统，所以观测值是 4 维$[\, e_1, e_2, \cos(\omega t), \sin(\omega t) \,]$
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.statex = self.np_random.uniform(low=-2.0, high=2.0, size=(2,))
        self.statey = self.np_random.uniform(low=-2.0, high=2.0, size=(2,))
        self.current_step = 0
        # self.time = 0.0
        # period = 2 * np.pi / 1.2
        # self.t = self.np_random.uniform(low=0.0, high=period) 
        self.theta = self.np_random.uniform(low=0.0, high=2*np.pi)
        # 然后在构建传给 Agent 的观测空间 (Observation) 时，带上此时的误差和相位
        e1 = self.statey[0] - self.statex[0]
        e2 = self.statey[1] - self.statex[1]
        obs = np.array([e1, e2, np.cos(self.theta), np.sin(self.theta)], dtype=np.float32)
        return obs, {}
    
    def get_derivatives(self, state, theta, action=None):
        if action is None:
            action = np.array([0.0], dtype=np.float32)

        x, y = state
        dx = y
        dy = -self.delta * y - self.alpha * x - self.beta * x**3 + self.gamma * np.cos( theta) + action[0]
        return np.array([dx, dy], dtype=np.float32)
    
    def step(self, action):
        self.current_step += 1
        real_action = np.asarray(action, dtype=np.float32) * self.scale  # 将动作缩放到物理控制力的范围
        x1, y1 = self.statex
        x2, y2 = self.statey
        # Duff 系统的动力学方程
        derivative_x = self.get_derivatives(self.statex, self.theta)
        derivative_y = self.get_derivatives(self.statey, self.theta, action=real_action)
        # 使用欧拉法更新状态
        self.statex += derivative_x * self.dt
        self.statey += derivative_y * self.dt
        self.theta += self.omiga * self.dt  # 更新驱动力的相位
        # 计算误差和奖励
        error = self.statey - self.statex

        obs = np.array([error[0], error[1], np.cos(self.theta), np.sin(self.theta)], dtype=np.float32)
        error_norm = np.linalg.norm(error, ord=1)

        reward = -np.log1p(error_norm)

        terminated = False
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}


