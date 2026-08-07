import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ContinuousHopfieldEnv(gym.Env):
    """
    3节点连续 Hopfield 混沌神经网络同步环境 (单输入牵制控制)。
    包含绝对误差与分数阶混合奖励，以及动作限幅与放缩机制。
    """
    
    def __init__(
        self,
        mismatch_scale: float = 0.0,
        pulse_step: int | None = None,
        pulse_vector: tuple[float, float, float] = (0.0, 0.0, 0.0),
        pulse_width: int = 1,
        pinning_node: int = 0,
        process_noise_std: float = 0.0
    ):
        super().__init__()

        # ==========================================
        # 1. 物理引擎基础参数
        # ==========================================
        self.dt = 0.01
        self.rk4_steps = 5
        self.max_steps = 400
        self.current_step = 0
        self.mismatch_scale = mismatch_scale
        self.B = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 控制输入矩阵 
        self.B[pinning_node] = 1.0  # 仅对指定节点施加控制输入
        # ==========================================
        # 2. 动作空间与推力映射
        # ==========================================
        self.scale = 4.0  # 物理推力真实放大倍数
        # 神经网络输出归一化范围设定为 [-1, 1]，匹配实际的激活空间与探索裕度
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.process_noise_std = process_noise_std

        # ==========================================
        # 3. 观测空间 (状态 + 同步误差)
        # ==========================================
        # 6维：3个驱动目标状态 + 3个同步误差 (除以 5.0 以防神经网络输入饱和)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # ==========================================
        # 4. 动力学权重矩阵定义
        # ==========================================
        self.W = np.array([
            [-1.4,  1.2, -7.0],
            [ 1.1,  0.0,  2.8],
            [ 0.8, -2.0,  4.0],
        ], dtype=np.float32)

        # # 响应系统失配矩阵 (用于测试控制算法的鲁棒性)
        # self.W_r = self.W + mismatch_scale * np.array([
        #     [ 0.1, -0.05,  0.02],
        #     [ 0.05,  0.1, -0.08],
        #     [-0.2,  0.05,  0.05]
        # ], dtype=np.float32)

        # ==========================================
        # 5. 脉冲干扰配置
        # ==========================================
        self.pulse_step = pulse_step
        self.pulse_width = max(int(pulse_width), 1)
        self.pulse_vector = np.asarray(pulse_vector, dtype=np.float32)

        # 状态初始化占位
        self.statex = np.zeros(3, dtype=np.float32)
        self.statey = np.zeros(3, dtype=np.float32)
        self._reset_scale_stats()

    def _reset_scale_stats(self):
        """重置诊断统计量 (用于监控评估回调)"""
        self.natural_abs_sum = 0.0
        self.control_abs_sum = 0.0
        self.control_ratio_samples = []
        self.error_sum = 0.0
        self.stats_steps = 0

    def _get_derivatives(self, state, W_matrix, B, control_force=0.0):
        """
        核心动力学方程 (优化：使用矩阵乘法代替逐元素展开，大幅提升运算效率)
        """
        x = np.tanh(state)
        dx = -state + W_matrix @ x
        # 牵制控制：外部控制力 U(t) 仅注入到指定节点
        dx += B*control_force 
        return dx.astype(np.float32)

    def _rk4_step(self, state, W_matrix, B, control_force=0.0):
        """标准的四阶龙格-库塔数值积分 (RK4)"""
        k1 = self._get_derivatives(state, W_matrix, B, control_force)
        k2 = self._get_derivatives(state + 0.5 * self.dt * k1, W_matrix, B, control_force)
        k3 = self._get_derivatives(state + 0.5 * self.dt * k2, W_matrix, B, control_force)
        k4 = self._get_derivatives(state + self.dt * k3, W_matrix, B, control_force)
        
        next_state = state + (self.dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        # 物理限幅：防止在极度失配或探索初期产生数值溢出
        return np.clip(next_state, -10.0, 10.0).astype(np.float32)

    def _get_obs(self):
        """获取归一化后的环境观测值"""
        error = self.statey - self.statex
        x1, x2, x3 = self.statex
        e1, e2, e3 = error
        return np.array([x1,x2,x3,e1,e2,e3]).astype(np.float32) / 5.0

    def reset(self, seed=None, options=None):
        """环境重置"""
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self._reset_scale_stats()

        # ==========================================
        # 动态生成基于物理拓扑的随机失配矩阵 (3% 容差)
        # ==========================================
        if self.mismatch_scale > 0.0:
            # 产生 [-0.03, 0.03] 之间的均匀分布随机扰动矩阵
            random_perturbation = self.np_random.uniform(
                low=-0.03, high=0.03, size=self.W.shape
            ).astype(np.float32)
            
            # 使用哈达玛乘积 (对应元素相乘)，保证 0 权重依然为 0
            self.W_r = self.W + self.mismatch_scale * (self.W * random_perturbation)
        else:
            # 理想情况，无失配
            self.W_r = self.W.copy()
        
        # 将初始状态随机散布在 [-1, 1] 的相空间中
        self.statex = self.np_random.uniform(low=-1.0, high=1.0, size=(3,)).astype(np.float32)
        self.statey = self.np_random.uniform(low=-1.0, high=1.0, size=(3,)).astype(np.float32)
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # ==========================================
        # 1. 解析动作并施加控制
        # ==========================================
        control_force = float(action[0]) * self.scale
        
        # 诊断：计算控制力与自然演化项的比例
        natural_terms = self._get_derivatives(self.statey, self.W_r, self.B, control_force=0.0)
        natural_abs = float(np.abs(natural_terms).sum())
        control_abs = float(abs(control_force))
        
        self.natural_abs_sum += natural_abs
        self.control_abs_sum += control_abs
        self.control_ratio_samples.append(control_abs / (natural_abs + 1e-6))
        self.stats_steps += 1

        # ==========================================
        # 2. 物理世界时间演化
        # ==========================================
        for _ in range(self.rk4_steps):
            self.statex = self._rk4_step(self.statex, self.W, self.B, control_force=0.0)
            self.statey = self._rk4_step(self.statey, self.W_r, self.B, control_force=control_force)

        if self.process_noise_std > 0.0:
            noise = self.np_random.normal(
                loc=0.0,
                scale=self.process_noise_std,
                size=self.statey.shape
            ).astype(np.float32)
            self.statey = np.clip(self.statey + noise, -10.0, 10.0)

        # 注入外部脉冲干扰 (若在指定步数内)
        pulse_applied = False
        if self.pulse_step is not None:
            if self.pulse_step <= self.current_step < self.pulse_step + self.pulse_width:
                self.statex = np.clip(self.statex + self.pulse_vector, -10.0, 10.0)
                pulse_applied = True

        # ==========================================
        # 3. 误差评估与复合奖励计算
        # ==========================================
        error = self.statey - self.statex
        error_norm = float(np.linalg.norm(error, ord=1))
        self.error_sum += error_norm
        
        # 核心奖励机制：绝对误差 
        reward = -0.1 * error_norm

        # 检查终止状态
        terminated = False
        truncated = self.current_step >= self.max_steps

        # ==========================================
        # 4. 组装评估信息 (Info Dict)
        # ==========================================
        info = {
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