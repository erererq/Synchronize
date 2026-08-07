from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Aihara parameters from env/aihara_env.py
K = 0.7        # 降低惯性，释放高频震荡
ALPHA = 1.0    # 强大的自身不应期惩罚
BETA = 0.05     
BIAS = 0.35    # 经典的混沌诱发偏置
EPSILON = 0.02 # 恢复极度陡峭的非线性折叠

STEPS = 5000
BURN_IN = 500

def activation(value: float) -> float:
    value = np.clip(value / EPSILON, -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-value)))

def aihara_step(state: np.ndarray) -> np.ndarray:
    x, y = state
    fx = activation(float(x))
    fy = activation(float(y))
    
    # 捕食者-猎物不对称耦合：注意 y_next 里的加号 (+)
    # X 被 Y 抑制 (-BETA)
    x_next = K * x - ALPHA * fx - BETA * fy + BIAS
    # Y 被 X 刺激 (+BETA)
    y_next = K * y - ALPHA * fy + BETA * fx + BIAS
    
    return np.array([x_next, y_next], dtype=np.float64)

def simulate(state0: np.ndarray, steps: int) -> np.ndarray:
    state = state0.astype(np.float64).copy()
    history = np.zeros((steps, 2), dtype=np.float64)
    for i in range(steps):
        history[i] = state
        state = aihara_step(state)
    return history

def plot_chaos_verification(traj_ss: np.ndarray, traj1: np.ndarray, traj2: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Time series
    t_ss = np.arange(len(traj_ss))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t_ss, traj_ss[:, 0], color="tab:blue", lw=0.8)
    axes[0].set_ylabel("x(t)")
    axes[0].set_title("Aihara System Time Series (Steady State)")
    axes[1].plot(t_ss, traj_ss[:, 1], color="tab:orange", lw=0.8)
    axes[1].set_ylabel("y(t)")
    axes[1].set_xlabel("time step")
    fig.tight_layout()
    fig.savefig(out_dir / "aihara_time_series.png", dpi=150)
    plt.close(fig)

    # 2. Phase plane
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # [可视化修正] 离散混沌系统千万不要用线(line)连起来，必须用极小的散点(scatter)画！
    ax.plot(traj_ss[:, 0], traj_ss[:, 1], '.', markersize=1.5, color="black", alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Phase Portrait: x-y Attractor")
    fig.tight_layout()
    fig.savefig(out_dir / "aihara_phase_plane.png", dpi=150)
    plt.close(fig)

    # 3. Sensitivity to Initial Conditions
    dist = np.linalg.norm(traj1 - traj2, axis=1)
    t_dist = np.arange(len(dist))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t_dist, dist, color="tab:red")
    axes[0].set_ylabel("Euclidean distance")
    axes[0].set_title("Sensitivity to Initial Conditions")

    axes[1].plot(t_dist, np.log(dist + 1e-12), color="tab:purple")
    axes[1].set_ylabel("log(distance)")
    axes[1].set_xlabel("time step")
    fig.tight_layout()
    fig.savefig(out_dir / "aihara_initial_sensitivity.png", dpi=150)
    plt.close(fig)

def main():
    # 强行打破初始对称性！
    state1 = np.array([0.1, 0.15]) 
    state2 = state1 + np.array([1e-6, 0.0])
    
    traj1 = simulate(state1, STEPS)
    traj2 = simulate(state2, STEPS)
    traj_ss = traj1[BURN_IN:]
    
    out_dir = Path("logs") / "aihara" / "chaos_check"
    plot_chaos_verification(traj_ss, traj1, traj2, out_dir)
    
    print(f"Finished. Plots saved to: {out_dir}")

import numpy as np

# --- 在你原有的代码中增加激活函数的导数计算 ---
def activation_derivative(value: float) -> float:
    # 导数公式: f'(x) = f(x) * (1 - f(x)) / EPSILON
    fx = activation(value)
    return (fx * (1.0 - fx)) / EPSILON

# --- 核心：计算当前状态下的局部撕裂力（雅可比矩阵） ---
# def get_jacobian(state: np.ndarray) -> np.ndarray:
#     x, y = state
#     df_dx = activation_derivative(float(x))
#     df_dy = activation_derivative(float(y))
    
#     # 按照刚才推导的数学公式构建 2x2 矩阵
#     J = np.array([
#         [K - ALPHA * df_dx,  -BETA * df_dy],
#         [-BETA * df_dx,      K - ALPHA * df_dy]
#     ], dtype=np.float64)
#     return J
def get_jacobian(state: np.ndarray) -> np.ndarray:
    x, y = state
    df_dx = activation_derivative(float(x))
    df_dy = activation_derivative(float(y))
    
    # 严格匹配上面方程的偏导数
    J = np.array([
        [K - ALPHA * df_dx,  -BETA * df_dy],
        [BETA * df_dx, K - ALPHA * df_dy]
    ], dtype=np.float64)
    return J

# --- 终极裁判：计算最大李雅普诺夫指数 (MLE) ---
def calculate_mle(state0: np.ndarray, steps: int = 10000, burn_in: int = 2000) -> float:
    state = state0.astype(np.float64).copy()
    
    # 1. 烧机阶段：让系统先跑进吸引子里，排除初始位置的干扰
    for _ in range(burn_in):
        state = aihara_step(state)
        
    # 2. 准备一个切空间的探测向量 (随便什么方向都可以，这里取 [1, 0])
    v = np.array([1.0, 0.0], dtype=np.float64)
    v = v / np.linalg.norm(v) # 初始归一化
    
    sum_log_expansion = 0.0
    
    # 3. 开始在切空间里追踪撕裂率
    for _ in range(steps):
        # 拿到当前的局部撕裂力矩阵
        J = get_jacobian(state)
        
        # 让探测向量被系统撕裂/挤压一次
        v_next = J @ v
        
        # 计算它被拉伸了多少倍
        expansion_factor = np.linalg.norm(v_next)
        
        # 极度危险的防御机制：防止坍缩到 0 导致 log 报错
        if expansion_factor < 1e-12:
            expansion_factor = 1e-12
            
        # 把这一步的指数发散率记在总账本上
        sum_log_expansion += np.log(expansion_factor)
        
        # 【Benettin 核心Hack】：把向量强行缩回长度 1，防止浮点数溢出！
        v = v_next / expansion_factor
        
        # 系统往前走一步
        state = aihara_step(state)
        
    # 计算平均每一步的指数发散率
    mle = sum_log_expansion / steps
    return mle

if __name__ == "__main__":
    main()
# 你可以在 main 函数里这样调用：
    mle_value = calculate_mle(np.array([0.1, 0.1]))
    print(f"Maximum Lyapunov Exponent: {mle_value:.5f}")
    # if mle_value > 0:
    #     print("结论：系统处于纯正的混沌状态 (Chaos)！")
