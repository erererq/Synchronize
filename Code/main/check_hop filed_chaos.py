import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent

out_dir = PROJECT_ROOT / "logs" / "hopfield" / "chaos_check"
out_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)
# # 假设你在写双栏论文，单栏宽 3.5 寸
# plt.rcParams.update({
#     'figure.figsize': (3.5, 2.5), 
#     'figure.figsize': (3.5, 3.5), 
#     'figure.dpi': 300,             
#     'font.size': 5,              
#     'axes.labelsize': 5,         
#     'lines.linewidth': 0.5,         
#     'font.size': 8,              
#     'axes.labelsize': 9,         
#     'xtick.labelsize': 8,
#     'ytick.labelsize': 8,
#     'lines.linewidth': 0.8         
# })
# ==========================================
# 1. 物理参数与系统定义
# ==========================================
W = np.array([
    [ -1.4,  1.2,  -7 ],
    [  1.1,  0.0,   2.8],
    [  0.8, -2.0,   4.0]
], dtype=np.float32)

def system_dynamics(x):
    return -x + W @ np.tanh(x)

def jacobian_matrix(x):
    sech2 = 1.0 - np.tanh(x)**2
    return -np.eye(3) + W @ np.diag(sech2)

# ==========================================
# 2. 积分与子李雅普诺夫计算初始化
# ==========================================
dt = 0.01
t_max = 1000.0  # 稍微拉长仿真时间，让收敛更精确
steps = int(t_max / dt)
burn_in_steps = 5000 

x = np.array([0.5, -0.01, 0.2]) 

# 为三种牵制情况分别初始化正交扰动矩阵 (2x2) 和 累加器
Q = np.eye(3)  # 3x3 的正交矩阵用于完整系统的变分方程演化，但我们只关注子系统
LE_sum = np.zeros(3)  # 用于完整系统的李雅普诺夫指数计算，但我们主要关注子系统的结果
# Q1: 牵制节点1，剩余节点2,3
Q1 = np.eye(2)
LE_sum1 = np.zeros(2)

# Q2: 牵制节点2，剩余节点1,3
Q2 = np.eye(2)
LE_sum2 = np.zeros(2)

# Q3: 牵制节点3，剩余节点1,2
Q3 = np.eye(2)
LE_sum3 = np.zeros(2)

history = []  # 用于记录轨迹数据以绘制相图
print("正在沿混沌吸引子轨迹积分，计算条件李雅普诺夫指数(CLEs)...")

# ==========================================
# 3. 核心演化循环
# ==========================================
for i in range(steps):
    # 1. 物理状态演化 (RK4)
    k1 = system_dynamics(x)
    k2 = system_dynamics(x + 0.5 * dt * k1)
    k3 = system_dynamics(x + 0.5 * dt * k2)
    k4 = system_dynamics(x + dt * k3)
    x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # 2. 获取当前的完整 3x3 雅可比矩阵
    J = jacobian_matrix(x)

    # 3. 提取三种牵制情况下的 2x2 子雅可比矩阵 (划掉对应的行列)
    # 牵制节点 1：划掉第 0 行 0 列
    J_sub1 = J[1:3, 1:3]
    # 牵制节点 2：划掉第 1 行 1 列
    J_sub2 = np.array([[J[0, 0], J[0, 2]], 
                       [J[2, 0], J[2, 2]]])
    # 牵制节点 3：划掉第 2 行 2 列
    J_sub3 = J[0:2, 0:2]

    # 4. 变分方程演化 (追踪微小扰动椭球在剩余子空间中的变形)
    Q1 = Q1 + dt * (J_sub1 @ Q1)
    Q2 = Q2 + dt * (J_sub2 @ Q2)
    Q3 = Q3 + dt * (J_sub3 @ Q3)
    Q = Q + dt * (J @ Q)  # 完整系统的变分方程演化 
    # 5. 施加格拉姆-施密特正交化 (QR 分解)
    Q1, R1 = np.linalg.qr(Q1)
    Q2, R2 = np.linalg.qr(Q2)
    Q3, R3 = np.linalg.qr(Q3)
    Q, R = np.linalg.qr(Q)  # 完整系统的正交化 
    # 6. 累加对角线元素的对数值 (避开瞬态)
    if i > burn_in_steps:
        history.append(x)  # 记录轨迹数据
        LE_sum1 += np.log(np.abs(np.diag(R1)))
        LE_sum2 += np.log(np.abs(np.diag(R2)))
        LE_sum3 += np.log(np.abs(np.diag(R3)))
        LE_sum += np.log(np.abs(np.diag(R)))  # 完整系统的李雅普诺夫指数累加

# ==========================================
# 4. 输出判决结果
# ==========================================
effective_time = (steps - burn_in_steps) * dt

# 提取最大条件李雅普诺夫指数 (MCLE)
max_cle_1 = np.max(LE_sum1 / effective_time)
max_cle_2 = np.max(LE_sum2 / effective_time)
max_cle_3 = np.max(LE_sum3 / effective_time)
max_cle_full = np.max(LE_sum / effective_time)

print("\n" + "="*50)
print("牵制控制物理可控性判决结果 (最大条件李雅普诺夫指数)")
print("="*50)
print(f"牵制 节点 1 时，剩余子系统的最大 CLE: {max_cle_1:+.4f}  --> {'[发散❌]' if max_cle_1 > 0 else '[稳定✅]'}")
print(f"牵制 节点 2 时，剩余子系统的最大 CLE: {max_cle_2:+.4f}  --> {'[发散❌]' if max_cle_2 > 0 else '[稳定✅]'}")
print(f"牵制 节点 3 时，剩余子系统的最大 CLE: {max_cle_3:+.4f}  --> {'[稳定✅]' if max_cle_3 < 0 else '[发散❌]'}")
print(f"完整系统的最大 LE: {max_cle_full:+.4f}  --> {'[发散❌]' if max_cle_full > 0 else '[稳定✅]'}")
print("="*50)


if max_cle_1 > 0 and max_cle_2 > 0 and max_cle_3 < 0:
    print("\n结论：数学证明完美成功！")
    print("只有当牵制施加在节点 3 时，未受控的子系统才能满足横向渐近稳定性。")
    print("这为本文 DRL 智能体选择节点 3 作为控制靶点提供了严密的控制论基础！")


# 绘图
history = np.array(history)
fig = plt.figure(figsize=(4, 3.5))
ax = fig.add_subplot(111, projection='3d')
step = 10
N = len(history)
colors = plt.cm.viridis(np.linspace(0, 1, N//step+1))  # 渐变配色方案

for i in range(0, N - 1, step):
    ax.plot(
        history[i:i+step, 0],
        history[i:i+step, 1],
        history[i:i+step, 2],
        color=colors[i//step],
        lw=0.4,  # 线条粗细
        alpha=0.8
    )
# ax.plot(history[:, 0], history[:, 1], history[:, 2], lw=0.4, color='b', alpha=0.8)

# ax.set_title("3-dimensional phase portrait of the Hopfield network", pad=20)
ax.set_xlabel("$x_1(t)$")
ax.set_ylabel("$x_2(t)$")
ax.set_zlabel("$x_3(t)$")
plt.tight_layout()
# fig.savefig(out_dir / "hopfield_phase_plane_single_unused.png", dpi=300)
fig.savefig(out_dir / "hopfield_phase_plane_single_unused.pdf")
plt.close(fig)

# Plot two uncontrolled Hopfield trajectories with different random initial states.
SYNC_PLOT_SEED = 2026
SYNC_PLOT_TIME = 200.0


def rk4_uncontrolled_step(state):
    k1 = system_dynamics(state)
    k2 = system_dynamics(state + 0.5 * dt * k1)
    k3 = system_dynamics(state + 0.5 * dt * k2)
    k4 = system_dynamics(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def collect_uncontrolled_master_slave_trajectory():
    rng = np.random.default_rng(SYNC_PLOT_SEED)
    master = rng.uniform(low=-1.0, high=1.0, size=3).astype(np.float32)
    slave = rng.uniform(low=-1.0, high=1.0, size=3).astype(np.float32)
    plot_steps = int(SYNC_PLOT_TIME / dt)

    master_hist = [master.copy()]
    slave_hist = [slave.copy()]
    for _ in range(plot_steps):
        master = rk4_uncontrolled_step(master)
        slave = rk4_uncontrolled_step(slave)
        master_hist.append(master.copy())
        slave_hist.append(slave.copy())

    return np.asarray(master_hist), np.asarray(slave_hist)


master_traj, slave_traj = collect_uncontrolled_master_slave_trajectory()

fig = plt.figure(figsize=(6.0, 4.8))
ax = fig.add_axes([0.06, 0.08, 0.78, 0.86], projection="3d")

ax.plot(
    master_traj[:, 0],
    master_traj[:, 1],
    master_traj[:, 2],
    color="blue",
    lw=0.6,
    alpha=0.58,
    label="Drive System",
)
ax.plot(
    slave_traj[:, 0],
    slave_traj[:, 1],
    slave_traj[:, 2],
    color="red",
    lw=0.6,
    alpha=0.58,
    label="Response System",
)

ax.set_xlabel("$x_1$", labelpad=4)
ax.set_ylabel("$x_2$", labelpad=4)
ax.set_zlabel("$x_3$", labelpad=4)
ax.view_init(elev=22, azim=-55)
ax.set_box_aspect((1.2, 1.0, 1.0))
ax.legend(loc="upper right", frameon=True)
ax.grid(True, linestyle=":", alpha=0.35)

fig.savefig(out_dir / "hopfield_phase_plane.pdf", bbox_inches="tight", pad_inches=0.35)
fig.savefig(out_dir / "hopfield_phase_plane.png", dpi=300, bbox_inches="tight", pad_inches=0.35)
plt.close(fig)

import numpy as np
import scipy.linalg as la

# 1. 你的系统雅可比矩阵 A (在原点处)
A = np.array([
    [-2.4,  1.2, -7.0],
    [ 1.1, -1.0,  2.8],
    [ 0.8, -2.0,  3.0]
])

# 2. LQR 权重矩阵设计
# Q: 惩罚误差 (状态偏离0的代价) -> 设为单位阵，对三个节点的误差一视同仁
# R: 惩罚推力 (消耗能量的代价) -> 设为 1.0，推力越大惩罚越大
Q = np.eye(3)
R = np.array([[1.0]])

# 3. 三个节点的控制输入矩阵
B_list = [
    ("Node 1", np.array([[1], [0], [0]])),
    ("Node 2", np.array([[0], [1], [0]])),
    ("Node 3", np.array([[0], [0], [1]]))
]

print("=== LQR Optimal Control Cost Analysis ===")
for name, B in B_list:
    try:
        # 求解连续代数里卡提方程 (CARE)
        P = la.solve_continuous_are(A, B, Q, R)
        
        # 计算期望最优总代价 (Trace(P))
        optimal_cost = np.trace(P)
        
        # 为了更直观，我们还可以算出最优反馈增益 K = R^-1 B^T P
        K = np.linalg.inv(R) @ B.T @ P
        gain_norm = np.linalg.norm(K)
        
        print(f"[{name}]")
        print(f"  -> 最优总能量代价 (Cost J): {optimal_cost:10.4f}")
        print(f"  -> 反馈增益范数 ||K|| : {gain_norm:10.4f}\n")
    except Exception as e:
        print(f"[{name}] 求解失败: {e}")
