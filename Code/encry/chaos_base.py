"""
chaos_base.py  —  混沌动力学轨迹采集与通用工具

职责：
  1. 加载预训练 PPO 模型并驱动 ContinuousHopfieldEnv 环境
  2. 采集主/从系统原始连续状态轨迹
  3. 提供通用的序列同步性校验函数
"""

import sys
from pathlib import Path
import numpy as np

# ---- 导入路径设置 ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Code.env.continuous_hopfield_env import ContinuousHopfieldEnv
from stable_baselines3 import PPO

# ==========================================
# 常量配置
# ==========================================

# 预训练模型路径（相对于项目根目录）
MODEL_DIR = "models/hopfield/ppo_continuous_hopfield_node3_final_seed_42.zip"

# 模型对应的牵制控制节点（node3 → pinning_node=2）
PINNING_NODE = 2

# 默认图像分块大小
DEFAULT_BLOCK_SIZE = 8

# 丢弃的瞬态步数
DEFAULT_TRANSIENT_STEPS = 1500


def get_sequence_pairs(env) -> list:
    """
    从环境中提取 4 对 [master, slave] 状态值。
    适配旧版 Lorenz 与新版 Hopfield 环境（3D 系统自动非线性交叉合成第 4 维）。
    """
    legacy_methods = ("get_current", "get_current1", "get_current2", "get_current3")
    if all(hasattr(env, method) for method in legacy_methods):
        return [getattr(env, method)() for method in legacy_methods]

    if hasattr(env, "statex") and hasattr(env, "statey"):
        master_state = np.asarray(env.statex, dtype=np.float32).reshape(-1)
        slave_state = np.asarray(env.statey, dtype=np.float32).reshape(-1)
        pair_count = min(len(master_state), len(slave_state))
        pairs = [[master_state[idx], slave_state[idx]] for idx in range(pair_count)]

        # 3D 系统：用非线性交叉项合成第 4 维
        if pair_count == 3:
            x4_master = master_state[0] * master_state[1] + master_state[2]
            x4_slave = slave_state[0] * slave_state[1] + slave_state[2]
            pairs.append([x4_master, x4_slave])

        if len(pairs) >= 4:
            return pairs[:4]

    raise AttributeError(
        "Environment must expose get_current methods or statex/statey arrays with >= 3 dims."
    )


def generate_raw(
    num: int,
    transient_steps: int = DEFAULT_TRANSIENT_STEPS,
    model_path: str = None,
    pinning_node: int = PINNING_NODE,
    seed: int = None
) -> tuple[np.ndarray, ...]:
    """
    运行 PPO 模型驱动 Hopfield 环境，采集原始连续混沌状态轨迹。

    Args:
        num: 需要的有效序列点数。
        transient_steps: 瞬态丢弃步数（默认 1500）。
        model_path: 模型路径，默认使用 MODEL_DIR。
        pinning_node: 牵制控制节点（0, 1, 2），默认 2。
        seed: 环境随机种子。

    Returns:
        8 个 ndarray：(raw_x1..raw_x4 为主系统，raw_x5..raw_x8 为从系统)。
    """
    resolved_model = model_path or str(PROJECT_ROOT / MODEL_DIR)
    env = ContinuousHopfieldEnv(pinning_node=pinning_node)
    model = PPO.load(resolved_model, device="cpu")

    list_obs = [[] for _ in range(8)]
    reset_result = env.reset(seed=seed)
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    total_steps = num + transient_steps
    for i in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, dones, info = step_result

        x1, x2, x3, x4 = get_sequence_pairs(env)

        if i >= transient_steps:
            list_obs[0].append(x1[0])
            list_obs[1].append(x2[0])
            list_obs[2].append(x3[0])
            list_obs[3].append(x4[0])
            list_obs[4].append(x1[1])
            list_obs[5].append(x2[1])
            list_obs[6].append(x3[1])
            list_obs[7].append(x4[1])

    return tuple(np.asarray(seq, dtype=np.float32) for seq in list_obs)


def check_synchronization(master_sequence: tuple, slave_sequence: tuple) -> bool:
    """
    检查主序列和从序列是否完全同步（逐元素相等）。
    """
    if len(master_sequence) != len(slave_sequence):
        return False
    for x_i, y_i in zip(master_sequence, slave_sequence):
        if not np.array_equal(x_i, y_i):
            return False
    return True
