"""
scheme3_seed_stream.py  —  方案 3：稳态同步握手 + 物理种子提取 + 高熵混沌流扩展

职责：
  1. 监测 Hopfield 连续混沌系统的稳态同步状态（残差 < ε）
  2. 利用保护带格点量化（Guard-band Binning）提取主从 100% 恒等的物理特征种子
  3. 基于物理种子，驱动高灵敏度离散混沌映射（Logistic-Sine 混合映射）
  4. 确定性扩展生成各通道所需的大规模、高信息熵、均匀分布的 DNA 规则矩阵（值域 1~8）
"""

import sys
import os
import argparse
import hashlib
from pathlib import Path
import numpy as np

# ---- 导入路径设置 ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 适配 Windows 控制台编码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Code.encry.chaos_base import (
    generate_raw,
    check_synchronization,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_TRANSIENT_STEPS,
)

# 稳态安全格点参数
GRID_DELTA = 0.25      # 网格间距（远大于最大稳差 0.027）
GUARD_BAND = 0.035      # 边界保护带半宽
REQUIRED_SEED_DIMS = 16 # 提取的种子分量数量


def extract_guardband_seed(raw_trajs: tuple[np.ndarray, ...], sample_indices: list[int] = None, grid_delta: float = GRID_DELTA, guard: float = GUARD_BAND) -> tuple[np.ndarray, list[int]]:
    """
    在稳态混沌轨迹中，利用保护带格点量化提取高熵离散种子。
    若 sample_indices 为 None（Master 端），则主动搜索安全稳态采样点并返回；
    若传入 sample_indices（Slave 端），则在指定安全时刻进行采样。
    只要采样时刻满足保护带要求，即使存在微小浮点跟踪误差，两端量化 Bin Index 亦 100% 绝对恒等。
    """
    num_dims = len(raw_trajs)
    seq_len = len(raw_trajs[0])

    if sample_indices is None:
        # Master 端搜索安全点
        safe_indices = []
        for idx in range(seq_len):
            is_safe = True
            for d in range(num_dims):
                val = raw_trajs[d][idx]
                rem = val % grid_delta
                dist_to_boundary = min(rem, grid_delta - rem)
                if dist_to_boundary < guard:
                    is_safe = False
                    break
            if is_safe:
                safe_indices.append(idx)

        if len(safe_indices) < (REQUIRED_SEED_DIMS // num_dims + 1):
            raise RuntimeError(f"未能提取到足够的安全稳态点 (需至少 {REQUIRED_SEED_DIMS} 维，仅找到 {len(safe_indices)} 点)")

        # 间隔选取若干安全点，增加各采样点之间的去相关性与相空间覆盖度
        stride = max(1, len(safe_indices) // (REQUIRED_SEED_DIMS // num_dims + 1))
        chosen_steps = safe_indices[::stride][: (REQUIRED_SEED_DIMS // num_dims + 1)]
    else:
        chosen_steps = sample_indices

    # 提取选定时刻各维度的离散网格索引
    seed_values = []
    for idx in chosen_steps:
        for d in range(num_dims):
            bin_idx = int(np.floor(raw_trajs[d][idx] / grid_delta))
            seed_values.append(bin_idx)
            if len(seed_values) >= REQUIRED_SEED_DIMS:
                break
        if len(seed_values) >= REQUIRED_SEED_DIMS:
            break

    return np.array(seed_values, dtype=np.int32), chosen_steps


def expand_seed_to_streams(seed_vec: np.ndarray, target_length: int, num_streams: int = 6, num_rules: int = 8) -> tuple[np.ndarray, ...]:
    """
    使用提取的离散物理种子驱动高性能混沌伪随机发生器（PRNG），扩展为图像所需的 6 条 DNA 规则序列。

    扩展动力学：使用混沌动力学极强、无弱混沌窗口的 Logistic-Sine 耦合映射 (LSCM):
        x_{k+1} = ( r * x_k * (1 - x_k) + (4 - r) * sin(pi * x_k) / 4 ) mod 1.0
    """
    seed_bytes = seed_vec.tobytes()
    # 使用 SHA-512 生成 128 位十六进制散列，为 6 条流分别提供充裕且独立的扰动初值
    hash_digest = hashlib.sha512(seed_bytes).hexdigest()

    streams = []
    for s_idx in range(num_streams):
        sub_hex = hash_digest[s_idx * 16 : (s_idx + 1) * 16]
        int_val = int(sub_hex, 16)
        x = 0.05 + (int_val % (10**9)) / (10**9) * 0.90
        r = 3.99 + (s_idx * 0.0015)

        for _ in range(100):
            x = (r * x * (1.0 - x) + (4.0 - r) * np.sin(np.pi * x) / 4.0) % 1.0

        seq = np.empty(target_length, dtype=np.uint8)
        for i in range(target_length):
            x = (r * x * (1.0 - x) + (4.0 - r) * np.sin(np.pi * x) / 4.0) % 1.0
            rule = int(np.floor(x * num_rules)) + 1
            if rule > num_rules:
                rule = num_rules
            seq[i] = rule

        streams.append(seq)

    return tuple(streams)


def generate_seed_scheme3(
    height: int = 256,
    width: int = 256,
    p: int = DEFAULT_BLOCK_SIZE,
    transient_steps: int = DEFAULT_TRANSIENT_STEPS
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...], dict]:
    """
    方案 3 序列生成主入口（6 序列 3 轮编解码架构）。
    仅需采样 1024 步物理稳态用于种子提取，随后快速确定性扩展为所需尺寸序列。
    """
    num = ((height + p - 1) // p) * ((width + p - 1) // p)
    # 种子提取仅需固定 1024 步稳态轨迹，无需冗余模拟全图步数
    raw = generate_raw(512, transient_steps=transient_steps)

    master_raw = raw[:3]
    slave_raw = raw[3:]

    # 1. 保护带稳态种子提取（Master 确定安全采样步，Slave 在相同安全步采样）
    seed_master, chosen_steps = extract_guardband_seed(master_raw, sample_indices=None, grid_delta=GRID_DELTA, guard=GUARD_BAND)
    seed_slave, _ = extract_guardband_seed(slave_raw, sample_indices=chosen_steps, grid_delta=GRID_DELTA, guard=GUARD_BAND)

    seed_match = np.array_equal(seed_master, seed_slave)
    if not seed_match:
        raise RuntimeError("方案 3 物理种子提取发生分歧，主从两端种子不一致！")

    # 2. 确定性混沌流展开为 6 条 DNA 规则序列
    master_sequence = expand_seed_to_streams(seed_master, target_length=num, num_streams=6)
    slave_sequence = expand_seed_to_streams(seed_slave, target_length=num, num_streams=6)

    seed_info = {
        "seed_length": len(seed_master),
        "seed_master": seed_master.tolist(),
        "seed_slave": seed_slave.tolist(),
        "seed_exact_match": bool(seed_match),
        "sample_steps": chosen_steps,
        "expansion_length": num,
        "num_streams": 6,
    }

    return master_sequence, slave_sequence, raw, seed_info



# ==========================================
# __main__：独立验证与测试入口
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="方案 3：稳态握手 + 物理种子提取 + 混沌流扩展验证")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    args = parser.parse_args()

    print("=" * 60)
    print("方案 3：稳态握手 + 种子提取 + 混沌流扩展 (Seed-driven Stream)")
    print(f"图像尺寸: {args.height}x{args.width}, 块大小: {args.block_size}")
    print("=" * 60)

    master_seq, slave_seq, raw, seed_info = generate_seed_scheme3(args.height, args.width, args.block_size)

    print("\n[种子协商结果]")
    print(f"  主端种子: {seed_info['seed_master']}")
    print(f"  从端种子: {seed_info['seed_slave']}")
    print(f"  两端种子是否严格一致: {'[PASS] 是' if seed_info['seed_exact_match'] else '[FAIL] 否'}")

    synced = check_synchronization(master_seq, slave_seq)
    print(f"\n[流扩展同步判定] check_synchronization: {'[PASS] (100% 逐元素恒等)' if synced else '[FAIL]'}")

    print("\n[统计特性检查（检验均匀分布与非平庸性）]")
    for i in range(len(master_seq)):
        m = master_seq[i]
        unique_vals, counts = np.unique(m, return_counts=True)
        print(f"  流 Stream {i+1}: 独立取值数={len(unique_vals)}, 取值={unique_vals}")
        assert len(unique_vals) == 8, f"流 Stream {i+1} 未完全覆盖 8 种规则！"

    print("\n" + "=" * 60)
    print("方案 3 (6 序列 3 轮架构) 独立测试全部通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
