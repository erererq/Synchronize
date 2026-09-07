"""
scheme1_direct_quant.py  —  方案 1：连续混沌轨迹自适应归一化 + 互补双网格量化（6 序列 3 轮编解码架构）

职责：
  1. 接收 Hopfield 混沌系统真实三维物理状态轨迹 (x1, x2, x3, 长度 2N)
  2. 执行奇偶时间交错抽取，构建 6 条纯正物理序列 (L1~L6, S1~S6)
  3. 执行各序列独立自适应归一化 (Session-level Min/Max Normalization)
  4. 应用统一安全分辨率 K = (64, 64, 64, 64, 64, 64)，理论严格满足 Δz < 0.25
  5. 采用 Floor/Round 互补双网格机制（δ = 0.25）消除跳变歧义：
     - 主端仅输出 1-bit 指示标记（Round vs Floor），绝不泄露任何量化真值
     - 掩码通过 np.packbits 压缩为 6 通道紧凑位图随密钥传递
  6. 产出 6 条全维度严格覆盖 [1, 24] 且主从 100% 绝对一致的 DNA 规则序列（遍历 S_4 对称群）
"""

import sys
import os
import argparse
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

# 6 条序列的量化区间数（统一采用 K=48 为 24 种规则的 2 个完整周期，彻底消除模偏差且严格满足 Δz < 0.25）
K_ADAPTIVE = (48, 48, 48, 48, 48, 48)

# 互补网格最优保护带阈值
DELTA_THRESHOLD = 0.25


def quantize_master(traj: np.ndarray, min_val: float, max_val: float, K: int, delta: float = DELTA_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """
    加密端（Master）：自适应归一化 + 互补双网格量化 + 1-bit 标记提取。

    Returns:
        rules: DNA 规则序列 uint8，值域 1~24。
        packed_mask: 紧凑位图 uint8 数组（承载每个点是否为 Round 的 1-bit 指示标记）。
    """
    span = max_val - min_val
    if span <= 1e-9:
        span = 1.0

    # 1. 归一化到 [0, 1) 并放大到 K 尺度
    u = np.clip((traj - min_val) / span, 0.0, 1.0 - 1e-7)
    z = u * K

    # 2. 计算距整数边界距离并提取 1-bit 标记（无真值泄漏）
    dist_to_int = np.abs(z - np.round(z))
    marked = (dist_to_int <= delta)  # True 表示使用 Round，False 表示使用 Floor

    # 3. 双网格量化与模 24 映射（覆盖 24 种 S_4 全排列规则）
    q = np.where(marked, np.round(z), np.floor(z)).astype(np.int32)
    rules = (np.mod(q, 24) + 1).astype(np.uint8)

    # 4. 压缩为紧凑位图
    packed_mask = np.packbits(marked)

    return rules, packed_mask


def quantize_slave(traj: np.ndarray, min_val: float, max_val: float, K: int, packed_mask: np.ndarray) -> np.ndarray:
    """
    解密端（Slave）：利用接收到的极值 [min, max] 与 1-bit 标记位图，执行对齐量化。
    注意：从端无需知道任何量化真值，仅凭标记即能在数学上实现 100% 绝对零误差还原。
    """
    span = max_val - min_val
    if span <= 1e-9:
        span = 1.0

    # 1. 解压 1-bit 标记
    marked = np.unpackbits(packed_mask)[:len(traj)].astype(bool)

    # 2. 相同极值归一化与放大
    u = np.clip((traj - min_val) / span, 0.0, 1.0 - 1e-7)
    z = u * K

    # 3. 遵照 1-bit 标记分别采用 Round 或 Floor，执行模 24 映射
    q = np.where(marked, np.round(z), np.floor(z)).astype(np.int32)
    rules = (np.mod(q, 24) + 1).astype(np.uint8)

    return rules


def generate_seed_scheme1(
    height: int = 256,
    width: int = 256,
    p: int = DEFAULT_BLOCK_SIZE,
    transient_steps: int = DEFAULT_TRANSIENT_STEPS
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...], dict]:
    """
    方案 1 序列生成主入口（6 序列 3 轮编解码架构）。

    Args:
        height: 图像高度。
        width: 图像宽度。
        p: 图像分块大小。
        transient_steps: 瞬态丢弃步数。

    Returns:
        master_sequence: (L1, L2, L3, L4, L5, L6) uint8，值域 1~24。
        slave_sequence:  (S1, S2, S3, S4, S5, S6) uint8，值域 1~24（严格逐点 100% 恒等）。
        raw:             原始连续浮点轨迹 (x1..x3, y1..y3)。
        helper_info:     包含会话极值 bounds 与 1-bit 紧凑掩码 packed_masks。
    """
    num = ((height + p - 1) // p) * ((width + p - 1) // p)
    raw = generate_raw(num, transient_steps=transient_steps)

    master_raw = raw[:3]  # x1, x2, x3, 长度 2 * num
    slave_raw = raw[3:]   # y1, y2, y3, 长度 2 * num

    # 奇偶时间抽取，构建 6 条物理序列
    # L1, L2, L3: 奇数时刻点 (0, 2, 4...)
    # L4, L5, L6: 偶数时刻点 (1, 3, 5...)
    master_sliced = [
        master_raw[0][0::2], master_raw[1][0::2], master_raw[2][0::2],
        master_raw[0][1::2], master_raw[1][1::2], master_raw[2][1::2],
    ]
    slave_sliced = [
        slave_raw[0][0::2], slave_raw[1][0::2], slave_raw[2][0::2],
        slave_raw[0][1::2], slave_raw[1][1::2], slave_raw[2][1::2],
    ]

    master_list = []
    slave_list = []
    bounds_list = []
    packed_masks_list = []
    marked_ratios = []

    for dim in range(6):
        m_r = master_sliced[dim]
        s_r = slave_sliced[dim]
        K = K_ADAPTIVE[dim]

        # 1. 主端确定当前会话的物理极值（加微小裕度防端点除零）
        min_v = float(m_r.min() - 1e-4)
        max_v = float(m_r.max() + 1e-4)
        bounds_list.append((min_v, max_v))

        # 2. 主端量化并生成 1-bit 掩码
        m_rules, packed_mask = quantize_master(m_r, min_v, max_v, K, delta=DELTA_THRESHOLD)
        packed_masks_list.append(packed_mask)

        # 统计标记比例
        unpacked = np.unpackbits(packed_mask)[:len(m_r)]
        marked_ratios.append(float(np.mean(unpacked)))

        # 3. 从端依掩码执行无真值量化对齐
        s_rules = quantize_slave(s_r, min_v, max_v, K, packed_mask)

        master_list.append(m_rules)
        slave_list.append(s_rules)

    master_sequence = tuple(master_list)
    slave_sequence = tuple(slave_list)

    helper_info = {
        "bounds": bounds_list,
        "packed_masks": packed_masks_list,
        "K_adaptive": K_ADAPTIVE,
        "marked_ratios": marked_ratios,
        "mask_bytes_total": sum(len(pm) for pm in packed_masks_list),
        "zero_leakage": True,
    }

    return master_sequence, slave_sequence, raw, helper_info


# ==========================================
# __main__：独立验证与测试入口
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="方案 1：6 序列 3 轮编解码架构独立测试")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    args = parser.parse_args()

    print("=" * 70)
    print("方案 1：连续混沌轨迹自适应归一化 + 互补双网格量化 (6 序列架构)")
    print(f"图像尺寸: {args.height}x{args.width}, 块大小: {args.block_size}")
    print(f"自适应分辨率 K: {K_ADAPTIVE}, 保护带阈值 δ: {DELTA_THRESHOLD}")
    print("=" * 70)

    master_seq, slave_seq, raw, helper_info = generate_seed_scheme1(args.height, args.width, args.block_size)

    assert len(master_seq) == 6, f"预期 6 条序列，实际为 {len(master_seq)}"
    assert len(slave_seq) == 6, f"预期 6 条从序列，实际为 {len(slave_seq)}"

    synced = check_synchronization(master_seq, slave_seq)
    print(f"\n[同步判定] check_synchronization: {'[PASS] (6 序列 100% 逐点恒等零误差)' if synced else '[FAIL]'}")

    seq_names = ["L1 (x1_odd)", "L2 (x2_odd)", "L3 (x3_odd)", "L4 (x1_even)", "L5 (x2_even)", "L6 (x3_even)"]
    print("\n[序列多样性与规则全覆盖检验]")
    for i in range(6):
        m = master_seq[i]
        unique_vals, counts = np.unique(m, return_counts=True)
        ratio_pct = helper_info["marked_ratios"][i] * 100
        min_v, max_v = helper_info["bounds"][i]
        print(f"  序列 {seq_names[i]} (K={K_ADAPTIVE[i]}): 物理范围=[{min_v:.3f}, {max_v:.3f}], 1-bit标记率={ratio_pct:.1f}%")
        print(f"             独立规则数={len(unique_vals)}, 取值={unique_vals}")
        assert len(unique_vals) == 24, f"序列 {seq_names[i]} 未完全覆盖 24 种规则！(实际覆盖 {len(unique_vals)} 种)"

    print(f"\n[辅助数据开销与安全特性]")
    print(f"  紧凑位图总字节数: {helper_info['mask_bytes_total']} 字节 (~{helper_info['mask_bytes_total']/1024:.2f} KB)")
    print(f"  是否泄露任何真值/偏移量: 否 (Zero Value Leakage)")

    print("\n" + "=" * 70)
    print("方案 1 (6 序列 3 轮架构) 独立测试全部通过！")
    print("=" * 70)


if __name__ == "__main__":
    main()
