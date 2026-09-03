"""
chaos_sequence.py  —  混沌同步序列生成门面与调度模块

职责：
  1. 作为统一调度门面（Facade），汇聚方案 1 与方案 3 的生成能力
  2. 支持自由切换方案 1（轨迹直接量化）与方案 3（稳态握手种子流扩展）
  3. 保持向后兼容性，供上层加解密流程（encry.py）及测试脚本无缝调用
  4. 命令行独立执行与序列诊断分析
"""

import os
import sys
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

# 引入底层共用模块
from Code.encry.chaos_base import (
    generate_raw,
    check_synchronization,
    MODEL_DIR,
    PINNING_NODE,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_TRANSIENT_STEPS,
)

# 引入解耦的方案 1 与方案 3 独立模块
from Code.encry.scheme1_direct_quant import generate_seed_scheme1
from Code.encry.scheme3_seed_stream import generate_seed_scheme3

# 全局配置开关
ENABLE_SYNC_CHECK = True
DEFAULT_SCHEME = "scheme3"


def generate_seed(
    height: int = 256,
    width: int = 256,
    p: int = DEFAULT_BLOCK_SIZE,
    scheme: str = DEFAULT_SCHEME
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """
    统一序列生成入口，支持方案 1 与方案 3 自由切换。

    Args:
        height: 图像高度。
        width: 图像宽度。
        p: 图像分块大小。
        scheme: "scheme1" 或 "scheme3"，默认 "scheme3"。

    Returns:
        master_sequence: (x1, x2, x3, x4) uint8，值域 1~8。
        slave_sequence:  (x5, x6, x7, x8) uint8，值域 1~8。
        raw_arrays:      原始连续浮点轨迹。
    """
    scheme = scheme.lower()
    if scheme in ("1", "scheme1", "direct", "quant"):
        master_seq, slave_seq, raw, _ = generate_seed_scheme1(height, width, p)
    elif scheme in ("3", "scheme3", "seed", "stream"):
        master_seq, slave_seq, raw, _ = generate_seed_scheme3(height, width, p)
    else:
        raise ValueError(f"未知的混沌序列处理方案: {scheme}。可选方案为 'scheme1' 或 'scheme3'。")

    return master_sequence_clean(master_seq), master_sequence_clean(slave_seq), raw


def master_sequence_clean(seq_tuple: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """确保序列格式为统一 uint8 ndarray。"""
    return tuple(np.asarray(s, dtype=np.uint8) for s in seq_tuple)


# ==========================================
# 保存与绘图分析辅助工具
# ==========================================

def _save_results(master_seq, slave_seq, raw_arrays, output_dir: str, scheme: str):
    """保存序列数据和对比图到实验目录。"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存量化后序列为 .npz
    npz_path = os.path.join(output_dir, f"chaotic_sequences_{scheme}.npz")
    save_dict = {f"master_L{i+1}": master_seq[i] for i in range(len(master_seq))}
    save_dict.update({f"slave_S{i+1}": slave_seq[i] for i in range(len(slave_seq))})
    np.savez(npz_path, **save_dict)

    default_npz_path = os.path.join(output_dir, "chaotic_sequences.npz")
    np.savez(default_npz_path, **save_dict)
    print(f"[SAVED] Quantized sequences → {npz_path}")

    # 2. 保存原始浮点序列
    raw_npz_path = os.path.join(output_dir, "raw_sequences.npz")
    np.savez(
        raw_npz_path,
        **{f"raw_x{i+1}": np.array(raw_arrays[i]) for i in range(len(raw_arrays))}
    )
    print(f"[SAVED] Raw float sequences → {raw_npz_path}")

    # 3. 绘制主/从序列对比图与误差图
    try:
        import matplotlib.pyplot as plt

        num_seqs = len(master_seq)
        fig, axes = plt.subplots(num_seqs, 1, figsize=(14, 2.2 * num_seqs), sharex=True)
        fig.suptitle(f"Master vs Slave Chaotic Sequences ({scheme.upper()}) [6-Sequence Architecture]", fontsize=14)

        for idx, ax in enumerate(axes):
            m = master_seq[idx]
            s = slave_seq[idx]
            n_points = min(len(m), 500)

            ax.plot(m[:n_points], label=f"Master L{idx+1}", linewidth=0.8, alpha=0.8)
            ax.plot(s[:n_points], label=f"Slave S{idx+1}", linewidth=0.8, alpha=0.8, linestyle='--')
            ax.set_ylabel(f"Pair {idx+1}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Block Index")
        plt.tight_layout()

        fig_path = os.path.join(output_dir, f"sequence_comparison_{scheme}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] Comparison plot → {fig_path}")

        # 4. 误差图
        fig2, axes2 = plt.subplots(num_seqs, 1, figsize=(14, 2 * num_seqs), sharex=True)
        fig2.suptitle(f"Synchronization Error (Master - Slave) [{scheme.upper()}]", fontsize=14)

        for idx, ax in enumerate(axes2):
            error = master_seq[idx].astype(np.int16) - slave_seq[idx].astype(np.int16)
            ax.plot(error, linewidth=0.5, color='red', alpha=0.7)
            ax.set_ylabel(f"Error {idx+1}")
            ax.axhline(y=0, color='black', linewidth=0.3)
            ax.grid(True, alpha=0.3)
            non_zero = np.count_nonzero(error)
            ax.set_title(f"Non-zero errors: {non_zero}/{len(error)} ({non_zero/len(error)*100:.2f}%)", fontsize=9)

        axes2[-1].set_xlabel("Block Index")
        plt.tight_layout()

        err_path = os.path.join(output_dir, f"sync_error_{scheme}.png")
        plt.savefig(err_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] Sync error plot → {err_path}")

    except ImportError:
        print("[WARN] matplotlib not available, skipping plots.")


# ==========================================
# __main__：独立测试与命令行入口
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="混沌同步序列生成与调度实验")
    parser.add_argument("--scheme", type=str, default=DEFAULT_SCHEME,
                        choices=["scheme1", "scheme3"],
                        help="选择混沌序列处理方案: scheme1 (轨迹直接量化) 或 scheme3 (稳态种子流扩展)")
    parser.add_argument("--height", type=int, default=256, help="图像高度 (default: 256)")
    parser.add_argument("--width", type=int, default=256, help="图像宽度 (default: 256)")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE, help="分块大小 (default: 8)")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(str(PROJECT_ROOT), "experiments", "seq_gen")

    print("=" * 60)
    print("混沌序列生成调度引擎 (6 序列 3 轮不对称编解码)")
    print(f"当前选定方案: {args.scheme}")
    print(f"图像尺寸: {args.height}x{args.width}, 分块: {args.block_size}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    master_seq, slave_seq, raw_arrays = generate_seed(
        height=args.height,
        width=args.width,
        p=args.block_size,
        scheme=args.scheme
    )

    is_synced = check_synchronization(master_seq, slave_seq)
    print(f"\n[同步检验] check_synchronization: {'[PASS] 成功' if is_synced else '[FAIL] 失败'}")

    # 统计独立值数量，确保无常数坍缩
    print("\n[序列多样性检验]")
    for i in range(len(master_seq)):
        u_count = len(np.unique(master_seq[i]))
        print(f"  序列 L{i+1}: 独立值计数 = {u_count} (值域 1~8)")
        assert u_count > 1, f"警告：序列 L{i+1} 发生了常数坍缩！"

    _save_results(master_seq, slave_seq, raw_arrays, args.output_dir, args.scheme)

    print("\n" + "=" * 60)
    print(f"方案 {args.scheme} (6 序列架构) 序列生成与保存已成功完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
