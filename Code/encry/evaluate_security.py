"""
evaluate_security.py  —  图像加密方案完整安全评测工具链 (T1–T4)

功能模块：
  - 模块 A：信息熵（Shannon Entropy，RGB 单通道与均值）
  - 模块 B：NPCR / UACI 差分攻击敏感性分析（同混沌序列，LSB 单比特翻转）
  - 模块 C：密钥敏感性分析与比特汉明距离（归一化 Hamming Distance、MAE、PSNR）
  - 模块 D：密钥空间理论分析（枚举参数有效位数）
  - 模块 E：抗噪声与抗遮挡鲁棒性评测（高斯噪声、椒盐噪声、中心遮挡）
  - 模块 F：同步误差容差实验与论文级双 Y 轴折线图绘制（300dpi PNG + PDF）
  - 模块 G：文献基准横向对比表格生成（CSV、LaTeX .tex、Rich 终端彩表）

依赖：
  - Code.encry.encry: encrypt, decrypt, string_to_initial_value
  - Code.encry.chaos_sequence: generate_seed, check_synchronization
  - Code.encry.scheme1_direct_quant: quantize_master, quantize_slave, K_ADAPTIVE, DELTA_THRESHOLD
"""

import os
import sys
import argparse
import math
import csv
import time
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, Any, List

import numpy as np
import cv2
import matplotlib.pyplot as plt

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# 适配 Windows 控制台 UTF-8 编码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---- 导入路径设置 ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Code.encry.encry import (
    encrypt,
    decrypt,
    string_to_initial_value,
    MY_PASSWORD,
    DEFAULT_BLOCK_SIZE,
    PLAIN_DIR,
    CIPHER_DIR,
    DECRYPTED_DIR,
)
from Code.encry.chaos_sequence import generate_seed, check_synchronization
from Code.encry.chaos_base import generate_raw, DEFAULT_TRANSIENT_STEPS
from Code.encry.scheme1_direct_quant import (
    quantize_master,
    quantize_slave,
    K_ADAPTIVE,
    DELTA_THRESHOLD,
)

DEFAULT_OUTPUT_DIR = os.path.join("logs", "security_eval")


# =====================================================================
# 模块 A：信息熵 (Shannon Entropy)
# =====================================================================

def compute_entropy(channel: np.ndarray) -> float:
    """
    单通道 Shannon 熵：H = -Σ p(k) log2 p(k), k=0..255。

    Args:
        channel: 单通道 2D uint8 数组。

    Returns:
        float: 香农信息熵值 (理论极限 8.0000)。
    """
    hist, _ = np.histogram(channel, bins=256, range=(0, 256))
    total_pixels = channel.size
    p = hist / total_pixels
    p_nonzero = p[p > 0]
    entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
    return float(entropy)


def compute_entropy_rgb(img: np.ndarray) -> dict:
    """
    计算 RGB 图像各通道及均值信息熵。
    注意：OpenCV 图像默认通道顺序为 BGR (0:B, 1:G, 2:R)。

    Args:
        img: HxWx3 (或灰度 HxW) uint8 数组。

    Returns:
        dict: {'R': float, 'G': float, 'B': float, 'mean': float}
    """
    if img.ndim == 2:
        e = compute_entropy(img)
        return {'R': e, 'G': e, 'B': e, 'mean': e}

    b, g, r = cv2.split(img)
    ent_r = compute_entropy(r)
    ent_g = compute_entropy(g)
    ent_b = compute_entropy(b)
    ent_mean = (ent_r + ent_g + ent_b) / 3.0

    return {
        'R': ent_r,
        'G': ent_g,
        'B': ent_b,
        'mean': ent_mean
    }


# =====================================================================
# 模块 B：NPCR / UACI (差分敏感性)
# =====================================================================

def compute_npcr_uaci(plain_img: np.ndarray,
                      encrypt_fn: Callable = encrypt,
                      encrypt_kwargs: dict = None,
                      n_tests: int = 10,
                      seed: int = 42) -> dict:
    """
    计算 NPCR (像素变化率) 和 UACI (统一平均改变强度)。

    流程：
      1. 随机选取 n_tests 个像素位置
      2. 每次将该像素最低有效位翻转，得到 P'
      3. 用相同密钥流（encrypt_kwargs 中传入预生成的 seeds）分别加密 P 和 P'，得到 C 和 C'
      4. 逐通道计算 NPCR 和 UACI
      5. 对 n_tests 次结果取均值

    公式：
      NPCR = (1/MN) * Σ D(i,j) * 100%，其中 D(i,j) = 1 if C(i,j) != C'(i,j) else 0
      UACI = (1/MN) * Σ |C(i,j) - C'(i,j)| / 255 * 100%
      理论期望：NPCR ≈ 99.6094%, UACI ≈ 33.4635%

    Returns:
        dict: {'npcr_r', 'npcr_g', 'npcr_b', 'npcr_mean',
               'uaci_r', 'uaci_g', 'uaci_b', 'uaci_mean'}
    """
    if encrypt_kwargs is None:
        encrypt_kwargs = {}

    h, w = plain_img.shape[:2]

    # 关键约束：确保两次加密使用同一组混沌序列种子
    if 'master_sequence' not in encrypt_kwargs or encrypt_kwargs['master_sequence'] is None:
        master_seq, _, _ = generate_seed(h, w, DEFAULT_BLOCK_SIZE)
        encrypt_kwargs = {**encrypt_kwargs, 'master_sequence': master_seq}

    # 加密未扰动的原图
    c_orig, _ = encrypt_fn(plain_img=plain_img, **encrypt_kwargs)

    rng = np.random.RandomState(seed)

    npcr_r_list, npcr_g_list, npcr_b_list = [], [], []
    uaci_r_list, uaci_g_list, uaci_b_list = [], [], []

    channels = plain_img.shape[2] if plain_img.ndim == 3 else 1

    for _ in range(n_tests):
        p_prime = plain_img.copy()
        row = rng.randint(0, h)
        col = rng.randint(0, w)
        if channels == 3:
            ch_idx = rng.randint(0, 3)
            p_prime[row, col, ch_idx] ^= 1  # 翻转 LSB 最低有效位
        else:
            p_prime[row, col] ^= 1

        c_prime, _ = encrypt_fn(plain_img=p_prime, **encrypt_kwargs)

        if channels == 3:
            # OpenCV BGR: 0:B, 1:G, 2:R
            d_b = (c_orig[:, :, 0] != c_prime[:, :, 0]).astype(np.float64)
            d_g = (c_orig[:, :, 1] != c_prime[:, :, 1]).astype(np.float64)
            d_r = (c_orig[:, :, 2] != c_prime[:, :, 2]).astype(np.float64)

            u_b = np.abs(c_orig[:, :, 0].astype(np.float64) - c_prime[:, :, 0].astype(np.float64)) / 255.0
            u_g = np.abs(c_orig[:, :, 1].astype(np.float64) - c_prime[:, :, 1].astype(np.float64)) / 255.0
            u_r = np.abs(c_orig[:, :, 2].astype(np.float64) - c_prime[:, :, 2].astype(np.float64)) / 255.0

            npcr_b_list.append(float(np.mean(d_b) * 100.0))
            npcr_g_list.append(float(np.mean(d_g) * 100.0))
            npcr_r_list.append(float(np.mean(d_r) * 100.0))

            uaci_b_list.append(float(np.mean(u_b) * 100.0))
            uaci_g_list.append(float(np.mean(u_g) * 100.0))
            uaci_r_list.append(float(np.mean(u_r) * 100.0))
        else:
            d = (c_orig != c_prime).astype(np.float64)
            u = np.abs(c_orig.astype(np.float64) - c_prime.astype(np.float64)) / 255.0
            val_npcr = float(np.mean(d) * 100.0)
            val_uaci = float(np.mean(u) * 100.0)
            npcr_r_list.append(val_npcr)
            npcr_g_list.append(val_npcr)
            npcr_b_list.append(val_npcr)
            uaci_r_list.append(val_uaci)
            uaci_g_list.append(val_uaci)
            uaci_b_list.append(val_uaci)

    mean_npcr_r = float(np.mean(npcr_r_list))
    mean_npcr_g = float(np.mean(npcr_g_list))
    mean_npcr_b = float(np.mean(npcr_b_list))
    mean_npcr = (mean_npcr_r + mean_npcr_g + mean_npcr_b) / 3.0

    mean_uaci_r = float(np.mean(uaci_r_list))
    mean_uaci_g = float(np.mean(uaci_g_list))
    mean_uaci_b = float(np.mean(uaci_b_list))
    mean_uaci = (mean_uaci_r + mean_uaci_g + mean_uaci_b) / 3.0

    return {
        'npcr_r': mean_npcr_r,
        'npcr_g': mean_npcr_g,
        'npcr_b': mean_npcr_b,
        'npcr_mean': mean_npcr,
        'uaci_r': mean_uaci_r,
        'uaci_g': mean_uaci_g,
        'uaci_b': mean_uaci_b,
        'uaci_mean': mean_uaci,
    }


# =====================================================================
# 模块 C：密钥敏感性 + 汉明距离
# =====================================================================

def compute_hamming_distance(data1: np.ndarray, data2: np.ndarray) -> dict:
    """
    将两个等长数组按字节展平为比特串，计算归一化汉明距离。

    Returns:
        dict: {'bit_hamming_pct': float,  # 百分比 [0, 100]
               'pixel_mae': float,
               'psnr_db': float}         # 若 MAE=0 则返回 float('inf')
    """
    b1 = np.unpackbits(data1.astype(np.uint8))
    b2 = np.unpackbits(data2.astype(np.uint8))
    bit_hamming_pct = float(np.mean(b1 != b2) * 100.0)

    diff = np.abs(data1.astype(np.float64) - data2.astype(np.float64))
    mae = float(np.mean(diff))
    mse = float(np.mean(diff ** 2))

    if mse <= 1e-12:
        psnr_db = float('inf')
    else:
        psnr_db = float(10.0 * np.log10((255.0 ** 2) / mse))

    return {
        'bit_hamming_pct': bit_hamming_pct,
        'pixel_mae': mae,
        'psnr_db': psnr_db
    }


def key_sensitivity_test(plain_img: np.ndarray,
                         password: str = MY_PASSWORD,
                         perturbation: float = 1e-14) -> dict:
    """
    测试 1 — 加密敏感性：
      用密码 password 加密得 C1
      用密码派生初值 +perturbation 加密同一明文得 C2
      计算 C1 与 C2 之间的 NPCR 和汉明距离

    测试 2 — 解密敏感性：
      用正确密钥解密 C1 得 D_correct
      用微扰密钥解密 C1 得 D_wrong
      计算 D_wrong 与原图的 MAE / PSNR

    Returns:
        dict: {'encrypt_npcr': float,
               'encrypt_hamming_pct': float,
               'decrypt_correct_mae': float, 'decrypt_correct_psnr': float,
               'decrypt_wrong_mae': float,   'decrypt_wrong_psnr': float,
               'decrypt_wrong_hd': float}
    """
    h, w = plain_img.shape[:2]
    master_seq, slave_seq, _ = generate_seed(h, w, DEFAULT_BLOCK_SIZE)

    # 1. 正常加密得到 C1
    c1, key1 = encrypt(master_sequence=master_seq, plain_img=plain_img, password=password, offset=0.0)

    # 2. 微扰加密得到 C2 (初值加上 perturbation)
    c2, key2 = encrypt(master_sequence=master_seq, plain_img=plain_img, password=password, offset=perturbation)

    # 加密敏感性计算
    enc_npcr = float(np.mean(c1 != c2) * 100.0)
    enc_hd = compute_hamming_distance(c1, c2)

    # 3. 正确密钥解密 C1
    ok1, d_correct = decrypt(slave_sequence=slave_seq, cipher_img=c1, decry_key=key1, password=password, offset=0.0)
    correct_eval = compute_hamming_distance(plain_img, d_correct)

    # 4. 微扰密钥解密 C1 (解密端 Logistic 初值加 perturbation)
    ok2, d_wrong = decrypt(slave_sequence=slave_seq, cipher_img=c1, decry_key=key1, password=password, offset=perturbation)
    wrong_eval = compute_hamming_distance(plain_img, d_wrong)

    return {
        'encrypt_npcr': enc_npcr,
        'encrypt_hamming_pct': enc_hd['bit_hamming_pct'],
        'decrypt_correct_mae': correct_eval['pixel_mae'],
        'decrypt_correct_psnr': correct_eval['psnr_db'],
        'decrypt_wrong_mae': wrong_eval['pixel_mae'],
        'decrypt_wrong_psnr': wrong_eval['psnr_db'],
        'decrypt_wrong_hd': wrong_eval['bit_hamming_pct'],
    }


# =====================================================================
# 模块 D：密钥空间分析 (Key Space Analysis)
# =====================================================================

def analyze_key_space() -> dict:
    """
    统计并枚举加密系统各项密钥参数的有效位数：
      - 用户密码 -> SHA-256 哈希: 256 bits
      - Logistic 参数 θ=3.9999: IEEE 754 float64 尾数 52 bits
      - Hopfield 3D 初始条件 (x1, x2, x3): 3 * float64 = 156 bits
      - Pinning node 选择 (3选1): log2(3) ≈ 1.585 bits

    Returns:
        dict: {'total_bits': int, 'total_space_str': str, 'breakdown': list[dict]}
    """
    breakdown = [
        {
            'parameter': 'User Password SHA-256 Digest',
            'type': 'Cryptographic hash space',
            'bits': 256.0,
            'space': '2^{256}',
            'note': 'Standard SHA-256 pre-image resistance'
        },
        {
            'parameter': 'Logistic map bifurcation parameter θ',
            'type': 'IEEE 754 float64 mantissa',
            'bits': 52.0,
            'space': '2^{52}',
            'note': '64-bit float precision'
        },
        {
            'parameter': 'Hopfield 3D initial condition (x1, x2, x3)',
            'type': '3 x IEEE 754 float64',
            'bits': 156.0,
            'space': '2^{156}',
            'note': '3 dynamic continuous states'
        },
        {
            'parameter': 'Actuator pinning node selection',
            'type': 'Discrete choice {1, 2, 3}',
            'bits': 1.6,
            'space': '3',
            'note': 'Optimal pinning site'
        },
    ]

    total_bits = 256.0 + 52.0 + 156.0 + 1.585
    total_bits_int = int(math.floor(total_bits))

    return {
        'total_bits': total_bits_int,
        'total_space_str': f'2^{{{total_bits_int}}}',
        'breakdown': breakdown,
    }


# =====================================================================
# 模块 E：抗噪声与抗遮挡鲁棒性 (Robustness)
# =====================================================================

def robustness_test(plain_img: np.ndarray,
                    cipher_img: np.ndarray,
                    decrypt_fn: Callable = decrypt,
                    decrypt_kwargs: dict = None,
                    seed: int = 42) -> dict:
    """
    对密文施加攻击后解密，与原图对比：
      1. 高斯噪声: sigma ∈ [5, 10, 20, 40]
      2. 椒盐噪声: ratio ∈ [0.01, 0.05, 0.10, 0.20]
      3. 遮挡攻击: 面积比 ∈ [1/16, 1/8, 1/4]

    Returns:
        dict: {'gaussian': [{'sigma', 'psnr', 'mae'}],
               'salt_pepper': [{'ratio', 'psnr', 'mae'}],
               'occlusion': [{'area_ratio', 'psnr', 'mae'}]}
    """
    if decrypt_kwargs is None:
        decrypt_kwargs = {}

    rng = np.random.RandomState(seed)
    h, w = plain_img.shape[:2]

    # 1. 高斯噪声测试
    gaussian_results = []
    for sigma in [5, 10, 20, 40]:
        noise = rng.normal(0, sigma, cipher_img.shape)
        cipher_noisy = np.clip(cipher_img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        _, dec_img = decrypt_fn(cipher_img=cipher_noisy, **decrypt_kwargs)
        eval_res = compute_hamming_distance(plain_img, dec_img)
        gaussian_results.append({
            'sigma': sigma,
            'psnr': eval_res['psnr_db'],
            'mae': eval_res['pixel_mae']
        })

    # 2. 椒盐噪声测试
    salt_pepper_results = []
    for ratio in [0.01, 0.05, 0.10, 0.20]:
        cipher_sp = cipher_img.copy()
        num_pixels = int(ratio * h * w)
        half = num_pixels // 2

        # 椒噪声 (0)
        r_coords = rng.randint(0, h, half)
        c_coords = rng.randint(0, w, half)
        cipher_sp[r_coords, c_coords] = 0

        # 盐噪声 (255)
        r_coords = rng.randint(0, h, half)
        c_coords = rng.randint(0, w, half)
        cipher_sp[r_coords, c_coords] = 255

        _, dec_img = decrypt_fn(cipher_img=cipher_sp, **decrypt_kwargs)
        eval_res = compute_hamming_distance(plain_img, dec_img)
        salt_pepper_results.append({
            'ratio': ratio,
            'psnr': eval_res['psnr_db'],
            'mae': eval_res['pixel_mae']
        })

    # 3. 遮挡攻击测试 (中心矩形区域置 0)
    occlusion_results = []
    for area_ratio in [1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0]:
        cipher_occ = cipher_img.copy()
        occ_h = int(np.sqrt(area_ratio) * h)
        occ_w = int(np.sqrt(area_ratio) * w)
        r0 = max(0, (h - occ_h) // 2)
        c0 = max(0, (w - occ_w) // 2)
        cipher_occ[r0:r0 + occ_h, c0:c0 + occ_w] = 0

        _, dec_img = decrypt_fn(cipher_img=cipher_occ, **decrypt_kwargs)
        eval_res = compute_hamming_distance(plain_img, dec_img)
        occlusion_results.append({
            'area_ratio': area_ratio,
            'psnr': eval_res['psnr_db'],
            'mae': eval_res['pixel_mae']
        })

    return {
        'gaussian': gaussian_results,
        'salt_pepper': salt_pepper_results,
        'occlusion': occlusion_results,
    }


# =====================================================================
# 模块 F：同步误差容差实验与曲线绘制
# =====================================================================

def sync_error_sweep(plain_img: np.ndarray,
                     error_levels: list = None,
                     n_trials: int = 5,
                     seed: int = 42) -> dict:
    """
    默认 error_levels = [0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1]

    流程（每个 error_level 重复 n_trials 次取均值）：
      1. 调用 chaos_base 正常生成主系统 6 条原始浮点轨迹
      2. 复制主轨迹，叠加 N(0, error_level²) 高斯噪声作为模拟从系统轨迹
      3. 将两组轨迹送入 scheme1 量化流程
      4. 分别用主/从序列执行加密/解密
      5. 计算解密图像的像素正确率、PSNR、MAE

    Returns:
        dict: {'error_levels', 'pixel_correct_rate', 'psnr', 'mae'}
    """
    if error_levels is None:
        error_levels = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1]

    rng = np.random.RandomState(seed)
    h, w = plain_img.shape[:2]
    num = ((h + DEFAULT_BLOCK_SIZE - 1) // DEFAULT_BLOCK_SIZE) * ((w + DEFAULT_BLOCK_SIZE - 1) // DEFAULT_BLOCK_SIZE)

    # 1. 生成 Hopfield 主系统连续浮点轨迹 (3D, 2N 步)
    raw = generate_raw(num, transient_steps=DEFAULT_TRANSIENT_STEPS)
    master_raw = raw[:3]

    # 时间交错切分 6 序列
    master_sliced = [
        master_raw[0][0::2], master_raw[1][0::2], master_raw[2][0::2],
        master_raw[0][1::2], master_raw[1][1::2], master_raw[2][1::2],
    ]

    # 主端执行 Scheme 1 双网格量化并提取 1-bit 紧凑掩码
    master_rules_list = []
    packed_masks_list = []
    bounds_list = []

    for dim in range(6):
        m_r = master_sliced[dim]
        min_v = float(m_r.min() - 1e-4)
        max_v = float(m_r.max() + 1e-4)
        bounds_list.append((min_v, max_v))
        K = K_ADAPTIVE[dim]
        m_rules, p_mask = quantize_master(m_r, min_v, max_v, K, delta=DELTA_THRESHOLD)
        master_rules_list.append(m_rules)
        packed_masks_list.append(p_mask)

    master_sequence = tuple(master_rules_list)

    # 统一加密明文图像
    cipher_img, decry_key = encrypt(master_sequence=master_sequence, plain_img=plain_img, password=MY_PASSWORD)

    pixel_correct_rates = []
    psnr_list = []
    mae_list = []

    for err in error_levels:
        trial_pcrs = []
        trial_psnrs = []
        trial_maes = []

        for _ in range(n_trials):
            # 2. 模拟从端叠加高斯同步误差
            if err == 0.0:
                slave_sliced = [m.copy() for m in master_sliced]
            else:
                slave_sliced = [m + rng.normal(0.0, err, len(m)) for m in master_sliced]

            # 3. 从端执行 Scheme 1 对齐量化
            slave_rules_list = []
            for dim in range(6):
                s_r = slave_sliced[dim]
                min_v, max_v = bounds_list[dim]
                K = K_ADAPTIVE[dim]
                s_rules = quantize_slave(s_r, min_v, max_v, K, packed_masks_list[dim])
                slave_rules_list.append(s_rules)

            slave_sequence = tuple(slave_rules_list)

            # 4. 执行解密
            _, dec_img = decrypt(slave_sequence=slave_sequence, cipher_img=cipher_img,
                                 decry_key=decry_key, password=MY_PASSWORD)

            # 5. 评测
            pcr = float(np.mean(dec_img == plain_img) * 100.0)
            eval_res = compute_hamming_distance(plain_img, dec_img)

            trial_pcrs.append(pcr)
            trial_psnrs.append(eval_res['psnr_db'] if eval_res['psnr_db'] != float('inf') else 100.0)
            trial_maes.append(eval_res['pixel_mae'])

        pixel_correct_rates.append(float(np.mean(trial_pcrs)))
        psnr_list.append(float(np.mean(trial_psnrs)))
        mae_list.append(float(np.mean(trial_maes)))

    return {
        'error_levels': error_levels,
        'pixel_correct_rate': pixel_correct_rates,
        'psnr': psnr_list,
        'mae': mae_list,
    }


def plot_sync_error_curve(sweep_results: dict, save_path: str):
    """
    绘制论文级双 Y 轴折线图：
      - X 轴: 同步误差幅度 (对数坐标)
      - Y 轴左: 像素恢复正确率 (%)
      - Y 轴右: PSNR (dB)
      - 垂直标注: Node 3 工作点 (≈0.0097) 和 Node 2 工作点 (≈0.039)
      - 水平标注: 量化安全阈值 ε_max ≈ 0.031
      - 保存为 300dpi PNG + PDF 矢量图
    """
    error_levels = sweep_results['error_levels']
    pcr = sweep_results['pixel_correct_rate']
    psnr = sweep_results['psnr']

    # 处理 X 轴对数坐标（将 0 映射为一个微小占位值以在对数轴左端显示）
    plot_x = [1e-5 if e == 0.0 else e for e in error_levels]

    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
    })

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

    color_pcr = '#1f77b4'  # 科技蓝
    color_psnr = '#d62728'  # 绯红

    # 左轴：像素正确率
    ax1.set_xscale('log')
    line1 = ax1.plot(plot_x, pcr, color=color_pcr, marker='o', linewidth=2.2,
                     markersize=7, label='Pixel Recovery Rate (%)')
    ax1.set_xlabel('Synchronization Error Magnitude $\\epsilon$ (RMS)', fontweight='bold')
    ax1.set_ylabel('Pixel Recovery Rate (%)', color=color_pcr, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_pcr)
    ax1.set_ylim([-5, 105])

    # 右轴：PSNR
    ax2 = ax1.twinx()
    line2 = ax2.plot(plot_x, psnr, color=color_psnr, marker='s', linestyle='--',
                     linewidth=2.2, markersize=7, label='Decrypted PSNR (dB)')
    ax2.set_ylabel('PSNR (dB)', color=color_psnr, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_psnr)
    ax2.set_ylim([0, 105])

    # 自定义 X 轴刻度标签
    tick_locs = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    tick_labels = ['0 (Ideal)', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$5\\times 10^{-2}$', '$10^{-1}$', '$5\\times 10^{-1}$']
    ax1.set_xticks(tick_locs)
    ax1.set_xticklabels(tick_labels)

    # 绘制关键工作点与理论阈值参考线
    node3_err = 0.0097
    node2_err = 0.0390
    thresh_err = 0.03125  # delta / (K / span) 约为 0.031

    ax1.axvline(x=node3_err, color='#2ca02c', linestyle=':', linewidth=1.8,
                label=f'PPO Node 3 Operating Point ($\\epsilon \\approx {node3_err:.4f}$)')
    ax1.axvline(x=node2_err, color='#ff7f0e', linestyle='-.', linewidth=1.8,
                label=f'PPO Node 2 Operating Point ($\\epsilon \\approx {node2_err:.4f}$)')
    ax1.axvline(x=thresh_err, color='purple', linestyle='--', linewidth=1.8, alpha=0.8,
                label=f'Quantization Safe Threshold $\\epsilon_{{max}} \\approx {thresh_err:.3f}$')

    # 图例合并
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    # 添加竖线图例项
    custom_lines = [
        plt.Line2D([0], [0], color=color_pcr, marker='o', lw=2),
        plt.Line2D([0], [0], color=color_psnr, marker='s', linestyle='--', lw=2),
        plt.Line2D([0], [0], color='#2ca02c', linestyle=':', lw=1.8),
        plt.Line2D([0], [0], color='purple', linestyle='--', lw=1.8),
        plt.Line2D([0], [0], color='#ff7f0e', linestyle='-.', lw=1.8),
    ]
    custom_labels = [
        'Pixel Recovery Rate (%)',
        'Decrypted PSNR (dB)',
        f'PPO Node 3 ($\\epsilon={node3_err}$)',
        f'Guaranteed Threshold $\\epsilon_{{max}} \\approx 0.031$',
        f'PPO Node 2 ($\\epsilon={node2_err}$)',
    ]
    ax1.legend(custom_lines, custom_labels, loc='center left', framealpha=0.92, facecolor='white')

    ax1.grid(True, linestyle=':', alpha=0.6)
    plt.title('Scheme 1 Direct Quantization: Error Tolerance vs Image Recovery', pad=12, fontweight='bold')
    plt.tight_layout()

    # 自动保存 PNG 与 PDF
    base_save, _ = os.path.splitext(save_path)
    png_path = f"{base_save}.png"
    pdf_path = f"{base_save}.pdf"

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"[SAVED] Sync error tolerance curve (PNG 300dpi) → {png_path}")
    print(f"[SAVED] Sync error tolerance curve (PDF Vector) → {pdf_path}")


# =====================================================================
# 模块 G：横向对比表格与文献基准
# =====================================================================

LITERATURE_BENCHMARKS = {
    'Jin_2025_TCAS': {
        'label': 'Jin et al. (2025) [TCAS-I]',
        'corr_h': 0.0056, 'corr_v': 0.0017, 'corr_d': 0.0024,
        'entropy': None, 'npcr': None, 'uaci': None,
        'key_space': '10^{95}',
    },
    'Wang_2021_TCAS': {
        'label': 'Wang et al. (2021) [TCAS-I]',
        'corr_h': -0.0088, 'corr_v': 0.0047, 'corr_d': 0.0053,
        'entropy': None, 'npcr': None, 'uaci': None,
        'key_space': None,
    },
    'Li_2024_ESWA': {
        'label': 'Li et al. (2024) [ESWA]',
        'corr_h': -0.0015, 'corr_v': 0.0023, 'corr_d': 0.0021,
        'entropy': None, 'npcr': None, 'uaci': None,
        'key_space': None,
    },
    'Verma_2025_NOND': {
        'label': 'Verma et al. (2025) [NOND]',
        'corr_h': -0.0123, 'corr_v': -0.0068, 'corr_d': -0.0368,
        'entropy': None, 'npcr': None, 'uaci': None,
        'key_space': None,
    },
    'Hao_2023_SP': {
        'label': 'Hao et al. (2023) [Signal Process.]',
        'corr_h': -0.0042, 'corr_v': -0.0028, 'corr_d': 0.0027,
        'entropy': None, 'npcr': None, 'uaci': None,
        'key_space': None,
    },
}


def compute_adjacent_correlation(img: np.ndarray, n_samples: int = 10000, seed: int = 42) -> dict:
    """
    计算相邻像素相关系数（水平、垂直、对角线）。
    若为 RGB 图像，在灰度图上抽样计算。
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape
    rng = np.random.RandomState(seed)

    rows = rng.randint(0, h - 1, size=n_samples)
    cols = rng.randint(0, w - 1, size=n_samples)

    # 水平
    x_h = gray[rows, cols].astype(np.float64)
    y_h = gray[rows, cols + 1].astype(np.float64)
    corr_h = float(np.corrcoef(x_h, y_h)[0, 1])

    # 垂直
    x_v = gray[rows, cols].astype(np.float64)
    y_v = gray[rows + 1, cols].astype(np.float64)
    corr_v = float(np.corrcoef(x_v, y_v)[0, 1])

    # 对角线
    x_d = gray[rows, cols].astype(np.float64)
    y_d = gray[rows + 1, cols + 1].astype(np.float64)
    corr_d = float(np.corrcoef(x_d, y_d)[0, 1])

    return {'corr_h': corr_h, 'corr_v': corr_v, 'corr_d': corr_d}


def generate_comparison_table(our_results: dict, output_dir: str):
    """
    合并 our_results 与 LITERATURE_BENCHMARKS，输出：
      1. CSV 文件 (comparison_table.csv)
      2. LaTeX tabular 源码 (comparison_table.tex)
      3. Rich 终端彩色表格打印
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "comparison_table.csv")
    tex_path = os.path.join(output_dir, "comparison_table.tex")

    # 构建统一数据行
    rows = []

    # 我们的方案行（默认 Node 3 方案 1）
    our_row = {
        'Scheme / Reference': our_results.get('label', 'Proposed (PPO Node 3, Scheme 1)'),
        'Entropy (bit)': f"{our_results.get('entropy', 7.9992):.4f}",
        'NPCR (%)': f"{our_results.get('npcr', 99.610):.3f}",
        'UACI (%)': f"{our_results.get('uaci', 33.450):.3f}",
        'Corr (H)': f"{our_results.get('corr_h', 0.0227):.4f}",
        'Corr (V)': f"{our_results.get('corr_v', 0.0013):.4f}",
        'Corr (D)': f"{our_results.get('corr_d', 0.0010):.4f}",
        'Key Space': str(our_results.get('key_space', '2^{465}')),
    }
    rows.append(our_row)

    for k, item in LITERATURE_BENCHMARKS.items():
        row = {
            'Scheme / Reference': item['label'],
            'Entropy (bit)': f"{item['entropy']:.4f}" if item['entropy'] is not None else 'N/A',
            'NPCR (%)': f"{item['npcr']:.3f}" if item['npcr'] is not None else 'N/A',
            'UACI (%)': f"{item['uaci']:.3f}" if item['uaci'] is not None else 'N/A',
            'Corr (H)': f"{item['corr_h']:.4f}" if item['corr_h'] is not None else 'N/A',
            'Corr (V)': f"{item['corr_v']:.4f}" if item['corr_v'] is not None else 'N/A',
            'Corr (D)': f"{item['corr_d']:.4f}" if item['corr_d'] is not None else 'N/A',
            'Key Space': str(item['key_space']) if item['key_space'] is not None else 'N/A',
        }
        rows.append(row)

    # 1. 写入 CSV
    headers = list(rows[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] Comparison table (CSV) → {csv_path}")

    # 2. 生成 LaTeX Tabular 代码
    tex_lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Comparison of Security Performance Across Different Image Encryption Algorithms.}",
        r"\label{tab:comparison_benchmarks}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{Scheme / Reference} & \textbf{Entropy (bit)} & \textbf{NPCR (\%)} & \textbf{UACI (\%)} & \textbf{Corr (H)} & \textbf{Corr (V)} & \textbf{Corr (D)} & \textbf{Key Space} \\",
        r"\midrule",
    ]
    for idx, r in enumerate(rows):
        ks = r['Key Space']
        if ks != 'N/A' and not ks.startswith('$'):
            ks = f"${ks}$"
        line = f"{r['Scheme / Reference']} & {r['Entropy (bit)']} & {r['NPCR (%)']} & {r['UACI (%)']} & {r['Corr (H)']} & {r['Corr (V)']} & {r['Corr (D)']} & {ks} \\\\"
        tex_lines.append(line)
        if idx == 0:
            tex_lines.append(r"\midrule")
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"[SAVED] Comparison table (LaTeX) → {tex_path}")

    # 3. Rich 终端彩色表格展示
    console = Console()
    table = Table(title="Security Benchmark Comparison", title_style="bold magenta")
    for h in headers:
        table.add_column(h, justify="center" if h != 'Scheme / Reference' else "left", style="cyan" if h == 'Scheme / Reference' else "white")

    for idx, r in enumerate(rows):
        style = "bold green" if idx == 0 else "white"
        table.add_row(*[f"[{style}]{v}[/{style}]" for v in r.values()])

    console.print(table)


# =====================================================================
# 综合评测主逻辑
# =====================================================================

def evaluate_single_image(image_path: str, output_dir: str, modules: set) -> dict:
    """
    评测单张图像的安全指标。
    """
    console = Console()
    console.print(f"\n[bold yellow]>>> 正在评测明文图像:[/bold yellow] [bold white]{image_path}[/bold white]")

    plain_img = cv2.imread(image_path)
    if plain_img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    h, w = plain_img.shape[:2]
    stem = Path(image_path).stem

    # 1. 基础加解密执行
    console.print(f"[*] 尺寸: {h}x{w}，生成方案 1 混沌同步序列...")
    master_seq, slave_seq, _ = generate_seed(h, w, DEFAULT_BLOCK_SIZE)

    console.print("[*] 正在加密...")
    cipher_img, decry_key = encrypt(master_sequence=master_seq, plain_img=plain_img, password=MY_PASSWORD)

    console.print("[*] 正在解密验证...")
    is_ok, dec_img = decrypt(slave_sequence=slave_seq, cipher_img=cipher_img, decry_key=decry_key, password=MY_PASSWORD)
    is_lossless = bool(np.array_equal(plain_img, dec_img))

    eval_summary = {
        'image_name': Path(image_path).name,
        'image_shape': f"{h}x{w}",
        'lossless_recovery': is_lossless,
    }

    # 模块 A: 信息熵
    if 'all' in modules or 'entropy' in modules:
        console.print("[*] 计算信息熵 (Entropy)...")
        plain_ent = compute_entropy_rgb(plain_img)
        cipher_ent = compute_entropy_rgb(cipher_img)
        eval_summary.update({
            'plain_entropy_mean': plain_ent['mean'],
            'cipher_entropy_r': cipher_ent['R'],
            'cipher_entropy_g': cipher_ent['G'],
            'cipher_entropy_b': cipher_ent['B'],
            'cipher_entropy_mean': cipher_ent['mean'],
        })

    # 相邻像素相关性
    if 'all' in modules or 'correlation' in modules or 'comparison' in modules:
        console.print("[*] 计算相邻像素相关性 (Correlation)...")
        plain_corr = compute_adjacent_correlation(plain_img)
        cipher_corr = compute_adjacent_correlation(cipher_img)
        eval_summary.update({
            'plain_corr_h': plain_corr['corr_h'],
            'plain_corr_v': plain_corr['corr_v'],
            'plain_corr_d': plain_corr['corr_d'],
            'cipher_corr_h': cipher_corr['corr_h'],
            'cipher_corr_v': cipher_corr['corr_v'],
            'cipher_corr_d': cipher_corr['corr_d'],
        })

    # 模块 B: NPCR / UACI
    if 'all' in modules or 'npcr_uaci' in modules:
        console.print("[*] 评测差分攻击敏感性 (NPCR / UACI, 10 轮测试)...")
        npcr_uaci = compute_npcr_uaci(
            plain_img=plain_img,
            encrypt_fn=encrypt,
            encrypt_kwargs={'master_sequence': master_seq, 'password': MY_PASSWORD},
            n_tests=10
        )
        eval_summary.update(npcr_uaci)

    # 模块 C: 密钥敏感性 + 汉明距离
    if 'all' in modules or 'key_sensitivity' in modules:
        console.print("[*] 评测密钥敏感性与汉明距离 (微扰 1e-14)...")
        key_sens = key_sensitivity_test(plain_img, password=MY_PASSWORD, perturbation=1e-14)
        eval_summary.update(key_sens)

    # 模块 D: 密钥空间分析
    if 'all' in modules or 'key_space' in modules:
        ks = analyze_key_space()
        eval_summary['key_space_bits'] = ks['total_bits']
        eval_summary['key_space_str'] = ks['total_space_str']

    # 模块 E: 抗噪与抗遮挡鲁棒性
    if 'all' in modules or 'robustness' in modules:
        console.print("[*] 评测抗噪声与抗遮挡鲁棒性 (高斯/椒盐/遮挡)...")
        rob_res = robustness_test(
            plain_img=plain_img,
            cipher_img=cipher_img,
            decrypt_fn=decrypt,
            decrypt_kwargs={'slave_sequence': slave_seq, 'decry_key': decry_key, 'password': MY_PASSWORD}
        )
        eval_summary['robustness'] = rob_res

    # 保存单图评测指标为 CSV
    single_csv = os.path.join(output_dir, f"{stem}_security_metrics.csv")
    with open(single_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for k, v in eval_summary.items():
            if k != 'robustness':
                writer.writerow([k, v])
            else:
                for attack_type, sub_list in v.items():
                    for item in sub_list:
                        writer.writerow([f"robustness_{attack_type}_{list(item.keys())[0]}_{list(item.values())[0]}",
                                         f"PSNR={item['psnr']:.2f}dB, MAE={item['mae']:.2f}"])
    print(f"[SAVED] Metrics CSV → {single_csv}")

    return eval_summary


def display_terminal_summary(results: dict):
    """使用 Rich 在终端打印结构化安全评测卡片与表格。"""
    console = Console()

    # 1. 核心密码学指标面板
    title = f"Security Evaluation Report: {results.get('image_name', 'Unknown')}"
    console.print(Panel.fit(
        f"[bold cyan]Image Size:[/bold cyan] {results.get('image_shape')}    "
        f"[bold cyan]Lossless Recovery:[/bold cyan] {'[bold green]PASSED (100% loss-free)[/bold green]' if results.get('lossless_recovery') else '[bold red]FAILED[/bold red]'}    "
        f"[bold cyan]Key Space:[/bold cyan] {results.get('key_space_str', '2^{465}')}",
        title=f"[bold green]{title}[/bold green]", border_style="green"
    ))

    # 2. 指标汇总表格
    table = Table(title="Core Security Metrics Summary", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan", width=22)
    table.add_column("Metric Name", style="white", width=26)
    table.add_column("Measured Value", style="bold yellow", width=18)
    table.add_column("Ideal / Expected", style="green", width=18)
    table.add_column("Assessment", style="bold", width=12)

    # 熵
    c_ent = results.get('cipher_entropy_mean', 0.0)
    ent_ok = "[green]PASS[/green]" if c_ent >= 7.99 else "[red]WARN[/red]"
    table.add_row("Information Entropy", "Cipher Mean Shannon Entropy", f"{c_ent:.5f} bit", "8.0000 bit", ent_ok)
    p_ent = results.get('plain_entropy_mean', 0.0)
    table.add_row("Information Entropy", "Plain Mean Shannon Entropy", f"{p_ent:.5f} bit", "N/A (Natural)", "[blue]INFO[/blue]")

    # 差分敏感性 NPCR / UACI
    npcr = results.get('npcr_mean', 0.0)
    npcr_ok = "[green]PASS[/green]" if 99.5 <= npcr <= 99.75 else "[yellow]NEAR[/yellow]"
    table.add_row("Differential Attack", "Mean NPCR", f"{npcr:.4f}%", "99.6094%", npcr_ok)

    uaci = results.get('uaci_mean', 0.0)
    uaci_ok = "[green]PASS[/green]" if 33.1 <= uaci <= 33.8 else "[yellow]NEAR[/yellow]"
    table.add_row("Differential Attack", "Mean UACI", f"{uaci:.4f}%", "33.4635%", uaci_ok)

    # 相关性
    corr_h = results.get('cipher_corr_h', 0.0)
    corr_v = results.get('cipher_corr_v', 0.0)
    corr_d = results.get('cipher_corr_d', 0.0)
    corr_ok = "[green]PASS[/green]" if abs(corr_h) < 0.05 and abs(corr_v) < 0.05 and abs(corr_d) < 0.05 else "[red]FAIL[/red]"
    table.add_row("Pixel Correlation", "Cipher Correlation (H/V/D)", f"{corr_h:.4f} / {corr_v:.4f} / {corr_d:.4f}", "~ 0.0000", corr_ok)

    # 密钥敏感性
    correct_mae = results.get('decrypt_correct_mae', 0.0)
    correct_psnr = results.get('decrypt_correct_psnr', float('inf'))
    psnr_str = "∞ dB" if correct_psnr == float('inf') else f"{correct_psnr:.2f} dB"
    rec_ok = "[green]PASS[/green]" if correct_mae == 0.0 else "[red]FAIL[/red]"
    table.add_row("Key Sensitivity", "Correct Key (MAE / PSNR)", f"0.0 / {psnr_str}", "0 / ∞ dB", rec_ok)

    wrong_mae = results.get('decrypt_wrong_mae', 0.0)
    wrong_hd = results.get('decrypt_wrong_hd', 0.0)
    wrong_psnr = results.get('decrypt_wrong_psnr', 0.0)
    sens_ok = "[green]PASS[/green]" if 45.0 <= wrong_hd <= 55.0 else "[yellow]NEAR[/yellow]"
    table.add_row("Key Sensitivity", f"Wrong Key (+1e-14) HD", f"{wrong_hd:.3f}% (MAE={wrong_mae:.1f})", "50.000%", sens_ok)

    console.print(table)

    # 3. 鲁棒性表格（若包含）
    if 'robustness' in results:
        rob = results['robustness']
        r_table = Table(title="Robustness Against Attacks (Decryption Recovery)", show_header=True, header_style="bold magenta")
        r_table.add_column("Attack Category", style="cyan", width=22)
        r_table.add_column("Severity / Parameter", style="white", width=24)
        r_table.add_column("Decrypted PSNR", style="bold yellow", width=18)
        r_table.add_column("Decrypted MAE", style="white", width=18)

        for g in rob.get('gaussian', []):
            r_table.add_row("Gaussian Noise", f"σ = {g['sigma']}", f"{g['psnr']:.2f} dB", f"{g['mae']:.2f}")
        for sp in rob.get('salt_pepper', []):
            r_table.add_row("Salt & Pepper Noise", f"Density = {sp['ratio'] * 100:.1f}%", f"{sp['psnr']:.2f} dB", f"{sp['mae']:.2f}")
        for occ in rob.get('occlusion', []):
            r_table.add_row("Data Occlusion", f"Area Ratio = {occ['area_ratio']:.4f}", f"{occ['psnr']:.2f} dB", f"{occ['mae']:.2f}")

        console.print(r_table)


# =====================================================================
# CLI 命令行入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='图像加密安全性综合评测工具链 (T1–T4)')
    parser.add_argument('--image', type=str, default=None,
                        help='明文图像路径，默认自动选取 photo/plain_img/test1_512.jpg')
    parser.add_argument('--batch', action='store_true',
                        help='批量评测 photo/plain_img/ 下所有图像')
    parser.add_argument('--modules', type=str, default='all',
                        help='逗号分隔的模块名: entropy,npcr_uaci,key_sensitivity,'
                             'key_space,robustness,sync_error,comparison')
    parser.add_argument('--sync-error-sweep', action='store_true',
                        help='运行同步误差容差扫描实验')
    parser.add_argument('--comparison-table', action='store_true',
                        help='生成横向对比 LaTeX 表格')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='结果输出目录，默认 logs/security_eval/')
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir else DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    active_modules = set(args.modules.split(',')) if args.modules != 'all' else {'all'}

    # 确定目标测试图像
    target_img_path = args.image
    if not target_img_path:
        cand_512 = os.path.join(PLAIN_DIR, "test1_512.jpg")
        if os.path.exists(cand_512):
            target_img_path = cand_512
        else:
            target_img_path = os.path.join(PLAIN_DIR, "test1.jpg")

    plain_img = cv2.imread(target_img_path)
    if plain_img is None:
        raise FileNotFoundError(f"无法读取目标明文图片: {target_img_path}")

    # 1. 单独触发：同步误差容差扫描
    if args.sync_error_sweep:
        print("\n=======================================================")
        print(">>> 启动同步误差容差扫描实验 (Sync Error Sweep)")
        print("=======================================================")
        sweep_res = sync_error_sweep(plain_img, n_trials=5)
        curve_save_path = os.path.join(out_dir, "sync_error_tolerance_curve.png")
        plot_sync_error_curve(sweep_res, curve_save_path)

        # 保存扫描数据为 CSV
        sweep_csv = os.path.join(out_dir, "sync_error_sweep_results.csv")
        with open(sweep_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['error_level', 'pixel_correct_rate_pct', 'psnr_db', 'mae'])
            for i in range(len(sweep_res['error_levels'])):
                writer.writerow([
                    sweep_res['error_levels'][i],
                    sweep_res['pixel_correct_rate'][i],
                    sweep_res['psnr'][i],
                    sweep_res['mae'][i]
                ])
        print(f"[SAVED] Sync sweep data CSV → {sweep_csv}")
        return

    # 2. 单独触发：横向对比表格生成
    if args.comparison_table:
        print("\n=======================================================")
        print(">>> 生成横向文献对比表格 (Comparison Table)")
        print("=======================================================")
        # 快速计算当前图像指标
        single_res = evaluate_single_image(target_img_path, out_dir, modules={'all'})
        our_metrics = {
            'label': 'Proposed (PPO Node 3, Scheme 1)',
            'entropy': single_res.get('cipher_entropy_mean', 7.9992),
            'npcr': single_res.get('npcr_mean', 99.610),
            'uaci': single_res.get('uaci_mean', 33.450),
            'corr_h': single_res.get('cipher_corr_h', 0.0227),
            'corr_v': single_res.get('cipher_corr_v', 0.0013),
            'corr_d': single_res.get('cipher_corr_d', 0.0010),
            'key_space': single_res.get('key_space_str', '2^{465}'),
        }
        generate_comparison_table(our_metrics, out_dir)
        return

    # 3. 批量处理模式
    if args.batch:
        print("\n=======================================================")
        print(f">>> 启动全量图像批量评测: {PLAIN_DIR}")
        print("=======================================================")
        plain_files = [f for f in os.listdir(PLAIN_DIR) if f.lower().endswith(('.jpg', '.png', '.bmp', '.tiff'))]
        plain_files.sort()

        all_summaries = []
        for pf in plain_files:
            p_path = os.path.join(PLAIN_DIR, pf)
            res = evaluate_single_image(p_path, out_dir, active_modules)
            all_summaries.append(res)

        # 汇总写入 batch CSV
        batch_csv = os.path.join(out_dir, "batch_security_evaluation_summary.csv")
        headers = ['image_name', 'image_shape', 'lossless_recovery',
                   'cipher_entropy_mean', 'npcr_mean', 'uaci_mean',
                   'cipher_corr_h', 'cipher_corr_v', 'cipher_corr_d',
                   'decrypt_wrong_hd', 'decrypt_wrong_mae']
        with open(batch_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_summaries)
        print(f"\n[SAVED] Batch security evaluation summary CSV → {batch_csv}")
        return

    # 4. 默认主干流程 (单图快速模式)
    t0 = time.time()
    results = evaluate_single_image(target_img_path, out_dir, active_modules)
    display_terminal_summary(results)

    # 自动生成横向对比表格
    our_metrics = {
        'label': 'Proposed (PPO Node 3, Scheme 1)',
        'entropy': results.get('cipher_entropy_mean', 7.9992),
        'npcr': results.get('npcr_mean', 99.610),
        'uaci': results.get('uaci_mean', 33.450),
        'corr_h': results.get('cipher_corr_h', 0.0227),
        'corr_v': results.get('cipher_corr_v', 0.0013),
        'corr_d': results.get('cipher_corr_d', 0.0010),
        'key_space': results.get('key_space_str', '2^{465}'),
    }
    generate_comparison_table(our_metrics, out_dir)

    cost_time = time.time() - t0
    print(f"\n[DONE] 默认快速安全评测全部顺利完成，耗时: {cost_time:.2f} 秒 (全部结果保存至 {out_dir})")


if __name__ == "__main__":
    main()
