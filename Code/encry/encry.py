"""
encry.py  —  图像加密/解密主流程

职责：
  1. 基于混沌同步序列的 RGB 图像加密（Q 置乱 + DNA 双轮编解码）
  2. 对称解密与验证
  3. 单图 / 批量处理入口

依赖：
  - chaos_sequence.generate_seed / check_synchronization 提供混沌序列
"""

import os
import traceback
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import hashlib

# 适配 Windows 控制台编码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---- 导入路径设置 ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Code.encry.chaos_sequence import (
    generate_seed,
    check_synchronization,
    ENABLE_SYNC_CHECK,
    DEFAULT_BLOCK_SIZE,
)

# ==========================================
# 路径配置
# ==========================================

DATA_EXPERIMENT_DIR = 'photo'
PLAIN_DIR = os.path.join(DATA_EXPERIMENT_DIR, "plain_img")
CIPHER_DIR = os.path.join(DATA_EXPERIMENT_DIR, "cipher_img")
DECRYPTED_DIR = os.path.join(DATA_EXPERIMENT_DIR, "decrypted_img")

# ==========================================
# 全局常量
# ==========================================

# 默认图像尺寸（批量模式会动态覆盖）
M_image = 512
N_image = 512
# 分块大小
p = DEFAULT_BLOCK_SIZE
# 用户密码
MY_PASSWORD = "password987"

# 24 种 DNA 碱基编码规则（S_4 对称群全排列，2bit → 碱基的双射）
# 规则 1~8 保持经典 Watson-Crick 互补配对；规则 9~24 拓展其余全部 16 种置换，彻底打破互补不变子空间缺陷
coding_rules = {
    1: {'00': 'A', '01': 'G', '10': 'C', '11': 'T'},  # Watson-Crick 互补
    2: {'00': 'A', '01': 'C', '10': 'G', '11': 'T'},  # Watson-Crick 互补
    3: {'00': 'T', '01': 'G', '10': 'C', '11': 'A'},  # Watson-Crick 互补
    4: {'00': 'T', '01': 'C', '10': 'G', '11': 'A'},  # Watson-Crick 互补
    5: {'00': 'C', '01': 'A', '10': 'T', '11': 'G'},  # Watson-Crick 互补
    6: {'00': 'G', '01': 'A', '10': 'T', '11': 'C'},  # Watson-Crick 互补
    7: {'00': 'C', '01': 'T', '10': 'A', '11': 'G'},  # Watson-Crick 互补
    8: {'00': 'G', '01': 'T', '10': 'A', '11': 'C'},  # Watson-Crick 互补
    9: {'00': 'A', '01': 'C', '10': 'T', '11': 'G'},  # S_4 拓展全排列
    10: {'00': 'A', '01': 'G', '10': 'T', '11': 'C'},  # S_4 拓展全排列
    11: {'00': 'A', '01': 'T', '10': 'C', '11': 'G'},  # S_4 拓展全排列
    12: {'00': 'A', '01': 'T', '10': 'G', '11': 'C'},  # S_4 拓展全排列
    13: {'00': 'C', '01': 'A', '10': 'G', '11': 'T'},  # S_4 拓展全排列
    14: {'00': 'C', '01': 'G', '10': 'A', '11': 'T'},  # S_4 拓展全排列
    15: {'00': 'C', '01': 'G', '10': 'T', '11': 'A'},  # S_4 拓展全排列
    16: {'00': 'C', '01': 'T', '10': 'G', '11': 'A'},  # S_4 拓展全排列
    17: {'00': 'G', '01': 'A', '10': 'C', '11': 'T'},  # S_4 拓展全排列
    18: {'00': 'G', '01': 'C', '10': 'A', '11': 'T'},  # S_4 拓展全排列
    19: {'00': 'G', '01': 'C', '10': 'T', '11': 'A'},  # S_4 拓展全排列
    20: {'00': 'G', '01': 'T', '10': 'C', '11': 'A'},  # S_4 拓展全排列
    21: {'00': 'T', '01': 'A', '10': 'C', '11': 'G'},  # S_4 拓展全排列
    22: {'00': 'T', '01': 'A', '10': 'G', '11': 'C'},  # S_4 拓展全排列
    23: {'00': 'T', '01': 'C', '10': 'A', '11': 'G'},  # S_4 拓展全排列
    24: {'00': 'T', '01': 'G', '10': 'A', '11': 'C'},  # S_4 拓展全排列
}


# ==========================================
# 密钥派生
# ==========================================

def string_to_initial_value(password: str, offset: float = 0.0) -> float:
    """
    将用户输入的字符串密码转换为 (0, 1) 之间的浮点数，用作混沌系统的初值。
    """
    hash_obj = hashlib.sha256(password.encode('utf-8'))
    hex_dig = hash_obj.hexdigest()
    int_val = int(hex_dig, 16)
    float_val = int_val / (2 ** 256) + offset

    # 避免正好是 0 或 1 或超出 (0, 1) 区间
    if float_val <= 0:
        float_val = 0.123456789
    if float_val >= 1:
        float_val = 0.987654321

    return float_val


def logistic_map(theta: float, initial_value: float, num_iterations: int) -> list:
    """
    生成 Logistic 映射序列：x_{n+1} = θ * x_n * (1 - x_n)

    Args:
        theta: 控制参数（3.9999 处于混沌区）。
        initial_value: 初始值。
        num_iterations: 迭代次数。

    Returns:
        包含 num_iterations+1 个元素的序列列表。
    """
    sequence = [initial_value]
    for _ in range(num_iterations):
        last_value = sequence[-1]
        next_value = theta * last_value * (1 - last_value)
        sequence.append(next_value)
    return sequence


# ==========================================
# 掩码矩阵
# ==========================================

def reshape_sequence_to_Q(logistic_sequence: list, height: int, width: int) -> np.ndarray:
    """
    将混沌序列量化为 0~255 整数并 reshape 为 h×w 掩码矩阵 Q。
    """
    K1_array = np.array(logistic_sequence)
    K1_prime = np.mod(np.round(K1_array * 10 ** 4), 256).astype(np.uint8)
    Q = K1_prime.reshape(height, width)
    return Q


# ==========================================
# 分块与重组
# ==========================================

def split_into_blocks(matrix: np.ndarray, block_size: int = p) -> list:
    """将二维矩阵拆分为一系列 block_size × block_size 的小块。"""
    blocks = []
    height, width = matrix.shape
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = matrix[i:i + block_size, j:j + block_size]
            blocks.append(block)
    return blocks


def reshape_blocks_to_channel(blocks, height: int, width: int, block_size: int = p) -> np.ndarray:
    """将一系列小块重新组合为 h×w 二维矩阵。"""
    reshaped_matrix = np.zeros((height, width), dtype=np.uint8)
    index = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            reshaped_matrix[i:i + block_size, j:j + block_size] = blocks[index]
            index += 1
    return reshaped_matrix


# ==========================================
# 二进制 ↔ 十进制转换
# ==========================================

def convert_to_8bit_binary(matrix_block_all: list) -> list:
    """将矩阵块中每个整数元素转换为 8 位二进制字符串。"""
    binary_block_all = []
    for matrix_block in matrix_block_all:
        binary_vectorizer = np.vectorize(
            lambda x: format(int(x), '08b') if str(x).isdigit() else ''
        )
        binary_block = binary_vectorizer(matrix_block)
        binary_block_all.append(binary_block.tolist())
    return binary_block_all


def convert_binary_to_decimal(matrix_block_all) -> list:
    """将 8 位二进制字符串转换回 0~255 整数。"""
    decimal_block_all = []
    for binary_block in matrix_block_all:
        decimal_block = []
        for row in binary_block:
            decimal_row = [
                int(binary_str, 2) if isinstance(binary_str, str) and len(binary_str) == 8 else 0
                for binary_str in row
            ]
            decimal_block.append(decimal_row)
        decimal_block_all.append(decimal_block)
    return decimal_block_all


# ==========================================
# DNA 编解码
# ==========================================

def binary_to_dna(binary_blocks, bd, rules):
    """
    二进制串 → DNA 碱基序列。每个块的编码规则由混沌序列 bd[i] 决定。
    """
    if len(bd) != len(binary_blocks):
        raise ValueError(
            f"Length mismatch: bd={len(bd)}, binary_blocks={len(binary_blocks)}"
        )

    dna_blocks = []
    for i, block in enumerate(binary_blocks):
        rule = rules.get(bd[i])
        if not rule:
            raise ValueError(f"No coding rule found for value {bd[i]}.")

        dna_block = []
        for row in block:
            dna_row = [
                ''.join([rule[binary_str[j:j + 2]] for j in range(0, len(binary_str), 2)])
                for binary_str in row
            ]
            dna_block.append(dna_row)
        dna_blocks.append(dna_block)

    return dna_blocks


def dna_to_binary(dna_blocks, db, rules):
    """
    DNA 碱基序列 → 二进制串。每个块的解码规则由混沌序列 db[i] 决定。
    编码/解码使用不同规则是扩散的核心机制。
    """
    if len(db) != len(dna_blocks):
        raise ValueError(
            f"Length mismatch: db={len(db)}, dna_blocks={len(dna_blocks)}"
        )

    binary_blocks = []
    for i, block in enumerate(dna_blocks):
        rule = rules.get(db[i])
        if not rule:
            raise ValueError(f"No coding rule found for value {db[i]}.")

        reverse_rule = {v: k for k, v in rule.items()}
        binary_block = []

        for row in block:
            binary_row = []
            for dna_str in row:
                if len(dna_str) != 4:
                    raise ValueError(f"DNA string {dna_str} is not exactly 4 characters long.")
                try:
                    binary_str = ''.join([reverse_rule[char] for char in dna_str])
                    binary_row.append(binary_str)
                except KeyError as e:
                    raise ValueError(f"Invalid DNA character {e} found in string {dna_str}.")
            binary_block.append(binary_row)
        binary_blocks.append(binary_block)

    return binary_blocks


# ==========================================
# 图像 I/O
# ==========================================

def save_image(multi_channel_img: np.ndarray, path: str):
    """保存多通道图像，自动创建目录。"""
    ndim = multi_channel_img.ndim
    if ndim == 2:
        channels = 1
    elif ndim == 3:
        channels = multi_channel_img.shape[2]
    else:
        raise ValueError(f"不支持的图像维度: {ndim}")

    if channels not in [1, 3, 4]:
        raise ValueError(f"不支持的通道数: {channels}")

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    success = cv2.imwrite(path, multi_channel_img)
    if not success:
        raise IOError(f"图像保存失败: {path}")

    print(f"成功保存 {channels} 通道图像至: {path}")


# ==========================================
# 加密
# ==========================================

def encrypt(master_sequence: tuple = None, plain_img: np.ndarray = None, password: str = None, offset: float = 0.0, **kwargs) -> tuple:
    """
    三阶段加密：Q 置乱 (XOR) → DNA 三轮编解码 → 输出密文。

    Args:
        master_sequence: (L1..L6) 量化后的混沌主序列。若为 None 则内部自动生成。
        plain_img: 明文图像 (BGR)。
        password: 用户密码，用于派生 Logistic Map 初值。
        offset: Logistic Map 初值微扰量（用于敏感性测试）。

    Returns:
        cipher_img: 密文图像。
        decry_key: 解密密钥 (img_word, initial_value)。
    """
    # 兼容处理：若第 1 个参数传入的是 plain_img (ndarray)，而第 2 个参数为 None 或序列
    if isinstance(master_sequence, np.ndarray) and plain_img is None:
        plain_img = master_sequence
        master_sequence = kwargs.get('master_sequence', None)

    if plain_img is None:
        raise ValueError("plain_img cannot be None.")

    height, width = plain_img.shape[:2]

    if master_sequence is None:
        master_sequence, _, _ = generate_seed(height, width, p)

    L1, L2, L3, L4, L5, L6 = master_sequence

    # ===== 阶段 1：Q 置乱 =====
    channels = cv2.split(plain_img)

    # 生成 Logistic Map 初值（密码 + 图像像素和混合）
    i1 = np.sum(channels[0])
    i2 = np.sum(channels[1])
    i3 = np.sum(channels[2])
    img_word = str(int(i1) + int(i2) + int(i3))

    if isinstance(password, str):
        raw_seed_str = password + img_word
    elif password is None:
        raw_seed_str = img_word
    else:
        raise ValueError("Password must be a string or None.")

    initial_value = string_to_initial_value(raw_seed_str, offset=offset)
    theta = 3.9999
    num_iterations = height * width

    logistic_sequence = logistic_map(theta, initial_value, num_iterations - 1)
    Q = reshape_sequence_to_Q(logistic_sequence, height, width)

    # 分块并 XOR 扩散
    channels_blocks = [split_into_blocks(ch) for ch in channels]
    blocks_Q = split_into_blocks(Q)

    diffused_channels = []
    for blocks_I in channels_blocks:
        diffused_channel = [np.bitwise_xor(block_I, block_Q)
                            for block_I, block_Q in zip(blocks_I, blocks_Q)]
        diffused_channels.append(diffused_channel)

    # ===== 阶段 2：DNA 三轮不对称编解码 =====
    bin_channels = [convert_to_8bit_binary(ch) for ch in diffused_channels]

    # 第 1 轮：L1 编码 → L2 解码
    dna_channels = [binary_to_dna(bin_ch, L1, coding_rules) for bin_ch in bin_channels]
    bin_channels = [dna_to_binary(dna_ch, L2, coding_rules) for dna_ch in dna_channels]

    # 第 2 轮：L3 编码 → L4 解码
    dna_channels = [binary_to_dna(bin_ch, L3, coding_rules) for bin_ch in bin_channels]
    bin_channels = [dna_to_binary(dna_ch, L4, coding_rules) for dna_ch in dna_channels]

    # 第 3 轮：L5 编码 → L6 解码
    dna_channels = [binary_to_dna(bin_ch, L5, coding_rules) for bin_ch in bin_channels]
    bin_channels = [dna_to_binary(dna_ch, L6, coding_rules) for dna_ch in dna_channels]

    # ===== 阶段 3：输出 =====
    channels_blocks = [convert_binary_to_decimal(bin_ch) for bin_ch in bin_channels]
    channels = [reshape_blocks_to_channel(ch, height, width) for ch in channels_blocks]

    cipher_img = cv2.merge(channels)
    decry_key = (img_word, initial_value)

    return cipher_img, decry_key


# ==========================================
# 解密
# ==========================================

def decrypt(slave_sequence: tuple, cipher_img: np.ndarray,
            decry_key: tuple, password: str = None, offset: float = 0.0) -> tuple:
    """
    解密算法：加密的严格逆过程。

    Args:
        slave_sequence: (S1..S6) — 量化后的混沌从序列。
        cipher_img: 密文图像。
        decry_key: 解密密钥。
        password: 用户密码。
        offset: 初值微扰量（用于敏感性测试）。

    Returns:
        (is_success, decrypted_img): 解密结果。
    """
    img_word = decry_key[0]

    # 重新生成 Q 矩阵
    if isinstance(password, str):
        raw_seed_str = password + img_word
    elif password is None:
        raw_seed_str = img_word
    else:
        raise ValueError("Password must be a string or None.")

    initial_value_of_Q = string_to_initial_value(raw_seed_str, offset=offset)

    cipher_channels = cv2.split(cipher_img)
    height, width = cipher_img.shape[:2]

    theta = 3.9999
    num_iterations = height * width
    logistic_sequence = logistic_map(theta, initial_value_of_Q, num_iterations - 1)
    Q = reshape_sequence_to_Q(logistic_sequence, height, width)
    blocks_Q = split_into_blocks(Q)

    S1, S2, S3, S4, S5, S6 = slave_sequence

    # DNA 逆序三轮解码：
    # 逆第 3 轮（对应加密第 3 轮）：S6 编码 → S5 解码
    # 逆第 2 轮（对应加密第 2 轮）：S4 编码 → S3 解码
    # 逆第 1 轮（对应加密第 1 轮）：S2 编码 → S1 解码
    cipher_channel_blocks = [split_into_blocks(ch) for ch in cipher_channels]
    bin_channel_blocks = [convert_to_8bit_binary(ch) for ch in cipher_channel_blocks]

    # 逆第 3 轮：S6 编码 → S5 解码
    dna_channel_blocks = [binary_to_dna(bin_ch, S6, coding_rules) for bin_ch in bin_channel_blocks]
    bin_channel_blocks = [dna_to_binary(dna_ch, S5, coding_rules) for dna_ch in dna_channel_blocks]

    # 逆第 2 轮：S4 编码 → S3 解码
    dna_channel_blocks = [binary_to_dna(bin_ch, S4, coding_rules) for bin_ch in bin_channel_blocks]
    bin_channel_blocks = [dna_to_binary(dna_ch, S3, coding_rules) for dna_ch in dna_channel_blocks]

    # 逆第 1 轮：S2 编码 → S1 解码
    dna_channel_blocks = [binary_to_dna(bin_ch, S2, coding_rules) for bin_ch in bin_channel_blocks]
    bin_channel_blocks = [dna_to_binary(dna_ch, S1, coding_rules) for dna_ch in dna_channel_blocks]

    # 十进制还原
    channels_blocks = [convert_binary_to_decimal(bin_ch) for bin_ch in bin_channel_blocks]

    # Q 异或还原
    decrypted_channels_blocks = []
    for channel_blocks in channels_blocks:
        decrypted = [np.bitwise_xor(block_I, block_Q)
                     for block_I, block_Q in zip(channel_blocks, blocks_Q)]
        decrypted_channels_blocks.append(decrypted)

    decrypted_channels = [reshape_blocks_to_channel(ch, height, width)
                          for ch in decrypted_channels_blocks]
    decrypted_img = cv2.merge(decrypted_channels)

    return (True, decrypted_img)


# ==========================================
# 解密验证
# ==========================================

def check_decryption_pixel(plain_path: str, decrypted_path: str) -> bool:
    """像素级精确验证：解密图与原图是否零差异。"""
    plain_img = cv2.imread(plain_path)
    decrypted_img = cv2.imread(decrypted_path)
    if plain_img.shape != decrypted_img.shape:
        return False
    difference = cv2.absdiff(plain_img, decrypted_img)
    for channel in cv2.split(difference):
        if cv2.countNonZero(channel) != 0:
            return False
    return True


def check_decryption_psnr(plain_path: str, decrypted_path: str) -> bool:
    """PSNR 验证：> 40dB 视为解密成功。"""
    plain_img = cv2.imread(plain_path)
    decrypted_img = cv2.imread(decrypted_path)
    if plain_img.shape != decrypted_img.shape:
        return False
    psnr_value = cv2.PSNR(plain_img, decrypted_img)
    return psnr_value > 40


# ==========================================
# 流程编排
# ==========================================

def encrypt_and_decrypt(plain_path: str, cipher_path: str, decrypted_path: str,
                        password: str = None, master_sequence: tuple = None,
                        slave_sequence: tuple = None, offset: float = 0.0, **kwargs) -> bool:
    """
    单图完整流程：生成序列 (方案 1) → 同步检查 → 加密 → 解密 → 像素级验证。

    Args:
        plain_path: 明文图片路径。
        cipher_path: 密文图片保存路径。
        decrypted_path: 解密还原图片保存路径。
        password: 用户口令密码。
        master_sequence: 外部传入的主序列（可选）。
        slave_sequence: 外部传入的从序列（可选）。
        offset: 初值微扰量（可选）。
    """
    if not all([plain_path, cipher_path, decrypted_path]):
        raise ValueError("File paths cannot be None.")

    plain_img = cv2.imread(plain_path)
    if plain_img is None:
        raise FileNotFoundError(f"Unable to load image at {plain_path}")
    height, width = plain_img.shape[:2]

    if master_sequence is None or slave_sequence is None:
        if ENABLE_SYNC_CHECK:
            cnt = 0
            attempt_num = 5
            while True:
                cnt += 1
                print(f"Generating chaotic sequences [scheme1], attempt {cnt}...")
                master_sequence, slave_sequence, _ = generate_seed(height, width, p)
                if check_synchronization(master_sequence, slave_sequence):
                    print("Chaotic sequences [scheme1] passed the synchronization check.")
                    break
                elif cnt >= attempt_num:
                    print(f"Error: Unable to generate valid chaotic sequences after {attempt_num} attempts.")
                    return False
        else:
            master_sequence, slave_sequence, _ = generate_seed(height, width, p)

    # 加密
    cipher_img, decry_key = encrypt(master_sequence, plain_img, password, offset=offset)
    save_image(cipher_img, cipher_path)

    # 解密
    is_success, decry_img = decrypt(slave_sequence, cipher_img, decry_key, password, offset=offset)
    print("Decryption attempt completed.")
    save_image(decry_img, decrypted_path)

    # 验证
    if not check_decryption_pixel(plain_path, decrypted_path):
        print("Warning: Decryption verification failed: Pixel values do not match exactly.")
        return False
    else:
        print("Verification succeed [scheme1]: Pixel values match exactly (100% loss-free).")
        return True


def process_images_in_folder(source_dir: str, cipher_dir: str, decrypted_dir: str):
    """
    批量加解密：遍历 source_dir 下图片，增量处理（已存在则跳过）。
    """
    global M_image, N_image

    os.makedirs(cipher_dir, exist_ok=True)
    os.makedirs(decrypted_dir, exist_ok=True)

    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    files = os.listdir(source_dir)
    count_processed = 0
    count_skipped = 0

    print(f"Starting batch processing in: {source_dir}\n" + "-" * 40)

    for file_name in files:
        if file_name.lower().endswith(('.tiff', '.png', '.jpg', '.bmp')):
            plain_path = os.path.join(source_dir, file_name)
            cipher_path = os.path.join(cipher_dir, file_name)
            decrypted_path = os.path.join(decrypted_dir, file_name)

            print()
            if os.path.exists(decrypted_path):
                print(f"[SKIP] {file_name} already exists in decrypted folder.")
                count_skipped += 1
                continue

            temp_img = cv2.imread(plain_path)
            if temp_img is None:
                print(f"[ERROR] Could not read {file_name}, skipping.")
                continue

            M_image = temp_img.shape[0]
            N_image = temp_img.shape[1]
            print(f"[INFO] Global size set to: {M_image}x{N_image} for {file_name}")

            print(f"[ACTION] Processing new image: {file_name}")
            try:
                encrypt_and_decrypt(plain_path, cipher_path, decrypted_path, password=MY_PASSWORD)
                count_processed += 1
            except Exception as e:
                print(f"[ERROR] Failed to process {file_name}. Reason: {str(e)}")
                traceback.print_exc()

    print("-" * 40)
    print(f"Batch task finished. New Processed: {count_processed}, Skipped: {count_skipped}.")


# ==========================================
# __main__
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图像加密与解密主流程 (方案 1 纯 3D 动力学 24 DNA 规则)")
    parser.add_argument("--image", type=str, default=None,
                        help="指定要加解密的单张图像路径 (default: photo/plain_img/test1_512.jpg)")
    parser.add_argument("--batch", action="store_true",
                        help="执行文件夹批量处理模式")
    args = parser.parse_args()

    plain_folder = PLAIN_DIR
    cipher_folder = CIPHER_DIR
    decrypted_folder = DECRYPTED_DIR

    if args.batch:
        process_images_in_folder(plain_folder, cipher_folder, decrypted_folder)
    else:
        # 单图加解密（默认优先使用 512x512 调试小图）
        target_image = args.image
        if not target_image:
            candidate_512 = os.path.join(plain_folder, "test1_512.jpg")
            if os.path.exists(candidate_512):
                target_image = candidate_512
            else:
                target_image = os.path.join(plain_folder, "test1.jpg")
        if not os.path.exists(target_image):
            # 兼容寻找 plain_folder 下任意图片
            if os.path.exists(plain_folder):
                candidates = [f for f in os.listdir(plain_folder) if f.lower().endswith(('.jpg', '.png', '.tiff', '.bmp'))]
                if candidates:
                    target_image = os.path.join(plain_folder, candidates[0])

        if os.path.exists(target_image):
            base_name = os.path.splitext(os.path.basename(target_image))[0]
            cipher_path = os.path.join(cipher_folder, f"{base_name}_scheme1.png")
            decrypted_path = os.path.join(decrypted_folder, f"{base_name}_scheme1.png")

            print("=== 运行图像加解密 (方案 1) ===")
            print(f"明文图片: {target_image}")
            print(f"密文输出: {cipher_path}")
            print(f"解密输出: {decrypted_path}")

            encrypt_and_decrypt(
                plain_path=target_image,
                cipher_path=cipher_path,
                decrypted_path=decrypted_path,
                password=MY_PASSWORD
            )
        else:
            print(f"Error: Target image '{target_image}' not found.")
