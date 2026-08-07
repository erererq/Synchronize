import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent

plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,

        # Font sizes
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,

        # Lines
        "lines.linewidth": 1.5,
        "axes.linewidth": 1,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.major.size": 3,
        "ytick.major.size": 3,

        # Tick style
        "xtick.direction": "in",
        "ytick.direction": "in",

        # Save quality
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # PDF font embedding
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

logs_dir = Path("logs") / "hopfield"
TIME_PER_STEP = 20 / 400  # 400 steps are approximately 20 s.
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))
from stable_baselines3 import PPO
from env.continuous_hopfield_env import ContinuousHopfieldEnv
# 假设你保存了 3 个不同 seed 训练出的最终模型
model_path=[
    ("models/hopfield/ppo_continuous_hopfield_node1_final_seed_42",0),
    ("models/hopfield/ppo_continuous_hopfield_node1_final_seed_123",0),
    ("models/hopfield/ppo_continuous_hopfield_node1_final_seed_1024",0),
    ("models/hopfield/ppo_continuous_hopfield_node2_final_seed_42",1),
    ("models/hopfield/ppo_continuous_hopfield_node2_final_seed_123",1),
    ("models/hopfield/ppo_continuous_hopfield_node2_final_seed_1024",1),
    ("models/hopfield/ppo_continuous_hopfield_node3_final_seed_42",2),
    ("models/hopfield/ppo_continuous_hopfield_node3_final_seed_123",2),
    ("models/hopfield/ppo_continuous_hopfield_node3_final_seed_1024",2)
]

start_seed = 1000
seed_list = list(range(start_seed, start_seed + 100))
seed_list = [42]  # 直接指定你想测试的 seed 列表
for seed in seed_list:
    all_seeds_error_hist = []
    for path, pinning_node in model_path:
        model=PPO.load(str(path),device="cpu")
        env = ContinuousHopfieldEnv(pinning_node=pinning_node)  # 根据模型对应的 pinning_node 创建环境
        # 重置环境，注意测试时一般用固定的 seed 以保证公平对比
        obs, _ = env.reset(seed=seed)

        error_hist=[]
        # 记录初始误差 (假设你要算向量的二范数，或者直接取绝对值)
        error_norm = np.linalg.norm(env.statey-env.statex,ord=1)  # 这里以 L1 范数为例，你也可以用 np.linalg.norm(obs) 来计算 L2 范数
        error_hist.append(error_norm)

        done=False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            error_norm = np.linalg.norm(env.statey-env.statex,ord=1)
            error_hist.append(error_norm)
        all_seeds_error_hist.append(error_hist)

    # 将列表转换为 Numpy 矩阵，形状为 (3个种子, 回合最大步数)
    all_seeds_error_hist = np.array(all_seeds_error_hist)

    # 绘制误差曲线
    # 1. 生成 X 轴坐标，单位为秒
    time_s = np.arange(all_seeds_error_hist.shape[1]) * TIME_PER_STEP

    # 2. 沿着“种子”维度 (axis=0) 计算均值和标准差
    mean_error1 = np.mean(all_seeds_error_hist[0:3], axis=0)
    std_error1 = np.std(all_seeds_error_hist[0:3], axis=0)

    mean_error2 = np.mean(all_seeds_error_hist[3:6], axis=0)
    std_error2 = np.std(all_seeds_error_hist[3:6], axis=0)

    mean_error3 = np.mean(all_seeds_error_hist[6:9], axis=0)
    std_error3 = np.std(all_seeds_error_hist[6:9], axis=0)

    # print(f"Mean Error at Final Step:Node 1: {mean_error1[-1]:.4f} | Node 2: {mean_error2[-1]:.4f} | Node 3: {mean_error3[-1]:.4f}")
    # print(f"MAE: Node 1: {np.mean(mean_error1):.4f} | Node 2: {np.mean(mean_error2):.4f} | Node 3: {np.mean(mean_error3):.4f}")
    # # 3. 开始画图
    plt.figure(figsize=(10, 5))

    plt.yscale("log")  # 设置 Y 轴为对数尺度，适合展示误差的指数级下降趋势
    # 调整 Y 轴下限，防止对数坐标报错或过于拉伸
    plt.ylim(bottom=1e-3,top=1e2)
    # 画出平均误差演化轨迹
    plt.plot(time_s, mean_error1, color="blue", label="PPO Models Node 1 (Mean)")
    plt.plot(time_s, mean_error2, color="green", label="PPO Models Node 2 (Mean)")
    plt.plot(time_s, mean_error3, color="red", label="PPO Models Node 3 (Mean)")

    # # 画出阴影区间 (这里用 1倍标准差 作为波动范围)
    # plt.fill_between(
    #     time_s, 
    #     mean_error - std_error, 
    #     mean_error + std_error, 
    #     color="blue", 
    #     alpha=0.2, 
    #     label="$\pm 1$ Standard Deviation"
    # )

    # 4. 美化图表
    # plt.title("Synchronization Error under Different Pinning Nodes")
    plt.xlabel("Time/s")
    plt.ylabel("Total Error Norm $e$")


    plt.legend(loc="upper right", frameon=True, framealpha=0.6)
    plt.grid(True, which="both", linestyle=':', alpha=0.2)
    plt.tight_layout()

    plt.savefig(logs_dir / "test_results" / "compare_different_nodes.png", dpi=300)
    plt.savefig(logs_dir / "test_results" / "compare_different_nodes.pdf", dpi=300)
    # plt.show()

    # # 5. 计算并追加写入 RMSE 和 MAE 数据
    # # ==========================================
    # # 准备目标文件路径
    # metrics_file = logs_dir / "test_results" / "compare_different_node_metrics_summary.csv"

    # # 确保目标文件夹存在
    # metrics_file.parent.mkdir(parents=True, exist_ok=True)

    # # 检查文件是否已经存在，如果不存在说明是第一次写，需要加上表头
    # file_exists = metrics_file.exists()

    # # 计算 MAE (Mean Absolute Error)
    # # 先对每条运行单独按时间求指标，再在对应 node 的多条运行之间取平均，
    # # 与 final_l1_error 的统计口径保持一致。
    # mae_node1 = np.mean(np.mean(all_seeds_error_hist[0:3], axis=1))
    # mae_node2 = np.mean(np.mean(all_seeds_error_hist[3:6], axis=1))
    # mae_node3 = np.mean(np.mean(all_seeds_error_hist[6:9], axis=1))
    # # 计算 RMSE (Root Mean Square Error)
    # rmse_node1 = np.mean(np.sqrt(np.mean(all_seeds_error_hist[0:3] ** 2, axis=1)))
    # rmse_node2 = np.mean(np.sqrt(np.mean(all_seeds_error_hist[3:6] ** 2, axis=1)))
    # rmse_node3 = np.mean(np.sqrt(np.mean(all_seeds_error_hist[6:9] ** 2, axis=1)))
    # # 计算 final l1 error
    # final_l1_error1 = np.mean(all_seeds_error_hist[0:3,-1])
    # final_l1_error2 = np.mean(all_seeds_error_hist[3:6,-1])
    # final_l1_error3 = np.mean(all_seeds_error_hist[6:9,-1])

    # # 以追加模式 ('a') 打开文件
    # with open(metrics_file, "a", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     # 如果文件不存在，写入表头
    #     if not file_exists:
    #         writer.writerow(["Experiment", "Node 1 MAE", "Node 2 MAE", "Node 3 MAE", "Node 1 RMSE", "Node 2 RMSE", "Node 3 RMSE","Final l1 error1","Final l1 error2","Final l1 error3"])

    #     # 追加写入这次实验的数据
    #     # 这里也可以加上日期、超参数等标识，方便以后区分
    #     experiment_name = "Hopfield Compare Different Nodes"
    #     writer.writerow([experiment_name, mae_node1, mae_node2, mae_node3, rmse_node1, rmse_node2, rmse_node3,final_l1_error1,final_l1_error2,final_l1_error3])

    # # print(f"Metrics successfully appended to {metrics_file}")
