import sys
from pathlib import Path

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

# 1. 路径设置与环境导入
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent
CODE_DIR = CURRENT_FILE.parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

import env  # 必须导入以注册 FHNEnv-v0


def main() -> None:
    # 2. 直接固定加载最终模型
    model_path = PROJECT_ROOT / "models" / "fhn" / "ppo_fhn_final.zip"
    if not model_path.exists():
        print(f"找不到模型文件: {model_path}，请先运行 train_fhn.py")
        return
        
    print(f"正在加载模型: {model_path}")
    model = PPO.load(model_path, device="cpu")

    # 3. 初始化环境
    env_inst = gymnasium.make("FHNEnv-v0")
    base_env = env_inst.unwrapped
    obs, _ = env_inst.reset(seed=2026)

    drive_hist, response_hist, error_hist = [], [], []

    # 4. 跑一个环境回合，保存数据
    done = False
    while not done:
        drive_hist.append(base_env.statex.copy())
        response_hist.append(base_env.statey.copy())
        error_hist.append((base_env.statey - base_env.statex).copy())

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env_inst.step(action)
        done = terminated or truncated

    env_inst.close()

    # 5. 数据处理与极简画图
    drive = np.array(drive_hist)
    response = np.array(response_hist)
    error = np.abs(np.array(error_hist))
    t = np.arange(len(drive))

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    axes[0].plot(t, drive[:, 0], label="Drive x", color="black")
    axes[0].plot(t, response[:, 0], label="Response x", color="tab:blue", linestyle="--")
    axes[0].set_ylabel("State x")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, drive[:, 1], label="Drive y", color="black")
    axes[1].plot(t, response[:, 1], label="Response y", color="tab:blue", linestyle="--")
    axes[1].set_ylabel("State y")
    axes[1].legend(loc="upper right")

    axes[2].plot(t, drive[:, 2], label="Drive z", color="black")
    axes[2].plot(t, response[:, 2], label="Response z", color="tab:blue", linestyle="--")
    axes[2].set_ylabel("State z")
    axes[2].legend(loc="upper right")

    axes[3].plot(t, error[:, 0], label="|Error x|", color="tab:red")
    axes[3].plot(t, error[:, 1], label="|Error y|", color="tab:orange")
    axes[3].plot(t, error[:, 2], label="|Error z|", color="tab:green")
    axes[3].set_xlabel("Time Step")
    axes[3].set_ylabel("Absolute Error")
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    out_dir = PROJECT_ROOT / "logs" / "fhn" / "test_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "fhn_simple_test.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"测试完成！图表已保存至: {save_path}")
    print(f"最终绝对误差 -> x: {error[-1, 0]:.6f}, y: {error[-1, 1]:.6f}, z: {error[-1, 2]:.6f}")


if __name__ == "__main__":
    main()
