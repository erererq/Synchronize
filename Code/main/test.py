from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent
CODE_DIR = CURRENT_FILE.parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

import env  # noqa: F401


def resolve_model_path(model_arg: str | None) -> Path:
    if model_arg:
        path = Path(model_arg).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path

    final_model = PROJECT_ROOT / "models" / "ppo_lorenz_final.zip"
    if final_model.exists():
        return final_model
    best_model = PROJECT_ROOT / "models" / "best" / "best_model.zip"
    if best_model.exists():
        return best_model
    raise FileNotFoundError("No model found. Tried models/ppo_lorenz_final.zip and models/best/best_model.zip")


def run_episode(model: PPO, seed: int, deterministic: bool) -> dict:
    env_inst = gymnasium.make("LorenzEnv-v0")
    base_env = env_inst.unwrapped
    obs, _ = env_inst.reset(seed=seed)

    error_vec_hist = []
    error_l1_hist = []
    reward_hist = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env_inst.step(action)
        done = terminated or truncated

        err_vec = base_env.statey - base_env.statex
        error_vec_hist.append(err_vec.copy())
        error_l1_hist.append(float(np.linalg.norm(err_vec, ord=1)))
        reward_hist.append(float(reward))

    env_inst.close()

    return {
        "error_vec": np.asarray(error_vec_hist),
        "error_l1": np.asarray(error_l1_hist),
        "reward": np.asarray(reward_hist),
    }


def save_plots(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(results["error_l1"]))

    abs_err = np.abs(results["error_vec"])
    threshold = 1e-2

    fig5, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, abs_err[:, i], linewidth=1.0, label=f"|e{i + 1}|")
        ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8, label="1e-2")
        ax.set_ylabel(f"|e{i + 1}|")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("step")
    fig5.suptitle("Absolute Component Errors |ex|, |ey|, |ez|, |ew|")
    fig5.tight_layout()
    fig5.savefig(out_dir / "error_components_abs.png", dpi=150)
    plt.close(fig5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model and plot test curves.")
    parser.add_argument("--model", type=str, default=None, help="Path to model zip file.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for test episode.")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy for inference.")
    parser.add_argument("--out", type=str, default="logs/test_plots", help="Output folder for plots.")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    model = PPO.load(str(model_path), device="cpu")
    results = run_episode(model, seed=args.seed, deterministic=not args.stochastic)

    out_dir = (PROJECT_ROOT / args.out).resolve()
    save_plots(results, out_dir)

    error_l1 = results["error_l1"]
    error_vec = results["error_vec"]
    abs_err = np.abs(error_vec)
    reward = results["reward"]
    threshold = 1e-2
    metric_start = min(500, len(error_vec))
    post_error_vec = error_vec[metric_start:]
    post_abs_err = abs_err[metric_start:]
    post_rmse = float(np.sqrt(np.mean(np.square(post_error_vec)))) if len(post_error_vec) > 0 else float("nan")
    post_mae = float(np.mean(post_abs_err)) if len(post_abs_err) > 0 else float("nan")
    tail_steps = min(200, len(abs_err))
    tail = abs_err[-tail_steps:]
    all_dims_under_ratio = float(np.mean(np.all(tail < threshold, axis=1)))

    print(f"Model: {model_path}")
    print(f"Episode steps: {len(error_l1)}")
    print(f"Final L1 error: {error_l1[-1]:.6f}")
    print(f"Mean L1 error: {error_l1.mean():.6f}")
    print(f"Median L1 error: {np.median(error_l1):.6f}")
    print(f"Mean RMSE after step {metric_start}: {post_rmse:.6f}")
    print(f"Mean MAE after step {metric_start}: {post_mae:.6f}")
    for i in range(4):
        comp = abs_err[:, i]
        print(
            f"|e{i + 1}| final={comp[-1]:.6f}, mean={comp.mean():.6f}, "
            f"p90={np.percentile(comp, 90):.6f}"
        )
    print(
        f"Last {tail_steps} steps with all |ex,ey,ez,ew| < 1e-2 ratio: "
        f"{all_dims_under_ratio:.3f}"
    )
    print(f"Mean reward: {reward.mean():.6f}")
    print(f"Saved plots to: {out_dir}")


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_lorenz_attractor():
    # 1. 实例化你的环境
    env = gymnasium.make("LorenzEnv-v0")
    base_env = env.unwrapped
    obs, _ = env.reset()
    
    # 2. 跑一段时间的“预热”，舍弃前 2000 步的瞬态轨迹
    for _ in range(2000):
        # 输入 0 动作，让驱动系统自由演化
        base_env.step(np.array([0.0, 0.0, 0.0]))
        
    # 3. 开始收集稳态下的系统轨迹
    steps_to_record = 50000  # 减小记录步数，防止 Matplotlib 3D 渲染卡死
    trajectory = np.zeros((steps_to_record, 4))
    
    for i in range(steps_to_record):
        base_env.step(np.array([0.0, 0.0, 0.0]))
        # 记录驱动系统的状态 (x, y, z, w)
        trajectory[i] = base_env.statex
        
    # 提取 x, y, z 坐标用于 3D 绘图 (使用 [::2] 降采样，既保证美观又加快渲染)
    x_vals = trajectory[::2, 0]
    y_vals = trajectory[::2, 1]
    z_vals = trajectory[::2, 2]

    # 4. 使用 Matplotlib 画 3D 相图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 当线条成千上万时，需要把线宽(lw)调得更细，透明度(alpha)调得更低，才能看出内部的层次感
    ax.plot(x_vals, y_vals, z_vals, lw=0.2, color='b', alpha=0.5)
    
    ax.set_title("Phase Portrait of 4D Hyperchaotic Lorenz/Chen System (x-y-z projection)")
    ax.set_xlabel("X state")
    ax.set_ylabel("Y state")
    ax.set_zlabel("Z state")
    
    out_path = PROJECT_ROOT / "logs" / "test_plots" / "lorenz_attractor.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Lorenz 混沌相图已保存至: {out_path}")
    plt.close(fig)


    

if __name__ == "__main__":
    main()
    plot_lorenz_attractor()
