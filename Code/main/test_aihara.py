from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable
import gymnasium
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO


current_file = Path(__file__).resolve()
project_dir = current_file.parent.parent.parent
code_dir = current_file.parent.parent
if str(code_dir) not in sys.path:
    sys.path.append(str(code_dir))

import env  


ENV_ID = "AiharaEnv-v0"


def resolve_model_path(model_arg: str | None) -> Path:
    if model_arg:
        path = Path(model_arg).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path

    final_model = project_dir / "models" / "aihara" / "ppo_aihara_final.zip"
    if final_model.exists():
        return final_model
    best_model = project_dir / "models" / "aihara" / "best" / "best_model.zip"
    if best_model.exists():
        return best_model
    raise FileNotFoundError("No Aihara model found. Tried models/aihara/ppo_aihara_final.zip and models/aihara/best/best_model.zip")


def run_episode(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    seed: int,
    deterministic: bool,
    mismatch_scale: float,
    pulse_step: int | None,
    pulse_width: int,
    pulse_vector: tuple[float, float],
) -> dict:
    env_inst = gymnasium.make(
        ENV_ID,
        mismatch_scale=mismatch_scale,
        pulse_step=pulse_step,
        pulse_width=pulse_width,
        pulse_vector=pulse_vector,
    )
    base_env = env_inst.unwrapped
    obs, _ = env_inst.reset(seed=seed)

    drive_hist = []
    response_hist = []
    error_hist = []
    reward_hist = []
    pulse_hist = []

    # 修复：先记录 t=0 的初始状态 (此时没有任何 action, reward 为 0, pulse 为 False)
    drive_hist.append(base_env.statex.copy())
    response_hist.append(base_env.statey.copy())
    error_hist.append((base_env.statey - base_env.statex).copy())
    reward_hist.append(0.0) 
    pulse_hist.append(False)

    done = False
    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env_inst.step(action)
        done = terminated or truncated

        drive_hist.append(base_env.statex.copy())
        response_hist.append(base_env.statey.copy())
        error_hist.append((base_env.statey - base_env.statex).copy())
        reward_hist.append(float(reward))
        pulse_hist.append(bool(info["pulse_applied"]))

    env_inst.close()
    return {
        "drive": np.asarray(drive_hist),
        "response": np.asarray(response_hist),
        "error": np.asarray(error_hist),
        "reward": np.asarray(reward_hist),
        "pulse": np.asarray(pulse_hist),
    }


def save_plots(results_ppo: dict, results_base: dict, results_linear: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(results_ppo["reward"]))

    abs_err_ppo = np.abs(results_ppo["error"])
    abs_err_base = np.abs(results_base["error"])
    abs_err_linear = np.abs(results_linear["error"])

    total_err_ppo = abs_err_ppo[:, 0] + abs_err_ppo[:, 1]
    total_err_base = abs_err_base[:, 0] + abs_err_base[:, 1]
    total_err_linear = abs_err_linear[:, 0] + abs_err_linear[:, 1]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, results_base["response"][:, 0], label="Baseline resp x", color="gray", linestyle="--", alpha=0.6)
    axes[0].plot(t, results_ppo["drive"][:, 0], label="Drive x", color="black", linewidth=2)
    axes[0].plot(t, results_ppo["response"][:, 0], label="PPO resp x", color="tab:blue", alpha=0.9)
    axes[0].plot(t, results_linear["response"][:, 0], label="Linear resp x", color="tab:orange", alpha=0.9)
    axes[0].set_ylabel("State X")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, results_base["response"][:, 1], label="Baseline resp y", color="gray", linestyle="--", alpha=0.6)
    axes[1].plot(t, results_ppo["drive"][:, 1], label="Drive y", color="black", linewidth=2)
    axes[1].plot(t, results_ppo["response"][:, 1], label="PPO resp y", color="tab:blue", alpha=0.9)
    axes[1].plot(t, results_linear["response"][:, 1], label="Linear resp y", color="tab:orange", alpha=0.9)
    axes[1].set_ylabel("State Y")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, total_err_base, label="Baseline total |e|", color="gray", linestyle="--", alpha=0.7)
    axes[2].plot(t, total_err_ppo, label="PPO total |e|", color="tab:blue", linewidth=2)
    axes[2].plot(t, total_err_linear, label="Linear total |e|", color="tab:orange", linewidth=2)

    if results_ppo["pulse"].any():
        pulse_idx = np.where(results_ppo["pulse"])[0]
        if len(pulse_idx) > 0:
            axes[2].axvline(pulse_idx[0], color="red", linestyle=":", alpha=0.8, label="Pulse")

    axes[2].set_ylabel("Total Abs Error")
    axes[2].set_xlabel("Step")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "aihara_sync_curves_with_baseline.png", dpi=200)
    plt.close(fig)


def build_case_dir(
    root_out_dir: Path,
    mismatch: float,
    pulse_step: int | None,
    pulse_width: int,
    pulse_vector: tuple[float, float],
) -> Path:
    case_dir = root_out_dir / f"mismatch_{mismatch:.1f}"
    if pulse_step is None:
        if abs(mismatch) < 1e-12:
            return case_dir / "nominal_system"  # 既无失配也无脉冲 (理想标称系统)
        else:
            return case_dir / "mismatch_only"   # 仅有失配，无脉冲

    if abs(mismatch) < 1e-12:
        scenario_dir = "pulse_only"
    else:
        scenario_dir = "combined_mismatch_pulse"

    pulse_x, pulse_y = pulse_vector
    pulse_tag = (
        f"pulse_step_{pulse_step}"
        f"_width_{pulse_width}"
        f"_x_{pulse_x:.2f}"
        f"_y_{pulse_y:.2f}"
    )
    return case_dir / scenario_dir / pulse_tag



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model on AiharaEnv-v0.")
    parser.add_argument("--model", type=str, default=None, help="Path to model zip file.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for test episode.")
    parser.add_argument("--out", type=str, default="logs/aihara/test_plots", help="Output folder for plots.")
    parser.add_argument("--mismatch", type=float, default=0.0, help="Mismatch scale for the response neuron.")
    parser.add_argument("--pulse-step", type=int, default=None, help="Apply a pulse disturbance to the drive neuron at this step.")
    parser.add_argument("--pulse-width", type=int, default=1, help="Number of steps for which the pulse stays active.")
    parser.add_argument("--pulse-x", type=float, default=0.0, help="Pulse injected into the drive x state.")
    parser.add_argument("--pulse-y", type=float, default=0.0, help="Pulse injected into the drive y state.")
    parser.add_argument("--sweep", action="store_true", help="Run mismatch robustness sweep at 0.0, 0.5, 1.0, 1.5, 2.0")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    model = PPO.load(str(model_path), device="cpu")
    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    def baseline_policy(obs):
        return np.zeros(1, dtype=np.float32)
    
    def linear_feedback_policy(obs):
        # 你的 obs 是 [drive_x, drive_y, e_x, e_y] / 3.0
        e_x = obs[2] * 3.0  # 还原真实的 x 误差
        e_y = obs[3] * 3.0  # 还原真实的 y 误差
        k = 1.5  # 随便调一个能让标称环境收敛的比例系数
        action = -k * (e_x + e_y) / 50.0  # 除以 50 是因为环境里会再乘以 self.scale
        # 必须和 PPO 享受同等待遇：硬限幅在正负 1 之间，保证公平对比控制能量
        return np.clip(np.array([action], dtype=np.float32), -1.0, 1.0)

    out_dir = (project_dir / args.out).resolve()
    pulse_vector = (args.pulse_x, args.pulse_y)

    sweep_values = [args.mismatch]
    if args.sweep:
        sweep_values = [0.0, 0.5, 1.0, 1.5, 2.0]

    print(f"Model: {model_path}")
    for mismatch in sweep_values:
        print(f"Testing mismatch={mismatch:.1f}, pulse_step={args.pulse_step}, pulse_vector={pulse_vector}")
        results_baseline = run_episode(
            baseline_policy,
            seed=args.seed,
            mismatch_scale=mismatch,
            pulse_step=args.pulse_step,
            pulse_width=args.pulse_width,
            pulse_vector=pulse_vector,
            deterministic=True,
        )

        results_ppo = run_episode(
            ppo_policy,
            seed=args.seed,
            mismatch_scale=mismatch,
            pulse_step=args.pulse_step,
            pulse_width=args.pulse_width,
            pulse_vector=pulse_vector,
            deterministic= True
        )

        results_linear = run_episode(
            linear_feedback_policy,
            seed=args.seed,
            mismatch_scale=mismatch,
            pulse_step=args.pulse_step,
            pulse_width=args.pulse_width,
            pulse_vector=pulse_vector,
            deterministic= True
        )

        error_ppo = results_ppo["error"]
        error_base = results_baseline["error"]
        error_linear = results_linear["error"]


        mean_l1_ppo = np.linalg.norm(error_ppo, ord=1, axis=1).mean()
        mean_l1_base = np.linalg.norm(error_base, ord=1, axis=1).mean()
        mean_l1_linear = np.linalg.norm(error_linear, ord=1, axis=1).mean()

        abs_err = np.abs(error_ppo)
        mismatch_out_dir = build_case_dir(
            out_dir,
            mismatch=mismatch,
            pulse_step=args.pulse_step,
            pulse_width=args.pulse_width,
            pulse_vector=pulse_vector,
        )
        save_plots(results_ppo, results_baseline, results_linear, mismatch_out_dir)

        pulse_steps = int(results_ppo["pulse"].sum())
        print(f"Mismatch scale: {mismatch:.1f}")
        print(f"Episode steps: {len(error_ppo)}")
        print(f"Baseline mean L1 error: {mean_l1_base:.6f}")
        print(f"PPO mean L1 error: {mean_l1_ppo:.6f}")
        print(f"Linear mean L1 error: {mean_l1_linear:.10f}")
        print(f"Final PPO L1 error: {np.linalg.norm(error_ppo[-1], ord=1):.6f}")
        print(f"Final Linear L1 error: {np.linalg.norm(error_linear[-1], ord=1):.10f}")
        print(f"Baseline final L1 error: {np.linalg.norm(error_base[-1], ord=1):.6f}")
        print(f"|ex| mean={abs_err[:, 0].mean():.6f}, p90={np.percentile(abs_err[:, 0], 90):.6f}")
        print(f"|ey| mean={abs_err[:, 1].mean():.6f}, p90={np.percentile(abs_err[:, 1], 90):.6f}")
        print(f"Mean reward: {results_ppo['reward'].mean():.6f}")
        print(f"Pulse-active steps: {pulse_steps}")
        print(f"Saved plots to: {mismatch_out_dir}")


if __name__ == "__main__":
    main()
