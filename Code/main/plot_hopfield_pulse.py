from __future__ import annotations

import argparse
import sys
from pathlib import Path

import re

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent

if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from env.continuous_hopfield_env import ContinuousHopfieldEnv


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

DEFAULT_PATTERN = "ppo_continuous_hopfield_node3_final_seed_*.zip"
nodes_to_plot = (2, 3)  # top subplot: node 2, bottom subplot: node 3

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pulse recovery curves for Hopfield control.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mismatch", type=float, default=0)
    parser.add_argument("--pulse-step", type=int, default=200)
    parser.add_argument("--pulse-width", type=int, default=1)
    parser.add_argument("--pulse-vector", type=float, nargs=3, default=[5.0, 5.0, 5.0])
    parser.add_argument("--out", type=str, default="logs/hopfield/test_results/pulse_response.png")
    parser.add_argument("--include-zero", action="store_true")
    return parser.parse_args()


def resolve_model_paths(args: argparse.Namespace) -> list[Path]:
    base_dir = PROJECT_ROOT / "models" / "hopfield"

    if args.model:
        model_paths = [Path(args.model).resolve()]
    else:
        model_dir = Path(args.model_dir) if args.model_dir else base_dir
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
        model_paths = sorted(model_dir.glob(args.pattern))

    if not model_paths:
        raise FileNotFoundError("No model files matched the given settings.")

    missing = [str(path) for path in model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Model file(s) not found:\n" + "\n".join(missing))

    return model_paths


def make_env(
    mismatch_scale: float,
    pulse_step: int,
    pulse_width: int,
    pulse_vector: tuple[float, float, float],
    pinning_node: int,
) -> ContinuousHopfieldEnv:
    return ContinuousHopfieldEnv(
        mismatch_scale=mismatch_scale,
        pulse_step=pulse_step,
        pulse_width=pulse_width,
        pulse_vector=pulse_vector,
        pinning_node=pinning_node,
    )


def run_episode(policy_fn, env: ContinuousHopfieldEnv, seed: int) -> dict[str, np.ndarray]:
    obs, _ = env.reset(seed=seed)

    error_hist = [(env.statey - env.statex).copy()]
    action_hist = [0.0]

    done = False
    while not done:
        action = policy_fn(obs, env)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        error_hist.append((env.statey - env.statex).copy())
        action_hist.append(float(action[0]) * env.scale)

    return {
        "error": np.asarray(error_hist, dtype=np.float32),
        "action": np.asarray(action_hist, dtype=np.float32),
    }


def evaluate_ppo_models(
    model_paths: list[Path],
    args: argparse.Namespace,
    pinning_node: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    error_histories: list[np.ndarray] = []
    action_curves: list[np.ndarray] = []
    step_time: float | None = None

    for model_path in model_paths:
        model = PPO.load(str(model_path), device="cpu")

        def ppo_policy(obs: np.ndarray, env: ContinuousHopfieldEnv) -> np.ndarray:
            action, _ = model.predict(obs, deterministic=True)
            return np.asarray(action, dtype=np.float32)

        env = make_env(
            mismatch_scale=args.mismatch,
            pulse_step=args.pulse_step,
            pulse_width=args.pulse_width,
            pulse_vector=tuple(args.pulse_vector),
            pinning_node=pinning_node,
        )
        if step_time is None:
            step_time = env.dt * env.rk4_steps

        result = run_episode(ppo_policy, env, args.seed)
        env.close()

        error_histories.append(result["error"])
        action_curves.append(result["action"])

    error_stack = np.stack(error_histories, axis=0)
    action_stack = np.stack(action_curves, axis=0)
    return (
        np.mean(error_stack, axis=0),
        np.std(error_stack, axis=0),
        np.mean(action_stack, axis=0),
        float(step_time),
    )


def evaluate_baseline(args: argparse.Namespace, mode: str, pinning_node: int) -> tuple[np.ndarray, np.ndarray]:
    env = make_env(
        mismatch_scale=args.mismatch,
        pulse_step=args.pulse_step,
        pulse_width=args.pulse_width,
        pulse_vector=tuple(args.pulse_vector),
        pinning_node=pinning_node,
    )

    if mode == "linear":
        def policy(obs: np.ndarray, env: ContinuousHopfieldEnv) -> np.ndarray:
            controlled_idx = int(np.argmax(env.B))
            error = env.statey[controlled_idx] - env.statex[controlled_idx]
            action = -8.0 * error / env.scale
            return np.clip(np.array([action], dtype=np.float32), -1.0, 1.0)
    elif mode == "zero":
        def policy(obs: np.ndarray, env: ContinuousHopfieldEnv) -> np.ndarray:
            return np.zeros(1, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported baseline mode: {mode}")

    result = run_episode(policy, env, args.seed)
    env.close()
    return result["error"], result["action"]


def l1_curve(error_history: np.ndarray) -> np.ndarray:
    return np.linalg.norm(error_history, ord=1, axis=1)


def plot_pulse_response(
    node_results: dict[int, dict[str, np.ndarray | float]],
    pulse_step: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True, sharey=True)
    pulse_label_added = False

    for subplot_index, node in enumerate(nodes_to_plot):
        axis = axes[subplot_index]
        node_data = node_results[node]
        ppo_mean_error = l1_curve(np.asarray(node_data["ppo_mean_error"]))
        linear_error = l1_curve(np.asarray(node_data["linear_error"]))
        step_time = float(node_data["step_time"])
        time_axis = np.arange(len(ppo_mean_error)) * step_time
        pulse_time = pulse_step * step_time

        axis.plot(
            time_axis,
            ppo_mean_error,
            color="#d62728",
            linestyle="-",
            label="PPO",
        )
        axis.plot(
            time_axis,
            linear_error,
            color="#1f77b4",
            linestyle="--",
            label="Linear",
        )

        if not pulse_label_added:
            axis.axvline(pulse_time, color="purple", linestyle=":", linewidth=2, label="Pulse")
            pulse_label_added = True
        else:
            axis.axvline(pulse_time, color="purple", linestyle=":", linewidth=2)

        if subplot_index == 0:
            axis.set_ylabel(r"$||e(t)||_1$")
        axis.grid(True, linestyle=":", alpha=0.5)
        if node==2:
            axis.set_title(f"(a) Node {node}", pad=6)
        else:
            axis.set_title(f"(b) Node {node}", pad=6)
        axis.set_yscale("log")
        axis.set_xlabel("Time/s")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.18, top=0.78, wspace=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Pulse response figure saved to: {out_path}")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    node_results: dict[int, dict[str, np.ndarray | float]] = {}
    for node in nodes_to_plot:
        node_args = argparse.Namespace(**vars(args))
        node_args.pattern = re.sub(r"node(\d+)", f"node{node}", args.pattern)
        model_paths = resolve_model_paths(node_args)
        pinning_node = node - 1

        print(f"Using {len(model_paths)} PPO model(s) for node {node}.")

        ppo_mean_error, ppo_std_error, ppo_mean_action, step_time = evaluate_ppo_models(
            model_paths, node_args, pinning_node
        )
        linear_error, linear_action = evaluate_baseline(node_args, mode="linear", pinning_node=pinning_node)
        node_results[node] = {
            "ppo_mean_error": ppo_mean_error,
            "ppo_std_error": ppo_std_error,
            "ppo_mean_action": ppo_mean_action,
            "linear_error": linear_error,
            "linear_action": linear_action,
            "step_time": step_time,
        }

    plot_pulse_response(
        node_results=node_results,
        pulse_step=args.pulse_step,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
