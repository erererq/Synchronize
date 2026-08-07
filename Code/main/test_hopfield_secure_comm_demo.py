from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent

if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

if TYPE_CHECKING:
    from stable_baselines3 import PPO


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


DEFAULT_PATTERN = "ppo_continuous_hopfield_node3_final_seed_*.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal chaotic masking communication demo based on the existing "
            "continuous-time Hopfield synchronization controller."
        )
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default="models/hopfield")
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN)
    parser.add_argument("--node", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--linear-gain", type=float, default=8.0)
    parser.add_argument("--message-amplitude", type=float, default=0.10)
    parser.add_argument("--message-frequency", type=float, default=0.80)
    parser.add_argument("--message-phase", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="logs/hopfield/secure_comm_demo")
    return parser.parse_args()


def resolve_output_dir(out: str) -> Path:
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path.resolve()


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        model_path = model_path.resolve()
    else:
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
        matches = sorted(model_dir.glob(args.pattern))
        if not matches:
            raise FileNotFoundError(
                f"No model files matched {args.pattern!r} in {model_dir}."
            )
        model_path = matches[0].resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model_path


def parse_model_metadata(model_path: Path) -> dict[str, str | int]:
    stem = model_path.stem
    seed_match = re.search(r"seed_(\d+)", stem)
    node_match = re.search(r"node(\d+)", stem)
    return {
        "model_name": stem,
        "train_seed": int(seed_match.group(1)) if seed_match else -1,
        "node": int(node_match.group(1)) if node_match else -1,
    }


def make_env(pinning_node: int, max_steps: int):
    try:
        from env.continuous_hopfield_env import ContinuousHopfieldEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gymnasium and the local Hopfield environment are required to run the communication demo."
        ) from exc

    env = ContinuousHopfieldEnv(
        pinning_node=pinning_node,
        mismatch_scale=0.0,
        pulse_step=None,
        pulse_width=1,
        pulse_vector=(0.0, 0.0, 0.0),
        process_noise_std=0.0,
    )
    env.max_steps = int(max_steps)
    return env


def run_episode(policy_fn, env, seed: int) -> dict[str, np.ndarray]:
    obs, _ = env.reset(seed=seed)

    drive_hist = [env.statex.copy()]
    response_hist = [env.statey.copy()]
    error_hist = [(env.statey - env.statex).copy()]
    action_hist = [0.0]

    done = False
    while not done:
        action = policy_fn(obs, env)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        drive_hist.append(env.statex.copy())
        response_hist.append(env.statey.copy())
        error_hist.append((env.statey - env.statex).copy())
        action_hist.append(float(action[0]) * env.scale)

    return {
        "drive": np.asarray(drive_hist, dtype=np.float32),
        "response": np.asarray(response_hist, dtype=np.float32),
        "error": np.asarray(error_hist, dtype=np.float32),
        "action": np.asarray(action_hist, dtype=np.float32),
    }


def build_linear_policy(linear_gain: float):
    def linear_policy(_obs: np.ndarray, env) -> np.ndarray:
        controlled_idx = int(np.argmax(env.B))
        error = env.statey[controlled_idx] - env.statex[controlled_idx]
        action = -linear_gain * error / env.scale
        return np.clip(np.array([action], dtype=np.float32), -1.0, 1.0)

    return linear_policy


def build_ppo_policy(model: "PPO"):
    def ppo_policy(obs: np.ndarray, _env) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return ppo_policy


def build_message(
    num_steps: int,
    step_time: float,
    amplitude: float,
    frequency: float,
    phase: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_axis = np.arange(num_steps, dtype=np.float64) * step_time
    message = amplitude * np.sin(2.0 * np.pi * frequency * time_axis + phase)
    return time_axis, message.astype(np.float32)


def build_communication_signals(
    result: dict[str, np.ndarray],
    message: np.ndarray,
) -> dict[str, np.ndarray]:
    x1 = result["drive"][:, 0]
    y1 = result["response"][:, 0]
    sent_signal = x1 + message
    recovered = sent_signal - y1
    recovery_error = recovered - message
    return {
        "x1": x1.astype(np.float32),
        "y1": y1.astype(np.float32),
        "message": message.astype(np.float32),
        "sent_signal": sent_signal.astype(np.float32),
        "recovered_message": recovered.astype(np.float32),
        "recovery_error": recovery_error.astype(np.float32),
    }


def compute_comm_metrics(signals: dict[str, np.ndarray]) -> dict[str, float]:
    recovery_error = signals["recovery_error"]
    message = signals["message"]
    recovered = signals["recovered_message"]
    return {
        "message_mae": float(np.mean(np.abs(recovery_error))),
        "message_rmse": float(np.sqrt(np.mean(np.square(recovery_error)))),
        "message_max_abs_error": float(np.max(np.abs(recovery_error))),
        "message_corrcoef": float(np.corrcoef(message, recovered)[0, 1]),
    }


def save_csv(records: list[dict[str, float | int | str]], out_path: Path) -> None:
    if not records:
        return
    fieldnames = list(dict.fromkeys(key for row in records for key in row))
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved: {out_path}")


def save_signal_csv(
    time_axis: np.ndarray,
    signal_bank: dict[str, dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    rows: list[dict[str, float | str]] = []
    for method, signals in signal_bank.items():
        for idx, t in enumerate(time_axis):
            rows.append(
                {
                    "method": method,
                    "time": float(t),
                    "message": float(signals["message"][idx]),
                    "sent_signal": float(signals["sent_signal"][idx]),
                    "recovered_message": float(signals["recovered_message"][idx]),
                    "recovery_error": float(signals["recovery_error"][idx]),
                    "x1": float(signals["x1"][idx]),
                    "y1": float(signals["y1"][idx]),
                }
            )
    save_csv(rows, out_path)


def plot_demo(
    time_axis: np.ndarray,
    signal_bank: dict[str, dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.4), sharex=True, constrained_layout=True)

    example_method = next(iter(signal_bank))
    axes[0].plot(time_axis, signal_bank[example_method]["sent_signal"], color="0.45", linewidth=1.4)
    axes[0].plot(time_axis, signal_bank[example_method]["message"], color="black", linewidth=1.2, linestyle="--")
    axes[0].set_ylabel("Signal")
    axes[0].legend(["Masked signal $s(t)$", "Message $m(t)$"], loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.4)

    specs = [("PPO", "#d62728"), ("Linear", "#1f77b4")]
    for axis, (method, color) in zip(axes[1:], specs):
        signals = signal_bank[method]
        axis.plot(time_axis, signals["message"], color="black", linewidth=1.1, linestyle="--")
        axis.plot(time_axis, signals["recovered_message"], color=color, linewidth=1.5)
        axis.set_ylabel("Message")
        axis.set_title(method, pad=4)
        axis.grid(True, linestyle=":", alpha=0.4)
        axis.legend(["Original $m(t)$", r"Recovered $\hat{m}(t)$"], loc="upper right")

    axes[-1].set_xlabel("Time")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_recovery_error(
    time_axis: np.ndarray,
    signal_bank: dict[str, dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.6), constrained_layout=True)
    colors = {"PPO": "#d62728", "Linear": "#1f77b4"}
    for method in ["PPO", "Linear"]:
        ax.plot(
            time_axis,
            signal_bank[method]["recovery_error"],
            color=colors[method],
            linewidth=1.5,
            label=method,
        )
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Recovery error $\hat{m}(t)-m(t)$")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper right")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = resolve_output_dir(args.out)
    model_path = resolve_model_path(args)
    metadata = parse_model_metadata(model_path)
    pinning_node = args.node - 1

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable_baselines3 is required to load the PPO model for the communication demo."
        ) from exc

    print("Running chaotic masking communication demo under standard conditions:")
    print("  mismatch_scale = 0.0")
    print("  pulse_step = None")
    print("  process_noise_std = 0.0")
    print(f"  pinning node = {args.node}")
    print(f"  PPO model = {model_path.name}")

    model = PPO.load(str(model_path), device="cpu")
    ppo_policy = build_ppo_policy(model)
    linear_policy = build_linear_policy(args.linear_gain)

    env = make_env(pinning_node=pinning_node, max_steps=args.max_steps)
    step_time = env.dt * env.rk4_steps
    env.close()

    method_results: dict[str, dict[str, np.ndarray]] = {}
    for method, policy in [("PPO", ppo_policy), ("Linear", linear_policy)]:
        env = make_env(pinning_node=pinning_node, max_steps=args.max_steps)
        result = run_episode(policy, env, seed=args.seed)
        env.close()
        method_results[method] = result

    time_axis, message = build_message(
        num_steps=len(next(iter(method_results.values()))["drive"]),
        step_time=step_time,
        amplitude=args.message_amplitude,
        frequency=args.message_frequency,
        phase=args.message_phase,
    )

    signal_bank: dict[str, dict[str, np.ndarray]] = {}
    metric_rows: list[dict[str, float | int | str]] = []
    for method, result in method_results.items():
        signals = build_communication_signals(result, message)
        signal_bank[method] = signals
        metrics = compute_comm_metrics(signals)
        metric_rows.append(
            {
                "method": method,
                "model_name": metadata["model_name"] if method == "PPO" else "linear_feedback",
                "train_seed": metadata["train_seed"] if method == "PPO" else -1,
                "node": args.node,
                "test_seed": args.seed,
                "message_amplitude": args.message_amplitude,
                "message_frequency": args.message_frequency,
                **metrics,
            }
        )

    save_csv(metric_rows, out_dir / "communication_metrics.csv")
    save_signal_csv(time_axis, signal_bank, out_dir / "communication_signals.csv")
    plot_demo(time_axis, signal_bank, out_dir / "masked_message_recovery.png")
    plot_recovery_error(time_axis, signal_bank, out_dir / "message_recovery_error.png")


if __name__ == "__main__":
    main()
