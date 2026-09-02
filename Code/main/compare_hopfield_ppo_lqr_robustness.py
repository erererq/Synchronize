"""Fair robustness comparison of full-observation PPO and saturated LQR."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from stable_baselines3 import PPO


CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from env.continuous_hopfield_env import ContinuousHopfieldEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PPO and LQR robustness on Nodes 2 and 3.")
    parser.add_argument("--test-seed-start", type=int, default=100)
    parser.add_argument("--test-seed-count", type=int, default=20)
    parser.add_argument("--mismatch-levels", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05])
    parser.add_argument("--pulse-step", type=int, default=200)
    parser.add_argument("--pulse-vector", type=float, nargs=3, default=[5.0, 5.0, 5.0])
    parser.add_argument("--out", type=str, default="logs/hopfield/ppo_lqr_robustness")
    return parser.parse_args()


def model_paths_by_node() -> dict[int, list[Path]]:
    model_dir = PROJECT_ROOT / "models" / "hopfield"
    result = {
        node: sorted(model_dir.glob(f"ppo_continuous_hopfield_node{node}_final_seed_*.zip"))
        for node in (2, 3)
    }
    for node, paths in result.items():
        if not paths:
            raise FileNotFoundError(f"No full-observation Node {node} PPO models found in {model_dir}")
    return result


def train_seed(model_path: Path) -> int:
    match = re.search(r"seed_(\d+)", model_path.stem)
    return int(match.group(1)) if match else -1


def lqr_gain(node: int) -> np.ndarray:
    w = np.array([[-1.4, 1.2, -7.0], [1.1, 0.0, 2.8], [0.8, -2.0, 4.0]])
    a = w - np.eye(3)
    b = np.zeros((3, 1))
    b[node - 1, 0] = 1.0
    p = solve_continuous_are(a, b, np.eye(3), np.ones((1, 1)))
    return (b.T @ p).reshape(3)


def run_episode(
    method: str,
    node: int,
    seed: int,
    mismatch: float,
    noise_std: float,
    pulse_step: int | None,
    pulse_vector: tuple[float, float, float],
    gain: np.ndarray,
    model: PPO | None,
) -> dict[str, float]:
    env = ContinuousHopfieldEnv(
        pinning_node=node - 1,
        mismatch_scale=mismatch,
        process_noise_std=noise_std,
        pulse_step=pulse_step,
        pulse_width=1,
        pulse_vector=pulse_vector,
    )
    obs, _ = env.reset(seed=seed)
    errors = [(env.statey - env.statex).copy()]
    controls = [0.0]
    done = False
    while not done:
        if method == "PPO":
            assert model is not None
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(1)
        else:
            error = (env.statey - env.statex).astype(np.float64)
            force = -float(gain @ error)
            action = np.array([np.clip(force / env.scale, -1.0, 1.0)], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        errors.append((env.statey - env.statex).copy())
        controls.append(float(action[0]) * env.scale)

    error = np.asarray(errors, dtype=np.float64)
    control = np.asarray(controls, dtype=np.float64)
    l1 = np.linalg.norm(error, ord=1, axis=1)
    steady_start = max(1, int(0.8 * len(l1)))
    step_time = env.dt * env.rk4_steps
    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_l1": float(np.mean(l1)),
        "steady_l1": float(np.mean(l1[steady_start:])),
        "final_l1": float(l1[-1]),
        "mean_abs_u": float(np.mean(np.abs(control))),
        "energy_u": float(np.sum(np.square(control)) * step_time),
        "saturation_fraction": float(np.mean(np.isclose(np.abs(control[1:]), env.scale, atol=1e-6))),
    }
    if pulse_step is not None:
        post = l1[pulse_step:]
        metrics["max_post_pulse"] = float(np.max(post))
        metrics["mean_post_pulse_error"] = float(np.mean(post))
    env.close()
    return metrics


def save_csv(rows: list[dict], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["scenario"], row["node"], row["method"], row["mismatch"], row["noise_std"])
        groups[key].append(row)
    metric_names = [
        "mae", "rmse", "mean_l1", "steady_l1", "final_l1",
        "mean_abs_u", "energy_u", "saturation_fraction",
        "max_post_pulse", "mean_post_pulse_error",
    ]
    output = []
    for key, group in sorted(groups.items()):
        scenario, node, method, mismatch, noise_std = key
        item = {
            "scenario": scenario,
            "node": node,
            "method": method,
            "mismatch": mismatch,
            "noise_std": noise_std,
            "num_runs": len(group),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group if metric in row], dtype=float)
            if not len(values):
                continue
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def plot_trend(summary: list[dict], scenario: str, x_key: str, x_label: str, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex="col", constrained_layout=True)
    colors = {"PPO": "#d62728", "LQR": "#1f77b4"}
    markers = {"PPO": "o", "LQR": "s"}
    for column, node in enumerate((2, 3)):
        for method in ("PPO", "LQR"):
            selected = sorted(
                [row for row in summary if row["scenario"] == scenario and int(row["node"]) == node and row["method"] == method],
                key=lambda row: float(row[x_key]),
            )
            x = [float(row[x_key]) for row in selected]
            axes[0, column].plot(x, [float(row["mae_mean"]) for row in selected], marker=markers[method], color=colors[method], label=method)
            axes[1, column].plot(x, [float(row["energy_u_mean"]) for row in selected], marker=markers[method], color=colors[method], label=method)
        axes[0, column].set_title(f"Node {node}")
        axes[0, column].set_ylabel("MAE")
        axes[1, column].set_ylabel(r"Control energy $E_u$")
        axes[1, column].set_xlabel(x_label)
        for row_axes in axes[:, column]:
            row_axes.grid(True, linestyle=":", alpha=0.4)
            row_axes.legend(frameon=False)
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    print(f"Saved: {path}")


def plot_pulse(summary: list[dict], path: Path) -> None:
    pulse_rows = [row for row in summary if row["scenario"] == "pulse"]
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), constrained_layout=True)
    x = np.arange(2)
    width = 0.34
    for offset, method, color in ((-width / 2, "PPO", "#d62728"), (width / 2, "LQR", "#1f77b4")):
        selected = [next(row for row in pulse_rows if int(row["node"]) == node and row["method"] == method) for node in (2, 3)]
        axes[0].bar(x + offset, [float(row["mean_post_pulse_error_mean"]) for row in selected], width, label=method, color=color)
        axes[1].bar(x + offset, [float(row["energy_u_mean"]) for row in selected], width, label=method, color=color)
    axes[0].set_ylabel("Mean post-pulse error")
    axes[1].set_ylabel(r"Control energy $E_u$")
    for axis in axes:
        axis.set_xticks(x, ["Node 2", "Node 3"])
        axis.grid(True, axis="y", linestyle=":", alpha=0.4)
        axis.legend(frameon=False)
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    print(f"Saved: {path}")


def append_condition(
    rows: list[dict],
    models: dict[int, list[tuple[Path, PPO]]],
    seeds: list[int],
    scenario: str,
    mismatch: float = 0.0,
    noise_std: float = 0.0,
    pulse_step: int | None = None,
    pulse_vector: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    print(f"Testing {scenario}: mismatch={mismatch}, noise={noise_std}, pulse={pulse_step}")
    for node in (2, 3):
        gain = lqr_gain(node)
        for seed in seeds:
            lqr_metrics = run_episode(
                "LQR", node, seed, mismatch, noise_std, pulse_step, pulse_vector, gain, None
            )
            rows.append({
                "model_name": f"lqr_node{node}_q1_r1", "train_seed": -1,
                "node": node, "method": "LQR", "test_seed": seed,
                "scenario": scenario, "mismatch": mismatch, "noise_std": noise_std,
                "pulse_step": pulse_step if pulse_step is not None else -1, **lqr_metrics,
            })
            for path, model in models[node]:
                ppo_metrics = run_episode(
                    "PPO", node, seed, mismatch, noise_std, pulse_step, pulse_vector, gain, model
                )
                rows.append({
                    "model_name": path.stem, "train_seed": train_seed(path),
                    "node": node, "method": "PPO", "test_seed": seed,
                    "scenario": scenario, "mismatch": mismatch, "noise_std": noise_std,
                    "pulse_step": pulse_step if pulse_step is not None else -1, **ppo_metrics,
                })


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.test_seed_start, args.test_seed_start + args.test_seed_count))
    models = {
        node: [(path, PPO.load(str(path), device="cpu")) for path in paths]
        for node, paths in model_paths_by_node().items()
    }
    rows: list[dict] = []
    for level in args.mismatch_levels:
        append_condition(rows, models, seeds, "mismatch", mismatch=level)
    append_condition(
        rows, models, seeds, "pulse", pulse_step=args.pulse_step,
        pulse_vector=tuple(args.pulse_vector),
    )
    for level in args.noise_levels:
        append_condition(rows, models, seeds, "noise", noise_std=level)
    summary = summarize(rows)
    save_csv(rows, output_dir / "raw_metrics.csv")
    save_csv(summary, output_dir / "metrics_summary.csv")
    plot_trend(summary, "mismatch", "mismatch", r"Mismatch scale $\eta$", output_dir / "mismatch_mae_energy.png")
    plot_trend(summary, "noise", "noise_std", r"Noise standard deviation $\sigma$", output_dir / "noise_mae_energy.png")
    plot_pulse(summary, output_dir / "pulse_error_energy.png")


if __name__ == "__main__":
    main()
