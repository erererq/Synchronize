"""Compare screened SAC policies with the existing PPO policies."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC


CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from env.continuous_hopfield_env import ContinuousHopfieldEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen PPO against SAC on Hopfield Nodes 2 and 3.")
    parser.add_argument("--sac-timesteps", type=int, default=100_000)
    parser.add_argument("--test-seed-start", type=int, default=100)
    parser.add_argument("--test-seed-count", type=int, default=20)
    parser.add_argument("--mismatch-level", type=float, default=2.0)
    parser.add_argument("--pulse-step", type=int, default=200)
    parser.add_argument("--pulse-vector", type=float, nargs=3, default=[5.0, 5.0, 5.0])
    parser.add_argument("--out", type=str, default="logs/hopfield/ppo_sac_screen")
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def models_by_node(sac_timesteps: int) -> dict[int, list[tuple[str, Path, object]]]:
    model_dir = PROJECT_ROOT / "models" / "hopfield"
    result: dict[int, list[tuple[str, Path, object]]] = {}
    for node in (2, 3):
        ppo_paths = sorted(model_dir.glob(f"ppo_continuous_hopfield_node{node}_final_seed_*.zip"))
        sac_paths = sorted(model_dir.glob(f"sac_continuous_hopfield_node{node}_screen_{sac_timesteps}steps_seed_*.zip"))
        if not ppo_paths or not sac_paths:
            raise FileNotFoundError(f"Missing PPO or SAC model for Node {node}")
        result[node] = (
            [("PPO", path, PPO.load(str(path), device="cpu")) for path in ppo_paths]
            + [("SAC", path, SAC.load(str(path), device="cpu")) for path in sac_paths]
        )
    return result


def run_episode(model: object, node: int, seed: int, mismatch: float, pulse_step: int | None, pulse_vector: tuple[float, float, float]) -> dict[str, float]:
    env = ContinuousHopfieldEnv(
        pinning_node=node - 1,
        mismatch_scale=mismatch,
        pulse_step=pulse_step,
        pulse_width=1,
        pulse_vector=pulse_vector,
    )
    observation, _ = env.reset(seed=seed)
    errors = [(env.statey - env.statex).copy()]
    controls = [0.0]
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(1)
        observation, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        errors.append((env.statey - env.statex).copy())
        controls.append(float(action[0]) * env.scale)
    error = np.asarray(errors, dtype=np.float64)
    control = np.asarray(controls, dtype=np.float64)
    l1 = np.linalg.norm(error, ord=1, axis=1)
    steady_start = max(1, int(0.8 * len(l1)))
    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "steady_l1": float(np.mean(l1[steady_start:])),
        "final_l1": float(l1[-1]),
        "energy_u": float(np.sum(np.square(control)) * env.dt * env.rk4_steps),
        "saturation_fraction": float(np.mean(np.isclose(np.abs(control[1:]), env.scale, atol=1e-6))),
    }
    if pulse_step is not None:
        metrics["mean_post_pulse_error"] = float(np.mean(l1[pulse_step:]))
    env.close()
    return metrics


def save_csv(rows: list[dict], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["node"], row["method"])].append(row)
    metrics = ["mae", "steady_l1", "final_l1", "energy_u", "saturation_fraction", "mean_post_pulse_error"]
    output: list[dict] = []
    for (scenario, node, method), group in sorted(groups.items()):
        item = {"scenario": scenario, "node": node, "method": method, "num_runs": len(group), "num_train_seeds": len(set(row["train_seed"] for row in group))}
        for metric in metrics:
            values = np.asarray([row[metric] for row in group if metric in row], dtype=np.float64)
            if len(values):
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def plot_summary(summary: list[dict], path: Path) -> None:
    methods = ("PPO", "SAC")
    colors = {"PPO": "#d62728", "SAC": "#9467bd"}
    scenarios = (("nominal", "mae_mean", "Nominal MAE"), ("mismatch", "mae_mean", "Mismatch MAE"), ("pulse", "mean_post_pulse_error_mean", "Post-pulse error"))
    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.0), constrained_layout=True)
    for row_index, node in enumerate((2, 3)):
        for column, (scenario, metric, title) in enumerate(scenarios):
            selected = [next(row for row in summary if row["scenario"] == scenario and int(row["node"]) == node and row["method"] == method) for method in methods]
            axes[row_index, column].bar(methods, [float(row[metric]) for row in selected], color=[colors[method] for method in methods])
            axes[row_index, column].set_title(f"Node {node}: {title}")
            axes[row_index, column].grid(True, axis="y", linestyle=":", alpha=0.4)
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models = models_by_node(args.sac_timesteps)
    seeds = range(args.test_seed_start, args.test_seed_start + args.test_seed_count)
    pulse_vector = tuple(float(value) for value in args.pulse_vector)
    conditions = (("nominal", 0.0, None), ("mismatch", args.mismatch_level, None), ("pulse", 0.0, args.pulse_step))
    rows: list[dict] = []
    for scenario, mismatch, pulse_step in conditions:
        print(f"Testing {scenario}")
        for node in (2, 3):
            for method, path, model in models[node]:
                for seed in seeds:
                    metrics = run_episode(model, node, seed, mismatch, pulse_step, pulse_vector)
                    rows.append({"scenario": scenario, "node": node, "method": method, "model_name": path.stem, "train_seed": seed_from_path(path), "test_seed": seed, "mismatch": mismatch, "pulse_step": pulse_step if pulse_step is not None else -1, **metrics})
    summary = summarize(rows)
    save_csv(rows, output_dir / "raw_metrics.csv")
    save_csv(summary, output_dir / "metrics_summary.csv")
    plot_summary(summary, output_dir / "ppo_sac_screen.png")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
