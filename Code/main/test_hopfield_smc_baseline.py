"""Tune and evaluate a saturated boundary-layer SMC baseline.

The controller uses the nominal Hopfield model and the complete synchronization
error.  Its scalar sliding surface is derived from the local LQR gain, but its
nonlinear equivalent-control term is evaluated on the original nonlinear
dynamics.  SMC parameters are selected on validation seeds and then frozen for
the held-out test seeds.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are
from stable_baselines3 import PPO


CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from env.continuous_hopfield_env import ContinuousHopfieldEnv


W_NOMINAL = np.array(
    [[-1.4, 1.2, -7.0], [1.1, 0.0, 2.8], [0.8, -2.0, 4.0]],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a boundary-layer SMC baseline.")
    parser.add_argument("--validation-seed-count", type=int, default=10)
    parser.add_argument("--test-seed-start", type=int, default=100)
    parser.add_argument("--test-seed-count", type=int, default=20)
    parser.add_argument("--reaching-gains", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 32.0])
    parser.add_argument("--boundary-widths", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--mismatch-levels", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    parser.add_argument("--pulse-step", type=int, default=200)
    parser.add_argument("--pulse-vector", type=float, nargs=3, default=[5.0, 5.0, 5.0])
    parser.add_argument("--out", type=str, default="logs/hopfield/smc_baseline")
    return parser.parse_args()


def lqr_gain(node: int) -> np.ndarray:
    a = W_NOMINAL - np.eye(3)
    b = np.zeros((3, 1), dtype=np.float64)
    b[node - 1, 0] = 1.0
    p = solve_continuous_are(a, b, np.eye(3), np.ones((1, 1)))
    return (b.T @ p).reshape(3)


def sliding_surface(node: int) -> np.ndarray:
    """Return c with c^T B = 1 for s=c^T e."""
    gain = lqr_gain(node)
    return gain / gain[node - 1]


def smc_force(env: ContinuousHopfieldEnv, surface: np.ndarray, reaching_gain: float, boundary_width: float) -> float:
    error = (env.statey - env.statex).astype(np.float64)
    nominal_error_drift = -error + W_NOMINAL @ (
        np.tanh(env.statey.astype(np.float64)) - np.tanh(env.statex.astype(np.float64))
    )
    sliding_variable = float(surface @ error)
    boundary_action = float(np.clip(sliding_variable / boundary_width, -1.0, 1.0))
    # c^T B = 1 by construction.  The first term is nonlinear nominal-model
    # compensation and the second term imposes s_dot=-gain*sat(s/width).
    return -float(surface @ nominal_error_drift) - reaching_gain * boundary_action


def model_paths(node: int) -> list[Path]:
    paths = sorted((PROJECT_ROOT / "models" / "hopfield").glob(f"ppo_continuous_hopfield_node{node}_final_seed_*.zip"))
    if not paths:
        raise FileNotFoundError(f"No Node {node} PPO models found")
    return paths


def train_seed(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def run_episode(
    method: str,
    node: int,
    seed: int,
    mismatch: float = 0.0,
    pulse_step: int | None = None,
    pulse_vector: tuple[float, float, float] = (0.0, 0.0, 0.0),
    reaching_gain: float = 1.0,
    boundary_width: float = 0.1,
    linear_gain: float = 8.0,
    model: PPO | None = None,
    noise_std: float = 0.0,
) -> dict[str, float]:
    env = ContinuousHopfieldEnv(
        pinning_node=node - 1,
        mismatch_scale=mismatch,
        pulse_step=pulse_step,
        pulse_width=1,
        pulse_vector=pulse_vector,
        process_noise_std=noise_std,
    )
    observation, _ = env.reset(seed=seed)
    surface = sliding_surface(node)
    gain = lqr_gain(node)
    errors = [(env.statey - env.statex).copy()]
    controls = [0.0]
    done = False
    while not done:
        if method == "PPO":
            if model is None:
                raise ValueError("PPO model is required")
            action, _ = model.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(1)
        elif method == "LQR":
            error = (env.statey - env.statex).astype(np.float64)
            force = -float(gain @ error)
            action = np.array([np.clip(force / env.scale, -1.0, 1.0)], dtype=np.float32)
        elif method == "SMC":
            force = smc_force(env, surface, reaching_gain, boundary_width)
            action = np.array([np.clip(force / env.scale, -1.0, 1.0)], dtype=np.float32)
        elif method == "Linear":
            error = (env.statey - env.statex).astype(np.float64)
            force = -linear_gain * float(error[node - 1])
            action = np.array([np.clip(force / env.scale, -1.0, 1.0)], dtype=np.float32)
        else:
            raise ValueError(f"Unknown method: {method}")

        observation, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        errors.append((env.statey - env.statex).copy())
        controls.append(float(action[0]) * env.scale)

    error = np.asarray(errors, dtype=np.float64)
    control = np.asarray(controls, dtype=np.float64)
    l1_error = np.linalg.norm(error, ord=1, axis=1)
    steady_start = max(1, int(0.8 * len(l1_error)))
    step_time = env.dt * env.rk4_steps
    result = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_l1": float(np.mean(l1_error)),
        "steady_l1": float(np.mean(l1_error[steady_start:])),
        "final_l1": float(l1_error[-1]),
        "mean_abs_u": float(np.mean(np.abs(control))),
        "energy_u": float(np.sum(np.square(control)) * step_time),
        "saturation_fraction": float(np.mean(np.isclose(np.abs(control[1:]), env.scale, atol=1e-6))),
    }
    if pulse_step is not None:
        post = l1_error[pulse_step:]
        result["max_post_pulse"] = float(np.max(post))
        result["mean_post_pulse_error"] = float(np.mean(post))
    env.close()
    return result


def validation_score(node: int, seeds: list[int], reaching_gain: float, boundary_width: float, pulse_step: int, pulse_vector: tuple[float, float, float]) -> tuple[float, dict[str, float]]:
    # The three equally weighted validation conditions prevent tuning solely to
    # the nominal model or solely to the large impulse experiment.
    conditions = [
        ("nominal", 0.0, None),
        ("mismatch", 1.0, None),
        ("pulse", 0.0, pulse_step),
    ]
    condition_mae: dict[str, float] = {}
    for name, mismatch, pulse in conditions:
        values = [
            run_episode("SMC", node, seed, mismatch, pulse, pulse_vector, reaching_gain, boundary_width)["mae"]
            for seed in seeds
        ]
        condition_mae[name] = float(np.mean(values))
    # Normalize each condition by a fixed scale so the impulse case does not
    # dominate parameter selection merely because its numerical error is larger.
    scales = {"nominal": 0.1, "mismatch": 0.15, "pulse": 0.45}
    score = float(np.mean([condition_mae[name] / scales[name] for name in condition_mae]))
    return score, condition_mae


def save_csv(rows: list[dict], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["node"], row["method"], row["mismatch"])].append(row)
    metrics = ["mae", "steady_l1", "final_l1", "energy_u", "saturation_fraction", "mean_post_pulse_error"]
    output: list[dict] = []
    for key, group in sorted(groups.items()):
        scenario, node, method, mismatch = key
        item = {"scenario": scenario, "node": node, "method": method, "mismatch": mismatch, "num_runs": len(group)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in group if metric in row], dtype=np.float64)
            if len(values):
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def plot_results(summary: list[dict], output_path: Path) -> None:
    methods = ["PPO", "LQR", "SMC"]
    colors = {"PPO": "#d62728", "LQR": "#1f77b4", "SMC": "#2ca02c"}
    scenarios = [("mismatch", 2.0, "Mismatch $\\eta=2$"), ("pulse", 0.0, "Global impulse")]
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), constrained_layout=True)
    x = np.arange(len(methods))
    for column, node in enumerate((2, 3)):
        for scenario, mismatch, label in scenarios:
            selected = [
                next(row for row in summary if row["scenario"] == scenario and int(row["node"]) == node and row["method"] == method and float(row["mismatch"]) == mismatch)
                for method in methods
            ]
            row_index = 0 if scenario == "mismatch" else 1
            metric = "mae_mean" if scenario == "mismatch" else "mean_post_pulse_error_mean"
            axes[row_index, column].bar(x, [float(row[metric]) for row in selected], color=[colors[m] for m in methods])
            axes[row_index, column].set_xticks(x, methods)
            axes[row_index, column].set_ylabel("MAE" if scenario == "mismatch" else "Mean post-pulse error")
            axes[row_index, column].set_title(f"Node {node}: {label}")
            axes[row_index, column].grid(True, axis="y", linestyle=":", alpha=0.4)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pulse_vector = tuple(float(value) for value in args.pulse_vector)

    validation_rows: list[dict] = []
    selected: dict[int, tuple[float, float]] = {}
    validation_seeds = list(range(args.validation_seed_count))
    for node in (2, 3):
        best: tuple[float, float, float] | None = None
        for reaching_gain in args.reaching_gains:
            for boundary_width in args.boundary_widths:
                score, details = validation_score(node, validation_seeds, reaching_gain, boundary_width, args.pulse_step, pulse_vector)
                row = {"node": node, "reaching_gain": reaching_gain, "boundary_width": boundary_width, "score": score, **{f"{key}_mae": value for key, value in details.items()}}
                validation_rows.append(row)
                candidate = (score, reaching_gain, boundary_width)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        selected[node] = (best[1], best[2])
        print(f"Node {node}: selected reaching_gain={best[1]:g}, boundary_width={best[2]:g}, score={best[0]:.6f}")
    save_csv(validation_rows, output_dir / "validation_grid.csv")

    ppo_models = {node: [(path, PPO.load(str(path), device="cpu")) for path in model_paths(node)] for node in (2, 3)}
    test_seeds = list(range(args.test_seed_start, args.test_seed_start + args.test_seed_count))
    rows: list[dict] = []
    conditions = [("mismatch", level, None) for level in args.mismatch_levels]
    conditions.append(("pulse", 0.0, args.pulse_step))
    for scenario, mismatch, pulse_step in conditions:
        print(f"Testing {scenario}: mismatch={mismatch}, pulse={pulse_step}")
        for node in (2, 3):
            reaching_gain, boundary_width = selected[node]
            for seed in test_seeds:
                for method in ("LQR", "SMC"):
                    metrics = run_episode(method, node, seed, mismatch, pulse_step, pulse_vector, reaching_gain, boundary_width)
                    rows.append({
                        "model_name": f"{method.lower()}_node{node}", "train_seed": -1, "node": node,
                        "method": method, "test_seed": seed, "scenario": scenario, "mismatch": mismatch,
                        "pulse_step": pulse_step if pulse_step is not None else -1,
                        "reaching_gain": reaching_gain if method == "SMC" else "",
                        "boundary_width": boundary_width if method == "SMC" else "", **metrics,
                    })
                for path, model in ppo_models[node]:
                    metrics = run_episode("PPO", node, seed, mismatch, pulse_step, pulse_vector, model=model)
                    rows.append({
                        "model_name": path.stem, "train_seed": train_seed(path), "node": node,
                        "method": "PPO", "test_seed": seed, "scenario": scenario, "mismatch": mismatch,
                        "pulse_step": pulse_step if pulse_step is not None else -1,
                        "reaching_gain": "", "boundary_width": "", **metrics,
                    })

    summary = summarize(rows)
    save_csv(rows, output_dir / "raw_metrics.csv")
    save_csv(summary, output_dir / "metrics_summary.csv")
    plot_results(summary, output_dir / "ppo_lqr_smc_robustness.png")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
