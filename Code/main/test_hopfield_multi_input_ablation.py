"""Nominal PPO synchronization ablation for one, two, or three inputs.

Expected model-name examples:
  ppo_continuous_hopfield_node2_final_seed_42.zip
  ppo_continuous_hopfield_nodes2-3_final_seed_42.zip
  ppo_continuous_hopfield_nodes1-2-3_final_seed_42.zip

Only nominal synchronization is evaluated here.  Parameter mismatch, impulses,
noise, and partial observation deliberately remain in their dedicated scripts.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np


CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from env.continuous_hopfield_multi_input_env import ContinuousHopfieldMultiInputEnv

if TYPE_CHECKING:
    from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nominal 3-D Hopfield PPO synchronization with different input counts."
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--model-dir", type=str, default=str(PROJECT_ROOT / "models" / "hopfield")
    )
    parser.add_argument(
        "--pattern", type=str, default="ppo_continuous_hopfield_node[123]_final_seed_*.zip"
    )
    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        default=None,
        help="Override controlled nodes when they cannot be inferred from a model name.",
    )
    parser.add_argument("--test-seed-start", type=int, default=0)
    parser.add_argument("--test-seed-count", type=int, default=20)
    parser.add_argument("--out", type=str, default="logs/hopfield/multi_input_ablation")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[list[Path], Path]:
    if args.models:
        model_paths = [Path(path) for path in args.models]
    elif args.model:
        model_paths = [Path(args.model)]
    else:
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
        model_paths = sorted(model_dir.glob(args.pattern))

    resolved_models = []
    for path in model_paths:
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved_models.append(path.resolve())
    if not resolved_models:
        raise FileNotFoundError("No PPO model files matched the requested settings.")
    missing = [str(path) for path in resolved_models if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))

    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return resolved_models, output_dir.resolve()


def parse_model_metadata(model_path: Path) -> tuple[list[int] | None, int]:
    stem = model_path.stem
    multi_match = re.search(r"nodes([1-3](?:-[1-3])*)", stem)
    single_match = re.search(r"node([1-3])", stem)
    seed_match = re.search(r"seed_(\d+)", stem)
    if multi_match:
        nodes = [int(value) for value in multi_match.group(1).split("-")]
    elif single_match:
        nodes = [int(single_match.group(1))]
    else:
        nodes = None
    train_seed = int(seed_match.group(1)) if seed_match else -1
    return nodes, train_seed


def compute_metrics(
    error: np.ndarray, control: np.ndarray, step_time: float, action_limit: float
) -> dict[str, float]:
    l1_error = np.linalg.norm(error, ord=1, axis=1)
    steady_start = max(1, int(0.8 * len(l1_error)))
    active_control = control[1:]
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_l1": float(np.mean(l1_error)),
        "steady_l1": float(np.mean(l1_error[steady_start:])),
        "final_l1": float(l1_error[-1]),
        # Time mean of ||u||_1; this reduces to |u| in the single-input case.
        "mean_abs_u": float(np.mean(np.sum(np.abs(control), axis=1))),
        # Integral of ||u||_2^2 over time, summed across all actuators.
        "energy_u": float(np.sum(np.square(control)) * step_time),
        "saturation_fraction": float(
            np.mean(np.isclose(np.abs(active_control), action_limit, atol=1e-6))
        ),
    }


def run_episode(model: "PPO", nodes: list[int], seed: int) -> tuple[dict[str, float], np.ndarray]:
    env = ContinuousHopfieldMultiInputEnv(controlled_nodes=nodes)
    if tuple(model.observation_space.shape) != tuple(env.observation_space.shape):
        raise ValueError(
            f"Observation mismatch: model expects {model.observation_space.shape}, "
            f"but nominal full-observation environment provides {env.observation_space.shape}."
        )
    if tuple(model.action_space.shape) != tuple(env.action_space.shape):
        raise ValueError(
            f"Action mismatch for nodes {nodes}: model expects {model.action_space.shape}, "
            f"environment expects {env.action_space.shape}. Check the model name or --nodes."
        )

    obs, _ = env.reset(seed=seed)
    errors = [(env.statey - env.statex).copy()]
    controls = [np.zeros(env.num_control_inputs, dtype=np.float32)]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(env.action_space.shape)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        errors.append((env.statey - env.statex).copy())
        controls.append(np.clip(action, -1.0, 1.0) * env.scale)

    error_array = np.asarray(errors, dtype=np.float64)
    control_array = np.asarray(controls, dtype=np.float64)
    metrics = compute_metrics(
        error_array,
        control_array,
        env.dt * env.rk4_steps,
        env.scale,
    )
    l1_curve = np.linalg.norm(error_array, ord=1, axis=1)
    env.close()
    return metrics, l1_curve


def save_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def aggregate(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        groups[str(row["controlled_nodes"])].append(row)
    metric_names = [
        "mae",
        "rmse",
        "mean_l1",
        "steady_l1",
        "final_l1",
        "mean_abs_u",
        "energy_u",
        "saturation_fraction",
    ]
    summaries = []
    for controlled_nodes, group in sorted(groups.items()):
        item: dict[str, float | int | str] = {
            "controlled_nodes": controlled_nodes,
            "num_control_inputs": int(group[0]["num_control_inputs"]),
            "num_runs": len(group),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summaries.append(item)
    return summaries


def plot_curves(curves: dict[str, list[np.ndarray]], step_time: float, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for label, samples in sorted(curves.items()):
        matrix = np.stack(samples)
        median = np.median(matrix, axis=0)
        lower_quartile = np.quantile(matrix, 0.25, axis=0)
        upper_quartile = np.quantile(matrix, 0.75, axis=0)
        time = np.arange(len(median)) * step_time
        axis.plot(time, np.clip(median, 1e-8, None), linewidth=1.8, label=label)
        axis.fill_between(
            time,
            np.clip(lower_quartile, 1e-8, None),
            np.clip(upper_quartile, 1e-8, None),
            alpha=0.14,
        )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(r"$\|e(t)\|_1$")
    axis.set_yscale("log")
    axis.grid(True, which="both", linestyle=":", alpha=0.4)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_tradeoff(
    summaries: list[dict[str, float | int | str]], path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    annotation_offsets = {
        "1": (-68, -10),
        "1-2": (6, 5),
        "1-3": (6, 5),
        "2-3": (6, -16),
        "1-2-3": (-82, 5),
    }
    for row in summaries:
        num_inputs = int(row["num_control_inputs"])
        energy = float(row["energy_u_mean"])
        mae = float(row["mae_mean"])
        label = str(row["controlled_nodes"])
        axis.scatter(
            energy,
            mae,
            s=75 + 45 * num_inputs,
            color=colors[num_inputs],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        offset = annotation_offsets.get(label, (6, 5))
        axis.annotate(
            f"nodes {label}",
            (energy, mae),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )
    for num_inputs, color in colors.items():
        axis.scatter([], [], s=75 + 45 * num_inputs, color=color, edgecolor="black", label=f"{num_inputs} input(s)")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Control energy $E_u$")
    axis.set_ylabel("MAE")
    axis.grid(True, which="both", linestyle=":", alpha=0.4)
    axis.legend(frameon=False, loc="upper left")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    model_paths, output_dir = resolve_paths(args)
    test_seeds = list(range(args.test_seed_start, args.test_seed_start + args.test_seed_count))
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("stable_baselines3 is required to load PPO models.") from exc

    rows: list[dict[str, float | int | str]] = []
    curves: dict[str, list[np.ndarray]] = defaultdict(list)
    for model_path in model_paths:
        inferred_nodes, train_seed = parse_model_metadata(model_path)
        nodes = args.nodes or inferred_nodes
        if nodes is None:
            raise ValueError(f"Cannot infer controlled nodes from {model_path.name}; pass --nodes.")
        if len(set(nodes)) != len(nodes) or any(node not in (1, 2, 3) for node in nodes):
            raise ValueError(f"Invalid controlled-node list: {nodes}")

        model = PPO.load(str(model_path), device="cpu")
        node_text = "-".join(str(node) for node in nodes)
        label = f"{len(nodes)} input(s): node {node_text}"
        print(f"\nTesting {model_path.name} as {label}")
        for test_seed in test_seeds:
            metrics, curve = run_episode(model, nodes, test_seed)
            curves[label].append(curve)
            rows.append(
                {
                    "model_name": model_path.stem,
                    "train_seed": train_seed,
                    "node": nodes[0] if len(nodes) == 1 else -1,
                    "method": "PPO",
                    "test_seed": test_seed,
                    "mismatch": 0.0,
                    "pulse_step": -1,
                    **metrics,
                    "max_post_pulse": float("nan"),
                    "mean_post_pulse_error": float("nan"),
                    "recovery_steps": float("nan"),
                    "recovery_time": float("nan"),
                    "controlled_nodes": node_text,
                    "observation_mode": "full",
                    "observation_dim": 6,
                    "num_control_inputs": len(nodes),
                    "action_limit": 4.0,
                    "noise_std": 0.0,
                }
            )

    summaries = aggregate(rows)
    save_csv(rows, output_dir / "raw_metrics.csv")
    save_csv(summaries, output_dir / "metrics_summary.csv")
    if not args.no_plot:
        plot_curves(curves, 0.05, output_dir / "multi_input_sync_comparison.png")
        plot_tradeoff(summaries, output_dir / "multi_input_energy_mae_tradeoff.png")


if __name__ == "__main__":
    main()
