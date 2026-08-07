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

from env.continuous_hopfield_5d_hohnn_env import ContinuousHopfield5DHOHNNEnv

if TYPE_CHECKING:
    from stable_baselines3 import PPO


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test PPO models on the 5D HOHNN synchronization task.")
    parser.add_argument("--model", type=str, default=None, help="Single model path.")
    parser.add_argument("--models", nargs="+", default=None, help="Multiple model paths.")
    parser.add_argument("--model-dir", type=str, default=str(PROJECT_ROOT / "models" / "5d_hohnn"))
    parser.add_argument(
        "--pattern",
        type=str,
        default="ppo_5d_hohnn_node*",
        help="Model filename pattern; both .zip and extensionless SB3 model files are supported.",
    )
    parser.add_argument(
        "--node",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5],
        help="Override the controlled node when it cannot be inferred from the model name.",
    )
    parser.add_argument("--test-seed-start", type=int, default=100)
    parser.add_argument("--test-seed-count", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--control-scale", type=float, default=20.0)
    parser.add_argument("--w11", type=float, default=1.0)
    parser.add_argument("--w22", type=float, default=3.26)
    parser.add_argument("--w44", type=float, default=170.0)
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "logs" / "5d" / "sync_test"))
    parser.add_argument("--include-zero", action="store_true", help="Also test the uncontrolled baseline.")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def resolve_output_dir(out: str) -> Path:
    out_dir = Path(out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir.resolve()


def resolve_model_paths(args: argparse.Namespace) -> list[Path]:
    if args.models:
        paths = [Path(path) for path in args.models]
    elif args.model:
        paths = [Path(args.model)]
    else:
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
        paths = sorted(path for path in model_dir.glob(args.pattern) if path.is_file())

        # Older training runs saved valid ZIP archives without the .zip suffix.
        if not paths and args.pattern.endswith(".zip"):
            fallback_pattern = args.pattern.removesuffix(".zip")
            paths = sorted(path for path in model_dir.glob(fallback_pattern) if path.is_file())

    resolved = []
    for path in paths:
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved.append(path.resolve())

    if not resolved:
        raise FileNotFoundError("No model files matched the given settings.")

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError("Model file(s) not found:\n" + "\n".join(missing))

    return resolved


def parse_node_from_model_name(model_path: Path) -> int | None:
    match = re.search(r"(?:^|_)node(\d+)(?:_|$)", model_path.stem)
    if match:
        return int(match.group(1))
    return None


def make_env(args: argparse.Namespace, node: int) -> ContinuousHopfield5DHOHNNEnv:
    env = ContinuousHopfield5DHOHNNEnv(
        w11=args.w11,
        w22=args.w22,
        w44=args.w44,
        pinning_node=node - 1,
        control_scale=args.control_scale,
    )
    env.max_steps = int(args.max_steps)
    return env


def run_episode(policy_fn, env: ContinuousHopfield5DHOHNNEnv, seed: int) -> dict[str, np.ndarray]:
    obs, _ = env.reset(seed=seed)
    error_history = [(env.statey - env.statex).copy()]
    action_history = [np.zeros(env.action_space.shape, dtype=np.float32)]

    done = False
    while not done:
        action = np.asarray(policy_fn(obs, env), dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        error_history.append((env.statey - env.statex).copy())
        action_history.append(action * env.scale)

    return {
        "error": np.asarray(error_history, dtype=np.float32),
        "action": np.asarray(action_history, dtype=np.float32),
    }


def compute_metrics(result: dict[str, np.ndarray], step_time: float) -> dict[str, float]:
    error = result["error"]
    action = result["action"]
    l1_error = np.linalg.norm(error, ord=1, axis=1)

    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_l1": float(np.mean(l1_error)),
        "final_l1": float(l1_error[-1]),
        "mean_abs_u": float(np.mean(np.abs(action))),
        "energy_u": float(np.sum(np.square(action)) * step_time),
    }


def save_csv(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with out_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_path}")


def aggregate_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        groups[str(row["model_label"])].append(row)

    metrics = ["mae", "rmse", "mean_l1", "final_l1", "mean_abs_u", "energy_u"]
    summary: list[dict[str, float | int | str]] = []
    for label, group in groups.items():
        item: dict[str, float | int | str] = {
            "model_label": label,
            "node": group[0]["node"],
            "n_runs": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values))
        summary.append(item)
    return summary


def plot_model_comparison(
    curves_by_label: dict[str, list[np.ndarray]],
    step_time: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(curves_by_label), 1)))

    for color, (label, curves) in zip(colors, curves_by_label.items()):
        stacked = np.stack(curves, axis=0)
        mean_curve = np.mean(stacked, axis=0)
        std_curve = np.std(stacked, axis=0)
        time = np.arange(mean_curve.size) * step_time

        ax.plot(time, mean_curve, color=color, linewidth=1.8, label=label)
        if len(curves) > 1:
            lower = np.clip(mean_curve - std_curve, 0.0, None)
            ax.fill_between(time, lower, mean_curve + std_curve, color=color, alpha=0.14, linewidth=0)

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$||e(t)||_1$")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {out_path}")


def evaluate_model(
    args: argparse.Namespace,
    model_path: Path,
    test_seeds: list[int],
    ppo_class: type["PPO"],
) -> tuple[list[dict[str, float | int | str]], str, list[np.ndarray], float]:
    node = args.node or parse_node_from_model_name(model_path)
    if node is None:
        raise ValueError(
            f"Could not infer the controlled node from {model_path.name}. "
            "Pass it explicitly, for example: --node 2"
        )

    model = ppo_class.load(str(model_path), device="cpu")
    label = f"node {node}"

    def policy(obs: np.ndarray, _env: ContinuousHopfield5DHOHNNEnv) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    rows: list[dict[str, float | int | str]] = []
    curves: list[np.ndarray] = []
    step_time: float | None = None

    print(f"\n=== Testing {model_path.name} ({label}) ===")
    for seed in test_seeds:
        env = make_env(args, node)
        if step_time is None:
            step_time = env.dt * env.rk4_steps

        result = run_episode(policy, env, seed)
        env.close()
        metrics = compute_metrics(result, step_time)
        curves.append(np.linalg.norm(result["error"], ord=1, axis=1))
        rows.append(
            {
                "model_name": model_path.stem,
                "model_label": label,
                "node": node,
                "test_seed": seed,
                **metrics,
            }
        )
        print(f"seed={seed} | final_l1={metrics['final_l1']:.6f} | mean_l1={metrics['mean_l1']:.6f}")

    return rows, label, curves, float(step_time)


def evaluate_zero_baseline(
    args: argparse.Namespace,
    node: int,
    test_seeds: list[int],
) -> tuple[list[dict[str, float | int | str]], str, list[np.ndarray], float]:
    label = "zero input"

    def zero_policy(_obs: np.ndarray, env: ContinuousHopfield5DHOHNNEnv) -> np.ndarray:
        return np.zeros(env.action_space.shape, dtype=np.float32)

    rows: list[dict[str, float | int | str]] = []
    curves: list[np.ndarray] = []
    step_time: float | None = None

    print(f"\n=== Testing {label} ===")
    for seed in test_seeds:
        env = make_env(args, node)
        if step_time is None:
            step_time = env.dt * env.rk4_steps

        result = run_episode(zero_policy, env, seed)
        env.close()
        metrics = compute_metrics(result, step_time)
        curves.append(np.linalg.norm(result["error"], ord=1, axis=1))
        rows.append(
            {
                "model_name": "zero_input",
                "model_label": label,
                "node": "none",
                "test_seed": seed,
                **metrics,
            }
        )
        print(f"seed={seed} | final_l1={metrics['final_l1']:.6f} | mean_l1={metrics['mean_l1']:.6f}")

    return rows, label, curves, float(step_time)


def main() -> None:
    args = parse_args()
    model_paths = resolve_model_paths(args)
    out_dir = resolve_output_dir(args.out)
    test_seeds = list(range(args.test_seed_start, args.test_seed_start + args.test_seed_count))

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("stable_baselines3 is required to load PPO models.") from exc

    all_rows: list[dict[str, float | int | str]] = []
    curves_by_label: dict[str, list[np.ndarray]] = {}
    step_time: float | None = None

    for model_path in model_paths:
        rows, label, curves, model_step_time = evaluate_model(args, model_path, test_seeds, PPO)
        all_rows.extend(rows)
        curves_by_label.setdefault(label, []).extend(curves)
        if step_time is None:
            step_time = model_step_time

    if args.include_zero:
        baseline_node = args.node or parse_node_from_model_name(model_paths[0]) or 1
        rows, label, curves, baseline_step_time = evaluate_zero_baseline(args, baseline_node, test_seeds)
        all_rows.extend(rows)
        curves_by_label[label] = curves
        if step_time is None:
            step_time = baseline_step_time

    save_csv(all_rows, out_dir / "raw_sync_metrics.csv")
    save_csv(aggregate_rows(all_rows), out_dir / "sync_metrics_summary.csv")

    if not args.no_plot and step_time is not None:
        plot_model_comparison(curves_by_label, step_time, out_dir / "model_sync_comparison.png")


if __name__ == "__main__":
    main()
