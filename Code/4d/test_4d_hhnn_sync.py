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

from env.continuous_hopfield_4d_hhnn_env import ContinuousHopfield4DHHNNEnv

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
    parser = argparse.ArgumentParser(description="Test trained PPO models on the 4D HHNN synchronization task.")
    parser.add_argument("--model", type=str, default=None, help="Single model path.")
    parser.add_argument("--models", nargs="+", default=None, help="Multiple model paths.")
    parser.add_argument("--model-dir", type=str, default=str(PROJECT_ROOT / "models" / "4d_hhnn"))
    parser.add_argument("--pattern", type=str, default="ppo_4d_hhnn*.zip")
    parser.add_argument("--nodes", type=int, nargs="+", default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--test-seed-start", type=int, default=100)
    parser.add_argument("--test-seed-count", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--control-scale", type=float, default=20.0)
    parser.add_argument("--w22", type=float, default=2.0)
    parser.add_argument("--w33", type=float, default=1.0)
    parser.add_argument("--w44", type=float, default=170.0)
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "logs" / "4d" / "sync_test"))
    parser.add_argument("--include-zero", action="store_true", help="Also plot the uncontrolled zero-input baseline.")
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
        paths = sorted(model_dir.glob(args.pattern))

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


def parse_nodes_from_model_name(model_path: Path) -> list[int] | None:
    stem = model_path.stem
    nodes_match = re.search(r"nodes([0-9-]+)", stem)
    if nodes_match:
        return [int(item) for item in nodes_match.group(1).split("-") if item]

    node_match = re.search(r"node(\d+)", stem)
    if node_match:
        return [int(node_match.group(1))]

    return None


def parse_model_label(model_path: Path, nodes: list[int]) -> str:
    seed_match = re.search(r"seed_(\d+)", model_path.stem)
    seed_text = seed_match.group(1) if seed_match else "?"
    node_text = ",".join(str(node) for node in nodes)
    # return f"nodes {node_text}, seed {seed_text}"
    return f"nodes {node_text}"


def make_env(args: argparse.Namespace, nodes: list[int]) -> ContinuousHopfield4DHHNNEnv:
    env = ContinuousHopfield4DHHNNEnv(
        w22=args.w22,
        w33=args.w33,
        w44=args.w44,
        pinning_nodes=[node - 1 for node in nodes],
        control_scale=args.control_scale,
    )
    env.max_steps = int(args.max_steps)
    return env


def run_episode(policy_fn, env: ContinuousHopfield4DHHNNEnv, seed: int) -> dict[str, np.ndarray]:
    obs, _ = env.reset(seed=seed)

    error_hist = [(env.statey - env.statex).copy()]
    action_hist = [np.zeros(env.action_space.shape, dtype=np.float32)]

    done = False
    while not done:
        action = np.asarray(policy_fn(obs, env), dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        error_hist.append((env.statey - env.statex).copy())
        action_hist.append(action * env.scale)

    return {
        "error": np.asarray(error_hist, dtype=np.float32),
        "action": np.asarray(action_hist, dtype=np.float32),
    }


def compute_metrics(result: dict[str, np.ndarray], step_time: float) -> dict[str, float]:
    error = result["error"]
    action = result["action"]
    l1 = np.linalg.norm(error, ord=1, axis=1)

    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_l1": float(np.mean(l1)),
        "final_l1": float(l1[-1]),
        "mean_abs_u": float(np.mean(np.abs(action))),
        "energy_u": float(np.sum(np.square(action)) * step_time),
    }


def l1_curve(result: dict[str, np.ndarray]) -> np.ndarray:
    return np.linalg.norm(result["error"], ord=1, axis=1)


def mean_std_curve(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(curves, axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def save_csv(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_path}")


def aggregate_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        groups[str(row["model_label"])].append(row)

    metric_names = ["mae", "rmse", "mean_l1", "final_l1", "mean_abs_u", "energy_u"]
    summary = []
    for label, group_rows in groups.items():
        item: dict[str, float | int | str] = {"model_label": label, "n_runs": len(group_rows)}
        item["nodes"] = str(group_rows[0]["nodes"])
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=np.float64)
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
        mean_curve, std_curve = mean_std_curve(curves)
        t = np.arange(mean_curve.shape[0]) * step_time
        ax.plot(t, mean_curve, color=color, linewidth=1.8, label=label)
        if len(curves) > 1:
            lower = np.clip(mean_curve - std_curve, 0.0, None)
            upper = mean_curve + std_curve
            ax.fill_between(t, lower, upper, color=color, alpha=0.14, linewidth=0)

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
    ppo_cls: type["PPO"],
) -> tuple[list[dict[str, float | int | str]], str, list[np.ndarray], float]:
    nodes = args.nodes or parse_nodes_from_model_name(model_path)
    if nodes is None:
        raise ValueError(
            f"Could not infer controlled nodes from model name: {model_path.name}. "
            "Please pass --nodes, for example --nodes 1 2 3."
        )

    model = ppo_cls.load(str(model_path), device="cpu")
    label = parse_model_label(model_path, nodes)

    def ppo_policy(obs: np.ndarray, _env: ContinuousHopfield4DHHNNEnv) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    rows: list[dict[str, float | int | str]] = []
    curves: list[np.ndarray] = []
    step_time: float | None = None

    print(f"\n=== Testing {model_path.name} ({label}) ===")
    for seed in test_seeds:
        env = make_env(args, nodes)
        if step_time is None:
            step_time = env.dt * env.rk4_steps

        result = run_episode(ppo_policy, env, seed)
        env.close()

        metrics = compute_metrics(result, step_time)
        curves.append(l1_curve(result))
        rows.append(
            {
                "model_name": model_path.stem,
                "model_label": label,
                "nodes": "-".join(str(node) for node in nodes),
                "test_seed": seed,
                **metrics,
            }
        )
        print(f"seed={seed} | final_l1={metrics['final_l1']:.6f} | mean_l1={metrics['mean_l1']:.6f}")

    return rows, label, curves, float(step_time)


def evaluate_zero_baseline(
    args: argparse.Namespace,
    nodes: list[int],
    test_seeds: list[int],
) -> tuple[list[dict[str, float | int | str]], str, list[np.ndarray], float]:
    label = f"zero input, nodes {','.join(str(node) for node in nodes)}"

    def zero_policy(_obs: np.ndarray, env: ContinuousHopfield4DHHNNEnv) -> np.ndarray:
        return np.zeros(env.action_space.shape, dtype=np.float32)

    rows: list[dict[str, float | int | str]] = []
    curves: list[np.ndarray] = []
    step_time: float | None = None

    print(f"\n=== Testing {label} ===")
    for seed in test_seeds:
        env = make_env(args, nodes)
        if step_time is None:
            step_time = env.dt * env.rk4_steps

        result = run_episode(zero_policy, env, seed)
        env.close()

        metrics = compute_metrics(result, step_time)
        curves.append(l1_curve(result))
        rows.append(
            {
                "model_name": "zero_input",
                "model_label": label,
                "nodes": "-".join(str(node) for node in nodes),
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
        curves_by_label[label] = curves
        if step_time is None:
            step_time = model_step_time

    if args.include_zero:
        baseline_nodes = args.nodes or parse_nodes_from_model_name(model_paths[0])
        if baseline_nodes is None:
            baseline_nodes = [4]
        rows, label, curves, baseline_step_time = evaluate_zero_baseline(args, baseline_nodes, test_seeds)
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
