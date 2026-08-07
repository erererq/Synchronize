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

if TYPE_CHECKING:
    from stable_baselines3 import PPO


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
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
            "Noise-only evaluation for the continuous-time Hopfield synchronization task. "
            "This script reuses the existing Hopfield environment and enforces the standard setting: "
            "no mismatch, no pulse, and Gaussian state perturbation injected once at the end of each control step."
        )
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model-dir", type=str, default="models/hopfield")
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN)
    parser.add_argument("--node", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--test-seed-start", type=int, default=100)
    parser.add_argument("--test-seed-count", type=int, default=10)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0,0.01, 0.02, 0.05])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--linear-gain", type=float, default=8.0)
    parser.add_argument(
        "--compare-noise",
        type=float,
        default=None,
        help="Noise level used for the PPO-vs-Linear comparison curve. Defaults to the largest tested noise.",
    )
    parser.add_argument("--out", type=str, default="logs/hopfield/noise_test")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def resolve_output_dir(out: str) -> Path:
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path.resolve()


def resolve_model_paths(args: argparse.Namespace) -> list[Path]:
    if args.models:
        model_paths = [Path(model) for model in args.models]
    elif args.model:
        model_paths = [Path(args.model)]
    else:
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
        model_paths = sorted(model_dir.glob(args.pattern))

    resolved_paths: list[Path] = []
    for path in model_paths:
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved_paths.append(path.resolve())

    if not resolved_paths:
        raise FileNotFoundError("No model files matched the given settings.")

    missing = [str(path) for path in resolved_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Model file(s) not found:\n" + "\n".join(missing))

    return resolved_paths


def parse_model_metadata(model_path: Path) -> dict[str, str | int]:
    stem = model_path.stem
    seed_match = re.search(r"seed_(\d+)", stem)
    node_match = re.search(r"node(\d+)", stem)
    return {
        "model_name": stem,
        "train_seed": int(seed_match.group(1)) if seed_match else -1,
        "node": int(node_match.group(1)) if node_match else -1,
    }


def make_env(pinning_node: int, process_noise_std: float, max_steps: int):
    try:
        from env.continuous_hopfield_env import ContinuousHopfieldEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gymnasium and the local Hopfield environment are required to run the noise test script."
        ) from exc

    env = ContinuousHopfieldEnv(
        pinning_node=pinning_node,
        mismatch_scale=0.0,
        pulse_step=None,
        pulse_width=1,
        pulse_vector=(0.0, 0.0, 0.0),
        process_noise_std=process_noise_std,
    )
    env.max_steps = int(max_steps)
    return env


def build_test_seeds(args: argparse.Namespace) -> list[int]:
    return list(range(args.test_seed_start, args.test_seed_start + args.test_seed_count))


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


def append_record(
    records: list[dict[str, float | int | str]],
    *,
    method: str,
    metadata: dict[str, str | int],
    test_seed: int,
    noise_std: float,
    metrics: dict[str, float],
) -> None:
    record: dict[str, float | int | str] = {
        "method": method,
        "model_name": metadata["model_name"],
        "train_seed": metadata["train_seed"],
        "node": metadata["node"],
        "test_seed": test_seed,
        "noise_std": noise_std,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mean_l1": metrics["mean_l1"],
        "final_l1": metrics["final_l1"],
        "mean_abs_u": metrics["mean_abs_u"],
        "energy_u": metrics["energy_u"],
    }
    records.append(record)


def save_csv(records: list[dict[str, float | int | str]], out_path: Path) -> None:
    if not records:
        return
    fieldnames = list(dict.fromkeys(key for row in records for key in row))
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved: {out_path}")


def aggregate_records(records: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, float], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in records:
        groups[(str(row["method"]), float(row["noise_std"]))].append(row)

    metric_names = ["mae", "rmse", "mean_l1", "final_l1", "mean_abs_u", "energy_u"]
    summary_rows: list[dict[str, float | int | str]] = []

    for (method, noise_std), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        summary_row: dict[str, float | int | str] = {
            "method": method,
            "noise_std": noise_std,
            "n_runs": len(rows),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            summary_row[f"{metric}_mean"] = float(np.mean(values))
            summary_row[f"{metric}_std"] = float(np.std(values))
        summary_rows.append(summary_row)

    return summary_rows


def build_paper_table(summary_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    by_noise: dict[float, dict[str, dict[str, float | int | str]]] = defaultdict(dict)
    for row in summary_rows:
        by_noise[float(row["noise_std"])][str(row["method"])] = row

    table_rows: list[dict[str, float | int | str]] = []
    for noise_std in sorted(by_noise):
        methods = by_noise[noise_std]
        ppo = methods.get("PPO", {})
        linear = methods.get("Linear", {})
        table_rows.append(
            {
                "sigma": noise_std,
                "PPO_MAE": ppo.get("mae_mean", np.nan),
                "Linear_MAE": linear.get("mae_mean", np.nan),
                "PPO_Final_L1": ppo.get("final_l1_mean", np.nan),
                "Linear_Final_L1": linear.get("final_l1_mean", np.nan),
                "PPO_Energy": ppo.get("energy_u_mean", np.nan),
                "Linear_Energy": linear.get("energy_u_mean", np.nan),
            }
        )
    return table_rows


def mean_std_curve(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(curves, axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def plot_summary(summary_rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    if not summary_rows:
        return

    methods = sorted({str(row["method"]) for row in summary_rows})
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    colors = {"PPO": "#d62728", "Linear": "#1f77b4"}
    markers = {"PPO": "o", "Linear": "s"}

    for method in methods:
        method_rows = [row for row in summary_rows if str(row["method"]) == method]
        method_rows.sort(key=lambda row: float(row["noise_std"]))
        noise = np.asarray([float(row["noise_std"]) for row in method_rows], dtype=np.float64)
        mae = np.asarray([float(row["mae_mean"]) for row in method_rows], dtype=np.float64)
        ax.plot(
            noise,
            mae,
            color=colors.get(method, None),
            marker=markers.get(method, "o"),
            linewidth=1.7,
            markersize=6.5,
            label=method,
        )

    ax.set_xlabel("Noise standard deviation $\sigma$")
    ax.set_ylabel("MAE")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper left")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_ppo_noise_curves(
    ppo_curves_by_noise: dict[float, list[np.ndarray]],
    step_time: float,
    out_path: Path,
) -> None:
    if not ppo_curves_by_noise:
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    cmap = plt.cm.plasma
    noise_levels = sorted(ppo_curves_by_noise)
    norm = plt.Normalize(vmin=min(noise_levels), vmax=max(noise_levels) if len(noise_levels) > 1 else min(noise_levels) + 1.0)

    for noise_std in noise_levels:
        curves = ppo_curves_by_noise[noise_std]
        mean_curve, std_curve = mean_std_curve(curves)
        t = np.arange(mean_curve.shape[0]) * step_time
        color = cmap(norm(noise_std))
        ax.plot(t, mean_curve, color=color, linewidth=1.7, label=fr"$\sigma={noise_std:g}$")
        if len(curves) > 1:
            ax.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.16, linewidth=0)

    ax.set_xlabel("Time/s")
    ax.set_ylabel(r"$||e(t)||_1$")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper right", ncol=2)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_method_comparison(
    ppo_curves: list[np.ndarray],
    linear_curves: list[np.ndarray],
    noise_std: float,
    step_time: float,
    out_path: Path,
) -> None:
    if not ppo_curves or not linear_curves:
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    specs = [
        ("PPO", ppo_curves, "#d62728"),
        ("Linear", linear_curves, "#1f77b4"),
    ]
    for label, curves, color in specs:
        mean_curve, std_curve = mean_std_curve(curves)
        t = np.arange(mean_curve.shape[0]) * step_time
        ax.plot(t, mean_curve, color=color, linewidth=1.8, label=label)
        if len(curves) > 1:
            ax.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.16, linewidth=0)

    ax.set_xlabel("Time/s")
    ax.set_ylabel("L1 Error")
    ax.set_title(fr"Noise level $\sigma={noise_std:g}$", pad=6)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper right")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04,dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = resolve_output_dir(args.out)
    model_paths = resolve_model_paths(args)
    test_seeds = build_test_seeds(args)
    pinning_node = args.node - 1

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable_baselines3 is required to load PPO models for the noise test script."
        ) from exc

    raw_records: list[dict[str, float | int | str]] = []
    ppo_curves_by_noise: dict[float, list[np.ndarray]] = defaultdict(list)
    linear_curves_by_noise: dict[float, list[np.ndarray]] = defaultdict(list)
    linear_policy = build_linear_policy(args.linear_gain)
    step_time: float | None = None

    print("Noise-only evaluation under standard conditions:")
    print("  mismatch_scale = 0.0")
    print("  pulse_step = None")
    print("  full observation = enabled")
    print(f"  pinning node = {args.node}")

    for noise_std in args.noise_levels:
        print(f"\n=== Noise std: {noise_std:.4f} ===")

        for seed in test_seeds:
            env = make_env(pinning_node=pinning_node, process_noise_std=noise_std, max_steps=args.max_steps)
            if step_time is None:
                step_time = env.dt * env.rk4_steps
            linear_result = run_episode(linear_policy, env, seed)
            env.close()
            linear_metrics = compute_metrics(linear_result, step_time)
            linear_curves_by_noise[noise_std].append(l1_curve(linear_result))
            append_record(
                raw_records,
                method="Linear",
                metadata={"model_name": "linear_feedback", "train_seed": -1, "node": args.node},
                test_seed=seed,
                noise_std=noise_std,
                metrics=linear_metrics,
            )

        for model_path in model_paths:
            metadata = parse_model_metadata(model_path)
            print(f"  Loading PPO model: {metadata['model_name']}")
            model = PPO.load(str(model_path), device="cpu")
            ppo_policy = build_ppo_policy(model)

            for seed in test_seeds:
                env = make_env(pinning_node=pinning_node, process_noise_std=noise_std, max_steps=args.max_steps)
                ppo_result = run_episode(ppo_policy, env, seed)
                env.close()
                ppo_metrics = compute_metrics(ppo_result, step_time)
                ppo_curves_by_noise[noise_std].append(l1_curve(ppo_result))
                append_record(
                    raw_records,
                    method="PPO",
                    metadata=metadata,
                    test_seed=seed,
                    noise_std=noise_std,
                    metrics=ppo_metrics,
                )

    raw_csv_path = out_dir / "raw_noise_metrics.csv"
    summary_csv_path = out_dir / "noise_metrics_summary.csv"
    paper_table_path = out_dir / "noise_paper_table.csv"
    save_csv(raw_records, raw_csv_path)

    summary_rows = aggregate_records(raw_records)
    save_csv(summary_rows, summary_csv_path)
    save_csv(build_paper_table(summary_rows), paper_table_path)

    if not args.no_plot and step_time is not None:
        plot_summary(summary_rows, out_dir / "noise_mae_trend.png")
        plot_ppo_noise_curves(ppo_curves_by_noise, step_time, out_dir / "ppo_noise_error_curves.png")

        compare_noise = args.compare_noise
        if compare_noise is None:
            compare_noise = max(args.noise_levels)
        plot_method_comparison(
            ppo_curves_by_noise.get(compare_noise, []),
            linear_curves_by_noise.get(compare_noise, []),
            compare_noise,
            step_time,
            out_dir / "ppo_vs_linear_high_noise.png",
        )


if __name__ == "__main__":
    main()
