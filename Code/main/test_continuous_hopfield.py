from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

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
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

node = 3  # manually set to 1, 2, or 3

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO on ContinuousHopfieldEnv.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*.zip")
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=str, default="logs/hopfield/test_results")
    parser.add_argument("--mismatch", type=float, default=0.0)
    parser.add_argument("--pulse-step", type=int, default=None)
    parser.add_argument("--pulse-width", type=int, default=1)
    parser.add_argument("--pulse-vector", type=float, nargs=3, default=[5.0, 5.0, 5.0])
    parser.add_argument("--sweep", action="store_true")
    return parser.parse_args()


def resolve_output_dir(out: str) -> Path:
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path.resolve()


def resolve_model_paths(args: argparse.Namespace) -> list[Path]:
    base_dir = PROJECT_ROOT / "models" / "hopfield"

    if args.models:
        model_paths = [Path(model).resolve() for model in args.models]
    elif args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
        model_paths = sorted(model_dir.glob(args.pattern))
    elif args.model:
        model_paths = [Path(args.model).resolve()]
    else:
        default_pattern = f"ppo_continuous_hopfield_node{node}_final_seed_*.zip"
        model_paths = sorted(base_dir.glob(default_pattern))

    missing = [str(path) for path in model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Model file(s) not found:\n" + "\n".join(missing))
    if not model_paths:
        raise FileNotFoundError("No model files matched the given settings.")
    return model_paths


def parse_model_metadata(model_path: Path) -> dict[str, str | int]:
    stem = model_path.stem
    seed_match = re.search(r"seed_(\d+)", stem)
    node_match = re.search(r"node(\d+)", stem)

    return {
        "model_name": stem,
        "train_seed": int(seed_match.group(1)) if seed_match else -1,
        "node": int(node_match.group(1)) if node_match else -1,
    }


def make_env(
    mismatch_scale: float,
    pulse_step: int | None,
    pulse_width: int,
    pulse_vector: tuple[float, float, float],
    pinning_node: int,
) -> ContinuousHopfieldEnv:
    env = ContinuousHopfieldEnv(
        mismatch_scale=mismatch_scale,
        pulse_step=pulse_step,
        pulse_width=pulse_width,
        pulse_vector=pulse_vector,
        pinning_node=pinning_node,
    )
    return env


def run_episode(
    policy_fn,
    env: ContinuousHopfieldEnv,
    seed: int,
) -> dict[str, np.ndarray]:
    obs, _ = env.reset(seed=seed)

    drive_hist = [env.statex.copy()]
    response_hist = [env.statey.copy()]
    error_hist = [(env.statey - env.statex).copy()]
    action_hist = [0.0]
    pulse_hist = [False]

    done = False
    while not done:
        action = policy_fn(obs, env)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        drive_hist.append(env.statex.copy())
        response_hist.append(env.statey.copy())
        error_hist.append((env.statey - env.statex).copy())
        action_hist.append(float(action[0]) * env.scale)
        pulse_hist.append(bool(info.get("pulse_applied", False)))

    return {
        "drive": np.asarray(drive_hist, dtype=np.float32),
        "response": np.asarray(response_hist, dtype=np.float32),
        "error": np.asarray(error_hist, dtype=np.float32),
        "action": np.asarray(action_hist, dtype=np.float32),
        "pulse": np.asarray(pulse_hist, dtype=bool),
    }


def evaluate_methods_for_mismatch(
    policies: dict[str, callable],
    seed: int,
    mismatch: float,
    pulse_step: int | None,
    pulse_width: int,
    pulse_vector: tuple[float, float, float],
    pinning_node: int,
) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    results: dict[str, dict[str, np.ndarray]] = {}
    step_time: float | None = None

    for method_name, policy_fn in policies.items():
        env = make_env(
            mismatch_scale=mismatch,
            pulse_step=pulse_step,
            pulse_width=pulse_width,
            pulse_vector=pulse_vector,
            pinning_node=pinning_node,
        )
        if step_time is None:
            step_time = env.dt * env.rk4_steps
        results[method_name] = run_episode(policy_fn, env, seed)
        env.close()

    return results, float(step_time)


def compute_metrics(
    result: dict[str, np.ndarray],
    step_time: float,
    pulse_step: int | None = None,
    recovery_threshold: float = 0.1,
) -> dict[str, float]:
    error = result["error"]
    action = result["action"]
    l1 = np.linalg.norm(error, ord=1, axis=1)
    steady_start = max(1, int(0.8 * len(l1)))

    metrics: dict[str, float] = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_l1": float(np.mean(l1)),
        "steady_l1": float(np.mean(l1[steady_start:])),
        "final_l1": float(l1[-1]),
        "mean_abs_u": float(np.mean(np.abs(action))),
        "energy_u": float(np.sum(np.square(action)) * step_time),
        "saturation_fraction": float(
            np.mean(np.isclose(np.abs(action[1:]), 4.0, atol=1e-6))
        ),
    }

    if pulse_step is not None and pulse_step < len(l1):
        post_pulse = l1[pulse_step:]
        metrics["max_post_pulse"] = float(np.max(post_pulse))

        recovery_steps = -1
        for idx in range(len(post_pulse)):
            if np.all(post_pulse[idx:] < recovery_threshold):
                recovery_steps = idx
                break
        
        post_pulse_mean_error = np.mean(l1[pulse_step:])
        metrics["mean_post_pulse_error"] = float(post_pulse_mean_error)
        metrics["recovery_steps"] = float(recovery_steps)
        metrics["recovery_time"] = (
            float(recovery_steps * step_time) if recovery_steps >= 0 else float("nan")
        )

    return metrics


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"  [{name}]")
    print(f"    MAE: {metrics['mae']:.4f}")
    print(f"    RMSE: {metrics['rmse']:.4f}")
    print(f"    Mean L1 Error: {metrics['mean_l1']:.4f}")
    print(f"    Final L1 Error: {metrics['final_l1']:.4f}")
    print(f"    Mean Absolute Control Force: {metrics['mean_abs_u']:.4f}")
    print(f"    Control Energy Eu: {metrics['energy_u']:.4f}")

    if "max_post_pulse" in metrics:
        print(f"    Max Post-Pulse Error: {metrics['max_post_pulse']:.4f}")
        recovery_steps = int(metrics["recovery_steps"])
        if recovery_steps >= 0:
            print(
                f"    Recovery -> Steps: {recovery_steps} | "
                f"Time: {metrics['recovery_time']:.4f}"
            )
        else:
            print("    Recovery -> Not recovered below threshold within the episode")


def build_metric_row(
    method: str,
    metrics: dict[str, float],
    metadata: dict[str, str | int],
    test_seed: int,
    mismatch: float,
    pulse_step: int | None,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "model_name": metadata["model_name"],
        "train_seed": metadata["train_seed"],
        "node": metadata["node"],
        "method": method,
        "test_seed": test_seed,
        "mismatch": mismatch,
        "pulse_step": pulse_step if pulse_step is not None else -1,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mean_l1": metrics["mean_l1"],
        "steady_l1": metrics["steady_l1"],
        "final_l1": metrics["final_l1"],
        "mean_abs_u": metrics["mean_abs_u"],
        "energy_u": metrics["energy_u"],
        "saturation_fraction": metrics["saturation_fraction"],
    }
    if "max_post_pulse" in metrics:
        row["max_post_pulse"] = metrics["max_post_pulse"]
        row["mean_post_pulse_error"] = metrics["mean_post_pulse_error"]
        row["recovery_steps"] = metrics["recovery_steps"]
        row["recovery_time"] = metrics["recovery_time"]
    return row


def append_method_rows(
    rows: list[dict[str, float | int | str]],
    method_metrics: dict[str, dict[str, float]],
    metadata: dict[str, str | int],
    test_seed: int,
    mismatch: float,
    pulse_step: int | None,
) -> None:
    for method, metrics in method_metrics.items():
        rows.append(build_metric_row(method, metrics, metadata, test_seed, mismatch, pulse_step))


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
    if not records:
        return []

    numeric_keys = [
        "mae",
        "rmse",
        "mean_l1",
        "steady_l1",
        "final_l1",
        "mean_abs_u",
        "energy_u",
        "saturation_fraction",
        "max_post_pulse",
        "recovery_steps",
        "recovery_time",
        "mean_post_pulse_error",
    ]
    group_map: dict[tuple[str, float, int], list[dict[str, float | int | str]]] = defaultdict(list)

    for row in records:
        key = (str(row["method"]), float(row["mismatch"]), int(row["pulse_step"]))
        group_map[key].append(row)

    summary_rows: list[dict[str, float | int | str]] = []
    for (method, mismatch, pulse_step), rows in sorted(group_map.items()):
        summary: dict[str, float | int | str] = {
            "method": method,
            "mismatch": mismatch,
            "pulse_step": pulse_step,
            "num_runs": len(rows),
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in rows if key in row and not np.isnan(float(row[key]))]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    return summary_rows


def plot_sweep_recovery_trend(
    mismatch_values: list[float],
    ppo_recovery: dict[float, list[float]],
    linear_recovery: dict[float, list[float]],
    out_dir: Path,
    pulse_step: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    ppo_mean = [np.nanmean(ppo_recovery[m]) if ppo_recovery[m] else np.nan for m in mismatch_values]
    ppo_std = [np.nanstd(ppo_recovery[m]) if ppo_recovery[m] else np.nan for m in mismatch_values]
    linear_mean = [np.nanmean(linear_recovery[m]) if linear_recovery[m] else np.nan for m in mismatch_values]
    linear_std = [np.nanstd(linear_recovery[m]) if linear_recovery[m] else np.nan for m in mismatch_values]

    ax.errorbar(
        mismatch_values,
        ppo_mean,
        yerr=ppo_std,
        color="#d62728",
        marker="s",
        linestyle="-",
        linewidth=2.5,
        capsize=4,
        label="PPO",
    )
    ax.errorbar(
        mismatch_values,
        linear_mean,
        yerr=linear_std,
        color="#1f77b4",
        marker="o",
        linestyle="--",
        linewidth=2.0,
        capsize=4,
        label="Linear Feedback",
    )
    ax.set_xlabel(r"Parameter Mismatch Scale ($\eta$)")
    ax.set_ylabel(r"Recovery Time $t_s$")
    ax.set_xticks(mismatch_values)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()

    save_path = out_dir / f"robustness_trend_analysis_pulse{pulse_step}.png"
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Recovery trend plot saved to: {save_path}")


def plot_ppo_mismatch_errors(
    mismatch_error_history: dict[float, list[np.ndarray]],
    out_dir: Path,
    pulse_step: int,
    step_time: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_yscale("log")

    sorted_mismatches = sorted(mismatch_error_history.keys())
    norm = plt.Normalize(vmin=min(sorted_mismatches), vmax=max(sorted_mismatches))
    cmap = plt.cm.plasma

    for mismatch in sorted_mismatches:
        histories = mismatch_error_history[mismatch]
        if not histories:
            continue
        stacked = np.stack(histories, axis=0)
        mean_curve = np.mean(stacked, axis=0)
        std_curve = np.std(stacked, axis=0)
        time_axis = np.arange(mean_curve.shape[0]) * step_time
        color = cmap(norm(mismatch))

        ax.plot(time_axis, mean_curve, label=fr"$\eta={mismatch:.1f}$", color=color, linewidth=1.8)
        if len(histories) > 1:
            lower = np.clip(mean_curve - std_curve, 1e-8, None)
            upper = np.clip(mean_curve + std_curve, 1e-8, None)
            ax.fill_between(time_axis, lower, upper, color=color, alpha=0.18)

    pulse_time = pulse_step * step_time
    ax.axvline(x=pulse_time, color="purple", linestyle=":", linewidth=2, alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Total L1 Synchronization Error $||e||_1$")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", frameon=False, title=r"Mismatch $\eta$")
    plt.tight_layout()

    save_path = out_dir / f"ppo_error_mismatch_comparison_pulse{pulse_step}.png"
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"PPO mismatch-error plot saved to: {save_path}")


def evaluate_model(
    model_path: Path,
    args: argparse.Namespace,
    mismatch_values: list[float],
    pinning_node: int,
) -> tuple[
    list[dict[str, float | int | str]],
    dict[float, list[float]],
    dict[float, list[float]],
    dict[float, list[np.ndarray]],
    float | None,
]:
    metadata = parse_model_metadata(model_path)
    print(f"\n=== Evaluating model: {metadata['model_name']} ===")
    model = PPO.load(str(model_path), device="cpu")

    def ppo_policy(obs: np.ndarray, env: ContinuousHopfieldEnv) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    def zero_policy(obs: np.ndarray, env: ContinuousHopfieldEnv) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    def linear_policy(obs: np.ndarray, env: ContinuousHopfieldEnv) -> np.ndarray:
        controlled_idx = int(np.argmax(env.B))
        error = env.statey[controlled_idx] - env.statex[controlled_idx]
        action = -8.0 * error / env.scale
        return np.clip(np.array([action], dtype=np.float32), -1.0, 1.0)

    policies = {
        "PPO": ppo_policy,
        "Linear": linear_policy,
        "Zero Input": zero_policy,
    }

    rows: list[dict[str, float | int | str]] = []
    ppo_recovery: dict[float, list[float]] = defaultdict(list)
    linear_recovery: dict[float, list[float]] = defaultdict(list)
    ppo_error_history: dict[float, list[np.ndarray]] = defaultdict(list)
    step_time: float | None = None

    for mismatch in mismatch_values:
        print(f"\n--- Testing Mismatch Scale: {mismatch:.1f} ---")

        results, step_time = evaluate_methods_for_mismatch(
            policies=policies,
            seed=args.seed,
            mismatch=mismatch,
            pulse_step=args.pulse_step,
            pulse_width=args.pulse_width,
            pulse_vector=tuple(args.pulse_vector),
            pinning_node=pinning_node,
        )

        method_metrics = {
            method: compute_metrics(result, step_time, args.pulse_step)
            for method, result in results.items()
        }

        # for method, metrics in method_metrics.items():
        #     print_metrics(method, metrics)

        append_method_rows(
            rows=rows,
            method_metrics=method_metrics,
            metadata=metadata,
            test_seed=args.seed,
            mismatch=mismatch,
            pulse_step=args.pulse_step,
        )

        ppo_error_history[mismatch].append(np.linalg.norm(results["PPO"]["error"], ord=1, axis=1))
        if args.pulse_step is not None:
            ppo_recovery[mismatch].append(method_metrics["PPO"].get("recovery_time", np.nan))
            linear_recovery[mismatch].append(method_metrics["Linear"].get("recovery_time", np.nan))

    return rows, ppo_recovery, linear_recovery, ppo_error_history, step_time


def merge_nested_lists(target: dict[float, list], source: dict[float, list]) -> None:
    for key, values in source.items():
        target[key].extend(values)


def main() -> None:
    args = parse_args()
    if node not in (1, 2, 3):
        raise ValueError(f"Global variable 'node' must be 1, 2, or 3, got: {node}")

    pinning_node = node - 1
    if args.model_dir and args.pattern:
        args.pattern = re.sub(r"node(\d+)", f"node{node}", args.pattern)

    model_paths = resolve_model_paths(args)
    out_dir = resolve_output_dir(args.out)
    mismatch_values = [0.0, 0.5, 1.0, 1.5, 2.0] if args.sweep else [args.mismatch]

    print(f"Found {len(model_paths)} model(s) to evaluate.")
    print(f"Using pinning node: {node}")

    raw_records: list[dict[str, float | int | str]] = []
    all_ppo_recovery: dict[float, list[float]] = defaultdict(list)
    all_linear_recovery: dict[float, list[float]] = defaultdict(list)
    all_ppo_error_history: dict[float, list[np.ndarray]] = defaultdict(list)
    step_time: float | None = None

    for model_path in model_paths:
        rows, ppo_recovery, linear_recovery, ppo_error_history, model_step_time = evaluate_model(
            model_path,
            args,
            mismatch_values,
            pinning_node,
        )
        raw_records.extend(rows)
        merge_nested_lists(all_ppo_recovery, ppo_recovery)
        merge_nested_lists(all_linear_recovery, linear_recovery)
        merge_nested_lists(all_ppo_error_history, ppo_error_history)
        if step_time is None:
            step_time = model_step_time

    save_csv(raw_records, out_dir / "raw_metrics.csv")
    save_csv(aggregate_records(raw_records), out_dir / "metrics_summary.csv")

    if args.sweep and args.pulse_step is not None and step_time is not None:
        plot_sweep_recovery_trend(
            mismatch_values,
            all_ppo_recovery,
            all_linear_recovery,
            out_dir,
            args.pulse_step,
        )
        plot_ppo_mismatch_errors(
            all_ppo_error_history,
            out_dir,
            args.pulse_step,
            step_time,
        )


if __name__ == "__main__":
    main()
