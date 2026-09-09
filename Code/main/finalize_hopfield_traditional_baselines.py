"""Finalize fair PPO/traditional-controller comparisons for the manuscript.

The manuscript default retains the original scalar-feedback gain k=8.  The
optional gain list is kept only for diagnostic sensitivity checks.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO


CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent
if str(CURRENT_FILE.parent) not in sys.path:
    sys.path.append(str(CURRENT_FILE.parent))

from test_hopfield_smc_baseline import model_paths, run_episode, train_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO and fixed traditional-control baselines.")
    parser.add_argument("--test-seed-start", type=int, default=100)
    parser.add_argument("--test-seed-count", type=int, default=20)
    parser.add_argument("--linear-gain", type=float, default=8.0)
    parser.add_argument("--mismatch-levels", type=float, nargs="+", default=[0, 0.5, 1, 1.5, 2])
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0, 0.01, 0.02, 0.05])
    parser.add_argument("--pulse-step", type=int, default=200)
    parser.add_argument("--pulse-vector", type=float, nargs=3, default=[5, 5, 5])
    parser.add_argument("--out", type=str, default="logs/hopfield/traditional_baselines_final")
    return parser.parse_args()


def save_csv(rows: list[dict], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["node"], row["method"], row["mismatch"], row["noise_std"])].append(row)
    metrics = ["mae", "rmse", "steady_l1", "final_l1", "energy_u", "saturation_fraction", "max_post_pulse", "mean_post_pulse_error"]
    output: list[dict] = []
    for (scenario, node, method, mismatch, noise_std), group in sorted(groups.items()):
        item = {
            "scenario": scenario,
            "node": node,
            "method": method,
            "mismatch": mismatch,
            "noise_std": noise_std,
            "num_runs": len(group),
            "num_controller_seeds": len(set(row["controller_seed"] for row in group)),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group if metric in row], dtype=float)
            if len(values):
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def plot_mismatch(summary: list[dict], path_base: Path) -> None:
    methods = ("PPO", "Linear", "LQR", "SMC")
    colors = {"PPO": "#d62728", "Linear": "#7f7f7f", "LQR": "#1f77b4", "SMC": "#2ca02c"}
    markers = {"PPO": "o", "Linear": "D", "LQR": "s", "SMC": "^"}
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex="col", constrained_layout=True)
    for column, node in enumerate((2, 3)):
        for method in methods:
            selected = sorted(
                [row for row in summary if row["scenario"] == "mismatch" and int(row["node"]) == node and row["method"] == method],
                key=lambda row: float(row["mismatch"]),
            )
            x = [float(row["mismatch"]) for row in selected]
            axes[0, column].plot(x, [float(row["mae_mean"]) for row in selected], color=colors[method], marker=markers[method], label=method)
            axes[1, column].plot(x, [float(row["energy_u_mean"]) for row in selected], color=colors[method], marker=markers[method], label=method)
        axes[0, column].set_title(f"Node {node}")
        axes[0, column].set_ylabel("MAE")
        axes[0, column].set_yscale("log")
        axes[1, column].set_ylabel(r"Control energy $E_u$")
        axes[1, column].set_xlabel(r"Mismatch scale $\eta$")
        for axis in axes[:, column]:
            axis.grid(True, linestyle=":", alpha=0.4)
            handles, labels = axis.get_legend_handles_labels()
            axis.legend(handles, ["Fixed (k=8)" if label == "Linear" else label for label in labels], frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(path_base.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_pulse(summary: list[dict], path_base: Path) -> None:
    methods = ("PPO", "Linear", "LQR", "SMC")
    colors = {"PPO": "#d62728", "Linear": "#7f7f7f", "LQR": "#1f77b4", "SMC": "#2ca02c"}
    pulse = [row for row in summary if row["scenario"] == "pulse"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), constrained_layout=True)
    x = np.arange(2)
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(methods))
    for offset, method in zip(offsets, methods):
        selected = [next(row for row in pulse if int(row["node"]) == node and row["method"] == method) for node in (2, 3)]
        axes[0].bar(x + offset, [float(row["mean_post_pulse_error_mean"]) for row in selected], width, color=colors[method], label=method)
        axes[1].bar(x + offset, [float(row["energy_u_mean"]) for row in selected], width, color=colors[method], label=method)
    axes[0].set_ylabel(r"Post-pulse mean error $E_{\mathrm{post}}$")
    axes[1].set_ylabel(r"Control energy $E_u$")
    for axis in axes:
        axis.set_xticks(x, ("Node 2", "Node 3"))
        axis.grid(True, axis="y", linestyle=":", alpha=0.4)
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles, ["Fixed (k=8)" if label == "Linear" else label for label in labels], frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(path_base.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_noise(summary: list[dict], path_base: Path) -> None:
    methods = ("PPO", "Linear", "LQR", "SMC")
    colors = {"PPO": "#d62728", "Linear": "#7f7f7f", "LQR": "#1f77b4", "SMC": "#2ca02c"}
    markers = {"PPO": "o", "Linear": "D", "LQR": "s", "SMC": "^"}
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex="col", constrained_layout=True)
    for column, node in enumerate((2, 3)):
        for method in methods:
            selected = sorted(
                [row for row in summary if row["scenario"] == "noise" and int(row["node"]) == node and row["method"] == method],
                key=lambda row: float(row["noise_std"]),
            )
            x = [float(row["noise_std"]) for row in selected]
            axes[0, column].plot(x, [float(row["mae_mean"]) for row in selected], color=colors[method], marker=markers[method], label=method)
            axes[1, column].plot(x, [float(row["energy_u_mean"]) for row in selected], color=colors[method], marker=markers[method], label=method)
        axes[0, column].set_title(f"Node {node}")
        axes[0, column].set_ylabel("MAE")
        axes[0, column].set_yscale("log")
        axes[1, column].set_ylabel(r"Control energy $E_u$")
        axes[1, column].set_xlabel(r"Noise standard deviation $\sigma$")
        for axis in axes[:, column]:
            axis.grid(True, linestyle=":", alpha=0.4)
            handles, labels = axis.get_legend_handles_labels()
            axis.legend(handles, ["Fixed (k=8)" if label == "Linear" else label for label in labels], frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(path_base.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pulse_vector = tuple(float(value) for value in args.pulse_vector)

    linear_gain = float(args.linear_gain)
    print(f"Using the fixed linear-feedback gain k={linear_gain:g} for both nodes")

    smc_parameters = {2: (16.0, 0.5), 3: (16.0, 0.5)}
    ppo_models = {
        node: [(path, PPO.load(str(path), device="cpu")) for path in model_paths(node)]
        for node in (2, 3)
    }
    test_seeds = list(range(args.test_seed_start, args.test_seed_start + args.test_seed_count))
    conditions = [("mismatch", float(level), None, 0.0) for level in args.mismatch_levels]
    conditions.append(("pulse", 0.0, args.pulse_step, 0.0))
    conditions.extend(("noise", 0.0, None, float(level)) for level in args.noise_levels)
    rows: list[dict] = []
    for scenario, mismatch, pulse_step, noise_std in conditions:
        print(f"Testing {scenario}: mismatch={mismatch}, pulse={pulse_step}, noise={noise_std}")
        for node in (2, 3):
            reaching_gain, boundary_width = smc_parameters[node]
            for seed in test_seeds:
                for method in ("Linear", "LQR", "SMC"):
                    metrics = run_episode(
                        method, node, seed, mismatch, pulse_step, pulse_vector,
                        reaching_gain=reaching_gain, boundary_width=boundary_width,
                        linear_gain=linear_gain, noise_std=noise_std,
                    )
                    rows.append({
                        "scenario": scenario, "node": node, "method": method,
                        "controller_seed": -1, "test_seed": seed, "mismatch": mismatch,
                        "noise_std": noise_std,
                        "pulse_step": pulse_step if pulse_step is not None else -1,
                        "linear_gain": linear_gain if method == "Linear" else "",
                        "smc_reaching_gain": reaching_gain if method == "SMC" else "",
                        "smc_boundary_width": boundary_width if method == "SMC" else "",
                        **metrics,
                    })
                for path, model in ppo_models[node]:
                    metrics = run_episode("PPO", node, seed, mismatch, pulse_step, pulse_vector, model=model, noise_std=noise_std)
                    rows.append({
                        "scenario": scenario, "node": node, "method": "PPO",
                        "controller_seed": train_seed(path), "test_seed": seed,
                        "mismatch": mismatch, "pulse_step": pulse_step if pulse_step is not None else -1,
                        "noise_std": noise_std,
                        "linear_gain": "", "smc_reaching_gain": "", "smc_boundary_width": "",
                        **metrics,
                    })

    summary = summarize(rows)
    save_csv(rows, output_dir / "raw_metrics.csv")
    save_csv(summary, output_dir / "metrics_summary.csv")
    plot_mismatch(summary, output_dir / "mismatch_methods")
    plot_pulse(summary, output_dir / "pulse_methods")
    plot_noise(summary, output_dir / "noise_methods")
    print(f"Saved final baseline results to {output_dir}")


if __name__ == "__main__":
    main()
