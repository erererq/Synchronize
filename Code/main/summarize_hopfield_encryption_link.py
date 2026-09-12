"""Aggregate the expanded synchronization-to-encryption smoke tests."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = PROJECT_ROOT / "logs" / "hopfield" / "encryption_link_expanded"
OUTPUT_ROOT = INPUT_ROOT / "aggregate"


def read_rows() -> list[dict]:
    rows: list[dict] = []
    pattern = re.compile(r"node(?P<node>[23])_eta(?P<eta>[012])_seed(?P<ppo_seed>42|123|1024)$")
    for directory in INPUT_ROOT.iterdir():
        match = pattern.fullmatch(directory.name)
        metrics_path = directory / "metrics.csv"
        if not match or not metrics_path.exists():
            continue
        with metrics_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row["ppo_seed"] = int(match.group("ppo_seed"))
                row["node"] = int(row["node"])
                row["mismatch"] = float(row["mismatch"])
                row["test_seed"] = int(row["test_seed"])
                rows.append(row)
    return rows


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(data))
    if len(data) < 2:
        return mean, mean, mean
    half_width = float(t.ppf(0.975, len(data) - 1) * np.std(data, ddof=1) / np.sqrt(len(data)))
    return mean, mean - half_width, mean + half_width


def aggregate(rows: list[dict]) -> list[dict]:
    # Use each test initial condition as the statistical unit.  The three PPO
    # policy-seed outcomes are averaged within a test seed.  Repeated classical
    # rows are identical, so only their first copy is retained.
    per_test: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metrics = (
        "continuous_mae",
        "direct_key_word_disagreement",
        "legacy_key_word_disagreement",
        "recovered_image_mae",
        "exact_recovery",
    )
    seen_classical: set[tuple] = set()
    for row in rows:
        key = (row["node"], row["mismatch"], row["method"], row["test_seed"])
        if row["method"] != "PPO":
            if key in seen_classical:
                continue
            seen_classical.add(key)
        for metric in metrics:
            value = row[metric]
            if metric == "exact_recovery":
                numeric = 1.0 if value == "True" else 0.0
            else:
                numeric = float(value)
            per_test[key][metric].append(numeric)

    grouped: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (node, mismatch, method, _), values in per_test.items():
        for metric, repetitions in values.items():
            grouped[(node, mismatch, method)][metric].append(float(np.mean(repetitions)))

    output: list[dict] = []
    for (node, mismatch, method), values in sorted(grouped.items()):
        item: dict = {"node": node, "mismatch": mismatch, "method": method, "num_test_seeds": len(values["continuous_mae"])}
        for metric in metrics:
            mean, low, high = mean_ci(values[metric])
            if metric in ("direct_key_word_disagreement", "legacy_key_word_disagreement", "exact_recovery"):
                low, high = max(0.0, low), min(1.0, high)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci95_low"] = low
            item[f"{metric}_ci95_high"] = high
        output.append(item)
    return output


def paired_comparisons(rows: list[dict]) -> list[dict]:
    values: dict[tuple, list[float]] = defaultdict(list)
    exact: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = (row["node"], row["mismatch"], row["method"], row["test_seed"])
        values[key].append(float(row["direct_key_word_disagreement"]))
        exact[key].append(1.0 if row["exact_recovery"] == "True" else 0.0)

    output: list[dict] = []
    for node, mismatch in ((2, 1.0), (2, 2.0), (3, 2.0)):
        for baseline in ("Fixed", "LQR", "SMC"):
            disagreement_differences: list[float] = []
            recovery_differences: list[float] = []
            ppo_levels: list[float] = []
            baseline_levels: list[float] = []
            for test_seed in range(100, 120):
                ppo_key = (node, mismatch, "PPO", test_seed)
                baseline_key = (node, mismatch, baseline, test_seed)
                if ppo_key not in values or baseline_key not in values:
                    continue
                ppo_value = float(np.mean(values[ppo_key]))
                baseline_value = float(np.mean(values[baseline_key]))
                ppo_levels.append(ppo_value)
                baseline_levels.append(baseline_value)
                disagreement_differences.append(ppo_value - baseline_value)
                recovery_differences.append(float(np.mean(exact[ppo_key])) - float(np.mean(exact[baseline_key])))
            disagreement_mean, disagreement_low, disagreement_high = mean_ci(disagreement_differences)
            recovery_mean, recovery_low, recovery_high = mean_ci(recovery_differences)
            baseline_mean = float(np.mean(baseline_levels))
            ppo_mean = float(np.mean(ppo_levels))
            output.append({
                "node": node,
                "mismatch": mismatch,
                "comparison": f"PPO - {baseline}",
                "num_paired_test_seeds": len(disagreement_differences),
                "ppo_direct_disagreement_mean": ppo_mean,
                "baseline_direct_disagreement_mean": baseline_mean,
                "relative_disagreement_reduction": (baseline_mean - ppo_mean) / baseline_mean if baseline_mean else "",
                "paired_disagreement_difference_mean": disagreement_mean,
                "paired_disagreement_difference_ci95_low": disagreement_low,
                "paired_disagreement_difference_ci95_high": disagreement_high,
                "paired_exact_recovery_difference_mean": recovery_mean,
                "paired_exact_recovery_difference_ci95_low": recovery_low,
                "paired_exact_recovery_difference_ci95_high": recovery_high,
            })
    return output


def ppo_sac_comparisons(ppo_rows: list[dict]) -> list[dict]:
    ppo_values: dict[tuple, list[dict]] = defaultdict(list)
    for row in ppo_rows:
        if row["method"] == "PPO":
            ppo_values[(row["node"], row["mismatch"], row["test_seed"])].append(row)

    output: list[dict] = []
    pattern = re.compile(r"node(?P<node>[23])_eta(?P<eta>[012])_sac42$")
    for directory in (INPUT_ROOT.parent / "encryption_link_sac_screen").iterdir():
        match = pattern.fullmatch(directory.name)
        if not match:
            continue
        node, mismatch = int(match.group("node")), float(match.group("eta"))
        with (directory / "metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
            sac_by_seed = {int(row["test_seed"]): row for row in csv.DictReader(handle) if row["method"] == "SAC"}
        direct_differences: list[float] = []
        continuous_differences: list[float] = []
        ppo_direct: list[float] = []
        sac_direct: list[float] = []
        ppo_continuous: list[float] = []
        sac_continuous: list[float] = []
        for test_seed, sac_row in sorted(sac_by_seed.items()):
            policies = ppo_values[(node, mismatch, test_seed)]
            if not policies:
                continue
            ppo_d = float(np.mean([float(row["direct_key_word_disagreement"]) for row in policies]))
            ppo_c = float(np.mean([float(row["continuous_mae"]) for row in policies]))
            sac_d = float(sac_row["direct_key_word_disagreement"])
            sac_c = float(sac_row["continuous_mae"])
            ppo_direct.append(ppo_d)
            sac_direct.append(sac_d)
            ppo_continuous.append(ppo_c)
            sac_continuous.append(sac_c)
            direct_differences.append(ppo_d - sac_d)
            continuous_differences.append(ppo_c - sac_c)
        d_mean, d_low, d_high = mean_ci(direct_differences)
        c_mean, c_low, c_high = mean_ci(continuous_differences)
        ppo_d_mean, ppo_d_low, ppo_d_high = mean_ci(ppo_direct)
        sac_d_mean, sac_d_low, sac_d_high = mean_ci(sac_direct)
        ppo_c_mean, ppo_c_low, ppo_c_high = mean_ci(ppo_continuous)
        sac_c_mean, sac_c_low, sac_c_high = mean_ci(sac_continuous)
        output.append({
            "node": node,
            "mismatch": mismatch,
            "num_paired_test_seeds": len(direct_differences),
            "ppo_direct_disagreement_mean": ppo_d_mean,
            "ppo_direct_disagreement_ci95_low": ppo_d_low,
            "ppo_direct_disagreement_ci95_high": ppo_d_high,
            "sac_direct_disagreement_mean": sac_d_mean,
            "sac_direct_disagreement_ci95_low": sac_d_low,
            "sac_direct_disagreement_ci95_high": sac_d_high,
            "relative_disagreement_reduction": (sac_d_mean - ppo_d_mean) / sac_d_mean,
            "paired_direct_difference_mean": d_mean,
            "paired_direct_difference_ci95_low": d_low,
            "paired_direct_difference_ci95_high": d_high,
            "ppo_continuous_mae_mean": ppo_c_mean,
            "ppo_continuous_mae_ci95_low": ppo_c_low,
            "ppo_continuous_mae_ci95_high": ppo_c_high,
            "sac_continuous_mae_mean": sac_c_mean,
            "sac_continuous_mae_ci95_low": sac_c_low,
            "sac_continuous_mae_ci95_high": sac_c_high,
            "paired_continuous_difference_mean": c_mean,
            "paired_continuous_difference_ci95_low": c_low,
            "paired_continuous_difference_ci95_high": c_high,
        })
    return sorted(output, key=lambda row: (row["node"], row["mismatch"]))


def plot_ppo_sac(rows: list[dict], path: Path) -> None:
    selected = [row for row in rows if row["node"] == 2]
    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), constrained_layout=True)
    x = np.asarray([row["mismatch"] for row in selected])
    for method, color in (("ppo", "#d62728"), ("sac", "#9467bd")):
        for axis, metric, ylabel in (
            (axes[0], "direct_disagreement", "Direct key disagreement (%)"),
            (axes[1], "continuous_mae", "Post-settling continuous MAE"),
        ):
            mean = np.asarray([row[f"{method}_{metric}_mean"] for row in selected])
            low = np.asarray([row[f"{method}_{metric}_ci95_low"] for row in selected])
            high = np.asarray([row[f"{method}_{metric}_ci95_high"] for row in selected])
            scale = 100.0 if metric == "direct_disagreement" else 1.0
            axis.errorbar(x, scale * mean, yerr=scale * np.vstack((mean - low, high - mean)), marker="o", capsize=3, color=color, label=method.upper())
            axis.set_xlabel(r"Mismatch scale $\eta$")
            axis.set_ylabel(ylabel)
            axis.grid(True, linestyle=":", alpha=0.4)
    axes[0].legend(frameon=False)
    figure.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(figure)


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_node2(rows: list[dict], path: Path) -> None:
    methods = ("PPO", "Fixed", "LQR", "SMC")
    colors = {"PPO": "#d62728", "Fixed": "#7f7f7f", "LQR": "#1f77b4", "SMC": "#2ca02c"}
    markers = {"PPO": "o", "Fixed": "D", "LQR": "s", "SMC": "^"}
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
    for method in methods:
        selected = sorted(
            [row for row in rows if row["node"] == 2 and row["method"] == method],
            key=lambda row: row["mismatch"],
        )
        x = np.asarray([row["mismatch"] for row in selected])
        for axis, metric, ylabel in (
            (axes[0], "direct_key_word_disagreement", "Direct key disagreement (%)"),
            (axes[1], "exact_recovery", "Exact recovery rate (%)"),
        ):
            y = 100 * np.asarray([row[f"{metric}_mean"] for row in selected])
            low = 100 * np.asarray([row[f"{metric}_ci95_low"] for row in selected])
            high = 100 * np.asarray([row[f"{metric}_ci95_high"] for row in selected])
            axis.errorbar(x, y, yerr=np.vstack((y - low, high - y)), color=colors[method], marker=markers[method], capsize=3, label=method)
            axis.set_xlabel(r"Mismatch scale $\eta$")
            axis.set_ylabel(ylabel)
            axis.grid(True, linestyle=":", alpha=0.4)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_ylim(-5, 105)
    figure.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = read_rows()
    summary = aggregate(all_rows)
    write_csv(summary, OUTPUT_ROOT / "expanded_summary.csv")
    write_csv(paired_comparisons(all_rows), OUTPUT_ROOT / "paired_comparisons.csv")
    sac_comparison = ppo_sac_comparisons(all_rows)
    write_csv(sac_comparison, OUTPUT_ROOT / "ppo_sac_screen_comparison.csv")
    plot_ppo_sac(sac_comparison, OUTPUT_ROOT / "ppo_sac_key_screen.png")
    plot_node2(summary, OUTPUT_ROOT / "node2_key_link_trends.png")
    print(f"Saved {len(summary)} aggregate rows to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
