"""Evaluate saturated scalar-feedback and single-input LQR baselines.

The two controllers use the same Hopfield environment, initial-condition seed,
parameter mismatch, sample-and-hold interval, and physical input limit.  The
scalar controller uses only the pinned-node error, whereas LQR uses the full
three-dimensional synchronization error and injects one scalar input into the
selected node.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_are


CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from env.continuous_hopfield_env import ContinuousHopfieldEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saturated scalar feedback and single-input LQR."
    )
    parser.add_argument("--nodes", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--mismatches", type=float, nargs="+", default=[0.0])
    parser.add_argument("--linear-gain", type=float, default=8.0)
    parser.add_argument("--recovery-threshold", type=float, default=0.1)
    parser.add_argument("--pulse-step", type=int, default=None)
    parser.add_argument("--pulse-width", type=int, default=1)
    parser.add_argument("--pulse-vector", type=float, nargs=3, default=[5.0, 5.0, 5.0])
    parser.add_argument(
        "--out", type=str, default="logs/hopfield/lqr_baseline"
    )
    return parser.parse_args()


def resolve_output_dir(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def compute_lqr_gain(node: int) -> tuple[np.ndarray, np.ndarray]:
    """Return K and closed-loop eigenvalues for u=-K e at the origin."""
    w = np.array(
        [
            [-1.4, 1.2, -7.0],
            [1.1, 0.0, 2.8],
            [0.8, -2.0, 4.0],
        ],
        dtype=np.float64,
    )
    a = w - np.eye(3)
    b = np.zeros((3, 1), dtype=np.float64)
    b[node - 1, 0] = 1.0
    q = np.eye(3, dtype=np.float64)
    r = np.ones((1, 1), dtype=np.float64)
    p = solve_continuous_are(a, b, q, r)
    k = np.linalg.solve(r, b.T @ p).reshape(3)
    eigvals = np.linalg.eigvals(a - b @ k.reshape(1, 3))
    return k, eigvals


def normalized_action(control_force: float, scale: float) -> np.ndarray:
    action = np.clip(control_force / scale, -1.0, 1.0)
    return np.array([action], dtype=np.float32)


def run_episode(
    method: str,
    node: int,
    seed: int,
    mismatch: float,
    linear_gain: float,
    lqr_gain: np.ndarray,
    pulse_step: int | None,
    pulse_width: int,
    pulse_vector: tuple[float, float, float],
    recovery_threshold: float,
) -> dict[str, float | int | str]:
    env = ContinuousHopfieldEnv(
        mismatch_scale=mismatch,
        pulse_step=pulse_step,
        pulse_width=pulse_width,
        pulse_vector=pulse_vector,
        pinning_node=node - 1,
    )
    env.reset(seed=seed)
    errors: list[np.ndarray] = [(env.statey - env.statex).copy()]
    controls: list[float] = [0.0]

    done = False
    while not done:
        error = (env.statey - env.statex).astype(np.float64)
        if method == "Linear":
            control_force = -linear_gain * float(error[node - 1])
        elif method == "LQR":
            control_force = -float(lqr_gain @ error)
        elif method == "Zero Input":
            control_force = 0.0
        else:
            raise ValueError(f"Unknown method: {method}")

        action = normalized_action(control_force, env.scale)
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        errors.append((env.statey - env.statex).copy())
        controls.append(float(action[0]) * env.scale)

    error_array = np.asarray(errors, dtype=np.float64)
    control_array = np.asarray(controls, dtype=np.float64)
    l1_error = np.linalg.norm(error_array, ord=1, axis=1)
    steady_start = max(1, int(0.8 * len(l1_error)))
    step_time = env.dt * env.rk4_steps
    saturation_fraction = float(
        np.mean(np.isclose(np.abs(control_array[1:]), env.scale, atol=1e-6))
    )
    env.close()

    if method == "LQR":
        model_name = f"lqr_node{node}_q1_r1"
    elif method == "Linear":
        model_name = f"linear_gain_{linear_gain:g}"
    else:
        model_name = "zero_input"

    result: dict[str, float | int | str] = {
        "model_name": model_name,
        "train_seed": -1,
        "node": node,
        "method": method,
        "test_seed": seed,
        "mismatch": mismatch,
        "pulse_step": pulse_step if pulse_step is not None else -1,
        "mae": float(np.mean(np.abs(error_array))),
        "rmse": float(np.sqrt(np.mean(np.square(error_array)))),
        "mean_l1": float(np.mean(l1_error)),
        "steady_l1": float(np.mean(l1_error[steady_start:])),
        "final_l1": float(l1_error[-1]),
        "mean_abs_u": float(np.mean(np.abs(control_array))),
        "energy_u": float(np.sum(np.square(control_array)) * step_time),
        "saturation_fraction": saturation_fraction,
    }
    if pulse_step is not None and pulse_step < len(l1_error):
        post_pulse = l1_error[pulse_step:]
        result["max_post_pulse"] = float(np.max(post_pulse))
        result["mean_post_pulse_error"] = float(np.mean(post_pulse))
        recovery_steps = -1
        for index in range(len(post_pulse)):
            if np.all(post_pulse[index:] < recovery_threshold):
                recovery_steps = index
                break
        result["recovery_steps"] = recovery_steps
        result["recovery_time"] = (
            float(recovery_steps * step_time) if recovery_steps >= 0 else float("nan")
        )
    return result


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[int, str, float, int], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                int(row["node"]),
                str(row["method"]),
                float(row["mismatch"]),
                int(row["pulse_step"]),
            )
        ].append(row)

    metrics = [
        "mae",
        "rmse",
        "mean_l1",
        "steady_l1",
        "final_l1",
        "mean_abs_u",
        "energy_u",
        "saturation_fraction",
    ]
    if any("max_post_pulse" in row for row in rows):
        metrics.extend(
            [
                "max_post_pulse",
                "mean_post_pulse_error",
                "recovery_steps",
                "recovery_time",
            ]
        )
    summaries: list[dict[str, float | int | str]] = []
    for (node, method, mismatch, pulse_step), group in sorted(groups.items()):
        item: dict[str, float | int | str] = {
            "node": node,
            "method": method,
            "mismatch": mismatch,
            "pulse_step": pulse_step,
            "num_runs": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            values = values[~np.isnan(values)]
            if not len(values):
                continue
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summaries.append(item)
    return summaries


def save_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    invalid_nodes = [node for node in args.nodes if node not in (1, 2, 3)]
    if invalid_nodes:
        raise ValueError(f"Nodes must be selected from 1, 2, 3; got {invalid_nodes}")

    output_dir = resolve_output_dir(args.out)
    gains: dict[int, np.ndarray] = {}
    gain_rows: list[dict[str, float | int | str]] = []
    for node in args.nodes:
        gain, eigvals = compute_lqr_gain(node)
        gains[node] = gain
        stable = bool(np.all(np.real(eigvals) < 0.0))
        print(f"Node {node}: K={gain}; closed-loop eigenvalues={eigvals}; stable={stable}")
        gain_rows.append(
            {
                "node": node,
                "k1": float(gain[0]),
                "k2": float(gain[1]),
                "k3": float(gain[2]),
                "gain_norm": float(np.linalg.norm(gain)),
                "linearized_closed_loop_stable": str(stable),
            }
        )

    rows: list[dict[str, float | int | str]] = []
    methods = ("Linear", "LQR", "Zero Input")
    for node in args.nodes:
        for mismatch in args.mismatches:
            for seed in args.seeds:
                for method in methods:
                    rows.append(
                        run_episode(
                            method=method,
                            node=node,
                            seed=seed,
                            mismatch=mismatch,
                            linear_gain=args.linear_gain,
                            lqr_gain=gains[node],
                            pulse_step=args.pulse_step,
                            pulse_width=args.pulse_width,
                            pulse_vector=tuple(args.pulse_vector),
                            recovery_threshold=args.recovery_threshold,
                        )
                    )

    summaries = summarize(rows)
    save_csv(gain_rows, output_dir / "lqr_gains.csv")
    save_csv(rows, output_dir / "raw_metrics.csv")
    save_csv(summaries, output_dir / "metrics_summary.csv")

    print("\nSummary (mean over test seeds)")
    for row in summaries:
        print(
            f"Node {row['node']} | mismatch={float(row['mismatch']):.2f} | "
            f"{row['method']}: MAE={float(row['mae_mean']):.6f}, "
            f"steady L1={float(row['steady_l1_mean']):.6f}, "
            f"energy={float(row['energy_u_mean']):.6f}, "
            f"saturation={100.0 * float(row['saturation_fraction_mean']):.2f}%"
        )


if __name__ == "__main__":
    main()
