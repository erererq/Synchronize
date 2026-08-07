from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

current_file = Path(__file__).resolve()
code_dir = current_file.parent.parent
project_root = code_dir.parent

if str(code_dir) not in sys.path:
    sys.path.append(str(code_dir))

from env.continuous_hopfield_env import ContinuousHopfieldEnv


# ==============================
# Fixed experiment settings
# ==============================
model_path = project_root / "models" / "hopfield" / "ppo_continuous_hopfield_node1_final_seed_42.zip"
output_dir = project_root / "logs" / "hopfield" / "multi_init_test"

test_seeds = list(range(100, 110))  # 10 different initial conditions
mismatch_values = [0.0]
pinning_node = 0


def make_env(mismatch_scale: float) -> ContinuousHopfieldEnv:
    return ContinuousHopfieldEnv(
        mismatch_scale=mismatch_scale,
        pinning_node=pinning_node,
    )


def run_episode(policy_fn, env: ContinuousHopfieldEnv, seed: int) -> dict[str, np.ndarray]:
    obs, _ = env.reset(seed=seed)

    error_hist = [(env.statey - env.statex).copy()]

    done = False
    while not done:
        action = policy_fn(obs, env)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        error_hist.append((env.statey - env.statex).copy())

    return {
        "error": np.asarray(error_hist, dtype=np.float32),
    }


def compute_metrics(result: dict[str, np.ndarray], step_time: float) -> dict[str, float]:
    error = result["error"]
    l1 = np.linalg.norm(error, ord=1, axis=1)

    metrics: dict[str, float] = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_l1": float(np.mean(l1)),
        "final_l1": float(l1[-1]),
    }

    return metrics


def save_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        


def aggregate_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    metric_keys = ["mae", "rmse", "mean_l1", "final_l1"]
    avg = {}
    for key in metric_keys:
        values = [float(r[key]) for r in rows]
        avg[key] = np.sum(values) / len(values)

    summary_rows = {
        "num_runs": len(rows),
        **avg
    }

    return [summary_rows]


def build_trajectory_rows(
    test_seed: int,
    mismatch: float,
    result: dict[str, np.ndarray],
) -> list[dict[str, float | int]]:
    error = result["error"]
    l1 = np.linalg.norm(error, ord=1, axis=1)

    rows: list[dict[str, float | int]] = []
    for step_idx, (error_vec, l1_value) in enumerate(zip(error, l1)):
        rows.append(
            {
                "test_seed": test_seed,
                "mismatch": mismatch,
                "step": step_idx,
                "e1": float(error_vec[0]),
                "e2": float(error_vec[1]),
                "e3": float(error_vec[2]),
                "error": float(error_vec[0] + error_vec[1] + error_vec[2]),
            }
        )
    return rows


def main() -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading PPO model: {model_path}")
    model = PPO.load(str(model_path), device="cpu")

    def ppo_policy(obs: np.ndarray, env: ContinuousHopfieldEnv) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    raw_rows: list[dict[str, float | int | str]] = []
    trajectory_rows: list[dict[str, float | int]] = []

    for mismatch in mismatch_values:
        print(f"\n=== Mismatch Scale: {mismatch:.1f} ===")
        for seed in test_seeds:
            env_ppo = make_env(mismatch)
            step_time = env_ppo.dt * env_ppo.rk4_steps

            res_ppo = run_episode(ppo_policy, env_ppo, seed)

            env_ppo.close()

            metrics_ppo = compute_metrics(res_ppo, step_time)
            trajectory_rows.extend(build_trajectory_rows(seed, mismatch, res_ppo))

            print(f"seed={seed} | PPO final={metrics_ppo['final_l1']:.4f}")

            raw_rows.append(
                {
                    "test_seed": seed,
                    "mismatch": mismatch,
                    **metrics_ppo,
                }
            )

    summary_rows = aggregate_rows(raw_rows)

    save_csv(raw_rows, output_dir / "node1_raw_init_metrics.csv")
    save_csv(summary_rows, output_dir / "node1_init_metrics_summary.csv")
    save_csv(trajectory_rows, output_dir / "node1_raw_init_trajectories.csv")

    print(f"\nSaved raw metrics to: {output_dir / 'raw_init_metrics.csv'}")
    print(f"Saved summary metrics to: {output_dir / 'init_metrics_summary.csv'}")
    print(f"Saved trajectories to: {output_dir / 'raw_init_trajectories.csv'}")


if __name__ == "__main__":
    main()
