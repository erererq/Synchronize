from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

current_file = Path(__file__).resolve()
code_dir = current_file.parent.parent
project_root = code_dir.parent

if str(code_dir) not in sys.path:
    sys.path.append(str(code_dir))

from env.continuous_hopfield_env import ContinuousHopfieldEnv


MODEL_PATH = project_root / "models" / "hopfield" / "ppo_continuous_hopfield_node3_final_seed_42.zip"
OUTPUT_DIR = project_root / "logs" / "hopfield" / "pulse_csv_test"
TEST_SEEDS = list(range(100, 110))
PINNING_NODE = 2
PULSE_STEP = 200
PULSE_WIDTH = 10
TARGET_SIGNAL = "error"  # choose from: e1, e2, e3, error
TIME_PER_STEP = 0.05


def make_env(pinning_node: int, pulse_step: int, pulse_width: int = 10) -> ContinuousHopfieldEnv:
    return ContinuousHopfieldEnv(
        pulse_step=pulse_step,
        pulse_width=pulse_width,
        pulse_vector=np.array([1.0 if i == pinning_node else 0.0 for i in range(3)], dtype=np.float32),
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

    return {"error": np.asarray(error_hist, dtype=np.float32)}


def build_trajectory_rows(test_index: int, result: dict[str, np.ndarray]) -> list[dict[str, float | int]]:
    error = result["error"]
    rows: list[dict[str, float | int]] = []

    for step_index, error_vec in enumerate(error):
        rows.append(
            {
                "test_index": test_index,
                "step": step_index,
                "time": step_index * TIME_PER_STEP,
                "e1": float(error_vec[0]),
                "e2": float(error_vec[1]),
                "e3": float(error_vec[2]),
                "error": float(np.sum(np.abs(error_vec))),
            }
        )

    return rows


def save_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_surface_csv(raw_csv_path: Path, target_signal: str, out_path: Path) -> None:
    df = pd.read_csv(raw_csv_path)
    if target_signal not in df.columns:
        raise KeyError(f"Column not found: {target_signal}")

    surface_df = df.pivot(index="test_index", columns="time", values=target_signal)
    surface_df.columns.name = "time"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    surface_df.to_csv(out_path, encoding="utf-8-sig")


def main() -> None:
    model = PPO.load(str(MODEL_PATH), device="cpu")

    def ppo_policy(obs: np.ndarray, env: ContinuousHopfieldEnv) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    trajectory_rows: list[dict[str, float | int]] = []
    for test_index, seed in enumerate(TEST_SEEDS):
        env = make_env(pinning_node=PINNING_NODE, pulse_step=PULSE_STEP, pulse_width=PULSE_WIDTH)
        result = run_episode(ppo_policy, env, seed)
        env.close()
        trajectory_rows.extend(build_trajectory_rows(test_index, result))

    raw_out_path = OUTPUT_DIR / f"raw_pulse_trajectories_{TARGET_SIGNAL}.csv"
    save_csv(raw_out_path, trajectory_rows)

    surface_out_path = OUTPUT_DIR / f"surface_pulse_{TARGET_SIGNAL}.csv"
    export_surface_csv(raw_out_path, TARGET_SIGNAL, surface_out_path)

    print(f"Saved raw trajectories to: {raw_out_path}")
    print(f"Saved surface matrix to: {surface_out_path}")


if __name__ == "__main__":
    main()
