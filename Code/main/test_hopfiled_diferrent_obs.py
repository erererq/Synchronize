import csv
import re
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

current_file = Path(__file__).resolve()
code_dir = current_file.parent.parent
project_root = code_dir.parent

if str(code_dir) not in sys.path:
    sys.path.append(str(code_dir))

from env.continuous_hopfield_env import ContinuousHopfieldEnv


model_path = project_root / "models" / "hopfield"
pattern = "ppo_continuous_hopfield_node3_final_seed_*.zip"
test_seeds = list(range(100, 110))


def get_model_files(model_dir: str, pattern: str) -> list[Path]:
    model_paths = sorted(Path(model_dir).glob(pattern))
    if not model_paths:
        raise FileNotFoundError(
            f"No files matching {pattern!r} were found in {model_dir!r}."
        )
    print(f"Found {len(model_paths)} model files.")
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


def make_env(pinning_node: int) -> ContinuousHopfieldEnv:
    return ContinuousHopfieldEnv(
        mismatch_scale=0.0,
        pulse_step=None,
        pulse_width=5,
        pulse_vector=[5.0, 5.0, 5.0],
        pinning_node=pinning_node,
    )


def run_episode(policy_fn, env: ContinuousHopfieldEnv, seed: int = 2026) -> dict[str, np.ndarray]:
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


def save_csv(records: list[dict[str, float | int | str]], out_path: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved: {out_path}")


def main():
    pinning_node = 2
    paths = get_model_files(model_path, pattern)
    all_step_records = []

    for path in paths:
        metadata = parse_model_metadata(path)
        print(f"\n=== Evaluating model: {metadata['model_name']} ===")

        model = PPO.load(str(path), device="cpu")

        def ppo_policy(obs: np.ndarray, env: ContinuousHopfieldEnv) -> np.ndarray:
            action, _ = model.predict(obs, deterministic=True)
            return np.asarray(action, dtype=np.float32)

        for test_seed in test_seeds:
            env = make_env(pinning_node)
            results = run_episode(ppo_policy, env, seed=test_seed)
            env.close()

            error_array = results["error"]
            action_array = results["action"]

            for step_idx, current_error_vector in enumerate(error_array):
                current_l1 = np.linalg.norm(current_error_vector, ord=1)
                step_record = {
                    "model_name": metadata["model_name"],
                    "train_seed": metadata["train_seed"],
                    "test_seed": test_seed,
                    "step": step_idx,
                    "error_l1": float(current_l1),
                    "action": float(action_array[step_idx]),
                }
                all_step_records.append(step_record)

    out_csv = project_root / "logs" / "hopfield" / "node3_obs6_step_by_step_data.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    save_csv(all_step_records, out_csv)


if __name__ == "__main__":
    main()
