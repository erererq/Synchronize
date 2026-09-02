"""Train exploratory nominal PPO models for multi-input ablation.

The PPO settings match the existing 3-D single-input training script.  Only the
number and location of controlled nodes change between configurations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from env.continuous_hopfield_multi_input_env import ContinuousHopfieldMultiInputEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train nominal multi-input Hopfield PPO models.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["1-2", "1-3", "2-3", "1-2-3"],
        help="Controlled-node configurations using one-based node numbers.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_nodes(text: str) -> list[int]:
    nodes = [int(value) for value in text.split("-")]
    if not nodes or len(set(nodes)) != len(nodes) or any(node not in (1, 2, 3) for node in nodes):
        raise ValueError(f"Invalid controlled-node configuration: {text}")
    return nodes


def make_env(nodes: list[int], seed: int, rank: int):
    def initialize():
        env = ContinuousHopfieldMultiInputEnv(controlled_nodes=nodes)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return initialize


def main() -> None:
    args = parse_args()
    if args.n_envs < 1:
        raise ValueError("--n-envs must be positive.")

    model_dir = PROJECT_ROOT / "models" / "hopfield" / "multi_input_exploration"
    log_dir = PROJECT_ROOT / "logs" / "hopfield" / "multi_input_training"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    for config_text in args.configs:
        nodes = parse_nodes(config_text)
        model_path = model_dir / (
            f"ppo_continuous_hopfield_nodes{config_text}_final_seed_{args.seed}.zip"
        )
        if model_path.exists() and not args.overwrite:
            print(f"Skipping existing model: {model_path.name}")
            continue

        print(
            f"\nTraining nodes {config_text}: inputs={len(nodes)}, "
            f"seed={args.seed}, timesteps={args.timesteps}"
        )
        train_env = DummyVecEnv(
            [make_env(nodes, args.seed, rank) for rank in range(args.n_envs)]
        )
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=args.verbose,
            tensorboard_log=str(log_dir / "tb"),
            learning_rate=3e-4,
            n_steps=2048 // args.n_envs,
            batch_size=128,
            gamma=0.99,
            seed=args.seed,
            device=args.device,
            ent_coef=0.001,
        )
        try:
            model.learn(
                total_timesteps=args.timesteps,
                progress_bar=True,
                tb_log_name=f"nodes{config_text}_seed_{args.seed}",
            )
            model.save(str(model_path.with_suffix("")))
            print(f"Saved: {model_path}")
        finally:
            train_env.close()


if __name__ == "__main__":
    main()
