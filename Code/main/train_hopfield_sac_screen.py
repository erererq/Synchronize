"""Low-cost SAC screening run for the continuous Hopfield task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from env.continuous_hopfield_env import ContinuousHopfieldEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train screening SAC policies on Hopfield Nodes 2 and 3.")
    parser.add_argument("--nodes", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--mismatch", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Small MLP updates are substantially faster without large BLAS thread pools.
    torch.set_num_threads(1)
    model_dir = PROJECT_ROOT / "models" / "hopfield"
    log_dir = PROJECT_ROOT / "logs" / "hopfield" / "sac_screen"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    for node in args.nodes:
        for seed in args.seeds:
            print(f"Training SAC: node={node}, seed={seed}, timesteps={args.timesteps}")
            env = Monitor(
                ContinuousHopfieldEnv(
                    mismatch_scale=args.mismatch,
                    pinning_node=node - 1,
                )
            )
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=100_000,
                learning_starts=5_000,
                batch_size=128,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
                policy_kwargs={"net_arch": [64, 64]},
                seed=seed,
                device=args.device,
                tensorboard_log=str(log_dir / "tb"),
                verbose=1,
            )
            model.learn(
                total_timesteps=args.timesteps,
                tb_log_name=f"node{node}_seed{seed}_{args.timesteps}steps",
                progress_bar=True,
            )
            output = model_dir / f"sac_continuous_hopfield_node{node}_screen_{args.timesteps}steps_seed_{seed}"
            model.save(str(output))
            env.close()
            print(f"Saved: {output}.zip")


if __name__ == "__main__":
    main()
