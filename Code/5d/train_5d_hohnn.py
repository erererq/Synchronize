from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent

if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from env.continuous_hopfield_5d_hohnn_env import ContinuousHopfield5DHOHNNEnv


DEFAULT_TIMESTEPS = 400_000
DEFAULT_SEEDS = [42, 123, 1024, 2048, 3072]


class ScaleDiagnosticsCallback(BaseCallback):
    def __init__(self, print_freq: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for idx, done in enumerate(dones):
            if not done:
                continue
            self.episode_count += 1
            diagnostics = infos[idx].get("scale_diagnostics")
            if diagnostics is None or self.episode_count % self.print_freq != 0:
                continue

            ratio = diagnostics["global_control_ratio"]
            if ratio < 0.15:
                verdict = "scale small, control may be weak"
            elif ratio > 1.50:
                verdict = "scale large, control may be strong"
            else:
                verdict = "scale looks reasonable"

            print(
                "\n[ScaleDiag] "
                f"episode={self.episode_count} "
                f"scale={diagnostics['scale']:.2f} "
                f"global_ratio={ratio:.3f} "
                f"max_ratio={diagnostics['max_control_ratio']:.3f} "
                f"mean_error={diagnostics['mean_error_norm']:.3f} "
                f"-> {verdict}"
            )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the 5D HOHNN synchronization task.")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--node", type=int, default=4, choices=[1, 2, 3, 4, 5], help="1-based pinning control node.")
    parser.add_argument("--mismatch", type=float, default=0.0)
    parser.add_argument("--process-noise", type=float, default=0.0)
    parser.add_argument("--control-scale", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--w11", type=float, default=1.0)
    parser.add_argument("--w22", type=float, default=3.26)
    parser.add_argument("--w44", type=float, default=170.0)
    parser.add_argument("--models-dir", type=str, default=str(PROJECT_ROOT / "models" / "5d_hohnn"))
    parser.add_argument("--logs-dir", type=str, default=str(PROJECT_ROOT / "logs" / "5d" / "train"))
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def make_env(args: argparse.Namespace, rank: int, seed: int):
    def _init():
        env = ContinuousHopfield5DHOHNNEnv(
            w11=args.w11,
            w22=args.w22,
            w44=args.w44,
            mismatch_scale=args.mismatch,
            process_noise_std=args.process_noise,
            pinning_node=args.node - 1,
            control_scale=args.control_scale,
        )
        env.max_steps = int(args.max_steps)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def model_name(args: argparse.Namespace, seed: int) -> str:
    return (
        "ppo_5d_hohnn"
        f"_node{args.node}"
        f"_w11_{args.w11:g}"
        f"_w22_{args.w22:g}"
        f"_w44_{args.w44:g}"
        f"_seed_{seed}"
    )


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    logs_dir = Path(args.logs_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("Training 5D HOHNN PPO synchronization model")
    print(f"  node={args.node}, mismatch={args.mismatch}, process_noise={args.process_noise}")
    print(f"  parameters: w11={args.w11}, w22={args.w22}, w44={args.w44}")
    print(f"  seeds={args.seeds}")

    for seed in args.seeds:
        print(f"\n========== 5D HOHNN seed {seed} ==========")
        train_env = DummyVecEnv([make_env(args, rank, seed) for rank in range(args.n_envs)])
        callback = ScaleDiagnosticsCallback(print_freq=10, verbose=1)

        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=str(logs_dir / "tb"),
            learning_rate=3e-4,
            n_steps=max(2048 // args.n_envs, 1),
            batch_size=128,
            gamma=0.99,
            seed=seed,
            device=args.device,
            ent_coef=0.001,
        )

        tb_name = (
            f"5d_node{args.node}"
            f"_w11_{args.w11:g}"
            f"_w22_{args.w22:g}"
            f"_w44_{args.w44:g}"
            f"_seed_{seed}"
        )
        model.learn(total_timesteps=args.timesteps, callback=[callback], progress_bar=True, tb_log_name=tb_name)

        save_path = models_dir / f"{model_name(args, seed)}.zip"
        model.save(str(save_path))
        print(f"Saved model: {save_path}")
        train_env.close()


if __name__ == "__main__":
    main()
