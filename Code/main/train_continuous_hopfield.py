from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from env.continuous_hopfield_env import ContinuousHopfieldEnv  


N_ENVS = 4
TOTAL_TIMESTEPS = 400_000
CHECKPOINT_FREQ = 25_000
EVAL_TARGET_STEPS = 5_000
ENV_ID = "ContinuousHopfieldEnv-v0"
pinning_node = 1
obs_dimention = 4

def make_env(rank: int, seed: int, mismatch_scale: float):
    def _init():
        env_inst = ContinuousHopfieldEnv(mismatch_scale=mismatch_scale,pinning_node=pinning_node)
        env_inst.reset(seed=seed + rank)
        return Monitor(env_inst)

    return _init


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
                verdict = "scale small, control too weak"
            elif ratio > 1.50:
                verdict = "scale large, control too strong"
            else:
                verdict = "scale appropriate"

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


def main():
    parser = argparse.ArgumentParser(description="Train PPO on ContinuousHopfieldEnv-v0.")
    parser.add_argument("--mismatch", type=float, default=0.0, help="Mismatch scale used during training.")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS, help="Total PPO timesteps.")
    parser.add_argument("--seeds",type=int,nargs="+",default=[42,123,1024,2048,3072],help="List of random seeds to use.")
    args = parser.parse_args()

    models_dir = Path("models") / "hopfield"
    logs_dir = Path("logs") / "hopfield"

    for path in (models_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    
    print(f"Start training Pinning Control on {ENV_ID} with mismatch_scale={args.mismatch:.2f}.")
    print(f"Will run for following seeds: {args.seeds}")

    for current_seed in args.seeds:
        print(f"\n{'='*20} Starting Seed: {current_seed} {'='*20}")

        # 1. 为当前种子配置独立的保存目录，防止模型相互覆盖
        best_dir = models_dir / f"best_seed_{current_seed}"
        best_dir.mkdir(parents=True, exist_ok=True)
        eval_log_dir = logs_dir / f"eval_seed_{current_seed}"

        train_env = DummyVecEnv([make_env(i, current_seed, args.mismatch) for i in range(N_ENVS)])
        # eval_env = DummyVecEnv([make_env(0, current_seed + 10_000, args.mismatch)])

        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=str(logs_dir / "tb"),
            # tb_log_name=f"node1_mismatch_{args.mismatch:.2f}",
            learning_rate=3e-4,
            n_steps=2048 // N_ENVS,
            batch_size=128,
            gamma=0.99,
            seed=current_seed,
            device="cpu",
            ent_coef=0.001
        )

        # eval_callback = EvalCallback(
        #     eval_env,
        #     best_model_save_path=str(best_dir),
        #     log_path=str(logs_dir / "eval"),
        #     eval_freq=max(EVAL_TARGET_STEPS // N_ENVS, 1),
        #     deterministic=True,
        #     render=False,
        # )
        scale_callback = ScaleDiagnosticsCallback(print_freq=10, verbose=1)

        print(f"Start training Pinning Control on {ENV_ID} with mismatch_scale={args.mismatch:.2f}.")
        model.learn(
            total_timesteps=args.timesteps,
            callback=[scale_callback],
            progress_bar=True,
            # 【关键】在 tb_log_name 中标明当前种子，这样在 TensorBoard 里可以清晰地区分不同种子的曲线
            tb_log_name=f"mismatch_{args.mismatch:.2f}_seed_{current_seed}"
        )

        model.save(str(models_dir / f"ppo_obs{obs_dimention}_continuous_hopfield_node{pinning_node+1}_final_seed_{current_seed}"))
        print(f"Training finished for seed {current_seed}. Models are saved in models/hopfield/.")

        # 7. 【非常重要】循环末尾一定要关闭环境释放资源，否则跑多个种子时内存会爆炸
        train_env.close()
        # eval_env.close()


if __name__ == "__main__":
    main()
