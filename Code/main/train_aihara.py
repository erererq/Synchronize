# from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os
import gymnasium
import numpy as np
import tqdm
import rich

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


current_dir = Path(__file__).resolve()
code_dir = current_dir.parent.parent
if str(code_dir) not in sys.path:
    sys.path.append(str(code_dir))

import env  # noqa: F401


SEED = 42
N_ENVS = 4
TOTAL_TIMESTEPS = 300_000
CHECKPOINT_FREQ = 25_000
EVAL_TARGET_STEPS = 5_000
ENV_ID = "AiharaEnv-v0"


def make_env(rank: int, seed: int, mismatch_scale: float):
    def _init():
        env_inst = gymnasium.make(ENV_ID, mismatch_scale=mismatch_scale)
        env_inst.reset(seed=seed + rank)
        return Monitor(env_inst)

    return _init


class MinErrorEvalCallback(BaseCallback):
    """
    自定义评估回调，记录每次评估的最小误差。
    """

    def __init__(self, eval_env, best_model_save_path: str,eval_freq=10000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.eval_freq = eval_freq
        self.best_mean_error = np.inf

        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # 运行评估回合
            obs = self.eval_env.reset()
            episode_errors = []
            current_episode_error = 0.0
            episode = 0
            n_episodes = 5
            while episode < n_episodes:
                action, _ =self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                assert "error_norm" in info[0], "致命错误:info 字典中丢失了 'error_norm' 指标！请检查环境的 step() 函数。"
                current_episode_error += info[0]["error_norm"]
                
                if done[0]:
                    episode_errors.append(current_episode_error)
                    current_episode_error = 0.0
                    # obs = self.eval_env.reset() 向量化环境会自动在 done=True 时重置，无需手动调用 reset()，直接继续下一步即可。
                    episode += 1

            mean_error = np.mean(episode_errors)

            if self.verbose > 0:
                print(f"\nEval num_timesteps={self.num_timesteps}, mean_error={mean_error:.2f}")
            # 核心逻辑：当当前误差 < 历史最小误差时，保存模型
            if mean_error < self.best_mean_error:
                if self.verbose > 0:
                    print(f"New best mean error: {mean_error:.2f} < {self.best_mean_error:.2f}. Saving model.")
                    self.best_mean_error = mean_error
                if self.best_model_save_path is not None:
                    save_path = os.path.join(self.best_model_save_path, "best_model.zip")
                    self.model.save(save_path)
        return True

def main():
    parser = argparse.ArgumentParser(description="Train PPO on AiharaEnv-v0.")
    parser.add_argument("--mismatch", type=float, default=0.0, help="Mismatch scale used during training.")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS, help="Total PPO timesteps.")
    args = parser.parse_args()

    models_dir = Path("models") / "aihara"
    logs_dir = Path("logs") / "aihara"
    best_dir = models_dir / "best"

    for path in (models_dir, logs_dir, best_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([make_env(i, SEED, args.mismatch) for i in range(N_ENVS)])
    eval_env = DummyVecEnv([make_env(0, SEED + 10_000, args.mismatch)])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(logs_dir / "tb"),
        learning_rate=3e-4,
        n_steps=1024 // N_ENVS,
        batch_size=64,
        gamma=0.99,
        seed=SEED,
        device="cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(models_dir),
        name_prefix="ppo_aihara_ckpt",
    )
    eval_callback = MinErrorEvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        eval_freq=max(EVAL_TARGET_STEPS // N_ENVS, 1),
    )

    print(f"Start training AiharaEnv-v0 with mismatch_scale={args.mismatch:.2f}.")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model.save(str(models_dir / "ppo_aihara_final"))
    print("Training finished. Aihara checkpoints are saved in models/aihara/.")


if __name__ == "__main__":
    main()
