from pathlib import Path

import sys

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

current_dir = Path(__file__).resolve()
code_dir = current_dir.parent.parent

sys.path.append(str(code_dir))

import env  # noqa: F401

SEED = 42
N_ENVS = 4
TOTAL_TIMESTEPS = 500_000
CHECKPOINT_FREQ = 50_000
EVAL_TARGET_STEPS = 10_000
ENV_ID = "ChuaEnv-v0"


def make_env(rank: int, seed: int):
    def _init():
        env = gymnasium.make(ENV_ID)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def main():
    models_dir = Path("models") / "chua"
    logs_dir = Path("logs") / "chua"
    best_dir = models_dir / "best"

    for path in (models_dir, logs_dir, best_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([make_env(i, SEED) for i in range(N_ENVS)])
    eval_env = DummyVecEnv([make_env(0, SEED + 10_000)])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(logs_dir / "tb"),
        learning_rate=3e-4,
        n_steps=2048 // N_ENVS,
        batch_size=64,
        gamma=0.99,
        seed=SEED,
        device="cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(models_dir),
        name_prefix="ppo_chua_ckpt",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(logs_dir / "eval"),
        eval_freq=max(EVAL_TARGET_STEPS // N_ENVS, 1),
        deterministic=True,
        render=False,
    )

    print("Start training ChuaEnv-v0.")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False,
    )

    model.save(str(models_dir / "ppo_chua_final"))
    print("Training finished. Chua checkpoints are saved in models/chua/.")


if __name__ == "__main__":
    main()
