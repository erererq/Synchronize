from pathlib import Path

import sys

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

current_dir = Path(__file__).resolve()
lorenz_dir = current_dir.parent.parent

sys.path.append(str(lorenz_dir))

import env  # noqa: F401

SEED = 42
N_ENVS = 4
TOTAL_TIMESTEPS = 600_000
CHECKPOINT_FREQ = 50_000
EVAL_TARGET_STEPS = 10_000


# Manual observation normalization is done in env (_get_obs), so VecNormalize-specific
# callback logic is not needed for the active training path.


def make_env(rank: int, seed: int):
    def _init():
        env = gymnasium.make("LorenzEnv-v0")
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def main():
    models_dir = Path("models")
    logs_dir = Path("logs")
    best_dir = models_dir / "best"
    # vecnorm_dir = models_dir / "vecnormalize"  # Not needed with manual normalization.

    for path in (models_dir, logs_dir, best_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([make_env(i, SEED) for i in range(N_ENVS)])
    # train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(0, SEED + 10_000)])
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # eval_env.obs_rms = train_env.obs_rms
    # eval_env.training = False


    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(logs_dir / "tb"),
        learning_rate=3e-4,
        n_steps=2048//N_ENVS,
        batch_size=64,
        gamma=0.99,
        seed=SEED,
        device="cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(models_dir),
        name_prefix="ppo_lorenz_ckpt",
    )
    # vecnormalize_callback = SaveVecNormalizeCallback(
    #     save_freq=CHECKPOINT_FREQ,
    #     save_path=vecnorm_dir,
    #     name_prefix="ppo_lorenz_vecnormalize",
    # )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(logs_dir / "eval"),
        eval_freq=max(EVAL_TARGET_STEPS // N_ENVS, 1),
        deterministic=True,
        render=False,
    )

    print("Start training (manual observation normalization is handled in env).")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback,
                #    vecnormalize_callback,
                     eval_callback],
        progress_bar=False,
    )

    model.save(str(models_dir / "ppo_lorenz_final"))
    # train_env.save(str(models_dir / "vecnormalize_final.pkl"))  # Not needed without VecNormalize.
    print("Training finished. Model checkpoints are saved in models/.")


if __name__ == "__main__":
    main()
    # current_file_abs = Path(__file__).resolve()
    # lorenz_dir = current_file_abs.parent.parent
    # print(f"当前文件绝对路径: {lorenz_dir}")
