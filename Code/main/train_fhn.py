from pathlib import Path
import sys
import tqdm
import rich
import gymnasium
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
current_dir = Path(__file__).resolve()
code_dir = current_dir.parent.parent
project_dir = code_dir.parent
if str(code_dir) not in sys.path:
    sys.path.append(str(code_dir))
import env  

def train_fhn_env():
    train_env = DummyVecEnv([lambda: Monitor(gymnasium.make("FHNEnv-v0", mismatch_scale=0.0))])

    model = PPO("MlpPolicy",
                 train_env,
                 learning_rate=3e-4,
                 verbose=1, 
                 seed=42,
                 device="cpu",
                 )
    models_dir = project_dir / "models" / "fhn" 
    models_dir.mkdir(parents=True, exist_ok=True)
    model.learn(total_timesteps=300000,progress_bar=True)
    model.save(str(models_dir/"ppo_fhn_final"))

if __name__ == "__main__":
    train_fhn_env()
