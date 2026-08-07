import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from env.continuous_hopfield_env import ContinuousHopfieldEnv

def quick_test():
    model_path="models/hopfield/ppo_obs2_continuous_hopfield_node3_final_seed_42.zip"
    model=PPO.load(str(model_path),device="cpu")
    env = ContinuousHopfieldEnv(pinning_node=2)
    obs,_ = env.reset(seed=42)
    error_hist=[]
    done=False
    while not done:
        action,_= model.predict(obs,deterministic=True)
        obs,reward,terminated,truncated,info=env.step(action)
        done=terminated or truncated
        error_hist.append(np.linalg.norm(env.statey-env.statex,ord=1))
    plt.plot(error_hist)
    plt.show()

if __name__ == "__main__":
    quick_test()