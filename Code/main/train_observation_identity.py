"""Same-dimension observation-subset experiment; final checkpoints only."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'logs/hopfield/observation_identity'
MODELS=ROOT/'models/hopfield/observation_identity'
SEEDS=(42,123,1024)
BUDGET=401408


def dump(path,data):
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def make_env(node,subset):
    import gymnasium as gym
    import numpy as np
    from Code.env.continuous_hopfield_env import ContinuousHopfieldEnv
    class SelectedObservation(gym.ObservationWrapper):
        def __init__(self):
            super().__init__(ContinuousHopfieldEnv(pinning_node=node-1))
            self.indices=([0,node-1,3,node+2] if subset=='original4' else
                          [1,2,4,5] if subset=='alternative4' else list(range(6)))
            self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(len(self.indices),),dtype=np.float32)
        def observation(self,obs):
            return obs[self.indices].copy()
    return SelectedObservation()


def worker(node,seed):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    torch.set_num_threads(1)
    tag=f'PPO_node{node}_obs23_seed{seed}'
    path=MODELS/f'{tag}.zip'
    started=time.time()
    if path.exists():
        saved=PPO.load(path,device='cpu')
        assert saved.num_timesteps==BUDGET and saved.seed==seed
        return
    def factory(rank):
        def initialize():
            env=make_env(node,'alternative4')
            env.reset(seed=seed+rank)
            return Monitor(env,filename=str(OUT/f'{tag}_env{rank}.monitor.csv'))
        return initialize
    class Progress(BaseCallback):
        def _on_step(self):
            if self.num_timesteps%8192==0:
                dump(OUT/f'{tag}.json',dict(status='training',node=node,seed=seed,
                    steps=self.num_timesteps,budget=BUDGET,elapsed=time.time()-started))
            return True
    env=DummyVecEnv([factory(r) for r in range(4)])
    model=PPO('MlpPolicy',env,learning_rate=3e-4,n_steps=512,batch_size=128,
        gamma=.99,ent_coef=.001,seed=seed,device='cpu',verbose=0)
    dump(OUT/f'{tag}.json',dict(status='training',node=node,seed=seed,steps=0,budget=BUDGET))
    model.learn(total_timesteps=BUDGET,callback=Progress())
    assert model.num_timesteps==BUDGET
    model.save(path)
    env.close()
    dump(OUT/f'{tag}.json',dict(status='complete',node=node,seed=seed,steps=model.num_timesteps,
        elapsed=time.time()-started,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def evaluate(include_new):
    import torch
    import numpy as np
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    rows=[]
    curves={}
    for node in [2,3]:
        for subset in (['full6','original4','alternative4'] if include_new else ['full6','original4']):
            for seed in SEEDS:
                if subset=='alternative4':
                    path=MODELS/f'PPO_node{node}_obs23_seed{seed}.zip'
                else:
                    prefix='ppo' if subset=='full6' else 'ppo_obs4'
                    path=ROOT/f'models/hopfield/{prefix}_continuous_hopfield_node{node}_final_seed_{seed}.zip'
                env=make_env(node,subset)
                model=PPO.load(path,device='cpu')
                assert model.observation_space.shape==env.observation_space.shape
                assert model.num_timesteps==BUDGET and model.seed==seed
                assert model.n_envs==4 and model.n_steps==512 and model.batch_size==128
                assert model.ent_coef==.001 and model.n_epochs==10 and model.gamma==.99
                for initial in range(100,110):
                    obs,_=env.reset(seed=initial)
                    e=[(env.unwrapped.statey-env.unwrapped.statex).astype(float)]
                    u=[]
                    for k in range(400):
                        action,_=model.predict(obs,deterministic=True)
                        obs,_,terminated,truncated,_=env.step(action)
                        assert not terminated and (not truncated or k==399)
                        assert np.isfinite(obs).all()
                        e.append((env.unwrapped.statey-env.unwrapped.statex).astype(float))
                        u.append(float(action[0])*4)
                    e=np.array(e); l1=np.abs(e).sum(axis=1)
                    rows.append(dict(node=node,subset=subset,training_seed=seed,initial_seed=initial,
                        mae=float(np.abs(e).mean()),rmse=float(np.square(e).mean()**.5),
                        steady_l1=float(l1[320:].mean()),final_l1=float(l1[-1]),
                        energy=float(np.square(u).sum()*.05),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    curves[f'{node}_{subset}_{seed}_{initial}']=l1
                env.close()
    summary=[]
    for node in [2,3]:
        for subset in (['full6','original4','alternative4'] if include_new else ['full6','original4']):
            rr=[r for r in rows if r['node']==node and r['subset']==subset]
            item=dict(node=node,subset=subset,policies=3,initial_conditions=10)
            for metric in ['mae','rmse','steady_l1','final_l1','energy']:
                means=[float(np.mean([r[metric] for r in rr if r['training_seed']==s])) for s in SEEDS]
                item[metric]=dict(mean=float(np.mean(means)),std=float(np.std(means,ddof=1)),seed_means=means)
            summary.append(item)
    name='results' if include_new else 'preflight'
    dump(OUT/f'{name}.json',dict(protocol=dict(training_seeds=SEEDS,test_initial_seeds=list(range(100,110)),
        budget=BUDGET,nominal_budget=400000,nominal_only=True,subset_original={2:[0,1,3,4],3:[0,2,3,5]},
        subset_alternative=[1,2,4,5],final_checkpoints_only=True,reward='-0.1 L1 full error',
        note='Old policies reused after metadata and replay checks. Different subsets trained separately; no deployment column swap.'),
        summary=summary,rows=rows))
    np.savez_compressed(OUT/f'{name}_curves.npz',**curves)
    print(json.dumps(summary,indent=2),flush=True)


def launch(job):
    node,seed=job
    with (OUT/f'PPO_node{node}_obs23_seed{seed}.log').open('w',encoding='utf-8') as f:
        result=subprocess.run([sys.executable,'-u',__file__,'--node',str(node),'--seed',str(seed)],
            stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return node,seed,result.returncode


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--node',type=int,choices=[2,3]);p.add_argument('--seed',type=int)
    p.add_argument('--preflight',action='store_true');p.add_argument('--all',action='store_true')
    p.add_argument('--evaluate',action='store_true');p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    if a.preflight:evaluate(False)
    elif a.evaluate:evaluate(True)
    elif a.all:
        jobs=[(n,s) for n in [2,3] for s in SEEDS]
        dump(OUT/'plan.json',dict(jobs=jobs,budget=BUDGET,only_changed_factor='four-dimensional observation coordinates',
            retain_all=True,train_from_scratch=True))
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            results=list(pool.map(launch,jobs))
        if any(code for _,_,code in results):raise RuntimeError(results)
        evaluate(True)
    else:worker(a.node,a.seed)


if __name__=='__main__':main()
