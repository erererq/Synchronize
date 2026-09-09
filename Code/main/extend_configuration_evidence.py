"""Paired-subset completion, post-hoc rule sensitivity, frozen-policy transfer."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,subprocess,argparse,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'logs/hopfield/configuration_extension'
MODELS=ROOT/'models/hopfield/configuration_extension'
SEEDS=[42,123,1024]

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def train(node,seed):
    import torch,gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from Code.env.continuous_hopfield_env import ContinuousHopfieldEnv
    torch.set_num_threads(1)
    path=MODELS/f'node{node}_missing4_seed{seed}.zip'
    if path.exists():
        m=PPO.load(path,device='cpu');assert m.seed==seed and m.num_timesteps==401408
        return
    indices=[0,2,3,5] if node==2 else [0,1,3,4]
    class Obs(gym.ObservationWrapper):
        def __init__(self):
            super().__init__(ContinuousHopfieldEnv(pinning_node=node-1))
            self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(4,),dtype=np.float32)
        def observation(self,o):return o[indices].copy()
    def factory(rank):
        def init():
            e=Obs();e.reset(seed=seed+rank);return Monitor(e)
        return init
    class Progress(BaseCallback):
        def _on_step(self):
            if self.num_timesteps%16384==0:save(f'node{node}_{seed}_progress.json',dict(steps=self.num_timesteps))
            return True
    env=DummyVecEnv([factory(i) for i in range(4)])
    m=PPO('MlpPolicy',env,learning_rate=3e-4,n_steps=512,batch_size=128,gamma=.99,ent_coef=.001,seed=seed,device='cpu')
    m.learn(total_timesteps=401408,callback=Progress());m.save(path);env.close()
    save(f'node{node}_{seed}_progress.json',dict(steps=m.num_timesteps,complete=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

def evaluate(node,subset,initials,eta=0.):
    import torch
    from stable_baselines3 import PPO
    from Code.env.continuous_hopfield_env import ContinuousHopfieldEnv
    from Code.main.evaluate_configuration_selection import checkpoint
    torch.set_num_threads(1)
    indices={'original4':[0,node-1,3,node+2],'alternative4':[1,2,4,5],
             'missing4':([0,2,3,5] if node==2 else [0,1,3,4])}[subset]
    class Audit(ContinuousHopfieldEnv):
        def _rk4_step(self,*args,**kwargs):
            y=super()._rk4_step(*args,**kwargs);self.hits+=int(np.any(np.abs(y)>=10));return y
    rows=[]
    for seed in SEEDS:
        path=MODELS/f'node{node}_missing4_seed{seed}.zip' if subset=='missing4' else checkpoint(node,subset,seed)
        m=PPO.load(path,device='cpu');assert m.seed==seed and m.num_timesteps==401408
        envs=[Audit(pinning_node=node-1) for _ in initials];obs=[];errors=[];states=[];matrices=[]
        for e,i in zip(envs,initials):
            e.hits=0;o,_=e.reset(seed=i)
            # Response-only elementwise relative mismatch. Independent RNG preserves paired states.
            direction=np.random.default_rng(i+7000000).uniform(-1,1,size=e.W.shape).astype(np.float32)
            e.W_r=(e.W*(1+eta*direction)).astype(np.float32)
            matrices.append(e.W_r.tolist());states.append(np.r_[e.statex,e.statey].tolist())
            obs.append(o[indices]);errors.append([e.statey-e.statex])
        energy=np.zeros(len(envs))
        for k in range(400):
            actions,_=m.predict(np.asarray(obs),deterministic=True)
            assert np.isfinite(actions).all() and np.max(np.abs(actions))<=1
            obs=[]
            for j,(e,a) in enumerate(zip(envs,actions)):
                o,_,term,trunc,_=e.step(a);assert not term and (not trunc or k==399)
                assert np.isfinite(o).all()
                obs.append(o[indices]);errors[j].append(e.statey-e.statex)
                energy[j]+=(float(a[0])*4)**2*.05
        for j,e in enumerate(envs):
            arr=np.asarray(errors[j],dtype=float);tail=np.abs(arr[320:]).sum(axis=1)
            rows.append(dict(node=node,subset=subset,seed=seed,initial=initials[j],eta=eta,
                mae=float(np.abs(arr).mean()),tail_l1=float(tail.mean()),energy=float(energy[j]),
                guard_hits=e.hits,initial_state=states[j],response_weights=matrices[j],model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            e.close()
    print(f'Evaluated Node {node} {subset}, eta={eta}',flush=True)
    return rows

def launch(job):
    n,s=job
    with (OUT/f'train_{n}_{s}.log').open('w') as f:
        p=subprocess.run([sys.executable,'-u',__file__,'--node',str(n),'--seed',str(s)],stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert p.returncode==0,(n,s,p.returncode)

def main():
    p=argparse.ArgumentParser();p.add_argument('--node',type=int);p.add_argument('--seed',type=int);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    if a.node:return train(a.node,a.seed)
    protocol=dict(training_seeds=SEEDS,budget=401408,retain_all=True,
        paired_subset_initials=list(range(930000,930020)),transfer_initials=list(range(940000,940040)),
        transfer_eta=[0,.01,.03,.05],transfer='response only W_r=W*(1+eta*U), U elementwise uniform[-1,1], independent RNG, same directions across eta; drive unchanged',
        frozen_transfer_configs=[[2,'alternative4',.1],[3,'original4',.05]],
        sensitivity='original 8 candidates and archived validation only, rates .8/.9/1.0, dimension-first vs energy-first; post-hoc, no new test claim',
        completion='2 nodes x all 3 paired four-dimensional subsets, not all arbitrary four-coordinate combinations; missing subsets exclude actuated-node pair')
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:save('protocol.json',protocol)
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(launch,[(n,s) for n in [2,3] for s in SEEDS]))
    if not (OUT/'paired_results.json').exists():
        rows=[]
        for n in [2,3]:
            for s in ['original4','alternative4','missing4']:rows+=evaluate(n,s,protocol['paired_subset_initials'])
        save('paired_results.json',rows)
    print('Stage 1 complete',flush=True)
    from Code.main.evaluate_configuration_selection import summary,SUBSETS
    old=json.loads((ROOT/'logs/hopfield/configuration_selection/validation.json').read_text())
    sensitivity=[]
    for scope in ['any','node2','node3']:
        for tol in [.01,.05,.1]:
            for rate in [.8,.9,1.]:
                for priority in ['dimension','energy']:
                    candidates=[]
                    for n in [2,3]:
                        if scope!='any' and scope!=f'node{n}':continue
                        for s,dim in SUBSETS.items():
                            stat=summary([r for r in old if r['node']==n and r['subset']==s],tol)
                            if min(stat['rates'])>=rate:candidates.append(dict(node=n,subset=s,dimension=dim,energy=stat['energy']))
                    keys=['dimension','energy'] if priority=='dimension' else ['energy','dimension']
                    selected=min(candidates,key=lambda c:(c[keys[0]],c[keys[1]],c['node'],c['subset'])) if candidates else None
                    sensitivity.append(dict(scope=scope,tolerance=tol,rate=rate,priority=priority,selected=selected))
    save('sensitivity.json',sensitivity);print('Stage 2 complete',flush=True)
    if not (OUT/'transfer_results.json').exists():
        rows=[]
        for n,s,tol in protocol['frozen_transfer_configs']:
            for eta in protocol['transfer_eta']:rows+=evaluate(n,s,protocol['transfer_initials'],eta)
        save('transfer_results.json',rows)
    print('Stage 3 complete',flush=True)

if __name__=='__main__':main()
