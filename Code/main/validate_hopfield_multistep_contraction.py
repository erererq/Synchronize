"""Expanded nominal multi-step tests. Numerical evidence, not a tube certificate."""
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Code.env.continuous_hopfield_env import ContinuousHopfieldEnv
from Code.main.check_hopfield_jacobian_condition import deployed_step_error_jacobian
from Code.main.estimate_hopfield_closed_loop_conditional_lyapunov import collect_rollout

W=np.array([[-1.4,1.2,-7],[1.1,0,2.8],[.8,-2,4]],dtype=np.float32)
HORIZONS=[1,5,10,20,40,80,160]


def jac(model,node,x,e):
    return deployed_step_error_jacobian(model,node,x,e,W,4.,5.,.01,5).astype(float)


def rollout(model,node,seed):
    env=ContinuousHopfieldEnv(pinning_node=node-1)
    env.max_steps=2000
    obs,_=env.reset(seed=seed)
    original=env._rk4_step
    guard=[0,0.]
    def monitored(*args,**kwargs):
        z=original(*args,**kwargs)
        guard[0]+=int(np.count_nonzero(np.abs(z)>=10.))
        guard[1]=max(guard[1],float(np.abs(z).max()))
        return z
    env._rk4_step=monitored
    x,e=[],[]
    for _ in range(2000):
        x.append(env.statex.copy())
        e.append((env.statey-env.statex).copy())
        action,_=model.predict(obs,deterministic=True)
        obs,_,_,_,_=env.step(action)
    env.close()
    return np.asarray(x),np.asarray(e),guard


def step_batch(model,node,x,e):
    action,_=model.predict(np.concatenate((x,e),axis=1)/5.,deterministic=True)
    force=np.zeros_like(x)
    force[:,node-1]=np.clip(action[:,0],-1,1)*4
    y=x+e
    def rk(z,u):
        def rhs(a): return -a+np.tanh(a)@W.T+u
        k1=rhs(z); k2=rhs(z+.005*k1); k3=rhs(z+.005*k2); k4=rhs(z+.01*k3)
        return z+(.01/6)*(k1+2*k2+2*k3+k4)
    for _ in range(5):
        x=rk(x,0)
        y=rk(y,force)
    return x,y-x


def nonlinear_check(model,node,x,e,p):
    starts=[100,500,1000,1500]
    radii=[.001,.01,.05]
    dirs=np.concatenate((np.eye(3),-np.eye(3)))
    xx=[]; ee=[]; baseline=[]; offsets=[]
    for start in starts:
        for radius in radii:
            for direction in dirs:
                xx.append(x[start]); baseline.append(e[start]); ee.append(e[start]+radius*direction)
                offsets.append(radius*direction)
    xb=np.array(xx,dtype=float); xp=xb.copy()
    eb=np.array(baseline,dtype=float); ep=np.array(ee,dtype=float)
    offsets=np.array(offsets)
    denom=np.sqrt(np.einsum('bi,ij,bj->b',offsets,p,offsets))
    result={}
    for step in range(1,161):
        xb,eb=step_batch(model,node,xb,eb)
        xp,ep=step_batch(model,node,xp,ep)
        if step in [40,80,160]:
            difference=ep-eb
            gains=np.sqrt(np.einsum('bi,ij,bj->b',difference,p,difference))/denom
            result[step]={'max_gain':float(gains.max()),'pairs':len(gains)}
    return result


def main():
    torch.set_num_threads(1)
    target=ROOT/'logs/hopfield/multistep_validation'
    target.mkdir(parents=True,exist_ok=True)
    rows=[]; metrics=[]
    for node in [2,3]:
        for training_seed in [42,123,1024]:
            path=ROOT/f'models/hopfield/ppo_continuous_hopfield_node{node}_final_seed_{training_seed}.zip'
            model=PPO.load(str(path),device='cpu')
            xf,ef,_,_=collect_rollout(model,node,1,700)
            average=jac(model,node,xf,ef)[100:].mean(axis=0)
            assert max(abs(np.linalg.eigvals(average)))<1
            p=solve_discrete_lyapunov(average.T,np.eye(3)); p=(p+p.T)/2
            assert np.linalg.eigvalsh(p).min()>0
            s=np.linalg.cholesky(p).T; sinv=np.linalg.inv(s)
            metrics.append(dict(node=node,training_seed=training_seed,P=p.tolist(),condition_number=float(np.linalg.cond(p)),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for seed in range(11,21):
                x,e,guard=rollout(model,node,seed)
                m=jac(model,node,x,e)[100:]
                row=dict(node=node,training_seed=training_seed,test_seed=seed,guard_hits=guard[0],max_abs_state=guard[1],linear={})
                for h in HORIZONS:
                    product=np.broadcast_to(np.eye(3),(len(m)-h+1,3,3)).copy()
                    for offset in range(h): product=m[offset:offset+len(product)]@product
                    weighted=np.linalg.svd(s@product@sinv,compute_uv=False)[:,0]
                    euclidean=np.linalg.svd(product,compute_uv=False)[:,0]
                    row['linear'][h]=dict(weighted_max=float(weighted.max()),euclidean_max=float(euclidean.max()),weighted_failures=int(np.count_nonzero(weighted>=1)),windows=len(product))
                row['nonlinear']=nonlinear_check(model,node,x,e,p)
                rows.append(row)
                print(f'node={node} model={training_seed} initial={seed} weighted80={row["linear"][80]["weighted_max"]:.4f}',flush=True)
    summary=[]
    for node in [2,3]:
        group=[r for r in rows if r['node']==node]
        summary.append(dict(node=node,trajectories=len(group),
            linear={h:dict(weighted_max=max(r['linear'][h]['weighted_max'] for r in group),euclidean_max=max(r['linear'][h]['euclidean_max'] for r in group),failures=sum(r['linear'][h]['weighted_failures'] for r in group),windows=sum(r['linear'][h]['windows'] for r in group)) for h in HORIZONS},
            nonlinear={h:max(r['nonlinear'][h]['max_gain'] for r in group) for h in [40,80,160]},guard_hits=sum(r['guard_hits'] for r in group)))
    result=dict(protocol=dict(steps=2000,burn_in=100,dt=.05,fit_seed=1,test_seeds=list(range(11,21)),fit_steps=700,
        horizons=HORIZONS,nonlinear_starts=[100,500,1000,1500],nonlinear_radii=[.001,.01,.05],directions='six signed coordinate axes',
        scope='Nominal sampled tests only. No uniform region verification, bias bound, invariance, disturbance or formal synchronization certificate. Overlapping windows are dependent.'),metrics=metrics,summary=summary,rows=rows)
    (target/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
