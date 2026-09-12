"""Bounded pilot for a common weighted multi-step contraction condition.

Finite trajectory points only, not certification of a region or error bound.
"""
import sys
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Code.main.estimate_hopfield_closed_loop_conditional_lyapunov import collect_rollout
from Code.main.check_hopfield_jacobian_condition import deployed_step_error_jacobian

W=np.array([[-1.4,1.2,-7],[1.1,0,2.8],[.8,-2,4]],dtype=np.float32)


def jacobians(model,node,seed):
    x,e,_,_=collect_rollout(model,node,seed,700)
    return deployed_step_error_jacobian(model,node,x,e,W,4.,5.,.01,5)[100:].astype(float)


def main():
    results=[]
    for node in [2,3]:
        for training_seed in [42,123,1024]:
            path=ROOT/f'models/hopfield/ppo_continuous_hopfield_node{node}_final_seed_{training_seed}.zip'
            model=PPO.load(str(path),device='cpu')
            fit=jacobians(model,node,1)
            test=jacobians(model,node,11)
            mean=fit.mean(axis=0)
            radius=float(max(abs(np.linalg.eigvals(mean))))
            metrics={'euclidean':np.eye(3)}
            if radius<1:
                p=solve_discrete_lyapunov(mean.T,np.eye(3))
                p=(p+p.T)/2
                if np.linalg.eigvalsh(p).min()>0:
                    metrics['mean_jacobian_lyapunov']=p
            for name,p in metrics.items():
                s=np.linalg.cholesky(p).T
                sinv=np.linalg.inv(s)
                row=dict(node=node,training_seed=training_seed,fit_initial_seed=1,
                         test_initial_seed=11,metric=name,mean_jacobian_spectral_radius=radius,
                         metric_condition_number=float(np.linalg.cond(p)),windows={})
                for h in [1,5,10,20,40,80,160]:
                    products=np.broadcast_to(np.eye(3),(len(test)-h+1,3,3)).copy()
                    for offset in range(h):
                        products=test[offset:offset+len(products)]@products
                    norms=np.linalg.svd(s@products@sinv,compute_uv=False)[:,0]
                    row['windows'][h]=dict(max_gain=float(norms.max()),fraction_below_one=float(np.mean(norms<1)),count=len(norms))
                results.append(row)
            print(f'finished node {node}, seed {training_seed}',flush=True)
    target=ROOT/'logs/hopfield/contraction_feasibility'
    target.mkdir(parents=True,exist_ok=True)
    (target/'results.json').write_text(json.dumps(dict(scope='One fit and one held-out nominal trajectory per policy, 700 steps, discard 100; frozen metric, overlapping windows. No tube verification, bias bound, disturbances or invariance.',results=results),indent=2),encoding='utf-8')
    for row in results:
        print(row['node'],row['training_seed'],row['metric'],{k:round(v['max_gain'],4) for k,v in row['windows'].items()})


if __name__=='__main__':
    main()
