"""Frozen-parameter nominal node comparison at short and extended horizons."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Code.main.validate_sync_application_bridge import AuditedEnv, check_integrator
from Code.main.test_hopfield_smc_baseline import lqr_gain, sliding_surface, smc_force


def main():
    target = ROOT/'logs/hopfield/node1_traditional_check'
    target.mkdir(parents=True, exist_ok=True)
    check_integrator()
    rows = []
    for node in [1,2,3]:
        gain, surface = lqr_gain(node), sliding_surface(node)
        for method in ['LQR','SMC']:
            for seed in range(100,120):
                env = AuditedEnv(pinning_node=node-1)
                env.max_steps = 2000
                env.reset(seed=seed)
                errors = [(env.statey-env.statex).astype(float)]
                controls = []
                for k in range(2000):
                    error = (env.statey-env.statex).astype(float)
                    force = -float(gain@error) if method=='LQR' else smc_force(env,surface,16,.5)
                    action = np.array([np.clip(force/4,-1,1)],dtype=np.float32)
                    obs,_,terminated,truncated,_ = env.step(action)
                    assert np.isfinite(obs).all() and not terminated
                    assert not truncated or k==1999
                    errors.append((env.statey-env.statex).astype(float))
                    controls.append(float(action[0])*4)
                for horizon in [400,2000]:
                    e = np.array(errors[:horizon+1])
                    u = np.array(controls[:horizon])
                    l1 = np.abs(e).sum(axis=1)
                    tail = l1[int(.8*len(l1)):]
                    rows.append(dict(node=node,method=method,initial_seed=seed,steps=horizon,
                        mae=float(np.abs(e).mean()),tail_mean_l1=float(tail.mean()),
                        tail_max_l1=float(tail.max()),final_l1=float(l1[-1]),
                        energy=float(np.square(u).sum()*.05),
                        saturation_fraction=float(np.isclose(np.abs(u),4,atol=1e-6).mean()),
                        tail_all_below_001=bool(np.all(tail<.01)),
                        guard_hits_full_run=env.guard_hits))
                np.savez_compressed(target/f'n{node}_{method}_{seed}.npz',errors=np.array(errors),controls=controls)
                env.close()
            print(f'Node {node} {method} done',flush=True)
    summary=[]
    for node in [1,2,3]:
        for method in ['LQR','SMC']:
            for horizon in [400,2000]:
                rr=[r for r in rows if (r['node'],r['method'],r['steps'])==(node,method,horizon)]
                item=dict(node=node,method=method,steps=horizon,runs=len(rr),
                    tail_all_below_001=sum(r['tail_all_below_001'] for r in rr),
                    guard_hits=sum(r['guard_hits_full_run'] for r in rr))
                for m in ['mae','tail_mean_l1','final_l1','energy','saturation_fraction']:
                    item[m+'_mean']=float(np.mean([r[m] for r in rr]))
                    item[m+'_std']=float(np.std([r[m] for r in rr],ddof=1))
                summary.append(item)
    result=dict(protocol=dict(nominal_only=True,initial_seeds=list(range(100,120)),
        nodes=[1,2,3],methods=['LQR','SMC'],horizons=[400,2000],dt=.05,input_limit=4,
        lqr_Q='I',lqr_R=1,smc_gain=16,smc_boundary=.5,no_retuning=True,
        criterion='All last-20-percent sampled L1 errors below 0.01; diagnostic threshold, not a theorem.',
        gains={str(n):lqr_gain(n).tolist() for n in [1,2,3]},
        note='Extended horizons share the short-run prefixes; no independent trial duplication.'),summary=summary,rows=rows)
    (target/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
