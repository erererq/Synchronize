"""Audit archived SAC observation results; generate a compact paper table."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from stable_baselines3 import SAC
from train_observation_identity import make_env

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'logs/hopfield/sac_observation_check'
SEEDS = (42, 123, 1024)


def main():
    torch.set_num_threads(1)
    result = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    rows = result['rows']
    keys = {(r['node'],r['subset'],r['training_seed'],r['initial_seed']) for r in rows}
    expected = {(n,o,s,i) for n in (2,3) for o in ('original4','alternative4')
                for s in SEEDS for i in range(930000,930020)}
    assert keys == expected and len(rows) == 240
    assert all(np.isfinite(r[m]) and r[m]>=0 for r in rows for m in ('mae','steady_l1','energy'))
    assert sum(r['guard_hits'] for r in rows) == result['guard_hits'] == 0
    ppo = json.loads((ROOT/'logs/hopfield/configuration_extension/paired_results.json').read_text(encoding='utf-8'))
    ppo = [r for r in ppo if r['subset'] in ('original4','alternative4')]
    assert {(r['node'],r['subset'],r['seed'],r['initial']) for r in ppo} == expected
    assert len(ppo) == 240
    summary = []
    max_diff = 0.
    for n in (2,3):
        for o in ('original4','alternative4'):
            for s in SEEDS:
                tag = f'SAC_node{n}_{o}_seed{s}_400000'
                path = ROOT/'models/hopfield/sac_observation_check'/f'{tag}.zip'
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                rr = [r for r in rows if (r['node'],r['subset'],r['training_seed'])==(n,o,s)]
                assert all(r['model_sha256']==digest for r in rr)
                model = SAC.load(path,device='cpu')
                assert model.seed==s and model.num_timesteps==400000
                assert model.observation_space.shape==(4,) and model.action_space.shape==(1,)
                assert model.learning_rate==3e-4 and model.batch_size==128 and model.buffer_size==100000
                assert model.learning_starts==5000 and model.gradient_steps==1 and model.gamma==.99
                assert model.tau==.005 and model.policy_kwargs['net_arch']==[64,64]
                env = make_env(n,o)
                obs,_ = env.reset(seed=930000)
                e=[(env.unwrapped.statey-env.unwrapped.statex).astype(float)]
                u=[]
                for k in range(400):
                    action,_=model.predict(obs,deterministic=True)
                    obs,_,terminated,truncated,_=env.step(action)
                    assert not terminated and truncated==(k==399)
                    e.append((env.unwrapped.statey-env.unwrapped.statex).astype(float))
                    u.append(float(action[0])*4)
                env.close()
                e=np.array(e)
                actual=dict(mae=float(np.abs(e).mean()),steady_l1=float(np.abs(e[320:]).sum(axis=1).mean()),
                            energy=float(np.square(u).sum()*.05))
                archived=next(r for r in rr if r['initial_seed']==930000)
                for metric,value in actual.items():
                    diff=abs(value-archived[metric]); max_diff=max(max_diff,diff)
                    assert diff<1e-9,(tag,metric,diff)
            for algorithm,data,seed_key in [('PPO',ppo,'seed'),('SAC',rows,'training_seed')]:
                item=dict(node=n,subset=o,algorithm=algorithm)
                for metric in ('mae','energy'):
                    means=[float(np.mean([r[metric] for r in data if
                           (r['node'],r['subset'],r[seed_key])==(n,o,s)])) for s in SEEDS]
                    item[metric]=dict(mean=float(np.mean(means)),sd=float(np.std(means,ddof=1)),runs=means)
                    if algorithm=='SAC':
                        recorded=next(r for r in result['summary'] if (r['node'],r['subset'])==(n,o))[metric]
                        assert abs(item[metric]['mean']-recorded['mean'])<1e-12
                        assert abs(item[metric]['sd']-recorded['std'])<1e-12
                summary.append(item)
    audit=dict(policies=12,trajectories=240,unique_records=240,guard_hits=0,replayed_trajectories=12,
               replay_max_metric_difference=max_diff,matched_ppo_records=240,summary=summary)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    table=[r'\begin{table}[pos=htbp,width=\linewidth]',r'\centering\small',
           r'\setlength{\tabcolsep}{3pt}',
           r'\caption{SAC four-dimensional observation cross-check. Each value is mean $\pm$ sample SD across three training-run means over 20 common initial conditions, shared with the PPO paired-subset tests in Table~\ref{tab:observation_identity}.}',
           r'\label{tab:sac_observation_check}',r'\begin{tabular}{@{}clcc@{}}',r'\toprule',
           r'Node & Subset & MAE & $E_u$ \\',r'\midrule']
    for item in summary:
        if item['algorithm'] != 'SAC':
            continue
        pair='23' if item['subset']=='alternative4' else ('12' if item['node']==2 else '13')
        m=item['mae'];e=item['energy']
        table.append(f"{item['node']} & $\\mathcal O_{{{pair}}}$ & "
                     f"${m['mean']:.4f}\\pm{m['sd']:.4f}$ & ${e['mean']:.2f}\\pm{e['sd']:.2f}$ "+r'\\')
    table.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (ROOT/'Manuscript/generated/sac_observation_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
