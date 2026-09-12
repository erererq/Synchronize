"""Paired exploratory amplitude test; frozen policies and baseline parameters.

The original environment draws U(-.03,.03). Scaling eta by amplitude/.03
is algebraically equivalent to changing that range and preserves its RNG draws.
All methods and amplitudes use the same initial states and perturbation directions.
"""
from pathlib import Path
import sys
import os
import json
import hashlib
import concurrent.futures
import subprocess
import argparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code'))
from evaluate_hopfield_revision import model_path, save_csv
OUT = ROOT / 'logs/hopfield/mismatch_amplitude_screen'


def run(method, node, seed):
    import numpy as np
    import torch
    from stable_baselines3 import PPO, SAC
    from scipy.linalg import solve_continuous_are
    from env.continuous_hopfield_env import ContinuousHopfieldEnv
    from test_hopfield_smc_baseline import W_NOMINAL, sliding_surface, smc_force
    from hopfield_revision_metrics import trajectory_metrics
    torch.set_num_threads(1)
    path = model_path(method, str(node), seed) if method in ('PPO', 'SAC') else None
    model = (PPO if method == 'PPO' else SAC).load(path, device='cpu') if path else None
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path else 'deterministic'
    b = np.eye(3)[:, node-1:node]
    r = .01 if method == 'LQR_R001' else 1.
    gain = (b.T @ solve_continuous_are(W_NOMINAL-np.eye(3), b, np.eye(3), [[r]]) / r).reshape(3)
    surface = sliding_surface(node)
    rows = []
    for eta in (1., 2.):
        for amplitude in (.03, .05):
            for test_seed in range(100, 120):
                env = ContinuousHopfieldEnv(pinning_node=node-1, mismatch_scale=eta*amplitude/.03)
                obs, _ = env.reset(seed=test_seed)
                initial = dict(initial_x=json.dumps(env.statex.tolist()), initial_y=json.dumps(env.statey.tolist()))
                # Verify the actual matrix against a direct draw with the requested range.
                rng = np.random.default_rng(test_seed)
                direct = rng.uniform(-amplitude, amplitude, (3,3)).astype(np.float32)
                assert np.allclose(env.W_r, env.W+eta*env.W*direct, atol=1e-6)
                hits = [0]
                rk4 = env._rk4_step
                def monitored(*args, **kwargs):
                    result = rk4(*args, **kwargs)
                    hits[0] += int(np.count_nonzero(np.abs(result) >= 10.))
                    return result
                env._rk4_step = monitored
                errors = [(env.statey-env.statex).copy()]
                controls = [np.zeros(1)]
                for _ in range(400):
                    error = (env.statey-env.statex).astype(np.float64)
                    if model is not None:
                        action, _ = model.predict(obs, deterministic=True)
                    else:
                        force = (-8.*error[node-1] if method == 'Fixed_k8' else
                                 smc_force(env, surface, 16., .5) if method == 'SMC' else -float(gain@error))
                        action = np.array([force/4.], dtype=np.float32)
                    action = np.clip(action, -1., 1.).astype(np.float32)
                    obs, _, terminated, truncated, _ = env.step(action)
                    errors.append((env.statey-env.statex).copy())
                    controls.append(action*4.)
                assert truncated and not terminated
                rows.append(dict(method=method,node=node,controller_seed=seed,test_seed=test_seed,
                    amplitude=amplitude,eta=eta,relative_bound=eta*amplitude,state_guard_hits=hits[0],
                    model_sha256=digest,**initial,**trajectory_metrics(errors, controls)))
                env.close()
            print(method,node,seed,eta,amplitude,flush=True)
    save_csv(rows, OUT/f'{method}_node{node}_seed{seed}.csv')


def launch(job):
    method,node,seed = job
    with (OUT/f'{method}_node{node}_seed{seed}.log').open('w') as stream:
        subprocess.run([sys.executable, __file__, '--job', method, str(node), str(seed)],
            stdout=stream,stderr=subprocess.STDOUT,check=True,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return job


def summarize():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.concat([pd.read_csv(p) for p in OUT.glob('*_node*_seed*.csv')], ignore_index=True)
    assert len(frame) == 1600
    assert frame.groupby(['test_seed']).initial_x.nunique().max() == 1
    assert frame.groupby(['test_seed']).initial_y.nunique().max() == 1
    # First average test ICs, then policy seeds; retain all frozen checkpoints.
    per = frame.groupby(['node','method','amplitude','eta','controller_seed'])[['mae','energy_u','steady_l1']].mean().reset_index()
    summary = per.groupby(['node','method','amplitude','eta']).agg(
        mae=('mae','mean'),mae_seed_sd=('mae','std'),energy_u=('energy_u','mean'),
        steady_l1=('steady_l1','mean')).reset_index()
    frame.to_csv(OUT/'episodes.csv',index=False)
    summary.to_csv(OUT/'summary.csv',index=False)
    fig, axes = plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for col,node in enumerate((2,3)):
        for method in ('PPO','SAC','LQR_R1','LQR_R001','SMC','Fixed_k8'):
            data=summary[(summary.node==node)&(summary.eta==2)].sort_values('amplitude')
            data=data[data.method==method]
            for row,metric in enumerate(('mae','energy_u')):
                axes[row,col].plot(data.amplitude*200,data[metric],marker='o',label=method)
        axes[0,col].set_title(f'Node {node}, eta = 2')
        for row in (0,1):
            axes[row,col].set_xlabel('Maximum relative mismatch (%)')
            axes[row,col].set_ylabel('MAE' if row==0 else 'Control effort')
            axes[row,col].set_yscale('log')
            axes[row,col].set_xticks([6,10])
            axes[row,col].grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    fig.savefig(OUT/'amplitude_comparison.png',dpi=180)
    print(summary.to_string(index=False))
    print('State guard hits:',frame.state_guard_hits.sum())


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',nargs=3)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.job:
        run(args.job[0],int(args.job[1]),int(args.job[2]))
    else:
        jobs=[(m,n,s) for m in ('PPO','SAC') for n in (2,3) for s in (42,123,1024)]
        jobs += [(m,n,-1) for m in ('Fixed_k8','LQR_R1','LQR_R001','SMC') for n in (2,3)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for job in pool.map(launch,jobs):
                print('Completed',job,flush=True)
        summarize()
