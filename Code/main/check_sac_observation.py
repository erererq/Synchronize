"""Bounded SAC cross-check of the existing PPO observation-composition result.

Reuses the original environment and SAC hyperparameters; does not tune or
discard runs. Outputs live progress and per-policy results for restartability.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train_observation_identity import make_env

OUT = ROOT / 'logs/hopfield/sac_observation_check'
MODELS = ROOT / 'models/hopfield/sac_observation_check'
SEEDS = (42, 123, 1024)
BUDGET = 400000
INITIALS = list(range(930000, 930020))


def dump(path, data):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def worker(node, subset, seed, budget):
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    torch.set_num_threads(1)
    tag = f'SAC_node{node}_{subset}_seed{seed}_{budget}'
    path = MODELS / f'{tag}.zip'
    start = time.time()
    class Progress(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 5000 == 0:
                dump(OUT / f'{tag}_progress.json', dict(status='training', steps=self.num_timesteps,
                     budget=budget, elapsed=time.time()-start))
            return True
    env = Monitor(make_env(node, subset))
    if not path.exists():
        model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=100000,
                    learning_starts=5000, batch_size=128, tau=.005, gamma=.99,
                    train_freq=1, gradient_steps=1, ent_coef='auto',
                    policy_kwargs={'net_arch': [64, 64]}, seed=seed, device='cpu', verbose=0)
        dump(OUT / f'{tag}_progress.json', dict(status='training', steps=0, budget=budget))
        model.learn(total_timesteps=budget, callback=Progress())
        model.save(path)
    else:
        model = SAC.load(path, device='cpu')
    assert model.num_timesteps == budget and model.seed == seed
    assert model.observation_space.shape == (4,)
    assert model.batch_size == 128 and model.gamma == .99
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    base = env.unwrapped
    # Count actual RK4 guard activations without changing the integrator.
    rk4 = base._rk4_step
    guard = [0]
    def counted(*args, **kwargs):
        state = rk4(*args, **kwargs)
        guard[0] += int(np.any(np.abs(state) >= 10))
        return state
    base._rk4_step = counted
    rows = []
    for initial in INITIALS:
        obs, _ = env.reset(seed=initial)
        guard[0] = 0
        errors = [(base.statey-base.statex).astype(float)]
        inputs = []
        for k in range(400):
            action, _ = model.predict(obs, deterministic=True)
            assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1
            obs, _, terminated, truncated, _ = env.step(action)
            assert not terminated and truncated == (k == 399)
            assert np.isfinite(obs).all()
            errors.append((base.statey-base.statex).astype(float))
            inputs.append(float(action[0])*base.scale)
        errors = np.array(errors)
        rows.append(dict(node=node, subset=subset, training_seed=seed, initial_seed=initial,
                         mae=float(np.abs(errors).mean()),
                         steady_l1=float(np.abs(errors[320:]).sum(axis=1).mean()),
                         energy=float(np.square(inputs).sum()*.05), guard_hits=guard[0],
                         model_sha256=digest))
    env.close()
    dump(OUT / f'{tag}_results.json', dict(rows=rows, budget=budget))
    dump(OUT / f'{tag}_progress.json', dict(status='complete', steps=budget, budget=budget,
         elapsed=time.time()-start, model_sha256=digest))


def launch(job):
    node, subset, seed = job
    with (OUT / f'node{node}_{subset}_seed{seed}.log').open('w', encoding='utf-8') as output:
        p = subprocess.run([sys.executable, '-u', __file__, '--node', str(node),
                            '--subset', subset, '--seed', str(seed)], stdout=output,
                           stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                    OPENBLAS_NUM_THREADS='1'),
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    print(f'{job}: exit {p.returncode}', flush=True)
    return p.returncode


def summarize():
    import numpy as np
    rows = []
    for node in (2, 3):
        for subset in ('original4', 'alternative4'):
            for seed in SEEDS:
                path = OUT / f'SAC_node{node}_{subset}_seed{seed}_{BUDGET}_results.json'
                rows.extend(json.loads(path.read_text(encoding='utf-8'))['rows'])
    assert len(rows) == 240
    summary = []
    for node in (2, 3):
        for subset in ('original4', 'alternative4'):
            item = dict(node=node, subset=subset)
            for metric in ('mae', 'steady_l1', 'energy'):
                means = [float(np.mean([r[metric] for r in rows if r['node']==node and
                         r['subset']==subset and r['training_seed']==seed])) for seed in SEEDS]
                item[metric] = dict(mean=float(np.mean(means)), std=float(np.std(means, ddof=1)),
                                    seed_means=means)
            summary.append(item)
    dump(OUT / 'results.json', dict(rows=rows, summary=summary,
         guard_hits=sum(r['guard_hits'] for r in rows)))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--all', action='store_true')
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--node', type=int, choices=[2, 3])
    p.add_argument('--subset', choices=['original4', 'alternative4'])
    p.add_argument('--seed', type=int)
    p.add_argument('--budget', type=int, default=BUDGET)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        summarize()
    elif a.all:
        assert a.budget == BUDGET
        jobs = [(n, o, s) for n in (2, 3) for s in SEEDS for o in ('original4', 'alternative4')]
        dump(OUT / 'protocol.json', dict(jobs=jobs, budget=BUDGET, test_initials=INITIALS,
             subsets={'node2_original4':[0,1,3,4], 'node3_original4':[0,2,3,5],
                      'alternative4':[1,2,4,5]}, final_checkpoints_only=True,
             nominal_only=True, selection_or_tuning=False,
             note='PPO has 401408 transitions due to rollout rounding; SAC has 400000. '
                  'Use the existing completed-pair PPO tests on these same initial conditions.'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            codes = list(pool.map(launch, jobs))
        if any(codes):
            raise RuntimeError(codes)
        summarize()
    else:
        assert a.node and a.subset and a.seed is not None
        worker(a.node, a.subset, a.seed, a.budget)


if __name__ == '__main__':
    main()
