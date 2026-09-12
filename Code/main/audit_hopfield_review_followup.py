"""Bounded review follow-up: MCLE discretization, long-rollout guards, LQR R.

No training, checkpoint selection, or changes to historical result files.
LQR protocol: Q=I, R in {.01,.1,1,10,100}; select separately for each node
on seeds 0..9, using the SAME nominal/eta=1/pulse normalized MAE score as
the SMC validation. Freeze before evaluating seeds 100..119. Report all
validation weights, even when they do not favor PPO.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Code/main'))
OUT = ROOT / 'logs/hopfield/review_followup'
W = np.array([[-1.4, 1.2, -7.], [1.1, 0., 2.8], [.8, -2., 4.]])


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2), encoding='utf-8')


def write_rows(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mcle():
    # Original script stored W as float32; reproduce those coefficients.
    w = W.astype(np.float32).astype(np.float64)
    indices = np.array([[1, 2], [0, 2], [0, 1]])

    def rhs(x, tangent):
        jac = -np.eye(3) + w * (1. - np.tanh(x)**2)[None, :]
        sub = jac[indices[:, :, None], indices[:, None, :]]
        return -x + w @ np.tanh(x), sub @ tangent

    rows = []
    for method, h in [('legacy_euler', .01), ('coupled_rk4', .01),
                      ('coupled_rk4', .005), ('coupled_rk4', .0025)]:
        x = np.array([.5, -.01, .2])
        tangent = np.tile(np.eye(2), (3, 1, 1))
        sums = np.zeros((3, 2))
        steps, burn = round(1000 / h), round(50 / h)
        for k in range(steps):
            a, da = rhs(x, tangent)
            b, db = rhs(x + h*a/2, tangent + h*da/2)
            c, dc = rhs(x + h*b/2, tangent + h*db/2)
            d, dd = rhs(x + h*c, tangent + h*dc)
            x = x + h*(a + 2*b + 2*c + d)/6
            if method == 'legacy_euler':
                tangent += h * rhs(x, tangent)[1]
            else:
                tangent += h*(da + 2*db + 2*dc + dd)/6
            tangent, upper = np.linalg.qr(tangent)
            # Legacy code excluded one extra interval but divided by 950.
            if (k > burn if method == 'legacy_euler' else k >= burn):
                sums += np.log(np.abs(np.diagonal(upper, axis1=1, axis2=2)))
        values = np.max(sums / 950., axis=1)
        for node, value in enumerate(values, 1):
            rows.append(dict(method=method, step=h, node=node, mcle=float(value),
                             total_time=1000., discarded_time=50.))
        print(method, h, values, flush=True)
        write_rows('mcle_steps.csv', rows)


def guards():
    import torch
    from stable_baselines3 import PPO
    from Code.env.continuous_hopfield_env import ContinuousHopfieldEnv
    torch.set_num_threads(1)
    source = ROOT / 'logs/jacobian_condition/long_horizon/closed_loop_conditional_lyapunov.csv'
    with source.open(encoding='utf-8-sig') as f:
        historical = list(csv.DictReader(f))
    assert len(historical) == 60
    models, rows = {}, []
    for old in historical:
        path = ROOT / 'models/hopfield' / old['model']
        if path.name not in models:
            models[path.name] = PPO.load(path, device='cpu')
        model = models[path.name]
        env = ContinuousHopfieldEnv(pinning_node=int(old['node'])-1)
        env.max_steps = 2001
        obs, _ = env.reset(seed=int(old['evaluation_seed']))
        original = env._rk4_step
        counts = [0, 0.]

        def monitored(*args, **kwargs):
            state = original(*args, **kwargs)
            counts[0] += int(np.count_nonzero(np.abs(state) >= 10.))
            counts[1] = max(counts[1], float(np.abs(state).max()))
            return state

        env._rk4_step = monitored
        errors = []
        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
            errors.append(float(np.linalg.norm(env.statey-env.statex)))
        mean = float(np.mean(errors[100:]))
        row = dict(model=path.name, model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   node=int(old['node']), evaluation_seed=int(old['evaluation_seed']),
                   state_guard_hits=counts[0], max_abs_substep_state=counts[1],
                   reproduced_residual=mean,
                   absolute_residual_difference=abs(mean-float(old['post_burn_error_mean'])))
        rows.append(row)
        print('guard', row['node'], row['evaluation_seed'], counts, flush=True)
    write_rows('long_rollout_guards.csv', rows)
    write_json('long_rollout_audit.json', dict(trajectories=len(rows),
        state_guard_hits=sum(r['state_guard_hits'] for r in rows),
        max_abs_substep_state=max(r['max_abs_substep_state'] for r in rows),
        max_residual_difference=max(r['absolute_residual_difference'] for r in rows)))


def lqr():
    from scipy.linalg import solve_continuous_are
    from Code.env.continuous_hopfield_env import ContinuousHopfieldEnv
    from hopfield_revision_metrics import trajectory_metrics
    candidates = [.01, .1, 1., 10., 100.]
    conditions = [('nominal', 0., None, 0., .1),
                  ('mismatch', 1., None, 0., .15), ('pulse', 0., 200, 0., .45)]
    write_json('lqr_protocol.json', dict(Q='I3', R=candidates, validation_seeds=list(range(10)),
        test_seeds=list(range(100,120)), validation_scenarios=conditions,
        objective='mean of nominal MAE/.1, eta=1 MAE/.15, pulse MAE/.45',
        tie_break='candidate order', action_limit=4, steps=400, dt_control=.05))

    def run(node, r, seed, scenario, eta, pulse, noise):
        b = np.eye(3)[:, node-1:node]
        p = solve_continuous_are(W-np.eye(3), b, np.eye(3), np.array([[r]]))
        gain = (b.T @ p / r).reshape(3)
        env = ContinuousHopfieldEnv(pinning_node=node-1, mismatch_scale=eta,
            pulse_step=pulse, pulse_vector=(5.,5.,5.), process_noise_std=noise)
        env.reset(seed=seed)
        errors, controls = [(env.statey-env.statex).copy()], [np.zeros(1)]
        for _ in range(400):
            action = np.array([np.clip(-gain @ (env.statey-env.statex).astype(float)/4., -1., 1.)], dtype=np.float32)
            env.step(action)
            errors.append((env.statey-env.statex).copy())
            controls.append(action*4.)
        return dict(node=node, R=r, test_seed=seed, scenario=scenario, mismatch=eta, noise_std=noise,
                    **trajectory_metrics(errors,controls,pulse_step=pulse))

    validation, scores, selected = [], [], {}
    for node in (2,3):
        for r in candidates:
            score = []
            for scenario, eta, pulse, noise, normalizer in conditions:
                trials = [run(node,r,seed,scenario,eta,pulse,noise) for seed in range(10)]
                validation.extend(trials)
                score.append(np.mean([t['mae'] for t in trials])/normalizer)
            scores.append(dict(node=node,R=r,score=float(np.mean(score))))
            print('validation', scores[-1], flush=True)
        selected[node] = min([s for s in scores if s['node']==node],key=lambda s:s['score'])['R']
    write_rows('lqr_validation_raw.csv',validation)
    write_rows('lqr_validation_scores.csv',scores)
    write_json('lqr_frozen_selection.json', selected)
    tests = []
    test_conditions = [('nominal',0.,None,0.),('mismatch',2.,None,0.),
                       ('pulse',0.,200,0.),('noise',0.,None,.05)]
    for node in (2,3):
        for r in dict.fromkeys([1., selected[node]]):
            for scenario, eta, pulse, noise in test_conditions:
                tests.extend(run(node,r,seed,scenario,eta,pulse,noise) for seed in range(100,120))
                print('test',node,r,scenario,flush=True)
    write_rows('lqr_test_raw.csv',tests)
    summaries = []
    for node in (2,3):
        for r in dict.fromkeys([1., selected[node]]):
            for scenario, eta, pulse, noise in test_conditions:
                trials = [t for t in tests if t['node']==node and t['R']==r and t['scenario']==scenario]
                summary = dict(node=node,R=r,scenario=scenario,n=len(trials))
                for key in ('mae','energy_u','steady_l1','mean_post_pulse_error'):
                    if key in trials[0]:
                        summary[key]=float(np.mean([t[key] for t in trials]))
                summaries.append(summary)
    write_rows('lqr_test_summary.csv',summaries)
    print(json.dumps(summaries,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=['mcle','guards','lqr'])
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    globals()[args.task]()
