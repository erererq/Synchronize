"""Paired functional check of a conditional synchronization-to-recovery bridge.

No training or tuning. The legacy image statistics are NOT regenerated here.
Outputs describe the separate centered-reconciliation/lane-permutation prototype.
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Code.env.continuous_hopfield_env import ContinuousHopfieldEnv
from Code.encry import photo_encry_checked as base
from Code.encry import photo_encry_reconciled as rec
from Code.main.test_hopfield_smc_baseline import lqr_gain, sliding_surface, smc_force

OUT = ROOT / 'logs/hopfield/sync_application_bridge'


class AuditedEnv(ContinuousHopfieldEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.guard_hits = 0

    def _rk4_step(self, state, W_matrix, B, control_force=0.0):
        k1 = self._get_derivatives(state, W_matrix, B, control_force)
        k2 = self._get_derivatives(state + .5*self.dt*k1, W_matrix, B, control_force)
        k3 = self._get_derivatives(state + .5*self.dt*k2, W_matrix, B, control_force)
        k4 = self._get_derivatives(state + self.dt*k3, W_matrix, B, control_force)
        raw = state + (self.dt/6.0)*(k1+2.0*k2+2.0*k3+k4)
        self.guard_hits += int(np.any(np.abs(raw) > 10))
        return np.clip(raw, -10, 10).astype(np.float32)


def check_integrator():
    a, b = ContinuousHopfieldEnv(pinning_node=1), AuditedEnv(pinning_node=1)
    a.reset(seed=100)
    b.reset(seed=100)
    for k in range(80):
        action = np.array([np.sin(k)*.7], dtype=np.float32)
        aa, *_ = a.step(action)
        bb, *_ = b.step(action)
        assert np.array_equal(aa, bb)
    a.close()
    b.close()


def streams(states):
    # Same float32 fourth-stream arithmetic as the standalone prototype.
    return np.array([states[:, 0], states[:, 1], states[:, 2],
                     states[:, 0]*states[:, 1]+states[:, 2]], dtype=np.float64)


def rollout(node, method, seed, model):
    env = AuditedEnv(pinning_node=node-1)
    env.max_steps = 1756
    obs, _ = env.reset(seed=seed)
    gain, surface = lqr_gain(node), sliding_surface(node)
    xs, ys = [], []
    effort = 0.0
    for k in range(env.max_steps):
        e = (env.statey-env.statex).astype(np.float64)
        if method == 'PPO':
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(1)
        else:
            force = (-8*e[node-1] if method == 'Fixed' else
                     -float(gain@e) if method == 'LQR' else
                     smc_force(env, surface, 16.0, .5))
            action = np.array([np.clip(force/4, -1, 1)], dtype=np.float32)
        assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1
        obs, _, terminated, truncated, _ = env.step(action)
        assert np.isfinite(obs).all()
        assert not terminated and (not truncated or k == env.max_steps-1)
        if k >= 1500:
            xs.append(env.statex.copy())
            ys.append(env.statey.copy())
            effort += float(action[0]*4)**2*.05
    guard_hits = env.guard_hits
    env.close()
    return np.array(xs), np.array(ys), effort, guard_hits


def candidate(cipher, rules, meta):
    return rec.transform(cipher, rules, 8, True) ^ base.mask(
        cipher.shape, meta['pixel_sum'], 'bridge-test-only')[:, :, None]


def trial(x, y, plain):
    master, response = streams(x), streams(y)
    cipher, meta = rec.encrypt(plain, master, 'bridge-test-only', delta=1.0)
    meta = json.loads(json.dumps(meta))
    rules, _ = rec.enroll(master)
    offsets = np.asarray(meta['helper']['offsets'])
    estimated = tuple((np.floor(response-offsets+.5).astype(np.int64)%8+1).astype(np.uint8))
    direct = tuple((np.floor(response+.5).astype(np.int64)%8+1).astype(np.uint8))
    coordinated_candidate = candidate(cipher, estimated, meta)
    direct_candidate = candidate(cipher, direct, meta)
    fixed_candidate = candidate(cipher, tuple(np.ones(256, dtype=np.uint8) for _ in range(4)), meta)
    accepted = True
    try:
        recovered = rec.decrypt(cipher, response, meta, 'bridge-test-only')
        assert np.array_equal(recovered, coordinated_candidate)
    except ValueError:
        accepted = False
    e = y.astype(float)-x.astype(float)
    # Exact algebraic state-to-four-stream bound before float32 stream evaluation.
    fourth_bound = np.abs(y[:, 1])*np.abs(e[:, 0])+np.abs(x[:, 0])*np.abs(e[:, 1])+np.abs(e[:, 2])
    pointwise_bound = np.maximum(np.max(np.abs(e), axis=1), fourth_bound)
    real_master4 = x[:, 0].astype(float)*x[:, 1].astype(float)+x[:, 2].astype(float)
    real_response4 = y[:, 0].astype(float)*y[:, 1].astype(float)+y[:, 2].astype(float)
    eval_error = np.abs(master[3]-real_master4)+np.abs(response[3]-real_response4)
    assert np.all(np.max(np.abs(master-response), axis=0) <= pointwise_bound+eval_error+1e-12)
    bound = float(np.max(pointwise_bound+eval_error))
    max_error = float(np.max(np.abs(master-response)))
    if bound < .5-1e-10:
        assert accepted and np.array_equal(plain, coordinated_candidate)
    return dict(state_mae=float(np.abs(e).mean()), state_max_error=float(np.abs(e).max()),
        four_stream_max_error=max_error, state_propagated_bound=bound,
        sufficient_condition_with_margin=bound < .5-1e-10,
        half_bin_margin=.5-max_error,
        direct_rule_mismatch=float(np.mean(np.asarray(rules)!=np.asarray(direct))),
        reconciled_rule_mismatch=float(np.mean(np.asarray(rules)!=np.asarray(estimated))),
        direct_exact=bool(np.array_equal(plain, direct_candidate)),
        direct_image_mae=float(np.abs(plain.astype(float)-direct_candidate.astype(float)).mean()),
        accepted=accepted,
        reconciled_exact=bool(accepted and np.array_equal(plain, coordinated_candidate)),
        candidate_image_mae=float(np.abs(plain.astype(float)-coordinated_candidate.astype(float)).mean()),
        candidate_bit_error=float(np.unpackbits(plain^coordinated_candidate).mean()),
        fixed_exact_bypassing_digest=bool(np.array_equal(plain, fixed_candidate)),
        helper_bytes=len(json.dumps(meta['helper'], separators=(',', ':')).encode()),
        image_bytes=int(plain.nbytes))


def structural_tests():
    rng = np.random.default_rng(9901)
    x = rng.uniform(-4, 4, (4, 256))
    plain = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    c, meta = rec.encrypt(plain, x, 'bridge-test-only')
    meta = json.loads(json.dumps(meta))
    rows = []
    sender, _ = rec.enroll(x)
    for shift in [-.51, -.49, 0, .49, .51, 8.0]:
        accepted = True
        try:
            recovered = rec.decrypt(c, x+shift, meta, 'bridge-test-only')
            exact = bool(np.array_equal(plain, recovered))
        except ValueError:
            accepted, exact = False, False
        rows.append(dict(shift=shift, accepted=accepted, exact=exact))
        assert exact == (abs(shift)<.5 or shift==8.0)
    # Exhaust all byte maps; genuine invertibility, not digest-only success.
    tables = set()
    for indices in itertools.product(range(1, 9), repeat=4):
        t, inv = rec.lane_table(indices), rec.lane_table(indices, True)
        assert np.array_equal(inv[t], np.arange(256))
        assert not np.any(t == np.arange(256))
        tables.add(t.tobytes())
    assert len(tables) == 4096
    return dict(boundary_checks=rows, unique_maps=4096, all_byte_roundtrips=True,
                audited_integrator_bitwise_check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check_integrator()
    structural = structural_tests()
    source = cv2.imread(str(ROOT/'Manuscript/figures/original_image.jpg'))
    if source is None:
        raise FileNotFoundError('Missing source image')
    plain = cv2.resize(source, (128, 128), interpolation=cv2.INTER_AREA)
    base.write_lossless(OUT/'plain.png', plain)
    rows, hashes = [], {}
    drive_by_seed = {}
    for node in [2, 3]:
        for method in ['PPO', 'Fixed', 'LQR', 'SMC']:
            for training_seed in ([42, 123, 1024] if method=='PPO' else [None]):
                model = None
                if training_seed is not None:
                    path = ROOT/f'models/hopfield/ppo_continuous_hopfield_node{node}_final_seed_{training_seed}.zip'
                    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                    model = PPO.load(path, device='cpu')
                for seed in [100, 101, 102]:
                    x, y, effort, hits = rollout(node, method, seed, model)
                    if seed in drive_by_seed:
                        assert np.array_equal(x, drive_by_seed[seed])
                    else:
                        drive_by_seed[seed] = x.copy()
                    row = dict(node=node, method=method, training_seed=training_seed, initial_seed=seed,
                               collection_effort=effort, guard_hits=hits, **trial(x, y, plain))
                    rows.append(row)
                    print(f'{node} {method} {training_seed} {seed}: bound={row["state_propagated_bound"]:.4g}, '
                          f'direct={row["direct_exact"]}, reconciled={row["reconciled_exact"]}', flush=True)
                    np.savez_compressed(OUT/f'trajectory_n{node}_{method}_{training_seed}_{seed}.npz', drive=x, response=y)
    summary=[]
    for node in [2,3]:
        for method in ['PPO','Fixed','LQR','SMC']:
            rr=[r for r in rows if r['node']==node and r['method']==method]
            summary.append(dict(node=node,method=method,trials=len(rr),
                state_mae_mean=float(np.mean([r['state_mae'] for r in rr])),
                max_stream_error=max(r['four_stream_max_error'] for r in rr),
                max_state_propagated_bound=max(r['state_propagated_bound'] for r in rr),
                direct_mismatch_mean=float(np.mean([r['direct_rule_mismatch'] for r in rr])),
                coordinated_mismatch_mean=float(np.mean([r['reconciled_rule_mismatch'] for r in rr])),
                direct_exact=sum(r['direct_exact'] for r in rr),
                coordinated_exact=sum(r['reconciled_exact'] for r in rr),
                condition_pass=sum(r['sufficient_condition_with_margin'] for r in rr),
                fixed_exact=sum(r['fixed_exact_bypassing_digest'] for r in rr),
                guard_hits=sum(r['guard_hits'] for r in rr)))
    result=dict(protocol=dict(nodes=[2,3],ppo_seeds=[42,123,1024],initial_seeds=[100,101,102],
        nominal_only=True,burn_in=1500,samples=256,dt=.05,image_size=[128,128,3],block_size=8,delta=1.0,
        fixed_gain=8,lqr_Q='I',lqr_R=1,smc_reaching_gain=16,smc_boundary_width=.5,
        no_training_or_retuning=True,paired_drive_check='bitwise equal',
        scope='Separate functional prototype, not legacy image statistics or cryptographic security.',
        image_sha256=hashlib.sha256(plain.tobytes()).hexdigest(),model_hashes=hashes),
        structural_tests=structural,summary=summary,rows=rows)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
