"""Evaluate every available revision checkpoint; cached independent job outputs.

Nominal actuator ablation: all seven nonempty subsets, 3 seeds, 20 test ICs.
SAC comparison: nominal, eta=2, impulse, sigma=.05; matched PPO checkpoints.
Traditional audit: repeat the complete previously reported protocol, preserving
original results. No test-based checkpoint selection or parameter adaptation.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/hopfield/revision_comparisons"
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def save_csv(rows, path):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def model_path(method, config, seed):
    if method == "PPO":
        if "-" not in config:
            return ROOT / f"models/hopfield/ppo_continuous_hopfield_node{config}_final_seed_{seed}.zip"
        if seed == 42:
            return ROOT / f"models/hopfield/multi_input_exploration/ppo_continuous_hopfield_nodes{config}_final_seed_42.zip"
    return ROOT / f"models/hopfield/revision_comparisons/{method}_nodes{config}_seed{seed}.zip"


def run_job(method, config, seed):
    import numpy as np
    import torch
    from stable_baselines3 import PPO, SAC
    from env.continuous_hopfield_env import ContinuousHopfieldEnv
    from env.continuous_hopfield_multi_input_env import ContinuousHopfieldMultiInputEnv
    from test_hopfield_smc_baseline import lqr_gain, sliding_surface, smc_force
    from hopfield_revision_metrics import trajectory_metrics
    torch.set_num_threads(1)
    nodes = [int(n) for n in config.split("-")]
    learned = method in ("PPO", "SAC")
    path = model_path(method, config, seed) if learned else None
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path else "deterministic"
    result_path = OUT / "evaluation" / f"{method}_nodes{config}_seed{seed}.csv"
    if result_path.exists():
        old = list(csv.DictReader(result_path.open(encoding="utf-8-sig")))
        assert old and all(row["model_sha256"] == digest for row in old)
        print("Cached", result_path.name, flush=True)
        return
    model = None
    if learned:
        data = json.loads(zipfile.ZipFile(path).read("data"))
        assert data["num_timesteps"] == 401408 and data["seed"] == seed, path
        model = (PPO if method == "PPO" else SAC).load(path, device="cpu")
        assert model.observation_space.shape == (6,) and model.action_space.shape == (len(nodes),)
    conditions = [("nominal", 0., None, 0.)]
    if len(nodes) == 1 and nodes[0] in (2, 3):
        if method == "SAC":
            conditions += [("mismatch", 2., None, 0.), ("pulse", 0., 200, 0.), ("noise", 0., None, .05)]
        else:
            conditions += [("mismatch", eta, None, 0.) for eta in (.5, 1., 1.5, 2.)]
            conditions += [("pulse", 0., 200, 0.)]
            conditions += [("noise", 0., None, sigma) for sigma in (.01, .02, .05)]
    rows = []
    for scenario, mismatch, pulse, noise in conditions:
        for test_seed in range(100, 120):
            if len(nodes) == 1:
                env = ContinuousHopfieldEnv(pinning_node=nodes[0]-1, mismatch_scale=mismatch,
                    pulse_step=pulse, pulse_vector=(5., 5., 5.) if pulse else (0., 0., 0.), process_noise_std=noise)
            else:
                env = ContinuousHopfieldMultiInputEnv(controlled_nodes=nodes)
            obs, _ = env.reset(seed=test_seed)
            initial_x, initial_y = env.statex.copy(), env.statey.copy()
            # A clipped coordinate equals exactly the guard boundary. Monitor
            # each RK4 substep as well as post-noise / post-impulse updates.
            guard_hits = [0]
            original_rk4 = env._rk4_step
            def monitored_rk4(*args, **kwargs):
                state = original_rk4(*args, **kwargs)
                guard_hits[0] += int(np.count_nonzero(np.abs(state) >= 10.))
                return state
            env._rk4_step = monitored_rk4
            errors = [(env.statey-env.statex).copy()]
            controls = [np.zeros(len(nodes))]
            gain = lqr_gain(nodes[0]) if method == "LQR" else None
            surface = sliding_surface(nodes[0]) if method == "SMC" else None
            for _ in range(400):
                if learned:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    error = (env.statey-env.statex).astype(np.float64)
                    if method == "Linear":
                        force = -8.*error[nodes[0]-1]
                    elif method == "LQR":
                        force = -float(gain@error)
                    else:
                        force = smc_force(env, surface, 16., .5)
                    action = np.array([force/4.], dtype=np.float32)
                action = np.clip(np.asarray(action, dtype=np.float32), -1., 1.)
                obs, _, terminated, truncated, _ = env.step(action)
                guard_hits[0] += int(np.count_nonzero(np.abs(env.statex) >= 10.) + np.count_nonzero(np.abs(env.statey) >= 10.))
                controls.append(action*4.)
                errors.append((env.statey-env.statex).copy())
            assert truncated and not terminated
            metrics = trajectory_metrics(errors, controls, pulse_step=pulse)
            rows.append(dict(method=method, controlled_nodes=config, num_control_inputs=len(nodes),
                node=nodes[0] if len(nodes)==1 else -1, controller_seed=seed, test_seed=test_seed,
                scenario=scenario, mismatch=mismatch, noise_std=noise, pulse_step=pulse or -1,
                observation_dim=6 if learned or method=="SMC" else 3 if method=="LQR" else 1,
                action_limit=4., state_guard_hits=guard_hits[0], model_sha256=digest,
                initial_x=json.dumps(initial_x.tolist()), initial_y=json.dumps(initial_y.tolist()), **metrics))
            env.close()
        print(method, config, seed, scenario, mismatch, noise, flush=True)
    save_csv(rows, result_path)


def launch(job):
    method, config, seed = job
    log = OUT / "evaluation" / f"{method}_nodes{config}_seed{seed}.log"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with log.open("w", encoding="utf-8") as f:
        result = subprocess.run([sys.executable, "-u", __file__, "--method", method,
            "--config", config, "--seed", str(seed)], stdout=f, stderr=subprocess.STDOUT, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"Evaluation failed: {log}")
    return log.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--available", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--method")
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    (OUT / "evaluation").mkdir(parents=True, exist_ok=True)
    if not args.available and not args.require_complete:
        run_job(args.method, args.config, args.seed)
        return
    configs = ("1", "2", "3", "1-2", "1-3", "2-3", "1-2-3")
    planned = [("PPO", c, s) for s in (42, 123, 1024) for c in configs]
    planned += [("SAC", c, s) for s in (42, 123, 1024) for c in ("2", "3")]
    missing = [job for job in planned if not model_path(*job).exists()]
    if args.require_complete and missing:
        raise FileNotFoundError(f"Incomplete training: {missing}")
    jobs = [job for job in planned if job not in missing]
    jobs += [(m, c, -1) for m in ("Linear", "LQR", "SMC") for c in ("2", "3")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for completed in pool.map(launch, jobs):
            print(completed, flush=True)
    print("Missing checkpoints:", missing, flush=True)


if __name__ == "__main__":
    main()
