"""Reproducible revision training; preserve all old models and final seeds.

Run ``--all --workers 6`` to fill six SAC and eight missing multi-input PPO
checkpoints. Every model receives 401408 environment transitions. JSON records
and per-worker logs permit progress inspection without selecting best runs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/hopfield/revision_comparisons"
MODELS = ROOT / "models/hopfield/revision_comparisons"
BUDGET = 401408
SEEDS = (42, 123, 1024)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def worker(algorithm, config, seed):
    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    sys.path.insert(0, str(ROOT / "Code"))
    from env.continuous_hopfield_env import ContinuousHopfieldEnv
    from env.continuous_hopfield_multi_input_env import ContinuousHopfieldMultiInputEnv

    torch.set_num_threads(1)
    tag = f"{algorithm}_nodes{config}_seed{seed}"
    status_path = OUT / f"{tag}.json"
    model_path = MODELS / f"{tag}.zip"
    started = time.time()
    nodes = [int(n) for n in config.split("-")]
    base = dict(algorithm=algorithm, nodes=nodes, seed=seed, budget=BUDGET)
    class Progress(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 8192 == 0:
                elapsed = time.time() - started
                write_json(status_path, dict(base, status="training", steps=self.num_timesteps,
                                            elapsed_seconds=elapsed, steps_per_second=self.num_timesteps/elapsed))
                print(tag, self.num_timesteps, round(elapsed), flush=True)
            return True
    if model_path.exists():
        data = json.loads(zipfile.ZipFile(model_path).read("data"))
        if data["num_timesteps"] != BUDGET or data["seed"] != seed:
            raise ValueError(f"Existing checkpoint mismatch: {model_path}")
        print("Already complete", model_path, flush=True)
        return
    write_json(status_path, dict(base, status="starting", steps=0))
    if algorithm == "SAC":
        env = Monitor(ContinuousHopfieldEnv(pinning_node=nodes[0]-1), filename=str(OUT / f"{tag}.monitor.csv"))
        model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100000,
                    learning_starts=5000, batch_size=128, tau=0.005, gamma=0.99,
                    train_freq=1, gradient_steps=1, ent_coef="auto",
                    policy_kwargs={"net_arch": [64, 64]}, seed=seed, device="cpu", verbose=0)
    else:
        def factory(rank):
            def initialize():
                env = ContinuousHopfieldMultiInputEnv(controlled_nodes=nodes)
                env.reset(seed=seed+rank)
                return Monitor(env, filename=str(OUT / f"{tag}_env{rank}.monitor.csv"))
            return initialize
        env = DummyVecEnv([factory(rank) for rank in range(4)])
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512,
                    batch_size=128, gamma=0.99, seed=seed, device="cpu",
                    ent_coef=0.001, verbose=0)
    model.learn(total_timesteps=BUDGET, callback=Progress())
    model.save(model_path)
    assert model.num_timesteps == BUDGET
    env.close()
    write_json(status_path, dict(base, status="complete", steps=model.num_timesteps,
                                elapsed_seconds=time.time()-started,
                                sha256=hashlib.sha256(model_path.read_bytes()).hexdigest()))
    print("Saved", model_path, flush=True)


def launch(job):
    algorithm, config, seed = job
    tag = f"{algorithm}_nodes{config}_seed{seed}"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (OUT / f"{tag}.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run([sys.executable, "-u", __file__, "--algorithm", algorithm,
                                 "--config", config, "--seed", str(seed)],
                                stdout=handle, stderr=subprocess.STDOUT, env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return tag, result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--algorithm", choices=["PPO", "SAC"])
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    if not args.all:
        worker(args.algorithm, args.config, args.seed)
        return
    # Interleave the algorithms so both kinds of evidence progress together.
    sac = [("SAC", str(node), seed) for seed in SEEDS for node in (2, 3)]
    multi = [("PPO", config, seed) for seed in (123, 1024) for config in ("1-2", "1-3", "2-3", "1-2-3")]
    jobs = []
    while sac or multi:
        if sac:
            jobs.append(sac.pop(0))
        if multi:
            jobs.append(multi.pop(0))
    write_json(OUT / "training_plan.json", dict(budget=BUDGET, seeds=SEEDS, jobs=jobs,
        checkpoint_selection="final only; retain every seed and configuration",
        reused="single-input PPO seeds 42,123,1024 and multi-input PPO seed 42"))
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for tag, code in pool.map(launch, jobs):
            print(tag, "exit", code, flush=True)
            if code:
                failures.append(tag)
    if failures:
        raise RuntimeError(f"Failed training jobs: {failures}")
    print("ALL TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
