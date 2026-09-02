"""Estimate post-training transverse Lyapunov exponents of deployed PPO loops.

The controller is evaluated exactly as in ``ContinuousHopfieldEnv``: one PPO
action is computed every control step and held for five RK4 substeps.  The
Jacobian of that sample-and-hold error map is propagated with QR
renormalization along each nominal closed-loop rollout.

The result is numerical trajectory evidence, not a global formal certificate.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Code.env.continuous_hopfield_env import ContinuousHopfieldEnv
from Code.main.check_hopfield_jacobian_condition import (
    deployed_step_error_jacobian,
    discover_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate conditional Lyapunov exponents of deployed Hopfield PPO loops."
    )
    parser.add_argument("--nodes", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "hopfield",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "jacobian_condition",
    )
    return parser.parse_args()


def collect_rollout(
    model: PPO,
    node: int,
    evaluation_seed: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    env = ContinuousHopfieldEnv(pinning_node=node - 1)
    env.max_steps = steps + 1
    observation, _ = env.reset(seed=evaluation_seed)
    drive_states = []
    errors = []
    actions = []
    error_norms = []

    for _ in range(steps):
        drive_states.append(env.statex.astype(np.float64).copy())
        errors.append((env.statey - env.statex).astype(np.float64).copy())
        action, _ = model.predict(observation, deterministic=True)
        actions.append(float(action[0]) * env.scale)
        observation, _, _, _, _ = env.step(action)
        error_norms.append(float(np.linalg.norm(env.statey - env.statex, ord=2)))

    return (
        np.asarray(drive_states),
        np.asarray(errors),
        np.asarray(actions),
        np.asarray(error_norms),
    )


def qr_lyapunov_exponents(
    jacobians: np.ndarray,
    burn_in: int,
    control_interval: float,
) -> np.ndarray:
    if not 0 <= burn_in < len(jacobians):
        raise ValueError("burn_in must be nonnegative and smaller than steps")
    basis = np.eye(3, dtype=np.float64)
    log_sums = np.zeros(3, dtype=np.float64)
    accumulated_steps = 0

    for step, jacobian in enumerate(jacobians):
        propagated = jacobian @ basis
        basis, upper = np.linalg.qr(propagated)
        diagonal = np.maximum(np.abs(np.diag(upper)), np.finfo(np.float64).tiny)
        if step >= burn_in:
            log_sums += np.log(diagonal)
            accumulated_steps += 1

    exponents = log_sums / (accumulated_steps * control_interval)
    return np.sort(exponents)[::-1]


def main() -> None:
    args = parse_args()
    models = discover_models(args.models_dir, args.nodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for model_path, node, training_seed in models:
        print(f"\nLoading {model_path.name}")
        model = PPO.load(model_path, device="cpu")
        for evaluation_seed in args.evaluation_seeds:
            drive, error, action, error_norm = collect_rollout(
                model=model,
                node=node,
                evaluation_seed=evaluation_seed,
                steps=args.steps,
            )
            env = ContinuousHopfieldEnv(pinning_node=node - 1)
            jacobians = deployed_step_error_jacobian(
                model=model,
                node=node,
                x=drive,
                error=error,
                w=env.W.astype(np.float64),
                action_scale=float(env.scale),
                observation_scale=5.0,
                dt=float(env.dt),
                rk4_steps=int(env.rk4_steps),
            )
            exponents = qr_lyapunov_exponents(
                jacobians=jacobians,
                burn_in=args.burn_in,
                control_interval=float(env.dt * env.rk4_steps),
            )
            post_burn_error = error_norm[args.burn_in :]
            post_burn_action = action[args.burn_in :]
            row = {
                "model": model_path.name,
                "node": node,
                "training_seed": training_seed,
                "evaluation_seed": evaluation_seed,
                "largest_conditional_exponent": float(exponents[0]),
                "middle_conditional_exponent": float(exponents[1]),
                "smallest_conditional_exponent": float(exponents[2]),
                "largest_exponent_negative": bool(exponents[0] < 0.0),
                "post_burn_error_mean": float(np.mean(post_burn_error)),
                "post_burn_error_p95": float(np.quantile(post_burn_error, 0.95)),
                "post_burn_error_max": float(np.max(post_burn_error)),
                "post_burn_action_abs_max": float(np.max(np.abs(post_burn_action))),
                "post_burn_action_abs_mean": float(np.mean(np.abs(post_burn_action))),
            }
            rows.append(row)
            print(
                f"  eval seed {evaluation_seed}: LCE={exponents[0]:.6f}; "
                f"mean error={row['post_burn_error_mean']:.6f}; "
                f"p95 error={row['post_burn_error_p95']:.6f}"
            )

    output_path = args.output_dir / "closed_loop_conditional_lyapunov.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {output_path}")
    print("Numerical trajectory assessment only; not a global formal certificate.")


if __name__ == "__main__":
    main()
