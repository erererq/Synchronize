"""Post-training sampled Jacobian assessment for the 3-D Hopfield PPO policies.

This script evaluates the sufficient Euclidean contraction condition

    lambda_max((J + J.T) / 2) < 0,

where J is the Jacobian of the *deployed* continuous-time closed-loop error
field with respect to the synchronization error.  The assessment includes the
environment's observation normalization, Stable-Baselines3 action clipping,
and the physical action scale.

The result is a sampled numerical assessment over a drive-attractor tube.  It
is not interval verification and therefore must not be described as a formal
certificate over the continuous tube.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Code.env.continuous_hopfield_env import ContinuousHopfieldEnv


MODEL_RE = re.compile(r"node(?P<node>\d+)_final_seed_(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample the closed-loop Jacobian contraction condition."
    )
    parser.add_argument("--nodes", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--radii", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1])
    parser.add_argument("--attractor-samples", type=int, default=500)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--transient-control-steps", type=int, default=1200)
    parser.add_argument("--sample-stride-control-steps", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=20260822)
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


def rk4_drive_step(state: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
    def derivative(z: np.ndarray) -> np.ndarray:
        return -z + w @ np.tanh(z)

    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def generate_attractor_samples(
    w: np.ndarray,
    dt: float,
    rk4_steps: int,
    transient_control_steps: int,
    sample_stride_control_steps: int,
    sample_count: int,
) -> np.ndarray:
    state = np.array([0.0, 0.01, 0.0], dtype=np.float64)

    def advance_control_step(z: np.ndarray) -> np.ndarray:
        for _ in range(rk4_steps):
            z = rk4_drive_step(z, w, dt)
        return z

    for _ in range(transient_control_steps):
        state = advance_control_step(state)

    samples = []
    for _ in range(sample_count):
        for _ in range(sample_stride_control_steps):
            state = advance_control_step(state)
        samples.append(state.copy())
    return np.asarray(samples, dtype=np.float64)


def discover_models(models_dir: Path, nodes: list[int]) -> list[tuple[Path, int, int]]:
    selected = []
    for path in sorted(models_dir.glob("ppo_continuous_hopfield_node*_final_seed_*.zip")):
        match = MODEL_RE.search(path.stem)
        if not match:
            continue
        node = int(match.group("node"))
        seed = int(match.group("seed"))
        if node in nodes:
            selected.append((path, node, seed))
    if not selected:
        raise FileNotFoundError(f"No matching full-observation PPO models found in {models_dir}")
    return selected


def sphere_directions(sample_count: int, direction_count: int, rng: np.random.Generator) -> np.ndarray:
    directions = rng.normal(size=(sample_count, direction_count, 3))
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    return directions / np.maximum(norms, np.finfo(np.float64).eps)


def policy_action_and_error_gradient(
    model: PPO,
    x: np.ndarray,
    error: np.ndarray,
    action_scale: float,
    observation_scale: float,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = []
    raw_actions = []
    gradients = []
    device = model.device
    model.policy.set_training_mode(False)

    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        x_tensor = torch.as_tensor(x[start:stop], dtype=torch.float32, device=device)
        error_tensor = torch.as_tensor(
            error[start:stop], dtype=torch.float32, device=device
        ).clone().detach().requires_grad_(True)
        observation = torch.cat((x_tensor, error_tensor), dim=1) / observation_scale

        raw_action = model.policy._predict(observation, deterministic=True)
        deployed_action = torch.clamp(raw_action, -1.0, 1.0) * action_scale
        gradient = torch.autograd.grad(deployed_action.sum(), error_tensor)[0]

        actions.append(deployed_action.detach().cpu().numpy().reshape(-1))
        raw_actions.append(raw_action.detach().cpu().numpy().reshape(-1))
        gradients.append(gradient.detach().cpu().numpy())

    return (
        np.concatenate(actions),
        np.concatenate(raw_actions),
        np.concatenate(gradients),
    )


def deployed_step_error_jacobian(
    model: PPO,
    node: int,
    x: np.ndarray,
    error: np.ndarray,
    w: np.ndarray,
    action_scale: float,
    observation_scale: float,
    dt: float,
    rk4_steps: int,
    batch_size: int = 1024,
) -> np.ndarray:
    """Differentiate the actual sample-and-hold RK4 control-step map."""
    jacobians = []
    device = model.device
    model.policy.set_training_mode(False)
    w_tensor = torch.as_tensor(w, dtype=torch.float32, device=device)
    pinning_vector = torch.zeros(3, dtype=torch.float32, device=device)
    pinning_vector[node - 1] = 1.0

    def rk4_step(
        state: torch.Tensor,
        control: torch.Tensor | None,
    ) -> torch.Tensor:
        def derivative(z: torch.Tensor) -> torch.Tensor:
            value = -z + torch.tanh(z) @ w_tensor.T
            if control is not None:
                value = value + control[:, None] * pinning_vector[None, :]
            return value

        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt * k1)
        k3 = derivative(state + 0.5 * dt * k2)
        k4 = derivative(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        x_tensor = torch.as_tensor(x[start:stop], dtype=torch.float32, device=device)
        error_tensor = torch.as_tensor(
            error[start:stop], dtype=torch.float32, device=device
        ).clone().detach().requires_grad_(True)
        observation = torch.cat((x_tensor, error_tensor), dim=1) / observation_scale
        raw_action = model.policy._predict(observation, deterministic=True).reshape(-1)
        control = torch.clamp(raw_action, -1.0, 1.0) * action_scale

        drive_state = x_tensor
        response_state = x_tensor + error_tensor
        for _ in range(rk4_steps):
            drive_state = rk4_step(drive_state, None)
            response_state = rk4_step(response_state, control)
        next_error = response_state - drive_state

        output_gradients = []
        for output_idx in range(3):
            gradient = torch.autograd.grad(
                next_error[:, output_idx].sum(),
                error_tensor,
                retain_graph=output_idx < 2,
            )[0]
            output_gradients.append(gradient.detach().cpu().numpy())
        jacobians.append(np.stack(output_gradients, axis=1))

    return np.concatenate(jacobians, axis=0)


def assess_points(
    model: PPO,
    node: int,
    x: np.ndarray,
    error: np.ndarray,
    w: np.ndarray,
    action_scale: float,
    observation_scale: float,
    dt: float,
    rk4_steps: int,
) -> dict[str, float | int | bool]:
    action, raw_action, du_de = policy_action_and_error_gradient(
        model=model,
        x=x,
        error=error,
        action_scale=action_scale,
        observation_scale=observation_scale,
    )

    response_state = x + error
    sech_squared = 1.0 / np.cosh(response_state) ** 2
    natural_jacobian = -np.eye(3)[None, :, :] + w[None, :, :] * sech_squared[:, None, :]
    pinning_vector = np.zeros(3, dtype=np.float64)
    pinning_vector[node - 1] = 1.0
    control_jacobian = pinning_vector[None, :, None] * du_de[:, None, :]
    jacobian = natural_jacobian + control_jacobian
    symmetric_jacobian = 0.5 * (jacobian + np.swapaxes(jacobian, 1, 2))
    lambda_max = np.linalg.eigvalsh(symmetric_jacobian)[:, -1]
    continuous_spectral_abscissa = np.max(np.linalg.eigvals(jacobian).real, axis=1)

    step_jacobian = deployed_step_error_jacobian(
        model=model,
        node=node,
        x=x,
        error=error,
        w=w,
        action_scale=action_scale,
        observation_scale=observation_scale,
        dt=dt,
        rk4_steps=rk4_steps,
    )
    step_sigma_max = np.linalg.svd(step_jacobian, compute_uv=False)[:, 0]
    step_spectral_radius = np.max(np.abs(np.linalg.eigvals(step_jacobian)), axis=1)

    max_lambda = float(np.max(lambda_max))
    return {
        "sample_count": int(len(lambda_max)),
        "lambda_max_worst": max_lambda,
        "lambda_max_p99": float(np.quantile(lambda_max, 0.99)),
        "lambda_max_p95": float(np.quantile(lambda_max, 0.95)),
        "lambda_max_median": float(np.median(lambda_max)),
        "fraction_strictly_negative": float(np.mean(lambda_max < 0.0)),
        "sampled_condition_passed": bool(max_lambda < 0.0),
        "sampled_alpha": float(-max_lambda) if max_lambda < 0.0 else float("nan"),
        "continuous_spectral_abscissa_worst": float(np.max(continuous_spectral_abscissa)),
        "fraction_continuous_jacobian_hurwitz": float(
            np.mean(continuous_spectral_abscissa < 0.0)
        ),
        "step_sigma_max_worst": float(np.max(step_sigma_max)),
        "step_sigma_max_p99": float(np.quantile(step_sigma_max, 0.99)),
        "fraction_step_euclidean_contractive": float(np.mean(step_sigma_max < 1.0)),
        "step_spectral_radius_worst": float(np.max(step_spectral_radius)),
        "fraction_step_locally_schur": float(np.mean(step_spectral_radius < 1.0)),
        "action_abs_max": float(np.max(np.abs(action))),
        "action_abs_p95": float(np.quantile(np.abs(action), 0.95)),
        "action_abs_median": float(np.median(np.abs(action))),
        "raw_action_abs_max": float(np.max(np.abs(raw_action))),
        "fraction_action_saturated": float(np.mean(np.abs(raw_action) >= 1.0)),
        "du_de_norm_max": float(np.max(np.linalg.norm(du_de, axis=1))),
        "du_de_norm_p95": float(np.quantile(np.linalg.norm(du_de, axis=1), 0.95)),
    }


def finite_difference_check(
    model: PPO,
    x: np.ndarray,
    error: np.ndarray,
    action_scale: float,
    observation_scale: float,
    epsilon: float = 1e-4,
) -> float:
    _, _, analytical = policy_action_and_error_gradient(
        model, x[None, :], error[None, :], action_scale, observation_scale
    )
    numerical = np.zeros(3, dtype=np.float64)
    for idx in range(3):
        delta = np.zeros(3, dtype=np.float64)
        delta[idx] = epsilon
        plus, _, _ = policy_action_and_error_gradient(
            model, x[None, :], (error + delta)[None, :], action_scale, observation_scale
        )
        minus, _, _ = policy_action_and_error_gradient(
            model, x[None, :], (error - delta)[None, :], action_scale, observation_scale
        )
        numerical[idx] = (plus[0] - minus[0]) / (2.0 * epsilon)
    return float(np.max(np.abs(analytical[0] - numerical)))


def main() -> None:
    args = parse_args()
    invalid_radii = [radius for radius in args.radii if radius < 0.0]
    if invalid_radii:
        raise ValueError(f"Radii must be nonnegative: {invalid_radii}")

    env = ContinuousHopfieldEnv()
    w = env.W.astype(np.float64)
    attractor = generate_attractor_samples(
        w=w,
        dt=float(env.dt),
        rk4_steps=int(env.rk4_steps),
        transient_control_steps=args.transient_control_steps,
        sample_stride_control_steps=args.sample_stride_control_steps,
        sample_count=args.attractor_samples,
    )
    rng = np.random.default_rng(args.random_seed)
    directions = sphere_directions(args.attractor_samples, args.directions, rng)
    models = discover_models(args.models_dir, args.nodes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    finite_difference_results: list[dict[str, object]] = []

    print(f"Attractor samples: {len(attractor)}")
    print(f"Attractor coordinate min: {np.min(attractor, axis=0)}")
    print(f"Attractor coordinate max: {np.max(attractor, axis=0)}")
    print(f"Models: {len(models)}")

    for model_path, node, training_seed in models:
        print(f"\nLoading {model_path.name}")
        model = PPO.load(model_path, device="cpu")
        if tuple(model.observation_space.shape) != (6,):
            raise ValueError(
                f"Expected a six-dimensional full-observation policy, got "
                f"{model.observation_space} for {model_path.name}"
            )
        if tuple(model.action_space.shape) != (1,):
            raise ValueError(f"Expected one scalar action for {model_path.name}")

        fd_error = finite_difference_check(
            model=model,
            x=attractor[len(attractor) // 2],
            error=np.zeros(3, dtype=np.float64),
            action_scale=float(env.scale),
            observation_scale=5.0,
        )
        finite_difference_results.append(
            {
                "model": model_path.name,
                "node": node,
                "training_seed": training_seed,
                "maximum_absolute_gradient_difference": fd_error,
            }
        )
        print(f"  finite-difference gradient max abs difference: {fd_error:.6g}")

        for radius in args.radii:
            if radius == 0.0:
                x_points = attractor
                error_points = np.zeros_like(attractor)
            else:
                x_points = np.repeat(attractor, args.directions, axis=0)
                error_points = (radius * directions).reshape(-1, 3)

            result = assess_points(
                model=model,
                node=node,
                x=x_points,
                error=error_points,
                w=w,
                action_scale=float(env.scale),
                observation_scale=5.0,
                dt=float(env.dt),
                rk4_steps=int(env.rk4_steps),
            )
            row = {
                "model": model_path.name,
                "node": node,
                "training_seed": training_seed,
                "tube_radius": radius,
                **result,
            }
            rows.append(row)
            verdict = "PASS" if result["sampled_condition_passed"] else "FAIL"
            print(
                f"  r={radius:g}: {verdict}; worst lambda_sym="
                f"{result['lambda_max_worst']:.6f}; negative fraction="
                f"{result['fraction_strictly_negative']:.3f}; "
                f"worst step sigma="
                f"{result['step_sigma_max_worst']:.6f}; "
                f"worst step rho={result['step_spectral_radius_worst']:.6f}; "
                f"max |u|={result['action_abs_max']:.6f}; "
                f"saturation={result['fraction_action_saturated']:.3f}"
            )

    csv_path = args.output_dir / "jacobian_condition_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "interpretation": (
            "Sampled numerical assessment only; sampled pass does not prove the "
            "condition on the full continuous attractor tube."
        ),
        "formula": "lambda_max((J + J.T)/2) < 0",
        "project_root": str(PROJECT_ROOT),
        "models_dir": str(args.models_dir.resolve()),
        "attractor_initial_state": [0.0, 0.01, 0.0],
        "attractor_samples": args.attractor_samples,
        "directions_per_state": args.directions,
        "transient_control_steps": args.transient_control_steps,
        "sample_stride_control_steps": args.sample_stride_control_steps,
        "radii": args.radii,
        "random_seed": args.random_seed,
        "state_weight_matrix": w.tolist(),
        "action_scale": float(env.scale),
        "observation_scale": 5.0,
        "attractor_coordinate_min": np.min(attractor, axis=0).tolist(),
        "attractor_coordinate_max": np.max(attractor, axis=0).tolist(),
        "finite_difference_checks": finite_difference_results,
    }
    metadata_path = args.output_dir / "jacobian_condition_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
