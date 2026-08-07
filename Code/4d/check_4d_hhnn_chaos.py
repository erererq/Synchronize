from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent

DT = 0.001
RK4_STEPS = 10
DELTA0 = 1e-8
STATE_CLIP = 20.0


def build_system(w22: float, w33: float, w44: float) -> tuple[np.ndarray, np.ndarray]:
    decay = np.array([1.0, 1.0, 1.0, 100.0], dtype=np.float64)
    weights = np.array(
        [
            [1.0, 0.5, -3.0, -1.0],
            [0.0, w22, 3.0, 0.0],
            [3.0, -3.0, w33, 0.0],
            [100.0, 0.0, 0.0, w44],
        ],
        dtype=np.float64,
    )
    return decay, weights


def derivatives(state: np.ndarray, decay: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return -decay * state + weights @ np.tanh(state)


def rk4_micro_step(state: np.ndarray, decay: np.ndarray, weights: np.ndarray) -> np.ndarray:
    k1 = derivatives(state, decay, weights)
    k2 = derivatives(state + 0.5 * DT * k1, decay, weights)
    k3 = derivatives(state + 0.5 * DT * k2, decay, weights)
    k4 = derivatives(state + DT * k3, decay, weights)
    next_state = state + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.clip(next_state, -STATE_CLIP, STATE_CLIP)


def rk4_step(state: np.ndarray, decay: np.ndarray, weights: np.ndarray) -> np.ndarray:
    next_state = state.astype(np.float64).copy()
    for _ in range(RK4_STEPS):
        next_state = rk4_micro_step(next_state, decay, weights)
    return next_state


def simulate(
    state0: np.ndarray,
    steps: int,
    decay: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    state = state0.astype(np.float64).copy()
    history = np.zeros((steps + 1, state0.size), dtype=np.float64)
    history[0] = state

    for step in range(1, steps + 1):
        state = rk4_step(state, decay, weights)
        history[step] = state

    return history


def estimate_lle(
    state0: np.ndarray,
    steps: int,
    burn_in: int,
    decay: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    reference = state0.astype(np.float64).copy()
    perturbed = reference + DELTA0 * np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    log_sum = 0.0
    valid_steps = 0
    running_lle = []
    step_time = DT * RK4_STEPS

    for step in range(steps):
        reference = rk4_step(reference, decay, weights)
        perturbed = rk4_step(perturbed, decay, weights)

        diff = perturbed - reference
        distance = float(np.linalg.norm(diff))
        if distance < 1e-16:
            diff = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            distance = 1e-16

        if step >= burn_in:
            log_sum += np.log(distance / DELTA0)
            valid_steps += 1
            running_lle.append(log_sum / (valid_steps * step_time))

        perturbed = reference + DELTA0 * diff / distance

    lle = log_sum / max(valid_steps, 1) / step_time
    return float(lle), np.asarray(running_lle, dtype=np.float64)


def plot_time_series(traj: np.ndarray, out_path: Path) -> None:
    t = np.arange(traj.shape[0]) * DT * RK4_STEPS
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 7.0), sharex=True, constrained_layout=True)
    for idx, ax in enumerate(axes):
        ax.plot(t, traj[:, idx], lw=0.7)
        ax.set_ylabel(f"x{idx + 1}")
        ax.grid(True, linestyle=":", alpha=0.35)
    axes[-1].set_xlabel("Time")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_phase_portraits(traj: np.ndarray, out_path: Path) -> None:
    pairs = [(0, 1), (0, 2), (0, 3), (1, 3)]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0), constrained_layout=True)
    for ax, (i, j) in zip(axes.ravel(), pairs):
        ax.plot(traj[:, i], traj[:, j], lw=0.35, color="black")
        ax.set_xlabel(f"x{i + 1}")
        ax.set_ylabel(f"x{j + 1}")
        ax.grid(True, linestyle=":", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_sensitivity(distances: np.ndarray, out_path: Path) -> None:
    t = np.arange(distances.size) * DT * RK4_STEPS
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    ax.plot(t, np.log(distances + 1e-16), lw=0.8, color="tab:purple")
    ax.set_xlabel("Time")
    ax.set_ylabel("log distance")
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_running_lle(running_lle: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    ax.plot(running_lle, lw=0.8, color="tab:green")
    ax.axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Post burn-in step")
    ax.set_ylabel("Running LLE")
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Uncontrolled chaos check for the 4D HHNN model.")
    parser.add_argument("--w22", type=float, default=2.0)
    parser.add_argument("--w33", type=float, default=1.0)
    parser.add_argument("--w44", type=float, default=170.0)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--burn-in", type=int, default=10000)
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "logs" / "4d" / "chaos_check"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    decay, weights = build_system(args.w22, args.w33, args.w44)
    state0 = np.array([0.01, -0.02, 0.0, 0.0], dtype=np.float64)
    near_state0 = state0 + np.array([1e-6, 0.0, 0.0, 0.0], dtype=np.float64)

    traj = simulate(state0, args.steps, decay, weights)
    near_traj = simulate(near_state0, args.steps, decay, weights)
    distances = np.linalg.norm(near_traj - traj, axis=1)
    lle, running_lle = estimate_lle(state0, args.steps, args.burn_in, decay, weights)

    traj_after_burn = traj[args.burn_in :]
    distance_after_burn = distances[args.burn_in :]
    verdict = "chaotic candidate" if lle > 0.0 else "not clearly chaotic"

    result_lines = [
        "4D HHNN uncontrolled chaos check",
        f"parameters: w22={args.w22}, w33={args.w33}, w44={args.w44}",
        f"dt={DT}, rk4_steps={RK4_STEPS}, steps={args.steps}, burn_in={args.burn_in}",
        f"largest Lyapunov exponent estimate: {lle:+.6f}",
        f"initial distance after burn-in: {distance_after_burn[0]:.6e}",
        f"final distance after burn-in: {distance_after_burn[-1]:.6e}",
        f"verdict: {verdict}",
    ]
    result_text = "\n".join(result_lines) + "\n"
    (out_dir / "result.txt").write_text(result_text, encoding="utf-8")
    print(result_text)

    if not args.no_plot:
        plot_time_series(traj_after_burn, out_dir / "time_series.png")
        plot_phase_portraits(traj_after_burn, out_dir / "phase_portraits.png")
        plot_sensitivity(distance_after_burn, out_dir / "initial_sensitivity.png")
        plot_running_lle(running_lle, out_dir / "running_lle.png")
        print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
