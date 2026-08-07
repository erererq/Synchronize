import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STATE_CLIP = 6.0
DELTA0 = 1e-8


def activation(state: np.ndarray) -> np.ndarray:
    return np.tanh(state).astype(np.float64)


def hopfield_step(state: np.ndarray, decay: float, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    next_state = decay * state
    next_state += weights @ activation(state)
    next_state += bias
    return np.clip(next_state, -STATE_CLIP, STATE_CLIP).astype(np.float64)


def simulate(state0: np.ndarray, steps: int, decay: float, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    state = state0.astype(np.float64).copy()
    history = np.zeros((steps, 2), dtype=np.float64)

    for idx in range(steps):
        history[idx] = state
        state = hopfield_step(state, decay, weights, bias)

    return history


def estimate_lle_series(
    state0: np.ndarray,
    steps: int,
    burn_in: int,
    decay: float,
    weights: np.ndarray,
    bias: np.ndarray,
    delta0: float = DELTA0,
) -> tuple[float, np.ndarray]:
    reference = state0.astype(np.float64).copy()
    perturbed = reference + delta0 * np.array([1.0, 0.0], dtype=np.float64)

    cumulative = []
    log_sum = 0.0
    valid_steps = 0

    for idx in range(steps):
        reference = hopfield_step(reference, decay, weights, bias)
        perturbed = hopfield_step(perturbed, decay, weights, bias)

        diff = perturbed - reference
        stretch = float(np.linalg.norm(diff))
        if stretch < 1e-16:
            diff = np.array([1.0, 0.0], dtype=np.float64)
            stretch = 1e-16

        if idx >= burn_in:
            log_sum += np.log(stretch / delta0)
            valid_steps += 1
            cumulative.append(log_sum / valid_steps)

        perturbed = reference + delta0 * diff / stretch

    lle = float(log_sum / max(valid_steps, 1))
    return lle, np.asarray(cumulative, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the largest Lyapunov exponent of the discrete Hopfield drive system.")
    parser.add_argument("--decay", type=float, required=True, help="Drive-system decay parameter.")
    parser.add_argument("--w11", type=float, required=True, help="Weight matrix entry (1,1).")
    parser.add_argument("--w12", type=float, required=True, help="Weight matrix entry (1,2).")
    parser.add_argument("--w21", type=float, required=True, help="Weight matrix entry (2,1).")
    parser.add_argument("--w22", type=float, required=True, help="Weight matrix entry (2,2).")
    parser.add_argument("--b1", type=float, required=True, help="Bias entry 1.")
    parser.add_argument("--b2", type=float, required=True, help="Bias entry 2.")
    parser.add_argument("--steps", type=int, default=100000, help="Total simulated steps.")
    parser.add_argument("--burn-in", type=int, default=10000, help="Transient steps discarded before LLE accumulation.")
    parser.add_argument("--out", type=str, default="logs/hopfield/lyapunov_check", help="Output folder.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = np.array([[args.w11, args.w12], [args.w21, args.w22]], dtype=np.float64)
    bias = np.array([args.b1, args.b2], dtype=np.float64)
    state0 = np.array([0.2, -0.1], dtype=np.float64)

    traj = simulate(state0, args.steps, args.decay, weights, bias)
    traj_ss = traj[args.burn_in :]

    near_state = state0 + np.array([1e-6, 0.0], dtype=np.float64)
    traj_a = simulate(state0, args.steps, args.decay, weights, bias)
    traj_b = simulate(near_state, args.steps, args.decay, weights, bias)
    dist = np.linalg.norm(traj_a - traj_b, axis=1)
    dist_ss = dist[args.burn_in :]

    lle, lle_series = estimate_lle_series(state0, args.steps, args.burn_in, args.decay, weights, bias)

    t = np.arange(len(traj_ss))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(t, traj_ss[:, 0], lw=0.6, color="tab:blue")
    axes[0, 0].set_title("x1(n)")

    axes[0, 1].plot(t, traj_ss[:, 1], lw=0.6, color="tab:orange")
    axes[0, 1].set_title("x2(n)")

    axes[1, 0].plot(traj_ss[:, 0], traj_ss[:, 1], lw=0.5, color="black")
    axes[1, 0].set_title("Phase Portrait")
    axes[1, 0].set_xlabel("x1")
    axes[1, 0].set_ylabel("x2")

    axes[1, 1].plot(np.arange(len(lle_series)), lle_series, lw=0.8, color="tab:green")
    axes[1, 1].axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[1, 1].set_title("Running Largest Lyapunov Exponent")
    axes[1, 1].set_xlabel("post burn-in step")

    fig.tight_layout()
    fig.savefig(out_dir / "hopfield_lyapunov_summary.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(dist_ss, color="tab:red", lw=0.6)
    axes[0].set_ylabel("distance")
    axes[0].set_title("Sensitivity to Initial Conditions")
    axes[1].plot(np.log(dist_ss + 1e-12), color="tab:purple", lw=0.6)
    axes[1].set_ylabel("log(distance)")
    axes[1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / "hopfield_lyapunov_distance.png", dpi=150)
    plt.close(fig)

    result_text = (
        f"decay={args.decay:.6f}\n"
        f"weights={weights.tolist()}\n"
        f"bias={bias.tolist()}\n"
        f"steps={args.steps}\n"
        f"burn_in={args.burn_in}\n"
        f"largest_lyapunov_exponent={lle:.8f}\n"
        f"x1_range=[{traj_ss[:, 0].min():.6f}, {traj_ss[:, 0].max():.6f}]\n"
        f"x2_range=[{traj_ss[:, 1].min():.6f}, {traj_ss[:, 1].max():.6f}]\n"
        f"final_separation={dist[-1]:.8e}\n"
    )
    (out_dir / "result.txt").write_text(result_text, encoding="utf-8")

    print(result_text.strip())
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
