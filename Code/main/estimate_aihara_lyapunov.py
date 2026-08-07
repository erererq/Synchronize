import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def activation(value: float, epsilon: float) -> float:
    scaled = np.clip(value / epsilon, -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-scaled)))


def aihara_step(state: np.ndarray, k: float, alpha: float, beta: float, bias: float, epsilon: float) -> np.ndarray:
    x, y = state
    fx = activation(float(x), epsilon)
    x_next = k * x - alpha * y + bias
    y_next = y + beta * (fx - y)
    return np.array([x_next, y_next], dtype=np.float64)


def simulate(
    state0: np.ndarray,
    steps: int,
    k: float,
    alpha: float,
    beta: float,
    bias: float,
    epsilon: float,
) -> np.ndarray:
    state = state0.astype(np.float64).copy()
    history = np.zeros((steps, 2), dtype=np.float64)

    for idx in range(steps):
        history[idx] = state
        state = aihara_step(state, k, alpha, beta, bias, epsilon)

    return history


def estimate_lle_series(
    state0: np.ndarray,
    steps: int,
    burn_in: int,
    k: float,
    alpha: float,
    beta: float,
    bias: float,
    epsilon: float,
    delta0: float = 1e-8,
) -> tuple[float, np.ndarray]:
    reference = state0.astype(np.float64).copy()
    perturbed = reference + delta0 * np.array([1.0, 0.0], dtype=np.float64)

    cumulative = []
    log_sum = 0.0
    valid_steps = 0

    for idx in range(steps):
        reference = aihara_step(reference, k, alpha, beta, bias, epsilon)
        perturbed = aihara_step(perturbed, k, alpha, beta, bias, epsilon)

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
    parser = argparse.ArgumentParser(description="Estimate the largest Lyapunov exponent of the Aihara neuron map.")
    parser.add_argument("--k", type=float, default=0.94, help="Internal decay parameter.")
    parser.add_argument("--alpha", type=float, default=0.85, help="Refractory scaling parameter.")
    parser.add_argument("--beta", type=float, default=0.15, help="Slow-variable coupling parameter.")
    parser.add_argument("--bias", type=float, default=0.25, help="Constant external bias.")
    parser.add_argument("--epsilon", type=float, default=0.06, help="Sigmoid steepness parameter.")
    parser.add_argument("--steps", type=int, default=20000, help="Total simulated map iterations.")
    parser.add_argument("--burn-in", type=int, default=2000, help="Transient steps discarded before LLE accumulation.")
    parser.add_argument("--out", type=str, default="logs/aihara/lyapunov_check", help="Output folder.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    state0 = np.array([0.1, 0.1], dtype=np.float64)
    traj = simulate(state0, args.steps, args.k, args.alpha, args.beta, args.bias, args.epsilon)
    traj_ss = traj[args.burn_in :]

    state1 = state0.copy()
    state2 = state0 + np.array([1e-6, 0.0], dtype=np.float64)
    traj1 = simulate(state1, args.steps, args.k, args.alpha, args.beta, args.bias, args.epsilon)
    traj2 = simulate(state2, args.steps, args.k, args.alpha, args.beta, args.bias, args.epsilon)
    dist = np.linalg.norm(traj1 - traj2, axis=1)
    dist_ss = dist[args.burn_in :]

    lle, lle_series = estimate_lle_series(
        state0,
        args.steps,
        args.burn_in,
        args.k,
        args.alpha,
        args.beta,
        args.bias,
        args.epsilon,
    )

    t = np.arange(len(traj_ss))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(t, traj_ss[:, 0], lw=0.6, color="tab:blue")
    axes[0, 0].set_title("x(t)")

    axes[0, 1].plot(t, traj_ss[:, 1], lw=0.6, color="tab:orange")
    axes[0, 1].set_title("y(t)")

    axes[1, 0].plot(traj_ss[:, 0], traj_ss[:, 1], lw=0.5, color="black")
    axes[1, 0].set_title("Phase Plane")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")

    axes[1, 1].plot(np.arange(len(lle_series)), lle_series, lw=0.8, color="tab:purple")
    axes[1, 1].axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[1, 1].set_title("Running Largest Lyapunov Exponent")
    axes[1, 1].set_xlabel("post burn-in step")

    fig.tight_layout()
    fig.savefig(out_dir / "aihara_lyapunov_summary.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(dist_ss, color="tab:red", lw=0.6)
    axes[0].set_ylabel("distance")
    axes[0].set_title("Sensitivity to Initial Conditions")
    axes[1].plot(np.log(dist_ss + 1e-12), color="tab:purple", lw=0.6)
    axes[1].set_ylabel("log(distance)")
    axes[1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / "aihara_lyapunov_distance.png", dpi=150)
    plt.close(fig)

    result_text = (
        f"k={args.k:.6f}\n"
        f"alpha={args.alpha:.6f}\n"
        f"beta={args.beta:.6f}\n"
        f"bias={args.bias:.6f}\n"
        f"epsilon={args.epsilon:.6f}\n"
        f"steps={args.steps}\n"
        f"burn_in={args.burn_in}\n"
        f"largest_lyapunov_exponent={lle:.8f}\n"
    )
    (out_dir / "result.txt").write_text(result_text, encoding="utf-8")

    print(result_text.strip())
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
