import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_K = 0.94
DEFAULT_ALPHA = 0.85
DEFAULT_BETA = 0.15
DEFAULT_BIAS = 0.25
DEFAULT_EPSILON = 0.06


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


def estimate_lle(
    state0: np.ndarray,
    steps: int,
    burn_in: int,
    k: float,
    alpha: float,
    beta: float,
    bias: float,
    epsilon: float,
    delta0: float = 1e-8,
) -> float:
    reference = state0.astype(np.float64).copy()
    direction = np.array([1.0, 0.0], dtype=np.float64)
    perturbed = reference + delta0 * direction
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

        perturbed = reference + delta0 * diff / stretch

    if valid_steps == 0:
        return float("-inf")

    return float(log_sum / valid_steps)


def estimate_peak_std(signal: np.ndarray) -> float:
    peaks = []
    for idx in range(1, len(signal) - 1):
        if signal[idx] > signal[idx - 1] and signal[idx] > signal[idx + 1]:
            peaks.append(signal[idx])

    if len(peaks) < 5:
        return 0.0

    return float(np.std(np.asarray(peaks, dtype=np.float64)))


def save_summary_plot(case: dict, out_dir: Path) -> None:
    tag = (
        f"k_{case['k']:.3f}_a_{case['alpha']:.3f}_b_{case['beta']:.3f}"
        f"_bias_{case['bias']:.3f}_eps_{case['epsilon']:.3f}"
    )
    case_dir = out_dir / tag
    case_dir.mkdir(parents=True, exist_ok=True)

    traj = case["traj"]
    dist = case["dist"]
    t = np.arange(len(traj))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(t, traj[:, 0], lw=0.6, color="tab:blue")
    axes[0, 0].set_title("x(t)")

    axes[0, 1].plot(t, traj[:, 1], lw=0.6, color="tab:orange")
    axes[0, 1].set_title("y(t)")

    axes[1, 0].plot(traj[:, 0], traj[:, 1], lw=0.5, color="black")
    axes[1, 0].set_title("Phase Plane")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")

    axes[1, 1].plot(np.log(dist + 1e-12), lw=0.6, color="tab:purple")
    axes[1, 1].set_title("log(distance)")

    fig.tight_layout()
    fig.savefig(case_dir / "summary.png", dpi=150)
    plt.close(fig)


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Aihara parameters and rank them by largest Lyapunov exponent.")
    parser.add_argument("--steps", type=int, default=12000, help="Total simulation steps for each candidate.")
    parser.add_argument("--burn-in", type=int, default=2000, help="Transient steps discarded before LLE accumulation.")
    parser.add_argument("--top-k", type=int, default=12, help="Number of top candidates to save plots for.")
    parser.add_argument("--out", type=str, default="logs/aihara/chaos_scan", help="Output folder.")
    parser.add_argument("--k-list", type=str, default="0.85,0.90,0.94,0.96", help="Comma-separated k values.")
    parser.add_argument("--alpha-list", type=str, default="0.80,0.85,0.90,1.0", help="Comma-separated alpha values.")
    parser.add_argument("--beta-list", type=str, default="0.10,0.15,0.20", help="Comma-separated beta values.")
    parser.add_argument("--bias-list", type=str, default="0.20,0.25,0.35,0.50", help="Comma-separated bias values.")
    parser.add_argument("--epsilon-list", type=str, default="0.02,0.04,0.06,0.08", help="Comma-separated epsilon values.")
    args = parser.parse_args()

    k_list = parse_float_list(args.k_list)
    alpha_list = parse_float_list(args.alpha_list)
    beta_list = parse_float_list(args.beta_list)
    bias_list = parse_float_list(args.bias_list)
    epsilon_list = parse_float_list(args.epsilon_list)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    state0 = np.array([0.1, 0.1], dtype=np.float64)
    state1 = state0.copy()
    state2 = state0 + np.array([1e-6, 0.0], dtype=np.float64)

    results = []
    for k, alpha, beta, bias, epsilon in product(k_list, alpha_list, beta_list, bias_list, epsilon_list):
        lle = estimate_lle(state0, args.steps, args.burn_in, k, alpha, beta, bias, epsilon)
        traj = simulate(state0, args.steps, k, alpha, beta, bias, epsilon)[args.burn_in :]
        traj_a = simulate(state1, args.steps, k, alpha, beta, bias, epsilon)[args.burn_in :]
        traj_b = simulate(state2, args.steps, k, alpha, beta, bias, epsilon)[args.burn_in :]
        dist = np.linalg.norm(traj_a - traj_b, axis=1)
        peak_std = estimate_peak_std(traj[:, 0])
        results.append(
            {
                "k": k,
                "alpha": alpha,
                "beta": beta,
                "bias": bias,
                "epsilon": epsilon,
                "lle": lle,
                "peak_std": peak_std,
                "dist_ratio": float(dist[-1] / (dist[0] + 1e-12)),
                "traj": traj,
                "dist": dist,
            }
        )
        print(
            f"k={k:.3f}, alpha={alpha:.3f}, beta={beta:.3f}, bias={bias:.3f}, epsilon={epsilon:.3f}, "
            f"lle={lle:.6f}, peak_std={peak_std:.6f}"
        )

    results.sort(key=lambda case: (case["lle"], case["peak_std"]), reverse=True)

    ranking_path = out_dir / "ranking.txt"
    with ranking_path.open("w", encoding="utf-8") as handle:
        for idx, case in enumerate(results[: args.top_k], start=1):
            line = (
                f"{idx}. k={case['k']:.3f}, alpha={case['alpha']:.3f}, beta={case['beta']:.3f}, "
                f"bias={case['bias']:.3f}, epsilon={case['epsilon']:.3f}, "
                f"lle={case['lle']:.6f}, peak_std={case['peak_std']:.6f}, "
                f"dist_ratio={case['dist_ratio']:.3e}\n"
            )
            handle.write(line)
            print(line.strip())
            save_summary_plot(case, out_dir)

    defaults_path = out_dir / "default_reference.txt"
    default_lle = estimate_lle(
        state0,
        args.steps,
        args.burn_in,
        DEFAULT_K,
        DEFAULT_ALPHA,
        DEFAULT_BETA,
        DEFAULT_BIAS,
        DEFAULT_EPSILON,
    )
    defaults_path.write_text(
        (
            f"default_k={DEFAULT_K}\n"
            f"default_alpha={DEFAULT_ALPHA}\n"
            f"default_beta={DEFAULT_BETA}\n"
            f"default_bias={DEFAULT_BIAS}\n"
            f"default_epsilon={DEFAULT_EPSILON}\n"
            f"default_lle={default_lle:.6f}\n"
        ),
        encoding="utf-8",
    )
    print(f"Saved ranking to: {ranking_path}")
    print(f"Saved default reference to: {defaults_path}")


if __name__ == "__main__":
    main()
