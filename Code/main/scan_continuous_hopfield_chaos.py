from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DT = 0.01
RK4_STEPS = 5
DELTA0 = 1e-8
STATE_CLIP = 40.0
DECAY = np.ones(3, dtype=np.float64)
BASE_WEIGHTS = np.array(
    [
        [1.5, 2.9, 0.8],
        [0.2, 1.18, 0.0],
        [2.977, -22.0, 0.47],
    ],
    dtype=np.float64,
)
# BASE_WEIGHTS = np.array([
#     [2.0,  -1.2,  0.0],
#     [1.2,   1.7,  1.15],
#     [-4.75, 0.0,  1.1]
# ], dtype=np.float64)

BIAS = np.zeros(3, dtype=np.float64)


def build_weights(alpha: float, beta: float) -> np.ndarray:
    weights = BASE_WEIGHTS.copy()
    weights[0, 2] = alpha
    weights[1, 0] = beta
    return weights


def activation(state: np.ndarray) -> np.ndarray:
    return np.tanh(state).astype(np.float64)


def derivatives(state: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (-DECAY * state + weights @ activation(state) + BIAS).astype(np.float64)


def rk4_micro_step(state: np.ndarray, weights: np.ndarray) -> np.ndarray:
    k1 = derivatives(state, weights)
    k2 = derivatives(state + 0.5 * DT * k1, weights)
    k3 = derivatives(state + 0.5 * DT * k2, weights)
    k4 = derivatives(state + DT * k3, weights)
    next_state = state + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.clip(next_state, -STATE_CLIP, STATE_CLIP).astype(np.float64)


def rk4_control_step(state: np.ndarray, weights: np.ndarray) -> np.ndarray:
    next_state = state.astype(np.float64).copy()
    for _ in range(RK4_STEPS):
        next_state = rk4_micro_step(next_state, weights)
    return next_state


def simulate(state0: np.ndarray, steps: int, weights: np.ndarray) -> np.ndarray:
    state = state0.astype(np.float64).copy()
    history = np.zeros((steps, 3), dtype=np.float64)

    for idx in range(steps):
        history[idx] = state
        state = rk4_control_step(state, weights)

    return history


def estimate_lle(
    state0: np.ndarray,
    steps: int,
    burn_in: int,
    weights: np.ndarray,
    delta0: float = DELTA0,
) -> tuple[float, np.ndarray]:
    reference = state0.astype(np.float64).copy()
    perturbed = reference + delta0 * np.array([1.0, 0.0, 0.0], dtype=np.float64)

    cumulative = []
    log_sum = 0.0
    valid_steps = 0

    for idx in range(steps):
        reference = rk4_control_step(reference, weights)
        perturbed = rk4_control_step(perturbed, weights)

        diff = perturbed - reference
        stretch = float(np.linalg.norm(diff))
        if stretch < 1e-16:
            diff = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            stretch = 1e-16

        if idx >= burn_in:
            log_sum += np.log(stretch / delta0)
            valid_steps += 1
            cumulative.append(log_sum / (valid_steps * DT * RK4_STEPS))

        perturbed = reference + delta0 * diff / stretch

    lle = float(log_sum / max(valid_steps, 1) / (DT * RK4_STEPS))
    return lle, np.asarray(cumulative, dtype=np.float64)


def estimate_peak_std(series: np.ndarray) -> float:
    peaks = []
    for idx in range(1, len(series) - 1):
        if series[idx] > series[idx - 1] and series[idx] > series[idx + 1]:
            peaks.append(series[idx])

    if len(peaks) < 5:
        return 0.0
    return float(np.std(np.asarray(peaks, dtype=np.float64)))


def summarize_case(alpha: float, beta: float, steps: int, burn_in: int) -> dict:
    weights = build_weights(alpha, beta)
    state0 = np.array([0.1, 0.0, -0.1], dtype=np.float64)
    traj = simulate(state0, steps, weights)
    traj_ss = traj[burn_in:]

    near_state = state0 + np.array([1e-6, 0.0, 0.0], dtype=np.float64)
    traj_a = simulate(state0, steps, weights)
    traj_b = simulate(near_state, steps, weights)
    dist = np.linalg.norm(traj_a - traj_b, axis=1)
    dist_ss = dist[burn_in:]

    lle, lle_series = estimate_lle(state0, steps, burn_in, weights)
    peak_std = estimate_peak_std(traj_ss[:, 0])
    dist_ratio = float(dist_ss[-1] / (dist_ss[0] + 1e-12))
    x_span = float(traj_ss[:, 0].max() - traj_ss[:, 0].min())
    score = lle + 0.02 * peak_std + 0.001 * np.log1p(max(dist_ratio, 0.0)) + 0.001 * x_span

    return {
        "alpha": alpha,
        "beta": beta,
        "weights": weights,
        "lle": lle,
        "peak_std": peak_std,
        "dist_ratio": dist_ratio,
        "x_span": x_span,
        "score": float(score),
        "traj": traj_ss,
        "dist": dist_ss,
        "lle_series": lle_series,
    }


def save_case_plot(case: dict, out_dir: Path) -> None:
    case_dir = out_dir / f"alpha_{case['alpha']:.3f}_beta_{case['beta']:.3f}"
    case_dir.mkdir(parents=True, exist_ok=True)

    traj = case["traj"]
    dist = case["dist"]
    lle_series = case["lle_series"]
    t = np.arange(len(traj)) * DT

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(t, traj[:, 0], lw=0.6, color="tab:blue")
    axes[0, 0].set_title("x1(t)")
    axes[0, 1].plot(traj[:, 0], traj[:, 1], lw=0.4, color="black")
    axes[0, 1].set_title("x1-x2")
    axes[1, 0].plot(traj[:, 0], traj[:, 2], lw=0.4, color="black")
    axes[1, 0].set_title("x1-x3")
    axes[1, 1].plot(np.log(dist + 1e-12), lw=0.6, color="tab:purple")
    axes[1, 1].set_title("log(distance)")
    fig.tight_layout()
    fig.savefig(case_dir / "summary.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(lle_series, color="tab:green", lw=0.8)
    ax.axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    ax.set_title("Running LLE")
    ax.set_xlabel("post burn-in step")
    ax.set_ylabel("LLE")
    fig.tight_layout()
    fig.savefig(case_dir / "running_lle.png", dpi=150)
    plt.close(fig)


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--burn-in", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--alpha-values", type=str, default="0.4,0.6,0.8,1.0,1.2")
    parser.add_argument("--beta-values", type=str, default="-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2")
    parser.add_argument("--out", type=str, default=str(Path("logs") / "hopfield" / "continuous_literature_scan"))
    args = parser.parse_args()

    alpha_values = parse_float_list(args.alpha_values)
    beta_values = parse_float_list(args.beta_values)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for alpha, beta in product(alpha_values, beta_values):
        case = summarize_case(alpha, beta, args.steps, args.burn_in)
        results.append(case)
        print(
            f"alpha={alpha:.3f}, beta={beta:.3f}, "
            f"lle={case['lle']:.6f}, peak_std={case['peak_std']:.6f}, "
            f"dist_ratio={case['dist_ratio']:.3e}"
        )

    results.sort(key=lambda item: item["score"], reverse=True)

    ranking_lines = []
    for idx, case in enumerate(results[: args.top_k], start=1):
        line = (
            f"{idx}. alpha={case['alpha']:.3f}, beta={case['beta']:.3f}, "
            f"lle={case['lle']:.6f}, peak_std={case['peak_std']:.6f}, "
            f"dist_ratio={case['dist_ratio']:.3e}, x_span={case['x_span']:.6f}, "
            f"score={case['score']:.6f}"
        )
        ranking_lines.append(line)
        print(line)
        save_case_plot(case, out_dir)

    (out_dir / "ranking.txt").write_text("\n".join(ranking_lines) + "\n", encoding="utf-8")
    reference_text = (
        "Model:\n"
        "dx1/dt = -x1 + 1.5*tanh(x1) + 2.9*tanh(x2) + alpha*tanh(x3)\n"
        "dx2/dt = -x2 + beta*tanh(x1) + 1.18*tanh(x2)\n"
        "dx3/dt = -x3 + 2.977*tanh(x1) - 22*tanh(x2) + 0.47*tanh(x3)\n"
    )
    (out_dir / "model_reference.txt").write_text(reference_text, encoding="utf-8")


if __name__ == "__main__":
    main()
