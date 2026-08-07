import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DECAY = 0.35
DEFAULT_WEIGHTS = np.array([[1.48, -1.62], [1.16, 0.18]], dtype=np.float64)
DEFAULT_BIAS = np.array([0.05, -0.02], dtype=np.float64)
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


def estimate_lle(
    state0: np.ndarray,
    steps: int,
    burn_in: int,
    decay: float,
    weights: np.ndarray,
    bias: np.ndarray,
    delta0: float = DELTA0,
) -> float:
    reference = state0.astype(np.float64).copy()
    perturbed = reference + delta0 * np.array([1.0, 0.0], dtype=np.float64)
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


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def save_summary_plot(case: dict, out_dir: Path) -> None:
    tag = (
        f"decay_{case['decay']:.3f}"
        f"_w11_{case['w11']:.3f}_w12_{case['w12']:.3f}"
        f"_w21_{case['w21']:.3f}_w22_{case['w22']:.3f}"
        f"_b1_{case['b1']:.3f}_b2_{case['b2']:.3f}"
    )
    case_dir = out_dir / tag
    case_dir.mkdir(parents=True, exist_ok=True)

    traj = case["traj"]
    dist = case["dist"]
    t = np.arange(len(traj))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(t, traj[:, 0], lw=0.6, color="tab:blue")
    axes[0, 0].set_title("x1(n)")

    axes[0, 1].plot(t, traj[:, 1], lw=0.6, color="tab:orange")
    axes[0, 1].set_title("x2(n)")

    axes[1, 0].plot(traj[:, 0], traj[:, 1], lw=0.5, color="black")
    axes[1, 0].set_title("Phase Plane")
    axes[1, 0].set_xlabel("x1")
    axes[1, 0].set_ylabel("x2")

    axes[1, 1].plot(np.log(dist + 1e-12), lw=0.6, color="tab:purple")
    axes[1, 1].set_title("log(distance)")

    fig.tight_layout()
    fig.savefig(case_dir / "summary.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Hopfield drive-system parameters and rank them by largest Lyapunov exponent.")
    parser.add_argument("--steps", type=int, default=20000, help="Total simulation steps for each candidate.")
    parser.add_argument("--burn-in", type=int, default=2000, help="Transient steps discarded before LLE accumulation.")
    parser.add_argument("--top-k", type=int, default=12, help="Number of top candidates to save plots for.")
    parser.add_argument("--out", type=str, default="logs/hopfield/chaos_scan", help="Output folder.")
    parser.add_argument("--decay-list", type=str, default="0.25,0.30,0.35,0.40,0.45", help="Comma-separated decay values.")
    parser.add_argument("--w11-list", type=str, default="1.20,1.48,1.80", help="Comma-separated w11 values.")
    parser.add_argument("--w12-list", type=str, default="-1.90,-1.62,-1.30", help="Comma-separated w12 values.")
    parser.add_argument("--w21-list", type=str, default="0.80,1.16,1.50", help="Comma-separated w21 values.")
    parser.add_argument("--w22-list", type=str, default="-0.20,0.18,0.50", help="Comma-separated w22 values.")
    parser.add_argument("--b1-list", type=str, default="-0.10,0.05,0.20", help="Comma-separated bias-1 values.")
    parser.add_argument("--b2-list", type=str, default="-0.20,-0.02,0.10", help="Comma-separated bias-2 values.")
    args = parser.parse_args()

    decay_list = parse_float_list(args.decay_list)
    w11_list = parse_float_list(args.w11_list)
    w12_list = parse_float_list(args.w12_list)
    w21_list = parse_float_list(args.w21_list)
    w22_list = parse_float_list(args.w22_list)
    b1_list = parse_float_list(args.b1_list)
    b2_list = parse_float_list(args.b2_list)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    state0 = np.array([0.2, -0.1], dtype=np.float64)
    near_state = state0 + np.array([1e-6, 0.0], dtype=np.float64)

    results = []
    for decay, w11, w12, w21, w22, b1, b2 in product(
        decay_list,
        w11_list,
        w12_list,
        w21_list,
        w22_list,
        b1_list,
        b2_list,
    ):
        weights = np.array([[w11, w12], [w21, w22]], dtype=np.float64)
        bias = np.array([b1, b2], dtype=np.float64)
        lle = estimate_lle(state0, args.steps, args.burn_in, decay, weights, bias)
        traj = simulate(state0, args.steps, decay, weights, bias)[args.burn_in :]
        traj_a = simulate(state0, args.steps, decay, weights, bias)[args.burn_in :]
        traj_b = simulate(near_state, args.steps, decay, weights, bias)[args.burn_in :]
        dist = np.linalg.norm(traj_a - traj_b, axis=1)
        peak_std = estimate_peak_std(traj[:, 0])
        results.append(
            {
                "decay": decay,
                "w11": w11,
                "w12": w12,
                "w21": w21,
                "w22": w22,
                "b1": b1,
                "b2": b2,
                "lle": lle,
                "peak_std": peak_std,
                "dist_ratio": float(dist[-1] / (dist[0] + 1e-12)),
                "traj": traj,
                "dist": dist,
            }
        )
        print(
            f"decay={decay:.3f}, weights=[[{w11:.3f},{w12:.3f}],[{w21:.3f},{w22:.3f}]], "
            f"bias=[{b1:.3f},{b2:.3f}], lle={lle:.6f}, peak_std={peak_std:.6f}"
        )

    results.sort(key=lambda case: (case["lle"], case["peak_std"]), reverse=True)

    ranking_path = out_dir / "ranking.txt"
    with ranking_path.open("w", encoding="utf-8") as handle:
        for idx, case in enumerate(results[: args.top_k], start=1):
            line = (
                f"{idx}. decay={case['decay']:.3f}, "
                f"weights=[[{case['w11']:.3f},{case['w12']:.3f}],[{case['w21']:.3f},{case['w22']:.3f}]], "
                f"bias=[{case['b1']:.3f},{case['b2']:.3f}], "
                f"lle={case['lle']:.6f}, peak_std={case['peak_std']:.6f}, "
                f"dist_ratio={case['dist_ratio']:.3e}\n"
            )
            handle.write(line)
            print(line.strip())
            save_summary_plot(case, out_dir)

    default_lle = estimate_lle(state0, args.steps, args.burn_in, DEFAULT_DECAY, DEFAULT_WEIGHTS, DEFAULT_BIAS)
    default_path = out_dir / "default_reference.txt"
    default_path.write_text(
        (
            f"default_decay={DEFAULT_DECAY:.6f}\n"
            f"default_weights={DEFAULT_WEIGHTS.tolist()}\n"
            f"default_bias={DEFAULT_BIAS.tolist()}\n"
            f"default_lle={default_lle:.6f}\n"
        ),
        encoding="utf-8",
    )

    print(f"Saved ranking to: {ranking_path}")
    print(f"Saved default reference to: {default_path}")


if __name__ == "__main__":
    main()
