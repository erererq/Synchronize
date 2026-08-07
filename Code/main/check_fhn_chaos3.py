from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


A = 0.7
B = 0.8
DT = 0.001
STEPS = 120000
BURN_IN = 20000

EPS_LIST = [0.03, 0.05, 0.08, 0.12]
C_LIST = [0.6, 0.9, 1.2, 1.5]
K_LIST = [0.6, 0.9, 1.2, 1.5]
D_LIST = [0.02, 0.05, 0.1, 0.2]


def deriv(state, eps, c, k, d):
    x, y, z = state
    dx = x - x**3 / 3.0 - y + c * z
    dy = eps * (x + A - B * y)
    dz = -k * x - d * z
    return np.array([dx, dy, dz], dtype=np.float64)


def simulate(state0, eps, c, k, d):
    state = state0.astype(np.float64).copy()
    hist = np.zeros((STEPS, 3), dtype=np.float64)
    for i in range(STEPS):
        hist[i] = state
        state = state + DT * deriv(state, eps, c, k, d)
    return hist


def estimate_peak_std(x):
    peaks = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            peaks.append(x[i])
    peaks = np.asarray(peaks, dtype=np.float64)
    if len(peaks) < 5:
        return 0.0
    return float(np.std(peaks))


def analyze_case(eps, c, k, d):
    traj = simulate(np.array([0.2, -0.1, 0.3]), eps, c, k, d)
    traj_ss = traj[BURN_IN:]

    traj1 = simulate(np.array([0.2, -0.1, 0.3]), eps, c, k, d)
    traj2 = simulate(np.array([0.200001, -0.1, 0.3]), eps, c, k, d)
    dist = np.linalg.norm(traj1 - traj2, axis=1)[BURN_IN:]

    peak_std = estimate_peak_std(traj_ss[:, 0])
    dist_ratio = float(dist[-1] / (dist[0] + 1e-12))
    x_range = float(traj_ss[:, 0].max() - traj_ss[:, 0].min())

    score = peak_std + 0.1 * np.log10(dist_ratio + 1e-12)

    return {
        "eps": eps,
        "c": c,
        "k": k,
        "d": d,
        "score": score,
        "peak_std": peak_std,
        "dist_ratio": dist_ratio,
        "x_range": x_range,
        "traj": traj_ss,
        "dist": dist,
    }


def save_case_plot(case, out_dir: Path):
    traj = case["traj"]
    dist = case["dist"]

    tag = f"eps_{case['eps']}_c_{case['c']}_k_{case['k']}_d_{case['d']}"
    case_dir = out_dir / tag
    case_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(len(traj)) * DT

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(t, traj[:, 0], lw=0.7)
    axes[0, 0].set_title("x(t)")

    axes[0, 1].plot(traj[:, 0], traj[:, 1], lw=0.5)
    axes[0, 1].set_title("x-y")

    axes[1, 0].plot(traj[:, 0], traj[:, 2], lw=0.5)
    axes[1, 0].set_title("x-z")

    axes[1, 1].plot(t, np.log(dist + 1e-12), lw=0.7)
    axes[1, 1].set_title("log distance")

    fig.tight_layout()
    fig.savefig(case_dir / "summary.png", dpi=150)
    plt.close(fig)


def main():
    out_dir = Path("logs") / "fhn" / "chaos_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for eps, c, k, d in product(EPS_LIST, C_LIST, K_LIST, D_LIST):
        case = analyze_case(eps, c, k, d)
        results.append(case)
        print(
            f"eps={eps:.3f}, c={c:.3f}, k={k:.3f}, d={d:.3f}, "
            f"score={case['score']:.6f}, peak_std={case['peak_std']:.6f}, "
            f"dist_ratio={case['dist_ratio']:.3e}"
        )

    results.sort(key=lambda x: x["score"], reverse=True)

    top_k = 12
    with open(out_dir / "ranking.txt", "w", encoding="utf-8") as f:
        for i, case in enumerate(results[:top_k], start=1):
            line = (
                f"{i}. eps={case['eps']}, c={case['c']}, k={case['k']}, d={case['d']}, "
                f"score={case['score']:.6f}, peak_std={case['peak_std']:.6f}, "
                f"dist_ratio={case['dist_ratio']:.3e}\n"
            )
            f.write(line)
            print(line.strip())
            save_case_plot(case, out_dir)


if __name__ == "__main__":
    main()
