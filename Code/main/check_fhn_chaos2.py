from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DT = 0.001
STEPS = 200000
BURN_IN = 50000

MODEL_KIND = "3d"

# Shared FHN parameters.
A_PARAM = 0.7
B_PARAM = 0.8
EPS = 0.08
BIAS = 0.50
FORCE_AMP = 0.50
OMEGA = 1.0

# Slow-modulation parameter for the 3D model.
ALPHA = 0.20


def deriv_2d(state: np.ndarray, time_value: float) -> np.ndarray:
    x, y = state
    forcing = FORCE_AMP * np.sin(OMEGA * time_value)
    dx = x - x**3 / 3.0 - y + BIAS + forcing
    dy = EPS * (x + A_PARAM - B_PARAM * y)
    return np.array([dx, dy], dtype=np.float64)


def deriv_3d(state: np.ndarray, time_value: float) -> np.ndarray:
    x, y, z = state
    forcing = FORCE_AMP * np.sin(OMEGA * time_value)
    dx = x - x**3 / 3.0 - y + z + BIAS
    dy = EPS * (x + A_PARAM - B_PARAM * y)
    dz = -ALPHA * z + forcing
    return np.array([dx, dy, dz], dtype=np.float64)


def simulate(state0: np.ndarray, steps: int, model_kind: str) -> np.ndarray:
    state = state0.astype(np.float64).copy()
    history = np.zeros((steps, len(state0)), dtype=np.float64)
    time_value = 0.0

    for i in range(steps):
        history[i] = state
        if model_kind == "2d":
            state = state + DT * deriv_2d(state, time_value)
        else:
            state = state + DT * deriv_3d(state, time_value)
        time_value += DT

    return history


def make_output_dir(model_kind: str) -> Path:
    suffix = "2d" if model_kind == "2d" else "3d"
    out_dir = Path("logs") / "fhn" / f"chaos_check_forced_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_time_series(traj_ss: np.ndarray, out_dir: Path, model_kind: str) -> None:
    t = np.arange(len(traj_ss)) * DT
    n_rows = traj_ss.shape[1]
    labels = ["x(t)", "y(t)"] if model_kind == "2d" else ["x(t)", "y(t)", "z(t)"]
    colors = ["black", "tab:blue", "tab:green"]

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 7 if model_kind == "2d" else 8), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for idx in range(n_rows):
        axes[idx].plot(t, traj_ss[:, idx], color=colors[idx])
        axes[idx].set_ylabel(labels[idx])
    axes[-1].set_xlabel("time")

    fig.tight_layout()
    fig.savefig(out_dir / "time_series.png", dpi=150)
    plt.close(fig)


def plot_phase_planes(traj_ss: np.ndarray, out_dir: Path, model_kind: str) -> None:
    if model_kind == "2d":
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(traj_ss[:, 0], traj_ss[:, 1], lw=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Phase Plane: x-y")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(traj_ss[:, 0], traj_ss[:, 1], lw=0.5)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Phase Plane: x-y")

        axes[1].plot(traj_ss[:, 0], traj_ss[:, 2], lw=0.5)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("z")
        axes[1].set_title("Phase Plane: x-z")

        axes[2].plot(traj_ss[:, 1], traj_ss[:, 2], lw=0.5)
        axes[2].set_xlabel("y")
        axes[2].set_ylabel("z")
        axes[2].set_title("Phase Plane: y-z")

    fig.tight_layout()
    fig.savefig(out_dir / "phase_planes.png", dpi=150)
    plt.close(fig)


def plot_initial_sensitivity(dist_ss: np.ndarray, out_dir: Path) -> None:
    t_dist = np.arange(len(dist_ss)) * DT
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t_dist, dist_ss, color="tab:red")
    axes[0].set_ylabel("distance")
    axes[0].set_title("Sensitivity to Initial Conditions")

    axes[1].plot(t_dist, np.log(dist_ss + 1e-12), color="tab:purple")
    axes[1].set_ylabel("log(distance)")
    axes[1].set_xlabel("time")

    fig.tight_layout()
    fig.savefig(out_dir / "initial_sensitivity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    model_kind = MODEL_KIND.lower()
    if model_kind not in {"2d", "3d"}:
        raise ValueError("MODEL_KIND must be either '2d' or '3d'.")

    out_dir = make_output_dir(model_kind)
    state0 = np.array([0.2, -0.1], dtype=np.float64) if model_kind == "2d" else np.array([0.2, -0.1, 0.3], dtype=np.float64)

    traj = simulate(state0, STEPS, model_kind)
    traj_ss = traj[BURN_IN:]
    plot_time_series(traj_ss, out_dir, model_kind)
    plot_phase_planes(traj_ss, out_dir, model_kind)

    state1 = state0.copy()
    state2 = state0.copy()
    state2[0] += 1e-6
    traj1 = simulate(state1, STEPS, model_kind)
    traj2 = simulate(state2, STEPS, model_kind)
    dist = np.linalg.norm(traj1 - traj2, axis=1)
    dist_ss = dist[BURN_IN:]
    plot_initial_sensitivity(dist_ss, out_dir)

    print("Finished.")
    print(f"Saved figures to: {out_dir}")
    print(f"Model kind: {model_kind}")
    print(f"State ranges after burn-in:")
    print(f"x: [{traj_ss[:, 0].min():.4f}, {traj_ss[:, 0].max():.4f}]")
    print(f"y: [{traj_ss[:, 1].min():.4f}, {traj_ss[:, 1].max():.4f}]")
    if model_kind == "3d":
        print(f"z: [{traj_ss[:, 2].min():.4f}, {traj_ss[:, 2].max():.4f}]")
    print(f"Final separation: {dist[-1]:.6e}")


if __name__ == "__main__":
    main()
