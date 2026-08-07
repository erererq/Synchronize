from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from stable_baselines3 import PPO

CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent

if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

if TYPE_CHECKING:
    from env.continuous_hopfield_env import ContinuousHopfieldEnv

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


DEFAULT_PATTERN = "ppo_continuous_hopfield_node3_final_seed_*.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the signed PPO control input over the (x1, x2) phase plane "
            "for the continuous Hopfield pinning-node-3 case."
        )
    )
    parser.add_argument("--model", type=str, default=None, help="Path to one PPO .zip model.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/hopfield",
        help="Directory used when --model is not set.",
    )
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of environment steps for the standard rollout.",
    )
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.0,
        help="Scatter point size. Use 0 for a clean colored trajectory line only.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional figure title. Empty by default because captions usually carry the description.",
    )
    parser.add_argument(
        "--color-percentiles",
        type=float,
        nargs=2,
        default=[2.0, 98.0],
        metavar=("LOW", "HIGH"),
        help=(
            "Percentile range used for the adaptive color scale. "
            "Use 0 100 to span the full observed u(t) range."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="logs/hopfield/test_results/hopfield_node3_control_phase_plane.png",
    )
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        model_path = model_path.resolve()
    else:
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = PROJECT_ROOT / model_dir
        matches = sorted(model_dir.glob(args.pattern))
        if not matches:
            raise FileNotFoundError(
                f"No node-3 PPO model matched {args.pattern!r} in {model_dir}."
            )
        model_path = matches[0].resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model_path


def make_node3_env(args: argparse.Namespace) -> "ContinuousHopfieldEnv":
    try:
        from env.continuous_hopfield_env import ContinuousHopfieldEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gymnasium and the local Hopfield environment dependencies are required "
            "to generate rollout data. Install the same environment used for PPO training."
        ) from exc

    env = ContinuousHopfieldEnv(
        mismatch_scale=0.0,
        pulse_step=None,
        pulse_width=1,
        pulse_vector=(0.0, 0.0, 0.0),
        pinning_node=2,  # node 3 in the paper, zero-based index in the code
    )
    env.max_steps = max(int(args.steps), 1)
    return env


def run_episode(
    model: "PPO",
    env: "ContinuousHopfieldEnv",
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs, _ = env.reset(seed=seed)

    x1_traj: list[float] = []
    x2_traj: list[float] = []
    u_traj: list[float] = []

    done = False
    while not done:
        statex = np.asarray(env.statex, dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)

        # This is already the physical control input u(t), not the normalized action.
        # Do not multiply by env.scale again when plotting.
        control_input = float(action[0]) * env.scale

        x1_traj.append(float(statex[0]))
        x2_traj.append(float(statex[1]))
        u_traj.append(control_input)

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return (
        np.asarray(x1_traj, dtype=np.float32),
        np.asarray(x2_traj, dtype=np.float32),
        np.asarray(u_traj, dtype=np.float32),
    )


def maybe_downsample(
    x1_traj: np.ndarray,
    x2_traj: np.ndarray,
    u_traj: np.ndarray,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_points is None or max_points <= 0 or len(x1_traj) <= max_points:
        return x1_traj, x2_traj, u_traj

    indices = np.linspace(0, len(x1_traj) - 1, max_points).astype(int)
    return x1_traj[indices], x2_traj[indices], u_traj[indices]


def make_signed_norm(u_traj: np.ndarray, color_percentiles: tuple[float, float]) -> Normalize:
    finite_u = u_traj[np.isfinite(u_traj)]
    if finite_u.size == 0:
        raise ValueError("u_traj contains no finite values.")

    low, high = sorted(color_percentiles)
    low = float(np.clip(low, 0.0, 100.0))
    high = float(np.clip(high, 0.0, 100.0))
    if np.isclose(low, high):
        low, high = 0.0, 100.0

    u_min, u_max = np.percentile(finite_u, [low, high])
    u_min = float(u_min)
    u_max = float(u_max)

    if np.isclose(u_min, u_max):
        center = float(np.median(finite_u))
        pad = max(abs(center) * 0.05, 1e-6)
        return Normalize(vmin=center - pad, vmax=center + pad, clip=True)

    if u_min < 0.0 < u_max:
        return TwoSlopeNorm(vmin=u_min, vcenter=0.0, vmax=u_max)
    return Normalize(vmin=u_min, vmax=u_max, clip=True)


def make_colorbar_ticks(norm: Normalize) -> np.ndarray:
    vmin = float(norm.vmin)
    vmax = float(norm.vmax)

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.array([])
    if np.isclose(vmin, vmax):
        return np.array([vmin])

    if vmin < 0.0 < vmax:
        ticks = np.array([vmin, 0.5 * vmin, 0.0, 0.5 * vmax, vmax])
    else:
        ticks = np.linspace(vmin, vmax, 5)

    # Remove near-duplicates while preserving order, which matters when one side
    # of the signed control range is much smaller than the other.
    unique_ticks: list[float] = []
    tolerance = max(abs(vmax - vmin) * 1e-6, 1e-12)
    for tick in ticks:
        tick = float(tick)
        if not any(abs(tick - existing) <= tolerance for existing in unique_ticks):
            unique_ticks.append(tick)
    return np.asarray(unique_ticks, dtype=float)


def format_colorbar_tick(value: float, _position: int) -> str:
    if abs(value) < 5e-5:
        return "0"
    if abs(value) < 0.1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def plot_control_phase_plane(
    x1_traj: np.ndarray,
    x2_traj: np.ndarray,
    u_traj: np.ndarray,
    out_path: Path,
    point_size: float = 0.0,
    title: str = "",
    color_percentiles: tuple[float, float] = (2.0, 98.0),
) -> None:
    if not (len(x1_traj) == len(x2_traj) == len(u_traj)):
        raise ValueError("x1_traj, x2_traj, and u_traj must have the same length.")
    if len(x1_traj) < 2:
        raise ValueError("At least two trajectory points are required for plotting.")

    points = np.column_stack([x1_traj, x2_traj]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = make_signed_norm(u_traj, color_percentiles)
    segment_u = 0.5 * (u_traj[:-1] + u_traj[1:])

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    line = LineCollection(
        segments,
        cmap="coolwarm",
        norm=norm,
        linewidth=0.65,
        alpha=0.95,
    )
    line.set_array(segment_u)
    ax.add_collection(line)

    colorbar_source = line
    if point_size > 0.0:
        scatter = ax.scatter(
            x1_traj,
            x2_traj,
            c=u_traj,
            cmap="coolwarm",
            norm=norm,
            s=point_size,
            alpha=0.45,
            edgecolors="none",
        )
        colorbar_source = scatter

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    if title:
        ax.set_title(title, pad=6)
    ax.autoscale()
    ax.margins(0.04)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.45)

    colorbar = fig.colorbar(
        colorbar_source,
        ax=ax,
        pad=0.02,
        shrink=0.88,
        ticks=make_colorbar_ticks(norm),
    )
    colorbar.set_label(r"Control input $u(t)$", labelpad=8)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(format_colorbar_tick))
    colorbar.ax.tick_params(labelsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure to: {out_path}")


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    print(f"Using PPO model: {model_path}")
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable_baselines3 is required to load the PPO model. "
            "Install it in the Python environment used to run this script."
        ) from exc

    model = PPO.load(str(model_path), device="cpu")
    env = make_node3_env(args)
    try:
        x1_traj, x2_traj, u_traj = run_episode(model, env, args.seed)
    finally:
        env.close()

    x1_traj, x2_traj, u_traj = maybe_downsample(
        x1_traj,
        x2_traj,
        u_traj,
        args.max_points,
    )
    plot_control_phase_plane(
        x1_traj,
        x2_traj,
        u_traj,
        out_path,
        point_size=args.point_size,
        title=args.title,
        color_percentiles=tuple(args.color_percentiles),
    )


if __name__ == "__main__":
    main()
