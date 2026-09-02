from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "logs" / "hopfield" / "multi_init_test"
FIGURE_DIR = PROJECT_ROOT / "Manuscript" / "figures"
TIME_PER_STEP = 0.05
DISPLAY_FRACTION = 0.30

NODE_FILES = {
    3: DATA_DIR / "raw_init_trajectories.csv",
    2: DATA_DIR / "node2_raw_init_trajectories.csv",
    1: DATA_DIR / "node1_raw_init_trajectories.csv",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 9,
        "figure.dpi": 180,
        "savefig.dpi": 300,
    }
)


def load_components(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    frame = pd.read_csv(csv_path)
    required = {"test_seed", "step", "e1", "e2", "e3"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {sorted(missing)}")

    matrices: list[np.ndarray] = []
    time_axis: np.ndarray | None = None
    seeds: np.ndarray | None = None
    for component in ("e1", "e2", "e3"):
        pivot = frame.pivot(index="test_seed", columns="step", values=component)
        if pivot.isnull().values.any():
            raise ValueError(f"Incomplete trajectory grid in {csv_path}: {component}")
        limit = int(DISPLAY_FRACTION * pivot.shape[1]) + 1
        pivot = pivot.iloc[:, :limit]
        if time_axis is None:
            time_axis = pivot.columns.to_numpy(dtype=float) * TIME_PER_STEP
            seeds = pivot.index.to_numpy()
        matrices.append(pivot.to_numpy())

    assert time_axis is not None and seeds is not None
    return time_axis, seeds, matrices


def main() -> None:
    loaded = {node: load_components(path) for node, path in NODE_FILES.items()}
    color_limit = max(
        float(np.max(np.abs(matrix)))
        for _, _, matrices in loaded.values()
        for matrix in matrices
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    comparison, comparison_axes = plt.subplots(3, 3, figsize=(8.2, 5.5), sharex=True, sharey=True)
    comparison_image = None
    for row, node in enumerate((3, 2, 1)):
        time_axis, seeds, matrices = loaded[node]
        extent = [float(time_axis[0]), float(time_axis[-1]), len(seeds) + 0.5, 0.5]
        for column, matrix in enumerate(matrices):
            axis = comparison_axes[row, column]
            comparison_image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-color_limit,
                vmax=color_limit,
                extent=extent,
            )
            if row == 0:
                axis.set_title(rf"$e_{column + 1}$")
            if column == 0:
                axis.set_ylabel(f"Node {node}\nTest index")
                axis.set_yticks((1, 5, 10))
            if row == 2:
                axis.set_xlabel("Time (s)")
    assert comparison_image is not None
    comparison_colorbar = comparison.colorbar(
        comparison_image, ax=comparison_axes, fraction=0.022, pad=0.018
    )
    comparison_colorbar.set_label("Signed synchronization error")
    comparison.subplots_adjust(left=0.10, right=0.88, bottom=0.10, top=0.95, wspace=0.08, hspace=0.12)
    comparison_output = FIGURE_DIR / "node_error_heatmap_comparison.png"
    comparison.savefig(comparison_output, bbox_inches="tight", pad_inches=0.04)
    plt.close(comparison)
    print(f"Saved {comparison_output}")

    for node, (time_axis, seeds, matrices) in loaded.items():
        fig, axes = plt.subplots(3, 1, figsize=(6.6, 5.8), sharex=True)
        image = None
        extent = [float(time_axis[0]), float(time_axis[-1]), len(seeds) + 0.5, 0.5]
        for component_index, (axis, matrix) in enumerate(zip(axes, matrices), start=1):
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-color_limit,
                vmax=color_limit,
                extent=extent,
            )
            axis.set_ylabel("Test index")
            axis.set_yticks(np.arange(1, len(seeds) + 1))
            axis.set_title(rf"$e_{component_index}$", loc="left", pad=2)
        axes[-1].set_xlabel("Time (s)")
        assert image is not None
        colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
        colorbar.set_label("Signed synchronization error")
        fig.subplots_adjust(left=0.12, right=0.86, bottom=0.09, top=0.98, hspace=0.20)
        output = FIGURE_DIR / f"node{node}_error_heatmap.png"
        fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
