from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

project_root = Path(__file__).resolve().parents[2]
csv_path = project_root / 'logs' / 'hopfield' / 'multi_init_test' / 'raw_init_trajectories.csv'
out_path = project_root / 'logs' / 'hopfield' / 'multi_init_test' / 'multi_init_surface_error.pdf'
value_col = 'error'  # choose from: e1, e2, e3, error
time_per_step = 0.05

plt.rcParams.update(
    {
        "text.usetex": True,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    }
)


def main() -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV file not found: {csv_path}')

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError('CSV file is empty.')
    if value_col not in df.columns:
        raise KeyError(f'Column not found in CSV: {value_col}')

    pivot = df.pivot(index='test_seed', columns='step', values=value_col)
    if pivot.isnull().values.any():
        raise ValueError('Trajectory grid contains missing values, cannot build a surface.')

    seed_axis = np.arange(len(pivot.index), dtype=int)
    step_axis = pivot.columns.to_numpy(dtype=float) * time_per_step
    lim=int(0.3 * len(step_axis))+1
    z_values = pivot.to_numpy()[:,:lim]
    x_grid, y_grid = np.meshgrid(step_axis[:lim], seed_axis)

    fig = plt.figure(figsize=(8.2, 6.0))
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_surface(
        x_grid,
        y_grid,
        z_values,
        cmap='jet',
        linewidth=0,
        antialiased=True,
        alpha=0.96,
    )

    z_range = float(z_values.max() - z_values.min())
    z_offset = float(z_values.min()) - 0.15 * (z_range + 1e-8)
    ax.contourf(
        x_grid,
        y_grid,
        z_values,
        zdir='z',
        offset=z_offset,
        cmap='jet',
        levels=40,
    )

    z_label_map = {
        'l1_error': r'$||e(t)||_1$',
        'e1': r'$e_1$',
        'e2': r'$e_2$',
        'e3': r'$e_3$',
        'error': r'$e(t)$',
    }

    ax.set_xlabel('Time', labelpad=0)
    ax.set_ylabel('Test Index', labelpad=0)
    ax.set_zlabel(z_label_map.get(value_col, value_col), labelpad=-2)
    ax.set_yticks(seed_axis)
    # === 强制锁死 X 和 Y 轴的物理边界，消除留白 ===
    ax.set_xlim(float(step_axis.min()))
    ax.set_ylim(float(seed_axis.min()), float(seed_axis.max()))
    ax.set_zlim(z_offset, float(z_values.max()))
    ax.set_box_aspect((1.5, 1.0, 0.9))
    ax.view_init(elev=15, azim=-60)
    # ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis._axinfo['juggled'] = (1, 2, 1)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.82, 0.82, 0.82, 1.0))

    ax.grid(True, alpha=0.35)

    colorbar = fig.colorbar(surface, ax=ax, shrink=0.74, pad=0.08, aspect=22)
    colorbar.set_label(z_label_map.get(value_col, value_col))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.2)
    # plt.show()

    print(f'3D surface figure saved to: {out_path}')


if __name__ == '__main__':
    main()
