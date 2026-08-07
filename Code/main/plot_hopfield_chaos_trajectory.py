import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pathlib import Path

out_dir = Path("logs") / "hopfield" / "chaos_check"

# # 假设你在写双栏论文，单栏宽 3.5 寸
plt.rcParams.update({
    'figure.figsize': (3.5, 2.5), 
    'figure.figsize': (3.5, 3.5), 
    'figure.dpi': 300,             
    'font.size': 5,              
    'axes.labelsize': 5,         
    'lines.linewidth': 0.5,         
    'font.size': 8,              
    'axes.labelsize': 9,         
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 0.8         
})

dt = 0.01
def get_derivative(state, W_matrix):
    # state 是 (3,)，np.tanh(state) 也是 (3,)
    # W_matrix 是 (3,3)，矩阵乘法 @ 之后直接得到 (3,) 向量
    # 彻底告别按分量手写，既快又准
    dx = -state + W_matrix @ np.tanh(state)
    return np.array(dx, dtype=np.float32)

def rk4(state,W_matrix):
    k1 = get_derivative(state,W_matrix)
    k2 = get_derivative(state+0.5 * dt * k1, W_matrix)
    k3 = get_derivative(state+0.5 * dt * k2, W_matrix)
    k4 = get_derivative(state+dt * k3, W_matrix)

    next_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return next_state

def simulate_trajectory(initial_state, W_matrix, steps):
    trajectory = np.zeros((steps, 3),dtype=np.float32)
    trajectory[0] = initial_state

    for t in range(1, steps):
        trajectory[t] = rk4(trajectory[t-1], W_matrix)

    return trajectory

initial_state = np.array([0.0, 0.01, 0.0],dtype=np.float32)
W_matrix = np.array([
    [ -1.4,  1.2,  -7 ],
    [ 1.1,   0.0,  2.8],
    [0.8,   -2,  4 ]
], dtype=np.float32)


step =np.arange(0,200,dt)
trajectory = simulate_trajectory(initial_state, W_matrix, len(step))
fig, ax = plt.subplots(nrows=3, ncols=1)
ax[0].plot(step, trajectory[:, 0], label='x1',color = "blue")
ax[0].set_ylabel("$x_1(t)$")
ax[1].plot(step, trajectory[:, 1], label='x2',color = "orange")
ax[1].set_ylabel("$x_2(t)$")
ax[2].plot(step, trajectory[:, 2], label='x3',color = "green")
ax[2].set_ylabel("$x_3(t)$")
ax[2].set_xlabel('t/s')
fig.tight_layout()
plt.show()
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / 'hopfield_chaos_trajectory.png')