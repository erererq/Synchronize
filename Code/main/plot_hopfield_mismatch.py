import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.6,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

df = pd.read_csv("logs/hopfield/test_results/metrics_summary.csv")

ppo = df[df["method"] == "PPO"].sort_values("mismatch")
linear = df[df["method"] == "Linear"].sort_values("mismatch")

plt.figure(figsize=(3.5, 2.6))
plt.plot(ppo["mismatch"], ppo["mae_mean"], marker="s", markersize=4.5, label="PPO")
plt.plot(linear["mismatch"], linear["mae_mean"], marker="o", markersize=4.5, linestyle="--", label="Linear")

plt.xlabel(r"Mismatch Scale $\eta$")
plt.ylabel("MAE")
plt.grid(True, linestyle=":", alpha=0.45)
plt.legend(frameon=True, framealpha=0.9)
plt.tight_layout()
plt.savefig("logs/hopfield/test_results/mismatch_mae.pdf")
# plt.savefig("logs/hopfield/test_results/node2_mismatch_mae.pdf")
plt.show()
