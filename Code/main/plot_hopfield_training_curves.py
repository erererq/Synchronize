from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
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

LOG_DIRS = {
    "Node 1": Path("logs/hopfield/tb/node1"),
    "Node 2": Path("logs/hopfield/tb/node2"),
    "Node 3": Path("logs/hopfield/tb/node3"),
}

TAG = "eval/mean_reward"
OUT_PATH = Path("logs/hopfield/training_curves_eval_reward.png")


def load_scalar(log_dir: Path, tag: str):
    event_files = sorted(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")

    ea = EventAccumulator(str(event_files[-1]))
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        raise KeyError(f"Tag '{tag}' not found in {log_dir}. Available: {ea.Tags()['scalars']}")

    events = ea.Scalars(tag)
    x = [e.step for e in events]
    y = [e.value for e in events]
    return x, y


def main():
    plt.figure(figsize=(8, 5))

    for label, log_dir in LOG_DIRS.items():
        x, y = load_scalar(log_dir, TAG)
        plt.plot(x, y, label=label)

    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Mean Reward")
    plt.title("Training Curves Under Different Pinning Nodes")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    plt.show()

    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
