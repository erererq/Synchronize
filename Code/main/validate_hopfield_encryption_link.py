"""Small end-to-end check from synchronization quality to image recovery.

This is intentionally a smoke test, not a paper-scale security experiment.  It
uses the project's existing DNA encrypt/decrypt functions and their current
quantization pipeline, while replacing only the trajectory generator so PPO,
fixed-gain feedback, LQR, and SMC can be compared under one common condition.
"""

from __future__ import annotations

import argparse
import csv
import runpy
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC


CURRENT_FILE = Path(__file__).resolve()
CODE_DIR = CURRENT_FILE.parent.parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from env.continuous_hopfield_env import ContinuousHopfieldEnv
from main.test_hopfield_smc_baseline import lqr_gain, sliding_surface, smc_force


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the synchronization-to-encryption link.")
    parser.add_argument("--node", type=int, default=2, choices=(2, 3))
    parser.add_argument("--mismatch", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--burn-in", type=int, default=400)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--linear-gain", type=float, default=8.0)
    parser.add_argument("--ppo-seed", type=int, default=42)
    parser.add_argument("--include-sac", action="store_true")
    parser.add_argument("--sac-seed", type=int, default=42)
    parser.add_argument("--sac-timesteps", type=int, default=100_000)
    parser.add_argument("--out", default="logs/hopfield/encryption_link_smoke")
    return parser.parse_args()


def controller_action(
    method: str,
    env: ContinuousHopfieldEnv,
    observation: np.ndarray,
    node: int,
    model: object,
    linear_gain: float,
) -> np.ndarray:
    if method in ("PPO", "SAC"):
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(1)
    error = (env.statey - env.statex).astype(np.float64)
    if method == "Fixed":
        force = -linear_gain * float(error[node - 1])
    elif method == "LQR":
        force = -float(lqr_gain(node) @ error)
    elif method == "SMC":
        force = smc_force(env, sliding_surface(node), reaching_gain=16.0, boundary_width=0.5)
    else:
        raise ValueError(method)
    return np.array([np.clip(force / env.scale, -1.0, 1.0)], dtype=np.float32)


def generate_trajectories(
    method: str,
    node: int,
    mismatch: float,
    seed: int,
    burn_in: int,
    sample_count: int,
    model: object,
    linear_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    env = ContinuousHopfieldEnv(pinning_node=node - 1, mismatch_scale=mismatch)
    observation, _ = env.reset(seed=seed)
    drive: list[np.ndarray] = []
    response: list[np.ndarray] = []
    for step in range(burn_in + sample_count):
        action = controller_action(method, env, observation, node, model, linear_gain)
        observation, _, _, _, _ = env.step(action)
        if step >= burn_in:
            drive.append(env.statex.astype(np.float64).copy())
            response.append(env.statey.astype(np.float64).copy())
    env.close()
    return np.asarray(drive), np.asarray(response)


def four_streams(states: np.ndarray) -> tuple[np.ndarray, ...]:
    fourth = states[:, 0] * states[:, 1] + states[:, 2]
    return states[:, 0], states[:, 1], states[:, 2], fourth


def legacy_quantize(streams: tuple[np.ndarray, ...], encryption_module: dict) -> tuple[np.ndarray, ...]:
    """Reproduce the quantization/filter loop currently used by photo_encry."""
    output: list[np.ndarray] = []
    kalman = encryption_module["process_array_with_kalman"]
    for stream in streams:
        quantized = (np.mod(np.round(stream), 8) + 1).astype(np.uint8)
        for _ in range(15):
            quantized = kalman(quantized)
            quantized = (np.mod(quantized, 8) + 1).astype(np.uint8)
        output.append(quantized)
    return tuple(output)


def direct_quantize(streams: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """The rule-index conversion before the legacy repeated Kalman/modulo loop."""
    return tuple((np.mod(np.round(stream), 8) + 1).astype(np.uint8) for stream in streams)


def key_metrics(master: tuple[np.ndarray, ...], slave: tuple[np.ndarray, ...]) -> tuple[float, float]:
    master_words = np.concatenate(master).astype(np.uint8) - 1
    slave_words = np.concatenate(slave).astype(np.uint8) - 1
    word_disagreement = float(np.mean(master_words != slave_words))
    xor = np.bitwise_xor(master_words, slave_words)
    bit_disagreement = float(np.unpackbits(xor[:, None], axis=1)[:, -3:].mean())
    return word_disagreement, bit_disagreement


def sequence_entropy(streams: tuple[np.ndarray, ...]) -> tuple[float, float]:
    unique_counts = [len(np.unique(stream)) for stream in streams]
    values = np.concatenate(streams).astype(np.int64)
    counts = np.bincount(values, minlength=9)[1:9].astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    return float(np.mean(unique_counts)), entropy


def save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    encryption_module = runpy.run_path(str(PROJECT_ROOT / "Code" / "encry" / "photo_encry"), run_name="photo_encry_module")
    source = cv2.imread(str(PROJECT_ROOT / "Manuscript" / "figures" / "original_image.jpg"))
    if source is None:
        raise FileNotFoundError("Manuscript/figures/original_image.jpg")
    plain = cv2.resize(source, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_dir / "plain_64.png"), plain)

    blocks_per_side = (args.image_size + args.block_size - 1) // args.block_size
    sample_count = blocks_per_side * blocks_per_side
    model_path = PROJECT_ROOT / "models" / "hopfield" / f"ppo_continuous_hopfield_node{args.node}_final_seed_{args.ppo_seed}.zip"
    model = PPO.load(str(model_path), device="cpu")
    sac_model = None
    if args.include_sac:
        sac_path = PROJECT_ROOT / "models" / "hopfield" / f"sac_continuous_hopfield_node{args.node}_screen_{args.sac_timesteps}steps_seed_{args.sac_seed}.zip"
        sac_model = SAC.load(str(sac_path), device="cpu")

    rows: list[dict] = []
    methods = ("PPO", "SAC", "Fixed", "LQR", "SMC") if args.include_sac else ("PPO", "Fixed", "LQR", "SMC")
    for seed in range(args.seed, args.seed + args.seed_count):
        for method in methods:
            active_model = sac_model if method == "SAC" else model
            drive, response = generate_trajectories(
                method, args.node, args.mismatch, seed, args.burn_in,
                sample_count, active_model, args.linear_gain,
            )
            master_direct = direct_quantize(four_streams(drive))
            slave_direct = direct_quantize(four_streams(response))
            direct_word_disagreement, direct_bit_disagreement = key_metrics(master_direct, slave_direct)
            direct_unique_mean, direct_entropy = sequence_entropy(master_direct)
            master = legacy_quantize(four_streams(drive), encryption_module)
            slave = legacy_quantize(four_streams(response), encryption_module)
            word_disagreement, bit_disagreement = key_metrics(master, slave)
            legacy_unique_mean, legacy_entropy = sequence_entropy(master)
            cipher, decryption_key = encryption_module["encrypt"](master, plain, "smoke-test-password")
            _, recovered = encryption_module["decrypt"](slave, cipher, decryption_key, "smoke-test-password")
            absolute_difference = cv2.absdiff(plain, recovered)
            image_mae = float(np.mean(absolute_difference))
            image_psnr = float(cv2.PSNR(plain, recovered))
            exact = bool(np.array_equal(plain, recovered))
            continuous_mae = float(np.mean(np.abs(response - drive)))
            row = {
                "node": args.node,
                "mismatch": args.mismatch,
                "test_seed": seed,
                "method": method,
                "continuous_mae": continuous_mae,
                "direct_key_word_disagreement": direct_word_disagreement,
                "direct_key_bit_disagreement": direct_bit_disagreement,
                "direct_master_unique_values_per_stream": direct_unique_mean,
                "direct_master_entropy_bits": direct_entropy,
                "legacy_key_word_disagreement": word_disagreement,
                "legacy_key_bit_disagreement": bit_disagreement,
                "legacy_master_unique_values_per_stream": legacy_unique_mean,
                "legacy_master_entropy_bits": legacy_entropy,
                "recovered_image_mae": image_mae,
                "recovered_image_psnr_db": image_psnr,
                "exact_recovery": exact,
            }
            rows.append(row)
            if seed == args.seed:
                cv2.imwrite(str(output_dir / f"{method.lower()}_cipher.png"), cipher)
                cv2.imwrite(str(output_dir / f"{method.lower()}_recovered.png"), recovered)
            print(row)

    save_csv(rows, output_dir / "metrics.csv")
    summary: list[dict] = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        summary.append({
            "method": method,
            "num_seeds": len(selected),
            "continuous_mae_mean": float(np.mean([row["continuous_mae"] for row in selected])),
            "direct_key_word_disagreement_mean": float(np.mean([row["direct_key_word_disagreement"] for row in selected])),
            "legacy_key_word_disagreement_mean": float(np.mean([row["legacy_key_word_disagreement"] for row in selected])),
            "recovered_image_mae_mean": float(np.mean([row["recovered_image_mae"] for row in selected])),
            "exact_recovery_rate": float(np.mean([row["exact_recovery"] for row in selected])),
        })
    save_csv(summary, output_dir / "summary.csv")
    for row in summary:
        print("SUMMARY", row)

    figure, axes = plt.subplots(1, len(methods) + 1, figsize=(2.4 * (len(methods) + 1), 2.8), constrained_layout=True)
    images = [("Plain", output_dir / "plain_64.png")]
    images.extend((method, output_dir / f"{method.lower()}_recovered.png") for method in methods)
    first_seed_rows = {row["method"]: row for row in rows if row["test_seed"] == args.seed}
    for axis, (label, path) in zip(axes, images):
        image = cv2.imread(str(path))
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if label == "Plain":
            title = label
        else:
            row = first_seed_rows[label]
            title = f"{label}\nkey mismatch={row['legacy_key_word_disagreement']:.1%}"
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.savefig(output_dir / "recovery_comparison_seed100.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved smoke-test outputs to {output_dir}")


if __name__ == "__main__":
    main()
