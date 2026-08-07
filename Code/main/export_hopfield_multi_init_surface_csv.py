from __future__ import annotations

from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
input_csv = project_root / "logs" / "hopfield" / "multi_init_test" / "node1_raw_init_trajectories.csv"
output_dir = project_root / "logs" / "hopfield" / "multi_init_test" / "node1"
value_columns = ["error", "e1", "e2", "e3"]
time_per_step = 0.05


def main() -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for value_col in value_columns:
        if value_col not in df.columns:
            print(f"Skip {value_col}: column not found.")
            continue

        surface_df = df.pivot(index="test_seed", columns="step", values=value_col)
        surface_df.columns = surface_df.columns.to_numpy(dtype=float) * time_per_step
        surface_df.columns.name = "time"
        out_path = output_dir / f"surface_{value_col}.csv"
        surface_df.to_csv(out_path, encoding="utf-8-sig")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
