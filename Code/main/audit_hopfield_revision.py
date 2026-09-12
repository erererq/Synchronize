"""Numerical and provenance checks for the manuscript comparison revision."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import solve_continuous_are

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))
from env.continuous_hopfield_env import ContinuousHopfieldEnv
from env.continuous_hopfield_multi_input_env import ContinuousHopfieldMultiInputEnv
from test_hopfield_smc_baseline import W_NOMINAL, lqr_gain, sliding_surface, run_episode
from hopfield_revision_metrics import trajectory_metrics


def main():
    result = {}
    rng = np.random.default_rng(1701)
    for node in (1, 2, 3):
        single = ContinuousHopfieldEnv(pinning_node=node-1)
        multi = ContinuousHopfieldMultiInputEnv(controlled_nodes=[node])
        obs1, _ = single.reset(seed=107)
        obs2, _ = multi.reset(seed=107)
        np.testing.assert_array_equal(obs1, obs2)
        max_difference = 0.
        for _ in range(400):
            action = rng.uniform(-1, 1, size=1).astype(np.float32)
            o1, r1, _, _, _ = single.step(action)
            o2, r2, _, _, _ = multi.step(action)
            max_difference = max(max_difference, float(np.abs(o1-o2).max()))
            np.testing.assert_allclose(o1, o2, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(r1, r2, atol=1e-5, rtol=1e-5)
        result[f"singleton_environment_node{node}_max_difference"] = max_difference
    for node in (2, 3):
        a = W_NOMINAL-np.eye(3)
        b = np.eye(3)[:, node-1:node]
        p = solve_continuous_are(a, b, np.eye(3), np.ones((1, 1)))
        k = lqr_gain(node)
        residual = a.T@p+p@a-p@b@b.T@p+np.eye(3)
        assert np.linalg.norm(residual) < 1e-8
        assert np.linalg.eigvals(a-b@k.reshape(1, -1)).real.max() < 0
        np.testing.assert_allclose(sliding_surface(node)@b[:, 0], 1.)
        result[f"lqr_node{node}"] = dict(K=k.tolist(), care_residual=float(np.linalg.norm(residual)),
                                      closed_loop_real_parts=np.linalg.eigvals(a-b@k.reshape(1,-1)).real.tolist())
    error = rng.normal(size=(401, 3))
    u = rng.uniform(-4, 4, size=(401, 3))
    u[0] = 0
    metrics = trajectory_metrics(error, u, pulse_step=200)
    np.testing.assert_allclose(metrics["energy_u"], sum(np.square(u[:, j]).sum()*.05 for j in range(3)))
    np.testing.assert_allclose(metrics["mae"], metrics["mean_l1"]/3)
    np.testing.assert_allclose(metrics["steady_l1"], np.abs(error[320:]).sum(axis=1).mean())
    np.testing.assert_allclose(metrics["mean_post_pulse_error"], np.abs(error[200:]).sum(axis=1).mean())
    result["metric_checks"] = "passed; initial sample, 81 tail samples, 201 post-pulse samples, summed energy"
    # Recover the size of the historical validation set from the stored values.
    grid_path = ROOT / "logs/hopfield/smc_baseline_extended/validation_grid.csv"
    grid = list(csv.DictReader(grid_path.open(encoding="utf-8-sig")))
    for node in (2, 3):
        target = next(row for row in grid if int(row["node"])==node and float(row["reaching_gain"])==16 and float(row["boundary_width"])==.5)
        samples = []
        matching_count = None
        for seed in range(20):
            samples.append(run_episode("SMC", node, seed, reaching_gain=16., boundary_width=.5)["mae"])
            if abs(np.mean(samples)-float(target["nominal_mae"])) < 1e-10:
                matching_count = seed+1
                break
        assert matching_count is not None, "Historical validation count could not be reproduced"
        result[f"smc_node{node}_validation_count"] = matching_count
        for name, mismatch, pulse in (("mismatch", 1., None), ("pulse", 0., 200)):
            values = [run_episode("SMC", node, seed, mismatch=mismatch, pulse_step=pulse,
                       pulse_vector=(5.,5.,5.), reaching_gain=16., boundary_width=.5)["mae"]
                      for seed in range(matching_count)]
            np.testing.assert_allclose(np.mean(values), float(target[f"{name}_mae"]), atol=1e-10)
        node_grid = [row for row in grid if int(row["node"])==node]
        best = min(node_grid, key=lambda row: float(row["score"]))
        result[f"smc_node{node}_grid_best"] = best
    out = ROOT / "logs/hopfield/revision_comparisons/protocol_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
