"""Shared sampled-data metrics, including initial error and all actuators."""
import numpy as np


def trajectory_metrics(error, control, step_time=0.05, action_limit=4.0, pulse_step=None):
    error = np.asarray(error, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64).reshape(len(error), -1)
    assert error.ndim == 2 and error.shape[1] == 3
    assert np.isfinite(error).all() and np.isfinite(control).all()
    assert np.max(np.abs(control)) <= action_limit + 1e-6
    l1 = np.abs(error).sum(axis=1)
    result = dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.square(error).mean())),
                  mean_l1=float(l1.mean()), steady_l1=float(l1[int(0.8*len(l1)):].mean()),
                  final_l1=float(l1[-1]), mean_abs_u=float(np.abs(control).sum(axis=1).mean()),
                  energy_u=float(np.square(control).sum()*step_time),
                  saturation_fraction=float(np.isclose(np.abs(control[1:]), action_limit, atol=1e-6).mean()))
    if pulse_step is not None:
        result.update(max_post_pulse=float(l1[pulse_step:].max()),
                      mean_post_pulse_error=float(l1[pulse_step:].mean()))
    return result
