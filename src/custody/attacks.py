import numpy as np

from custody.coevasion import CoEvasionTest


def naive_tamper(genuine_delta: np.ndarray, magnitude: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return genuine_delta + rng.normal(0, magnitude, size=genuine_delta.shape)


def heuristic_coevasion(genuine_joint_mean: np.ndarray, noise: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return genuine_joint_mean + rng.normal(0, noise, size=genuine_joint_mean.shape)


def adaptive_coevasion(
    coevasion_test: CoEvasionTest,
    target_mean: np.ndarray,
    n_steps: int = 200,
    step_size: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target_distance = (coevasion_test.tau_low + coevasion_test.tau_high) / 2

    best = target_mean + rng.normal(0, 0.5, size=target_mean.shape)
    best_error = abs(coevasion_test.score(best) - target_distance)

    for _ in range(n_steps):
        trial = best + rng.normal(0, step_size, size=best.shape)
        trial_error = abs(coevasion_test.score(trial) - target_distance)
        if trial_error < best_error:
            best, best_error = trial, trial_error

    return best
