import numpy as np


def conflict_score(declared: np.ndarray, inferred: np.ndarray) -> float:
    return float(np.linalg.norm(declared - inferred))


def snapshot_baseline_flag(declared: np.ndarray, inferred: np.ndarray, threshold: float) -> bool:
    return conflict_score(declared, inferred) > threshold
