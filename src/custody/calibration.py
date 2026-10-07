import numpy as np


def scores_to_pseudo_probabilities(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo == 0:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - lo) / (hi - lo)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == 0 else (probs > lo) & (probs <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_confidence = float(probs[mask].mean())
        bin_accuracy = float(labels[mask].mean())
        ece += (count / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)
