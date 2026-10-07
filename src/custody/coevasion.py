import numpy as np
from scipy.spatial.distance import mahalanobis


class CoEvasionTest:
    def __init__(self):
        self.mean: np.ndarray | None = None
        self.inv_cov: np.ndarray | None = None
        self.tau_low: float | None = None
        self.tau_high: float | None = None

    def fit(
        self, joint_samples: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0
    ) -> None:
        self.mean = joint_samples.mean(axis=0)
        cov = np.cov(joint_samples, rowvar=False)
        self.inv_cov = np.linalg.pinv(np.atleast_2d(cov))
        distances = np.array([self.score(s) for s in joint_samples])
        self.tau_low = float(np.percentile(distances, low_percentile))
        self.tau_high = float(np.percentile(distances, high_percentile))

    def score(self, sample: np.ndarray) -> float:
        return float(mahalanobis(sample, self.mean, self.inv_cov))

    def flag(self, sample: np.ndarray) -> str | None:
        distance = self.score(sample)
        if distance > self.tau_high:
            return "mismatch"
        if distance < self.tau_low:
            return "too_perfect"
        return None

    def anomaly_score(self, sample: np.ndarray) -> float:
        distance = self.score(sample)
        return float(max(distance - self.tau_high, self.tau_low - distance, 0.0))
